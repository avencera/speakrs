//! cuDNN forward convolutions: one planner and one plan type for every model area
//!
//! [`ConvPlanner`] picks each algorithm from cuDNN's own ranking, so the same GPU, cuDNN
//! version and shape always run the same algorithm with the same numerics. A
//! [`ConvPlan`] owns its descriptors but no workspace: a model allocates one workspace
//! for its largest plan and passes it to every convolution, because a model's
//! convolutions never run at once

use std::ffi::c_int;

use cudarc::cudnn::sys::{
    cudnnActivationMode_t, cudnnConvolutionDescriptor_t, cudnnConvolutionFwdAlgo_t,
    cudnnConvolutionFwdAlgoPerf_t, cudnnConvolutionMode_t, cudnnDataType_t,
    cudnnFilterDescriptor_t, cudnnHandle_t, cudnnMathType_t, cudnnNanPropagation_t, cudnnStatus_t,
    cudnnTensorDescriptor_t, cudnnTensorFormat_t,
};
use cudarc::cudnn::{
    ActivationDescriptor, ConvBiasActivationForward, ConvDescriptor, ConvForward, FilterDescriptor,
    TensorDescriptor, result,
};
use cudarc::driver::{CudaView, CudaViewMut};
use tracing::debug;

use super::error::{check_len, element_count, to_c_int};
use super::{CudaError, CudaMath, CudaRuntime};

/// Number of forward algorithms cuDNN ranks
const ALGORITHM_COUNT: usize = cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_COUNT as usize;

/// An NCHW 2-D convolution on FP32 buffers (cross-correlation, as in PyTorch `Conv2d`)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Conv2d {
    /// Batch size `n`
    pub batch: usize,
    /// Input channels `c`
    pub in_channels: usize,
    /// Output channels `k`
    pub out_channels: usize,
    /// Input height and width
    pub input: [usize; 2],
    /// Filter height and width
    pub kernel: [usize; 2],
    /// Zero padding on each side of height and width
    pub padding: [usize; 2],
    /// Stride along height and width
    pub stride: [usize; 2],
    /// Dilation along height and width
    pub dilation: [usize; 2],
    /// Precision of the convolution; FP32 unless the caller opts into TF32
    pub math: CudaMath,
}

impl Conv2d {
    /// Output height and width, using the PyTorch formula
    pub fn output(&self) -> [usize; 2] {
        [0, 1].map(|axis| {
            let span = self.dilation[axis] * (self.kernel[axis].saturating_sub(1)) + 1;
            (self.input[axis] + 2 * self.padding[axis]).saturating_sub(span)
                / self.stride[axis].max(1)
                + 1
        })
    }

    /// Input shape `[n, c, h, w]`
    pub fn input_shape(&self) -> [usize; 4] {
        [self.batch, self.in_channels, self.input[0], self.input[1]]
    }

    /// Filter shape `[k, c, r, s]`
    pub fn filter_shape(&self) -> [usize; 4] {
        [
            self.out_channels,
            self.in_channels,
            self.kernel[0],
            self.kernel[1],
        ]
    }

    /// Output shape `[n, k, p, q]`
    pub fn output_shape(&self) -> [usize; 4] {
        let [p, q] = self.output();
        [self.batch, self.out_channels, p, q]
    }
}

/// What a fused convolution adds before its ReLU besides the bias
#[derive(Debug)]
pub(crate) enum Residual<'a, 'b> {
    /// Nothing. cuDNN still takes a `z` operand of the output's shape, scaled by
    /// zero; pass any buffer of that size holding finite values, since `0 * NaN`
    /// would still be NaN
    None {
        /// A finite buffer of the output's size
        scratch: &'a CudaView<'b, f32>,
    },
    /// A residual of the output's shape
    Add(&'a CudaView<'b, f32>),
}

/// A cuDNN forward convolution with an algorithm chosen by [`ConvPlanner`]
///
/// The plan owns no workspace; see the module docs
#[derive(Debug)]
pub(crate) struct ConvPlan {
    spec: Conv2d,
    conv: ConvDescriptor<f32>,
    x: TensorDescriptor<f32>,
    w: FilterDescriptor<f32>,
    y: TensorDescriptor<f32>,
    bias: TensorDescriptor<f32>,
    relu: ActivationDescriptor<f32>,
    algo: cudnnConvolutionFwdAlgo_t,
    workspace_bytes: usize,
}

impl ConvPlan {
    /// The convolution this plan runs
    pub fn spec(&self) -> &Conv2d {
        &self.spec
    }

    /// Bytes of workspace the chosen algorithm needs
    pub fn workspace_bytes(&self) -> usize {
        self.workspace_bytes
    }

    /// Runs `y = conv(x, w)` with `workspace`, which must hold at least
    /// [`Self::workspace_bytes`]
    pub fn forward(
        &self,
        workspace: &mut CudaViewMut<'_, u8>,
        x: &CudaView<'_, f32>,
        w: &CudaView<'_, f32>,
        y: &mut CudaViewMut<'_, f32>,
    ) -> Result<(), CudaError> {
        #[cfg(test)]
        let _library = super::test_support::call("cudnn.conv");
        self.check(workspace, x, w, y)?;
        let forward = ConvForward {
            conv: &self.conv,
            x: &self.x,
            w: &self.w,
            y: &self.y,
        };
        let mut workspace = workspace.slice_mut(..self.workspace_bytes);
        // SAFETY: the buffers hold exactly the FP32 NCHW shapes of the descriptors,
        // checked above, and the workspace is the size cuDNN asked for this algorithm
        unsafe {
            forward.launch(
                self.algo,
                (self.workspace_bytes > 0).then_some(&mut workspace),
                (1.0, 0.0),
                x,
                w,
                y,
            )
        }?;

        Ok(())
    }

    /// Runs `y = relu(conv(x, w) + residual + bias)` as one cuDNN call
    ///
    /// For the implicit-GEMM algorithms cuDNN applies the bias, residual and ReLU
    /// in the convolution's own epilogue, which saves a full read and write of the
    /// activation compared with a separate kernel
    pub fn forward_bias_relu(
        &self,
        workspace: &mut CudaViewMut<'_, u8>,
        x: &CudaView<'_, f32>,
        w: &CudaView<'_, f32>,
        bias: &CudaView<'_, f32>,
        residual: Residual<'_, '_>,
        y: &mut CudaViewMut<'_, f32>,
    ) -> Result<(), CudaError> {
        #[cfg(test)]
        let _library = super::test_support::call("cudnn.conv");
        self.check(workspace, x, w, y)?;
        check_len("conv2d bias", self.spec.out_channels, bias.len())?;
        let (z, scale) = match residual {
            Residual::None { scratch } => (scratch, 0.0),
            Residual::Add(z) => (z, 1.0),
        };
        check_len("conv2d residual", y.len(), z.len())?;

        let forward = ConvBiasActivationForward {
            conv: &self.conv,
            act: &self.relu,
            x: &self.x,
            w: &self.w,
            z: &self.y,
            bias: &self.bias,
            y: &self.y,
        };
        let mut workspace = workspace.slice_mut(..self.workspace_bytes);
        // SAFETY: `x`, `w`, `y` and `z` hold exactly the FP32 NCHW shapes of the
        // descriptors and `bias` one value per output channel, all checked above;
        // the workspace is the size cuDNN asked for this algorithm
        unsafe {
            forward.launch(
                self.algo,
                (self.workspace_bytes > 0).then_some(&mut workspace),
                (1.0, scale),
                x,
                w,
                z,
                bias,
                y,
            )
        }?;

        Ok(())
    }

    fn check(
        &self,
        workspace: &CudaViewMut<'_, u8>,
        x: &CudaView<'_, f32>,
        w: &CudaView<'_, f32>,
        y: &CudaViewMut<'_, f32>,
    ) -> Result<(), CudaError> {
        let context = "conv2d";
        let spec = &self.spec;
        check_len(
            "conv2d input",
            element_count(context, &spec.input_shape())?,
            x.len(),
        )?;
        check_len(
            "conv2d filter",
            element_count(context, &spec.filter_shape())?,
            w.len(),
        )?;
        check_len(
            "conv2d output",
            element_count(context, &spec.output_shape())?,
            y.len(),
        )?;
        if workspace.len() < self.workspace_bytes {
            return Err(CudaError::BufferLength {
                context: "conv2d workspace",
                expected: self.workspace_bytes,
                actual: workspace.len(),
            });
        }

        Ok(())
    }
}

/// Chooses cuDNN forward algorithms from cuDNN's own ranking, without timing runs,
/// so the same GPU, cuDNN version and shape always get the same algorithm and the
/// same numerics
///
/// The rule takes the first entry of `cudnnGetConvolutionForwardAlgorithm_v7` that
/// cuDNN reports as supported, is not FFT based, and runs in a math type
/// [`CudaMath`] allows. The FFT algorithms are excluded because their FP32 rounding
/// error is larger and their workspace reaches gigabytes at batch 32; on the RTX 5070
/// Ti they were also the slow picks (FP32 256 channels: FFT tiling 3.4 ms against
/// 1.2 ms for non-fused Winograd at batch 32)
///
/// cudarc keeps the raw descriptors of its safe types private, so the ranking is
/// queried with a short-lived handle and descriptors of its own
pub(crate) struct ConvPlanner<'a> {
    runtime: &'a CudaRuntime,
    handle: QueryHandle,
}

impl<'a> ConvPlanner<'a> {
    /// A planner for convolutions on `runtime`'s cuDNN handle
    pub fn new(runtime: &'a CudaRuntime) -> Result<Self, CudaError> {
        // the query handle belongs to the context current on this thread
        runtime.context().bind_to_thread()?;
        Ok(Self {
            runtime,
            handle: QueryHandle(result::create_handle()?),
        })
    }

    /// Plans `spec` with the chosen algorithm
    pub fn plan(&self, spec: Conv2d) -> Result<ConvPlan, CudaError> {
        let ranked = self.rank(&spec)?;
        let (algo, math_type) = choose(&ranked, spec.math).unwrap_or((
            // implicit GEMM supports every FP32 NCHW convolution without workspace
            cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
            cudnnMathType_t::CUDNN_FMA_MATH,
        ));

        let dnn = self.runtime.dnn();
        let mut conv = dnn.create_conv2d::<f32>(
            pair(spec.padding)?,
            pair(spec.stride)?,
            pair(spec.dilation)?,
            cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
        )?;
        // cuDNN expects the descriptor to carry the math type of the chosen entry
        conv.set_math_type(math_type)?;

        let nchw = cudnnTensorFormat_t::CUDNN_TENSOR_NCHW;
        let x = dnn.create_4d_tensor::<f32>(nchw, dims(spec.input_shape())?)?;
        let w = dnn.create_4d_filter::<f32>(nchw, dims(spec.filter_shape())?)?;
        let y = dnn.create_4d_tensor::<f32>(nchw, dims(spec.output_shape())?)?;
        let channels = to_c_int("conv2d", spec.out_channels)?;
        let bias = dnn.create_4d_tensor::<f32>(nchw, [1, channels, 1, 1])?;
        // NaN propagates, matching the ONNX graphs' Relu
        let relu = dnn.create_activation::<f32>(
            cudnnActivationMode_t::CUDNN_ACTIVATION_RELU,
            cudnnNanPropagation_t::CUDNN_PROPAGATE_NAN,
            0.0,
        )?;
        let workspace_bytes = ConvForward {
            conv: &conv,
            x: &x,
            w: &w,
            y: &y,
        }
        .get_workspace_size(algo)?;

        debug!(
            input = ?spec.input_shape(),
            filter = ?spec.filter_shape(),
            stride = ?spec.stride,
            math = ?spec.math,
            ranked = ?ranked
                .iter()
                .map(|entry| (entry.algo as u32, entry.status as u32, entry.mathType as u32))
                .collect::<Vec<_>>(),
            algo = algo as u32,
            ?math_type,
            workspace_bytes,
            "Planned CUDA convolution"
        );

        Ok(ConvPlan {
            spec,
            conv,
            x,
            w,
            y,
            bias,
            relu,
            algo,
            workspace_bytes,
        })
    }

    /// cuDNN's heuristic ranking for `spec`, best first
    fn rank(&self, spec: &Conv2d) -> Result<Vec<cudnnConvolutionFwdAlgoPerf_t>, CudaError> {
        let descriptors = QueryDescriptors::new(spec)?;
        let mut returned: c_int = 0;
        // SAFETY: zeroed perf structs are valid plain data that cuDNN overwrites
        let mut perf: [cudnnConvolutionFwdAlgoPerf_t; ALGORITHM_COUNT] =
            unsafe { std::mem::zeroed() };
        // SAFETY: the handle and descriptors are live, fully set, and `perf` holds
        // the requested number of entries
        unsafe {
            result::get_convolution_forward_algorithm(
                self.handle.0,
                descriptors.x,
                descriptors.w,
                descriptors.conv,
                descriptors.y,
                ALGORITHM_COUNT as c_int,
                &mut returned,
                perf.as_mut_ptr(),
            )
        }?;

        let returned = usize::try_from(returned).unwrap_or(0).min(ALGORITHM_COUNT);
        Ok(perf[..returned].to_vec())
    }
}

/// The first ranked entry the selection rule accepts, with the math type to run it in
fn choose(
    ranked: &[cudnnConvolutionFwdAlgoPerf_t],
    math: CudaMath,
) -> Option<(cudnnConvolutionFwdAlgo_t, cudnnMathType_t)> {
    ranked
        .iter()
        .filter(|entry| entry.status == cudnnStatus_t::CUDNN_STATUS_SUCCESS)
        .filter(|entry| {
            !matches!(
                entry.algo,
                cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_FFT
                    | cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING
            )
        })
        .find(|entry| allows(math, entry.mathType))
        .map(|entry| (entry.algo, entry.mathType))
}

/// Whether an algorithm entry's math type stays within `math`
fn allows(math: CudaMath, math_type: cudnnMathType_t) -> bool {
    match math {
        // the default math type permits TF32 on Ampere and newer, so FP32 only
        // accepts entries that run plain FMA
        CudaMath::Fp32 => math_type == cudnnMathType_t::CUDNN_FMA_MATH,
        // `_ALLOW_CONVERSION` may down-convert to FP16, which is past TF32
        CudaMath::Tf32 => matches!(
            math_type,
            cudnnMathType_t::CUDNN_FMA_MATH | cudnnMathType_t::CUDNN_TENSOR_OP_MATH
        ),
    }
}

/// A cuDNN handle used only for heuristic queries
struct QueryHandle(cudnnHandle_t);

impl Drop for QueryHandle {
    fn drop(&mut self) {
        // SAFETY: the handle came from `create_handle` and is destroyed once
        let _ = unsafe { result::destroy_handle(self.0) };
    }
}

/// Raw descriptors for one heuristic query, destroyed on drop
struct QueryDescriptors {
    x: cudnnTensorDescriptor_t,
    w: cudnnFilterDescriptor_t,
    conv: cudnnConvolutionDescriptor_t,
    y: cudnnTensorDescriptor_t,
}

impl QueryDescriptors {
    fn new(spec: &Conv2d) -> Result<Self, CudaError> {
        let mut descriptors = Self {
            x: std::ptr::null_mut(),
            w: std::ptr::null_mut(),
            conv: std::ptr::null_mut(),
            y: std::ptr::null_mut(),
        };
        // created one at a time into `descriptors`, so a failure part way still
        // destroys the ones that exist
        descriptors.x = result::create_tensor_descriptor()?;
        descriptors.y = result::create_tensor_descriptor()?;
        descriptors.w = result::create_filter_descriptor()?;
        descriptors.conv = result::create_convolution_descriptor()?;

        let format = cudnnTensorFormat_t::CUDNN_TENSOR_NCHW;
        let float = cudnnDataType_t::CUDNN_DATA_FLOAT;
        let [padding, stride, dilation] = [
            pair(spec.padding)?,
            pair(spec.stride)?,
            pair(spec.dilation)?,
        ];
        let x = dims(spec.input_shape())?;
        let y = dims(spec.output_shape())?;
        let w = dims(spec.filter_shape())?;
        // SAFETY: every descriptor was just created and is set once with valid
        // dimensions
        unsafe {
            result::set_tensor4d_descriptor(descriptors.x, format, float, x)?;
            result::set_tensor4d_descriptor(descriptors.y, format, float, y)?;
            result::set_filter4d_descriptor(descriptors.w, float, format, w)?;
            result::set_convolution2d_descriptor(
                descriptors.conv,
                padding[0],
                padding[1],
                stride[0],
                stride[1],
                dilation[0],
                dilation[1],
                cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
                float,
            )?;
            result::set_convolution_math_type(descriptors.conv, spec.math.cudnn())?;
        }

        Ok(descriptors)
    }
}

impl Drop for QueryDescriptors {
    fn drop(&mut self) {
        // SAFETY: each non-null descriptor was created by this struct and is
        // destroyed exactly once
        unsafe {
            if !self.x.is_null() {
                let _ = result::destroy_tensor_descriptor(self.x);
            }
            if !self.y.is_null() {
                let _ = result::destroy_tensor_descriptor(self.y);
            }
            if !self.w.is_null() {
                let _ = result::destroy_filter_descriptor(self.w);
            }
            if !self.conv.is_null() {
                let _ = result::destroy_convolution_descriptor(self.conv);
            }
        }
    }
}

fn dims(shape: [usize; 4]) -> Result<[c_int; 4], CudaError> {
    let [a, b, c, d] = shape;
    Ok([
        to_c_int("conv2d", a)?,
        to_c_int("conv2d", b)?,
        to_c_int("conv2d", c)?,
        to_c_int("conv2d", d)?,
    ])
}

fn pair(values: [usize; 2]) -> Result<[c_int; 2], CudaError> {
    Ok([
        to_c_int("conv2d", values[0])?,
        to_c_int("conv2d", values[1])?,
    ])
}
