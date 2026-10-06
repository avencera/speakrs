//! Native CUDA PyanNet segmentation: what `segmentation-3.0.onnx` and
//! `segmentation-3.0-b32.onnx` compute, without ONNX Runtime
//!
//! - waveform instance normalization, then SincNet: 80 generated band-pass filters
//!   (selected convolution, stride 10), `abs`, max pool, instance normalization and
//!   LeakyReLU, then two learned convolutions (cuDNN), each followed by max pool,
//!   instance normalization and LeakyReLU (`segmentation_pool_norm`)
//! - four bidirectional LSTM layers through one selected Oxide or Library stack
//! - three linear layers in cuBLAS with LeakyReLU, then log-softmax over the 7
//!   powerset classes
//!
//! [`CudaSegmentation`] holds the weights and one [`SegmentationWorkspace`] per batch
//! shape it has run, so buffers, plans and captured graphs are allocated once per
//! batch class and reused

mod dispatch;
mod graph;
mod kernels;
#[cfg(test)]
pub(super) use kernels::REQUIRED_KERNELS;
#[cfg(feature = "cuda")]
mod rnn;
mod shape;
mod weights;

use cudarc::driver::CudaSlice;

use self::graph::CapturedGraph;
use self::kernels::{PoolNorm, RowLayout, SegmentationKernels};
#[cfg(feature = "cuda")]
use self::rnn::{CudnnLstm, LstmPlan};
use self::shape::{
    CONV_KERNEL, FEATURES, HIDDEN, LEAKY_SLOPE, LINEAR, NORM_EPSILON, POOL, SINC_CHANNELS,
};
#[cfg(feature = "cuda")]
use self::shape::{SINC_KERNEL, SINC_STRIDE};
use self::weights::{LstmLayer, SegmentationWeights};
#[cfg(feature = "cuda")]
use super::CudaLstmAlgorithm;
use super::candidate::{LstmOxide, SincCandidate, SincOutput, SincOxide};
use super::dnn::Conv2d;
#[cfg(feature = "cuda")]
use super::dnn::{ConvPlan, ConvPlanner};
use super::implementation::{AreaTarget, LibraryNeed, MODEL_BATCHES, Selected, plan_selection};
use super::{CudaError, CudaMath, CudaRuntime, DeviceTensor, PtxTier, SafetensorsFile, Sgemm};
use super::{CudaLibrary, KernelModule};

use self::shape::CLASSES;
pub use self::shape::SegmentationShape;

/// How the segmentation forward pass runs
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SegmentationOptions {
    /// Precision of every cuBLAS, cuDNN convolution and cuDNN RNN call
    pub math: CudaMath,
    /// cuDNN RNN algorithm for the LSTM stack
    #[cfg(feature = "cuda")]
    pub lstm_algo: CudaLstmAlgorithm,
    /// Capture the forward pass of each workspace as a CUDA graph on its first run and
    /// replay it afterwards
    pub cuda_graph: bool,
}

/// The segmentation model on one [`CudaRuntime`]: device weights, kernels and a
/// workspace per batch shape
///
/// Use it only with the runtime it was created on
#[derive(Debug)]
pub struct CudaSegmentation {
    workspaces: Vec<SegmentationWorkspace>,
    network: Network,
}

#[cfg(all(test, feature = "cuda", not(feature = "cuda-driver-only")))]
pub use test_support::SegmentationTensor;

/// Buffers, convolution plans, LSTM plan and captured graph for one batch shape
#[derive(Debug)]
pub struct SegmentationWorkspace {
    shape: SegmentationShape,
    sinc: SincPlan,
    conv1: ConvStage,
    conv2: ConvStage,
    /// cuDNN workspace shared by the three convolutions, sized for the largest
    conv_workspace: CudaSlice<u8>,
    lstm: LstmStage,
    tensors: Tensors,
    graph: Option<CapturedGraph>,
}

/// One owner for the Sinc producer
#[derive(Debug)]
enum SincPlan {
    #[cfg(feature = "cuda")]
    Library(ConvPlan),
    Oxide(SincOxide),
    #[cfg(all(test, feature = "cuda", not(feature = "cuda-driver-only")))]
    Mutant {
        library: ConvPlan,
        mutant: super::test_support::Mutant,
    },
}

impl SincPlan {
    fn workspace_bytes(&self) -> usize {
        match self {
            #[cfg(feature = "cuda")]
            Self::Library(plan) => plan.workspace_bytes(),
            Self::Oxide(_) => 0,
            #[cfg(all(test, feature = "cuda", not(feature = "cuda-driver-only")))]
            Self::Mutant { library, .. } => library.workspace_bytes(),
        }
    }
}

/// Learned convolutions have no qualified Oxide implementation yet
#[derive(Debug)]
enum ConvStage {
    #[cfg(feature = "cuda")]
    Library(ConvPlan),
}

impl ConvStage {
    fn new(runtime: &CudaRuntime, boundary: &str, spec: Conv2d) -> Result<Self, CudaError> {
        LibraryNeed::new(
            KernelModule::Segmentation,
            boundary,
            spec.batch,
            spec.math,
            AreaTarget::for_area(runtime, KernelModule::Segmentation)?,
            CudaLibrary::Cudnn,
        )
        .prepare(runtime)?;
        #[cfg(feature = "cuda")]
        return Ok(Self::Library(ConvPlanner::new(runtime)?.plan(spec)?));
        #[cfg(not(feature = "cuda"))]
        Err(CudaError::LibraryUnavailable {
            library: CudaLibrary::Cudnn,
        })
    }

    fn workspace_bytes(&self) -> usize {
        match *self {
            #[cfg(feature = "cuda")]
            Self::Library(ref plan) => plan.workspace_bytes(),
        }
    }

    fn forward(
        &self,
        workspace: &mut cudarc::driver::CudaViewMut<'_, u8>,
        input: &cudarc::driver::CudaView<'_, f32>,
        weight: &cudarc::driver::CudaView<'_, f32>,
        output: &mut cudarc::driver::CudaViewMut<'_, f32>,
    ) -> Result<(), CudaError> {
        #[cfg(not(feature = "cuda"))]
        let _ = (workspace, input, weight, output);
        match *self {
            #[cfg(feature = "cuda")]
            Self::Library(ref plan) => plan.forward(workspace, input, weight, output),
        }
    }
}

/// One owner for the complete LSTM stack
#[derive(Debug)]
enum LstmStage {
    #[cfg(feature = "cuda")]
    Library(LstmPlan),
    Oxide {
        candidate: Box<LstmOxide>,
        rows: usize,
    },
    #[cfg(all(test, feature = "cuda", not(feature = "cuda-driver-only")))]
    Mutant {
        library: LstmPlan,
        mutant: super::test_support::Mutant,
    },
}

#[derive(Debug)]
struct Tensors {
    input: DeviceTensor,
    wave_norm: DeviceTensor,
    sinc: DeviceTensor,
    /// `[batch, 80, pool0]` written by a Sinc candidate that pools, allocated only for one
    sinc_pooled: Option<DeviceTensor>,
    stage0: DeviceTensor,
    conv1: DeviceTensor,
    stage1: DeviceTensor,
    conv2: DeviceTensor,
    lstm_input: DeviceTensor,
    lstm_output: DeviceTensor,
    linear0: DeviceTensor,
    linear1: DeviceTensor,
    output: DeviceTensor,
}

/// The weights and kernels; separate from the workspaces so a forward pass can
/// borrow both
#[derive(Debug)]
struct Network {
    options: SegmentationOptions,
    kernels: SegmentationKernels,
    #[cfg(feature = "cuda")]
    lstm: Option<CudnnLstm>,
    /// Waveform normalization scale and shift, one value each
    wav_norm: [CudaSlice<f32>; 2],
    /// `[80, 1, 1, 251]`
    sinc_filters: CudaSlice<f32>,
    /// Learned convolution weights `[60, c, 1, 5]` and biases `[60]`
    convs: [[CudaSlice<f32>; 2]; 2],
    /// Scale and shift of the three SincNet normalizations
    norms: [[CudaSlice<f32>; 2]; 3],
    /// `MatMul` weights `[in, out]` and biases of the three linear layers
    linear: [[CudaSlice<f32>; 2]; 3],
    /// Zero bias for the stages without a convolution bias
    zero_bias: CudaSlice<f32>,
    /// Host LSTM weights for candidate plans
    lstm_weights: Vec<LstmLayer>,
}

impl CudaSegmentation {
    /// Powerset classes per output frame (3 speakers, at most 2 active)
    pub const CLASSES: usize = CLASSES;

    /// Uploads the weights of `segmentation-3.0` (or its `-b32` export) from a
    /// safetensors file keyed by ONNX initializer names
    pub fn new(
        runtime: &CudaRuntime,
        weights: &SafetensorsFile,
        options: SegmentationOptions,
    ) -> Result<Self, CudaError> {
        Ok(Self {
            network: Network::new(runtime, &SegmentationWeights::load(weights)?, options)?,
            workspaces: Vec::new(),
        })
    }

    /// The PTX tier the segmentation kernels were loaded from
    pub fn kernel_tier(&self) -> PtxTier {
        self.network.kernels.tier()
    }

    /// The workspace for `batch` windows of `samples` samples, allocated on first use
    pub fn workspace(
        &mut self,
        runtime: &CudaRuntime,
        batch: usize,
        samples: usize,
    ) -> Result<&mut SegmentationWorkspace, CudaError> {
        let index = self.workspace_index(runtime, batch, samples)?;
        Ok(&mut self.workspaces[index])
    }

    /// Uploads `batch` windows (`[batch, samples]`, `samples = input.len() / batch`),
    /// runs the model and returns the `[batch, frames, 7]` powerset log-probabilities
    pub fn run(
        &mut self,
        runtime: &CudaRuntime,
        batch: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, CudaError> {
        let samples = input.len().checked_div(batch).unwrap_or(0);
        if samples * batch != input.len() {
            return Err(CudaError::BufferLength {
                context: "segmentation input (batch * samples)",
                expected: samples * batch,
                actual: input.len(),
            });
        }

        let index = self.workspace_index(runtime, batch, samples)?;
        let workspace = &mut self.workspaces[index];
        workspace.upload_input(runtime, input)?;
        self.network.forward(runtime, workspace)?;
        workspace.download_output(runtime)
    }

    fn workspace_index(
        &mut self,
        runtime: &CudaRuntime,
        batch: usize,
        samples: usize,
    ) -> Result<usize, CudaError> {
        let existing = self
            .workspaces
            .iter()
            .position(|workspace| workspace.matches(batch, samples));
        if let Some(index) = existing {
            return Ok(index);
        }

        let shape = SegmentationShape::new(batch, samples)?;
        self.workspaces
            .push(self.network.workspace(runtime, shape)?);
        Ok(self.workspaces.len() - 1)
    }
}

impl Network {
    fn new(
        runtime: &CudaRuntime,
        weights: &SegmentationWeights,
        options: SegmentationOptions,
    ) -> Result<Self, CudaError> {
        let mut needs = Vec::new();
        for batch in MODEL_BATCHES {
            for (area, boundary) in [
                (KernelModule::Sincnet, dispatch::SINC_LAYER),
                (KernelModule::Segmentation, "sincnet.conv1"),
                (KernelModule::Segmentation, "sincnet.conv2"),
                (KernelModule::Lstm, dispatch::LSTM_LAYER),
            ] {
                let target = AreaTarget::for_area(runtime, area)?;
                let selected = plan_selection(
                    runtime,
                    area,
                    boundary,
                    batch,
                    options.math,
                    #[cfg(all(test, feature = "cuda", not(feature = "cuda-driver-only")))]
                    None,
                )?;
                if matches!(selected, Selected::Library) {
                    needs.push(LibraryNeed::new(
                        area,
                        boundary,
                        batch,
                        options.math,
                        target,
                        CudaLibrary::Cudnn,
                    ));
                    #[cfg(feature = "cuda")]
                    if boundary == dispatch::LSTM_LAYER
                        && options.lstm_algo == CudaLstmAlgorithm::PersistDynamic
                    {
                        needs.push(LibraryNeed::new(
                            area,
                            boundary,
                            batch,
                            options.math,
                            target,
                            CudaLibrary::Nvrtc,
                        ));
                    }
                } else if boundary == dispatch::LSTM_LAYER {
                    needs.push(LibraryNeed::new(
                        area,
                        "lstm.stack.input_proj",
                        batch,
                        options.math,
                        target,
                        CudaLibrary::Cublas,
                    ));
                }
            }
            for boundary in ["linear0", "linear1", "linear2"] {
                needs.push(LibraryNeed::new(
                    KernelModule::Segmentation,
                    boundary,
                    batch,
                    options.math,
                    AreaTarget::for_area(runtime, KernelModule::Segmentation)?,
                    CudaLibrary::Cublas,
                ));
            }
        }
        if super::driver_only() {
            for need in &needs {
                need.prepare(runtime)?;
            }
        }
        LibraryNeed::new(
            KernelModule::Segmentation,
            "linear0",
            1,
            options.math,
            AreaTarget::for_area(runtime, KernelModule::Segmentation)?,
            CudaLibrary::Cublas,
        )
        .prepare(runtime)?;
        // selected handles precede model buffers because their allocations affect recurrence locality
        for need in &needs {
            need.prepare_handle(runtime)?;
        }
        let stream = runtime.stream();
        let upload = |values: &[f32]| stream.clone_htod(values);
        let pair = |first: &[f32], second: &[f32]| -> Result<[CudaSlice<f32>; 2], CudaError> {
            Ok([upload(first)?, upload(second)?])
        };

        let [norm0, norm1, norm2] = &weights.norms;
        let [linear0, linear1, linear2] = &weights.linear;
        Ok(Self {
            options,
            kernels: SegmentationKernels::load(runtime)?,
            #[cfg(feature = "cuda")]
            lstm: None,
            wav_norm: pair(&weights.wav_norm.gamma, &weights.wav_norm.beta)?,
            sinc_filters: upload(&weights.sinc_filters)?,
            convs: [
                pair(&weights.conv1.weight, &weights.conv1.bias)?,
                pair(&weights.conv2.weight, &weights.conv2.bias)?,
            ],
            norms: [
                pair(&norm0.gamma, &norm0.beta)?,
                pair(&norm1.gamma, &norm1.beta)?,
                pair(&norm2.gamma, &norm2.beta)?,
            ],
            linear: [
                pair(&linear0.weight, &linear0.bias)?,
                pair(&linear1.weight, &linear1.bias)?,
                pair(&linear2.weight, &linear2.bias)?,
            ],
            zero_bias: stream.alloc_zeros(SINC_CHANNELS)?,
            lstm_weights: weights.lstm.to_vec(),
        })
    }

    /// Allocates the buffers and plans for one batch shape
    fn workspace(
        &mut self,
        runtime: &CudaRuntime,
        shape: SegmentationShape,
    ) -> Result<SegmentationWorkspace, CudaError> {
        let batch = shape.batch;
        let math = self.options.math;
        let conv = |[in_channels, out_channels]: [usize; 2], input, kernel, stride| Conv2d {
            batch,
            in_channels,
            out_channels,
            input: [1, input],
            kernel: [1, kernel],
            padding: [0, 0],
            stride: [1, stride],
            dilation: [1, 1],
            math,
        };
        let stream = runtime.stream();
        let tensor = |dims: &[usize]| DeviceTensor::zeros(stream, dims);
        let tensors = Tensors {
            input: tensor(&[batch, 1, shape.samples])?,
            wave_norm: tensor(&[batch, 1, shape.samples])?,
            sinc: tensor(&[batch, SINC_CHANNELS, shape.sinc])?,
            sinc_pooled: None,
            stage0: tensor(&[batch, SINC_CHANNELS, shape.pool0])?,
            conv1: tensor(&[batch, FEATURES, shape.conv1])?,
            stage1: tensor(&[batch, FEATURES, shape.pool1])?,
            conv2: tensor(&[batch, FEATURES, shape.conv2])?,
            lstm_input: tensor(&[batch, shape.frames, FEATURES])?,
            lstm_output: tensor(&[batch, shape.frames, 2 * HIDDEN])?,
            linear0: tensor(&[batch, shape.frames, LINEAR[0][1]])?,
            linear1: tensor(&[batch, shape.frames, LINEAR[1][1]])?,
            output: tensor(&[batch, shape.frames, CLASSES])?,
        };

        let sinc_selected = plan_selection(
            runtime,
            KernelModule::Sincnet,
            dispatch::SINC_LAYER,
            batch,
            math,
            #[cfg(all(test, feature = "cuda", not(feature = "cuda-driver-only")))]
            None,
        )?;
        let lstm_selected = plan_selection(
            runtime,
            KernelModule::Lstm,
            dispatch::LSTM_LAYER,
            batch,
            math,
            #[cfg(all(test, feature = "cuda", not(feature = "cuda-driver-only")))]
            None,
        )?;
        let sinc = self.plan_sinc(runtime, shape, sinc_selected)?;
        let conv1 = ConvStage::new(
            runtime,
            "sincnet.conv1",
            conv([SINC_CHANNELS, FEATURES], shape.pool0, CONV_KERNEL, 1),
        )?;
        let conv2 = ConvStage::new(
            runtime,
            "sincnet.conv2",
            conv([FEATURES, FEATURES], shape.pool1, CONV_KERNEL, 1),
        )?;
        let workspace_bytes = sinc
            .workspace_bytes()
            .max(conv1.workspace_bytes())
            .max(conv2.workspace_bytes());
        let mut tensors = tensors;
        if matches!(sinc, SincPlan::Oxide(_)) {
            tensors.sinc_pooled = pooled_tensor(runtime, shape)?;
        }
        // keep convolution scratch before recurrence state; allocation order affects replay latency
        let conv_workspace = stream.alloc_zeros(workspace_bytes.max(1))?;
        let lstm = self.plan_lstm(runtime, shape, lstm_selected)?;
        Ok(SegmentationWorkspace {
            shape,
            sinc,
            conv1,
            conv2,
            conv_workspace,
            lstm,
            tensors,
            graph: None,
        })
    }

    fn forward(
        &self,
        runtime: &CudaRuntime,
        workspace: &mut SegmentationWorkspace,
    ) -> Result<(), CudaError> {
        if !self.options.cuda_graph {
            return self.enqueue(runtime, workspace);
        }

        if workspace.graph.is_none() {
            let graph = CapturedGraph::capture(runtime, || self.enqueue(runtime, workspace))?;
            workspace.graph = Some(graph);
        }

        match &workspace.graph {
            Some(graph) => graph.launch(),
            None => Ok(()),
        }
    }

    /// Queues every layer of one forward pass on the runtime's stream
    fn enqueue(
        &self,
        runtime: &CudaRuntime,
        workspace: &mut SegmentationWorkspace,
    ) -> Result<(), CudaError> {
        let shape = workspace.shape;
        let batch = shape.batch;
        let SegmentationWorkspace {
            sinc,
            conv1,
            conv2,
            conv_workspace,
            lstm,
            tensors: t,
            ..
        } = workspace;
        let mut conv_workspace = conv_workspace.as_view_mut();
        let norm = |channels: usize, in_len: usize, pool: usize| PoolNorm {
            batch,
            channels,
            in_len,
            pool,
            abs_input: false,
            slope: LEAKY_SLOPE,
            epsilon: NORM_EPSILON,
            layout: RowLayout::channels_first(channels, in_len / pool),
        };
        let k = &self.kernels;

        // waveform instance normalization: no pooling and no activation
        let wave = PoolNorm {
            slope: 1.0,
            ..norm(1, shape.samples, 1)
        };
        let [gamma, beta] = &self.wav_norm;
        let zero_bias = &self.zero_bias;
        k.pool_norm(
            runtime,
            wave,
            t.input.data(),
            zero_bias,
            gamma,
            beta,
            t.wave_norm.data_mut(),
        )?;

        self.sinc_forward(
            runtime,
            shape,
            sinc,
            &mut conv_workspace,
            dispatch::SincIo {
                input: t.wave_norm.data(),
                raw: t.sinc.data_mut(),
                pooled: t.sinc_pooled.as_mut().map(DeviceTensor::data_mut),
                stage0: t.stage0.data_mut(),
            },
        )?;

        // the convolution bias is added inside the pooling kernel
        let [weight, bias] = &self.convs[0];
        conv1.forward(
            &mut conv_workspace,
            &t.stage0.data().as_view(),
            &weight.as_view(),
            &mut t.conv1.data_mut().as_view_mut(),
        )?;
        let stage1 = norm(FEATURES, shape.conv1, POOL);
        let [gamma, beta] = &self.norms[1];
        k.pool_norm(
            runtime,
            stage1,
            t.conv1.data(),
            bias,
            gamma,
            beta,
            t.stage1.data_mut(),
        )?;

        let [weight, bias] = &self.convs[1];
        conv2.forward(
            &mut conv_workspace,
            &t.stage1.data().as_view(),
            &weight.as_view(),
            &mut t.conv2.data_mut().as_view_mut(),
        )?;
        // the last stage writes the batch-major `[batch, frames, 60]` LSTM input
        let stage2 = PoolNorm {
            layout: RowLayout::time_major_rows(FEATURES, shape.frames),
            ..norm(FEATURES, shape.conv2, POOL)
        };
        let [gamma, beta] = &self.norms[2];
        k.pool_norm(
            runtime,
            stage2,
            t.conv2.data(),
            bias,
            gamma,
            beta,
            t.lstm_input.data_mut(),
        )?;

        self.lstm_forward(runtime, lstm, t.lstm_input.data(), t.lstm_output.data_mut())?;

        let rows = batch * shape.frames;
        let gemm = |index: usize| Sgemm {
            math: self.options.math,
            ..Sgemm::new(rows, LINEAR[index][1], LINEAR[index][0])
        };
        let [weight, bias] = &self.linear[0];
        runtime.sgemm(gemm(0), t.lstm_output.data(), weight, t.linear0.data_mut())?;
        k.bias_leaky(runtime, bias, LEAKY_SLOPE, t.linear0.data_mut())?;

        let [weight, bias] = &self.linear[1];
        runtime.sgemm(gemm(1), t.linear0.data(), weight, t.linear1.data_mut())?;
        k.bias_leaky(runtime, bias, LEAKY_SLOPE, t.linear1.data_mut())?;

        let [weight, bias] = &self.linear[2];
        runtime.sgemm(gemm(2), t.linear1.data(), weight, t.output.data_mut())?;
        k.bias_log_softmax(runtime, bias, t.output.data_mut())
    }
}

impl SegmentationWorkspace {
    fn matches(&self, batch: usize, samples: usize) -> bool {
        self.shape.batch == batch && self.shape.samples == samples
    }

    /// Copies `[batch, samples]` waveforms into the input buffer
    pub fn upload_input(&mut self, runtime: &CudaRuntime, input: &[f32]) -> Result<(), CudaError> {
        self.tensors.input.copy_from_host(runtime.stream(), input)
    }

    /// Copies the `[batch, frames, 7]` log-probabilities back; waits for the stream
    pub fn download_output(&self, runtime: &CudaRuntime) -> Result<Vec<f32>, CudaError> {
        self.tensors.output.download(runtime.stream())
    }
}

#[cfg(all(test, feature = "cuda", not(feature = "cuda-driver-only")))]
#[path = "../../../tests/cuda_qualify/segmentation.rs"]
pub(crate) mod test_support;

/// The pooled Sinc tensor, for a candidate whose declared output is pooled
fn pooled_tensor(
    runtime: &CudaRuntime,
    shape: SegmentationShape,
) -> Result<Option<DeviceTensor>, CudaError> {
    if SincOxide::OUTPUT != SincOutput::Pooled {
        return Ok(None);
    }

    let dims = [shape.batch, SINC_CHANNELS, shape.pool0];
    Ok(Some(DeviceTensor::zeros(runtime.stream(), &dims)?))
}
