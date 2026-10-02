//! Native CUDA PyanNet segmentation: what `segmentation-3.0.onnx` and
//! `segmentation-3.0-b32.onnx` compute, without ONNX Runtime
//!
//! - waveform instance normalization, then SincNet: 80 generated band-pass filters
//!   (cuDNN convolution, stride 10), `abs`, max pool, instance normalization and
//!   LeakyReLU, then two learned convolutions (cuDNN), each followed by max pool,
//!   instance normalization and LeakyReLU (`segmentation_pool_norm`)
//! - four bidirectional LSTM layers as one cuDNN RNN ([`CudaLstmAlgorithm`])
//! - three linear layers in cuBLAS with LeakyReLU, then log-softmax over the 7
//!   powerset classes
//!
//! [`CudaSegmentation`] holds the weights and one [`SegmentationWorkspace`] per batch
//! shape it has run, so buffers, plans and captured graphs are allocated once per
//! batch class and reused

mod graph;
mod kernels;
mod rnn;
mod shape;
mod weights;

use cudarc::driver::CudaSlice;

use self::graph::CapturedGraph;
use self::kernels::{PoolNorm, RowLayout, SegmentationKernels};
use self::rnn::{CudnnLstm, LstmPlan};
use self::shape::{
    CONV_KERNEL, FEATURES, HIDDEN, LEAKY_SLOPE, LINEAR, NORM_EPSILON, POOL, SINC_CHANNELS,
    SINC_KERNEL, SINC_STRIDE,
};
use self::weights::SegmentationWeights;
use super::dnn::{Conv2d, ConvPlan, ConvPlanner};
use super::{
    CudaError, CudaLstmAlgorithm, CudaMath, CudaRuntime, DeviceTensor, PtxTier, SafetensorsFile,
    Sgemm,
};

use self::shape::CLASSES;
pub use self::shape::SegmentationShape;
#[cfg(test)]
use self::shape::WINDOW_SAMPLES;

/// How the segmentation forward pass runs
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SegmentationOptions {
    /// Precision of every cuBLAS, cuDNN convolution and cuDNN RNN call
    pub math: CudaMath,
    /// cuDNN RNN algorithm for the LSTM stack
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
    network: Network,
    workspaces: Vec<SegmentationWorkspace>,
}

#[cfg(test)]
/// A tensor of one forward pass, kept in its [`SegmentationWorkspace`]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SegmentationTensor {
    /// `[batch, 1, samples]` waveforms
    Input,
    /// `[batch, 1, samples]` after the waveform instance normalization
    WaveNorm,
    /// `[batch, 80, sinc]` SincNet convolution output, before `abs`
    SincConv,
    /// `[batch, 80, pool0]` after pool, normalization and LeakyReLU
    Stage0,
    /// `[batch, 60, conv1]` first learned convolution, without its bias
    Conv1,
    /// `[batch, 60, pool1]` after bias, pool, normalization and LeakyReLU
    Stage1,
    /// `[batch, 60, conv2]` second learned convolution, without its bias
    Conv2,
    /// `[batch, frames, 60]` LSTM input
    LstmInput,
    /// `[batch, frames, 256]` output of the last LSTM layer
    LstmOutput,
    /// `[batch, frames, 128]` after the first linear layer and LeakyReLU
    Linear0,
    /// `[batch, frames, 128]` after the second linear layer and LeakyReLU
    Linear1,
    /// `[batch, frames, 7]` powerset log-probabilities
    Output,
}

#[cfg(test)]
impl SegmentationTensor {
    /// Every tensor, in forward order
    pub const ALL: [Self; 12] = [
        Self::Input,
        Self::WaveNorm,
        Self::SincConv,
        Self::Stage0,
        Self::Conv1,
        Self::Stage1,
        Self::Conv2,
        Self::LstmInput,
        Self::LstmOutput,
        Self::Linear0,
        Self::Linear1,
        Self::Output,
    ];
}

/// Buffers, convolution plans, LSTM plan and captured graph for one batch shape
#[derive(Debug)]
pub struct SegmentationWorkspace {
    shape: SegmentationShape,
    sinc: ConvPlan,
    conv1: ConvPlan,
    conv2: ConvPlan,
    /// cuDNN workspace shared by the three convolutions, sized for the largest
    conv_workspace: CudaSlice<u8>,
    lstm: LstmPlan,
    tensors: Tensors,
    graph: Option<CapturedGraph>,
}

#[derive(Debug)]
struct Tensors {
    input: DeviceTensor,
    wave_norm: DeviceTensor,
    sinc: DeviceTensor,
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
    lstm: CudnnLstm,
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
}

impl CudaSegmentation {
    #[cfg(test)]
    /// Samples in the 10 s, 16 kHz window speakrs segments
    pub const WINDOW_SAMPLES: usize = WINDOW_SAMPLES;
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

    #[cfg(test)]
    /// The generated `[80, 1, 251]` SincNet filters, for parity checks
    pub fn sinc_filters(&self) -> &CudaSlice<f32> {
        &self.network.sinc_filters
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

    #[cfg(test)]
    /// The workspace for `batch` windows of `samples` samples, if one was allocated;
    /// its tensors hold the last forward pass with that shape
    pub fn find_workspace(&self, batch: usize, samples: usize) -> Option<&SegmentationWorkspace> {
        self.workspaces
            .iter()
            .find(|workspace| workspace.matches(batch, samples))
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

    #[cfg(test)]
    /// Runs the model on the input already uploaded to the `(batch, samples)`
    /// workspace and leaves the result on the device
    ///
    /// With [`SegmentationOptions::cuda_graph`] the first call per workspace captures
    /// a graph and later calls replay it
    pub fn forward(
        &mut self,
        runtime: &CudaRuntime,
        batch: usize,
        samples: usize,
    ) -> Result<(), CudaError> {
        let index = self.workspace_index(runtime, batch, samples)?;
        self.network.forward(runtime, &mut self.workspaces[index])
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
            lstm: CudnnLstm::new(runtime, &weights.lstm, options.math, options.lstm_algo)?,
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
        })
    }

    /// Allocates the buffers and plans for one batch shape
    fn workspace(
        &self,
        runtime: &CudaRuntime,
        shape: SegmentationShape,
    ) -> Result<SegmentationWorkspace, CudaError> {
        let batch = shape.batch;
        let math = self.options.math;
        let planner = ConvPlanner::new(runtime)?;
        let conv = |[in_channels, out_channels]: [usize; 2], input, kernel, stride| {
            planner.plan(Conv2d {
                batch,
                in_channels,
                out_channels,
                input: [1, input],
                kernel: [1, kernel],
                padding: [0, 0],
                stride: [1, stride],
                dilation: [1, 1],
                math,
            })
        };

        let stream = runtime.stream();
        let tensor = |dims: &[usize]| DeviceTensor::zeros(stream, dims);
        let tensors = Tensors {
            input: tensor(&[batch, 1, shape.samples])?,
            wave_norm: tensor(&[batch, 1, shape.samples])?,
            sinc: tensor(&[batch, SINC_CHANNELS, shape.sinc])?,
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

        let sinc = conv([1, SINC_CHANNELS], shape.samples, SINC_KERNEL, SINC_STRIDE)?;
        let conv1 = conv([SINC_CHANNELS, FEATURES], shape.pool0, CONV_KERNEL, 1)?;
        let conv2 = conv([FEATURES, FEATURES], shape.pool1, CONV_KERNEL, 1)?;
        let workspace_bytes = [&sinc, &conv1, &conv2]
            .map(ConvPlan::workspace_bytes)
            .into_iter()
            .max()
            .unwrap_or(0);

        Ok(SegmentationWorkspace {
            shape,
            sinc,
            conv1,
            conv2,
            // at least one byte, because CUDA cannot allocate zero bytes
            conv_workspace: stream.alloc_zeros(workspace_bytes.max(1))?,
            lstm: self.lstm.plan(runtime, batch, shape.frames)?,
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

        // SincNet: abs of the band-pass outputs, then pool, normalize and activate
        sinc.forward(
            &mut conv_workspace,
            &t.wave_norm.data().as_view(),
            &self.sinc_filters.as_view(),
            &mut t.sinc.data_mut().as_view_mut(),
        )?;
        let stage0 = PoolNorm {
            abs_input: true,
            ..norm(SINC_CHANNELS, shape.sinc, POOL)
        };
        let [gamma, beta] = &self.norms[0];
        k.pool_norm(
            runtime,
            stage0,
            t.sinc.data(),
            zero_bias,
            gamma,
            beta,
            t.stage0.data_mut(),
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

        self.lstm
            .forward(runtime, lstm, t.lstm_input.data(), t.lstm_output.data_mut())?;

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
    #[cfg(test)]
    /// The activation lengths of this workspace
    pub fn shape(&self) -> SegmentationShape {
        self.shape
    }

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

    #[cfg(test)]
    /// One tensor of the last forward pass, for parity checks
    pub fn tensor(&self, tensor: SegmentationTensor) -> &DeviceTensor {
        let t = &self.tensors;
        match tensor {
            SegmentationTensor::Input => &t.input,
            SegmentationTensor::WaveNorm => &t.wave_norm,
            SegmentationTensor::SincConv => &t.sinc,
            SegmentationTensor::Stage0 => &t.stage0,
            SegmentationTensor::Conv1 => &t.conv1,
            SegmentationTensor::Stage1 => &t.stage1,
            SegmentationTensor::Conv2 => &t.conv2,
            SegmentationTensor::LstmInput => &t.lstm_input,
            SegmentationTensor::LstmOutput => &t.lstm_output,
            SegmentationTensor::Linear0 => &t.linear0,
            SegmentationTensor::Linear1 => &t.linear1,
            SegmentationTensor::Output => &t.output,
        }
    }

    #[cfg(test)]
    /// Device bytes of the activation buffers
    pub fn activation_bytes(&self) -> usize {
        SegmentationTensor::ALL
            .into_iter()
            .map(|tensor| self.tensor(tensor).len() * size_of::<f32>())
            .sum()
    }

    #[cfg(test)]
    /// Device bytes of the cuDNN RNN workspace
    pub fn lstm_workspace_bytes(&self) -> usize {
        self.lstm.workspace_bytes()
    }
}
