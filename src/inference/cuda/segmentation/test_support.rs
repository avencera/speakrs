//! Test-only accessors and explicit LSTM selection for the segmentation GPU tests

use cudarc::driver::CudaSlice;

use super::dispatch::LSTM;
use super::shape::WINDOW_SAMPLES;
use super::{CudaSegmentation, LstmStage, SegmentationShape, SegmentationWorkspace};
use crate::inference::cuda::implementation::Choice;
use crate::inference::cuda::{CudaError, CudaRuntime, DeviceTensor};

impl CudaSegmentation {
    /// Plans the LSTM stack with an explicit choice and drops any captured graph
    pub(crate) fn select_lstm(
        &mut self,
        runtime: &CudaRuntime,
        shape: [usize; 2],
        choice: Choice,
    ) -> Result<(), CudaError> {
        let index = self.workspace_index(runtime, shape[0], shape[1])?;
        let shape = self.workspaces[index].shape;
        let selected = crate::inference::cuda::implementation::plan_selection(
            runtime,
            LSTM,
            shape.batch,
            self.network.options.math,
            Some(choice),
        )?;
        let plan = self.network.plan_lstm(runtime, shape, selected)?;
        let workspace = &mut self.workspaces[index];
        workspace.lstm = plan;
        workspace.graph = None;
        Ok(())
    }

    /// Run only the selected LSTM stack with device inputs for direct parity checks
    pub(crate) fn run_lstm(
        &mut self,
        runtime: &CudaRuntime,
        batch: usize,
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
    ) -> Result<(), CudaError> {
        let index = self.workspace_index(runtime, batch, WINDOW_SAMPLES)?;
        self.network
            .lstm_forward(runtime, &mut self.workspaces[index].lstm, input, output)
    }

    /// The generated `[80, 1, 251]` SincNet filters, for parity checks
    pub fn sinc_filters(&self) -> &CudaSlice<f32> {
        &self.network.sinc_filters
    }

    /// The workspace for `batch` windows of `samples` samples, if one was allocated;
    /// its tensors hold the last forward pass with that shape
    pub fn find_workspace(&self, batch: usize, samples: usize) -> Option<&SegmentationWorkspace> {
        self.workspaces
            .iter()
            .find(|workspace| workspace.matches(batch, samples))
    }

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
}

impl SegmentationWorkspace {
    /// The activation lengths of this workspace
    pub fn shape(&self) -> SegmentationShape {
        self.shape
    }

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

    /// Device bytes of the activation buffers
    pub fn activation_bytes(&self) -> usize {
        SegmentationTensor::ALL
            .into_iter()
            .map(|tensor| self.tensor(tensor).len() * size_of::<f32>())
            .sum()
    }

    /// Device bytes of the cuDNN RNN workspace
    pub fn lstm_workspace_bytes(&self) -> usize {
        match &self.lstm {
            LstmStage::Library(plan) => plan.workspace_bytes(),
            LstmStage::Oxide { .. } | LstmStage::Projected { .. } => 0,
        }
    }
}

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

impl CudaSegmentation {
    /// Samples in the 10 s, 16 kHz window speakrs segments
    pub const WINDOW_SAMPLES: usize = WINDOW_SAMPLES;
}
