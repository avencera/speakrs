//! Test-only stack and Sinc operators owned by the segmentation stage

use super::dispatch::{LSTM, LSTM_LAYER, SINC, SINC_LAYER, SincIo};
use super::shape::WINDOW_SAMPLES;
use super::{
    CudaSegmentation, LstmPlan, LstmStage, Network, SINC_CHANNELS, SegmentationShape, SincPlan,
};
use crate::inference::cuda::candidate::{
    LstmCandidate, LstmOxide, SincCandidate, SincOutput, SincOxide,
};
use crate::inference::cuda::implementation::Choice;
use crate::inference::cuda::test_support::{self, Mutant};
use crate::inference::cuda::{CudaError, CudaRuntime};
use cudarc::driver::{CudaSlice, CudaViewMut};

pub(super) fn mutant_lstm(
    network: &Network,
    runtime: &CudaRuntime,
    plan: &mut LstmPlan,
    input: &CudaSlice<f32>,
    output: &mut CudaSlice<f32>,
    mutant: Mutant,
) -> Result<(), CudaError> {
    if mutant == Mutant::Precision {
        test_support::round_input(runtime, input)?;
    }
    // a lookup answers from the first input it saw, whatever it is given now
    let table = if mutant == Mutant::Lookup {
        Some(test_support::lookup(runtime, "lstm.stack/input", input)?)
    } else {
        None
    };
    let input = table.as_ref().unwrap_or(input);
    let iterations = if mutant == Mutant::Slow { 3 } else { 1 };
    for _ in 0..iterations {
        network
            .lstm
            .as_ref()
            .expect("mutant Library stack")
            .forward(runtime, plan, input, output)?;
    }
    test_support::post(runtime, output, plan.batch(), mutant)
}

/// The planted Sinc producer writes the raw convolution; the locked consumer follows
#[allow(clippy::too_many_arguments)]
pub(super) fn mutant_sinc(
    network: &Network,
    runtime: &CudaRuntime,
    shape: SegmentationShape,
    plan: &super::ConvPlan,
    workspace: &mut CudaViewMut<'_, u8>,
    input: &CudaSlice<f32>,
    raw: &mut CudaSlice<f32>,
    mutant: Mutant,
) -> Result<(), CudaError> {
    if mutant == Mutant::Precision {
        test_support::round_input(runtime, input)?;
    }
    let table = if mutant == Mutant::Lookup {
        Some(test_support::lookup(
            runtime,
            "sincnet.conv0.abs_pool/input",
            input,
        )?)
    } else {
        None
    };
    let input = table.as_ref().unwrap_or(input);
    let iterations = if mutant == Mutant::Slow { 3 } else { 1 };
    for _ in 0..iterations {
        plan.forward(
            workspace,
            &input.as_view(),
            &network.sinc_filters.as_view(),
            &mut raw.as_view_mut(),
        )?;
    }
    test_support::post(runtime, raw, shape.batch, mutant)
}

/// One isolated boundary with two input sets and its own output buffers
pub(crate) struct Isolated {
    target: &'static str,
    batch: usize,
    inputs: [CudaSlice<f32>; 2],
    saved: [CudaSlice<f32>; 2],
    raw: CudaSlice<f32>,
    pooled: CudaSlice<f32>,
    stage0: CudaSlice<f32>,
    output: CudaSlice<f32>,
}

impl Isolated {
    pub(crate) fn is_lstm(&self) -> bool {
        self.target == LSTM_LAYER
    }
}

/// Exact Sinc or LSTM weights with no CUDA owner
pub(crate) struct HostReference(ReferenceWeights);

enum ReferenceWeights {
    Sinc(Vec<f32>),
    Lstm(Vec<super::weights::LstmLayer>),
}

impl HostReference {
    /// Computes independent f64 truth using only host data
    pub(crate) fn evaluate(
        &self,
        input: &[f32],
        batch: usize,
        state: &mut u64,
    ) -> super::super::test_support::qualify::reference::Sample {
        use super::super::test_support::qualify::reference;
        match &self.0 {
            ReferenceWeights::Lstm(weights) => {
                let layers: Vec<_> = weights
                    .iter()
                    .map(|layer| reference::Lstm {
                        input: layer.input,
                        w: &layer.w,
                        r: &layer.r,
                        b: &layer.b,
                    })
                    .collect();
                reference::lstm(input, 589, &layers, &reference::batch_rows(batch, state))
            }
            ReferenceWeights::Sinc(filters) => reference::sinc(
                input,
                filters,
                reference::indices(&[batch, 80, 5325], state),
            ),
        }
    }
}

impl CudaSegmentation {
    /// Copies exact operator weights into a host-only f64 truth owner
    pub(crate) fn f64_snapshot(
        &self,
        runtime: &CudaRuntime,
        target: &str,
    ) -> Result<HostReference, CudaError> {
        if target == "lstm" {
            return Ok(HostReference(ReferenceWeights::Lstm(
                self.network.lstm_weights.clone(),
            )));
        }
        Ok(HostReference(ReferenceWeights::Sinc(
            runtime.stream().clone_dtoh(&self.network.sinc_filters)?,
        )))
    }

    pub(crate) fn qualification_has_graph(&self, batch: usize) -> bool {
        self.find_workspace(batch, WINDOW_SAMPLES)
            .is_some_and(|workspace| workspace.graph.is_some())
    }

    /// The stage's captured graph, for the locked driver's own replays
    pub(crate) fn qualification_graph(&self, batch: usize) -> Option<&cudarc::driver::CudaGraph> {
        self.find_workspace(batch, WINDOW_SAMPLES)?
            .graph
            .as_ref()
            .map(|graph| graph.inner())
    }

    /// Plant a stage-only accuracy defect in the actual device output
    pub(crate) fn qualification_round_stage(
        &self,
        runtime: &CudaRuntime,
        batch: usize,
    ) -> Result<(), CudaError> {
        let workspace = self
            .find_workspace(batch, WINDOW_SAMPLES)
            .expect("stage workspace");
        test_support::round_input(runtime, workspace.tensors.output.data())
    }

    /// Diagnostic per-layer stack taps from a candidate; never gated
    pub(crate) fn diagnostic_lstm_outputs(
        &self,
        runtime: &CudaRuntime,
        batch: usize,
    ) -> Vec<(usize, Vec<f32>)> {
        self.find_workspace(batch, WINDOW_SAMPLES)
            .and_then(|workspace| match &workspace.lstm {
                LstmStage::Oxide { candidate, .. } => Some(candidate),
                _ => None,
            })
            .map(|candidate| candidate.diagnostic_layers(runtime.stream()))
            .unwrap_or_default()
    }

    /// Whether the boundary runs a candidate or planted fault at this batch
    pub(crate) fn isolated_declared(&self, batch: usize, target: &str) -> bool {
        let Some(workspace) = self.find_workspace(batch, WINDOW_SAMPLES) else {
            return false;
        };
        if target == "lstm" {
            return !matches!(workspace.lstm, LstmStage::Library(_));
        }
        !matches!(workspace.sinc, SincPlan::Library(_))
    }

    pub(crate) fn isolated(
        &mut self,
        runtime: &CudaRuntime,
        batch: usize,
        target: &str,
        inputs: [&[f32]; 2],
    ) -> Result<Isolated, CudaError> {
        let shape = self.workspace(runtime, batch, WINDOW_SAMPLES)?.shape;
        let stream = runtime.stream();
        let target = if target == "lstm" {
            LSTM_LAYER
        } else {
            SINC_LAYER
        };
        let [first, second] = inputs;
        Ok(Isolated {
            target,
            batch,
            inputs: [stream.clone_htod(first)?, stream.clone_htod(second)?],
            saved: [stream.clone_htod(first)?, stream.clone_htod(second)?],
            raw: stream.alloc_zeros(batch * SINC_CHANNELS * shape.sinc)?,
            pooled: stream.alloc_zeros(batch * SINC_CHANNELS * shape.pool0)?,
            stage0: stream.alloc_zeros(batch * SINC_CHANNELS * shape.pool0)?,
            output: stream.alloc_zeros(batch * shape.frames * 256)?,
        })
    }

    /// Restores input set `which` after a mutant changed it in place
    pub(crate) fn isolated_restore(
        &self,
        runtime: &CudaRuntime,
        op: &mut Isolated,
        which: usize,
    ) -> Result<(), CudaError> {
        let [first, second] = &mut op.inputs;
        let input = if which == 0 { first } else { second };
        runtime.stream().memcpy_dtod(&op.saved[which], input)?;
        Ok(())
    }

    pub(crate) fn isolated_run(
        &mut self,
        runtime: &CudaRuntime,
        op: &mut Isolated,
        which: usize,
    ) -> Result<(), CudaError> {
        let index = self.workspace_index(runtime, op.batch, WINDOW_SAMPLES)?;
        let w = &mut self.workspaces[index];
        if op.target == SINC_LAYER {
            return self.network.sinc_forward(
                runtime,
                w.shape,
                &w.sinc,
                &mut w.conv_workspace.as_view_mut(),
                SincIo {
                    input: &op.inputs[which],
                    raw: &mut op.raw,
                    pooled: Some(&mut op.pooled),
                    stage0: &mut op.stage0,
                },
            );
        }

        self.network
            .lstm_forward(runtime, &mut w.lstm, &op.inputs[which], &mut op.output)
    }

    /// The gated boundary: the stack output, or the pooled Sinc tensor before the
    /// consumer, read outside the timed interval
    pub(crate) fn isolated_output(
        &self,
        runtime: &CudaRuntime,
        op: &Isolated,
    ) -> Result<Vec<f32>, CudaError> {
        if op.target == LSTM_LAYER {
            return Ok(runtime.stream().clone_dtoh(&op.output)?);
        }

        let w = self
            .find_workspace(op.batch, WINDOW_SAMPLES)
            .expect("isolated workspace");
        let pooled_by_candidate =
            matches!(w.sinc, SincPlan::Oxide(_)) && SincOxide::OUTPUT == SincOutput::Pooled;
        if pooled_by_candidate {
            return Ok(runtime.stream().clone_dtoh(&op.pooled)?);
        }

        let mut output = runtime.stream().alloc_zeros(op.pooled.len())?;
        test_support::pool(runtime, &op.raw, w.shape.sinc, &mut output)?;
        Ok(runtime.stream().clone_dtoh(&output)?)
    }

    /// Reports the coverage a candidate declares, for the result
    pub(crate) fn coverage(
        target: &str,
        tier: crate::inference::cuda::PtxTier,
    ) -> crate::inference::cuda::candidate::Coverage {
        if target == "lstm" {
            LstmOxide::coverage(tier)
        } else {
            SincOxide::coverage(tier)
        }
    }
}

#[test]
#[ignore = "requires reference fixtures, CPU only"]
fn f64_reference_matches_fixture_rounding() -> Result<(), CudaError> {
    use super::weights::SegmentationWeights;
    use crate::inference::cuda::SafetensorsFile;
    use crate::inference::cuda::test_support::qualify::reference;
    let weights = SegmentationWeights::load(&SafetensorsFile::open(
        "/workspace/models-native/segmentation-3.0.safetensors",
    )?)?;
    let file = SafetensorsFile::open("/workspace/ref/segmentation-3.0/test_first_b1.safetensors")?;
    let input = file.read_f32(
        "tensor//sincnet/wav_norm1d/InstanceNormalization_output_0",
        &[1, 1, 160000],
    )?;
    let truth = reference::sinc(
        &input,
        &weights.sinc_filters,
        reference::indices(&[1, 80, 5325], &mut 73),
    );
    let raw = file.read_f32("tensor//sincnet/Abs_output_0", &[1, 80, 15975])?;
    let expected: Vec<f32> = raw
        .as_chunks::<3>()
        .0
        .iter()
        .map(|v| v[0].max(v[1]).max(v[2]))
        .collect();
    reference::assert_rounding(&truth, &expected, 2e-6, 3e-6);

    let cf = file.read_f32("tensor//sincnet/LeakyRelu_2_output_0", &[1, 60, 589])?;
    let input: Vec<f32> = (0..589 * 60).map(|i| cf[(i % 60) * 589 + i / 60]).collect();
    let layers: Vec<_> = weights
        .lstm
        .iter()
        .map(|layer| reference::Lstm {
            input: layer.input,
            w: &layer.w,
            r: &layer.r,
            b: &layer.b,
        })
        .collect();
    let truth = reference::lstm(&input, 589, &layers, &[0]);
    let expected = file.read_f32("tensor//lstm/Transpose_5_output_0", &[1, 589, 256])?;
    reference::assert_rounding(&truth, &expected, 1e-5, 4e-5);
    Ok(())
}

use super::{SegmentationWorkspace, pooled_tensor};
use crate::inference::cuda::DeviceTensor;

impl CudaSegmentation {
    pub(crate) fn select_sinc(
        &mut self,
        runtime: &CudaRuntime,
        batch: usize,
        samples: usize,
        choice: Choice,
    ) -> Result<(), CudaError> {
        let index = self.workspace_index(runtime, batch, samples)?;
        let shape = self.workspaces[index].shape;
        let selected = crate::inference::cuda::implementation::plan_selection(
            runtime,
            SINC,
            batch,
            self.network.options.math,
            Some(choice),
        )?;
        let plan = self.network.plan_sinc(runtime, shape, selected)?;
        let workspace = &mut self.workspaces[index];
        if matches!(plan, SincPlan::Oxide(_)) && workspace.tensors.sinc_pooled.is_none() {
            workspace.tensors.sinc_pooled = pooled_tensor(runtime, workspace.shape)?;
        }
        let bytes = plan
            .workspace_bytes()
            .max(workspace.conv1.workspace_bytes())
            .max(workspace.conv2.workspace_bytes())
            .max(1);
        workspace.conv_workspace = runtime.stream().alloc_zeros(bytes)?;
        workspace.sinc = plan;
        workspace.graph = None;
        Ok(())
    }

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

    /// Runs the model eagerly on the input already uploaded to the `(batch, samples)`
    /// workspace, never replaying or capturing a graph
    pub fn forward_eager(
        &mut self,
        runtime: &CudaRuntime,
        batch: usize,
        samples: usize,
    ) -> Result<(), CudaError> {
        let index = self.workspace_index(runtime, batch, samples)?;
        self.network.enqueue(runtime, &mut self.workspaces[index])
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
            LstmStage::Library(plan) | LstmStage::Mutant { library: plan, .. } => {
                plan.workspace_bytes()
            }
            LstmStage::Oxide { .. } => 0,
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
