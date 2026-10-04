//! Isolated convolution qualification through the locked production dispatcher

use super::{ConvLayer, ConvPlan, ConvPlanner, Convs, ResNetEmbedding, Residual};
use crate::inference::cuda::candidate::{ConvCandidate, ConvOxide};
use crate::inference::cuda::implementation::Choice;
use crate::inference::cuda::test_support::{self, Mutant};
use crate::inference::cuda::{CudaError, CudaRuntime, SafetensorsFile};
use cudarc::driver::{CudaSlice, CudaView, CudaViewMut};

pub(super) fn mutant_conv(
    convs: &mut Convs<'_>,
    layer: &ConvLayer,
    x: &CudaView<'_, f32>,
    residual: Residual<'_, '_>,
    y: &mut CudaViewMut<'_, f32>,
    mutant: Mutant,
) -> Result<(), CudaError> {
    if mutant == Mutant::Precision {
        test_support::round_input(convs.runtime, x)?;
    }

    // a lookup answers with the first weights it saw, whatever it is given now
    let table = if mutant == Mutant::Lookup {
        let weight = format!("{}/weight", layer.name());
        let bias = format!("{}/bias", layer.name());
        Some((
            test_support::lookup(convs.runtime, &weight, layer.weight().data())?,
            test_support::lookup(convs.runtime, &bias, layer.bias().data())?,
        ))
    } else {
        None
    };
    let (weight, bias) = match &table {
        Some((weight, bias)) => (weight.as_view(), bias.as_view()),
        None => (
            layer.weight().data().as_view(),
            layer.bias().data().as_view(),
        ),
    };

    let iterations = if mutant == Mutant::Slow { 3 } else { 1 };
    for _ in 0..iterations {
        let residual = match &residual {
            Residual::None { scratch } => Residual::None { scratch },
            Residual::Add(value) => Residual::Add(value),
        };
        convs.plan(layer).forward_bias_relu(
            &mut convs.workspace.as_view_mut(),
            x,
            &weight,
            &bias,
            residual,
            y,
        )?;
    }

    test_support::post(convs.runtime, y, convs.chunks, mutant)
}

/// The reference tensors of one convolution
fn names(block: usize, second: bool, shortcut: bool) -> [String; 3] {
    let input = if second {
        format!("tensor/relu_{}", 2 * block + 1)
    } else if block == 0 {
        "tensor/relu".into()
    } else {
        format!("tensor/relu_{}", 2 * block)
    };
    let output = format!("tensor/relu_{}", 2 * block + if second { 2 } else { 1 });
    let residual = if shortcut {
        "tensor/getitem_27".into()
    } else if block == 0 {
        "tensor/relu".into()
    } else {
        format!("tensor/relu_{}", 2 * block)
    };
    [input, output, residual]
}

/// One input set: the live input, a pristine copy for a mutant that rounds its input
/// in place, and the residual
struct Inputs {
    input: CudaSlice<f32>,
    saved: Option<CudaSlice<f32>>,
    residual: CudaSlice<f32>,
}

/// Host data of one input set: the input and, for a second convolution, the residual
pub(crate) struct HostInputs {
    pub(crate) input: Vec<f32>,
    pub(crate) residual: Option<Vec<f32>>,
}

/// One layer with two reference input sets, separate from the full-stage buffers
pub(crate) struct Operator<'a> {
    model: &'a ResNetEmbedding,
    layer: &'a ConvLayer,
    batch: usize,
    plans: Vec<ConvPlan>,
    workspace: CudaSlice<u8>,
    candidates: Vec<(String, ConvOxide)>,
    inputs: Vec<Inputs>,
    output: CudaSlice<f32>,
    adds_residual: bool,
    /// Reference outputs for each input set
    pub(crate) references: Vec<Vec<f32>>,
}

impl<'a> Operator<'a> {
    pub(crate) fn new(
        model: &'a ResNetEmbedding,
        runtime: &CudaRuntime,
        files: [&SafetensorsFile; 2],
        batch: usize,
        block: usize,
        second: bool,
    ) -> Result<Self, CudaError> {
        let b = &model.0.trunk.blocks[block];
        let [input_name, output_name, residual_name] = names(block, second, b.shortcut.is_some());
        let read =
            |file, name: &str| super::super::test_support::qualify::read_batch(file, name, batch);
        let mut sets = Vec::new();
        let mut references = Vec::new();
        for file in files {
            sets.push(HostInputs {
                input: read(file, &input_name)?,
                residual: if second {
                    Some(read(file, &residual_name)?)
                } else {
                    None
                },
            });
            references.push(read(file, &output_name)?);
        }

        Self::from_host(model, runtime, &sets, references, batch, block, second)
    }

    /// The element counts of one input and one output at `batch`
    pub(crate) fn lens(
        model: &ResNetEmbedding,
        batch: usize,
        block: usize,
        second: bool,
    ) -> (usize, usize) {
        let b = &model.0.trunk.blocks[block];
        let layer = if second { &b.conv2 } else { &b.conv1 };
        let conv = layer.conv(batch, model.0.math);
        (
            conv.input_shape().iter().product(),
            conv.output_shape().iter().product(),
        )
    }

    /// One operator on the given input sets, for the fixtures or an in-process input
    pub(crate) fn from_host(
        model: &'a ResNetEmbedding,
        runtime: &CudaRuntime,
        sets: &[HostInputs],
        references: Vec<Vec<f32>>,
        batch: usize,
        block: usize,
        second: bool,
    ) -> Result<Self, CudaError> {
        let b = &model.0.trunk.blocks[block];
        let layer = if second { &b.conv2 } else { &b.conv1 };
        let stream = runtime.stream();
        let (_, output_len) = Self::lens(model, batch, block, second);
        // only the precision mutant changes its input in place and needs a restore copy
        let restores = layer.choice(batch, model.0.math) == Choice::Mutant(Mutant::Precision);
        let mut inputs = Vec::new();
        for set in sets {
            inputs.push(Inputs {
                input: stream.clone_htod(&set.input)?,
                saved: if restores {
                    Some(stream.clone_htod(&set.input)?)
                } else {
                    None
                },
                residual: match &set.residual {
                    Some(residual) => stream.clone_htod(residual)?,
                    None => stream.alloc_zeros(output_len)?,
                },
            });
        }

        let planner = ConvPlanner::new(runtime)?;
        let plans = model
            .0
            .trunk
            .shapes()
            .iter()
            .map(|shape| planner.plan(shape.conv(batch, model.0.math)))
            .collect::<Result<Vec<_>, _>>()?;
        let workspace_bytes = plans
            .iter()
            .map(ConvPlan::workspace_bytes)
            .max()
            .unwrap_or(1)
            .max(1);
        let candidates =
            super::dispatch::plan_candidates(runtime, &model.0.trunk, batch, model.0.math)?;
        Ok(Self {
            model,
            layer,
            batch,
            plans,
            workspace: stream.alloc_zeros(workspace_bytes)?,
            candidates,
            output: stream.alloc_zeros(output_len)?,
            inputs,
            adds_residual: second,
            references,
        })
    }

    /// Independent f64 truth from the same host inputs and uploaded folded weights
    pub(crate) fn f64_reference(
        &self,
        runtime: &CudaRuntime,
        input: &HostInputs,
        state: &mut u64,
    ) -> Result<super::super::test_support::qualify::reference::Sample, CudaError> {
        use super::super::test_support::qualify::reference;
        let spec = self.layer.conv(self.batch, self.model.0.math);
        let weight = self.layer.weight().download(runtime.stream())?;
        let bias = self.layer.bias().download(runtime.stream())?;
        Ok(reference::conv(
            &spec,
            &input.input,
            &weight,
            &bias,
            input.residual.as_deref(),
            reference::indices(&spec.output_shape(), state),
        ))
    }

    pub(crate) fn name(&self) -> &str {
        self.layer.name()
    }

    /// Whether the candidate runs this pair; undeclared pairs run the Library path
    pub(crate) fn declared(&self) -> bool {
        match self.layer.choice(self.batch, self.model.0.math) {
            Choice::Library => false,
            Choice::Oxide => ConvOxide::COVERAGE.covers(self.name(), self.batch, self.model.0.math),
            Choice::Mutant(_) => true,
        }
    }

    /// Restores input set `which` after a mutant changed it in place
    pub(crate) fn restore(&mut self, runtime: &CudaRuntime, which: usize) -> Result<(), CudaError> {
        let set = &mut self.inputs[which];
        if let Some(saved) = &set.saved {
            runtime.stream().memcpy_dtod(saved, &mut set.input)?;
        }
        Ok(())
    }

    pub(crate) fn run(&mut self, runtime: &CudaRuntime, which: usize) -> Result<(), CudaError> {
        let mut convs = Convs {
            runtime,
            kernels: &self.model.0.kernels,
            plans: &self.plans,
            workspace: &mut self.workspace,
            chunks: self.batch,
            math: self.model.0.math,
            candidates: &self.candidates,
        };
        let set = &self.inputs[which];
        let r = set.residual.as_view();
        let residual = if self.adds_residual {
            Residual::Add(&r)
        } else {
            Residual::None { scratch: &r }
        };
        convs.conv_bias_relu(
            self.layer,
            &set.input.as_view(),
            residual,
            &mut self.output.as_view_mut(),
        )
    }

    pub(crate) fn output(&self, runtime: &CudaRuntime) -> Result<Vec<f32>, CudaError> {
        Ok(runtime.stream().clone_dtoh(&self.output)?)
    }
}

use super::{EmbeddingBatch, EmbeddingTapFn, ForwardGraph, SPEAKERS_PER_CHUNK};
use cudarc::driver::CudaGraph;
use std::sync::Arc;

impl ResNetEmbedding {
    pub(crate) fn select_conv(&mut self, name: &str, choice: Choice) -> bool {
        let Some(model) = Arc::get_mut(&mut self.0) else {
            return false;
        };

        model
            .trunk
            .blocks
            .iter_mut()
            .any(|block| block.conv1.select(name, choice) || block.conv2.select(name, choice))
    }
}

impl EmbeddingBatch {
    /// Runs the forward pass eagerly and calls `tap` with every intermediate
    /// activation listed in [`EmbeddingTap`], in execution order
    ///
    /// The view is only valid during the call; download it there if needed. This
    /// never replays a captured graph
    pub fn forward_with_taps(
        &mut self,
        runtime: &CudaRuntime,
        tap: &mut EmbeddingTapFn<'_>,
    ) -> Result<(), CudaError> {
        self.run(runtime, tap)
    }

    /// Copies host fbank and masks into the batch, runs the forward pass and
    /// downloads the embeddings
    ///
    /// `fbank` is `[chunks, 998, 80]` and `masks` is `[chunks * 3, 589]`; the result
    /// is `[chunks * 3, 256]`, all row-major
    pub fn embed(
        &mut self,
        runtime: &CudaRuntime,
        fbank: &[f32],
        masks: &[f32],
    ) -> Result<Vec<f32>, CudaError> {
        let stream = runtime.stream();
        self.fbank.copy_from_host(stream, fbank)?;
        self.masks.copy_from_host(stream, masks)?;
        self.forward(runtime)?;
        self.download_output(runtime)
    }

    /// Embeddings per forward pass, `chunks * 3`
    pub fn rows(&self) -> usize {
        self.chunks * SPEAKERS_PER_CHUNK
    }

    /// Distinct convolution shapes, one cuDNN plan each
    pub fn plan_count(&self) -> usize {
        self.plans.len()
    }

    /// Whether [`Self::forward`] replays a captured CUDA graph
    pub fn has_graph(&self) -> bool {
        self.graph.is_some()
    }

    /// The captured forward graph, for the locked qualification driver's replays
    pub fn graph(&self) -> Option<&CudaGraph> {
        self.graph.as_ref().map(|ForwardGraph(graph)| graph)
    }

    /// Bytes of the shared cuDNN workspace
    pub fn workspace_bytes(&self) -> usize {
        self.workspace.len()
    }

    /// Bytes held by this batch's activation, input and output buffers, excluding
    /// the cuDNN workspace
    pub fn buffer_bytes(&self) -> usize {
        let floats = self.fbank.len()
            + self.masks.len()
            + self.stem_input.len()
            + self.trunk.iter().map(CudaSlice::len).sum::<usize>()
            + self.hidden.len()
            + self.shortcut.len()
            + self.pooled.len()
            + self.output.len();
        floats * size_of::<f32>()
    }
}
