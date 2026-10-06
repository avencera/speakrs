//! The candidate interface: one trait per qualified boundary, implemented by kernel work
//! outside the locked harness
//!
//! A candidate registers a plan type, never a closure. Locked dispatch code
//! (`embedding/dispatch.rs` and `segmentation/dispatch.rs`) creates the plan when a
//! batch class is set up, outside every timed and traced interval, and then calls
//! [`ConvCandidate::enqueue`], [`SincCandidate::enqueue`] or
//! [`LstmCandidate::enqueue`] inside a scope it owns. The candidate only enqueues
//! device work on the stream it is given and on registered [`SideStream`]s, and opens
//! sub-scopes only through the locked [`Phases`] and [`LstmPhases`] handles
//!
//! Each trait declares a [`Coverage`]: the layer and batch pairs the candidate
//! implements, per math mode. Dispatch runs the candidate for exactly those triples and the
//! Library path for every other triple, and the harness qualifies exactly the declared
//! triples
//!
//! Candidate code lives in `candidate/` and its kernels in the `resnet`, `lstm` and
//! `sincnet` PTX areas. The harness scans those files before it builds anything; see
//! `scripts/cuda/qualify/README.md` for what the scan refuses. This file and the
//! dispatch files are locked. Production runs a candidate only where the locked
//! `implementation::PRODUCTION` table selects it, which the root sets at integration

use std::cell::{RefCell, RefMut};
use std::sync::Arc;

use cudarc::driver::{
    CudaEvent, CudaSlice, CudaStream, CudaView, CudaViewMut, DeviceRepr, ValidAsZeroBits,
};

use super::dnn::Conv2d;
use super::{CudaError, CudaMath, CudaRuntime, PtxTier, Sgemm};

mod conv;
mod lstm;
mod sinc;

#[cfg(test)]
pub(super) use kernel_inventory::conv_kernel_inventory;

#[cfg(test)]
mod kernel_inventory {
    use super::conv::{REQUIRED_KERNELS, SMALL_BATCH_WAVES, Shape, select_tiling};

    pub(crate) fn conv_kernel_inventory() -> Vec<&'static str> {
        let mut entries = REQUIRED_KERNELS.to_vec();
        // exercise both sides of the production device-dependent selection threshold
        for shape in [Shape::C32, Shape::C64, Shape::C32Stride2] {
            let (large, small) = shape.tilings();
            let output = [16, 64];
            let threshold = large.blocks(1, output).div_ceil(SMALL_BATCH_WAVES);
            for multiprocessors in [1, threshold, threshold + 1] {
                entries.push(select_tiling(large, small, 1, output, multiprocessors).entry);
            }
        }

        entries
    }
}
#[cfg(test)]
pub(super) use lstm::REQUIRED_KERNELS as LSTM_KERNELS;
#[cfg(test)]
pub(super) use sinc::REQUIRED_KERNELS as SINC_KERNELS;

pub(crate) use conv::Oxide as ConvOxide;
pub(crate) use lstm::Oxide as LstmOxide;
pub(crate) use sinc::Oxide as SincOxide;

/// A planning refusal that is distinct from a CUDA or model error
#[derive(Debug, thiserror::Error)]
pub(crate) enum PlanError {
    /// The device cannot host this candidate; only production may fall back
    #[error("{reason}")]
    DeviceUnsupported {
        /// The device constraint that prevents this plan
        reason: String,
    },
    /// A real error, which dispatch must propagate in every selection mode
    #[error(transparent)]
    Cuda(#[from] CudaError),
}

impl From<cudarc::driver::DriverError> for PlanError {
    fn from(error: cudarc::driver::DriverError) -> Self {
        Self::Cuda(error.into())
    }
}

/// Model batches that accepted records can authorize for production
pub(crate) const QUALIFIED_BATCHES: [usize; 2] = [1, 32];
/// Test samples, including stress batches that never grant production coverage
pub(crate) const TEST_BATCHES: [usize; 5] = [1, 7, 32, 33, 64];

/// Batch sizes a candidate implements
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Batches {
    /// Every positive batch size, tested at [`TEST_BATCHES`]; production remains
    /// restricted by the independent accepted table
    All,
    /// Exactly these batch sizes; every other batch runs the Library path
    Only(&'static [usize]),
}

/// Math modes a candidate implements
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Maths {
    /// FP32 and TF32
    All,
    /// Exactly these modes; the other mode runs the Library path
    Only(&'static [CudaMath]),
}

/// One product of layers, batch sizes and math modes a candidate implements
///
/// A triple is in the entry when its layer is listed, its batch is in `batches` and
/// its mode is in `maths`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CoverageEntry {
    /// Boundary names: ResNet convolution names such as `resnet.layer1.0.conv1`,
    /// `lstm.stack` or `sincnet.conv0.abs_pool`
    pub layers: &'static [&'static str],
    /// Batch sizes, for every layer listed above
    pub batches: Batches,
    /// Math modes, for every layer and batch above
    pub maths: Maths,
}

impl CoverageEntry {
    fn covers(&self, layer: &str, batch: usize, math: CudaMath) -> bool {
        let batch_covered = match self.batches {
            Batches::All => batch > 0,
            Batches::Only(batches) => batches.contains(&batch),
        };
        let math_covered = match self.maths {
            Maths::All => true,
            Maths::Only(maths) => maths.contains(&math),
        };
        batch_covered && math_covered && self.layers.contains(&layer)
    }
}

/// The layer, batch size and math mode triples a candidate implements: the union of
/// its entries, for example 32-channel layers at every batch plus 64-channel layers at
/// b32 and b64 only
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Coverage(pub &'static [CoverageEntry]);

impl Coverage {
    /// Covers nothing: every triple runs the Library path
    pub(crate) const NONE: Self = Self(&[]);

    /// Whether the candidate implements `layer` at `batch` in `math`
    pub(crate) fn covers(&self, layer: &str, batch: usize, math: CudaMath) -> bool {
        self.0.iter().any(|entry| entry.covers(layer, batch, math))
    }

    /// The entries whose union is the coverage
    pub(crate) fn entries(&self) -> &'static [CoverageEntry] {
        self.0
    }
}

/// One eligible ResNet 3x3 convolution with folded batch norm, as a candidate plans it
#[derive(Debug, Clone, Copy)]
pub(crate) struct ConvLayerSpec<'a> {
    /// The layer name, such as `resnet.layer2.0.conv1`
    pub name: &'a str,
    /// Shape, padding, stride, batch and math mode
    pub conv: Conv2d,
    /// Whether the output adds a residual before the ReLU
    pub residual: bool,
    /// Folded weights `[k, c, 3, 3]`
    pub weight: &'a CudaSlice<f32>,
    /// Folded bias `[k]`
    pub bias: &'a CudaSlice<f32>,
}

/// Device inputs of one convolution call
#[derive(Debug)]
pub(crate) struct ConvInputs<'a, 'b> {
    /// `[n, c, h, w]`
    pub x: &'a CudaView<'b, f32>,
    /// `[n, k, p, q]` when the layer adds a residual
    pub residual: Option<&'a CudaView<'b, f32>>,
    /// Folded weights `[k, c, 3, 3]`
    pub weight: &'a CudaView<'b, f32>,
    /// Folded bias `[k]`
    pub bias: &'a CudaView<'b, f32>,
}

/// A candidate for `y = relu(conv(x, w) + bias [+ residual])`
pub(crate) trait ConvCandidate: Sized {
    /// Convolution names and batch sizes this candidate implements
    const COVERAGE: Coverage;

    /// Coverage for the actual loaded tier; override when variants cover different tuples
    fn coverage(_tier: PtxTier) -> Coverage {
        Self::COVERAGE
    }

    /// Prepares one layer for one batch size; runs once per batch class, untimed
    fn plan(runtime: &CudaRuntime, layer: ConvLayerSpec<'_>) -> Result<Self, PlanError>;

    /// Enqueues the layer on `stream`, writing every element of `y` `[n, k, p, q]`
    fn enqueue(
        &self,
        inputs: ConvInputs<'_, '_>,
        y: &mut CudaViewMut<'_, f32>,
        phases: &Phases,
        stream: &CudaStream,
    ) -> Result<(), CudaError>;
}

/// What a Sinc candidate writes; the locked dispatch then runs the shared
/// `segmentation_pool_norm` consumer with the matching parameters
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SincOutput {
    /// The raw convolution `[n, 80, sinc]`; the consumer takes `abs`, pools by 3 and
    /// normalizes, as on the Library path
    RawConv,
    /// `max(|conv|)` over each window of 3, `[n, 80, pool0]`; the consumer only
    /// normalizes (pool 1, no `abs`)
    Pooled,
}

/// The Sinc producer for one batch size
#[derive(Debug, Clone, Copy)]
pub(crate) struct SincSpec<'a> {
    /// Windows per batch
    pub batch: usize,
    /// Samples per window, 160000
    pub samples: usize,
    /// Convolution output steps, 15975
    pub sinc: usize,
    /// Pooled steps, 5325
    pub pooled: usize,
    /// Math mode of the boundary
    pub math: CudaMath,
    /// Generated filters `[80, 1, 251]`
    pub filters: &'a CudaSlice<f32>,
}

/// Device inputs of one Sinc call
#[derive(Debug)]
pub(crate) struct SincInputs<'a, 'b> {
    /// Normalized waveforms `[n, 1, samples]`
    pub waveform: &'a CudaView<'b, f32>,
    /// Generated filters `[80, 1, 251]`
    pub filters: &'a CudaView<'b, f32>,
}

/// A candidate for the Sinc convolution, optionally fused with `abs` and max pool
pub(crate) trait SincCandidate: Sized {
    /// `sincnet.conv0.abs_pool` and the batch sizes this candidate implements
    const COVERAGE: Coverage;

    /// Coverage for the actual loaded tier; override when variants cover different tuples
    fn coverage(_tier: PtxTier) -> Coverage {
        Self::COVERAGE
    }
    /// The tensor [`Self::enqueue`] writes
    const OUTPUT: SincOutput;

    /// Prepares one batch size; runs once per batch class, untimed
    fn plan(runtime: &CudaRuntime, spec: SincSpec<'_>) -> Result<Self, PlanError>;

    /// Enqueues the producer on `stream`, writing every element of `output`
    fn enqueue(
        &self,
        inputs: SincInputs<'_, '_>,
        output: &mut CudaViewMut<'_, f32>,
        phases: &Phases,
        stream: &CudaStream,
    ) -> Result<(), CudaError>;
}

/// Host weights of one bidirectional LSTM layer, in ONNX layout
#[derive(Debug, Clone, Copy)]
pub(crate) struct LstmLayerWeights<'a> {
    /// Input size: 60 for layer 0, 256 above
    pub input: usize,
    /// `W`, `[2, 512, input]`, gates `[i, o, f, c]`
    pub w: &'a [f32],
    /// `R`, `[2, 512, 128]`
    pub r: &'a [f32],
    /// `B`, `[2, 1024]`: input biases then recurrent biases
    pub b: &'a [f32],
}

/// The four-layer bidirectional stack for one batch size
#[derive(Debug, Clone, Copy)]
pub(crate) struct LstmSpec<'a> {
    /// Sequences per batch
    pub batch: usize,
    /// Steps per sequence, 589
    pub frames: usize,
    /// Math mode of the boundary
    pub math: CudaMath,
    /// The four layers, input first
    pub layers: [LstmLayerWeights<'a>; 4],
}

/// A candidate for the complete stack: `[n, frames, 60]` in, `[n, frames, 256]` out
/// with the forward direction in the first 128 features
pub(crate) trait LstmCandidate: Sized {
    /// `lstm.stack` and the batch sizes this candidate implements
    const COVERAGE: Coverage;

    /// Coverage for the actual loaded tier; override when variants cover different tuples
    fn coverage(_tier: PtxTier) -> Coverage {
        Self::COVERAGE
    }

    /// Prepares one batch size; runs once per batch class, untimed
    fn plan(runtime: &CudaRuntime, spec: LstmSpec<'_>) -> Result<Self, PlanError>;

    /// Enqueues the stack on `stream`, every input projection inside
    /// [`LstmPhases::input_proj`] and every recurrence inside [`LstmPhases::recurrence`]
    fn enqueue(
        &self,
        input: &CudaView<'_, f32>,
        output: &mut CudaViewMut<'_, f32>,
        phases: &LstmPhases<'_>,
        stream: &CudaStream,
    ) -> Result<(), CudaError>;

    /// Batch-major `[n, frames, 256]` outputs of layers 0 to 2 after the last
    /// [`Self::enqueue`], for diagnostics only; they are never gated
    fn diagnostic_layers(&self, _stream: &CudaStream) -> Vec<(usize, Vec<f32>)> {
        Vec::new()
    }
}

/// LSTM direction of an input projection or recurrence
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Direction {
    /// Left to right
    Forward,
    /// Right to left
    Reverse,
}

impl Direction {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Forward => "forward",
            Self::Reverse => "reverse",
        }
    }
}

/// The fixed names of a candidate's optional sub-scopes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Op {
    /// Work before the main computation, such as clearing scratch
    Prologue,
    /// Repacking inputs or weights into a kernel layout
    Pack,
    /// The main computation
    Main,
    /// A cross-block reduction
    Reduce,
    /// Work after the main computation, such as an epilogue or unpacking
    Epilogue,
}

impl Op {
    fn name(self) -> &'static str {
        match self {
            Self::Prologue => "prologue",
            Self::Pack => "pack",
            Self::Main => "main",
            Self::Reduce => "reduce",
            Self::Epilogue => "epilogue",
        }
    }
}

/// Locked sub-scopes inside a candidate's `enqueue`
///
/// The scope names come from fixed methods and the [`Op`] enum, never from candidate
/// text, so a candidate cannot forge a harness range
#[derive(Debug)]
pub(crate) struct Phases(());

impl Phases {
    pub(super) fn new() -> Self {
        Self(())
    }

    /// Runs `enqueue` inside the locked scope for `op`
    pub(crate) fn op<T>(
        &self,
        op: Op,
        enqueue: impl FnOnce() -> Result<T, CudaError>,
    ) -> Result<T, CudaError> {
        let _scope = sub_scope(|| format!("op.{}", op.name()));
        enqueue()
    }
}

/// Locked sub-scopes of the LSTM stack
///
/// The harness requires all eight `input_proj` and all eight `recurrence` scopes, one
/// per layer and direction, each with at least one launch and none overlapping. The
/// projection helper exists only inside `input_proj`, and `recurrence` must be
/// library-free
#[derive(Debug)]
pub(crate) struct LstmPhases<'a> {
    phases: Phases,
    projection: Projection<'a>,
}

impl<'a> LstmPhases<'a> {
    pub(super) fn new(projection: Projection<'a>) -> Self {
        Self {
            phases: Phases::new(),
            projection,
        }
    }

    /// Runs `enqueue` inside the locked scope for `op`
    pub(crate) fn op<T>(
        &self,
        op: Op,
        enqueue: impl FnOnce() -> Result<T, CudaError>,
    ) -> Result<T, CudaError> {
        self.phases.op(op, enqueue)
    }

    /// Runs the input projection of `layer` in `direction`, with the projection helper
    pub(crate) fn input_proj<T>(
        &self,
        layer: usize,
        direction: Direction,
        enqueue: impl FnOnce(&Projection<'_>) -> Result<T, CudaError>,
    ) -> Result<T, CudaError> {
        check_layer(layer)?;
        let _scope = sub_scope(|| format!("input_proj.L{layer}.{}", direction.name()));
        enqueue(&self.projection)
    }

    /// Runs the recurrence of `layer` in `direction`
    pub(crate) fn recurrence<T>(
        &self,
        layer: usize,
        direction: Direction,
        enqueue: impl FnOnce() -> Result<T, CudaError>,
    ) -> Result<T, CudaError> {
        check_layer(layer)?;
        let _scope = sub_scope(|| format!("recurrence.L{layer}.{}", direction.name()));
        enqueue()
    }
}

#[cfg(all(test, feature = "cuda", not(feature = "cuda-driver-only")))]
#[path = "candidate_test_support.rs"]
pub(crate) mod test_support;

#[cfg(all(test, feature = "cuda", not(feature = "cuda-driver-only")))]
use test_support::{projection_scope, sub_scope};

/// Production opens no harness scope
#[cfg(not(all(test, feature = "cuda", not(feature = "cuda-driver-only"))))]
struct NoScope;

#[cfg(not(all(test, feature = "cuda", not(feature = "cuda-driver-only"))))]
fn sub_scope(_name: impl FnOnce() -> String) -> NoScope {
    NoScope
}

#[cfg(not(all(test, feature = "cuda", not(feature = "cuda-driver-only"))))]
fn projection_scope(_stream: &CudaStream, _layer: usize, _direction: Direction) -> NoScope {
    NoScope
}

fn check_layer(layer: usize) -> Result<(), CudaError> {
    if layer < 4 {
        return Ok(());
    }

    Err(CudaError::Unsupported {
        context: "LSTM phase",
        reason: format!("layer {layer} is not one of the four stack layers"),
    })
}

/// A device buffer a plan owns and its `enqueue` writes through `&self`
///
/// Only device memory changes after `plan`; candidate code holds no other mutable state
#[derive(Debug)]
pub(crate) struct Scratch<T>(RefCell<CudaSlice<T>>);

impl<T: DeviceRepr + ValidAsZeroBits> Scratch<T> {
    /// A zeroed buffer of `len` elements on the runtime's stream
    pub(crate) fn zeros(runtime: &CudaRuntime, len: usize) -> Result<Self, CudaError> {
        Ok(Self(RefCell::new(runtime.stream().alloc_zeros(len)?)))
    }

    /// A buffer holding `values`, for example repacked weights
    pub(crate) fn from_host(runtime: &CudaRuntime, values: &[T]) -> Result<Self, CudaError> {
        Ok(Self(RefCell::new(runtime.stream().clone_htod(values)?)))
    }

    /// The buffer, for reading or writing during one call
    pub(crate) fn get(&self) -> RefMut<'_, CudaSlice<T>> {
        self.0.borrow_mut()
    }
}

/// A side stream a candidate creates in `plan` and forks from the given stream in
/// `enqueue`, for example to run the reverse direction concurrently
///
/// The locked type records the stream for the harness, so the profile and the
/// captured-graph checks cover work on it like work on the main stream
#[derive(Debug)]
pub(crate) struct SideStream {
    stream: Arc<CudaStream>,
    forked: CudaEvent,
    joined: CudaEvent,
}

impl SideStream {
    /// Creates a stream on the runtime's context, with its fork and join events
    pub(crate) fn new(runtime: &CudaRuntime) -> Result<Self, CudaError> {
        let context = runtime.context();
        let stream = context.new_stream()?;
        #[cfg(all(test, feature = "cuda", not(feature = "cuda-driver-only")))]
        super::test_support::register_side_stream(&stream);
        Ok(Self {
            stream,
            forked: context.new_event(None)?,
            joined: context.new_event(None)?,
        })
    }

    /// The stream to launch on between [`Self::split`] and [`Self::merge`]
    pub(crate) fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// Makes the side stream wait for everything already queued on `parent`
    pub(crate) fn split(&self, parent: &CudaStream) -> Result<(), CudaError> {
        self.forked.record(parent)?;
        self.stream.wait(&self.forked)?;
        Ok(())
    }

    /// Makes `parent` wait for everything already queued on the side stream; every
    /// split must be merged before `enqueue` returns
    pub(crate) fn merge(&self, parent: &CudaStream) -> Result<(), CudaError> {
        self.joined.record(&self.stream)?;
        parent.wait(&self.joined)?;
        Ok(())
    }
}

/// The shape options of one input projection GEMM
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ProjectionGemm {
    /// Output columns: 128, 256, 384 or 512 gate rows
    pub n: usize,
    /// The weight is stored `[n, k]`, as ONNX `W` is
    pub weight_transposed: bool,
    /// 0 overwrites the output, 1 accumulates into it (for example onto a bias)
    pub beta: f32,
}

/// The locked cuBLAS input projection, available inside [`LstmPhases::input_proj`]
///
/// Library controls and the pinned production stack use this helper. Fresh Oxide
/// qualifications require custom projection launches instead of library calls
///
/// It computes `c[m, n] = a[m, k] · w + beta * c` with `m = batch * frames`, `k` the
/// layer's input size and the boundary's math mode, and refuses any other shape
#[derive(Debug)]
pub(crate) struct Projection<'a> {
    runtime: &'a CudaRuntime,
    rows: usize,
    math: CudaMath,
}

impl<'a> Projection<'a> {
    /// Allowed output columns, one to four gates of one direction
    pub(crate) const COLUMNS: [usize; 4] = [128, 256, 384, 512];

    pub(super) fn new(runtime: &'a CudaRuntime, rows: usize, math: CudaMath) -> Self {
        Self {
            runtime,
            rows,
            math,
        }
    }

    /// Runs one input projection of `layer` in `direction`
    pub(crate) fn project(
        &self,
        layer: usize,
        direction: Direction,
        gemm: ProjectionGemm,
        a: &CudaView<'_, f32>,
        weight: &CudaView<'_, f32>,
        c: &mut CudaViewMut<'_, f32>,
    ) -> Result<(), CudaError> {
        let context = "LSTM input projection";
        let k = match layer {
            0 => 60,
            1..=3 => 256,
            _ => {
                return Err(CudaError::Unsupported {
                    context,
                    reason: format!("layer {layer} is not one of the four stack layers"),
                });
            }
        };
        if !Self::COLUMNS.contains(&gemm.n) || ![0.0, 1.0].contains(&gemm.beta) {
            return Err(CudaError::Unsupported {
                context,
                reason: format!("n = {}, beta = {} is not a projection", gemm.n, gemm.beta),
            });
        }

        let _scope = projection_scope(self.runtime.stream(), layer, direction);
        let spec = Sgemm {
            b_transposed: gemm.weight_transposed,
            beta: gemm.beta,
            math: self.math,
            ..Sgemm::new(self.rows, gemm.n, k)
        };
        self.runtime.sgemm(spec, a, weight, c)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Batches, Coverage, CoverageEntry, CudaError, CudaMath, CudaRuntime, CudaStream,
        CudaViewMut, Maths, Phases, PlanError, PtxTier, SincCandidate, SincInputs, SincOutput,
        SincSpec,
    };

    struct TierFixture;

    impl SincCandidate for TierFixture {
        const COVERAGE: Coverage = Coverage::NONE;
        const OUTPUT: SincOutput = SincOutput::Pooled;

        fn coverage(tier: PtxTier) -> Coverage {
            match tier {
                PtxTier::Sm75 => Coverage(&[CoverageEntry {
                    layers: &["sincnet.conv0.abs_pool"],
                    batches: Batches::Only(&[1]),
                    maths: Maths::Only(&[CudaMath::Fp32]),
                }]),
                PtxTier::Sm80 => Coverage(&[CoverageEntry {
                    layers: &["sincnet.conv0.abs_pool"],
                    batches: Batches::Only(&[32]),
                    maths: Maths::Only(&[CudaMath::Tf32]),
                }]),
                _ => Coverage::NONE,
            }
        }

        fn plan(_runtime: &CudaRuntime, _spec: SincSpec<'_>) -> Result<Self, PlanError> {
            unreachable!("coverage-only fixture does not construct GPU plans")
        }

        fn enqueue(
            &self,
            _inputs: SincInputs<'_, '_>,
            _output: &mut CudaViewMut<'_, f32>,
            _phases: &Phases,
            _stream: &CudaStream,
        ) -> Result<(), CudaError> {
            unreachable!("coverage-only fixture does not launch kernels")
        }
    }

    #[test]
    fn per_tier_coverage_is_not_the_static_default_or_another_tiers_triples() {
        let layer = "sincnet.conv0.abs_pool";
        let baseline = TierFixture::coverage(PtxTier::Sm75);
        let higher = TierFixture::coverage(PtxTier::Sm80);
        assert!(TierFixture::COVERAGE.entries().is_empty());
        assert!(baseline.covers(layer, 1, CudaMath::Fp32));
        assert!(!baseline.covers(layer, 32, CudaMath::Tf32));
        assert!(higher.covers(layer, 32, CudaMath::Tf32));
        assert!(!higher.covers(layer, 1, CudaMath::Fp32));
        assert!(TierFixture::coverage(PtxTier::Sm120).entries().is_empty());
    }
}
