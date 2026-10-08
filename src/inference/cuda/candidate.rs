//! The candidate interface: one trait per kernel boundary
//!
//! A candidate registers a plan type, never a closure. Locked dispatch code
//! (`embedding/dispatch.rs` and `segmentation/dispatch.rs`) creates the plan when a
//! batch class is set up, and then calls
//! [`ConvCandidate::enqueue`], [`SincCandidate::enqueue`], [`LstmCandidate::enqueue`]
//! or [`FbankCandidate::enqueue`]. The candidate only enqueues
//! device work on the stream it is given and on [`SideStream`]s. Phase execution
//! uses the [`Phases`] and [`LstmPhases`] handles
//!
//! Each trait declares a [`Coverage`]: the layer and batch pairs the candidate
//! implements, per math mode. Driver-only routing uses this implemented coverage.
//! Hybrid routing requires a port speed scope or a qualified production tuple;
//! unmeasured device-sensitive tuples use Library. Each plan owns exactly
//! the declared triples
//!
//! A plan is built from a [`ConfigPin`] that names its complete execution choice.
//! A qualified tuple supplies its record pin; a complete port supplies its fixed
//! device rule result. Direct development tests use the implemented pin. Each trait
//! also states its [`SpecialValues`]
//! contract. The locked owner passes the exact loaded module from the selection
//! token; a plan never resolves production module policy again
//!
//! Candidate code lives in `candidate/` and its kernels in the `resnet`, `lstm`,
//! `sincnet` and `fbankdft` PTX areas. Production selection uses typed artifact
//! bindings and measured speed scopes

use std::cell::{RefCell, RefMut};
use std::sync::Arc;

use cudarc::driver::{
    CudaEvent, CudaSlice, CudaStream, CudaView, CudaViewMut, DeviceRepr, ValidAsZeroBits,
};

use super::device::DeviceAttributes;
pub(crate) use super::error::{GeometryError, WeightFault};
use super::geometry::Conv2d;
use super::{CudaError, CudaMath, CudaRuntime, KernelModule, LoadedKernels, PtxTier, Sgemm};

mod conv;
mod fbank;
mod lstm;
mod lstmproj;
mod segdense;
mod sinc;
mod wideconv;

#[cfg(test)]
pub(super) use kernel_inventory::conv_kernel_inventory;

#[cfg(test)]
mod kernel_inventory {
    pub(crate) const WIDECONV_KERNELS: [&str; 41] = [
        "spk_wideconv_c128",
        "spk_wideconv_c128s2",
        "spk_wideconv_c256",
        "spk_wideconv_c64s2",
        "spk_wideconv_gemm",
        "spk_wideconv_pack_tc",
        "spk_wideconv_pack_tc3",
        "spk_wideconv_pack_wbf",
        "spk_wideconv_pack_weights",
        "spk_wideconv_pack_winograd",
        "spk_wideconv_pack_wtc",
        "spk_wideconv_reduce",
        "spk_wideconv_shortcut_c128",
        "spk_wideconv_shortcut_c128_wide",
        "spk_wideconv_shortcut_c32",
        "spk_wideconv_shortcut_c64",
        "spk_wideconv_shortcut_c64_wide",
        "spk_wideconv_stem",
        "spk_wideconv_stem_wide",
        "spk_wideconv_tc3_c128s2",
        "spk_wideconv_tc3_c128s2_wide",
        "spk_wideconv_tc3_c64s2",
        "spk_wideconv_tc3_c64s2_wide",
        "spk_wideconv_tc_c128",
        "spk_wideconv_tc_c128s2",
        "spk_wideconv_tc_c128s2_narrow",
        "spk_wideconv_tc_c128s2_slim",
        "spk_wideconv_tc_c256",
        "spk_wideconv_tc_c64s2",
        "spk_wideconv_tc_c64s2_narrow",
        "spk_wideconv_tc_c64s2_slim",
        "spk_wideconv_wbf_c128",
        "spk_wideconv_wbf_c256",
        "spk_wideconv_wino_c128",
        "spk_wideconv_wino_c128_sweep2",
        "spk_wideconv_wino_c256",
        "spk_wideconv_wino_fixup",
        "spk_wideconv_wtc2_c128",
        "spk_wideconv_wtc2_c256",
        "spk_wideconv_wtc3_c128",
        "spk_wideconv_wtc3_c256",
    ];

    /// Every kernel entry a plan can launch, for the PTX inventory check
    pub(crate) const SEGDENSE_KERNELS: [&str; 32] = [
        "spk_segdense_pack",
        "spk_segdense_pack_conv_mma",
        "spk_segdense_conv1_b1",
        "spk_segdense_conv1_b32",
        "spk_segdense_conv1_b32_tc",
        "spk_segdense_conv1_b32_x3",
        "spk_segdense_conv2_b1",
        "spk_segdense_conv2_b32",
        "spk_segdense_conv2_b32_tc",
        "spk_segdense_conv2_b32_x3",
        "spk_segdense_linear0_b1",
        "spk_segdense_linear0_b1_tf32",
        "spk_segdense_linear0_b32",
        "spk_segdense_linear0_b32_tf32",
        "spk_segdense_linear1_b1",
        "spk_segdense_linear1_b1_tf32",
        "spk_segdense_linear1_b32",
        "spk_segdense_linear1_b32_tf32",
        "spk_segdense_classifier_b1",
        "spk_segdense_classifier_b32",
        "spk_segdense_embed_b1",
        "spk_segdense_embed_b32",
        "spk_segdense_embed_b32_tf32",
        "spk_segdense_embed_b32_tf32_k2",
        "spk_segdense_embed_b32_tf32_e64",
        "spk_segdense_embed_b32_x3",
        "spk_segdense_embed_b32_f16",
        "spk_segdense_reduce_embed",
        "spk_segdense_reduce_embed_flat",
        "spk_segdense_reduce_e18",
        "spk_segdense_reduce_e34",
        "spk_segdense_reduce_e36",
    ];

    use super::ConvShape as Shape;
    use super::conv::{REQUIRED_KERNELS, SMALL_BATCH_WAVES, select_tiling};

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
pub(super) use fbank::REQUIRED_KERNELS as FBANK_DFT_KERNELS;
#[cfg(test)]
pub(super) use kernel_inventory::{SEGDENSE_KERNELS, WIDECONV_KERNELS};
#[cfg(test)]
pub(super) use lstm::REQUIRED_KERNELS as LSTM_KERNELS;
#[cfg(test)]
pub(super) use lstmproj::REQUIRED_KERNELS as LSTMPROJ_KERNELS;
#[cfg(test)]
pub(super) use sinc::REQUIRED_KERNELS as SINC_KERNELS;

pub(crate) use conv::Oxide as ConvOxide;
pub(crate) use fbank::Oxide as FbankOxide;
pub(crate) use lstm::Oxide as LstmOxide;
// the routing port selects these plans
pub(crate) use segdense::{Area as SegdenseArea, SegdensePin};
pub(crate) use segdense::{DenseOxide, SegConvOxide};
// the root's routing for builds without libraries consumes this export
pub(crate) use lstmproj::Oxide as LstmProjOxide;
pub(crate) use sinc::Oxide as SincOxide;
// the GPU development checks force selections made for other devices
#[cfg(all(test, feature = "_cuda-libraries"))]
pub(crate) use wideconv::{Config as WideconvConfig, Device as WideconvDevice};
pub(crate) use wideconv::{Oxide as WideconvOxide, Pin as WideconvPin};

/// A planning refusal that is distinct from a CUDA or model error
///
/// Only production outside driver-only mode may fall back to Library, and only for
/// refusals that do not indicate a host bug: a device capability limit, weights
/// outside the candidate's numeric contract, or a valid geometry the candidate does
/// not implement. An invalid geometry or violated plan invariant is a hard error
#[derive(Debug, thiserror::Error)]
pub(crate) enum PlanError {
    /// The device cannot host this candidate, such as a resource limit
    #[error("{reason}")]
    DeviceUnsupported {
        /// The device constraint that prevents this plan
        reason: String,
    },
    /// The weights violate the candidate's numeric contract
    #[error("{layer}: {fault}")]
    WeightsOutOfContract {
        /// The weight tensor that was refused
        layer: &'static str,
        /// The first violation found
        fault: WeightFault,
    },
    /// The requested geometry is invalid for the pin, or valid but unimplemented
    #[error(transparent)]
    Geometry(GeometryError),
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

/// The complete execution choice of one candidate plan
///
/// A plan is constructed from its pin and validates device support; it never
/// reselects a configuration and compares labels afterwards
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ConfigPin {
    /// A ResNet 3x3 convolution
    Conv(ConvPin),
    /// A wide trunk convolution
    Wideconv(WideconvPin),
    /// The four-layer LSTM stack
    Lstm(LstmPin),
    /// The Sinc producer
    Sinc(SincPin),
    /// The filterbank energy producer before log and temporal normalization
    Fbank(FbankPin),
    /// A segmentation convolution, dense head or the embedding projection
    Segdense(SegdensePin),
}

impl ConfigPin {
    /// The candidate area whose kernels execute this pin
    pub(crate) const fn area(self) -> KernelModule {
        match self {
            Self::Conv(_) => KernelModule::Resnet,
            Self::Lstm(LstmPin::LegacyCooperative) => KernelModule::Lstm,
            Self::Lstm(LstmPin::Projected(_)) => KernelModule::LstmProj,
            Self::Wideconv(_) => KernelModule::Wideconv,
            Self::Sinc(_) => KernelModule::Sincnet,
            Self::Fbank(_) => KernelModule::FbankDft,
            Self::Segdense(_) => KernelModule::Segdense,
        }
    }

    /// Whether the pin names a device-dependent selection rule rather than one fixed
    /// configuration; only capability-wide legacy evidence may carry such a rule
    pub(crate) const fn is_device_rule(self) -> bool {
        matches!(
            self,
            Self::Conv(ConvPin::LegacyWaves(_))
                | Self::Wideconv(WideconvPin::DeviceRule)
                | Self::Lstm(LstmPin::LegacyCooperative)
        )
    }
}

/// A complete filterbank producer configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FbankPin {
    /// `fbankdft_fft_mel_accurate`: one 256-thread block per eight frames of a row,
    /// grid `(ceil(998 / 8), batch)`, a packed 512-point real FFT in two-term FP32
    /// expansions with host-f64 twiddles rounded once, and the mel filters as runs of
    /// 16 staged bins; FP32 arithmetic in both math modes
    FftMelAccurate,
}

/// The fixed filterbank geometry and host tables a producer shares with the Library
/// path, so a candidate cannot drift from the window and mel filters it replaces
pub(crate) use super::fbank::{FBANK_FRAMES, FBANK_MEL_BINS, FBANK_WINDOW_SAMPLES, FbankConstants};

/// The supported filterbank batches, independent of model and stress batch sets
pub(crate) const FBANK_BATCHES: [usize; 32] = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
    27, 28, 29, 30, 31, 32,
];

/// A validated `[batch,160000]` waveform producer shape
#[derive(Debug, Clone, Copy)]
pub(crate) struct FbankSpec {
    batch: usize,
    math: CudaMath,
}

impl FbankSpec {
    /// Reject batches outside the boundary's independent production set
    pub(crate) fn new(batch: usize, math: CudaMath) -> Result<Self, PlanError> {
        if !super::implementation::BoundaryId::named("fbank.dft")
            .batches()
            .contains(batch)
        {
            return Err(PlanError::Geometry(GeometryError::Invalid {
                context: "fbank.dft",
                reason: format!("fbank.dft batch {batch} is outside 1..=32"),
            }));
        }
        Ok(Self { batch, math })
    }

    /// Waveform rows in this plan
    pub(crate) const fn batch(self) -> usize {
        self.batch
    }

    /// Library multiply mode used for this comparison
    pub(crate) const fn math(self) -> CudaMath {
        self.math
    }
}

/// A candidate for `fbank.dft`, including framing, windowing and mel projection
///
/// The producer writes every energy in `[B,998,80]`. The locked consumer owns
/// the unchanged log/CMN pass. Enqueue uses only the runtime's stream and registered
/// side streams; a Library control implements the same interface in locked test code
pub(crate) trait FbankCandidate: Sized {
    /// The plan's complete pin; the Library control has no candidate configuration
    type Pin;
    /// Implemented tuples; acceptance remains independent
    const COVERAGE: Coverage;
    /// Numeric contract of the energy producer
    const SPECIAL_VALUES: SpecialValues;
    /// Implemented coverage at the actual loaded tier
    fn coverage(_tier: PtxTier) -> Coverage {
        Self::COVERAGE
    }
    /// The complete configuration used by qualification
    fn implemented_pin(spec: FbankSpec) -> Result<Self::Pin, PlanError>;
    /// Build one validated batch plan from `pin` outside measured intervals, with the
    /// module the selection token already loaded
    fn plan(
        runtime: &CudaRuntime,
        kernels: &LoadedKernels,
        spec: FbankSpec,
        pin: Self::Pin,
    ) -> Result<Self, PlanError>;
    /// Write all energies before the shared log/CMN consumer
    fn enqueue(
        &self,
        waveform: &CudaView<'_, f32>,
        energies: &mut CudaViewMut<'_, f32>,
        phases: &Phases,
        runtime: &CudaRuntime,
    ) -> Result<(), CudaError>;
}

/// A ResNet 3x3 convolution shape with fused kernels: padding 1, no dilation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ConvShape {
    /// 32 -> 32 channels, stride 1
    C32,
    /// 64 -> 64 channels, stride 1
    C64,
    /// 32 -> 64 channels, stride 2
    C32Stride2,
}

/// One fused ResNet kernel entry with its fixed block and tile
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ConvKernel {
    /// `spk_resnet_conv3x3_c32`: 256 threads, 8 output rows by 64 columns
    C32,
    /// `spk_resnet_conv3x3_c64`: 256 threads, 4 rows by 64 columns
    C64,
    /// `spk_resnet_conv3x3_c64_small`: 128 threads, 1 row by 64 columns
    C64Small,
    /// `spk_resnet_conv3x3_c32s2`: 256 threads, 4 rows by 64 columns
    C32Stride2,
    /// `spk_resnet_conv3x3_c32s2_small`: 128 threads, 2 rows by 64 columns
    C32Stride2Small,
    /// `spk_resnet_tc_c32`: TF32 tensor cores, 128 threads, 4 rows by 112 columns;
    /// sm80 tier and TF32 mode only
    C32Tensor,
    /// `spk_resnet_tc_c64`: TF32 tensor cores, 128 threads, 4 rows by 56 columns;
    /// sm80 tier and TF32 mode only
    C64Tensor,
    /// `spk_resnet_tc_c32s2`: TF32 tensor cores, 128 threads, 4 rows by 32 columns;
    /// sm80 tier and TF32 mode only
    C32Stride2Tensor,
}

impl ConvKernel {
    /// The shape this entry computes
    pub(crate) const fn shape(self) -> ConvShape {
        match self {
            Self::C32 | Self::C32Tensor => ConvShape::C32,
            Self::C64 | Self::C64Small | Self::C64Tensor => ConvShape::C64,
            Self::C32Stride2 | Self::C32Stride2Small | Self::C32Stride2Tensor => {
                ConvShape::C32Stride2
            }
        }
    }
}

impl ConvShape {
    /// The shape's TF32 tensor-core entry, which only the sm80 tier exports
    pub(crate) const fn tensor_kernel(self) -> ConvKernel {
        match self {
            Self::C32 => ConvKernel::C32Tensor,
            Self::C64 => ConvKernel::C64Tensor,
            Self::C32Stride2 => ConvKernel::C32Stride2Tensor,
        }
    }
}

/// The execution choice of a ResNet convolution: weights packed once per plan, NCHW
/// activations, FP32 FMA in both math modes except on the TF32 tensor-core entries
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ConvPin {
    /// Exactly this entry and tile
    Kernel(ConvKernel),
    /// The PR #36 rule: the shape's small-block entry when the large entry's grid
    /// gives fewer than two blocks per SM, otherwise the large entry
    LegacyWaves(ConvShape),
}

impl ConvPin {
    /// The shape every entry this pin can run computes
    pub(crate) const fn shape(self) -> ConvShape {
        match self {
            Self::Kernel(kernel) => kernel.shape(),
            Self::LegacyWaves(shape) => shape,
        }
    }
}

/// The execution choice of the LSTM stack
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LstmPin {
    /// The PR #36 stack: locked cuBLAS input projections, `spk_lstm_recurrence` with
    /// 128-thread blocks, and the cooperative tile schedule the device's resident
    /// capacity allows
    LegacyCooperative,
    /// The library-free stack in the `lstmproj` area: the named input projection, then
    /// `spk_lstm_recurrence` with eight units per 256-thread block on the cooperative
    /// tile schedule the device's resident capacity allows
    Projected(LstmProjection),
}

/// The input projection of the library-free LSTM stack
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LstmProjection {
    /// FP32 with a 64 by 64 tile, for row counts too small to fill the larger tiles
    Small,
    /// FP32 with a 128 by 128 tile
    Large,
    /// TF32 matrix fragments with a 128 by 256 tile; needs the sm80 tier
    Tensor,
}

/// The execution choice of the Sinc producer
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SincPin {
    /// `spk_sincnet_conv_abs_pool`: 16-channel groups, 256 pooled outputs per
    /// 256-thread block, filters packed once per plan, pooled output
    ConvAbsPool,
}

/// What a candidate guarantees for special values, against the f64 reference result
/// of its operator
///
/// Each field is a separate contract; none implies another
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SpecialValues {
    /// When finite inputs and weights give finite outputs
    pub finite: FiniteContract,
    /// NaN in an input or weight
    pub nan: NanContract,
    /// An infinite input or weight
    pub infinity: InfinityContract,
    /// The sign of zero outputs
    pub signed_zero: SignedZeroContract,
}

/// The input and weight bound under which finite operands give finite outputs
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FiniteContract {
    /// Filterbank energies remain finite for waveforms with magnitude at most one,
    /// using the fixed 400-sample window, DFT and nonnegative mel filters
    UnitWaveform,
    /// Every output is finite when, for each output, the f64 sum of the absolute
    /// values of all its terms, `Σ|w·x| + |bias| (+ |residual|)`, is at most
    /// `f32::MAX / headroom`. The headroom covers FP32 rounding growth along the
    /// fixed accumulation order
    AbsoluteSum {
        /// Divisor of `f32::MAX`
        headroom: u32,
    },
    /// Outputs are activations bounded by one in magnitude when every gate
    /// pre-activation's absolute term sum is at most `f32::MAX / headroom`
    BoundedActivation {
        /// Divisor of `f32::MAX`
        headroom: u32,
    },
}

/// NaN behaviour
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NanContract {
    /// A NaN term makes every output whose sum contains it NaN
    Propagates,
    /// The pooling maximum ignores a NaN term; a window of only NaN terms gives
    /// negative infinity
    PoolingIgnores,
    /// Not established; NaN operands are outside the contract
    Unspecified,
}

/// Infinity behaviour
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum InfinityContract {
    /// IEEE FP32 arithmetic in the fixed order: an infinite term gives an infinite
    /// sum, and opposite infinities give NaN
    Ieee,
    /// Not established; infinite operands are outside the contract
    Unspecified,
}

/// Signed-zero behaviour
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SignedZeroContract {
    /// The ReLU keeps a negative-zero pre-activation as negative zero
    ReluKeepsNegative,
    /// Outputs are absolute values, so a zero output is positive zero
    Positive,
    /// Not established
    Unspecified,
}
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

/// The operation after a convolution's bias addition
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Epilogue {
    /// Bias only, as used by shortcut convolutions
    Bias,
    /// Bias followed by ReLU
    BiasRelu,
    /// Residual and bias followed by ReLU
    BiasReluResidual,
}

/// One NCHW convolution with folded batch norm and its complete epilogue
#[derive(Debug, Clone, Copy)]
pub(crate) struct ConvLayerSpec<'a> {
    /// The layer name, such as `resnet.layer2.0.conv1`
    pub name: &'a str,
    /// Shape, padding, stride, batch and math mode
    pub conv: Conv2d,
    /// The bias, activation and residual operation after convolution
    pub epilogue: Epilogue,
    /// Folded weights `[k, c, kernel_height, kernel_width]`
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
    /// Folded weights `[k, c, kernel_height, kernel_width]`
    pub weight: &'a CudaView<'b, f32>,
    /// Folded bias `[k]`
    pub bias: &'a CudaView<'b, f32>,
}

/// A candidate for convolution and the spec's complete bias, activation and residual epilogue
pub(crate) trait ConvCandidate: Sized {
    /// The complete configuration, or unit for the Library control
    type Pin;
    /// Convolution names and batch sizes this candidate implements
    const COVERAGE: Coverage;

    /// The special-value contract of every plan
    const SPECIAL_VALUES: SpecialValues;

    /// Coverage for the actual loaded tier; override when variants cover different tuples
    fn coverage(_tier: PtxTier) -> Coverage {
        Self::COVERAGE
    }

    /// The configuration qualification plans for this layer
    fn implemented_pin(layer: &ConvLayerSpec<'_>) -> Result<Self::Pin, PlanError>;

    /// Prepares one layer for one batch size from `pin`; runs once per batch class,
    /// untimed
    fn plan(
        runtime: &CudaRuntime,
        kernels: &LoadedKernels,
        layer: ConvLayerSpec<'_>,
        pin: Self::Pin,
    ) -> Result<Self, PlanError>;

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

    /// The special-value contract of every plan
    const SPECIAL_VALUES: SpecialValues;

    /// Coverage for the actual loaded tier; override when variants cover different tuples
    fn coverage(_tier: PtxTier) -> Coverage {
        Self::COVERAGE
    }
    /// The tensor [`Self::enqueue`] writes
    const OUTPUT: SincOutput;

    /// The configuration qualification plans for this batch size
    fn implemented_pin(spec: &SincSpec<'_>) -> Result<SincPin, PlanError>;

    /// Prepares one batch size from `pin`; runs once per batch class, untimed
    fn plan(
        runtime: &CudaRuntime,
        kernels: &LoadedKernels,
        spec: SincSpec<'_>,
        pin: SincPin,
    ) -> Result<Self, PlanError>;

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

    /// The special-value contract of every plan
    const SPECIAL_VALUES: SpecialValues;

    /// Coverage for the actual loaded tier; override when variants cover different tuples
    fn coverage(_tier: PtxTier) -> Coverage {
        Self::COVERAGE
    }

    /// The configuration qualification plans for this batch size
    fn implemented_pin(spec: &LstmSpec<'_>) -> Result<LstmPin, PlanError>;

    /// The configuration to run on `device` with the loaded `tier` when no accepted
    /// record names one, such as in a build without libraries; a deterministic rule
    fn device_pin(
        device: &DeviceAttributes,
        tier: PtxTier,
        spec: &LstmSpec<'_>,
    ) -> Result<LstmPin, PlanError>;

    /// Prepares one batch size from `pin`; runs once per batch class, untimed
    fn plan(
        runtime: &CudaRuntime,
        kernels: &LoadedKernels,
        spec: LstmSpec<'_>,
        pin: LstmPin,
    ) -> Result<Self, PlanError>;

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

/// Typed phases inside a candidate's `enqueue`
#[derive(Debug)]
pub(crate) struct Phases(());

impl Phases {
    pub(super) fn new() -> Self {
        Self(())
    }

    /// Run the enqueue operation for one typed phase
    pub(crate) fn op<T>(
        &self,
        _op: Op,
        enqueue: impl FnOnce() -> Result<T, CudaError>,
    ) -> Result<T, CudaError> {
        enqueue()
    }
}

/// Phases of the LSTM stack, with a projection helper and checked layer indices
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

    /// Run the enqueue operation for one typed phase
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
        _direction: Direction,
        enqueue: impl FnOnce(&Projection<'_>) -> Result<T, CudaError>,
    ) -> Result<T, CudaError> {
        check_layer(layer)?;
        enqueue(&self.projection)
    }

    /// Runs the recurrence of `layer` in `direction`
    pub(crate) fn recurrence<T>(
        &self,
        layer: usize,
        _direction: Direction,
        enqueue: impl FnOnce() -> Result<T, CudaError>,
    ) -> Result<T, CudaError> {
        check_layer(layer)?;
        enqueue()
    }
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
/// Fork and join events keep side-stream work ordered with the main stream
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
/// Direct Library comparisons and the pinned production stack use this helper
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
        _direction: Direction,
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
        CudaViewMut, FiniteContract, InfinityContract, LoadedKernels, Maths, NanContract, Phases,
        PlanError, PtxTier, SignedZeroContract, SincCandidate, SincInputs, SincOutput, SincPin,
        SincSpec, SpecialValues,
    };

    struct TierFixture;

    impl SincCandidate for TierFixture {
        const COVERAGE: Coverage = Coverage::NONE;
        const SPECIAL_VALUES: SpecialValues = SpecialValues {
            finite: FiniteContract::AbsoluteSum { headroom: 2 },
            nan: NanContract::Unspecified,
            infinity: InfinityContract::Unspecified,
            signed_zero: SignedZeroContract::Unspecified,
        };
        const OUTPUT: SincOutput = SincOutput::Pooled;

        fn implemented_pin(_spec: &SincSpec<'_>) -> Result<SincPin, PlanError> {
            Ok(SincPin::ConvAbsPool)
        }

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

        fn plan(
            _runtime: &CudaRuntime,
            _kernels: &LoadedKernels,
            _spec: SincSpec<'_>,
            _pin: SincPin,
        ) -> Result<Self, PlanError> {
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

#[cfg(all(test, feature = "_cuda-libraries"))]
#[path = "candidate_tests.rs"]
mod candidate_tests;

/// A fixed dense boundary in the loaded model
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DenseSite {
    /// First segmentation linear layer
    Linear0,
    /// Second segmentation linear layer
    Linear1,
    /// Segmentation classifier
    Classifier,
    /// Embedding projection
    Embedding,
}

impl DenseSite {
    /// The fixed model boundary implemented by this site
    pub(crate) fn boundary(self) -> super::implementation::BoundaryId {
        super::implementation::BoundaryId::named(match self {
            Self::Linear0 => "linear0",
            Self::Linear1 => "linear1",
            Self::Classifier => "linear2",
            Self::Embedding => "resnet.seg_1",
        })
    }
}

/// The operation after dense multiplication
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DenseEpilogue {
    /// Add bias only
    Bias,
    /// Add bias and apply leaky ReLU
    BiasLeakyRelu,
    /// Add bias and apply row log softmax
    BiasLogSoftmax,
}

/// Validated dense geometry; weights and epilogue follow the fixed site
#[derive(Debug, Clone, Copy)]
pub(crate) struct DenseSpec {
    site: DenseSite,
    batch: usize,
    math: CudaMath,
}

impl DenseSpec {
    /// Build a fixed model shape with a positive batch and bounded buffer sizes
    pub(crate) fn new(site: DenseSite, batch: usize, math: CudaMath) -> Result<Self, PlanError> {
        let spec = Self { site, batch, math };
        let (m, n, k) = spec.dimensions();
        validate_fixed_batch("dense plan", batch, &[m, n, k, m * n, m * k, n * k])?;
        Ok(spec)
    }

    /// Refuse a valid variable-length window that this fixed kernel does not implement
    pub(crate) fn check_rows(self, rows: usize) -> Result<(), PlanError> {
        if rows == self.dimensions().0 {
            return Ok(());
        }
        Err(PlanError::Geometry(GeometryError::Unimplemented {
            context: "dense plan",
            reason: format!(
                "{} requires {} rows per item, got {rows}",
                self.site.boundary(),
                self.dimensions().0
            ),
        }))
    }

    /// The fixed model boundary
    pub(crate) const fn site(self) -> DenseSite {
        self.site
    }
    /// Items in this plan
    pub(crate) const fn batch(self) -> usize {
        self.batch
    }
    /// Library multiply mode
    pub(crate) const fn math(self) -> CudaMath {
        self.math
    }
    /// Rows, output columns and input columns per batch item
    pub(crate) const fn dimensions(self) -> (usize, usize, usize) {
        match self.site {
            DenseSite::Linear0 => (589, 128, 256),
            DenseSite::Linear1 => (589, 128, 128),
            DenseSite::Classifier => (589, 7, 128),
            DenseSite::Embedding => (3, 256, 5120),
        }
    }
    /// The fixed operation after multiplication
    pub(crate) const fn epilogue(self) -> DenseEpilogue {
        match self.site {
            DenseSite::Linear0 | DenseSite::Linear1 => DenseEpilogue::BiasLeakyRelu,
            DenseSite::Classifier => DenseEpilogue::BiasLogSoftmax,
            DenseSite::Embedding => DenseEpilogue::Bias,
        }
    }
    /// Whether weights use transposed `[n,k]` storage and output starts with bias
    pub(crate) const fn transposed_weights(self) -> bool {
        matches!(self.site, DenseSite::Embedding)
    }
    /// GEMM beta after seeding the output bias; other sites add bias afterwards
    pub(crate) const fn beta(self) -> f32 {
        if self.transposed_weights() { 1.0 } else { 0.0 }
    }
    /// Input buffer elements
    pub(crate) fn input_len(self) -> usize {
        let (m, _, k) = self.dimensions();
        self.batch * m * k
    }
    /// Shared weight buffer elements
    pub(crate) fn weight_len(self) -> usize {
        let (_, n, k) = self.dimensions();
        n * k
    }
    /// Shared bias buffer elements
    pub(crate) fn bias_len(self) -> usize {
        self.dimensions().1
    }
    /// Output buffer elements
    pub(crate) fn output_len(self) -> usize {
        let (m, n, _) = self.dimensions();
        self.batch * m * n
    }
}

/// A complete dense operator, including its fixed bias and activation epilogue
pub(crate) trait DenseCandidate: Sized {
    /// Complete candidate configuration, or unit for Library
    type Pin;
    /// Implemented tuples, independent of production acceptance
    const COVERAGE: Coverage;
    /// Numeric contract of the complete operator
    const SPECIAL_VALUES: SpecialValues;
    /// Implemented coverage at the actual loaded tier
    fn coverage(_tier: PtxTier) -> Coverage {
        Self::COVERAGE
    }
    /// The configuration a deterministic rule picks for the loaded module's `tier`
    /// and `device`
    fn implemented_pin(
        spec: DenseSpec,
        tier: PtxTier,
        device: &DeviceAttributes,
    ) -> Result<Self::Pin, PlanError>;
    /// Build a plan outside measured intervals; weights may be packed here once
    fn plan(
        runtime: &CudaRuntime,
        kernels: &LoadedKernels,
        spec: DenseSpec,
        weight: &CudaSlice<f32>,
        bias: &CudaSlice<f32>,
        pin: Self::Pin,
    ) -> Result<Self, PlanError>;
    /// Write the complete output on the runtime stream, with the planned weights
    fn enqueue(
        &self,
        x: &CudaView<'_, f32>,
        output: &mut CudaViewMut<'_, f32>,
        phases: &Phases,
        runtime: &CudaRuntime,
    ) -> Result<(), CudaError>;
}

/// A fixed segmentation convolution producer
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SegConvSite {
    /// The 80 to 60 channel convolution
    Conv1,
    /// The 60 to 60 channel convolution
    Conv2,
}

impl SegConvSite {
    /// The fixed model boundary implemented by this site
    pub(crate) fn boundary(self) -> super::implementation::BoundaryId {
        super::implementation::BoundaryId::named(match self {
            Self::Conv1 => "sincnet.conv1",
            Self::Conv2 => "sincnet.conv2",
        })
    }
}

/// Validated five-tap, stride-one NCW convolution without bias
#[derive(Debug, Clone, Copy)]
pub(crate) struct SegConvSpec {
    site: SegConvSite,
    batch: usize,
    math: CudaMath,
}

impl SegConvSpec {
    /// Build a fixed model shape with a positive batch and bounded buffer sizes
    pub(crate) fn new(site: SegConvSite, batch: usize, math: CudaMath) -> Result<Self, PlanError> {
        let spec = Self { site, batch, math };
        validate_fixed_batch(
            "segmentation conv plan",
            batch,
            &[
                spec.in_channels() * spec.input_steps(),
                spec.out_channels() * spec.output_steps(),
            ],
        )?;
        Ok(spec)
    }
    /// Verify that the model's temporal shape is the fixed kernel shape
    pub(crate) fn check_conv(self, conv: Conv2d) -> Result<(), PlanError> {
        if conv.batch == self.batch
            && conv.math == self.math
            && conv.in_channels == self.in_channels()
            && conv.out_channels == self.out_channels()
            && conv.input == [1, self.input_steps()]
            && conv.kernel == [1, self.kernel()]
            && conv.padding == [0, 0]
            && conv.stride == [1, 1]
            && conv.dilation == [1, 1]
        {
            return Ok(());
        }
        Err(PlanError::Geometry(GeometryError::Unimplemented {
            context: "temporal plan",
            reason: format!("{} does not implement shape {conv:?}", self.site.boundary()),
        }))
    }

    /// The fixed model boundary
    pub(crate) const fn site(self) -> SegConvSite {
        self.site
    }
    /// Items in this plan
    pub(crate) const fn batch(self) -> usize {
        self.batch
    }
    /// Library multiply mode
    pub(crate) const fn math(self) -> CudaMath {
        self.math
    }
    /// Input channels
    pub(crate) const fn in_channels(self) -> usize {
        match self.site {
            SegConvSite::Conv1 => 80,
            SegConvSite::Conv2 => 60,
        }
    }
    /// Output channels
    pub(crate) const fn out_channels(self) -> usize {
        60
    }
    /// Input steps per channel
    pub(crate) const fn input_steps(self) -> usize {
        match self.site {
            SegConvSite::Conv1 => 5325,
            SegConvSite::Conv2 => 1773,
        }
    }
    /// Output steps per channel
    pub(crate) const fn output_steps(self) -> usize {
        self.input_steps() - 4
    }
    /// Filter taps
    pub(crate) const fn kernel(self) -> usize {
        5
    }
    /// Input buffer elements in NCW layout
    pub(crate) fn input_len(self) -> usize {
        self.batch * self.in_channels() * self.input_steps()
    }
    /// Shared weights in output-channel, input-channel, tap order
    pub(crate) fn weight_len(self) -> usize {
        self.out_channels() * self.in_channels() * self.kernel()
    }
    /// Output buffer elements in NCW layout
    pub(crate) fn output_len(self) -> usize {
        self.batch * self.out_channels() * self.output_steps()
    }
}

/// A raw segmentation convolution; the shared pool consumer owns bias
pub(crate) trait SegConvCandidate: Sized {
    /// Complete candidate configuration, or unit for Library
    type Pin;
    /// Implemented tuples, independent of production acceptance
    const COVERAGE: Coverage;
    /// Numeric contract of the producer
    const SPECIAL_VALUES: SpecialValues;
    /// Implemented coverage at the actual loaded tier
    fn coverage(_tier: PtxTier) -> Coverage {
        Self::COVERAGE
    }
    /// The configuration a deterministic rule picks for the loaded module's `tier`
    /// and `device`
    fn implemented_pin(
        spec: SegConvSpec,
        tier: PtxTier,
        device: &DeviceAttributes,
    ) -> Result<Self::Pin, PlanError>;
    /// Build a plan outside measured intervals; weights may be packed here once
    fn plan(
        runtime: &CudaRuntime,
        kernels: &LoadedKernels,
        spec: SegConvSpec,
        weight: &CudaSlice<f32>,
        pin: Self::Pin,
    ) -> Result<Self, PlanError>;
    /// Write every raw NCW output value, without bias, on the runtime stream, with
    /// the planned weights
    fn enqueue(
        &self,
        x: &CudaView<'_, f32>,
        output: &mut CudaViewMut<'_, f32>,
        phases: &Phases,
        runtime: &CudaRuntime,
    ) -> Result<(), CudaError>;
}

fn validate_fixed_batch(
    context: &'static str,
    batch: usize,
    lengths: &[usize],
) -> Result<(), PlanError> {
    if batch > 0
        && lengths.iter().all(|length| {
            batch
                .checked_mul(*length)
                .is_some_and(|value| value <= i32::MAX as usize)
        })
    {
        return Ok(());
    }

    Err(PlanError::Geometry(GeometryError::Invalid {
        context,
        reason: format!("batch {batch} is zero or exceeds the buffer index range"),
    }))
}

/// Library-free coverage and a deterministic configuration chosen before module loading
///
/// Area ports implement this next to their candidate trait implementation and register
/// it in `implementation::driver::areas`. Library-dependent legacy plans do not opt in
pub(crate) trait DriverCandidate {
    /// The kernel module that owns this candidate
    const AREA: KernelModule;
    /// Only tuples whose complete operation needs no numerical library
    fn driver_coverage(tier: PtxTier) -> Coverage;
    /// Structural speed evidence, if this complete port is accepted on all devices
    fn broad_evidence() -> Option<&'static super::implementation::BroadEvidence> {
        None
    }
    /// Speed policy of the complete port, separate from implemented coverage
    fn speed_scope(
        _boundary: super::implementation::BoundaryId,
        _batch: usize,
        _math: CudaMath,
        _device: &DeviceAttributes,
        _tier: PtxTier,
    ) -> Option<super::implementation::SpeedScope> {
        Self::broad_evidence().map(super::implementation::SpeedScope::AllDevices)
    }
    /// The development reports supporting this port's speed policy
    fn speed_summary(_math: CudaMath) -> &'static str {
        Self::broad_evidence().map_or("device-sensitive port measurements", |evidence| {
            evidence.summary()
        })
    }
    /// One complete pin, selected from cached device facts without GPU allocation
    fn driver_pin(
        boundary: super::implementation::BoundaryId,
        batch: usize,
        math: CudaMath,
        device: &super::device::DeviceAttributes,
        tier: PtxTier,
    ) -> Result<ConfigPin, PlanError>;
}
