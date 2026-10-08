//! The wide convolution candidate: the trunk convolutions outside the PR #36 shapes
//!
//! It covers the stem, the three 1x1 stride-2 shortcuts, the 64 -> 128 and 128 -> 256
//! stride-2 layers and the 128- and 256-channel layers, each with its complete bias,
//! ReLU and residual epilogue on the same NCHW buffers as the cuDNN plan. Kernels live
//! in `crates/speakrs-cuda-kernels/src/wideconv.rs`: spatial tiles, a fused Winograd
//! F(2x2, 3x3) with FFMA, 3xTF32, 2xTF32 or BF16x3 products, and `mma.sync` direct
//! tiles in the sm80 PTX tier
//!
//! A plan runs one [`Config`]: a kernel family plus an input-channel partition whose
//! planes a fixed-order pass reduces, never atomics. [`Config::select`] chooses it as a
//! deterministic function of the device attributes, the loaded PTX tier and the
//! convolution, from development timings on an RTX 4060 Ti, an RTX 5060 Ti and an
//! A100. Partitioned plans own their partial-sum planes, allocated in `plan`, so
//! enqueueing allocates nothing and is capturable

use cudarc::driver::sys::CUfunction_attribute;
use cudarc::driver::{
    CudaFunction, CudaSlice, CudaStream, CudaView, CudaViewMut, DevicePtr, LaunchConfig,
    PushKernelArg,
};

use super::{
    Batches, ConvCandidate, ConvInputs, ConvLayerSpec, Coverage, CoverageEntry, Epilogue,
    FiniteContract, GeometryError, InfinityContract, Maths, NanContract, Op, Phases, PlanError,
    Scratch, SignedZeroContract, SpecialValues,
};
use crate::inference::cuda::device::DeviceAttributes;
use crate::inference::cuda::error::{check_len, element_count};
use crate::inference::cuda::geometry::Conv2d;
use crate::inference::cuda::{
    ComputeCapability, CudaError, CudaMath, CudaRuntime, LoadedKernels, PtxTier,
};

// keep launch names and the shipped-entry inventory in the same definition
macro_rules! kernel_entries {
    ($($name:ident => $entry:literal),+ $(,)?) => {
        pub(super) mod entries {
            $(pub(super) const $name: &str = $entry;)+
            #[cfg(test)]
            pub(crate) const ALL: &[&str] = &[$($name),+];
        }
    };
}

kernel_entries! {
    C128 => "spk_wideconv_c128",
    C128S2 => "spk_wideconv_c128s2",
    C256 => "spk_wideconv_c256",
    C64S2 => "spk_wideconv_c64s2",
    GEMM => "spk_wideconv_gemm",
    PACK_TC => "spk_wideconv_pack_tc",
    PACK_TC3 => "spk_wideconv_pack_tc3",
    PACK_WBF => "spk_wideconv_pack_wbf",
    PACK_WEIGHTS => "spk_wideconv_pack_weights",
    PACK_WINOGRAD => "spk_wideconv_pack_winograd",
    PACK_WTC => "spk_wideconv_pack_wtc",
    REDUCE => "spk_wideconv_reduce",
    SHORTCUT_C128 => "spk_wideconv_shortcut_c128",
    SHORTCUT_C128_WIDE => "spk_wideconv_shortcut_c128_wide",
    SHORTCUT_C32 => "spk_wideconv_shortcut_c32",
    SHORTCUT_C64 => "spk_wideconv_shortcut_c64",
    SHORTCUT_C64_WIDE => "spk_wideconv_shortcut_c64_wide",
    STEM => "spk_wideconv_stem",
    STEM_WIDE => "spk_wideconv_stem_wide",
    TC3_C128S2 => "spk_wideconv_tc3_c128s2",
    TC3_C128S2_WIDE => "spk_wideconv_tc3_c128s2_wide",
    TC3_C64S2 => "spk_wideconv_tc3_c64s2",
    TC3_C64S2_WIDE => "spk_wideconv_tc3_c64s2_wide",
    TC_C128 => "spk_wideconv_tc_c128",
    TC_C128S2 => "spk_wideconv_tc_c128s2",
    TC_C128S2_NARROW => "spk_wideconv_tc_c128s2_narrow",
    TC_C128S2_SLIM => "spk_wideconv_tc_c128s2_slim",
    TC_C256 => "spk_wideconv_tc_c256",
    TC_C64S2 => "spk_wideconv_tc_c64s2",
    TC_C64S2_NARROW => "spk_wideconv_tc_c64s2_narrow",
    TC_C64S2_SLIM => "spk_wideconv_tc_c64s2_slim",
    WBF_C128 => "spk_wideconv_wbf_c128",
    WBF_C256 => "spk_wideconv_wbf_c256",
    WINO_C128 => "spk_wideconv_wino_c128",
    WINO_C128_SWEEP2 => "spk_wideconv_wino_c128_sweep2",
    WINO_C256 => "spk_wideconv_wino_c256",
    WINO_C64 => "spk_wideconv_wino_c64",
    WINO_FIXUP => "spk_wideconv_wino_fixup",
    WTC1_C128 => "spk_wideconv_wtc1_c128",
    WTC1_C256 => "spk_wideconv_wtc1_c256",
    WTC2_C128 => "spk_wideconv_wtc2_c128",
    WTC2_C256 => "spk_wideconv_wtc2_c256",
    WTC3_C128 => "spk_wideconv_wtc3_c128",
    WTC3_C256 => "spk_wideconv_wtc3_c256",
    WTP1_C128 => "spk_wideconv_wtp1_c128",
    WTP1_C256 => "spk_wideconv_wtp1_c256",
    H16_C128 => "spk_wideconv_h16_c128",
    H16_C128_NARROW => "spk_wideconv_h16_c128_narrow",
    H16_C256 => "spk_wideconv_h16_c256",
    H16_C256_NARROW => "spk_wideconv_h16_c256_narrow",
    H16_C32 => "spk_wideconv_h16_c32",
    H16_C64 => "spk_wideconv_h16_c64",
    PACK_H16 => "spk_wideconv_pack_h16",
}

/// The 22 trunk convolutions with a wideconv kernel, in trunk order
const LAYERS: [&str; 22] = [
    "resnet.conv1",
    "resnet.layer2.0.shortcut.0",
    "resnet.layer3.0.conv1",
    "resnet.layer3.0.conv2",
    "resnet.layer3.0.shortcut.0",
    "resnet.layer3.1.conv1",
    "resnet.layer3.1.conv2",
    "resnet.layer3.2.conv1",
    "resnet.layer3.2.conv2",
    "resnet.layer3.3.conv1",
    "resnet.layer3.3.conv2",
    "resnet.layer3.4.conv1",
    "resnet.layer3.4.conv2",
    "resnet.layer3.5.conv1",
    "resnet.layer3.5.conv2",
    "resnet.layer4.0.conv1",
    "resnet.layer4.0.conv2",
    "resnet.layer4.0.shortcut.0",
    "resnet.layer4.1.conv1",
    "resnet.layer4.1.conv2",
    "resnet.layer4.2.conv1",
    "resnet.layer4.2.conv2",
];

/// The 64-channel same-shape trunk convolutions on Turing or a measured FP16 recipe
///
/// On a T4 at batch 32 the direct ResNet kernel ran these at 0.85-0.89x of cuDNN,
/// which picks its non-fused Winograd there; FFMA Winograd F(2x2, 3x3) cuts their
/// multiplies 2.25x, so Turing runs them here in FP32 mode too. TF32 mode takes the
/// FP16 tiles. Other parts keep the direct kernel, so their routes do not move
const C64_LAYERS: [&str; 7] = [
    "resnet.layer2.0.conv2",
    "resnet.layer2.1.conv1",
    "resnet.layer2.1.conv2",
    "resnet.layer2.2.conv1",
    "resnet.layer2.2.conv2",
    "resnet.layer2.3.conv1",
    "resnet.layer2.3.conv2",
];

/// The 32-channel same-shape trunk convolutions, which wideconv runs in TF32 mode only,
/// with direct FP16 tensor-core tiles on Turing or a measured FP16 recipe
///
/// FP32 mode keeps the direct ResNet kernel
const C32_LAYERS: [&str; 6] = [
    "resnet.layer1.0.conv1",
    "resnet.layer1.0.conv2",
    "resnet.layer1.1.conv1",
    "resnet.layer1.1.conv2",
    "resnet.layer1.2.conv1",
    "resnet.layer1.2.conv2",
];

/// The capability whose FP32 mode runs `C64_LAYERS` here, in FFMA Winograd
const TURING: ComputeCapability = ComputeCapability::new(7, 5);

/// Fixed input-channel partitions, reduced without atomics
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Partition {
    Whole,
    Two,
    Four,
    Eight,
}

impl Partition {
    /// Input-channel partitions, each reduced by its own CTAs
    pub(crate) fn count(self) -> u32 {
        match self {
            Self::Whole => 1,
            Self::Two => 2,
            Self::Four => 4,
            Self::Eight => 8,
        }
    }
}

/// Which cells of a Winograd launch split their input channels into the partition's
/// planes; every other algorithm splits all of its tiles
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SplitCells {
    /// Every cell splits
    All,
    /// Cells from this index on split; earlier cells, whole waves of CTAs, reduce
    /// every input channel themselves
    From(u16),
}

/// Spatial window reuse or implicit matrix tiling, chosen explicitly in development
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Algorithm {
    Spatial,
    /// The stem's spatial tiles with eight pixels per thread of a 256-thread CTA and
    /// streaming stores; stem only
    WideStem,
    ImplicitGemm,
    /// `mma.sync` implicit GEMM for the 3x3 wide convolutions, in the products and
    /// tiles of `TensorKernel`; sm80 tier only
    TensorCore(TensorKernel),
    /// Fused Winograd F(2x2, 3x3) for the same-channel stride-1 3x3 convolutions, with
    /// the element-wise products of `WinogradProducts`
    Winograd(WinogradProducts),
    /// Direct `mma.sync` implicit GEMM for the same-channel stride-1 3x3 convolutions
    /// with FP16 operands scaled by 2^10 and FP32 accumulation, in the output-channel
    /// tiles of `Fp16Tiles`; TF32 mode only, every tier
    Fp16(Fp16Tiles),
}

/// Output channels per CTA of a direct FP16 kernel
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Fp16Tiles {
    /// Every channel of the 32- and 64-channel shapes, 128 of the wider ones
    Wide,
    /// 64 channels of the 128- and 256-channel shapes, which doubles the CTAs of a batch
    Narrow,
}

/// Largest operand magnitude the FP16 tiles convert without saturating: the largest
/// finite FP16 value, 65504, over the 2^10 operand scale
pub(crate) const FP16_OPERAND_LIMIT: f32 = 65504.0 / 1024.0;

/// Whether selection may choose FP16 tiles for a boundary
///
/// FP16 tiles saturate any operand above [`FP16_OPERAND_LIMIT`], which would silently
/// change the convolution, so a boundary whose weights exceed it, and the recomputation
/// of a batch whose activations did, select as if FP16 tiles did not exist
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) enum Fp16Policy {
    /// Recipes, defaults and tune files may choose FP16 tiles
    #[default]
    Allowed,
    /// Every source takes the choice it makes without FP16 tiles
    Excluded,
}

impl Fp16Policy {
    /// Excluded when any weight is above [`FP16_OPERAND_LIMIT`] in magnitude or is NaN
    ///
    /// The FP16 kernels pack the raw folded weights, so these are the values they convert
    pub(crate) fn of_weights(weights: &[f32]) -> Self {
        if weights
            .iter()
            .all(|weight| weight.abs() <= FP16_OPERAND_LIMIT)
        {
            Self::Allowed
        } else {
            Self::Excluded
        }
    }

    pub(crate) const fn allows(self) -> bool {
        matches!(self, Self::Allowed)
    }
}

/// Products and stride-2 tiles of a direct tensor-core kernel
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TensorKernel {
    /// One TF32 product, both operands rounded; TF32 mode only, every 3x3 wide shape.
    /// Stride-2 tiles cover 128 output columns from `TC_WIDE_BATCH`, narrower ones below
    Tf32,
    /// As `Tf32` on stride-2 shapes only, with slim tiles: twice the CTAs of the narrow
    /// tiles, for parts with more SMs than narrow tiles
    Tf32Slim,
    /// Three TF32 products per term (3xTF32) with FP32-level error; both math modes,
    /// stride-2 shapes only, 32-column tiles with the tap loop rolled. Selected for
    /// FP32-mode stride-2 layers of TF32-rich parts below `TC_WIDE_BATCH`
    Tf32x3,
    /// As `Tf32x3` with 48-column tiles, selected from `TC_WIDE_BATCH`
    Tf32x3Wide,
}

impl TensorKernel {
    /// Whether the kernel splits both operands for FP32-level error
    fn split(self) -> bool {
        matches!(self, Self::Tf32x3 | Self::Tf32x3Wide)
    }
}

/// How a fused Winograd kernel forms its element-wise products
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WinogradProducts {
    /// FFMA in FP32; every tier and math mode
    Fp32,
    /// As `Fp32`, with the input channels in two sweeps whose output-domain partial
    /// sums join in order, which halves the FP32 accumulation chains; 128 channels only
    Fp32Sweep2,
    /// Three TF32 tensor-core products per term (3xTF32) with FP32-level error; sm80
    /// tier, both math modes
    Tf32x3,
    /// Two TF32 tensor-core products per term, weights split and transformed inputs
    /// rounded; sm80 tier, TF32 mode only
    Tf32x2,
    /// One TF32 tensor-core product per term, both operands rounded; sm80 tier, TF32
    /// mode only. Not selected: `Tf32x1Staged` ran faster on every measured part;
    /// forced runs keep it as the unstaged baseline
    Tf32x1,
    /// As `Tf32x1` with three raw stages and the warps in two phases, so the input
    /// transform overlaps the products and a chunk takes one barrier
    Tf32x1Staged,
    /// Three BF16 tensor-core products per term, both operands split into high and low
    /// BF16 parts; sm80 tier, TF32 mode only. Not selected yet: it awaits an A100
    /// timing; forced runs test it
    Bf16x3,
}

impl WinogradProducts {
    fn tensor(self) -> bool {
        !matches!(self, Self::Fp32 | Self::Fp32Sweep2)
    }

    /// Dynamic shared bytes of a launch, as `WINO_SHARED_BYTES`, `WTC_SHARED_BYTES` and
    /// `WTP_SHARED_BYTES`
    fn shared_bytes(self) -> u32 {
        match self {
            Self::Fp32 | Self::Fp32Sweep2 => 62_464,
            Self::Tf32x3 | Self::Tf32x2 | Self::Tf32x1 => 66_560,
            Self::Tf32x1Staged => 71_168,
            Self::Bf16x3 => 88_064,
        }
    }

    /// Input channels every partition covers a whole multiple of: one pipeline stage,
    /// or one per sweep for the swept kernels
    fn chunk(self) -> usize {
        match self {
            Self::Fp32 => 4,
            Self::Fp32Sweep2 => 8,
            Self::Tf32x3 | Self::Tf32x2 | Self::Tf32x1 | Self::Tf32x1Staged => 8,
            Self::Bf16x3 => 16,
        }
    }
}

/// Output channels per tensor-core CTA, as `TC_CHANNELS` in the device crate
const TC_CHANNELS: u32 = 128;
/// Flattened output pixels per stride-1 tensor-core CTA, as `TC_PIXELS`
const TC_PIXELS: u32 = 112;
/// Output columns per stride-2 tensor-core CTA, as `TC_COLUMNS`
const TC_COLUMNS: u32 = 128;
/// Output columns per narrow stride-2 tensor-core CTA, as `TC_NARROW_COLUMNS`
const TC_NARROW_COLUMNS: u32 = 96;
/// Output columns per narrow 128 -> 256 stride-2 tensor-core CTA, as
/// `TC_C128S2_NARROW_COLUMNS`
const TC_C128S2_NARROW_COLUMNS: u32 = 64;
/// Output columns per 3xTF32 stride-2 CTA, as `TC3_COLUMNS`
const TC3_COLUMNS: u32 = 32;
/// Output columns per wide 3xTF32 stride-2 CTA, as `TC3_WIDE_COLUMNS`
const TC3_WIDE_COLUMNS: u32 = 48;
/// Output columns per slim 64 -> 128 stride-2 CTA, as `TC_C64S2_SLIM_COLUMNS`
const TC_C64S2_SLIM_COLUMNS: u32 = 64;
/// Output columns per slim 128 -> 256 stride-2 CTA, as `TC_C128S2_SLIM_COLUMNS`
const TC_C128S2_SLIM_COLUMNS: u32 = 32;
/// Batch from which the stride-2 tensor-core layers use 128-column tiles
///
/// Narrow tiles balance batch 1 across the SMs, and wide tiles win at batch 32; the
/// crossover between them is not measured
const TC_WIDE_BATCH: u32 = 8;
/// Output pixels per shortcut CTA, as `SHORTCUT_PIXELS` in the device crate
const SHORTCUT_PIXELS: u32 = 64;
/// Batch from which shortcut tiles cover 128 output channels instead of 64
///
/// Production runs batches 1 and 32, where 64- and 128-channel tiles win
/// respectively; the crossover between them is not measured
const SHORTCUT_WIDE_BATCH: u32 = 8;
/// Output channels per Winograd CTA, as `WINO_CHANNELS` in the device crate
const WINO_CHANNELS: u32 = 64;
/// 2x2 output tiles per Winograd CTA, consecutive in one tile row, as `WINO_TILES`
const WINO_TILES: u32 = 32;
/// Threads per Winograd CTA
const WINO_THREADS: u32 = 256;
/// Threads per wide stem CTA, as in `spk_wideconv_stem_wide`
const WIDE_STEM_THREADS: u32 = 256;
/// Pixels per wide stem thread, as in `spk_wideconv_stem_wide`
const WIDE_STEM_PIXELS: u32 = 8;
/// Waves of wide stem CTAs from which the stem uses them
///
/// The stem only writes, and with many CTAs DRAM bounds it: on a 4060 Ti at batch 32
/// (1248 wide CTAs, 37 waves) the wide tiles ran 1.255 ms against 1.303 ms for the narrow
/// ones and cuDNN's 1.288 ms. At batch 1 their 39 CTAs filled about one wave and left
/// the SMs latency-bound, 0.032 against 0.013 ms; four times as many narrow CTAs hide
/// the latency. The crossover between them is not measured
const WIDE_STEM_WAVES: u32 = 8;
/// Output columns per direct FP16 CTA, as `8 * n_tiles` in the device crate
const FP16_COLUMNS: u32 = 64;
/// Waves of wide FP16 CTAs below which the 128- and 256-channel shapes use narrow ones
const FP16_WIDE_WAVES: u32 = 2;

/// Largest dynamic shared allocation a launch may request without opting in
const DEFAULT_SHARED_LIMIT: u32 = 48 * 1024;

/// Device attributes that select a wideconv configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Device {
    /// Compute capability of the GPU
    pub(crate) capability: ComputeCapability,
    /// Streaming multiprocessors
    pub(crate) sms: u32,
    /// PTX tier of the loaded wideconv module
    pub(crate) tier: PtxTier,
}

impl Device {
    /// The runtime's cached attributes and the tier of the module a plan runs
    pub(crate) fn new(attributes: &DeviceAttributes, tier: PtxTier) -> Self {
        Self {
            capability: attributes.capability(),
            sms: attributes.multiprocessors().get(),
            tier,
        }
    }

    /// Whether dense TF32 tensor cores outrun FP32 FFMA by enough for three-product
    /// TF32 arithmetic to win
    ///
    /// Data-centre Ampere (8.0), Hopper (9.0) and Blackwell (10.x) issue TF32 at 8x or
    /// more of their FP32 rate, so 3xTF32 Winograd beats FFMA Winograd there. GeForce
    /// and workstation Ampere and Ada (8.6, 8.9) and consumer Blackwell (12.x) run TF32
    /// at one or two times their FP32 rate, where three products lose; they keep the
    /// FP32 kernels in FP32 mode and run one product in TF32 mode. Unknown
    /// capabilities fall back to FP32, which is never wrong
    fn tensor_rich(self) -> bool {
        self.tier >= PtxTier::Sm80
            && matches!(
                (self.capability.major, self.capability.minor),
                (8, 0) | (9, 0) | (10, _)
            )
    }
}

/// A runnable configuration: kernel family and input-channel partitions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Config {
    pub(crate) algorithm: Algorithm,
    pub(crate) partition: Partition,
    pub(crate) split_cells: SplitCells,
}

/// The execution choice of a wide convolution
///
/// Production routes fix an exact configuration before loading; development
/// checks can also force an exact [`Config`] through [`Oxide::with_config`]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Pin {
    /// [`Config::select`] for the plan's device and loaded tier
    DeviceRule,
    /// Fixed configuration selected from cached attributes before loading
    Configured(Config),
}

impl Pin {
    /// Wide FP16 tiles for a compiled same-channel stride-1 boundary in TF32 mode
    pub(crate) fn fp16_wide(name: &str, batch: usize, math: CudaMath) -> Option<Self> {
        let conv = model_conv(name, batch, math).ok()?;
        let shape = Shape::of(conv).ok()?;
        if math != CudaMath::Tf32 || shape.fp16_entry(Fp16Tiles::Wide).is_none() {
            return None;
        }

        Some(Self::Configured(Config {
            algorithm: Algorithm::Fp16(Fp16Tiles::Wide),
            partition: Partition::Whole,
            split_cells: SplitCells::All,
        }))
    }

    /// The T4 recipe uses the same tile rule as the cc 7.5 default, at its 40 SM point
    pub(crate) fn measured_t4_fp16(name: &str, batch: usize, math: CudaMath) -> Option<Self> {
        Self::fp16_wide(name, batch, math)?;
        let conv = model_conv(name, batch, math).ok()?;
        Config::select(
            Device {
                capability: TURING,
                sms: 40,
                tier: PtxTier::Sm75,
            },
            conv,
            Fp16Policy::Allowed,
        )
        .ok()
        .map(Self::Configured)
    }

    /// The retained A100 TF32 recipe pins at the measured batch classes
    pub(crate) fn measured_a100(name: &str, batch: usize) -> Option<Self> {
        if !matches!(batch, 1 | 32) || !LAYERS.contains(&name) {
            return None;
        }
        let (algorithm, partition) = match name {
            "resnet.layer3.0.conv1" | "resnet.layer4.0.conv1" => return None,
            _ if name.ends_with("shortcut.0") => return None,
            _ if name.starts_with("resnet.layer3.") => (
                Algorithm::Winograd(WinogradProducts::Tf32x1),
                Partition::Whole,
            ),
            _ if name.starts_with("resnet.layer4.") && batch == 1 => (
                Algorithm::Winograd(WinogradProducts::Tf32x3),
                Partition::Two,
            ),
            _ if name.starts_with("resnet.layer4.") => {
                (Algorithm::TensorCore(TensorKernel::Tf32), Partition::Whole)
            }
            _ => return None,
        };
        Some(Self::Configured(Config {
            algorithm,
            partition,
            split_cells: SplitCells::All,
        }))
    }
}

/// Most whole waves of Winograd CTAs before which a launch splits its last, partial wave
///
/// A launch of `tiles` CTAs on `sms` SMs runs `tiles / sms` whole waves and a partial
/// one. Splitting the partial wave's cells over their input channels lets it fill the
/// SMs for a fraction of a CTA's time, for the price of a fixup over those cells only:
/// on a 4060 Ti (34 SMs) the 80 batch-1 CTAs of the 128-channel layers take two whole
/// waves and 12 CTAs split in two, against three waves unsplit. Beyond three whole waves
/// the partial wave is too small a share of the time to pay for the fixup
const WINOGRAD_TAIL_WAVES: u32 = 3;
/// CTAs per SM a fully split Winograd launch aims for when its whole launch is less
/// than one wave: with three waves the last, partial wave is a small share of the time
const WINOGRAD_WAVES: u32 = 3;
/// Whole waves of FFMA Winograd CTAs below which every cell splits, not only the cells of
/// the partial wave
///
/// The partition count also bounds the length of the FP32 accumulation chains, and the
/// development check compares error per layer over its batches. On a 4060 Ti the
/// 256-channel batch-1 layers (40 CTAs on 34 SMs) ran 1.56x cuDNN with four partitions on the
/// partial wave only, at cuDNN's maximum error, and 1.30x with every cell in four
/// partitions at 0.4-0.6x of it. The 128-channel layers (80 CTAs) take two whole waves,
/// where splitting every cell in two ran no faster than cuDNN
const WINOGRAD_FULL_SPLIT_WAVES: u32 = 2;
/// Fewest input channels per Winograd partition: below 32 the per-CTA prologue and
/// epilogue outweigh the products
const WINOGRAD_MIN_SPLIT_CHANNELS: usize = 32;

impl Config {
    /// The configuration that measured fastest for `conv` on parts like `device`
    ///
    /// A deterministic function of the device attributes and the convolution contract;
    /// no runtime timing. `fp16` excludes the FP16 tiles, leaving the choice made
    /// without them
    pub(crate) fn select(
        device: Device,
        conv: Conv2d,
        fp16: Fp16Policy,
    ) -> Result<Self, CudaError> {
        let shape = Shape::of(conv)?;
        let tf32 = conv.math == CudaMath::Tf32;
        let tensor_tier = device.tier >= PtxTier::Sm80;
        let whole = |algorithm| Self {
            algorithm,
            partition: Partition::Whole,
            split_cells: SplitCells::All,
        };
        if fp16.allows()
            && let Some(tiles) = Self::fp16(device, shape, conv)
        {
            return Ok(whole(Algorithm::Fp16(tiles)));
        }
        Ok(match shape {
            Shape::Stem => whole(Self::stem(device, conv)),
            Shape::Shortcut => whole(Algorithm::Spatial),
            Shape::C64Stride2 | Shape::C128Stride2 if tf32 && tensor_tier => {
                whole(Algorithm::TensorCore(Self::strided_tf32(device, conv)?))
            }
            Shape::C64Stride2 | Shape::C128Stride2
                if device.tensor_rich() && Self::strided_fp32_tensor(shape, conv) =>
            {
                let kernel = if conv.batch >= TC_WIDE_BATCH as usize {
                    TensorKernel::Tf32x3Wide
                } else {
                    TensorKernel::Tf32x3
                };
                whole(Algorithm::TensorCore(kernel))
            }
            // four partitions give the 128 -> 256 layer's 40 batch-1 CTAs enough
            // parallelism; the others have enough CTAs whole
            Shape::C128Stride2 if conv.batch == 1 => Self {
                algorithm: Algorithm::Spatial,
                partition: Partition::Four,
                split_cells: SplitCells::All,
            },
            Shape::C64Stride2 | Shape::C128Stride2 => whole(Algorithm::Spatial),
            Shape::C32 => {
                return Err(unsupported(
                    "32-channel layers run here only with FP16 tiles in TF32 mode",
                ));
            }
            // only FFMA Winograd covers this shape in FP32 mode, and only Turing routes it here
            Shape::C64 if device.capability != TURING => {
                return Err(unsupported(
                    "64-channel Winograd is selected only on Turing",
                ));
            }
            Shape::C64 => {
                let products = WinogradProducts::Fp32;
                let (partition, split_cells) = Self::winograd_split(device, conv, products)?;
                Self {
                    algorithm: Algorithm::Winograd(products),
                    partition,
                    split_cells,
                }
            }
            Shape::C128 | Shape::C256 => {
                let products = Self::winograd_products(device, shape, conv);
                let Some(products) = products else {
                    return Ok(whole(Algorithm::TensorCore(TensorKernel::Tf32)));
                };
                let (partition, split_cells) = Self::winograd_split(device, conv, products)?;
                let products = Self::swept(products, shape, conv);
                Self {
                    algorithm: Algorithm::Winograd(products),
                    partition,
                    split_cells,
                }
            }
        })
    }

    /// Turing has no TF32 hardware; its TF32 mode uses FP16 tensor cores instead
    fn fp16(device: Device, shape: Shape, conv: Conv2d) -> Option<Fp16Tiles> {
        if conv.math != CudaMath::Tf32
            || device.capability != TURING
            || !matches!(shape, Shape::C32 | Shape::C64 | Shape::C128 | Shape::C256)
        {
            return None;
        }

        Self::fp16_turing_tiles(device, shape, conv)
    }

    /// Turing's FP16 tiles: the 128- and 256-channel shapes take narrow tiles when wide
    /// ones would give fewer than `FP16_WIDE_WAVES` CTAs per SM
    fn fp16_turing_tiles(device: Device, shape: Shape, conv: Conv2d) -> Option<Fp16Tiles> {
        let [oh, ow] = conv.output().map(|size| size as u32);
        let (grid, _) = shape.fp16_launch(Fp16Tiles::Wide, conv.batch as u32, oh, ow)?;
        let ctas = grid.0 * grid.1 * grid.2;
        let narrow =
            matches!(shape, Shape::C128 | Shape::C256) && ctas < FP16_WIDE_WAVES * device.sms;
        Some(if narrow {
            Fp16Tiles::Narrow
        } else {
            Fp16Tiles::Wide
        })
    }

    /// Wide stem tiles where they give at least `WIDE_STEM_WAVES` waves of CTAs
    fn stem(device: Device, conv: Conv2d) -> Algorithm {
        let pixels = (conv.output()[0] * conv.output()[1]) as u64;
        let ctas =
            pixels.div_ceil(u64::from(WIDE_STEM_THREADS * WIDE_STEM_PIXELS)) * conv.batch as u64;
        if ctas >= u64::from(WIDE_STEM_WAVES * device.sms) {
            Algorithm::WideStem
        } else {
            Algorithm::Spatial
        }
    }

    /// Whether 3xTF32 tiles run a FP32-mode stride-2 layer on TF32-rich parts
    ///
    /// On an A100 at batch 32 the 48-column tiles ran both layers at 1.42-1.49x of
    /// cuDNN FP32 and the 32-column ones at 1.37-1.42x, against 0.91-0.94x for the
    /// spatial kernels. At batch 1 the 32-column tiles ran 64 -> 128 at 1.56x, against
    /// 1.07x spatial, but 128 -> 256 at 1.59x, against 1.71x for the spatial kernel in
    /// four partitions, which keeps that layer. Their error was 0.5-0.7x of cuDNN FP32
    fn strided_fp32_tensor(shape: Shape, conv: Conv2d) -> bool {
        conv.math == CudaMath::Fp32
            && (shape == Shape::C64Stride2 || conv.batch >= TC_WIDE_BATCH as usize)
    }

    /// One-product stride-2 tiles: slim where the SMs outnumber the CTAs of the
    /// batch's tiles, which happens only for narrow tiles at small batches
    ///
    /// At batch 1 the narrow tiles give 60 (64 -> 128) and 40 (128 -> 256) CTAs. The
    /// 34 and 36 SMs of a 4060 Ti or 5060 Ti run those in about two waves and keep
    /// them; the 108 SMs of an A100 would leave half their SMs idle, and slim tiles
    /// double the CTAs
    fn strided_tf32(device: Device, conv: Conv2d) -> Result<TensorKernel, CudaError> {
        let layout = Layout::new(
            conv,
            Partition::Whole,
            SplitCells::All,
            Algorithm::TensorCore(TensorKernel::Tf32),
        )?;
        let (x, y, z) = layout.config.grid_dim;
        let narrow = (conv.batch as u32) < TC_WIDE_BATCH;
        Ok(if narrow && x * y * z < device.sms {
            TensorKernel::Tf32Slim
        } else {
            TensorKernel::Tf32
        })
    }

    /// Winograd products for a same-channel layer, or `None` where the direct
    /// tensor-core kernel is faster
    fn winograd_products(device: Device, shape: Shape, conv: Conv2d) -> Option<WinogradProducts> {
        let tf32 = conv.math == CudaMath::Tf32;
        if device.tensor_rich() {
            return Some(match (tf32, shape) {
                (false, _) => WinogradProducts::Tf32x3,
                // on the A100 one staged product is the measured choice at both widths
                // and batches: a second product only held per-layer error to that of a
                // direct TF32 convolution, which DER does not need. The 128-channel
                // layers ran 2-4% faster than unstaged; the 256-channel layers ran
                // 0.031 ms at batch 1 against 0.051 ms for 3xTF32 in two partitions, and
                // 0.50-0.52 ms at batch 32 against 0.54-0.60 ms direct
                (true, Shape::C128 | Shape::C256)
                    if device.capability == ComputeCapability::new(8, 0) =>
                {
                    WinogradProducts::Tf32x1Staged
                }
                // unmeasured tensor-rich parts use the approved staged product
                (true, _) => WinogradProducts::Tf32x1Staged,
            });
        }
        // on parts with TF32 at the FP32 rate one staged product still beats both FFMA
        // Winograd and the direct tensor-core kernel at both batches: on a 5060 Ti the
        // 128-channel layers ran 1.73-1.75 ms at batch 32 against 2.46-2.49 ms direct,
        // and 0.064 ms at batch 1 against 0.101 ms FFMA; the 256-channel layers 1.57 ms
        // against 2.41-2.42 ms and 0.059 ms against 0.121-0.123 ms. A 4060 Ti gained
        // the same. The unstaged kernel ran 2-5% slower at every point
        if tf32 && device.tier >= PtxTier::Sm80 {
            return Some(WinogradProducts::Tf32x1Staged);
        }
        Some(WinogradProducts::Fp32)
    }

    /// FFMA Winograd on the 128-channel layers runs two channel sweeps in FP32 mode
    ///
    /// cuDNN's FP32 mode runs its own fused Winograd on these layers. On a 4060 Ti the
    /// single 128-term FP32 chain of the plain kernel sat at its error (per-layer
    /// maximum-error geomeans up to 1.18); two sweeps, in whole and split cells at both
    /// batches, bring every layer to 0.84 or below on the reference activations, for
    /// about 8% more time at batch 32. Four sweeps cost 25%, which left the layers
    /// without a residual inside cuDNN's noise margin. The 256-channel layers stay
    /// below cuDNN's error without sweeps, and TF32 mode compares against cuDNN's TF32
    /// error; both keep the plain kernel
    fn swept(products: WinogradProducts, shape: Shape, conv: Conv2d) -> WinogradProducts {
        if products == WinogradProducts::Fp32 && shape == Shape::C128 && conv.math == CudaMath::Fp32
        {
            WinogradProducts::Fp32Sweep2
        } else {
            products
        }
    }

    /// Input-channel split of a Winograd launch
    ///
    /// - one or more but at most `WINOGRAD_TAIL_WAVES` whole waves: the cells of the
    ///   partial wave split into the largest power of two that still fits one wave
    /// - less than one wave of tensor-core products: every cell splits into the
    ///   largest power of two that still fits one wave, and in FP32 mode into at least
    ///   two
    /// - less than `WINOGRAD_FULL_SPLIT_WAVES` waves of FFMA products: every cell splits
    ///   into the smallest power of two that offers `WINOGRAD_WAVES` CTAs per SM
    /// - otherwise, or where no split fits: whole
    ///
    /// Every partition keeps at least `WINOGRAD_MIN_SPLIT_CHANNELS` input channels
    fn winograd_split(
        device: Device,
        conv: Conv2d,
        products: WinogradProducts,
    ) -> Result<(Partition, SplitCells), CudaError> {
        let layout = Layout::new(
            conv,
            Partition::Whole,
            SplitCells::All,
            Algorithm::Winograd(products),
        )?;
        let blocks = layout.config.grid_dim.0;
        let cells = layout.cells;
        let tiles = blocks * cells;
        let sms = device.sms;
        let fits = |partition: Partition| {
            conv.in_channels / (partition.count() as usize) >= WINOGRAD_MIN_SPLIT_CHANNELS
        };
        let splits = [Partition::Two, Partition::Four, Partition::Eight];
        let whole = (Partition::Whole, SplitCells::All);
        if tiles < sms && products.tensor() {
            // tensor-core products finish a CTA's input channels several times faster
            // than FFMA, so past one wave the partial sums' round trip and the fixup
            // cost more than the idle SMs: on an A100 at batch 1 the 128-channel
            // layers ran 1.41x cuDNN whole against 1.06-1.08x in four partitions, the
            // 256-channel layers 2.31x in two against 1.89-1.97x in eight
            let partition = splits
                .into_iter()
                .take_while(|&partition| tiles * partition.count() <= sms && fits(partition))
                .last();
            // FP32 mode is measured against cuDNN's FP32 error: on an A100 at batch 1 the
            // 128-channel layers whole reached 1.23x of its maximum error on one layer,
            // and 0.79x in two partitions, which still ran 1.10x cuDNN
            let floor =
                (conv.math == CudaMath::Fp32 && fits(Partition::Two)).then_some(Partition::Two);
            let partition = partition.or(floor).unwrap_or(Partition::Whole);
            return Ok((partition, SplitCells::All));
        }
        // one tensor-core wave and a partial one keep the tail split below: on a 5060 Ti
        // the 256-channel batch-1 layers (40 one-product CTAs on 36 SMs) ran 0.059 ms
        // with eight partitions on the partial wave, against 0.082-0.084 ms with every
        // cell in four
        if tiles < WINOGRAD_FULL_SPLIT_WAVES * sms && !products.tensor() {
            let target = WINOGRAD_WAVES * sms;
            let mut partition = Partition::Whole;
            for next in splits {
                if tiles * partition.count() >= target || !fits(next) {
                    break;
                }
                partition = next;
            }
            return Ok((partition, SplitCells::All));
        }
        let waves = tiles / sms;
        if waves > WINOGRAD_TAIL_WAVES || tiles % sms == 0 {
            return Ok(whole);
        }
        let from = waves * sms / blocks;
        let tail = (cells - from) * blocks;
        let partition = splits
            .into_iter()
            .take_while(|&partition| tail * partition.count() <= sms && fits(partition))
            .last();
        // a split tail past the 16-bit cell index cannot fit the 65535-row grid either
        let Ok(from) = u16::try_from(from) else {
            return Ok(whole);
        };
        Ok(partition.map_or(whole, |partition| (partition, SplitCells::From(from))))
    }
}

/// The convolution contract and immutable folded weights used to create a plan
struct Spec<'a, 'b> {
    conv: Conv2d,
    weight: &'a CudaView<'b, f32>,
    config: Config,
}

/// Live NCHW input and folded bias; absence of a residual never reads scratch
struct Inputs<'a, 'b> {
    x: &'a CudaView<'b, f32>,
    bias: &'a CudaView<'b, f32>,
    residual: Option<&'a CudaView<'b, f32>>,
}

/// Device operands of one launch; the residual aliases `x` when `add_residual` is zero
#[derive(Clone, Copy)]
struct Operands<'a, 'b> {
    x: &'a CudaView<'b, f32>,
    bias: &'a CudaView<'b, f32>,
    residual: &'a CudaView<'b, f32>,
    add_residual: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Shape {
    Stem,
    C32,
    C64,
    C128,
    C256,
    C64Stride2,
    C128Stride2,
    Shortcut,
}

impl Shape {
    fn of(conv: Conv2d) -> Result<Self, CudaError> {
        if conv.dilation != [1, 1] {
            return Err(unsupported("requires dilation 1"));
        }
        if conv.kernel == [1, 1]
            && conv.padding == [0, 0]
            && conv.stride == [2, 2]
            && matches!(
                (conv.in_channels, conv.out_channels),
                (32, 64) | (64, 128) | (128, 256)
            )
        {
            return Ok(Self::Shortcut);
        }
        if conv.kernel != [3, 3] || conv.padding != [1, 1] {
            return Err(unsupported("requires a known 3x3 or shortcut contract"));
        }
        match (conv.in_channels, conv.out_channels, conv.stride) {
            (1, 32, [1, 1]) => Ok(Self::Stem),
            (32, 32, [1, 1]) => Ok(Self::C32),
            (64, 64, [1, 1]) => Ok(Self::C64),
            (128, 128, [1, 1]) => Ok(Self::C128),
            (256, 256, [1, 1]) => Ok(Self::C256),
            (64, 128, [2, 2]) => Ok(Self::C64Stride2),
            (128, 256, [2, 2]) => Ok(Self::C128Stride2),
            _ => Err(unsupported("no kernel for this channel or stride contract")),
        }
    }

    /// Spatial-tile entry; the 32- and 64-channel shapes have only FP16 and Winograd
    /// kernels
    fn entry(self, in_channels: u32, batch: u32) -> Option<&'static str> {
        Some(match self {
            Self::Stem => entries::STEM,
            Self::C32 | Self::C64 => return None,
            Self::C128 => entries::C128,
            Self::C256 => entries::C256,
            Self::C64Stride2 => entries::C64S2,
            Self::C128Stride2 => entries::C128S2,
            Self::Shortcut => shortcut_tile(in_channels, batch).0,
        })
    }

    /// Input size compiled into the tensor-core and shortcut-tile kernels
    fn compiled_input(self, in_channels: usize) -> Option<[usize; 2]> {
        match (self, in_channels) {
            (Self::C128 | Self::C128Stride2, _) | (Self::Shortcut, 128) => Some([20, 250]),
            (Self::C32, _) => Some([80, 998]),
            (Self::C64, _) => Some([40, 499]),
            (Self::C256, _) => Some([10, 125]),
            (Self::C64Stride2, _) | (Self::Shortcut, 64) => Some([40, 499]),
            (Self::Shortcut, 32) => Some([80, 998]),
            _ => None,
        }
    }

    /// Tensor-core entry for the 3x3 wide shapes
    fn tensor_entry(self, kernel: TensorKernel, batch: u32) -> Option<&'static str> {
        match (kernel, self) {
            (TensorKernel::Tf32x3, Self::C64Stride2) => return Some(entries::TC3_C64S2),
            (TensorKernel::Tf32x3, Self::C128Stride2) => return Some(entries::TC3_C128S2),
            (TensorKernel::Tf32x3Wide, Self::C64Stride2) => {
                return Some(entries::TC3_C64S2_WIDE);
            }
            (TensorKernel::Tf32x3Wide, Self::C128Stride2) => {
                return Some(entries::TC3_C128S2_WIDE);
            }
            (TensorKernel::Tf32Slim, Self::C64Stride2) => {
                return Some(entries::TC_C64S2_SLIM);
            }
            (TensorKernel::Tf32Slim, Self::C128Stride2) => {
                return Some(entries::TC_C128S2_SLIM);
            }
            (TensorKernel::Tf32x3 | TensorKernel::Tf32x3Wide | TensorKernel::Tf32Slim, _) => {
                return None;
            }
            (TensorKernel::Tf32, _) => {}
        }
        match self {
            Self::C128 => Some(entries::TC_C128),
            Self::C256 => Some(entries::TC_C256),
            Self::C64Stride2 if batch < TC_WIDE_BATCH => Some(entries::TC_C64S2_NARROW),
            Self::C64Stride2 => Some(entries::TC_C64S2),
            Self::C128Stride2 if batch < TC_WIDE_BATCH => Some(entries::TC_C128S2_NARROW),
            Self::C128Stride2 => Some(entries::TC_C128S2),
            Self::Stem | Self::C32 | Self::C64 | Self::Shortcut => None,
        }
    }

    /// Direct FP16 entry for the same-channel stride-1 shapes
    fn fp16_entry(self, tiles: Fp16Tiles) -> Option<&'static str> {
        match (self, tiles) {
            (Self::C32, Fp16Tiles::Wide) => Some(entries::H16_C32),
            (Self::C64, Fp16Tiles::Wide) => Some(entries::H16_C64),
            (Self::C128, Fp16Tiles::Wide) => Some(entries::H16_C128),
            (Self::C128, Fp16Tiles::Narrow) => Some(entries::H16_C128_NARROW),
            (Self::C256, Fp16Tiles::Wide) => Some(entries::H16_C256),
            (Self::C256, Fp16Tiles::Narrow) => Some(entries::H16_C256_NARROW),
            _ => None,
        }
    }

    /// Direct FP16 grid and dynamic shared bytes, mirroring the device instances: output
    /// rows and channels per CTA, `FP16_COLUMNS` columns, and two stages of the CTA's
    /// input rows and columns with their halo at 16 bytes per pixel
    fn fp16_launch(
        self,
        tiles: Fp16Tiles,
        batch: u32,
        oh: u32,
        ow: u32,
    ) -> Option<((u32, u32, u32), u32)> {
        let (rows, cta_channels, channels) = match (self, tiles) {
            (Self::C32, Fp16Tiles::Wide) => (4, 32, 32),
            (Self::C64, Fp16Tiles::Wide) => (4, 64, 64),
            (Self::C128, Fp16Tiles::Wide) => (2, 128, 128),
            (Self::C128, Fp16Tiles::Narrow) => (2, 64, 128),
            (Self::C256, Fp16Tiles::Wide) => (2, 128, 256),
            (Self::C256, Fp16Tiles::Narrow) => (2, 64, 256),
            _ => return None,
        };
        let grid = (
            ow.div_ceil(FP16_COLUMNS),
            oh.div_ceil(rows),
            batch * (channels / cta_channels),
        );
        Some((grid, 2 * (rows + 2) * (FP16_COLUMNS + 2) * 16))
    }

    /// Fused Winograd entry for the same-channel stride-1 shapes
    fn winograd_entry(self, products: WinogradProducts) -> Option<&'static str> {
        use WinogradProducts::{Bf16x3, Fp32, Fp32Sweep2, Tf32x1, Tf32x1Staged, Tf32x2, Tf32x3};

        match (self, products) {
            (Self::C128, Bf16x3) => Some(entries::WBF_C128),
            (Self::C256, Bf16x3) => Some(entries::WBF_C256),
            (Self::C64, Fp32) => Some(entries::WINO_C64),
            (Self::C128, Fp32) => Some(entries::WINO_C128),
            (Self::C256, Fp32) => Some(entries::WINO_C256),
            (Self::C128, Fp32Sweep2) => Some(entries::WINO_C128_SWEEP2),
            (Self::C128, Tf32x3) => Some(entries::WTC3_C128),
            (Self::C256, Tf32x3) => Some(entries::WTC3_C256),
            (Self::C128, Tf32x2) => Some(entries::WTC2_C128),
            (Self::C256, Tf32x2) => Some(entries::WTC2_C256),
            (Self::C128, Tf32x1) => Some(entries::WTC1_C128),
            (Self::C256, Tf32x1) => Some(entries::WTC1_C256),
            (Self::C128, Tf32x1Staged) => Some(entries::WTP1_C128),
            (Self::C256, Tf32x1Staged) => Some(entries::WTP1_C256),
            _ => None,
        }
    }

    /// Output columns of one stride-2 tensor-core CTA, matching `tensor_entry`
    fn tensor_columns(self, kernel: TensorKernel, batch: u32) -> u32 {
        match (kernel, self) {
            (TensorKernel::Tf32x3, _) => return TC3_COLUMNS,
            (TensorKernel::Tf32x3Wide, _) => return TC3_WIDE_COLUMNS,
            (TensorKernel::Tf32Slim, Self::C64Stride2) => return TC_C64S2_SLIM_COLUMNS,
            (TensorKernel::Tf32Slim, _) => return TC_C128S2_SLIM_COLUMNS,
            (TensorKernel::Tf32, _) => {}
        }
        match self {
            Self::C64Stride2 if batch < TC_WIDE_BATCH => TC_NARROW_COLUMNS,
            Self::C128Stride2 if batch < TC_WIDE_BATCH => TC_C128S2_NARROW_COLUMNS,
            _ => TC_COLUMNS,
        }
    }

    /// Tensor-core grid and dynamic shared bytes, mirroring the device constants:
    /// two stages of eight input channels, each channel padded to 8 mod 32 words
    fn tensor_launch(
        self,
        products: TensorKernel,
        batch: u32,
        input_w: u32,
        oh: u32,
        ow: u32,
        channels: u32,
    ) -> ((u32, u32, u32), u32) {
        let channel_words = |words: u32| (words + 23) / 32 * 32 + 8;
        if matches!(self, Self::C64Stride2 | Self::C128Stride2) {
            let columns = self.tensor_columns(products, batch);
            let words = channel_words(3 * (2 * columns + 1));
            let grid = (channels / TC_CHANNELS, batch * oh * ow.div_ceil(columns), 1);
            return (grid, 2 * 8 * words * 4);
        }
        let words = channel_words(TC_PIXELS + 2 * (input_w + 2) + 8);
        let grid = (
            channels / TC_CHANNELS,
            (batch * oh * ow).div_ceil(TC_PIXELS),
            1,
        );
        (grid, 2 * 8 * words * 4)
    }

    fn spatial_grid(
        self,
        batch: u32,
        in_channels: u32,
        channels: u32,
        oh: u32,
        ow: u32,
        splits: u32,
    ) -> (u32, u32, u32) {
        match self {
            // eight flat pixels per thread of a 256-thread CTA, in all 32 channels
            Self::Stem => ((oh * ow).div_ceil(512), batch, 1),
            Self::C64Stride2 => (ow.div_ceil(64), oh.div_ceil(2), batch * 4 * splits),
            Self::Shortcut => (
                (oh * ow).div_ceil(SHORTCUT_PIXELS),
                batch,
                channels / shortcut_tile(in_channels, batch).1,
            ),
            _ => (ow.div_ceil(64), oh, batch * (channels / 64) * splits),
        }
    }
}

/// Validated indexing and launch geometry, before any device allocation
#[derive(Debug)]
pub(super) struct Layout {
    shape: Shape,
    pub(super) config: LaunchConfig,
    pub(super) input_len: usize,
    pub(super) output_len: usize,
    pub(super) workspace_len: usize,
    /// First Winograd cell that splits; the cell count when none does
    pub(super) split_from: u32,
    /// Winograd cells: 64 output channels by one row of 32 tiles, without the channel blocks
    pub(super) cells: u32,
}

impl Layout {
    pub(super) fn new(
        conv: Conv2d,
        partition: Partition,
        split_cells: SplitCells,
        algorithm: Algorithm,
    ) -> Result<Self, CudaError> {
        let shape = Shape::of(conv)?;
        if split_cells != SplitCells::All && !matches!(algorithm, Algorithm::Winograd(_)) {
            return Err(unsupported("only Winograd launches split a tail of cells"));
        }
        if conv.batch == 0 || conv.input.contains(&0) {
            return Err(unsupported("batch and spatial dimensions must be positive"));
        }
        if matches!(shape, Shape::Stem | Shape::Shortcut) && partition.count() != 1 {
            return Err(unsupported("stem and shortcuts require a whole reduction"));
        }
        if algorithm == Algorithm::WideStem && shape != Shape::Stem {
            return Err(unsupported("wide stem tiles cover only the stem"));
        }
        if shape == Shape::Stem && algorithm == Algorithm::ImplicitGemm {
            return Err(unsupported("the stem uses spatial tiles"));
        }
        let compiled = matches!(
            algorithm,
            Algorithm::TensorCore(_) | Algorithm::Winograd(_) | Algorithm::Fp16(_)
        ) || (algorithm == Algorithm::Spatial && shape == Shape::Shortcut);
        if compiled
            && shape
                .compiled_input(conv.in_channels)
                .is_some_and(|input| input != conv.input)
        {
            return Err(unsupported("the kernel is compiled for another input size"));
        }
        if algorithm == Algorithm::Spatial
            && shape
                .entry(conv.in_channels as u32, conv.batch as u32)
                .is_none()
        {
            return Err(unsupported("no spatial tiles for this shape"));
        }
        if algorithm == Algorithm::ImplicitGemm && matches!(shape, Shape::C32 | Shape::C64) {
            return Err(unsupported(
                "the 32- and 64-channel shapes have only FP16 and Winograd tiles",
            ));
        }
        if let Algorithm::Fp16(tiles) = algorithm {
            if shape.fp16_entry(tiles).is_none() {
                return Err(unsupported(
                    "FP16 tiles cover only the same-channel stride-1 shapes, narrow ones only the 128- and 256-channel shapes",
                ));
            }
            if conv.math != CudaMath::Tf32 {
                return Err(unsupported("FP16 tiles round both operands to FP16"));
            }
            if partition.count() != 1 {
                return Err(unsupported("FP16 tiles reduce the whole input"));
            }
        }
        if let Algorithm::TensorCore(products) = algorithm {
            if shape.tensor_entry(products, conv.batch as u32).is_none() {
                return Err(unsupported(
                    "tensor-core tiles cover only the 3x3 wide shapes, slim and 3xTF32 tiles only the stride-2 ones",
                ));
            }
            if !products.split() && conv.math != CudaMath::Tf32 {
                return Err(unsupported("one-product tensor-core tiles round to TF32"));
            }
            if partition.count() != 1 {
                return Err(unsupported("tensor-core tiles reduce the whole input"));
            }
        }
        if let Algorithm::Winograd(products) = algorithm {
            if shape.winograd_entry(products).is_none() {
                return Err(unsupported(
                    "Winograd tiles cover only the same-channel stride-1 shapes",
                ));
            }
            if matches!(
                products,
                WinogradProducts::Tf32x2
                    | WinogradProducts::Tf32x1
                    | WinogradProducts::Tf32x1Staged
                    | WinogradProducts::Bf16x3
            ) && conv.math != CudaMath::Tf32
            {
                return Err(unsupported(
                    "two-product TF32 and BF16 Winograd keep TF32-level error",
                ));
            }
            if !conv
                .in_channels
                .is_multiple_of(partition.count() as usize * products.chunk())
            {
                return Err(unsupported(
                    "Winograd partitions cover whole four-channel stages",
                ));
            }
        }
        let input_len = element_count("wideconv input", &conv.input_shape())?;
        to_u32(input_len)?;
        let output_len = element_count("wideconv output", &conv.output_shape())?;
        let splits = partition.count();
        let workspace_len = if splits == 1 {
            0
        } else {
            element_count("wideconv partitions", &[output_len, splits as usize])?
        };
        to_u32(input_len.max(output_len).max(workspace_len))?;
        let [oh, ow] = conv.output().map(|size| size as u32);
        let batch = conv.batch as u32;
        let channels = conv.out_channels as u32;
        let mut shared_mem_bytes = 0;
        let mut threads = 128;
        let cells = batch * oh.div_ceil(2) * ow.div_ceil(2).div_ceil(WINO_TILES);
        let split_from = match (splits, split_cells) {
            (1, _) => cells,
            (_, SplitCells::All) => 0,
            (_, SplitCells::From(cell)) if u32::from(cell) <= cells => u32::from(cell),
            _ => return Err(unsupported("the split tail starts past the last cell")),
        };
        let grid_dim = if algorithm == Algorithm::Spatial {
            shape.spatial_grid(batch, conv.in_channels as u32, channels, oh, ow, splits)
        } else if algorithm == Algorithm::WideStem {
            threads = WIDE_STEM_THREADS;
            (
                (oh * ow).div_ceil(WIDE_STEM_THREADS * WIDE_STEM_PIXELS),
                batch,
                1,
            )
        } else if let Algorithm::TensorCore(products) = algorithm {
            let (grid, shared) =
                shape.tensor_launch(products, batch, conv.input[1] as u32, oh, ow, channels);
            shared_mem_bytes = shared;
            grid
        } else if let Algorithm::Fp16(tiles) = algorithm {
            // checked above
            let (grid, shared) = shape
                .fp16_launch(tiles, batch, oh, ow)
                .ok_or_else(|| unsupported("no FP16 tiles for this shape"))?;
            shared_mem_bytes = shared;
            grid
        } else if let Algorithm::Winograd(products) = algorithm {
            threads = WINO_THREADS;
            shared_mem_bytes = products.shared_bytes();
            // one CTA per output-channel block and whole cell, and `splits` per split cell
            (
                channels / WINO_CHANNELS,
                split_from + (cells - split_from) * splits,
                1,
            )
        } else {
            ((oh * ow).div_ceil(64), 1, batch * (channels / 32) * splits)
        };
        if grid_dim.1 > u16::MAX.into() || grid_dim.2 > u16::MAX.into() {
            return Err(unsupported(
                "grid y and z must fit the CUDA 65535-block limit",
            ));
        }
        let config = LaunchConfig {
            grid_dim,
            block_dim: (threads, 1, 1),
            shared_mem_bytes,
        };
        Ok(Self {
            shape,
            config,
            input_len,
            output_len,
            workspace_len,
            split_from,
            cells,
        })
    }
}

/// One layer's wide convolution at one batch size: its configuration, packed weights
/// and partial-sum planes
#[derive(Debug)]
pub(crate) struct Oxide {
    conv: Conv2d,
    config: Config,
    algorithm: Algorithm,
    partition: Partition,
    epilogue: Epilogue,
    function: CudaFunction,
    reduce: CudaFunction,
    fixup: CudaFunction,
    packed: CudaSlice<f32>,
    /// partition planes, `layout.workspace_len` elements used; one element when whole
    workspace: Scratch<f32>,
    layout: Layout,
    tier: PtxTier,
}

impl Oxide {
    /// Packs immutable weights once and rejects contracts without a matching kernel
    fn build(
        runtime: &CudaRuntime,
        kernels: &LoadedKernels,
        spec: Spec<'_, '_>,
        epilogue: Epilogue,
    ) -> Result<Self, CudaError> {
        let conv = spec.conv;
        let Config {
            algorithm,
            partition,
            split_cells,
        } = spec.config;
        let layout = Layout::new(conv, partition, split_cells, algorithm)?;
        let shape = layout.shape;
        let entry = match algorithm {
            Algorithm::ImplicitGemm => entries::GEMM,
            // checked by `Layout::new`, as are the tensor-core and Winograd entries
            Algorithm::Spatial => shape
                .entry(conv.in_channels as u32, conv.batch as u32)
                .unwrap_or_default(),
            Algorithm::WideStem => entries::STEM_WIDE,
            Algorithm::TensorCore(products) => shape
                .tensor_entry(products, conv.batch as u32)
                .unwrap_or_default(),
            Algorithm::Winograd(products) => shape.winograd_entry(products).unwrap_or_default(),
            Algorithm::Fp16(tiles) => shape.fp16_entry(tiles).unwrap_or_default(),
        };
        let weight_len = element_count("wideconv weights", &conv.filter_shape())?;
        check_len("wideconv weights", weight_len, spec.weight.len())?;
        // the sm75 variant of these entries traps; `Layout::new` keeps one-product tiles
        // out of FP32 mode
        if matches!(algorithm, Algorithm::TensorCore(_)) && kernels.tier() < PtxTier::Sm80 {
            return Err(unsupported("tensor-core tiles need the sm80 PTX tier"));
        }
        if let Algorithm::Winograd(products) = algorithm
            && products.tensor()
            && kernels.tier() < PtxTier::Sm80
        {
            // the sm75 variant of these entries traps
            return Err(unsupported("tensor-core Winograd needs the sm80 PTX tier"));
        }
        let pack = kernels.function(match algorithm {
            Algorithm::TensorCore(TensorKernel::Tf32 | TensorKernel::Tf32Slim) => entries::PACK_TC,
            Algorithm::TensorCore(TensorKernel::Tf32x3 | TensorKernel::Tf32x3Wide) => {
                entries::PACK_TC3
            }
            Algorithm::Winograd(WinogradProducts::Fp32 | WinogradProducts::Fp32Sweep2) => {
                entries::PACK_WINOGRAD
            }
            Algorithm::Winograd(WinogradProducts::Bf16x3) => entries::PACK_WBF,
            Algorithm::Winograd(_) => entries::PACK_WTC,
            Algorithm::Fp16(_) => entries::PACK_H16,
            // the stem's weights are copied, not packed
            Algorithm::Spatial | Algorithm::WideStem | Algorithm::ImplicitGemm => {
                entries::PACK_WEIGHTS
            }
        })?;
        // Winograd weights hold a 4x4 transform per 3x3 filter, tensor-core ones a high
        // and a low part of each; 3xTF32 direct weights a high and a low part per tap
        let packed_len = match algorithm {
            Algorithm::Winograd(
                WinogradProducts::Fp32 | WinogradProducts::Fp32Sweep2 | WinogradProducts::Bf16x3,
            ) => weight_len / 9 * 16,
            Algorithm::Winograd(_) => weight_len / 9 * 32,
            Algorithm::TensorCore(TensorKernel::Tf32x3 | TensorKernel::Tf32x3Wide) => {
                weight_len * 2
            }
            // two FP16 weights per word
            Algorithm::Fp16(_) => weight_len / 2,
            _ => weight_len,
        };
        let mut packed = runtime.stream().alloc_zeros::<f32>(packed_len)?;
        if matches!(algorithm, Algorithm::Winograd(_)) {
            let ci = to_u32(conv.in_channels)?;
            let co = to_u32(conv.out_channels)?;
            let len = weight_len as u64;
            let out_len = packed_len as u64;
            let mut launch = runtime.stream().launch_builder(&pack);
            launch
                .arg(spec.weight)
                .arg(&len)
                .arg(&ci)
                .arg(&co)
                .arg(&mut packed)
                .arg(&out_len);
            // safety: the packed buffer holds the kernel's 16 or 32 words per filter and the
            // ABI matches
            unsafe { launch.launch(linear_config(to_u32(packed_len)?)) }?;
        } else if matches!(algorithm, Algorithm::TensorCore(_) | Algorithm::Fp16(_)) {
            let ci = to_u32(conv.in_channels)?;
            let co = to_u32(conv.out_channels)?;
            let len = weight_len as u64;
            let out_len = packed_len as u64;
            let mut launch = runtime.stream().launch_builder(&pack);
            launch
                .arg(spec.weight)
                .arg(&len)
                .arg(&ci)
                .arg(&co)
                .arg(&mut packed)
                .arg(&out_len);
            // safety: the packed buffer holds the kernel's half, one or two words per
            // weight, one thread per word, and the ABI matches
            unsafe { launch.launch(linear_config(to_u32(packed_len)?)) }?;
        } else if shape == Shape::Stem {
            runtime.stream().memcpy_dtod(spec.weight, &mut packed)?;
        } else {
            let ci = to_u32(conv.in_channels)?;
            let co = to_u32(conv.out_channels)?;
            let weight_len_u32 = to_u32(weight_len)?;
            let len = weight_len as u64;
            let taps = (conv.kernel[0] * conv.kernel[1]) as u32;
            let mut launch = runtime.stream().launch_builder(&pack);
            launch
                .arg(spec.weight)
                .arg(&len)
                .arg(&ci)
                .arg(&co)
                .arg(&taps)
                .arg(&mut packed)
                .arg(&len);
            // safety: packed and source weights have equal checked lengths and the ABI matches
            unsafe { launch.launch(linear_config(weight_len_u32)) }?;
        }

        let function = kernels.function(entry)?;
        let shared = layout.config.shared_mem_bytes;
        if shared > DEFAULT_SHARED_LIMIT {
            function.set_attribute(
                CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                shared as i32,
            )?;
        }

        Ok(Self {
            conv,
            config: spec.config,
            algorithm,
            partition,
            epilogue,
            function,
            reduce: kernels.function(entries::REDUCE)?,
            fixup: kernels.function(entries::WINO_FIXUP)?,
            packed,
            // CUDA cannot allocate zero bytes
            workspace: Scratch::zeros(runtime, layout.workspace_len.max(1))?,
            layout,
            tier: kernels.tier(),
        })
    }

    /// FP32 elements of the plan's partial-sum planes
    pub(crate) fn workspace_len(&self) -> usize {
        self.layout.workspace_len
    }

    /// The configuration this plan runs
    pub(crate) fn config(&self) -> Config {
        self.config
    }

    /// PTX tier selected by the runtime for this kernel area
    pub(crate) fn tier(&self) -> PtxTier {
        self.tier
    }

    /// Driver JIT resource counts: registers, local bytes and static shared bytes
    pub(crate) fn resources(&self) -> Result<[i32; 3], CudaError> {
        Ok([
            self.function.num_regs()?,
            self.function.local_size_bytes()?,
            self.function.shared_size_bytes()?,
        ])
    }

    /// Maximum resident CTAs per SM for this plan's actual launch geometry
    pub(crate) fn resident_blocks(&self) -> Result<u32, CudaError> {
        Ok(self
            .function
            .occupancy_max_active_blocks_per_multiprocessor(
                self.layout.config.block_dim.0,
                self.layout.config.shared_mem_bytes as usize,
                None,
            )?)
    }

    /// Enqueues convolution and, for partitioned plans, the deterministic epilogue;
    /// FP16 tiles also set `range` nonzero when an activation saturates
    ///
    /// All allocations and packing occur in `plan`, so this method is capturable
    fn launch<'a>(
        &self,
        inputs: Inputs<'_, '_>,
        output: &mut CudaViewMut<'a, f32>,
        workspace: &mut CudaViewMut<'a, f32>,
        range: Option<&mut CudaViewMut<'_, f32>>,
        stream: &CudaStream,
    ) -> Result<(), CudaError> {
        check_len("wideconv input", self.layout.input_len, inputs.x.len())?;
        check_len("wideconv bias", self.conv.out_channels, inputs.bias.len())?;
        check_len("wideconv output", self.layout.output_len, output.len())?;
        check_len(
            "wideconv workspace",
            self.layout.workspace_len,
            workspace.len(),
        )?;
        if matches!(self.layout.shape, Shape::Stem | Shape::Shortcut) && inputs.residual.is_some() {
            return Err(unsupported("stem and shortcuts have no residual operand"));
        }
        if let Some(residual) = inputs.residual {
            check_len("wideconv residual", self.layout.output_len, residual.len())?;
        }

        // the inactive residual descriptor aliases read-only input, not uninitialized scratch
        let residual = inputs.residual.unwrap_or(inputs.x);
        let add_residual = u32::from(inputs.residual.is_some());
        let [h, w] = self.conv.input.map(|size| size as u32);
        let batch = self.conv.batch as u32;
        let channels = self.conv.out_channels as u32;
        let splits = self.partition.count();
        let lengths = [
            inputs.x.len() as u64,
            self.packed.len() as u64,
            inputs.bias.len() as u64,
            residual.len() as u64,
            output.len() as u64,
            workspace.len() as u64,
        ];
        if self.layout.shape == Shape::Stem {
            // each output plane takes 16-byte stores only when every thread's pixels stay
            // inside one plane and every plane stays aligned
            let pixels = if self.algorithm == Algorithm::WideStem {
                WIDE_STEM_PIXELS
            } else {
                4
            };
            let (out_ptr, _out) = output.device_ptr(stream);
            let vector = u32::from((h * w).is_multiple_of(pixels) && out_ptr.is_multiple_of(16));
            drop(_out);
            let mut launch = stream.launch_builder(&self.function);
            launch
                .arg(inputs.x)
                .arg(&lengths[0])
                .arg(&self.packed)
                .arg(&lengths[1])
                .arg(inputs.bias)
                .arg(&lengths[2])
                .arg(&h)
                .arg(&w)
                .arg(&vector)
                .arg(output)
                .arg(&lengths[4]);
            // safety: validated stem buffers, `vector` only with aligned planes, and a
            // disjoint run of pixels per thread
            unsafe { launch.launch(self.layout.config) }?;
            return Ok(());
        }
        let ci = self.conv.in_channels as u32;
        if self.layout.shape == Shape::Shortcut && self.algorithm == Algorithm::Spatial {
            // tiles store even pixel pairs with 8-byte accesses
            let (out_ptr, _out) = output.device_ptr(stream);
            if !out_ptr.is_multiple_of(8) {
                return Err(unsupported("shortcut outputs need 8-byte alignment"));
            }
            drop(_out);
            let mut launch = stream.launch_builder(&self.function);
            launch
                .arg(inputs.x)
                .arg(&lengths[0])
                .arg(&self.packed)
                .arg(&lengths[1])
                .arg(inputs.bias)
                .arg(&lengths[2])
                .arg(&batch)
                .arg(output)
                .arg(&lengths[4]);
            // safety: checked compiled-size NCHW buffers and aligned outputs; each pixel
            // pair has one writer
            unsafe { launch.launch(self.layout.config) }?;
            return Ok(());
        }
        if matches!(self.algorithm, Algorithm::Winograd(_)) {
            let operands = Operands {
                x: inputs.x,
                bias: inputs.bias,
                residual,
                add_residual,
            };
            return self.enqueue_winograd(&operands, output, workspace, stream);
        }
        if matches!(
            self.algorithm,
            Algorithm::TensorCore(_) | Algorithm::Fp16(_)
        ) {
            // stride-1 TF32 epilogues move even pixel pairs with 8-byte accesses; the FP16
            // ones store single words
            if matches!(self.algorithm, Algorithm::TensorCore(_)) {
                let aligned = |pointer: u64| pointer.is_multiple_of(8);
                let (out_ptr, _out) = output.device_ptr(stream);
                let (res_ptr, _res) = residual.device_ptr(stream);
                if !aligned(out_ptr) || !aligned(res_ptr) {
                    return Err(unsupported("tensor-core outputs need 8-byte alignment"));
                }
                drop((_out, _res));
            }
            let range = match (self.algorithm, range) {
                (Algorithm::Fp16(_), Some(range)) => Some(range),
                (Algorithm::Fp16(_), None) => {
                    return Err(unsupported("FP16 tiles need an out-of-range word"));
                }
                _ => None,
            };
            let range_len = range.as_ref().map_or(0, |range| range.len() as u64);
            let mut launch = stream.launch_builder(&self.function);
            launch
                .arg(inputs.x)
                .arg(&lengths[0])
                .arg(&self.packed)
                .arg(&lengths[1])
                .arg(inputs.bias)
                .arg(&lengths[2])
                .arg(residual)
                .arg(&lengths[3])
                .arg(&add_residual)
                .arg(&batch)
                .arg(output)
                .arg(&lengths[4]);
            if let Some(range) = range {
                launch.arg(range).arg(&range_len);
            }
            // safety: lengths and alignment are checked, the dynamic shared size is the
            // kernel's, each output element has one writer, and FP16 launches get their
            // range word
            unsafe { launch.launch(self.layout.config) }?;
            return Ok(());
        }
        let target = if splits == 1 {
            &mut *output
        } else {
            &mut *workspace
        };
        let target_len = if splits == 1 { lengths[4] } else { lengths[5] };
        let mut launch = stream.launch_builder(&self.function);
        launch
            .arg(inputs.x)
            .arg(&lengths[0])
            .arg(&self.packed)
            .arg(&lengths[1])
            .arg(inputs.bias)
            .arg(&lengths[2])
            .arg(residual)
            .arg(&lengths[3])
            .arg(&add_residual);
        let stride = self.conv.stride[0] as u32;
        let tf32 =
            u32::from(self.conv.math == CudaMath::Tf32 && self.layout.shape != Shape::Shortcut);
        let shortcut = u32::from(self.layout.shape == Shape::Shortcut);
        if self.algorithm == Algorithm::ImplicitGemm {
            launch
                .arg(&batch)
                .arg(&ci)
                .arg(&channels)
                .arg(&h)
                .arg(&w)
                .arg(&stride)
                .arg(&splits)
                .arg(&tf32)
                .arg(&shortcut);
        } else {
            launch.arg(&h).arg(&w).arg(&batch).arg(&splits);
        }
        launch.arg(target).arg(&target_len);
        // safety: lengths are checked, partitions are disjoint, and tile launch matches the ABI
        unsafe { launch.launch(self.layout.config) }?;
        if splits == 1 {
            return Ok(());
        }

        let operands = Operands {
            x: inputs.x,
            bias: inputs.bias,
            residual,
            add_residual,
        };
        self.reduce(&operands, workspace, output, stream)
    }
}

impl Oxide {
    /// Launches the fused Winograd kernel, and for partitioned plans the fixed-order
    /// sum of the split cells' partial planes
    fn enqueue_winograd<'a>(
        &self,
        operands: &Operands<'_, '_>,
        output: &mut CudaViewMut<'a, f32>,
        workspace: &mut CudaViewMut<'a, f32>,
        stream: &CudaStream,
    ) -> Result<(), CudaError> {
        let Operands {
            x,
            bias,
            residual,
            add_residual,
        } = *operands;
        let splits = self.partition.count();
        let split_from = self.layout.split_from;
        let batch = self.conv.batch as u32;
        // epilogues move even pixel pairs with 8-byte accesses
        let aligned = |pointer: u64| pointer.is_multiple_of(8);
        let (out_ptr, _out) = output.device_ptr(stream);
        let (work_ptr, _work) = workspace.device_ptr(stream);
        let (res_ptr, _res) = residual.device_ptr(stream);
        if !aligned(out_ptr) || !aligned(res_ptr) || (splits > 1 && !aligned(work_ptr)) {
            return Err(unsupported("Winograd outputs need 8-byte alignment"));
        }
        drop((_out, _work, _res));
        let lengths = [
            x.len() as u64,
            self.packed.len() as u64,
            bias.len() as u64,
            residual.len() as u64,
            output.len() as u64,
            workspace.len() as u64,
        ];
        let mut launch = stream.launch_builder(&self.function);
        launch
            .arg(x)
            .arg(&lengths[0])
            .arg(&self.packed)
            .arg(&lengths[1])
            .arg(bias)
            .arg(&lengths[2])
            .arg(residual)
            .arg(&lengths[3])
            .arg(&add_residual)
            .arg(&batch)
            .arg(&splits)
            .arg(&split_from)
            .arg(&mut *output)
            .arg(&lengths[4])
            .arg(&mut *workspace)
            .arg(&lengths[5]);
        // safety: lengths, partition planes and alignment are checked, the dynamic shared
        // size is the kernel's, and each output and plane element has one writer
        unsafe { launch.launch(self.layout.config) }?;
        if split_from == self.layout.cells {
            return Ok(());
        }

        let partial = workspace.as_view();
        let channels = self.conv.out_channels as u32;
        let [h, w] = self.conv.output().map(|size| size as u32);
        let mut launch = stream.launch_builder(&self.fixup);
        launch
            .arg(&partial)
            .arg(&lengths[5])
            .arg(bias)
            .arg(&lengths[2])
            .arg(residual)
            .arg(&lengths[3])
            .arg(&add_residual)
            .arg(&batch)
            .arg(&channels)
            .arg(&h)
            .arg(&w)
            .arg(&splits)
            .arg(&split_from)
            .arg(output)
            .arg(&lengths[4]);
        // one thread per output word of a split cell: 64 channels by 2 rows by 64 columns
        let config = LaunchConfig {
            grid_dim: (
                WINO_CHANNELS * 4 * WINO_TILES / WINO_THREADS,
                self.layout.cells - split_from,
                channels / WINO_CHANNELS,
            ),
            block_dim: (WINO_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        // safety: the split cells wrote every word of their planes before this
        // same-stream launch, and each output word of a split cell has one writer
        unsafe { launch.launch(config) }?;
        Ok(())
    }

    /// Sums the partition planes in fixed order, then adds residual and bias and applies ReLU
    fn reduce(
        &self,
        operands: &Operands<'_, '_>,
        workspace: &CudaViewMut<'_, f32>,
        output: &mut CudaViewMut<'_, f32>,
        stream: &CudaStream,
    ) -> Result<(), CudaError> {
        let Operands {
            bias,
            residual,
            add_residual,
            ..
        } = *operands;
        let [oh, ow] = self.conv.output().map(|size| size as u32);
        let plane = oh * ow;
        let channels = self.conv.out_channels as u32;
        let splits = self.partition.count();
        let lengths = [
            workspace.len() as u64,
            bias.len() as u64,
            residual.len() as u64,
            output.len() as u64,
        ];
        let partial = workspace.as_view();
        let mut launch = stream.launch_builder(&self.reduce);
        launch
            .arg(&partial)
            .arg(&lengths[0])
            .arg(bias)
            .arg(&lengths[1])
            .arg(residual)
            .arg(&lengths[2])
            .arg(&add_residual)
            .arg(&splits)
            .arg(&plane)
            .arg(&channels)
            .arg(output)
            .arg(&lengths[3]);
        // safety: all partition planes were written before this same-stream reduction
        unsafe { launch.launch(linear_config(self.layout.output_len as u32)) }?;
        Ok(())
    }
}

impl ConvCandidate for Oxide {
    type Pin = Pin;
    const COVERAGE: Coverage = Coverage(&[CoverageEntry {
        layers: &LAYERS,
        batches: Batches::All,
        maths: Maths::All,
    }]);

    // Winograd transforms mix every input of a 4x4 tile, tensor-core products round to
    // TF32 or BF16 parts, and partitioned sums reorder terms, so only finiteness is
    // stated. The headroom covers the Winograd input and output transforms, which each
    // sum up to 16 terms of the tile before the products
    const SPECIAL_VALUES: SpecialValues = SpecialValues {
        finite: FiniteContract::AbsoluteSum { headroom: 256 },
        nan: NanContract::Unspecified,
        infinity: InfinityContract::Unspecified,
        signed_zero: SignedZeroContract::Unspecified,
    };

    fn implemented_pin(layer: &ConvLayerSpec<'_>) -> Result<Pin, PlanError> {
        check_epilogue(layer)?;
        Ok(Pin::DeviceRule)
    }

    fn plan(
        runtime: &CudaRuntime,
        kernels: &LoadedKernels,
        layer: ConvLayerSpec<'_>,
        pin: Pin,
    ) -> Result<Self, PlanError> {
        let config = match pin {
            Pin::DeviceRule => Config::select(
                Device::new(runtime.device(), kernels.tier()),
                layer.conv,
                Fp16Policy::Allowed,
            )
            .map_err(refusal)?,
            Pin::Configured(config)
                if model_conv(layer.name, layer.conv.batch, layer.conv.math)? == layer.conv =>
            {
                config
            }
            Pin::Configured(_) => {
                return Err(PlanError::Geometry(GeometryError::Unimplemented {
                    context: "wideconv fixed geometry",
                    reason: "the model shape differs from the selected pin".into(),
                }));
            }
        };
        Self::with_config(runtime, kernels, layer, config)
    }

    /// Refuses FP16 plans, which need the out-of-range word of [`Oxide::enqueue_checked`]
    fn enqueue(
        &self,
        inputs: ConvInputs<'_, '_>,
        y: &mut CudaViewMut<'_, f32>,
        phases: &Phases,
        stream: &CudaStream,
    ) -> Result<(), CudaError> {
        self.enqueue_with(inputs, y, None, phases, stream)
    }
}

impl Oxide {
    /// Whether this plan runs FP16 tiles, whose launches report saturating activations
    pub(crate) fn is_fp16(&self) -> bool {
        matches!(self.algorithm, Algorithm::Fp16(_))
    }

    /// Enqueues the layer as [`ConvCandidate::enqueue`] does; FP16 tiles also set
    /// `range[0]` nonzero when an activation saturates, and other plans never write it
    ///
    /// The caller clears the word and reads it after the stream reaches this launch
    pub(crate) fn enqueue_checked(
        &self,
        inputs: ConvInputs<'_, '_>,
        y: &mut CudaViewMut<'_, f32>,
        range: &mut CudaViewMut<'_, f32>,
        phases: &Phases,
        stream: &CudaStream,
    ) -> Result<(), CudaError> {
        check_len("wideconv out-of-range word", 1, range.len())?;
        self.enqueue_with(inputs, y, Some(range), phases, stream)
    }

    fn enqueue_with(
        &self,
        inputs: ConvInputs<'_, '_>,
        y: &mut CudaViewMut<'_, f32>,
        range: Option<&mut CudaViewMut<'_, f32>>,
        phases: &Phases,
        stream: &CudaStream,
    ) -> Result<(), CudaError> {
        if inputs.residual.is_some() != (self.epilogue == Epilogue::BiasReluResidual) {
            return Err(CudaError::CandidateGeometry {
                area: "wideconv",
                boundary: "wideconv residual".into(),
                batch: self.conv.batch,
                math: self.conv.math,
                error: GeometryError::Invalid {
                    context: "wideconv residual",
                    reason: "residual input does not match the planned epilogue".into(),
                },
            });
        }

        let mut planes = self.workspace.get();
        // reborrow both so they share one lifetime
        let mut workspace = planes.slice_mut(..self.layout.workspace_len);
        let mut output = y.slice_mut(..);
        let inputs = Inputs {
            x: inputs.x,
            bias: inputs.bias,
            residual: inputs.residual,
        };
        phases.op(Op::Main, || {
            self.launch(inputs, &mut output, &mut workspace, range, stream)
        })
    }
}

impl Oxide {
    /// Plans exactly `config`, as the device rule does after selecting it; development
    /// checks also force configurations selected for other devices
    pub(crate) fn with_config(
        runtime: &CudaRuntime,
        kernels: &LoadedKernels,
        layer: ConvLayerSpec<'_>,
        config: Config,
    ) -> Result<Self, PlanError> {
        check_epilogue(&layer)?;
        let tier = kernels.tier();
        let tensor = match config.algorithm {
            Algorithm::TensorCore(_) => true,
            Algorithm::Winograd(products) => products.tensor(),
            _ => false,
        };
        // the sm75 variant of the tensor-core entries traps
        if tensor && tier < PtxTier::Sm80 {
            return Err(PlanError::DeviceUnsupported {
                reason: format!("{config:?} needs the sm80 wideconv PTX tier, loaded {tier}"),
            });
        }
        let spec = Spec {
            conv: layer.conv,
            weight: &layer.weight.as_view(),
            config,
        };
        Self::build(runtime, kernels, spec, layer.epilogue).map_err(refusal)
    }
}

impl Oxide {
    /// Driver-only coverage on a part of `capability`; development checks also plan the
    /// Turing coverage on other parts
    pub(crate) fn coverage_on(
        tier: PtxTier,
        capability: ComputeCapability,
        fp16: Fp16Policy,
    ) -> Coverage {
        // only FP16 tiles cover the 32-channel layers
        const TURING_FP32_OPERANDS: Coverage = Coverage(&[
            CoverageEntry {
                layers: &LAYERS,
                batches: Batches::All,
                maths: Maths::All,
            },
            CoverageEntry {
                layers: &C64_LAYERS,
                batches: Batches::All,
                maths: Maths::All,
            },
        ]);
        const TURING_COVERAGE: Coverage = Coverage(&[
            CoverageEntry {
                layers: &LAYERS,
                batches: Batches::All,
                maths: Maths::All,
            },
            CoverageEntry {
                layers: &C64_LAYERS,
                batches: Batches::All,
                maths: Maths::All,
            },
            CoverageEntry {
                layers: &C32_LAYERS,
                batches: Batches::All,
                maths: Maths::Only(&[CudaMath::Tf32]),
            },
        ]);
        match (capability == TURING, fp16) {
            (true, Fp16Policy::Allowed) => TURING_COVERAGE,
            (true, Fp16Policy::Excluded) => TURING_FP32_OPERANDS,
            (false, _) => <Self as ConvCandidate>::coverage(tier),
        }
    }
}

/// The kernels fix the epilogue by shape: shortcuts add bias only, the stem adds bias
/// and ReLU, and the other 3x3 layers may also add a residual
fn check_epilogue(layer: &ConvLayerSpec<'_>) -> Result<(), PlanError> {
    let shape = Shape::of(layer.conv).map_err(refusal)?;
    let fits = match shape {
        Shape::Shortcut => layer.epilogue == Epilogue::Bias,
        Shape::Stem => layer.epilogue == Epilogue::BiasRelu,
        _ => layer.epilogue != Epilogue::Bias,
    };
    if fits {
        return Ok(());
    }

    Err(PlanError::Geometry(GeometryError::Invalid {
        context: "wideconv plan",
        reason: format!(
            "{} is {shape:?}, whose kernel has no {:?} epilogue",
            layer.name, layer.epilogue
        ),
    }))
}

/// A refusal from configuration or layout is an unimplemented geometry; device and
/// size errors stay CUDA errors
fn refusal(error: CudaError) -> PlanError {
    match error {
        CudaError::Unsupported { context, reason } => {
            PlanError::Geometry(GeometryError::Unimplemented { context, reason })
        }
        other => PlanError::Cuda(other),
    }
}

/// Shortcut-tile entry and its output channels per CTA
fn shortcut_tile(in_channels: u32, batch: u32) -> (&'static str, u32) {
    match (in_channels, batch >= SHORTCUT_WIDE_BATCH) {
        (32, _) => (entries::SHORTCUT_C32, 64),
        (64, false) => (entries::SHORTCUT_C64, 64),
        (64, true) => (entries::SHORTCUT_C64_WIDE, 128),
        (128, false) => (entries::SHORTCUT_C128, 64),
        _ => (entries::SHORTCUT_C128_WIDE, 128),
    }
}

fn unsupported(reason: &str) -> CudaError {
    CudaError::Unsupported {
        context: "wideconv plan",
        reason: reason.into(),
    }
}

fn to_u32(value: usize) -> Result<u32, CudaError> {
    u32::try_from(value).map_err(|_| CudaError::DimensionOverflow {
        context: "wideconv index",
        value,
    })
}

fn linear_config(elements: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (elements.div_ceil(256), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Trunk totals from do-wideconv sm80 dev runs, not a whole-area universal claim
pub(super) const TRUNK_SPEED_SUMMARY: &str = "do-wideconv sm80 trunk totals: Blackwell b1/b32 both maths >=1.34x; Ada FP32 b32 1.29x and TF32 b1/b32 >=1.39x; Ada FP32 b1 1.01x uses Library";

pub(super) fn trunk_speed_scope(
    _boundary: super::super::implementation::BoundaryId,
    batch: usize,
    math: CudaMath,
    device: &DeviceAttributes,
) -> Option<super::super::implementation::SpeedScope> {
    let capability = device.capability();
    let measured = matches!(batch, 1 | 32)
        && (capability == ComputeCapability::new(12, 0)
            || capability == ComputeCapability::new(8, 9)
                && (math == CudaMath::Tf32 || batch == 32));
    measured.then_some(super::super::implementation::SpeedScope::MeasuredCapability { capability })
}

impl super::DriverCandidate for Oxide {
    const AREA: super::KernelModule = super::KernelModule::Wideconv;

    fn hybrid_fp16(
        device: &DeviceAttributes,
        tier: PtxTier,
        recipe: Option<super::super::implementation::policy::Recipe>,
        fp16: Fp16Policy,
    ) -> Fp16Policy {
        use crate::inference::cuda::implementation::policy::Recipe;
        // the 4060 Ti FP16 measurements belong to its whole-pipeline recipe, not
        // the non-FP16 trunk totals used as this port's speed evidence
        if Recipe::fp16_device(device, tier) == Some(Recipe::Rtx4060Ti)
            && recipe != Some(Recipe::Rtx4060Ti)
        {
            return Fp16Policy::Excluded;
        }

        fp16
    }

    fn driver_coverage(tier: PtxTier, device: &DeviceAttributes, fp16: Fp16Policy) -> Coverage {
        use crate::inference::cuda::implementation::policy::Recipe;
        if fp16.allows() && Recipe::fp16_device(device, tier) == Some(Recipe::Rtx4060Ti) {
            return Coverage(&[
                CoverageEntry {
                    layers: &LAYERS,
                    batches: Batches::All,
                    maths: Maths::All,
                },
                CoverageEntry {
                    layers: &C64_LAYERS,
                    batches: Batches::All,
                    maths: Maths::Only(&[CudaMath::Tf32]),
                },
                CoverageEntry {
                    layers: &C32_LAYERS,
                    batches: Batches::All,
                    maths: Maths::Only(&[CudaMath::Tf32]),
                },
            ]);
        }

        Self::coverage_on(tier, device.capability(), fp16)
    }

    fn speed_scope(
        boundary: super::super::implementation::BoundaryId,
        batch: usize,
        math: CudaMath,
        device: &DeviceAttributes,
        tier: PtxTier,
    ) -> Option<super::super::implementation::SpeedScope> {
        trunk_speed_scope(boundary, batch, math, device).filter(|_| tier >= PtxTier::Sm80)
    }

    fn speed_summary(_math: CudaMath) -> &'static str {
        TRUNK_SPEED_SUMMARY
    }

    fn tuning_fp16_pin(
        boundary: super::super::implementation::BoundaryId,
        batch: usize,
        math: CudaMath,
        device: &DeviceAttributes,
        tier: PtxTier,
    ) -> Result<Option<super::ConfigPin>, PlanError> {
        // fp16 entries are shipped in every tier for Turing and newer devices
        if device.capability() < TURING {
            return Ok(None);
        }
        let Some(wide) = Pin::fp16_wide(boundary.name(), batch, math) else {
            return Ok(None);
        };
        // retain the startup tile size where measured, without limiting discovery
        if let Ok(startup) =
            Self::driver_pin(boundary, batch, math, device, tier, Fp16Policy::Allowed)
            && startup.is_fp16()
        {
            return Ok(Some(startup));
        }
        Ok(Some(super::ConfigPin::Wideconv(wide)))
    }

    fn tuning_fp32_pin(
        boundary: super::super::implementation::BoundaryId,
        batch: usize,
        math: CudaMath,
        _device: &DeviceAttributes,
        _tier: PtxTier,
    ) -> Result<Option<super::ConfigPin>, PlanError> {
        let conv = model_conv(boundary.name(), batch, math)?;
        let config = Config {
            algorithm: Algorithm::Spatial,
            partition: Partition::Whole,
            split_cells: SplitCells::All,
        };
        if Layout::new(conv, config.partition, config.split_cells, config.algorithm).is_err() {
            return Ok(None);
        }

        Ok(Some(super::ConfigPin::Wideconv(Pin::Configured(config))))
    }

    fn driver_pin(
        boundary: super::super::implementation::BoundaryId,
        batch: usize,
        math: CudaMath,
        device: &DeviceAttributes,
        tier: PtxTier,
        fp16: Fp16Policy,
    ) -> Result<super::ConfigPin, PlanError> {
        use crate::inference::cuda::implementation::policy::Recipe;
        if let Some(pin) = Recipe::fp16_device(device, tier)
            .filter(|_| fp16.allows())
            .and_then(|recipe| recipe.fp16_pin(boundary, batch, math))
        {
            return Ok(pin);
        }

        let conv = model_conv(boundary.name(), batch, math)?;
        let config = Config::select(Device::new(device, tier), conv, fp16).map_err(refusal)?;
        Ok(super::ConfigPin::Wideconv(Pin::Configured(config)))
    }
}

/// Geometry compiled into the model's 22 wide convolution boundaries and the 32- and
/// 64-channel ones of the FP16 routes
fn model_conv(name: &str, batch: usize, math: CudaMath) -> Result<Conv2d, PlanError> {
    if !LAYERS.contains(&name) && !C64_LAYERS.contains(&name) && !C32_LAYERS.contains(&name) {
        return Err(PlanError::Geometry(GeometryError::Unimplemented {
            context: "wideconv boundary",
            reason: name.into(),
        }));
    }
    let (in_channels, out_channels, input, kernel, stride) = match name {
        "resnet.conv1" => (1, 32, [80, 998], 3, 1),
        "resnet.layer2.0.shortcut.0" => (32, 64, [80, 998], 1, 2),
        "resnet.layer3.0.shortcut.0" => (64, 128, [40, 499], 1, 2),
        "resnet.layer4.0.shortcut.0" => (128, 256, [20, 250], 1, 2),
        "resnet.layer3.0.conv1" => (64, 128, [40, 499], 3, 2),
        "resnet.layer4.0.conv1" => (128, 256, [20, 250], 3, 2),
        _ if C32_LAYERS.contains(&name) => (32, 32, [80, 998], 3, 1),
        _ if C64_LAYERS.contains(&name) => (64, 64, [40, 499], 3, 1),
        _ if name.starts_with("resnet.layer3.") => (128, 128, [20, 250], 3, 1),
        _ => (256, 256, [10, 125], 3, 1),
    };
    Ok(Conv2d {
        batch,
        in_channels,
        out_channels,
        input,
        kernel: [kernel; 2],
        padding: [kernel / 2; 2],
        stride: [stride; 2],
        dilation: [1; 2],
        math,
    })
}
