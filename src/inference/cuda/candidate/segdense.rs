//! Fixed-shape plans for the segmentation convolutions, the segmentation dense heads
//! and the embedding `seg_1` projection, on the `segdense` kernels
//!
//! Each plan replaces one library boundary, including the epilogue kernel it fuses
//! away. FP32 math runs full-FP32 kernels, or 3xTF32 tensor-core kernels on GPUs
//! whose TF32 tensor rate is several times their FP32 rate. TF32 math runs TF32
//! tensor-core kernels where the sm80 tier provides a faster one on this device,
//! and otherwise the FP32 kernel, which is at least as exact as the TF32 library
//! call it replaces. The configuration is a fixed function of the loaded PTX tier and
//! of [`DeviceAttributes`], never of run-time timing
//!
//! Every kernel bakes in its batch, so the plans implement exactly the model batches
//! 1 and 32

use std::sync::Arc;

use cudarc::driver::sys::CUfunction_attribute;
use cudarc::driver::{
    CudaFunction, CudaSlice, CudaStream, CudaView, CudaViewMut, LaunchConfig, PushKernelArg,
};

use super::{
    Batches, Coverage, CoverageEntry, DenseCandidate, DenseSite, DenseSpec, FiniteContract,
    GeometryError, InfinityContract, Maths, NanContract, Op, Phases, PlanError, Scratch,
    SegConvCandidate, SegConvSite, SegConvSpec, SignedZeroContract, SpecialValues,
};
use crate::inference::cuda::device::DeviceAttributes;
use crate::inference::cuda::error::check_len;
use crate::inference::cuda::{
    ComputeCapability, CudaError, CudaMath, CudaRuntime, KernelModule, LoadedKernels, PtxTier,
};

const CONTEXT: &str = "segdense plan";

/// The batches every kernel is compiled for
const BATCHES: [usize; 2] = [1, 32];

/// Output channels of a convolution padded to whole thread tiles
const CONV_PADDED: usize = 64;

/// Convolution taps
const TAPS: usize = 5;

/// Static shared memory any block may use without opting in
const DEFAULT_SHARED: u32 = 48 * 1024;

/// Most partial planes `spk_segdense_reduce_embed` adds: eight warps of up to 16
/// planes each, so the reduction takes one round trip to memory
const EMBED_MAX_SPLITS: u32 = 128;
/// Most partial planes `spk_segdense_reduce_embed_flat` adds, all held by one
/// thread
const FLAT_MAX_SPLITS: u32 = 40;

/// Reduction rows a `Select` kernel recomputes in 3xTF32, as `segdense_precision!`
/// fixes it in the kernel crate
const SELECT_ROWS: u32 = 16;

/// Reduction terms per stage of `spk_segdense_embed_b32_f16`: two warp groups of 16
const HALF_STAGE: u32 = 32;
/// Most reduction terms one `spk_segdense_embed_b32_f16` slice keeps in shared
/// memory
const HALF_SLICE: u32 = 320;
/// Fewest slices that keep every `spk_segdense_embed_b32_f16` slice within
/// [`HALF_SLICE`] terms
const HALF_MIN_SPLITS: u32 = (5120 / HALF_STAGE).div_ceil(HALF_SLICE / HALF_STAGE);
/// Dynamic shared memory of `spk_segdense_embed_b32_f16`, as `mma_half_split!`
/// documents it: 96 row scales, the FP16 slice of 96 rows and a three-stage weight
/// ring, which the cross-group reduction of 192 threads' 64 accumulators reuses
pub(super) const HALF_SHARED: u32 = {
    let ring = 96 * (HALF_SLICE / 2 + 4) + 3 * (HALF_STAGE / 2) * (128 + 8);
    let reduction = 192 * 32 * 64 / 32;
    4 * (96 + if ring > reduction { ring } else { reduction })
};

/// Finite operands give finite outputs within the FP32 range; NaN, infinity and the
/// sign of zero are not established, since the FP16 and TF32 splits and the fused
/// log-softmax do not follow IEEE order for them
const SPECIAL_VALUES: SpecialValues = SpecialValues {
    finite: FiniteContract::AbsoluteSum { headroom: 2 },
    nan: NanContract::Unspecified,
    infinity: InfinityContract::Unspecified,
    signed_zero: SignedZeroContract::Unspecified,
};

/// Device attributes that choose among a site's kernel configurations
///
/// The same sm80 PTX runs on parts whose dense TF32 tensor rate is about 8x their
/// FP32 rate and on parts where the two are about equal, with 34 to more than 100
/// SMs, so the best kernel differs by device
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Hardware {
    pub(super) capability: ComputeCapability,
    /// Streaming multiprocessors this context may use
    pub(super) multiprocessors: u32,
    /// Shared memory one block may opt in to, in bytes
    pub(super) shared_optin: u32,
}

impl Hardware {
    fn of(device: &DeviceAttributes) -> Self {
        Self {
            capability: device.capability(),
            multiprocessors: device.multiprocessors().get(),
            shared_optin: device.shared_optin_bytes(),
        }
    }

    /// Whether dense TF32 tensor throughput is several times the FP32 SIMT rate
    ///
    /// Compute capabilities 8.0 (A100, A30), 9.0 (H100) and 10.0 (B200) are the
    /// data-center parts with full-rate tensor cores. GeForce and workstation parts
    /// (8.6, 8.9, 12.x) run dense TF32 at about their FP32 rate, where three TF32
    /// products cost more than one FP32 SIMT product, so FP32 math keeps the SIMT
    /// kernels there; unknown capabilities keep them too, as they do not depend on
    /// that ratio
    fn tensor_heavy(self) -> bool {
        matches!(
            (self.capability.major, self.capability.minor),
            (8, 0) | (9, 0) | (10, 0)
        )
    }

    /// Whether this is a consumer Blackwell part (12.x), where the embedding's
    /// TF32 split-K runs fastest as one 384-thread block per SM with an in-block
    /// k-split: measured on a 36-SM RTX 5060 Ti at 1.01x the library against 0.87x
    /// for the two 192-thread blocks per SM that win on Ada (1.06x against 0.98x on
    /// a 34-SM RTX 4060 Ti)
    fn consumer_blackwell(self) -> bool {
        self.capability.major == 12
    }

    /// Whether `spk_segdense_embed_b32_f16` fits: one resident block per SM must
    /// give it at least [`HALF_MIN_SPLITS`] slices, and a block must be able to opt
    /// in to its shared memory
    fn half_split_fits(self) -> bool {
        self.splits(2, 1) >= HALF_MIN_SPLITS && self.shared_optin >= HALF_SHARED
    }

    /// Reduction slices for a split-K kernel with `tiles` output tiles and
    /// `resident` blocks per SM: one block per resident slot, so every slice of the
    /// single wave starts at once and none waits for a second wave
    fn splits(self, tiles: u32, resident: u32) -> u32 {
        (self.multiprocessors * resident / tiles).clamp(1, EMBED_MAX_SPLITS)
    }
}

/// The boundaries this area implements and their layout contracts
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Site {
    /// Valid five-tap NCW convolution, 80 to 60 channels, without bias
    Conv1,
    /// Valid five-tap NCW convolution, 60 to 60 channels, without bias
    Conv2,
    /// Row-major 256-to-128 projection `[k, n]` with bias and LeakyReLU
    Linear0,
    /// Row-major 128-to-128 projection `[k, n]` with bias and LeakyReLU
    Linear1,
    /// Row-major 128-to-7 projection `[k, n]` with bias and log-softmax
    Linear2,
    /// Three masks per chunk, 5120-to-256 projection with weight `[n, k]` and bias
    Embedding,
}

impl From<DenseSite> for Site {
    fn from(site: DenseSite) -> Self {
        match site {
            DenseSite::Linear0 => Self::Linear0,
            DenseSite::Linear1 => Self::Linear1,
            DenseSite::Classifier => Self::Linear2,
            DenseSite::Embedding => Self::Embedding,
        }
    }
}

impl From<SegConvSite> for Site {
    fn from(site: SegConvSite) -> Self {
        match site {
            SegConvSite::Conv1 => Self::Conv1,
            SegConvSite::Conv2 => Self::Conv2,
        }
    }
}

impl Site {
    /// Resolve the model boundary without duplicating the site's ownership table
    fn from_boundary(
        boundary: crate::inference::cuda::implementation::BoundaryId,
    ) -> Result<Self, PlanError> {
        Ok(match boundary.name() {
            "sincnet.conv1" => Site::Conv1,
            "sincnet.conv2" => Site::Conv2,
            "linear0" => Site::Linear0,
            "linear1" => Site::Linear1,
            "linear2" => Site::Linear2,
            "resnet.seg_1" => Site::Embedding,
            _ => {
                return Err(PlanError::Geometry(GeometryError::Unimplemented {
                    context: CONTEXT,
                    reason: format!("unknown boundary {boundary}"),
                }));
            }
        })
    }

    /// `(rows per batch item, columns, reduction, NCW input length)`; convolution
    /// rows are output positions
    pub(crate) const fn dimensions(self) -> (usize, usize, usize, usize) {
        match self {
            Self::Conv1 => (5321, 60, 400, 5325),
            Self::Conv2 => (1769, 60, 300, 1773),
            Self::Linear0 => (589, 128, 256, 0),
            Self::Linear1 => (589, 128, 128, 0),
            Self::Linear2 => (589, 7, 128, 0),
            Self::Embedding => (3, 256, 5120, 0),
        }
    }

    const fn is_conv(self) -> bool {
        matches!(self, Self::Conv1 | Self::Conv2)
    }

    /// The kernel of this site for `batch`, `math`, the loaded PTX `tier` and the
    /// device, if the batch is one the kernels are compiled for
    pub(super) fn choice(
        self,
        batch: usize,
        math: CudaMath,
        tier: PtxTier,
        hardware: Hardware,
    ) -> Option<Choice> {
        if tier >= PtxTier::Sm80
            && let Some(choice) = self.tensor_choice(batch, math, hardware)
        {
            return Some(choice);
        }

        self.scalar_choice(batch, math, hardware)
    }

    /// The non-tensor algorithm, with the same fixed-order reduction as normal routing
    fn scalar_choice(self, batch: usize, math: CudaMath, hardware: Hardware) -> Option<Choice> {
        let entry = match (self, batch) {
            (Self::Conv1, 1) => Entry::Conv1B1,
            (Self::Conv1, 32) => Entry::Conv1B32,
            (Self::Conv2, 1) => Entry::Conv2B1,
            (Self::Conv2, 32) => Entry::Conv2B32,
            (Self::Linear0, 1) => Entry::Linear0B1,
            (Self::Linear0, 32) => Entry::Linear0B32,
            // the FP32 kernel is compensated, which costs time and buys nothing against
            // TF32 library error; the TF32 kernel's sm75 expansion is the plain SIMT one
            (Self::Linear1, 1) if math == CudaMath::Tf32 => Entry::Linear1B1Tf32,
            (Self::Linear1, 1) => Entry::Linear1B1,
            (Self::Linear1, 32) => Entry::Linear1B32,
            (Self::Linear2, 1) => Entry::ClassifierB1,
            (Self::Linear2, 32) => Entry::ClassifierB32,
            (Self::Embedding, 1) => Entry::EmbedB1,
            // two resident blocks per SM
            (Self::Embedding, 32) => {
                return Some(Choice::split(Entry::EmbedB32, hardware.splits(2, 2)));
            }
            _ => return None,
        };

        Some(Choice::fixed(entry))
    }

    /// The sm80-tier tensor-core kernel of this site, where one beats both the
    /// library and the SIMT kernel on this device
    fn tensor_choice(self, batch: usize, math: CudaMath, hardware: Hardware) -> Option<Choice> {
        let heavy = hardware.tensor_heavy();
        let entry = match (self, batch, math) {
            // where TF32 runs at about the FP32 rate, the FP32 SIMT convolution
            // already beats the TF32 library and stays exact, while cuDNN's TF32 mode
            // can pick an FP32-accurate algorithm (conv2 on the RTX 5060 Ti) that a
            // TF32 kernel cannot match in accuracy
            (Self::Conv1, 32, CudaMath::Tf32) if heavy => Entry::Conv1B32Tc,
            (Self::Conv1, 32, CudaMath::Fp32) if heavy => Entry::Conv1B32X3,
            (Self::Conv2, 32, CudaMath::Tf32) if heavy => Entry::Conv2B32Tc,
            (Self::Conv2, 32, CudaMath::Fp32) if heavy => Entry::Conv2B32X3,
            (Self::Linear0, 1, CudaMath::Tf32) => Entry::Linear0B1Tf32,
            // batch 1 keeps plain TF32: the residual products' extra latency cost it
            // most of its margin (1.01x the library on Ada against 1.21x)
            (Self::Linear0, 32, CudaMath::Tf32) => Entry::Linear0B32Select,
            (Self::Linear1, 32, CudaMath::Tf32) => Entry::Linear1B32Tf32,
            // where TF32 mma runs at several times the FP32 rate, the main kernel is
            // short and the split-K reduction's cost grows with its slice count: 96 x 64
            // tiles fill one wave with half the slices (54 against 108 on an A100,
            // measured 1.07x the library against 0.91x)
            (Self::Embedding, 32, CudaMath::Tf32) if heavy => {
                return Some(Choice::split(Entry::EmbedB32Tf32E64, hardware.splits(4, 2)));
            }
            // where TF32 runs at about the FP32 rate, FP16 products run at twice it;
            // one resident 384-thread block per SM
            (Self::Embedding, 32, CudaMath::Tf32) if hardware.half_split_fits() => {
                return Some(Choice::split(Entry::EmbedB32F16, hardware.splits(2, 1)));
            }
            (Self::Embedding, 32, CudaMath::Tf32) => return Some(Self::tf32_embedding(hardware)),
            // four 96 x 64 tiles, two resident blocks per SM
            (Self::Embedding, 32, CudaMath::Fp32) if heavy => {
                return Some(Choice::split(Entry::EmbedB32X3, hardware.splits(4, 2)));
            }
            _ => return None,
        };

        Some(Choice::fixed(entry))
    }

    /// The batch-32 TF32 embedding kernel where TF32 runs at about the FP32 rate
    fn tf32_embedding(hardware: Hardware) -> Choice {
        if hardware.consumer_blackwell() {
            // one resident block per SM
            return Choice::split(Entry::EmbedB32Tf32K2, hardware.splits(2, 1));
        }

        // two resident blocks per SM
        Choice::split(Entry::EmbedB32Tf32, hardware.splits(2, 2))
    }

    /// The TF32 kernel that FP16 products replace, so the tuner can time an approved
    /// kernel of ours where the device rule picks the FP16 one
    fn tuning_tf32_choice(
        self,
        batch: usize,
        math: CudaMath,
        hardware: Hardware,
    ) -> Option<Choice> {
        let default = self.tensor_choice(batch, math, hardware)?;
        (default.entry == Entry::EmbedB32F16).then(|| Self::tf32_embedding(hardware))
    }
}

/// One kernel entry of the area; its site, batch, tile, block and shared memory are
/// fixed, and split-K entries also take a slice count
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Entry {
    Conv1B1,
    Conv1B32,
    /// sm80 tier: TF32 tensor cores
    Conv1B32Tc,
    /// sm80 tier: 3xTF32 tensor cores
    Conv1B32X3,
    Conv2B1,
    Conv2B32,
    /// sm80 tier: TF32 tensor cores
    Conv2B32Tc,
    /// sm80 tier: 3xTF32 tensor cores
    Conv2B32X3,
    Linear0B1,
    /// sm80 tier: TF32 tensor cores
    Linear0B1Tf32,
    Linear0B32,
    /// sm80 tier: TF32 tensor cores with the `Select` 3xTF32 rows
    Linear0B32Select,
    Linear1B1,
    /// TF32 tensor cores on the sm80 tier, plain SIMT on sm75
    Linear1B1Tf32,
    Linear1B32,
    /// sm80 tier: TF32 tensor cores
    Linear1B32Tf32,
    ClassifierB1,
    ClassifierB32,
    EmbedB1,
    /// Split-K FP32 SIMT
    EmbedB32,
    /// sm80 tier: split-K TF32, two resident blocks per SM
    EmbedB32Tf32,
    /// sm80 tier: split-K TF32 with an in-block k-split, one block per SM
    EmbedB32Tf32K2,
    /// sm80 tier: split-K TF32 on 96 x 64 tiles
    EmbedB32Tf32E64,
    /// sm80 tier: split-K 3xTF32 on 96 x 64 tiles
    EmbedB32X3,
    /// sm80 tier: split-K scaled FP16 products
    EmbedB32F16,
}

impl Entry {
    pub(super) const fn site(self) -> Site {
        match self {
            Self::Conv1B1 | Self::Conv1B32 | Self::Conv1B32Tc | Self::Conv1B32X3 => Site::Conv1,
            Self::Conv2B1 | Self::Conv2B32 | Self::Conv2B32Tc | Self::Conv2B32X3 => Site::Conv2,
            Self::Linear0B1 | Self::Linear0B1Tf32 | Self::Linear0B32 | Self::Linear0B32Select => {
                Site::Linear0
            }
            Self::Linear1B1 | Self::Linear1B1Tf32 | Self::Linear1B32 | Self::Linear1B32Tf32 => {
                Site::Linear1
            }
            Self::ClassifierB1 | Self::ClassifierB32 => Site::Linear2,
            Self::EmbedB1
            | Self::EmbedB32
            | Self::EmbedB32Tf32
            | Self::EmbedB32Tf32K2
            | Self::EmbedB32Tf32E64
            | Self::EmbedB32X3
            | Self::EmbedB32F16 => Site::Embedding,
        }
    }

    pub(super) const fn batch(self) -> usize {
        match self {
            Self::Conv1B1
            | Self::Conv2B1
            | Self::Linear0B1
            | Self::Linear0B1Tf32
            | Self::Linear1B1
            | Self::Linear1B1Tf32
            | Self::ClassifierB1
            | Self::EmbedB1 => 1,
            _ => 32,
        }
    }

    /// The launch shape, with `splits` reduction slices for a split-K entry
    pub(super) const fn config(self, splits: u32) -> Config {
        match self {
            Self::Conv1B1 => Config::conv("spk_segdense_conv1_b1", 64, 256),
            Self::Conv1B32 => Config::conv("spk_segdense_conv1_b32", 128, 128),
            Self::Conv1B32Tc => Config::tensor_conv("spk_segdense_conv1_b32_tc", 128, 80, 1, 128),
            Self::Conv1B32X3 => Config::tensor_conv("spk_segdense_conv1_b32_x3", 128, 80, 2, 256),
            Self::Conv2B1 => Config::conv("spk_segdense_conv2_b1", 32, 256),
            Self::Conv2B32 => Config::conv("spk_segdense_conv2_b32", 128, 128),
            Self::Conv2B32Tc => Config::tensor_conv("spk_segdense_conv2_b32_tc", 128, 64, 1, 128),
            Self::Conv2B32X3 => Config::tensor_conv("spk_segdense_conv2_b32_x3", 128, 64, 2, 256),
            Self::Linear0B1 => Config::gemm("spk_segdense_linear0_b1", 16, 32, 128),
            Self::Linear0B1Tf32 => Config::gemm("spk_segdense_linear0_b1_tf32", 32, 32, 128),
            Self::Linear0B32 => Config::gemm("spk_segdense_linear0_b32", 64, 128, 128),
            Self::Linear0B32Select => {
                Config::select_gemm("spk_segdense_linear0_b32_tf32", 64, 128, 128)
            }
            Self::Linear1B1 => Config::gemm("spk_segdense_linear1_b1", 16, 32, 128),
            Self::Linear1B1Tf32 => Config::gemm("spk_segdense_linear1_b1_tf32", 16, 32, 128),
            Self::Linear1B32 => Config::gemm("spk_segdense_linear1_b32", 64, 128, 128),
            Self::Linear1B32Tf32 => Config::gemm("spk_segdense_linear1_b32_tf32", 64, 128, 128),
            Self::ClassifierB1 => Config::classifier("spk_segdense_classifier_b1", 8, 64),
            Self::ClassifierB32 => Config::classifier("spk_segdense_classifier_b32", 128, 256),
            Self::EmbedB1 => Config {
                kernel: "spk_segdense_embed_b1",
                kind: Kind::Gemv { cols: 2 },
                block: 256,
                shared: 0,
            },
            Self::EmbedB32 => Config::split("spk_segdense_embed_b32", 128, splits, 192),
            Self::EmbedB32Tf32 => Config::split("spk_segdense_embed_b32_tf32", 128, splits, 192),
            Self::EmbedB32Tf32K2 => {
                Config::split("spk_segdense_embed_b32_tf32_k2", 128, splits, 384)
            }
            Self::EmbedB32Tf32E64 => {
                Config::split("spk_segdense_embed_b32_tf32_e64", 64, splits, 192)
            }
            Self::EmbedB32X3 => Config::split("spk_segdense_embed_b32_x3", 64, splits, 192),
            Self::EmbedB32F16 => Config::half_split(splits),
        }
    }

    pub(super) const fn is_split(self) -> bool {
        matches!(self.config(1).kind, Kind::Split { .. })
    }
}

/// A kernel entry and, for split-K entries, its reduction slice count
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Choice {
    pub(super) entry: Entry,
    /// 1 to [`EMBED_MAX_SPLITS`] for split-K entries, 0 otherwise
    pub(super) splits: u8,
}

impl Choice {
    pub(super) const fn fixed(entry: Entry) -> Self {
        Self { entry, splits: 0 }
    }

    /// `splits` is within 1 to [`EMBED_MAX_SPLITS`], as [`Hardware::splits`] clamps it
    const fn split(entry: Entry, splits: u32) -> Self {
        Self {
            entry,
            splits: splits as u8,
        }
    }

    pub(super) const fn config(self) -> Config {
        self.entry.config(self.splits as u32)
    }
}

/// How a kernel tiles its boundary, which fixes its grid and operand layouts
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Kind {
    /// One block per `positions` output positions of one batch item; weights
    /// packed `[k, 64]`
    Conv { positions: u32 },
    /// One block per `positions` output positions of one batch item on tensor
    /// cores; weights packed as `mma` fragments over `cin_pad` input channels, in
    /// `planes` TF32 planes (2 for the 3xTF32 split)
    TensorConv {
        positions: u32,
        cin_pad: u32,
        planes: u32,
    },
    /// One block per `rows x cols` output tile; weights `[k, n]`, followed by the
    /// indices of the `select` reduction rows the kernel recomputes in 3xTF32
    Gemm { rows: u32, cols: u32, select: u32 },
    /// One block per `rows` output rows; weights `[k, n]`
    Classifier { rows: u32 },
    /// One block per `cols` output columns; weights `[n, k]`
    Gemv { cols: u32 },
    /// One block per `rows x cols` tile and reduction slice, then the fixed-order
    /// reduction of the `splits` partial planes
    Split {
        rows: u32,
        cols: u32,
        splits: u32,
        weights: SplitWeights,
    },
}

/// How a split-K kernel reads its weights
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SplitWeights {
    /// FP32 `[k, n]`
    Fp32,
    /// FP16 pairs of adjacent reduction rows as `[k / 2, n]` words, scaled by a
    /// power of two per column, then the `n` inverse scales ([`pack_half_columns`])
    ScaledHalf,
}

/// One site kernel and its launch shape
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Config {
    pub kernel: &'static str,
    pub kind: Kind,
    pub block: u32,
    /// Dynamic shared memory per block, in bytes
    pub shared: u32,
}

impl Config {
    const fn conv(kernel: &'static str, positions: u32, block: u32) -> Self {
        Self {
            kernel,
            kind: Kind::Conv { positions },
            block,
            shared: 0,
        }
    }

    /// A tensor-core convolution with `cin_pad` input channels, in stages of 8
    /// input channels three deep, as `mma_conv!` documents its shared memory
    const fn tensor_conv(
        kernel: &'static str,
        positions: u32,
        cin_pad: u32,
        planes: u32,
        block: u32,
    ) -> Self {
        const STAGES: u32 = 3;
        let stage = 8 * (positions + 16) + 2560 * planes;
        let epilogue = 64 * (positions + 8);
        let floats = if STAGES * stage > epilogue {
            STAGES * stage
        } else {
            epilogue
        };

        Self {
            kernel,
            kind: Kind::TensorConv {
                positions,
                cin_pad,
                planes,
            },
            block,
            shared: 4 * floats,
        }
    }

    const fn gemm(kernel: &'static str, rows: u32, cols: u32, block: u32) -> Self {
        Self {
            kernel,
            kind: Kind::Gemm {
                rows,
                cols,
                select: 0,
            },
            block,
            shared: 0,
        }
    }

    /// A TF32 GEMM that recomputes [`SELECT_ROWS`] reduction rows in 3xTF32, the
    /// `Select` precision
    const fn select_gemm(kernel: &'static str, rows: u32, cols: u32, block: u32) -> Self {
        Self {
            kernel,
            kind: Kind::Gemm {
                rows,
                cols,
                select: SELECT_ROWS,
            },
            block,
            shared: 0,
        }
    }

    const fn classifier(kernel: &'static str, rows: u32, block: u32) -> Self {
        Self {
            kernel,
            kind: Kind::Classifier { rows },
            block,
            shared: 0,
        }
    }

    /// An embedding split-K kernel with 96-row tiles of `cols` columns
    const fn split(kernel: &'static str, cols: u32, splits: u32, block: u32) -> Self {
        Self {
            kernel,
            kind: Kind::Split {
                rows: 96,
                cols,
                splits,
                weights: SplitWeights::Fp32,
            },
            block,
            shared: 0,
        }
    }

    /// `spk_segdense_embed_b32_f16` with `splits` reduction slices
    const fn half_split(splits: u32) -> Self {
        Self {
            kernel: "spk_segdense_embed_b32_f16",
            kind: Kind::Split {
                rows: 96,
                cols: 128,
                splits,
                weights: SplitWeights::ScaledHalf,
            },
            block: 384,
            shared: HALF_SHARED,
        }
    }
}

/// The complete execution choice of one segdense plan: the kernel entry the device
/// rule picked, its split count, and the math mode and loaded tier it was picked for
///
/// The entry fixes the site and batch. The tier is the one the rule saw: sm80-tier
/// kernels are trapping stubs in the sm75 PTX, so a plan refuses a module below it
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SegdensePin {
    choice: Choice,
    math: CudaMath,
    tier: PtxTier,
}

impl SegdensePin {
    /// The configuration the device rule picks for this boundary on the module
    /// `tier` and `device`
    pub(crate) fn select(
        site: Site,
        batch: usize,
        math: CudaMath,
        tier: PtxTier,
        device: &DeviceAttributes,
    ) -> Result<Self, PlanError> {
        let choice = site
            .choice(batch, math, tier, Hardware::of(device))
            .ok_or_else(|| {
                PlanError::Geometry(GeometryError::Unimplemented {
                    context: CONTEXT,
                    reason: format!("{site:?} batch {batch}; the kernels compile batches 1 and 32"),
                })
            })?;

        Ok(Self { choice, math, tier })
    }

    /// The TF32 algorithm that FP16 products replace on this device, if they do
    fn select_tuning_tf32(
        site: Site,
        batch: usize,
        math: CudaMath,
        tier: PtxTier,
        device: &DeviceAttributes,
    ) -> Option<Self> {
        if tier < PtxTier::Sm80 {
            return None;
        }

        let choice = site.tuning_tf32_choice(batch, math, Hardware::of(device))?;
        Some(Self { choice, math, tier })
    }

    /// A scalar FP32 algorithm with the requested pipeline mode retained in its pin
    fn select_fp32(
        site: Site,
        batch: usize,
        math: CudaMath,
        tier: PtxTier,
        device: &DeviceAttributes,
    ) -> Result<Self, PlanError> {
        let choice = site
            .scalar_choice(batch, CudaMath::Fp32, Hardware::of(device))
            .ok_or_else(|| {
                PlanError::Geometry(GeometryError::Unimplemented {
                    context: CONTEXT,
                    reason: format!("{site:?} batch {batch}; the kernels compile batches 1 and 32"),
                })
            })?;

        Ok(Self { choice, math, tier })
    }

    /// The arithmetic algorithm, independent of the launch tile and split count
    pub(crate) const fn entry(self) -> Entry {
        self.choice.entry
    }

    /// The kernel configuration
    pub(crate) const fn config(self) -> Config {
        self.choice.config()
    }

    /// The split-K slice count, for split-K kernels
    pub(crate) const fn splits(self) -> Option<u32> {
        if self.choice.entry.is_split() {
            Some(self.choice.splits as u32)
        } else {
            None
        }
    }

    /// Refuse a pin for another boundary or a module that lacks its kernels
    pub(super) fn check(
        self,
        site: Site,
        batch: usize,
        math: CudaMath,
        tier: PtxTier,
    ) -> Result<(), PlanError> {
        let entry = self.choice.entry;
        if (entry.site(), entry.batch(), self.math) != (site, batch, math) {
            return Err(PlanError::Geometry(GeometryError::Invalid {
                context: CONTEXT,
                reason: format!(
                    "pin {entry:?} {:?} cannot plan {site:?} batch {batch} {math:?}",
                    self.math
                ),
            }));
        }
        if self.tier > tier {
            return Err(PlanError::DeviceUnsupported {
                reason: format!(
                    "{} needs the {} segdense module, the loaded one is {tier}",
                    self.config().kernel,
                    self.tier
                ),
            });
        }
        Ok(())
    }
}

/// A packed-weight producer with no library dependency at enqueue time
#[derive(Debug)]
struct Producer {
    main: CudaFunction,
    /// The fixed-order reduction of the partial planes, for split-K
    reduce: Option<Reduction>,
    config: Config,
    weight: CudaSlice<f32>,
    bias: CudaSlice<f32>,
    /// Split-K partial planes, written by the main kernel and read by the reduction
    partials: Scratch<f32>,
    batch: usize,
    rows: usize,
    n: usize,
    k: usize,
    input_len: usize,
}

impl Producer {
    /// Packs weights and makes the fixed-shape launch plan before capture
    ///
    /// `bias` is `None` for the convolutions, whose bias the shared pool consumer adds
    fn plan(
        runtime: &CudaRuntime,
        kernels: &LoadedKernels,
        weight: &CudaSlice<f32>,
        bias: Option<&CudaSlice<f32>>,
        pin: SegdensePin,
    ) -> Result<Self, PlanError> {
        if kernels.request().area() != KernelModule::Segdense {
            return Err(PlanError::Geometry(GeometryError::Invalid {
                context: CONTEXT,
                reason: format!(
                    "the {} module cannot run segdense plans",
                    kernels.request().area().name()
                ),
            }));
        }

        let config = pin.config();
        let site = pin.choice.entry.site();
        let batch = pin.choice.entry.batch();
        let (rows, n, k, input_len) = site.dimensions();
        check_len(CONTEXT, n * k, weight.len())?;
        if site.is_conv() != bias.is_none() {
            return Err(PlanError::Geometry(GeometryError::Invalid {
                context: CONTEXT,
                reason: format!("{site:?} takes a bias exactly when it is not a convolution"),
            }));
        }
        if let Some(bias) = bias {
            check_len(CONTEXT, n, bias.len())?;
        }

        if let Kind::Split { splits, .. } = config.kind
            && !(1..=EMBED_MAX_SPLITS).contains(&splits)
        {
            return Err(PlanError::Geometry(GeometryError::Invalid {
                context: CONTEXT,
                reason: format!(
                    "{splits} split-K slices, the reduction adds 1 to {EMBED_MAX_SPLITS}"
                ),
            }));
        }
        if let Kind::Split {
            splits,
            weights: SplitWeights::ScaledHalf,
            ..
        } = config.kind
            && splits < HALF_MIN_SPLITS
        {
            return Err(PlanError::Geometry(GeometryError::Invalid {
                context: CONTEXT,
                reason: format!(
                    "{splits} split-K slices, the FP16 kernel needs at least {HALF_MIN_SPLITS}"
                ),
            }));
        }

        let main = kernels.function(config.kernel)?;
        if config.shared > DEFAULT_SHARED {
            let optin = runtime.device().shared_optin_bytes();
            if config.shared > optin {
                return Err(PlanError::DeviceUnsupported {
                    reason: format!(
                        "{} needs {} bytes of shared memory, the device allows {optin}",
                        config.kernel, config.shared
                    ),
                });
            }

            main.set_attribute(
                CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                config.shared as i32,
            )?;
        }

        let stream = runtime.stream();
        let pack = kernels.function("spk_segdense_pack")?;
        let packed = match config.kind {
            Kind::Conv { .. } => {
                pack_on_device(stream, &pack, weight, k * CONV_PADDED, [n, k, CONV_PADDED])?
            }
            Kind::Split {
                weights: SplitWeights::ScaledHalf,
                ..
            } => {
                let weight = stream.clone_dtoh(weight)?;
                stream.clone_htod(&pack_half_columns(&weight, n, k))?
            }
            Kind::Split { .. } => pack_on_device(stream, &pack, weight, k * n, [n, k, n])?,
            Kind::TensorConv {
                cin_pad, planes, ..
            } => {
                let (cin_pad, planes) = (cin_pad as usize, planes as usize);
                let len = planes * cin_pad * TAPS * CONV_PADDED;
                let function = kernels.function("spk_segdense_pack_conv_mma")?;
                pack_on_device(stream, &function, weight, len, [k / TAPS, cin_pad, planes])?
            }
            Kind::Gemm { select, .. } if select > 0 => {
                let mut weight = stream.clone_dtoh(weight)?;
                let rows = selected_rows(&weight, n, select as usize);
                weight.extend(rows.into_iter().map(f32::from_bits));
                stream.clone_htod(&weight)?
            }
            Kind::Gemm { .. } | Kind::Classifier { .. } | Kind::Gemv { .. } => {
                stream.clone_dtod(weight)?
            }
        };

        let bias = match bias {
            Some(bias) => stream.clone_dtod(bias)?,
            None => stream.alloc_zeros(0)?,
        };
        let workspace = match config.kind {
            Kind::Split { splits, .. } => splits as usize * batch * rows * n,
            _ => 0,
        };
        let partials = Scratch::zeros(runtime, workspace)?;
        let reduce = match config.kind {
            Kind::Split { splits, .. } => {
                let shape = Reduction::shape(splits);
                Some(Reduction {
                    function: kernels.function(shape.kernel)?,
                    shape,
                })
            }
            _ => None,
        };

        // packing can use another stream than the eventual captured graph
        runtime.synchronize()?;
        Ok(Self {
            main,
            reduce,
            config,
            weight: packed,
            bias,
            partials,
            batch,
            rows,
            n,
            k,
            input_len,
        })
    }

    /// Queues the producer and, for split-K, its fixed-order reduction
    fn enqueue(
        &self,
        input: &CudaView<'_, f32>,
        output: &mut CudaViewMut<'_, f32>,
        phases: &Phases,
        stream: &CudaStream,
    ) -> Result<(), CudaError> {
        let input_elements = if self.input_len == 0 {
            self.batch * self.rows * self.k
        } else {
            self.batch * (self.k / TAPS) * self.input_len
        };

        check_len(CONTEXT, input_elements, input.len())?;
        check_len(CONTEXT, self.batch * self.rows * self.n, output.len())?;

        let m = (self.batch * self.rows) as u32;
        let grid = match self.config.kind {
            Kind::Conv { positions } | Kind::TensorConv { positions, .. } => (
                self.rows.div_ceil(positions as usize) as u32,
                self.batch as u32,
                1,
            ),
            Kind::Gemm { rows, cols, .. } => (m.div_ceil(rows), self.n as u32 / cols, 1),
            Kind::Classifier { rows } => (m.div_ceil(rows), 1, 1),
            Kind::Gemv { cols } => (self.n as u32 / cols, 1, 1),
            Kind::Split {
                rows, cols, splits, ..
            } => (m.div_ceil(rows), self.n as u32 / cols, splits),
        };
        let config = LaunchConfig {
            grid_dim: grid,
            block_dim: (self.config.block, 1, 1),
            shared_mem_bytes: self.config.shared,
        };

        let input_len = input.len() as u64;
        let weight_len = self.weight.len() as u64;
        let bias_len = self.bias.len() as u64;
        let output_len = output.len() as u64;
        let Some(reduce) = &self.reduce else {
            return phases.op(Op::Main, || {
                let mut launch = stream.launch_builder(&self.main);
                launch
                    .arg(input)
                    .arg(&input_len)
                    .arg(&self.weight)
                    .arg(&weight_len);
                if !matches!(
                    self.config.kind,
                    Kind::Conv { .. } | Kind::TensorConv { .. }
                ) {
                    launch.arg(&self.bias).arg(&bias_len);
                }
                launch.arg(&mut *output).arg(&output_len);
                // safety: the plan fixes every dimension, the checks above cover each
                // buffer, and the grid gives every output element one writer
                unsafe { launch.launch(config) }?;
                Ok(())
            });
        };

        let mut partials = self.partials.get();
        let partials_len = partials.len() as u64;
        phases.op(Op::Main, || {
            let mut launch = stream.launch_builder(&self.main);
            launch
                .arg(input)
                .arg(&input_len)
                .arg(&self.weight)
                .arg(&weight_len)
                .arg(&self.bias)
                .arg(&bias_len)
                .arg(&mut *partials)
                .arg(&partials_len);
            // safety: as above; block `z` writes only partial plane `z`
            unsafe { launch.launch(config) }?;
            Ok(())
        })?;

        let shape = reduce.shape;
        phases.op(Op::Reduce, || {
            let mut launch = stream.launch_builder(&reduce.function);
            launch
                .arg(&*partials)
                .arg(&partials_len)
                .arg(&self.bias)
                .arg(&bias_len)
                .arg(&mut *output)
                .arg(&output_len)
                .arg(&shape.splits);
            let config = LaunchConfig {
                grid_dim: (shape.grid, 1, 1),
                block_dim: (shape.block, 1, 1),
                shared_mem_bytes: 0,
            };
            // safety: partials hold `splits` output-sized planes, within the kernel's
            // limit, and the shape covers each four output elements once
            unsafe { launch.launch(config) }?;
            Ok(())
        })
    }
}

/// Runs a pack kernel over a new buffer of `len` elements, with the source weight and
/// the three `dims` as its remaining arguments
fn pack_on_device(
    stream: &Arc<CudaStream>,
    function: &CudaFunction,
    source: &CudaSlice<f32>,
    len: usize,
    dims: [usize; 3],
) -> Result<CudaSlice<f32>, CudaError> {
    let mut weight = stream.alloc_zeros(len)?;
    let lens = [source.len() as u64, weight.len() as u64];
    let dims = dims.map(|value| value as u32);
    let mut launch = stream.launch_builder(function);
    launch
        .arg(source)
        .arg(&lens[0])
        .arg(&mut weight)
        .arg(&lens[1])
        .arg(&dims[0])
        .arg(&dims[1])
        .arg(&dims[2]);
    // safety: the source has its checked `n * k` elements, both pack kernels read only
    // source elements their dimensions name, and the destination has one thread per
    // element
    unsafe { launch.launch(LaunchConfig::for_num_elems(len as u32)) }?;
    Ok(weight)
}

/// The `count` rows of a row-major `[k, n]` weight with the largest sum of squares,
/// in ascending order, ties going to the lower index
///
/// TF32 rounding error of a product term grows with the weight's magnitude, so
/// recomputing these rows in 3xTF32 removes the largest share of the error a fixed
/// number of rows can. The choice depends on the weights alone, never on the inputs,
/// so it holds for every input the model sees
pub(super) fn selected_rows(weight: &[f32], n: usize, count: usize) -> Vec<u32> {
    let mut energy: Vec<(f64, u32)> = weight
        .chunks_exact(n)
        .zip(0u32..)
        .map(|(row, index)| {
            let sum = row.iter().map(|&value| f64::from(value).powi(2)).sum();
            (sum, index)
        })
        .collect();

    energy.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    let mut rows: Vec<u32> = energy
        .into_iter()
        .take(count)
        .map(|(_, index)| index)
        .collect();
    rows.sort_unstable();
    rows
}

/// Packs `weight [n, k]` for `spk_segdense_embed_b32_f16`: word `p * n + col` holds
/// the FP16 pair `(w[col][2p] s, w[col][2p + 1] s)` with the low half first, and
/// float `k / 2 * n + col` holds `1 / s`, where the power of two `s` puts the
/// column's largest magnitude in `[2^14, 2^15)` ([`half_scale`])
pub(super) fn pack_half_columns(weight: &[f32], n: usize, k: usize) -> Vec<f32> {
    let mut packed = vec![0.0; k / 2 * n + n];
    for (col, row) in weight.chunks_exact(k).take(n).enumerate() {
        let largest = row
            .iter()
            .fold(0.0f32, |largest, value| largest.max(value.abs()));
        let (scale, inverse) = half_scale(largest);
        for (pair, &[first, second]) in row.as_chunks::<2>().0.iter().enumerate() {
            let low = u32::from(f16_bits(first * scale));
            let high = u32::from(f16_bits(second * scale));
            packed[pair * n + col] = f32::from_bits(low | (high << 16));
        }
        packed[k / 2 * n + col] = inverse;
    }

    packed
}

/// The power of two `s` that puts `largest` in `[2^14, 2^15)`, below the FP16 maximum
/// of 65504 even after rounding, and `1 / s`; the exponent is clamped so both are
/// normal floats. The kernel crate's `half_scale` scales activation rows the same way
pub(super) fn half_scale(largest: f32) -> (f32, f32) {
    let exponent = ((largest.to_bits() >> 23) & 0xff) as i32 - 127;
    let shift = (14 - exponent).clamp(-113, 126);
    (
        f32::from_bits(((shift + 127) as u32) << 23),
        f32::from_bits(((127 - shift) as u32) << 23),
    )
}

/// `value` rounded to the nearest FP16, ties to even, as binary16 bits, which is what
/// the kernels' `cvt.rn.f16x2.f32` gives
pub(super) fn f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x7f_ffff;
    if exponent == 0xff {
        // infinities stay infinite and NaNs stay NaN
        return sign | 0x7c00 | if mantissa == 0 { 0 } else { 0x200 };
    }

    let half_exponent = exponent - 127 + 15;
    if half_exponent >= 1 {
        let base = ((half_exponent as u32) << 10) | (mantissa >> 13);
        let rounded = base + u32::from(round_up(mantissa, 13, base));
        // a carry past the largest finite value gives the infinity encoding
        return sign | rounded.min(0x7c00) as u16;
    }

    // below the normal range the result counts steps of 2^-24, the smallest FP16
    // subnormal; FP32 subnormals and anything under half a step round to zero
    let shift = 126 - exponent;
    if exponent == 0 || shift > 24 {
        return sign;
    }

    let significand = mantissa | 0x80_0000;
    let steps = significand >> shift;
    sign | (steps + u32::from(round_up(significand, shift as u32, steps))) as u16
}

/// Whether dropping the low `bits` bits of `value`, leaving `kept`, rounds up under
/// ties-to-even
fn round_up(value: u32, bits: u32, kept: u32) -> bool {
    let rest = value & ((1 << bits) - 1);
    let half = 1 << (bits - 1);
    rest > half || (rest == half && kept & 1 == 1)
}

/// The split-K reduction kernel for a number of partial planes, and its launch
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ReductionShape {
    pub(super) kernel: &'static str,
    pub(super) splits: u32,
    pub(super) grid: u32,
    pub(super) block: u32,
}

#[derive(Debug)]
pub(super) struct Reduction {
    function: CudaFunction,
    shape: ReductionShape,
}

impl Reduction {
    /// Up to [`FLAT_MAX_SPLITS`] planes, one thread loads all planes of four outputs
    /// before adding them, which needs no barrier; more planes would not fit in its
    /// registers, so eight warps share them and meet in shared memory. The split
    /// counts `Site::config` picks on the development GPUs have fixed-count forms
    /// with no per-plane test
    pub(super) const fn shape(splits: u32) -> ReductionShape {
        const OUTPUTS: u32 = 96 * 256;
        let fixed = match splits {
            18 => Some("spk_segdense_reduce_e18"),
            34 => Some("spk_segdense_reduce_e34"),
            36 => Some("spk_segdense_reduce_e36"),
            _ => None,
        };
        if let Some(kernel) = fixed {
            return ReductionShape {
                kernel,
                splits,
                grid: OUTPUTS / 4 / 128,
                block: 128,
            };
        }

        if splits <= FLAT_MAX_SPLITS {
            // 64-thread blocks spread the loads over more SMs than the 48 blocks of
            // 128 threads would
            return ReductionShape {
                kernel: "spk_segdense_reduce_embed_flat",
                splits,
                grid: OUTPUTS / 4 / 64,
                block: 64,
            };
        }

        ReductionShape {
            kernel: "spk_segdense_reduce_embed",
            splits,
            grid: OUTPUTS / 128,
            block: 256,
        }
    }
}

/// Segmentation `linear0`, `linear1` and classifier, and embedding `seg_1`, with
/// their bias, LeakyReLU or log-softmax epilogues fused
#[derive(Debug)]
pub(crate) struct DenseOxide(Producer);

impl DenseCandidate for DenseOxide {
    type Pin = SegdensePin;
    const COVERAGE: Coverage = Coverage(&[CoverageEntry {
        layers: &["linear0", "linear1", "linear2", "resnet.seg_1"],
        batches: Batches::Only(&BATCHES),
        maths: Maths::All,
    }]);
    const SPECIAL_VALUES: SpecialValues = SPECIAL_VALUES;

    fn implemented_pin(
        spec: DenseSpec,
        tier: PtxTier,
        device: &DeviceAttributes,
    ) -> Result<SegdensePin, PlanError> {
        SegdensePin::select(spec.site().into(), spec.batch(), spec.math(), tier, device)
    }

    fn plan(
        runtime: &CudaRuntime,
        kernels: &LoadedKernels,
        spec: DenseSpec,
        weight: &CudaSlice<f32>,
        bias: &CudaSlice<f32>,
        pin: SegdensePin,
    ) -> Result<Self, PlanError> {
        pin.check(
            spec.site().into(),
            spec.batch(),
            spec.math(),
            kernels.tier(),
        )?;
        Producer::plan(runtime, kernels, weight, Some(bias), pin).map(Self)
    }

    fn enqueue(
        &self,
        x: &CudaView<'_, f32>,
        output: &mut CudaViewMut<'_, f32>,
        phases: &Phases,
        runtime: &CudaRuntime,
    ) -> Result<(), CudaError> {
        self.0.enqueue(x, output, phases, runtime.stream())
    }
}

/// The two raw segmentation convolutions; the shared pool consumer adds their bias
#[derive(Debug)]
pub(crate) struct SegConvOxide(Producer);

impl SegConvCandidate for SegConvOxide {
    type Pin = SegdensePin;
    const COVERAGE: Coverage = Coverage(&[CoverageEntry {
        layers: &["sincnet.conv1", "sincnet.conv2"],
        batches: Batches::Only(&BATCHES),
        maths: Maths::All,
    }]);
    const SPECIAL_VALUES: SpecialValues = SPECIAL_VALUES;

    fn implemented_pin(
        spec: SegConvSpec,
        tier: PtxTier,
        device: &DeviceAttributes,
    ) -> Result<SegdensePin, PlanError> {
        SegdensePin::select(spec.site().into(), spec.batch(), spec.math(), tier, device)
    }

    fn plan(
        runtime: &CudaRuntime,
        kernels: &LoadedKernels,
        spec: SegConvSpec,
        weight: &CudaSlice<f32>,
        pin: SegdensePin,
    ) -> Result<Self, PlanError> {
        pin.check(
            spec.site().into(),
            spec.batch(),
            spec.math(),
            kernels.tier(),
        )?;
        Producer::plan(runtime, kernels, weight, None, pin).map(Self)
    }

    fn enqueue(
        &self,
        x: &CudaView<'_, f32>,
        output: &mut CudaViewMut<'_, f32>,
        phases: &Phases,
        runtime: &CudaRuntime,
    ) -> Result<(), CudaError> {
        self.0.enqueue(x, output, phases, runtime.stream())
    }
}

/// One registry owner for the complete segmentation and embedding dense area
pub(crate) struct Area;
impl super::DriverCandidate for Area {
    const AREA: KernelModule = KernelModule::Segdense;
    fn driver_coverage(
        _tier: PtxTier,
        _device: &DeviceAttributes,
        _fp16: super::Fp16Policy,
    ) -> Coverage {
        Coverage(&[CoverageEntry {
            layers: &[
                "sincnet.conv1",
                "sincnet.conv2",
                "linear0",
                "linear1",
                "linear2",
                "resnet.seg_1",
            ],
            batches: Batches::Only(&BATCHES),
            maths: Maths::All,
        }])
    }
    fn broad_evidence() -> Option<&'static crate::inference::cuda::implementation::BroadEvidence> {
        use crate::inference::cuda::implementation::{ArchitectureSpeed, BroadEvidence};
        const EVIDENCE: BroadEvidence = BroadEvidence::with_limits(
            &[
                ArchitectureSpeed {
                    capability: ComputeCapability::new(8, 9),
                    minimum_speedup_milli: 1060,
                },
                ArchitectureSpeed {
                    capability: ComputeCapability::new(12, 0),
                    minimum_speedup_milli: 1060,
                },
            ],
            "segdense sm80: fused conv/pool producers and packed dense epilogues; do-segdense dev reports on Ada and Blackwell, every measured case >=1.06x; A100 24/24 dev",
            ComputeCapability::new(8, 0),
            PtxTier::Sm80,
        );
        Some(&EVIDENCE)
    }
    fn tuning_fp32_pin(
        boundary: crate::inference::cuda::implementation::BoundaryId,
        batch: usize,
        math: CudaMath,
        device: &DeviceAttributes,
        tier: PtxTier,
    ) -> Result<Option<super::ConfigPin>, PlanError> {
        SegdensePin::select_fp32(Site::from_boundary(boundary)?, batch, math, tier, device)
            .map(super::ConfigPin::Segdense)
            .map(Some)
    }

    fn tuning_tf32_pin(
        boundary: crate::inference::cuda::implementation::BoundaryId,
        batch: usize,
        math: CudaMath,
        device: &DeviceAttributes,
        tier: PtxTier,
    ) -> Result<Option<super::ConfigPin>, PlanError> {
        let site = Site::from_boundary(boundary)?;
        Ok(
            SegdensePin::select_tuning_tf32(site, batch, math, tier, device)
                .map(super::ConfigPin::Segdense),
        )
    }

    fn driver_pin(
        boundary: crate::inference::cuda::implementation::BoundaryId,
        batch: usize,
        math: CudaMath,
        device: &DeviceAttributes,
        tier: PtxTier,
        _fp16: super::Fp16Policy,
    ) -> Result<super::ConfigPin, PlanError> {
        let site = Site::from_boundary(boundary)?;
        SegdensePin::select(site, batch, math, tier, device).map(super::ConfigPin::Segdense)
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
pub(crate) mod test_support;
