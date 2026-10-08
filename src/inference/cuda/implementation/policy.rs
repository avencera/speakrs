//! Measured whole-plan recipes and device-class defaults are distinct speed claims

use super::{BoundaryId, SpeedScope};
use crate::inference::cuda::candidate::{ConfigPin, Fp16Policy, WideconvPin};
use crate::inference::cuda::device::DeviceAttributes;
use crate::inference::cuda::{ComputeCapability, CudaMath, KernelModule, PtxTier};

/// The source of a boundary choice, in decreasing policy precedence
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Source {
    TuneFile,
    Recipe,
    Default,
    Library,
}

impl Source {
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::TuneFile => "tune file",
            Self::Recipe => "recipe",
            Self::Default => "default",
            Self::Library => "library",
        }
    }
}

/// Pipeline precision is part of a whole-plan recipe, not a per-layer speed claim
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) enum RecipeMode {
    #[default]
    Disabled,
    Fp32SegmentationTf32Embedding,
}

impl RecipeMode {
    pub(crate) fn new(segmentation: CudaMath, embedding: CudaMath) -> Self {
        match (segmentation, embedding) {
            (CudaMath::Fp32, CudaMath::Tf32) => Self::Fp32SegmentationTf32Embedding,
            _ => Self::Disabled,
        }
    }
}

/// A measured plan at an exact device point
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Recipe {
    Rtx4060TiSinc,
    Rtx4060Ti,
    Rtx5060Ti,
    TeslaT4,
    A100Pcie,
    A100Sxm4,
}

impl Recipe {
    /// Whether this device has a built-in recipe at the compiled tier
    pub(crate) fn measured_device(device: &DeviceAttributes, tier: PtxTier) -> bool {
        [
            Self::Rtx4060TiSinc,
            Self::Rtx4060Ti,
            Self::Rtx5060Ti,
            Self::TeslaT4,
            Self::A100Pcie,
            Self::A100Sxm4,
        ]
        .into_iter()
        .any(|recipe| recipe.scope().contains(device) && recipe.allows_tier_limit(tier))
    }

    pub(crate) const fn scope(self) -> SpeedScope {
        let (capability, multiprocessors, name) = match self {
            Self::Rtx4060TiSinc | Self::Rtx4060Ti => (
                ComputeCapability::new(8, 9),
                34,
                "NVIDIA GeForce RTX 4060 Ti",
            ),
            Self::Rtx5060Ti => (
                ComputeCapability::new(12, 0),
                36,
                "NVIDIA GeForce RTX 5060 Ti",
            ),
            Self::TeslaT4 => (ComputeCapability::new(7, 5), 40, "Tesla T4"),
            Self::A100Pcie => (ComputeCapability::new(8, 0), 108, "NVIDIA A100-PCIE-40GB"),
            Self::A100Sxm4 => (ComputeCapability::new(8, 0), 108, "NVIDIA A100-SXM4-40GB"),
        };
        SpeedScope::Point {
            capability,
            multiprocessors,
            device_name: name,
        }
    }

    pub(crate) const fn summary(self) -> &'static str {
        match self {
            Self::Rtx4060TiSinc => {
                "hybrid-profile cea1cbe: FP32 b1/b32 fused SincNet; 30-file driver 319.73x versus hybrid 313.91x; identical RTTMs"
            }
            Self::Rtx4060Ti => {
                "FP16 stride-1 C32/C64 at every batch and C128/C256 from batch 8; 30-file median 52.30 to 48.35 s with identical RTTMs; FP32 segmentation and TF32 embedding"
            }
            Self::Rtx5060Ti => {
                "fc55d67 boundary measurements: fixed driver pins at embedding b1/b4/b8/b16/b32, staged one-product C128/C256 and 8-window LSTM tiles; FP32 segmentation and TF32 embedding"
            }
            Self::TeslaT4 => {
                "FP16 stride-1 trunk at every batch; 216-file dev DER 7.0125 to 7.0118; 10-file driver 154.5x versus FP32 driver 92.2x; FP32 segmentation and TF32 embedding"
            }
            Self::A100Pcie | Self::A100Sxm4 => {
                "108-SM A100 class defaults; PCIe 40GB measurement: 10-file median 8.787 s versus retained pins 8.938 s, hard-file median 1.117 s versus 1.218 s; identical driver RTTMs and per-file DER; FP32 segmentation and TF32 embedding"
            }
        }
    }

    /// Recipes require the compiled tier used by their measured configurations
    pub(crate) fn allows_tier_limit(self, tier: PtxTier) -> bool {
        match self {
            Self::Rtx4060TiSinc => true,
            Self::TeslaT4 => cfg!(feature = "cuda-sm75"),
            _ => tier >= PtxTier::Sm80 && cfg!(feature = "cuda-sm80"),
        }
    }

    /// Fixed measured exceptions to the current device configuration rule
    pub(crate) fn fixed_pin(
        self,
        boundary: BoundaryId,
        batch: usize,
        math: CudaMath,
        fp16: Fp16Policy,
    ) -> Option<ConfigPin> {
        self.fp16_pin(boundary, batch, math)
            .filter(|_| fp16.allows())
    }

    /// Exact measured FP16 points, also used by builds without CUDA libraries
    pub(crate) fn fp16_device(device: &DeviceAttributes, tier: PtxTier) -> Option<Self> {
        [Self::TeslaT4, Self::Rtx4060Ti].into_iter().find(|recipe| {
            recipe.scope().contains(device) && (*recipe == Self::TeslaT4 || tier >= PtxTier::Sm80)
        })
    }

    /// FP16 changes only same-channel stride-1 trunk layers in TF32 mode
    pub(crate) fn fp16_pin(
        self,
        boundary: BoundaryId,
        batch: usize,
        math: CudaMath,
    ) -> Option<ConfigPin> {
        let pin = match self {
            Self::TeslaT4 => WideconvPin::measured_t4_fp16(boundary.name(), batch, math),
            Self::Rtx4060Ti => {
                // C128/C256 lose at small batches; the measured crossover starts at eight
                if (boundary.name().starts_with("resnet.layer3.")
                    || boundary.name().starts_with("resnet.layer4."))
                    && batch < 8
                {
                    return None;
                }

                WideconvPin::fp16_wide(boundary.name(), batch, math)
            }
            _ => None,
        }?;
        Some(ConfigPin::Wideconv(pin))
    }

    pub(crate) fn select(
        boundary: BoundaryId,
        batch: usize,
        math: CudaMath,
        device: &DeviceAttributes,
        mode: RecipeMode,
    ) -> Option<Self> {
        if !boundary.batches().contains(batch) {
            return None;
        }
        if boundary == BoundaryId::named("sincnet.conv0.abs_pool")
            && math == CudaMath::Fp32
            && Self::Rtx4060TiSinc.scope().contains(device)
        {
            return Some(Self::Rtx4060TiSinc);
        }
        if mode != RecipeMode::Fp32SegmentationTf32Embedding {
            return None;
        }
        let expected = match boundary.area() {
            KernelModule::Resnet | KernelModule::Embedding => CudaMath::Tf32,
            _ => CudaMath::Fp32,
        };
        if math != expected {
            return None;
        }
        [
            Self::Rtx4060Ti,
            Self::Rtx5060Ti,
            Self::TeslaT4,
            Self::A100Pcie,
            Self::A100Sxm4,
        ]
        .into_iter()
        .filter(|recipe| {
            !matches!(recipe, Self::A100Pcie | Self::A100Sxm4) || matches!(batch, 1 | 32)
        })
        .find(|recipe| recipe.scope().contains(device))
    }
}

/// An explicit user-approved class default for the early tensor-core trunk
///
/// The evidence covers trunk group totals on cc 8.0, 8.9 and 12.0, not every
/// individual layer or the unmeasured wide trunk on other devices
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DeviceDefault {
    AmpereTf32EarlyTrunk,
    TuringFp16Trunk,
}

impl DeviceDefault {
    pub(crate) fn select(
        area: KernelModule,
        boundary: BoundaryId,
        batch: usize,
        math: CudaMath,
        device: &DeviceAttributes,
        tier: PtxTier,
        fp16: Fp16Policy,
    ) -> Option<Self> {
        if area == KernelModule::Wideconv
            && fp16.allows()
            && math == CudaMath::Tf32
            && device.capability() == ComputeCapability::new(7, 5)
            && WideconvPin::fp16_wide(boundary.name(), batch, math).is_some()
        {
            return Some(Self::TuringFp16Trunk);
        }

        (area == KernelModule::Resnet
            && matches!(batch, 1 | 32)
            && math == CudaMath::Tf32
            && device.capability() >= ComputeCapability::new(8, 0)
            && tier >= PtxTier::Sm80)
            .then_some(Self::AmpereTf32EarlyTrunk)
    }

    pub(crate) const fn summary(self) -> &'static str {
        match self {
            Self::TuringFp16Trunk => {
                "cc 7.5 TF32-mode FP16 trunk default: T4 trunk 1.95x faster; 216-file dev DER 7.0125 to 7.0118; no TF32 hardware"
            }
            Self::AmpereTf32EarlyTrunk => {
                "TF32 early C32/C64 trunk group: A100, RTX 4060 Ti 2.18x/1.90x and RTX 5060 Ti 1.58x/2.03x versus cuDNN b1/b32; Ampere+ class default; not a universal per-layer claim"
            }
        }
    }
}

#[cfg(test)]
mod tests;
