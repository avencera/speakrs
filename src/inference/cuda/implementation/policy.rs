//! Measured whole-plan recipes and device-class defaults are distinct speed claims

use super::{BoundaryId, SpeedScope};
use crate::inference::cuda::candidate::{ConfigPin, ConvKernel, ConvPin, WideconvPin};
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

/// A measured recipe can keep Library, follow the driver rule, or fix a pin
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RecipeChoice {
    Library,
    DriverPin,
    FixedPin(ConfigPin),
}

impl Recipe {
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
                "fc55d67 boundary measurements: fixed driver pins at embedding b1/b4/b8/b16/b32, scalar C32 conv2 at 14 exact tuples; FP32 segmentation and TF32 embedding"
            }
            Self::Rtx5060Ti => {
                "fc55d67 boundary measurements: fixed driver pins at embedding b1/b4/b8/b16/b32, staged one-product C128/C256 and 8-window LSTM tiles; FP32 segmentation and TF32 embedding"
            }
            Self::TeslaT4 => {
                "fc55d67 T4 boundary measurements: 173 driver-pin choices and 55 Library choices across embedding b1/b4/b8/b16/b32; FP32 segmentation and TF32 embedding"
            }
            Self::A100Pcie => {
                "hybrid-profile cea1cbe: complete FP32 segmentation/TF32 embedding driver recipe; 10-file driver 648.06x versus hybrid 547.35x; identical RTTMs; short-file Library startup wins; not a per-layer speed claim"
            }
            Self::A100Sxm4 => {
                "do-a100 de89ae2: complete FP32 segmentation/TF32 embedding driver recipe; 10-file median 9.82 s versus Library 11.32 s; identical RTTMs; not a per-layer speed claim"
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

    /// Choose execution from the measured boundary, batch and arithmetic mode
    pub(crate) fn choice(self, boundary: BoundaryId, batch: usize, math: CudaMath) -> RecipeChoice {
        if self == Self::TeslaT4 {
            let library = match math {
                CudaMath::Tf32 => matches!(
                    (boundary.name(), batch),
                    ("resnet.conv1", 16 | 32)
                        | ("resnet.layer1.0.conv1" | "resnet.layer1.0.conv2", 1)
                        | (
                            "resnet.layer2.0.conv1"
                                | "resnet.layer3.0.conv1"
                                | "resnet.layer4.0.conv1",
                            1 | 4 | 8 | 16 | 32
                        )
                        | ("resnet.layer3.0.shortcut.0", 1)
                        | ("resnet.layer4.0.shortcut.0", 4)
                        | (
                            "resnet.layer3.0.conv2"
                                | "resnet.layer3.1.conv1"
                                | "resnet.layer3.1.conv2"
                                | "resnet.layer3.2.conv1"
                                | "resnet.layer3.2.conv2"
                                | "resnet.layer3.3.conv1"
                                | "resnet.layer3.3.conv2"
                                | "resnet.layer3.4.conv1"
                                | "resnet.layer3.4.conv2"
                                | "resnet.layer3.5.conv1"
                                | "resnet.layer3.5.conv2"
                                | "resnet.layer4.0.conv2"
                                | "resnet.layer4.1.conv1"
                                | "resnet.layer4.1.conv2"
                                | "resnet.layer4.2.conv1"
                                | "resnet.layer4.2.conv2",
                            16 | 32
                        )
                ),
                CudaMath::Fp32 => matches!((boundary.name(), batch), ("linear0" | "linear1", 1)),
            };
            if library {
                return RecipeChoice::Library;
            }
        }

        self.fixed_pin(boundary, batch, math)
            .map_or(RecipeChoice::DriverPin, RecipeChoice::FixedPin)
    }

    /// Fixed measured exceptions to the current device configuration rule
    pub(crate) fn fixed_pin(
        self,
        boundary: BoundaryId,
        batch: usize,
        math: CudaMath,
    ) -> Option<ConfigPin> {
        if self == Self::Rtx4060Ti && math == CudaMath::Tf32 {
            let scalar = match boundary.name() {
                "resnet.layer1.0.conv2" | "resnet.layer1.1.conv2" => {
                    matches!(batch, 1 | 4 | 8 | 16 | 32)
                }
                "resnet.layer1.2.conv2" => matches!(batch, 4 | 8 | 16 | 32),
                _ => false,
            };
            return scalar.then_some(ConfigPin::Conv(ConvPin::Kernel(ConvKernel::C32)));
        }
        if !matches!(self, Self::A100Pcie | Self::A100Sxm4)
            || !matches!(batch, 1 | 32)
            || math != CudaMath::Tf32
        {
            return None;
        }
        WideconvPin::measured_a100(boundary.name(), batch).map(ConfigPin::Wideconv)
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
}

impl DeviceDefault {
    pub(crate) fn select(
        area: KernelModule,
        batch: usize,
        math: CudaMath,
        device: &DeviceAttributes,
        tier: PtxTier,
    ) -> Option<Self> {
        (area == KernelModule::Resnet
            && matches!(batch, 1 | 32)
            && math == CudaMath::Tf32
            && device.capability() >= ComputeCapability::new(8, 0)
            && tier >= PtxTier::Sm80)
            .then_some(Self::AmpereTf32EarlyTrunk)
    }

    pub(crate) const fn summary(self) -> &'static str {
        "TF32 early C32/C64 trunk group: A100, RTX 4060 Ti 2.18x/1.90x and RTX 5060 Ti 1.58x/2.03x versus cuDNN b1/b32; Ampere+ class default; not a universal per-layer claim"
    }
}

#[cfg(test)]
mod tests;
