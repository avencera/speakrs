use std::fmt;
use std::str::FromStr;

use super::CudaError;

/// Environment variable that forces a lower compiled-in PTX tier, for testing and as
/// an escape hatch when a higher tier misbehaves on some card
pub const PTX_TIER_ENV: &str = "SPEAKRS_CUDA_PTX_TIER";

/// A device compute capability, ordered by major then minor version
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ComputeCapability {
    /// Major version, 7 for Turing and Volta
    pub major: u32,
    /// Minor version, 5 for Turing
    pub minor: u32,
}

impl ComputeCapability {
    /// A compute capability from its major and minor version
    pub const fn new(major: u32, minor: u32) -> Self {
        Self { major, minor }
    }
}

impl fmt::Display for ComputeCapability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}", self.major, self.minor)
    }
}

/// A GPU generation that committed PTX can target
///
/// The driver JIT-compiles PTX for its target and every newer GPU, so a tier runs on
/// any device at or above [`Self::min_capability`]. [`Self::Sm75`] is the baseline
/// that every kernel area ships today. GPU target features embed each area's best
/// shipped variant for the target; `cuda` enables all GPU targets
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum PtxTier {
    /// `sm_75`: Turing (T4, RTX 20xx) and newer
    Sm75,
    /// `sm_80`: Ampere (A100, RTX 30xx) and newer
    Sm80,
    /// `sm_90`: Hopper and newer
    Sm90,
    /// `sm_120`: consumer Blackwell (RTX 50xx) and newer
    Sm120,
}

impl PtxTier {
    /// The tier every area ships and every supported GPU runs
    pub const BASELINE: Self = Self::Sm75;

    /// Tiers with shipped production kernels, independent of Cargo features
    pub const SHIPPED: [Self; 1] = [Self::Sm75];

    /// Highest shipped tier that the device can execute
    pub fn native(device: ComputeCapability) -> Self {
        Self::SHIPPED
            .into_iter()
            .filter(|tier| tier.min_capability() <= device)
            .max()
            .unwrap_or(Self::BASELINE)
    }

    /// Cargo feature for this GPU target
    pub const fn feature(self) -> &'static str {
        match self {
            Self::Sm75 => "cuda-sm75",
            Self::Sm80 => "cuda-sm80",
            Self::Sm90 => "cuda-sm90",
            Self::Sm120 => "cuda-sm120",
        }
    }

    /// Every tier, lowest first
    pub const ALL: [Self; 4] = [Self::Sm75, Self::Sm80, Self::Sm90, Self::Sm120];

    /// The name used in PTX file names and in `SPEAKRS_CUDA_PTX_TIER`, such as `sm75`
    pub const fn name(self) -> &'static str {
        match self {
            Self::Sm75 => "sm75",
            Self::Sm80 => "sm80",
            Self::Sm90 => "sm90",
            Self::Sm120 => "sm120",
        }
    }

    /// The oldest compute capability that can run this tier's PTX
    pub const fn min_capability(self) -> ComputeCapability {
        match self {
            Self::Sm75 => ComputeCapability::new(7, 5),
            Self::Sm80 => ComputeCapability::new(8, 0),
            Self::Sm90 => ComputeCapability::new(9, 0),
            Self::Sm120 => ComputeCapability::new(12, 0),
        }
    }

    /// Whether this GPU target is enabled in this build
    pub const fn is_compiled_in(self) -> bool {
        match self {
            Self::Sm75 => cfg!(feature = "cuda-sm75"),
            Self::Sm80 => cfg!(feature = "cuda-sm80"),
            Self::Sm90 => cfg!(feature = "cuda-sm90"),
            Self::Sm120 => cfg!(feature = "cuda-sm120"),
        }
    }

    /// The enabled GPU targets, lowest first
    pub fn compiled_in() -> impl Iterator<Item = Self> {
        Self::ALL.into_iter().filter(|tier| tier.is_compiled_in())
    }

    /// Reads `SPEAKRS_CUDA_PTX_TIER`; unset or empty means no override
    pub fn from_env() -> Result<Option<Self>, CudaError> {
        match std::env::var(PTX_TIER_ENV) {
            Ok(value) if !value.trim().is_empty() => value.parse().map(Some),
            _ => Ok(None),
        }
    }

    /// The highest tier a device may load: the highest compiled-in tier it supports,
    /// or `requested` when given
    ///
    /// Fails when the device is below the baseline, or when `requested` is not
    /// compiled in or is above what the device supports
    pub fn select(
        capability: ComputeCapability,
        requested: Option<Self>,
    ) -> Result<Self, CudaError> {
        if capability < Self::BASELINE.min_capability() {
            return Err(CudaError::UnsupportedDevice {
                capability,
                baseline: Self::BASELINE,
            });
        }

        let Some(tier) = requested else {
            let supported = Self::compiled_in().filter(|tier| tier.min_capability() <= capability);
            return supported.max().ok_or_else(|| {
                let tier = Self::native(capability);
                CudaError::TierNotCompiledIn {
                    tier,
                    device: capability,
                    feature: tier.feature(),
                }
            });
        };

        if !tier.is_compiled_in() {
            return Err(CudaError::TierNotCompiledIn {
                tier,
                device: capability,
                feature: tier.feature(),
            });
        }

        if tier.min_capability() > capability {
            return Err(CudaError::PtxTierAboveDevice { tier, capability });
        }

        Ok(tier)
    }
}

impl fmt::Display for PtxTier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl FromStr for PtxTier {
    type Err = CudaError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let value = value.trim();
        Self::ALL
            .into_iter()
            .find(|tier| tier.name() == value)
            .ok_or_else(|| CudaError::InvalidPtxTier {
                value: value.to_string(),
            })
    }
}

#[cfg(test)]
mod tests {
    use super::{ComputeCapability, CudaError, PtxTier};

    #[test]
    fn automatic_target_requires_an_enabled_tier_the_device_supports() {
        for device in [
            ComputeCapability::new(7, 5),
            ComputeCapability::new(8, 0),
            ComputeCapability::new(8, 6),
            ComputeCapability::new(9, 0),
            ComputeCapability::new(12, 0),
        ] {
            let expected = PtxTier::compiled_in()
                .filter(|tier| tier.min_capability() <= device)
                .max();
            let selected = PtxTier::select(device, None);
            match expected {
                Some(tier) => assert_eq!(selected.unwrap(), tier),
                None => assert!(matches!(selected, Err(CudaError::TierNotCompiledIn {
                    device: actual, feature: "cuda-sm75", ..
                }) if actual == device)),
            }
        }
        assert!(matches!(
            PtxTier::select(ComputeCapability::new(7, 0), None),
            Err(CudaError::UnsupportedDevice { .. })
        ));
    }

    #[test]
    fn overrides_require_the_target_feature_and_device_capability() {
        let device = ComputeCapability::new(12, 0);
        for tier in PtxTier::ALL {
            let selected = PtxTier::select(device, Some(tier));
            if tier.is_compiled_in() {
                assert_eq!(selected.unwrap(), tier);
                if tier != PtxTier::Sm75 {
                    assert!(matches!(
                        PtxTier::select(ComputeCapability::new(7, 5), Some(tier)),
                        Err(CudaError::PtxTierAboveDevice { .. })
                    ));
                }
                continue;
            }
            assert!(matches!(selected, Err(CudaError::TierNotCompiledIn {
                tier: actual, feature, device: actual_device
            }) if actual == tier && feature == tier.feature() && actual_device == device));
        }
    }
}
