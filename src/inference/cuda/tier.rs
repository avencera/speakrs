//! speakrs's GPU target policy over cudakit's tiers: Cargo features, shipped tiers and
//! the environment override

use std::ffi::CStr;

pub use cudakit::{ComputeCapability, PtxTier};
use cudakit::{ExtraSymbols, LibraryPolicy, Profile, TierSet};

use super::CudaError;

/// Environment variable that forces a lower compiled-in PTX tier, for testing and as
/// an escape hatch when a higher tier misbehaves on some card
pub const PTX_TIER_ENV: &str = "SPEAKRS_CUDA_PTX_TIER";

/// cuDNN entry points the segmentation LSTM calls directly, beyond cudakit's wrappers
pub(super) const CUDNN_RNN_SYMBOLS: &[&CStr] = &[
    c"cudnnSetTensorNdDescriptor",
    c"cudnnGetTensorNdDescriptor",
    c"cudnnCreateRNNDescriptor",
    c"cudnnDestroyRNNDescriptor",
    c"cudnnSetRNNDescriptor_v8",
    c"cudnnCreateRNNDataDescriptor",
    c"cudnnDestroyRNNDataDescriptor",
    c"cudnnSetRNNDataDescriptor",
    c"cudnnGetRNNWeightSpaceSize",
    c"cudnnGetRNNWeightParams",
    c"cudnnGetRNNTempSpaceSizes",
    c"cudnnBuildRNNDynamic",
    c"cudnnRNNForward",
];

/// Tiers with shipped production kernels, independent of Cargo features
const SHIPPED: [PtxTier; 1] = [PtxTier::Sm75];

/// The build's CUDA policy as cudakit sees it
pub(super) static PROFILE: Profile = Profile {
    consumer: "speakrs",
    compiled: TierSet::EMPTY
        .with_if(PtxTier::Sm75, cfg!(feature = "cuda-sm75"))
        .with_if(PtxTier::Sm80, cfg!(feature = "cuda-sm80"))
        .with_if(PtxTier::Sm90, cfg!(feature = "cuda-sm90"))
        .with_if(PtxTier::Sm120, cfg!(feature = "cuda-sm120")),
    shipped: TierSet::of(&SHIPPED),
    tier_features: &[
        (PtxTier::Sm75, "cuda-sm75"),
        (PtxTier::Sm80, "cuda-sm80"),
        (PtxTier::Sm90, "cuda-sm90"),
        (PtxTier::Sm120, "cuda-sm120"),
    ],
    libraries: if super::driver_only() {
        LibraryPolicy::DriverOnly
    } else {
        LibraryPolicy::Allowed
    },
    extra_symbols: ExtraSymbols {
        cublas: &[],
        cudnn: CUDNN_RNN_SYMBOLS,
        nvrtc: &[],
    },
};

/// speakrs's Cargo feature and environment override for each PTX tier
pub(crate) trait TierPolicy: Sized {
    /// Cargo feature for this GPU target; `None` for a cudakit tier speakrs does not ship
    fn feature(self) -> Option<&'static str>;

    /// Reads `SPEAKRS_CUDA_PTX_TIER`; unset or empty means no override
    fn from_env() -> Result<Option<Self>, CudaError>;
}

impl TierPolicy for PtxTier {
    fn feature(self) -> Option<&'static str> {
        PROFILE.feature(self)
    }

    fn from_env() -> Result<Option<Self>, CudaError> {
        match std::env::var(PTX_TIER_ENV) {
            Ok(value) if !value.trim().is_empty() => Ok(Some(value.parse()?)),
            _ => Ok(None),
        }
    }
}

/// Build-policy queries that only tests make, since [`cudakit::Gpu::open`] selects the
/// tier in production
#[cfg(test)]
pub(crate) mod test_support {
    use super::{ComputeCapability, PROFILE, PtxTier};
    use crate::inference::cuda::CudaError;

    pub(crate) use super::TierPolicy;

    /// The build's tier policy, as the runtime applies it
    pub(crate) trait BuildTiers: Sized {
        /// Whether this GPU target is enabled in this build
        fn is_compiled_in(&self) -> bool;

        /// The enabled GPU targets, lowest first
        fn compiled_in() -> impl Iterator<Item = Self>;

        /// The highest tier a device may load; see [`cudakit::Profile::select`]
        fn select(
            capability: ComputeCapability,
            requested: Option<Self>,
        ) -> Result<Self, CudaError>;
    }

    impl BuildTiers for PtxTier {
        fn is_compiled_in(&self) -> bool {
            PROFILE.is_compiled_in(*self)
        }

        fn compiled_in() -> impl Iterator<Item = Self> {
            PROFILE.compiled.iter()
        }

        fn select(
            capability: ComputeCapability,
            requested: Option<Self>,
        ) -> Result<Self, CudaError> {
            Ok(PROFILE.select(capability, requested)?)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::{BuildTiers, TierPolicy};
    use super::{ComputeCapability, PtxTier};
    use crate::inference::cuda::CudaError;

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
                    device: actual, feature: Some("cuda-sm75"), ..
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
        for &tier in PtxTier::ALL {
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

    #[cfg(feature = "_cuda-libraries")]
    #[test]
    fn direct_rnn_calls_have_preflight_symbols() {
        let symbols: Vec<_> = cudakit::symbols::CUDNN
            .iter()
            .chain(super::CUDNN_RNN_SYMBOLS)
            .collect();
        for suffix in include_str!("segmentation/rnn.rs").split("dnn::").skip(1) {
            let name = suffix
                .split(|ch: char| !ch.is_ascii_alphanumeric() && ch != '_')
                .next()
                .unwrap();
            if name.starts_with("cudnn") && suffix[name.len()..].starts_with('(') {
                assert!(
                    symbols
                        .iter()
                        .any(|symbol| symbol.to_bytes() == name.as_bytes()),
                    "missing preflight for {name}"
                );
            }
        }
    }

    #[test]
    fn every_target_feature_maps_to_its_tier() {
        // later cudakit tiers have no speakrs feature until speakrs targets them
        for tier in [PtxTier::Sm75, PtxTier::Sm80, PtxTier::Sm90, PtxTier::Sm120] {
            assert_eq!(
                tier.feature(),
                Some(format!("cuda-{}", tier.name()).as_str())
            );
        }
    }
}
