//! Model-load advice for a device without measured kernel choices

use std::sync::Once;

use super::super::PtxTier;
use super::super::device::DeviceAttributes;
use super::super::implementation::policy::Recipe;

static HINT: Once = Once::new();

pub(super) fn log(device: &DeviceAttributes, tier: PtxTier, tune_loaded: bool) {
    if !eligible(device, tier, tune_loaded, super::super::driver_only()) {
        return;
    }

    HINT.call_once(|| {
        let name = device.name();
        let capability = device.capability();
        let (major, minor) = (capability.major, capability.minor);
        tracing::info!("No measured CUDA recipe for {name} (cc {major}.{minor}); run `speakrs cuda tune` to tune kernel choices for this GPU");
    });
}

fn eligible(
    device: &DeviceAttributes,
    tier: PtxTier,
    tune_loaded: bool,
    driver_only: bool,
) -> bool {
    !driver_only && !tune_loaded && !Recipe::measured_device(device, tier)
}

#[cfg(test)]
mod tests {
    use super::eligible;
    use crate::inference::cuda::device::test_support::Builder;
    use crate::inference::cuda::{ComputeCapability, PtxTier};

    #[test]
    fn hint_is_only_for_an_untuned_unmeasured_hybrid_device() {
        let device = Builder::new(ComputeCapability::new(9, 0))
            .name("unknown GPU")
            .build();
        assert!(eligible(&device, PtxTier::Sm80, false, false));
        assert!(!eligible(&device, PtxTier::Sm80, true, false));
        assert!(!eligible(&device, PtxTier::Sm80, false, true));
        assert!(!eligible(&device, PtxTier::Sm80, true, true));
    }

    #[test]
    fn measured_device_points_stay_silent_at_their_recipe_tier() {
        for (cc, sms, name, tier) in [
            (
                ComputeCapability::new(8, 9),
                34,
                "NVIDIA GeForce RTX 4060 Ti",
                PtxTier::Sm80,
            ),
            (
                ComputeCapability::new(12, 0),
                36,
                "NVIDIA GeForce RTX 5060 Ti",
                PtxTier::Sm80,
            ),
            (ComputeCapability::new(7, 5), 40, "Tesla T4", PtxTier::Sm75),
            (
                ComputeCapability::new(8, 0),
                108,
                "NVIDIA A100-PCIE-40GB",
                PtxTier::Sm80,
            ),
            (
                ComputeCapability::new(8, 0),
                108,
                "NVIDIA A100-SXM4-40GB",
                PtxTier::Sm80,
            ),
        ] {
            let device = Builder::new(cc).multiprocessors(sms).name(name).build();
            if !tier.is_compiled_in() {
                continue;
            }
            assert!(!eligible(&device, tier, false, false), "{name}");
            let unknown = Builder::new(cc).multiprocessors(sms + 1).name(name).build();
            assert!(eligible(&unknown, tier, false, false), "{name}");
        }
    }
}
