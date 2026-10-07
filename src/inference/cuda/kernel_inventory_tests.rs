//! Host-only evidence that every selectable host kernel is shipped in its PTX area

use std::collections::BTreeSet;

use super::{ComputeCapability, KernelModule, PtxTier, candidate, embedding, fbank, segmentation};

fn entries(ptx: &str) -> BTreeSet<&str> {
    ptx.lines()
        .filter_map(|line| line.trim().strip_prefix(".visible .entry "))
        .map(|entry| entry.split(['(', ' ', '\t']).next().unwrap())
        .collect()
}

fn missing<'a>(ptx: &str, required: &'a [&'a str]) -> Vec<&'a str> {
    let entries = entries(ptx);
    required
        .iter()
        .copied()
        .filter(|name| !entries.contains(name))
        .collect()
}

fn plans() -> Vec<(KernelModule, Vec<&'static str>)> {
    let plans = vec![
        (KernelModule::Fbank, fbank::REQUIRED_KERNELS.to_vec()),
        (
            KernelModule::Embedding,
            embedding::REQUIRED_KERNELS.to_vec(),
        ),
        (
            KernelModule::Segmentation,
            segmentation::REQUIRED_KERNELS.to_vec(),
        ),
        (KernelModule::Resnet, candidate::conv_kernel_inventory()),
        (KernelModule::Lstm, candidate::LSTM_KERNELS.to_vec()),
        (KernelModule::Sincnet, candidate::SINC_KERNELS.to_vec()),
        (KernelModule::LstmProj, candidate::LSTMPROJ_KERNELS.to_vec()),
    ];
    #[cfg(all(feature = "cuda", not(feature = "cuda-driver-only")))]
    let plans = {
        let mut plans = plans;
        plans.push((KernelModule::Probe, super::probe::REQUIRED_KERNELS.to_vec()));
        plans
    };
    plans
}

#[test]
fn every_host_configuration_has_entries_in_every_paired_ptx_tier() {
    let mut pairings = 0;
    for (area, required) in plans() {
        assert!(
            area.variants().iter().next().is_some(),
            "{} has no embedded PTX",
            area.name()
        );
        assert!(
            !required.is_empty(),
            "{} has no host requirements",
            area.name()
        );
        for device in [
            ComputeCapability::new(7, 5),
            ComputeCapability::new(8, 0),
            ComputeCapability::new(8, 6),
            ComputeCapability::new(8, 9),
            ComputeCapability::new(9, 0),
            ComputeCapability::new(12, 0),
            ComputeCapability::new(13, 0),
        ] {
            for requested in std::iter::once(None).chain(PtxTier::ALL.map(Some)) {
                let Ok(limit) = PtxTier::select(device, requested) else {
                    continue;
                };
                // a lower shipped variant can pair with this target through fallback
                for (tier, ptx) in area.variants().iter() {
                    if tier > limit || tier.min_capability() > device {
                        continue;
                    }
                    pairings += 1;
                    let absent = missing(ptx, &required);
                    assert!(
                        absent.is_empty(),
                        "{} {tier} on {device}: missing {absent:?}",
                        area.name()
                    );
                }
            }
        }
    }

    assert!(pairings > 0, "enable a CUDA target feature for this test");
}

#[test]
fn missing_entry_is_detected_in_an_in_memory_ptx_fixture() {
    for (area, required) in plans() {
        for (tier, ptx) in area.variants().iter() {
            assert!(missing(ptx, &required).is_empty(), "{} {tier}", area.name());
            for name in &required {
                let fixture = ptx.replace(
                    &format!(".visible .entry {name}("),
                    &format!(".visible .entry removed_{name}("),
                );
                assert_ne!(fixture, ptx, "entry parser fixture did not remove {name}");
                assert!(
                    missing(&fixture, &required).contains(name),
                    "missing {name} was not detected"
                );
            }
        }
    }
}
