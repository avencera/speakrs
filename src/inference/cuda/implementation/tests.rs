//! Pin production to the triples accepted for integration, independent of declarations

use super::{PRODUCTION, Selected, Target, select};
use crate::inference::cuda::CudaMath;
use crate::inference::cuda::{ComputeCapability, PtxTier};

#[test]
fn production_selects_exactly_the_qualified_triples() {
    let c32 = [
        "resnet.layer1.0.conv1",
        "resnet.layer1.0.conv2",
        "resnet.layer1.1.conv1",
        "resnet.layer1.1.conv2",
        "resnet.layer1.2.conv1",
        "resnet.layer1.2.conv2",
        "resnet.layer2.0.conv1",
    ];
    let c64 = [
        "resnet.layer2.0.conv2",
        "resnet.layer2.1.conv1",
        "resnet.layer2.1.conv2",
        "resnet.layer2.2.conv1",
        "resnet.layer2.2.conv2",
        "resnet.layer2.3.conv1",
        "resnet.layer2.3.conv2",
    ];
    assert_eq!(
        crate::inference::cuda::candidate::QUALIFIED_BATCHES,
        [1, 32]
    );
    let declared = PRODUCTION
        .iter()
        .flat_map(|coverage| coverage.coverage.entries())
        .flat_map(|entry| entry.layers.iter().copied());
    let layers = c32.into_iter().chain(c64).chain(declared).chain([
        "lstm.stack",
        "sincnet.conv0.abs_pool",
        "resnet.layer3.0.conv1",
        "resnet.conv1",
        "unknown",
    ]);
    for layer in layers {
        for batch in (1..=66).chain([128, usize::MAX]) {
            for math in [CudaMath::Fp32, CudaMath::Tf32] {
                for tier in PtxTier::ALL {
                    for device in [
                        ComputeCapability::new(7, 5),
                        ComputeCapability::new(8, 0),
                        ComputeCapability::new(8, 6),
                        ComputeCapability::new(8, 9),
                        ComputeCapability::new(9, 0),
                        ComputeCapability::new(10, 0),
                        ComputeCapability::new(12, 0),
                        ComputeCapability::new(12, 1),
                        ComputeCapability::new(13, 0),
                    ] {
                        let expected = [1, 32].contains(&batch)
                            && tier == PtxTier::Sm75
                            && device == ComputeCapability::new(12, 0)
                            && (c32.contains(&layer)
                                || c64.contains(&layer) && (batch != 1 || math == CudaMath::Fp32)
                                || ["lstm.stack", "sincnet.conv0.abs_pool"].contains(&layer)
                                    && math == CudaMath::Fp32)
                            && !(layer == "resnet.layer2.0.conv1"
                                && batch == 1
                                && math == CudaMath::Fp32);
                        let selected = select(layer, batch, math, Target { tier, device }).unwrap();
                        assert_eq!(
                            matches!(selected, Selected::Oxide(_)),
                            expected,
                            "{layer} b{batch} {math:?} {tier} {device}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn regressed_resnet_tuple_uses_library_without_dropping_siblings() {
    let target = Target {
        tier: PtxTier::Sm75,
        device: ComputeCapability::new(12, 0),
    };
    let layer = "resnet.layer2.0.conv1";
    assert!(matches!(
        select(layer, 1, CudaMath::Fp32, target).unwrap(),
        Selected::Library
    ));
    for (batch, math) in [
        (1, CudaMath::Tf32),
        (32, CudaMath::Fp32),
        (32, CudaMath::Tf32),
    ] {
        let Selected::Oxide(token) = select(layer, batch, math, target).unwrap() else {
            panic!("unaffected sibling must keep its production token")
        };
        assert_eq!(token.boundary, layer);
        assert_eq!(token.batch, batch);
        assert_eq!(token.math, math);
        assert_eq!(token.target, target);
        assert_eq!(token.record, super::RESNET_RECORD);
        assert_eq!(token.selection, super::Selection::Production);
    }
}

#[test]
fn invalid_requests_cannot_make_tokens() {
    let target = Target {
        tier: PtxTier::Sm75,
        device: ComputeCapability::new(12, 0),
    };
    assert!(select("", 1, CudaMath::Fp32, target).is_err());
    assert!(select("lstm.stack", 0, CudaMath::Fp32, target).is_err());
}

#[test]
fn stage_tail_base_coverage_is_pinned_not_candidate_declared() {
    use crate::inference::cuda::KernelModule;
    let target = Target {
        tier: PtxTier::Sm75,
        device: ComputeCapability::new(12, 0),
    };
    for area in [KernelModule::Resnet, KernelModule::Sincnet] {
        let coverage = super::production_coverage(area, target);
        assert!(!coverage.entries().is_empty());
        for entry in coverage.entries() {
            for layer in entry.layers {
                for batch in [1, 32] {
                    assert_eq!(
                        coverage.covers(layer, batch, CudaMath::Fp32),
                        matches!(
                            select(layer, batch, CudaMath::Fp32, target).unwrap(),
                            Selected::Oxide(_)
                        )
                    );
                }
                for batch in [7, 33, 64] {
                    assert!(!coverage.covers(layer, batch, CudaMath::Fp32));
                }
            }
        }
        assert_eq!(
            super::production_coverage(
                area,
                Target {
                    device: ComputeCapability::new(8, 0),
                    ..target
                }
            ),
            crate::inference::cuda::candidate::Coverage::NONE
        );
        assert_eq!(
            super::production_coverage(
                area,
                Target {
                    tier: PtxTier::Sm80,
                    ..target
                }
            ),
            crate::inference::cuda::candidate::Coverage::NONE
        );
    }
}

#[test]
fn direct_pinned_requests_use_the_production_token() {
    use super::super::KernelModule;
    use super::{Choice, Selection};
    let target = Target {
        tier: PtxTier::Sm75,
        device: ComputeCapability::new(12, 0),
    };
    for choice in [Choice::StageTail, Choice::StageTailControl] {
        for (area, layer) in [
            (KernelModule::Resnet, "resnet.layer1.0.conv1"),
            (KernelModule::Sincnet, "sincnet.conv0.abs_pool"),
        ] {
            for batch in [1, 32] {
                let Selected::Oxide(token) = super::qualification_selection(
                    choice,
                    area,
                    layer,
                    batch,
                    CudaMath::Fp32,
                    target,
                )
                .unwrap() else {
                    panic!("pinned request must not silently select Library")
                };
                let Selected::Oxide(expected) =
                    select(layer, batch, CudaMath::Fp32, target).unwrap()
                else {
                    panic!("production token")
                };
                assert_eq!(token.record, expected.record);
                assert_eq!(token.der, expected.der);
                assert_eq!(token.boundary, layer);
                assert_eq!(token.batch, batch);
                assert_eq!(token.math, CudaMath::Fp32);
                assert_eq!(token.selection, Selection::Production);
                assert!(matches!(
                    super::qualification_selection(
                        choice,
                        area,
                        layer,
                        batch,
                        CudaMath::Tf32,
                        target,
                    )
                    .unwrap(),
                    Selected::Library
                ));
            }
            for batch in [7, 33, 64] {
                assert!(matches!(
                    super::qualification_selection(
                        choice,
                        area,
                        layer,
                        batch,
                        CudaMath::Fp32,
                        target,
                    )
                    .unwrap(),
                    Selected::Library
                ));
            }
        }
    }
}

#[test]
fn records_and_integrated_evidence_are_pinned() {
    let pins = [
        "8f8fa3e3c158771e354aad83f4e42fca6fac998aa192b39a966067a4b0035758",
        "3badc1aec939b0e8f7312786d695bec6445de1dacb1f85e44124bf3ac20356f8",
        "a4d1a2692b78a814cd2da16f084801c3641bfd11c6d095d610f3a8182ad6f675",
    ];
    for (entry, pin) in PRODUCTION.iter().zip(pins) {
        assert_eq!(entry.record, pin);
        assert_eq!(
            entry.der,
            "8066268031afba058d93e305206d5b8225e40c6ab2ebb1052874d607e055646f"
        );
    }
}

/// Export evaluated const entries, so Python never guesses what Rust expressions mean
#[test]
fn export_production_table() {
    let Ok(path) = std::env::var("SPEAKRS_QUALIFY_TABLE_OUTPUT") else {
        return;
    };
    let entries: Vec<_> = PRODUCTION
        .iter()
        .map(|entry| {
            let candidate = match entry.area {
                super::KernelModule::Resnet => {
                    <super::ConvOxide as super::ConvCandidate>::coverage(entry.tier)
                }
                super::KernelModule::Lstm => {
                    <super::LstmOxide as super::LstmCandidate>::coverage(entry.tier)
                }
                super::KernelModule::Sincnet => {
                    <super::SincOxide as super::SincCandidate>::coverage(entry.tier)
                }
                _ => panic!("production area has no candidate coverage export"),
            };
            serde_json::json!({
                "area": entry.area.name(),
                "candidate_coverage": super::super::test_support::qualify::coverage_json(candidate),
                "coverage": super::super::test_support::qualify::coverage_json(entry.coverage),
                "tier": entry.tier.to_string(),
                "devices": entry.devices.iter().map(ToString::to_string).collect::<Vec<_>>(),
                "record": entry.record,
                "der": entry.der,
            })
        })
        .collect();
    std::fs::write(
        path,
        serde_json::to_vec_pretty(&entries).expect("finite table"),
    )
    .expect("table output");
}

#[test]
fn planning_refusals_preserve_selection_and_policy() {
    use super::super::{CudaError, KernelModule};
    use super::{PlanError, Selection};
    let Selected::Oxide(mut token) = select(
        "lstm.stack",
        1,
        CudaMath::Fp32,
        Target {
            tier: PtxTier::Sm75,
            device: ComputeCapability::new(12, 0),
        },
    )
    .unwrap() else {
        panic!("production token")
    };
    for selection in [Selection::Production, Selection::Explicit] {
        token.selection = selection;
        for driver_only in [false, true] {
            assert_eq!(
                token
                    .finish(KernelModule::Lstm, driver_only, Ok(42))
                    .unwrap(),
                Some(42)
            );
            let refusal = || PlanError::DeviceUnsupported {
                reason: "grid exceeds device capacity".to_owned(),
            };
            let result = token.finish::<()>(KernelModule::Lstm, driver_only, Err(refusal()));
            if selection == Selection::Production && !driver_only {
                assert!(result.unwrap().is_none());
            } else {
                assert!(matches!(result, Err(CudaError::CandidateDeviceUnsupported {
                    area: "lstm", boundary, batch: 1, math: CudaMath::Fp32,
                    tier: PtxTier::Sm75, device, reason,
                }) if boundary == "lstm.stack" && device == ComputeCapability::new(12, 0)
                    && reason == "grid exceeds device capacity"));
            }
            assert!(matches!(
                token.finish::<()>(
                    KernelModule::Lstm,
                    driver_only,
                    Err(PlanError::Cuda(CudaError::Unsupported {
                        context: "invalid shape",
                        reason: "not a device limit".to_owned(),
                    }))
                ),
                Err(CudaError::Unsupported {
                    context: "invalid shape",
                    ..
                })
            ));
        }
    }
}
