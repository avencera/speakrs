//! Pin production to the triples accepted for integration, independent of declarations

use super::{PRODUCTION, Selected, Target, select};
use crate::inference::cuda::kernels::{ArtifactHash, LoadedArtifact};
use crate::inference::cuda::{ComputeCapability, CudaMath, KernelModule, PtxTier};

#[test]
fn library_and_mutant_requests_do_not_load_candidate_artifacts() {
    use super::{AreaTarget, Choice, PlanRequest};
    use crate::inference::cuda::test_support::Mutant;
    let location = AreaTarget {
        tier: PtxTier::Sm75,
        device: ComputeCapability::new(12, 0),
    };
    for (area, boundary) in [
        (KernelModule::Resnet, "resnet.layer1.0.conv1"),
        (KernelModule::Lstm, "lstm.stack"),
        (KernelModule::Sincnet, "sincnet.conv0.abs_pool"),
    ] {
        for choice in [Choice::Library, Choice::Mutant(Mutant::Precision)] {
            let selected = PlanRequest::Qualification(choice)
                .resolve(area, boundary, 1, CudaMath::Fp32, location, |_| {
                    panic!("Library-backed request must not load a candidate module")
                })
                .unwrap();
            match (choice, selected) {
                (Choice::Library, Selected::Library)
                | (Choice::Mutant(Mutant::Precision), Selected::Mutant(Mutant::Precision)) => {}
                other => panic!("wrong owner: {other:?}"),
            }
        }
        assert!(
            PlanRequest::Qualification(Choice::Library)
                .resolve(area, "", 1, CudaMath::Fp32, location, |_| panic!(
                    "invalid tuple"
                ))
                .is_err()
        );
        assert!(
            PlanRequest::Qualification(Choice::Library)
                .resolve(area, boundary, 0, CudaMath::Fp32, location, |_| panic!(
                    "invalid tuple"
                ))
                .is_err()
        );
    }
}

#[test]
fn uncovered_candidate_requests_do_not_load_artifacts() {
    use super::{AreaTarget, Choice, PlanRequest, Selection};
    let location = AreaTarget {
        tier: PtxTier::Sm75,
        device: ComputeCapability::new(12, 0),
    };
    for choice in [
        Choice::Oxide(Selection::Explicit),
        Choice::StageTail,
        Choice::StageTailControl,
    ] {
        assert!(matches!(
            PlanRequest::Qualification(choice)
                .resolve(
                    KernelModule::Lstm,
                    "lstm.stack",
                    1,
                    CudaMath::Tf32,
                    location,
                    |_| panic!("uncovered request must not load an artifact")
                )
                .unwrap(),
            Selected::Library
        ));
    }
    for (batch, math, device) in [
        (7, CudaMath::Fp32, ComputeCapability::new(12, 0)),
        (1, CudaMath::Tf32, ComputeCapability::new(12, 0)),
        (1, CudaMath::Fp32, ComputeCapability::new(8, 9)),
    ] {
        assert!(matches!(
            PlanRequest::Production
                .resolve(
                    KernelModule::Lstm,
                    "lstm.stack",
                    batch,
                    math,
                    AreaTarget { device, ..location },
                    |_| panic!("uncovered production tuple must not load an artifact")
                )
                .unwrap(),
            Selected::Library
        ));
    }
}

#[test]
fn sinc_stage_tail_owner_resolves_coverage_before_loading() {
    use super::{AreaTarget, Choice, PlanRequest};
    let device = ComputeCapability::new(12, 0);
    let tier = PtxTier::Sm75;
    for choice in [Choice::StageTail, Choice::StageTailControl] {
        for (batch, math, device, expected_loads) in [
            (1, CudaMath::Fp32, device, 1),
            (32, CudaMath::Fp32, device, 1),
            (7, CudaMath::Fp32, device, 0),
            (1, CudaMath::Tf32, device, 0),
            (32, CudaMath::Tf32, device, 0),
            (1, CudaMath::Fp32, ComputeCapability::new(8, 9), 0),
        ] {
            let mut loads = 0;
            let target = Target {
                tier,
                device,
                artifact: legacy_artifact("sincnet"),
            };
            let selected = PlanRequest::Qualification(choice)
                .resolve(
                    KernelModule::Sincnet,
                    "sincnet.conv0.abs_pool",
                    batch,
                    math,
                    AreaTarget { tier, device },
                    |request| {
                        assert_eq!(request, super::ArtifactRequest::Pinned(target.artifact));
                        loads += 1;
                        Ok(target)
                    },
                )
                .unwrap();
            assert_eq!(loads, expected_loads);
            match selected {
                Selected::Oxide(token) if expected_loads == 1 => {
                    assert_eq!(token.target, target);
                    assert_eq!(token.boundary, "sincnet.conv0.abs_pool");
                    assert_eq!(token.batch, batch);
                    assert_eq!(token.math, math);
                    assert_eq!(token.selection, super::Selection::Production);
                }
                Selected::Library if expected_loads == 0 => {}
                other => panic!("wrong fixture owner: {other:?}"),
            }
        }
    }
}

#[test]
fn covered_selection_loads_and_matches_the_actual_artifact() {
    use super::{AreaTarget, PlanRequest};
    use crate::inference::cuda::kernels::{ArtifactHash, LoadedArtifact};
    let location = AreaTarget {
        tier: PtxTier::Sm75,
        device: ComputeCapability::new(12, 0),
    };
    for artifact in [
        legacy_artifact("lstm"),
        LoadedArtifact::Cubin {
            arch: location.device,
            sha256: ArtifactHash::from_hex(
                "089961e8e8e2f97fae86947e1c5cd14ef422c4de6141b03ce729ad9949009f6e",
            ),
        },
    ] {
        let mut loads = 0;
        let selected = PlanRequest::Production
            .resolve(
                KernelModule::Lstm,
                "lstm.stack",
                1,
                CudaMath::Fp32,
                location,
                |request| {
                    assert_eq!(
                        request,
                        super::ArtifactRequest::Pinned(legacy_artifact("lstm"))
                    );
                    loads += 1;
                    Ok(Target {
                        tier: location.tier,
                        device: location.device,
                        artifact,
                    })
                },
            )
            .unwrap();
        assert_eq!(loads, 1);
        match artifact {
            LoadedArtifact::PtxJit { .. } => {
                let Selected::Oxide(token) = selected else {
                    panic!("legacy JIT pin")
                };
                assert_eq!(token.target.artifact, artifact);
            }
            LoadedArtifact::Cubin { .. } => assert!(matches!(selected, Selected::Library)),
        }
    }
}

#[test]
fn explicit_candidate_keeps_its_loaded_identity_and_library_errors_need_no_artifact() {
    use super::{AreaTarget, Choice, LibraryNeed, PlanRequest, Selection};
    use crate::inference::cuda::{CudaError, CudaLibrary};
    let location = AreaTarget {
        tier: PtxTier::Sm75,
        device: ComputeCapability::new(8, 9),
    };
    let artifact = LoadedArtifact::Cubin {
        arch: location.device,
        sha256: ArtifactHash::from_hex(
            "5b8e7918ea9d0fbfd556a3e08c21b841f70355ab7743779f483ec59289162c21",
        ),
    };
    let mut loads = 0;
    let selected = PlanRequest::Qualification(Choice::Oxide(Selection::Explicit))
        .resolve(
            KernelModule::Lstm,
            "lstm.stack",
            1,
            CudaMath::Fp32,
            location,
            |request| {
                assert_eq!(request, super::ArtifactRequest::EmbeddedExact);
                loads += 1;
                Ok(Target {
                    tier: location.tier,
                    device: location.device,
                    artifact,
                })
            },
        )
        .unwrap();
    assert_eq!(loads, 1);
    let Selected::Oxide(token) = selected else {
        panic!("covered explicit request")
    };
    assert_eq!(token.target.artifact, artifact);
    assert_eq!(token.selection, Selection::Explicit);
    let error = LibraryNeed::new(
        KernelModule::Lstm,
        "lstm.stack",
        1,
        CudaMath::Fp32,
        location,
        CudaLibrary::Cudnn,
    )
    .error();
    let CudaError::NotDriverOnly {
        area,
        boundary,
        batch,
        math,
        tier,
        device,
        library,
    } = error
    else {
        panic!("typed Library refusal")
    };
    assert_eq!(
        (area, boundary.as_str(), batch, math, tier, device, library),
        (
            "lstm",
            "lstm.stack",
            1,
            CudaMath::Fp32,
            location.tier,
            location.device,
            CudaLibrary::Cudnn
        )
    );
}

fn legacy_artifact(area: &str) -> LoadedArtifact {
    let ptx = match area {
        "lstm" => include_str!("../ptx/lstm.sm75.ptx"),
        "sincnet" => include_str!("../ptx/sincnet.sm75.ptx"),
        _ => include_str!("../ptx/resnet.sm75.ptx"),
    };
    LoadedArtifact::PtxJit {
        sha256: ArtifactHash::of(ptx.as_bytes()),
    }
}

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
        let artifact = legacy_artifact(layer.split('.').next().unwrap());
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
                        let selected = select(
                            layer,
                            batch,
                            math,
                            Target {
                                tier,
                                device,
                                artifact,
                            },
                        )
                        .unwrap();
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
        artifact: legacy_artifact("resnet"),
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
        artifact: legacy_artifact("resnet"),
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
        artifact: legacy_artifact("resnet"),
    };
    for area in [KernelModule::Resnet, KernelModule::Sincnet] {
        let target = Target {
            artifact: legacy_artifact(area.name()),
            ..target
        };
        let location = super::AreaTarget {
            tier: target.tier,
            device: target.device,
        };
        let super::LoadedArtifact::PtxJit { sha256 } = target.artifact else {
            panic!("legacy PTX fixture")
        };
        assert_eq!(
            super::legacy_fixture_coverage(area, location, super::ArtifactHash::of(b"stale PTX")),
            crate::inference::cuda::candidate::Coverage::NONE
        );
        let coverage = super::legacy_fixture_coverage(area, location, sha256);
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
            super::legacy_fixture_coverage(
                area,
                super::AreaTarget {
                    device: ComputeCapability::new(8, 0),
                    ..location
                },
                sha256
            ),
            crate::inference::cuda::candidate::Coverage::NONE
        );
        assert_eq!(
            super::legacy_fixture_coverage(
                area,
                super::AreaTarget {
                    tier: PtxTier::Sm80,
                    ..location
                },
                sha256
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
        artifact: legacy_artifact("resnet"),
    };
    for choice in [Choice::StageTail, Choice::StageTailControl] {
        for (area, layer) in [
            (KernelModule::Resnet, "resnet.layer1.0.conv1"),
            (KernelModule::Sincnet, "sincnet.conv0.abs_pool"),
        ] {
            let target = Target {
                artifact: legacy_artifact(area.name()),
                ..target
            };
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
                "artifact": super::super::test_support::artifact_json(entry.artifact),
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
            artifact: legacy_artifact("lstm"),
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

#[test]
fn cubin_qualification_never_matches_jit_or_another_binary() {
    let sha256 = ArtifactHash::of(b"qualified cubin");
    let device = ComputeCapability::new(12, 0);
    let artifact = LoadedArtifact::Cubin {
        arch: device,
        sha256,
    };
    let entry = super::Production {
        area: super::KernelModule::Resnet,
        coverage: PRODUCTION[0].coverage,
        tier: PtxTier::Sm75,
        devices: super::DEVICES,
        artifact,
        record: super::RESNET_RECORD,
        der: super::INTEGRATED_DER,
    };
    let target = Target {
        tier: PtxTier::Sm75,
        device,
        artifact,
    };
    assert!(entry.matches_target(target));
    for artifact in [
        LoadedArtifact::PtxJit { sha256 },
        LoadedArtifact::Cubin {
            arch: ComputeCapability::new(8, 9),
            sha256,
        },
        LoadedArtifact::Cubin {
            arch: device,
            sha256: ArtifactHash::of(b"other cubin"),
        },
    ] {
        assert!(!entry.matches_target(Target { artifact, ..target }));
    }
    for entry in PRODUCTION {
        let target = Target {
            artifact: entry.artifact,
            ..target
        };
        assert!(entry.matches_target(target));
        assert!(!entry.matches_target(Target { artifact, ..target }));
    }
}

#[test]
fn every_production_tuple_requests_its_qualified_ptx_jit() {
    use super::{AreaTarget, ArtifactRequest, PlanRequest};
    let device = ComputeCapability::new(12, 0);
    let mut selected_count = 0;
    for entry in PRODUCTION {
        let location = AreaTarget {
            tier: entry.tier,
            device,
        };
        assert_eq!(
            super::production_artifact(entry.area, location),
            Some(entry.artifact)
        );
        assert!(matches!(entry.artifact, LoadedArtifact::PtxJit { .. }));
        assert_eq!(entry.artifact, legacy_artifact(entry.area.name()));
        for layer in entry
            .coverage
            .entries()
            .iter()
            .flat_map(|row| row.layers.iter().copied())
            .collect::<std::collections::BTreeSet<_>>()
        {
            for batch in super::MODEL_BATCHES {
                for math in [CudaMath::Fp32, CudaMath::Tf32] {
                    if !entry.coverage.covers(layer, batch, math) {
                        continue;
                    }
                    let mut loads = 0;
                    let selected = PlanRequest::Production
                        .resolve(entry.area, layer, batch, math, location, |request| {
                            assert_eq!(request, ArtifactRequest::Pinned(entry.artifact));
                            loads += 1;
                            Ok(Target {
                                tier: location.tier,
                                device,
                                artifact: entry.artifact,
                            })
                        })
                        .unwrap();
                    let Selected::Oxide(token) = selected else {
                        panic!("qualified production tuple fell back: {layer} b{batch} {math:?}")
                    };
                    assert_eq!(token.target.artifact, entry.artifact);
                    assert_eq!(token.record, entry.record);
                    assert_eq!(loads, 1);
                    selected_count += 1;
                }
            }
        }
    }
    assert_eq!(selected_count, 52);
}

#[test]
fn hypothetical_cubin_owner_requests_only_its_pinned_binary() {
    use super::{AreaTarget, Production};
    use crate::inference::cuda::kernels::{EmbeddedCubin, load_artifact};
    let device = ComputeCapability::new(12, 0);
    let embedded = EmbeddedCubin {
        arch: device,
        bytes: b"qualified cubin",
    };
    let artifact = LoadedArtifact::Cubin {
        arch: device,
        sha256: ArtifactHash::of(embedded.bytes),
    };
    let entries = [Production {
        artifact,
        ..Production {
            area: KernelModule::Sincnet,
            coverage: PRODUCTION[2].coverage,
            tier: PtxTier::Sm75,
            devices: super::DEVICES,
            artifact,
            record: "qualified-cubin",
            der: "qualified-der",
        }
    }];
    super::validate_production_owners(&entries);
    let owner = super::production_owner(
        &entries,
        KernelModule::Sincnet,
        AreaTarget {
            tier: PtxTier::Sm75,
            device,
        },
    )
    .unwrap();
    let (loaded, actual) = load_artifact(
        owner.artifact,
        Some(embedded),
        ArtifactHash::of(b"source ptx"),
        |bytes| Ok::<_, ()>(bytes.to_vec()),
        || panic!("cubin record cannot request PTX"),
    )
    .unwrap();
    assert_eq!(loaded, embedded.bytes);
    assert_eq!(actual, artifact);
}

#[test]
fn conflicting_production_owners_are_rejected() {
    let duplicate = [
        super::Production {
            artifact: LoadedArtifact::Cubin {
                arch: ComputeCapability::new(12, 0),
                sha256: ArtifactHash::of(b"cubin"),
            },
            ..super::Production {
                area: PRODUCTION[0].area,
                coverage: PRODUCTION[0].coverage,
                tier: PRODUCTION[0].tier,
                devices: PRODUCTION[0].devices,
                artifact: PRODUCTION[0].artifact,
                record: PRODUCTION[0].record,
                der: PRODUCTION[0].der,
            }
        },
        super::Production {
            area: PRODUCTION[0].area,
            coverage: PRODUCTION[0].coverage,
            tier: PRODUCTION[0].tier,
            devices: PRODUCTION[0].devices,
            artifact: PRODUCTION[0].artifact,
            record: PRODUCTION[0].record,
            der: PRODUCTION[0].der,
        },
    ];
    assert!(std::panic::catch_unwind(|| super::validate_production_owners(&duplicate)).is_err());
}

#[test]
fn requested_artifact_rejection_uses_typed_policy_without_swapping() {
    use super::{AreaTarget, PlanRequest};
    use crate::inference::cuda::{
        CudaError,
        kernels::{ArtifactLoadError, EmbeddedCubin, load_artifact},
    };
    use cudarc::driver::{DriverError, sys::CUresult};
    let artifact = legacy_artifact("sincnet");
    let make_error = || CudaError::ArtifactLoad {
        module: "sincnet",
        artifact,
        source: DriverError(CUresult::CUDA_ERROR_INVALID_IMAGE),
    };
    let mut loads = 0;
    let selected = PlanRequest::Production
        .resolve(
            KernelModule::Sincnet,
            "sincnet.conv0.abs_pool",
            1,
            CudaMath::Fp32,
            AreaTarget {
                tier: PtxTier::Sm75,
                device: ComputeCapability::new(12, 0),
            },
            |request| {
                assert_eq!(request, super::ArtifactRequest::Pinned(artifact));
                loads += 1;
                let cubin = EmbeddedCubin {
                    arch: ComputeCapability::new(12, 0),
                    bytes: b"must not load",
                };
                let LoadedArtifact::PtxJit { sha256 } = artifact else {
                    unreachable!()
                };
                let error = load_artifact::<(), _>(
                    artifact,
                    Some(cubin),
                    sha256,
                    |_| panic!("no cubin substitution"),
                    || Err(make_error()),
                )
                .unwrap_err();
                let ArtifactLoadError::Driver(error) = error else {
                    panic!("must preserve driver rejection")
                };
                Err(error)
            },
        )
        .unwrap();
    assert!(matches!(selected, Selected::Library));
    assert_eq!(loads, 1);
    assert!(
        matches!(super::artifact_refusal(make_error(), false), Err(CudaError::ArtifactLoad { artifact: actual, .. }) if actual == artifact)
    );
}

#[test]
#[ignore = "short GPU proof; run under the shared GPU flock without diagnostic overrides"]
fn default_production_loads_record_pinned_jit() -> Result<(), crate::inference::cuda::CudaError> {
    use crate::inference::cuda::CudaRuntime;
    use std::fs::{OpenOptions, TryLockError};
    for name in [
        crate::inference::cuda::kernels::FORCE_PTX_JIT_ENV,
        crate::inference::cuda::tier::PTX_TIER_ENV,
        "SPEAKRS_QUALIFY_PHASE",
    ] {
        assert!(
            std::env::var_os(name).is_none(),
            "default production proof forbids {name}"
        );
    }
    let lock = OpenOptions::new()
        .read(true)
        .write(true)
        .open("/workspace/gpu-bench.lock")
        .expect("shared GPU lock exists");
    assert!(matches!(lock.try_lock(), Err(TryLockError::WouldBlock)));
    let runtime = CudaRuntime::new(0)?;
    assert_eq!(runtime.compute_capability(), ComputeCapability::new(12, 0));
    for (area, boundary) in [
        (KernelModule::Sincnet, "sincnet.conv0.abs_pool"),
        (KernelModule::Resnet, "resnet.layer1.0.conv1"),
    ] {
        let selected = super::plan_selection(&runtime, area, boundary, 1, CudaMath::Fp32, None)?;
        let Selected::Oxide(token) = selected else {
            panic!("default production must select the qualified candidate")
        };
        let expected = legacy_artifact(area.name());
        assert_eq!(token.target.artifact, expected);
        let loaded = runtime.load_kernels(area)?;
        assert_eq!(loaded.artifact(), expected);
        let (_, ptx) = runtime.area_ptx(area)?;
        for line in ptx
            .lines()
            .filter_map(|line| line.trim().strip_prefix(".visible .entry "))
        {
            loaded.function(line.split(['(', ' ', '\t']).next().unwrap())?;
        }
        println!(
            "production_artifact_proof {}",
            serde_json::json!({ "area": area.name(), "boundary": boundary, "batch": 1, "math": "fp32", "candidate": true, "tier": token.target.tier.to_string(), "device": token.target.device.to_string(), "artifact": crate::inference::cuda::test_support::artifact_json(loaded.artifact()), "embedded_ptx_sha256": loaded.ptx_sha256().to_string(), "record": token.record, "name": runtime.context().name()?, "sm_count": runtime.multiprocessor_count()?, "l2_bytes": runtime.l2_cache_size()? })
        );
    }
    for area in super::ALWAYS_ON {
        let (tier, ptx) = runtime.area_ptx(*area)?;
        let loaded = runtime.load_kernels(*area)?;
        let expected = LoadedArtifact::PtxJit {
            sha256: ArtifactHash::of(ptx.as_bytes()),
        };
        assert_eq!(loaded.artifact(), expected);
        for line in ptx
            .lines()
            .filter_map(|line| line.trim().strip_prefix(".visible .entry "))
        {
            loaded.function(line.split(['(', ' ', '\t']).next().unwrap())?;
        }
        println!(
            "always_on_artifact_proof {}",
            serde_json::json!({
                "area": area.name(), "tier": tier.to_string(),
                "device": runtime.compute_capability().to_string(),
                "artifact": crate::inference::cuda::test_support::artifact_json(loaded.artifact()),
                "embedded_ptx_sha256": loaded.ptx_sha256().to_string(),
            })
        );
    }
    let modules = crate::inference::cuda::test_support::loaded_modules();
    let modules = modules.as_array().expect("recorded module array");
    assert_eq!(modules.len(), 5);
    assert!(
        modules
            .iter()
            .all(|module| module["artifact"]["kind"] == "PtxJit")
    );
    runtime.synchronize()
}

#[test]
fn always_on_owners_request_embedded_ptx_jit_on_both_devices() {
    for device in [ComputeCapability::new(12, 0), ComputeCapability::new(8, 9)] {
        for area in super::ALWAYS_ON {
            let (tier, ptx) = area
                .variants()
                .resolve(*area, PtxTier::Sm75, device)
                .unwrap();
            let owner = super::artifact_owner(*area, super::AreaTarget { tier, device }).unwrap();
            assert!(matches!(
                owner,
                super::ProductionArtifactOwner::AlwaysOnPtxJit
            ));
            assert_eq!(
                owner.request(ptx),
                super::ArtifactRequest::Pinned(LoadedArtifact::PtxJit {
                    sha256: ArtifactHash::of(ptx.as_bytes())
                })
            );
        }
        let location = super::AreaTarget {
            tier: PtxTier::Sm75,
            device,
        };
        assert!(super::artifact_owner(KernelModule::Probe, location).is_none());
    }
}
