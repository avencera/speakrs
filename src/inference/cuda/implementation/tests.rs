//! Production bindings, module requests and token policy, independent of declarations

use super::{
    Binding, BoundaryId, Choice, LibraryNeed, Modules, PRODUCTION, PlanRequest, RecordHash,
    Selected, Selection, SpeedEvidence, SpeedScope, SpeedStatus, TokenEvidence, TupleProof, select,
};
use crate::inference::cuda::candidate::{
    ConfigPin, ConvKernel, ConvPin, ConvShape, LstmPin, PlanError, SincPin,
};
use crate::inference::cuda::device::DeviceAttributes;
use crate::inference::cuda::device::test_support::Builder;
use crate::inference::cuda::kernels::{AreaPtx, ArtifactHash, LoadedArtifact, ModuleRequest};
use crate::inference::cuda::test_support::Mutant;
use crate::inference::cuda::{
    ComputeCapability, CudaError, CudaLibrary, CudaMath, GeometryError, KernelModule, PtxTier,
    WeightFault,
};

const BLACKWELL: ComputeCapability = ComputeCapability::new(12, 0);
const ADA: ComputeCapability = ComputeCapability::new(8, 9);

fn device(capability: ComputeCapability) -> DeviceAttributes {
    Builder::new(capability)
        .multiprocessors(36)
        .name("NVIDIA GeForce RTX 5060 Ti")
        .build()
}

fn legacy_ptx(area: KernelModule) -> &'static str {
    match area {
        KernelModule::Lstm => include_str!("../ptx/lstm.sm75.ptx"),
        KernelModule::Sincnet => include_str!("../ptx/sincnet.sm75.ptx"),
        _ => include_str!("../ptx/resnet.sm75.ptx"),
    }
}

fn legacy_module(area: KernelModule) -> ModuleRequest {
    ModuleRequest::new(
        area,
        PtxTier::Sm75,
        LoadedArtifact::PtxJit {
            sha256: ArtifactHash::of(legacy_ptx(area).as_bytes()),
        },
    )
}

/// How the scripted loader answers a request
#[derive(Debug, Clone, Copy)]
enum Loader {
    /// Accept exactly the request
    Exact,
    /// Accept, but report this artifact instead, as a diagnostic override would
    Other(LoadedArtifact),
    /// Refuse with the driver's rejection
    Refuses,
}

/// A host-only runtime with cached attributes and a scripted loader
struct Fixture {
    device: DeviceAttributes,
    limit: PtxTier,
    loader: Loader,
    loads: Vec<ModuleRequest>,
    explicit_artifact: Option<LoadedArtifact>,
}

impl Fixture {
    fn new(capability: ComputeCapability) -> Self {
        Self {
            device: device(capability),
            limit: PtxTier::Sm120,
            loader: Loader::Exact,
            loads: Vec::new(),
            explicit_artifact: None,
        }
    }

    fn loader(mut self, loader: Loader) -> Self {
        self.loader = loader;
        self
    }

    fn resolve(
        &mut self,
        request: PlanRequest,
        boundary: &str,
        batch: usize,
        math: CudaMath,
    ) -> Result<Selected, CudaError> {
        request.resolve(BoundaryId::named(boundary), batch, math, self)
    }
}

impl Modules for &mut Fixture {
    fn device(&self) -> &DeviceAttributes {
        &self.device
    }

    fn tier_limit(&self) -> PtxTier {
        self.limit
    }

    fn load(&mut self, request: ModuleRequest) -> Result<ModuleRequest, CudaError> {
        self.loads.push(request);
        match self.loader {
            Loader::Exact => Ok(request),
            Loader::Other(artifact) => {
                Ok(ModuleRequest::new(request.area(), request.tier(), artifact))
            }
            Loader::Refuses => Err(CudaError::ArtifactLoad {
                module: request.area().name(),
                artifact: request.artifact(),
                source: cudarc::driver::DriverError(
                    cudarc::driver::sys::CUresult::CUDA_ERROR_INVALID_IMAGE,
                ),
            }),
        }
    }

    fn embedded_exact(&self, area: KernelModule) -> Result<ModuleRequest, CudaError> {
        let request = legacy_module(area);
        Ok(self.explicit_artifact.map_or(request, |artifact| {
            ModuleRequest::new(area, request.tier(), artifact)
        }))
    }
}

fn token(selected: Selected) -> super::Qualified {
    let Selected::Oxide(token) = selected else {
        panic!("expected a candidate token, got {selected:?}")
    };
    token
}

#[test]
fn library_and_mutant_requests_do_not_load_candidate_artifacts() {
    for boundary in [
        "resnet.layer1.0.conv1",
        "lstm.stack",
        "sincnet.conv0.abs_pool",
    ] {
        for choice in [Choice::Library, Choice::Mutant(Mutant::Precision)] {
            let mut fixture = Fixture::new(BLACKWELL);
            let selected = fixture
                .resolve(
                    PlanRequest::Qualification(choice),
                    boundary,
                    1,
                    CudaMath::Fp32,
                )
                .unwrap();
            match (choice, selected) {
                (Choice::Library, Selected::Library)
                | (Choice::Mutant(Mutant::Precision), Selected::Mutant(Mutant::Precision)) => {}
                other => panic!("wrong owner: {other:?}"),
            }
            assert!(fixture.loads.is_empty());
        }
        let mut fixture = Fixture::new(BLACKWELL);
        assert!(
            fixture
                .resolve(
                    PlanRequest::Qualification(Choice::Library),
                    boundary,
                    0,
                    CudaMath::Fp32
                )
                .is_err()
        );
    }
}

#[test]
fn uncovered_requests_do_not_load_artifacts() {
    for choice in [
        Choice::Oxide(Selection::Explicit),
        Choice::StageTail,
        Choice::StageTailControl,
    ] {
        let mut fixture = Fixture::new(BLACKWELL);
        let selected = fixture
            .resolve(
                PlanRequest::Qualification(choice),
                "lstm.stack",
                1,
                CudaMath::Tf32,
            )
            .unwrap();
        assert!(matches!(selected, Selected::Library));
        assert!(fixture.loads.is_empty());
    }
    for (batch, math, capability) in [
        (7, CudaMath::Fp32, BLACKWELL),
        (1, CudaMath::Tf32, BLACKWELL),
        (1, CudaMath::Fp32, ADA),
    ] {
        let mut fixture = Fixture::new(capability);
        let selected = fixture
            .resolve(PlanRequest::Production, "lstm.stack", batch, math)
            .unwrap();
        assert!(matches!(selected, Selected::Library));
        assert!(fixture.loads.is_empty());
    }
    // a forced tier limit below the binding's tier leaves the tuple on Library
    let mut fixture = Fixture::new(BLACKWELL);
    let bound = super::bound_module(
        PRODUCTION,
        KernelModule::Lstm,
        &fixture.device,
        PtxTier::Sm75,
    );
    assert_eq!(bound, Some(legacy_module(KernelModule::Lstm)));
    fixture.limit = PtxTier::Sm75;
    assert!(matches!(
        fixture
            .resolve(PlanRequest::Production, "lstm.stack", 1, CudaMath::Fp32)
            .unwrap(),
        Selected::Oxide(_)
    ));
}

#[test]
fn stage_tail_resolves_pinned_coverage_before_loading() {
    for choice in [Choice::StageTail, Choice::StageTailControl] {
        for (batch, math, capability, expected_loads) in [
            (1, CudaMath::Fp32, BLACKWELL, 1),
            (32, CudaMath::Fp32, BLACKWELL, 1),
            (7, CudaMath::Fp32, BLACKWELL, 0),
            (1, CudaMath::Tf32, BLACKWELL, 0),
            (32, CudaMath::Tf32, BLACKWELL, 0),
            (1, CudaMath::Fp32, ADA, 0),
        ] {
            let mut fixture = Fixture::new(capability);
            let selected = fixture
                .resolve(
                    PlanRequest::Qualification(choice),
                    "sincnet.conv0.abs_pool",
                    batch,
                    math,
                )
                .unwrap();
            assert_eq!(fixture.loads.len(), expected_loads);
            if expected_loads == 0 {
                assert!(matches!(selected, Selected::Library));
                continue;
            }
            assert_eq!(fixture.loads, [legacy_module(KernelModule::Sincnet)]);
            let token = token(selected);
            assert_eq!(token.target.module, legacy_module(KernelModule::Sincnet));
            assert_eq!(token.boundary.name(), "sincnet.conv0.abs_pool");
            assert_eq!((token.batch, token.math), (batch, math));
            assert_eq!(token.selection, Selection::Production);
        }
    }
}

#[test]
fn production_token_requires_the_bound_module_identity() {
    // a diagnostic override that loads other bytes cannot obtain the production token
    let other = LoadedArtifact::Cubin {
        arch: BLACKWELL,
        sha256: ArtifactHash::from_hex(
            "089961e8e8e2f97fae86947e1c5cd14ef422c4de6141b03ce729ad9949009f6e",
        ),
    };
    for (loader, selected) in [(Loader::Exact, true), (Loader::Other(other), false)] {
        let mut fixture = Fixture::new(BLACKWELL).loader(loader);
        let result = fixture
            .resolve(PlanRequest::Production, "lstm.stack", 1, CudaMath::Fp32)
            .unwrap();
        assert_eq!(fixture.loads, [legacy_module(KernelModule::Lstm)]);
        assert_eq!(matches!(result, Selected::Oxide(_)), selected);
    }
    // production keeps today's Library fallback after a refusal; the explicit request errors
    let mut fixture = Fixture::new(BLACKWELL).loader(Loader::Refuses);
    assert!(matches!(
        fixture
            .resolve(
                PlanRequest::Production,
                "sincnet.conv0.abs_pool",
                1,
                CudaMath::Fp32
            )
            .unwrap(),
        Selected::Library
    ));
    assert!(matches!(
        fixture.resolve(
            PlanRequest::Qualification(Choice::Oxide(Selection::Explicit)),
            "sincnet.conv0.abs_pool",
            1,
            CudaMath::Fp32
        ),
        Err(CudaError::ArtifactLoad { .. })
    ));
    assert!(matches!(
        super::artifact_refusal(
            CudaError::ArtifactUnavailable {
                module: "sincnet",
                artifact: legacy_module(KernelModule::Sincnet).artifact(),
            },
            false
        ),
        Err(CudaError::ArtifactUnavailable { .. })
    ));
}

#[test]
fn explicit_plan_keeps_the_loaded_module_instead_of_resolving_production_again() {
    for capability in [ADA, BLACKWELL] {
        let mut fixture = Fixture::new(capability);
        if capability == BLACKWELL {
            fixture.explicit_artifact = Some(LoadedArtifact::Cubin {
                arch: capability,
                sha256: ArtifactHash::of(b"explicit exact-architecture cubin"),
            });
        }
        let token = token(
            fixture
                .resolve(
                    PlanRequest::Qualification(Choice::Oxide(Selection::Explicit)),
                    "lstm.stack",
                    1,
                    CudaMath::Fp32,
                )
                .unwrap(),
        );
        let request = token.plan_module();
        let loaded_for_plan = (&mut fixture).load(request).unwrap();
        assert_eq!(loaded_for_plan, request);
        assert_eq!(fixture.loads.last(), Some(&request));
        let production = super::production_module(
            KernelModule::Lstm,
            &fixture.device,
            fixture.limit,
            KernelModule::Lstm.variants(),
        )
        .unwrap();
        if capability == ADA {
            assert!(production.is_none());
        } else {
            assert_ne!(production, Some(request));
        }
    }
}

#[test]
fn explicit_candidate_plans_its_implemented_pin_and_library_needs_no_artifact() {
    let mut fixture = Fixture::new(ADA);
    let token = token(
        fixture
            .resolve(
                PlanRequest::Qualification(Choice::Oxide(Selection::Explicit)),
                "lstm.stack",
                1,
                CudaMath::Fp32,
            )
            .unwrap(),
    );
    assert_eq!(fixture.loads, [legacy_module(KernelModule::Lstm)]);
    assert_eq!(token.target.module, legacy_module(KernelModule::Lstm));
    assert_eq!(token.selection, Selection::Explicit);
    assert_eq!(token.pin, super::PlanPin::Implemented);
    assert_eq!(token.evidence, TokenEvidence::Qualification);

    let target = super::AreaTarget {
        tier: PtxTier::Sm75,
        device: ADA,
    };
    let error = LibraryNeed::new(
        BoundaryId::named("lstm.stack.input_proj"),
        1,
        CudaMath::Fp32,
        target,
        CudaLibrary::Cublas,
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
            "lstm.stack.input_proj",
            1,
            CudaMath::Fp32,
            PtxTier::Sm75,
            ADA,
            CudaLibrary::Cublas
        )
    );
}

#[test]
fn explicit_fbank_dft_plans_the_record_owned_area_and_production_stays_library() {
    for math in [CudaMath::Fp32, CudaMath::Tf32] {
        for batch in 1..=32 {
            let mut fixture = Fixture::new(ADA);
            let token = token(
                fixture
                    .resolve(
                        PlanRequest::Qualification(Choice::Oxide(Selection::Explicit)),
                        "fbank.dft",
                        batch,
                        math,
                    )
                    .unwrap(),
            );
            // the route is the record-owned area, never the always-on Fbank module
            assert_eq!(fixture.loads, [legacy_module(KernelModule::FbankDft)]);
            assert_eq!(token.target.module.area(), KernelModule::FbankDft);
            assert_eq!(token.pin, super::PlanPin::Implemented);
            assert_eq!(token.evidence, TokenEvidence::Qualification);

            // no binding exists, so production needs no artifact
            let mut fixture = Fixture::new(ADA);
            let selected = fixture
                .resolve(PlanRequest::Production, "fbank.dft", batch, math)
                .unwrap();
            assert!(matches!(selected, Selected::Library));
            assert!(fixture.loads.is_empty());
        }
        let mut fixture = Fixture::new(ADA);
        let selected = fixture
            .resolve(
                PlanRequest::Qualification(Choice::Oxide(Selection::Explicit)),
                "fbank.dft",
                33,
                math,
            )
            .unwrap();
        assert!(matches!(selected, Selected::Library));
        assert!(fixture.loads.is_empty());
    }
}

#[test]
fn production_tokens_require_tier_device_and_artifact_of_the_binding() {
    let device = device(BLACKWELL);
    let boundary = BoundaryId::named("resnet.layer1.0.conv1");
    let bound = legacy_module(KernelModule::Resnet);
    let token = token(select(boundary, 1, CudaMath::Fp32, &device, bound).unwrap());
    let TokenEvidence::Production { accuracy, speed } = token.evidence else {
        panic!("production evidence")
    };
    assert_eq!(accuracy, super::production::resnet::RECORD);
    assert_eq!(speed.record, super::production::resnet::RECORD);
    assert_eq!(speed.integrated, super::production::INTEGRATED_DER);
    assert_eq!(speed.scope, super::production::LEGACY_SCOPE);
    assert_eq!(
        token.pin,
        super::PlanPin::Pinned(ConfigPin::Conv(ConvPin::LegacyWaves(ConvShape::C32)))
    );
    let sha256 = ArtifactHash::of(legacy_ptx(KernelModule::Resnet).as_bytes());
    for loaded in [
        ModuleRequest::new(KernelModule::Resnet, PtxTier::Sm80, bound.artifact()),
        ModuleRequest::new(
            KernelModule::Resnet,
            PtxTier::Sm75,
            LoadedArtifact::Cubin {
                arch: BLACKWELL,
                sha256,
            },
        ),
        ModuleRequest::new(
            KernelModule::Resnet,
            PtxTier::Sm75,
            LoadedArtifact::PtxJit {
                sha256: ArtifactHash::of(b"other PTX"),
            },
        ),
        legacy_module(KernelModule::Lstm),
    ] {
        assert!(matches!(
            select(boundary, 1, CudaMath::Fp32, &device, loaded).unwrap(),
            Selected::Library
        ));
    }
    for other in [ComputeCapability::new(12, 1), ADA] {
        assert!(matches!(
            select(
                boundary,
                1,
                CudaMath::Fp32,
                &Builder::new(other).build(),
                bound
            )
            .unwrap(),
            Selected::Library
        ));
    }
    assert!(select(boundary, 0, CudaMath::Fp32, &device, bound).is_err());
}

#[test]
fn regressed_resnet_tuple_uses_library_without_dropping_siblings() {
    let device = device(BLACKWELL);
    let layer = BoundaryId::named("resnet.layer2.0.conv1");
    let bound = legacy_module(KernelModule::Resnet);
    assert!(matches!(
        select(layer, 1, CudaMath::Fp32, &device, bound).unwrap(),
        Selected::Library
    ));
    for (batch, math) in [
        (1, CudaMath::Tf32),
        (32, CudaMath::Fp32),
        (32, CudaMath::Tf32),
    ] {
        let token = token(select(layer, batch, math, &device, bound).unwrap());
        assert_eq!(
            (token.boundary, token.batch, token.math),
            (layer, batch, math)
        );
        assert_eq!(token.target.module, bound);
        assert_eq!(
            token.pin,
            super::PlanPin::Pinned(ConfigPin::Conv(ConvPin::LegacyWaves(ConvShape::C32Stride2)))
        );
    }
}

#[test]
fn stage_tail_coverage_is_pinned_not_candidate_declared() {
    let device = device(BLACKWELL);
    for area in [KernelModule::Resnet, KernelModule::Sincnet] {
        let embedded = AreaPtx::fixture(&[(PtxTier::Sm75, legacy_ptx(area))]);
        let stale = AreaPtx::fixture(&[(PtxTier::Sm75, "stale PTX")]);
        assert!(super::legacy_fixture_coverage(area, &device, stale).is_empty());
        let coverage = super::legacy_fixture_coverage(area, &device, embedded);
        assert!(!coverage.is_empty());
        let coverage = crate::inference::cuda::candidate::Coverage(coverage.leak());
        for layer in coverage.entries().iter().flat_map(|entry| entry.layers) {
            let boundary = BoundaryId::named(layer);
            for batch in [1, 32] {
                assert_eq!(
                    coverage.covers(layer, batch, CudaMath::Fp32),
                    matches!(
                        select(
                            boundary,
                            batch,
                            CudaMath::Fp32,
                            &device,
                            legacy_module(area)
                        )
                        .unwrap(),
                        Selected::Oxide(_)
                    )
                );
            }
            for batch in [7, 33, 64] {
                assert!(!coverage.covers(layer, batch, CudaMath::Fp32));
            }
        }
        assert!(
            super::legacy_fixture_coverage(area, &Builder::new(ADA).build(), embedded).is_empty()
        );
    }
}

#[test]
fn direct_pinned_requests_use_the_production_token() {
    let device = device(BLACKWELL);
    for choice in [Choice::StageTail, Choice::StageTailControl] {
        for (area, layer) in [
            (KernelModule::Resnet, "resnet.layer1.0.conv1"),
            (KernelModule::Sincnet, "sincnet.conv0.abs_pool"),
        ] {
            let boundary = BoundaryId::named(layer);
            let loaded = legacy_module(area);
            let direct = |batch, math| {
                super::qualification_selection(choice, boundary, batch, math, &device, loaded)
                    .unwrap()
            };
            for batch in [1, 32] {
                let actual = token(direct(batch, CudaMath::Fp32));
                let expected =
                    token(select(boundary, batch, CudaMath::Fp32, &device, loaded).unwrap());
                assert_eq!(actual.evidence, expected.evidence);
                assert_eq!(actual.pin, expected.pin);
                assert_eq!((actual.boundary, actual.batch), (boundary, batch));
                assert_eq!(actual.selection, Selection::Production);
                assert!(matches!(direct(batch, CudaMath::Tf32), Selected::Library));
            }
            for batch in [7, 33, 64] {
                assert!(matches!(direct(batch, CudaMath::Fp32), Selected::Library));
            }
        }
    }
}

#[test]
fn records_and_integrated_evidence_are_pinned() {
    let pins = [
        (
            KernelModule::Resnet,
            "8f8fa3e3c158771e354aad83f4e42fca6fac998aa192b39a966067a4b0035758",
        ),
        (
            KernelModule::Lstm,
            "3badc1aec939b0e8f7312786d695bec6445de1dacb1f85e44124bf3ac20356f8",
        ),
        (
            KernelModule::Sincnet,
            "a4d1a2692b78a814cd2da16f084801c3641bfd11c6d095d610f3a8182ad6f675",
        ),
    ];
    assert_eq!(PRODUCTION.len(), pins.len());
    for (binding, (area, pin)) in PRODUCTION.iter().zip(pins) {
        assert_eq!(binding.area(), area);
        assert_eq!(binding.module, legacy_module(area));
        // legacy approval stays capability-wide and is never copied into point evidence
        assert_eq!(binding.scope, super::production::LEGACY_SCOPE);
        for proof in binding.proofs {
            assert_eq!(proof.accuracy.to_string(), pin);
            let SpeedStatus::Measured(speed) = proof.speed else {
                panic!("legacy proofs are measured")
            };
            assert_eq!(speed.record.to_string(), pin);
            assert_eq!(
                speed.integrated.to_string(),
                "8066268031afba058d93e305206d5b8225e40c6ab2ebb1052874d607e055646f"
            );
            assert_eq!(speed.scope, binding.scope);
        }
    }
}

/// Export evaluated const entries, so Python never guesses what Rust expressions mean
///
fn measured_record_pairs(proofs: &[TupleProof]) -> Vec<(SpeedEvidence, RecordHash)> {
    let mut records = Vec::new();
    for proof in proofs {
        let SpeedStatus::Measured(speed) = proof.speed else {
            continue;
        };
        let pair = (speed, proof.accuracy);
        if !records.contains(&pair) {
            records.push(pair);
        }
    }
    records
}

#[test]
fn export_groups_distinct_accuracy_records_without_losing_speed_evidence() {
    let first = PRODUCTION[0].proofs[0];
    let second = TupleProof {
        accuracy: RecordHash::from_hex(
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        ),
        ..first
    };
    let records = measured_record_pairs(&[first, second]);
    let SpeedStatus::Measured(speed) = first.speed else {
        panic!("measured fixture")
    };
    assert_eq!(records, [(speed, first.accuracy), (speed, second.accuracy)]);
}

/// Each export entry is one binding's measured proofs with one accuracy record
/// and one speed record; each tuple also exports its complete configuration pin
#[test]
fn export_production_table() {
    let Ok(path) = std::env::var("SPEAKRS_QUALIFY_TABLE_OUTPUT") else {
        return;
    };
    let mut entries = Vec::new();
    for binding in PRODUCTION {
        let candidate = super::candidate_coverage(binding.area(), binding.module.tier());
        assert!(
            !candidate.entries().is_empty(),
            "production area has no candidate coverage export"
        );
        let records = measured_record_pairs(binding.proofs);
        for (speed, accuracy) in records {
            let tuples: Vec<_> = binding
                .proofs
                .iter()
                .filter(|proof| {
                    proof.speed == SpeedStatus::Measured(speed) && proof.accuracy == accuracy
                })
                .map(|proof| {
                    serde_json::json!({
                        "layers": [proof.boundary.name()],
                        "batches": [proof.batch],
                        "maths": [match proof.math {
                            CudaMath::Fp32 => "fp32",
                            CudaMath::Tf32 => "tf32",
                        }],
                    })
                })
                .collect();
            let configurations: Vec<_> = binding.proofs.iter()
                .filter(|proof| proof.speed == SpeedStatus::Measured(speed) && proof.accuracy == accuracy)
                .map(|proof| serde_json::json!({
                    "tuple": [proof.boundary.name(), proof.batch, match proof.math { CudaMath::Fp32 => "fp32", CudaMath::Tf32 => "tf32" }],
                    "pin": super::super::test_support::configuration::pin_json(proof.pin),
                })).collect();
            let scope = match speed.scope {
                SpeedScope::LegacyCapability { capability } => {
                    serde_json::json!({"kind": "LegacyCapability", "capability": capability.to_string()})
                }
                SpeedScope::Point {
                    capability,
                    multiprocessors,
                    device_name,
                } => serde_json::json!({
                    "kind": "Point",
                    "capability": capability.to_string(),
                    "sm_count": multiprocessors,
                    "device_name": device_name,
                }),
            };
            entries.push(serde_json::json!({
                "area": binding.area().name(),
                "candidate_coverage": super::super::test_support::qualify::coverage_json(candidate),
                "coverage": {"entries": tuples},
                "tier": binding.module.tier().to_string(),
                "devices": [speed.scope.capability().to_string()],
                "speed_scope": scope,
                "artifact": super::super::test_support::artifact_json(binding.module.artifact()),
                "record": speed.record.to_string(),
                "accuracy_record": accuracy.to_string(),
                "configurations": configurations,
                "boundary_domain": super::super::test_support::configuration::boundary_domain(),
                "models": super::super::test_support::configuration::model_identity(),
                "library_artifacts": super::super::test_support::configuration::library_artifacts(),
                "der": speed.integrated.to_string(),
            }));
        }
    }
    std::fs::write(
        path,
        serde_json::to_vec_pretty(&entries).expect("finite table"),
    )
    .expect("table output");
}

#[test]
fn planning_refusals_follow_the_d7_policy() {
    let device = device(BLACKWELL);
    let mut token = token(
        select(
            BoundaryId::named("lstm.stack"),
            1,
            CudaMath::Fp32,
            &device,
            legacy_module(KernelModule::Lstm),
        )
        .unwrap(),
    );
    let refusals = || {
        [
            PlanError::DeviceUnsupported {
                reason: "grid exceeds device capacity".to_owned(),
            },
            PlanError::WeightsOutOfContract {
                layer: "lstm.layer0.w",
                fault: WeightFault::NonFinite { index: 3 },
            },
            PlanError::Geometry(GeometryError::Unimplemented {
                context: "test plan",
                reason: "valid but unimplemented".to_owned(),
            }),
        ]
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
            let fallback = selection == Selection::Production && !driver_only;
            for refusal in refusals() {
                let result = token.finish::<()>(KernelModule::Lstm, driver_only, Err(refusal));
                if fallback {
                    assert!(result.unwrap().is_none());
                    continue;
                }
                match result.unwrap_err() {
                    CudaError::CandidateDeviceUnsupported {
                        area: "lstm",
                        boundary,
                        batch: 1,
                        math: CudaMath::Fp32,
                        tier: PtxTier::Sm75,
                        device,
                        reason,
                    } => {
                        assert_eq!(boundary, "lstm.stack");
                        assert_eq!(device, BLACKWELL);
                        assert_eq!(reason, "grid exceeds device capacity");
                    }
                    CudaError::CandidateWeightsOutOfContract {
                        area: "lstm",
                        layer: "lstm.layer0.w",
                        fault: WeightFault::NonFinite { index: 3 },
                        tier: PtxTier::Sm75,
                        ..
                    }
                    | CudaError::CandidateGeometry {
                        area: "lstm",
                        error: GeometryError::Unimplemented { .. },
                        ..
                    } => {}
                    other => panic!("wrong typed refusal: {other:?}"),
                }
            }
            // an invalid geometry and a real CUDA error are hard errors in every mode
            assert!(matches!(
                token.finish::<()>(
                    KernelModule::Lstm,
                    driver_only,
                    Err(PlanError::Geometry(GeometryError::Invalid {
                        context: "test plan",
                        reason: "violated invariant".to_owned(),
                    }))
                ),
                Err(CudaError::CandidateGeometry {
                    error: GeometryError::Invalid { .. },
                    ..
                })
            ));
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
fn every_production_tuple_requests_its_bound_ptx_jit() {
    let mut selected = 0;
    for binding in PRODUCTION {
        assert!(matches!(
            binding.module.artifact(),
            LoadedArtifact::PtxJit { .. }
        ));
        for proof in binding.proofs {
            let mut fixture = Fixture::new(BLACKWELL);
            let token = token(
                PlanRequest::Production
                    .resolve(proof.boundary, proof.batch, proof.math, &mut fixture)
                    .unwrap(),
            );
            assert_eq!(fixture.loads, [binding.module]);
            assert_eq!(token.target.module, binding.module);
            assert_eq!(token.pin, super::PlanPin::Pinned(proof.pin));
            selected += 1;
        }
    }
    assert_eq!(selected, 52);
}

/// Accuracy-accepted tuples must be implemented by the candidate at the bound tier;
/// speed acceptance lives inside each proof, so it can never exceed accuracy
#[test]
fn coverage_layers_nest() {
    for binding in PRODUCTION {
        let implemented = super::candidate_coverage(binding.area(), binding.module.tier());
        for proof in binding.proofs {
            assert!(
                implemented.covers(proof.boundary.name(), proof.batch, proof.math),
                "{proof:?} is accepted but not implemented"
            );
            assert_eq!(proof.pin.area(), binding.area());
        }
    }
}

#[test]
fn legacy_bindings_ignore_a_newly_embedded_higher_tier() {
    // the old loader picked the highest embedded variant at or below the limit
    for area in [
        KernelModule::Resnet,
        KernelModule::Lstm,
        KernelModule::Sincnet,
    ] {
        let embedded = AreaPtx::fixture(&[
            (PtxTier::Sm75, legacy_ptx(area)),
            (PtxTier::Sm80, "// a newly shipped sm80 variant"),
        ]);
        assert_eq!(embedded.select(PtxTier::Sm120).unwrap().0, PtxTier::Sm80);
        let device = device(BLACKWELL);
        let request = super::production_module(area, &device, PtxTier::Sm120, embedded).unwrap();
        assert_eq!(request, Some(legacy_module(area)));
        assert!(!super::legacy_fixture_coverage(area, &device, embedded).is_empty());
    }
    let mut selected = 0;
    for binding in PRODUCTION {
        for proof in binding.proofs {
            let mut fixture = Fixture::new(BLACKWELL);
            let token = token(
                PlanRequest::Production
                    .resolve(proof.boundary, proof.batch, proof.math, &mut fixture)
                    .unwrap(),
            );
            assert_eq!(token.target.module.tier(), PtxTier::Sm75);
            selected += 1;
        }
    }
    assert_eq!(selected, 52);
}

#[test]
fn always_on_areas_request_embedded_baseline_ptx_jit_on_every_device() {
    for capability in [BLACKWELL, ADA, ComputeCapability::new(7, 5)] {
        let device = device(capability);
        for area in super::ALWAYS_ON {
            let text = area.variants().embedded(PtxTier::Sm75).unwrap().text;
            assert_eq!(
                super::production_module(*area, &device, PtxTier::Sm120, area.variants()).unwrap(),
                Some(ModuleRequest::new(
                    *area,
                    PtxTier::Sm75,
                    LoadedArtifact::PtxJit {
                        sha256: ArtifactHash::of(text.as_bytes())
                    }
                ))
            );
            // a newly embedded higher tier does not move an always-on area either
            let embedded = AreaPtx::fixture(&[(PtxTier::Sm75, text), (PtxTier::Sm80, "sm80")]);
            assert_eq!(
                super::production_module(*area, &device, PtxTier::Sm120, embedded)
                    .unwrap()
                    .map(ModuleRequest::tier),
                Some(PtxTier::Sm75)
            );
        }
        assert_eq!(
            super::production_module(
                KernelModule::Probe,
                &device,
                PtxTier::Sm120,
                KernelModule::Probe.variants()
            )
            .unwrap(),
            None
        );
    }
}

const POINT: SpeedScope = SpeedScope::Point {
    capability: BLACKWELL,
    multiprocessors: 36,
    device_name: "NVIDIA GeForce RTX 5060 Ti",
};
const RECORD: RecordHash =
    RecordHash::from_hex("1111111111111111111111111111111111111111111111111111111111111111");
const C64_PIN: ConfigPin = ConfigPin::Conv(ConvPin::Kernel(ConvKernel::C64));

const fn fixture_proof(name: &str, batch: usize, pin: ConfigPin, speed: SpeedStatus) -> TupleProof {
    TupleProof {
        boundary: BoundaryId::named(name),
        batch,
        math: CudaMath::Fp32,
        pin,
        accuracy: RECORD,
        speed,
    }
}

const fn measured(scope: SpeedScope) -> SpeedStatus {
    SpeedStatus::Measured(SpeedEvidence {
        scope,
        record: RECORD,
        integrated: RECORD,
    })
}

fn cubin(area: KernelModule, tier: PtxTier) -> ModuleRequest {
    ModuleRequest::new(
        area,
        tier,
        LoadedArtifact::Cubin {
            arch: BLACKWELL,
            sha256: ArtifactHash::of(b"cubin"),
        },
    )
}

fn validate(bindings: &[Binding]) -> bool {
    let bindings = bindings.to_vec();
    std::panic::catch_unwind(move || {
        super::evidence::validate(&bindings, super::ALWAYS_ON, super::ROUTE_PRECEDENCE)
    })
    .is_ok()
}

#[test]
fn validator_rejects_conflicting_bindings_and_allows_many_proofs() {
    static ONE: [TupleProof; 1] = [fixture_proof(
        "resnet.layer2.1.conv1",
        32,
        C64_PIN,
        measured(POINT),
    )];
    static OTHER: [TupleProof; 1] = [fixture_proof(
        "resnet.layer2.1.conv2",
        32,
        C64_PIN,
        measured(POINT),
    )];
    let point = Binding {
        scope: POINT,
        module: cubin(KernelModule::Resnet, PtxTier::Sm80),
        proofs: &ONE,
    };
    assert!(validate(&[point]));
    // several records for one binding are allowed, but not two proofs of one tuple
    assert!(validate(&[
        point,
        Binding {
            proofs: &OTHER,
            ..point
        }
    ]));
    assert!(!validate(&[point, point]));
    // a second module for one area on an overlapping device scope cannot load
    assert!(!validate(&[
        point,
        Binding {
            module: cubin(KernelModule::Resnet, PtxTier::Sm75),
            proofs: &OTHER,
            ..point
        }
    ]));
    assert!(!validate(&[
        super::production::resnet::BINDING,
        Binding {
            proofs: &OTHER,
            ..point
        }
    ]));
    // another card of the same capability is a separate point scope
    static OTHER_CARD: [TupleProof; 1] = [fixture_proof(
        "resnet.layer2.1.conv1",
        32,
        C64_PIN,
        measured(SpeedScope::Point {
            capability: BLACKWELL,
            multiprocessors: 70,
            device_name: "NVIDIA GeForce RTX 5070 Ti",
        }),
    )];
    assert!(validate(&[
        point,
        Binding {
            scope: SpeedScope::Point {
                capability: BLACKWELL,
                multiprocessors: 70,
                device_name: "NVIDIA GeForce RTX 5070 Ti",
            },
            module: cubin(KernelModule::Resnet, PtxTier::Sm75),
            proofs: &OTHER_CARD,
        }
    ]));
}

#[test]
fn validator_rejects_misstated_evidence() {
    let point = |proofs: &'static [TupleProof]| Binding {
        scope: POINT,
        module: cubin(KernelModule::Resnet, PtxTier::Sm80),
        proofs,
    };
    static RULE: [TupleProof; 1] = [fixture_proof(
        "resnet.layer2.1.conv1",
        32,
        ConfigPin::Conv(ConvPin::LegacyWaves(ConvShape::C64)),
        measured(POINT),
    )];
    static LEGACY_EVIDENCE: [TupleProof; 1] = [fixture_proof(
        "resnet.layer2.1.conv1",
        32,
        C64_PIN,
        measured(SpeedScope::LegacyCapability {
            capability: BLACKWELL,
        }),
    )];
    static STRESS: [TupleProof; 1] = [fixture_proof(
        "resnet.layer2.1.conv1",
        7,
        C64_PIN,
        measured(POINT),
    )];
    static FOREIGN: [TupleProof; 1] = [fixture_proof(
        "resnet.layer2.1.conv1",
        32,
        ConfigPin::Sinc(SincPin::ConvAbsPool),
        measured(POINT),
    )];
    static UNMEASURED: [TupleProof; 1] = [fixture_proof(
        "resnet.layer2.1.conv1",
        32,
        C64_PIN,
        SpeedStatus::Unmeasured,
    )];
    // a selection rule is not a pin; legacy approval is not point evidence
    assert!(!validate(&[point(&RULE)]));
    assert!(!validate(&[point(&LEGACY_EVIDENCE)]));
    assert!(!validate(&[point(&STRESS)]));
    assert!(!validate(&[point(&FOREIGN)]));
    assert!(!validate(&[point(&[])]));
    assert!(validate(&[point(&UNMEASURED)]));
    // a cubin binding must target its exact device, and always-on areas have no bindings
    assert!(!validate(&[Binding {
        module: ModuleRequest::new(
            KernelModule::Resnet,
            PtxTier::Sm80,
            LoadedArtifact::Cubin {
                arch: ADA,
                sha256: ArtifactHash::of(b"cubin"),
            },
        ),
        ..point(&UNMEASURED)
    }]));
    assert!(!validate(&[Binding {
        module: cubin(KernelModule::Fbank, PtxTier::Sm75),
        ..point(&UNMEASURED)
    }]));
}

#[test]
fn point_scope_needs_the_exact_card() {
    let card = device(BLACKWELL);
    assert!(POINT.contains(&card));
    for other in [
        Builder::new(BLACKWELL)
            .multiprocessors(70)
            .name("NVIDIA GeForce RTX 5060 Ti")
            .build(),
        Builder::new(BLACKWELL)
            .multiprocessors(36)
            .name("NVIDIA GeForce RTX 5070")
            .build(),
        Builder::new(ADA)
            .multiprocessors(36)
            .name("NVIDIA GeForce RTX 5060 Ti")
            .build(),
    ] {
        assert!(!POINT.contains(&other));
        assert_eq!(
            super::production::LEGACY_SCOPE.contains(&other),
            other.capability() == BLACKWELL
        );
    }
}

#[test]
fn route_precedence_and_unmeasured_speed_decide_explicitly() {
    // two routes for one boundary: a fixed ResNet kernel and a hypothetical LSTM-area one
    static MEASURED: [TupleProof; 1] = [fixture_proof(
        "resnet.layer2.1.conv1",
        32,
        C64_PIN,
        measured(POINT),
    )];
    static UNMEASURED: [TupleProof; 1] = [fixture_proof(
        "resnet.layer2.1.conv1",
        32,
        C64_PIN,
        SpeedStatus::Unmeasured,
    )];
    static SECOND: [TupleProof; 1] = [fixture_proof(
        "resnet.layer2.1.conv1",
        32,
        ConfigPin::Lstm(LstmPin::LegacyCooperative),
        SpeedStatus::Measured(SpeedEvidence {
            scope: SpeedScope::LegacyCapability {
                capability: BLACKWELL,
            },
            record: RECORD,
            integrated: RECORD,
        }),
    )];
    const RESNET_ROUTE: Binding = Binding {
        scope: POINT,
        module: ModuleRequest::new(
            KernelModule::Resnet,
            PtxTier::Sm80,
            LoadedArtifact::Cubin {
                arch: BLACKWELL,
                sha256: ArtifactHash::from_hex(
                    "2222222222222222222222222222222222222222222222222222222222222222",
                ),
            },
        ),
        proofs: &MEASURED,
    };
    const LSTM_ROUTE: Binding = Binding {
        scope: SpeedScope::LegacyCapability {
            capability: BLACKWELL,
        },
        module: ModuleRequest::new(
            KernelModule::Lstm,
            PtxTier::Sm75,
            LoadedArtifact::PtxJit {
                sha256: ArtifactHash::from_hex(
                    "3333333333333333333333333333333333333333333333333333333333333333",
                ),
            },
        ),
        proofs: &SECOND,
    };
    static TABLE: [Binding; 2] = [RESNET_ROUTE, LSTM_ROUTE];
    static UNMEASURED_TABLE: [Binding; 2] = [
        Binding {
            proofs: &UNMEASURED,
            ..RESNET_ROUTE
        },
        LSTM_ROUTE,
    ];
    assert!(validate(&TABLE));
    let device = device(BLACKWELL);
    let route = |table: &'static [Binding], precedence: &[KernelModule]| {
        super::production_route(
            table,
            precedence,
            BoundaryId::named("resnet.layer2.1.conv1"),
            32,
            CudaMath::Fp32,
            &device,
        )
        .map(|route| route.module.area())
    };
    let resnet_first = [KernelModule::Resnet, KernelModule::Lstm];
    let lstm_first = [KernelModule::Lstm, KernelModule::Resnet];
    assert_eq!(route(&TABLE, &resnet_first), Some(KernelModule::Resnet));
    assert_eq!(route(&TABLE, &lstm_first), Some(KernelModule::Lstm));
    // the highest route decides: unmeasured speed means Library, not the next route
    assert_eq!(route(&UNMEASURED_TABLE, &resnet_first), None);
    assert_eq!(
        route(&UNMEASURED_TABLE, &lstm_first),
        Some(KernelModule::Lstm)
    );
    // a card outside the point scope never sees the point route
    let other = Builder::new(BLACKWELL).multiprocessors(70).build();
    assert_eq!(
        super::production_route(
            &TABLE,
            &resnet_first,
            BoundaryId::named("resnet.layer2.1.conv1"),
            32,
            CudaMath::Fp32,
            &other,
        )
        .map(|route| route.module.area()),
        Some(KernelModule::Lstm)
    );
}

#[test]
#[ignore = "short GPU proof; run under the shared GPU flock without diagnostic overrides"]
fn default_production_loads_record_pinned_jit() -> Result<(), CudaError> {
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
    assert_eq!(runtime.compute_capability(), BLACKWELL);
    let device = runtime.device();
    println!(
        "device_attributes {}",
        serde_json::json!({
            "name": device.name(), "capability": device.capability().to_string(),
            "sm_count": device.multiprocessors().get(), "l2_bytes": device.l2_bytes(),
            "shared_optin_bytes": device.shared_optin_bytes(),
        })
    );
    let mut selected = 0;
    for binding in PRODUCTION {
        for proof in binding.proofs {
            let token = token(super::plan_selection(
                &runtime,
                proof.boundary,
                proof.batch,
                proof.math,
                None,
            )?);
            assert_eq!(token.target.module, binding.module);
            assert_eq!(token.pin, super::PlanPin::Pinned(proof.pin));
            assert!(
                plan_from_pin(&runtime, proof, token)?,
                "{proof:?} fell back to Library at plan time"
            );
            selected += 1;
        }
        let loaded = runtime.load_kernels(binding.area())?;
        assert_eq!(loaded.request(), binding.module);
        let ptx = binding
            .area()
            .variants()
            .embedded(binding.module.tier())
            .unwrap()
            .text;
        for line in ptx
            .lines()
            .filter_map(|line| line.trim().strip_prefix(".visible .entry "))
        {
            loaded.function(line.split(['(', ' ', '\t']).next().unwrap())?;
        }
        println!(
            "production_artifact_proof {}",
            serde_json::json!({
                "area": binding.area().name(), "tuples": binding.proofs.len(),
                "tier": loaded.tier().to_string(), "device": runtime.compute_capability().to_string(),
                "artifact": crate::inference::cuda::test_support::artifact_json(loaded.artifact()),
                "embedded_ptx_sha256": loaded.ptx_sha256().to_string(),
            })
        );
    }
    assert_eq!(selected, 52);
    println!(
        "planned_from_pins {}",
        serde_json::json!({ "tuples": selected })
    );
    for area in super::ALWAYS_ON {
        let expected = runtime.production_module(*area)?.expect("always-on module");
        let loaded = runtime.load_kernels(*area)?;
        assert_eq!(loaded.request(), expected);
        assert_eq!(
            loaded.artifact(),
            LoadedArtifact::PtxJit {
                sha256: loaded.ptx_sha256()
            }
        );
        let ptx = area.variants().embedded(expected.tier()).unwrap().text;
        for line in ptx
            .lines()
            .filter_map(|line| line.trim().strip_prefix(".visible .entry "))
        {
            loaded.function(line.split(['(', ' ', '\t']).next().unwrap())?;
        }
        println!(
            "always_on_artifact_proof {}",
            serde_json::json!({
                "area": area.name(), "tier": loaded.tier().to_string(),
                "device": runtime.compute_capability().to_string(),
                "artifact": crate::inference::cuda::test_support::artifact_json(loaded.artifact()),
                "embedded_ptx_sha256": loaded.ptx_sha256().to_string(),
            })
        );
    }
    let modules = crate::inference::cuda::test_support::loaded_modules();
    let modules = modules.as_array().expect("recorded module array");
    assert_eq!(modules.len(), 6);
    assert!(
        modules
            .iter()
            .all(|module| module["artifact"]["kind"] == "PtxJit")
    );
    runtime.synchronize()
}

/// Build one accepted tuple's plan from its pin on the live device, with zero weights
/// at the model's shapes; `false` means production fell back to Library
fn plan_from_pin(
    runtime: &crate::inference::cuda::CudaRuntime,
    proof: &TupleProof,
    token: super::Qualified,
) -> Result<bool, CudaError> {
    use crate::inference::cuda::candidate::{ConvLayerSpec, LstmLayerWeights, LstmSpec, SincSpec};
    use crate::inference::cuda::dnn::Conv2d;
    use crate::inference::cuda::{FBANK_FRAMES, FBANK_MEL_BINS};

    let stream = runtime.stream();
    let (batch, math) = (proof.batch, proof.math);
    match proof.pin {
        ConfigPin::Fbank(_) => {
            let spec = crate::inference::cuda::candidate::FbankSpec::new(batch, math).map_err(
                |error| CudaError::Unsupported {
                    context: "fbank.dft production proof",
                    reason: error.to_string(),
                },
            )?;
            Ok(token.fbank(runtime, spec)?.is_some())
        }
        ConfigPin::Conv(pin) => {
            let trunk = [FBANK_MEL_BINS, FBANK_FRAMES];
            let ([in_channels, out_channels], stride, input) = match pin.shape() {
                ConvShape::C32 => ([32, 32], 1, trunk),
                ConvShape::C32Stride2 => ([32, 64], 2, trunk),
                ConvShape::C64 => ([64, 64], 1, trunk.map(|size| size.div_ceil(2))),
            };
            let weight = stream.alloc_zeros::<f32>(out_channels * in_channels * 9)?;
            let bias = stream.alloc_zeros::<f32>(out_channels)?;
            let conv = Conv2d {
                batch,
                in_channels,
                out_channels,
                input,
                kernel: [3, 3],
                padding: [1, 1],
                stride: [stride; 2],
                dilation: [1, 1],
                math,
            };
            let spec = ConvLayerSpec {
                name: proof.boundary.name(),
                conv,
                residual: proof.boundary.name().ends_with("conv2"),
                weight: &weight,
                bias: &bias,
            };
            Ok(token.conv(runtime, spec)?.is_some())
        }
        ConfigPin::Lstm(_) => {
            let first = (vec![0.0; 2 * 512 * 60], vec![0.0; 2 * 512 * 128]);
            let upper = (vec![0.0; 2 * 512 * 256], vec![0.0; 2 * 512 * 128]);
            let bias = vec![0.0; 2 * 1024];
            let (w0, r0) = (first.0.as_slice(), first.1.as_slice());
            let (w1, r1) = (upper.0.as_slice(), upper.1.as_slice());
            let weights = |input, w, r| LstmLayerWeights {
                input,
                w,
                r,
                b: bias.as_slice(),
            };
            let spec = LstmSpec {
                batch,
                frames: 589,
                math,
                layers: [
                    weights(60, w0, r0),
                    weights(256, w1, r1),
                    weights(256, w1, r1),
                    weights(256, w1, r1),
                ],
            };
            Ok(token.lstm(runtime, spec)?.is_some())
        }
        ConfigPin::Sinc(_) => {
            let filters = stream.alloc_zeros::<f32>(80 * 251)?;
            let spec = SincSpec {
                batch,
                samples: 160_000,
                sinc: 15_975,
                pooled: 5_325,
                math,
                filters: &filters,
            };
            Ok(token.sinc(runtime, spec)?.is_some())
        }
    }
}
