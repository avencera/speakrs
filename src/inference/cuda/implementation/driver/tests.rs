//! Host routing proofs without a CUDA context or an optional library

use super::{Area, select_from};
use crate::inference::cuda::candidate::{
    Batches, ConfigPin, Coverage, CoverageEntry, DriverCandidate, Maths, PlanError, SincPin,
};
use crate::inference::cuda::device::{DeviceAttributes, test_support::Builder};
use crate::inference::cuda::implementation::{
    BoundaryId, Modules, PlanPin, PlanRequest, Selected, TokenEvidence,
};
use crate::inference::cuda::kernels::ModuleRequest;
use crate::inference::cuda::{ComputeCapability, CudaError, CudaMath, KernelModule, PtxTier};

struct Fixture {
    device: DeviceAttributes,
    loads: Vec<ModuleRequest>,
    limit: PtxTier,
    force_jit: bool,
}

impl Fixture {
    fn new() -> Self {
        Self {
            device: Builder::new(ComputeCapability::new(12, 0))
                .multiprocessors(36)
                .shared_optin_bytes(99 << 10)
                .name("unmeasured GPU")
                .build(),
            loads: vec![],
            limit: PtxTier::Sm120,
            force_jit: false,
        }
    }
}

impl Modules for &mut Fixture {
    fn effective_request(&self, request: ModuleRequest) -> Result<ModuleRequest, CudaError> {
        request.diagnostic_request(self.force_jit)
    }

    fn device(&self) -> &DeviceAttributes {
        &self.device
    }
    fn tier_limit(&self) -> PtxTier {
        self.limit
    }
    fn load(&mut self, request: ModuleRequest) -> Result<ModuleRequest, CudaError> {
        self.loads.push(request);
        Ok(request)
    }
    #[cfg(feature = "_cuda-libraries")]
    fn embedded_exact(&self, _area: KernelModule) -> Result<ModuleRequest, CudaError> {
        unreachable!()
    }
}

// the stub provides a library-free route through the same interface as an area port
struct Stub;
impl DriverCandidate for Stub {
    const AREA: KernelModule = KernelModule::Sincnet;
    fn driver_coverage(_tier: PtxTier) -> Coverage {
        Coverage(&[CoverageEntry {
            layers: &["sincnet.conv0.abs_pool"],
            batches: Batches::Only(&[1]),
            maths: Maths::Only(&[CudaMath::Fp32]),
        }])
    }
    fn driver_pin(
        _boundary: BoundaryId,
        _batch: usize,
        _math: CudaMath,
        _device: &DeviceAttributes,
        _tier: PtxTier,
    ) -> Result<ConfigPin, PlanError> {
        Ok(ConfigPin::Sinc(SincPin::ConvAbsPool))
    }
}

#[test]
fn stub_area_routes_implemented_coverage_and_marks_speed_unmeasured() {
    let mut fixture = Fixture::new();
    let selected = select_from(
        &[Area::candidate::<Stub>()],
        BoundaryId::named("sincnet.conv0.abs_pool"),
        1,
        CudaMath::Fp32,
        &mut &mut fixture,
        super::Selection::DriverOnly,
    )
    .unwrap();
    let Some(Selected::Oxide(token)) = selected else {
        panic!("covered tuple must use the stub")
    };
    assert!(matches!(
        token.pin,
        PlanPin::Pinned(ConfigPin::Sinc(SincPin::ConvAbsPool))
    ));
    assert_eq!(token.evidence, TokenEvidence::Implemented);
    assert_eq!(fixture.loads, [token.target.module]);
}

#[test]
fn missing_area_fails_typed_before_any_artifact_load() {
    let mut fixture = Fixture::new();
    for (boundary, batch, math) in [
        ("resnet.conv1", 1, CudaMath::Fp32),
        ("linear0", 2, CudaMath::Tf32),
        ("sincnet.conv1", 3, CudaMath::Fp32),
    ] {
        let result = select_from(
            &[],
            BoundaryId::named(boundary),
            batch,
            math,
            &mut &mut fixture,
            super::Selection::DriverOnly,
        );
        assert!(
            matches!(result, Err(CudaError::MissingKernel { boundary: actual, batch: b, math: m }) if actual == boundary && b == batch && m == math)
        );
    }
    assert!(fixture.loads.is_empty());
}

#[test]
fn stub_uncovered_tuple_is_not_selected() {
    let mut fixture = Fixture::new();
    let result = select_from(
        &[Area::candidate::<Stub>()],
        BoundaryId::named("sincnet.conv0.abs_pool"),
        32,
        CudaMath::Fp32,
        &mut &mut fixture,
        super::Selection::DriverOnly,
    );
    assert!(matches!(
        result,
        Err(CudaError::MissingKernel { batch: 32, .. })
    ));
    assert!(fixture.loads.is_empty());
}

#[test]
fn driver_configuration_is_fixed_from_cached_device_attributes() {
    use crate::inference::cuda::candidate::{ConvKernel, ConvOxide, ConvPin};
    let boundary = BoundaryId::named("resnet.layer2.0.conv1");
    let small_device = Builder::new(ComputeCapability::new(12, 0))
        .multiprocessors(1)
        .build();
    let large_device = Builder::new(ComputeCapability::new(12, 0))
        .multiprocessors(1000)
        .build();
    let pin =
        |device| ConvOxide::driver_pin(boundary, 1, CudaMath::Fp32, device, PtxTier::Sm75).unwrap();
    assert_eq!(
        pin(&small_device),
        ConfigPin::Conv(ConvPin::Kernel(ConvKernel::C32Stride2))
    );
    assert_eq!(
        pin(&large_device),
        ConfigPin::Conv(ConvPin::Kernel(ConvKernel::C32Stride2Small))
    );
    assert_eq!(pin(&large_device), pin(&large_device));
}

struct BroadStub;
impl DriverCandidate for BroadStub {
    const AREA: KernelModule = KernelModule::Sincnet;
    fn driver_coverage(tier: PtxTier) -> Coverage {
        Stub::driver_coverage(tier)
    }
    fn broad_evidence() -> Option<&'static crate::inference::cuda::implementation::BroadEvidence> {
        use crate::inference::cuda::implementation::{ArchitectureSpeed, BroadEvidence};
        static SUMMARY: BroadEvidence = BroadEvidence::with_minimum(
            &[
                ArchitectureSpeed {
                    capability: ComputeCapability::new(8, 0),
                    minimum_speedup_milli: 1200,
                },
                ArchitectureSpeed {
                    capability: ComputeCapability::new(8, 9),
                    minimum_speedup_milli: 1500,
                },
            ],
            "fixture: fused producer removes an intermediate buffer on Ampere and Ada",
            crate::inference::cuda::ComputeCapability::new(7, 5),
        );
        Some(&SUMMARY)
    }
    fn driver_pin(
        boundary: BoundaryId,
        batch: usize,
        math: CudaMath,
        device: &DeviceAttributes,
        tier: PtxTier,
    ) -> Result<ConfigPin, PlanError> {
        Stub::driver_pin(boundary, batch, math, device, tier)
    }
}

#[test]
fn hybrid_broad_port_uses_explicit_evidence_on_an_unmeasured_device() {
    let mut fixture = Fixture::new();
    let selected = select_from(
        &[Area::candidate::<BroadStub>()],
        BoundaryId::named("sincnet.conv0.abs_pool"),
        1,
        CudaMath::Fp32,
        &mut &mut fixture,
        super::Selection::Production,
    )
    .unwrap();
    let Some(Selected::Oxide(token)) = selected else {
        panic!("broad port must run on an unmeasured GPU")
    };
    let TokenEvidence::Port { scope, .. } = token.evidence else {
        panic!("broad port must carry its evidence")
    };
    assert!(scope.contains(&fixture.device));
    assert!(!scope.measured_on_device(fixture.device.capability()));
    assert_eq!(token.selection, super::Selection::Production);
    assert_eq!(fixture.loads, [token.target.module]);
}

#[cfg(feature = "_cuda-libraries")]
struct RefusingBroad;
#[cfg(feature = "_cuda-libraries")]
impl DriverCandidate for RefusingBroad {
    const AREA: KernelModule = KernelModule::Sincnet;
    fn driver_coverage(tier: PtxTier) -> Coverage {
        Stub::driver_coverage(tier)
    }
    fn broad_evidence() -> Option<&'static crate::inference::cuda::implementation::BroadEvidence> {
        BroadStub::broad_evidence()
    }
    fn driver_pin(
        _boundary: BoundaryId,
        _batch: usize,
        _math: CudaMath,
        _device: &DeviceAttributes,
        _tier: PtxTier,
    ) -> Result<ConfigPin, PlanError> {
        Err(PlanError::DeviceUnsupported {
            reason: "stub resource refusal".to_owned(),
        })
    }
}

#[test]
#[cfg(feature = "_cuda-libraries")]
fn hybrid_refusal_is_final_even_when_the_legacy_table_has_a_route() {
    let mut fixture = Fixture::new();
    let selected = PlanRequest::Hybrid
        .resolve_with_candidates(
            BoundaryId::named("sincnet.conv0.abs_pool"),
            1,
            CudaMath::Fp32,
            &mut fixture,
            &[Area::candidate::<RefusingBroad>()],
        )
        .unwrap();
    assert!(matches!(selected, Selected::Library));
    assert!(fixture.loads.is_empty());
}

#[test]
#[cfg(feature = "_cuda-libraries")]
fn merged_ports_obey_broad_and_device_sensitive_speed_scopes() {
    let mut fixture = Fixture::new();
    fixture.device = Builder::new(ComputeCapability::new(9, 0))
        .name("unmeasured Hopper")
        .build();
    for boundary in ["linear0", "sincnet.conv1", "resnet.seg_1", "fbank.dft"] {
        let selected = PlanRequest::Hybrid
            .resolve(BoundaryId::named(boundary), 1, CudaMath::Fp32, &mut fixture)
            .unwrap();
        let Selected::Oxide(token) = selected else {
            panic!("broad port must run on Hopper")
        };
        let TokenEvidence::Port { scope, .. } = token.evidence else {
            panic!("missing speed policy")
        };
        assert!(matches!(
            scope,
            crate::inference::cuda::implementation::SpeedScope::AllDevices(_)
        ));
        assert!(!scope.measured_on_device(fixture.device.capability()));
    }
    let selected = PlanRequest::Hybrid
        .resolve(
            BoundaryId::named("lstm.stack"),
            1,
            CudaMath::Fp32,
            &mut fixture,
        )
        .unwrap();
    assert!(matches!(selected, Selected::Library));
    fixture.device = Builder::new(ComputeCapability::new(7, 5)).build();
    let selected = PlanRequest::Hybrid
        .resolve(
            BoundaryId::named("linear0"),
            1,
            CudaMath::Fp32,
            &mut fixture,
        )
        .unwrap();
    assert!(matches!(selected, Selected::Library));
}

#[test]
#[cfg(feature = "_cuda-libraries")]
fn fbank_speed_scope_depends_on_math_and_segdense_requires_the_measured_tier() {
    let mut fixture = Fixture::new();
    for capability in [
        ComputeCapability::new(12, 0),
        ComputeCapability::new(8, 9),
        ComputeCapability::new(7, 5),
    ] {
        fixture.device = Builder::new(capability).build();
        for math in [CudaMath::Fp32, CudaMath::Tf32] {
            let selected = PlanRequest::Hybrid
                .resolve(BoundaryId::named("fbank.dft"), 1, math, &mut fixture)
                .unwrap();
            let kernel = capability >= ComputeCapability::new(8, 0)
                && (math == CudaMath::Fp32 || capability == ComputeCapability::new(8, 9));
            assert_eq!(
                matches!(selected, Selected::Oxide(_)),
                kernel,
                "fbank {capability:?} {math:?}"
            );
        }
    }
    fixture.device = Builder::new(ComputeCapability::new(8, 9)).build();
    fixture.limit = PtxTier::Sm75;
    let selected = PlanRequest::Hybrid
        .resolve(
            BoundaryId::named("linear0"),
            1,
            CudaMath::Fp32,
            &mut fixture,
        )
        .unwrap();
    assert!(matches!(selected, Selected::Library));
    let selected = PlanRequest::DriverOnly
        .resolve(
            BoundaryId::named("linear0"),
            1,
            CudaMath::Fp32,
            &mut fixture,
        )
        .unwrap();
    let Selected::Oxide(token) = selected else {
        panic!("driver-only uses the unmeasured sm75 implementation")
    };
    assert_eq!(token.evidence, TokenEvidence::Implemented);
}

#[test]
fn resnet_point_binding_shares_one_artifact_across_routes_and_modes() {
    use crate::inference::cuda::candidate::{ConvKernel, ConvPin};
    use crate::inference::cuda::kernels::{ArtifactHash, LoadedArtifact};

    let mut fixture = Fixture::new();
    fixture.device = Builder::new(ComputeCapability::new(12, 0))
        .multiprocessors(36)
        .name("NVIDIA GeForce RTX 5060 Ti")
        .build();
    let production = super::super::production_module(
        KernelModule::Resnet,
        &fixture.device,
        fixture.limit,
        KernelModule::Resnet.variants(),
    )
    .unwrap()
    .unwrap();
    assert_eq!(production.tier(), PtxTier::Sm80);
    assert_eq!(
        production.artifact(),
        LoadedArtifact::Cubin {
            arch: ComputeCapability::new(12, 0),
            sha256: ArtifactHash::from_hex(
                "0950b0d84cd9fa3d9053cd30399fce14a6aa6c3ff8777485598dd8deeba89078"
            ),
        }
    );
    for batch in [1, 32] {
        for (name, fp32, tf32) in [
            (
                "resnet.layer1.0.conv1",
                ConvKernel::C32,
                ConvKernel::C32Tensor,
            ),
            (
                "resnet.layer2.0.conv1",
                ConvKernel::C32Stride2,
                ConvKernel::C32Stride2Tensor,
            ),
            (
                "resnet.layer2.3.conv2",
                ConvKernel::C64,
                ConvKernel::C64Tensor,
            ),
        ] {
            for (math, kernel) in [(CudaMath::Fp32, fp32), (CudaMath::Tf32, tf32)] {
                let requests = [
                    PlanRequest::DriverOnly,
                    #[cfg(feature = "_cuda-libraries")]
                    PlanRequest::Hybrid,
                ];
                for request in requests {
                    let Selected::Oxide(token) = request
                        .resolve(BoundaryId::named(name), batch, math, &mut fixture)
                        .unwrap()
                    else {
                        panic!("missing measured ResNet plan")
                    };
                    assert_eq!(token.target.module, production);
                    assert_eq!(
                        token.pin,
                        PlanPin::Pinned(ConfigPin::Conv(ConvPin::Kernel(kernel)))
                    );
                    assert!(
                        matches!(token.evidence, TokenEvidence::Port { scope, .. } if scope == crate::inference::cuda::candidate::ConvOxide::RTX50_SCOPE)
                    );
                    assert!(production.check_cached(token.target.module).is_ok());
                }
            }
        }
    }
    assert!(fixture.loads.iter().all(|request| *request == production));
}

#[test]
#[cfg(feature = "_cuda-libraries")]
fn resnet_speed_does_not_extend_to_other_blackwell_cards_or_tiers() {
    let boundary = BoundaryId::named("resnet.layer1.0.conv1");
    for (sms, name) in [
        (70, "NVIDIA GeForce RTX 5070 Ti"),
        (36, "another card"),
        (70, "NVIDIA GeForce RTX 5060 Ti"),
    ] {
        let mut fixture = Fixture::new();
        fixture.device = Builder::new(ComputeCapability::new(12, 0))
            .multiprocessors(sms)
            .name(name)
            .build();
        for math in [CudaMath::Fp32, CudaMath::Tf32] {
            assert!(matches!(
                PlanRequest::Hybrid
                    .resolve(boundary, 1, math, &mut fixture)
                    .unwrap(),
                Selected::Library
            ));
            let Selected::Oxide(token) = PlanRequest::DriverOnly
                .resolve(boundary, 1, math, &mut fixture)
                .unwrap()
            else {
                panic!("driver route still covers unmeasured devices")
            };
            assert_eq!(token.evidence, TokenEvidence::Implemented);
        }
    }
    let mut fixture = Fixture::new();
    fixture.device = Builder::new(ComputeCapability::new(12, 0))
        .multiprocessors(36)
        .name("NVIDIA GeForce RTX 5060 Ti")
        .build();
    fixture.limit = PtxTier::Sm75;
    for math in [CudaMath::Fp32, CudaMath::Tf32] {
        assert!(matches!(
            PlanRequest::Hybrid
                .resolve(boundary, 1, math, &mut fixture)
                .unwrap(),
            Selected::Library
        ));
        let Selected::Oxide(token) = PlanRequest::DriverOnly
            .resolve(boundary, 1, math, &mut fixture)
            .unwrap()
        else {
            panic!("baseline implementation")
        };
        assert_eq!(token.target.module.tier(), PtxTier::Sm75);
        assert_eq!(token.evidence, TokenEvidence::Implemented);
    }
}

#[test]
#[cfg(feature = "_cuda-libraries")]
fn trunk_scope_matches_measured_totals_and_library_fallbacks() {
    let mut fixture = Fixture::new();
    for capability in [
        ComputeCapability::new(12, 0),
        ComputeCapability::new(8, 9),
        ComputeCapability::new(9, 0),
    ] {
        fixture.device = Builder::new(capability)
            .multiprocessors(36)
            .name("NVIDIA GeForce RTX 5060 Ti")
            .build();
        for batch in [1, 7, 32, 33] {
            for math in [CudaMath::Fp32, CudaMath::Tf32] {
                for boundary in [
                    "resnet.conv1",
                    "resnet.layer2.0.conv2",
                    "resnet.layer3.0.conv1",
                ] {
                    let selected = PlanRequest::Hybrid
                        .resolve(BoundaryId::named(boundary), batch, math, &mut fixture)
                        .unwrap();
                    let expected = matches!(batch, 1 | 32)
                        && (capability == ComputeCapability::new(12, 0)
                            || capability == ComputeCapability::new(8, 9)
                                && (batch == 32 || math == CudaMath::Tf32));
                    assert_eq!(
                        matches!(selected, Selected::Oxide(_)),
                        expected,
                        "{capability:?} {boundary} b{batch} {math:?}"
                    );
                    assert!(matches!(
                        PlanRequest::DriverOnly
                            .resolve(BoundaryId::named(boundary), batch, math, &mut fixture)
                            .unwrap(),
                        Selected::Oxide(_)
                    ));
                }
            }
        }
    }
}

#[test]
fn every_model_boundary_has_a_driver_route_for_model_batches() {
    let mut fixture = Fixture::new();
    for boundary in
        BoundaryId::all().filter(|boundary| *boundary != BoundaryId::named("lstm.stack.input_proj"))
    {
        // the projected stack owns input projection; only the legacy stack names it separately
        for batch in [1, 32] {
            for math in [CudaMath::Fp32, CudaMath::Tf32] {
                assert!(
                    matches!(
                        PlanRequest::DriverOnly
                            .resolve(boundary, batch, math, &mut fixture)
                            .unwrap(),
                        Selected::Oxide(_)
                    ),
                    "{boundary:?} b{batch} {math:?}"
                );
            }
        }
    }
}

#[test]
fn tensor_core_trunk_kernels_are_selected_only_for_tf32_on_measured_capabilities() {
    use crate::inference::cuda::candidate::{ConvKernel, ConvOxide, ConvPin};
    let a100 = Builder::new(ComputeCapability::new(8, 0))
        .multiprocessors(108)
        .build();
    let ada = Builder::new(ComputeCapability::new(8, 9))
        .multiprocessors(34)
        .build();
    let pin = |name, batch, math, device: &_, tier| {
        ConvOxide::driver_pin(BoundaryId::named(name), batch, math, device, tier).unwrap()
    };
    let kernel = |kernel| ConfigPin::Conv(ConvPin::Kernel(kernel));
    for (device, batch) in [(&a100, 1), (&a100, 32), (&ada, 1), (&ada, 32)] {
        assert_eq!(
            pin(
                "resnet.layer1.0.conv1",
                batch,
                CudaMath::Tf32,
                device,
                PtxTier::Sm80
            ),
            kernel(ConvKernel::C32Tensor)
        );
        assert_eq!(
            pin(
                "resnet.layer2.3.conv2",
                batch,
                CudaMath::Tf32,
                device,
                PtxTier::Sm80
            ),
            kernel(ConvKernel::C64Tensor)
        );
    }
    for (device, batch) in [(&a100, 1), (&a100, 32), (&ada, 1), (&ada, 32)] {
        assert_eq!(
            pin(
                "resnet.layer2.0.conv1",
                batch,
                CudaMath::Tf32,
                device,
                PtxTier::Sm80
            ),
            kernel(ConvKernel::C32Stride2Tensor)
        );
    }
    // FP32 mode and the sm75 tier keep the FP32 kernels
    assert_eq!(
        pin(
            "resnet.layer1.0.conv1",
            32,
            CudaMath::Fp32,
            &a100,
            PtxTier::Sm80
        ),
        kernel(ConvKernel::C32)
    );
    assert_eq!(
        pin(
            "resnet.layer2.1.conv1",
            32,
            CudaMath::Tf32,
            &a100,
            PtxTier::Sm75
        ),
        kernel(ConvKernel::C64)
    );
    // other sm80-tier parts keep their current selection
    for (major, minor, sms) in [(8, 6, 84), (9, 0, 132), (12, 0, 36)] {
        let device = Builder::new(ComputeCapability::new(major, minor))
            .multiprocessors(sms)
            .build();
        for name in [
            "resnet.layer1.0.conv1",
            "resnet.layer2.0.conv1",
            "resnet.layer2.1.conv1",
        ] {
            let selected = pin(name, 32, CudaMath::Tf32, &device, PtxTier::Sm80);
            assert!(
                !matches!(
                    selected,
                    ConfigPin::Conv(ConvPin::Kernel(
                        ConvKernel::C32Tensor
                            | ConvKernel::C64Tensor
                            | ConvKernel::C32Stride2Tensor
                    ))
                ),
                "{major}.{minor} {name}: {selected:?}"
            );
        }
    }
}

#[test]
fn forced_jit_fixes_driver_and_hybrid_identity_without_cubin_speed_evidence() {
    use crate::inference::cuda::kernels::LoadedArtifact;

    for selection in [super::Selection::DriverOnly, super::Selection::Production] {
        let mut fixture = Fixture::new();
        fixture.force_jit = true;
        fixture.device = Builder::new(ComputeCapability::new(8, 9)).build();
        let selected_request = super::super::production_module(
            KernelModule::Sincnet,
            &fixture.device,
            fixture.limit,
            KernelModule::Sincnet.variants(),
        )
        .unwrap()
        .unwrap();
        assert!(matches!(
            selected_request.artifact(),
            LoadedArtifact::Cubin { .. }
        ));
        let selected = select_from(
            &[Area::candidate::<BroadStub>()],
            BoundaryId::named("sincnet.conv0.abs_pool"),
            1,
            CudaMath::Fp32,
            &mut &mut fixture,
            selection,
        )
        .unwrap();
        let Some(Selected::Oxide(token)) = selected else {
            panic!("forced JIT must select the port")
        };
        assert_eq!(
            token.target.module,
            selected_request.diagnostic_request(true).unwrap()
        );
        assert!(matches!(
            token.target.module.artifact(),
            LoadedArtifact::PtxJit { .. }
        ));
        assert_eq!(fixture.loads, [token.target.module]);
        assert_eq!(token.evidence, TokenEvidence::Implemented);
        assert!(!token.speed_measured());
    }
}
