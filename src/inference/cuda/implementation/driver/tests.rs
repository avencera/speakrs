//! Host routing proofs without a CUDA context or an optional library

use super::{Area, select_from};
use crate::inference::cuda::candidate::{
    Batches, ConfigPin, Coverage, CoverageEntry, DriverCandidate, Fp16Policy, Maths, PlanError,
    SincPin,
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
    refuse_load: bool,
    benchmarking: bool,
    limit: PtxTier,
    recipe_mode: super::super::policy::RecipeMode,
    tuned: Option<crate::inference::cuda::tuning::ApprovedChoice>,
    fp16: Fp16Policy,
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
            refuse_load: false,
            benchmarking: false,
            limit: PtxTier::Sm120,
            recipe_mode: super::super::policy::RecipeMode::Disabled,
            tuned: None,
            fp16: Fp16Policy::Allowed,
            force_jit: false,
        }
    }
}

impl Modules for &mut Fixture {
    fn is_tuning(&self) -> bool {
        self.benchmarking
    }

    fn tune_choice(
        &self,
        _boundary: BoundaryId,
        _batch: usize,
        _math: CudaMath,
    ) -> Result<Option<crate::inference::cuda::tuning::ApprovedChoice>, CudaError> {
        Ok(self.tuned.clone())
    }

    fn recipe_mode(&self) -> super::super::policy::RecipeMode {
        self.recipe_mode
    }

    fn fp16(&self) -> Fp16Policy {
        self.fp16
    }

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
        if self.refuse_load {
            return Err(CudaError::ArtifactUnavailable {
                module: request.area().name(),
                artifact: request.artifact(),
            });
        }
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
    fn driver_coverage(_tier: PtxTier, _device: &DeviceAttributes, _fp16: Fp16Policy) -> Coverage {
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
        _fp16: Fp16Policy,
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
    let pin = |device| {
        ConvOxide::driver_pin(
            boundary,
            1,
            CudaMath::Fp32,
            device,
            PtxTier::Sm75,
            Fp16Policy::Allowed,
        )
        .unwrap()
    };
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
    fn driver_coverage(tier: PtxTier, device: &DeviceAttributes, fp16: Fp16Policy) -> Coverage {
        Stub::driver_coverage(tier, device, fp16)
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
        fp16: Fp16Policy,
    ) -> Result<ConfigPin, PlanError> {
        Stub::driver_pin(boundary, batch, math, device, tier, fp16)
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
    fn driver_coverage(tier: PtxTier, device: &DeviceAttributes, fp16: Fp16Policy) -> Coverage {
        Stub::driver_coverage(tier, device, fp16)
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
        _fp16: Fp16Policy,
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
fn merged_ports_use_kernels_without_device_speed_evidence() {
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
    assert!(matches!(selected, Selected::Oxide(_)));
    fixture.device = Builder::new(ComputeCapability::new(7, 5)).build();
    let selected = PlanRequest::Hybrid
        .resolve(
            BoundaryId::named("linear0"),
            1,
            CudaMath::Fp32,
            &mut fixture,
        )
        .unwrap();
    assert!(matches!(selected, Selected::Oxide(_)));
}

#[test]
#[cfg(feature = "_cuda-libraries")]
fn fbank_and_segdense_use_implemented_coverage_at_each_tier() {
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
            let kernel = true;
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
    assert!(matches!(selected, Selected::Oxide(_)));
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
                "f9cada62aab90b3a641ee5fb6232fa951bd6a14d4c38926d550fdded7b2a00c3"
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
fn resnet_recipe_does_not_extend_but_tf32_class_default_does() {
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
            let selected = PlanRequest::Hybrid
                .resolve(boundary, 1, math, &mut fixture)
                .unwrap();
            match (math, selected) {
                (CudaMath::Fp32, Selected::Oxide(token)) => {
                    assert_eq!(token.evidence, TokenEvidence::Implemented)
                }
                (CudaMath::Tf32, Selected::Oxide(token)) => assert_eq!(
                    token.evidence,
                    TokenEvidence::DeviceDefault(
                        super::super::policy::DeviceDefault::AmpereTf32EarlyTrunk
                    )
                ),
                (_, selected) => panic!("unexpected class selection: {selected:?}"),
            }
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
            Selected::Oxide(_)
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
fn trunk_defaults_cover_unmeasured_devices_and_batch_sizes() {
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
                    let expected = true;
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
fn tensor_core_trunk_kernels_are_selected_only_for_tf32_on_ampere_and_newer() {
    use crate::inference::cuda::candidate::{ConvKernel, ConvOxide, ConvPin};
    let a100 = Builder::new(ComputeCapability::new(8, 0))
        .multiprocessors(108)
        .build();
    let ada = Builder::new(ComputeCapability::new(8, 9))
        .multiprocessors(34)
        .build();
    let pin = |name, batch, math, device: &_, tier| {
        ConvOxide::driver_pin(
            BoundaryId::named(name),
            batch,
            math,
            device,
            tier,
            Fp16Policy::Allowed,
        )
        .unwrap()
    };
    let kernel = |kernel| ConfigPin::Conv(ConvPin::Kernel(kernel));
    // the A100's 64-channel batch-1 grid would leave SMs idle in 56-column tiles
    for (device, batch, c64) in [
        (&a100, 1, ConvKernel::C64TensorSlim),
        (&a100, 32, ConvKernel::C64Tensor),
        (&ada, 1, ConvKernel::C64Tensor),
        (&ada, 32, ConvKernel::C64Tensor),
    ] {
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
            kernel(c64)
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
    // unmeasured Ampere and newer parts use the same class-default tensor entries
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
                matches!(
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
fn fp16_routes_send_the_early_trunk_layers_to_wideconv() {
    use crate::inference::cuda::candidate::{
        WideconvAlgorithm, WideconvFp16Tiles, WideconvPin, WideconvProducts,
    };

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Route {
        Turing,
        Ada,
        None,
    }

    let devices = [
        (
            ComputeCapability::new(7, 5),
            40,
            PtxTier::Sm75,
            Route::Turing,
        ),
        // an sm75-tier build on a newer part keeps the direct kernel
        (ComputeCapability::new(8, 6), 84, PtxTier::Sm75, Route::None),
        (ComputeCapability::new(8, 6), 84, PtxTier::Sm80, Route::None),
        // Ada's route was measured with the sm80 tier only
        (ComputeCapability::new(8, 9), 34, PtxTier::Sm75, Route::None),
        (ComputeCapability::new(8, 9), 34, PtxTier::Sm80, Route::Ada),
        (ComputeCapability::new(8, 9), 46, PtxTier::Sm80, Route::None),
        (
            ComputeCapability::new(12, 0),
            36,
            PtxTier::Sm120,
            Route::None,
        ),
    ];
    // a build without the device's tier, such as `cuda-rtx50` on Turing, has no such route
    let devices = devices
        .into_iter()
        .filter(|(_, _, limit, _)| limit.is_compiled_in());
    let wide = Some(WideconvAlgorithm::Fp16(WideconvFp16Tiles::Wide));
    let narrow = Some(WideconvAlgorithm::Fp16(WideconvFp16Tiles::Narrow));
    let winograd = Some(WideconvAlgorithm::Winograd(WideconvProducts::Fp32));
    // per layer, batch and math: the Turing and Ada wideconv algorithms; `None` keeps the
    // direct ResNet kernel for the 32- and 64-channel input layers and any non-FP16
    // wideconv kernel for the wider ones. FP32 mode never takes FP16 tiles, and only
    // Turing takes them on the stride-2 layers
    let cases = [
        ("resnet.layer1.0.conv1", 1, CudaMath::Tf32, wide, wide),
        ("resnet.layer1.2.conv2", 32, CudaMath::Tf32, wide, wide),
        ("resnet.layer1.0.conv1", 32, CudaMath::Fp32, None, None),
        ("resnet.layer2.0.conv2", 32, CudaMath::Tf32, wide, wide),
        ("resnet.layer2.3.conv2", 1, CudaMath::Fp32, winograd, None),
        ("resnet.layer3.1.conv1", 1, CudaMath::Tf32, narrow, None),
        ("resnet.layer3.1.conv1", 32, CudaMath::Tf32, wide, wide),
        ("resnet.layer4.2.conv2", 1, CudaMath::Tf32, narrow, None),
        ("resnet.layer4.2.conv2", 7, CudaMath::Tf32, wide, None),
        ("resnet.layer4.2.conv2", 8, CudaMath::Tf32, wide, wide),
        ("resnet.layer4.2.conv2", 32, CudaMath::Fp32, None, None),
        ("resnet.layer2.0.conv1", 1, CudaMath::Tf32, wide, None),
        ("resnet.layer2.0.conv1", 32, CudaMath::Tf32, wide, None),
        ("resnet.layer2.0.conv1", 32, CudaMath::Fp32, None, None),
        ("resnet.layer3.0.conv1", 1, CudaMath::Tf32, narrow, None),
        ("resnet.layer3.0.conv1", 32, CudaMath::Tf32, wide, None),
        ("resnet.layer4.0.conv1", 1, CudaMath::Tf32, narrow, None),
        ("resnet.layer4.0.conv1", 4, CudaMath::Tf32, wide, None),
        ("resnet.layer4.0.conv1", 32, CudaMath::Fp32, None, None),
    ];
    for (capability, sms, limit, route) in devices {
        let mut fixture = Fixture::new();
        fixture.device = Builder::new(capability)
            .multiprocessors(sms)
            .shared_optin_bytes(64 << 10)
            .name(if route == Route::Ada {
                "NVIDIA GeForce RTX 4060 Ti"
            } else {
                "unmeasured GPU"
            })
            .build();
        fixture.limit = limit;
        for (name, batch, math, turing, ada) in cases {
            let Selected::Oxide(token) = PlanRequest::DriverOnly
                .resolve(BoundaryId::named(name), batch, math, &mut fixture)
                .unwrap()
            else {
                panic!("{capability:?} {name}: no driver route")
            };
            let context = format!("{capability:?} {limit:?} {name} b{batch} {math:?}");
            let expected = match route {
                Route::Turing => turing,
                Route::Ada => ada,
                Route::None => None,
            };
            let early = name.starts_with("resnet.layer1.") || name.starts_with("resnet.layer2.");
            if expected.is_none() && early {
                assert_eq!(token.area(), KernelModule::Resnet, "{context}");
                continue;
            }

            let PlanPin::Pinned(ConfigPin::Wideconv(WideconvPin::Configured(config))) = token.pin
            else {
                panic!("{context}: unexpected pin {:?}", token.pin)
            };
            match expected {
                Some(algorithm) => assert_eq!(config.algorithm, algorithm, "{context}"),
                None => assert!(
                    !matches!(config.algorithm, WideconvAlgorithm::Fp16(_)),
                    "{context}: {:?}",
                    config.algorithm
                ),
            }
        }
    }
}

#[test]
#[cfg(feature = "_cuda-libraries")]
fn measured_ada_sinc_recipe_matches_driver_pin_and_keeps_blackwell_binding() {
    use super::super::policy::Recipe;
    let mut fixture = Fixture::new();
    fixture.device = Builder::new(ComputeCapability::new(8, 9))
        .multiprocessors(34)
        .name("NVIDIA GeForce RTX 4060 Ti")
        .build();
    let boundary = BoundaryId::named("sincnet.conv0.abs_pool");
    for batch in [1, 32] {
        let Selected::Oxide(hybrid) = PlanRequest::Hybrid
            .resolve(boundary, batch, CudaMath::Fp32, &mut fixture)
            .unwrap()
        else {
            panic!("measured Ada recipe")
        };
        let Selected::Oxide(driver) = PlanRequest::DriverOnly
            .resolve(boundary, batch, CudaMath::Fp32, &mut fixture)
            .unwrap()
        else {
            panic!("driver recipe")
        };
        assert_eq!(hybrid.pin, driver.pin);
        assert_eq!(hybrid.target, driver.target);
        assert_eq!(
            hybrid.evidence,
            TokenEvidence::Recipe(Recipe::Rtx4060TiSinc)
        );
    }
    fixture.device = Builder::new(ComputeCapability::new(12, 0)).build();
    let Selected::Oxide(token) = PlanRequest::Hybrid
        .resolve(boundary, 32, CudaMath::Fp32, &mut fixture)
        .unwrap()
    else {
        panic!("implemented SincNet default")
    };
    assert_eq!(token.evidence, TokenEvidence::Implemented);
}

#[test]
#[cfg(feature = "_cuda-libraries")]
fn measured_a100_recipe_uses_class_defaults_for_every_pipeline_boundary() {
    use super::super::policy::{Recipe, RecipeMode};
    for (name, recipe) in [
        ("NVIDIA A100-PCIE-40GB", Recipe::A100Pcie),
        ("NVIDIA A100-SXM4-40GB", Recipe::A100Sxm4),
    ] {
        let mut fixture = Fixture::new();
        fixture.device = Builder::new(ComputeCapability::new(8, 0))
            .multiprocessors(108)
            .name(name)
            .build();
        fixture.recipe_mode = RecipeMode::Fp32SegmentationTf32Embedding;
        for boundary in BoundaryId::all().filter(|id| id.name() != "lstm.stack.input_proj") {
            let math = match boundary.area() {
                KernelModule::Resnet | KernelModule::Embedding => CudaMath::Tf32,
                _ => CudaMath::Fp32,
            };
            for batch in [1, 32] {
                let Selected::Oxide(hybrid) = PlanRequest::Hybrid
                    .resolve(boundary, batch, math, &mut fixture)
                    .unwrap()
                else {
                    panic!("{name} {boundary} b{batch}")
                };
                let Selected::Oxide(driver) = PlanRequest::DriverOnly
                    .resolve(boundary, batch, math, &mut fixture)
                    .unwrap()
                else {
                    panic!("driver boundary")
                };
                assert_eq!(hybrid.pin, driver.pin, "{name} {boundary} b{batch}");
                if let Some(fp16) = recipe.fp16_pin(boundary, batch, math) {
                    assert_eq!(driver.pin, super::super::PlanPin::Pinned(fp16));
                }
                assert_eq!(hybrid.target, driver.target);
                assert_eq!(hybrid.evidence, TokenEvidence::Recipe(recipe));
            }
        }
        fixture.recipe_mode = RecipeMode::Disabled;
        for boundary in [
            "sincnet.conv0.abs_pool",
            "lstm.stack",
            "resnet.layer3.1.conv1",
        ] {
            let boundary = BoundaryId::named(boundary);
            assert!(matches!(
                PlanRequest::Hybrid
                    .resolve(boundary, 32, CudaMath::Fp32, &mut fixture)
                    .unwrap(),
                Selected::Oxide(_)
            ));
        }
    }
}

#[test]
#[cfg(feature = "_cuda-libraries")]
fn tune_file_wins_over_recipe_and_class_default_before_artifact_loading() {
    use crate::inference::cuda::tuning::ApprovedChoice;
    let mut fixture = Fixture::new();
    fixture.device = Builder::new(ComputeCapability::new(8, 9))
        .multiprocessors(34)
        .name("NVIDIA GeForce RTX 4060 Ti")
        .build();
    fixture.tuned = Some(ApprovedChoice::Library);
    let sinc = BoundaryId::named("sincnet.conv0.abs_pool");
    assert!(matches!(
        PlanRequest::Hybrid
            .resolve(sinc, 32, CudaMath::Fp32, &mut fixture)
            .unwrap(),
        Selected::Library
    ));
    assert!(fixture.loads.is_empty());
    fixture.device = Builder::new(ComputeCapability::new(9, 0))
        .multiprocessors(132)
        .name("NVIDIA H100")
        .build();
    let conv = BoundaryId::named("resnet.layer1.0.conv1");
    assert!(matches!(
        PlanRequest::Hybrid
            .resolve(conv, 32, CudaMath::Tf32, &mut fixture)
            .unwrap(),
        Selected::Library
    ));
    assert!(fixture.loads.is_empty());
    fixture.tuned = None;
    let Selected::Oxide(token) = PlanRequest::Hybrid
        .resolve(conv, 32, CudaMath::Tf32, &mut fixture)
        .unwrap()
    else {
        panic!("class default")
    };
    assert_eq!(token.source(), super::super::policy::Source::Default);
}

#[test]
fn validated_tune_pin_wins_and_cannot_be_reused_for_another_batch() {
    let mut fixture = Fixture::new();
    fixture.device = Builder::new(ComputeCapability::new(8, 9))
        .multiprocessors(34)
        .name("NVIDIA GeForce RTX 4060 Ti")
        .build();
    let boundary = BoundaryId::named("resnet.layer1.0.conv1");
    let approved = crate::inference::cuda::tuning::tests::approved_choice(
        &fixture.device,
        boundary,
        32,
        CudaMath::Tf32,
        fixture.limit,
    );
    let crate::inference::cuda::tuning::ApprovedChoice::Kernel(config) = &approved else {
        panic!("approved kernel")
    };
    let pin = config.pin();
    fixture.tuned = Some(approved);
    let Selected::Oxide(token) = PlanRequest::Hybrid
        .resolve(boundary, 32, CudaMath::Tf32, &mut fixture)
        .unwrap()
    else {
        panic!("tuned kernel")
    };
    assert_eq!(token.pin, PlanPin::Pinned(pin));
    assert_eq!(token.source(), super::super::policy::Source::TuneFile);
    assert!(matches!(token.evidence, TokenEvidence::Tuned { .. }));
    assert!(
        PlanRequest::Hybrid
            .resolve(boundary, 1, CudaMath::Tf32, &mut fixture)
            .is_err()
    );
}

#[test]
fn forced_jit_loads_ptx_for_a_tuned_cubin_without_its_speed_evidence() {
    use crate::inference::cuda::kernels::LoadedArtifact;

    let mut fixture = Fixture::new();
    fixture.device = Builder::new(ComputeCapability::new(8, 9))
        .multiprocessors(34)
        .name("NVIDIA GeForce RTX 4060 Ti")
        .build();
    let boundary = BoundaryId::named("resnet.layer1.0.conv1");
    let approved = crate::inference::cuda::tuning::tests::approved_choice(
        &fixture.device,
        boundary,
        32,
        CudaMath::Tf32,
        fixture.limit,
    );
    let crate::inference::cuda::tuning::ApprovedChoice::Kernel(config) = &approved else {
        panic!("approved kernel")
    };
    let saved = config.module();
    let pin = config.pin();
    assert!(matches!(saved.artifact(), LoadedArtifact::Cubin { .. }));
    fixture.tuned = Some(approved);
    fixture.force_jit = true;

    let Selected::Oxide(token) = PlanRequest::Hybrid
        .resolve(boundary, 32, CudaMath::Tf32, &mut fixture)
        .unwrap()
    else {
        panic!("tuned kernel")
    };
    let jit = saved.diagnostic_request(true).unwrap();
    assert!(matches!(jit.artifact(), LoadedArtifact::PtxJit { .. }));
    assert_eq!(token.target.module, jit);
    assert_eq!(fixture.loads, [jit]);
    // the tuned configuration stays, but the cubin timing no longer backs it
    assert_eq!(token.pin, PlanPin::Pinned(pin));
    assert_eq!(token.evidence, TokenEvidence::Implemented);
}

#[test]
#[cfg(feature = "_cuda-libraries")]
fn tuner_cannot_time_an_artifact_fallback_as_the_selected_kernel() {
    use crate::inference::cuda::tuning::tests::approved_choice;

    let mut fixture = Fixture::new();
    let boundary = BoundaryId::named("resnet.layer1.0.conv1");
    fixture.tuned = Some(approved_choice(
        &fixture.device,
        boundary,
        32,
        CudaMath::Tf32,
        fixture.limit,
    ));
    fixture.refuse_load = true;
    assert!(matches!(
        PlanRequest::Hybrid.resolve(boundary, 32, CudaMath::Tf32, &mut fixture),
        Err(CudaError::ArtifactUnavailable { .. })
    ));

    fixture.benchmarking = true;
    assert!(matches!(
        PlanRequest::Hybrid.resolve(boundary, 32, CudaMath::Tf32, &mut fixture),
        Err(CudaError::ArtifactUnavailable { .. })
    ));
}

#[test]
#[cfg(feature = "_cuda-libraries")]
fn a100_lower_tier_cannot_claim_the_measured_whole_plan() {
    use super::super::policy::RecipeMode;
    for name in ["NVIDIA A100-PCIE-40GB", "NVIDIA A100-SXM4-40GB"] {
        let mut fixture = Fixture::new();
        fixture.device = Builder::new(ComputeCapability::new(8, 0))
            .multiprocessors(108)
            .name(name)
            .build();
        fixture.recipe_mode = RecipeMode::Fp32SegmentationTf32Embedding;
        fixture.limit = PtxTier::Sm75;
        for (name, math) in [
            ("resnet.layer1.0.conv1", CudaMath::Tf32),
            ("sincnet.conv0.abs_pool", CudaMath::Fp32),
            ("lstm.stack", CudaMath::Fp32),
        ] {
            let boundary = BoundaryId::named(name);
            assert!(matches!(
                PlanRequest::Hybrid
                    .resolve(boundary, 32, math, &mut fixture)
                    .unwrap(),
                Selected::Oxide(_)
            ));
        }
    }
}

#[test]
#[cfg(feature = "_cuda-libraries")]
fn tuner_plan_refusal_after_loading_cannot_be_a_library_timing() {
    use crate::inference::cuda::tuning::tests::approved_choice;
    let mut fixture = Fixture::new();
    let boundary = BoundaryId::named("resnet.layer1.0.conv1");
    fixture.tuned = Some(approved_choice(
        &fixture.device,
        boundary,
        32,
        CudaMath::Tf32,
        fixture.limit,
    ));
    fixture.benchmarking = true;
    let Selected::Oxide(token) = PlanRequest::Hybrid
        .resolve(boundary, 32, CudaMath::Tf32, &mut fixture)
        .unwrap()
    else {
        panic!("approved benchmark token")
    };
    assert_eq!(fixture.loads.len(), 1);
    assert!(matches!(
        token.finish::<()>(
            KernelModule::Resnet,
            false,
            Err(PlanError::DeviceUnsupported {
                reason: "test plan refusal".into(),
            })
        ),
        Err(CudaError::CandidateDeviceUnsupported { .. })
    ));
}

#[test]
fn driver_4060_fp16_pins_do_not_inherit_non_fp16_speed() {
    use crate::inference::cuda::candidate::{WideconvAlgorithm, WideconvFp16Tiles, WideconvPin};

    if !PtxTier::Sm80.is_compiled_in() {
        return;
    }

    let mut fixture = Fixture::new();
    fixture.device = Builder::new(ComputeCapability::new(8, 9))
        .multiprocessors(34)
        .name("NVIDIA GeForce RTX 4060 Ti")
        .build();
    fixture.limit = PtxTier::Sm80;
    for (name, batches) in [
        ("resnet.layer1.0.conv1", &[1, 32][..]),
        ("resnet.layer2.1.conv2", &[1, 32][..]),
        ("resnet.layer3.1.conv1", &[32][..]),
        ("resnet.layer4.2.conv2", &[32][..]),
    ] {
        let boundary = BoundaryId::named(name);
        for &batch in batches {
            let Selected::Oxide(token) = PlanRequest::DriverOnly
                .resolve(boundary, batch, CudaMath::Tf32, &mut fixture)
                .unwrap()
            else {
                panic!("{boundary} b{batch}: no driver route")
            };
            let PlanPin::Pinned(ConfigPin::Wideconv(WideconvPin::Configured(config))) = token.pin
            else {
                panic!("{boundary} b{batch}: unexpected pin {:?}", token.pin)
            };
            assert_eq!(
                config.algorithm,
                WideconvAlgorithm::Fp16(WideconvFp16Tiles::Wide),
                "{boundary} b{batch}"
            );
            assert_eq!(
                token.evidence,
                TokenEvidence::Implemented,
                "{boundary} b{batch}"
            );
            assert!(!token.speed_measured(), "{boundary} b{batch}");
            assert_eq!(
                token.source(),
                super::super::policy::Source::Default,
                "{boundary} b{batch}"
            );
        }
    }

    fixture.fp16 = Fp16Policy::Excluded;
    let Selected::Oxide(token) = PlanRequest::DriverOnly
        .resolve(
            BoundaryId::named("resnet.layer3.1.conv1"),
            32,
            CudaMath::Tf32,
            &mut fixture,
        )
        .unwrap()
    else {
        panic!("non-FP16 reference route")
    };
    assert!(matches!(token.pin, PlanPin::Pinned(pin) if !pin.is_fp16()));
    assert!(matches!(token.evidence, TokenEvidence::Port { .. }));
    assert!(token.speed_measured());
    assert_eq!(token.source(), super::super::policy::Source::Recipe);
}

#[test]
#[cfg(feature = "_cuda-libraries")]
fn disabled_4060_recipe_uses_non_fp16_coverage_and_pins() {
    use crate::inference::cuda::candidate::WideconvOxide;

    let mut fixture = Fixture::new();
    fixture.device = Builder::new(ComputeCapability::new(8, 9))
        .multiprocessors(34)
        .name("NVIDIA GeForce RTX 4060 Ti")
        .build();
    fixture.limit = PtxTier::Sm80;
    for name in [
        "resnet.layer1.0.conv1",
        "resnet.layer2.1.conv2",
        "resnet.layer3.1.conv1",
        "resnet.layer4.2.conv2",
    ] {
        let boundary = BoundaryId::named(name);
        for batch in [1, 32] {
            fixture.fp16 = Fp16Policy::Excluded;
            let expected = PlanRequest::Hybrid
                .resolve(boundary, batch, CudaMath::Tf32, &mut fixture)
                .unwrap();
            fixture.fp16 = Fp16Policy::Allowed;
            let Selected::Oxide(actual) = PlanRequest::Hybrid
                .resolve(boundary, batch, CudaMath::Tf32, &mut fixture)
                .unwrap()
            else {
                panic!("{boundary} b{batch}: non-FP16 route")
            };
            let Selected::Oxide(expected) = expected else {
                panic!("{boundary} b{batch}: excluded reference route")
            };
            assert_eq!(actual.pin, expected.pin, "{boundary} b{batch}");
            assert_eq!(actual.evidence, expected.evidence, "{boundary} b{batch}");
            assert!(
                !is_fp16(Some((actual.area(), actual.pin))),
                "{boundary} b{batch}"
            );

            if name.starts_with("resnet.layer1.") || name.starts_with("resnet.layer2.") {
                // no wideconv coverage means no FP16 port token or speed claim;
                // the measured non-FP16 ResNet route owns these boundaries instead
                assert_eq!(actual.area(), KernelModule::Resnet);
                assert!(
                    select_from(
                        &[Area::candidate::<WideconvOxide>()],
                        boundary,
                        batch,
                        CudaMath::Tf32,
                        &mut &mut fixture,
                        super::Selection::Production,
                    )
                    .unwrap()
                    .is_none(),
                    "{boundary} b{batch}: no FP16 port speed claim"
                );
            }

            let Selected::Oxide(driver) = PlanRequest::DriverOnly
                .resolve(boundary, batch, CudaMath::Tf32, &mut fixture)
                .unwrap()
            else {
                panic!("driver route")
            };
            if batch == 32
                || name.starts_with("resnet.layer1.")
                || name.starts_with("resnet.layer2.")
            {
                assert!(
                    is_fp16(Some((driver.area(), driver.pin))),
                    "{boundary} b{batch}"
                );
            }
        }
    }
}

#[test]
#[cfg(feature = "_cuda-libraries")]
fn measured_rtx_recipes_use_fp16_only_at_the_4060_ti_point() {
    use super::super::policy::RecipeMode;
    use crate::inference::cuda::candidate::{
        ConvKernel, ConvPin, WideconvAlgorithm, WideconvFp16Tiles, WideconvPin, WideconvProducts,
    };
    for (cc, sms, name) in [
        (
            ComputeCapability::new(8, 9),
            34,
            "NVIDIA GeForce RTX 4060 Ti",
        ),
        (
            ComputeCapability::new(12, 0),
            36,
            "NVIDIA GeForce RTX 5060 Ti",
        ),
    ] {
        let mut fixture = Fixture::new();
        fixture.device = Builder::new(cc).multiprocessors(sms).name(name).build();
        fixture.recipe_mode = RecipeMode::Fp32SegmentationTf32Embedding;
        for batch in [1, 4, 8, 16, 32] {
            let Selected::Oxide(wide) = PlanRequest::Hybrid
                .resolve(
                    BoundaryId::named("resnet.layer3.1.conv1"),
                    batch,
                    CudaMath::Tf32,
                    &mut fixture,
                )
                .unwrap()
            else {
                panic!("measured wide trunk")
            };
            let PlanPin::Pinned(ConfigPin::Wideconv(WideconvPin::Configured(config))) = wide.pin
            else {
                panic!("fixed Winograd configuration")
            };
            assert_eq!(
                config.algorithm,
                if cc == ComputeCapability::new(8, 9) && batch >= 8 {
                    WideconvAlgorithm::Fp16(WideconvFp16Tiles::Wide)
                } else {
                    WideconvAlgorithm::Winograd(WideconvProducts::Tf32x1Staged)
                }
            );
            let Selected::Oxide(c32) = PlanRequest::Hybrid
                .resolve(
                    BoundaryId::named("resnet.layer1.0.conv2"),
                    batch,
                    CudaMath::Tf32,
                    &mut fixture,
                )
                .unwrap()
            else {
                panic!("measured C32 trunk")
            };
            if cc == ComputeCapability::new(8, 9) {
                let PlanPin::Pinned(ConfigPin::Wideconv(WideconvPin::Configured(config))) = c32.pin
                else {
                    panic!("FP16 early trunk")
                };
                assert_eq!(
                    config.algorithm,
                    WideconvAlgorithm::Fp16(WideconvFp16Tiles::Wide)
                );
            } else {
                assert_eq!(
                    c32.pin,
                    PlanPin::Pinned(ConfigPin::Conv(ConvPin::Kernel(ConvKernel::C32Tensor)))
                );
            }
        }
    }
}

/// The 4090 recipe reaches the 32->64 stride-2 layer through its own coverage, keeps the
/// TF32 rule below each crossover batch, and resolves alike with and without libraries
#[test]
#[cfg(feature = "_cuda-libraries")]
fn measured_rtx4090_recipe_routes_fp16_from_each_crossover_batch() {
    use super::super::policy::{Recipe, RecipeMode};
    use crate::inference::cuda::candidate::{WideconvAlgorithm, WideconvFp16Tiles, WideconvPin};
    let rtx4090 = |sms| {
        Builder::new(ComputeCapability::new(8, 9))
            .multiprocessors(sms)
            .shared_optin_bytes(101376)
            .name("NVIDIA GeForce RTX 4090")
            .build()
    };
    let mut fixture = Fixture::new();
    fixture.device = rtx4090(128);
    fixture.recipe_mode = RecipeMode::Fp32SegmentationTf32Embedding;
    let fp16 = |name: &str, batch, fixture: &mut Fixture| {
        let boundary = BoundaryId::named(name);
        let Selected::Oxide(hybrid) = PlanRequest::Hybrid
            .resolve(boundary, batch, CudaMath::Tf32, &mut *fixture)
            .unwrap()
        else {
            panic!("{name} b{batch}: measured trunk")
        };
        let Selected::Oxide(driver) = PlanRequest::DriverOnly
            .resolve(boundary, batch, CudaMath::Tf32, fixture)
            .unwrap()
        else {
            panic!("{name} b{batch}: driver trunk")
        };
        assert_eq!(hybrid.pin, driver.pin, "{name} b{batch}");
        match hybrid.pin {
            PlanPin::Pinned(ConfigPin::Wideconv(WideconvPin::Configured(config))) => {
                match config.algorithm {
                    WideconvAlgorithm::Fp16(tiles) => Some(tiles),
                    _ => None,
                }
            }
            _ => None,
        }
    };

    for batch in [1, 4, 8, 16, 32] {
        for name in [
            "resnet.layer1.0.conv1",
            "resnet.layer2.0.conv1",
            "resnet.layer2.1.conv2",
        ] {
            assert_eq!(
                fp16(name, batch, &mut fixture),
                Some(WideconvFp16Tiles::Wide),
                "{name} b{batch}"
            );
        }
    }
    assert_eq!(fp16("resnet.layer3.1.conv1", 1, &mut fixture), None);
    assert_eq!(
        fp16("resnet.layer3.1.conv1", 4, &mut fixture),
        Some(WideconvFp16Tiles::Narrow)
    );
    assert_eq!(fp16("resnet.layer4.1.conv1", 4, &mut fixture), None);
    assert_eq!(
        fp16("resnet.layer4.1.conv1", 8, &mut fixture),
        Some(WideconvFp16Tiles::Narrow)
    );
    assert_eq!(
        fp16("resnet.layer4.0.conv1", 32, &mut fixture),
        Some(WideconvFp16Tiles::Wide)
    );
    assert_eq!(fp16("resnet.conv1", 32, &mut fixture), None);

    // another SM count is not the measured point
    fixture.device = rtx4090(126);
    assert_eq!(Recipe::fp16_device(&fixture.device, fixture.limit), None);
    assert_eq!(fp16("resnet.layer2.0.conv1", 32, &mut fixture), None);
}

#[test]
#[cfg(all(feature = "_cuda-libraries", feature = "cuda-sm75"))]
fn measured_t4_recipe_contains_only_driver_kernel_choices() {
    use super::super::policy::{Recipe, RecipeMode};
    let mut fixture = Fixture::new();
    fixture.device = Builder::new(ComputeCapability::new(7, 5))
        .multiprocessors(40)
        .shared_optin_bytes(64 << 10)
        .name("Tesla T4")
        .build();
    fixture.limit = PtxTier::Sm75;
    fixture.recipe_mode = RecipeMode::Fp32SegmentationTf32Embedding;
    let mut counts = (0, 0);
    for boundary in BoundaryId::all().filter(|id| id.name() != "lstm.stack.input_proj") {
        let math = match boundary.area() {
            KernelModule::Resnet | KernelModule::Embedding => CudaMath::Tf32,
            _ => CudaMath::Fp32,
        };
        for batch in boundary.batches().iter() {
            let before = fixture.loads.len();
            let hybrid = PlanRequest::Hybrid
                .resolve(boundary, batch, math, &mut fixture)
                .unwrap();
            if matches!(hybrid, Selected::Library) {
                assert_eq!(fixture.loads.len(), before);
            }
            let Selected::Oxide(driver) = PlanRequest::DriverOnly
                .resolve(boundary, batch, math, &mut fixture)
                .unwrap()
            else {
                panic!("driver {boundary} b{batch}")
            };
            match hybrid {
                Selected::Library => counts.1 += 1,
                Selected::Oxide(hybrid) => {
                    counts.0 += 1;
                    if boundary == BoundaryId::named("resnet.conv1") && matches!(batch, 16 | 32) {
                        use crate::inference::cuda::candidate::{
                            WideconvAlgorithm, WideconvConfig, WideconvPartition, WideconvPin,
                            WideconvSplitCells,
                        };
                        assert_eq!(
                            hybrid.pin,
                            PlanPin::Pinned(ConfigPin::Wideconv(WideconvPin::Configured(
                                WideconvConfig {
                                    algorithm: WideconvAlgorithm::Spatial,
                                    partition: WideconvPartition::Whole,
                                    split_cells: WideconvSplitCells::All,
                                }
                            )))
                        );
                    } else {
                        assert_eq!(hybrid.pin, driver.pin, "{boundary} b{batch}");
                    }
                    assert_eq!(hybrid.target, driver.target);
                    assert_eq!(hybrid.evidence, TokenEvidence::Recipe(Recipe::TeslaT4));
                }
            }
        }
    }
    assert_eq!(counts, (228, 0));

    // an exact user tune choice still has priority over a kernel recipe
    let boundary = BoundaryId::named("resnet.conv1");
    fixture.tuned = Some(crate::inference::cuda::tuning::tests::approved_choice(
        &fixture.device,
        boundary,
        32,
        CudaMath::Tf32,
        fixture.limit,
    ));
    let Selected::Oxide(tuned) = PlanRequest::Hybrid
        .resolve(boundary, 32, CudaMath::Tf32, &mut fixture)
        .unwrap()
    else {
        panic!("tune file overrides kernel recipe")
    };
    assert_eq!(tuned.source(), super::super::policy::Source::TuneFile);
}

/// The pin and area a request selects, or `None` for Library
#[cfg(feature = "_cuda-libraries")]
fn route(
    request: PlanRequest,
    fixture: &mut Fixture,
    name: &str,
    batch: usize,
) -> Option<(KernelModule, PlanPin)> {
    match request
        .resolve(BoundaryId::named(name), batch, CudaMath::Tf32, fixture)
        .unwrap()
    {
        Selected::Oxide(token) => Some((token.area(), token.pin)),
        Selected::Library => None,
    }
}

#[cfg(feature = "_cuda-libraries")]
fn is_fp16(route: Option<(KernelModule, PlanPin)>) -> bool {
    matches!(route, Some((_, PlanPin::Pinned(pin))) if pin.is_fp16())
}

#[test]
#[cfg(feature = "_cuda-libraries")]
fn excluded_fp16_takes_each_source_choice_without_fp16_tiles() {
    use super::super::policy::RecipeMode;
    use crate::inference::cuda::candidate::{WideconvAlgorithm, WideconvPin, WideconvProducts};

    let algorithm = |route: Option<(KernelModule, PlanPin)>| match route {
        Some((_, PlanPin::Pinned(ConfigPin::Wideconv(WideconvPin::Configured(config))))) => {
            Some(config.algorithm)
        }
        _ => None,
    };
    let ffma = Some(WideconvAlgorithm::Winograd(WideconvProducts::Fp32));
    // per layer and batch: the Turing and 4060 Ti choices without FP16 tiles, where
    // `None` is the direct ResNet kernel
    let cases = [
        ("resnet.layer1.0.conv1", 1, None, None),
        ("resnet.layer2.1.conv2", 32, ffma, None),
        (
            "resnet.layer3.1.conv1",
            32,
            ffma,
            Some(WideconvAlgorithm::Winograd(WideconvProducts::Tf32x1Staged)),
        ),
    ];
    let devices = [
        (ComputeCapability::new(7, 5), 40, "Tesla T4", PtxTier::Sm75),
        (
            ComputeCapability::new(8, 9),
            34,
            "NVIDIA GeForce RTX 4060 Ti",
            PtxTier::Sm80,
        ),
    ];
    for (capability, sms, name, limit) in devices {
        if !limit.is_compiled_in() {
            continue;
        }
        let mut fixture = Fixture::new();
        fixture.device = Builder::new(capability)
            .multiprocessors(sms)
            .shared_optin_bytes(64 << 10)
            .name(name)
            .build();
        fixture.limit = limit;
        fixture.recipe_mode = RecipeMode::Fp32SegmentationTf32Embedding;
        let turing = capability == ComputeCapability::new(7, 5);
        for request in [PlanRequest::Hybrid, PlanRequest::DriverOnly] {
            for (layer, batch, turing_choice, ada_choice) in cases {
                let context = format!("{name} {request:?} {layer} b{batch}");
                fixture.fp16 = Fp16Policy::Allowed;
                let allowed = route(request, &mut fixture, layer, batch);
                assert!(is_fp16(allowed), "{context}: the recipe picks FP16 tiles");

                fixture.fp16 = Fp16Policy::Excluded;
                let excluded = route(request, &mut fixture, layer, batch);
                let expected = if turing { turing_choice } else { ada_choice };
                match expected {
                    Some(choice) => assert_eq!(algorithm(excluded), Some(choice), "{context}"),
                    None => assert_eq!(
                        excluded.map(|(area, _)| area),
                        Some(KernelModule::Resnet),
                        "{context}"
                    ),
                }
            }
        }

        // no tuple of any boundary keeps an FP16 pin once excluded, in either build
        fixture.fp16 = Fp16Policy::Excluded;
        for boundary in BoundaryId::all().filter(|id| id.name() != "lstm.stack.input_proj") {
            for batch in boundary.batches().iter() {
                for request in [PlanRequest::Hybrid, PlanRequest::DriverOnly] {
                    let selected = route(request, &mut fixture, boundary.name(), batch);
                    assert!(!is_fp16(selected), "{name} {request:?} {boundary} b{batch}");
                }
            }
        }

        // an FP16 tune choice gives way to the selection made without it
        let boundary = BoundaryId::named("resnet.layer2.1.conv2");
        let tuned = crate::inference::cuda::tuning::tests::approved_fp16_choice(
            &fixture.device,
            boundary,
            32,
            CudaMath::Tf32,
            fixture.limit,
        );
        assert!(tuned.is_fp16(), "{name}: the approved choice is FP16");
        fixture.tuned = Some(tuned);
        fixture.fp16 = Fp16Policy::Allowed;
        let Selected::Oxide(token) = PlanRequest::Hybrid
            .resolve(boundary, 32, CudaMath::Tf32, &mut fixture)
            .unwrap()
        else {
            panic!("{name}: tune choice")
        };
        assert_eq!(token.source(), super::super::policy::Source::TuneFile);
        fixture.fp16 = Fp16Policy::Excluded;
        let Selected::Oxide(token) = PlanRequest::Hybrid
            .resolve(boundary, 32, CudaMath::Tf32, &mut fixture)
            .unwrap()
        else {
            panic!("{name}: an excluded tune choice keeps the recipe's kernel")
        };
        assert!(!is_fp16(Some((token.area(), token.pin))), "{name}");
        assert_eq!(
            token.source(),
            super::super::policy::Source::Recipe,
            "{name}"
        );
        fixture.benchmarking = true;
        let refusal = PlanRequest::Hybrid
            .resolve(boundary, 32, CudaMath::Tf32, &mut fixture)
            .unwrap_err();
        assert!(
            refusal
                .to_string()
                .contains("FP16 tuning candidate is excluded by the layer weights"),
            "{name}: {refusal}"
        );
    }
}

#[test]
#[cfg(all(feature = "_cuda-libraries", feature = "cuda-sm75"))]
fn excluded_fp16_drops_the_turing_class_default() {
    let mut fixture = Fixture::new();
    fixture.device = Builder::new(ComputeCapability::new(7, 5))
        .multiprocessors(40)
        .shared_optin_bytes(64 << 10)
        .name("unmeasured Turing GPU")
        .build();
    fixture.limit = PtxTier::Sm75;
    let boundary = BoundaryId::named("resnet.layer2.1.conv2");
    let Selected::Oxide(token) = PlanRequest::Hybrid
        .resolve(boundary, 32, CudaMath::Tf32, &mut fixture)
        .unwrap()
    else {
        panic!("the cc 7.5 class default routes FP16 tiles")
    };
    assert!(matches!(
        token.evidence,
        TokenEvidence::DeviceDefault(super::super::policy::DeviceDefault::TuringFp16Trunk)
    ));
    assert!(is_fp16(Some((token.area(), token.pin))));

    // excluded FP16 tiles use the implemented FP32 kernel instead
    fixture.fp16 = Fp16Policy::Excluded;
    let excluded = PlanRequest::Hybrid
        .resolve(boundary, 32, CudaMath::Tf32, &mut fixture)
        .unwrap();
    let Selected::Oxide(token) = excluded else {
        panic!("implemented FP32 default")
    };
    assert_eq!(token.evidence, TokenEvidence::Implemented);
    assert!(!is_fp16(Some((token.area(), token.pin))));
}

#[test]
#[cfg(feature = "_cuda-libraries")]
fn unknown_device_uses_implemented_kernel_without_a_speed_claim() {
    let mut fixture = Fixture::new();
    let selected = PlanRequest::Hybrid
        .resolve_with_candidates(
            BoundaryId::named("sincnet.conv0.abs_pool"),
            1,
            CudaMath::Fp32,
            &mut fixture,
            &[Area::candidate::<Stub>()],
        )
        .unwrap();
    let Selected::Oxide(token) = selected else {
        panic!("unknown device kernel")
    };
    assert_eq!(
        token.pin,
        PlanPin::Pinned(ConfigPin::Sinc(SincPin::ConvAbsPool))
    );
    assert_eq!(token.evidence, TokenEvidence::Implemented);
    assert!(!token.speed_measured());
    assert_eq!(fixture.loads, [token.target.module]);
}

#[test]
#[cfg(feature = "_cuda-libraries")]
fn coverage_gap_uses_library_without_loading_a_kernel() {
    let mut fixture = Fixture::new();
    fixture.device = Builder::new(ComputeCapability::new(9, 0))
        .name("unknown GPU")
        .build();
    let selected = PlanRequest::Hybrid
        .resolve_with_candidates(
            BoundaryId::named("sincnet.conv0.abs_pool"),
            32,
            CudaMath::Fp32,
            &mut fixture,
            &[Area::candidate::<Stub>()],
        )
        .unwrap();
    assert!(matches!(selected, Selected::Library));
    assert!(fixture.loads.is_empty());
}

#[test]
#[cfg(feature = "_cuda-libraries")]
fn kernel_load_failure_is_not_a_capability_fallback() {
    let mut fixture = Fixture::new();
    fixture.refuse_load = true;
    let selected = PlanRequest::Hybrid.resolve_with_candidates(
        BoundaryId::named("sincnet.conv0.abs_pool"),
        1,
        CudaMath::Fp32,
        &mut fixture,
        &[Area::candidate::<Stub>()],
    );
    assert!(matches!(
        selected,
        Err(CudaError::ArtifactUnavailable { .. })
    ));
    assert_eq!(fixture.loads.len(), 1);
}

#[test]
#[cfg(feature = "_cuda-libraries")]
fn measured_hybrid_and_driver_select_the_same_kernel_for_every_covered_tuple() {
    use super::super::policy::RecipeMode;
    for (cc, sms, name, tier) in [
        (
            ComputeCapability::new(8, 9),
            34,
            "NVIDIA GeForce RTX 4060 Ti",
            PtxTier::Sm80,
        ),
        (
            ComputeCapability::new(8, 9),
            128,
            "NVIDIA GeForce RTX 4090",
            PtxTier::Sm80,
        ),
        (
            ComputeCapability::new(12, 0),
            36,
            "NVIDIA GeForce RTX 5060 Ti",
            PtxTier::Sm120,
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
        let mut fixture = Fixture::new();
        fixture.device = Builder::new(cc)
            .multiprocessors(sms)
            .shared_optin_bytes(99 << 10)
            .name(name)
            .build();
        fixture.limit = tier;
        let mut compared = 0;
        for mode in [
            RecipeMode::Disabled,
            RecipeMode::Fp32SegmentationTf32Embedding,
        ] {
            fixture.recipe_mode = mode;
            for fp16 in [Fp16Policy::Allowed, Fp16Policy::Excluded] {
                fixture.fp16 = fp16;
                for boundary in BoundaryId::all() {
                    for batch in boundary.batches().iter() {
                        for math in [CudaMath::Fp32, CudaMath::Tf32] {
                            let driver =
                                match super::select_default(boundary, batch, math, &mut fixture) {
                                    Ok(Selected::Oxide(driver)) => driver,
                                    Err(CudaError::MissingKernel { .. }) => continue,
                                    result => {
                                        panic!("{name} {boundary} b{batch} {math:?}: {result:?}")
                                    }
                                };
                            compared += 1;
                            let Selected::Oxide(hybrid) = PlanRequest::Hybrid
                                .resolve(boundary, batch, math, &mut fixture)
                                .unwrap()
                            else {
                                panic!("{name} {boundary} b{batch} {math:?}")
                            };
                            assert_eq!(
                                hybrid.pin, driver.pin,
                                "{name} {boundary} b{batch} {math:?} {mode:?} {fp16:?}"
                            );
                            assert_eq!(hybrid.target, driver.target);
                        }
                    }
                }
            }
        }
        assert!(compared >= 228, "{name}: covered tuples were not checked");
    }
}

#[test]
fn unmeasured_defaults_obey_accuracy_policy() {
    use super::super::policy::{Recipe, RecipeMode};

    let mut violations = Vec::new();
    for (cc, sms, name) in [
        (ComputeCapability::new(7, 5), 40, "unmeasured Turing"),
        (ComputeCapability::new(8, 0), 108, "unmeasured Ampere"),
        (
            ComputeCapability::new(8, 6),
            84,
            "unmeasured Ampere consumer",
        ),
        (ComputeCapability::new(8, 9), 34, "unmeasured Ada"),
        (ComputeCapability::new(9, 0), 132, "NVIDIA H100"),
        (
            ComputeCapability::new(10, 0),
            148,
            "unmeasured Blackwell server",
        ),
        (
            ComputeCapability::new(12, 0),
            36,
            "unmeasured Blackwell consumer",
        ),
    ] {
        let mut checked = 0;
        for tier in [PtxTier::Sm75, PtxTier::Sm80, PtxTier::Sm90, PtxTier::Sm120] {
            let mut fixture = Fixture::new();
            fixture.device = Builder::new(cc)
                .multiprocessors(sms)
                .shared_optin_bytes(match cc.major {
                    7 => 65536,
                    8 if cc.minor == 0 => 163840,
                    9 | 10 => 227328,
                    _ => 101376,
                })
                .name(name)
                .build();
            fixture.limit = tier;
            if Recipe::measured_device(&fixture.device, tier) {
                violations.push(format!("{name} {tier:?}: fixture must be unmeasured"));
                continue;
            }

            for mode in [
                RecipeMode::Disabled,
                RecipeMode::Fp32SegmentationTf32Embedding,
            ] {
                fixture.recipe_mode = mode;
                for fp16 in [Fp16Policy::Allowed, Fp16Policy::Excluded] {
                    fixture.fp16 = fp16;
                    for boundary in BoundaryId::all() {
                        for batch in boundary.batches().iter() {
                            for math in [CudaMath::Fp32, CudaMath::Tf32] {
                                let context = format!(
                                    "{name} {tier:?} {boundary} b{batch} {math:?} {mode:?} {fp16:?}"
                                );
                                let selected = select_from(
                                    &super::areas(),
                                    boundary,
                                    batch,
                                    math,
                                    &mut &mut fixture,
                                    super::super::Selection::Production,
                                );
                                let selected = match selected {
                                    Ok(Some(Selected::Oxide(selected))) => selected,
                                    Ok(_) => continue,
                                    Err(error) => {
                                        violations.push(format!("{context}: {error}"));
                                        continue;
                                    }
                                };
                                #[cfg(feature = "_cuda-libraries")]
                                let PlanPin::Pinned(pin) = selected.pin else {
                                    violations
                                        .push(format!("{context}: default must have a fixed pin"));
                                    continue;
                                };
                                #[cfg(not(feature = "_cuda-libraries"))]
                                let PlanPin::Pinned(pin) = selected.pin;
                                checked += 1;
                                if super::Selection::Production
                                    .accuracy()
                                    .approve(boundary, math, pin)
                                    .is_none()
                                {
                                    violations.push(format!("{context}: {pin:?}"));
                                }
                            }
                        }
                    }
                }
            }
        }

        if checked == 0 {
            violations.push(format!("{name}: no defaults checked"));
        }
    }

    assert!(
        violations.is_empty(),
        "unmeasured default violations:\n{}",
        violations.join("\n")
    );
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

#[test]
#[cfg(feature = "cuda-sm80")]
fn production_rtx4090_dense_head_uses_the_approved_fp32_pin() {
    use super::super::policy::RecipeMode;
    use crate::inference::cuda::candidate::SegdenseEntry;
    use crate::inference::cuda::tuning::accuracy::Policy;

    let mut fixture = Fixture::new();
    fixture.device = Builder::new(ComputeCapability::new(8, 9))
        .multiprocessors(128)
        .shared_optin_bytes(101376)
        .name("NVIDIA GeForce RTX 4090")
        .build();
    fixture.limit = PtxTier::Sm80;
    fixture.recipe_mode = RecipeMode::Fp32SegmentationTf32Embedding;
    let boundary = BoundaryId::named("resnet.seg_1");
    let Selected::Oxide(selected) =
        super::select_default(boundary, 32, CudaMath::Tf32, &mut fixture)
            .expect("the 4090 dense head has an approved production kernel")
    else {
        panic!("the 4090 dense head must use a kernel")
    };
    let PlanPin::Pinned(ConfigPin::Segdense(pin)) = selected.pin else {
        panic!("the production dense head must have a fixed segdense pin")
    };
    assert_eq!(pin.entry(), SegdenseEntry::EmbedB32);
    assert_eq!(pin.splits(), Some(128));
    assert!(Policy::approve(boundary, CudaMath::Tf32, ConfigPin::Segdense(pin)).is_some());
    assert_eq!(fixture.loads, [selected.target.module]);
}

#[cfg(feature = "_cuda-libraries")]
#[test]
fn unlisted_measured_device_boundary_cannot_bypass_production_accuracy() {
    use super::super::policy::{Recipe, RecipeMode};
    use crate::inference::cuda::tuning::accuracy::Policy;

    for (cc, sms, name) in [
        (
            ComputeCapability::new(8, 9),
            34,
            "NVIDIA GeForce RTX 4060 Ti",
        ),
        (ComputeCapability::new(8, 9), 128, "NVIDIA GeForce RTX 4090"),
        (
            ComputeCapability::new(12, 0),
            36,
            "NVIDIA GeForce RTX 5060 Ti",
        ),
        (ComputeCapability::new(7, 5), 40, "Tesla T4"),
        (ComputeCapability::new(8, 0), 108, "NVIDIA A100-PCIE-40GB"),
        (ComputeCapability::new(8, 0), 108, "NVIDIA A100-SXM4-40GB"),
    ] {
        let mut fixture = Fixture::new();
        fixture.device = Builder::new(cc).multiprocessors(sms).name(name).build();
        fixture.recipe_mode = RecipeMode::Fp32SegmentationTf32Embedding;
        let boundary = BoundaryId::named("sincnet.conv0.abs_pool");
        let pin = ConfigPin::Sinc(SincPin::ConvAbsPool);
        assert_eq!(Policy::approve(boundary, CudaMath::Tf32, pin), None);
        assert_eq!(
            Recipe::accuracy_exception(
                boundary,
                32,
                CudaMath::Tf32,
                pin,
                &fixture.device,
                fixture.limit
            ),
            None
        );
        let result = super::select_default(boundary, 32, CudaMath::Tf32, &mut fixture);
        assert!(
            matches!(
                result,
                Err(CudaError::MissingKernel {
                    batch: 32,
                    math: CudaMath::Tf32,
                    ..
                })
            ),
            "{name}: {result:?}"
        );
        assert!(
            fixture.loads.is_empty(),
            "{name}: an unapproved pin must not load"
        );
    }
}

#[test]
#[cfg(feature = "cuda-sm80")]
fn production_measured_dense_heads_keep_only_their_exact_exceptions() {
    use super::super::policy::{Recipe, RecipeMode};
    use crate::inference::cuda::candidate::SegdensePin;
    use crate::inference::cuda::tuning::accuracy::Policy;

    for (cc, sms, name, expected, recipe) in [
        (
            ComputeCapability::new(8, 9),
            34,
            "NVIDIA GeForce RTX 4060 Ti",
            SegdensePin::measured_rtx4060ti_embedding(),
            Recipe::Rtx4060Ti,
        ),
        (
            ComputeCapability::new(12, 0),
            36,
            "NVIDIA GeForce RTX 5060 Ti",
            SegdensePin::measured_rtx5060ti_embedding(),
            Recipe::Rtx5060Ti,
        ),
    ] {
        let mut fixture = Fixture::new();
        fixture.device = Builder::new(cc)
            .multiprocessors(sms)
            .shared_optin_bytes(101376)
            .name(name)
            .build();
        fixture.limit = PtxTier::Sm80;
        fixture.recipe_mode = RecipeMode::Fp32SegmentationTf32Embedding;
        let boundary = BoundaryId::named("resnet.seg_1");
        let expected = ConfigPin::Segdense(expected);
        let Selected::Oxide(selected) =
            super::select_default(boundary, 32, CudaMath::Tf32, &mut fixture).unwrap()
        else {
            panic!("the measured dense head must use a kernel")
        };
        assert_eq!(selected.pin, PlanPin::Pinned(expected));
        assert_eq!(Policy::approve(boundary, CudaMath::Tf32, expected), None);
        assert_eq!(
            Recipe::accuracy_exception(
                boundary,
                32,
                CudaMath::Tf32,
                expected,
                &fixture.device,
                fixture.limit
            ),
            Some(recipe)
        );
        let different = Builder::new(ComputeCapability::new(8, 9))
            .multiprocessors(64)
            .shared_optin_bytes(101376)
            .build();
        let different_pin = ConfigPin::Segdense(
            SegdensePin::forced_embed_b32("EmbedB32F16", CudaMath::Tf32, PtxTier::Sm80, &different)
                .unwrap(),
        );
        assert_ne!(expected, different_pin);
        for (boundary, batch, math, pin) in [
            (boundary, 32, CudaMath::Tf32, different_pin),
            (boundary, 1, CudaMath::Tf32, expected),
            (boundary, 32, CudaMath::Fp32, expected),
            (BoundaryId::named("linear0"), 32, CudaMath::Tf32, expected),
        ] {
            assert_eq!(
                Recipe::accuracy_exception(
                    boundary,
                    batch,
                    math,
                    pin,
                    &fixture.device,
                    fixture.limit
                ),
                None
            );
        }
    }
}

#[test]
#[cfg(feature = "_cuda-libraries")]
fn measured_pipeline_pins_change_only_the_4090_head_and_t4_large_stem() {
    use super::super::policy::RecipeMode;
    use crate::inference::cuda::candidate::SegdenseEntry;

    let mut changes = 0;
    for (cc, sms, name, tier) in [
        (
            ComputeCapability::new(8, 9),
            34,
            "NVIDIA GeForce RTX 4060 Ti",
            PtxTier::Sm80,
        ),
        (
            ComputeCapability::new(8, 9),
            128,
            "NVIDIA GeForce RTX 4090",
            PtxTier::Sm80,
        ),
        (
            ComputeCapability::new(12, 0),
            36,
            "NVIDIA GeForce RTX 5060 Ti",
            PtxTier::Sm120,
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
        let mut fixture = Fixture::new();
        fixture.device = Builder::new(cc)
            .multiprocessors(sms)
            .shared_optin_bytes(99 << 10)
            .name(name)
            .build();
        fixture.limit = tier;
        fixture.recipe_mode = RecipeMode::Fp32SegmentationTf32Embedding;
        for fp16 in [Fp16Policy::Allowed, Fp16Policy::Excluded] {
            fixture.fp16 = fp16;
            for boundary in BoundaryId::all().filter(|id| id.name() != "lstm.stack.input_proj") {
                let math = match boundary.area() {
                    KernelModule::Resnet | KernelModule::Embedding => CudaMath::Tf32,
                    _ => CudaMath::Fp32,
                };
                for batch in boundary.batches().iter() {
                    let Selected::Oxide(previous) = PlanRequest::DriverOnly
                        .resolve(boundary, batch, math, &mut fixture)
                        .unwrap()
                    else {
                        panic!("the previous measured route must use a kernel")
                    };
                    let Selected::Oxide(current) =
                        super::select_default(boundary, batch, math, &mut fixture).unwrap()
                    else {
                        panic!("the measured production route must use a kernel")
                    };
                    assert_eq!(previous.target, current.target);
                    if name == "NVIDIA GeForce RTX 4090"
                        && boundary == BoundaryId::named("resnet.seg_1")
                        && batch == 32
                    {
                        let PlanPin::Pinned(ConfigPin::Segdense(before)) = previous.pin else {
                            panic!("previous dense head")
                        };
                        let PlanPin::Pinned(ConfigPin::Segdense(after)) = current.pin else {
                            panic!("current dense head")
                        };
                        assert_eq!(before.entry(), SegdenseEntry::EmbedB32F16);
                        assert_eq!(after.entry(), SegdenseEntry::EmbedB32);
                        assert_eq!(after.splits(), Some(128));
                        changes += 1;
                    } else if name == "Tesla T4"
                        && boundary == BoundaryId::named("resnet.conv1")
                        && matches!(batch, 16 | 32)
                    {
                        use crate::inference::cuda::candidate::{
                            WideconvAlgorithm, WideconvConfig, WideconvPartition, WideconvPin,
                            WideconvSplitCells,
                        };
                        for (pin, algorithm) in [
                            (previous.pin, WideconvAlgorithm::WideStem),
                            (current.pin, WideconvAlgorithm::Spatial),
                        ] {
                            assert_eq!(
                                pin,
                                PlanPin::Pinned(ConfigPin::Wideconv(WideconvPin::Configured(
                                    WideconvConfig {
                                        algorithm,
                                        partition: WideconvPartition::Whole,
                                        split_cells: WideconvSplitCells::All,
                                    }
                                )))
                            );
                        }
                        changes += 1;
                    } else {
                        assert_eq!(
                            previous.pin, current.pin,
                            "{name} {boundary} b{batch} {math:?} {fp16:?}"
                        );
                    }
                }
            }
        }
    }
    assert_eq!(
        changes, 6,
        "the 4090 head and T4 b16/b32 stem change with either trunk FP16 guard state"
    );
}

#[test]
#[cfg(feature = "cuda-sm80")]
fn accuracy_replacement_does_not_inherit_the_old_pins_speed_evidence() {
    use crate::inference::cuda::candidate::{
        WideconvAlgorithm, WideconvConfig, WideconvPartition, WideconvPin, WideconvSplitCells,
    };
    let mut fixture = Fixture::new();
    fixture.device = Builder::new(ComputeCapability::new(8, 9))
        .multiprocessors(34)
        .name("NVIDIA GeForce RTX 4060 Ti")
        .build();
    fixture.limit = PtxTier::Sm80;
    let selected = super::select_default(
        BoundaryId::named("resnet.layer3.1.conv1"),
        32,
        CudaMath::Fp32,
        &mut fixture,
    )
    .unwrap();
    let Selected::Oxide(token) = selected else {
        panic!("approved spatial replacement")
    };
    assert_eq!(
        token.pin,
        PlanPin::Pinned(ConfigPin::Wideconv(WideconvPin::Configured(
            WideconvConfig {
                algorithm: WideconvAlgorithm::Spatial,
                partition: WideconvPartition::Whole,
                split_cells: WideconvSplitCells::All,
            }
        )))
    );
    assert_eq!(token.evidence, TokenEvidence::Implemented);
    assert!(!token.speed_measured());
}

#[test]
#[cfg(not(feature = "_cuda-libraries"))]
fn target_only_tf32_segmentation_selects_every_stage_boundary() {
    use super::super::policy::RecipeMode;
    use crate::inference::cuda::tuning::accuracy::Policy;

    let mut checked = 0;
    for (cc, tier, shared) in [
        (ComputeCapability::new(7, 5), PtxTier::Sm75, 65536),
        (ComputeCapability::new(8, 0), PtxTier::Sm80, 163840),
        (ComputeCapability::new(8, 9), PtxTier::Sm80, 101376),
        (ComputeCapability::new(9, 0), PtxTier::Sm90, 227328),
        (ComputeCapability::new(12, 0), PtxTier::Sm120, 101376),
    ] {
        let compiled = match tier {
            PtxTier::Sm75 => cfg!(feature = "cuda-sm75"),
            PtxTier::Sm80 => cfg!(feature = "cuda-sm80"),
            PtxTier::Sm90 => cfg!(feature = "cuda-sm90"),
            PtxTier::Sm120 => cfg!(feature = "cuda-sm120"),
        };
        if !compiled {
            continue;
        }

        let mut fixture = Fixture::new();
        fixture.device = Builder::new(cc).shared_optin_bytes(shared).build();
        fixture.limit = tier;
        fixture.recipe_mode = RecipeMode::new(CudaMath::Tf32, CudaMath::Tf32);
        for batch in [1, 32] {
            // the projected LSTM stack owns its input projection as part of this boundary
            for (name, area) in [
                ("sincnet.conv0.abs_pool", KernelModule::Sincnet),
                ("sincnet.conv1", KernelModule::Segdense),
                ("sincnet.conv2", KernelModule::Segdense),
                ("lstm.stack", KernelModule::LstmProj),
                ("linear0", KernelModule::Segdense),
                ("linear1", KernelModule::Segdense),
                ("linear2", KernelModule::Segdense),
            ] {
                let boundary = BoundaryId::named(name);
                let Selected::Oxide(token) =
                    super::select_default(boundary, batch, CudaMath::Tf32, &mut fixture)
                        .unwrap_or_else(|error| panic!("{cc:?} {name} b{batch}: {error}"))
                else {
                    panic!("{cc:?} {name} b{batch}: target-only needs a kernel")
                };
                checked += 1;
                let PlanPin::Pinned(pin) = token.pin;
                assert_eq!(pin.area(), area, "{cc:?} {name} b{batch}");
                assert_eq!(token.math, CudaMath::Tf32);
                assert!(
                    super::Selection::Production
                        .accuracy()
                        .approve(boundary, CudaMath::Tf32, pin)
                        .is_some()
                );
                if name == "sincnet.conv0.abs_pool" {
                    assert_eq!(pin, ConfigPin::Sinc(SincPin::ConvAbsPool));
                    assert_eq!(Policy::approve(boundary, CudaMath::Tf32, pin), None);
                }
            }
        }
        assert_eq!(fixture.loads.len(), 14);
    }
    assert!(checked >= 14);
}
