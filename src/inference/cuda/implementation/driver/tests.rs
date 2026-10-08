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
}

impl Fixture {
    fn new() -> Self {
        Self {
            device: Builder::new(ComputeCapability::new(12, 0))
                .multiprocessors(36)
                .name("unmeasured GPU")
                .build(),
            loads: vec![],
        }
    }
}

impl Modules for &mut Fixture {
    fn device(&self) -> &DeviceAttributes {
        &self.device
    }
    fn tier_limit(&self) -> PtxTier {
        PtxTier::Sm120
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
    let Selected::Oxide(token) = selected else {
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
        ("fbank.dft", 1, CudaMath::Fp32),
        ("lstm.stack", 32, CudaMath::Tf32),
        ("linear0", 32, CudaMath::Fp32),
    ] {
        let result =
            PlanRequest::DriverOnly.resolve(BoundaryId::named(boundary), batch, math, &mut fixture);
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
        use crate::inference::cuda::implementation::evidence::{ArchitectureSpeed, BroadEvidence};
        static SUMMARY: BroadEvidence = BroadEvidence::new(
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
    let Selected::Oxide(token) = selected else {
        panic!("broad port must run on an unmeasured GPU")
    };
    let TokenEvidence::Broad { scope } = token.evidence else {
        panic!("broad port must carry its evidence")
    };
    assert!(scope.contains(&fixture.device));
    assert!(!scope.measured_on_device(fixture.device.capability()));
    assert_eq!(token.selection, super::Selection::Production);
    assert_eq!(fixture.loads, [token.target.module]);
}
