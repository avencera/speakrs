//! Qualification-backed selection before any optional library state is created
//!
//! A production binding owns one module request per area and device scope. Each of
//! its tuple proofs pins a complete configuration and carries a separate speed status.
//! Selection resolves the module request from the binding before any load, so adding
//! another embedded tier never changes what an existing binding loads

// runtime wiring is deferred until override policy and input ownership are defined
#[allow(dead_code)]
pub(crate) mod overrides;

mod boundary;
mod driver;
mod evidence;

/// Accepted production bindings, one file per candidate area
mod production {
    pub(super) mod lstm;
    pub(super) mod resnet;
    pub(super) mod sincnet;

    use super::{RecordHash, SpeedEvidence, SpeedScope, SpeedStatus, TupleProof};
    use crate::inference::cuda::candidate::ConfigPin;
    use crate::inference::cuda::implementation::BoundaryId;
    use crate::inference::cuda::{ComputeCapability, CudaMath};

    pub(super) const FP32: CudaMath = CudaMath::Fp32;
    pub(super) const TF32: CudaMath = CudaMath::Tf32;

    /// SHA256 of int-k/ab-summary.json for the integrated configuration
    pub(super) const INTEGRATED_DER: RecordHash =
        RecordHash::from_hex("8066268031afba058d93e305206d5b8225e40c6ab2ebb1052874d607e055646f");

    /// PR #36 approval: sm75 PTX forced on an RTX 5070 Ti, granted for the whole
    /// capability before SM count was a selection key
    pub(super) const LEGACY_SCOPE: SpeedScope = SpeedScope::LegacyCapability {
        capability: ComputeCapability::new(12, 0),
    };

    /// A PR #36 tuple: one record proves both accuracy and capability-wide speed
    pub(super) const fn legacy_proof(
        boundary: BoundaryId,
        batch: usize,
        math: CudaMath,
        pin: ConfigPin,
        record: RecordHash,
    ) -> TupleProof {
        TupleProof {
            boundary,
            batch,
            math,
            pin,
            accuracy: record,
            speed: SpeedStatus::Measured(SpeedEvidence {
                scope: LEGACY_SCOPE,
                record,
                integrated: INTEGRATED_DER,
            }),
        }
    }
}

pub(crate) use boundary::{BoundaryId, ProductionBatches};
pub(crate) use evidence::{
    ArchitectureSpeed, Binding, BroadEvidence, RecordHash, SpeedEvidence, SpeedScope, SpeedStatus,
    TupleProof,
};

#[cfg(all(test, feature = "_cuda-libraries"))]
use super::candidate::{Batches, Coverage, CoverageEntry, Maths};
use super::candidate::{
    ConfigPin, ConvCandidate, ConvLayerSpec, ConvOxide, DenseCandidate, DenseOxide, DenseSpec,
    FbankCandidate, FbankOxide, FbankSpec, LstmCandidate, LstmOxide, LstmProjOxide, LstmSpec,
    PlanError, SegConvCandidate, SegConvOxide, SegConvSpec, SincCandidate, SincOxide, SincSpec,
};
use super::device::DeviceAttributes;
use super::error::GeometryError;
use super::kernels::{AreaPtx, ArtifactHash, LoadedArtifact, ModuleRequest};
use super::{
    ComputeCapability, CudaError, CudaLibrary, CudaMath, CudaRuntime, KernelModule, PtxTier,
};

/// A loaded module and the exact device that executes it
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Target {
    pub module: ModuleRequest,
    pub device: ComputeCapability,
}

/// The tier Library diagnostics report for an area, without loading anything
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AreaTarget {
    pub tier: PtxTier,
    pub device: ComputeCapability,
}

impl AreaTarget {
    /// The tier of the area's production module on this device; an area that loads no
    /// candidate module here reports the baseline tier every area ships
    pub(crate) fn for_area(runtime: &CudaRuntime, area: KernelModule) -> Result<Self, CudaError> {
        Ok(Self {
            tier: production_tier(area, runtime.device(), runtime.ptx_tier())
                .unwrap_or(PtxTier::BASELINE),
            device: runtime.compute_capability(),
        })
    }
}

/// Why a candidate was selected; only production permits a device fallback
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Selection {
    /// An accepted production-table entry
    Production,
    /// Complete implemented coverage; speed is not a selection gate
    DriverOnly,
    /// An explicit or qualification request
    #[cfg(all(test, feature = "_cuda-libraries"))]
    Explicit,
}

/// Implementation requests used only by the qualification controls
#[cfg(all(test, feature = "_cuda-libraries"))]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Choice {
    #[default]
    Library,
    Oxide(Selection),
    /// The pinned FP32 production path with a stage-only timing fault
    StageTail,
    /// The same pinned FP32 production path without the timing fault
    StageTailControl,
    Mutant(super::test_support::Mutant),
}

/// One selected owner; an Oxide token can only come from this module
#[derive(Debug)]
pub(crate) enum Selected {
    Library,
    Oxide(Qualified),
    #[cfg(all(test, feature = "_cuda-libraries"))]
    Mutant(super::test_support::Mutant),
}

/// Every accepted production binding
const PRODUCTION: &[Binding] = &[
    production::resnet::BINDING,
    production::lstm::BINDING,
    production::sincnet::BINDING,
];

// always-on areas retain the artifact policy used before cubins were shipped
const ALWAYS_ON: &[KernelModule] = &[
    KernelModule::Fbank,
    KernelModule::Embedding,
    KernelModule::Segmentation,
];

/// Routes that may implement one boundary, highest precedence first; the first route
/// with a proof for a tuple decides it, even when that proof's speed is unmeasured
const ROUTE_PRECEDENCE: &[KernelModule] = &[
    KernelModule::Wideconv,
    KernelModule::Resnet,
    KernelModule::Segdense,
    KernelModule::LstmProj,
    KernelModule::Lstm,
    KernelModule::Sincnet,
    KernelModule::FbankDft,
];

// one module per area and overlapping device scope, complete pins and scoped evidence
const _: () = evidence::validate(PRODUCTION, ALWAYS_ON, ROUTE_PRECEDENCE);

/// Production model batch classes, excluding the harness stress classes
pub(crate) const MODEL_BATCHES: [usize; 2] = ProductionBatches::MODEL;

/// The module production loads for an area on this device, resolved before any load
///
/// Always-on areas request driver JIT of their embedded baseline PTX on every device.
/// Candidate areas request exactly their binding's module, or nothing when no binding
/// covers the device or the runtime's tier limit is below the binding's tier
pub(crate) fn production_module(
    area: KernelModule,
    device: &DeviceAttributes,
    limit: PtxTier,
    variants: AreaPtx,
) -> Result<Option<ModuleRequest>, CudaError> {
    if super::driver_only() || driver::uses_port_artifact(area) {
        return Ok(variants.driver_request(area, limit, device.capability()));
    }
    if !ALWAYS_ON.contains(&area) {
        return Ok(bound_module(PRODUCTION, area, device, limit));
    }
    let tier = PtxTier::BASELINE;
    let ptx = variants
        .embedded(tier)
        .ok_or(CudaError::AreaTierNotCompiledIn {
            area: area.name(),
            tier,
            device: device.capability(),
            feature: tier.feature(),
        })?;
    Ok(Some(ModuleRequest::new(
        area,
        tier,
        LoadedArtifact::PtxJit {
            sha256: ArtifactHash::of(ptx.text.as_bytes()),
        },
    )))
}

/// The production module's tier, without hashing embedded bytes
fn production_tier(
    area: KernelModule,
    device: &DeviceAttributes,
    limit: PtxTier,
) -> Option<PtxTier> {
    if super::driver_only() || driver::uses_port_artifact(area) {
        return area
            .variants()
            .driver_request(area, limit, device.capability())
            .map(ModuleRequest::tier);
    }
    if ALWAYS_ON.contains(&area) {
        return Some(PtxTier::BASELINE);
    }
    bound_module(PRODUCTION, area, device, limit).map(ModuleRequest::tier)
}

fn bound_module(
    bindings: &[Binding],
    area: KernelModule,
    device: &DeviceAttributes,
    limit: PtxTier,
) -> Option<ModuleRequest> {
    // the validator guarantees one module for every overlapping binding of an area
    bindings
        .iter()
        .find(|binding| binding.area() == area && binding.scope.contains(device))
        .map(|binding| binding.module)
        .filter(|module| module.tier() <= limit)
}

/// An accepted production route: the binding's module and the tuple's proof
#[derive(Debug, Clone, Copy)]
struct Route {
    module: ModuleRequest,
    proof: &'static TupleProof,
    speed: SpeedEvidence,
}

/// The highest-precedence route with a proof for this tuple on this device
///
/// That route decides: a measured proof selects it, and an unmeasured one selects
/// Library while the libraries exist. A lower route is never tried instead
fn production_route(
    bindings: &'static [Binding],
    precedence: &[KernelModule],
    boundary: BoundaryId,
    batch: usize,
    math: CudaMath,
    device: &DeviceAttributes,
) -> Option<Route> {
    if !boundary.batches().contains(batch) {
        return None;
    }
    for area in precedence {
        let Some((binding, proof)) = bindings
            .iter()
            .filter(|binding| binding.area() == *area && binding.scope.contains(device))
            .find_map(|binding| Some((binding, binding.proof(boundary, batch, math)?)))
        else {
            continue;
        };
        return match proof.speed {
            SpeedStatus::Measured(speed) if speed.scope.contains(device) => Some(Route {
                module: binding.module,
                proof,
                speed,
            }),
            SpeedStatus::Measured(_) | SpeedStatus::Unmeasured => None,
        };
    }
    None
}

/// The configuration a token plans
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PlanPin {
    /// The pin an accepted proof names
    Pinned(ConfigPin),
    /// The candidate's own implemented configuration, for qualification only
    #[cfg(all(test, feature = "_cuda-libraries"))]
    Implemented,
}

/// The evidence behind a token
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TokenEvidence {
    /// An accepted proof with measured speed for this device
    Production {
        accuracy: RecordHash,
        speed: SpeedEvidence,
    },
    /// Implemented library-free coverage, without a speed claim
    Implemented,
    /// A complete broad-winner port, carrying its structural speed evidence
    Port {
        scope: SpeedScope,
        summary: &'static str,
    },
    /// An explicit qualification control, which grants no production evidence
    #[cfg(all(test, feature = "_cuda-libraries"))]
    Qualification,
}

/// A boundary accepted by a pinned record, or an explicit qualification control
#[derive(Debug)]
pub(crate) struct Qualified {
    boundary: BoundaryId,
    batch: usize,
    math: CudaMath,
    target: Target,
    pin: PlanPin,
    evidence: TokenEvidence,
    selection: Selection,
}

impl Qualified {
    /// The area that owns this selected plan
    pub(crate) fn area(&self) -> KernelModule {
        self.target.module.area()
    }

    fn speed_measured(&self) -> bool {
        match self.evidence {
            TokenEvidence::Production { speed, .. } => {
                speed.scope.measured_on_device(self.target.device)
            }
            TokenEvidence::Port { scope, .. } => scope.measured_on_device(self.target.device),
            _ => false,
        }
    }

    fn production(
        boundary: BoundaryId,
        batch: usize,
        math: CudaMath,
        device: &DeviceAttributes,
        route: Route,
    ) -> Self {
        Self {
            boundary,
            batch,
            math,
            target: Target {
                module: route.module,
                device: device.capability(),
            },
            pin: PlanPin::Pinned(route.proof.pin),
            evidence: TokenEvidence::Production {
                accuracy: route.proof.accuracy,
                speed: route.speed,
            },
            selection: Selection::Production,
        }
    }

    /// The module already selected and loaded before this plan is constructed
    fn plan_module(&self) -> ModuleRequest {
        self.target.module
    }

    fn check(
        &self,
        runtime: &CudaRuntime,
        area: KernelModule,
        boundary: &str,
        batch: usize,
        math: CudaMath,
    ) -> Result<super::LoadedKernels, CudaError> {
        let mismatch = || CudaError::Unsupported {
            context: "qualified plan",
            reason: "qualification token does not match the requested plan".to_owned(),
        };
        if self.boundary.name() != boundary
            || self.batch != batch
            || self.math != math
            || self.target.module.area() != area
            || self.target.device != runtime.compute_capability()
        {
            return Err(mismatch());
        }
        // the cached module must be exactly the token's; a conflicting cache is an error
        let loaded = runtime.load_module(self.plan_module())?;
        if loaded.request() != self.plan_module() {
            return Err(mismatch());
        }
        match self.evidence {
            TokenEvidence::Production { accuracy, speed } => tracing::debug!(
                boundary = self.boundary.name(),
                batch,
                ?math,
                %accuracy,
                speed_record = %speed.record,
                integrated = %speed.integrated,
                scope = ?speed.scope,
                "CUDA production evidence"
            ),
            TokenEvidence::Port { scope, .. } => {
                if let SpeedScope::AllDevices(evidence) = scope {
                    tracing::debug!(
                        summary = evidence.summary(),
                        ?scope,
                        "CUDA broad-winner evidence"
                    );
                }
            }
            TokenEvidence::Implemented => {}
            #[cfg(all(test, feature = "_cuda-libraries"))]
            TokenEvidence::Qualification => {}
        }
        Ok(loaded)
    }

    /// Resolve a refusal without constructing a dormant Library plan
    ///
    /// An invalid geometry is a host bug in every mode. Production outside driver-only
    /// mode uses Library for a capability refusal, out-of-contract weights or an
    /// unimplemented geometry; every other mode returns the typed refusal
    pub(super) fn finish<T>(
        &self,
        area: KernelModule,
        driver_only: bool,
        result: Result<T, PlanError>,
    ) -> Result<Option<T>, CudaError> {
        let library_allowed = self.selection == Selection::Production && !driver_only;
        match result {
            Ok(plan) => {
                let measured = self.speed_measured();
                tracing::info!(boundary = self.boundary.name(), batch = self.batch,
                    math = ?self.math, area = area.name(), speed_measured = measured,
                    evidence = ?self.evidence, "CUDA route implementation=Oxide");
                Ok(Some(plan))
            }
            Err(PlanError::Cuda(error)) => Err(error),
            Err(PlanError::Geometry(error @ GeometryError::Invalid { .. })) => {
                Err(self.geometry(area, error))
            }
            Err(refusal) if library_allowed => {
                tracing::warn!(
                    boundary = self.boundary.name(),
                    batch = self.batch,
                    "CUDA candidate unavailable reason={refusal}; using Library"
                );
                Ok(None)
            }
            Err(PlanError::DeviceUnsupported { reason }) => {
                Err(CudaError::CandidateDeviceUnsupported {
                    area: area.name(),
                    boundary: self.boundary.name().to_owned(),
                    batch: self.batch,
                    math: self.math,
                    tier: self.target.module.tier(),
                    device: self.target.device,
                    reason,
                })
            }
            Err(PlanError::WeightsOutOfContract { layer, fault }) => {
                Err(CudaError::CandidateWeightsOutOfContract {
                    area: area.name(),
                    boundary: self.boundary.name().to_owned(),
                    batch: self.batch,
                    math: self.math,
                    tier: self.target.module.tier(),
                    device: self.target.device,
                    layer,
                    fault,
                })
            }
            Err(PlanError::Geometry(error)) => Err(self.geometry(area, error)),
        }
    }

    fn geometry(&self, area: KernelModule, error: GeometryError) -> CudaError {
        CudaError::CandidateGeometry {
            area: area.name(),
            boundary: self.boundary.name().to_owned(),
            batch: self.batch,
            math: self.math,
            error,
        }
    }

    /// A pin for another area is a violated table invariant, never a fallback
    fn foreign_pin(&self, pin: ConfigPin) -> CudaError {
        CudaError::Unsupported {
            context: "qualified plan",
            reason: format!("{} pin {pin:?} cannot plan this area", self.boundary),
        }
    }

    /// Build exactly the accepted convolution
    pub(crate) fn conv(
        self,
        runtime: &CudaRuntime,
        spec: ConvLayerSpec<'_>,
    ) -> Result<Option<ConvOxide>, CudaError> {
        let area = KernelModule::Resnet;
        let kernels = self.check(runtime, area, spec.name, spec.conv.batch, spec.conv.math)?;
        let pin = match self.pin {
            PlanPin::Pinned(ConfigPin::Conv(pin)) => Ok(pin),
            PlanPin::Pinned(other) => return Err(self.foreign_pin(other)),
            #[cfg(all(test, feature = "_cuda-libraries"))]
            PlanPin::Implemented => ConvOxide::implemented_pin(&spec),
        };
        let plan = pin.and_then(|pin| {
            let plan = ConvOxide::plan(runtime, &kernels, spec, pin)?;
            #[cfg(all(test, feature = "_cuda-libraries"))]
            super::test_support::configuration::record(
                self.boundary.name(),
                self.batch,
                self.math,
                ConfigPin::Conv(pin),
            );
            Ok(plan)
        });
        self.finish(area, super::driver_only(), plan)
    }

    /// Build exactly the accepted Sinc producer
    pub(crate) fn sinc(
        self,
        runtime: &CudaRuntime,
        spec: SincSpec<'_>,
    ) -> Result<Option<SincOxide>, CudaError> {
        let area = KernelModule::Sincnet;
        let kernels = self.check(
            runtime,
            area,
            "sincnet.conv0.abs_pool",
            spec.batch,
            spec.math,
        )?;
        let pin = match self.pin {
            PlanPin::Pinned(ConfigPin::Sinc(pin)) => Ok(pin),
            PlanPin::Pinned(other) => return Err(self.foreign_pin(other)),
            #[cfg(all(test, feature = "_cuda-libraries"))]
            PlanPin::Implemented => SincOxide::implemented_pin(&spec),
        };
        let plan = pin.and_then(|pin| {
            let plan = SincOxide::plan(runtime, &kernels, spec, pin)?;
            #[cfg(all(test, feature = "_cuda-libraries"))]
            super::test_support::configuration::record(
                self.boundary.name(),
                self.batch,
                self.math,
                ConfigPin::Sinc(pin),
            );
            Ok(plan)
        });
        self.finish(area, super::driver_only(), plan)
    }

    /// Build the accepted stack; qualification forbids nested library calls
    pub(crate) fn lstm(
        self,
        runtime: &CudaRuntime,
        spec: LstmSpec<'_>,
    ) -> Result<Option<LstmOxide>, CudaError> {
        let area = KernelModule::Lstm;
        let kernels = self.check(runtime, area, "lstm.stack", spec.batch, spec.math)?;
        let pin = match self.pin {
            PlanPin::Pinned(ConfigPin::Lstm(pin)) => Ok(pin),
            PlanPin::Pinned(other) => return Err(self.foreign_pin(other)),
            #[cfg(all(test, feature = "_cuda-libraries"))]
            PlanPin::Implemented => LstmOxide::implemented_pin(&spec),
        };
        let plan = pin.and_then(|pin| {
            let plan = LstmOxide::plan(runtime, &kernels, spec, pin)?;
            #[cfg(all(test, feature = "_cuda-libraries"))]
            super::test_support::configuration::record(
                self.boundary.name(),
                self.batch,
                self.math,
                ConfigPin::Lstm(pin),
            );
            Ok(plan)
        });
        self.finish(area, super::driver_only(), plan)
    }

    /// Build the library-free projected LSTM stack
    pub(crate) fn projected_lstm(
        self,
        runtime: &CudaRuntime,
        spec: LstmSpec<'_>,
    ) -> Result<Option<LstmProjOxide>, CudaError> {
        let area = KernelModule::LstmProj;
        let kernels = self.check(runtime, area, "lstm.stack", spec.batch, spec.math)?;
        let pin = match self.pin {
            PlanPin::Pinned(ConfigPin::Lstm(pin)) => Ok(pin),
            PlanPin::Pinned(other) => return Err(self.foreign_pin(other)),
            #[cfg(all(test, feature = "_cuda-libraries"))]
            PlanPin::Implemented => {
                LstmProjOxide::device_pin(runtime.device(), kernels.tier(), &spec)
            }
        };
        let plan = pin.and_then(|pin| LstmProjOxide::plan(runtime, &kernels, spec, pin));
        self.finish(area, super::driver_only(), plan)
    }

    /// Build a complete dense operation from its selected module and packed weights
    pub(crate) fn dense(
        self,
        runtime: &CudaRuntime,
        spec: DenseSpec,
        weight: &cudarc::driver::CudaSlice<f32>,
        bias: &cudarc::driver::CudaSlice<f32>,
    ) -> Result<Option<DenseOxide>, CudaError> {
        let area = KernelModule::Segdense;
        let kernels = self.check(
            runtime,
            area,
            spec.site().boundary().name(),
            spec.batch(),
            spec.math(),
        )?;
        let pin = match self.pin {
            PlanPin::Pinned(ConfigPin::Segdense(pin)) => Ok(pin),
            PlanPin::Pinned(other) => return Err(self.foreign_pin(other)),
            #[cfg(all(test, feature = "_cuda-libraries"))]
            PlanPin::Implemented => {
                DenseOxide::implemented_pin(spec, kernels.tier(), runtime.device())
            }
        };
        let plan = pin.and_then(|pin| DenseOxide::plan(runtime, &kernels, spec, weight, bias, pin));
        self.finish(area, super::driver_only(), plan)
    }

    /// Build a raw temporal convolution from its selected module and packed weights
    pub(crate) fn segconv(
        self,
        runtime: &CudaRuntime,
        spec: SegConvSpec,
        weight: &cudarc::driver::CudaSlice<f32>,
    ) -> Result<Option<SegConvOxide>, CudaError> {
        let area = KernelModule::Segdense;
        let kernels = self.check(
            runtime,
            area,
            spec.site().boundary().name(),
            spec.batch(),
            spec.math(),
        )?;
        let pin = match self.pin {
            PlanPin::Pinned(ConfigPin::Segdense(pin)) => Ok(pin),
            PlanPin::Pinned(other) => return Err(self.foreign_pin(other)),
            #[cfg(all(test, feature = "_cuda-libraries"))]
            PlanPin::Implemented => {
                SegConvOxide::implemented_pin(spec, kernels.tier(), runtime.device())
            }
        };
        let plan = pin.and_then(|pin| SegConvOxide::plan(runtime, &kernels, spec, weight, pin));
        self.finish(area, super::driver_only(), plan)
    }

    /// Build exactly the accepted filterbank energy producer
    pub(crate) fn fbank(
        self,
        runtime: &CudaRuntime,
        spec: FbankSpec,
    ) -> Result<Option<FbankOxide>, CudaError> {
        let area = KernelModule::FbankDft;
        let kernels = self.check(runtime, area, "fbank.dft", spec.batch(), spec.math())?;
        let pin = match self.pin {
            PlanPin::Pinned(ConfigPin::Fbank(pin)) => Ok(pin),
            PlanPin::Pinned(other) => return Err(self.foreign_pin(other)),
            #[cfg(all(test, feature = "_cuda-libraries"))]
            PlanPin::Implemented => FbankOxide::implemented_pin(spec),
        };
        let plan = pin.and_then(|pin| {
            let plan = FbankOxide::plan(runtime, &kernels, spec, pin)?;
            #[cfg(all(test, feature = "_cuda-libraries"))]
            super::test_support::configuration::record(
                self.boundary.name(),
                self.batch,
                self.math,
                ConfigPin::Fbank(pin),
            );
            Ok(plan)
        });
        self.finish(area, super::driver_only(), plan)
    }
}

/// Invalid selection requests cannot create an Oxide token
#[derive(Debug, thiserror::Error)]
#[error("cannot select a CUDA boundary at batch zero")]
pub(crate) struct SelectionError;

/// Select a pinned tuple for an already loaded module; uncovered tuples use Library
#[cfg(all(test, feature = "_cuda-libraries"))]
pub(crate) fn select(
    boundary: BoundaryId,
    batch: usize,
    math: CudaMath,
    device: &DeviceAttributes,
    loaded: ModuleRequest,
) -> Result<Selected, SelectionError> {
    if batch == 0 {
        return Err(SelectionError);
    }
    Ok(
        match production_route(PRODUCTION, ROUTE_PRECEDENCE, boundary, batch, math, device) {
            Some(route) if route.module == loaded => {
                Selected::Oxide(Qualified::production(boundary, batch, math, device, route))
            }
            _ => Selected::Library,
        },
    )
}

/// What selection needs from a runtime: the cached device, the tier limit and a
/// strict loader. Host tests supply fixtures
pub(crate) trait Modules {
    /// The runtime's cached device attributes
    fn device(&self) -> &DeviceAttributes;

    /// The runtime's PTX tier limit
    fn tier_limit(&self) -> PtxTier;

    /// Whether model-load policy forces every replaceable boundary to Library
    fn force_library(&self) -> bool {
        false
    }

    /// Load exactly `request` and return the identity the driver accepted
    fn load(&mut self, request: ModuleRequest) -> Result<ModuleRequest, CudaError>;

    /// The best embedded artifact an explicit qualification request asks for,
    /// resolved without loading
    #[cfg(all(test, feature = "_cuda-libraries"))]
    fn embedded_exact(&self, area: KernelModule) -> Result<ModuleRequest, CudaError>;
}

impl Modules for &CudaRuntime {
    fn force_library(&self) -> bool {
        CudaRuntime::force_library(self)
    }

    fn device(&self) -> &DeviceAttributes {
        CudaRuntime::device(self)
    }

    fn tier_limit(&self) -> PtxTier {
        self.ptx_tier()
    }

    fn load(&mut self, request: ModuleRequest) -> Result<ModuleRequest, CudaError> {
        Ok(self.load_module(request)?.request())
    }

    #[cfg(all(test, feature = "_cuda-libraries"))]
    fn embedded_exact(&self, area: KernelModule) -> Result<ModuleRequest, CudaError> {
        self.embedded_exact_request(area)
    }
}

/// Selection for a concrete plan, with test controls kept outside production
pub(crate) fn plan_selection(
    runtime: &CudaRuntime,
    boundary: BoundaryId,
    batch: usize,
    math: CudaMath,
    #[cfg(all(test, feature = "_cuda-libraries"))] override_choice: Option<Choice>,
) -> Result<Selected, CudaError> {
    let request = PlanRequest::Hybrid;
    #[cfg(all(test, feature = "_cuda-libraries"))]
    let request = override_choice.map_or_else(
        || match super::test_support::default_choice(Choice::Oxide(Selection::Production)) {
            Choice::Oxide(Selection::Production) => request,
            choice => PlanRequest::Qualification(choice),
        },
        PlanRequest::Qualification,
    );
    let selected = request.resolve(boundary, batch, math, runtime)?;
    match &selected {
        Selected::Oxide(token) => {
            tracing::info!(boundary = boundary.name(), batch, ?math, area = token.area().name(), speed_measured = token.speed_measured(), evidence = ?token.evidence, "CUDA route selected implementation=Oxide")
        }
        Selected::Library => tracing::info!(
            boundary = boundary.name(),
            batch,
            ?math,
            speed_measured = false,
            "CUDA route selected implementation=Library"
        ),
        #[cfg(all(test, feature = "_cuda-libraries"))]
        Selected::Mutant(_) => {}
    }
    Ok(selected)
}

/// Selection intent precedes artifact loading; only an Oxide token needs artifact identity
#[derive(Debug, Clone, Copy)]
enum PlanRequest {
    Hybrid,
    #[cfg_attr(not(all(test, feature = "_cuda-libraries")), allow(dead_code))]
    Production,
    #[cfg(test)]
    DriverOnly,
    #[cfg(all(test, feature = "_cuda-libraries"))]
    Qualification(Choice),
}

impl PlanRequest {
    fn resolve(
        self,
        boundary: BoundaryId,
        batch: usize,
        math: CudaMath,
        modules: impl Modules,
    ) -> Result<Selected, CudaError> {
        self.resolve_with_candidates(boundary, batch, math, modules, &driver::areas())
    }

    fn resolve_with_candidates(
        self,
        boundary: BoundaryId,
        batch: usize,
        math: CudaMath,
        mut modules: impl Modules,
        candidates: &[driver::Area],
    ) -> Result<Selected, CudaError> {
        if batch == 0 {
            return Err(CudaError::Unsupported {
                context: "CUDA selection",
                reason: SelectionError.to_string(),
            });
        }

        let driver = super::driver_only();
        #[cfg(test)]
        let driver = driver || matches!(self, Self::DriverOnly);
        if driver {
            return driver::select(boundary, batch, math, modules);
        }
        if modules.force_library() && matches!(self, Self::Production | Self::Hybrid) {
            return Ok(Selected::Library);
        }

        if matches!(self, Self::Hybrid)
            && let Some(selected) =
                driver::select_ports(boundary, batch, math, candidates, &mut modules)?
        {
            return Ok(selected);
        }

        #[cfg(all(test, feature = "_cuda-libraries"))]
        if let Self::Qualification(choice) = self {
            match choice {
                Choice::Library => return Ok(Selected::Library),
                Choice::Mutant(mutant) => return Ok(Selected::Mutant(mutant)),
                Choice::StageTail | Choice::StageTailControl if math != CudaMath::Fp32 => {
                    return Ok(Selected::Library);
                }
                Choice::Oxide(Selection::Explicit) => {
                    let Some(request) = explicit_request(&modules, boundary, batch, math)? else {
                        return Ok(Selected::Library);
                    };
                    let loaded = modules.load(request)?;
                    return qualification_selection(
                        choice,
                        boundary,
                        batch,
                        math,
                        modules.device(),
                        loaded,
                    );
                }
                _ => {}
            }
        }

        // uncovered production tuples cannot need a loaded artifact, even for diagnostics
        let device = modules.device();
        let Some(route) =
            production_route(PRODUCTION, ROUTE_PRECEDENCE, boundary, batch, math, device)
                .filter(|route| route.module.tier() <= modules.tier_limit())
        else {
            return Ok(Selected::Library);
        };
        let loaded = match modules.load(route.module) {
            Ok(loaded) => loaded,
            Err(error) => {
                return artifact_refusal(
                    error,
                    matches!(self, Self::Production | Self::Hybrid) && !super::driver_only(),
                );
            }
        };
        // a diagnostic override can load other bytes, which never match the binding
        Ok(if loaded == route.module {
            Selected::Oxide(Qualified::production(
                boundary,
                batch,
                math,
                modules.device(),
                route,
            ))
        } else {
            Selected::Library
        })
    }
}

/// Only production may use the established Library policy after an artifact refusal
fn artifact_refusal(error: CudaError, library_allowed: bool) -> Result<Selected, CudaError> {
    if library_allowed
        && matches!(
            error,
            CudaError::ArtifactLoad { .. } | CudaError::ArtifactUnavailable { .. }
        )
    {
        tracing::warn!(%error, "CUDA qualified artifact unavailable; using Library");
        return Ok(Selected::Library);
    }
    Err(error)
}

#[cfg(all(test, feature = "_cuda-libraries"))]
fn candidate_coverage(area: KernelModule, tier: PtxTier) -> Coverage {
    match area {
        KernelModule::Resnet => ConvOxide::coverage(tier),
        KernelModule::Lstm => LstmOxide::coverage(tier),
        KernelModule::Sincnet => SincOxide::coverage(tier),
        KernelModule::FbankDft => FbankOxide::coverage(tier),
        _ => Coverage::NONE,
    }
}

/// The explicit request of the highest-precedence candidate whose implemented
/// coverage declares this tuple at its best embedded tier, resolved before loading
#[cfg(all(test, feature = "_cuda-libraries"))]
fn explicit_request(
    modules: &impl Modules,
    boundary: BoundaryId,
    batch: usize,
    math: CudaMath,
) -> Result<Option<ModuleRequest>, CudaError> {
    explicit_route(boundary, batch, math, |area| {
        // areas without a port have no implemented tuple or artifact to resolve
        if matches!(
            area,
            KernelModule::Wideconv | KernelModule::Segdense | KernelModule::LstmProj
        ) {
            return Ok(None);
        }
        let request = modules.embedded_exact(area)?;
        Ok(Some((request, candidate_coverage(area, request.tier()))))
    })
}

/// Resolve explicit routes from implemented coverage before loading an artifact
#[cfg(all(test, feature = "_cuda-libraries"))]
fn explicit_route(
    boundary: BoundaryId,
    batch: usize,
    math: CudaMath,
    mut implemented: impl FnMut(KernelModule) -> Result<Option<(ModuleRequest, Coverage)>, CudaError>,
) -> Result<Option<ModuleRequest>, CudaError> {
    for area in ROUTE_PRECEDENCE {
        let Some((request, coverage)) = implemented(*area)? else {
            continue;
        };
        if coverage.covers(boundary.name(), batch, math) {
            return Ok(Some(request));
        }
    }
    Ok(None)
}

#[cfg(all(test, feature = "_cuda-libraries"))]
fn qualification_selection(
    choice: Choice,
    boundary: BoundaryId,
    batch: usize,
    math: CudaMath,
    device: &DeviceAttributes,
    loaded: ModuleRequest,
) -> Result<Selected, CudaError> {
    let coverage = candidate_coverage(loaded.area(), loaded.tier());
    Ok(match choice {
        Choice::Oxide(selection) if coverage.covers(boundary.name(), batch, math) => {
            Selected::Oxide(Qualified {
                boundary,
                batch,
                math,
                target: Target {
                    module: loaded,
                    device: device.capability(),
                },
                pin: PlanPin::Implemented,
                evidence: TokenEvidence::Qualification,
                selection,
            })
        }
        Choice::Mutant(mutant) => Selected::Mutant(mutant),
        Choice::StageTail | Choice::StageTailControl if math == CudaMath::Fp32 => {
            // isolated segmentation selectors use this owner directly, without plan_selection
            select(boundary, batch, math, device, loaded).map_err(|error| {
                CudaError::Unsupported {
                    context: "CUDA selection",
                    reason: error.to_string(),
                }
            })?
        }
        Choice::Library | Choice::Oxide(_) | Choice::StageTail | Choice::StageTailControl => {
            Selected::Library
        }
    })
}

/// Declare legacy fixture coverage without loading an otherwise unused module
///
/// This only enumerates the speed-accepted tuples of a binding whose PTX JIT pin is the
/// embedded PTX of its tier. Actual plans still need the successful loaded artifact to
/// obtain a production token through `plan_selection`
#[cfg(all(test, feature = "_cuda-libraries"))]
pub(crate) fn legacy_fixture_coverage(
    area: KernelModule,
    device: &DeviceAttributes,
    variants: AreaPtx,
) -> Vec<CoverageEntry> {
    PRODUCTION
        .iter()
        .filter(|binding| binding.area() == area && binding.scope.contains(device))
        .filter(|binding| {
            variants.embedded(binding.module.tier()).is_some_and(|ptx| {
                binding.module.artifact()
                    == LoadedArtifact::PtxJit {
                        sha256: ArtifactHash::of(ptx.text.as_bytes()),
                    }
            })
        })
        .flat_map(|binding| binding.proofs)
        .filter(|proof| {
            matches!(proof.speed, SpeedStatus::Measured(speed) if speed.scope.contains(device))
        })
        .map(|proof| CoverageEntry {
            layers: proof.boundary.name_slice(),
            batches: Batches::Only(std::slice::from_ref(&proof.batch)),
            maths: Maths::Only(std::slice::from_ref(&proof.math)),
        })
        .collect()
}

/// A required library at a specific boundary, including nested products
#[derive(Debug, Clone)]
pub(crate) struct LibraryNeed {
    boundary: BoundaryId,
    batch: usize,
    math: CudaMath,
    target: AreaTarget,
    library: CudaLibrary,
}

impl LibraryNeed {
    /// Describe required Library state without constructing a candidate artifact key
    pub(crate) fn new(
        boundary: BoundaryId,
        batch: usize,
        math: CudaMath,
        target: AreaTarget,
        library: CudaLibrary,
    ) -> Self {
        Self {
            boundary,
            batch,
            math,
            target,
            library,
        }
    }

    pub(crate) fn error(&self) -> CudaError {
        if super::driver_only() {
            return driver::missing(self.boundary, self.batch, self.math);
        }
        CudaError::NotDriverOnly {
            area: self.boundary.area().name(),
            boundary: self.boundary.name().to_owned(),
            batch: self.batch,
            math: self.math,
            tier: self.target.tier,
            device: self.target.device,
            library: self.library,
        }
    }

    /// Resolve optional libraries before buffers, forward execution or graph capture
    pub(crate) fn prepare(&self, runtime: &CudaRuntime) -> Result<(), CudaError> {
        if super::driver_only() {
            return Err(self.error());
        }
        #[cfg(feature = "_cuda-libraries")]
        {
            runtime.prepare_library(self.library)?;
            tracing::info!(boundary = self.boundary.name(), batch = self.batch,
                math = ?self.math, speed_measured = false, "CUDA route implementation=Library");
            Ok(())
        }
        #[cfg(not(feature = "_cuda-libraries"))]
        {
            let _ = runtime;
            Err(self.error())
        }
    }

    /// Prepare selected handles before model buffers, leaving NVRTC to its Library RNN plan
    pub(crate) fn prepare_handle(&self, runtime: &CudaRuntime) -> Result<(), CudaError> {
        match self.library {
            CudaLibrary::Nvrtc if !super::driver_only() => Ok(()),
            _ => self.prepare(runtime),
        }
    }
}

#[cfg(all(test, feature = "_cuda-libraries"))]
mod golden_tests;
#[cfg(all(test, feature = "_cuda-libraries"))]
mod tests;
