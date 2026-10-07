//! Driver-only routing uses complete implemented coverage, never speed acceptance

use super::{BoundaryId, Modules, PlanPin, Qualified, Selected, Selection, Target, TokenEvidence};
use crate::inference::cuda::candidate::{
    ConfigPin, ConvOxide, Coverage, DriverCandidate, FbankOxide, LstmProjOxide, PlanError,
    SegdenseArea, SincOxide,
};
use crate::inference::cuda::device::DeviceAttributes;
use crate::inference::cuda::{CudaError, CudaMath, KernelModule, PtxTier};

/// One area's library-free candidate interface, independent of artifact loading
pub(super) struct Area {
    area: KernelModule,
    coverage: fn(PtxTier) -> Coverage,
    scope: fn(usize, CudaMath, &DeviceAttributes) -> Option<super::SpeedScope>,
    summary: fn(CudaMath) -> &'static str,
    pin:
        fn(BoundaryId, usize, CudaMath, &DeviceAttributes, PtxTier) -> Result<ConfigPin, PlanError>,
}

impl Area {
    /// Register a complete port; library-dependent legacy candidates do not register
    pub(super) fn candidate<C: DriverCandidate>() -> Self {
        Self {
            area: C::AREA,
            coverage: C::driver_coverage,
            scope: C::speed_scope,
            summary: C::speed_summary,
            pin: C::driver_pin,
        }
    }
}

// ports add registrations here once their complete boundary is library-free
pub(super) fn areas() -> [Area; 5] {
    [
        Area::candidate::<ConvOxide>(),
        Area::candidate::<SincOxide>(),
        Area::candidate::<SegdenseArea>(),
        Area::candidate::<FbankOxide>(),
        Area::candidate::<LstmProjOxide>(),
    ]
}

pub(super) fn missing(boundary: BoundaryId, batch: usize, math: CudaMath) -> CudaError {
    CudaError::MissingKernel {
        boundary: boundary.name().to_owned(),
        batch,
        math,
    }
}

pub(super) fn select(
    boundary: BoundaryId,
    batch: usize,
    math: CudaMath,
    mut modules: impl Modules,
) -> Result<Selected, CudaError> {
    select_from(
        &areas(),
        boundary,
        batch,
        math,
        &mut modules,
        Selection::DriverOnly,
    )
    .map(|selected| selected.expect("driver route returns a selection or an error"))
}

pub(super) fn uses_port_artifact(area: KernelModule) -> bool {
    areas().iter().any(|candidate| {
        candidate.area == area
            && matches!(
                area,
                KernelModule::FbankDft | KernelModule::Segdense | KernelModule::LstmProj
            )
    })
}

pub(super) fn select_ports(
    boundary: BoundaryId,
    batch: usize,
    math: CudaMath,
    candidates: &[Area],
    modules: &mut impl Modules,
) -> Result<Option<Selected>, CudaError> {
    select_from(
        candidates,
        boundary,
        batch,
        math,
        modules,
        Selection::Production,
    )
}

pub(super) fn select_from(
    areas: &[Area],
    boundary: BoundaryId,
    batch: usize,
    math: CudaMath,
    modules: &mut impl Modules,
    selection: Selection,
) -> Result<Option<Selected>, CudaError> {
    for area in super::ROUTE_PRECEDENCE {
        let Some(candidate) = areas.iter().find(|candidate| candidate.area == *area) else {
            continue;
        };

        let Some(request) = area.variants().driver_request(
            *area,
            modules.tier_limit(),
            modules.device().capability(),
        ) else {
            continue;
        };
        if !(candidate.coverage)(request.tier()).covers(boundary.name(), batch, math) {
            continue;
        }
        let scope = (candidate.scope)(batch, math, modules.device())
            .filter(|scope| scope.contains(modules.device()) && scope.allows_tier(request.tier()));
        if selection == Selection::Production && scope.is_none() {
            // a complete port owns its covered tuple even when speed is unmeasured
            return Ok(Some(Selected::Library));
        }
        let pin = (candidate.pin)(boundary, batch, math, modules.device(), request.tier());
        if selection == Selection::Production
            && matches!(
                &pin,
                Err(PlanError::DeviceUnsupported { .. }
                    | PlanError::WeightsOutOfContract { .. }
                    | PlanError::Geometry(
                        crate::inference::cuda::GeometryError::Unimplemented { .. }
                    ))
            )
        {
            return Ok(Some(Selected::Library));
        }
        let pin = pin.map_err(|error| match error {
            PlanError::Cuda(error) => error,
            PlanError::Geometry(error) => CudaError::CandidateGeometry {
                area: area.name(),
                boundary: boundary.name().to_owned(),
                batch,
                math,
                error,
            },
            other => CudaError::CandidateDeviceUnsupported {
                area: area.name(),
                boundary: boundary.name().to_owned(),
                batch,
                math,
                tier: request.tier(),
                device: modules.device().capability(),
                reason: other.to_string(),
            },
        })?;
        if pin.area() != *area {
            return Err(CudaError::Unsupported {
                context: "driver-only route",
                reason: "candidate returned a foreign area pin".to_owned(),
            });
        }
        let loaded = match modules.load(request) {
            Ok(loaded) => loaded,
            Err(error) => {
                return super::artifact_refusal(error, selection == Selection::Production)
                    .map(Some);
            }
        };
        if loaded != request {
            return Err(CudaError::Unsupported {
                context: "driver-only route",
                reason: "loaded artifact differs from selected artifact".to_owned(),
            });
        }
        return Ok(Some(Selected::Oxide(Qualified {
            boundary,
            batch,
            math,
            target: Target {
                module: loaded,
                device: modules.device().capability(),
            },
            pin: PlanPin::Pinned(pin),
            evidence: scope.map_or(TokenEvidence::Implemented, |scope| TokenEvidence::Port {
                scope,
                summary: (candidate.summary)(math),
            }),
            selection,
        })));
    }
    if selection == Selection::Production {
        return Ok(None);
    }
    Err(missing(boundary, batch, math))
}

#[cfg(test)]
mod tests;
