//! Driver-only routing uses complete implemented coverage, never speed acceptance

use super::{BoundaryId, Modules, PlanPin, Qualified, Selected, Selection, Target, TokenEvidence};
use crate::inference::cuda::candidate::{
    ConfigPin, ConvOxide, Coverage, DriverCandidate, PlanError, SincOxide,
};
use crate::inference::cuda::device::DeviceAttributes;
use crate::inference::cuda::{CudaError, CudaMath, KernelModule, PtxTier};

/// One area's library-free candidate interface, independent of artifact loading
pub(super) struct Area {
    area: KernelModule,
    coverage: fn(PtxTier) -> Coverage,
    broad: Option<&'static super::BroadEvidence>,
    pin:
        fn(BoundaryId, usize, CudaMath, &DeviceAttributes, PtxTier) -> Result<ConfigPin, PlanError>,
}

impl Area {
    /// Register a complete port; library-dependent legacy candidates do not register
    pub(super) fn candidate<C: DriverCandidate>() -> Self {
        Self {
            area: C::AREA,
            coverage: C::driver_coverage,
            broad: C::broad_evidence(),
            pin: C::driver_pin,
        }
    }
}

// ports add registrations here once their complete boundary is library-free
fn areas() -> [Area; 2] {
    [
        Area::candidate::<ConvOxide>(),
        Area::candidate::<SincOxide>(),
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
}

pub(super) fn is_broad(area: KernelModule) -> bool {
    areas()
        .iter()
        .any(|candidate| candidate.area == area && candidate.broad.is_some())
}

pub(super) fn select_broad(
    boundary: BoundaryId,
    batch: usize,
    math: CudaMath,
    modules: &mut impl Modules,
) -> Result<Selected, CudaError> {
    select_from(
        &areas(),
        boundary,
        batch,
        math,
        modules,
        Selection::Production,
    )
}

fn select_from(
    areas: &[Area],
    boundary: BoundaryId,
    batch: usize,
    math: CudaMath,
    modules: &mut impl Modules,
    selection: Selection,
) -> Result<Selected, CudaError> {
    for area in super::ROUTE_PRECEDENCE {
        let Some(candidate) = areas.iter().find(|candidate| candidate.area == *area) else {
            continue;
        };
        if selection == Selection::Production && candidate.broad.is_none() {
            continue;
        }
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
            return Ok(Selected::Library);
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
                return super::artifact_refusal(error, selection == Selection::Production);
            }
        };
        if loaded != request {
            return Err(CudaError::Unsupported {
                context: "driver-only route",
                reason: "loaded artifact differs from selected artifact".to_owned(),
            });
        }
        return Ok(Selected::Oxide(Qualified {
            boundary,
            batch,
            math,
            target: Target {
                module: loaded,
                device: modules.device().capability(),
            },
            pin: PlanPin::Pinned(pin),
            evidence: candidate
                .broad
                .map_or(TokenEvidence::Implemented, |summary| TokenEvidence::Broad {
                    scope: super::SpeedScope::AllDevices(summary),
                }),
            selection,
        }));
    }
    if selection == Selection::Production {
        return Ok(Selected::Library);
    }
    Err(missing(boundary, batch, math))
}

#[cfg(test)]
mod tests;
