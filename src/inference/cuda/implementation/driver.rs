//! Driver-only routing uses complete implemented coverage, never speed acceptance

use super::policy::{DeviceDefault, Recipe, RecipeChoice};
use super::{BoundaryId, Modules, PlanPin, Qualified, Selected, Selection, Target, TokenEvidence};
use crate::inference::cuda::candidate::{
    ConfigPin, ConvOxide, Coverage, DriverCandidate, FbankOxide, LstmProjOxide, PlanError,
    SegdenseArea, SincOxide, WideconvOxide,
};
use crate::inference::cuda::device::DeviceAttributes;
use crate::inference::cuda::{CudaError, CudaMath, KernelModule, PtxTier};

/// Candidate-owned scalar pin construction for the requested pipeline tuple
type TuningFp32Pin = fn(
    BoundaryId,
    usize,
    CudaMath,
    &DeviceAttributes,
    PtxTier,
) -> Result<Option<ConfigPin>, PlanError>;

/// One area's library-free candidate interface, independent of artifact loading
pub(super) struct Area {
    area: KernelModule,
    hybrid: HybridPolicy,
    coverage: fn(PtxTier, &DeviceAttributes) -> Coverage,
    scope: fn(BoundaryId, usize, CudaMath, &DeviceAttributes, PtxTier) -> Option<super::SpeedScope>,
    summary: fn(CudaMath) -> &'static str,
    pin:
        fn(BoundaryId, usize, CudaMath, &DeviceAttributes, PtxTier) -> Result<ConfigPin, PlanError>,
    tuning_fp32_pin: TuningFp32Pin,
}

/// A port either owns speed selection or retains the frozen qualified table
#[derive(Clone, Copy, PartialEq, Eq)]
enum HybridPolicy {
    Scoped,
    QualifiedTable,
}

impl Area {
    /// Register a complete port; library-dependent legacy candidates do not register
    pub(super) fn candidate<C: DriverCandidate>() -> Self {
        Self {
            area: C::AREA,
            hybrid: HybridPolicy::Scoped,
            coverage: C::driver_coverage,
            scope: C::speed_scope,
            summary: C::speed_summary,
            pin: C::driver_pin,
            tuning_fp32_pin: C::tuning_fp32_pin,
        }
    }

    /// Register implementation coverage without replacing qualified-table speed policy
    fn qualified<C: DriverCandidate>() -> Self {
        Self {
            hybrid: HybridPolicy::QualifiedTable,
            ..Self::candidate::<C>()
        }
    }
}

// ports add registrations here once their complete boundary is library-free
pub(super) fn areas() -> [Area; 6] {
    [
        Area::candidate::<WideconvOxide>(),
        Area::candidate::<ConvOxide>(),
        Area::qualified::<SincOxide>(),
        Area::candidate::<SegdenseArea>(),
        Area::candidate::<FbankOxide>(),
        Area::candidate::<LstmProjOxide>(),
    ]
}

/// Enumerate fixed port pins without treating implemented coverage as approval
pub(super) fn tuning_configurations(
    device: &DeviceAttributes,
    limit: PtxTier,
) -> Result<Vec<super::TuningConfiguration>, CudaError> {
    let mut configurations = Vec::new();
    for area in areas() {
        let Some(module) =
            super::production_module(area.area, device, limit, area.area.variants())?
        else {
            continue;
        };
        for boundary in
            BoundaryId::all().filter(|id| *id != BoundaryId::named("lstm.stack.input_proj"))
        {
            let batches = boundary.batches().iter();
            for batch in batches {
                for math in [CudaMath::Fp32, CudaMath::Tf32] {
                    // sincnet tf32 has no retained accuracy evidence for tuning
                    if area.area == KernelModule::Sincnet && math == CudaMath::Tf32 {
                        continue;
                    }
                    if !(area.coverage)(module.tier(), device).covers(boundary.name(), batch, math)
                    {
                        continue;
                    }
                    let Ok(pin) = (area.pin)(boundary, batch, math, device, module.tier()) else {
                        continue;
                    };
                    configurations.push((boundary, batch, math, module, pin, "default"));
                    // only the candidate owner can construct a pin for this math mode;
                    // the independent policy still has to approve its arithmetic
                    if let Ok(Some(fp32)) =
                        (area.tuning_fp32_pin)(boundary, batch, math, device, module.tier())
                        && fp32 != pin
                    {
                        configurations.push((boundary, batch, math, module, fp32, "fp32"));
                    }
                }
            }
        }
    }
    Ok(configurations)
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
    let recipe = (selection == Selection::Production)
        .then(|| {
            Recipe::select(
                boundary,
                batch,
                math,
                modules.device(),
                modules.recipe_mode(),
            )
            .filter(|recipe| recipe.allows_tier_limit(modules.tier_limit()))
        })
        .flatten();
    let recipe_choice = recipe.map(|recipe| recipe.choice(boundary, batch, math));
    if recipe_choice == Some(RecipeChoice::Library) {
        return Ok(Some(Selected::Library));
    }

    for area in super::ROUTE_PRECEDENCE {
        let Some(candidate) = areas.iter().find(|candidate| candidate.area == *area) else {
            continue;
        };

        if selection == Selection::Production
            && candidate.hybrid == HybridPolicy::QualifiedTable
            && recipe.is_none()
        {
            continue;
        }
        let Some(request) = super::production_module(
            *area,
            modules.device(),
            modules.tier_limit(),
            area.variants(),
        )?
        else {
            continue;
        };
        if !(candidate.coverage)(request.tier(), modules.device()).covers(
            boundary.name(),
            batch,
            math,
        ) {
            continue;
        }
        let scope = (candidate.scope)(boundary, batch, math, modules.device(), request.tier())
            .filter(|scope| scope.contains(modules.device()) && scope.allows_tier(request.tier()));
        let default = (selection == Selection::Production)
            .then(|| DeviceDefault::select(*area, batch, math, modules.device(), request.tier()))
            .flatten();
        if selection == Selection::Production
            && recipe.is_none()
            && scope.is_none()
            && default.is_none()
        {
            // a complete port owns its covered tuple even when speed is unmeasured
            return Ok(Some(Selected::Library));
        }
        let pin = match recipe_choice {
            Some(RecipeChoice::FixedPin(pin)) => Ok(pin),
            _ => (candidate.pin)(boundary, batch, math, modules.device(), request.tier()),
        };
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
        return Ok(Some(Selected::Oxide(Box::new(Qualified {
            boundary,
            batch,
            math,
            target: Target {
                module: loaded,
                device: modules.device().capability(),
            },
            pin: PlanPin::Pinned(pin),
            evidence: if selection == Selection::Production
                && let Some(recipe) = recipe
            {
                TokenEvidence::Recipe(recipe)
            } else if let Some(scope) = scope {
                TokenEvidence::Port {
                    scope,
                    summary: (candidate.summary)(math),
                }
            } else if let Some(default) = default {
                TokenEvidence::DeviceDefault(default)
            } else {
                TokenEvidence::Implemented
            },
            selection,
        }))));
    }
    if selection == Selection::Production {
        return Ok(None);
    }
    Err(missing(boundary, batch, math))
}

#[cfg(test)]
mod tests;
