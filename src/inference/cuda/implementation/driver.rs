//! Kernel routing uses complete implemented coverage, never speed acceptance

use super::policy::{DeviceDefault, Recipe};
use super::{BoundaryId, Modules, PlanPin, Qualified, Selected, Selection, Target, TokenEvidence};
use crate::inference::cuda::candidate::{
    ConfigPin, ConvOxide, Coverage, DriverCandidate, FbankOxide, Fp16Policy, LstmProjOxide,
    PlanError, SegdenseArea, SincOxide, WideconvOxide,
};
use crate::inference::cuda::device::DeviceAttributes;
use crate::inference::cuda::{CudaError, CudaMath, KernelModule, PtxTier};

/// Candidate-owned distinct tuning choices, independent of startup routing
type TuningPins =
    fn(BoundaryId, usize, CudaMath, &DeviceAttributes, PtxTier) -> Vec<(ConfigPin, &'static str)>;

/// Candidate-owned scalar alternative for independent production accuracy checks
type TuningFp32Pin = fn(
    BoundaryId,
    usize,
    CudaMath,
    &DeviceAttributes,
    PtxTier,
) -> Result<Option<ConfigPin>, PlanError>;

/// Candidate-owned pin construction under a selection's FP16 policy
type DriverPin = fn(
    BoundaryId,
    usize,
    CudaMath,
    &DeviceAttributes,
    PtxTier,
    Fp16Policy,
) -> Result<ConfigPin, PlanError>;

/// Candidate-owned operand policy shared by hybrid coverage and pin selection
type HybridFp16 = fn(&DeviceAttributes, PtxTier, Option<Recipe>, Fp16Policy) -> Fp16Policy;

/// One area's library-free candidate interface, independent of artifact loading
pub(super) struct Area {
    area: KernelModule,
    coverage: fn(PtxTier, &DeviceAttributes, Fp16Policy) -> Coverage,
    hybrid_fp16: HybridFp16,
    scope: fn(BoundaryId, usize, CudaMath, &DeviceAttributes, PtxTier) -> Option<super::SpeedScope>,
    summary: fn(CudaMath) -> &'static str,
    pin: DriverPin,
    tuning_pins: TuningPins,
    tuning_fp32_pin: TuningFp32Pin,
}

impl Area {
    /// Register a complete port; library-dependent legacy candidates do not register
    pub(super) fn candidate<C: DriverCandidate>() -> Self {
        Self {
            area: C::AREA,
            coverage: C::driver_coverage,
            hybrid_fp16: C::hybrid_fp16,
            scope: C::speed_scope,
            summary: C::speed_summary,
            pin: C::driver_pin,
            tuning_pins: C::tuning_pins,
            tuning_fp32_pin: C::tuning_fp32_pin,
        }
    }
}

// ports add registrations here once their complete boundary is library-free
pub(super) fn areas() -> [Area; 6] {
    [
        Area::candidate::<WideconvOxide>(),
        Area::candidate::<ConvOxide>(),
        Area::candidate::<SincOxide>(),
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
                    // implementation support belongs to the candidate; accuracy
                    // approval remains an independent catalogue step
                    for (pin, family) in
                        (area.tuning_pins)(boundary, batch, math, device, module.tier())
                    {
                        configurations.push((boundary, batch, math, module, pin, family));
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

/// Apply production default accuracy policy without permitting a Library result
pub(super) fn select_default(
    boundary: BoundaryId,
    batch: usize,
    math: CudaMath,
    mut modules: impl Modules,
) -> Result<Selected, CudaError> {
    match select_from(
        &areas(),
        boundary,
        batch,
        math,
        &mut modules,
        Selection::Production,
    )? {
        Some(selected @ Selected::Oxide(_)) => Ok(selected),
        _ => Err(missing(boundary, batch, math)),
    }
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
    let fp16 = modules.fp16();
    for area in super::ROUTE_PRECEDENCE {
        let Some(candidate) = areas.iter().find(|candidate| candidate.area == *area) else {
            continue;
        };

        let Some(request) = super::production_module(
            *area,
            modules.device(),
            modules.tier_limit(),
            area.variants(),
        )?
        else {
            continue;
        };
        let selected_request = request;
        let request = modules.effective_request(request)?;
        let fp16 = if selection == Selection::Production {
            (candidate.hybrid_fp16)(modules.device(), request.tier(), recipe, fp16)
        } else {
            fp16
        };
        if !(candidate.coverage)(request.tier(), modules.device(), fp16).covers(
            boundary.name(),
            batch,
            math,
        ) {
            continue;
        }
        let scope = (candidate.scope)(boundary, batch, math, modules.device(), request.tier())
            .filter(|scope| scope.contains(modules.device()) && scope.allows_tier(request.tier()));
        let default = (selection == Selection::Production)
            .then(|| {
                DeviceDefault::select(
                    *area,
                    boundary,
                    batch,
                    math,
                    modules.device(),
                    request.tier(),
                    fp16,
                )
            })
            .flatten();
        let pin = recipe
            .and_then(|recipe| recipe.fixed_pin(boundary, batch, math, fp16))
            .map_or_else(
                || {
                    (candidate.pin)(
                        boundary,
                        batch,
                        math,
                        modules.device(),
                        request.tier(),
                        fp16,
                    )
                },
                Ok,
            );
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
        let pin_error = |error| match error {
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
        };
        let mut pin = pin.map_err(pin_error)?;

        if selection == Selection::Production
            && Recipe::accuracy_exception(
                boundary,
                batch,
                math,
                pin,
                modules.device(),
                modules.tier_limit(),
            )
            .is_none()
            && crate::inference::cuda::tuning::accuracy::Policy::approve(boundary, math, pin)
                .is_none()
        {
            // unlisted recipe tuples and class defaults need independent approval
            let alternate = (candidate.tuning_fp32_pin)(
                boundary,
                batch,
                math,
                modules.device(),
                request.tier(),
            )
            .map_err(pin_error)?;
            let Some(approved) = alternate else {
                continue;
            };
            if crate::inference::cuda::tuning::accuracy::Policy::approve(boundary, math, approved)
                .is_none()
            {
                continue;
            }

            pin = approved;
        }
        if pin.area() != *area {
            return Err(CudaError::Unsupported {
                context: "driver-only route",
                reason: "candidate returned a foreign area pin".to_owned(),
            });
        }
        let loaded = modules.load(request)?;
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
            evidence: if request != selected_request {
                // measured evidence describes the selected artifact, not forced JIT
                TokenEvidence::Implemented
            } else if selection == Selection::Production
                && let Some(recipe) = recipe
            {
                TokenEvidence::Recipe(recipe)
            } else if let Some(scope) = scope {
                TokenEvidence::non_fp16_port(scope, (candidate.summary)(math), pin)
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
pub(super) mod test_support;
#[cfg(test)]
mod tests;
