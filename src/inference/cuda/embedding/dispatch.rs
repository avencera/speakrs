//! Locked dispatch of plans selected before any forward execution

#[cfg(feature = "cuda")]
use std::rc::Rc;

use cudarc::driver::{CudaView, CudaViewMut};

use super::super::candidate::{ConvCandidate, ConvInputs, ConvLayerSpec, ConvOxide, Phases};
use super::super::dnn::Residual;
#[cfg(feature = "cuda")]
use super::super::dnn::{ConvPlan, ConvPlanner};
use super::super::implementation::{AreaTarget, LibraryNeed, Selected, plan_selection};
use super::super::{CudaLibrary, CudaMath, CudaRuntime, KernelModule};
use super::trunk::{ConvLayer, Trunk};
use super::{Convs, CudaError};

/// Exactly one implementation for each convolution
#[derive(Debug)]
pub(super) enum Plan {
    #[cfg(feature = "cuda")]
    Library(Rc<ConvPlan>),
    Oxide(ConvOxide),
    #[cfg(all(test, feature = "cuda", not(feature = "cuda-driver-only")))]
    Mutant {
        library: Rc<ConvPlan>,
        mutant: super::super::test_support::Mutant,
    },
}

impl Plan {
    pub(super) fn workspace_bytes(&self) -> usize {
        match self {
            #[cfg(feature = "cuda")]
            Self::Library(plan) => plan.workspace_bytes(),
            Self::Oxide(_) => 0,
            #[cfg(all(test, feature = "cuda", not(feature = "cuda-driver-only")))]
            Self::Mutant { library, .. } => library.workspace_bytes(),
        }
    }

    #[cfg(all(test, feature = "cuda", not(feature = "cuda-driver-only")))]
    pub(super) fn choice(&self) -> super::super::implementation::Choice {
        use super::super::implementation::Choice;
        match self {
            Self::Library(_) => Choice::Library,
            Self::Oxide(_) => Choice::Oxide(super::super::implementation::Selection::Explicit),
            Self::Mutant { mutant, .. } => Choice::Mutant(*mutant),
        }
    }

    #[cfg(feature = "cuda")]
    pub(super) fn library(&self) -> Result<&ConvPlan, CudaError> {
        match self {
            Self::Library(plan) => Ok(plan),
            #[cfg(all(test, feature = "cuda", not(feature = "cuda-driver-only")))]
            Self::Mutant { library, .. } => Ok(library),
            Self::Oxide(_) => Err(CudaError::Unsupported {
                context: "convolution plan",
                reason: "Oxide plan has no Library state".to_owned(),
            }),
        }
    }
}

impl Convs<'_> {
    /// Run only the owner built for this layer and batch class
    pub(super) fn conv_bias_relu(
        &mut self,
        layer: &ConvLayer,
        x: &CudaView<'_, f32>,
        residual: Residual<'_, '_>,
        y: &mut CudaViewMut<'_, f32>,
    ) -> Result<(), CudaError> {
        match self.plan(layer)? {
            #[cfg(feature = "cuda")]
            Plan::Library(plan) => {
                #[cfg(all(test, feature = "cuda", not(feature = "cuda-driver-only")))]
                let _scope = layer.eligible().then(|| {
                    super::super::test_support::library(self.runtime.stream(), layer.name())
                });
                plan.forward_bias_relu(
                    &mut self.workspace.as_view_mut(),
                    x,
                    &layer.weight().data().as_view(),
                    &layer.bias().data().as_view(),
                    residual,
                    y,
                )?;
                #[cfg(all(test, feature = "cuda", not(feature = "cuda-driver-only")))]
                super::super::test_support::perturb(self.runtime, layer.name(), y)?;
                Ok(())
            }
            Plan::Oxide(plan) => {
                let residual = match residual {
                    Residual::Add(value) => Some(value),
                    Residual::None { .. } => None,
                };
                let stream = self.runtime.stream();
                #[cfg(all(test, feature = "cuda", not(feature = "cuda-driver-only")))]
                super::super::test_support::poison(self.runtime)?;
                #[cfg(all(test, feature = "cuda", not(feature = "cuda-driver-only")))]
                let _scope = super::super::test_support::candidate(stream, layer.name());
                plan.enqueue(
                    ConvInputs {
                        x,
                        residual,
                        weight: &layer.weight().data().as_view(),
                        bias: &layer.bias().data().as_view(),
                    },
                    y,
                    &Phases::new(),
                    stream,
                )
            }
            #[cfg(all(test, feature = "cuda", not(feature = "cuda-driver-only")))]
            Plan::Mutant { mutant, .. } => {
                let mutant = *mutant;
                super::super::test_support::poison(self.runtime)?;
                {
                    let _scope =
                        super::super::test_support::candidate(self.runtime.stream(), layer.name());
                    if !mutant.skips() {
                        super::test_support::mutant_conv(self, layer, x, residual, y, mutant)?;
                    }
                }
                super::super::test_support::unscoped(self.runtime, mutant)
            }
        }
    }
}

/// Select each layer first; deduplicate only the Library shapes that were selected
pub(super) fn plan_layers(
    runtime: &CudaRuntime,
    trunk: &Trunk,
    batch: usize,
    math: CudaMath,
) -> Result<Vec<(String, Plan)>, CudaError> {
    #[cfg(feature = "cuda")]
    let mut library_plans: Vec<Rc<ConvPlan>> = Vec::new();
    #[cfg(feature = "cuda")]
    let mut planner = None;
    let mut plans = Vec::new();
    for (layer, residual) in trunk.layers() {
        let selected = plan_selection(
            runtime,
            layer.boundary(),
            batch,
            math,
            #[cfg(all(test, feature = "cuda", not(feature = "cuda-driver-only")))]
            layer.override_choice(),
        )?;
        let selected = match selected {
            Selected::Oxide(token) => {
                #[cfg(all(test, feature = "cuda", not(feature = "cuda-driver-only")))]
                let _scope = super::super::test_support::plan(layer.name());
                if let Some(plan) = token.conv(
                    runtime,
                    ConvLayerSpec {
                        name: layer.name(),
                        conv: layer.conv(batch, math),
                        residual,
                        weight: layer.weight().data(),
                        bias: layer.bias().data(),
                    },
                )? {
                    plans.push((layer.name().to_owned(), Plan::Oxide(plan)));
                    continue;
                }
                Selected::Library
            }
            other => other,
        };
        LibraryNeed::new(
            layer.boundary(),
            batch,
            math,
            AreaTarget::for_area(runtime, KernelModule::Resnet)?,
            CudaLibrary::Cudnn,
        )
        .prepare(runtime)?;
        #[cfg(feature = "cuda")]
        {
            if planner.is_none() {
                planner = Some(ConvPlanner::new(runtime)?);
            }
            let spec = layer.conv(batch, math);
            let existing = library_plans.iter().find(|plan| *plan.spec() == spec);
            let library = match existing {
                Some(plan) => Rc::clone(plan),
                None => {
                    let plan = Rc::new(planner.as_ref().expect("planner initialized").plan(spec)?);
                    library_plans.push(Rc::clone(&plan));
                    plan
                }
            };
            let plan = match selected {
                #[cfg(all(test, feature = "cuda", not(feature = "cuda-driver-only")))]
                Selected::Mutant(mutant) => Plan::Mutant { library, mutant },
                _ => Plan::Library(library),
            };
            plans.push((layer.name().to_owned(), plan));
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = selected;
            return Err(CudaError::LibraryUnavailable {
                library: CudaLibrary::Cudnn,
            });
        }
    }
    Ok(plans)
}
