//! Locked dispatch of plans selected before any forward execution

#[cfg(feature = "_cuda-libraries")]
use std::rc::Rc;

use cudarc::driver::{CudaView, CudaViewMut};

use super::super::candidate::{
    ConvCandidate, ConvInputs, ConvLayerSpec, ConvOxide, Phases, WideconvOxide,
};
#[cfg(feature = "_cuda-libraries")]
use super::super::dnn::{ConvPlan, ConvPlanner};
use super::super::geometry::Residual;
use super::super::implementation::{AreaTarget, LibraryNeed, Selected, plan_selection};
use super::super::{CudaLibrary, CudaMath, CudaRuntime, KernelModule};
use super::trunk::{ConvLayer, Trunk};
use super::{Convs, CudaError};

/// Exactly one implementation for each convolution
#[derive(Debug)]
pub(super) enum Plan {
    #[cfg(feature = "_cuda-libraries")]
    Library(Rc<ConvPlan>),
    Oxide(ConvOxide),
    Wideconv(WideconvOxide),
    #[cfg(all(test, feature = "_cuda-libraries"))]
    Mutant {
        library: Rc<ConvPlan>,
        mutant: super::super::test_support::Mutant,
    },
}

impl Plan {
    pub(super) fn workspace_bytes(&self) -> usize {
        match self {
            #[cfg(feature = "_cuda-libraries")]
            Self::Library(plan) => plan.workspace_bytes(),
            // partitioned wideconv plans own their partial-sum planes
            Self::Oxide(_) | Self::Wideconv(_) => 0,
            #[cfg(all(test, feature = "_cuda-libraries"))]
            Self::Mutant { library, .. } => library.workspace_bytes(),
        }
    }

    #[cfg(all(test, feature = "_cuda-libraries"))]
    pub(super) fn choice(&self) -> super::super::implementation::Choice {
        use super::super::implementation::Choice;
        match self {
            Self::Library(_) => Choice::Library,
            Self::Oxide(_) | Self::Wideconv(_) => {
                Choice::Oxide(super::super::implementation::Selection::Explicit)
            }
            Self::Mutant { mutant, .. } => Choice::Mutant(*mutant),
        }
    }

    #[cfg(feature = "_cuda-libraries")]
    pub(super) fn library(&self) -> Result<&ConvPlan, CudaError> {
        match self {
            Self::Library(plan) => Ok(plan),
            #[cfg(all(test, feature = "_cuda-libraries"))]
            Self::Mutant { library, .. } => Ok(library),
            Self::Oxide(_) | Self::Wideconv(_) => Err(CudaError::Unsupported {
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
        if let Some(probe) = self.range_probe {
            probe.observe(self.runtime, layer, self.chunks, x, false)?;
        }
        #[cfg(all(test, feature = "_cuda-libraries"))]
        if let Some(owner) = self
            .qualification
            .and_then(|owners| owners.get(layer.name()))
        {
            let add = match &residual {
                Residual::Add(value) => Some(value.slice(..)),
                Residual::None { .. } => None,
            };
            let operation = super::super::test_support::candidate_seam::Operation::Spatial {
                boundary: layer.boundary(),
                conv: layer.conv(self.chunks, self.math),
                epilogue: layer.epilogue(add.is_some()),
            };
            return owner.run(
                self.runtime,
                operation,
                super::super::test_support::candidate_seam::Views {
                    input: x.slice(..),
                    weight: layer.weight().data().as_view(),
                    bias: Some(layer.bias().data().as_view()),
                    residual: add,
                },
                y,
                |y| {
                    let library = self.plan(layer)?.library()?;
                    let residual = match &residual {
                        Residual::Add(value) => Residual::Add(value),
                        Residual::None { scratch } => Residual::None { scratch },
                    };
                    library.forward_bias_relu(
                        &mut self.workspace.as_view_mut(),
                        x,
                        &layer.weight().data().as_view(),
                        &layer.bias().data().as_view(),
                        residual,
                        y,
                    )
                },
            );
        }
        match self.plan(layer)? {
            #[cfg(feature = "_cuda-libraries")]
            Plan::Library(plan) => {
                #[cfg(all(test, feature = "_cuda-libraries"))]
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
                #[cfg(all(test, feature = "_cuda-libraries"))]
                super::super::test_support::perturb(self.runtime, layer.name(), y)?;
                Ok(())
            }
            Plan::Oxide(plan) => self.candidate(plan, layer, x, residual, y),
            Plan::Wideconv(plan) => self.candidate(plan, layer, x, residual, y),
            #[cfg(all(test, feature = "_cuda-libraries"))]
            Plan::Mutant { mutant, .. } => {
                let mutant = *mutant;
                super::super::test_support::poison(self.runtime)?;
                {
                    let _scope = super::super::test_support::mutant_scope(
                        self.runtime.stream(),
                        layer.name(),
                        mutant,
                    );
                    if !mutant.skips() {
                        super::test_support::mutant_conv(self, layer, x, residual, y, mutant)?;
                    }
                }
                super::super::test_support::unscoped(self.runtime, mutant)
            }
        }
    }
}

impl Convs<'_> {
    /// `y = conv(x) + bias` for a 1x1 shortcut: Library runs the bare convolution and
    /// the shared bias kernel, a candidate its fused bias epilogue
    pub(super) fn shortcut(
        &mut self,
        layer: &ConvLayer,
        x: &CudaView<'_, f32>,
        y: &mut CudaViewMut<'_, f32>,
    ) -> Result<(), CudaError> {
        if let Some(probe) = self.range_probe {
            probe.observe(self.runtime, layer, self.chunks, x, false)?;
        }
        match self.plan(layer)? {
            Plan::Wideconv(plan) => {
                let none = Residual::None {
                    #[cfg(feature = "_cuda-libraries")]
                    scratch: x,
                };
                self.candidate(plan, layer, x, none, y)
            }
            _ => {
                #[cfg(all(test, feature = "_cuda-libraries"))]
                {
                    self.conv_bias(layer, x, y)
                }
                #[cfg(not(all(test, feature = "_cuda-libraries")))]
                {
                    self.conv(layer, x, y)?;
                    self.bias(layer, y)
                }
            }
        }
    }

    /// Enqueue a candidate plan with the layer's weights; a candidate never reads the
    /// Library scratch residual
    fn candidate(
        &self,
        plan: &impl ConvCandidate,
        layer: &ConvLayer,
        x: &CudaView<'_, f32>,
        residual: Residual<'_, '_>,
        y: &mut CudaViewMut<'_, f32>,
    ) -> Result<(), CudaError> {
        let residual = match residual {
            Residual::Add(value) => Some(value),
            Residual::None { .. } => None,
        };
        let stream = self.runtime.stream();
        #[cfg(all(test, feature = "_cuda-libraries"))]
        super::super::test_support::poison(self.runtime)?;
        #[cfg(all(test, feature = "_cuda-libraries"))]
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
}

/// Select each layer first; deduplicate only the Library shapes that were selected
pub(super) fn plan_layers(
    runtime: &CudaRuntime,
    trunk: &Trunk,
    batch: usize,
    math: CudaMath,
) -> Result<Vec<(String, Plan)>, CudaError> {
    #[cfg(feature = "_cuda-libraries")]
    let mut library_plans: Vec<Rc<ConvPlan>> = Vec::new();
    #[cfg(feature = "_cuda-libraries")]
    let mut planner = None;
    let mut plans = Vec::new();
    for (layer, residual) in trunk.layers() {
        let selected = plan_selection(
            runtime,
            layer.boundary(),
            batch,
            math,
            #[cfg(all(test, feature = "_cuda-libraries"))]
            layer.override_choice(),
        )?;
        let selected = match selected {
            Selected::Oxide(token) => {
                #[cfg(all(test, feature = "_cuda-libraries"))]
                let _scope = super::super::test_support::plan(layer.name());
                let spec = ConvLayerSpec {
                    name: layer.name(),
                    conv: layer.conv(batch, math),
                    epilogue: layer.epilogue(residual),
                    weight: layer.weight().data(),
                    bias: layer.bias().data(),
                };
                let plan = match token.area() {
                    KernelModule::Wideconv => token.wideconv(runtime, spec)?.map(Plan::Wideconv),
                    _ => token.conv(runtime, spec)?.map(Plan::Oxide),
                };
                if let Some(plan) = plan {
                    plans.push((layer.name().to_owned(), plan));
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
        #[cfg(feature = "_cuda-libraries")]
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
                #[cfg(all(test, feature = "_cuda-libraries"))]
                Selected::Mutant(mutant) => Plan::Mutant { library, mutant },
                _ => Plan::Library(library),
            };
            plans.push((layer.name().to_owned(), plan));
        }
        #[cfg(not(feature = "_cuda-libraries"))]
        {
            let _ = selected;
            return Err(CudaError::LibraryUnavailable {
                library: CudaLibrary::Cudnn,
            });
        }
    }
    Ok(plans)
}
