//! Test-only owners for new operation and stage collection paths

use super::candidate_seam::{self, Operation, RuntimePlan, Slices, Views};
use cudarc::driver::{CudaSlice, CudaViewMut, DevicePtr, DevicePtrMut};

use super::{Mutant, library, mutant_scope, perturb, poison, post, round_input, unscoped};
use crate::inference::cuda::{CudaError, CudaRuntime};

/// One prepared route, independent of current and secret inputs
pub(crate) struct Owner<C = Box<dyn RuntimePlan>> {
    name: &'static str,
    batch: usize,
    route: Route<C>,
}

enum Route<C> {
    Library,
    Candidate(C),
    Lookup(CudaSlice<f32>),
    Mutant(Mutant),
}

impl<C> std::fmt::Debug for Owner<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let route = match &self.route {
            Route::Library => "Library",
            Route::Candidate(_) => "Candidate",
            Route::Lookup(_) => "Lookup",
            Route::Mutant(_) => "Mutant",
        };
        f.debug_struct("Owner")
            .field("name", &self.name)
            .field("batch", &self.batch)
            .field("route", &route)
            .finish()
    }
}

impl<C> Owner<C> {
    /// Retain the prepared candidate in the same owner used by both collection paths
    pub(crate) fn candidate(operation: Operation, candidate: C) -> Self {
        Self {
            name: operation.boundary().name(),
            batch: operation.tuple().0,
            route: Route::Candidate(candidate),
        }
    }
    /// Exactly one route receives the invocation and its output owner
    pub(crate) fn dispatch<A, T>(
        &self,
        args: A,
        candidate: impl FnOnce(&C, A) -> Result<T, CudaError>,
        control: impl FnOnce(A) -> Result<T, CudaError>,
    ) -> Result<T, CudaError> {
        match &self.route {
            Route::Candidate(plan) => candidate(plan, args),
            _ => control(args),
        }
    }
}

impl Owner {
    /// Prepare fixed fixture answers outside capture; candidate ports use their own plan
    pub(crate) fn new(
        runtime: &CudaRuntime,
        name: &'static str,
        batch: usize,
        choice: &str,
        fixture: impl FnOnce() -> Result<Vec<f32>, CudaError>,
    ) -> Result<Self, CudaError> {
        let route = match choice {
            "Library" => Route::Library,
            "Lookup" => Route::Lookup(runtime.stream().clone_htod(&fixture()?)?),
            "Oxide" => {
                return Err(CudaError::Unsupported {
                    context: "new boundary qualification",
                    reason: "no candidate port exists".to_owned(),
                });
            }
            name => Route::Mutant(Mutant::parse(name).expect("applicable boundary mutant")),
        };
        Ok(Self { name, batch, route })
    }

    /// Prepare an injectable candidate from the actual model operation
    pub(crate) fn prepare(
        runtime: &CudaRuntime,
        operation: Operation,
        choice: &str,
        weight: &CudaSlice<f32>,
        bias: &CudaSlice<f32>,
        fixture: impl FnOnce() -> Result<Vec<f32>, CudaError>,
    ) -> Result<Self, CudaError> {
        let (batch, _) = operation.tuple();
        if choice != "Oxide" {
            return Self::new(runtime, operation.boundary().name(), batch, choice, fixture);
        }
        Ok(Self::candidate(
            operation,
            candidate_seam::prepare(runtime, operation, weight, bias)?,
        ))
    }

    /// Execute a prepared candidate on views, or keep the original control route
    pub(crate) fn run(
        &self,
        runtime: &CudaRuntime,
        operation: Operation,
        views: Views<'_>,
        output: &mut CudaViewMut<'_, f32>,
        produce: impl FnMut(&mut CudaViewMut<'_, f32>) -> Result<(), CudaError>,
    ) -> Result<(), CudaError> {
        self.dispatch(
            (views, output, produce),
            |plan, (views, output, _)| {
                plan.validate(operation, &views, output)?;
                poison(runtime)?;
                let _scope = super::candidate(runtime.stream(), self.name);
                plan.run(runtime, operation, views, output.slice_mut(..))
            },
            |(views, output, produce)| self.run_control(runtime, &views.input, output, produce),
        )
    }

    /// Slice-based Library callbacks retain the original buffers and operations
    pub(crate) fn run_slices(
        &self,
        runtime: &CudaRuntime,
        operation: Operation,
        inputs: Slices<'_>,
        output: &mut CudaSlice<f32>,
        produce: impl FnMut(&mut CudaSlice<f32>) -> Result<(), CudaError>,
    ) -> Result<(), CudaError> {
        self.dispatch(
            (inputs, output, produce),
            |plan, (inputs, output, _)| {
                let views = inputs.views();
                plan.validate(operation, &views, &output.as_view_mut())?;
                poison(runtime)?;
                let _scope = super::candidate(runtime.stream(), self.name);
                plan.run(runtime, operation, views, output.as_view_mut())
            },
            |(inputs, output, produce)| self.run_control(runtime, inputs.input, output, produce),
        )
    }

    /// Whether this plan declares candidate execution
    pub(crate) fn declared(&self) -> bool {
        !matches!(self.route, Route::Library)
    }

    /// Execute one locked operator; Lookup never calls or refreshes its Library producer
    fn run_control<X: DevicePtr<f32>, Y: DevicePtrMut<f32>>(
        &self,
        runtime: &CudaRuntime,
        input: &X,
        output: &mut Y,
        mut produce: impl FnMut(&mut Y) -> Result<(), CudaError>,
    ) -> Result<(), CudaError> {
        match &self.route {
            Route::Candidate(_) => unreachable!("candidate uses its retained adapter"),
            Route::Library => {
                let _scope = library(runtime.stream(), self.name);
                produce(output)?;
                perturb(runtime, self.name, output)
            }
            Route::Lookup(fixture) => {
                poison(runtime)?;
                let _scope = mutant_scope(runtime.stream(), self.name, Mutant::Lookup);
                runtime.stream().memcpy_dtod(fixture, output)?;
                post(runtime, output, self.batch, Mutant::Lookup)
            }
            Route::Mutant(mutant) => {
                poison(runtime)?;
                {
                    let _scope = mutant_scope(runtime.stream(), self.name, *mutant);
                    if !mutant.skips() {
                        if *mutant == Mutant::Precision {
                            round_input(runtime, input)?;
                        }
                        let repeats = if *mutant == Mutant::Slow { 3 } else { 1 };
                        for _ in 0..repeats {
                            produce(output)?;
                        }
                        post(runtime, output, self.batch, *mutant)?;
                    }
                }
                unscoped(runtime, *mutant)
            }
        }
    }
}

/// Run Library unchanged when no test-only owner is installed
pub(crate) fn run(
    owner: Option<&Owner>,
    runtime: &CudaRuntime,
    operation: Operation,
    views: Views<'_>,
    output: &mut CudaViewMut<'_, f32>,
    mut produce: impl FnMut(&mut CudaViewMut<'_, f32>) -> Result<(), CudaError>,
) -> Result<(), CudaError> {
    match owner {
        Some(owner) => owner.run(runtime, operation, views, output, produce),
        None => produce(output),
    }
}

/// Keep slice-based controls unchanged, with candidate views borrowed only once
pub(crate) fn run_slices(
    owner: Option<&Owner>,
    runtime: &CudaRuntime,
    operation: Operation,
    inputs: Slices<'_>,
    output: &mut CudaSlice<f32>,
    mut produce: impl FnMut(&mut CudaSlice<f32>) -> Result<(), CudaError>,
) -> Result<(), CudaError> {
    match owner {
        Some(owner) => owner.run_slices(runtime, operation, inputs, output, produce),
        None => produce(output),
    }
}
