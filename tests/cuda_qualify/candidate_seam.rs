//! Prepared test-only candidate routes shared by isolated and full-stage owners

use std::cell::RefCell;
use std::marker::PhantomData;
use std::rc::Rc;

use cudarc::driver::{CudaSlice, CudaView, CudaViewMut};
use serde_json::{Value, json};

use crate::inference::cuda::candidate::{
    ConvCandidate, ConvInputs, ConvLayerSpec, Coverage, DenseCandidate, DenseSite, DenseSpec,
    Epilogue, Phases, SegConvCandidate, SegConvSite, SegConvSpec,
};
use crate::inference::cuda::dnn::Conv2d;
use crate::inference::cuda::implementation::BoundaryId;
use crate::inference::cuda::kernels::{LoadedArtifact, ModuleRequest};
use crate::inference::cuda::{
    CudaError, CudaMath, CudaRuntime, KernelModule, LoadedKernels, PtxTier,
};

fn error(reason: impl Into<String>) -> CudaError {
    CudaError::Unsupported {
        context: "prepared candidate boundary",
        reason: reason.into(),
    }
}

/// Typed families prevent a dense registration from serving a convolution
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Family {
    Dense,
    Temporal,
    Spatial,
}

/// Fixed model operation, independent of the implementation and its module
#[derive(Debug, Clone, Copy)]
pub(crate) enum Operation {
    Dense(DenseSpec),
    Temporal(SegConvSpec),
    Spatial {
        boundary: BoundaryId,
        conv: Conv2d,
        epilogue: Epilogue,
    },
}

impl PartialEq for Operation {
    fn eq(&self, other: &Self) -> bool {
        match (*self, *other) {
            (Self::Dense(a), Self::Dense(b)) => {
                a.site() == b.site() && a.batch() == b.batch() && a.math() == b.math()
            }
            (Self::Temporal(a), Self::Temporal(b)) => {
                a.site() == b.site() && a.batch() == b.batch() && a.math() == b.math()
            }
            (
                Self::Spatial {
                    boundary: a,
                    conv: ac,
                    epilogue: ae,
                },
                Self::Spatial {
                    boundary: b,
                    conv: bc,
                    epilogue: be,
                },
            ) => a == b && ac == bc && ae == be,
            _ => false,
        }
    }
}

impl Operation {
    /// Exact boundary from the typed site, never from a name substring
    pub(crate) fn boundary(self) -> BoundaryId {
        match self {
            Self::Dense(spec) => BoundaryId::named(match spec.site() {
                DenseSite::Linear0 => "linear0",
                DenseSite::Linear1 => "linear1",
                DenseSite::Classifier => "linear2",
                DenseSite::Embedding => "resnet.seg_1",
            }),
            Self::Temporal(spec) => BoundaryId::named(match spec.site() {
                SegConvSite::Conv1 => "sincnet.conv1",
                SegConvSite::Conv2 => "sincnet.conv2",
            }),
            Self::Spatial { boundary, .. } => boundary,
        }
    }
    /// The operation family this owner can execute
    pub(crate) fn family(self) -> Family {
        match self {
            Self::Dense(_) => Family::Dense,
            Self::Temporal(_) => Family::Temporal,
            Self::Spatial { .. } => Family::Spatial,
        }
    }
    /// Candidate module for the family, not the Library diagnostic area
    pub(crate) fn area(self) -> KernelModule {
        match self.family() {
            Family::Dense | Family::Temporal => KernelModule::Segdense,
            Family::Spatial => KernelModule::Wideconv,
        }
    }
    /// Fixed batch and multiply mode
    pub(crate) fn tuple(self) -> (usize, CudaMath) {
        match self {
            Self::Dense(s) => (s.batch(), s.math()),
            Self::Temporal(s) => (s.batch(), s.math()),
            Self::Spatial { conv, .. } => (conv.batch, conv.math),
        }
    }
    fn lengths(self) -> [usize; 4] {
        match self {
            Self::Dense(s) => [s.input_len(), s.weight_len(), s.bias_len(), s.output_len()],
            Self::Temporal(s) => [s.input_len(), s.weight_len(), 0, s.output_len()],
            Self::Spatial { conv, .. } => [
                conv.input_shape().iter().product(),
                conv.filter_shape().iter().product(),
                conv.out_channels,
                conv.output_shape().iter().product(),
            ],
        }
    }
    fn residual(self) -> bool {
        matches!(
            self,
            Self::Spatial {
                epilogue: Epilogue::BiasReluResidual,
                ..
            }
        )
    }
}

/// Complete, stable pin evidence supplied by the port, without future kernel names
pub(crate) trait PinEvidence: Clone + PartialEq {
    /// All execution choices, including entry, tile, splits, packing and layout
    fn evidence(&self) -> Value;
}

/// Metadata checked before any candidate enqueue
pub(crate) struct Call {
    pub(crate) operation: Operation,
    pub(crate) lengths: [usize; 4],
    pub(crate) residual_len: Option<usize>,
}

impl Call {
    fn check(&self, selected: Operation) -> Result<(), CudaError> {
        if self.operation != selected {
            return Err(error(
                "operation family, boundary or epilogue differs from the prepared plan",
            ));
        }
        if self.lengths != selected.lengths() {
            return Err(error(
                "input, weight, bias or output length differs from the prepared plan",
            ));
        }
        if self.residual_len.is_some() != selected.residual()
            || self.residual_len.is_some_and(|len| len != self.lengths[3])
        {
            return Err(error("residual differs from the prepared epilogue"));
        }
        Ok(())
    }
}

/// A concrete executor owns the plan; erasure occurs only after selection
pub(crate) trait Executor {
    type Context: ?Sized;
    type Inputs<'a>;
    type Output<'a>;
    /// Execute directly on the caller's views
    fn enqueue(
        &self,
        context: &Self::Context,
        inputs: Self::Inputs<'_>,
        output: Self::Output<'_>,
    ) -> Result<(), CudaError>;
}

/// One injectable typed plan factory, used by the GPU and CPU seam
pub(crate) trait Factory {
    type Pin: PinEvidence;
    type Plan: Executor;
    type Resources<'a>;
    const FAMILY: Family;
    /// Implemented coverage for the selected loaded tier
    fn coverage(tier: PtxTier) -> Coverage;
    /// Return the plan and the exact pin passed to its constructor
    fn build(
        resources: Self::Resources<'_>,
        operation: Operation,
    ) -> Result<(Self::Plan, Self::Pin), CudaError>;
}

/// A registration binds one implementation to an exact module request
pub(crate) struct Registration<F> {
    request: ModuleRequest,
    marker: PhantomData<F>,
}

impl<F: Factory> Registration<F> {
    /// Register a real selected module, without loading it in the candidate
    pub(crate) fn new(request: ModuleRequest) -> Self {
        Self {
            request,
            marker: PhantomData,
        }
    }
    /// Check identity and coverage before constructing a plan
    pub(crate) fn prepare(
        &self,
        loaded: ModuleRequest,
        operation: Operation,
        resources: F::Resources<'_>,
    ) -> Result<Prepared<F::Plan, F::Pin>, CudaError> {
        if operation.family() != F::FAMILY || self.request.area() != operation.area() {
            return Err(error("wrong operation family or candidate area"));
        }
        if matches!(operation, Operation::Spatial { boundary, .. } if boundary.area() != KernelModule::Resnet)
        {
            return Err(error(
                "spatial route requires an actual convolution boundary",
            ));
        }
        self.request.check_cached(loaded)?;
        let (batch, math) = operation.tuple();
        if !F::coverage(loaded.tier()).covers(operation.boundary().name(), batch, math) {
            return Err(error(
                "candidate does not implement the tuple at the loaded tier",
            ));
        }
        let (plan, pin) = F::build(resources, operation)?;
        Ok(Prepared {
            plan,
            pin,
            operation,
            request: loaded,
        })
    }
}

/// Selected owner retains the full typed pin, fixed operation and loaded identity
pub(crate) struct Prepared<E, P> {
    plan: E,
    pin: P,
    operation: Operation,
    request: ModuleRequest,
}

impl<E: Executor, P: PinEvidence> Prepared<E, P> {
    /// Reject a changed operation before running the concrete executor
    pub(crate) fn run(
        &self,
        context: &E::Context,
        call: &Call,
        inputs: E::Inputs<'_>,
        output: E::Output<'_>,
    ) -> Result<(), CudaError> {
        call.check(self.operation)?;
        self.plan.enqueue(context, inputs, output)
    }
    /// Locked receipt from the selected request and retained complete pin
    pub(crate) fn receipt(&self) -> Value {
        let (batch, math) = self.operation.tuple();
        json!({"tuple":[self.operation.boundary().name(),batch,match math { CudaMath::Fp32 => "fp32", CudaMath::Tf32 => "tf32" }],
            "area":self.request.area().name(), "tier":self.request.tier().to_string(),
            "artifact":match self.request.artifact() { LoadedArtifact::PtxJit { sha256 } => json!({"kind":"PtxJit", "sha256":sha256.to_string()}), LoadedArtifact::Cubin { arch, sha256 } => json!({"kind":"Cubin", "arch":[arch.major,arch.minor], "sha256":sha256.to_string()}) }, "pin":self.pin.evidence()})
    }
}

/// Borrowed actual operation buffers, with no staging allocation or copy
pub(crate) struct Views<'a> {
    pub(crate) input: CudaView<'a, f32>,
    pub(crate) weight: CudaView<'a, f32>,
    pub(crate) bias: Option<CudaView<'a, f32>>,
    pub(crate) residual: Option<CudaView<'a, f32>>,
}

impl Views<'_> {
    /// Check all actual buffer lengths with the fixed operation
    pub(crate) fn call(&self, operation: Operation, output: &CudaViewMut<'_, f32>) -> Call {
        Call {
            operation,
            lengths: [
                self.input.len(),
                self.weight.len(),
                self.bias.as_ref().map_or(0, CudaView::len),
                output.len(),
            ],
            residual_len: self.residual.as_ref().map(CudaView::len),
        }
    }
}

/// Runtime erasure for heterogeneous stage maps; the underlying owner stays typed
pub(crate) trait RuntimePlan {
    /// Validate before poison or any other device work is enqueued
    fn validate(
        &self,
        operation: Operation,
        views: &Views<'_>,
        output: &CudaViewMut<'_, f32>,
    ) -> Result<(), CudaError>;
    /// Dispatch the retained concrete plan, never a Library producer
    fn run(
        &self,
        runtime: &CudaRuntime,
        operation: Operation,
        views: Views<'_>,
        output: CudaViewMut<'_, f32>,
    ) -> Result<(), CudaError>;
    /// Exact selection evidence from the prepared owner
    fn receipt(&self) -> Value;
}

impl<E, P> RuntimePlan for Prepared<E, P>
where
    E: for<'a> Executor<
            Context = CudaRuntime,
            Inputs<'a> = Views<'a>,
            Output<'a> = CudaViewMut<'a, f32>,
        >,
    P: PinEvidence,
{
    fn validate(
        &self,
        operation: Operation,
        views: &Views<'_>,
        output: &CudaViewMut<'_, f32>,
    ) -> Result<(), CudaError> {
        views.call(operation, output).check(self.operation)
    }
    fn run(
        &self,
        runtime: &CudaRuntime,
        operation: Operation,
        views: Views<'_>,
        output: CudaViewMut<'_, f32>,
    ) -> Result<(), CudaError> {
        let call = views.call(operation, &output);
        Prepared::run(self, runtime, &call, views, output)
    }
    fn receipt(&self) -> Value {
        Prepared::receipt(self)
    }
}

struct DenseAdapter<C>(C);
struct TemporalAdapter<C>(C);
struct SpatialAdapter<C>(C);

impl<C: DenseCandidate> Executor for DenseAdapter<C> {
    type Context = CudaRuntime;
    type Inputs<'a> = Views<'a>;
    type Output<'a> = CudaViewMut<'a, f32>;
    fn enqueue(
        &self,
        runtime: &CudaRuntime,
        views: Views<'_>,
        mut output: CudaViewMut<'_, f32>,
    ) -> Result<(), CudaError> {
        self.0.enqueue(
            &views.input,
            &views.weight,
            views
                .bias
                .as_ref()
                .ok_or_else(|| error("dense bias is absent"))?,
            &mut output,
            &Phases::new(),
            runtime,
        )
    }
}
impl<C: SegConvCandidate> Executor for TemporalAdapter<C> {
    type Context = CudaRuntime;
    type Inputs<'a> = Views<'a>;
    type Output<'a> = CudaViewMut<'a, f32>;
    fn enqueue(
        &self,
        runtime: &CudaRuntime,
        views: Views<'_>,
        mut output: CudaViewMut<'_, f32>,
    ) -> Result<(), CudaError> {
        self.0.enqueue(
            &views.input,
            &views.weight,
            &mut output,
            &Phases::new(),
            runtime,
        )
    }
}
impl<C: ConvCandidate> Executor for SpatialAdapter<C> {
    type Context = CudaRuntime;
    type Inputs<'a> = Views<'a>;
    type Output<'a> = CudaViewMut<'a, f32>;
    fn enqueue(
        &self,
        runtime: &CudaRuntime,
        views: Views<'_>,
        mut output: CudaViewMut<'_, f32>,
    ) -> Result<(), CudaError> {
        self.0.enqueue(
            ConvInputs {
                x: &views.input,
                weight: &views.weight,
                bias: views
                    .bias
                    .as_ref()
                    .ok_or_else(|| error("spatial bias is absent"))?,
                residual: views.residual.as_ref(),
            },
            &mut output,
            &Phases::new(),
            runtime.stream(),
        )
    }
}

/// Plan resources from the actual selected model operation
pub(crate) struct Resources<'a> {
    pub(crate) runtime: &'a CudaRuntime,
    pub(crate) kernels: &'a LoadedKernels,
    pub(crate) weight: &'a CudaSlice<f32>,
    pub(crate) bias: &'a CudaSlice<f32>,
}

struct DenseFactory<C>(PhantomData<C>);
struct TemporalFactory<C>(PhantomData<C>);
struct SpatialFactory<C>(PhantomData<C>);

impl<C: DenseCandidate> Factory for DenseFactory<C>
where
    C::Pin: PinEvidence,
{
    type Pin = C::Pin;
    type Plan = DenseAdapter<C>;
    type Resources<'a> = Resources<'a>;
    const FAMILY: Family = Family::Dense;
    fn coverage(tier: PtxTier) -> Coverage {
        C::coverage(tier)
    }
    fn build(r: Resources<'_>, operation: Operation) -> Result<(Self::Plan, Self::Pin), CudaError> {
        let Operation::Dense(spec) = operation else {
            return Err(error("not a dense operation"));
        };
        let pin = C::implemented_pin(spec).map_err(|e| error(e.to_string()))?;
        let plan =
            C::plan(r.runtime, r.kernels, spec, pin.clone()).map_err(|e| error(e.to_string()))?;
        Ok((DenseAdapter(plan), pin))
    }
}
impl<C: SegConvCandidate> Factory for TemporalFactory<C>
where
    C::Pin: PinEvidence,
{
    type Pin = C::Pin;
    type Plan = TemporalAdapter<C>;
    type Resources<'a> = Resources<'a>;
    const FAMILY: Family = Family::Temporal;
    fn coverage(tier: PtxTier) -> Coverage {
        C::coverage(tier)
    }
    fn build(r: Resources<'_>, operation: Operation) -> Result<(Self::Plan, Self::Pin), CudaError> {
        let Operation::Temporal(spec) = operation else {
            return Err(error("not a temporal operation"));
        };
        let pin = C::implemented_pin(spec).map_err(|e| error(e.to_string()))?;
        let plan =
            C::plan(r.runtime, r.kernels, spec, pin.clone()).map_err(|e| error(e.to_string()))?;
        Ok((TemporalAdapter(plan), pin))
    }
}
impl<C: ConvCandidate> Factory for SpatialFactory<C>
where
    C::Pin: PinEvidence,
{
    type Pin = C::Pin;
    type Plan = SpatialAdapter<C>;
    type Resources<'a> = Resources<'a>;
    const FAMILY: Family = Family::Spatial;
    fn coverage(tier: PtxTier) -> Coverage {
        C::coverage(tier)
    }
    fn build(r: Resources<'_>, operation: Operation) -> Result<(Self::Plan, Self::Pin), CudaError> {
        let Operation::Spatial {
            boundary,
            conv,
            epilogue,
        } = operation
        else {
            return Err(error("not a spatial operation"));
        };
        let spec = ConvLayerSpec {
            name: boundary.name(),
            conv,
            epilogue,
            weight: r.weight,
            bias: r.bias,
        };
        let pin = C::implemented_pin(&spec).map_err(|e| error(e.to_string()))?;
        let plan =
            C::plan(r.runtime, r.kernels, spec, pin.clone()).map_err(|e| error(e.to_string()))?;
        Ok((SpatialAdapter(plan), pin))
    }
}

trait RuntimeFactory {
    fn family(&self) -> Family;
    fn request(&self) -> ModuleRequest;
    fn coverage(&self, tier: PtxTier) -> Coverage;
    fn prepare(
        &self,
        r: Resources<'_>,
        operation: Operation,
    ) -> Result<Box<dyn RuntimePlan>, CudaError>;
}
impl<F> RuntimeFactory for Registration<F>
where
    F: for<'a> Factory<Resources<'a> = Resources<'a>> + 'static,
    F::Plan: for<'a> Executor<
            Context = CudaRuntime,
            Inputs<'a> = Views<'a>,
            Output<'a> = CudaViewMut<'a, f32>,
        > + 'static,
    F::Pin: 'static,
{
    fn family(&self) -> Family {
        F::FAMILY
    }
    fn request(&self) -> ModuleRequest {
        self.request
    }
    fn coverage(&self, tier: PtxTier) -> Coverage {
        F::coverage(tier)
    }
    fn prepare(
        &self,
        r: Resources<'_>,
        operation: Operation,
    ) -> Result<Box<dyn RuntimePlan>, CudaError> {
        Ok(Box::new(Registration::prepare(
            self,
            r.kernels.request(),
            operation,
            r,
        )?))
    }
}

thread_local! {
    static FACTORIES: RefCell<Vec<Rc<dyn RuntimeFactory>>> = const { RefCell::new(Vec::new()) };
    static RECEIPTS: RefCell<Vec<Value>> = const { RefCell::new(Vec::new()) };
}

fn register(factory: Rc<dyn RuntimeFactory>) -> Result<(), CudaError> {
    FACTORIES.with(|factories| {
        let mut factories = factories.borrow_mut();
        if factories
            .iter()
            .any(|known| known.family() == factory.family())
        {
            return Err(error("one registration per operation family"));
        }
        factories.push(factory);
        Ok(())
    })
}

/// Register a context-aware typed factory when a complete pin needs device facts
pub(crate) fn register_factory<F>(request: ModuleRequest) -> Result<(), CudaError>
where
    F: for<'a> Factory<Resources<'a> = Resources<'a>> + 'static,
    F::Plan: for<'a> Executor<
            Context = CudaRuntime,
            Inputs<'a> = Views<'a>,
            Output<'a> = CudaViewMut<'a, f32>,
        > + 'static,
    F::Pin: 'static,
{
    register(Rc::new(Registration::<F>::new(request)))
}

/// Register a dense port and its exact selected module once before collection
pub(crate) fn register_dense<C: DenseCandidate + 'static>(
    request: ModuleRequest,
) -> Result<(), CudaError>
where
    C::Pin: PinEvidence + 'static,
{
    register_factory::<DenseFactory<C>>(request)
}
/// Register a raw five-tap port once before collection
pub(crate) fn register_temporal<C: SegConvCandidate + 'static>(
    request: ModuleRequest,
) -> Result<(), CudaError>
where
    C::Pin: PinEvidence + 'static,
{
    register_factory::<TemporalFactory<C>>(request)
}
/// Register a native NCHW convolution port once before collection
pub(crate) fn register_spatial<C: ConvCandidate + 'static>(
    request: ModuleRequest,
) -> Result<(), CudaError>
where
    C::Pin: PinEvidence + 'static,
{
    register_factory::<SpatialFactory<C>>(request)
}

fn selected_factory(operation: Operation) -> Result<Rc<dyn RuntimeFactory>, CudaError> {
    let factory = FACTORIES
        .with(|factories| {
            factories
                .borrow()
                .iter()
                .find(|f| f.family() == operation.family())
                .cloned()
        })
        .ok_or_else(|| error("no candidate port is registered"))?;
    let request = factory.request();
    let (batch, math) = operation.tuple();
    if request.area() != operation.area()
        || !factory
            .coverage(request.tier())
            .covers(operation.boundary().name(), batch, math)
    {
        return Err(error(
            "registration area or implemented coverage does not match",
        ));
    }
    Ok(factory)
}

/// Implemented coverage from the registered request, without loading candidate bytes
pub(crate) fn coverage(family: Family, tier: PtxTier) -> Coverage {
    FACTORIES.with(|factories| {
        factories
            .borrow()
            .iter()
            .find(|factory| factory.family() == family && factory.request().tier() == tier)
            .map_or(Coverage::NONE, |factory| factory.coverage(tier))
    })
}

/// Preload selected registrations before clocks, capture and profile observation
pub(crate) fn preload(runtime: &CudaRuntime, operation: Operation) -> Result<(), CudaError> {
    let factory = selected_factory(operation)?;
    let kernels = runtime.load_module(factory.request())?;
    factory.request().check_cached(kernels.request())
}

/// Complete prepared receipts, without widening the closed production pin enum
pub(crate) fn planned() -> Value {
    RECEIPTS.with(|receipts| Value::Array(receipts.borrow().clone()))
}

/// Prepare from the same registration for both Operator and Stage construction
pub(crate) fn prepare(
    runtime: &CudaRuntime,
    operation: Operation,
    weight: &CudaSlice<f32>,
    bias: &CudaSlice<f32>,
) -> Result<Box<dyn RuntimePlan>, CudaError> {
    let factory = selected_factory(operation)?;
    let request = factory.request();
    let _scope = super::plan(operation.boundary().name());
    // only the locked owner loads the selected request; a candidate receives this handle
    let kernels = runtime.load_module(request)?;
    let plan = factory.prepare(
        Resources {
            runtime,
            kernels: &kernels,
            weight,
            bias,
        },
        operation,
    )?;
    let receipt = plan.receipt();
    record_receipt(receipt)?;
    Ok(plan)
}

fn record_receipt(receipt: Value) -> Result<(), CudaError> {
    RECEIPTS.with(|receipts| {
        let mut receipts = receipts.borrow_mut();
        if let Some(known) = receipts
            .iter()
            .find(|known| known["tuple"] == receipt["tuple"])
        {
            if known != &receipt {
                return Err(error("two identities or pins for one tuple"));
            }
        } else {
            receipts.push(receipt);
        }
        Ok(())
    })
}

/// Actual whole buffers for slice-based Library hooks
pub(crate) struct Slices<'a> {
    pub(crate) input: &'a CudaSlice<f32>,
    pub(crate) weight: &'a CudaSlice<f32>,
    pub(crate) bias: Option<&'a CudaSlice<f32>>,
    pub(crate) residual: Option<&'a CudaSlice<f32>>,
}

impl Slices<'_> {
    /// Borrow without a device allocation or copy
    pub(crate) fn views(&self) -> Views<'_> {
        Views {
            input: self.input.as_view(),
            weight: self.weight.as_view(),
            bias: self.bias.map(CudaSlice::as_view),
            residual: self.residual.map(CudaSlice::as_view),
        }
    }
}

#[cfg(test)]
#[path = "candidate_seam_tests.rs"]
mod tests;
