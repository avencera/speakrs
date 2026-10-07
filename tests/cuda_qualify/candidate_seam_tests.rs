use super::super::boundaries::Owner;
use super::{
    Call, ConvCandidate, ConvInputs, ConvLayerSpec, Coverage, CudaError, CudaMath, CudaRuntime,
    CudaView, CudaViewMut, DenseCandidate, DenseSite, DenseSpec, Epilogue, Executor, FACTORIES,
    Factory, Family, KernelModule, LoadedKernels, ModuleRequest, Operation, Phases, PinEvidence,
    PtxTier, Registration, SegConvCandidate, SegConvSite, SegConvSpec, Value, error, json,
    register_dense, register_spatial, register_temporal, selected_factory,
};
use crate::inference::cuda::candidate::{
    Batches, CoverageEntry, FiniteContract, InfinityContract, Maths, NanContract, PlanError,
    SignedZeroContract, SpecialValues,
};
use crate::inference::cuda::geometry::Conv2d;
use crate::inference::cuda::implementation::BoundaryId;
use crate::inference::cuda::kernels::{ArtifactHash, LoadedArtifact};
use std::cell::Cell;
use std::cell::RefCell;

const COVERAGE: Coverage = Coverage(&[CoverageEntry {
    layers: &["linear0", "sincnet.conv1", "resnet.layer3.0.conv2"],
    batches: Batches::Only(&[1]),
    maths: Maths::Only(&[CudaMath::Fp32]),
}]);
const CONTRACT: SpecialValues = SpecialValues {
    finite: FiniteContract::AbsoluteSum { headroom: 2 },
    nan: NanContract::Unspecified,
    infinity: InfinityContract::Unspecified,
    signed_zero: SignedZeroContract::Unspecified,
};

#[derive(Debug, Clone, PartialEq)]
struct FakePin {
    factor: u32,
}
impl PinEvidence for FakePin {
    fn evidence(&self) -> Value {
        json!({"test_only_factor":self.factor})
    }
}
fn request(area: KernelModule, tier: PtxTier, bytes: &[u8]) -> ModuleRequest {
    ModuleRequest::new(
        area,
        tier,
        LoadedArtifact::PtxJit {
            sha256: ArtifactHash::of(bytes),
        },
    )
}
fn operation(family: Family) -> Operation {
    match family {
        Family::Dense => {
            Operation::Dense(DenseSpec::new(DenseSite::Linear0, 1, CudaMath::Fp32).unwrap())
        }
        Family::Temporal => {
            Operation::Temporal(SegConvSpec::new(SegConvSite::Conv1, 1, CudaMath::Fp32).unwrap())
        }
        Family::Spatial => Operation::Spatial {
            boundary: BoundaryId::named("resnet.layer3.0.conv2"),
            conv: Conv2d {
                batch: 1,
                in_channels: 128,
                out_channels: 128,
                input: [20, 250],
                kernel: [3, 3],
                padding: [1, 1],
                stride: [1, 1],
                dilation: [1, 1],
                math: CudaMath::Fp32,
            },
            epilogue: Epilogue::BiasReluResidual,
        },
    }
}

/// CPU implementation of the same factory, receipt and prepared-owner protocol
struct FakeFactory<const KIND: u8 = 2>;
struct FakePlan(FakePin);
struct HostViews<'a> {
    input: &'a [f32],
    weight: &'a [f32],
    bias: Option<&'a [f32]>,
    residual: Option<&'a [f32]>,
}
type HostContext = RefCell<Vec<[usize; 4]>>;
impl Executor for FakePlan {
    type Context = HostContext;
    type Inputs<'a> = HostViews<'a>;
    type Output<'a> = &'a mut [f32];
    fn enqueue(
        &self,
        context: &HostContext,
        inputs: HostViews<'_>,
        output: &mut [f32],
    ) -> Result<(), CudaError> {
        context.borrow_mut().push([
            inputs.input.as_ptr() as usize,
            inputs.weight.as_ptr() as usize,
            inputs.bias.map_or(0, |v| v.as_ptr() as usize),
            inputs.residual.map_or(0, |v| v.as_ptr() as usize),
        ]);
        output.fill(
            inputs.input[0] * inputs.weight[0] * self.0.factor as f32
                + inputs.bias.map_or(0.0, |v| v[0])
                + inputs.residual.map_or(0.0, |v| v[0]),
        );
        Ok(())
    }
}
impl<const KIND: u8> Factory for FakeFactory<KIND> {
    type Pin = FakePin;
    type Plan = FakePlan;
    type Resources<'a> = &'a Cell<usize>;
    const FAMILY: Family = match KIND {
        0 => Family::Dense,
        1 => Family::Temporal,
        _ => Family::Spatial,
    };
    fn coverage(tier: PtxTier) -> Coverage {
        if tier == PtxTier::Sm80 {
            COVERAGE
        } else {
            Coverage::NONE
        }
    }
    fn build(count: &Cell<usize>, _operation: Operation) -> Result<(FakePlan, FakePin), CudaError> {
        count.set(count.get() + 1);
        let pin = FakePin { factor: 3 };
        Ok((FakePlan(pin.clone()), pin))
    }
}

#[test]
fn shared_factory_and_owner_execute_actual_weights_bias_and_residual_without_library() {
    let op = operation(Family::Spatial);
    let module = request(op.area(), PtxTier::Sm80, b"CPU test identity only");
    let factory = Registration::<FakeFactory>::new(module);
    let builds = Cell::new(0);
    let selected = factory.prepare(module, op, &builds).unwrap();
    assert_eq!(builds.get(), 1);
    assert_eq!(selected.request, module);
    assert_eq!(selected.pin, FakePin { factor: 3 });
    assert_eq!(selected.receipt()["pin"]["test_only_factor"], 3);
    assert_eq!(selected.receipt()["area"], "wideconv");
    let owner = Owner::candidate(op, selected);
    let lengths = op.lengths();
    let input = vec![2.0; lengths[0]];
    let weight = vec![5.0; lengths[1]];
    let bias = vec![7.0; lengths[2]];
    let residual = vec![11.0; lengths[3]];
    let other_weight = vec![-2.0; lengths[1]];
    let other_bias = vec![13.0; lengths[2]];
    let other_residual = vec![17.0; lengths[3]];
    let mut output = vec![0.0; lengths[3]];
    let context = HostContext::default();
    let library_calls = Cell::new(0);
    for (weights, biases, residuals, expected) in [
        (&weight, &bias, &residual, 48.0),
        (&other_weight, &other_bias, &other_residual, 18.0),
    ] {
        let call = Call {
            operation: op,
            lengths,
            residual_len: Some(residuals.len()),
        };
        owner
            .dispatch(
                (
                    HostViews {
                        input: &input,
                        weight: weights,
                        bias: Some(biases),
                        residual: Some(residuals),
                    },
                    output.as_mut_slice(),
                ),
                |plan, (views, output)| plan.run(&context, &call, views, output),
                |(_, output)| {
                    library_calls.set(library_calls.get() + 1);
                    output.fill(-100.0);
                    Ok(())
                },
            )
            .unwrap();
        assert!(output.iter().all(|value| *value == expected));
    }
    assert_eq!(library_calls.get(), 0);
    assert_eq!(
        *context.borrow(),
        [
            [
                input.as_ptr() as usize,
                weight.as_ptr() as usize,
                bias.as_ptr() as usize,
                residual.as_ptr() as usize
            ],
            [
                input.as_ptr() as usize,
                other_weight.as_ptr() as usize,
                other_bias.as_ptr() as usize,
                other_residual.as_ptr() as usize
            ]
        ]
    );
}

#[test]
fn selection_rejects_wrong_family_area_identity_and_actual_tier_coverage_before_factory() {
    let op = operation(Family::Spatial);
    let module = request(op.area(), PtxTier::Sm80, b"CPU identity");
    let factory = Registration::<FakeFactory>::new(module);
    let builds = Cell::new(0);
    assert!(
        factory
            .prepare(module, operation(Family::Dense), &builds)
            .is_err()
    );
    assert!(
        Registration::<FakeFactory>::new(request(
            KernelModule::Segdense,
            PtxTier::Sm80,
            b"CPU identity"
        ))
        .prepare(module, op, &builds)
        .is_err()
    );
    assert!(
        factory
            .prepare(
                request(op.area(), PtxTier::Sm80, b"different CPU identity"),
                op,
                &builds
            )
            .is_err()
    );
    let lower = request(op.area(), PtxTier::Sm75, b"CPU identity");
    assert!(
        Registration::<FakeFactory>::new(lower)
            .prepare(lower, op, &builds)
            .is_err()
    );
    let Operation::Spatial {
        boundary,
        mut conv,
        epilogue,
    } = op
    else {
        unreachable!()
    };
    conv.batch = 7;
    assert!(
        factory
            .prepare(
                module,
                Operation::Spatial {
                    boundary,
                    conv,
                    epilogue
                },
                &builds
            )
            .is_err()
    );
    assert_eq!(builds.get(), 0);
}

#[test]
fn prepared_owner_rejects_wrong_family_epilogue_and_residual_before_execution() {
    let op = operation(Family::Spatial);
    let module = request(op.area(), PtxTier::Sm80, b"CPU identity");
    let selected = Registration::<FakeFactory>::new(module)
        .prepare(module, op, &Cell::new(0))
        .unwrap();
    let owner = Owner::candidate(op, selected);
    let context = HostContext::default();
    let library_calls = Cell::new(0);
    let mut out = [123.0];
    let views = || HostViews {
        input: &[2.0],
        weight: &[5.0],
        bias: Some(&[7.0]),
        residual: Some(&[11.0]),
    };
    let lengths = op.lengths();
    let Operation::Spatial { boundary, conv, .. } = op else {
        unreachable!()
    };
    for call in [
        Call {
            operation: operation(Family::Dense),
            lengths,
            residual_len: Some(lengths[3]),
        },
        Call {
            operation: Operation::Spatial {
                boundary,
                conv,
                epilogue: Epilogue::BiasRelu,
            },
            lengths,
            residual_len: Some(lengths[3]),
        },
        Call {
            operation: op,
            lengths,
            residual_len: None,
        },
        Call {
            operation: op,
            lengths,
            residual_len: Some(1),
        },
        Call {
            operation: op,
            lengths: [1; 4],
            residual_len: Some(1),
        },
    ] {
        assert!(
            owner
                .dispatch(
                    (views(), out.as_mut_slice()),
                    |plan, (views, output)| plan.run(&context, &call, views, output),
                    |(_, _)| {
                        library_calls.set(library_calls.get() + 1);
                        Ok(())
                    }
                )
                .is_err()
        );
        assert_eq!(out, [123.0]);
    }
    assert!(context.borrow().is_empty());
    assert_eq!(library_calls.get(), 0);
}

// these adapters have no artifacts and cannot be selected outside this CPU test
struct FakePort;
fn no_gpu<T>() -> Result<T, PlanError> {
    Err(PlanError::Cuda(error(
        "CPU registration fixture has no GPU port",
    )))
}
impl DenseCandidate for FakePort {
    type Pin = FakePin;
    const COVERAGE: Coverage = COVERAGE;
    const SPECIAL_VALUES: SpecialValues = CONTRACT;
    fn coverage(tier: PtxTier) -> Coverage {
        FakeFactory::<2>::coverage(tier)
    }
    fn implemented_pin(
        _spec: DenseSpec,
        _tier: PtxTier,
        _device: &crate::inference::cuda::device::DeviceAttributes,
    ) -> Result<FakePin, PlanError> {
        Ok(FakePin { factor: 3 })
    }
    fn plan(
        _runtime: &CudaRuntime,
        _kernels: &LoadedKernels,
        _spec: DenseSpec,
        _weight: &cudarc::driver::CudaSlice<f32>,
        _bias: &cudarc::driver::CudaSlice<f32>,
        _pin: FakePin,
    ) -> Result<Self, PlanError> {
        no_gpu()
    }
    fn enqueue(
        &self,
        _x: &CudaView<'_, f32>,
        _output: &mut CudaViewMut<'_, f32>,
        _phases: &Phases,
        _runtime: &CudaRuntime,
    ) -> Result<(), CudaError> {
        Err(error("no GPU fixture"))
    }
}
impl SegConvCandidate for FakePort {
    type Pin = FakePin;
    const COVERAGE: Coverage = COVERAGE;
    const SPECIAL_VALUES: SpecialValues = CONTRACT;
    fn coverage(tier: PtxTier) -> Coverage {
        FakeFactory::<2>::coverage(tier)
    }
    fn implemented_pin(
        _spec: SegConvSpec,
        _tier: PtxTier,
        _device: &crate::inference::cuda::device::DeviceAttributes,
    ) -> Result<FakePin, PlanError> {
        Ok(FakePin { factor: 3 })
    }
    fn plan(
        _runtime: &CudaRuntime,
        _kernels: &LoadedKernels,
        _spec: SegConvSpec,
        _weight: &cudarc::driver::CudaSlice<f32>,
        _pin: FakePin,
    ) -> Result<Self, PlanError> {
        no_gpu()
    }
    fn enqueue(
        &self,
        _x: &CudaView<'_, f32>,
        _output: &mut CudaViewMut<'_, f32>,
        _phases: &Phases,
        _runtime: &CudaRuntime,
    ) -> Result<(), CudaError> {
        Err(error("no GPU fixture"))
    }
}
impl ConvCandidate for FakePort {
    type Pin = FakePin;
    const COVERAGE: Coverage = COVERAGE;
    const SPECIAL_VALUES: SpecialValues = CONTRACT;
    fn coverage(tier: PtxTier) -> Coverage {
        FakeFactory::<2>::coverage(tier)
    }
    fn implemented_pin(_spec: &ConvLayerSpec<'_>) -> Result<FakePin, PlanError> {
        Ok(FakePin { factor: 3 })
    }
    fn plan(
        _runtime: &CudaRuntime,
        _kernels: &LoadedKernels,
        _spec: ConvLayerSpec<'_>,
        _pin: FakePin,
    ) -> Result<Self, PlanError> {
        no_gpu()
    }
    fn enqueue(
        &self,
        _inputs: ConvInputs<'_, '_>,
        _output: &mut CudaViewMut<'_, f32>,
        _phases: &Phases,
        _stream: &cudarc::driver::CudaStream,
    ) -> Result<(), CudaError> {
        Err(error("no GPU fixture"))
    }
}

#[test]
fn all_three_typed_registrations_share_selection_and_no_port_default_is_closed() {
    FACTORIES.with(|f| f.borrow_mut().clear());
    for family in [Family::Dense, Family::Temporal, Family::Spatial] {
        assert!(selected_factory(operation(family)).is_err());
    }
    let dense = request(KernelModule::Segdense, PtxTier::Sm80, b"CPU fixture only");
    let spatial = request(KernelModule::Wideconv, PtxTier::Sm80, b"CPU fixture only");
    register_dense::<FakePort>(dense).unwrap();
    register_temporal::<FakePort>(dense).unwrap();
    register_spatial::<FakePort>(spatial).unwrap();
    for family in [Family::Dense, Family::Temporal, Family::Spatial] {
        let op = operation(family);
        let factory = selected_factory(op).unwrap();
        assert_eq!(factory.family(), family);
        assert_eq!(
            factory.request(),
            if family == Family::Spatial {
                spatial
            } else {
                dense
            }
        );
    }
    assert!(register_dense::<FakePort>(dense).is_err());
    FACTORIES.with(|f| f.borrow_mut().clear());
    register_dense::<FakePort>(spatial).unwrap();
    assert!(selected_factory(operation(Family::Dense)).is_err());
    FACTORIES.with(|f| f.borrow_mut().clear());
}

fn execute_without_residual<F>(op: Operation, expected: f32)
where
    F: for<'a> Factory<Pin = FakePin, Plan = FakePlan, Resources<'a> = &'a Cell<usize>>,
{
    let module = request(op.area(), PtxTier::Sm80, b"CPU test identity only");
    let builds = Cell::new(0);
    let selected = Registration::<F>::new(module)
        .prepare(module, op, &builds)
        .unwrap();
    assert_eq!(builds.get(), 1);
    let owner = Owner::candidate(op, selected);
    let lengths = op.lengths();
    let input = vec![2.0; lengths[0]];
    let weight = vec![5.0; lengths[1]];
    let bias = vec![7.0; lengths[2]];
    let bias = if bias.is_empty() {
        None
    } else {
        Some(bias.as_slice())
    };
    let mut output = vec![0.0; lengths[3]];
    let context = HostContext::default();
    let call = Call {
        operation: op,
        lengths,
        residual_len: None,
    };
    owner
        .dispatch(
            (
                HostViews {
                    input: &input,
                    weight: &weight,
                    bias,
                    residual: None,
                },
                output.as_mut_slice(),
            ),
            |plan, (views, output)| plan.run(&context, &call, views, output),
            |_| panic!("candidate cannot call Library"),
        )
        .unwrap();
    assert!(output.iter().all(|value| *value == expected));
    assert_eq!(
        *context.borrow(),
        [[
            input.as_ptr() as usize,
            weight.as_ptr() as usize,
            bias.map_or(0, |v| v.as_ptr() as usize),
            0
        ]]
    );
}

#[test]
fn dense_and_temporal_cpu_owners_execute_the_same_registered_factory_protocol() {
    execute_without_residual::<FakeFactory<0>>(operation(Family::Dense), 37.0);
    execute_without_residual::<FakeFactory<1>>(operation(Family::Temporal), 30.0);
}

#[test]
fn coverage_and_receipt_queries_keep_the_no_port_default_closed() {
    FACTORIES.with(|factories| factories.borrow_mut().clear());
    super::RECEIPTS.with(|receipts| receipts.borrow_mut().clear());
    assert!(
        super::coverage(Family::Dense, PtxTier::Sm80)
            .entries()
            .is_empty()
    );
    assert_eq!(super::planned(), json!([]));
    let module = request(KernelModule::Segdense, PtxTier::Sm80, b"CPU fixture only");
    register_dense::<FakePort>(module).unwrap();
    assert!(super::coverage(Family::Dense, PtxTier::Sm80).covers("linear0", 1, CudaMath::Fp32));
    assert!(
        super::coverage(Family::Dense, PtxTier::Sm75)
            .entries()
            .is_empty()
    );
    assert!(
        super::coverage(Family::Spatial, PtxTier::Sm80)
            .entries()
            .is_empty()
    );
    // the preload entry accepts a runtime only; no fake GPU identity is constructed here
    let _preload: fn(&CudaRuntime, Operation) -> Result<(), CudaError> = super::preload;
    FACTORIES.with(|factories| factories.borrow_mut().clear());
}

#[test]
fn receipts_retain_the_selected_pin_and_reject_conflicting_tuple_identity() {
    super::RECEIPTS.with(|receipts| receipts.borrow_mut().clear());
    let op = operation(Family::Spatial);
    let module = request(op.area(), PtxTier::Sm80, b"CPU fixture only");
    let selected = Registration::<FakeFactory>::new(module)
        .prepare(module, op, &Cell::new(0))
        .unwrap();
    let receipt = selected.receipt();
    super::record_receipt(receipt.clone()).unwrap();
    super::record_receipt(receipt.clone()).unwrap();
    assert_eq!(super::planned(), json!([receipt]));
    let mut changed_pin = selected.receipt();
    changed_pin["pin"]["test_only_factor"] = json!(99);
    assert!(super::record_receipt(changed_pin).is_err());
    let mut changed_artifact = selected.receipt();
    changed_artifact["artifact"]["sha256"] =
        json!(ArtifactHash::of(b"another CPU fixture").to_string());
    assert!(super::record_receipt(changed_artifact).is_err());
    assert_eq!(super::planned(), json!([selected.receipt()]));
    super::RECEIPTS.with(|receipts| receipts.borrow_mut().clear());
}
