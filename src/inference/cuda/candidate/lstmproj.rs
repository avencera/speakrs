//! The four bidirectional LSTM layers on cuda-oxide input projections and a
//! persistent recurrence, with no library call
//!
//! Per layer, a tiled projection computes `x · Wᵀ` for every step of each direction,
//! with the gate columns reordered so each hidden unit's four gates are adjacent.
//! Then one cooperative launch per direction runs that direction's whole recurrence
//! with the recurrent weights held on chip; see the kernel crate's `lstmproj` area for
//! the work split and the hidden-state exchange. The reverse direction runs on a side
//! stream, so both directions share the GPU instead of running one after the other;
//! the next layer waits for both
//!
//! [`RecurrencePlan`] picks one of three recurrence kernels with bitwise identical
//! results: the wide kernel's sixteen eight-unit blocks per tile for up to eight
//! windows, eight-window tiles of sixteen-unit blocks for larger batches whose grids
//! fit the GPU together, and on the A100 one block per window and direction that holds
//! the whole recurrent matrix and needs no exchange
//!
//! The exchange owns an allocation with a 2 MiB-aligned active subview. Producer
//! lines, step parities, batch tiles and directions have disjoint padded regions.
//! The gate stream and recurrent weights also have aligned active bases, so no
//! allocation order is assumed
//!
//! Every layer writes into the stack output. That is safe because a layer's
//! projections read the previous layer's complete output before its recurrence
//! overwrites it

pub(super) mod layout;

use cudarc::driver::sys::CUfunction_attribute;
use cudarc::driver::{
    CudaFunction, CudaSlice, CudaStream, CudaView, CudaViewMut, DevicePtr, DevicePtrMut,
    DeviceRepr, LaunchConfig, PushKernelArg, ValidAsZeroBits,
};

use self::layout::{
    AlignedSpan, ExchangeLayout, GATE_COLUMNS, GROUPS, HIDDEN, KERNEL, ProjectionLayout,
    RESIDENT_KERNEL, RESIDENT_SHARED_BYTES, RESIDENT_STATIC_SHARED_BYTES, RESIDENT_THREADS,
    STATE_TILE, Schedule, TENSOR_SHARED_BYTES, THREADS, TILED_GROUPS, TILED_KERNEL, TILED_ROWS,
    TILED_THREADS, pack_bias, pack_directions, pack_input_directions, padded_input, tiled_exchange,
};
use super::{
    Batches, Coverage, CoverageEntry, DeviceAttributes, Direction, FiniteContract, GeometryError,
    InfinityContract, LstmCandidate, LstmPhases, LstmPin, LstmProjection, LstmSpec, Maths,
    NanContract, Op, PlanError, Scratch, SideStream, SignedZeroContract, SpecialValues,
};
use crate::inference::cuda::{
    ComputeCapability, CudaError, CudaMath, CudaRuntime, KernelModule, LoadedKernels, PtxTier,
};

const SPK_LSTM_CLEAR: &str = "spk_lstm_clear";
const SPK_LSTM_PROJECTION: &str = "spk_lstm_projection";
const SPK_LSTM_PROJECTION_SMALL: &str = "spk_lstm_projection_small";
const SPK_LSTM_PROJECTION_TF32: &str = "spk_lstm_projection_tf32";

/// Kernel entries this host plan may load, in every tier
pub(crate) const REQUIRED_KERNELS: [&str; 7] = [
    KERNEL,
    TILED_KERNEL,
    RESIDENT_KERNEL,
    SPK_LSTM_CLEAR,
    SPK_LSTM_PROJECTION,
    SPK_LSTM_PROJECTION_SMALL,
    SPK_LSTM_PROJECTION_TF32,
];

/// LSTM input width of the first layer
const FEATURES: usize = 60;
const DIRECTIONS: [Direction; 2] = [Direction::Forward, Direction::Reverse];
/// Projection rows below which the FP32 64 by 64 tile wins: smaller grids do not
/// amortize the larger tiles' register footprint
const SMALL_PROJECTION_ROWS: usize = 2048;
/// Fewest windows for which the eight-window recurrence ran faster than the wide one
/// on the RTX 5060 Ti and RTX 4060 Ti. It needs two tiles: one tile occupies only 16
/// SMs, and up to eight windows ran as fast or faster on the wide kernel
const TILED_MIN_BATCH: usize = 9;
/// The capability whose SMs ran the single-block recurrence faster than both exchange
/// kernels: on an A100 at batch 32 it took 0.97 ms per layer against 1.38 ms for the
/// tiled kernel. Hopper SMs also hold the matrix but were not measured
const RESIDENT_CAPABILITY: ComputeCapability = ComputeCapability::new(8, 0);
/// Fewest windows that run the single-block recurrence. From two windows up it beat the
/// wide kernel by 18% or more on an A100. At one window the two were close and runs
/// disagreed: one TF32 stack took 3.21 ms on the wide kernel against 4.00 ms resident,
/// another 4.13 against 4.03 ms. So a lone window keeps the wide kernel
const RESIDENT_MIN_BATCH: usize = 2;

/// One layer's weights in the packed gate order, both directions stacked
#[derive(Debug)]
struct PackedLayer {
    /// Input width: 60 for the first layer, 256 after it
    input: usize,
    /// `[2, padded_input, 512]` for SIMT, `[2, 512, padded_input]` for TF32
    w: CudaSlice<f32>,
    /// `[2, 512, 128]`
    r: AlignedScratch<f32>,
    /// `[2, 512]`, `Wb + Rb`
    bias: CudaSlice<f32>,
}

/// The library-free stack for one batch size: packed weights, the projection and
/// recurrence kernels, the tile schedule, and the buffers and side stream its passes
/// reuse
#[derive(Debug)]
pub(crate) struct Oxide {
    projection: InputProjection,
    recurrence: CudaFunction,
    clear: CudaFunction,
    layers: [PackedLayer; 4],
    batch: usize,
    steps: usize,
    plan: RecurrencePlan,
    /// `[2, batch * steps, 512]`: each direction's input projection
    gates: AlignedScratch<f32>,
    /// Aligned, padded flag and value words, cleared before each pass; the
    /// single-block recurrence exchanges nothing
    state: Option<Exchange>,
    /// Runs the reverse direction when the schedule makes the directions concurrent
    side: SideStream,
}

impl LstmCandidate for Oxide {
    // the recurrence computes in FP32 in both modes; TF32 changes only the projection
    // where `device_pin` selects tensor fragments
    const COVERAGE: Coverage = Coverage(&[CoverageEntry {
        layers: &["lstm.stack"],
        batches: Batches::All,
        maths: Maths::All,
    }]);

    // the gates saturate, but `exp_f32` clamps its argument before the exponent, so how
    // NaN and infinite pre-activations surface has not been established
    const SPECIAL_VALUES: SpecialValues = SpecialValues {
        finite: FiniteContract::BoundedActivation { headroom: 2 },
        nan: NanContract::Unspecified,
        infinity: InfinityContract::Unspecified,
        signed_zero: SignedZeroContract::Unspecified,
    };

    /// FP32 projections, which every tier and device runs
    fn implemented_pin(spec: &LstmSpec<'_>) -> Result<LstmPin, PlanError> {
        Ok(LstmPin::Projected(fp32_projection(
            spec.batch * spec.frames,
        )))
    }

    fn device_pin(
        device: &DeviceAttributes,
        tier: PtxTier,
        spec: &LstmSpec<'_>,
    ) -> Result<LstmPin, PlanError> {
        Ok(LstmPin::Projected(projection_rule(
            ProjectionTarget {
                capability: device.capability(),
                shared_optin_bytes: device.shared_optin_bytes(),
                tier,
            },
            spec.batch,
            spec.batch * spec.frames,
            spec.math,
        )))
    }

    fn plan(
        runtime: &CudaRuntime,
        kernels: &LoadedKernels,
        spec: LstmSpec<'_>,
        pin: LstmPin,
    ) -> Result<Self, PlanError> {
        let recurrence = RecurrencePlan::select(runtime, kernels, spec.batch)?;
        Self::with_recurrence(runtime, kernels, spec, pin, recurrence)
    }

    fn enqueue(
        &self,
        input: &CudaView<'_, f32>,
        output: &mut CudaViewMut<'_, f32>,
        phases: &LstmPhases<'_>,
        stream: &CudaStream,
    ) -> Result<(), CudaError> {
        let rows = self.batch * self.steps;
        check_len("lstmproj input", rows * FEATURES, input.len())?;
        check_len("lstmproj output", rows * 2 * HIDDEN, output.len())?;

        phases.op(Op::Prologue, || self.clear_state(stream))?;
        for (index, layer) in self.layers.iter().enumerate() {
            for direction in DIRECTIONS {
                // the library helper stays unused; this scope runs the custom projection
                phases.input_proj(index, direction, |_| {
                    let d = direction_index(direction);
                    let weights = GATE_COLUMNS * padded_input(layer.input);
                    let w = layer.w.slice(d * weights..(d + 1) * weights);
                    let mut storage = self.gates.storage.get();
                    let mut gates = storage.slice_mut(self.gates.span.range());
                    let mut gates =
                        gates.slice_mut(d * rows * GATE_COLUMNS..(d + 1) * rows * GATE_COLUMNS);
                    let previous = output.as_view();
                    let x = if index == 0 { input } else { &previous };
                    self.projection
                        .enqueue(stream, x, &w, &mut gates, rows, layer.input)
                })?;
            }
            self.recur(index, layer, output, phases, stream)?;
        }

        Ok(())
    }
}

/// What the projection rule reads from the device and the loaded module
#[derive(Debug, Clone, Copy)]
pub(super) struct ProjectionTarget {
    pub(super) capability: ComputeCapability,
    pub(super) shared_optin_bytes: u32,
    pub(super) tier: PtxTier,
}

/// TF32 mode uses tensor projections where the loaded tier has them, the device can
/// host their tile, enough rows amortize it and they were accurate enough; every other
/// case uses FP32
pub(super) fn projection_rule(
    target: ProjectionTarget,
    batch: usize,
    rows: usize,
    math: CudaMath,
) -> LstmProjection {
    let tensor = math == CudaMath::Tf32
        && rows >= SMALL_PROJECTION_ROWS
        && target.tier >= PtxTier::Sm80
        && target.shared_optin_bytes >= TENSOR_SHARED_BYTES
        && tf32_projection_accurate(target.capability, batch);
    if tensor {
        return LstmProjection::Tensor;
    }

    fp32_projection(rows)
}

/// The FP32 projection tile for `rows` projection rows
fn fp32_projection(rows: usize) -> LstmProjection {
    if rows < SMALL_PROJECTION_ROWS {
        LstmProjection::Small
    } else {
        LstmProjection::Large
    }
}

/// Whether TF32 tensor projections stayed within the accuracy checks on this device
///
/// Development checks against an f64 reference and the cuDNN TF32 stack found two
/// exceptions, where FP32 projections run in TF32 mode instead. Other devices were not
/// measured and keep tensor projections
fn tf32_projection_accurate(capability: ComputeCapability, batch: usize) -> bool {
    // RTX 5060 Ti: the stage output was less accurate than cuDNN TF32 at every batch,
    // from both the sm80 and the sm120 PTX
    if capability == ComputeCapability::new(12, 0) {
        return false;
    }

    // RTX 4060 Ti: cuDNN runs two of the eight projections in FP32 at b32, and all-TF32
    // projections exceeded 1.1 times its f64 error there; b33 and b64 stayed within it
    !(capability == ComputeCapability::new(8, 9) && batch == 32)
}

impl Oxide {
    /// [`LstmCandidate::plan`] with an explicit recurrence plan, so both kernels can run
    /// the same batch
    pub(crate) fn with_recurrence(
        runtime: &CudaRuntime,
        kernels: &LoadedKernels,
        spec: LstmSpec<'_>,
        pin: LstmPin,
        plan: RecurrencePlan,
    ) -> Result<Self, PlanError> {
        let LstmPin::Projected(projection) = pin else {
            return Err(PlanError::Geometry(GeometryError::Invalid {
                context: "lstmproj plan",
                reason: format!("pin {pin:?} belongs to the lstm area"),
            }));
        };
        let projection = InputProjection::new(runtime.device(), kernels, projection)?;
        let clear = kernels.function(SPK_LSTM_CLEAR)?;
        let recurrence = plan.function(kernels)?;
        if plan == RecurrencePlan::Resident {
            recurrence.set_attribute(
                CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                RESIDENT_SHARED_BYTES as i32,
            )?;
        }

        let stream = runtime.stream();
        let layout = projection.layout();
        let pack = |index: usize| -> Result<PackedLayer, CudaError> {
            let layer = spec.layers[index];
            let input = if index == 0 { FEATURES } else { 2 * HIDDEN };
            check_len("lstmproj input width", input, layer.input)?;
            check_len("lstmproj W", 2 * GATE_COLUMNS * input, layer.w.len())?;
            check_len("lstmproj R", 2 * GATE_COLUMNS * HIDDEN, layer.r.len())?;
            check_len("lstmproj B", 4 * GATE_COLUMNS, layer.b.len())?;
            Ok(PackedLayer {
                input,
                w: stream.clone_htod(&pack_input_directions(layer.w, input, layout))?,
                r: AlignedScratch::from_host(runtime, &pack_directions(layer.r, HIDDEN))?,
                bias: stream.clone_htod(&pack_bias(layer.b))?,
            })
        };

        Ok(Self {
            projection,
            recurrence,
            clear,
            layers: [pack(0)?, pack(1)?, pack(2)?, pack(3)?],
            batch: spec.batch,
            steps: spec.frames,
            plan,
            gates: AlignedScratch::zeros(runtime, 2 * spec.batch * spec.frames * GATE_COLUMNS)?,
            state: plan
                .exchange_layout()
                .map(|layout| Exchange::new(runtime, layout))
                .transpose()?,
            side: SideStream::new(runtime)?,
        })
    }

    /// Zeroes the exchange state, so no word from an earlier pass carries a flag this
    /// pass expects
    fn clear_state(&self, stream: &CudaStream) -> Result<(), CudaError> {
        let Some(exchange) = &self.state else {
            return Ok(());
        };
        let mut storage = exchange.buffer.storage.get();
        let mut state = storage.slice_mut(exchange.buffer.span.range());
        let len = state.len();
        let len_arg = len as u64;
        let mut builder = stream.launch_builder(&self.clear);
        builder.arg(&mut state).arg(&len_arg);
        // SAFETY: `spk_lstm_clear` takes one slice, pointer and length, and writes only
        // inside it
        unsafe { builder.launch(LaunchConfig::for_num_elems(to_u32("lstmproj clear", len)?)) }?;
        Ok(())
    }

    /// Both directions' recurrences of one layer, the reverse one on the side stream
    /// when the schedule allows it
    fn recur(
        &self,
        index: usize,
        layer: &PackedLayer,
        output: &mut CudaViewMut<'_, f32>,
        phases: &LstmPhases<'_>,
        stream: &CudaStream,
    ) -> Result<(), CudaError> {
        let context = "lstmproj recurrence";
        let rows = self.batch * self.steps;
        let Some(exchange) = &self.state else {
            // one launch runs both directions
            return phases.recurrence(index, Direction::Forward, || {
                self.launch_resident(stream, layer, output)
            });
        };
        let mut storage = exchange.buffer.storage.get();
        let mut state = storage.slice_mut(exchange.buffer.span.range());
        let gate_storage = self.gates.storage.get();
        let gates = gate_storage.slice(self.gates.span.range());
        // cudarc orders accesses per allocation, so taking these pointers on the side
        // stream would make it wait for the forward kernel's write and serialize the
        // directions. Both are taken on the given stream, and their write events are
        // recorded there after the merge
        let (output, _output_record) = output.device_ptr_mut(stream);
        let (state, _state_record) = state.device_ptr_mut(stream);
        let concurrent = self.plan.concurrent();
        if concurrent {
            self.side.split(stream)?;
        }

        for direction in DIRECTIONS {
            let d = direction_index(direction);
            let launch = DirectionLaunch {
                gates: gates.slice(d * rows * GATE_COLUMNS..(d + 1) * rows * GATE_COLUMNS),
                direction: d,
                // the kernel takes this direction's first output column; adding it in
                // the kernel made ptxas (CUDA 13.0, sm_120) truncate the pointer to
                // 32 bits
                output: output + (d * HIDDEN * size_of::<f32>()) as u64,
                state: state + (exchange.layout.direction(d) * size_of::<u64>()) as u64,
                flag_base: to_u32(context, index * self.steps)?,
            };
            let on: &CudaStream = if concurrent && d == 1 {
                self.side.stream()
            } else {
                stream
            };
            phases.recurrence(index, direction, || self.launch(on, layer, launch))?;
        }

        if concurrent {
            self.side.merge(stream)?;
        }
        Ok(())
    }

    /// The cooperative launches of one direction of one layer
    fn launch(
        &self,
        stream: &CudaStream,
        layer: &PackedLayer,
        launch: DirectionLaunch<'_>,
    ) -> Result<(), CudaError> {
        let context = "lstmproj recurrence";
        let DirectionLaunch {
            gates,
            direction,
            output,
            state,
            flag_base,
        } = launch;
        let recurrent = GATE_COLUMNS * HIDDEN;
        let r_storage = layer.r.storage.get();
        let r = r_storage.slice(layer.r.span.range());
        let r = r.slice(direction * recurrent..(direction + 1) * recurrent);
        let bias = layer
            .bias
            .slice(direction * GATE_COLUMNS..(direction + 1) * GATE_COLUMNS);
        let lens = [gates.len(), r.len(), bias.len()].map(|len| len as u64);
        let batch = to_u32(context, self.batch)?;
        let steps = to_u32(context, self.steps)?;
        let direction = to_u32(context, direction)?;
        let schedule = match self.plan {
            RecurrencePlan::Wide(schedule) => schedule,
            RecurrencePlan::Resident => {
                return Err(CudaError::Unsupported {
                    context,
                    reason: "the single-block recurrence has no per-direction launch".to_owned(),
                });
            }
            RecurrencePlan::Tiled { tiles } => {
                let first = 0u32;
                let mut builder = stream.launch_builder(&self.recurrence);
                builder
                    .arg(&gates)
                    .arg(&lens[0])
                    .arg(&r)
                    .arg(&lens[1])
                    .arg(&bias)
                    .arg(&lens[2])
                    .arg(&output)
                    .arg(&state)
                    .arg(&batch)
                    .arg(&steps)
                    .arg(&direction)
                    .arg(&first)
                    .arg(&flag_base);
                let config = LaunchConfig {
                    grid_dim: (TILED_GROUPS as u32, to_u32(context, tiles)?, 1),
                    block_dim: (TILED_THREADS, 1, 1),
                    shared_mem_bytes: 0,
                };
                // SAFETY: the arguments follow the PTX signature of
                // `spk_lstm_recurrence_tiled`, which is `spk_lstm_recurrence`'s without
                // `tile_rows`. Views, alignment and flags are as for the wide launch
                // below; the state holds `tiled_exchange::TILE` words per tile. The plan
                // chose this kernel only where both directions' grids fit the context's
                // joint cooperative budget
                unsafe { builder.launch_cooperative(config) }?;
                return Ok(());
            }
        };
        let tile_rows = to_u32(context, schedule.tile_rows)?;

        for (first, tiles) in schedule.launches() {
            let first = to_u32(context, first)?;
            let mut builder = stream.launch_builder(&self.recurrence);
            builder
                .arg(&gates)
                .arg(&lens[0])
                .arg(&r)
                .arg(&lens[1])
                .arg(&bias)
                .arg(&lens[2])
                .arg(&output)
                .arg(&state)
                .arg(&batch)
                .arg(&steps)
                .arg(&direction)
                .arg(&first)
                .arg(&tile_rows)
                .arg(&flag_base);
            let config = LaunchConfig {
                grid_dim: (GROUPS as u32, to_u32(context, tiles)?, 1),
                block_dim: (THREADS, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: the arguments follow the PTX signature of `spk_lstm_recurrence`
            // (pointer and length per slice, the output and state pointers, then six
            // scalars). The gate, weight and bias views hold exactly one direction;
            // the output holds `[batch, steps, 256]` and the state this direction's
            // padded tiles, both checked or sized by the plan. The gate and R views
            // start on aligned bases, as the kernel's vector loads need. Flags grow by
            // layer, so no word from an earlier layer of this pass, or from before the
            // pass's zeroing, can match. The launch is cooperative and the schedule
            // keeps every block that may run at the same time within the GPU's
            // resident capacity, which the kernel's cross-block waits need
            unsafe { builder.launch_cooperative(config) }?;
        }

        Ok(())
    }
}

impl Oxide {
    /// The single launch of both directions of one layer, one block per window and
    /// direction
    fn launch_resident(
        &self,
        stream: &CudaStream,
        layer: &PackedLayer,
        output: &mut CudaViewMut<'_, f32>,
    ) -> Result<(), CudaError> {
        let context = "lstmproj resident recurrence";
        let gate_storage = self.gates.storage.get();
        let gates = gate_storage.slice(self.gates.span.range());
        let r_storage = layer.r.storage.get();
        let r = r_storage.slice(layer.r.span.range());
        let bias = layer.bias.as_view();
        let lens = [gates.len(), r.len(), bias.len()].map(|len| len as u64);
        let (output, _output_record) = output.device_ptr_mut(stream);
        // the kernel takes each direction's first output column; adding it in the
        // kernel made ptxas (CUDA 13.0, sm_120) truncate the pointer to 32 bits
        let reverse = output + (HIDDEN * size_of::<f32>()) as u64;
        let batch = to_u32(context, self.batch)?;
        let steps = to_u32(context, self.steps)?;
        let mut builder = stream.launch_builder(&self.recurrence);
        builder
            .arg(&gates)
            .arg(&lens[0])
            .arg(&r)
            .arg(&lens[1])
            .arg(&bias)
            .arg(&lens[2])
            .arg(&output)
            .arg(&reverse)
            .arg(&batch)
            .arg(&steps);
        let config = LaunchConfig {
            grid_dim: (batch, 2, 1),
            block_dim: (RESIDENT_THREADS, 1, 1),
            shared_mem_bytes: RESIDENT_SHARED_BYTES,
        };
        // SAFETY: the arguments follow the PTX signature of
        // `spk_lstm_recurrence_resident` (pointer and length per slice, both output
        // pointers, then two scalars). The gate view holds `[2, batch * steps, 512]`,
        // R `[2, 512, 128]` on an aligned base and the bias `[2, 512]`; the output holds
        // `[batch, steps, 256]`. Blocks never wait for each other, so the launch needs
        // no residency guarantee, and the plan raised the function's dynamic shared
        // limit to the bytes requested here
        unsafe { builder.launch(config) }?;
        Ok(())
    }
}

/// Owns a naturally allocated buffer and its typed, explicitly aligned subview
#[derive(Debug)]
struct AlignedScratch<T> {
    storage: Scratch<T>,
    span: AlignedSpan<T>,
}

impl<T: DeviceRepr + ValidAsZeroBits> AlignedScratch<T> {
    /// Copies immutable weights into the aligned active subview
    fn from_host(runtime: &CudaRuntime, values: &[T]) -> Result<Self, CudaError> {
        let buffer = Self::zeros(runtime, values.len())?;
        {
            let mut storage = buffer.storage.get();
            let mut view = storage.slice_mut(buffer.span.range());
            runtime.stream().memcpy_htod(values, &mut view)?;
        }
        Ok(buffer)
    }

    fn zeros(runtime: &CudaRuntime, len: usize) -> Result<Self, CudaError> {
        let storage = Scratch::zeros(runtime, AlignedSpan::<T>::allocation_len(len))?;
        let span = {
            let allocation = storage.get();
            let (address, _record) = allocation.device_ptr(runtime.stream());
            AlignedSpan::new(address, len)
        };
        Ok(Self { storage, span })
    }
}

/// The aligned flag and value allocation and its padded ownership layout
#[derive(Debug)]
struct Exchange {
    buffer: AlignedScratch<u64>,
    layout: ExchangeLayout,
}

impl Exchange {
    fn new(runtime: &CudaRuntime, layout: ExchangeLayout) -> Result<Self, CudaError> {
        let buffer = AlignedScratch::zeros(runtime, layout.words())?;
        Ok(Self { buffer, layout })
    }
}

/// The recurrence kernel a plan runs and how its batch tiles launch
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RecurrencePlan {
    /// One block per window and direction holding the whole recurrent matrix, on
    /// [`RESIDENT_CAPABILITY`] parts that can host it, from [`RESIDENT_MIN_BATCH`]
    /// windows; no exchange and no cooperative launch
    Resident,
    /// Sixteen eight-unit blocks per tile of 8, 16 or 32 windows; the faster kernel for
    /// a lone window, and the fallback when the tiled grids do not fit together
    Wide(Schedule),
    /// Eight sixteen-unit blocks per eight-window tile, both directions resident at once
    Tiled { tiles: usize },
}

impl RecurrencePlan {
    /// The measured choice for `batch` windows on this context
    pub(crate) fn select(
        runtime: &CudaRuntime,
        kernels: &LoadedKernels,
        batch: usize,
    ) -> Result<Self, PlanError> {
        if batch >= RESIDENT_MIN_BATCH && Self::resident_fits(runtime, kernels)? {
            return Ok(Self::Resident);
        }
        if let Some(plan) = Self::tiled(batch, Self::tiled_budget(runtime, kernels)?) {
            return Ok(plan);
        }

        Self::wide(runtime, kernels, batch)
    }

    /// Whether this context runs the single-block recurrence: a measured capability
    /// whose blocks can take its dynamic shared memory, and at least one resident block
    /// per SM once the function allows that much
    pub(crate) fn resident_fits(
        runtime: &CudaRuntime,
        kernels: &LoadedKernels,
    ) -> Result<bool, PlanError> {
        let device = runtime.device();
        if !Self::resident_device(device.capability(), device.shared_optin_bytes()) {
            return Ok(false);
        }

        let resident = kernels.function(RESIDENT_KERNEL)?;
        resident.set_attribute(
            CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            RESIDENT_SHARED_BYTES as i32,
        )?;
        let per_sm = resident.occupancy_max_active_blocks_per_multiprocessor(
            RESIDENT_THREADS,
            RESIDENT_SHARED_BYTES as usize,
            None,
        )?;
        Ok(per_sm >= 1)
    }

    /// The device rule of the single-block recurrence, before the occupancy check
    pub(crate) fn resident_device(capability: ComputeCapability, shared_optin_bytes: u32) -> bool {
        capability == RESIDENT_CAPABILITY
            && shared_optin_bytes >= RESIDENT_SHARED_BYTES + RESIDENT_STATIC_SHARED_BYTES
    }

    /// Blocks of the tiled kernel that both directions' cooperative grids can keep
    /// resident together, or `None` when the context's budget is unknown
    pub(crate) fn tiled_budget(
        runtime: &CudaRuntime,
        kernels: &LoadedKernels,
    ) -> Result<Option<usize>, PlanError> {
        let tiled = kernels.function(TILED_KERNEL)?;
        let capacity = runtime.cooperative_capacity(&tiled, TILED_THREADS, 0)?;
        Ok(runtime
            .concurrent_cooperative_capacity(&tiled, TILED_THREADS, 0)?
            .map(|joint| joint.min(capacity)))
    }

    /// The wide kernel's tile schedule for `batch` windows on this context
    pub(crate) fn wide(
        runtime: &CudaRuntime,
        kernels: &LoadedKernels,
        batch: usize,
    ) -> Result<Self, PlanError> {
        let wide = kernels.function(KERNEL)?;
        let capacity = runtime.cooperative_capacity(&wide, THREADS, 0)?;
        let concurrent_capacity = runtime.concurrent_cooperative_capacity(&wide, THREADS, 0)?;
        let schedule = Schedule::for_groups(GROUPS, batch, capacity, concurrent_capacity)?;
        Ok(Self::Wide(schedule))
    }

    fn function(self, kernels: &LoadedKernels) -> Result<CudaFunction, PlanError> {
        let name = match self {
            Self::Wide(_) => KERNEL,
            Self::Tiled { .. } => TILED_KERNEL,
            Self::Resident => RESIDENT_KERNEL,
        };
        Ok(kernels.function(name)?)
    }

    /// The tiled plan when `batch` is large enough and both directions' grids fit
    /// `budget` resident blocks together; an unknown budget keeps the wide kernel
    pub(crate) fn tiled(batch: usize, budget: Option<usize>) -> Option<Self> {
        if batch < TILED_MIN_BATCH {
            return None;
        }

        Self::tiled_fit(batch, budget)
    }

    /// The tiled plan for any batch whose grids fit `budget` together
    pub(crate) fn tiled_fit(batch: usize, budget: Option<usize>) -> Option<Self> {
        let tiles = batch.div_ceil(TILED_ROWS);
        let fits = budget.is_some_and(|budget| 2 * TILED_GROUPS * tiles <= budget);
        fits.then_some(Self::Tiled { tiles })
    }

    fn concurrent(self) -> bool {
        match self {
            Self::Wide(schedule) => schedule.concurrent,
            Self::Tiled { .. } => true,
            // one launch runs both directions without the side stream
            Self::Resident => false,
        }
    }

    /// The hidden-state exchange the kernel needs, if any
    fn exchange_layout(self) -> Option<ExchangeLayout> {
        match self {
            Self::Wide(schedule) => Some(ExchangeLayout::new(schedule.tiles, STATE_TILE)),
            Self::Tiled { tiles } => Some(ExchangeLayout::new(tiles, tiled_exchange::TILE)),
            Self::Resident => None,
        }
    }
}

/// The pinned projection and its kernel
#[derive(Debug)]
struct InputProjection {
    function: CudaFunction,
    projection: LstmProjection,
}

impl InputProjection {
    /// Loads the pinned projection's entry; tensor projections need the sm80 tier and
    /// the opt-in shared memory of their tile
    fn new(
        device: &DeviceAttributes,
        kernels: &LoadedKernels,
        projection: LstmProjection,
    ) -> Result<Self, PlanError> {
        let name = match projection {
            LstmProjection::Small => SPK_LSTM_PROJECTION_SMALL,
            LstmProjection::Large => SPK_LSTM_PROJECTION,
            LstmProjection::Tensor => SPK_LSTM_PROJECTION_TF32,
        };
        let function = kernels.function(name)?;
        if projection != LstmProjection::Tensor {
            return Ok(Self {
                function,
                projection,
            });
        }

        // the sm75 entry of the same name is an FP32 fallback with another launch shape
        if kernels.tier() < PtxTier::Sm80 {
            return Err(PlanError::DeviceUnsupported {
                reason: format!(
                    "tensor projections need the sm80 tier, but {} is loaded",
                    kernels.tier()
                ),
            });
        }
        if device.shared_optin_bytes() < TENSOR_SHARED_BYTES {
            return Err(PlanError::DeviceUnsupported {
                reason: format!(
                    "tensor projections need {TENSOR_SHARED_BYTES} shared bytes per block, \
                     but the GPU allows {}",
                    device.shared_optin_bytes()
                ),
            });
        }

        function.set_attribute(
            CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            TENSOR_SHARED_BYTES as i32,
        )?;
        Ok(Self {
            function,
            projection,
        })
    }

    fn layout(&self) -> ProjectionLayout {
        match self.projection {
            LstmProjection::Tensor => ProjectionLayout::Tensor,
            LstmProjection::Small | LstmProjection::Large => ProjectionLayout::Simt,
        }
    }

    fn enqueue(
        &self,
        stream: &CudaStream,
        input: &CudaView<'_, f32>,
        weights: &CudaView<'_, f32>,
        output: &mut CudaViewMut<'_, f32>,
        rows: usize,
        columns: usize,
    ) -> Result<(), CudaError> {
        // output tile: gate columns by rows
        let (tile_columns, tile_rows, shared_mem_bytes) = match self.projection {
            LstmProjection::Small => (64, 64, 0),
            LstmProjection::Large => (128, 128, 0),
            LstmProjection::Tensor => (256, 128, TENSOR_SHARED_BYTES),
        };
        let dimensions = [
            to_u32("lstmproj projection rows", rows)?,
            to_u32("lstmproj projection width", columns)?,
            to_u32("lstmproj projection padded width", padded_input(columns))?,
        ];
        let config = LaunchConfig {
            grid_dim: (
                GATE_COLUMNS as u32 / tile_columns,
                dimensions[0].div_ceil(tile_rows),
                1,
            ),
            block_dim: (256, 1, 1),
            shared_mem_bytes,
        };
        let mut builder = stream.launch_builder(&self.function);
        builder.arg(input).arg(weights).arg(output);
        for dimension in &dimensions {
            builder.arg(dimension);
        }
        // SAFETY: the packed weights, input and gate views hold this direction's
        // complete matrices. The pin fixes the entry, weight layout and launch together,
        // and the plan refused tensor tiles without the sm80 tier. Every global address,
        // including tails, is inside a checked allocation
        unsafe { builder.launch(config) }?;
        Ok(())
    }
}

/// Everything one direction's launches of one layer need besides the weights
struct DirectionLaunch<'a> {
    /// `[batch * steps, 512]`, this direction's input projection
    gates: CudaView<'a, f32>,
    /// 0 forward, 1 reverse
    direction: usize,
    /// Address of this direction's first column of the `[batch, steps, 256]` stack
    /// output, taken on the given stream
    output: u64,
    /// Address of this direction's exchange state
    state: u64,
    /// The layer's first step flag, `layer * steps`
    flag_base: u32,
}

fn direction_index(direction: Direction) -> usize {
    match direction {
        Direction::Forward => 0,
        Direction::Reverse => 1,
    }
}

/// Requires a buffer or dimension to hold exactly what the stack needs
fn check_len(context: &'static str, expected: usize, actual: usize) -> Result<(), CudaError> {
    if expected == actual {
        return Ok(());
    }

    Err(CudaError::BufferLength {
        context,
        expected,
        actual,
    })
}

fn to_u32(context: &'static str, value: usize) -> Result<u32, CudaError> {
    u32::try_from(value).map_err(|_| CudaError::DimensionOverflow { context, value })
}

impl super::DriverCandidate for Oxide {
    const AREA: KernelModule = KernelModule::LstmProj;
    fn driver_coverage(
        tier: PtxTier,
        _device: &DeviceAttributes,
        _fp16: super::Fp16Policy,
    ) -> Coverage {
        Self::coverage(tier)
    }
    fn speed_scope(
        _boundary: crate::inference::cuda::implementation::BoundaryId,
        batch: usize,
        _math: CudaMath,
        device: &DeviceAttributes,
        _tier: PtxTier,
    ) -> Option<crate::inference::cuda::implementation::SpeedScope> {
        let capability = device.capability();
        ([1, 32].contains(&batch)
            && [ComputeCapability::new(8, 9), ComputeCapability::new(12, 0)].contains(&capability))
        .then_some(
            crate::inference::cuda::implementation::SpeedScope::MeasuredCapability { capability },
        )
    }
    fn speed_summary(_math: CudaMath) -> &'static str {
        "do-lstm dev: library-free projection and recurrence faster on Ada 8.9 and Blackwell 12.0 at b1/b32 in both maths"
    }
    fn driver_pin(
        _boundary: crate::inference::cuda::implementation::BoundaryId,
        batch: usize,
        math: CudaMath,
        device: &DeviceAttributes,
        tier: PtxTier,
        _fp16: super::Fp16Policy,
    ) -> Result<super::ConfigPin, PlanError> {
        Ok(super::ConfigPin::Lstm(LstmPin::Projected(projection_rule(
            ProjectionTarget {
                capability: device.capability(),
                shared_optin_bytes: device.shared_optin_bytes(),
                tier,
            },
            batch,
            batch * 589,
            math,
        ))))
    }
}
