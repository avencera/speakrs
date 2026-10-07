//! The four bidirectional LSTM layers on cuBLAS input projections and the cuda-oxide
//! persistent recurrence kernel `spk_lstm_recurrence`
//!
//! Per layer, the locked projection helper computes `x · Wᵀ` for every step of each
//! direction, with the gate rows of `W` reordered so each hidden unit's four gates
//! are adjacent. Then one cooperative launch per direction runs that direction's
//! whole recurrence with the recurrent weights held on chip; see the kernel crate's
//! `lstm` area for the work split and the hidden-state exchange. The reverse
//! direction runs on a side stream, so both directions share the GPU instead of
//! running one after the other; the next layer waits for both
//!
//! Every layer writes into the stack output. That is safe because a layer's
//! projections read the previous layer's complete output before its recurrence
//! overwrites it

pub(super) mod layout;

use cudarc::driver::{
    CudaFunction, CudaSlice, CudaStream, CudaView, CudaViewMut, DevicePtrMut, LaunchConfig,
    PushKernelArg,
};

use self::layout::{
    GATE_COLUMNS, GROUPS, HIDDEN, KERNEL, STATE_TILE, Schedule, pack_bias, pack_directions,
};
use super::{
    Batches, Coverage, CoverageEntry, Direction, FiniteContract, InfinityContract, LstmCandidate,
    LstmPhases, LstmPin, LstmSpec, Maths, NanContract, Op, PlanError, ProjectionGemm, Scratch,
    SideStream, SignedZeroContract, SpecialValues,
};
use crate::inference::cuda::{CudaError, CudaMath, CudaRuntime, KernelModule};

const SPK_LSTM_CLEAR: &str = "spk_lstm_clear";

/// Kernel entries loaded by this host plan
pub(crate) const REQUIRED_KERNELS: [&str; 2] = [KERNEL, SPK_LSTM_CLEAR];

/// Threads per block, `THREADS` in the kernel crate's `lstm` area
const THREADS: u32 = 128;
/// LSTM input width of the first layer
const FEATURES: usize = 60;
const DIRECTIONS: [Direction; 2] = [Direction::Forward, Direction::Reverse];

/// One layer's weights in the packed gate order, both directions stacked
#[derive(Debug)]
struct PackedLayer {
    /// Input width: 60 for the first layer, 256 after it
    input: usize,
    /// `[2, 512, input]`
    w: CudaSlice<f32>,
    /// `[2, 512, 128]`
    r: CudaSlice<f32>,
    /// `[2, 512]`, `Wb + Rb`
    bias: CudaSlice<f32>,
}

/// The oxide stack for one batch size: packed weights, the recurrence kernel, the
/// tile schedule, and the buffers and side stream its passes reuse
#[derive(Debug)]
pub(crate) struct Oxide {
    recurrence: CudaFunction,
    clear: CudaFunction,
    layers: [PackedLayer; 4],
    batch: usize,
    steps: usize,
    schedule: Schedule,
    /// `[2, batch * steps, 512]`: each direction's input projection
    gates: Scratch<f32>,
    /// `[2, tiles, STATE_TILE]` flagged hidden-state words the blocks of one
    /// direction exchange; zeroed at the start of every pass, and each layer flags its
    /// words above every earlier layer's
    state: Scratch<u64>,
    /// Runs the reverse direction when the schedule makes the directions concurrent
    side: SideStream,
}

impl LstmCandidate for Oxide {
    // FP32 only: the recurrence computes in FP32 in both modes, so in TF32 its logits
    // leave the TF32 Library's 1-ulp noise band even where they are more accurate, and
    // segmentation runs FP32 in production. Measured faster than cuDNN Standard and
    // PersistStaticSmallH at every harness batch
    const COVERAGE: Coverage = Coverage(&[CoverageEntry {
        layers: &["lstm.stack"],
        batches: Batches::All,
        maths: Maths::Only(&[CudaMath::Fp32]),
    }]);

    // the gates saturate, but `exp_f32` clamps its argument before the exponent, so how
    // NaN and infinite pre-activations surface has not been established
    const SPECIAL_VALUES: SpecialValues = SpecialValues {
        finite: FiniteContract::BoundedActivation { headroom: 2 },
        nan: NanContract::Unspecified,
        infinity: InfinityContract::Unspecified,
        signed_zero: SignedZeroContract::Unspecified,
    };

    fn implemented_pin(_spec: &LstmSpec<'_>) -> Result<LstmPin, PlanError> {
        Ok(LstmPin::LegacyCooperative)
    }

    fn plan(runtime: &CudaRuntime, spec: LstmSpec<'_>, pin: LstmPin) -> Result<Self, PlanError> {
        let LstmPin::LegacyCooperative = pin;
        let kernels = runtime.load_kernels(KernelModule::Lstm)?;
        let recurrence = kernels.function(KERNEL)?;
        let clear = kernels.function(SPK_LSTM_CLEAR)?;
        let capacity = runtime.cooperative_capacity(&recurrence, THREADS, 0)?;
        let concurrent_capacity =
            runtime.concurrent_cooperative_capacity(&recurrence, THREADS, 0)?;
        let schedule = Schedule::new(spec.batch, capacity, concurrent_capacity)?;

        let stream = runtime.stream();
        let pack = |index: usize| -> Result<PackedLayer, CudaError> {
            let layer = spec.layers[index];
            let input = if index == 0 { FEATURES } else { 2 * HIDDEN };
            check_len("oxide LSTM input width", input, layer.input)?;
            check_len("oxide LSTM W", 2 * GATE_COLUMNS * input, layer.w.len())?;
            check_len("oxide LSTM R", 2 * GATE_COLUMNS * HIDDEN, layer.r.len())?;
            check_len("oxide LSTM B", 4 * GATE_COLUMNS, layer.b.len())?;
            Ok(PackedLayer {
                input,
                w: stream.clone_htod(&pack_directions(layer.w, input))?,
                r: stream.clone_htod(&pack_directions(layer.r, HIDDEN))?,
                bias: stream.clone_htod(&pack_bias(layer.b))?,
            })
        };

        Ok(Self {
            recurrence,
            clear,
            layers: [pack(0)?, pack(1)?, pack(2)?, pack(3)?],
            batch: spec.batch,
            steps: spec.frames,
            schedule,
            gates: Scratch::zeros(runtime, 2 * spec.batch * spec.frames * GATE_COLUMNS)?,
            state: Scratch::zeros(runtime, 2 * schedule.tiles * STATE_TILE)?,
            side: SideStream::new(runtime)?,
        })
    }

    fn enqueue(
        &self,
        input: &CudaView<'_, f32>,
        output: &mut CudaViewMut<'_, f32>,
        phases: &LstmPhases<'_>,
        stream: &CudaStream,
    ) -> Result<(), CudaError> {
        let rows = self.batch * self.steps;
        check_len("oxide LSTM input", rows * FEATURES, input.len())?;
        check_len("oxide LSTM output", rows * 2 * HIDDEN, output.len())?;

        phases.op(Op::Prologue, || self.clear_state(stream))?;
        for (index, layer) in self.layers.iter().enumerate() {
            for direction in DIRECTIONS {
                phases.input_proj(index, direction, |projection| {
                    let d = direction_index(direction);
                    let weights = GATE_COLUMNS * layer.input;
                    let w = layer.w.slice(d * weights..(d + 1) * weights);
                    let mut gates = self.gates.get();
                    let mut gates =
                        gates.slice_mut(d * rows * GATE_COLUMNS..(d + 1) * rows * GATE_COLUMNS);
                    let gemm = ProjectionGemm {
                        n: GATE_COLUMNS,
                        weight_transposed: true,
                        beta: 0.0,
                    };
                    if index == 0 {
                        projection.project(index, direction, gemm, input, &w, &mut gates)
                    } else {
                        projection.project(
                            index,
                            direction,
                            gemm,
                            &output.as_view(),
                            &w,
                            &mut gates,
                        )
                    }
                })?;
            }
            self.recur(index, layer, output, phases, stream)?;
        }

        Ok(())
    }
}

impl Oxide {
    /// Zeroes the exchange state, so no word from an earlier pass carries a flag this
    /// pass expects
    fn clear_state(&self, stream: &CudaStream) -> Result<(), CudaError> {
        let mut state = self.state.get();
        let len = state.len();
        let len_arg = len as u64;
        let mut builder = stream.launch_builder(&self.clear);
        builder.arg(&mut *state).arg(&len_arg);
        // SAFETY: `spk_lstm_clear` takes one slice, pointer and length, and writes only
        // inside it
        unsafe {
            builder.launch(LaunchConfig::for_num_elems(to_u32(
                "oxide LSTM clear",
                len,
            )?))
        }?;
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
        let context = "oxide LSTM recurrence";
        let rows = self.batch * self.steps;
        let mut state = self.state.get();
        let gates = self.gates.get();
        // cudarc orders accesses per allocation, so taking these pointers on the side
        // stream would make it wait for the forward kernel's write and serialize the
        // directions. Both are taken on the given stream, and their write events are
        // recorded there after the merge
        let (output, _output_record) = output.device_ptr_mut(stream);
        let (state, _state_record) = state.device_ptr_mut(stream);
        let concurrent = self.schedule.concurrent;
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
                state: state + (d * self.schedule.tiles * STATE_TILE * size_of::<u64>()) as u64,
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
        let context = "oxide LSTM recurrence";
        let DirectionLaunch {
            gates,
            direction,
            output,
            state,
            flag_base,
        } = launch;
        let recurrent = GATE_COLUMNS * HIDDEN;
        let r = layer
            .r
            .slice(direction * recurrent..(direction + 1) * recurrent);
        let bias = layer
            .bias
            .slice(direction * GATE_COLUMNS..(direction + 1) * GATE_COLUMNS);
        let lens = [gates.len(), r.len(), bias.len()].map(|len| len as u64);
        let batch = to_u32(context, self.batch)?;
        let steps = to_u32(context, self.steps)?;
        let direction = to_u32(context, direction)?;
        let tile_rows = to_u32(context, self.schedule.tile_rows)?;

        for (first, tiles) in self.schedule.launches() {
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
            // the output holds `[batch, steps, 256]` and the state STATE_TILE words
            // per tile of this direction, both checked or sized by the plan. Flags
            // grow by layer, so no word from an earlier layer of this pass, or from
            // before the pass's zeroing, can match. The launch is cooperative and the
            // schedule keeps every block that may run at the same time within the
            // GPU's resident capacity, which the kernel's cross-block waits need
            unsafe { builder.launch_cooperative(config) }?;
        }

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
