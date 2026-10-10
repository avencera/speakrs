//! The recurrence with one window of one direction per block and its recurrent
//! weights held on one SM, so no block waits for another
//!
//! [`spk_lstm_recurrence_resident`] computes exactly what
//! [`super::spk_lstm_recurrence`] computes, bit for bit. The other kernels split a
//! direction's 256 KiB recurrent matrix over eight or sixteen blocks, which publish
//! their hidden units through L2 at every step. On an A100 that exchange, not the
//! arithmetic, set the step time: the tiled kernel took 1.9 µs per step for a single
//! window, with one eighth of the products per SM. This kernel keeps the whole matrix
//! on one SM, half in registers and half in shared memory, and exchanges the hidden
//! vector through shared memory behind one block barrier per step. An A100 SM has the
//! 256 KiB register file and up to 164 KiB of shared memory this needs, which GPUs
//! with 100 KiB of shared memory per SM do not
//!
//! # Product and reduction
//!
//! Half-warp `p` owns units `8p..8p + 8`, the packed gate rows `32p..32p + 32`. Lane
//! `s` runs the 32 eight-term FMA chains of those rows over hidden inputs
//! `8s..8s + 8`, the same chains the other kernels run. Register slot `i` of lane `s`
//! holds row `i ^ 2s`, so at each reduction level every lane keeps its lower slots and
//! sends its upper ones: the slots a lane and its partner at offset 8 add hold the
//! same row, and so on down to offset 1. The levels add lanes 8, 4, 2 and 1 apart, the
//! order of the other kernels. Afterwards lane `s` holds rows `2s` and `2s + 1`, two
//! gates of unit `8p + s / 2`, and one shuffle with its pair completes the four gates
//!
//! Slots 0 to 15 keep their weights in registers. Slots 16 to 31 read theirs from
//! shared memory each step, in lane-major vectors that one warp reads as 512
//! contiguous bytes
//!
//! # Gate inputs
//!
//! The next step's gate inputs are loaded before this step's barrier, so their DRAM
//! latency overlaps the product

use cuda_device::{DynamicSharedArray, SharedArray, kernel, launch_bounds, thread, warp};

use super::{
    GATE_COLUMNS, GATES, HIDDEN, K_LANES, LANE_K, OUTPUT_COLUMNS, Vec4, fma_rn_f32, sigmoid_f32,
    stream_gates, stream_output, swizzle, tanh_f32,
};

/// Threads per block; the host launches exactly this many
pub const THREADS: usize = 256;
/// Packed gate rows owned by one half-warp
const ROWS: usize = 32;
/// Row slots whose weights stay in registers; the rest stream from shared memory
const REGISTER_SLOTS: usize = 16;
/// Warps per block
const WARPS: usize = THREADS / 32;
/// Dynamic shared memory the host provides: the shared slots' weights of every lane
pub const SHARED_BYTES: usize = WARPS * (ROWS - REGISTER_SLOTS) * 2 * 32 * size_of::<Vec4>();

// the shared slots hold half of the 256 KiB matrix; the host requests exactly this
const _: () = assert!(SHARED_BYTES == GATE_COLUMNS * HIDDEN * size_of::<f32>() / 2);
const _: () = assert!(2 * WARPS * ROWS == GATE_COLUMNS);
const _: () = assert!(THREADS == 2 * HIDDEN);
const _: () = assert!(ROWS == 2 * K_LANES);

/// Shared-memory vector of half `q` of slot `16 + i` for `lane` of `warp`
#[inline(always)]
const fn shared_vector(warp: usize, slot: usize, half: usize, lane: usize) -> usize {
    ((warp * (ROWS - REGISTER_SLOTS) + slot) * 2 + half) * 32 + lane
}

/// One launch: every step of both directions of one LSTM layer, one window and
/// direction per block
///
/// - `gates_x`: `[2, batch * steps, 512]`, each direction's input projection with
///   packed columns `unit * 4 + gate`, gates `[i, f, c, o]`
/// - `weights`: `[2, 512, 128]`, each direction's recurrent matrix with the same
///   packed rows
/// - `bias`: `[2, 512]`, `Wb + Rb` with the same packed columns
/// - `forward` and `reverse`: each direction's first column of the `[batch, steps,
///   256]` layer output, so the host adds 128 for the reverse one; hidden unit `u` of
///   step `t` goes to column `u` from there. Forming that offset in a kernel made
///   ptxas (CUDA 13.0, sm_120) drop a pointer's high 32 bits
///
/// The initial hidden and cell states are zero. Launch `(batch, 2, 1)` blocks of
/// [`THREADS`] threads with [`SHARED_BYTES`] of dynamic shared memory
#[kernel]
#[launch_bounds(256, 1)]
pub fn spk_lstm_recurrence_resident(
    gates_x: &[f32],
    weights: &[f32],
    bias: &[f32],
    forward: *mut f32,
    reverse: *mut f32,
    batch: u32,
    steps: u32,
) {
    static mut HIDDEN_VECTORS: SharedArray<f32, { 2 * HIDDEN }, 16> = SharedArray::UNINIT;
    // SAFETY: a raw pointer to this block's shared array; every access below is
    // separated from conflicting ones by a block barrier
    let hidden_vectors = unsafe { SharedArray::as_raw_mut_ptr(&raw mut HIDDEN_VECTORS) };
    let shared_weights = DynamicSharedArray::<Vec4, 16>::get();

    let tid = thread::threadIdx_x() as usize;
    let window = thread::blockIdx_x() as usize;
    let direction = thread::blockIdx_y() as usize;
    let batch = batch as usize;
    let steps = steps as usize;
    let output = if direction == 0 { forward } else { reverse };
    let reverse = direction != 0;
    let warp_index = tid / 32;
    let lane = tid % 32;
    let slice = lane % K_LANES;
    let first_row = (2 * warp_index + lane / K_LANES) * ROWS;
    let recurrent = direction * GATE_COLUMNS * HIDDEN;

    let mut w = [[0.0f32; LANE_K]; REGISTER_SLOTS];
    unroll!(I in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] {
        let row = first_row + (I ^ (2 * slice));
        unroll!(K in [0, 1, 2, 3, 4, 5, 6, 7] {
            w[I][K] = weights[recurrent + row * HIDDEN + slice * LANE_K + K];
        });
    });
    unroll!(I in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] {
        let row = first_row + ((REGISTER_SLOTS + I) ^ (2 * slice));
        unroll!(Q in [0, 1] {
            let base = recurrent + row * HIDDEN + slice * LANE_K + 4 * Q;
            let vector = Vec4([weights[base], weights[base + 1], weights[base + 2], weights[base + 3]]);
            // SAFETY: the host provides SHARED_BYTES, which cover every warp, slot,
            // half and lane
            unsafe { *shared_weights.add(shared_vector(warp_index, I, Q, lane)) = vector };
        });
    });
    // 256 threads clear both parity vectors in one straight-line store
    // SAFETY: tid < 2 * HIDDEN
    unsafe { *hidden_vectors.add(tid) = 0.0 };

    // after the reduction this lane holds gates `2 * (slice % 2)` and the next one of
    // `unit`; the even lane of each pair publishes the unit
    let unit = first_row / GATES + slice / 2;
    let even = slice % 2 == 0;
    let gate_column = direction * GATE_COLUMNS + unit * GATES;
    let gate_bias = [
        bias[gate_column],
        bias[gate_column + 1],
        bias[gate_column + 2],
        bias[gate_column + 3],
    ];
    let projection = direction * batch * steps * GATE_COLUMNS;
    let gates_at = |s: usize| -> [f32; GATES] {
        if s >= steps {
            return [0.0; GATES];
        }
        let t = if reverse { steps - 1 - s } else { s };
        // gate columns and row strides are multiples of four floats, so the vector is
        // aligned and inside the direction's projection
        // SAFETY: `window < batch`, `t < steps` and the unit's four gates are in range
        unsafe {
            stream_gates(
                gates_x
                    .as_ptr()
                    .add(projection + (window * steps + t) * GATE_COLUMNS + unit * GATES),
            )
        }
    };
    thread::sync_threads();

    let mut next_gates = gates_at(0);
    let mut cell = 0.0f32;
    let mut s = 0;
    while s < steps {
        let t = if reverse { steps - 1 - s } else { s };
        let x_gates = next_gates;
        next_gates = gates_at(s + 1);

        let mut recurrent_sums = [0.0f32; GATES];
        if s > 0 {
            // every lane published step s - 1 before this barrier, and no lane writes
            // this parity again before passing the next one
            thread::sync_threads();
            // SAFETY: the parity's vector is inside the shared array
            let hidden = unsafe { hidden_vectors.add(((s - 1) % 2) * HIDDEN) };
            recurrent_sums = recurrent_product(hidden, &w, shared_weights, warp_index, lane, even);
        }

        let mut pre = [0.0f32; GATES];
        unroll!(G in [0, 1, 2, 3] {
            pre[G] = (x_gates[G] + recurrent_sums[G]) + gate_bias[G];
        });

        let input_gate = sigmoid_f32(pre[0]);
        let forget_gate = sigmoid_f32(pre[1]);
        let candidate = tanh_f32(pre[2]);
        let output_gate = sigmoid_f32(pre[3]);
        cell = forget_gate * cell + input_gate * candidate;
        let hidden = output_gate * tanh_f32(cell);

        if even {
            // SAFETY: the swizzle permutes within the parity's vector, and only this
            // lane writes the unit
            unsafe { *hidden_vectors.add((s % 2) * HIDDEN + swizzle(unit)) = hidden };
            let index = (window * steps + t) * OUTPUT_COLUMNS + unit;
            // SAFETY: `window < batch` and `t < steps`, so the index is inside this
            // direction's half of the `[batch, steps, 256]` output, and only this
            // thread writes it
            unsafe { stream_output(output.add(index), hidden) };
        }
        s += 1;
    }
}

/// The four recurrent gate sums of this lane's unit
///
/// Each lane adds its eight inputs in order with FMAs; the reduction then adds the 16
/// lanes of the half-warp pairwise at offsets 8, 4, 2 and 1, the order the other
/// kernels use
#[inline(always)]
fn recurrent_product(
    hidden: *const f32,
    w: &[[f32; LANE_K]; REGISTER_SLOTS],
    shared_weights: *const Vec4,
    warp_index: usize,
    lane: usize,
    even: bool,
) -> [f32; GATES] {
    let slice = lane % K_LANES;
    // SAFETY: both vectors are inside the parity's swizzled hidden vector
    let h = [
        unsafe { *(hidden.add(swizzle(slice * LANE_K)) as *const Vec4) }.0,
        unsafe { *(hidden.add(swizzle(slice * LANE_K + 4)) as *const Vec4) }.0,
    ];

    // value `slot` holds row `slot ^ 2 * slice` of the half-warp
    let mut values = [0.0f32; ROWS];
    unroll!(I in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] {
        let mut a = h[0][0] * w[I][0];
        a = fma_rn_f32(h[0][1], w[I][1], a);
        a = fma_rn_f32(h[0][2], w[I][2], a);
        a = fma_rn_f32(h[0][3], w[I][3], a);
        a = fma_rn_f32(h[1][0], w[I][4], a);
        a = fma_rn_f32(h[1][1], w[I][5], a);
        a = fma_rn_f32(h[1][2], w[I][6], a);
        a = fma_rn_f32(h[1][3], w[I][7], a);
        values[I] = a;
    });
    unroll!(I in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] {
        // SAFETY: inside the dynamic shared weights, written before the first barrier
        let low = unsafe { *shared_weights.add(shared_vector(warp_index, I, 0, lane)) }.0;
        // SAFETY: as above
        let high = unsafe { *shared_weights.add(shared_vector(warp_index, I, 1, lane)) }.0;
        let mut a = h[0][0] * low[0];
        a = fma_rn_f32(h[0][1], low[1], a);
        a = fma_rn_f32(h[0][2], low[2], a);
        a = fma_rn_f32(h[0][3], low[3], a);
        a = fma_rn_f32(h[1][0], high[0], a);
        a = fma_rn_f32(h[1][1], high[1], a);
        a = fma_rn_f32(h[1][2], high[2], a);
        a = fma_rn_f32(h[1][3], high[3], a);
        values[REGISTER_SLOTS + I] = a;
    });

    // the slot permutation pairs value i of each lane with value i + half of its
    // partner, both of the same row
    scatter!(
        values,
        8,
        16,
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    );
    scatter!(values, 4, 8, [0, 1, 2, 3, 4, 5, 6, 7]);
    scatter!(values, 2, 4, [0, 1, 2, 3]);
    scatter!(values, 1, 2, [0, 1]);

    // the even lane holds gates 0 and 1, the odd lane gates 2 and 3
    let partner = [
        warp::shuffle_xor_f32(values[0], 1),
        warp::shuffle_xor_f32(values[1], 1),
    ];
    if even {
        [values[0], values[1], partner[0], partner[1]]
    } else {
        [partner[0], partner[1], values[0], values[1]]
    }
}
