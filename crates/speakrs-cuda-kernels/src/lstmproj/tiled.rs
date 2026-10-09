//! The recurrence for batched windows: eight-window tiles of sixteen-unit blocks
//!
//! [`spk_lstm_recurrence_tiled`] computes exactly what [`super::spk_lstm_recurrence`]
//! computes, bit for bit, with a work split measured faster on full batches. A tile has
//! [`TILE_ROWS`] windows and [`GROUPS`] blocks; block `g` owns units `16g..16g + 16`
//! of every window in the tile, so each consumer polls eight producers instead of
//! sixteen
//!
//! # Product and reduction
//!
//! A half-warp owns one unit. Lane `s` holds the four gate rows of that unit over
//! hidden inputs `8s..8s + 8`, as in the wide kernel, and runs the same 16 eight-term
//! FMA chains for each of the tile's eight windows. The reduction adds lane pairs at
//! offsets 8, 4, 2 and 1 in the same order, but its first three levels scatter windows
//! instead of gates. Afterwards lane `s` holds all four gate sums of window `s / 2`, so
//! the cell update needs no gather and only lane pairs repeat it
//!
//! Register slot `w` of lane `s` holds window `w ^ (s / 2)`. With that permutation
//! every scatter level keeps value `i` and sends value `i + half`, on every lane, so no
//! level selects between halves
//!
//! # Gate inputs
//!
//! The gate inputs of step `s + 2` are loaded after step `s`'s hidden state has
//! arrived. Issued earlier, their DRAM misses delay the exchange loads behind them

use cuda_device::{SharedArray, kernel, launch_bounds, thread, warp};

use super::{
    GATE_COLUMNS, GATES, HIDDEN, K_LANES, LANE_K, OUTPUT_COLUMNS, Vec4, fma_rn_f32, publish,
    read_state, ready, sigmoid_f32, stream_gates, stream_output, swizzle, tanh_f32,
    tiled_exchange,
};

/// Hidden units owned by one block
const UNITS: usize = 16;
/// Blocks per batch tile; the host launches exactly this many along x
pub const GROUPS: usize = HIDDEN / UNITS;
/// Threads per block; the host launches exactly this many
pub const THREADS: usize = UNITS * K_LANES;
/// Windows per batch tile
pub const TILE_ROWS: usize = tiled_exchange::ROWS;
/// Hidden words of one tile
const WORDS: usize = TILE_ROWS * HIDDEN;
/// Hidden words each thread polls
const PER_THREAD: usize = WORDS / THREADS;
/// Gate sums of one lane before the reduction: eight windows of four gates
const VALUES: usize = TILE_ROWS * GATES;

const _: () = assert!(PER_THREAD * THREADS == WORDS);
const _: () = assert!(GROUPS * UNITS == HIDDEN);
const _: () = assert!(UNITS == tiled_exchange::LINE);
// the straight-line zero fill writes two vectors per thread
const _: () = assert!(2 * TILE_ROWS * HIDDEN == THREADS * 8);

/// One cooperative launch: every step of one direction of one LSTM layer for
/// eight-window tiles
///
/// Arguments match [`super::spk_lstm_recurrence`] without `tile_rows`, which is
/// [`TILE_ROWS`]. `state` holds `tiled_exchange::TILE` zeroed or older-flagged words per
/// batch tile, private to this direction. Launch `(GROUPS, tiles, 1)` blocks of
/// `THREADS` threads cooperatively, with `first_tile + tiles <= ceil(batch / 8)`
#[kernel]
#[launch_bounds(256, 2)]
#[allow(clippy::too_many_arguments)]
pub fn spk_lstm_recurrence_tiled(
    gates_x: &[f32],
    weights: &[f32],
    bias: &[f32],
    output: *mut f32,
    state: *mut u64,
    batch: u32,
    steps: u32,
    direction: u32,
    first_tile: u32,
    flag_base: u32,
) {
    static mut HIDDEN_TILES: SharedArray<f32, { 2 * TILE_ROWS * HIDDEN }, 16> =
        SharedArray::UNINIT;
    // SAFETY: a raw pointer to this block's shared array; every access below is
    // separated from conflicting ones by a block barrier
    let hidden_tiles = unsafe { SharedArray::as_raw_mut_ptr(&raw mut HIDDEN_TILES) };

    let tid = thread::threadIdx_x() as usize;
    let group = thread::blockIdx_x() as usize;
    let tile = (first_tile + thread::blockIdx_y()) as usize;
    let batch = batch as usize;
    let steps = steps as usize;
    let reverse = direction != 0;
    let row0 = tile * TILE_ROWS;
    let rows = (batch - row0).min(TILE_ROWS);
    // SAFETY: the host passes tiled_exchange::TILE words for every tile of this direction
    let tile_state = unsafe { state.add(tile * tiled_exchange::TILE) };

    let slice = tid % K_LANES;
    let unit = group * UNITS + tid / K_LANES;
    let mut w = [[0.0f32; LANE_K]; GATES];
    unroll!(G in [0, 1, 2, 3] {
        let row = unit * GATES + G;
        unroll!(K in [0, 1, 2, 3, 4, 5, 6, 7] {
            w[G][K] = weights[row * HIDDEN + slice * LANE_K + K];
        });
    });

    // after the reduction this lane holds the four gates of tile row `cell_row`
    let cell_row = slice / 2;
    let owner = slice % 2 == 0;
    let active = cell_row < rows;
    let window = row0 + cell_row;
    let gate_column = unit * GATES;
    let gate_bias = [
        bias[gate_column],
        bias[gate_column + 1],
        bias[gate_column + 2],
        bias[gate_column + 3],
    ];
    let gates_at = |s: usize| -> [f32; GATES] {
        if !active || s >= steps {
            return [0.0; GATES];
        }
        let t = if reverse { steps - 1 - s } else { s };
        // gate columns and row strides are multiples of four floats; the active window
        // and valid unit give a full aligned vector
        // SAFETY: the four packed gates are inside the active projection view
        unsafe { stream_gates(gates_x.as_ptr().add((window * steps + t) * GATE_COLUMNS + gate_column)) }
    };

    // rows past the tile's windows stay zero, so the product needs no row checks.
    // Straight-line stores, so every shared word is written on every path before any read
    let zero = Vec4([0.0; 4]);
    unroll!(J in [0, 1] {
        // SAFETY: 256 threads x two vectors cover both hidden tiles
        unsafe { *(hidden_tiles.add((J * THREADS + tid) * 4) as *mut Vec4) = zero };
    });
    thread::sync_threads();

    let mut current_gates = gates_at(0);
    let mut next_gates = gates_at(1);
    let mut cell = 0.0f32;
    let mut s = 0;
    while s < steps {
        let t = if reverse { steps - 1 - s } else { s };
        let x_gates = current_gates;
        current_gates = next_gates;

        let mut recurrent = [0.0f32; GATES];
        if s > 0 {
            let parity = (s - 1) % 2;
            // SAFETY: the slot is inside this tile's state
            let slot = unsafe { tile_state.add(parity * tiled_exchange::SLOT) };
            // SAFETY: the parity's tile is inside the shared array
            let hidden_tile = unsafe { hidden_tiles.add(parity * WORDS) };
            load_hidden(slot, hidden_tile, rows, flag_base + s as u32);
            thread::sync_threads();
            // issued after the exchange loads returned, so its DRAM miss cannot delay them
            next_gates = gates_at(s + 2);
            recurrent = recurrent_product(hidden_tile, &w, slice);
        } else {
            next_gates = gates_at(2);
        }

        if active {
            let mut pre = [0.0f32; GATES];
            unroll!(G in [0, 1, 2, 3] {
                pre[G] = (x_gates[G] + recurrent[G]) + gate_bias[G];
            });

            let input_gate = sigmoid_f32(pre[0]);
            let forget_gate = sigmoid_f32(pre[1]);
            let candidate = tanh_f32(pre[2]);
            let output_gate = sigmoid_f32(pre[3]);
            cell = forget_gate * cell + input_gate * candidate;
            let hidden = output_gate * tanh_f32(cell);

            if owner {
                // the last step has no reader
                if s + 1 < steps {
                    let word = (s % 2) * tiled_exchange::SLOT + tiled_exchange::word(cell_row, unit);
                    // SAFETY: the word is inside this tile's state and only this thread writes it
                    unsafe { publish(tile_state.add(word), hidden, flag_base + s as u32 + 1) };
                }
                let index = (window * steps + t) * OUTPUT_COLUMNS + unit;
                // SAFETY: `window < batch` and `t < steps`, so the index is inside this
                // direction's half of the `[batch, steps, 256]` output, and only this
                // thread writes it
                unsafe { stream_output(output.add(index), hidden) };
            }
        }
        s += 1;
    }
}

/// Copies the previous step's hidden vectors of the tile's live rows into shared
/// memory, waiting until every word carries `flag`
///
/// Thread `tid` polls words `tid + 256j`, which are row `2j + tid / 128`, unit
/// `tid % 128`
#[inline(always)]
fn load_hidden(slot: *const u64, hidden_tile: *mut f32, rows: usize, flag: u32) {
    let tid = thread::threadIdx_x() as usize;
    let unit = tid % HIDDEN;
    let mut words = [0u64; PER_THREAD];
    unroll!(J in [0, 1, 2, 3] {
        let row = J * (THREADS / HIDDEN) + tid / HIDDEN;
        if row < rows {
            // SAFETY: the word is inside the slot
            words[J] = unsafe { read_state(slot.add(tiled_exchange::word(row, unit))) };
        }
    });

    loop {
        let mut waiting = false;
        unroll!(J in [0, 1, 2, 3] {
            let row = J * (THREADS / HIDDEN) + tid / HIDDEN;
            if row < rows && !ready(words[J], flag) {
                // SAFETY: as above; step s+2 cannot overwrite the word before every block
                // read step s
                words[J] = unsafe { read_state(slot.add(tiled_exchange::word(row, unit))) };
                waiting = true;
            }
        });
        if !waiting {
            break;
        }
    }

    unroll!(J in [0, 1, 2, 3] {
        let row = J * (THREADS / HIDDEN) + tid / HIDDEN;
        if row < rows {
            // SAFETY: `row < TILE_ROWS` and the swizzle permutes within the row
            unsafe { *hidden_tile.add(row * HIDDEN + swizzle(unit)) = f32::from_bits(words[J] as u32) };
        }
    });
}

/// Adds lane pairs `$offset` apart, keeping values `0..$half` and sending the rest
macro_rules! scatter {
    ($values:ident, $offset:literal, $half:literal, [$($i:literal),*]) => {
        $({
            $values[$i] += warp::shuffle_xor_f32($values[$i + $half], $offset);
        })*
    };
}

/// The four recurrent gate sums of tile row `slice / 2` for this half-warp's unit
///
/// Each lane adds its eight inputs in order with FMAs; the reduction then adds the 16
/// lanes of the half-warp pairwise at offsets 8, 4, 2 and 1, the order the wide kernel
/// uses
#[inline(always)]
fn recurrent_product(
    hidden_tile: *const f32,
    w: &[[f32; LANE_K]; GATES],
    slice: usize,
) -> [f32; GATES] {
    let permutation = slice / 2;
    // value `slot * 4 + gate` for tile row `slot ^ permutation`
    let mut values = [0.0f32; VALUES];
    unroll!(SLOT in [0, 1, 2, 3, 4, 5, 6, 7] {
        let row = (SLOT ^ permutation) * HIDDEN;
        unroll!(Q in [0, 1] {
            let offset = row + swizzle((2 * slice + Q) * 4);
            // SAFETY: the row is below TILE_ROWS and the vector inside it
            let h = unsafe { *(hidden_tile.add(offset) as *const Vec4) }.0;
            unroll!(G in [0, 1, 2, 3] {
                let mut a = if Q == 0 {
                    h[0] * w[G][0]
                } else {
                    fma_rn_f32(h[0], w[G][4], values[SLOT * GATES + G])
                };
                a = fma_rn_f32(h[1], w[G][4 * Q + 1], a);
                a = fma_rn_f32(h[2], w[G][4 * Q + 2], a);
                a = fma_rn_f32(h[3], w[G][4 * Q + 3], a);
                values[SLOT * GATES + G] = a;
            });
        });
    });

    // the slot permutation pairs value i of each lane with value i + half of its partner,
    // both of the same window
    scatter!(values, 8, 16, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
    scatter!(values, 4, 8, [0, 1, 2, 3, 4, 5, 6, 7]);
    scatter!(values, 2, 4, [0, 1, 2, 3]);
    // the last level keeps all four gates on both lanes of the pair
    unroll!(G in [0, 1, 2, 3] {
        values[G] += warp::shuffle_xor_f32(values[G], 1);
    });
    [values[0], values[1], values[2], values[3]]
}
