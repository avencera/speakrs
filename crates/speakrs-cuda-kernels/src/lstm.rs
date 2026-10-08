//! Batch-parallel persistent recurrence for the segmentation BiLSTM stack
//!
//! cuBLAS computes the input projection `x · Wᵀ` of every step before these kernels
//! run. One cooperative launch of [`spk_lstm_recurrence`] then runs every step of one
//! direction of one layer: `h_{t-1} · Rᵀ`, the gates, the cell update and the hidden
//! output, so the recurrent weights are read from global memory once per launch
//! instead of once per step
//!
//! # Work split
//!
//! A launch has [`GROUPS`] blocks per batch tile of up to `tile_rows` windows; the
//! host picks 8, 16 or 32 rows. Block `(g, tile)` owns hidden units `4g..4g + 4` of
//! every window in its tile, so all four gates of a unit stay in one block and the
//! cell state never leaves registers. Windows are independent recurrences that only
//! share weights, so small tiles give each SM several blocks whose waits and
//! arithmetic overlap.
//!
//! The block's 16 gate rows of `R` (16 × 128 FP32) stay in registers for the whole
//! sequence. Warp `u` computes the four gates of the block's unit `u`; lane `l` holds
//! their weights for hidden inputs `8(l % 16)..8(l % 16) + 8`, and the two half-warps
//! take alternate rows. The 16 lanes of a row read different hidden values from
//! shared memory, through a swizzle that keeps the reads free of bank conflicts, and
//! each value serves four gates. A butterfly over the 16 lanes adds the partial
//! products.
//!
//! # Exchanging the hidden state
//!
//! Each step needs the complete previous hidden vector of its windows, which 32
//! blocks produce. A block publishes each hidden value as one 64-bit word, the value's
//! bits in the low half and a step flag in the high half, into a state buffer with
//! one slot per step parity. Each relaxed load has the same size and address as its
//! naturally aligned 64-bit store, so single-copy atomicity keeps the flag and value
//! together. A reader needs no fence,
//! counter or second round trip to L2. The slot of step `s` is rewritten at step
//! `s + 2` only after every block has read it, because no block can finish step
//! `s + 1` before all blocks published it, which they do after reading step `s`.
//!
//! Readers spin until every flag they need matches. The host uses cooperative
//! launches and checks the single-grid limit and the context's joint SM budget.
//! Pre-Hopper GPUs can start partly resident cooperative grids, so multiple pipelines
//! in one process or other spinning cooperative work on the same device can still
//! contend, as with persistent library RNN algorithms. Flags
//! are `flag_base + step + 1`; the host zeroes the buffer before each forward pass and
//! gives every layer its own flag range, so no stale word can match
//!
//! # Numerics
//!
//! Everything is FP32 with round-to-nearest; the compiler may fuse a multiply and
//! an add into one FMA, which stays deterministic. Each gate pre-activation is
//! `(x·Wᵀ + h·Rᵀ) + (Wb + Rb)`, in that order, as ONNX Runtime's CPU LSTM adds the
//! recurrent product into the input projection and the combined bias afterwards.
//! The recurrent dot product is 16 eight-term FMA chains, one per lane, added in a
//! fixed butterfly, so repeated runs are bitwise identical. `exp` follows CUDA's
//! accurate `expf` (about 2 ulp) and the reciprocals take one Newton step, so neither
//! uses a fast approximation on its own

use cuda_device::{DisjointSlice, SharedArray, kernel, launch_bounds, ptx_asm, thread, warp};

/// Hidden units per direction
const HIDDEN: usize = 128;
/// LSTM gates per hidden unit, packed `[i, f, c, o]`
const GATES: usize = 4;
/// Packed gate columns of one direction: `unit * GATES + gate`
const GATE_COLUMNS: usize = HIDDEN * GATES;
/// Hidden units owned by one block
const UNITS: usize = 4;
/// Gate columns owned by one block
const COLUMNS: usize = UNITS * GATES;
/// Blocks per batch tile; the host launches exactly this many along x
pub const GROUPS: usize = HIDDEN / UNITS;
/// Threads per block; the host launches exactly this many
pub const THREADS: usize = 128;
/// Lanes that split the hidden inputs of one row in the recurrent product
const K_LANES: usize = 16;
/// Hidden inputs per lane of the recurrent product
const LANE_K: usize = HIDDEN / K_LANES;
/// Most windows per batch tile; the host picks the tile height at run time
pub const TILE_ROWS: usize = 32;
/// Windows per pass of the recurrent product, which bounds its accumulators
const CHUNK: usize = 8;
/// Layer output row: forward then reverse hidden units
const OUTPUT_COLUMNS: usize = 2 * HIDDEN;
/// State words each thread has in flight while loading the hidden tile
const LOAD_BATCH: usize = 8;
/// State words of one tile and step parity: `[TILE_ROWS, HIDDEN]`
const STATE_SLOT: usize = TILE_ROWS * HIDDEN;
/// State words of one tile: two step parities; the host allocates this many per tile
pub const STATE_TILE: usize = 2 * STATE_SLOT;

// the kernel's straight-line zero-fill covers both shared arrays exactly
const _: () = assert!(8 * THREADS * 4 == TILE_ROWS * HIDDEN);
const _: () = assert!(THREADS * 4 == TILE_ROWS * COLUMNS);

/// Four floats with 16-byte alignment, so loads and stores use one vector access
#[derive(Clone, Copy)]
#[repr(C, align(16))]
struct Vec4([f32; 4]);

/// Repeats `$body` with `$i` bound to each listed constant
///
/// The loops below must unroll completely: an array indexed by a loop variable lives
/// in local memory, and a load whose result is stored before the next load issues
/// serializes the hidden-state reads. Constant indices keep every array in registers
macro_rules! unroll {
    ($i:ident in [$($n:literal),*] $body:block) => {
        $({
            const $i: usize = $n;
            $body
        })*
    };
}

/// `a * b + c` with one rounding
///
/// cuda-oxide's `fma_rn_f32` intrinsic requires sm_80, though `fma.rn.f32` exists on
/// every target
#[inline(always)]
fn fma_rn_f32(a: f32, b: f32, c: f32) -> f32 {
    let result: f32;
    // SAFETY: a pure register instruction
    unsafe {
        ptx_asm!(
            "fma.rn.f32 %0, %1, %2, %3;",
            out("=f") result,
            in("f") a,
            in("f") b,
            in("f") c,
            options(register_only),
        );
    }
    result
}

/// `2^x` from the special function unit, within 2 ulp
#[inline(always)]
fn ex2_approx(x: f32) -> f32 {
    let result: f32;
    // SAFETY: a pure register instruction
    unsafe {
        ptx_asm!(
            "ex2.approx.ftz.f32 %0, %1;",
            out("=f") result,
            in("f") x,
            options(register_only),
        );
    }
    result
}

/// `exp(x)` within about 2 ulp, for `x` in `[-87, 88]`
///
/// The construction of CUDA's accurate `expf` (not `__expf`): `x · log2(e) = n + r`
/// with `n` an integer and `|r| <= 0.5`, where two FMAs with a split `log2(e)` keep
/// `r` exact to FP32 precision, then `2^r` from the special function unit, scaled
/// exactly by `2^n`. `__expf` instead feeds the rounded product to `ex2`, which
/// loses accuracy as `|x|` grows
#[inline(always)]
fn exp_f32(x: f32) -> f32 {
    // keeps `2^n` a normal number
    let x = x.clamp(-87.0, 88.0);
    // round to nearest by adding and removing 1.5 * 2^23
    let shifted = fma_rn_f32(x, core::f32::consts::LOG2_E, 12_582_912.0);
    let n = shifted - 12_582_912.0;
    // `LOG2_E` is log2(e) rounded to FP32; 1.925963e-8 is the rest
    let r = fma_rn_f32(x, core::f32::consts::LOG2_E, -n);
    let r = fma_rn_f32(x, 1.925_963e-8, r);
    let scale = f32::from_bits(((n as i32 + 127) as u32) << 23);
    ex2_approx(r) * scale
}

/// `1 / y` within 1 ulp for finite `y >= 1`, without a branch
///
/// IEEE division lowers to a reciprocal with a slow-path call, and that branch
/// stops the compiler from overlapping the four gate activations. One Newton step
/// on the hardware estimate is enough in this range: `y` is never zero, infinite
/// or subnormal, because `exp_f32` clamps its argument
#[inline(always)]
fn recip_f32(y: f32) -> f32 {
    let estimate: f32;
    // SAFETY: a pure register instruction
    unsafe {
        ptx_asm!(
            "rcp.approx.ftz.f32 %0, %1;",
            out("=f") estimate,
            in("f") y,
            options(register_only),
        );
    }
    fma_rn_f32(estimate, fma_rn_f32(-y, estimate, 1.0), estimate)
}

/// `1 / (1 + exp(-x))`
#[inline(always)]
fn sigmoid_f32(x: f32) -> f32 {
    recip_f32(1.0 + exp_f32(-x))
}

/// `tanh(x)` within a few ulp: the Cephes `tanhf` polynomial below 0.625 and
/// `1 - 2 / (e^{2|x|} + 1)` above it
///
/// Both forms are computed and one selected, so the lanes of a warp never split
#[inline(always)]
fn tanh_f32(x: f32) -> f32 {
    let z = x.abs();
    let s = x * x;
    let mut p = -5.704_988_7e-3;
    p = fma_rn_f32(p, s, 2.063_908_9e-2);
    p = fma_rn_f32(p, s, -5.373_971_6e-2);
    p = fma_rn_f32(p, s, 1.333_144_2e-1);
    p = fma_rn_f32(p, s, -3.333_328_2e-1);
    let small = fma_rn_f32(p * s, x, x);

    // tanh rounds to exactly one in FP32 above about 9
    let e = exp_f32(2.0 * z.min(20.0));
    let large = (1.0 - 2.0 * recip_f32(e + 1.0)).copysign(x);
    if z < 0.625 { small } else { large }
}

/// Publishes one hidden value with its step flag
///
/// # Safety
///
/// `address` must point to a writable, 8-byte aligned global word
#[inline(always)]
unsafe fn publish(address: *mut u64, value: f32, flag: u32) {
    let word = (u64::from(flag) << 32) | u64::from(value.to_bits());
    // SAFETY: the caller guarantees a valid aligned global address
    unsafe {
        ptx_asm!(
            "st.relaxed.gpu.global.u64 [%0], %1;",
            in("l") address as u64,
            in("l") word,
        );
    }
}

/// Reads one state word with the same size and address as its publication
///
/// # Safety
///
/// `address` must point to a readable, 8-byte aligned global word
#[inline(always)]
unsafe fn read_state(address: *const u64) -> u64 {
    let word: u64;
    // SAFETY: the caller guarantees a valid aligned global address; matching the
    // store's size preserves single-copy atomicity of the value and flag
    unsafe {
        ptx_asm!(
            "ld.relaxed.gpu.global.u64 %0, [%1];",
            out("=l") word,
            in("l") address as u64,
        );
    }
    word
}

/// Whether the word carries `flag`
#[inline(always)]
fn ready(word: u64, flag: u32) -> bool {
    (word >> 32) as u32 == flag
}

/// Zeroes the hidden-state exchange words before a forward pass
///
/// Launch one thread per word with a 1-D grid
#[kernel]
pub fn spk_lstm_clear(mut state: DisjointSlice<u64>) {
    let idx = thread::index_1d();
    if let Some(word) = state.get_mut(idx) {
        *word = 0;
    }
}

/// One cooperative launch: every step of one direction of one LSTM layer
///
/// - `gates_x`: `[batch * steps, 512]`, this direction's input projection
///   `x · Wᵀ` with packed columns `unit * 4 + gate`, gates `[i, f, c, o]`
/// - `weights`: `[512, 128]`, this direction's recurrent matrix `R` with the same
///   packed rows
/// - `bias`: `[512]`, `Wb + Rb` with the same packed columns
/// - `output`: this direction's first column of the `[batch, steps, 256]` layer
///   output, so the host adds `direction * 128`; the launch writes hidden unit `u`
///   of step `t` to column `u` from there. Forming that offset here made ptxas
///   (CUDA 13.0, sm_120) drop the pointer's high 32 bits
/// - `state`: `STATE_TILE` zeroed or older-flagged words per batch tile, private to
///   this direction
/// - `direction`: 0 runs `t = 0..steps`, 1 runs `t = steps - 1..=0`
/// - `first_tile`: the batch tile of block row 0, so the tiles of one direction can
///   be split over several launches when the GPU cannot hold them all at once
/// - `tile_rows`: windows per batch tile, at most `TILE_ROWS`. Windows are
///   independent recurrences that only share weights, so smaller tiles let the
///   blocks of one SM overlap one tile's waits with another's arithmetic
/// - `flag_base`: larger than every flag an earlier launch since the last zeroing
///   wrote to `state`; the host passes `layer * steps`
///
/// The initial hidden and cell states are zero. Launch `(GROUPS, tiles, 1)` blocks of
/// `THREADS` threads cooperatively, with `first_tile + tiles <= ceil(batch /
/// tile_rows)`
#[kernel]
#[launch_bounds(128)]
#[allow(clippy::too_many_arguments)]
pub fn spk_lstm_recurrence(
    gates_x: &[f32],
    weights: &[f32],
    bias: &[f32],
    output: *mut f32,
    state: *mut u64,
    batch: u32,
    steps: u32,
    direction: u32,
    first_tile: u32,
    tile_rows: u32,
    flag_base: u32,
) {
    static mut HIDDEN_TILE: SharedArray<f32, { TILE_ROWS * HIDDEN }, 16> = SharedArray::UNINIT;
    static mut SUMS: SharedArray<f32, { TILE_ROWS * COLUMNS }, 16> = SharedArray::UNINIT;
    // SAFETY: raw pointers to this block's shared arrays; every access below is
    // separated from conflicting ones by a block barrier
    let hidden_tile = unsafe { SharedArray::as_raw_mut_ptr(&raw mut HIDDEN_TILE) };
    // SAFETY: as above
    let sums = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SUMS) };

    let tid = thread::threadIdx_x() as usize;
    let group = thread::blockIdx_x() as usize;
    let tile = (first_tile + thread::blockIdx_y()) as usize;
    let batch = batch as usize;
    let steps = steps as usize;
    let reverse = direction != 0;
    let tile_rows = (tile_rows as usize).min(TILE_ROWS);
    let row0 = tile * tile_rows;
    let rows = (batch - row0).min(tile_rows);
    let chunks = rows.div_ceil(CHUNK);
    // SAFETY: the host passes STATE_TILE words for every tile of this direction
    let tile_state = unsafe { state.add(tile * STATE_TILE) };

    // product role: the four gate rows of the block's unit `warp` over hidden inputs
    // `8 * slice..8 * slice + 8`, for the rows of parity `half` in each chunk
    let lane = tid % 32;
    let slice = lane % K_LANES;
    let half = lane / K_LANES;
    let product = Product {
        unit: tid / 32,
        slice,
        half,
    };
    let mut w = [[0.0f32; LANE_K]; GATES];
    unroll!(G in [0, 1, 2, 3] {
        let row = group * COLUMNS + product.unit * GATES + G;
        unroll!(K in [0, 1, 2, 3, 4, 5, 6, 7] {
            w[G][K] = weights[row * HIDDEN + slice * LANE_K + K];
        });
    });

    // cell role: hidden unit `unit` of window `cell_row` in the tile
    let cell_row = tid / UNITS;
    let unit = tid % UNITS;
    let active = cell_row < rows;
    let window = row0 + cell_row;
    let hidden_unit = group * UNITS + unit;
    let gate_column = hidden_unit * GATES;
    let gate_bias = [
        bias[gate_column],
        bias[gate_column + 1],
        bias[gate_column + 2],
        bias[gate_column + 3],
    ];
    let mut cell = 0.0f32;

    // rows past the tile's windows stay zero, so whole chunks need no row checks.
    // Straight-line stores, so every shared array is written on every path before any
    // read, which a dominance check on the PTX can see; a loop would carry a guard
    let zero = Vec4([0.0; 4]);
    unroll!(J in [0, 1, 2, 3, 4, 5, 6, 7] {
        // SAFETY: 128 threads x 8 vectors x 4 floats cover the TILE_ROWS * HIDDEN tile
        unsafe { *(hidden_tile.add((J * THREADS + tid) * 4) as *mut Vec4) = zero };
    });
    // SAFETY: 128 threads x 4 floats cover the TILE_ROWS * COLUMNS sums
    unsafe { *(sums.add(tid * 4) as *mut Vec4) = zero };
    thread::sync_threads();

    let mut s = 0;
    while s < steps {
        let t = if reverse { steps - 1 - s } else { s };

        // the input projection does not depend on the previous step, so its load
        // overlaps the wait below
        let mut x_gates = [0.0f32; GATES];
        if active {
            let base = (window * steps + t) * GATE_COLUMNS + gate_column;
            x_gates = [
                gates_x[base],
                gates_x[base + 1],
                gates_x[base + 2],
                gates_x[base + 3],
            ];
        }

        let mut recurrent = [0.0f32; GATES];
        if s > 0 {
            // step `s - 1` wrote flag `flag_base + s` into parity `(s - 1) % 2`
            // SAFETY: the slot is inside this tile's state
            let slot = unsafe { tile_state.add(((s - 1) % 2) * STATE_SLOT) };
            load_hidden_tile(slot, hidden_tile, rows, flag_base + s as u32);
            thread::sync_threads();

            recurrent_product(hidden_tile, sums, &w, product, chunks);
            thread::sync_threads();

            if active {
                // SAFETY: inside the sums array and 16-byte aligned
                recurrent =
                    unsafe { *(sums.add(cell_row * COLUMNS + unit * GATES) as *const Vec4) }.0;
            }
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

            // the last step has no reader
            if s + 1 < steps {
                let word = (s % 2) * STATE_SLOT + cell_row * HIDDEN + hidden_unit;
                // SAFETY: inside this tile's state; only this thread writes the word
                // during this step
                unsafe { publish(tile_state.add(word), hidden, flag_base + s as u32 + 1) };
            }

            let index = (window * steps + t) * OUTPUT_COLUMNS + hidden_unit;
            // SAFETY: `window < batch` and `t < steps`, so the index is inside this
            // direction's half of the `[batch, steps, 256]` output, and only this
            // thread writes it
            unsafe { *output.add(index) = hidden };
        }
        s += 1;
    }
}

/// Copies the previous step's hidden vectors of the tile's windows into shared
/// memory, waiting until every word carries `flag`
#[inline(always)]
fn load_hidden_tile(slot: *const u64, hidden_tile: *mut f32, rows: usize, flag: u32) {
    let tid = thread::threadIdx_x() as usize;
    let total = rows * HIDDEN;

    // wait on one word per thread before reading the rest, so the first spin
    // does not flood L2 while the producers still write
    if tid < total {
        // SAFETY: `tid < rows * HIDDEN`, so the word is inside the slot
        let mut word = unsafe { read_state(slot.add(tid)) };
        while !ready(word, flag) {
            // SAFETY: as above
            word = unsafe { read_state(slot.add(tid)) };
        }
    }

    // adjacent lanes read adjacent words, so scalar loads use all of each L2
    // sector. Eight independent words keep as many values in flight as four pairs
    let mut first = tid;
    while first < total {
        let mut words = [0u64; LOAD_BATCH];
        unroll!(J in [0, 1, 2, 3, 4, 5, 6, 7] {
            let i = first + J * THREADS;
            if i < total {
                // SAFETY: `i < rows * HIDDEN`, so the word is inside the slot
                words[J] = unsafe { read_state(slot.add(i)) };
            }
        });

        loop {
            let mut waiting = false;
            unroll!(J in [0, 1, 2, 3, 4, 5, 6, 7] {
                let i = first + J * THREADS;
                if i < total && !ready(words[J], flag) {
                    // SAFETY: as above
                    words[J] = unsafe { read_state(slot.add(i)) };
                    waiting = true;
                }
            });
            if !waiting {
                break;
            }
        }

        unroll!(J in [0, 1, 2, 3, 4, 5, 6, 7] {
            let i = first + J * THREADS;
            if i < total {
                let row = i / HIDDEN;
                let offset = row * HIDDEN + swizzle(i % HIDDEN);
                // SAFETY: `row < rows` and the swizzle permutes within the row
                unsafe { *hidden_tile.add(offset) = f32::from_bits(words[J] as u32) };
            }
        });
        first += LOAD_BATCH * THREADS;
    }
}

/// Halves the values a lane holds by adding them with the lane `offset` away
///
/// The lane with the `offset` bit set keeps the upper half. After the offsets 8, 4,
/// 2 and 1, lane `l` of each half-warp holds the 16-lane sum of value `l`. Each sum is computed in
/// exactly one lane, so the order is fixed
macro_rules! reduce_scatter {
    ($values:ident, $lane:expr, $offset:literal, [$($i:literal),*]) => {{
        let upper = $lane & $offset != 0;
        $({
            let low = $values[$i];
            let high = $values[$i + $offset];
            let (keep, send) = if upper { (high, low) } else { (low, high) };
            $values[$i] = keep + warp::shuffle_xor_f32(send, $offset);
        })*
    }};
}

/// Where hidden input `k` of a row lives in the shared tile
///
/// Four-float groups 8..15 of each 32-group half swap neighbours, so the eight
/// lanes of a quarter warp, which read groups `2 * slice + q`, hit eight different
/// 16-byte bank groups
#[inline(always)]
fn swizzle(k: usize) -> usize {
    let group = k / 4;
    (group ^ ((group >> 3) & 1)) * 4 + k % 4
}

/// One thread's part of the recurrent product
#[derive(Clone, Copy)]
struct Product {
    /// The block's hidden unit whose four gates this warp computes
    unit: usize,
    /// This lane's hidden inputs: `8 * slice..8 * slice + 8`
    slice: usize,
    /// The row parity this half-warp takes within a chunk
    half: usize,
}

/// `sums[row][unit * 4 + gate] = sum_k hidden[row][k] * R[unit * 4 + gate][k]` for
/// this warp's unit and the rows of the first `chunks` chunks
///
/// Each lane adds its eight inputs in order with FMAs, then a butterfly over the 16
/// lanes of the half-warp adds the lane sums in a fixed order
#[inline(always)]
fn recurrent_product(
    hidden_tile: *const f32,
    sums: *mut f32,
    w: &[[f32; LANE_K]; GATES],
    product: Product,
    chunks: usize,
) {
    let Product { unit, slice, half } = product;
    let mut chunk = 0;
    while chunk < chunks {
        let first = chunk * CHUNK + half;
        // value `j * 4 + gate` for row `first + 2 * j`
        let mut values = [0.0f32; 4 * GATES];
        unroll!(J in [0, 1, 2, 3] {
            let row = (first + 2 * J) * HIDDEN;
            unroll!(Q in [0, 1] {
                let offset = row + swizzle((2 * slice + Q) * 4);
                // SAFETY: the row is below TILE_ROWS and the vector inside it
                let h = unsafe { *(hidden_tile.add(offset) as *const Vec4) }.0;
                unroll!(G in [0, 1, 2, 3] {
                    let mut a = if Q == 0 {
                        h[0] * w[G][0]
                    } else {
                        fma_rn_f32(h[0], w[G][4], values[J * GATES + G])
                    };
                    a = fma_rn_f32(h[1], w[G][4 * Q + 1], a);
                    a = fma_rn_f32(h[2], w[G][4 * Q + 2], a);
                    a = fma_rn_f32(h[3], w[G][4 * Q + 3], a);
                    values[J * GATES + G] = a;
                });
            });
        });

        reduce_scatter!(values, slice, 8, [0, 1, 2, 3, 4, 5, 6, 7]);
        reduce_scatter!(values, slice, 4, [0, 1, 2, 3]);
        reduce_scatter!(values, slice, 2, [0, 1]);
        reduce_scatter!(values, slice, 1, [0]);

        // lane `slice` holds row `first + 2 * (slice / 4)`, gate `slice % 4`
        let offset = (first + 2 * (slice / GATES)) * COLUMNS + unit * GATES + slice % GATES;
        // SAFETY: inside the sums array; each (row, column) has one writer
        unsafe { *sums.add(offset) = values[0] };
        chunk += 1;
    }
}
