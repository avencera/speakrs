//! Register-tiled FP32 input projections in the recurrence's packed gate layout

use cuda_device::{SharedArray, kernel, launch_bounds, thread};

use super::{Vec4, fma_rn_f32};

const K_TILE: usize = 8;
const A_STRIDE: usize = K_TILE;
const MAX_TILE: usize = 128;
const COLUMNS: usize = 512;

/// Computes one direction's input projection without changing the bias order
///
/// `input` is row-major `[rows, columns]`, `weights` is `[padded_columns, 512]`
/// in packed `[unit, i/f/c/o]` column order, and `output` is `[rows, 512]`
/// The recurrence adds the combined bias after the recurrent product
///
/// # Safety
///
/// The three buffers have the stated lengths and do not overlap. `columns` is
/// 60 or 256, and `padded_columns` is its multiple-of-16 padded width. Padded
/// weight rows are zero. Launch 256 threads per block, four blocks along
/// x and `ceil(rows / 128)` along y, with no dynamic shared memory
/// The host supplies naturally aligned FP32 allocations
#[kernel]
#[launch_bounds(256, 2)]
pub unsafe fn spk_lstm_projection(
    input: *const f32,
    weights: *const f32,
    output: *mut f32,
    rows: u32,
    columns: u32,
    padded_columns: u32,
) {
    static mut STAGING: SharedArray<f32, { MAX_TILE * (A_STRIDE + K_TILE) }, 16> =
        SharedArray::UNINIT;
    // safety: this block owns the array; load and compute phases have block barriers
    let shared = unsafe { SharedArray::as_raw_mut_ptr(&raw mut STAGING) };
    let dimensions = Dimensions {
        rows: rows as usize,
        columns: columns as usize,
        padded_columns: padded_columns as usize,
    };
    // safety: launch and buffer contracts are those of this entry
    unsafe { project::<true>(input, weights, output, shared, dimensions) };
}

/// Computes the same projection with a 64-by-64 tile for smaller row counts
///
/// # Safety
///
/// Buffer and width contracts match [`spk_lstm_projection`]. Launch 256 threads
/// per block, eight blocks along x and `ceil(rows / 64)` along y, with no dynamic
/// shared memory. Separate entries give each tile its own register
/// and shared-memory budget
#[kernel]
#[launch_bounds(256, 4)]
pub unsafe fn spk_lstm_projection_small(
    input: *const f32,
    weights: *const f32,
    output: *mut f32,
    rows: u32,
    columns: u32,
    padded_columns: u32,
) {
    static mut STAGING_SMALL: SharedArray<f32, { 64 * (A_STRIDE + K_TILE) }, 16> =
        SharedArray::UNINIT;
    // safety: this block owns the array; tile consumers and stores have block barriers
    let shared = unsafe { SharedArray::as_raw_mut_ptr(&raw mut STAGING_SMALL) };
    let dimensions = Dimensions {
        rows: rows as usize,
        columns: columns as usize,
        padded_columns: padded_columns as usize,
    };
    // safety: launch and buffer contracts are those of this entry
    unsafe { project::<false>(input, weights, output, shared, dimensions) };
}

/// Computes the projection through TF32 matrix fragments on sm80 and newer
///
/// # Safety
///
/// On sm80+, launch a 128-by-256 output tile with 256 threads and 61,440 dynamic
/// shared bytes. Weights are column-major `[512, padded_columns]` and already
/// rounded to TF32. On sm75, use [`spk_lstm_projection`]'s launch and buffer contracts
#[kernel]
#[launch_bounds(256, 1)]
pub unsafe fn spk_lstm_projection_tf32(
    input: *const f32,
    weights: *const f32,
    output: *mut f32,
    rows: u32,
    columns: u32,
    padded_columns: u32,
) {
    #[cfg(feature = "tier-sm80")]
    unsafe {
        super::tensor::project(
            input,
            weights,
            output,
            rows as usize,
            columns as usize,
            padded_columns as usize,
        );
    }
    #[cfg(not(feature = "tier-sm80"))]
    {
        static mut STAGING_TF32: SharedArray<f32, { MAX_TILE * (A_STRIDE + K_TILE) }, 16> =
            SharedArray::UNINIT;
        let shared = unsafe { SharedArray::as_raw_mut_ptr(&raw mut STAGING_TF32) };
        let dimensions = Dimensions {
            rows: rows as usize,
            columns: columns as usize,
            padded_columns: padded_columns as usize,
        };
        unsafe { project::<true>(input, weights, output, shared, dimensions) };
    }
}

#[derive(Clone, Copy)]
struct Dimensions {
    rows: usize,
    columns: usize,
    padded_columns: usize,
}

#[derive(Clone, Copy)]
struct TileLoad {
    a: Vec4,
    b: Vec4,
}

#[inline(always)]
unsafe fn load_tile<const LARGE: bool>(
    input: *const f32,
    weights: *const f32,
    dimensions: Dimensions,
    inner: usize,
) -> TileLoad {
    let tile = if LARGE { 128 } else { 64 };
    let tid = thread::threadIdx_x() as usize;
    let first_row = thread::blockIdx_y() as usize * tile;
    let first_col = thread::blockIdx_x() as usize * tile;
    let mut loaded = TileLoad {
        a: Vec4([0.0; 4]),
        b: Vec4([0.0; 4]),
    };
    if inner >= dimensions.padded_columns || tid >= tile * K_TILE / 4 {
        return loaded;
    }

    let global_row = first_row + tid / (K_TILE / 4);
    let global_k = inner + tid % (K_TILE / 4) * 4;
    if global_row < dimensions.rows && global_k < dimensions.columns {
        // safety: widths and offsets are multiples of four and this row is live
        loaded.a =
            unsafe { *(input.add(global_row * dimensions.columns + global_k) as *const Vec4) };
    }
    // safety: padded weight rows exist, including the last input-width tile
    loaded.b = unsafe {
        *(weights.add((inner + tid / (tile / 4)) * COLUMNS + first_col + tid % (tile / 4) * 4)
            as *const Vec4)
    };
    loaded
}

#[inline(always)]
unsafe fn store_tile<const LARGE: bool>(a: *mut f32, b: *mut f32, loaded: TileLoad) {
    let tile = if LARGE { 128 } else { 64 };
    let tid = thread::threadIdx_x() as usize;
    if tid >= tile * K_TILE / 4 {
        return;
    }

    // safety: every live tile vector has one loader and both addresses are aligned
    unsafe {
        unroll!(J in [0, 1, 2, 3] {
            *a.add((tid % (K_TILE / 4) * 4 + J) * tile + tid / (K_TILE / 4)) = loaded.a.0[J];
        });
        *(b.add(tid * 4) as *mut Vec4) = loaded.b;
    }
}

#[inline(always)]
unsafe fn project<const LARGE: bool>(
    input: *const f32,
    weights: *const f32,
    output: *mut f32,
    shared: *mut f32,
    dimensions: Dimensions,
) {
    let tile = if LARGE { 128 } else { 64 };
    let tid = thread::threadIdx_x() as usize;
    let first_row = thread::blockIdx_y() as usize * tile;
    let first_col = thread::blockIdx_x() as usize * tile;
    let row = tid / 16 * 4;
    let col = tid % 16 * 4;
    // transposed A rows let each thread read four row values with one vector load
    let a = shared;
    // safety: the A tile precedes the B tile within STAGING
    let b = unsafe { shared.add(tile * A_STRIDE) };
    let mut sums = [[0.0f32; 8]; 8];
    let mut inner = 0;
    // safety: load and store contracts are the entry's buffers and this block's tile
    unsafe { store_tile::<LARGE>(a, b, load_tile::<LARGE>(input, weights, dimensions, 0)) };
    while inner < dimensions.padded_columns {
        thread::sync_threads();
        // register prefetch overlaps global reads with the current shared-tile FMAs
        // safety: the final iteration returns zero vectors without reading past W
        let next = unsafe { load_tile::<LARGE>(input, weights, dimensions, inner + K_TILE) };

        let mut k = 0;
        while k < K_TILE {
            let low_a = unsafe { *(a.add(k * tile + row) as *const Vec4) }.0;
            let mut high_a = [0.0; 4];
            if LARGE {
                high_a = unsafe { *(a.add(k * tile + row + 64) as *const Vec4) }.0;
            }
            // safety: col is a multiple of four within this B row
            let low = unsafe { *(b.add(k * tile + col) as *const Vec4) }.0;
            let mut high = [0.0; 4];
            if LARGE {
                // safety: the upper four columns are within the 128-column B tile
                high = unsafe { *(b.add(k * tile + col + 64) as *const Vec4) }.0;
            }
            unroll!(R in [0, 1, 2, 3] {
                unroll!(C in [0, 1, 2, 3] {
                    sums[R][C] = fma_rn_f32(low_a[R], low[C], sums[R][C]);
                    if LARGE {
                        sums[R][C + 4] = fma_rn_f32(low_a[R], high[C], sums[R][C + 4]);
                        sums[R + 4][C] = fma_rn_f32(high_a[R], low[C], sums[R + 4][C]);
                        sums[R + 4][C + 4] = fma_rn_f32(high_a[R], high[C], sums[R + 4][C + 4]);
                    }
                });
            });
            k += 1;
        }
        // all readers finish before a loader overwrites the shared tile
        thread::sync_threads();
        // safety: the barrier completed every reader of the previous tile
        unsafe { store_tile::<LARGE>(a, b, next) };
        inner += K_TILE;
    }

    unroll!(R in [0, 1, 2, 3] {
        let global_row = first_row + row + R;
        if global_row < dimensions.rows {
            // safety: one thread owns these four aligned columns of this live row
            unsafe {
                *(output.add(global_row * COLUMNS + first_col + col) as *mut Vec4) =
                    Vec4([sums[R][0], sums[R][1], sums[R][2], sums[R][3]]);
                if LARGE {
                    *(output.add(global_row * COLUMNS + first_col + col + 64) as *mut Vec4) =
                        Vec4([sums[R][4], sums[R][5], sums[R][6], sums[R][7]]);
                }
            }
        }
        if LARGE && global_row + 64 < dimensions.rows {
            // safety: the upper row is live and its columns have the same single owner
            unsafe {
                *(output.add((global_row + 64) * COLUMNS + first_col + col) as *mut Vec4) =
                    Vec4([sums[R + 4][0], sums[R + 4][1], sums[R + 4][2], sums[R + 4][3]]);
                *(output.add((global_row + 64) * COLUMNS + first_col + col + 64) as *mut Vec4) =
                    Vec4([sums[R + 4][4], sums[R + 4][5], sums[R + 4][6], sums[R + 4][7]]);
            }
        }
    });
}
