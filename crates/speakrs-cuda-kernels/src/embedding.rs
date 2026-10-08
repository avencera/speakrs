//! ResNet34 multi-mask embedding kernels
//!
//! cuDNN runs the convolutions, fused with their bias, residual and ReLU, and
//! cuBLAS the embedding GEMM. These kernels cover the rest of
//! `wespeaker-multimask-tail.onnx`: the fbank transpose in front of the stem, the
//! bias of the 1x1 shortcut convolutions, the multi-mask weighted mean and standard
//! deviation pooling, and the GEMM bias
//!
//! Index arithmetic is 32-bit where it can be: the host rejects tensors with more
//! than `u32::MAX` elements

use cuda_device::{DisjointSlice, kernel, thread, warp};

/// Lanes per warp; the pooling kernel assigns one warp per output column
const WARP: usize = 32;

/// Elements each thread of a channel epilogue handles
pub const EPILOGUE_ITEMS: u32 = 4;

/// Clamp applied to the weighted variance before the square root (`val_428`)
const VARIANCE_FLOOR: f32 = 1e-10;

/// Standard deviation written for a speaker whose resized mask sums to zero
/// (`cat_1` in the graph)
const EMPTY_STD: f32 = 1e-5;

/// Transposes fbank features `[b, frames, bins]` into the NCHW stem input
/// `[b, 1, bins, frames]`
///
/// Launch one thread per element of `out`
#[kernel]
pub fn embedding_fbank_transpose(
    fbank: &[f32],
    frames: u32,
    bins: u32,
    mut out: DisjointSlice<f32>,
) {
    let idx = thread::index_1d();
    let i = idx.get() as u32;
    let plane = frames * bins;
    let item = i / plane;
    let within = i - item * plane;
    let bin = within / frames;
    let frame = within - bin * frames;
    if let Some(out_elem) = out.get_mut(idx) {
        *out_elem = fbank[(item * plane + frame * bins + bin) as usize];
    }
}

/// `y = y + bias[channel]` in place over an NCHW tensor, for the 1x1 shortcut
/// convolutions, which have no ReLU of their own
///
/// Launch with `grid.y = n * channels`, one row of blocks per channel plane of
/// `plane = h * w` elements, and `grid.x = ceil(plane / (blockDim.x *
/// EPILOGUE_ITEMS))`. Each thread handles [`EPILOGUE_ITEMS`] elements a block apart
/// and issues their loads before any store, so several loads are in flight; the
/// kernel is memory bound
#[kernel]
pub fn embedding_bias(bias: &[f32], channels: u32, plane: u32, mut y: DisjointSlice<f32>) {
    let row = thread::blockIdx_y();
    let shift = bias[(row % channels) as usize];
    let base = row as usize * plane as usize;
    let block = thread::blockDim_x();
    let first = thread::blockIdx_x() * block * EPILOGUE_ITEMS + thread::threadIdx_x();
    let len = y.len();

    let mut values = [0.0f32; EPILOGUE_ITEMS as usize];
    let mut item = 0;
    while item < EPILOGUE_ITEMS {
        let i = first + item * block;
        let index = base + i as usize;
        if i < plane && index < len {
            // SAFETY: bounds checked above; only this thread touches `index`,
            // because each (row, block, thread, item) maps to a distinct element
            values[item as usize] = unsafe { *y.get_unchecked_mut(index) };
        }
        item += 1;
    }

    let mut item = 0;
    while item < EPILOGUE_ITEMS {
        let i = first + item * block;
        let index = base + i as usize;
        if i < plane && index < len {
            // SAFETY: as for the load above
            unsafe {
                *y.get_unchecked_mut(index) = values[item as usize] + shift;
            }
        }
        item += 1;
    }
}

/// Fills a row-major `[rows, cols]` matrix with `bias` in every row, so the
/// embedding GEMM can accumulate onto it with `beta = 1`
///
/// Launch one thread per element of `out`
#[kernel]
pub fn embedding_broadcast_rows(bias: &[f32], cols: u32, mut out: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    let col = idx.get() as u32 % cols;
    if let Some(out_elem) = out.get_mut(idx) {
        *out_elem = bias[col as usize];
    }
}

/// Multi-mask statistics pooling
///
/// `features` is the trunk output viewed as `[b, columns, frames]` (channels times
/// frequency bins by time), `masks` is `[b * speakers, mask_frames]` in chunk-major,
/// speaker-major order, and `pooled` is `[b * speakers, 2 * columns]` with the
/// weighted means first and the weighted standard deviations second
///
/// Each mask row is resized to `frames` by nearest neighbour with the ONNX
/// `asymmetric` and `floor` rules, so frame `t` reads mask frame
/// `floor(t * mask_frames / frames)`. With `w` the resized mask:
///
/// - `mean = Σ w·x / d` where `d = Σ w` when that sum is positive, otherwise 1
/// - `var = Σ (x - mean)²·w / (d - Σ w² / d)`, floored at `1e-10`; a NaN
///   variance passes through the floor unchanged
/// - `std = sqrt(var)`
/// - a row whose mask sums to zero or less pools to `mean = 0`, `std = 1e-5`
///
/// One warp computes one `(chunk, column)` pair for every speaker of that chunk,
/// so the feature row is read from global memory once. Launch
/// `b * columns * 32` threads with a block size that is a multiple of 32
#[kernel]
pub fn embedding_mask_pool(
    features: &[f32],
    masks: &[f32],
    speakers: u32,
    columns: u32,
    frames: u32,
    mask_frames: u32,
    mut pooled: DisjointSlice<f32>,
) {
    let idx = thread::index_1d();
    let warp_index = idx.get() as u32 / WARP as u32;
    let lane = warp::lane_id() as usize;
    let chunk = warp_index / columns;
    let column = (warp_index - chunk * columns) as usize;
    let row_start = (warp_index * frames) as usize;
    let chunk = chunk as usize;
    let columns = columns as usize;
    let speakers = speakers as usize;
    // a partial last warp cannot happen for a whole number of warps, but a host
    // bug must not read past the features
    if row_start + frames as usize > features.len() {
        return;
    }

    let mut speaker = 0;
    while speaker < speakers {
        let mask_row = (chunk * speakers + speaker) * mask_frames as usize;

        let mut sum_w = 0.0f32;
        let mut sum_w2 = 0.0f32;
        let mut sum_wx = 0.0f32;
        let mut t = lane as u32;
        while t < frames {
            let w = masks[mask_row + (t * mask_frames / frames) as usize];
            let x = features[row_start + t as usize];
            sum_w += w;
            sum_w2 += w * w;
            sum_wx += x * w;
            t += WARP as u32;
        }
        let sum_w = warp::reduce_sum_f32(sum_w);
        let sum_w2 = warp::reduce_sum_f32(sum_w2);
        let sum_wx = warp::reduce_sum_f32(sum_wx);

        let denom = if sum_w > 0.0 { sum_w } else { 1.0 };
        let mean = sum_wx / denom;

        let mut sum_sq = 0.0f32;
        let mut t = lane as u32;
        while t < frames {
            let w = masks[mask_row + (t * mask_frames / frames) as usize];
            let diff = features[row_start + t as usize] - mean;
            sum_sq += (diff * diff) * w;
            t += WARP as u32;
        }
        let sum_sq = warp::reduce_sum_f32(sum_sq);

        let variance = sum_sq / (denom - sum_w2 / denom);
        let variance = if variance < VARIANCE_FLOOR {
            VARIANCE_FLOOR
        } else {
            variance
        };
        let (mean, std) = if sum_w > 0.0 {
            (mean, variance.sqrt())
        } else {
            (0.0, EMPTY_STD)
        };

        let out_row = (chunk * speakers + speaker) * 2 * columns;
        if lane == 0 && out_row + columns + column < pooled.len() {
            // SAFETY: only lane 0 of each warp writes, each warp owns a distinct
            // (chunk, column) pair and so distinct mean and std slots, and the
            // bound is checked above
            unsafe {
                *pooled.get_unchecked_mut(out_row + column) = mean;
                *pooled.get_unchecked_mut(out_row + columns + column) = std;
            }
        }
        speaker += 1;
    }
}
