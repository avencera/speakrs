//! Segmentation (SincNet, BiLSTM head) kernels
//!
//! The SincNet convolutions run in cuDNN, the BiLSTM stack in the cuDNN RNN API and the
//! linear layers in cuBLAS. These kernels cover what is left: max pooling, instance
//! normalization and LeakyReLU after each convolution (and the waveform instance
//! normalization in front), the bias and LeakyReLU after each linear layer, and the
//! bias and log-softmax of the classifier

use cuda_device::{DisjointSlice, SharedArray, kernel, thread, warp};

/// Upper bound on warps per block for the block reductions
const MAX_WARPS: usize = 32;

/// Powerset classes of the segmentation model (3 speakers, at most 2 active)
const CLASSES: usize = 7;

/// Sums `value` over the whole block and returns the total to every thread
///
/// The block size must be a multiple of 32 and at most 1024. Every thread adds the
/// warp partials in the same order, so all threads get bit-identical totals
#[inline(always)]
fn block_sum(value: f32, scratch: *mut f32) -> f32 {
    let warp_index = thread::threadIdx_x() as usize / 32;
    let warps = thread::blockDim_x() as usize / 32;
    let partial = warp::reduce_sum_f32(value);
    if warp::lane_id() == 0 {
        // SAFETY: `warp_index < MAX_WARPS` because the block has at most 1024 threads
        unsafe { *scratch.add(warp_index) = partial };
    }
    thread::sync_threads();

    let mut total = 0.0f32;
    let mut w = 0;
    while w < warps {
        // SAFETY: every warp wrote its slot before the barrier above
        total += unsafe { *scratch.add(w) };
        w += 1;
    }

    // the scratch slots are rewritten by the next reduction
    thread::sync_threads();
    total
}

/// Max pooling, instance normalization and LeakyReLU over one `(batch, channel)` row
/// per block
///
/// For row `r = b * channels + c` and output step `t`, the pooled value is
/// `max_k f(input[r * in_len + t * pool + k] + conv_bias[c])` over `k < pool`, where
/// `f` is `abs` when `abs_input != 0` and the identity otherwise. The row is then
/// normalized with its biased variance, scaled by `gamma[c]`, shifted by `beta[c]`,
/// passed through LeakyReLU with `slope` and written to
/// `out[b * out_batch_stride + c * out_channel_stride + t * out_time_stride]`.
/// `pool = 1`, `abs_input = 0` and `slope = 1` give a plain instance normalization.
///
/// `out` holds the pooled values between passes, so the output layout must give every
/// `(b, c, t)` its own element. Launch `batch * channels` blocks of 32 to 1024 threads
/// (a multiple of 32), with `out_len = in_len / pool`
#[kernel]
pub fn segmentation_pool_norm(
    input: &[f32],
    conv_bias: &[f32],
    gamma: &[f32],
    beta: &[f32],
    mut out: DisjointSlice<f32>,
    channels: u32,
    in_len: u32,
    out_len: u32,
    pool: u32,
    abs_input: u32,
    slope: f32,
    epsilon: f32,
    out_batch_stride: u32,
    out_channel_stride: u32,
    out_time_stride: u32,
) {
    static mut SCRATCH: SharedArray<f32, MAX_WARPS> = SharedArray::UNINIT;
    // SAFETY: a raw pointer to the block's shared array; only `block_sum` touches it,
    // between barriers
    let scratch = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SCRATCH) };

    let row = thread::blockIdx_x() as usize;
    let channels = channels as usize;
    let batch = row / channels;
    let channel = row % channels;
    let in_len = in_len as usize;
    let out_len = out_len as usize;
    let pool = pool as usize;
    let in_base = row * in_len;
    let out_base = batch * out_batch_stride as usize + channel * out_channel_stride as usize;
    let out_time_stride = out_time_stride as usize;
    let out_capacity = out.len();
    let bias = conv_bias[channel];
    let step = thread::blockDim_x() as usize;
    let first = thread::threadIdx_x() as usize;

    // pass 1: pool into `out` and sum
    let mut sum = 0.0f32;
    let mut t = first;
    while t < out_len {
        let start = in_base + t * pool;
        let mut pooled = f32::NEG_INFINITY;
        let mut k = 0;
        while k < pool {
            let mut value = input[start + k] + bias;
            if abs_input != 0 {
                value = value.abs();
            }
            pooled = pooled.max(value);
            k += 1;
        }

        let o = out_base + t * out_time_stride;
        if o < out_capacity {
            // SAFETY: `o` is in bounds, and each (row, t) is written only by this thread
            unsafe { *out.get_unchecked_mut(o) = pooled };
        }
        sum += pooled;
        t += step;
    }
    let mean = block_sum(sum, scratch) / out_len as f32;

    // pass 2: biased variance around the mean, as ONNX Runtime computes it
    let mut squares = 0.0f32;
    let mut t = first;
    while t < out_len {
        let o = out_base + t * out_time_stride;
        if o < out_capacity {
            // SAFETY: in bounds, and this thread wrote the element in pass 1
            let centered = unsafe { *out.get_unchecked_mut(o) } - mean;
            squares += centered * centered;
        }
        t += step;
    }
    let variance = block_sum(squares, scratch) / out_len as f32;

    // pass 3: y = x * scale + shift, then LeakyReLU
    let scale = 1.0f32 / (variance + epsilon).sqrt() * gamma[channel];
    let shift = beta[channel] - mean * scale;
    let mut t = first;
    while t < out_len {
        let o = out_base + t * out_time_stride;
        if o < out_capacity {
            // SAFETY: in bounds, and this thread owns the element
            let element = unsafe { out.get_unchecked_mut(o) };
            let normalized = *element * scale + shift;
            *element = if normalized >= 0.0 {
                normalized
            } else {
                normalized * slope
            };
        }
        t += step;
    }
}

/// In place `x[i] = leaky_relu(x[i] + bias[i % bias.len()])` over a row-major
/// matrix whose row length is `bias.len()`
///
/// Launch one thread per element of `x` with a 1-D grid
#[kernel]
pub fn segmentation_bias_leaky(bias: &[f32], slope: f32, mut x: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    let column = idx.get() % bias.len();
    if let Some(element) = x.get_mut(idx) {
        let value = *element + bias[column];
        *element = if value >= 0.0 { value } else { value * slope };
    }
}

/// In place `rows[r] = log_softmax(rows[r] + bias)` over the 7 powerset classes
///
/// Launch one thread per row with a 1-D grid; `bias` holds 7 values
#[kernel]
pub fn segmentation_bias_log_softmax(bias: &[f32], mut rows: DisjointSlice<[f32; CLASSES]>) {
    let idx = thread::index_1d();
    if let Some(row) = rows.get_mut(idx) {
        let mut logits = [0.0f32; CLASSES];
        let mut max = f32::NEG_INFINITY;
        let mut k = 0;
        while k < CLASSES {
            logits[k] = row[k] + bias[k];
            max = max.max(logits[k]);
            k += 1;
        }

        let mut sum = 0.0f32;
        let mut k = 0;
        while k < CLASSES {
            sum += (logits[k] - max).exp();
            k += 1;
        }

        let log_sum = sum.ln();
        let mut k = 0;
        while k < CLASSES {
            row[k] = logits[k] - max - log_sum;
            k += 1;
        }
    }
}
