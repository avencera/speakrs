//! SincNet front-end producer: the band-pass convolution, `abs` and max pool in one
//! kernel, so the raw `[batch, 80, sinc]` convolution output is never stored
//!
//! The instance normalization that follows stays a separate pass over the pooled
//! tensor, because its channel-wide mean and variance need every pooled value of a
//! row before any output can be written
//!
//! The host side lives in `src/inference/cuda/candidate/sinc.rs` and is qualified by
//! `cargo xtask cuda-qualify sincnet`

use cuda_device::vector::F32x4;
use cuda_device::{DisjointSlice, SharedArray, kernel, launch_bounds, thread};

/// Taps of every SincNet filter
pub const TAPS: usize = 251;
/// Convolution stride in samples
pub const STRIDE: usize = 10;
/// Max pool kernel and stride
pub const POOL: usize = 3;
/// SincNet filters
pub const CHANNELS: usize = 80;
/// Channels of one block; [`spk_sincnet_pack_filters`] packs the filters in groups of
/// this many
pub const GROUP: usize = 16;
/// Pooled outputs of one block
pub const TILE: usize = 256;
/// Pooled outputs of one thread, `THREADS` apart; two ran no faster at b32 and
/// slower at b7 with 168 registers, so one keeps 96
pub const PER_THREAD: usize = 1;
/// Threads of one block
pub const THREADS: usize = TILE / PER_THREAD;

/// Float4 quads of one packed filter group: `[TAPS][GROUP]`
const GROUP_QUADS: usize = TAPS * GROUP / 4;
/// Raw convolution positions of one block
const TILE_RAW: usize = TILE * POOL;
/// Waveform entries of one phase in shared memory: every raw position plus the
/// furthest tap offset `(TAPS - 1) / STRIDE`
const PHASE_LEN: usize = TILE_RAW + (TAPS - 1) / STRIDE;
/// Waveform samples staged per block, split into [`STRIDE`] phases
const SPAN: usize = STRIDE * PHASE_LEN;
/// Waveform loads of one thread while staging
const SPAN_LOADS: usize = SPAN.div_ceil(THREADS);
/// Filter quad loads of one thread while staging
const QUAD_LOADS: usize = GROUP_QUADS.div_ceil(THREADS);
/// Raw positions of one thread
const RAW: usize = PER_THREAD * POOL;
/// Taps covered by the unrolled phase loop; the last tap runs on its own
const FULL_TAPS: usize = TAPS - TAPS % STRIDE;

/// `packed[(g * TAPS + k) * GROUP + lane] = filters[(g * GROUP + lane) * TAPS + k]`:
/// the `[CHANNELS][TAPS]` filters as `[CHANNELS / GROUP][TAPS][GROUP]`, so
/// [`spk_sincnet_conv_abs_pool`] stages a group with contiguous quad loads and every
/// tap's group is one broadcast row
///
/// The host packs on the device once per plan from the filters the plan is given.
/// Launch one thread per packed element, `CHANNELS * TAPS` in all; `filters` and
/// `packed` both hold that many
#[kernel]
pub fn spk_sincnet_pack_filters(filters: &[f32], mut packed: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    let o = idx.get();
    let group = o / (TAPS * GROUP);
    let rest = o % (TAPS * GROUP);
    let source = (group * GROUP + rest % GROUP) * TAPS + rest / GROUP;
    if source < filters.len()
        && let Some(element) = packed.get_mut(idx)
    {
        // SAFETY: `source < filters.len()`
        *element = unsafe { *filters.get_unchecked(source) };
    }
}

/// `pooled[b, c, p] = max_d |conv[b, c, POOL * p + d]|` for `d < POOL`, where
/// `conv[b, c, j] = Σ_k filters[c, k] * wave[b, STRIDE * j + k]`
///
/// Each output sums its taps in ascending order with one fused multiply-add per
/// tap, and the maximum starts at negative infinity and applies `f32::max` in tap
/// order, like the library pooling kernel. `pooled` is `[batch, 80, pool_len]`,
/// `wave` is `[batch, samples]` and `filters` is the packing
/// `[CHANNELS / GROUP][TAPS][GROUP]` of [`spk_sincnet_pack_filters`]. `pool_len` must be at most
/// `((samples - TAPS) / STRIDE + 1) / POOL`, so every valid output reads only
/// samples of its own row
///
/// Launch `grid = (ceil(pool_len / TILE), CHANNELS / GROUP, batch)` with
/// [`THREADS`] threads. Thread `t` owns pooled outputs `blockIdx.x * TILE + t + m *
/// THREADS` for `m <` [`PER_THREAD`] and the block's [`GROUP`] channels. The block
/// stages its waveform span in shared memory split by sample phase, `xs[r][i] =
/// wave[span_start + STRIDE * i + r]`, so the three raw positions of a pooled
/// output read consecutive words and a warp reads 32 distinct banks. Every lane
/// reads the same filter quads, so the filter loads are broadcasts; the pooled
/// outputs of a thread share each of them. Samples past the row end are staged as
/// zero; only outputs past `pool_len`, which are never stored, read them
#[kernel]
#[launch_bounds(256, 2)]
pub fn spk_sincnet_conv_abs_pool(
    wave: &[f32],
    filters: &[F32x4],
    mut pooled: DisjointSlice<f32>,
    samples: u32,
    pool_len: u32,
) {
    static mut XS: SharedArray<f32, SPAN> = SharedArray::UNINIT;
    static mut WS: SharedArray<F32x4, GROUP_QUADS, 16> = SharedArray::UNINIT;
    // SAFETY: raw pointers to this block's shared arrays; every write happens
    // before the barrier below and every read after it
    let xs = unsafe { SharedArray::as_raw_mut_ptr(&raw mut XS) };
    let ws = unsafe { SharedArray::as_raw_mut_ptr(&raw mut WS) };

    let t = thread::threadIdx_x() as usize;
    let tile = thread::blockIdx_x() as usize;
    let group = thread::blockIdx_y() as usize;
    let row = thread::blockIdx_z() as usize;
    let samples = samples as usize;
    let pool_len = pool_len as usize;

    // stage the waveform span and the block's filter group; every load of a thread
    // is issued before its first shared store, so the loads overlap instead of
    // each waiting a full memory latency
    let span_start = tile * TILE_RAW * STRIDE;
    let row_base = row * samples;
    let wave_len = wave.len();
    let mut values = [0.0f32; SPAN_LOADS];
    let mut i = 0;
    #[unroll]
    while i < SPAN_LOADS {
        let sample = span_start + t + i * THREADS;
        let index = row_base + sample;
        if t + i * THREADS < SPAN && sample < samples && index < wave_len {
            // SAFETY: `index < wave.len()`
            values[i] = unsafe { *wave.get_unchecked(index) };
        }
        i += 1;
    }

    let quads = group * GROUP_QUADS;
    let mut weights = [F32x4::splat(0.0); QUAD_LOADS];
    let mut i = 0;
    #[unroll]
    while i < QUAD_LOADS {
        let q = t + i * THREADS;
        if q < GROUP_QUADS && quads + q < filters.len() {
            // SAFETY: `quads + q < filters.len()`
            weights[i] = unsafe { *filters.get_unchecked(quads + q) };
        }
        i += 1;
    }

    // samples past the row end are staged as zero
    let mut i = 0;
    #[unroll]
    while i < SPAN_LOADS {
        let s = t + i * THREADS;
        if s < SPAN {
            // SAFETY: `s < SPAN`, so the phase index is below `STRIDE` and the
            // entry below `PHASE_LEN`
            unsafe { *xs.add((s % STRIDE) * PHASE_LEN + s / STRIDE) = values[i] };
        }
        i += 1;
    }

    let mut i = 0;
    #[unroll]
    while i < QUAD_LOADS {
        let q = t + i * THREADS;
        if q < GROUP_QUADS {
            // SAFETY: `q < GROUP_QUADS`
            unsafe { *ws.add(q) = weights[i] };
        }
        i += 1;
    }
    thread::sync_threads();

    // `acc[c * RAW + m * POOL + d]` sums channel `c` at raw position `d` of the
    // thread's pooled output `m`
    let mut acc = [0.0f32; GROUP * RAW];
    // one tap for every channel and raw position: `x` points at the thread's first
    // phase entry and `w` at the tap's filter quads; a closure, so the `#[kernel]`
    // macro rewrites its `#[unroll]` loops and `acc` stays in registers
    let mut tap = |x: *const f32, w: *const F32x4| {
        let mut xv = [0.0f32; RAW];
        let mut r = 0;
        #[unroll]
        while r < RAW {
            // SAFETY: the caller's bound covers `POOL` consecutive entries at each
            // of the thread's pooled outputs
            xv[r] = unsafe { *x.add((r / POOL) * THREADS * POOL + r % POOL) };
            r += 1;
        }

        let mut quad = 0;
        #[unroll]
        while quad < GROUP / 4 {
            // SAFETY: the caller's bound covers the tap's `GROUP / 4` quads
            let wq = unsafe { *w.add(quad) }.to_array();
            let mut i = 0;
            #[unroll]
            while i < 4 * RAW {
                let a = (quad * 4 + i / RAW) * RAW + i % RAW;
                acc[a] = wq[i / RAW].mul_add(xv[i % RAW], acc[a]);
                i += 1;
            }
            quad += 1;
        }
    };

    // SAFETY (every `tap` below): the largest entry read is `POOL * (TILE - 1) +
    // POOL - 1 + (TAPS - 1) / STRIDE = PHASE_LEN - 1` within a phase, and the
    // largest filter quad is `GROUP_QUADS - 1`
    let x_thread = unsafe { xs.add(POOL * t) };
    let mut step = 0;
    while step < FULL_TAPS / STRIDE {
        let mut phase = 0;
        #[unroll]
        while phase < STRIDE {
            let x = unsafe { x_thread.add(phase * PHASE_LEN + step) };
            let w = unsafe { ws.add((step * STRIDE + phase) * (GROUP / 4)) };
            tap(x, w);
            phase += 1;
        }
        step += 1;
    }
    let mut k = FULL_TAPS;
    while k < TAPS {
        let x = unsafe { x_thread.add((k % STRIDE) * PHASE_LEN + k / STRIDE) };
        let w = unsafe { ws.add(k * (GROUP / 4)) };
        tap(x, w);
        k += 1;
    }

    let out_len = pooled.len();
    let mut m = 0;
    #[unroll]
    while m < PER_THREAD {
        let p = tile * TILE + t + m * THREADS;
        let mut c = 0;
        #[unroll]
        while c < GROUP {
            let mut value = f32::NEG_INFINITY;
            let mut d = 0;
            #[unroll]
            while d < POOL {
                value = value.max(acc[c * RAW + m * POOL + d].abs());
                d += 1;
            }

            let o = (row * CHANNELS + group * GROUP + c) * pool_len + p;
            if p < pool_len && o < out_len {
                // SAFETY: in bounds, and each (row, channel, p) has one owning
                // thread
                unsafe { *pooled.get_unchecked_mut(o) = value };
            }
            c += 1;
        }
        m += 1;
    }
}
