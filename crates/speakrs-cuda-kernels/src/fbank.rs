//! Filterbank (fbank) front-end kernels
//!
//! Together with two cuBLAS GEMMs on the host side, these compute what
//! `wespeaker-fbank.onnx` computes: Kaldi-style 80-bin log-mel features with temporal
//! cepstral mean normalization. The host keeps the frame layout in one place:
//!
//! - `frames` is `[rows · frames_per_row, FRAME_LENGTH]`, windowed and without the
//!   zero padding to 512, which contributes nothing to the DFT
//! - `spectrum` is `[frames, SPECTRUM_COLUMNS]`: the real parts of bins 0..=256 in
//!   columns 0..=256, then the imaginary parts of bins 1..=255 in columns 257..=511
//!   (bins 0 and 256 of a real signal have no imaginary part)
//! - `energies` and `features` are `[frames, mel_bins]`, so `features` read per row is
//!   `[rows, frames_per_row, mel_bins]`

use cuda_device::{DisjointSlice, SharedArray, kernel, thread, warp};

/// Samples per analysis frame (25 ms at 16 kHz)
const FRAME_LENGTH: usize = 400;
/// Samples between frame starts (10 ms at 16 kHz)
const FRAME_SHIFT: usize = 160;
/// Real DFT size; frames are zero padded from 400 to 512
const FFT_SIZE: usize = 512;
/// Columns of the spectrum matrix: 257 real parts and 255 imaginary parts
const SPECTRUM_COLUMNS: usize = FFT_SIZE;
/// Warps per block in [`fbank_frame_window`], one frame per warp
const FRAMES_PER_BLOCK: usize = 8;
/// Mel bins per block in [`fbank_log_cmn`]
const CMN_MEL_TILE: usize = 16;
/// Frame groups per block in [`fbank_log_cmn`]; each thread strides over frames
const CMN_FRAME_GROUPS: usize = 16;
/// Threads per block in [`fbank_log_cmn`]
const CMN_THREADS: usize = CMN_MEL_TILE * CMN_FRAME_GROUPS;

/// Frames, mean-normalizes, pre-emphasizes and windows the waveform
///
/// Launch with 256 threads per block and `ceil(frames / 8)` blocks: each warp handles
/// one frame. `waveform` is `[rows, samples_per_row]` in [-1, 1); `frames` holds
/// `rows · frames_per_row` frames of [`FRAME_LENGTH`] samples. For frame sample
/// `x[j] = scale · waveform[...] - mean(x)`, the output is
/// `(x[j] - preemphasis · x[max(j, 1) - 1]) · window[j]`, so the first sample uses
/// itself as its predecessor, like Kaldi's replicate padding
#[kernel]
pub fn fbank_frame_window(
    samples_per_row: u32,
    frames_per_row: u32,
    scale: f32,
    preemphasis: f32,
    waveform: &[f32],
    window: &[f32],
    mut frames: DisjointSlice<f32>,
) {
    let lane = warp::lane_id() as usize;
    let frame =
        thread::blockIdx_x() as usize * FRAMES_PER_BLOCK + thread::threadIdx_x() as usize / 32;
    // the whole warp leaves together, so the shuffles below see all 32 lanes
    if frame >= frames.len() / FRAME_LENGTH {
        return;
    }

    let row = frame / frames_per_row as usize;
    let start = row * samples_per_row as usize + (frame % frames_per_row as usize) * FRAME_SHIFT;

    // PCM16-derived samples scale to integers, so these partial sums stay exact in f32
    let mut sum = 0.0_f32;
    let mut j = lane;
    while j < FRAME_LENGTH {
        sum += waveform[start + j] * scale;
        j += 32;
    }
    let mean = warp::reduce_sum_f32(sum) / FRAME_LENGTH as f32;

    let out_start = frame * FRAME_LENGTH;
    let mut j = lane;
    while j < FRAME_LENGTH {
        let current = waveform[start + j] * scale - mean;
        let previous = if j == 0 {
            current
        } else {
            waveform[start + j - 1] * scale - mean
        };
        let value = (current - preemphasis * previous) * window[j];
        // SAFETY: `frame < frames.len() / FRAME_LENGTH` and `j < FRAME_LENGTH`, so the
        // index is in bounds, and each (frame, j) pair belongs to exactly one lane
        unsafe { *frames.get_unchecked_mut(out_start + j) = value };
        j += 32;
    }
}

/// Squared magnitude of DFT bin `bin` (0..=256) of the frame starting at `base`
#[inline(always)]
fn bin_power(spectrum: &[f32], base: usize, bin: usize) -> f32 {
    let real = spectrum[base + bin];
    let has_imaginary = bin > 0 && bin < FFT_SIZE / 2;
    let imaginary = if has_imaginary {
        spectrum[base + FFT_SIZE / 2 + bin]
    } else {
        0.0
    };
    real * real + imaginary * imaginary
}

/// Power spectrum for the dense mel GEMM: `power` is `[frames, bins]`
///
/// Launch one thread per element of `power`. `bins` is at most 257; the mel GEMM
/// passes 256 because the Nyquist bin has no mel weight
#[kernel]
pub fn fbank_power(bins: u32, spectrum: &[f32], mut power: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    let i = idx.get();
    let bins = bins as usize;
    if let Some(out) = power.get_mut(idx) {
        *out = bin_power(spectrum, (i / bins) * SPECTRUM_COLUMNS, i % bins);
    }
}

/// Mel energies straight from the spectrum, using each mel filter's contiguous run of
/// non-zero weights
///
/// Launch one thread per element of `energies` (`[frames, mel_bins]`, with
/// `mel_bins = mel_first.len()`). Filter `m` covers DFT bins
/// `mel_first[m] .. mel_first[m] + mel_count[m]` with weights
/// `mel_weights[m · mel_width ..]`
#[kernel]
pub fn fbank_mel_sparse(
    mel_width: u32,
    spectrum: &[f32],
    mel_first: &[u32],
    mel_count: &[u32],
    mel_weights: &[f32],
    mut energies: DisjointSlice<f32>,
) {
    let idx = thread::index_1d();
    let i = idx.get();
    let mel_bins = mel_first.len();
    let mel = i % mel_bins;
    let base = (i / mel_bins) * SPECTRUM_COLUMNS;
    let first = mel_first[mel] as usize;
    let weights = mel * mel_width as usize;

    if let Some(out) = energies.get_mut(idx) {
        let mut sum = 0.0_f32;
        for offset in 0..mel_count[mel] as usize {
            sum += mel_weights[weights + offset] * bin_power(spectrum, base, first + offset);
        }
        *out = sum;
    }
}

/// Log of the floored mel energies minus their mean over each row's frames
///
/// Launch with [`CMN_THREADS`] (256) threads per block and
/// `rows · mel_bins / CMN_MEL_TILE` blocks; `mel_bins` must be a multiple of 16.
/// `energies` and `features` are `[rows · frames_per_row, mel_bins]`. The mean is
/// accumulated in f64 so that it does not depend on the summation order
#[kernel]
pub fn fbank_log_cmn(
    frames_per_row: u32,
    mel_bins: u32,
    floor: f32,
    energies: &[f32],
    mut features: DisjointSlice<f32>,
) {
    static mut PARTIAL: SharedArray<f64, CMN_THREADS> = SharedArray::UNINIT;
    static mut MEAN: SharedArray<f32, CMN_MEL_TILE> = SharedArray::UNINIT;

    let frames_per_row = frames_per_row as usize;
    let mel_bins = mel_bins as usize;
    let tiles = mel_bins / CMN_MEL_TILE;
    let block = thread::blockIdx_x() as usize;
    let tid = thread::threadIdx_x() as usize;
    let lane = tid % CMN_MEL_TILE;
    let group = tid / CMN_MEL_TILE;
    let row_start = (block / tiles) * frames_per_row;
    let mel = (block % tiles) * CMN_MEL_TILE + lane;

    let mut sum = 0.0_f64;
    let mut frame = group;
    while frame < frames_per_row {
        let energy = energies[(row_start + frame) * mel_bins + mel];
        sum += log_floor(energy, floor) as f64;
        frame += CMN_FRAME_GROUPS;
    }

    // SAFETY: both statics are this kernel's shared memory, accessed only through
    // these pointers
    let partial = unsafe { SharedArray::as_raw_mut_ptr(&raw mut PARTIAL) };
    let mean_slot = unsafe { SharedArray::as_raw_mut_ptr(&raw mut MEAN) };
    // SAFETY: each thread writes only its own slot, and the barrier below orders the
    // writes before the reads
    unsafe { partial.add(tid).write(sum) };
    thread::sync_threads();

    if group == 0 {
        let mut total = 0.0_f64;
        for other in 0..CMN_FRAME_GROUPS {
            // SAFETY: every slot was written before the barrier above
            total += unsafe { partial.add(other * CMN_MEL_TILE + lane).read() };
        }
        // SAFETY: lane < CMN_MEL_TILE, and only group 0 writes, once per lane
        unsafe {
            mean_slot
                .add(lane)
                .write((total / frames_per_row as f64) as f32)
        };
    }
    thread::sync_threads();

    // SAFETY: written by group 0 before the barrier above
    let mean = unsafe { mean_slot.add(lane).read() };
    let mut frame = group;
    while frame < frames_per_row {
        let index = (row_start + frame) * mel_bins + mel;
        let value = log_floor(energies[index], floor) - mean;
        // SAFETY: `index` was just read from `energies`, which has the same length as
        // `features` (checked by the host), and each (frame, mel) pair belongs to
        // exactly one thread
        unsafe { *features.get_unchecked_mut(index) = value };
        frame += CMN_FRAME_GROUPS;
    }
}

/// `ln(max(energy, floor))`
#[inline(always)]
fn log_floor(energy: f32, floor: f32) -> f32 {
    let clamped = if energy < floor { floor } else { energy };
    clamped.ln()
}
