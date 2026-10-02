//! Fixed tensors of the WeSpeaker fbank front end, computed on the host
//!
//! These replace the initializers of `wespeaker-fbank.onnx`, so the native backend needs
//! no weights file for this stage

/// Samples per waveform row: one 10 s window at 16 kHz, the fbank model's input length
pub const FBANK_WINDOW_SAMPLES: usize = 160_000;
/// Samples per analysis frame (25 ms)
pub(super) const FBANK_FRAME_LENGTH: usize = 400;
/// Samples between frame starts (10 ms)
pub(super) const FBANK_FRAME_SHIFT: usize = 160;
/// Frames per waveform row, the time axis of the fbank output
pub const FBANK_FRAMES: usize = (FBANK_WINDOW_SAMPLES - FBANK_FRAME_LENGTH) / FBANK_FRAME_SHIFT + 1;
/// Mel bins per frame, the feature axis of the fbank output
pub const FBANK_MEL_BINS: usize = 80;

// the CUDA filterbank must produce exactly the fbank contract the embedding models share
const _: () = assert!(FBANK_FRAMES == crate::inference::embedding::FBANK_FRAMES);
const _: () = assert!(FBANK_MEL_BINS == crate::inference::embedding::FBANK_FEATURES);

/// Real DFT size; frames are zero padded from 400 to 512 samples
const FFT_SIZE: usize = 512;
/// One-sided spectrum bins, 0 to Nyquist
const SPECTRUM_BINS: usize = FFT_SIZE / 2 + 1;
/// Columns of the DFT basis: real parts of bins 0..=256, then imaginary parts of bins
/// 1..=255, which are the only bins of a real signal with an imaginary part
pub(super) const SPECTRUM_COLUMNS: usize = SPECTRUM_BINS + (SPECTRUM_BINS - 2);
/// Spectrum bins the dense mel GEMM reads; the Nyquist bin has no mel weight
pub(super) const MEL_GEMM_BINS: usize = SPECTRUM_BINS - 1;

/// Converts [-1, 1) samples to the 16-bit integer range Kaldi expects
pub(super) const WAVEFORM_SCALE: f32 = 32_768.0;
/// Pre-emphasis coefficient
pub(super) const PREEMPHASIS: f32 = 0.97;
/// Floor applied to mel energies before the log, `torch.finfo(torch.float32).eps`
pub(super) const ENERGY_FLOOR: f32 = f32::EPSILON;

const SAMPLE_RATE: f64 = 16_000.0;
const LOW_FREQUENCY: f64 = 20.0;

/// The window, DFT basis and mel filters of the fbank front end
///
/// The window is the symmetric Hamming window. The mel filters follow
/// `torchaudio.compliance.kaldi.get_mel_banks(80, 512, 16000, 20, 0, ...)` step by step
/// in f32, as the exported model did. A few roundings still differ from torch's, so 6
/// of the 20560 weights differ from the ONNX `mel` initializer, by at most 7.1e-6; on
/// the reference windows that moves log-mel features by at most 2.5e-5
#[derive(Debug, Clone)]
pub struct FbankConstants {
    window: Vec<f32>,
    dft_basis: Vec<f32>,
    mel: Vec<f32>,
    mel_table: MelTable,
}

/// The mel filters as one contiguous run of non-zero weights per filter
#[derive(Debug, Clone)]
pub(super) struct MelTable {
    /// First DFT bin of each filter
    pub(super) first: Vec<u32>,
    /// Number of DFT bins each filter covers
    pub(super) count: Vec<u32>,
    /// Row stride of [`Self::weights`], the widest filter
    pub(super) width: usize,
    /// `[FBANK_MEL_BINS, width]` weights, zero past each filter's count
    pub(super) weights: Vec<f32>,
}

impl FbankConstants {
    /// Computes every fixed tensor
    pub fn new() -> Self {
        let mel = mel_filters();
        let mel_table = MelTable::new(&mel);
        Self {
            window: hamming_window(),
            dft_basis: dft_basis(),
            mel,
            mel_table,
        }
    }

    /// The symmetric Hamming window, [`FBANK_FRAME_LENGTH`] values
    pub fn window(&self) -> &[f32] {
        &self.window
    }

    /// `[FBANK_FRAME_LENGTH, 512]` row-major: column `k` for `k` in 0..=256 is
    /// `cos(2πkt/512)` and column `256 + k` for `k` in 1..=255 is `sin(2πkt/512)`, so a
    /// frame times the basis gives every real and imaginary part of the one-sided DFT
    pub fn dft_basis(&self) -> &[f32] {
        &self.dft_basis
    }

    /// `[257, FBANK_MEL_BINS]` row-major, the layout of the ONNX `mel` initializer
    pub fn mel(&self) -> &[f32] {
        &self.mel
    }

    pub(super) fn mel_table(&self) -> &MelTable {
        &self.mel_table
    }
}

impl Default for FbankConstants {
    fn default() -> Self {
        Self::new()
    }
}

impl MelTable {
    fn new(mel: &[f32]) -> Self {
        let weight = |bin: usize, filter: usize| mel[bin * FBANK_MEL_BINS + filter];
        let runs: Vec<(usize, usize)> = (0..FBANK_MEL_BINS)
            .map(|filter| {
                let mut bins = (0..SPECTRUM_BINS).filter(|&bin| weight(bin, filter) != 0.0);
                let first = bins.next().unwrap_or(0);
                let last = bins.next_back().unwrap_or(first);
                (first, last + 1 - first)
            })
            .collect();

        let width = runs.iter().map(|&(_, count)| count).max().unwrap_or(0);
        let mut weights = vec![0.0; FBANK_MEL_BINS * width];
        for (filter, &(first, count)) in runs.iter().enumerate() {
            for offset in 0..count {
                weights[filter * width + offset] = weight(first + offset, filter);
            }
        }

        // bins are below 257 and runs at most 257 long, so both fit in u32
        Self {
            first: runs.iter().map(|&(first, _)| first as u32).collect(),
            count: runs.iter().map(|&(_, count)| count as u32).collect(),
            width,
            weights,
        }
    }
}

fn hamming_window() -> Vec<f32> {
    let denominator = (FBANK_FRAME_LENGTH - 1) as f64;
    (0..FBANK_FRAME_LENGTH)
        .map(|n| (0.54 - 0.46 * (std::f64::consts::TAU * n as f64 / denominator).cos()) as f32)
        .collect()
}

fn dft_basis() -> Vec<f32> {
    // reducing k·t modulo the DFT size first keeps the angle exact; computing
    // 2πkt/512 directly in f32 would lose about 1e-4 at the largest products
    let angle = |bin: usize, sample: usize| {
        std::f64::consts::TAU * ((bin * sample) % FFT_SIZE) as f64 / FFT_SIZE as f64
    };

    let mut basis = Vec::with_capacity(FBANK_FRAME_LENGTH * SPECTRUM_COLUMNS);
    for sample in 0..FBANK_FRAME_LENGTH {
        basis.extend((0..SPECTRUM_BINS).map(|bin| angle(bin, sample).cos() as f32));
        basis.extend((1..SPECTRUM_BINS - 1).map(|bin| angle(bin, sample).sin() as f32));
    }

    basis
}

/// `torchaudio.compliance.kaldi.get_mel_banks` without VTLN, transposed and padded with
/// a zero Nyquist row like `FbankWrapper` in `scripts/export_models.py`
///
/// torchaudio computes the scalar mel edges in f64 and everything else in f32; this
/// keeps the same split and operation order
fn mel_filters() -> Vec<f32> {
    let mel_scalar = |frequency: f64| 1127.0 * (1.0 + frequency / 700.0).ln();
    let mel_low = mel_scalar(LOW_FREQUENCY);
    let mel_high = mel_scalar(SAMPLE_RATE / 2.0);
    let delta = ((mel_high - mel_low) / (FBANK_MEL_BINS + 1) as f64) as f32;
    let mel_low = mel_low as f32;
    let bin_width = (SAMPLE_RATE / FFT_SIZE as f64) as f32;

    let mut mel = vec![0.0_f32; SPECTRUM_BINS * FBANK_MEL_BINS];
    for filter in 0..FBANK_MEL_BINS {
        let index = filter as f32;
        let left = mel_low + index * delta;
        let center = mel_low + (index + 1.0) * delta;
        let right = mel_low + (index + 2.0) * delta;

        for bin in 0..SPECTRUM_BINS - 1 {
            let frequency = bin_width * bin as f32;
            // a correctly rounded f32 log, so the filters do not depend on the
            // platform's `logf`
            let bin_mel = 1127.0_f32 * (f64::from(1.0_f32 + frequency / 700.0).ln() as f32);
            let up = (bin_mel - left) / (center - left);
            let down = (right - bin_mel) / (right - center);
            mel[bin * FBANK_MEL_BINS + filter] = up.min(down).max(0.0);
        }
    }

    mel
}
