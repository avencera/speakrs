//! Fixed tensors of the WeSpeaker fbank front end, computed on the host
//!
//! These replace the initializers of `wespeaker-fbank.onnx`, so the native backend needs
//! no weights file for this stage

/// Samples per waveform row: one 10 s window at 16 kHz, the fbank model's input length
pub const FBANK_WINDOW_SAMPLES: usize = 160_000;
/// Samples per analysis frame (25 ms)
pub(crate) const FBANK_FRAME_LENGTH: usize = 400;
/// Samples between frame starts (10 ms)
pub(crate) const FBANK_FRAME_SHIFT: usize = 160;
/// Frames per waveform row, the time axis of the fbank output
pub const FBANK_FRAMES: usize = (FBANK_WINDOW_SAMPLES - FBANK_FRAME_LENGTH) / FBANK_FRAME_SHIFT + 1;
/// Mel bins per frame, the feature axis of the fbank output
pub const FBANK_MEL_BINS: usize = 80;

/// Real DFT size; frames are zero padded from 400 to 512 samples
pub(crate) const FFT_SIZE: usize = 512;
/// One-sided spectrum bins, 0 to Nyquist
pub(crate) const SPECTRUM_BINS: usize = FFT_SIZE / 2 + 1;
/// Converts [-1, 1) samples to the 16-bit integer range Kaldi expects
pub(crate) const WAVEFORM_SCALE: f32 = 32_768.0;
/// Pre-emphasis coefficient
pub(crate) const PREEMPHASIS: f32 = 0.97;
/// Floor applied to mel energies before the log, `torch.finfo(torch.float32).eps`
pub(crate) const ENERGY_FLOOR: f32 = f32::EPSILON;

const SAMPLE_RATE: f64 = 16_000.0;
const LOW_FREQUENCY: f64 = 20.0;

/// The mel filters as one contiguous run of non-zero weights per filter
#[derive(Debug, Clone)]
pub(crate) struct MelTable {
    /// First DFT bin of each filter
    pub(crate) first: Vec<u32>,
    /// Number of DFT bins each filter covers
    pub(crate) count: Vec<u32>,
    /// Row stride of [`Self::weights`], the widest filter
    pub(crate) width: usize,
    /// `[FBANK_MEL_BINS, width]` weights, zero past each filter's count
    pub(crate) weights: Vec<f32>,
}

impl MelTable {
    /// Builds the shared sparse layout from the fixed dense mel recipe
    pub(crate) fn new(mel: &[f32]) -> Self {
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

/// Computes the symmetric 400-sample Hamming window
pub(crate) fn hamming_window() -> Vec<f32> {
    let denominator = (FBANK_FRAME_LENGTH - 1) as f64;
    (0..FBANK_FRAME_LENGTH)
        .map(|n| (0.54 - 0.46 * (std::f64::consts::TAU * n as f64 / denominator).cos()) as f32)
        .collect()
}

/// `torchaudio.compliance.kaldi.get_mel_banks` without VTLN, transposed and padded with
/// a zero Nyquist row like `FbankWrapper` in `scripts/export_models.py`
///
/// torchaudio computes the scalar mel edges in f64 and everything else in f32; this
/// keeps the same split and operation order
pub(crate) fn mel_filters() -> Vec<f32> {
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

#[cfg(test)]
mod tests {
    use super::{FBANK_MEL_BINS, MelTable, SPECTRUM_BINS, mel_filters};

    #[test]
    fn sparse_mel_runs_reproduce_dense_filter_energy() {
        let mel = mel_filters();
        let table = MelTable::new(&mel);
        let spectrum: Vec<f32> = (0..SPECTRUM_BINS)
            .map(|bin| ((bin * 17 + 3) % 101) as f32)
            .collect();
        for filter in 0..FBANK_MEL_BINS {
            let dense: f32 = spectrum
                .iter()
                .enumerate()
                .map(|(bin, value)| value * mel[bin * FBANK_MEL_BINS + filter])
                .sum();
            let first = table.first[filter] as usize;
            let count = table.count[filter] as usize;
            let sparse: f32 = (0..count)
                .map(|offset| {
                    spectrum[first + offset] * table.weights[filter * table.width + offset]
                })
                .sum();
            assert_eq!(dense.to_bits(), sparse.to_bits(), "filter={filter}");
            assert!(
                table.weights[filter * table.width + count..(filter + 1) * table.width]
                    .iter()
                    .all(|value| *value == 0.0)
            );
        }
    }
}
