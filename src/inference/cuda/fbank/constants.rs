use crate::inference::native_model::fbank::{
    FFT_SIZE, MelTable, SPECTRUM_BINS, hamming_window, mel_filters,
};

pub(super) use crate::inference::native_model::fbank::{
    ENERGY_FLOOR, FBANK_FRAME_LENGTH, FBANK_FRAME_SHIFT, PREEMPHASIS, WAVEFORM_SCALE,
};
pub use crate::inference::native_model::fbank::{
    FBANK_FRAMES, FBANK_MEL_BINS, FBANK_WINDOW_SAMPLES,
};

// the CUDA filterbank must produce exactly the fbank contract the embedding models share
#[cfg(any(feature = "migraphx", feature = "coreml"))]
const _: () = assert!(FBANK_FRAMES == crate::inference::embedding::FBANK_FRAMES);
const _: () = assert!(FBANK_MEL_BINS == crate::inference::embedding::FBANK_FEATURES);

/// Columns of the DFT basis: real parts of bins 0..=256, then imaginary parts of bins
/// 1..=255, which are the only bins of a real signal with an imaginary part
pub(super) const SPECTRUM_COLUMNS: usize = SPECTRUM_BINS + (SPECTRUM_BINS - 2);
/// Spectrum bins the dense mel GEMM reads; the Nyquist bin has no mel weight
pub(super) const MEL_GEMM_BINS: usize = SPECTRUM_BINS - 1;

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

    /// Contiguous filter runs
    ///
    /// The Library's sparse mel kernel and the record-owned DFT producer read the same runs
    pub(in crate::inference::cuda) fn mel_table(&self) -> &MelTable {
        &self.mel_table
    }
}

impl Default for FbankConstants {
    fn default() -> Self {
        Self::new()
    }
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
