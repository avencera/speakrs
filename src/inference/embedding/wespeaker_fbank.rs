//! Host implementation of the `wespeaker_fbank_v1` log-mel frontend
//!
//! The equations follow `FbankWrapper` in `scripts/export_models.py`, which is the
//! frontend inside the fixed embedding model. The Hamming window and the mel matrix
//! are not recomputed here: they come from the model that the tail is cut from, so
//! both paths share the same constants

use std::sync::Arc;

use ndarray::{ArrayView2, ArrayViewMut2};
use realfft::num_complex::Complex;
use realfft::{RealFftPlanner, RealToComplex};

/// Samples in one analysis frame (25 ms at 16 kHz)
pub(crate) const FRAME_SAMPLES: usize = 400;
/// Samples between consecutive frame starts (10 ms at 16 kHz)
pub(crate) const HOP_SAMPLES: usize = 160;
/// Real FFT length after zero padding one frame
pub(crate) const FFT_SAMPLES: usize = 512;
/// One-sided spectrum bins of one padded frame
pub(crate) const SPECTRUM_BINS: usize = FFT_SAMPLES / 2 + 1;
/// Mel bands in one feature frame
pub(crate) const MEL_BANDS: usize = 80;
// float audio is scaled to the 16-bit integer range before framing, as in Kaldi
const WAVEFORM_SCALE: f32 = 32_768.0;
const PREEMPHASIS: f32 = 0.97;

/// Errors raised while building or running the host fbank frontend
#[derive(Debug, thiserror::Error)]
pub enum FbankFrontendError {
    /// The analysis window does not cover one frame
    #[error("fbank window has {actual} samples, expected {FRAME_SAMPLES}")]
    WindowLength {
        /// Samples in the supplied window
        actual: usize,
    },
    /// The mel matrix does not map the one-sided spectrum to the mel bands
    #[error("fbank mel matrix has shape {actual:?}, expected [{SPECTRUM_BINS}, {MEL_BANDS}]")]
    MelShape {
        /// Shape of the supplied mel matrix
        actual: [usize; 2],
    },
    /// A frontend constant is NaN or infinite
    #[error("fbank {constant} contains a non-finite value")]
    NonFiniteConstant {
        /// Name of the constant
        constant: &'static str,
    },
    /// The waveform is shorter than one analysis frame
    #[error("fbank input has {samples} samples, fewer than one {FRAME_SAMPLES}-sample frame")]
    TooShort {
        /// Samples in the waveform
        samples: usize,
    },
    /// The output buffer does not match the frame count of the waveform
    #[error("fbank output has shape {actual:?}, expected {expected:?}")]
    OutputShape {
        /// Frames and bands implied by the waveform
        expected: [usize; 2],
        /// Shape of the supplied output buffer
        actual: [usize; 2],
    },
    /// The FFT rejected its buffers
    #[error("fbank FFT failed: {0}")]
    Fft(String),
}

/// Nonzero span of one triangular mel filter over the spectrum bins
#[derive(Debug)]
struct MelBand {
    first_bin: usize,
    weights: Box<[f32]>,
}

/// Log-mel frontend that turns one waveform window into mean-normalized fbank frames
pub(crate) struct WeSpeakerFbank {
    window: Box<[f32]>,
    bands: Box<[MelBand]>,
    fft: Arc<dyn RealToComplex<f32>>,
    frame: Vec<f32>,
    spectrum: Vec<Complex<f32>>,
    scratch: Vec<Complex<f32>>,
    power: Vec<f32>,
}

impl WeSpeakerFbank {
    /// Build the frontend from the model's Hamming window and `[257, 80]` mel matrix
    pub(crate) fn new(
        window: &[f32],
        mel: ArrayView2<'_, f32>,
    ) -> Result<Self, FbankFrontendError> {
        if window.len() != FRAME_SAMPLES {
            return Err(FbankFrontendError::WindowLength {
                actual: window.len(),
            });
        }
        if mel.dim() != (SPECTRUM_BINS, MEL_BANDS) {
            return Err(FbankFrontendError::MelShape {
                actual: [mel.nrows(), mel.ncols()],
            });
        }
        if !window.iter().all(|value| value.is_finite()) {
            return Err(FbankFrontendError::NonFiniteConstant { constant: "window" });
        }
        if !mel.iter().all(|value| value.is_finite()) {
            return Err(FbankFrontendError::NonFiniteConstant { constant: "mel" });
        }

        let bands = mel.columns().into_iter().map(mel_band).collect();
        let fft = RealFftPlanner::<f32>::new().plan_fft_forward(FFT_SAMPLES);
        let scratch = fft.make_scratch_vec();
        Ok(Self {
            window: window.into(),
            bands,
            fft,
            frame: vec![0.0; FFT_SAMPLES],
            spectrum: vec![Complex::default(); SPECTRUM_BINS],
            scratch,
            power: vec![0.0; SPECTRUM_BINS],
        })
    }

    /// Number of snip-edges frames for a waveform length
    pub(crate) const fn frame_count(samples: usize) -> usize {
        if samples < FRAME_SAMPLES {
            return 0;
        }
        (samples - FRAME_SAMPLES) / HOP_SAMPLES + 1
    }

    /// Write `[frames, 80]` mean-normalized log-mel features of `waveform` into `out`
    pub(crate) fn compute_into(
        &mut self,
        waveform: &[f32],
        mut out: ArrayViewMut2<'_, f32>,
    ) -> Result<(), FbankFrontendError> {
        let frames = Self::frame_count(waveform.len());
        if frames == 0 {
            return Err(FbankFrontendError::TooShort {
                samples: waveform.len(),
            });
        }
        if out.dim() != (frames, MEL_BANDS) {
            return Err(FbankFrontendError::OutputShape {
                expected: [frames, MEL_BANDS],
                actual: [out.nrows(), out.ncols()],
            });
        }

        for (index, mut row) in out.rows_mut().into_iter().enumerate() {
            let start = index * HOP_SAMPLES;
            self.power_spectrum(&waveform[start..start + FRAME_SAMPLES])?;
            for (value, band) in row.iter_mut().zip(self.bands.iter()) {
                let power = &self.power[band.first_bin..band.first_bin + band.weights.len()];
                let energy = power
                    .iter()
                    .zip(band.weights.iter())
                    .fold(0.0_f32, |sum, (power, weight)| sum + power * weight);
                *value = energy.max(f32::EPSILON).ln();
            }
        }

        // cepstral mean normalization over the whole window, per mel band
        let scale = frames as f32;
        for mut column in out.columns_mut() {
            let mean = column.iter().sum::<f32>() / scale;
            column.iter_mut().for_each(|value| *value -= mean);
        }
        Ok(())
    }

    fn power_spectrum(&mut self, samples: &[f32]) -> Result<(), FbankFrontendError> {
        let frame = &mut self.frame[..FRAME_SAMPLES];
        for (scaled, sample) in frame.iter_mut().zip(samples) {
            *scaled = sample * WAVEFORM_SCALE;
        }
        let mean = frame.iter().sum::<f32>() / FRAME_SAMPLES as f32;
        frame.iter_mut().for_each(|value| *value -= mean);

        // walk backwards so each sample still reads its unemphasized predecessor;
        // the first sample uses itself, as the replicate padding does in the model
        for index in (1..FRAME_SAMPLES).rev() {
            frame[index] -= PREEMPHASIS * frame[index - 1];
        }
        frame[0] -= PREEMPHASIS * frame[0];
        for (value, weight) in frame.iter_mut().zip(self.window.iter()) {
            *value *= weight;
        }
        self.frame[FRAME_SAMPLES..].fill(0.0);

        self.fft
            .process_with_scratch(&mut self.frame, &mut self.spectrum, &mut self.scratch)
            .map_err(|error| FbankFrontendError::Fft(error.to_string()))?;
        for (power, bin) in self.power.iter_mut().zip(&self.spectrum) {
            // the model takes the L2 norm and then squares it
            let magnitude = (bin.re * bin.re + bin.im * bin.im).sqrt();
            *power = magnitude * magnitude;
        }
        Ok(())
    }
}

fn mel_band(column: ndarray::ArrayView1<'_, f32>) -> MelBand {
    let first_bin = column.iter().position(|weight| *weight != 0.0);
    let last_bin = column.iter().rposition(|weight| *weight != 0.0);
    let (Some(first_bin), Some(last_bin)) = (first_bin, last_bin) else {
        return MelBand {
            first_bin: 0,
            weights: Box::default(),
        };
    };

    MelBand {
        first_bin,
        weights: column
            .slice(ndarray::s![first_bin..=last_bin])
            .iter()
            .copied()
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use ndarray::{Array1, Array2};
    use ndarray_npy::read_npy;

    use super::*;

    fn fixture<T: ndarray_npy::ReadableElement, D: ndarray::Dimension>(
        name: &str,
    ) -> ndarray::Array<T, D>
    where
        ndarray::Array<T, D>: ndarray_npy::ReadNpyExt,
    {
        read_npy(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("fixtures")
                .join(name),
        )
        .unwrap_or_else(|error| panic!("failed to read fixture {name}: {error}"))
    }

    fn frontend() -> WeSpeakerFbank {
        let window: Array1<f32> = fixture("wespeaker_fbank_window.npy");
        let mel: Array2<f32> = fixture("wespeaker_fbank_mel.npy");
        WeSpeakerFbank::new(window.as_slice().unwrap(), mel.view()).unwrap()
    }

    #[test]
    fn matches_the_pytorch_frontend() {
        let waveform: Array1<f32> = fixture("wespeaker_fbank_input.npy");
        let expected: Array2<f32> = fixture("wespeaker_fbank_expected.npy");
        let mut fbank = frontend();
        let mut actual = Array2::zeros(expected.dim());

        fbank
            .compute_into(waveform.as_slice().unwrap(), actual.view_mut())
            .unwrap();

        let max_error = actual
            .iter()
            .zip(&expected)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0_f32, f32::max);
        // FFT and reduction order differ from PyTorch, so values agree to float
        // round-off on log energies of order ten
        assert!(max_error < 2e-4, "max fbank error {max_error}");
    }

    #[test]
    fn repeated_calls_do_not_leak_state() {
        let waveform: Array1<f32> = fixture("wespeaker_fbank_input.npy");
        let frames = WeSpeakerFbank::frame_count(waveform.len());
        let mut fbank = frontend();
        let mut first = Array2::zeros((frames, MEL_BANDS));
        let mut second = Array2::zeros((frames, MEL_BANDS));

        fbank
            .compute_into(waveform.as_slice().unwrap(), first.view_mut())
            .unwrap();
        fbank
            .compute_into(&vec![0.25; waveform.len()], second.view_mut())
            .unwrap();
        fbank
            .compute_into(waveform.as_slice().unwrap(), second.view_mut())
            .unwrap();

        assert_eq!(first, second);
    }

    #[test]
    fn frame_count_uses_snip_edges() {
        assert_eq!(WeSpeakerFbank::frame_count(399), 0);
        assert_eq!(WeSpeakerFbank::frame_count(400), 1);
        assert_eq!(WeSpeakerFbank::frame_count(559), 1);
        assert_eq!(WeSpeakerFbank::frame_count(560), 2);
        assert_eq!(WeSpeakerFbank::frame_count(128_000), 798);
    }

    #[test]
    fn rejects_bad_constants_and_buffers() {
        let mel: Array2<f32> = fixture("wespeaker_fbank_mel.npy");
        assert!(matches!(
            WeSpeakerFbank::new(&[1.0; 399], mel.view()),
            Err(FbankFrontendError::WindowLength { actual: 399 })
        ));
        assert!(matches!(
            WeSpeakerFbank::new(&[1.0; FRAME_SAMPLES], mel.t()),
            Err(FbankFrontendError::MelShape { .. })
        ));
        assert!(matches!(
            WeSpeakerFbank::new(&[f32::NAN; FRAME_SAMPLES], mel.view()),
            Err(FbankFrontendError::NonFiniteConstant { constant: "window" })
        ));

        let mut fbank = frontend();
        let mut out = Array2::zeros((2, MEL_BANDS));
        assert!(matches!(
            fbank.compute_into(&[0.0; 399], out.view_mut()),
            Err(FbankFrontendError::TooShort { samples: 399 })
        ));
        assert!(matches!(
            fbank.compute_into(&[0.0; 400], out.view_mut()),
            Err(FbankFrontendError::OutputShape { .. })
        ));
    }
}
