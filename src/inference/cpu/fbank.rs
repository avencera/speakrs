//! Planned FFT filterbank with immutable recipes and private per-handle scratch

use std::sync::Arc;

use ndarray::ArrayViewMut2;
use rustfft::{Fft, FftPlanner, num_complex::Complex64};

use crate::inference::native_model::fbank::{
    ENERGY_FLOOR, FBANK_FRAME_LENGTH, FBANK_FRAME_SHIFT, FBANK_FRAMES, FBANK_MEL_BINS,
    FBANK_WINDOW_SAMPLES, FFT_SIZE, MelTable, PREEMPHASIS, SPECTRUM_BINS, WAVEFORM_SCALE,
    hamming_window, mel_filters,
};
use crate::inference::{InferenceError, TensorShapeError};

struct Recipe {
    fft: Arc<dyn Fft<f64>>,
    hamming: Vec<f32>,
    mel: MelTable,
}

/// Shared immutable FFT plan and neutral filterbank recipe
#[derive(Clone)]
pub(crate) struct CpuFbank(Arc<Recipe>);

/// Private waveform, FFT and feature scratch for one inference handle
pub(crate) struct FbankWorkspace {
    signal: SignalWorkspace,
}

struct SignalWorkspace {
    audio: Vec<f32>,
    fft: Vec<Complex64>,
    scratch: Vec<Complex64>,
}

impl CpuFbank {
    /// Plans the unnormalized 512-point f64 forward transform once
    pub(crate) fn new() -> Self {
        let mut planner = FftPlanner::<f64>::new();
        Self(Arc::new(Recipe {
            fft: planner.plan_fft_forward(FFT_SIZE),
            hamming: hamming_window(),
            mel: MelTable::new(&mel_filters()),
        }))
    }

    /// Creates independent reusable buffers for this fixed plan
    pub(crate) fn workspace(&self) -> FbankWorkspace {
        FbankWorkspace {
            signal: SignalWorkspace {
                audio: vec![0.0; FBANK_WINDOW_SAMPLES],
                fft: vec![Complex64::default(); FFT_SIZE],
                scratch: vec![Complex64::default(); self.0.fft.get_inplace_scratch_len()],
            },
        }
    }

    /// Writes `[998,80]` caller storage without an intermediate output allocation
    pub(crate) fn compute_into(
        &self,
        audio: &[f32],
        output: ArrayViewMut2<'_, f32>,
        workspace: &mut FbankWorkspace,
    ) -> Result<(), InferenceError> {
        if output.dim() != (FBANK_FRAMES, FBANK_MEL_BINS) {
            return Err(TensorShapeError::ShapeMismatch {
                context: "CPU filterbank output",
                expected: vec![FBANK_FRAMES, FBANK_MEL_BINS],
                actual: output.shape().to_vec(),
            }
            .into());
        }
        self.fill(audio, &mut workspace.signal, output);
        Ok(())
    }

    fn fill(
        &self,
        audio: &[f32],
        workspace: &mut SignalWorkspace,
        mut output: ArrayViewMut2<'_, f32>,
    ) {
        let count = audio.len().min(FBANK_WINDOW_SAMPLES);
        for (destination, source) in workspace.audio[..count].iter_mut().zip(&audio[..count]) {
            *destination = source * WAVEFORM_SCALE;
        }
        workspace.audio[count..].fill(0.0);
        let mut spectrum = [0.0; SPECTRUM_BINS];
        for frame in 0..FBANK_FRAMES {
            let first = frame * FBANK_FRAME_SHIFT;
            let samples = &workspace.audio[first..first + FBANK_FRAME_LENGTH];
            // quiet mel bands are sensitive to cancellation in frame preprocessing
            // and FFT butterflies, so keep both in f64 before the f32 spectrum
            let mean = samples.iter().map(|sample| f64::from(*sample)).sum::<f64>()
                / FBANK_FRAME_LENGTH as f64;
            let mut predecessor = f64::from(samples[0]) - mean;
            for (index, sample) in samples.iter().enumerate() {
                let centered = f64::from(*sample) - mean;
                workspace.fft[index] = Complex64::new(
                    (centered - f64::from(PREEMPHASIS) * predecessor)
                        * f64::from(self.0.hamming[index]),
                    0.0,
                );
                predecessor = centered;
            }
            workspace.fft[FBANK_FRAME_LENGTH..].fill(Complex64::default());
            self.0
                .fft
                .process_with_scratch(&mut workspace.fft, &mut workspace.scratch);
            for (destination, value) in spectrum.iter_mut().zip(&workspace.fft) {
                *destination = (value.re * value.re + value.im * value.im) as f32;
            }
            for bin in 0..FBANK_MEL_BINS {
                let first = self.0.mel.first[bin] as usize;
                let count = self.0.mel.count[bin] as usize;
                let weights =
                    &self.0.mel.weights[bin * self.0.mel.width..bin * self.0.mel.width + count];
                let energy: f32 = spectrum[first..first + count]
                    .iter()
                    .zip(weights)
                    .map(|(value, weight)| value * weight)
                    .sum();
                output[[frame, bin]] = energy.max(ENERGY_FLOOR).ln();
            }
        }
        for mut column in output.columns_mut() {
            // f64 avoids accumulated reduction error across the full padded window
            let mean = (column.iter().map(|value| f64::from(*value)).sum::<f64>()
                / FBANK_FRAMES as f64) as f32;
            column.mapv_inplace(|value| value - mean);
        }
    }
}

#[cfg(test)]
mod tests;
