use ndarray::{Array1, Array3};
use ort::value::TensorRef;

use super::{EmbeddingModel, first_output};
use crate::inference::ModelLoadError;

/// One waveform window prepared once and shared by all of its local speakers
///
/// Split models compute the fbank features here, so a window with several active
/// speakers runs the frontend once
pub struct EmbeddingWindow<'a> {
    audio: &'a [f32],
    features: WindowFeatures,
}

enum WindowFeatures {
    /// The fused model reads the waveform itself
    Waveform,
    /// `[1, frames, 80]` host fbank features for the split tail
    Fbank(Array3<f32>),
}

// a split whose embedding differs by more than this from the fused model on the
// probe window was cut or computed wrongly; float and TF32 noise stay far below it
const MAX_SPLIT_RELATIVE_ERROR: f32 = 1e-2;

impl EmbeddingModel {
    /// Extract a speaker embedding from raw audio with a uniform mask
    pub fn embed(&mut self, audio: &[f32]) -> Result<Array1<f32>, ort::Error> {
        let weights = vec![1.0; self.meta.geometry.mask_frames()];
        let window = self.prepare_window(audio)?;
        self.embed_window(&window, &weights)
    }

    /// Extract a speaker embedding weighted by a segmentation mask
    pub fn embed_masked(
        &mut self,
        audio: &[f32],
        mask: &[f32],
        clean_mask: Option<&[f32]>,
    ) -> Result<Array1<f32>, ort::Error> {
        self.validate_input(audio, mask, clean_mask)?;
        let window = self.prepare_window(audio)?;
        self.embed_window_masked(&window, mask, clean_mask)
    }

    /// Prepare one waveform window for any number of masked embeddings
    pub fn prepare_window<'a>(
        &mut self,
        audio: &'a [f32],
    ) -> Result<EmbeddingWindow<'a>, ort::Error> {
        self.validate_audio_input(audio)?;
        let Some(split) = self.ort.fixed_split.as_mut() else {
            return Ok(EmbeddingWindow {
                audio,
                features: WindowFeatures::Waveform,
            });
        };

        // the fused model sees the window zero-padded to its full length
        Self::prepare_waveform(
            0,
            audio,
            self.meta.geometry.window_samples(),
            &mut self.buffers.waveform_buffer.view_mut(),
        )?;
        let waveform = self
            .buffers
            .waveform_buffer
            .as_slice()
            .ok_or_else(|| ort::Error::new("embedding waveform buffer is not contiguous"))?;
        let mut features = Array3::zeros((1, split.feature_frames, super::FBANK_FEATURES));
        split
            .frontend
            .compute_into(waveform, features.index_axis_mut(ndarray::Axis(0), 0))
            .map_err(|error| ort::Error::new(error.to_string()))?;

        Ok(EmbeddingWindow {
            audio,
            features: WindowFeatures::Fbank(features),
        })
    }

    /// Extract a speaker embedding for one prepared window weighted by a segmentation mask
    pub fn embed_window_masked(
        &mut self,
        window: &EmbeddingWindow<'_>,
        mask: &[f32],
        clean_mask: Option<&[f32]>,
    ) -> Result<Array1<f32>, ort::Error> {
        self.validate_input(window.audio, mask, clean_mask)?;
        let selection = self.select_embedding_mask(mask, clean_mask, window.audio.len());
        let used_mask = selection.mask().ok_or_else(|| {
            ort::Error::new(format!(
                "selected {:?} embedding mask has no active frame after nearest resize to {}",
                selection.source(),
                self.meta.pooling_frames
            ))
        })?;
        self.embed_window(window, used_mask)
    }

    fn embed_window(
        &mut self,
        window: &EmbeddingWindow<'_>,
        weights: &[f32],
    ) -> Result<Array1<f32>, ort::Error> {
        match &window.features {
            WindowFeatures::Waveform => self.run_fused(window.audio, weights),
            WindowFeatures::Fbank(features) => self.run_split_tail(features, weights),
        }
    }

    fn run_fused(&mut self, audio: &[f32], weights: &[f32]) -> Result<Array1<f32>, ort::Error> {
        Self::prepare_waveform(
            0,
            audio,
            self.meta.geometry.window_samples(),
            &mut self.buffers.waveform_buffer.view_mut(),
        )?;
        self.prepare_single_weights(weights)?;

        let waveform_tensor = TensorRef::from_array_view(self.buffers.waveform_buffer.view())?;
        let weights_tensor = TensorRef::from_array_view(self.buffers.weights_buffer.view())?;
        let outputs = self
            .ort
            .session
            .run(ort::inputs!["waveform" => waveform_tensor, "weights" => weights_tensor])?;
        let output = first_output(outputs.values(), "masked embedding output")?;
        let (shape, data) = output.try_extract_tensor::<f32>()?;
        super::tensor::embedding_vector_from_ort_with_width(
            shape,
            data,
            self.meta.geometry.embedding_width(),
            "masked embedding output",
        )
    }

    fn run_split_tail(
        &mut self,
        features: &Array3<f32>,
        weights: &[f32],
    ) -> Result<Array1<f32>, ort::Error> {
        self.prepare_single_weights(weights)?;
        let split =
            self.ort.fixed_split.as_mut().ok_or_else(|| {
                ort::Error::new("window was prepared for a split embedding model")
            })?;

        let features_tensor = TensorRef::from_array_view(features.view())?;
        let weights_tensor = TensorRef::from_array_view(self.buffers.weights_buffer.view())?;
        let outputs = split.tail.run(ort::inputs![
            split.feature_input.as_str() => features_tensor,
            "weights" => weights_tensor
        ])?;
        let output = first_output(outputs.values(), "split embedding output")?;
        let (shape, data) = output.try_extract_tensor::<f32>()?;
        super::tensor::embedding_vector_from_ort_with_width(
            shape,
            data,
            self.meta.geometry.embedding_width(),
            "split embedding output",
        )
    }

    /// Check the split path against the fused model on a fixed probe window
    pub(super) fn verify_fixed_split_parity(&mut self) -> Result<(), ModelLoadError> {
        if self.ort.fixed_split.is_none() {
            return Ok(());
        }

        let audio = parity_probe(self.meta.geometry.window_samples());
        let weights = vec![1.0; self.meta.geometry.mask_frames()];
        let fused = self.run_fused(&audio, &weights)?;
        let window = self.prepare_window(&audio)?;
        let split = self.embed_window(&window, &weights)?;

        let difference = fused
            .iter()
            .zip(&split)
            .map(|(fused, split)| (fused - split).powi(2))
            .sum::<f32>()
            .sqrt();
        let norm = fused.iter().map(|value| value.powi(2)).sum::<f32>().sqrt();
        let relative_error = difference / norm;
        // a NaN error fails this comparison too
        if relative_error <= MAX_SPLIT_RELATIVE_ERROR {
            return Ok(());
        }
        Err(ModelLoadError::FixedSplitParity { relative_error })
    }
}

/// Deterministic speech-band probe: two tones, a chirp, and pseudo-random noise
fn parity_probe(samples: usize) -> Vec<f32> {
    let mut state = 0x2545_f491_u32;
    (0..samples)
        .map(|index| {
            let time = index as f32 / 16_000.0;
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let noise = state as f32 / u32::MAX as f32 - 0.5;
            0.3 * (std::f32::consts::TAU * 220.0 * time).sin()
                + 0.1 * (std::f32::consts::TAU * 1_375.0 * time).sin()
                + 0.05 * (std::f32::consts::TAU * (100.0 + 400.0 * time) * time).sin()
                + 0.02 * noise
        })
        .collect()
}
