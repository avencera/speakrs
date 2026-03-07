use std::fs;
use std::path::Path;

use ndarray::Array1;
use ort::session::Session;
use ort::value::Tensor;

pub struct EmbeddingModel {
    session: Session,
    sample_rate: usize,
    window_samples: usize,
    mask_frames: usize,
    min_num_samples: usize,
}

impl EmbeddingModel {
    /// Load the fixed-shape WeSpeaker embedding model
    pub fn new(model_path: &str) -> Result<Self, ort::Error> {
        let session = Session::builder()?.commit_from_file(model_path)?;
        let metadata_path = Path::new(model_path)
            .with_extension("min_num_samples.txt")
            .to_string_lossy()
            .into_owned();

        Ok(Self {
            session,
            sample_rate: 16_000,
            window_samples: 160_000,
            mask_frames: 589,
            min_num_samples: read_min_num_samples(&metadata_path).unwrap_or(400),
        })
    }

    pub fn sample_rate(&self) -> usize {
        self.sample_rate
    }

    pub fn min_num_samples(&self) -> usize {
        self.min_num_samples
    }

    pub fn embed(&mut self, audio: &[f32]) -> Result<Array1<f32>, ort::Error> {
        self.embed_waveform(audio, &Array1::ones(self.mask_frames))
    }

    pub fn embed_masked(
        &mut self,
        audio: &[f32],
        mask: &[f32],
        clean_mask: Option<&[f32]>,
    ) -> Result<Array1<f32>, ort::Error> {
        let used_mask = select_mask(mask, clean_mask, audio.len(), self.min_num_samples);
        self.embed_waveform(audio, &Array1::from_vec(used_mask.to_vec()))
    }

    fn embed_waveform(
        &mut self,
        audio: &[f32],
        weights: &Array1<f32>,
    ) -> Result<Array1<f32>, ort::Error> {
        let waveform = self.prepare_waveform(audio);
        let waveform_tensor = Tensor::from_array(
            waveform
                .view()
                .insert_axis(ndarray::Axis(0))
                .insert_axis(ndarray::Axis(0))
                .to_owned(),
        )?;
        let weights_tensor = Tensor::from_array(
            self.prepare_weights(weights)
                .view()
                .insert_axis(ndarray::Axis(0))
                .to_owned(),
        )?;

        let outputs = self
            .session
            .run(ort::inputs!["waveform" => waveform_tensor, "weights" => weights_tensor])?;
        let (_shape, data) = outputs[0].try_extract_tensor::<f32>()?;
        Ok(Array1::from_vec(data.to_vec()))
    }

    fn prepare_waveform(&self, audio: &[f32]) -> Array1<f32> {
        let mut waveform = vec![0.0f32; self.window_samples];
        let copy_len = audio.len().min(self.window_samples);
        waveform[..copy_len].copy_from_slice(&audio[..copy_len]);
        Array1::from_vec(waveform)
    }

    fn prepare_weights(&self, weights: &Array1<f32>) -> Array1<f32> {
        if weights.len() == self.mask_frames {
            return weights.clone();
        }

        let mut padded = vec![0.0f32; self.mask_frames];
        let copy_len = weights.len().min(self.mask_frames);
        if let Some(values) = weights.as_slice() {
            padded[..copy_len].copy_from_slice(&values[..copy_len]);
        }
        Array1::from_vec(padded)
    }
}

fn read_min_num_samples(path: &str) -> Option<usize> {
    fs::read_to_string(path).ok()?.trim().parse().ok()
}

fn select_mask<'a>(
    mask: &'a [f32],
    clean_mask: Option<&'a [f32]>,
    num_samples: usize,
    min_num_samples: usize,
) -> &'a [f32] {
    let Some(clean_mask) = clean_mask else {
        return mask;
    };

    if clean_mask.len() != mask.len() || num_samples == 0 {
        return mask;
    }

    let min_mask_frames = (mask.len() * min_num_samples).div_ceil(num_samples) as f32;
    let clean_weight: f32 = clean_mask.iter().copied().sum();
    if clean_weight > min_mask_frames {
        clean_mask
    } else {
        mask
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn select_mask_prefers_clean_mask_when_it_is_long_enough() {
        let mask = [1.0, 1.0, 1.0, 0.0];
        let clean = [1.0, 1.0, 1.0, 0.0];

        let selected = select_mask(&mask, Some(&clean), 16_000, 6_000);

        assert_eq!(selected, clean);
    }

    #[test]
    fn select_mask_falls_back_to_full_mask_when_clean_mask_is_too_short() {
        let mask = [1.0, 1.0, 1.0, 0.0];
        let clean = [1.0, 0.0, 0.0, 0.0];

        let selected = select_mask(&mask, Some(&clean), 16_000, 6_000);

        assert_eq!(selected, mask);
    }
}
