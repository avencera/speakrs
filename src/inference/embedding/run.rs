use ndarray::{Array1, s};
use ort::value::TensorRef;

use super::{EmbeddingModel, select_mask};

impl EmbeddingModel {
    /// Extract a speaker embedding from raw audio with a uniform mask
    pub fn embed(&mut self, audio: &[f32]) -> Result<Array1<f32>, ort::Error> {
        let weights = vec![1.0; self.meta.mask_frames];
        self.embed_single(audio, &weights)
    }

    /// Extract a speaker embedding weighted by a segmentation mask
    pub fn embed_masked(
        &mut self,
        audio: &[f32],
        mask: &[f32],
        clean_mask: Option<&[f32]>,
    ) -> Result<Array1<f32>, ort::Error> {
        let used_mask = select_mask(mask, clean_mask, audio.len(), self.meta.min_num_samples);
        self.embed_single(audio, used_mask)
    }

    fn embed_single(&mut self, audio: &[f32], weights: &[f32]) -> Result<Array1<f32>, ort::Error> {
        let copy_len = audio.len().min(self.meta.window_samples);
        self.buffers
            .waveform_buffer
            .slice_mut(s![0, 0, ..copy_len])
            .assign(&ndarray::ArrayView1::from(&audio[..copy_len]));
        if copy_len < self.meta.window_samples {
            self.buffers
                .waveform_buffer
                .slice_mut(s![0, 0, copy_len..])
                .fill(0.0);
        }
        self.prepare_single_weights(weights);

        let waveform_tensor = TensorRef::from_array_view(self.buffers.waveform_buffer.view())?;
        let weights_tensor = TensorRef::from_array_view(self.buffers.weights_buffer.view())?;
        let outputs = self
            .ort
            .session
            .run(ort::inputs!["waveform" => waveform_tensor, "weights" => weights_tensor])?;
        let (_shape, data) = outputs[0].try_extract_tensor::<f32>()?;
        Ok(Array1::from_vec(data.to_vec()))
    }
}
