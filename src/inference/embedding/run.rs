use ndarray::Array1;
use ort::value::TensorRef;

use super::{EmbeddingModel, first_output, select_mask};

impl EmbeddingModel {
    /// Extract a speaker embedding from raw audio with a uniform mask
    pub fn embed(&mut self, audio: &[f32]) -> Result<Array1<f32>, ort::Error> {
        let weights = vec![1.0; self.meta.geometry.mask_frames()];
        self.embed_single(audio, &weights)
    }

    /// Extract a speaker embedding weighted by a segmentation mask
    pub fn embed_masked(
        &mut self,
        audio: &[f32],
        mask: &[f32],
        clean_mask: Option<&[f32]>,
    ) -> Result<Array1<f32>, ort::Error> {
        self.validate_input(audio, mask, clean_mask)?;
        let used_mask = select_mask(
            mask,
            clean_mask,
            self.mask_selection_window_samples(audio.len()),
            self.meta.min_num_samples,
        );
        self.embed_single(audio, used_mask)
    }

    fn embed_single(&mut self, audio: &[f32], weights: &[f32]) -> Result<Array1<f32>, ort::Error> {
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
}
