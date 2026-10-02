use ndarray::{Array1, Array2, ArrayView2, s};

use crate::inference::InferenceError;

use super::super::tensor::{
    array2_from_shape_vec, array2_slice, array3_slice, embedding_batch_from_coreml,
    embedding_vector_from_coreml, fbank_hw_from_shape, push_fbank_batch_results,
};
use super::super::{
    CHUNK_SPEAKER_BATCH_SIZE, EmbeddingMeta, FBANK_BATCH_SIZE, FBANK_FEATURES, FBANK_FRAMES,
    MULTI_MASK_BATCH_SIZE, NUM_SPEAKERS, PRIMARY_BATCH_SIZE, SplitTailInput,
};
use super::{CoreMlEmbedding, load_fbank, load_fbank_batched, load_multi_mask, load_tail};

impl CoreMlEmbedding {
    /// Native filterbank followed by the native single tail
    ///
    /// The fused ONNX model is the same filterbank and ResNet graph, so this is the CoreML
    /// equivalent of the ORT single-window path
    pub(in crate::inference::embedding) fn embed_single(
        &mut self,
        meta: &EmbeddingMeta,
        audio: &[f32],
        weights: &[f32],
    ) -> Result<Array1<f32>, InferenceError> {
        let fbank = self.compute_chunk_fbank(meta, audio)?;
        self.embed_tail_single(meta, &fbank, weights)
    }

    pub(in crate::inference::embedding) fn compute_chunk_fbank(
        &mut self,
        meta: &EmbeddingMeta,
        audio: &[f32],
    ) -> Result<Array2<f32>, InferenceError> {
        self.buffers.fill_split_waveform(audio, meta.window_samples);

        let native = load_fbank(&mut self.fbank)?;
        let input_data = array3_slice(
            &self.buffers.split_waveform_buffer,
            "native chunk fbank input",
        )?;
        let tensor = native.predict_cached(&[(&self.cached_fbank_single_shape, input_data)])?;
        let (frames, features) =
            fbank_hw_from_shape(tensor.layout().dims(), "native chunk fbank output")?;
        array2_from_shape_vec(
            frames,
            features,
            tensor.into_data(),
            "native chunk fbank output",
        )
    }

    pub(in crate::inference::embedding) fn compute_chunk_fbanks_batch(
        &mut self,
        meta: &EmbeddingMeta,
        audios: &[&[f32]],
    ) -> Result<Vec<Array2<f32>>, InferenceError> {
        let mut results = Vec::with_capacity(audios.len());
        for batch in audios.chunks(FBANK_BATCH_SIZE) {
            if let [audio] = batch {
                results.push(self.compute_chunk_fbank(meta, audio)?);
                continue;
            }

            // the native batched filterbank runs zero-padded partial batches
            self.buffers.fill_fbank_batch(batch, meta.window_samples);
            let native = load_fbank_batched(&mut self.fbank_batched)?;
            let input_data = array3_slice(
                &self.buffers.split_fbank_batch_buffer,
                "native batched fbank input",
            )?;
            let tensor = native.predict_cached(&[(&self.cached_fbank_batch_shape, input_data)])?;
            let (frames, features) =
                fbank_hw_from_shape(tensor.layout().dims(), "native batched fbank output")?;
            push_fbank_batch_results(
                &mut results,
                &tensor.into_data(),
                frames,
                features,
                batch.len(),
            )?;
        }

        Ok(results)
    }

    pub(in crate::inference::embedding) fn embed_tail_single(
        &mut self,
        meta: &EmbeddingMeta,
        fbank: &Array2<f32>,
        weights: &[f32],
    ) -> Result<Array1<f32>, InferenceError> {
        self.buffers
            .fill_tail_single(fbank, weights, meta.mask_frames);

        let native = load_tail(
            &mut self.tail,
            self.embedding_compute_units,
            "Lazy loaded native tail",
        )?;
        let feature_slice = self
            .buffers
            .split_feature_batch_buffer
            .slice(s![0..1, .., ..]);
        let weight_slice = self.buffers.split_weights_batch_buffer.slice(s![0..1, ..]);
        let fbank_data = feature_slice
            .as_slice()
            .ok_or(InferenceError::NonContiguousBuffer {
                context: "native tail fbank input",
            })?;
        let weights_data = weight_slice
            .as_slice()
            .ok_or(InferenceError::NonContiguousBuffer {
                context: "native tail weights input",
            })?;
        let tensor = native.predict(&[
            ("fbank", &[1, FBANK_FRAMES, FBANK_FEATURES], fbank_data),
            ("weights", &[1, meta.mask_frames], weights_data),
        ])?;
        embedding_vector_from_coreml(tensor, "native tail output")
    }

    pub(in crate::inference::embedding) fn embed_tail_batch(
        &mut self,
        meta: &EmbeddingMeta,
        fbank: &Array2<f32>,
        segmentations: &ArrayView2<'_, f32>,
        clean_masks: &Array2<f32>,
        num_samples: usize,
    ) -> Result<Array2<f32>, InferenceError> {
        self.buffers.fill_tail_batch(
            fbank,
            segmentations,
            clean_masks,
            num_samples,
            meta.mask_frames,
            meta.min_num_samples,
        )?;

        let native = load_tail(
            &mut self.tail_batched,
            self.embedding_compute_units,
            "Lazy loaded native tail b32",
        )?;
        let fbank_data = array3_slice(
            &self.buffers.split_feature_batch_buffer,
            "native tail batch fbank input",
        )?;
        let weights_data = array2_slice(
            &self.buffers.split_weights_batch_buffer,
            "native tail batch weights input",
        )?;
        let batch = CHUNK_SPEAKER_BATCH_SIZE;
        let tensor = native.predict(&[
            ("fbank", &[batch, FBANK_FRAMES, FBANK_FEATURES], fbank_data),
            ("weights", &[batch, meta.mask_frames], weights_data),
        ])?;
        embedding_batch_from_coreml(
            tensor,
            CHUNK_SPEAKER_BATCH_SIZE,
            segmentations.ncols(),
            "native tail batch output",
        )
    }

    pub(in crate::inference::embedding) fn embed_tail_batch_inputs(
        &mut self,
        meta: &EmbeddingMeta,
        inputs: &[SplitTailInput<'_>],
    ) -> Result<Array2<f32>, InferenceError> {
        self.buffers.fill_tail_primary(inputs, meta.mask_frames)?;

        let slot = self
            .tail_primary_batched
            .as_mut()
            .ok_or(InferenceError::ModelUnavailable {
                model: "native batch-64 embedding tail",
            })?;
        let native = load_tail(
            slot,
            self.embedding_compute_units,
            "Lazy loaded native tail b64",
        )?;
        let fbank_data = array3_slice(
            &self.buffers.split_primary_feature_batch_buffer,
            "native primary tail fbank input",
        )?;
        let weights_data = array2_slice(
            &self.buffers.split_primary_weights_batch_buffer,
            "native primary tail weights input",
        )?;
        let tensor = native.predict_cached(&[
            (&self.cached_tail_fbank_shape, fbank_data),
            (&self.cached_tail_weights_shape, weights_data),
        ])?;
        embedding_batch_from_coreml(
            tensor,
            PRIMARY_BATCH_SIZE,
            inputs.len(),
            "native primary tail output",
        )
    }

    pub(in crate::inference::embedding) fn embed_multi_mask_batch(
        &mut self,
        meta: &EmbeddingMeta,
        fbanks: &[&Array2<f32>],
        masks: &[&[f32]],
    ) -> Result<Array2<f32>, InferenceError> {
        self.buffers
            .fill_multi_mask(fbanks, masks, meta.mask_frames)?;

        let native = load_multi_mask(&mut self.multi_mask, self.embedding_compute_units)?;
        let fbank_data = array3_slice(
            &self.buffers.multi_mask_fbank_buffer,
            "native multi-mask fbank input",
        )?;
        let masks_data = array2_slice(
            &self.buffers.multi_mask_masks_buffer,
            "native multi-mask masks input",
        )?;
        let tensor = native.predict_cached(&[
            (&self.cached_multi_mask_fbank_shape, fbank_data),
            (&self.cached_multi_mask_masks_shape, masks_data),
        ])?;
        embedding_batch_from_coreml(
            tensor,
            MULTI_MASK_BATCH_SIZE * NUM_SPEAKERS,
            masks.len(),
            "native multi-mask output",
        )
    }
}
