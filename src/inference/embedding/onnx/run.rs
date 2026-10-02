use ndarray::{Array1, Array2, ArrayView2, s};
use ort::value::TensorRef;

use crate::inference::InferenceError;

use super::super::buffers::{prepare_waveform, prepare_weights};
use super::super::tensor::{
    array2_from_shape_vec, embedding_batch_from_ort, embedding_vector_from_ort, fbank_hw_from_i64,
    first_output, push_fbank_batch_results,
};
use super::super::{
    CHUNK_SPEAKER_BATCH_SIZE, EMBEDDING_WIDTH, EmbeddingMeta, FBANK_BATCH_SIZE,
    MULTI_MASK_BATCH_SIZE, MaskedEmbeddingInput, NUM_SPEAKERS, PRIMARY_BATCH_SIZE, SplitTailInput,
    select_mask,
};
use super::OrtEmbedding;
use super::fbank_pool::{FbankPoolRoute, fbank_pool_route};

impl OrtEmbedding {
    /// Run the fused waveform-to-embedding model on one window
    pub(in crate::inference::embedding) fn embed_single(
        &mut self,
        meta: &EmbeddingMeta,
        audio: &[f32],
        weights: &[f32],
    ) -> Result<Array1<f32>, InferenceError> {
        let buffers = &mut self.fused_buffers;
        prepare_waveform(
            0,
            audio,
            meta.window_samples,
            &mut buffers.waveform_buffer.view_mut(),
        );
        prepare_weights(
            0,
            weights,
            meta.mask_frames,
            &mut buffers.weights_buffer.view_mut(),
        );

        let waveform_tensor = TensorRef::from_array_view(buffers.waveform_buffer.view())?;
        let weights_tensor = TensorRef::from_array_view(buffers.weights_buffer.view())?;
        let mut session = self.sessions.fused.lock()?;
        let outputs = session
            .run(ort::inputs!["waveform" => waveform_tensor, "weights" => weights_tensor])?;
        let output = first_output(outputs.values(), "masked embedding output")?;
        let (shape, data) = output.try_extract_tensor::<f32>()?;
        embedding_vector_from_ort(shape, data, "masked embedding output")
    }

    /// Run a full batch through the fused batch-64 model when it is loaded
    pub(in crate::inference::embedding) fn try_embed_primary_batch(
        &mut self,
        meta: &EmbeddingMeta,
        inputs: &[MaskedEmbeddingInput<'_>],
    ) -> Result<Option<Array2<f32>>, InferenceError> {
        let Some(session) = self
            .sessions
            .fused_batched
            .as_ref()
            .filter(|_| inputs.len() == PRIMARY_BATCH_SIZE)
        else {
            return Ok(None);
        };

        let buffers = &mut self.fused_buffers;
        for (batch_idx, input) in inputs.iter().enumerate() {
            let used_mask = select_mask(
                input.mask,
                input.clean_mask,
                input.audio.len(),
                meta.min_num_samples,
            );
            prepare_waveform(
                batch_idx,
                input.audio,
                meta.window_samples,
                &mut buffers.primary_batch_waveform_buffer.view_mut(),
            );
            prepare_weights(
                batch_idx,
                used_mask,
                meta.mask_frames,
                &mut buffers.primary_batch_weights_buffer.view_mut(),
            );
        }

        let waveform_tensor =
            TensorRef::from_array_view(buffers.primary_batch_waveform_buffer.view())?;
        let weights_tensor =
            TensorRef::from_array_view(buffers.primary_batch_weights_buffer.view())?;
        let ort_inputs = ort::inputs!["waveform" => waveform_tensor, "weights" => weights_tensor];
        let mut session = session.lock()?;
        let outputs = match &self.primary_batch_run_options {
            Some(options) => session.run_with_options(ort_inputs, options)?,
            None => session.run(ort_inputs)?,
        };
        let output = first_output(outputs.values(), "primary embedding batch output")?;
        let (shape, data) = output.try_extract_tensor::<f32>()?;
        embedding_batch_from_ort(
            shape,
            data,
            PRIMARY_BATCH_SIZE,
            inputs.len(),
            "primary embedding batch output",
        )
        .map(Some)
    }

    pub(in crate::inference::embedding) fn compute_chunk_fbank(
        &mut self,
        meta: &EmbeddingMeta,
        audio: &[f32],
    ) -> Result<Array2<f32>, InferenceError> {
        self.buffers.fill_split_waveform(audio, meta.window_samples);

        let waveform_tensor =
            TensorRef::from_array_view(self.buffers.split_waveform_buffer.view())?;
        let mut session = self
            .sessions
            .fbank
            .as_ref()
            .ok_or(InferenceError::ModelUnavailable {
                model: "split filterbank session",
            })?
            .lock()?;
        let outputs = session.run(ort::inputs!["waveform" => waveform_tensor])?;
        let output = first_output(outputs.values(), "chunk fbank output")?;
        let (shape, data) = output.try_extract_tensor::<f32>()?;
        let (frames, features) = fbank_hw_from_i64(shape, "chunk fbank output")?;
        array2_from_shape_vec(frames, features, data.to_vec(), "chunk fbank output")
    }

    pub(in crate::inference::embedding) fn compute_chunk_fbanks_batch(
        &mut self,
        meta: &EmbeddingMeta,
        audios: &[&[f32]],
    ) -> Result<Vec<Array2<f32>>, InferenceError> {
        let has_batched = self.has_batched_fbank();
        if fbank_pool_route(audios.len(), self.fbank_pool.len(), has_batched)
            == FbankPoolRoute::SessionPool
        {
            return self.fbank_pool.run(audios, meta.window_samples);
        }

        if !has_batched {
            tracing::debug!(
                count = audios.len(),
                "fbank: no batched session, falling back to per-window"
            );
            return audios
                .iter()
                .map(|audio| self.compute_chunk_fbank(meta, audio))
                .collect();
        }

        let mut results = Vec::with_capacity(audios.len());
        for batch in audios.chunks(FBANK_BATCH_SIZE) {
            // the ONNX batched filterbank has a fixed batch, so partial batches run per window
            if batch.len() < FBANK_BATCH_SIZE {
                for audio in batch {
                    results.push(self.compute_chunk_fbank(meta, audio)?);
                }
                continue;
            }

            self.buffers.fill_fbank_batch(batch, meta.window_samples);
            let waveform_tensor =
                TensorRef::from_array_view(self.buffers.split_fbank_batch_buffer.view())?;
            let mut session = self
                .sessions
                .fbank_batched
                .as_ref()
                .ok_or(InferenceError::ModelUnavailable {
                    model: "batched split filterbank session",
                })?
                .lock()?;
            let outputs = session.run(ort::inputs!["waveform" => waveform_tensor])?;
            let output = first_output(outputs.values(), "batched chunk fbank output")?;
            let (shape, data) = output.try_extract_tensor::<f32>()?;
            let (frames, features) = fbank_hw_from_i64(shape, "batched chunk fbank output")?;
            push_fbank_batch_results(&mut results, data, frames, features, batch.len())?;
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

        let feature_slice = self
            .buffers
            .split_feature_batch_buffer
            .slice(s![0..1, .., ..]);
        let weight_slice = self.buffers.split_weights_batch_buffer.slice(s![0..1, ..]);
        let fbank_tensor = TensorRef::from_array_view(feature_slice.view())?;
        let weights_tensor = TensorRef::from_array_view(weight_slice.view())?;
        let mut session = self
            .sessions
            .tail
            .as_ref()
            .ok_or(InferenceError::ModelUnavailable {
                model: "split tail session",
            })?
            .lock()?;
        let outputs =
            session.run(ort::inputs!["fbank" => fbank_tensor, "weights" => weights_tensor])?;
        let output = first_output(outputs.values(), "split tail output")?;
        let (shape, data) = output.try_extract_tensor::<f32>()?;
        embedding_vector_from_ort(shape, data, "split tail output")
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

        let fbank_tensor =
            TensorRef::from_array_view(self.buffers.split_feature_batch_buffer.view())?;
        let weights_tensor =
            TensorRef::from_array_view(self.buffers.split_weights_batch_buffer.view())?;
        let mut session = self
            .sessions
            .tail_batched
            .as_ref()
            .ok_or(InferenceError::ModelUnavailable {
                model: "split tail batched session",
            })?
            .lock()?;
        let outputs =
            session.run(ort::inputs!["fbank" => fbank_tensor, "weights" => weights_tensor])?;
        let output = first_output(outputs.values(), "tail batch output")?;
        let (shape, data) = output.try_extract_tensor::<f32>()?;
        embedding_batch_from_ort(
            shape,
            data,
            CHUNK_SPEAKER_BATCH_SIZE,
            segmentations.ncols(),
            "tail batch output",
        )
    }

    pub(in crate::inference::embedding) fn embed_tail_batch_inputs(
        &mut self,
        meta: &EmbeddingMeta,
        inputs: &[SplitTailInput<'_>],
    ) -> Result<Array2<f32>, InferenceError> {
        self.buffers.fill_tail_primary(inputs, meta.mask_frames)?;

        let fbank_tensor =
            TensorRef::from_array_view(self.buffers.split_primary_feature_batch_buffer.view())?;
        let weights_tensor =
            TensorRef::from_array_view(self.buffers.split_primary_weights_batch_buffer.view())?;
        let mut session = self
            .sessions
            .tail_primary_batched
            .as_ref()
            .ok_or(InferenceError::ModelUnavailable {
                model: "primary tail batched session",
            })?
            .lock()?;
        let outputs =
            session.run(ort::inputs!["fbank" => fbank_tensor, "weights" => weights_tensor])?;
        let output = first_output(outputs.values(), "primary tail batched output")?;
        let (shape, data) = output.try_extract_tensor::<f32>()?;
        embedding_batch_from_ort(
            shape,
            data,
            PRIMARY_BATCH_SIZE,
            inputs.len(),
            "primary tail batched output",
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
        let num_fbanks = fbanks.len();
        let num_masks = masks.len();

        if num_fbanks == MULTI_MASK_BATCH_SIZE
            && let Some(session) = self.sessions.multi_mask_batched.as_ref()
        {
            let fbank_tensor =
                TensorRef::from_array_view(self.buffers.multi_mask_fbank_buffer.view())?;
            let masks_tensor =
                TensorRef::from_array_view(self.buffers.multi_mask_masks_buffer.view())?;
            let mut session = session.lock()?;
            let outputs =
                session.run(ort::inputs!["fbank" => fbank_tensor, "masks" => masks_tensor])?;
            let output = first_output(outputs.values(), "multi-mask batched output")?;
            let (shape, data) = output.try_extract_tensor::<f32>()?;
            return embedding_batch_from_ort(
                shape,
                data,
                MULTI_MASK_BATCH_SIZE * NUM_SPEAKERS,
                num_masks,
                "multi-mask batched output",
            );
        }

        let session =
            self.sessions
                .multi_mask
                .as_ref()
                .ok_or(InferenceError::ModelUnavailable {
                    model: "multi-mask session",
                })?;
        let mut all_embeddings = Array2::<f32>::zeros((num_masks, EMBEDDING_WIDTH));
        for fbank_idx in 0..num_fbanks {
            let fbank_slice =
                self.buffers
                    .multi_mask_fbank_buffer
                    .slice(s![fbank_idx..fbank_idx + 1, .., ..]);
            let mask_start = fbank_idx * NUM_SPEAKERS;
            let mask_end = mask_start + NUM_SPEAKERS;
            let masks_slice = self
                .buffers
                .multi_mask_masks_buffer
                .slice(s![mask_start..mask_end, ..]);
            let fbank_tensor = TensorRef::from_array_view(fbank_slice.view())?;
            let masks_tensor = TensorRef::from_array_view(masks_slice.view())?;
            let mut session = session.lock()?;
            let outputs =
                session.run(ort::inputs!["fbank" => fbank_tensor, "masks" => masks_tensor])?;
            let output = first_output(outputs.values(), "multi-mask output")?;
            let (shape, data) = output.try_extract_tensor::<f32>()?;
            let decoded = embedding_batch_from_ort(
                shape,
                data,
                NUM_SPEAKERS,
                NUM_SPEAKERS,
                "multi-mask output",
            )?;
            for (local_idx, row_idx) in (mask_start..mask_end).enumerate() {
                all_embeddings
                    .row_mut(row_idx)
                    .assign(&decoded.row(local_idx));
            }
        }

        Ok(all_embeddings)
    }
}
