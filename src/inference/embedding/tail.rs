use ndarray::{Array1, Array2, ArrayView2, s};
use ort::value::TensorRef;

#[cfg(feature = "coreml")]
use super::tensor::{array2_slice, array3_slice};
use super::*;

impl EmbeddingModel {
    /// Extract per-speaker embeddings for one audio chunk using segmentation masks
    pub fn embed_chunk_speakers(
        &mut self,
        audio: &[f32],
        segmentations: ArrayView2<'_, f32>,
        clean_masks: &Array2<f32>,
    ) -> Result<Array2<f32>, ort::Error> {
        let speaker_count = segmentations.ncols();
        let mut embeddings = Array2::<f32>::zeros((speaker_count, 256));
        if !self.prefers_chunk_embedding_path() {
            for speaker_idx in 0..speaker_count {
                let mask = segmentations.column(speaker_idx).to_owned();
                let clean_mask = clean_masks.column(speaker_idx).to_owned();
                let embedding = self.embed_masked(
                    audio,
                    array1_slice(&mask, "chunk speaker mask")?,
                    Some(array1_slice(&clean_mask, "chunk speaker clean mask")?),
                )?;
                embeddings.row_mut(speaker_idx).assign(&embedding);
            }
            return Ok(embeddings);
        }

        let fbank = self.compute_chunk_fbank(audio)?;
        let has_batched_tail = self.split_tail_batched_session.is_some();
        #[cfg(feature = "coreml")]
        let has_batched_tail = has_batched_tail
            || Self::has_native_tail_model(&self.model_path, self.mode, CHUNK_SPEAKER_BATCH_SIZE);
        if speaker_count == CHUNK_SPEAKER_BATCH_SIZE && has_batched_tail {
            return self.embed_tail_batch(&fbank, &segmentations, clean_masks, audio.len());
        }

        for speaker_idx in 0..speaker_count {
            let mask = segmentations.column(speaker_idx).to_owned();
            let clean_mask = clean_masks.column(speaker_idx).to_owned();
            let used_mask = select_mask(
                array1_slice(&mask, "chunk tail mask")?,
                Some(array1_slice(&clean_mask, "chunk tail clean mask")?),
                audio.len(),
                self.min_num_samples,
            );
            let embedding = self.embed_tail_single(&fbank, used_mask)?;
            embeddings.row_mut(speaker_idx).assign(&embedding);
        }

        Ok(embeddings)
    }

    fn embed_tail_single(
        &mut self,
        fbank: &Array2<f32>,
        weights: &[f32],
    ) -> Result<Array1<f32>, ort::Error> {
        self.split_feature_batch_buffer
            .slice_mut(s![0, ..fbank.nrows(), ..fbank.ncols()])
            .assign(fbank);
        Self::prepare_weights(
            0,
            weights,
            self.mask_frames,
            &mut self.split_weights_batch_buffer.view_mut(),
        );

        #[cfg(feature = "coreml")]
        {
            let _ = self.ensure_native_tail_loaded();
        }
        #[cfg(feature = "coreml")]
        if let Some(native) = self.native_tail_session.as_mut() {
            let feature_slice = self.split_feature_batch_buffer.slice(s![0..1, .., ..]);
            let weight_slice = self.split_weights_batch_buffer.slice(s![0..1, ..]);
            let fbank_data = feature_slice.as_slice().ok_or_else(|| {
                ort::Error::new("native tail fbank input: array view was not contiguous")
            })?;
            let weights_data = weight_slice.as_slice().ok_or_else(|| {
                ort::Error::new("native tail weights input: array view was not contiguous")
            })?;
            let (data, _) = native
                .predict(&[
                    ("fbank", &[1, FBANK_FRAMES, FBANK_FEATURES], fbank_data),
                    ("weights", &[1, self.mask_frames], weights_data),
                ])
                .map_err(|e| ort::Error::new(e.to_string()))?;
            return Ok(Array1::from_vec(data));
        }

        let feature_slice = self.split_feature_batch_buffer.slice(s![0..1, .., ..]);
        let weight_slice = self.split_weights_batch_buffer.slice(s![0..1, ..]);
        let fbank_tensor = TensorRef::from_array_view(feature_slice.view())?;
        let weights_tensor = TensorRef::from_array_view(weight_slice.view())?;
        let outputs = self
            .split_tail_session
            .as_mut()
            .ok_or_else(|| ort::Error::new("missing split tail session"))?
            .run(ort::inputs!["fbank" => fbank_tensor, "weights" => weights_tensor])?;
        let (_shape, data) = outputs[0].try_extract_tensor::<f32>()?;
        Ok(Array1::from_vec(data.to_vec()))
    }

    fn embed_tail_batch(
        &mut self,
        fbank: &Array2<f32>,
        segmentations: &ArrayView2<'_, f32>,
        clean_masks: &Array2<f32>,
        num_samples: usize,
    ) -> Result<Array2<f32>, ort::Error> {
        self.split_feature_batch_buffer
            .slice_mut(s![0, ..fbank.nrows(), ..fbank.ncols()])
            .assign(fbank);
        let row_stride = FBANK_FRAMES * FBANK_FEATURES;
        let fbank_elems = fbank.nrows() * fbank.ncols();
        let buf = array3_slice_mut(
            &mut self.split_feature_batch_buffer,
            "split feature batch buffer",
        )?;
        for speaker_idx in 1..segmentations.ncols() {
            buf.copy_within(0..fbank_elems, speaker_idx * row_stride);
        }

        for speaker_idx in 0..segmentations.ncols() {
            let mask_col = segmentations.column(speaker_idx);
            let clean_col = clean_masks.column(speaker_idx);
            let use_clean = should_use_clean_mask(
                &clean_col,
                mask_col.len(),
                num_samples,
                self.min_num_samples,
            );
            let weights: Vec<f32> = if use_clean {
                clean_col.iter().copied().collect()
            } else {
                mask_col.iter().copied().collect()
            };
            Self::prepare_weights(
                speaker_idx,
                &weights,
                self.mask_frames,
                &mut self.split_weights_batch_buffer.view_mut(),
            );
        }

        #[cfg(feature = "coreml")]
        {
            let _ = self.ensure_native_tail_batched_loaded();
        }
        #[cfg(feature = "coreml")]
        if let Some(native) = self.native_tail_batched_session.as_mut() {
            let fbank_data = array3_slice(
                &self.split_feature_batch_buffer,
                "native tail batch fbank input",
            )?;
            let weights_data = array2_slice(
                &self.split_weights_batch_buffer,
                "native tail batch weights input",
            )?;
            let batch = CHUNK_SPEAKER_BATCH_SIZE;
            let (data, _) = native
                .predict(&[
                    ("fbank", &[batch, FBANK_FRAMES, FBANK_FEATURES], fbank_data),
                    ("weights", &[batch, self.mask_frames], weights_data),
                ])
                .map_err(|e| ort::Error::new(e.to_string()))?;
            return array2_from_shape_vec(
                segmentations.ncols(),
                256,
                data,
                "native tail batch output",
            );
        }

        let fbank_tensor = TensorRef::from_array_view(self.split_feature_batch_buffer.view())?;
        let weights_tensor = TensorRef::from_array_view(self.split_weights_batch_buffer.view())?;
        let outputs = self
            .split_tail_batched_session
            .as_mut()
            .ok_or_else(|| ort::Error::new("missing split tail batched session"))?
            .run(ort::inputs!["fbank" => fbank_tensor, "weights" => weights_tensor])?;
        let (_shape, data) = outputs[0].try_extract_tensor::<f32>()?;
        array2_from_shape_vec(
            segmentations.ncols(),
            256,
            data.to_vec(),
            "tail batch output",
        )
    }
}
