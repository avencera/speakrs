use ndarray::{Array2, s};
use ort::value::TensorRef;

#[cfg(feature = "coreml")]
use super::tensor::{array2_slice, array3_slice};
use super::*;

impl EmbeddingModel {
    /// Extract speaker embeddings for a batch of masked audio windows
    pub fn embed_batch(
        &mut self,
        inputs: &[MaskedEmbeddingInput<'_>],
    ) -> Result<Array2<f32>, ort::Error> {
        if let Some(sess) = self
            .primary_batched_session
            .as_mut()
            .filter(|_| inputs.len() == PRIMARY_BATCH_SIZE)
        {
            for (batch_idx, input) in inputs.iter().enumerate() {
                let used_mask = select_mask(
                    input.mask,
                    input.clean_mask,
                    input.audio.len(),
                    self.min_num_samples,
                );
                Self::prepare_waveform(
                    batch_idx,
                    input.audio,
                    self.window_samples,
                    &mut self.primary_batch_waveform_buffer.view_mut(),
                );
                Self::prepare_weights(
                    batch_idx,
                    used_mask,
                    self.mask_frames,
                    &mut self.primary_batch_weights_buffer.view_mut(),
                );
            }

            let waveform_tensor =
                TensorRef::from_array_view(self.primary_batch_waveform_buffer.view())?;
            let weights_tensor =
                TensorRef::from_array_view(self.primary_batch_weights_buffer.view())?;
            let ort_inputs =
                ort::inputs!["waveform" => waveform_tensor, "weights" => weights_tensor];
            let outputs = if let Some(opts) = &self.primary_batch_run_options {
                sess.run_with_options(ort_inputs, opts)?
            } else {
                sess.run(ort_inputs)?
            };
            let (_shape, data) = outputs[0].try_extract_tensor::<f32>()?;
            let n = inputs.len();
            let mut result = Array2::<f32>::zeros((n, 256));
            array2_slice_mut(&mut result, "batched embedding output")?
                .copy_from_slice(&data[..n * 256]);
            return Ok(result);
        }

        let mut stacked = Array2::<f32>::zeros((inputs.len(), 256));
        for (idx, input) in inputs.iter().enumerate() {
            let embedding = self.embed_masked(input.audio, input.mask, input.clean_mask)?;
            stacked.row_mut(idx).assign(&embedding);
        }
        Ok(stacked)
    }

    pub(crate) fn embed_multi_mask_batch(
        &mut self,
        fbanks: &[&Array2<f32>],
        masks: &[&[f32]],
    ) -> Result<Array2<f32>, ort::Error> {
        let num_fbanks = fbanks.len();
        let num_masks = masks.len();
        debug_assert_eq!(num_masks, num_fbanks * NUM_SPEAKERS);
        debug_assert!(num_fbanks <= MULTI_MASK_BATCH_SIZE);

        let fbank_row_stride = FBANK_FRAMES * FBANK_FEATURES;
        for (idx, fbank) in fbanks.iter().enumerate() {
            self.multi_mask_fbank_buffer
                .slice_mut(s![idx, ..fbank.nrows(), ..fbank.ncols()])
                .assign(fbank);
        }

        for (idx, mask) in masks.iter().enumerate() {
            Self::prepare_weights(
                idx,
                mask,
                self.mask_frames,
                &mut self.multi_mask_masks_buffer.view_mut(),
            );
        }
        if num_fbanks < MULTI_MASK_BATCH_SIZE {
            let start = num_fbanks * fbank_row_stride;
            let buf = array3_slice_mut(
                &mut self.multi_mask_fbank_buffer,
                "multi-mask fbank scratch buffer",
            )?;
            buf[start..].fill(0.0);
        }
        if num_masks < MULTI_MASK_BATCH_SIZE * NUM_SPEAKERS {
            self.multi_mask_masks_buffer
                .slice_mut(s![num_masks.., ..])
                .fill(0.0);
        }

        let full_mask_batch = MULTI_MASK_BATCH_SIZE * NUM_SPEAKERS;

        #[cfg(feature = "coreml")]
        {
            let _ = self.ensure_native_multi_mask_loaded();
        }
        #[cfg(feature = "coreml")]
        if let Some(native) = self.native_multi_mask_session.as_ref() {
            let fbank_data = array3_slice(
                &self.multi_mask_fbank_buffer,
                "native multi-mask fbank input",
            )?;
            let masks_data = array2_slice(
                &self.multi_mask_masks_buffer,
                "native multi-mask masks input",
            )?;
            let (data, _) = native
                .predict_cached(&[
                    (&self.cached_multi_mask_fbank_shape, fbank_data),
                    (&self.cached_multi_mask_masks_shape, masks_data),
                ])
                .map_err(|e| ort::Error::new(e.to_string()))?;
            let batch =
                array2_from_shape_vec(full_mask_batch, 256, data, "native multi-mask output")?;
            return Ok(batch.slice(s![0..num_masks, ..]).to_owned());
        }

        let use_batched =
            num_fbanks == MULTI_MASK_BATCH_SIZE && self.multi_mask_batched_session.is_some();

        if use_batched {
            let fbank_tensor = TensorRef::from_array_view(self.multi_mask_fbank_buffer.view())?;
            let masks_tensor = TensorRef::from_array_view(self.multi_mask_masks_buffer.view())?;
            let outputs = self
                .multi_mask_batched_session
                .as_mut()
                .ok_or_else(|| ort::Error::new("missing multi-mask batched session"))?
                .run(ort::inputs!["fbank" => fbank_tensor, "masks" => masks_tensor])?;
            let (_shape, data) = outputs[0].try_extract_tensor::<f32>()?;
            let batch = array2_from_shape_vec(
                full_mask_batch,
                256,
                data.to_vec(),
                "multi-mask batched output",
            )?;
            Ok(batch.slice(s![0..num_masks, ..]).to_owned())
        } else {
            let mut all_embeddings = Array2::<f32>::zeros((num_masks, 256));
            for fbank_idx in 0..num_fbanks {
                let fbank_slice =
                    self.multi_mask_fbank_buffer
                        .slice(s![fbank_idx..fbank_idx + 1, .., ..]);
                let mask_start = fbank_idx * NUM_SPEAKERS;
                let mask_end = mask_start + NUM_SPEAKERS;
                let masks_slice = self
                    .multi_mask_masks_buffer
                    .slice(s![mask_start..mask_end, ..]);
                let fbank_tensor = TensorRef::from_array_view(fbank_slice.view())?;
                let masks_tensor = TensorRef::from_array_view(masks_slice.view())?;
                let outputs = self
                    .multi_mask_session
                    .as_mut()
                    .ok_or_else(|| ort::Error::new("missing multi-mask session"))?
                    .run(ort::inputs!["fbank" => fbank_tensor, "masks" => masks_tensor])?;
                let (_shape, data) = outputs[0].try_extract_tensor::<f32>()?;
                for (local_idx, row_idx) in (mask_start..mask_end).enumerate() {
                    let start = local_idx * 256;
                    all_embeddings
                        .row_mut(row_idx)
                        .assign(&ndarray::ArrayView1::from(&data[start..start + 256]));
                }
            }
            Ok(all_embeddings)
        }
    }

    pub(crate) fn embed_tail_batch_inputs(
        &mut self,
        inputs: &[SplitTailInput<'_>],
    ) -> Result<Array2<f32>, ort::Error> {
        debug_assert!(inputs.len() <= PRIMARY_BATCH_SIZE);

        let row_stride = FBANK_FRAMES * FBANK_FEATURES;
        for (batch_idx, input) in inputs.iter().enumerate() {
            debug_assert_eq!(input.fbank.ncols(), FBANK_FEATURES);

            if batch_idx > 0 && std::ptr::eq(input.fbank, inputs[batch_idx - 1].fbank) {
                let buf = self
                    .split_primary_feature_batch_buffer
                    .as_slice_mut()
                    .ok_or_else(|| {
                        ort::Error::new("split primary feature batch buffer was not contiguous")
                    })?;
                let prev_start = (batch_idx - 1) * row_stride;
                buf.copy_within(prev_start..prev_start + row_stride, batch_idx * row_stride);
            } else {
                self.split_primary_feature_batch_buffer
                    .slice_mut(s![batch_idx, ..input.fbank.nrows(), ..input.fbank.ncols()])
                    .assign(input.fbank);
            }

            Self::prepare_weights(
                batch_idx,
                input.weights,
                self.mask_frames,
                &mut self.split_primary_weights_batch_buffer.view_mut(),
            );
        }
        if inputs.len() < PRIMARY_BATCH_SIZE {
            self.split_primary_weights_batch_buffer
                .slice_mut(s![inputs.len().., ..])
                .fill(0.0);
        }

        #[cfg(feature = "coreml")]
        {
            let _ = self.ensure_native_tail_primary_batched_loaded();
        }
        #[cfg(feature = "coreml")]
        if let Some(native) = self.native_tail_primary_batched_session.as_mut() {
            let fbank_data = array3_slice(
                &self.split_primary_feature_batch_buffer,
                "native primary tail fbank input",
            )?;
            let weights_data = array2_slice(
                &self.split_primary_weights_batch_buffer,
                "native primary tail weights input",
            )?;
            let (data, _) = native
                .predict_cached(&[
                    (&self.cached_tail_fbank_shape, fbank_data),
                    (&self.cached_tail_weights_shape, weights_data),
                ])
                .map_err(|e| ort::Error::new(e.to_string()))?;
            let batch =
                array2_from_shape_vec(PRIMARY_BATCH_SIZE, 256, data, "native primary tail output")?;
            return Ok(batch.slice(s![0..inputs.len(), ..]).to_owned());
        }

        let fbank_tensor =
            TensorRef::from_array_view(self.split_primary_feature_batch_buffer.view())?;
        let weights_tensor =
            TensorRef::from_array_view(self.split_primary_weights_batch_buffer.view())?;
        let outputs = self
            .split_primary_tail_batched_session
            .as_mut()
            .ok_or_else(|| ort::Error::new("missing primary tail batched session"))?
            .run(ort::inputs!["fbank" => fbank_tensor, "weights" => weights_tensor])?;
        let (_shape, data) = outputs[0].try_extract_tensor::<f32>()?;
        let batch = array2_from_shape_vec(
            PRIMARY_BATCH_SIZE,
            256,
            data.to_vec(),
            "primary tail batched output",
        )?;
        Ok(batch.slice(s![0..inputs.len(), ..]).to_owned())
    }
}
