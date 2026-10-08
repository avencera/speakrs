use ndarray::{Array2, Array3, ArrayView2, ArrayViewMut3, s};

use crate::inference::{InferenceError, TensorShapeError};

use super::prepare_weights;
use super::tensor::array3_slice_mut;
use super::{
    CHUNK_SPEAKER_BATCH_SIZE, FBANK_BATCH_SIZE, FBANK_FEATURES, FBANK_FRAMES, MASK_FRAMES,
    MULTI_MASK_BATCH_SIZE, NUM_SPEAKERS, PRIMARY_BATCH_SIZE, SplitTailInput, should_use_clean_mask,
};

/// Input staging for the split filterbank, tail, and multi-mask models
///
/// Both backends feed these models the same padded tensors, so filling lives here and each
/// backend only runs its sessions on the filled buffers. Buffers are private to one model
/// handle and must not be shared across concurrent runs
pub(super) struct EmbeddingBuffers {
    pub(super) multi_mask_fbank_buffer: Array3<f32>,
    pub(super) multi_mask_masks_buffer: Array2<f32>,
    pub(super) split_waveform_buffer: Array3<f32>,
    pub(super) split_fbank_batch_buffer: Array3<f32>,
    pub(super) split_feature_batch_buffer: Array3<f32>,
    pub(super) split_weights_batch_buffer: Array2<f32>,
    pub(super) split_primary_feature_batch_buffer: Array3<f32>,
    pub(super) split_primary_weights_batch_buffer: Array2<f32>,
}

impl EmbeddingBuffers {
    pub(super) fn fresh() -> Self {
        Self {
            multi_mask_fbank_buffer: Array3::zeros((
                MULTI_MASK_BATCH_SIZE,
                FBANK_FRAMES,
                FBANK_FEATURES,
            )),
            multi_mask_masks_buffer: Array2::zeros((
                MULTI_MASK_BATCH_SIZE * NUM_SPEAKERS,
                MASK_FRAMES,
            )),
            split_waveform_buffer: Array3::zeros((1, 1, 160_000)),
            split_fbank_batch_buffer: Array3::zeros((FBANK_BATCH_SIZE, 1, 160_000)),
            split_feature_batch_buffer: Array3::zeros((
                CHUNK_SPEAKER_BATCH_SIZE,
                FBANK_FRAMES,
                FBANK_FEATURES,
            )),
            split_weights_batch_buffer: Array2::zeros((CHUNK_SPEAKER_BATCH_SIZE, 589)),
            split_primary_feature_batch_buffer: Array3::zeros((
                PRIMARY_BATCH_SIZE,
                FBANK_FRAMES,
                FBANK_FEATURES,
            )),
            split_primary_weights_batch_buffer: Array2::zeros((PRIMARY_BATCH_SIZE, 589)),
        }
    }

    /// Stage one zero-padded waveform for the single filterbank model
    pub(super) fn fill_split_waveform(&mut self, audio: &[f32], window_samples: usize) {
        prepare_waveform(
            0,
            audio,
            window_samples,
            &mut self.split_waveform_buffer.view_mut(),
        );
    }

    /// Stage up to one batch of zero-padded waveforms for the batched filterbank model
    pub(super) fn fill_fbank_batch(&mut self, audios: &[&[f32]], window_samples: usize) {
        self.split_fbank_batch_buffer.fill(0.0);
        for (idx, audio) in audios.iter().enumerate() {
            let copy_len = audio.len().min(window_samples);
            self.split_fbank_batch_buffer
                .slice_mut(s![idx, 0, ..copy_len])
                .assign(&ndarray::ArrayView1::from(&audio[..copy_len]));
        }
    }

    /// Stage one filterbank and its weights in row 0 of the tail batch
    pub(super) fn fill_tail_single(
        &mut self,
        fbank: &Array2<f32>,
        weights: &[f32],
        mask_frames: usize,
    ) {
        self.split_feature_batch_buffer
            .slice_mut(s![0, ..fbank.nrows(), ..fbank.ncols()])
            .assign(fbank);
        prepare_weights(
            0,
            weights,
            mask_frames,
            &mut self.split_weights_batch_buffer.view_mut(),
        );
    }

    /// Stage one filterbank repeated per speaker plus each speaker's selected mask
    pub(super) fn fill_tail_batch(
        &mut self,
        fbank: &Array2<f32>,
        segmentations: &ArrayView2<'_, f32>,
        clean_masks: &Array2<f32>,
        num_samples: usize,
        mask_frames: usize,
        min_num_samples: usize,
    ) -> Result<(), InferenceError> {
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
            let use_clean =
                should_use_clean_mask(&clean_col, mask_col.len(), num_samples, min_num_samples);
            let weights: Vec<f32> = if use_clean {
                clean_col.iter().copied().collect()
            } else {
                mask_col.iter().copied().collect()
            };
            prepare_weights(
                speaker_idx,
                &weights,
                mask_frames,
                &mut self.split_weights_batch_buffer.view_mut(),
            );
        }

        Ok(())
    }

    /// Stage up to [`PRIMARY_BATCH_SIZE`] filterbank and weight rows for the batch-64 tail
    pub(super) fn fill_tail_primary(
        &mut self,
        inputs: &[SplitTailInput<'_>],
        mask_frames: usize,
    ) -> Result<(), InferenceError> {
        if inputs.len() > PRIMARY_BATCH_SIZE {
            return Err(InferenceError::BatchTooLarge {
                context: "primary tail batch",
                rows: inputs.len(),
                capacity: PRIMARY_BATCH_SIZE,
            });
        }

        let row_stride = FBANK_FRAMES * FBANK_FEATURES;
        for (batch_idx, input) in inputs.iter().enumerate() {
            debug_assert_eq!(input.fbank.ncols(), FBANK_FEATURES);

            if batch_idx > 0 && std::ptr::eq(input.fbank, inputs[batch_idx - 1].fbank) {
                let buf = array3_slice_mut(
                    &mut self.split_primary_feature_batch_buffer,
                    "split primary feature batch buffer",
                )?;
                let prev_start = (batch_idx - 1) * row_stride;
                buf.copy_within(prev_start..prev_start + row_stride, batch_idx * row_stride);
            } else {
                self.split_primary_feature_batch_buffer
                    .slice_mut(s![batch_idx, ..input.fbank.nrows(), ..input.fbank.ncols()])
                    .assign(input.fbank);
            }

            prepare_weights(
                batch_idx,
                input.weights,
                mask_frames,
                &mut self.split_primary_weights_batch_buffer.view_mut(),
            );
        }
        if inputs.len() < PRIMARY_BATCH_SIZE {
            self.split_primary_weights_batch_buffer
                .slice_mut(s![inputs.len().., ..])
                .fill(0.0);
        }

        Ok(())
    }

    /// Stage filterbank rows and their [`NUM_SPEAKERS`] masks for the multi-mask tail
    pub(super) fn fill_multi_mask(
        &mut self,
        fbanks: &[&Array2<f32>],
        masks: &[&[f32]],
        mask_frames: usize,
    ) -> Result<(), InferenceError> {
        let num_fbanks = fbanks.len();
        let num_masks = masks.len();
        if num_fbanks > MULTI_MASK_BATCH_SIZE {
            return Err(InferenceError::BatchTooLarge {
                context: "multi-mask batch",
                rows: num_fbanks,
                capacity: MULTI_MASK_BATCH_SIZE,
            });
        }
        let expected_masks =
            num_fbanks
                .checked_mul(NUM_SPEAKERS)
                .ok_or(TensorShapeError::Overflow {
                    context: "multi-mask batch",
                })?;
        if num_masks != expected_masks {
            return Err(InferenceError::MaskCountMismatch {
                fbanks: num_fbanks,
                expected: expected_masks,
                actual: num_masks,
            });
        }

        let fbank_row_stride = FBANK_FRAMES * FBANK_FEATURES;
        for (idx, fbank) in fbanks.iter().enumerate() {
            self.multi_mask_fbank_buffer
                .slice_mut(s![idx, ..fbank.nrows(), ..fbank.ncols()])
                .assign(fbank);
        }

        for (idx, mask) in masks.iter().enumerate() {
            prepare_weights(
                idx,
                mask,
                mask_frames,
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

        Ok(())
    }
}

/// Copy one waveform into a batch row, truncating or zero-padding to `window_samples`
pub(super) fn prepare_waveform(
    batch_idx: usize,
    audio: &[f32],
    window_samples: usize,
    waveform_buffer: &mut ArrayViewMut3<f32>,
) {
    let copy_len = audio.len().min(window_samples);
    waveform_buffer
        .slice_mut(s![batch_idx, 0, ..copy_len])
        .assign(&ndarray::ArrayView1::from(&audio[..copy_len]));
    if copy_len < window_samples {
        waveform_buffer
            .slice_mut(s![batch_idx, 0, copy_len..])
            .fill(0.0);
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::prepare_weights;

    #[test]
    fn prepare_weights_clears_tail_when_mask_is_shorter_than_buffer() {
        let mut buffer = ndarray::Array2::from_elem((2, 4), 9.0);

        prepare_weights(0, &[1.0, 2.0], 4, &mut buffer.view_mut());
        prepare_weights(1, &[3.0, 4.0, 5.0, 6.0, 7.0], 4, &mut buffer.view_mut());

        assert_eq!(buffer, array![[1.0, 2.0, 0.0, 0.0], [3.0, 4.0, 5.0, 6.0]]);
    }
}
