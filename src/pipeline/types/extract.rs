use ndarray::{Array2, Array3, s};
use tracing::debug;

use crate::inference::embedding::{EmbeddingModel, MaskedEmbeddingInput, SplitTailInput};
use crate::pipeline::{MIN_SPEAKER_ACTIVITY, clean_masks, select_speaker_weights};
use crate::reconstruct::aggregate_speaker_count;

use super::{
    ChunkEmbeddings, ChunkLayout, DecodedSegmentations, EmbeddingPath, PendingEmbedding,
    PendingSplitEmbedding, PipelineError, SpeakerCountTrack,
};

impl DecodedSegmentations {
    pub(in crate::pipeline) fn nchunks(&self) -> usize {
        self.0.shape()[0]
    }

    pub(in crate::pipeline) fn num_speakers(&self) -> usize {
        if self.0.ndim() < 3 {
            return 0;
        }
        self.0.shape()[2]
    }

    pub(in crate::pipeline) fn speaker_count(&self, layout: &ChunkLayout) -> SpeakerCountTrack {
        aggregate_speaker_count(self, &layout.start_frames, layout.output_frames)
    }

    pub(in crate::pipeline) fn extract_embeddings(
        &self,
        audio: &[f32],
        emb_model: &mut EmbeddingModel,
        layout: &ChunkLayout,
        embedding_path: EmbeddingPath,
    ) -> Result<ChunkEmbeddings, PipelineError> {
        let num_chunks = self.0.shape()[0];
        let num_speakers = self.0.shape()[2];
        let mut embeddings = Array3::<f32>::from_elem((num_chunks, num_speakers, 256), f32::NAN);

        match embedding_path {
            EmbeddingPath::MultiMask => {
                self.extract_multi_mask_embeddings(audio, emb_model, layout, &mut embeddings)?
            }
            EmbeddingPath::Split => {
                self.extract_split_embeddings(audio, emb_model, layout, &mut embeddings)?
            }
            EmbeddingPath::Masked => {
                if emb_model.prefers_chunk_embedding_path() {
                    self.extract_chunk_embeddings(audio, emb_model, layout, &mut embeddings)?;
                } else {
                    self.extract_masked_embeddings(audio, emb_model, layout, &mut embeddings)?;
                }
            }
        }

        Ok(ChunkEmbeddings(embeddings))
    }

    fn extract_chunk_embeddings(
        &self,
        audio: &[f32],
        emb_model: &mut EmbeddingModel,
        layout: &ChunkLayout,
        embeddings: &mut Array3<f32>,
    ) -> Result<(), PipelineError> {
        for chunk_idx in 0..self.0.shape()[0] {
            let chunk_audio = layout.chunk_audio(audio, chunk_idx);
            let chunk_segmentations = self.0.slice(s![chunk_idx, .., ..]);
            let clean_masks = clean_masks(&chunk_segmentations);
            let chunk_embeddings =
                emb_model.embed_chunk_speakers(chunk_audio, chunk_segmentations, &clean_masks)?;
            embeddings
                .slice_mut(s![chunk_idx, .., ..])
                .assign(&chunk_embeddings);
        }

        Ok(())
    }

    fn extract_masked_embeddings(
        &self,
        audio: &[f32],
        emb_model: &mut EmbeddingModel,
        layout: &ChunkLayout,
        embeddings: &mut Array3<f32>,
    ) -> Result<(), PipelineError> {
        let mut storage = Array3Writer(embeddings);
        let mut pending = Vec::with_capacity(emb_model.primary_batch_size());

        for chunk_idx in 0..self.0.shape()[0] {
            let chunk_audio = layout.chunk_audio(audio, chunk_idx);
            let chunk_segmentations = self.0.slice(s![chunk_idx, .., ..]);
            let clean_masks = clean_masks(&chunk_segmentations);

            for speaker_idx in 0..self.0.shape()[2] {
                let mask = chunk_segmentations.column(speaker_idx);
                let activity: f32 = mask.iter().sum();
                if activity < MIN_SPEAKER_ACTIVITY {
                    continue;
                }

                pending.push(PendingEmbedding {
                    chunk_idx,
                    speaker_idx,
                    audio: chunk_audio,
                    mask: mask.to_vec(),
                    clean_mask: clean_masks.column(speaker_idx).to_vec(),
                });
                if pending.len() == emb_model.primary_batch_size() {
                    flush_masked(emb_model, &pending, &mut storage)?;
                    pending.clear();
                }
            }
        }

        while !pending.is_empty() {
            let batch_len = emb_model.best_batch_len(pending.len());
            flush_masked(emb_model, &pending[..batch_len], &mut storage)?;
            pending.drain(..batch_len);
        }

        Ok(())
    }

    fn extract_split_embeddings(
        &self,
        audio: &[f32],
        emb_model: &mut EmbeddingModel,
        layout: &ChunkLayout,
        embeddings: &mut Array3<f32>,
    ) -> Result<(), PipelineError> {
        let batch_size = emb_model.split_primary_batch_size();
        let num_speakers = self.0.shape()[2];
        let min_num_samples = emb_model.min_num_samples();

        let mut storage = Array3Writer(embeddings);
        let mut batch = SplitEmbeddingBatch::new(batch_size);
        let mut tail_batches = 0usize;
        let mut active_items = 0usize;

        for chunk_idx in 0..self.0.shape()[0] {
            let chunk_audio = layout.chunk_audio(audio, chunk_idx);
            let chunk_segmentations = self.0.slice(s![chunk_idx, .., ..]);
            let clean_masks = clean_masks(&chunk_segmentations);

            for speaker_idx in 0..num_speakers {
                let Some(weights) = select_speaker_weights(
                    &chunk_segmentations,
                    &clean_masks,
                    speaker_idx,
                    chunk_audio.len(),
                    min_num_samples,
                ) else {
                    continue;
                };
                active_items += 1;
                batch.push(chunk_idx, speaker_idx, weights, || {
                    Ok(emb_model.compute_chunk_fbank(chunk_audio)?)
                })?;
                if batch.is_full() {
                    batch.flush(|pending, fbanks| {
                        flush_split(emb_model, pending, fbanks, &mut storage)
                    })?;
                    tail_batches += 1;
                }
            }
        }

        if !batch.is_empty() {
            batch.flush(|pending, fbanks| flush_split(emb_model, pending, fbanks, &mut storage))?;
            tail_batches += 1;
        }

        debug!(
            batches = tail_batches,
            active_items,
            total_items = self.0.shape()[0] * num_speakers,
            "Split embeddings complete (fbank+tail streaming)"
        );

        Ok(())
    }

    fn extract_multi_mask_embeddings(
        &self,
        audio: &[f32],
        emb_model: &mut EmbeddingModel,
        layout: &ChunkLayout,
        embeddings: &mut Array3<f32>,
    ) -> Result<(), PipelineError> {
        let batch_size = emb_model.multi_mask_batch_size();
        let num_speakers = self.0.shape()[2];
        let num_chunks = self.0.shape()[0];
        let min_num_samples = emb_model.min_num_samples();

        let mut storage = Array3Writer(embeddings);
        let mut fbank_buffer: Vec<Array2<f32>> = Vec::with_capacity(batch_size);
        let mut masks_buffer: Vec<Vec<f32>> = Vec::with_capacity(batch_size * num_speakers);
        let mut chunk_indices: Vec<usize> = Vec::with_capacity(batch_size);
        let mut batches = 0usize;

        for chunk_idx in 0..num_chunks {
            let chunk_audio = layout.chunk_audio(audio, chunk_idx);
            let fbank = emb_model.compute_chunk_fbank(chunk_audio)?;
            fbank_buffer.push(fbank);
            chunk_indices.push(chunk_idx);

            let chunk_segmentations = self.0.slice(s![chunk_idx, .., ..]);
            let clean_masks_arr = clean_masks(&chunk_segmentations);

            for speaker_idx in 0..num_speakers {
                let Some(weights) = select_speaker_weights(
                    &chunk_segmentations,
                    &clean_masks_arr,
                    speaker_idx,
                    chunk_audio.len(),
                    min_num_samples,
                ) else {
                    masks_buffer.push(vec![0.0; 589]);
                    continue;
                };
                masks_buffer.push(weights);
            }

            if fbank_buffer.len() == batch_size {
                flush_multi_mask(
                    emb_model,
                    &fbank_buffer,
                    &masks_buffer,
                    &chunk_indices,
                    num_speakers,
                    &mut storage,
                )?;
                batches += 1;
                fbank_buffer.clear();
                masks_buffer.clear();
                chunk_indices.clear();
            }
        }

        if !fbank_buffer.is_empty() {
            flush_multi_mask(
                emb_model,
                &fbank_buffer,
                &masks_buffer,
                &chunk_indices,
                num_speakers,
                &mut storage,
            )?;
            batches += 1;
        }

        debug!(
            batches,
            total_chunks = num_chunks,
            "Multi-mask embeddings complete"
        );

        Ok(())
    }
}

pub(in crate::pipeline) trait EmbeddingStorage {
    fn store(&mut self, chunk_idx: usize, speaker_idx: usize, embedding: &[f32]);
}

fn store_row<S: EmbeddingStorage>(
    storage: &mut S,
    chunk_idx: usize,
    speaker_idx: usize,
    row: ndarray::ArrayView1<'_, f32>,
) {
    if let Some(values) = row.as_slice() {
        storage.store(chunk_idx, speaker_idx, values);
        return;
    }

    let values = row.to_vec();
    storage.store(chunk_idx, speaker_idx, &values);
}

/// Writes embeddings into a pre-allocated Array3 by (chunk, speaker) index
pub(in crate::pipeline) struct Array3Writer<'a>(pub &'a mut Array3<f32>);

impl EmbeddingStorage for Array3Writer<'_> {
    fn store(&mut self, chunk_idx: usize, speaker_idx: usize, embedding: &[f32]) {
        self.0
            .slice_mut(s![chunk_idx, speaker_idx, ..])
            .assign(&ndarray::ArrayView1::from(embedding));
    }
}

pub(in crate::pipeline) fn flush_masked<S: EmbeddingStorage>(
    emb_model: &mut EmbeddingModel,
    pending: &[PendingEmbedding<'_>],
    storage: &mut S,
) -> Result<(), PipelineError> {
    let batch_inputs: Vec<_> = pending
        .iter()
        .map(|item| MaskedEmbeddingInput {
            audio: item.audio,
            mask: &item.mask,
            clean_mask: Some(&item.clean_mask),
        })
        .collect();
    let batch_embeddings = emb_model.embed_batch(&batch_inputs)?;

    for (batch_idx, item) in pending.iter().enumerate() {
        store_row(
            storage,
            item.chunk_idx,
            item.speaker_idx,
            batch_embeddings.row(batch_idx),
        );
    }

    Ok(())
}

/// Pending split-path embeddings and the chunk filterbanks they read
///
/// A chunk's filterbank is computed when its first active speaker is queued, so silent
/// chunks never hold one, and the filterbanks are released when their batch flushes. Only
/// the newest filterbank outlives a flush, because later speakers of the same chunk may
/// still be queued against it
pub(in crate::pipeline) struct SplitEmbeddingBatch {
    pending: Vec<PendingSplitEmbedding>,
    fbanks: Vec<Array2<f32>>,
    capacity: usize,
    /// The chunk whose filterbank is last in `fbanks`
    fbank_chunk: Option<usize>,
}

impl SplitEmbeddingBatch {
    pub(in crate::pipeline) fn new(capacity: usize) -> Self {
        Self {
            pending: Vec::with_capacity(capacity),
            fbanks: Vec::new(),
            capacity,
            fbank_chunk: None,
        }
    }

    /// Queue one active speaker, computing the chunk filterbank on the chunk's first speaker
    pub(in crate::pipeline) fn push(
        &mut self,
        chunk_idx: usize,
        speaker_idx: usize,
        weights: Vec<f32>,
        compute_fbank: impl FnOnce() -> Result<Array2<f32>, PipelineError>,
    ) -> Result<(), PipelineError> {
        if self.fbank_chunk != Some(chunk_idx) {
            // a filterbank kept across the last flush has no pending reader once its chunk ends
            if self.pending.is_empty() {
                self.fbanks.clear();
            }
            self.fbanks.push(compute_fbank()?);
            self.fbank_chunk = Some(chunk_idx);
        }

        self.pending.push(PendingSplitEmbedding {
            chunk_idx,
            speaker_idx,
            fbank_idx: self.fbanks.len() - 1,
            weights,
        });
        Ok(())
    }

    pub(in crate::pipeline) fn is_full(&self) -> bool {
        self.pending.len() >= self.capacity
    }

    pub(in crate::pipeline) fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    /// Run the pending batch, then keep only the newest chunk filterbank
    pub(in crate::pipeline) fn flush(
        &mut self,
        run: impl FnOnce(&[PendingSplitEmbedding], &[Array2<f32>]) -> Result<(), PipelineError>,
    ) -> Result<(), PipelineError> {
        run(&self.pending, &self.fbanks)?;
        self.pending.clear();

        let newest = self.fbanks.len().saturating_sub(1);
        self.fbanks.drain(..newest);
        Ok(())
    }
}

pub(in crate::pipeline) fn flush_split<S: EmbeddingStorage>(
    emb_model: &mut EmbeddingModel,
    pending: &[PendingSplitEmbedding],
    fbanks: &[Array2<f32>],
    storage: &mut S,
) -> Result<(), PipelineError> {
    let batch_inputs: Vec<_> = pending
        .iter()
        .map(|item| SplitTailInput {
            fbank: &fbanks[item.fbank_idx],
            weights: &item.weights,
        })
        .collect();
    let batch_embeddings = emb_model.embed_tail_batch_inputs(&batch_inputs)?;

    for (batch_idx, item) in pending.iter().enumerate() {
        store_row(
            storage,
            item.chunk_idx,
            item.speaker_idx,
            batch_embeddings.row(batch_idx),
        );
    }

    Ok(())
}

pub(in crate::pipeline) fn flush_multi_mask<S: EmbeddingStorage>(
    emb_model: &mut EmbeddingModel,
    fbanks: &[Array2<f32>],
    masks: &[Vec<f32>],
    chunk_indices: &[usize],
    num_speakers: usize,
    storage: &mut S,
) -> Result<(), PipelineError> {
    let fbank_refs: Vec<_> = fbanks.iter().collect();
    let mask_refs: Vec<_> = masks.iter().map(|m| m.as_slice()).collect();
    let batch_embeddings = emb_model.embed_multi_mask_batch(&fbank_refs, &mask_refs)?;

    for (fbank_idx, &chunk_idx) in chunk_indices.iter().enumerate() {
        for speaker_idx in 0..num_speakers {
            let mask_idx = fbank_idx * num_speakers + speaker_idx;
            let is_active = masks[mask_idx].iter().any(|&v| v > 0.0);
            if !is_active {
                continue;
            }
            store_row(
                storage,
                chunk_idx,
                speaker_idx,
                batch_embeddings.row(mask_idx),
            );
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;

    use super::{PipelineError, SplitEmbeddingBatch};

    /// Queues `(chunk, speaker)` pairs, flushing whenever the batch fills, and records the
    /// filterbank computations and what each flush saw
    fn run(capacity: usize, active: &[(usize, usize)]) -> (Vec<usize>, Vec<Vec<(usize, f32)>>) {
        let mut batch = SplitEmbeddingBatch::new(capacity);
        let mut computed = Vec::new();
        let mut flushes = Vec::new();
        let mut flush = |batch: &mut SplitEmbeddingBatch| {
            batch
                .flush(|pending, fbanks| {
                    // each filterbank holds its chunk index, so a pending item reads its own chunk
                    let seen = pending
                        .iter()
                        .map(|item| (item.chunk_idx, fbanks[item.fbank_idx][[0, 0]]))
                        .collect();
                    flushes.push(seen);
                    Ok::<(), PipelineError>(())
                })
                .unwrap();
        };

        for &(chunk_idx, speaker_idx) in active {
            batch
                .push(chunk_idx, speaker_idx, vec![1.0], || {
                    computed.push(chunk_idx);
                    Ok(Array2::from_elem((1, 1), chunk_idx as f32))
                })
                .unwrap();
            if batch.is_full() {
                flush(&mut batch);
            }
        }
        if !batch.is_empty() {
            flush(&mut batch);
        }
        assert!(batch.fbanks.len() <= 1);
        (computed, flushes)
    }

    #[test]
    fn silent_chunks_compute_and_hold_no_filterbank() {
        // chunks 0-1, 3-7 and 9+ are silent and never reach the batch
        let (computed, flushes) = run(2, &[(2, 0), (2, 1), (8, 0)]);

        assert_eq!(computed, [2, 8]);
        assert_eq!(flushes, [vec![(2, 2.0), (2, 2.0)], vec![(8, 8.0)]]);
    }

    #[test]
    fn speakers_after_a_mid_chunk_flush_reuse_the_chunk_filterbank() {
        let (computed, flushes) = run(1, &[(3, 0), (3, 1), (4, 2)]);

        assert_eq!(computed, [3, 4]);
        assert_eq!(flushes, [vec![(3, 3.0)], vec![(3, 3.0)], vec![(4, 4.0)]]);
    }
}
