use ndarray::{Array2, Array3, s};
use tracing::{info, trace};

use crate::inference::embedding::EmbeddingModel;
use crate::powerset::PowersetMapping;

use super::config::MIN_SPEAKER_ACTIVITY;
use super::types::{
    Array3Writer, EmbeddingStorage, PendingEmbedding, PendingSplitEmbedding, PipelineError,
    chunk_audio_raw, flush_masked, flush_split,
};
use super::{clean_masks, select_speaker_weights, write_speaker_mask_to_slice};

pub(super) struct ConcurrentEmbeddingResult {
    pub segmentations: Array3<f32>,
    pub embeddings: Array3<f32>,
    pub num_chunks: usize,
}

impl ConcurrentEmbeddingResult {
    pub fn is_empty(&self) -> bool {
        self.num_chunks == 0
    }
}

/// Compute total window count matching the streaming segmentation sliding-window logic
/// Includes the zero-padded tail window. Returns 0 when audio is shorter than one window
fn streaming_total_windows(audio_len: usize, window_samples: usize, step_samples: usize) -> usize {
    let full_windows = if audio_len >= window_samples {
        (audio_len - window_samples) / step_samples + 1
    } else {
        return 0;
    };
    let offset_after_full = full_windows * step_samples;
    let has_tail = offset_after_full < audio_len;
    full_windows + has_tail as usize
}

struct MultiMaskBatch<'a> {
    audio_slices: &'a [&'a [f32]],
    flat_masks: &'a [f32],
    mask_stride: usize,
    active_flags: &'a [bool],
    chunk_indices: &'a [usize],
}

pub(super) struct ConcurrentEmbeddingRunner<'a> {
    pub powerset: &'a PowersetMapping,
    pub audio: &'a [f32],
    pub step_samples: usize,
    pub window_samples: usize,
    pub num_speakers: usize,
}

impl<'a> ConcurrentEmbeddingRunner<'a> {
    fn total_windows(&self) -> usize {
        streaming_total_windows(self.audio.len(), self.window_samples, self.step_samples)
    }

    pub fn run_split(
        &self,
        receiver: crossbeam_channel::Receiver<Array2<f32>>,
        embedding_model: &mut EmbeddingModel,
        batch_size: usize,
        min_num_samples: usize,
    ) -> Result<ConcurrentEmbeddingResult, PipelineError> {
        let total_windows = self.total_windows();
        let mut seg_array: Option<Array3<f32>> = None;
        let mut emb_array: Option<Array3<f32>> = None;
        let mut pending: Vec<PendingSplitEmbedding> = Vec::with_capacity(batch_size);
        let mut fbanks: Vec<Array2<f32>> = Vec::new();
        let mut chunk_idx = 0usize;

        for raw_window in receiver {
            let decoded = self.powerset.hard_decode(&raw_window);
            let seg = seg_array.get_or_insert_with(|| {
                Array3::zeros((total_windows, decoded.nrows(), self.num_speakers))
            });
            seg.slice_mut(s![chunk_idx, .., ..]).assign(&decoded);
            drop(decoded);

            let seg_view = seg.slice(s![chunk_idx, .., ..]);
            let chunk_audio = chunk_audio_raw(
                self.audio,
                self.step_samples,
                self.window_samples,
                chunk_idx,
            );
            let clean = clean_masks(&seg_view);

            let fbank = embedding_model.compute_chunk_fbank(chunk_audio)?;
            fbanks.push(fbank);
            let mut current_fbank_idx = fbanks.len() - 1;

            for speaker_idx in 0..self.num_speakers {
                let Some(weights) = select_speaker_weights(
                    &seg_view,
                    &clean,
                    speaker_idx,
                    chunk_audio.len(),
                    min_num_samples,
                ) else {
                    continue;
                };
                pending.push(PendingSplitEmbedding {
                    chunk_idx,
                    speaker_idx,
                    fbank_idx: current_fbank_idx,
                    weights,
                });
                if pending.len() == batch_size {
                    let emb = emb_array.get_or_insert_with(|| {
                        Array3::from_elem((total_windows, self.num_speakers, 256), f32::NAN)
                    });
                    flush_split(embedding_model, &pending, &fbanks, &mut Array3Writer(emb))?;
                    pending.clear();

                    // keep the current chunk fbank alive if later speakers in this chunk still need it
                    if speaker_idx + 1 < self.num_speakers {
                        let kept_fbank = fbanks.swap_remove(current_fbank_idx);
                        fbanks.clear();
                        fbanks.push(kept_fbank);
                        current_fbank_idx = 0;
                    } else {
                        fbanks.clear();
                    }
                }
            }
            chunk_idx += 1;
        }

        if !pending.is_empty() {
            let emb = emb_array.get_or_insert_with(|| {
                Array3::from_elem((total_windows, self.num_speakers, 256), f32::NAN)
            });
            flush_split(embedding_model, &pending, &fbanks, &mut Array3Writer(emb))?;
        }

        self.finalize(seg_array, emb_array, chunk_idx, total_windows)
    }

    pub fn run_multi_mask(
        &self,
        receiver: crossbeam_channel::Receiver<Array2<f32>>,
        embedding_model: &mut EmbeddingModel,
        batch_size: usize,
        min_num_samples: usize,
    ) -> Result<ConcurrentEmbeddingResult, PipelineError> {
        let total_windows = self.total_windows();
        let mut seg_array: Option<Array3<f32>> = None;
        let mut emb_array: Option<Array3<f32>> = None;
        let mut num_frames: Option<usize> = None;

        // buffer audio slices instead of pre-computed fbanks — fbanks are computed
        // in a single batched call per flush, avoiding redundant per-window computation
        let mut audio_buffer: Vec<&[f32]> = Vec::with_capacity(batch_size);
        let mask_capacity = batch_size * self.num_speakers;
        // flat mask buffer: one contiguous allocation, indexed by [slot * nf .. (slot+1) * nf]
        let mut flat_masks: Vec<f32> = Vec::new();
        let mut active_flags: Vec<bool> = Vec::with_capacity(mask_capacity);
        let mut chunk_indices: Vec<usize> = Vec::with_capacity(batch_size);
        let mut chunk_idx = 0usize;

        let mut total_recv_wait_us = 0u64;
        let mut total_decode_us = 0u64;
        let mut total_fbank_us = 0u64;
        let mut total_gpu_predict_us = 0u64;
        let mut flush_count = 0u32;

        loop {
            let recv_start = std::time::Instant::now();
            let raw_window = match receiver.recv() {
                Ok(w) => w,
                Err(_) => break,
            };
            total_recv_wait_us += recv_start.elapsed().as_micros() as u64;

            let decode_start = std::time::Instant::now();
            let decoded = self.powerset.hard_decode(&raw_window);

            let nf = *num_frames.get_or_insert(decoded.nrows());
            let seg = seg_array
                .get_or_insert_with(|| Array3::zeros((total_windows, nf, self.num_speakers)));
            if flat_masks.is_empty() {
                flat_masks.resize(mask_capacity * nf, 0.0);
            }

            seg.slice_mut(s![chunk_idx, .., ..]).assign(&decoded);
            drop(decoded);

            let seg_view = seg.slice(s![chunk_idx, .., ..]);
            let chunk_audio = chunk_audio_raw(
                self.audio,
                self.step_samples,
                self.window_samples,
                chunk_idx,
            );
            audio_buffer.push(chunk_audio);
            chunk_indices.push(chunk_idx);

            let mask_base = (audio_buffer.len() - 1) * self.num_speakers;
            for speaker_idx in 0..self.num_speakers {
                let slot = mask_base + speaker_idx;
                let offset = slot * nf;
                let dest = &mut flat_masks[offset..offset + nf];
                let active = write_speaker_mask_to_slice(
                    &seg_view,
                    speaker_idx,
                    chunk_audio.len(),
                    min_num_samples,
                    dest,
                );
                active_flags.push(active);
            }
            total_decode_us += decode_start.elapsed().as_micros() as u64;

            if audio_buffer.len() == batch_size {
                let emb = emb_array.get_or_insert_with(|| {
                    Array3::from_elem((total_windows, self.num_speakers, 256), f32::NAN)
                });
                let batch = MultiMaskBatch {
                    audio_slices: &audio_buffer,
                    flat_masks: &flat_masks,
                    mask_stride: nf,
                    active_flags: &active_flags,
                    chunk_indices: &chunk_indices,
                };
                let (fbank_us, gpu_us) =
                    self.flush_multi_mask_flat(embedding_model, &batch, &mut Array3Writer(emb))?;
                total_fbank_us += fbank_us;
                total_gpu_predict_us += gpu_us;
                flush_count += 1;
                audio_buffer.clear();
                flat_masks.fill(0.0);
                active_flags.clear();
                chunk_indices.clear();
            }
            chunk_idx += 1;
        }

        if !audio_buffer.is_empty() {
            let nf = self.require_num_frames(num_frames, chunk_idx)?;
            let emb = emb_array.get_or_insert_with(|| {
                Array3::from_elem((total_windows, self.num_speakers, 256), f32::NAN)
            });
            let batch = MultiMaskBatch {
                audio_slices: &audio_buffer,
                flat_masks: &flat_masks,
                mask_stride: nf,
                active_flags: &active_flags,
                chunk_indices: &chunk_indices,
            };
            let (fbank_us, gpu_us) =
                self.flush_multi_mask_flat(embedding_model, &batch, &mut Array3Writer(emb))?;
            total_fbank_us += fbank_us;
            total_gpu_predict_us += gpu_us;
            flush_count += 1;
        }

        trace!(
            flushes = flush_count,
            chunks = chunk_idx,
            recv_wait_ms = total_recv_wait_us / 1000,
            decode_ms = total_decode_us / 1000,
            fbank_ms = total_fbank_us / 1000,
            gpu_predict_ms = total_gpu_predict_us / 1000,
            "Multi-mask embedding timing"
        );

        self.finalize(seg_array, emb_array, chunk_idx, total_windows)
    }

    fn flush_multi_mask_flat<S: EmbeddingStorage>(
        &self,
        embedding_model: &mut EmbeddingModel,
        batch: &MultiMaskBatch<'_>,
        storage: &mut S,
    ) -> Result<(u64, u64), PipelineError> {
        let fbank_start = std::time::Instant::now();
        let fbanks = embedding_model.compute_chunk_fbanks_batch(batch.audio_slices)?;
        let fbank_us = fbank_start.elapsed().as_micros() as u64;

        let fbank_refs: Vec<_> = fbanks.iter().collect();
        let num_masks = batch.audio_slices.len() * self.num_speakers;
        let mask_refs: Vec<&[f32]> = batch
            .flat_masks
            .chunks(batch.mask_stride)
            .take(num_masks)
            .collect();

        let predict_start = std::time::Instant::now();
        let batch_embeddings = embedding_model.embed_multi_mask_batch(&fbank_refs, &mask_refs)?;

        for (fbank_idx, &chunk_idx) in batch.chunk_indices.iter().enumerate() {
            for speaker_idx in 0..self.num_speakers {
                let mask_idx = fbank_idx * self.num_speakers + speaker_idx;
                if !batch.active_flags[mask_idx] {
                    continue;
                }
                self.store_embedding_row(
                    storage,
                    chunk_idx,
                    speaker_idx,
                    batch_embeddings.row(mask_idx),
                );
            }
        }
        let predict_us = predict_start.elapsed().as_micros() as u64;

        Ok((fbank_us, predict_us))
    }

    pub fn run_masked(
        &self,
        receiver: crossbeam_channel::Receiver<Array2<f32>>,
        embedding_model: &mut EmbeddingModel,
        batch_size: usize,
    ) -> Result<ConcurrentEmbeddingResult, PipelineError> {
        let total_windows = self.total_windows();
        let mut seg_array: Option<Array3<f32>> = None;
        let mut emb_array: Option<Array3<f32>> = None;
        let mut pending: Vec<PendingEmbedding<'_>> = Vec::with_capacity(batch_size);
        let mut chunk_idx = 0usize;
        let mut emb_calls = 0u32;
        let mut emb_batched = 0u32;
        let mut emb_single = 0u32;
        let mut total_speakers = 0u32;
        let mut skipped_speakers = 0u32;
        let mut channel_wait = std::time::Duration::ZERO;
        let mut decode_time = std::time::Duration::ZERO;
        let mut embed_time = std::time::Duration::ZERO;
        let emb_start = std::time::Instant::now();

        loop {
            let recv_start = std::time::Instant::now();
            let raw_window = match receiver.recv() {
                Ok(w) => w,
                Err(_) => break,
            };
            channel_wait += recv_start.elapsed();

            let decode_start = std::time::Instant::now();
            let decoded = self.powerset.hard_decode(&raw_window);
            let seg = seg_array.get_or_insert_with(|| {
                Array3::zeros((total_windows, decoded.nrows(), self.num_speakers))
            });
            seg.slice_mut(s![chunk_idx, .., ..]).assign(&decoded);
            drop(decoded);

            let seg_view = seg.slice(s![chunk_idx, .., ..]);
            let chunk_audio = chunk_audio_raw(
                self.audio,
                self.step_samples,
                self.window_samples,
                chunk_idx,
            );
            let clean = clean_masks(&seg_view);

            for speaker_idx in 0..self.num_speakers {
                total_speakers += 1;
                let mask_col = seg_view.column(speaker_idx);
                let activity: f32 = mask_col.iter().sum();
                if activity < MIN_SPEAKER_ACTIVITY {
                    skipped_speakers += 1;
                    continue;
                }

                pending.push(PendingEmbedding {
                    chunk_idx,
                    speaker_idx,
                    audio: chunk_audio,
                    mask: mask_col.to_vec(),
                    clean_mask: clean.column(speaker_idx).to_vec(),
                });
                if pending.len() == batch_size {
                    decode_time += decode_start.elapsed();
                    let flush_start = std::time::Instant::now();
                    let emb = emb_array.get_or_insert_with(|| {
                        Array3::from_elem((total_windows, self.num_speakers, 256), f32::NAN)
                    });
                    flush_masked(embedding_model, &pending, &mut Array3Writer(emb))?;
                    embed_time += flush_start.elapsed();
                    emb_calls += 1;
                    emb_batched += 1;
                    pending.clear();
                }
            }
            decode_time += decode_start.elapsed();
            chunk_idx += 1;
        }

        while !pending.is_empty() {
            let batch_len = embedding_model.best_batch_len(pending.len());
            let flush_start = std::time::Instant::now();
            let emb = emb_array.get_or_insert_with(|| {
                Array3::from_elem((total_windows, self.num_speakers, 256), f32::NAN)
            });
            flush_masked(
                embedding_model,
                &pending[..batch_len],
                &mut Array3Writer(emb),
            )?;
            embed_time += flush_start.elapsed();
            emb_calls += 1;
            emb_single += 1;
            pending.drain(..batch_len);
        }

        let total_emb = emb_start.elapsed();
        info!(
            chunks = chunk_idx,
            total_speakers,
            skipped_speakers,
            active_speakers = total_speakers - skipped_speakers,
            emb_calls,
            emb_batched,
            emb_single,
            channel_wait_ms = channel_wait.as_millis(),
            decode_ms = decode_time.as_millis(),
            embed_ms = embed_time.as_millis(),
            total_emb_ms = total_emb.as_millis(),
            "Embedding thread profile"
        );

        self.finalize(seg_array, emb_array, chunk_idx, total_windows)
    }

    fn finalize(
        &self,
        seg_array: Option<Array3<f32>>,
        emb_array: Option<Array3<f32>>,
        chunk_idx: usize,
        total_windows: usize,
    ) -> Result<ConcurrentEmbeddingResult, PipelineError> {
        if chunk_idx == 0 {
            return Ok(ConcurrentEmbeddingResult {
                segmentations: Array3::zeros((0, 0, 0)),
                embeddings: Array3::from_elem((0, 0, 0), f32::NAN),
                num_chunks: 0,
            });
        }

        debug_assert_eq!(
            chunk_idx, total_windows,
            "streaming window count mismatch: got {chunk_idx}, expected {total_windows}"
        );

        let seg = self.require_segmentations(seg_array, chunk_idx)?;
        // all speakers inactive → no flush happened, emb_array stays None
        let emb = emb_array
            .unwrap_or_else(|| Array3::from_elem((chunk_idx, self.num_speakers, 256), f32::NAN));

        // truncate if fewer windows arrived than predicted (shouldn't happen)
        let (seg, emb) = if chunk_idx < seg.shape()[0] {
            (
                seg.slice(s![..chunk_idx, .., ..]).to_owned(),
                emb.slice(s![..chunk_idx, .., ..]).to_owned(),
            )
        } else {
            (seg, emb)
        };

        Ok(ConcurrentEmbeddingResult {
            segmentations: seg,
            embeddings: emb,
            num_chunks: chunk_idx,
        })
    }

    fn require_num_frames(
        &self,
        num_frames: Option<usize>,
        chunk_idx: usize,
    ) -> Result<usize, PipelineError> {
        num_frames.ok_or_else(|| {
            PipelineError::Invariant(format!(
                "multi-mask path buffered audio without any decoded frames at chunk {chunk_idx}"
            ))
        })
    }

    fn require_segmentations(
        &self,
        segmentations: Option<Array3<f32>>,
        chunk_idx: usize,
    ) -> Result<Array3<f32>, PipelineError> {
        segmentations.ok_or_else(|| {
            PipelineError::Invariant(format!(
                "embedding path processed {chunk_idx} chunks without storing segmentations"
            ))
        })
    }

    fn store_embedding_row<S: EmbeddingStorage>(
        &self,
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
}
