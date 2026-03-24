use ndarray::{Array2, Array3, ArrayView2, s};
use tracing::{info, trace};

use crate::inference::embedding::{EmbeddingModel, MaskedEmbeddingInput, SplitTailInput};
use crate::powerset::PowersetMapping;

use super::config::*;
use super::types::*;
use super::{clean_masks, select_speaker_weights};

pub(super) struct ConcurrentEmbeddingResult {
    pub decoded_windows: Vec<Array2<f32>>,
    pub embeddings: Vec<SpeakerEmbedding>,
    pub num_speakers: usize,
}

impl ConcurrentEmbeddingResult {
    pub fn is_empty(&self) -> bool {
        self.decoded_windows.is_empty()
    }

    pub fn into_arrays(self) -> (Array3<f32>, Array3<f32>) {
        let num_chunks = self.decoded_windows.len();
        let num_frames = self.decoded_windows[0].nrows();

        let mut segmentations = Array3::<f32>::zeros((num_chunks, num_frames, self.num_speakers));
        for (i, w) in self.decoded_windows.iter().enumerate() {
            segmentations.slice_mut(s![i, .., ..]).assign(w);
        }

        let mut embeddings =
            Array3::<f32>::from_elem((num_chunks, self.num_speakers, 256), f32::NAN);
        for emb in &self.embeddings {
            embeddings
                .slice_mut(s![emb.chunk_idx, emb.speaker_idx, ..])
                .assign(&ndarray::ArrayView1::from(emb.embedding.as_slice()));
        }

        (segmentations, embeddings)
    }
}

pub(super) struct ConcurrentEmbeddingRunner<'a> {
    pub powerset: &'a PowersetMapping,
    pub audio: &'a [f32],
    pub step_samples: usize,
    pub window_samples: usize,
    pub num_speakers: usize,
}

impl<'a> ConcurrentEmbeddingRunner<'a> {
    fn decode_chunk<'chunk>(
        &self,
        raw_window: &Array2<f32>,
        decoded_windows: &'chunk mut Vec<Array2<f32>>,
        chunk_idx: usize,
    ) -> (ArrayView2<'chunk, f32>, &'a [f32], Array2<f32>) {
        decoded_windows.push(self.powerset.hard_decode(raw_window));
        let segmentation_view = decoded_windows.last().unwrap().view();
        let chunk_audio = chunk_audio_raw(
            self.audio,
            self.step_samples,
            self.window_samples,
            chunk_idx,
        );
        let clean_masks = clean_masks(&segmentation_view);
        (segmentation_view, chunk_audio, clean_masks)
    }

    pub fn run_split(
        &self,
        receiver: crossbeam_channel::Receiver<Array2<f32>>,
        embedding_model: &mut EmbeddingModel,
        batch_size: usize,
        min_num_samples: usize,
    ) -> Result<ConcurrentEmbeddingResult, PipelineError> {
        let mut decoded_windows: Vec<Array2<f32>> = Vec::new();
        let mut embeddings: Vec<SpeakerEmbedding> = Vec::new();
        let mut pending: Vec<PendingSplitEmbedding> = Vec::with_capacity(batch_size);
        let mut fbanks: Vec<Array2<f32>> = Vec::new();
        let mut chunk_idx = 0usize;

        for raw_window in receiver {
            let (segmentation_view, chunk_audio, clean_masks) =
                self.decode_chunk(&raw_window, &mut decoded_windows, chunk_idx);
            let fbank = embedding_model.compute_chunk_fbank(chunk_audio)?;
            fbanks.push(fbank);
            let mut current_fbank_idx = fbanks.len() - 1;

            for speaker_idx in 0..self.num_speakers {
                let Some(weights) = select_speaker_weights(
                    &segmentation_view,
                    &clean_masks,
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
                    self.flush_split_pending(embedding_model, &pending, &fbanks, &mut embeddings)?;
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
            self.flush_split_pending(embedding_model, &pending, &fbanks, &mut embeddings)?;
        }

        Ok(ConcurrentEmbeddingResult {
            decoded_windows,
            embeddings,
            num_speakers: self.num_speakers,
        })
    }

    pub fn run_multi_mask(
        &self,
        receiver: crossbeam_channel::Receiver<Array2<f32>>,
        embedding_model: &mut EmbeddingModel,
        batch_size: usize,
        min_num_samples: usize,
    ) -> Result<ConcurrentEmbeddingResult, PipelineError> {
        let mut decoded_windows: Vec<Array2<f32>> = Vec::new();
        let mut embeddings: Vec<SpeakerEmbedding> = Vec::new();
        // buffer audio slices instead of pre-computed fbanks — fbanks are computed
        // in a single batched call per flush, avoiding redundant per-window computation
        let mut audio_buffer: Vec<&[f32]> = Vec::with_capacity(batch_size);
        let mut masks_buffer: Vec<Vec<f32>> = Vec::with_capacity(batch_size * self.num_speakers);
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
            let (segmentation_view, chunk_audio, clean_masks) =
                self.decode_chunk(&raw_window, &mut decoded_windows, chunk_idx);
            audio_buffer.push(chunk_audio);
            chunk_indices.push(chunk_idx);

            for speaker_idx in 0..self.num_speakers {
                let Some(weights) = select_speaker_weights(
                    &segmentation_view,
                    &clean_masks,
                    speaker_idx,
                    chunk_audio.len(),
                    min_num_samples,
                ) else {
                    masks_buffer.push(vec![0.0; 589]);
                    continue;
                };
                masks_buffer.push(weights);
            }
            total_decode_us += decode_start.elapsed().as_micros() as u64;

            if audio_buffer.len() == batch_size {
                let (fbank_us, gpu_us) = self.flush_multi_mask_timed(
                    embedding_model,
                    &audio_buffer,
                    &masks_buffer,
                    &chunk_indices,
                    &mut embeddings,
                )?;
                total_fbank_us += fbank_us;
                total_gpu_predict_us += gpu_us;
                flush_count += 1;
                audio_buffer.clear();
                masks_buffer.clear();
                chunk_indices.clear();
            }
            chunk_idx += 1;
        }

        if !audio_buffer.is_empty() {
            let (fbank_us, gpu_us) = self.flush_multi_mask_timed(
                embedding_model,
                &audio_buffer,
                &masks_buffer,
                &chunk_indices,
                &mut embeddings,
            )?;
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

        Ok(ConcurrentEmbeddingResult {
            decoded_windows,
            embeddings,
            num_speakers: self.num_speakers,
        })
    }

    fn flush_multi_mask_timed(
        &self,
        embedding_model: &mut EmbeddingModel,
        audio_slices: &[&[f32]],
        masks: &[Vec<f32>],
        chunk_indices: &[usize],
        embeddings: &mut Vec<SpeakerEmbedding>,
    ) -> Result<(u64, u64), PipelineError> {
        let fbank_start = std::time::Instant::now();
        let fbanks = embedding_model.compute_chunk_fbanks_batch(audio_slices)?;
        let fbank_us = fbank_start.elapsed().as_micros() as u64;

        let fbank_refs: Vec<_> = fbanks.iter().collect();
        let mask_refs: Vec<_> = masks.iter().map(|m| m.as_slice()).collect();

        let predict_start = std::time::Instant::now();
        let batch_embeddings = embedding_model.embed_multi_mask_batch(&fbank_refs, &mask_refs)?;
        let predict_us = predict_start.elapsed().as_micros() as u64;

        for (fbank_idx, &chunk_idx) in chunk_indices.iter().enumerate() {
            for speaker_idx in 0..self.num_speakers {
                let mask_idx = fbank_idx * self.num_speakers + speaker_idx;
                let is_active = masks[mask_idx].iter().any(|&v| v > 0.0);
                if !is_active {
                    continue;
                }
                embeddings.push(SpeakerEmbedding {
                    chunk_idx,
                    speaker_idx,
                    embedding: batch_embeddings.row(mask_idx).to_vec(),
                });
            }
        }

        Ok((fbank_us, predict_us))
    }

    fn flush_split_pending(
        &self,
        embedding_model: &mut EmbeddingModel,
        pending: &[PendingSplitEmbedding],
        fbanks: &[Array2<f32>],
        embeddings: &mut Vec<SpeakerEmbedding>,
    ) -> Result<(), PipelineError> {
        let batch_inputs: Vec<_> = pending
            .iter()
            .map(|item| SplitTailInput {
                fbank: &fbanks[item.fbank_idx],
                weights: &item.weights,
            })
            .collect();
        let batch_embeddings = embedding_model.embed_tail_batch_inputs(&batch_inputs)?;
        for (batch_idx, item) in pending.iter().enumerate() {
            embeddings.push(SpeakerEmbedding {
                chunk_idx: item.chunk_idx,
                speaker_idx: item.speaker_idx,
                embedding: batch_embeddings.row(batch_idx).to_vec(),
            });
        }
        Ok(())
    }

    pub fn run_masked(
        &self,
        receiver: crossbeam_channel::Receiver<Array2<f32>>,
        embedding_model: &mut EmbeddingModel,
        batch_size: usize,
    ) -> Result<ConcurrentEmbeddingResult, PipelineError> {
        let mut decoded_windows: Vec<Array2<f32>> = Vec::new();
        let mut embeddings: Vec<SpeakerEmbedding> = Vec::new();
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
            let (segmentation_view, chunk_audio, clean_masks) =
                self.decode_chunk(&raw_window, &mut decoded_windows, chunk_idx);

            for speaker_idx in 0..self.num_speakers {
                total_speakers += 1;
                let mask_col = segmentation_view.column(speaker_idx);
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
                    clean_mask: clean_masks.column(speaker_idx).to_vec(),
                });
                if pending.len() == batch_size {
                    decode_time += decode_start.elapsed();
                    let flush_start = std::time::Instant::now();
                    self.flush_masked_pending(embedding_model, &pending, &mut embeddings)?;
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
            self.flush_masked_pending(embedding_model, &pending[..batch_len], &mut embeddings)?;
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

        Ok(ConcurrentEmbeddingResult {
            decoded_windows,
            embeddings,
            num_speakers: self.num_speakers,
        })
    }

    fn flush_masked_pending(
        &self,
        embedding_model: &mut EmbeddingModel,
        pending: &[PendingEmbedding<'_>],
        embeddings: &mut Vec<SpeakerEmbedding>,
    ) -> Result<(), PipelineError> {
        let batch_inputs: Vec<_> = pending
            .iter()
            .map(|item| MaskedEmbeddingInput {
                audio: item.audio,
                mask: &item.mask,
                clean_mask: Some(&item.clean_mask),
            })
            .collect();
        let batch_embeddings = embedding_model.embed_batch(&batch_inputs)?;
        for (batch_idx, item) in pending.iter().enumerate() {
            embeddings.push(SpeakerEmbedding {
                chunk_idx: item.chunk_idx,
                speaker_idx: item.speaker_idx,
                embedding: batch_embeddings.row(batch_idx).to_vec(),
            });
        }
        Ok(())
    }
}
