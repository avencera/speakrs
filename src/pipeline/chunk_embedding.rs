use std::sync::Arc;

use crossbeam_channel::{Receiver, Sender};
use ndarray::Array2;
use tracing::{debug, info, trace};

use crate::inference::coreml::{CachedInputShape, SharedCoreMlModel};
use crate::inference::embedding::EmbeddingModel;
use crate::inference::segmentation::SegmentationModel;
use crate::powerset::PowersetMapping;

use super::types::*;
use super::{clean_masks, select_speaker_weights};

#[derive(Clone)]
struct ChunkSessionHandle {
    cached_fbank_shape: Arc<CachedInputShape>,
    cached_masks_shape: Arc<CachedInputShape>,
    model: Arc<SharedCoreMlModel>,
}

#[derive(Clone)]
struct ChunkEmbeddingResources {
    chunk_sessions: Vec<ChunkSessionHandle>,
    chunk_lookup: Vec<(usize, usize, usize)>,
    fbank_30s: Option<Arc<SharedCoreMlModel>>,
    fbank_10s: Option<Arc<SharedCoreMlModel>>,
}

struct DecodedChunk {
    global_start: usize,
    decoded_chunk: Vec<Array2<f32>>,
}

struct PreparedChunk {
    global_start: usize,
    decoded_chunk: Vec<Array2<f32>>,
    fbank: Vec<f32>,
    masks: Vec<f32>,
    active: Vec<(usize, usize)>,
    num_masks: usize,
}

struct EmbeddedChunk {
    global_start: usize,
    decoded_chunk: Vec<Array2<f32>>,
    data: Vec<f32>,
    active: Vec<(usize, usize)>,
    num_masks: usize,
    predict_us: u64,
}

#[derive(Default)]
struct PrepStats {
    chunks: u32,
    fbank_us: u64,
}

struct GpuStats {
    predict_us: u64,
    chunks: u32,
    self_prep_us: u64,
}

/// Common parameters for chunk embedding execution
struct ChunkParams {
    step_samples: usize,
    window_samples: usize,
    num_speakers: usize,
    min_num_samples: usize,
    chunk_win_capacity: usize,
    total_windows: usize,
}

/// Timing summary from either pipelined or sequential execution
struct EmbeddingSummary {
    decoded_all: Vec<Array2<f32>>,
    embeddings_vec: Vec<SpeakerEmbedding>,
    gpu_predict_us: u64,
    prep_fbank_us: u64,
    prep_mask_us: u64,
}

// --- ChunkPrep actor: encapsulates the 14 parameters of prep_decoded_chunk ---

#[derive(Clone)]
struct ChunkPrep {
    step_samples: usize,
    window_samples: usize,
    num_speakers: usize,
    min_num_samples: usize,
    largest_fbank_frames: usize,
    largest_num_masks: usize,
    max_active: usize,
    fbank_30s: Option<Arc<SharedCoreMlModel>>,
    fbank_10s: Option<Arc<SharedCoreMlModel>>,
}

/// Per-worker mutable scratch buffers for chunk prep
struct PrepScratch {
    fbank_30s_buf: Vec<f32>,
    waveform_10s_buf: Vec<f32>,
    fbank_30s_shape: CachedInputShape,
    fbank_10s_shape: CachedInputShape,
}

impl PrepScratch {
    fn new(window_samples: usize) -> Self {
        Self {
            fbank_30s_buf: vec![0.0f32; 480_000],
            waveform_10s_buf: vec![0.0f32; window_samples],
            fbank_30s_shape: CachedInputShape::new("waveform", &[1, 1, 480_000]),
            fbank_10s_shape: CachedInputShape::new("waveform", &[1, 1, window_samples]),
        }
    }
}

impl ChunkPrep {
    fn prep(
        &self,
        decoded: &DecodedChunk,
        audio: &[f32],
        scratch: &mut PrepScratch,
    ) -> Result<PreparedChunk, PipelineError> {
        let global_start = decoded.global_start;
        let wins = decoded.decoded_chunk.len();
        let chunk_audio_start = global_start * self.step_samples;
        let chunk_audio_len = self.window_samples + (wins - 1) * self.step_samples;
        let chunk_audio_end = (chunk_audio_start + chunk_audio_len).min(audio.len());
        let chunk_audio = &audio[chunk_audio_start..chunk_audio_end];

        let mut fbank = vec![0.0f32; self.largest_fbank_frames * 80];

        if chunk_audio.len() <= 480_000 {
            if let Some(fbank_model) = &self.fbank_30s {
                scratch.fbank_30s_buf[..chunk_audio.len()].copy_from_slice(chunk_audio);
                scratch.fbank_30s_buf[chunk_audio.len()..].fill(0.0);
                let (data, out_shape) = fbank_model
                    .predict_cached(&[(&scratch.fbank_30s_shape, &*scratch.fbank_30s_buf)])
                    .map_err(|e| PipelineError::Other(e.to_string()))?;
                let copy_frames = out_shape[1].min(self.largest_fbank_frames);
                for r in 0..copy_frames {
                    let off = r * 80;
                    fbank[off..off + 80].copy_from_slice(&data[off..off + 80]);
                }
            }
        } else if let Some(fbank_model) = &self.fbank_10s {
            let mut fb_off = 0usize;
            let mut au_off = 0usize;
            while fb_off < self.largest_fbank_frames && au_off < chunk_audio.len() {
                let seg_end = (au_off + self.window_samples).min(chunk_audio.len());
                let seg_len = seg_end - au_off;
                scratch.waveform_10s_buf[..seg_len].copy_from_slice(&chunk_audio[au_off..seg_end]);
                if seg_len < self.window_samples {
                    scratch.waveform_10s_buf[seg_len..].fill(0.0);
                }
                let (data, out_shape) = fbank_model
                    .predict_cached(&[(&scratch.fbank_10s_shape, &*scratch.waveform_10s_buf)])
                    .map_err(|e| PipelineError::Other(e.to_string()))?;
                let copy = out_shape[1].min(self.largest_fbank_frames - fb_off);
                for r in 0..copy {
                    let src = r * 80;
                    let dst = (fb_off + r) * 80;
                    fbank[dst..dst + 80].copy_from_slice(&data[src..src + 80]);
                }
                fb_off += 998;
                au_off += self.window_samples;
            }
        }

        let mut masks = vec![0.0f32; self.largest_num_masks * 589];
        let mut active: Vec<(usize, usize)> = Vec::with_capacity(self.max_active);
        for (local, dec) in decoded.decoded_chunk.iter().enumerate() {
            let global_idx = global_start + local;
            let win_audio =
                chunk_audio_raw(audio, self.step_samples, self.window_samples, global_idx);
            let clean = clean_masks(&dec.view());
            for speaker_idx in 0..self.num_speakers {
                let mask_idx = local * self.num_speakers + speaker_idx;
                if mask_idx >= self.largest_num_masks {
                    break;
                }
                if let Some(weights) = select_speaker_weights(
                    &dec.view(),
                    &clean,
                    speaker_idx,
                    win_audio.len(),
                    self.min_num_samples,
                ) {
                    let dst = mask_idx * 589;
                    let cl = weights.len().min(589);
                    masks[dst..dst + cl].copy_from_slice(&weights[..cl]);
                    active.push((local, speaker_idx));
                }
            }
        }

        Ok(PreparedChunk {
            global_start,
            decoded_chunk: decoded.decoded_chunk.clone(),
            fbank,
            masks,
            active,
            num_masks: self.largest_num_masks,
        })
    }
}

// --- PrepWorker actor: runs on CPU prep threads ---

struct PrepWorker {
    prep: ChunkPrep,
    scratch: PrepScratch,
}

impl PrepWorker {
    fn run(
        mut self,
        audio: &[f32],
        step_samples: usize,
        window_samples: usize,
        chunk_rx: &Receiver<DecodedChunk>,
        prep_tx: Sender<PreparedChunk>,
    ) -> Result<PrepStats, PipelineError> {
        let mut stats = PrepStats::default();

        while let Ok(decoded) = chunk_rx.recv() {
            let chunk_audio_start = decoded.global_start * step_samples;
            if chunk_audio_start + window_samples > audio.len() {
                continue;
            }
            let prep_start = std::time::Instant::now();
            let prepared = self.prep.prep(&decoded, audio, &mut self.scratch)?;
            stats.fbank_us += prep_start.elapsed().as_micros() as u64;
            stats.chunks += 1;
            if prep_tx.send(prepared).is_err() {
                break;
            }
        }
        Ok(stats)
    }
}

// --- GpuWorker actor: work-stealing GPU predict loop ---

struct GpuWorker {
    model: Arc<SharedCoreMlModel>,
    fbank_shape: Arc<CachedInputShape>,
    masks_shape: Arc<CachedInputShape>,
    prep: ChunkPrep,
    scratch: PrepScratch,
}

impl GpuWorker {
    /// Try to get the next prepared chunk, falling back to self-prep from decoded queue
    fn next_prepared(
        &mut self,
        audio: &[f32],
        prep_rx: &Receiver<PreparedChunk>,
        chunk_rx: &Receiver<DecodedChunk>,
        decoded_done: &mut bool,
        total_prep_us: &mut u64,
    ) -> Result<Option<PreparedChunk>, PipelineError> {
        match prep_rx.try_recv() {
            Ok(p) => return Ok(Some(p)),
            Err(crossbeam_channel::TryRecvError::Disconnected) => return Ok(None),
            Err(crossbeam_channel::TryRecvError::Empty) => {}
        }

        if *decoded_done {
            return match prep_rx.recv() {
                Ok(p) => Ok(Some(p)),
                Err(_) => Ok(None),
            };
        }

        match chunk_rx.try_recv() {
            Ok(decoded) => {
                let prep_start = std::time::Instant::now();
                let p = self.prep.prep(&decoded, audio, &mut self.scratch)?;
                *total_prep_us += prep_start.elapsed().as_micros() as u64;
                Ok(Some(p))
            }
            Err(crossbeam_channel::TryRecvError::Empty) => {
                crossbeam_channel::select! {
                    recv(prep_rx) -> msg => match msg {
                        Ok(p) => Ok(Some(p)),
                        Err(_) => Ok(None),
                    },
                    recv(chunk_rx) -> msg => match msg {
                        Ok(decoded) => {
                            let prep_start = std::time::Instant::now();
                            let p = self.prep.prep(&decoded, audio, &mut self.scratch)?;
                            *total_prep_us += prep_start.elapsed().as_micros() as u64;
                            Ok(Some(p))
                        }
                        Err(_) => {
                            *decoded_done = true;
                            match prep_rx.recv() {
                                Ok(p) => Ok(Some(p)),
                                Err(_) => Ok(None),
                            }
                        }
                    },
                }
            }
            Err(crossbeam_channel::TryRecvError::Disconnected) => {
                *decoded_done = true;
                match prep_rx.recv() {
                    Ok(p) => Ok(Some(p)),
                    Err(_) => Ok(None),
                }
            }
        }
    }

    fn predict(&self, prepared: &PreparedChunk) -> Result<(Vec<f32>, u64), PipelineError> {
        let predict_start = std::time::Instant::now();
        let (data, _) = self
            .model
            .predict_cached(&[
                (&*self.fbank_shape, &prepared.fbank),
                (&*self.masks_shape, &prepared.masks),
            ])
            .map_err(|e| PipelineError::Other(e.to_string()))?;
        let predict_us = predict_start.elapsed().as_micros() as u64;
        Ok((data, predict_us))
    }

    fn run(
        mut self,
        audio: &[f32],
        prep_rx: Receiver<PreparedChunk>,
        chunk_rx: Receiver<DecodedChunk>,
        emb_tx: Sender<EmbeddedChunk>,
    ) -> Result<GpuStats, PipelineError> {
        let mut total_predict_us = 0u64;
        let mut total_prep_us = 0u64;
        let mut chunk_num = 0u32;
        let mut decoded_done = false;

        loop {
            let Some(prepared) = self.next_prepared(
                audio,
                &prep_rx,
                &chunk_rx,
                &mut decoded_done,
                &mut total_prep_us,
            )?
            else {
                break;
            };

            let (data, predict_us) = self.predict(&prepared)?;
            total_predict_us += predict_us;

            trace!(
                chunk_num,
                chunk_start = prepared.global_start,
                predict_ms = predict_us / 1000,
                "GPU chunk"
            );
            chunk_num += 1;

            if emb_tx
                .send(EmbeddedChunk {
                    global_start: prepared.global_start,
                    decoded_chunk: prepared.decoded_chunk,
                    data,
                    active: prepared.active,
                    num_masks: prepared.num_masks,
                    predict_us,
                })
                .is_err()
            {
                break;
            }
        }
        Ok(GpuStats {
            predict_us: total_predict_us,
            chunks: chunk_num,
            self_prep_us: total_prep_us,
        })
    }
}

// --- Resource construction ---

fn chunk_embedding_resources(emb_model: &mut EmbeddingModel) -> Option<ChunkEmbeddingResources> {
    let bundle = emb_model.prepare_chunk_resources()?;

    let chunk_sessions = bundle
        .sessions
        .iter()
        .map(|s| ChunkSessionHandle {
            cached_fbank_shape: Arc::clone(&s.cached_fbank_shape),
            cached_masks_shape: Arc::clone(&s.cached_masks_shape),
            model: Arc::clone(&s.model),
        })
        .collect();

    let chunk_lookup = bundle
        .sessions
        .iter()
        .map(|s| (s.num_windows, s.fbank_frames, s.num_masks))
        .collect();

    Some(ChunkEmbeddingResources {
        chunk_sessions,
        chunk_lookup,
        fbank_30s: bundle.fbank_30s,
        fbank_10s: bundle.fbank_10s,
    })
}

fn build_chunk_artifacts(
    step_seconds: f64,
    step_samples: usize,
    window_samples: usize,
    num_speakers: usize,
    decoded_all: Vec<Array2<f32>>,
    embeddings_vec: Vec<SpeakerEmbedding>,
) -> Option<InferenceArtifacts> {
    let num_chunks = decoded_all.len();
    let (segmentations, embeddings) =
        build_inference_arrays(decoded_all, embeddings_vec, num_speakers)?;
    Some(InferenceArtifacts {
        layout: ChunkLayout::new(step_seconds, step_samples, window_samples, num_chunks),
        segmentations: DecodedSegmentations(segmentations),
        embeddings: ChunkEmbeddings(embeddings),
    })
}

/// Number of parallel segmentation workers for CoreML
fn seg_worker_count() -> usize {
    std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(4)
        .min(8)
}

// --- Setup ---

/// Check capabilities and build resources. Returns None if chunk embedding is not possible
fn setup_chunk_embedding(
    seg_model: &SegmentationModel,
    emb_model: &mut EmbeddingModel,
    audio: &[f32],
) -> Option<(usize, usize, bool, Option<ChunkEmbeddingResources>)> {
    let window_samples = seg_model.window_samples();
    if audio.len() < window_samples {
        return None;
    }

    let chunk_win_capacity = emb_model.chunk_window_capacity()?;
    let step_samples = seg_model.step_samples();
    let total_windows = audio.len().saturating_sub(window_samples) / step_samples + 1;
    let est_chunks = total_windows.div_ceil(chunk_win_capacity);
    let use_pipelined = est_chunks >= 2;

    let resources = use_pipelined
        .then(|| chunk_embedding_resources(emb_model))
        .flatten();

    Some((chunk_win_capacity, total_windows, use_pipelined, resources))
}

// --- Pipelined execution ---

/// Pipelined multi-chunk execution: seg -> bridge -> prep -> GPU -> collect
fn run_pipelined<'scope>(
    scope: &'scope std::thread::Scope<'scope, '_>,
    emb_start: std::time::Instant,
    resources: ChunkEmbeddingResources,
    chunk_rx: &'scope Receiver<DecodedChunk>,
    audio: &'scope [f32],
    params: &ChunkParams,
) -> Result<EmbeddingSummary, PipelineError> {
    let largest_session = resources.chunk_sessions.last().unwrap().clone();
    let largest_fbank_frames = resources.chunk_lookup.last().unwrap().1;
    let largest_num_masks = resources.chunk_lookup.last().unwrap().2;
    let max_active = params.chunk_win_capacity * params.num_speakers;

    let step_samples = params.step_samples;
    let window_samples = params.window_samples;
    let num_speakers = params.num_speakers;

    let prep_config = ChunkPrep {
        step_samples,
        window_samples,
        num_speakers,
        min_num_samples: params.min_num_samples,
        largest_fbank_frames,
        largest_num_masks,
        max_active,
        fbank_30s: resources.fbank_30s.clone(),
        fbank_10s: resources.fbank_10s.clone(),
    };

    let (prep_tx, prep_rx) = crossbeam_channel::bounded::<PreparedChunk>(48);
    let (emb_tx, emb_rx) = crossbeam_channel::bounded::<EmbeddedChunk>(8);

    // CPU prep threads
    let num_prep_threads = 2usize;
    let mut prep_handles = Vec::with_capacity(num_prep_threads);
    for _prep_id in 0..num_prep_threads {
        let prep_tx = prep_tx.clone();
        let worker = PrepWorker {
            prep: prep_config.clone(),
            scratch: PrepScratch::new(window_samples),
        };
        prep_handles.push(scope.spawn(move || -> Result<PrepStats, PipelineError> {
            worker.run(audio, step_samples, window_samples, chunk_rx, prep_tx)
        }));
    }
    drop(prep_tx);

    // GPU worker
    let gpu_emb_tx = emb_tx.clone();
    let gpu_prep_rx = prep_rx;
    let gpu_chunk_rx = chunk_rx.clone();
    let gpu_worker = GpuWorker {
        model: largest_session.model,
        fbank_shape: largest_session.cached_fbank_shape,
        masks_shape: largest_session.cached_masks_shape,
        prep: prep_config,
        scratch: PrepScratch::new(window_samples),
    };
    let gpu_handle = scope.spawn(move || -> Result<GpuStats, PipelineError> {
        gpu_worker.run(audio, gpu_prep_rx, gpu_chunk_rx, gpu_emb_tx)
    });

    drop(emb_tx);

    // collect results (out-of-order from workers)
    let mut decoded_slots: Vec<Option<Array2<f32>>> = std::iter::repeat_with(|| None)
        .take(params.total_windows + params.chunk_win_capacity)
        .collect();
    let mut embeddings_vec: Vec<SpeakerEmbedding> = Vec::new();
    let mut total_predict_us = 0u64;
    let mut total_chunks = 0u32;

    while let Ok(embedded) = emb_rx.recv() {
        total_predict_us += embedded.predict_us;

        let batch_emb = Array2::from_shape_vec((embedded.num_masks, 256), embedded.data).unwrap();

        for &(local, speaker_idx) in &embedded.active {
            let mask_idx = local * num_speakers + speaker_idx;
            embeddings_vec.push(SpeakerEmbedding {
                chunk_idx: embedded.global_start + local,
                speaker_idx,
                embedding: batch_emb.row(mask_idx).to_vec(),
            });
        }

        for (local, decoded) in embedded.decoded_chunk.into_iter().enumerate() {
            let slot = embedded.global_start + local;
            if slot < decoded_slots.len() {
                decoded_slots[slot] = Some(decoded);
            }
        }

        total_chunks += 1;
    }

    let gpu_stats = gpu_handle.join().unwrap()?;

    let mut total_prep_fbank_us = 0u64;
    for handle in prep_handles {
        let stats = handle.join().unwrap()?;
        total_prep_fbank_us += stats.fbank_us;
    }

    // reconstruct ordered decoded_all from slots, remapping
    // embedding chunk_idx from absolute window position to sequential index
    let mut abs_to_seq: Vec<usize> = vec![0; decoded_slots.len()];
    let mut seq_idx = 0usize;
    for (abs_idx, slot) in decoded_slots.iter().enumerate() {
        if slot.is_some() {
            abs_to_seq[abs_idx] = seq_idx;
            seq_idx += 1;
        }
    }
    let decoded_all: Vec<Array2<f32>> = decoded_slots.into_iter().flatten().collect();
    for emb in &mut embeddings_vec {
        emb.chunk_idx = abs_to_seq[emb.chunk_idx];
    }

    debug!(
        total_chunks,
        gpu_chunks = gpu_stats.chunks,
        gpu_predict_ms = gpu_stats.predict_us / 1000,
        gpu_self_prep_ms = gpu_stats.self_prep_us / 1000,
        cpu_prep_ms = total_prep_fbank_us / 1000,
        predict_ms = total_predict_us / 1000,
        emb_wall_ms = emb_start.elapsed().as_millis(),
        "Work-stealing breakdown"
    );

    Ok(EmbeddingSummary {
        decoded_all,
        embeddings_vec,
        gpu_predict_us: total_predict_us,
        prep_fbank_us: total_prep_fbank_us + gpu_stats.self_prep_us,
        prep_mask_us: 0,
    })
}

// --- Sequential fallback ---

/// Sequential fallback for short audio (single chunk)
fn run_sequential_chunks(
    emb_model: &mut EmbeddingModel,
    chunk_rx: &Receiver<DecodedChunk>,
    audio: &[f32],
    params: &ChunkParams,
    emb_start: std::time::Instant,
) -> Result<EmbeddingSummary, PipelineError> {
    let step_samples = params.step_samples;
    let window_samples = params.window_samples;
    let num_speakers = params.num_speakers;
    let min_num_samples = params.min_num_samples;

    let mut decoded_all: Vec<Array2<f32>> = Vec::new();
    let mut embeddings_vec: Vec<SpeakerEmbedding> = Vec::new();
    let mut seq_fbank_us = 0u64;
    let mut seq_mask_us = 0u64;
    let mut seq_predict_us = 0u64;
    let mut seq_chunks = 0u32;

    for DecodedChunk {
        global_start,
        decoded_chunk,
    } in chunk_rx
    {
        let wins = decoded_chunk.len();
        let session = emb_model.chunk_session_for_windows(wins).unwrap();
        let sess_fbank_frames = session.fbank_frames;
        let sess_num_masks = session.num_masks;
        let _ = session;

        let chunk_audio_start = global_start * step_samples;
        if chunk_audio_start + window_samples > audio.len() {
            continue;
        }
        let chunk_audio_len = window_samples + (wins - 1) * step_samples;
        let chunk_audio_end = (chunk_audio_start + chunk_audio_len).min(audio.len());
        let chunk_audio = &audio[chunk_audio_start..chunk_audio_end];

        let mut fbank = vec![0.0f32; sess_fbank_frames * 80];
        let fbank_start = std::time::Instant::now();

        if let Some(result) = emb_model.compute_chunk_fbank_30s(chunk_audio) {
            let full_fbank = result?;
            let copy_frames = full_fbank.nrows().min(sess_fbank_frames);
            for r in 0..copy_frames {
                let dst = r * 80;
                fbank[dst..dst + 80].copy_from_slice(full_fbank.row(r).as_slice().unwrap());
            }
        } else {
            let mut fb_off = 0usize;
            let mut au_off = 0usize;
            while fb_off < sess_fbank_frames && au_off < chunk_audio.len() {
                let seg_end = (au_off + window_samples).min(chunk_audio.len());
                let seg_fbank = emb_model.compute_chunk_fbank(&chunk_audio[au_off..seg_end])?;
                let copy = seg_fbank.nrows().min(sess_fbank_frames - fb_off);
                for r in 0..copy {
                    let dst = (fb_off + r) * 80;
                    fbank[dst..dst + 80].copy_from_slice(seg_fbank.row(r).as_slice().unwrap());
                }
                fb_off += 998;
                au_off += window_samples;
            }
        }

        seq_fbank_us += fbank_start.elapsed().as_micros() as u64;

        // masks
        let mask_start = std::time::Instant::now();
        let mut masks = vec![0.0f32; sess_num_masks * 589];
        let mut active: Vec<(usize, usize)> = Vec::new();
        for (local, decoded) in decoded_chunk.iter().enumerate() {
            let global_idx = global_start + local;
            let win_audio = chunk_audio_raw(audio, step_samples, window_samples, global_idx);
            let clean = clean_masks(&decoded.view());
            for speaker_idx in 0..num_speakers {
                let mask_idx = local * num_speakers + speaker_idx;
                if mask_idx >= sess_num_masks {
                    break;
                }
                if let Some(weights) = select_speaker_weights(
                    &decoded.view(),
                    &clean,
                    speaker_idx,
                    win_audio.len(),
                    min_num_samples,
                ) {
                    let dst = mask_idx * 589;
                    let cl = weights.len().min(589);
                    masks[dst..dst + cl].copy_from_slice(&weights[..cl]);
                    active.push((local, speaker_idx));
                }
            }
        }

        seq_mask_us += mask_start.elapsed().as_micros() as u64;

        let predict_start = std::time::Instant::now();
        let session = emb_model.chunk_session_for_windows(wins).unwrap();
        let batch_emb = EmbeddingModel::embed_chunk_session(session, &fbank, &masks)?;
        seq_predict_us += predict_start.elapsed().as_micros() as u64;
        seq_chunks += 1;

        for &(local, speaker_idx) in &active {
            let mask_idx = local * num_speakers + speaker_idx;
            embeddings_vec.push(SpeakerEmbedding {
                chunk_idx: global_start + local,
                speaker_idx,
                embedding: batch_emb.row(mask_idx).to_vec(),
            });
        }

        decoded_all.extend(decoded_chunk);
    }

    trace!(
        seq_chunks,
        fbank_ms = seq_fbank_us / 1000,
        mask_ms = seq_mask_us / 1000,
        predict_ms = seq_predict_us / 1000,
        wall_ms = emb_start.elapsed().as_millis(),
        "EMB sequential",
    );

    Ok(EmbeddingSummary {
        decoded_all,
        embeddings_vec,
        gpu_predict_us: seq_predict_us,
        prep_fbank_us: seq_fbank_us,
        prep_mask_us: seq_mask_us,
    })
}

// --- Coordinator ---

/// Pipelined chunk embedding: seg (CPUOnly) and emb (GPU) run on separate
/// threads processing 30s chunks. While emb processes chunk N, seg produces
/// chunk N+1
pub(super) fn try_chunk_embedding(
    seg_model: &mut SegmentationModel,
    emb_model: &mut EmbeddingModel,
    powerset: &PowersetMapping,
    audio: &[f32],
) -> Result<Option<InferenceArtifacts>, PipelineError> {
    let Some((chunk_win_capacity, total_windows, use_pipelined, chunk_resources)) =
        setup_chunk_embedding(seg_model, emb_model, audio)
    else {
        return Ok(None);
    };

    let inference_start = std::time::Instant::now();
    let step_seconds = seg_model.step_seconds();
    let params = ChunkParams {
        step_samples: seg_model.step_samples(),
        window_samples: seg_model.window_samples(),
        num_speakers: 3,
        min_num_samples: emb_model.min_num_samples(),
        chunk_win_capacity,
        total_windows,
    };

    let (seg_tx, seg_rx) = crossbeam_channel::bounded::<Array2<f32>>(100);
    let (chunk_tx, chunk_rx) = crossbeam_channel::bounded::<DecodedChunk>(100);

    let result: Result<Option<InferenceArtifacts>, PipelineError> = std::thread::scope(|scope| {
        // seg thread: batched CPU-only segmentation feeding the bridge
        let seg_start = std::time::Instant::now();
        let seg_warm_start_windows = chunk_win_capacity;
        let seg_handle = scope.spawn(move || -> Result<std::time::Duration, PipelineError> {
            seg_model.run_streaming_parallel(
                audio,
                seg_tx,
                seg_worker_count(),
                Some(seg_warm_start_windows),
            )?;
            Ok(seg_start.elapsed())
        });

        // bridge thread: groups raw windows into decoded chunks
        let bridge_handle = scope.spawn(move || {
            let mut group: Vec<Array2<f32>> = Vec::with_capacity(chunk_win_capacity);
            let mut global_start = 0usize;

            for raw_window in &seg_rx {
                group.push(powerset.hard_decode(&raw_window));

                if group.len() == chunk_win_capacity {
                    if chunk_tx
                        .send(DecodedChunk {
                            global_start,
                            decoded_chunk: std::mem::take(&mut group),
                        })
                        .is_err()
                    {
                        break;
                    }
                    global_start += chunk_win_capacity;
                    group = Vec::with_capacity(chunk_win_capacity);
                }
            }
            if !group.is_empty() {
                let _ = chunk_tx.send(DecodedChunk {
                    global_start,
                    decoded_chunk: group,
                });
            }
        });

        // dispatch to pipelined or sequential execution
        let emb_start = std::time::Instant::now();
        let summary = if use_pipelined {
            let Some(resources) = chunk_resources else {
                return Ok(None);
            };

            run_pipelined(scope, emb_start, resources, &chunk_rx, audio, &params)?
        } else {
            run_sequential_chunks(emb_model, &chunk_rx, audio, &params, emb_start)?
        };
        let emb_elapsed = emb_start.elapsed();

        let seg_thread_elapsed = seg_handle.join().unwrap()?;
        bridge_handle.join().unwrap();
        trace!(
            seg_thread_ms = seg_thread_elapsed.as_millis(),
            seg_wall_ms = seg_start.elapsed().as_millis(),
            "SEG timing"
        );

        let num_chunks = summary.decoded_all.len();
        let Some(artifacts) = build_chunk_artifacts(
            step_seconds,
            params.step_samples,
            params.window_samples,
            params.num_speakers,
            summary.decoded_all,
            summary.embeddings_vec,
        ) else {
            return Ok(None);
        };

        let inference_elapsed = inference_start.elapsed();
        let audio_secs = audio.len() as f64 / 16_000.0;
        info!(
            chunks = num_chunks,
            chunk_capacity = params.chunk_win_capacity,
            pipelined = use_pipelined,
            seg_ms = seg_thread_elapsed.as_millis(),
            emb_ms = emb_elapsed.as_millis(),
            predict_ms = summary.gpu_predict_us / 1000,
            prep_fbank_ms = summary.prep_fbank_us / 1000,
            prep_mask_ms = summary.prep_mask_us / 1000,
            total_ms = inference_elapsed.as_millis(),
            audio_secs = audio_secs as u64,
            "Chunk embedding complete"
        );

        Ok(Some(artifacts))
    });

    result
}
