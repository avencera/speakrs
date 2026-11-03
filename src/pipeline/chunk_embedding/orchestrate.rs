use crossbeam_channel::Receiver;
use ndarray::{ArrayViewMut2, s};
use tracing::{debug, trace};

use super::collect::{CollectionPlan, FileCollector};
use super::fbank_plan::{ChunkFbankRequest, FbankModel, THIRTY_SECOND_SAMPLES};
use super::gpu::{ChunkEmbeddingResources, GpuWorker, chunk_embedding_resources};
use super::prep::{ChunkJob, ChunkPrep, PrepScratch, PrepWorker, SpeakerMaskLayout};
use super::{
    ChunkParams, EmbeddingModel, EmbeddingSummary, PipelineError, SegmentationModel,
    chunk_session_for_windows, join_scoped_result,
};
use crate::inference::embedding::FBANK_FEATURES;

pub(super) enum ChunkExecution {
    Sequential,
    Pipelined { resources: ChunkEmbeddingResources },
}

pub(super) struct ChunkExecutionPlan {
    pub layout: crate::pipeline::types::ChunkLayout,
    pub collection: CollectionPlan,
    pub execution: ChunkExecution,
}

impl ChunkExecutionPlan {
    pub(super) fn resolve(
        seg_model: &SegmentationModel,
        emb_model: &mut EmbeddingModel,
        audio: &[f32],
    ) -> Result<Option<Self>, PipelineError> {
        let window_samples = seg_model.window_samples();
        if audio.len() < window_samples {
            return Ok(None);
        }

        let Some(chunk_win_capacity) = emb_model.chunk_window_capacity() else {
            return Ok(None);
        };
        let total_windows = seg_model.window_count(audio.len());
        if total_windows == 0 {
            return Ok(None);
        }
        let est_chunks = total_windows.div_ceil(chunk_win_capacity);
        let collection = CollectionPlan::new(0, total_windows, chunk_win_capacity, 3)?;
        let execution = if chunk_schedule_is_pipelined(est_chunks) {
            match chunk_embedding_resources(emb_model)? {
                Some(resources) => ChunkExecution::Pipelined { resources },
                None => ChunkExecution::Sequential,
            }
        } else {
            ChunkExecution::Sequential
        };

        Ok(Some(Self {
            layout: crate::pipeline::types::ChunkLayout::from_spec(
                seg_model.window_spec(),
                seg_model.step_seconds(),
                0,
            ),
            collection,
            execution,
        }))
    }

    pub(super) const fn is_pipelined(&self) -> bool {
        matches!(self.execution, ChunkExecution::Pipelined { .. })
    }
}

fn chunk_schedule_is_pipelined(estimated_chunks: usize) -> bool {
    #[cfg(test)]
    {
        crate::pipeline::test_support::select_chunk_schedule(estimated_chunks)
    }
    #[cfg(not(test))]
    {
        estimated_chunks >= 2
    }
}

pub(super) fn run_pipelined<'scope>(
    scope: &'scope std::thread::Scope<'scope, '_>,
    emb_start: std::time::Instant,
    resources: ChunkEmbeddingResources,
    chunk_rx: Receiver<ChunkJob>,
    audio: &'scope [f32],
    params: &'scope ChunkParams,
    collection_plan: CollectionPlan,
) -> Result<EmbeddingSummary, PipelineError> {
    let largest = resources.largest_session()?.clone();
    let max_active = largest.num_windows * params.num_speakers;
    let prep_config = ChunkPrep {
        step_samples: params.step_samples,
        window_samples: params.window_samples,
        num_speakers: params.num_speakers,
        min_num_samples: params.min_num_samples,
        largest_fbank_frames: largest.fbank_frames,
        largest_num_masks: largest.num_masks,
        max_active,
        fbank_30s: resources.fbank_30s.clone(),
        fbank_10s: resources.fbank_10s.clone(),
        fbank_normalization_scope: params.fbank_normalization_scope,
    };
    let audios = [audio];

    let (prep_tx, prep_rx) = crossbeam_channel::bounded(48);
    let (emb_tx, emb_rx) = crossbeam_channel::bounded(8);

    let mut prep_handles = Vec::with_capacity(params.fbank_preparation_workers);
    for _ in 0..params.fbank_preparation_workers {
        let worker = PrepWorker {
            prep: prep_config.clone(),
            scratch: PrepScratch::new(params.window_samples),
        };
        let prep_tx = prep_tx.clone();
        let chunk_rx = chunk_rx.clone();
        prep_handles.push(scope.spawn(move || worker.run(&audios, chunk_rx, prep_tx)));
    }
    drop(prep_tx);

    let gpu_emb_tx = emb_tx.clone();
    let gpu_chunk_rx = chunk_rx.clone();
    let gpu_handle = scope.spawn(move || {
        GpuWorker {
            model: largest.handle.model,
            fbank_shape: largest.handle.cached_fbank_shape,
            masks_shape: largest.handle.cached_masks_shape,
            prep: prep_config,
            scratch: PrepScratch::new(params.window_samples),
        }
        .run(&audios, prep_rx, gpu_chunk_rx, gpu_emb_tx)
    });
    drop(chunk_rx);
    drop(emb_tx);

    let mut collector = FileCollector::new(collection_plan);
    let mut collect_error = None;
    let mut total_predict_us = 0u64;
    let mut total_chunks = 0u32;
    while let Ok(embedded) = emb_rx.recv() {
        total_predict_us += embedded.predict_us;
        total_chunks += 1;
        if let Err(error) = collector.add(embedded) {
            collect_error = Some(error);
            break;
        }
    }
    drop(emb_rx);

    let gpu_stats = join_scoped_result("chunk embedding gpu", gpu_handle)?;
    let mut total_prep_fbank_us = 0u64;
    for handle in prep_handles {
        total_prep_fbank_us += join_scoped_result("chunk embedding prep", handle)?.fbank_us;
    }
    if let Some(error) = collect_error {
        return Err(error);
    }
    let collected = collector.finish()?;
    let num_chunks = collected.num_windows;

    debug!(
        total_chunks,
        num_chunks,
        gpu_chunks = gpu_stats.chunks,
        gpu_predict_ms = gpu_stats.predict_us / 1000,
        gpu_self_prep_ms = gpu_stats.self_prep_us / 1000,
        cpu_prep_ms = total_prep_fbank_us / 1000,
        predict_ms = total_predict_us / 1000,
        emb_wall_ms = emb_start.elapsed().as_millis(),
        "Chunk embedding worker breakdown"
    );

    Ok(EmbeddingSummary {
        segmentations: collected.segmentations,
        embeddings: collected.embeddings,
        num_chunks,
        gpu_predict_us: total_predict_us,
        prep_fbank_us: total_prep_fbank_us + gpu_stats.self_prep_us,
        prep_mask_us: 0,
    })
}

pub(super) fn run_sequential_chunks(
    emb_model: &mut EmbeddingModel,
    chunk_rx: &Receiver<ChunkJob>,
    audio: &[f32],
    params: &ChunkParams,
    emb_start: std::time::Instant,
    collection_plan: CollectionPlan,
) -> Result<EmbeddingSummary, PipelineError> {
    let mut collector = FileCollector::new(collection_plan);
    let mut seq_fbank_us = 0u64;
    let mut seq_mask_us = 0u64;
    let mut seq_predict_us = 0u64;
    let mut seq_chunks = 0u32;

    for job in chunk_rx {
        let window_start = job.window_start;
        let decoded_chunk = job.decoded;
        let wins = decoded_chunk.len();
        let session = chunk_session_for_windows(emb_model, wins)?;
        let sess_fbank_frames = session.fbank_frames;
        let sess_num_masks = session.num_masks;

        let fbank_start = std::time::Instant::now();
        let fbank = sequential_chunk_fbank(
            emb_model,
            ChunkFbankRequest {
                chunk_start: window_start * params.step_samples,
                frame_capacity: sess_fbank_frames,
                audio_len: audio.len(),
                ten_second_samples: params.window_samples,
                allow_thirty_second: params.fbank_normalization_scope.uses_chunk_scope()
                    && emb_model.has_chunk_fbank_30s(),
            },
            audio,
        )?;

        seq_fbank_us += fbank_start.elapsed().as_micros() as u64;
        let mask_start = std::time::Instant::now();
        let (masks, active) = SpeakerMaskLayout {
            step_samples: params.step_samples,
            window_samples: params.window_samples,
            num_speakers: params.num_speakers,
            min_num_samples: params.min_num_samples,
            num_masks: sess_num_masks,
            max_active: sess_num_masks,
        }
        .collect(window_start, &decoded_chunk, audio);
        seq_mask_us += mask_start.elapsed().as_micros() as u64;

        let predict_start = std::time::Instant::now();
        let session = chunk_session_for_windows(emb_model, wins)?;
        let batch_emb = EmbeddingModel::embed_chunk_session(session, &fbank, &masks)?;
        seq_predict_us += predict_start.elapsed().as_micros() as u64;
        seq_chunks += 1;

        let num_masks = batch_emb.nrows();
        let data = batch_emb.iter().copied().collect();
        collector.add(super::gpu::EmbeddedChunk {
            file_index: 0,
            window_start,
            decoded: decoded_chunk,
            data,
            active,
            num_masks,
            predict_us: 0,
        })?;
    }

    let collected = collector.finish()?;
    let num_chunks = collected.num_windows;

    trace!(
        seq_chunks,
        num_chunks,
        fbank_ms = seq_fbank_us / 1000,
        mask_ms = seq_mask_us / 1000,
        predict_ms = seq_predict_us / 1000,
        wall_ms = emb_start.elapsed().as_millis(),
        "EMB sequential",
    );

    Ok(EmbeddingSummary {
        segmentations: collected.segmentations,
        embeddings: collected.embeddings,
        num_chunks,
        gpu_predict_us: seq_predict_us,
        prep_fbank_us: seq_fbank_us,
        prep_mask_us: seq_mask_us,
    })
}

/// Stitched chunk filterbank from the embedding model's own filterbank models
fn sequential_chunk_fbank(
    emb_model: &mut EmbeddingModel,
    request: ChunkFbankRequest,
    audio: &[f32],
) -> Result<Vec<f32>, PipelineError> {
    let mut fbank = vec![0.0f32; request.frame_capacity * FBANK_FEATURES];

    for call in request.plan() {
        let output = match call.model {
            FbankModel::ThirtySecond => emb_model
                .compute_chunk_fbank_30s(call.audio(audio, THIRTY_SECOND_SAMPLES))?
                .ok_or_else(|| super::invariant_error("chunk fbank plan needs the 30s model"))?,
            FbankModel::TenSecond => {
                emb_model.compute_chunk_fbank(call.audio(audio, request.ten_second_samples))?
            }
        };
        if output.nrows() < call.skip_frames + call.frames || output.ncols() != FBANK_FEATURES {
            return Err(super::invariant_error(format!(
                "chunk fbank call returned {:?}, expected at least {} frames of {FBANK_FEATURES} features",
                output.dim(),
                call.skip_frames + call.frames
            )));
        }

        let dest = &mut fbank
            [call.dest_frame * FBANK_FEATURES..(call.dest_frame + call.frames) * FBANK_FEATURES];
        ArrayViewMut2::from_shape((call.frames, FBANK_FEATURES), dest)
            .map_err(|error| super::invariant_error(format!("chunk fbank layout: {error}")))?
            .assign(&output.slice(s![call.skip_frames..call.skip_frames + call.frames, ..]));
    }

    Ok(fbank)
}
