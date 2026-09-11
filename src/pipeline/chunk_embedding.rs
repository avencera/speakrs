use ndarray::{Array2, Array3};
use tracing::{debug, trace};

use crate::clustering::plda::PldaTransform;
use crate::inference::embedding::{
    ChunkEmbeddingSession, EmbeddingModel, FBANK_FRAMES, FBANK_HOP_SAMPLES,
};
use crate::inference::segmentation::SegmentationModel;
use crate::powerset::PowersetMapping;

use super::config::ChunkFbankNormalizationScope;
use super::config::CoreMlChunkExecutionPolicy;
use super::config::PipelineConfig;
use super::post_inference::post_inference;
#[cfg(feature = "_metrics")]
use super::types::InternalInferenceStageTimings;
use super::types::{
    BatchInput, DecodedSegmentations, DiarizationResult, InferenceArtifacts, PipelineError,
};
pub(super) use super::types::{ChunkEmbeddings, ChunkLayout, chunk_audio_raw};
pub(super) use super::write_speaker_mask_to_slice;

mod collect;
mod error;
mod gpu;
mod orchestrate;
mod prep;

use collect::{CollectionPlan, FileCollector, build_chunk_artifacts};
use error::{backend_error, invariant_error, worker_panic};
use gpu::{GpuWorker, chunk_embedding_resources};
use orchestrate::{ChunkExecution, ChunkExecutionPlan, run_pipelined, run_sequential_chunks};
use prep::{ChunkJob, ChunkPrep, PrepScratch, PrepWorker};

/// Audio consumed per 10s fbank call when stitching a long fbank from segments.
/// Each call yields FBANK_FRAMES frames, which covers fewer samples than the 10s
/// window itself. Advancing by the full window would slip the stitched fbank by two
/// frames per segment relative to the speaker masks
pub(super) const FBANK_SEGMENT_SAMPLES: usize = FBANK_FRAMES * FBANK_HOP_SAMPLES;

struct ChunkParams {
    step_samples: usize,
    window_samples: usize,
    num_speakers: usize,
    min_num_samples: usize,
    segmentation_workers: usize,
    fbank_preparation_workers: usize,
    fbank_normalization_scope: ChunkFbankNormalizationScope,
}

struct EmbeddingSummary {
    segmentations: Array3<f32>,
    embeddings: Array3<f32>,
    num_chunks: usize,
    gpu_predict_us: u64,
    prep_fbank_us: u64,
    prep_mask_us: u64,
}

fn join_scoped_result<T>(
    name: &str,
    handle: std::thread::ScopedJoinHandle<'_, Result<T, PipelineError>>,
) -> Result<T, PipelineError> {
    handle.join().map_err(|_| worker_panic(name))?
}

fn chunk_session_for_windows(
    emb_model: &mut EmbeddingModel,
    wins: usize,
) -> Result<&ChunkEmbeddingSession, PipelineError> {
    emb_model.chunk_session_for_windows(wins)?.ok_or_else(|| {
        invariant_error(format!(
            "missing chunk embedding session for {wins} windows"
        ))
    })
}

pub(super) fn try_chunk_embedding(
    seg_model: &mut SegmentationModel,
    emb_model: &mut EmbeddingModel,
    powerset: &PowersetMapping,
    audio: &[f32],
    execution_policy: CoreMlChunkExecutionPolicy,
) -> Result<Option<InferenceArtifacts>, PipelineError> {
    let Some(plan) = ChunkExecutionPlan::resolve(seg_model, emb_model, audio)? else {
        return Ok(None);
    };
    let use_pipelined = plan.is_pipelined();
    let collection_plan = plan.collection;
    let chunk_win_capacity = collection_plan.group_capacity();

    let inference_start = std::time::Instant::now();
    let step_seconds = seg_model.step_seconds();
    let params = ChunkParams {
        step_samples: plan.layout.step_samples,
        window_samples: plan.layout.window_samples,
        num_speakers: 3,
        min_num_samples: emb_model.min_num_samples(),
        segmentation_workers: execution_policy.segmentation_workers,
        fbank_preparation_workers: execution_policy.fbank_preparation_workers,
        fbank_normalization_scope: execution_policy.fbank_normalization_scope,
    };

    let (seg_tx, seg_rx) = crossbeam_channel::bounded::<Array2<f32>>(100);
    let (chunk_tx, chunk_rx) = crossbeam_channel::bounded::<ChunkJob>(100);

    std::thread::scope(|scope| {
        let seg_start = std::time::Instant::now();
        let seg_warm_start_windows = chunk_win_capacity;
        let seg_handle = scope.spawn(move || -> Result<std::time::Duration, PipelineError> {
            seg_model.run_streaming_parallel(
                audio,
                seg_tx,
                params.segmentation_workers,
                Some(seg_warm_start_windows),
            )?;
            Ok(seg_start.elapsed())
        });

        let bridge_handle = scope.spawn(move || -> Result<(), PipelineError> {
            let mut group = Vec::with_capacity(chunk_win_capacity);
            let mut global_start = 0usize;

            for raw_window in &seg_rx {
                group.push(powerset.hard_decode(&raw_window)?);

                if group.len() == chunk_win_capacity {
                    if chunk_tx
                        .send(ChunkJob::new(0, global_start, std::mem::take(&mut group))?)
                        .is_err()
                    {
                        break;
                    }
                    global_start += chunk_win_capacity;
                    group = Vec::with_capacity(chunk_win_capacity);
                }
            }

            if !group.is_empty() {
                let _ = chunk_tx.send(ChunkJob::new(0, global_start, group)?);
            }
            Ok(())
        });

        let emb_start = std::time::Instant::now();
        let summary = match plan.execution {
            ChunkExecution::Pipelined { resources } => run_pipelined(
                scope,
                emb_start,
                resources,
                chunk_rx,
                audio,
                &params,
                collection_plan,
            )?,
            ChunkExecution::Sequential => run_sequential_chunks(
                emb_model,
                &chunk_rx,
                audio,
                &params,
                emb_start,
                collection_plan,
            )?,
        };
        let emb_elapsed = emb_start.elapsed();

        let seg_thread_elapsed = seg_handle
            .join()
            .map_err(|_| worker_panic("segmentation"))??;
        bridge_handle
            .join()
            .map_err(|_| worker_panic("segmentation bridge"))??;
        trace!(
            seg_thread_ms = seg_thread_elapsed.as_millis(),
            seg_wall_ms = seg_start.elapsed().as_millis(),
            "SEG timing"
        );

        let num_chunks = summary.num_chunks;
        let gpu_predict_us = summary.gpu_predict_us;
        let prep_fbank_us = summary.prep_fbank_us;
        let prep_mask_us = summary.prep_mask_us;
        let inference_elapsed = inference_start.elapsed();
        #[cfg(feature = "_metrics")]
        let stage_timings = InternalInferenceStageTimings {
            segmentation_seconds: seg_thread_elapsed.as_secs_f64(),
            embedding_seconds: emb_elapsed.as_secs_f64(),
            prediction_seconds: gpu_predict_us as f64 / 1_000_000.0,
            filterbank_preparation_seconds: prep_fbank_us as f64 / 1_000_000.0,
            mask_preparation_seconds: prep_mask_us as f64 / 1_000_000.0,
            total_seconds: inference_elapsed.as_secs_f64(),
            chunk_count: num_chunks,
            pipelined: use_pipelined,
        };
        let artifacts = build_chunk_artifacts(
            step_seconds,
            params.step_samples,
            params.window_samples,
            summary,
            #[cfg(feature = "_metrics")]
            stage_timings,
        )?;

        let audio_secs = audio.len() as f64 / 16_000.0;
        debug!(
            chunks = num_chunks,
            chunk_capacity = chunk_win_capacity,
            pipelined = use_pipelined,
            seg_ms = seg_thread_elapsed.as_millis(),
            emb_ms = emb_elapsed.as_millis(),
            predict_ms = gpu_predict_us / 1000,
            prep_fbank_ms = prep_fbank_us / 1000,
            prep_mask_ms = prep_mask_us / 1000,
            total_ms = inference_elapsed.as_millis(),
            audio_secs = audio_secs as u64,
            "Chunk embedding complete"
        );

        Ok(Some(artifacts))
    })
}

pub(super) fn try_batch_chunk_embedding(
    seg_model: &mut SegmentationModel,
    emb_model: &mut EmbeddingModel,
    powerset: &PowersetMapping,
    plda: &PldaTransform,
    files: &[BatchInput<'_>],
    config: &PipelineConfig,
    execution_policy: CoreMlChunkExecutionPolicy,
) -> Result<Option<Vec<DiarizationResult>>, PipelineError> {
    if files.is_empty() {
        return Ok(Some(Vec::new()));
    }

    let chunk_win_capacity = match emb_model.chunk_window_capacity() {
        Some(capacity) => capacity,
        None => return Ok(None),
    };

    let step_samples = seg_model.step_samples();
    let window_samples = seg_model.window_samples();
    let step_seconds = seg_model.step_seconds();
    let num_speakers = 3usize;
    let min_num_samples = emb_model.min_num_samples();
    let segmentation_workers = execution_policy.segmentation_workers;
    let fbank_preparation_workers = execution_policy.fbank_preparation_workers;

    if files.iter().any(|file| file.audio.len() < window_samples) {
        return Ok(None);
    }

    let collection_plans: Vec<CollectionPlan> = files
        .iter()
        .enumerate()
        .map(|(file_index, file)| {
            CollectionPlan::new(
                file_index,
                seg_model.window_count(file.audio.len()),
                chunk_win_capacity,
                num_speakers,
            )
        })
        .collect::<Result<_, _>>()?;

    let total_groups = collection_plans
        .iter()
        .map(|plan| plan.group_count())
        .try_fold(0usize, usize::checked_add)
        .ok_or_else(|| invariant_error("batch chunk group count overflowed"))?;
    if total_groups < 2 {
        return Ok(None);
    }

    let Some(resources) = chunk_embedding_resources(emb_model)? else {
        return Ok(None);
    };

    let largest = resources.largest_session()?.clone();
    let prep_config = ChunkPrep {
        step_samples,
        window_samples,
        num_speakers,
        min_num_samples,
        largest_fbank_frames: largest.fbank_frames,
        largest_num_masks: largest.num_masks,
        max_active: largest.num_windows * num_speakers,
        fbank_30s: resources.fbank_30s.clone(),
        fbank_10s: resources.fbank_10s.clone(),
        fbank_normalization_scope: execution_policy.fbank_normalization_scope,
    };

    let audios: Vec<&[f32]> = files.iter().map(|file| file.audio).collect();
    let batch_start = std::time::Instant::now();

    let (decoded_tx, decoded_rx) = crossbeam_channel::bounded::<ChunkJob>(100);
    let (prepared_tx, prepared_rx) = crossbeam_channel::bounded(48);
    let (embedded_tx, embedded_rx) = crossbeam_channel::bounded(16);

    std::thread::scope(|scope| {
        let audios_ref = &audios;

        let decoded_tx_seg = decoded_tx.clone();
        let seg_handle = scope.spawn(move || -> Result<(), PipelineError> {
            for (file_idx, file) in files.iter().enumerate() {
                let (seg_tx, seg_rx) = crossbeam_channel::bounded::<Array2<f32>>(100);

                std::thread::scope(|inner| {
                    let decoded_tx_bridge = &decoded_tx_seg;
                    let bridge_handle = inner.spawn(move || -> Result<(), PipelineError> {
                        let mut group = Vec::with_capacity(chunk_win_capacity);
                        let mut window_start = 0usize;

                        for raw_window in &seg_rx {
                            group.push(powerset.hard_decode(&raw_window)?);
                            if group.len() == chunk_win_capacity {
                                if decoded_tx_bridge
                                    .send(ChunkJob::new(
                                        file_idx,
                                        window_start,
                                        std::mem::take(&mut group),
                                    )?)
                                    .is_err()
                                {
                                    return Ok(());
                                }
                                window_start += chunk_win_capacity;
                                group = Vec::with_capacity(chunk_win_capacity);
                            }
                        }

                        if !group.is_empty() {
                            let _ = decoded_tx_bridge.send(ChunkJob::new(
                                file_idx,
                                window_start,
                                group,
                            )?);
                        }
                        Ok(())
                    });

                    seg_model.run_streaming_parallel(
                        file.audio,
                        seg_tx,
                        segmentation_workers,
                        Some(chunk_win_capacity),
                    )?;

                    bridge_handle
                        .join()
                        .map_err(|_| worker_panic("batch segmentation bridge"))??;
                    Ok::<(), PipelineError>(())
                })?;
            }
            drop(decoded_tx_seg);
            Ok(())
        });
        drop(decoded_tx);

        let mut prep_handles = Vec::with_capacity(fbank_preparation_workers);
        for _ in 0..fbank_preparation_workers {
            let worker = PrepWorker {
                prep: prep_config.clone(),
                scratch: PrepScratch::new(window_samples),
            };
            let prepared_tx = prepared_tx.clone();
            let decoded_rx = decoded_rx.clone();
            prep_handles.push(scope.spawn(move || worker.run(audios_ref, decoded_rx, prepared_tx)));
        }
        drop(prepared_tx);

        let gpu_worker = GpuWorker {
            model: largest.handle.model,
            fbank_shape: largest.handle.cached_fbank_shape,
            masks_shape: largest.handle.cached_masks_shape,
            prep: prep_config,
            scratch: PrepScratch::new(window_samples),
        };
        let gpu_embedded_tx = embedded_tx.clone();
        let gpu_decoded_rx = decoded_rx.clone();
        let gpu_handle = scope.spawn(move || {
            gpu_worker.run(audios_ref, prepared_rx, gpu_decoded_rx, gpu_embedded_tx)
        });
        drop(decoded_rx);
        drop(embedded_tx);

        let mut collectors: Vec<Option<FileCollector>> = collection_plans
            .iter()
            .copied()
            .map(FileCollector::new)
            .map(Some)
            .collect();
        let mut results: Vec<Option<DiarizationResult>> =
            std::iter::repeat_with(|| None).take(files.len()).collect();
        let mut files_complete = 0usize;
        let mut collect_error = None;

        for embedded in std::iter::from_fn(|| embedded_rx.recv().ok()) {
            let file_idx = embedded.file_index;
            let Some(collector) = collectors.get_mut(file_idx).and_then(Option::as_mut) else {
                collect_error = Some(invariant_error(format!(
                    "chunk payload file index {file_idx} has no active collector"
                )));
                break;
            };

            if let Err(error) = collector.add(embedded) {
                collect_error = Some(error);
                break;
            }

            if collector.is_complete() {
                let collector = match collectors[file_idx].take() {
                    Some(collector) => collector,
                    None => {
                        collect_error = Some(invariant_error(format!(
                            "collector for file {file_idx} completed without state"
                        )));
                        break;
                    }
                };
                match collector
                    .into_artifacts(step_seconds, step_samples, window_samples)
                    .and_then(|artifacts| post_inference(artifacts, config, plda))
                {
                    Ok(result) => results[file_idx] = Some(result),
                    Err(error) => {
                        collect_error = Some(error);
                        break;
                    }
                }
                files_complete += 1;
            }
        }
        drop(embedded_rx);

        join_scoped_result("batch segmentation", seg_handle)?;
        let _gpu_stats = join_scoped_result("batch chunk embedding gpu", gpu_handle)?;
        for handle in prep_handles {
            join_scoped_result("batch chunk embedding prep", handle)?;
        }
        if let Some(error) = collect_error {
            return Err(error);
        }

        for (file_idx, collector) in collectors.into_iter().enumerate() {
            if results[file_idx].is_some() {
                continue;
            }
            let collector = collector.ok_or_else(|| {
                invariant_error(format!(
                    "file {file_idx} has neither a collector nor a completed result"
                ))
            })?;
            collector.finish()?;

            return Err(invariant_error(format!(
                "file {file_idx} completed collection without a diarization result"
            )));
        }

        debug!(
            files = files.len(),
            files_complete,
            batch_ms = batch_start.elapsed().as_millis(),
            "Batch chunk embedding complete"
        );

        completed_file_results(results)
    })
    .map(Some)
}

fn completed_file_results(
    results: Vec<Option<DiarizationResult>>,
) -> Result<Vec<DiarizationResult>, PipelineError> {
    let mut completed = Vec::with_capacity(results.len());
    for (file_idx, result) in results.into_iter().enumerate() {
        let Some(result) = result else {
            return Err(invariant_error(format!(
                "missing batch chunk result for file {file_idx}"
            )));
        };
        completed.push(result);
    }
    Ok(completed)
}

#[cfg(test)]
mod tests {
    use super::completed_file_results;

    #[test]
    fn missing_batch_file_result_is_an_error() {
        assert!(completed_file_results(vec![None]).is_err());
        assert!(completed_file_results(Vec::new()).is_ok());
    }
}
