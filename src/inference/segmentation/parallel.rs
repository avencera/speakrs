#![cfg(feature = "coreml")]

use std::sync::{Arc, Mutex};

use crossbeam_channel::Sender;
use ndarray::Array2;
use tracing::{debug, trace};

use super::SegmentationError;
use super::{PRIMARY_BATCH_SIZE, SegmentationModel};
use crate::inference::coreml::CachedInputShape;
use crate::inference::segmentation::tensor::{
    padded_window, segmentation_array, segmentation_array_from_slice, worker_panic,
};

#[derive(Clone, Default)]
pub(super) struct WorkerErrorSlot(Arc<Mutex<Option<SegmentationError>>>);

impl WorkerErrorSlot {
    pub(super) fn record(&self, error: SegmentationError) {
        if let Ok(mut slot) = self.0.lock()
            && slot.is_none()
        {
            *slot = Some(error);
        }
    }

    pub(super) fn take(&self) -> Result<Option<SegmentationError>, SegmentationError> {
        self.0
            .lock()
            .map(|mut slot| slot.take())
            .map_err(|_| SegmentationError::Invariant {
                context: "parallel segmentation worker error slot",
                message: "worker error slot was poisoned".to_owned(),
            })
    }
}

impl SegmentationModel {
    /// Run segmentation with N parallel workers, each with a fresh CoreML model
    ///
    /// Workers use CPUOnly compute units to avoid GPU contention with embedding
    /// Results are sent through `tx` in chunk_idx order
    pub fn run_streaming_parallel(
        &mut self,
        audio: &[f32],
        tx: Sender<Array2<f32>>,
        num_workers: usize,
        warm_start_target_windows: Option<usize>,
    ) -> Result<usize, SegmentationError> {
        use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

        let mut offsets = Vec::new();
        let mut offset = 0;
        while offset + self.window_samples <= audio.len() {
            offsets.push(offset);
            offset += self.step_samples;
        }

        let padded = if offset < audio.len() && audio.len() > self.window_samples {
            let mut p = vec![0.0f32; self.window_samples];
            let remaining = audio.len() - offset;
            p[..remaining].copy_from_slice(&audio[offset..]);
            Some(p)
        } else {
            None
        };

        let total_windows = offsets.len() + padded.is_some() as usize;
        if total_windows == 0 {
            return Ok(0);
        }

        let Some((shared_model, batch_size)) = self.select_parallel_native_model(total_windows)
        else {
            return self.run_streaming(audio, tx);
        };

        let seg_start = std::time::Instant::now();
        let win_samples = self.window_samples;
        let seg_predict_us = AtomicU64::new(0);
        let seg_batched_calls = AtomicU64::new(0);
        let seg_batched_windows = AtomicU64::new(0);
        let seg_single_calls = AtomicU64::new(0);
        let est_embed_chunks = warm_start_target_windows
            .and_then(|target| target.checked_div(2))
            .filter(|&chunk_windows| chunk_windows > 0)
            .map(|chunk_windows| total_windows.div_ceil(chunk_windows));
        let upper_medium_warm_start = est_embed_chunks
            .map(|chunks| chunks >= 12)
            .unwrap_or(total_windows >= 640);
        let default_warm_start_small_windows = if upper_medium_warm_start {
            PRIMARY_BATCH_SIZE * 2
        } else {
            140
        };
        let warm_start_small_windows =
            warm_start_target_windows.unwrap_or(default_warm_start_small_windows);
        let warm_start_batch_capacity = if upper_medium_warm_start {
            PRIMARY_BATCH_SIZE / 2
        } else {
            28
        };
        let use_warm_start_b32 = batch_size == super::LARGE_BATCH_SIZE
            && total_windows < 1024
            && total_windows > PRIMARY_BATCH_SIZE
            && warm_start_small_windows > PRIMARY_BATCH_SIZE
            && self.native_batched_session.is_some();

        if batch_size > 1 {
            struct BatchTask<'a> {
                batch_idx: usize,
                start: usize,
                end: usize,
                batch_capacity: usize,
                model: &'a crate::inference::coreml::SharedCoreMlModel,
            }

            let mut batch_tasks: Vec<BatchTask<'_>> = Vec::new();
            if use_warm_start_b32 {
                let Some(small_model) = self.native_batched_session.as_ref() else {
                    return Err(SegmentationError::Invariant {
                        context: "parallel segmentation warm start",
                        message: "missing native b32 model".to_owned(),
                    });
                };
                let mut start = 0usize;
                let mut batch_idx = 0usize;
                let warm_start_end = total_windows.min(warm_start_small_windows);

                while start < warm_start_end {
                    let end = (start + warm_start_batch_capacity).min(total_windows);
                    batch_tasks.push(BatchTask {
                        batch_idx,
                        start,
                        end,
                        batch_capacity: warm_start_batch_capacity,
                        model: small_model,
                    });
                    start = end;
                    batch_idx += 1;
                }

                while start < total_windows {
                    let end = (start + super::LARGE_BATCH_SIZE).min(total_windows);
                    batch_tasks.push(BatchTask {
                        batch_idx,
                        start,
                        end,
                        batch_capacity: super::LARGE_BATCH_SIZE,
                        model: shared_model,
                    });
                    start = end;
                    batch_idx += 1;
                }
            } else {
                for batch_idx in 0..total_windows.div_ceil(batch_size) {
                    let start = batch_idx * batch_size;
                    let end = (start + batch_size).min(total_windows);
                    batch_tasks.push(BatchTask {
                        batch_idx,
                        start,
                        end,
                        batch_capacity: batch_size,
                        model: shared_model,
                    });
                }
            }

            let (batch_tx, batch_rx) = crossbeam_channel::unbounded::<(usize, Vec<Array2<f32>>)>();

            std::thread::scope(|scope| {
                let merge_handle = scope.spawn(move || -> Result<(), SegmentationError> {
                    let mut next_batch = 0usize;
                    let mut pending = std::collections::BTreeMap::<usize, Vec<Array2<f32>>>::new();

                    for (batch_idx, results) in batch_rx {
                        pending.insert(batch_idx, results);
                        while let Some(results) = pending.remove(&next_batch) {
                            for result in results {
                                tx.send(result)?;
                            }
                            next_batch += 1;
                        }
                    }

                    Ok(())
                });

                let worker_error = WorkerErrorSlot::default();
                rayon::scope(|rscope| {
                    let seg_predict_us = &seg_predict_us;
                    let seg_batched_calls = &seg_batched_calls;
                    let seg_batched_windows = &seg_batched_windows;
                    let next_task = Arc::new(AtomicUsize::new(0));
                    let worker_count = batch_tasks.len().min(num_workers.max(1));

                    for _worker_idx in 0..worker_count {
                        let batch_tasks = &batch_tasks;
                        let offsets = &offsets;
                        let padded = &padded;
                        let batch_tx = batch_tx.clone();
                        let next_task = Arc::clone(&next_task);
                        let worker_error = worker_error.clone();

                        rscope.spawn(move |_| {
                            let mut scratch_by_capacity = std::collections::BTreeMap::<
                                usize,
                                (CachedInputShape, Vec<f32>),
                            >::new();

                            loop {
                                let task_idx = next_task.fetch_add(1, Ordering::Relaxed);
                                let Some(task) = batch_tasks.get(task_idx) else {
                                    break;
                                };
                                let actual_batch = task.end - task.start;
                                let (cached_batch, batch_buf) = scratch_by_capacity
                                    .entry(task.batch_capacity)
                                    .or_insert_with(|| {
                                        (
                                            CachedInputShape::new(
                                                "input",
                                                &[task.batch_capacity, 1, win_samples],
                                            ),
                                            vec![0.0f32; task.batch_capacity * win_samples],
                                        )
                                    });

                                batch_buf.fill(0.0);
                                for (b, widx) in (task.start..task.end).enumerate() {
                                    let window = if widx < offsets.len() {
                                        &audio[offsets[widx]..offsets[widx] + win_samples]
                                    } else {
                                        let Ok(window) =
                                            padded_window(padded, "parallel segmentation batch")
                                        else {
                                            worker_error.record(SegmentationError::Invariant {
                                                context: "parallel segmentation batch",
                                                message: "missing padded window".to_owned(),
                                            });
                                            return;
                                        };
                                        window
                                    };
                                    let dst = b * win_samples;
                                    batch_buf[dst..dst + window.len()].copy_from_slice(window);
                                }

                                let batch_start = std::time::Instant::now();
                                let Ok((data, out_shape)) = task
                                    .model
                                    .predict_cached(&[(&*cached_batch, batch_buf.as_slice())])
                                    .map_err(|e| {
                                        SegmentationError::Ort(ort::Error::new(e.to_string()))
                                    })
                                else {
                                    worker_error.record(SegmentationError::Ort(ort::Error::new(
                                        "parallel segmentation batch predict failed",
                                    )));
                                    return;
                                };
                                let batch_us = batch_start.elapsed().as_micros() as u64;
                                seg_predict_us.fetch_add(batch_us, Ordering::Relaxed);
                                seg_batched_calls.fetch_add(1, Ordering::Relaxed);
                                seg_batched_windows
                                    .fetch_add(actual_batch as u64, Ordering::Relaxed);
                                trace!(
                                    batch_idx = task.batch_idx,
                                    batch_capacity = task.batch_capacity,
                                    batch_size = actual_batch,
                                    batch_ms = batch_us / 1000,
                                    "Seg batch profile"
                                );

                                let frames = out_shape[1];
                                let classes = out_shape[2];
                                let stride = frames * classes;
                                let mut results = Vec::with_capacity(actual_batch);
                                for b in 0..actual_batch {
                                    let s = b * stride;
                                    let Ok(result) = segmentation_array_from_slice(
                                        frames,
                                        classes,
                                        &data[s..s + stride],
                                        "parallel segmentation batched output",
                                    ) else {
                                        worker_error.record(SegmentationError::Invariant {
                                            context: "parallel segmentation batched output",
                                            message: "invalid segmentation output shape".to_owned(),
                                        });
                                        return;
                                    };
                                    results.push(result);
                                }
                                let _ = batch_tx.send((task.batch_idx, results));
                            }
                        });
                    }
                });

                if let Some(error) = worker_error.take()? {
                    return Err(error);
                }

                drop(batch_tx);
                merge_handle
                    .join()
                    .map_err(|_| worker_panic("parallel segmentation merge"))??;
                Ok::<(), SegmentationError>(())
            })?;
        } else {
            let chunk_size = total_windows.div_ceil(num_workers);
            let actual_workers = total_windows.div_ceil(chunk_size).min(num_workers);

            let mut worker_txs = Vec::with_capacity(actual_workers);
            let mut worker_rxs = Vec::with_capacity(actual_workers);
            for _ in 0..actual_workers {
                let (wtx, wrx) = crossbeam_channel::unbounded::<Array2<f32>>();
                worker_txs.push(wtx);
                worker_rxs.push(wrx);
            }

            std::thread::scope(|scope| {
                let merge_handle = scope.spawn(move || -> Result<(), SegmentationError> {
                    for wrx in &worker_rxs {
                        for result in wrx {
                            tx.send(result)?;
                        }
                    }
                    Ok(())
                });

                let worker_error = WorkerErrorSlot::default();
                rayon::scope(|rscope| {
                    let seg_predict_us = &seg_predict_us;
                    let seg_single_calls = &seg_single_calls;
                    for (worker_idx, worker_tx) in worker_txs.into_iter().enumerate() {
                        let model = shared_model;
                        let start = worker_idx * chunk_size;
                        let end = (start + chunk_size).min(total_windows);
                        let offsets = &offsets;
                        let padded = &padded;
                        let worker_error = worker_error.clone();

                        rscope.spawn(move |_| {
                            let cached_shape = CachedInputShape::new("input", &[1, 1, win_samples]);
                            let mut buffer = ndarray::Array3::<f32>::zeros((1, 1, win_samples));

                            for idx in start..end {
                                let window = if idx < offsets.len() {
                                    &audio[offsets[idx]..offsets[idx] + win_samples]
                                } else {
                                    let Ok(window) =
                                        padded_window(padded, "parallel segmentation worker")
                                    else {
                                        worker_error.record(SegmentationError::Invariant {
                                            context: "parallel segmentation worker",
                                            message: "missing padded window".to_owned(),
                                        });
                                        return;
                                    };
                                    window
                                };

                                buffer.fill(0.0);
                                buffer
                                    .slice_mut(ndarray::s![0, 0, ..window.len()])
                                    .assign(&ndarray::ArrayView1::from(window));
                                let Ok(input_data) =
                                    crate::inference::segmentation::tensor::array3_slice(
                                        &buffer,
                                        "parallel segmentation worker input",
                                    )
                                else {
                                    worker_error.record(SegmentationError::Invariant {
                                        context: "parallel segmentation worker input",
                                        message: "input buffer was not contiguous".to_owned(),
                                    });
                                    return;
                                };

                                let predict_start = std::time::Instant::now();
                                let Ok((data, out_shape)) = model
                                    .predict_cached(&[(&cached_shape, input_data)])
                                    .map_err(|e| {
                                        SegmentationError::Ort(ort::Error::new(e.to_string()))
                                    })
                                else {
                                    worker_error.record(SegmentationError::Ort(ort::Error::new(
                                        "parallel segmentation worker predict failed",
                                    )));
                                    return;
                                };
                                let predict_us = predict_start.elapsed().as_micros() as u64;
                                seg_predict_us.fetch_add(predict_us, Ordering::Relaxed);
                                seg_single_calls.fetch_add(1, Ordering::Relaxed);
                                trace!(
                                    worker_idx,
                                    batch_size = 1,
                                    batch_ms = predict_us / 1000,
                                    "Seg batch profile"
                                );

                                let frames = out_shape[1];
                                let classes = out_shape[2];
                                let Ok(result) = segmentation_array(
                                    frames,
                                    classes,
                                    data,
                                    "parallel segmentation worker output",
                                ) else {
                                    worker_error.record(SegmentationError::Invariant {
                                        context: "parallel segmentation worker output",
                                        message: "invalid segmentation output shape".to_owned(),
                                    });
                                    return;
                                };
                                let _ = worker_tx.send(result);
                            }
                        });
                    }
                });

                if let Some(error) = worker_error.take()? {
                    return Err(error);
                }

                merge_handle
                    .join()
                    .map_err(|_| worker_panic("parallel segmentation merge"))??;
                Ok::<(), SegmentationError>(())
            })?;
        }

        let total_seg = seg_start.elapsed();
        let seg_predict_ms = seg_predict_us.load(Ordering::Relaxed) / 1000;
        debug!(
            windows = total_windows,
            workers = num_workers,
            seg_est_embed_chunks = est_embed_chunks.unwrap_or(0),
            seg_warm_start_b32 = use_warm_start_b32,
            seg_warm_start_windows = if use_warm_start_b32 {
                total_windows.min(warm_start_small_windows)
            } else {
                0
            },
            seg_warm_start_batch_capacity = if use_warm_start_b32 {
                warm_start_batch_capacity
            } else {
                0
            },
            seg_batched_calls = seg_batched_calls.load(Ordering::Relaxed),
            seg_batched_windows = seg_batched_windows.load(Ordering::Relaxed),
            seg_single_calls = seg_single_calls.load(Ordering::Relaxed),
            seg_predict_ms_sum = seg_predict_ms,
            seg_total_ms = total_seg.as_millis(),
            "Parallel segmentation complete"
        );

        Ok(total_windows)
    }
}
