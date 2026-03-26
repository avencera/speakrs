#![cfg(feature = "coreml")]

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use crossbeam_channel::Sender;
use ndarray::{Array2, Array3, s};
use tracing::{debug, trace};

use super::SegmentationError;
use super::{LARGE_BATCH_SIZE, PRIMARY_BATCH_SIZE, SegmentationModel};
use crate::inference::coreml::{CachedInputShape, SharedCoreMlModel};
use crate::inference::segmentation::tensor::{
    array3_slice, padded_window, segmentation_array, segmentation_array_from_slice, worker_panic,
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

struct ParallelWindows<'a> {
    audio: &'a [f32],
    offsets: Vec<usize>,
    padded: Option<Vec<f32>>,
    window_samples: usize,
}

impl<'a> ParallelWindows<'a> {
    fn collect(audio: &'a [f32], window_samples: usize, step_samples: usize) -> Self {
        let mut offsets = Vec::new();
        let mut offset = 0;
        while offset + window_samples <= audio.len() {
            offsets.push(offset);
            offset += step_samples;
        }

        let padded = if offset < audio.len() && audio.len() > window_samples {
            let mut padded = vec![0.0f32; window_samples];
            let remaining = audio.len() - offset;
            padded[..remaining].copy_from_slice(&audio[offset..]);
            Some(padded)
        } else {
            None
        };

        Self {
            audio,
            offsets,
            padded,
            window_samples,
        }
    }

    fn is_empty(&self) -> bool {
        self.total_windows() == 0
    }

    fn total_windows(&self) -> usize {
        self.offsets.len() + self.padded.is_some() as usize
    }

    fn window<'b>(
        &'b self,
        idx: usize,
        context: &'static str,
    ) -> Result<&'b [f32], SegmentationError> {
        if idx < self.offsets.len() {
            let start = self.offsets[idx];
            return Ok(&self.audio[start..start + self.window_samples]);
        }
        if idx == self.offsets.len() {
            return padded_window(&self.padded, context);
        }

        Err(SegmentationError::Invariant {
            context,
            message: format!(
                "window index {idx} exceeded total window count {}",
                self.total_windows()
            ),
        })
    }
}

struct ParallelProfile {
    predict_us: AtomicU64,
    batched_calls: AtomicU64,
    batched_windows: AtomicU64,
    single_calls: AtomicU64,
}

struct ParallelRunSummary {
    total_windows: usize,
    num_workers: usize,
    est_embed_chunks: Option<usize>,
    use_warm_start_b32: bool,
    warm_start_small_windows: usize,
    warm_start_batch_capacity: usize,
    total_seg: std::time::Duration,
}

impl ParallelProfile {
    fn new() -> Self {
        Self {
            predict_us: AtomicU64::new(0),
            batched_calls: AtomicU64::new(0),
            batched_windows: AtomicU64::new(0),
            single_calls: AtomicU64::new(0),
        }
    }

    fn record_batch(
        &self,
        batch_idx: usize,
        batch_capacity: usize,
        batch_size: usize,
        batch_us: u64,
    ) {
        self.predict_us.fetch_add(batch_us, Ordering::Relaxed);
        self.batched_calls.fetch_add(1, Ordering::Relaxed);
        self.batched_windows
            .fetch_add(batch_size as u64, Ordering::Relaxed);
        trace!(
            batch_idx,
            batch_capacity,
            batch_size,
            batch_ms = batch_us / 1000,
            "Seg batch profile"
        );
    }

    fn record_single(&self, worker_idx: usize, predict_us: u64) {
        self.predict_us.fetch_add(predict_us, Ordering::Relaxed);
        self.single_calls.fetch_add(1, Ordering::Relaxed);
        trace!(
            worker_idx,
            batch_size = 1,
            batch_ms = predict_us / 1000,
            "Seg batch profile"
        );
    }

    fn log_completion(&self, summary: ParallelRunSummary) {
        debug!(
            windows = summary.total_windows,
            workers = summary.num_workers,
            seg_est_embed_chunks = summary.est_embed_chunks.unwrap_or(0),
            seg_warm_start_b32 = summary.use_warm_start_b32,
            seg_warm_start_windows = if summary.use_warm_start_b32 {
                summary.total_windows.min(summary.warm_start_small_windows)
            } else {
                0
            },
            seg_warm_start_batch_capacity = if summary.use_warm_start_b32 {
                summary.warm_start_batch_capacity
            } else {
                0
            },
            seg_batched_calls = self.batched_calls.load(Ordering::Relaxed),
            seg_batched_windows = self.batched_windows.load(Ordering::Relaxed),
            seg_single_calls = self.single_calls.load(Ordering::Relaxed),
            seg_predict_ms_sum = self.predict_us.load(Ordering::Relaxed) / 1000,
            seg_total_ms = summary.total_seg.as_millis(),
            "Parallel segmentation complete"
        );
    }
}

struct BatchTask<'a> {
    batch_idx: usize,
    start: usize,
    end: usize,
    batch_capacity: usize,
    model: &'a SharedCoreMlModel,
}

struct BatchTaskPlanner<'a> {
    shared_model: &'a SharedCoreMlModel,
    small_model: Option<&'a SharedCoreMlModel>,
    total_windows: usize,
    batch_size: usize,
    use_warm_start_b32: bool,
    warm_start_small_windows: usize,
    warm_start_batch_capacity: usize,
}

impl<'a> BatchTaskPlanner<'a> {
    fn build(self) -> Result<Vec<BatchTask<'a>>, SegmentationError> {
        if !self.use_warm_start_b32 {
            return Ok((0..self.total_windows.div_ceil(self.batch_size))
                .map(|batch_idx| {
                    let start = batch_idx * self.batch_size;
                    let end = (start + self.batch_size).min(self.total_windows);
                    BatchTask {
                        batch_idx,
                        start,
                        end,
                        batch_capacity: self.batch_size,
                        model: self.shared_model,
                    }
                })
                .collect());
        }

        let Some(small_model) = self.small_model else {
            return Err(SegmentationError::Invariant {
                context: "parallel segmentation warm start",
                message: "missing native b32 model".to_owned(),
            });
        };

        let mut tasks = Vec::new();
        let mut start = 0usize;
        let mut batch_idx = 0usize;
        let warm_start_end = self.total_windows.min(self.warm_start_small_windows);

        while start < warm_start_end {
            let end = (start + self.warm_start_batch_capacity).min(self.total_windows);
            tasks.push(BatchTask {
                batch_idx,
                start,
                end,
                batch_capacity: self.warm_start_batch_capacity,
                model: small_model,
            });
            start = end;
            batch_idx += 1;
        }

        while start < self.total_windows {
            let end = (start + LARGE_BATCH_SIZE).min(self.total_windows);
            tasks.push(BatchTask {
                batch_idx,
                start,
                end,
                batch_capacity: LARGE_BATCH_SIZE,
                model: self.shared_model,
            });
            start = end;
            batch_idx += 1;
        }

        Ok(tasks)
    }
}

struct ParallelBatchExecutor<'a> {
    windows: &'a ParallelWindows<'a>,
    tx: Sender<Array2<f32>>,
    tasks: Vec<BatchTask<'a>>,
    num_workers: usize,
    window_samples: usize,
    profile: &'a ParallelProfile,
}

impl<'a> ParallelBatchExecutor<'a> {
    fn run(self) -> Result<(), SegmentationError> {
        let (batch_tx, batch_rx) = crossbeam_channel::unbounded::<(usize, Vec<Array2<f32>>)>();

        std::thread::scope(|scope| {
            let merge_handle = scope.spawn(move || -> Result<(), SegmentationError> {
                let mut next_batch = 0usize;
                let mut pending = BTreeMap::<usize, Vec<Array2<f32>>>::new();

                for (batch_idx, results) in batch_rx {
                    pending.insert(batch_idx, results);
                    while let Some(results) = pending.remove(&next_batch) {
                        for result in results {
                            self.tx.send(result)?;
                        }
                        next_batch += 1;
                    }
                }

                Ok(())
            });

            let worker_error = WorkerErrorSlot::default();
            rayon::scope(|rscope| {
                let next_task = Arc::new(AtomicUsize::new(0));
                let worker_count = self.tasks.len().min(self.num_workers.max(1));

                for _worker_idx in 0..worker_count {
                    let tasks = &self.tasks;
                    let windows = self.windows;
                    let batch_tx = batch_tx.clone();
                    let next_task = Arc::clone(&next_task);
                    let worker_error = worker_error.clone();
                    let profile = self.profile;
                    let window_samples = self.window_samples;

                    rscope.spawn(move |_| {
                        let mut scratch_by_capacity =
                            BTreeMap::<usize, (CachedInputShape, Vec<f32>)>::new();

                        loop {
                            let task_idx = next_task.fetch_add(1, Ordering::Relaxed);
                            let Some(task) = tasks.get(task_idx) else {
                                break;
                            };
                            let actual_batch = task.end - task.start;
                            let (cached_batch, batch_buf) = scratch_by_capacity
                                .entry(task.batch_capacity)
                                .or_insert_with(|| {
                                    (
                                        CachedInputShape::new(
                                            "input",
                                            &[task.batch_capacity, 1, window_samples],
                                        ),
                                        vec![0.0f32; task.batch_capacity * window_samples],
                                    )
                                });

                            batch_buf.fill(0.0);
                            for (batch_offset, window_idx) in (task.start..task.end).enumerate() {
                                let Ok(window) =
                                    windows.window(window_idx, "parallel segmentation batch")
                                else {
                                    worker_error.record(SegmentationError::Invariant {
                                        context: "parallel segmentation batch",
                                        message: format!(
                                            "failed to resolve window {window_idx} for batch {}",
                                            task.batch_idx
                                        ),
                                    });
                                    return;
                                };
                                let dst = batch_offset * window_samples;
                                batch_buf[dst..dst + window.len()].copy_from_slice(window);
                            }

                            let batch_start = std::time::Instant::now();
                            let predict = task
                                .model
                                .predict_cached(&[(&*cached_batch, batch_buf.as_slice())])
                                .map_err(|error| {
                                    SegmentationError::Ort(ort::Error::new(error.to_string()))
                                });
                            let Ok((data, out_shape)) = predict else {
                                worker_error.record(predict.unwrap_err());
                                return;
                            };
                            let batch_us = batch_start.elapsed().as_micros() as u64;
                            profile.record_batch(
                                task.batch_idx,
                                task.batch_capacity,
                                actual_batch,
                                batch_us,
                            );

                            let frames = out_shape[1];
                            let classes = out_shape[2];
                            let stride = frames * classes;
                            let mut results = Vec::with_capacity(actual_batch);
                            for batch_offset in 0..actual_batch {
                                let start = batch_offset * stride;
                                let result = segmentation_array_from_slice(
                                    frames,
                                    classes,
                                    &data[start..start + stride],
                                    "parallel segmentation batched output",
                                );
                                let Ok(result) = result else {
                                    worker_error.record(result.unwrap_err());
                                    return;
                                };
                                results.push(result);
                            }

                            if batch_tx.send((task.batch_idx, results)).is_err() {
                                return;
                            }
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
        })
    }
}

struct ParallelSingleExecutor<'a> {
    windows: &'a ParallelWindows<'a>,
    tx: Sender<Array2<f32>>,
    model: &'a SharedCoreMlModel,
    num_workers: usize,
    window_samples: usize,
    profile: &'a ParallelProfile,
}

impl<'a> ParallelSingleExecutor<'a> {
    fn run(self) -> Result<(), SegmentationError> {
        let total_windows = self.windows.total_windows();
        let chunk_size = total_windows.div_ceil(self.num_workers);
        let actual_workers = total_windows.div_ceil(chunk_size).min(self.num_workers);

        let mut worker_txs = Vec::with_capacity(actual_workers);
        let mut worker_rxs = Vec::with_capacity(actual_workers);
        for _ in 0..actual_workers {
            let (worker_tx, worker_rx) = crossbeam_channel::unbounded::<Array2<f32>>();
            worker_txs.push(worker_tx);
            worker_rxs.push(worker_rx);
        }

        std::thread::scope(|scope| {
            let merge_handle = scope.spawn(move || -> Result<(), SegmentationError> {
                for worker_rx in &worker_rxs {
                    for result in worker_rx {
                        self.tx.send(result)?;
                    }
                }
                Ok(())
            });

            let worker_error = WorkerErrorSlot::default();
            rayon::scope(|rscope| {
                for (worker_idx, worker_tx) in worker_txs.into_iter().enumerate() {
                    let start = worker_idx * chunk_size;
                    let end = (start + chunk_size).min(total_windows);
                    let windows = self.windows;
                    let model = self.model;
                    let worker_error = worker_error.clone();
                    let profile = self.profile;
                    let window_samples = self.window_samples;

                    rscope.spawn(move |_| {
                        let cached_shape = CachedInputShape::new("input", &[1, 1, window_samples]);
                        let mut buffer = Array3::<f32>::zeros((1, 1, window_samples));

                        for window_idx in start..end {
                            let Ok(window) =
                                windows.window(window_idx, "parallel segmentation worker")
                            else {
                                worker_error.record(SegmentationError::Invariant {
                                    context: "parallel segmentation worker",
                                    message: format!(
                                        "failed to resolve window {window_idx} for worker {worker_idx}"
                                    ),
                                });
                                return;
                            };

                            buffer.fill(0.0);
                            buffer
                                .slice_mut(s![0, 0, ..window.len()])
                                .assign(&ndarray::ArrayView1::from(window));
                            let input_data =
                                array3_slice(&buffer, "parallel segmentation worker input");
                            let Ok(input_data) = input_data else {
                                worker_error.record(input_data.unwrap_err());
                                return;
                            };

                            let predict_start = std::time::Instant::now();
                            let predict = model.predict_cached(&[(&cached_shape, input_data)]).map_err(
                                |error| SegmentationError::Ort(ort::Error::new(error.to_string())),
                            );
                            let Ok((data, out_shape)) = predict else {
                                worker_error.record(predict.unwrap_err());
                                return;
                            };
                            let predict_us = predict_start.elapsed().as_micros() as u64;
                            profile.record_single(worker_idx, predict_us);

                            let result = segmentation_array(
                                out_shape[1],
                                out_shape[2],
                                data,
                                "parallel segmentation worker output",
                            );
                            let Ok(result) = result else {
                                worker_error.record(result.unwrap_err());
                                return;
                            };

                            if worker_tx.send(result).is_err() {
                                return;
                            }
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
        let windows = ParallelWindows::collect(audio, self.window_samples, self.step_samples);
        let total_windows = windows.total_windows();
        if windows.is_empty() {
            return Ok(0);
        }

        let Some((shared_model, batch_size)) = self.select_parallel_native_model(total_windows)
        else {
            return self.run_streaming(audio, tx);
        };

        let seg_start = std::time::Instant::now();
        let profile = ParallelProfile::new();
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
        let use_warm_start_b32 = batch_size == LARGE_BATCH_SIZE
            && total_windows < 1024
            && total_windows > PRIMARY_BATCH_SIZE
            && warm_start_small_windows > PRIMARY_BATCH_SIZE
            && self.native_batched_session.is_some();

        if batch_size > 1 {
            let tasks = BatchTaskPlanner {
                shared_model,
                small_model: self.native_batched_session.as_ref(),
                total_windows,
                batch_size,
                use_warm_start_b32,
                warm_start_small_windows,
                warm_start_batch_capacity,
            }
            .build()?;

            ParallelBatchExecutor {
                windows: &windows,
                tx,
                tasks,
                num_workers,
                window_samples: self.window_samples,
                profile: &profile,
            }
            .run()?;
        } else {
            ParallelSingleExecutor {
                windows: &windows,
                tx,
                model: shared_model,
                num_workers: num_workers.max(1),
                window_samples: self.window_samples,
                profile: &profile,
            }
            .run()?;
        }

        profile.log_completion(ParallelRunSummary {
            total_windows,
            num_workers,
            est_embed_chunks,
            use_warm_start_b32,
            warm_start_small_windows,
            warm_start_batch_capacity,
            total_seg: seg_start.elapsed(),
        });

        Ok(total_windows)
    }
}
