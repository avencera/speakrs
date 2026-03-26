#![cfg(feature = "coreml")]

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crossbeam_channel::Sender;
use ndarray::Array2;
use tracing::{debug, trace};

use super::SegmentationError;
use super::{LARGE_BATCH_SIZE, PRIMARY_BATCH_SIZE, SegmentationModel};
use crate::inference::coreml::SharedCoreMlModel;
use crate::inference::segmentation::tensor::{
    SegmentationWindows, segmentation_array, segmentation_array_from_slice, worker_panic,
};

mod batch;
mod single;

use batch::ParallelBatchExecutor;
use single::ParallelSingleExecutor;

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

pub(super) struct ParallelProfile {
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

pub(super) struct BatchTask<'a> {
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
        let windows = SegmentationWindows::collect(audio, self.window_samples, self.step_samples);
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
