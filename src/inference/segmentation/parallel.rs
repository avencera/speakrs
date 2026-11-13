#![cfg(feature = "coreml")]

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crossbeam_channel::Sender;
use ndarray::Array2;
use tracing::{debug, trace};

use super::SegmentationError;
use super::native::ParallelModel;
use super::{PRIMARY_BATCH_SIZE, SegmentationModel};
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
    warm_start: Option<WarmStart>,
    total_seg: std::time::Duration,
}

/// Leading batch-32 predictions that get the first windows of a batch-64 run to the
/// embedding stage sooner
#[derive(Clone, Copy)]
struct WarmStart {
    /// Windows the batch-32 model runs before the batch-64 model takes over
    small_windows: usize,
    /// Useful windows in each warm-start prediction
    batch_capacity: usize,
}

impl WarmStart {
    fn plan(
        total_windows: usize,
        warm_start_target_windows: Option<usize>,
        est_embed_chunks: Option<usize>,
    ) -> Option<Self> {
        let upper_medium_warm_start = est_embed_chunks
            .map(|chunks| chunks >= 12)
            .unwrap_or(total_windows >= 640);
        let default_small_windows = if upper_medium_warm_start {
            PRIMARY_BATCH_SIZE * 2
        } else {
            140
        };
        let small_windows = warm_start_target_windows.unwrap_or(default_small_windows);
        let batch_capacity = if upper_medium_warm_start {
            PRIMARY_BATCH_SIZE / 2
        } else {
            28
        };

        let applies = total_windows < 1024
            && total_windows > PRIMARY_BATCH_SIZE
            && small_windows > PRIMARY_BATCH_SIZE;
        applies.then_some(Self {
            small_windows,
            batch_capacity,
        })
    }
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
            seg_warm_start_b32 = summary.warm_start.is_some(),
            seg_warm_start_windows = summary.warm_start.map_or(0, |warm_start| {
                summary.total_windows.min(warm_start.small_windows)
            }),
            seg_warm_start_batch_capacity = summary
                .warm_start
                .map_or(0, |warm_start| warm_start.batch_capacity),
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
    model: &'a SharedCoreMlModel,
    total_windows: usize,
    batch_size: usize,
    warm_start: Option<(&'a SharedCoreMlModel, WarmStart)>,
}

impl<'a> BatchTaskPlanner<'a> {
    fn build(self) -> Vec<BatchTask<'a>> {
        let mut tasks = Vec::new();
        let mut start = 0usize;

        if let Some((small_model, warm_start)) = self.warm_start {
            let warm_start_end = self.total_windows.min(warm_start.small_windows);
            while start < warm_start_end {
                let end = (start + warm_start.batch_capacity).min(self.total_windows);
                tasks.push(BatchTask {
                    batch_idx: tasks.len(),
                    start,
                    end,
                    batch_capacity: warm_start.batch_capacity,
                    model: small_model,
                });
                start = end;
            }
        }

        while start < self.total_windows {
            let end = (start + self.batch_size).min(self.total_windows);
            tasks.push(BatchTask {
                batch_idx: tasks.len(),
                start,
                end,
                batch_capacity: self.batch_size,
                model: self.model,
            });
            start = end;
        }

        tasks
    }
}

impl SegmentationModel {
    /// Run segmentation with up to `num_workers` workers sharing the loaded CoreML models
    ///
    /// The batched path runs at most three predictions at once, because CoreML keeps a set of
    /// GPU buffers for every concurrent prediction a model has served
    ///
    /// Results are sent through `tx` in chunk_idx order
    pub fn run_streaming_parallel(
        &mut self,
        audio: &[f32],
        tx: Sender<Array2<f32>>,
        num_workers: usize,
        warm_start_target_windows: Option<usize>,
    ) -> Result<usize, SegmentationError> {
        let windows = SegmentationWindows::collect(audio, self.window_spec());
        let total_windows = windows.total_windows();
        if windows.is_empty() {
            return Ok(0);
        }

        let Some(backend) = self.coreml_backend() else {
            return self.run_streaming(audio, tx);
        };
        let parallel_model = backend.select_parallel_model(total_windows);

        let seg_start = std::time::Instant::now();
        let profile = ParallelProfile::new();
        let est_embed_chunks = warm_start_target_windows
            .and_then(|target| target.checked_div(2))
            .filter(|&chunk_windows| chunk_windows > 0)
            .map(|chunk_windows| total_windows.div_ceil(chunk_windows));
        let warm_start = match parallel_model {
            ParallelModel::Batched {
                warm_start: Some(small_model),
                ..
            } => WarmStart::plan(total_windows, warm_start_target_windows, est_embed_chunks)
                .map(|warm_start| (small_model, warm_start)),
            ParallelModel::Batched {
                warm_start: None, ..
            }
            | ParallelModel::Single(_) => None,
        };

        let workers = match parallel_model {
            ParallelModel::Batched {
                model, batch_size, ..
            } => {
                let tasks = BatchTaskPlanner {
                    model,
                    total_windows,
                    batch_size,
                    warm_start,
                }
                .build();

                ParallelBatchExecutor {
                    windows: &windows,
                    tx,
                    tasks,
                    num_workers,
                    window_samples: self.window_samples(),
                    profile: &profile,
                }
                .run()?
            }
            ParallelModel::Single(model) => ParallelSingleExecutor {
                windows: &windows,
                tx,
                model,
                num_workers: num_workers.max(1),
                window_samples: self.window_samples(),
                profile: &profile,
            }
            .run()?,
        };

        profile.log_completion(ParallelRunSummary {
            total_windows,
            num_workers: workers,
            est_embed_chunks,
            warm_start: warm_start.map(|(_, warm_start)| warm_start),
            total_seg: seg_start.elapsed(),
        });

        Ok(total_windows)
    }
}
