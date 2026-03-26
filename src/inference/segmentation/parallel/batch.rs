use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use crossbeam_channel::{Receiver, Sender};
use ndarray::Array2;

use super::{
    BatchTask, ParallelProfile, SegmentationError, SegmentationWindows, WorkerErrorSlot,
    segmentation_array_from_slice, worker_panic,
};
use crate::inference::coreml::CachedInputShape;

type QueuedBatch = (usize, Vec<Array2<f32>>);

pub(super) struct ParallelBatchExecutor<'a> {
    pub(super) windows: &'a SegmentationWindows<'a>,
    pub(super) tx: Sender<Array2<f32>>,
    pub(super) tasks: Vec<BatchTask<'a>>,
    pub(super) num_workers: usize,
    pub(super) window_samples: usize,
    pub(super) profile: &'a ParallelProfile,
}

struct OrderedBatchMerger {
    tx: Sender<Array2<f32>>,
    next_batch: usize,
    pending: BTreeMap<usize, Vec<Array2<f32>>>,
}

impl OrderedBatchMerger {
    fn new(tx: Sender<Array2<f32>>) -> Self {
        Self {
            tx,
            next_batch: 0,
            pending: BTreeMap::new(),
        }
    }

    fn run(mut self, batch_rx: Receiver<QueuedBatch>) -> Result<(), SegmentationError> {
        for (batch_idx, results) in batch_rx {
            self.insert(batch_idx, results)?;
        }

        Ok(())
    }

    fn insert(
        &mut self,
        batch_idx: usize,
        results: Vec<Array2<f32>>,
    ) -> Result<(), SegmentationError> {
        self.pending.insert(batch_idx, results);
        self.drain_ready_batches()
    }

    fn drain_ready_batches(&mut self) -> Result<(), SegmentationError> {
        while let Some(results) = self.pending.remove(&self.next_batch) {
            self.send_results(results)?;
            self.next_batch += 1;
        }

        Ok(())
    }

    fn send_results(&self, results: Vec<Array2<f32>>) -> Result<(), SegmentationError> {
        for result in results {
            self.tx.send(result)?;
        }

        Ok(())
    }
}

#[derive(Default)]
struct BatchScratch {
    by_capacity: BTreeMap<usize, (CachedInputShape, Vec<f32>)>,
}

impl BatchScratch {
    fn buffer_for(
        &mut self,
        batch_capacity: usize,
        window_samples: usize,
    ) -> (&CachedInputShape, &mut Vec<f32>) {
        let (cached_batch, batch_buf) =
            self.by_capacity.entry(batch_capacity).or_insert_with(|| {
                (
                    CachedInputShape::new("input", &[batch_capacity, 1, window_samples]),
                    vec![0.0f32; batch_capacity * window_samples],
                )
            });

        (&*cached_batch, batch_buf)
    }
}

struct BatchWorker<'ctx, 'a> {
    tasks: &'ctx [BatchTask<'a>],
    windows: &'ctx SegmentationWindows<'a>,
    batch_tx: Sender<QueuedBatch>,
    next_task: Arc<AtomicUsize>,
    worker_error: WorkerErrorSlot,
    profile: &'ctx ParallelProfile,
    window_samples: usize,
}

impl<'ctx, 'a> BatchWorker<'ctx, 'a> {
    fn run(self) {
        let mut scratch = BatchScratch::default();

        while let Some(task) = self.claim_next_task() {
            let results = match self.process_task(task, &mut scratch) {
                Ok(results) => results,
                Err(error) => {
                    self.worker_error.record(error);
                    return;
                }
            };

            if self.batch_tx.send((task.batch_idx, results)).is_err() {
                return;
            }
        }
    }

    fn claim_next_task(&self) -> Option<&BatchTask<'a>> {
        let task_idx = self.next_task.fetch_add(1, Ordering::Relaxed);
        self.tasks.get(task_idx)
    }

    fn process_task(
        &self,
        task: &BatchTask<'a>,
        scratch: &mut BatchScratch,
    ) -> Result<Vec<Array2<f32>>, SegmentationError> {
        let (cached_batch, batch_buf) =
            scratch.buffer_for(task.batch_capacity, self.window_samples);
        self.fill_input(task, batch_buf)?;
        let (data, frames, classes) = self.predict(task, cached_batch, batch_buf.as_slice())?;
        self.decode_results(task, &data, frames, classes)
    }

    fn fill_input(
        &self,
        task: &BatchTask<'a>,
        batch_buf: &mut [f32],
    ) -> Result<(), SegmentationError> {
        batch_buf.fill(0.0);
        for (batch_offset, window_idx) in (task.start..task.end).enumerate() {
            let window = self
                .windows
                .window(window_idx, "parallel segmentation batch")
                .map_err(|_| SegmentationError::Invariant {
                    context: "parallel segmentation batch",
                    message: format!(
                        "failed to resolve window {window_idx} for batch {}",
                        task.batch_idx
                    ),
                })?;
            let dst = batch_offset * self.window_samples;
            batch_buf[dst..dst + window.len()].copy_from_slice(window);
        }

        Ok(())
    }

    fn predict(
        &self,
        task: &BatchTask<'a>,
        cached_batch: &CachedInputShape,
        batch_buf: &[f32],
    ) -> Result<(Vec<f32>, usize, usize), SegmentationError> {
        let actual_batch = task.end - task.start;
        let batch_start = std::time::Instant::now();
        let (data, out_shape) = task
            .model
            .predict_cached(&[(cached_batch, batch_buf)])
            .map_err(|error| SegmentationError::Ort(ort::Error::new(error.to_string())))?;
        let batch_us = batch_start.elapsed().as_micros() as u64;
        self.profile
            .record_batch(task.batch_idx, task.batch_capacity, actual_batch, batch_us);

        Ok((data, out_shape[1], out_shape[2]))
    }

    fn decode_results(
        &self,
        task: &BatchTask<'a>,
        data: &[f32],
        frames: usize,
        classes: usize,
    ) -> Result<Vec<Array2<f32>>, SegmentationError> {
        let actual_batch = task.end - task.start;
        let stride = frames * classes;
        let mut results = Vec::with_capacity(actual_batch);

        for batch_offset in 0..actual_batch {
            let start = batch_offset * stride;
            let result = segmentation_array_from_slice(
                frames,
                classes,
                &data[start..start + stride],
                "parallel segmentation batched output",
            )?;
            results.push(result);
        }

        Ok(results)
    }
}

impl<'a> ParallelBatchExecutor<'a> {
    pub(super) fn run(self) -> Result<(), SegmentationError> {
        let Self {
            windows,
            tx,
            tasks,
            num_workers,
            window_samples,
            profile,
        } = self;
        let (batch_tx, batch_rx) = crossbeam_channel::unbounded::<QueuedBatch>();

        std::thread::scope(|scope| {
            let merge_handle = scope.spawn(move || OrderedBatchMerger::new(tx).run(batch_rx));

            let worker_error = WorkerErrorSlot::default();
            Self::run_workers(
                &tasks,
                windows,
                &batch_tx,
                num_workers,
                window_samples,
                profile,
                &worker_error,
            );

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

    fn run_workers(
        tasks: &[BatchTask<'a>],
        windows: &SegmentationWindows<'a>,
        batch_tx: &Sender<QueuedBatch>,
        num_workers: usize,
        window_samples: usize,
        profile: &ParallelProfile,
        worker_error: &WorkerErrorSlot,
    ) {
        rayon::scope(|rscope| {
            let next_task = Arc::new(AtomicUsize::new(0));
            let worker_count = tasks.len().min(num_workers.max(1));

            for _worker_idx in 0..worker_count {
                let worker = BatchWorker {
                    tasks,
                    windows,
                    batch_tx: batch_tx.clone(),
                    next_task: Arc::clone(&next_task),
                    worker_error: worker_error.clone(),
                    profile,
                    window_samples,
                };

                rscope.spawn(move |_| worker.run());
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn batch(values: &[f32]) -> Vec<Array2<f32>> {
        values
            .iter()
            .map(|value| Array2::from_elem((1, 1), *value))
            .collect()
    }

    fn received_values(rx: &Receiver<Array2<f32>>) -> Vec<f32> {
        rx.try_iter().map(|result| result[[0, 0]]).collect()
    }

    #[test]
    fn merger_preserves_batch_item_order() {
        let (tx, rx) = crossbeam_channel::unbounded();
        let mut merger = OrderedBatchMerger::new(tx);

        merger.insert(0, batch(&[1.0, 2.0, 3.0])).unwrap();

        assert_eq!(received_values(&rx), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn merger_drains_contiguous_pending_batches() {
        let (tx, rx) = crossbeam_channel::unbounded();
        let mut merger = OrderedBatchMerger::new(tx);

        merger.insert(1, batch(&[10.0, 11.0])).unwrap();
        merger.insert(3, batch(&[30.0])).unwrap();
        assert!(received_values(&rx).is_empty());

        merger.insert(0, batch(&[0.0, 1.0])).unwrap();
        assert_eq!(received_values(&rx), vec![0.0, 1.0, 10.0, 11.0]);

        merger.insert(2, batch(&[20.0, 21.0])).unwrap();
        assert_eq!(received_values(&rx), vec![20.0, 21.0, 30.0]);
    }
}
