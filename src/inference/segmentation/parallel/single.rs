use crossbeam_channel::{Receiver, Sender};
use ndarray::{Array2, Array3, s};

use super::{
    ParallelProfile, SegmentationError, SegmentationWindows, SharedCoreMlModel, WorkerErrorSlot,
    segmentation_array, worker_panic,
};
use crate::inference::coreml::CachedInputShape;
use crate::inference::segmentation::tensor::array3_slice;

type WorkerTx = Sender<Array2<f32>>;
type WorkerRx = Receiver<Array2<f32>>;

pub(super) struct ParallelSingleExecutor<'a> {
    pub(super) windows: &'a SegmentationWindows<'a>,
    pub(super) tx: Sender<Array2<f32>>,
    pub(super) model: &'a SharedCoreMlModel,
    pub(super) num_workers: usize,
    pub(super) window_samples: usize,
    pub(super) profile: &'a ParallelProfile,
}

struct WorkerResultMerger {
    tx: Sender<Array2<f32>>,
    worker_rxs: Vec<WorkerRx>,
}

impl WorkerResultMerger {
    fn new(tx: Sender<Array2<f32>>, worker_rxs: Vec<WorkerRx>) -> Self {
        Self { tx, worker_rxs }
    }

    fn run(self) -> Result<(), SegmentationError> {
        for worker_rx in &self.worker_rxs {
            for result in worker_rx {
                self.tx.send(result)?;
            }
        }

        Ok(())
    }
}

struct SingleScratch {
    cached_shape: CachedInputShape,
    buffer: Array3<f32>,
}

impl SingleScratch {
    fn new(window_samples: usize) -> Self {
        Self {
            cached_shape: CachedInputShape::new("input", &[1, 1, window_samples]),
            buffer: Array3::<f32>::zeros((1, 1, window_samples)),
        }
    }

    fn load_window(&mut self, window: &[f32]) {
        self.buffer.fill(0.0);
        self.buffer
            .slice_mut(s![0, 0, ..window.len()])
            .assign(&ndarray::ArrayView1::from(window));
    }

    fn input_data(&self) -> Result<&[f32], SegmentationError> {
        array3_slice(&self.buffer, "parallel segmentation worker input")
    }
}

struct WorkerChannels {
    txs: Vec<WorkerTx>,
    rxs: Vec<WorkerRx>,
}

impl WorkerChannels {
    fn new(worker_count: usize) -> Self {
        let mut txs = Vec::with_capacity(worker_count);
        let mut rxs = Vec::with_capacity(worker_count);

        for _ in 0..worker_count {
            let (worker_tx, worker_rx) = crossbeam_channel::unbounded::<Array2<f32>>();
            txs.push(worker_tx);
            rxs.push(worker_rx);
        }

        Self { txs, rxs }
    }
}

struct SingleWorker<'ctx, 'a> {
    worker_idx: usize,
    start: usize,
    end: usize,
    windows: &'ctx SegmentationWindows<'a>,
    model: &'ctx SharedCoreMlModel,
    worker_tx: Sender<Array2<f32>>,
    worker_error: WorkerErrorSlot,
    profile: &'ctx ParallelProfile,
    window_samples: usize,
}

impl<'ctx, 'a> SingleWorker<'ctx, 'a> {
    fn run(self) {
        let mut scratch = SingleScratch::new(self.window_samples);

        for window_idx in self.start..self.end {
            let result = match self.process_window(window_idx, &mut scratch) {
                Ok(result) => result,
                Err(error) => {
                    self.worker_error.record(error);
                    return;
                }
            };

            if self.worker_tx.send(result).is_err() {
                return;
            }
        }
    }

    fn process_window(
        &self,
        window_idx: usize,
        scratch: &mut SingleScratch,
    ) -> Result<Array2<f32>, SegmentationError> {
        let window = self.resolve_window(window_idx)?;
        scratch.load_window(window);
        let (data, frames, classes) = self.predict(scratch)?;
        self.decode_result(data, frames, classes)
    }

    fn resolve_window(&self, window_idx: usize) -> Result<&[f32], SegmentationError> {
        self.windows
            .window(window_idx, "parallel segmentation worker")
            .map_err(|_| SegmentationError::Invariant {
                context: "parallel segmentation worker",
                message: format!(
                    "failed to resolve window {window_idx} for worker {}",
                    self.worker_idx
                ),
            })
    }

    fn predict(
        &self,
        scratch: &SingleScratch,
    ) -> Result<(Vec<f32>, usize, usize), SegmentationError> {
        let predict_start = std::time::Instant::now();
        let (data, out_shape) = self
            .model
            .predict_cached(&[(&scratch.cached_shape, scratch.input_data()?)])
            .map_err(|error| SegmentationError::Ort(ort::Error::new(error.to_string())))?;
        let predict_us = predict_start.elapsed().as_micros() as u64;
        self.profile.record_single(self.worker_idx, predict_us);

        Ok((data, out_shape[1], out_shape[2]))
    }

    fn decode_result(
        &self,
        data: Vec<f32>,
        frames: usize,
        classes: usize,
    ) -> Result<Array2<f32>, SegmentationError> {
        segmentation_array(frames, classes, data, "parallel segmentation worker output")
    }
}

struct SingleWorkerPool<'ctx, 'a> {
    windows: &'ctx SegmentationWindows<'a>,
    model: &'ctx SharedCoreMlModel,
    chunk_size: usize,
    total_windows: usize,
    window_samples: usize,
    profile: &'ctx ParallelProfile,
    worker_error: WorkerErrorSlot,
}

impl<'ctx, 'a> SingleWorkerPool<'ctx, 'a> {
    fn run(&self, worker_txs: Vec<WorkerTx>) {
        rayon::scope(|rscope| {
            for (worker_idx, worker_tx) in worker_txs.into_iter().enumerate() {
                let worker = self.worker(worker_idx, worker_tx);
                rscope.spawn(move |_| worker.run());
            }
        });
    }

    fn worker(&self, worker_idx: usize, worker_tx: WorkerTx) -> SingleWorker<'ctx, 'a> {
        let start = worker_idx * self.chunk_size;
        let end = (start + self.chunk_size).min(self.total_windows);

        SingleWorker {
            worker_idx,
            start,
            end,
            windows: self.windows,
            model: self.model,
            worker_tx,
            worker_error: self.worker_error.clone(),
            profile: self.profile,
            window_samples: self.window_samples,
        }
    }
}

impl<'a> ParallelSingleExecutor<'a> {
    pub(super) fn run(self) -> Result<(), SegmentationError> {
        let Self {
            windows,
            tx,
            model,
            num_workers,
            window_samples,
            profile,
        } = self;
        let total_windows = windows.total_windows();
        let chunk_size = total_windows.div_ceil(num_workers);
        let actual_workers = total_windows.div_ceil(chunk_size).min(num_workers);
        let WorkerChannels { txs, rxs } = WorkerChannels::new(actual_workers);

        std::thread::scope(|scope| {
            let merge_handle = scope.spawn(move || WorkerResultMerger::new(tx, rxs).run());

            let worker_error = WorkerErrorSlot::default();
            SingleWorkerPool {
                windows,
                model,
                chunk_size,
                total_windows,
                window_samples,
                profile,
                worker_error: worker_error.clone(),
            }
            .run(txs);

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
