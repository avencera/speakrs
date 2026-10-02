use crossbeam_channel::Sender;
use ndarray::Array2;
use tracing::debug;

use super::{PRIMARY_BATCH_SIZE, SegmentationBackend, SegmentationError, SegmentationModel};
use crate::inference::segmentation::tensor::SegmentationWindows;

impl SegmentationModel {
    /// Run segmentation on audio, streaming raw logits through a channel
    ///
    /// Same logic as `run()`, but sends each decoded window through `tx` as it's produced
    /// instead of collecting into a Vec. Returns total window count
    pub fn run_streaming(
        &mut self,
        audio: &[f32],
        tx: Sender<Array2<f32>>,
    ) -> Result<usize, SegmentationError> {
        let windows = SegmentationWindows::collect(audio, self.window_spec());
        let total_windows = windows.total_windows();
        if windows.is_empty() {
            return Ok(0);
        }

        let seg_start = std::time::Instant::now();
        let mut seg_infer_time = std::time::Duration::ZERO;
        let mut seg_batched = 0u32;
        let mut seg_single = 0u32;

        let has_batched = self.has_batched();
        let zeros = vec![0.0f32; self.window_samples()];

        let mut next_idx = 0;
        while next_idx < total_windows {
            let remaining = total_windows - next_idx;

            if remaining >= PRIMARY_BATCH_SIZE && has_batched {
                let batch: Vec<&[f32]> = (next_idx..next_idx + PRIMARY_BATCH_SIZE)
                    .map(|idx| windows.window(idx, "streaming segmentation batch"))
                    .collect::<Result<_, _>>()?;

                let t = std::time::Instant::now();
                let results = self.run_batch(&batch)?;
                seg_infer_time += t.elapsed();
                seg_batched += 1;
                for r in results {
                    tx.send(r)?;
                }
                next_idx += PRIMARY_BATCH_SIZE;
                continue;
            }

            if remaining > 1 && has_batched {
                let mut batch: Vec<&[f32]> = (next_idx..total_windows)
                    .map(|idx| windows.window(idx, "streaming segmentation tail batch"))
                    .collect::<Result<_, _>>()?;
                batch.resize(PRIMARY_BATCH_SIZE, &zeros[..]);

                let t = std::time::Instant::now();
                let results = self.run_batch(&batch)?;
                seg_infer_time += t.elapsed();
                seg_batched += 1;
                for r in results.into_iter().take(remaining) {
                    tx.send(r)?;
                }
                next_idx = total_windows;
                continue;
            }

            let t = std::time::Instant::now();
            let result =
                self.run_window(windows.window(next_idx, "streaming segmentation single")?)?;
            seg_infer_time += t.elapsed();
            seg_single += 1;
            tx.send(result)?;
            next_idx += 1;
        }

        let total_seg = seg_start.elapsed();
        debug!(
            windows = total_windows,
            seg_batched,
            seg_single,
            seg_infer_ms = seg_infer_time.as_millis(),
            seg_total_ms = total_seg.as_millis(),
            seg_overhead_ms = (total_seg - seg_infer_time).as_millis(),
            "Segmentation thread profile"
        );

        Ok(total_windows)
    }

    /// Run segmentation on audio, returning raw logits per window
    ///
    /// Returns `Vec<Array2<f32>>` where each element is [frames, 7] logits
    pub fn run(&mut self, audio: &[f32]) -> Result<Vec<Array2<f32>>, SegmentationError> {
        let windows = SegmentationWindows::collect(audio, self.window_spec());
        let total_windows = windows.total_windows();
        let has_batched = self.has_batched();
        let mut results = Vec::with_capacity(total_windows);
        let mut next_idx = 0;

        while next_idx < total_windows {
            let remaining = total_windows - next_idx;
            if remaining >= PRIMARY_BATCH_SIZE && has_batched {
                let batch: Vec<&[f32]> = (next_idx..next_idx + PRIMARY_BATCH_SIZE)
                    .map(|idx| windows.window(idx, "segmentation run batch window"))
                    .collect::<Result<_, _>>()?;
                results.extend(self.run_batch(&batch)?);
                next_idx += PRIMARY_BATCH_SIZE;
                continue;
            }

            let window = windows.window(next_idx, "segmentation run tail window")?;
            results.push(self.run_window(window)?);
            next_idx += 1;
        }

        Ok(results)
    }

    /// Whether the backend has a batch-32 model for full and padded tail batches
    fn has_batched(&self) -> bool {
        match &self.backend {
            #[cfg(feature = "_ort")]
            SegmentationBackend::Ort(backend) => backend.has_batched(),
            #[cfg(feature = "coreml")]
            SegmentationBackend::CoreMl(_) => true,
            #[cfg(feature = "cuda")]
            SegmentationBackend::Cuda(_) => true,
        }
    }

    fn run_window(&mut self, window: &[f32]) -> Result<Array2<f32>, SegmentationError> {
        match &mut self.backend {
            #[cfg(feature = "_ort")]
            SegmentationBackend::Ort(backend) => backend.run_window(window),
            #[cfg(feature = "coreml")]
            SegmentationBackend::CoreMl(backend) => backend.run_window(window),
            #[cfg(feature = "cuda")]
            SegmentationBackend::Cuda(backend) => backend.run_window(window),
        }
    }

    fn run_batch(&mut self, windows: &[&[f32]]) -> Result<Vec<Array2<f32>>, SegmentationError> {
        match &mut self.backend {
            #[cfg(feature = "_ort")]
            SegmentationBackend::Ort(backend) => backend.run_batch(windows),
            #[cfg(feature = "coreml")]
            SegmentationBackend::CoreMl(backend) => backend.run_batch(windows),
            #[cfg(feature = "cuda")]
            SegmentationBackend::Cuda(backend) => backend.run_batch(windows),
        }
    }
}
