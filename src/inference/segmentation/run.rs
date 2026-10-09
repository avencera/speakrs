use std::num::NonZeroUsize;

use crossbeam_channel::Sender;
use ndarray::Array2;
use tracing::debug;

#[cfg(any(feature = "migraphx", feature = "coreml", feature = "_cuda"))]
use super::PRIMARY_BATCH_SIZE;
use super::{SegmentationBackend, SegmentationError, SegmentationModel};
use crate::inference::segmentation::tensor::SegmentationWindows;

mod batching;

use batching::{BatchPlan, Batching, Delivery};

impl SegmentationModel {
    /// Run segmentation on audio, streaming raw logits through a channel
    ///
    /// Sends raw logits through `tx` as each window is produced
    /// Fixed-size backends pad multi-window tails; [`Self::run`] uses single-window tails
    /// Returns the total window count
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

        let batching = self.batching();
        let zeros = vec![0.0f32; self.window_samples()];

        let mut next_idx = 0;
        while let Some(remaining) = NonZeroUsize::new(total_windows - next_idx) {
            let plan = batching.plan(remaining, Delivery::Streaming);
            let t = std::time::Instant::now();
            let results = self.run_planned(&windows, next_idx, plan, &zeros)?;
            seg_infer_time += t.elapsed();
            if plan.is_single() {
                seg_single += 1;
            } else {
                seg_batched += 1;
            }

            for result in results {
                tx.send(result)?;
            }
            next_idx += plan.useful();
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
        let batching = self.batching();
        let mut results = Vec::with_capacity(total_windows);
        let mut next_idx = 0;

        while let Some(remaining) = NonZeroUsize::new(total_windows - next_idx) {
            let plan = batching.plan(remaining, Delivery::Collected);
            results.extend(self.run_planned(&windows, next_idx, plan, &[])?);
            next_idx += plan.useful();
        }

        Ok(results)
    }

    /// Batch geometry owned by the loaded backend
    fn batching(&self) -> Batching {
        #[cfg(any(feature = "migraphx", feature = "coreml", feature = "_cuda"))]
        let fixed = Batching::Fixed(
            NonZeroUsize::new(PRIMARY_BATCH_SIZE).expect("nonzero fixed batch size"),
        );
        match &self.backend {
            #[cfg(feature = "cpu")]
            SegmentationBackend::Cpu(backend) => Batching::Useful(
                NonZeroUsize::new(backend.capacity()).expect("native CPU capacity is nonzero"),
            ),
            #[cfg(feature = "migraphx")]
            SegmentationBackend::Ort(backend) => {
                if backend.has_batched() {
                    fixed
                } else {
                    Batching::Single
                }
            }
            #[cfg(feature = "coreml")]
            SegmentationBackend::CoreMl(_) => fixed,
            #[cfg(feature = "_cuda")]
            SegmentationBackend::Cuda(_) => fixed,
        }
    }

    fn run_planned(
        &mut self,
        windows: &SegmentationWindows<'_>,
        next: usize,
        plan: BatchPlan,
        zeros: &[f32],
    ) -> Result<Vec<Array2<f32>>, SegmentationError> {
        if plan.is_single() {
            return self
                .run_window(windows.window(next, "segmentation single window")?)
                .map(|output| vec![output]);
        }

        let mut outputs = self.run_windows(windows, next, plan, zeros)?;
        if outputs.len() != plan.model() {
            return Err(SegmentationError::MalformedOutput {
                context: "segmentation batch output count",
                message: format!("expected {} windows, got {}", plan.model(), outputs.len()),
            });
        }

        outputs.truncate(plan.useful());
        Ok(outputs)
    }

    fn run_window(&mut self, window: &[f32]) -> Result<Array2<f32>, SegmentationError> {
        match &mut self.backend {
            #[cfg(feature = "cpu")]
            SegmentationBackend::Cpu(backend) => backend.run_window(window),
            #[cfg(feature = "migraphx")]
            SegmentationBackend::Ort(backend) => backend.run_window(window),
            #[cfg(feature = "coreml")]
            SegmentationBackend::CoreMl(backend) => backend.run_window(window),
            #[cfg(feature = "_cuda")]
            SegmentationBackend::Cuda(backend) => backend.run_window(window),
        }
    }

    /// Runs the planned windows from `next` as one model batch, zero padded to its size
    fn run_windows(
        &mut self,
        windows: &SegmentationWindows<'_>,
        next: usize,
        plan: BatchPlan,
        zeros: &[f32],
    ) -> Result<Vec<Array2<f32>>, SegmentationError> {
        #[cfg(feature = "_cuda")]
        let window_samples = self.window_samples();
        match &mut self.backend {
            #[cfg(feature = "cpu")]
            SegmentationBackend::Cpu(backend) => {
                backend.run_batch(&window_batch(windows, next, plan, zeros)?)
            }
            #[cfg(feature = "migraphx")]
            SegmentationBackend::Ort(backend) => {
                backend.run_batch(&window_batch(windows, next, plan, zeros)?)
            }
            #[cfg(feature = "coreml")]
            SegmentationBackend::CoreMl(backend) => {
                backend.run_batch(&window_batch(windows, next, plan, zeros)?)
            }
            // the windows of one recording overlap, so the audio they cover goes to the
            // device once and the windows are cut out of it there
            #[cfg(feature = "_cuda")]
            SegmentationBackend::Cuda(backend) => {
                let (span, starts) = windows.span(next, plan.useful(), plan.model());
                // a step longer than the window leaves gaps that the span would upload
                if span.len() > plan.useful() * window_samples {
                    return backend.run_batch(&window_batch(windows, next, plan, zeros)?);
                }
                backend.run_span(span, &starts)
            }
        }
    }
}

/// The planned windows from `next`, zero padded to the model batch
fn window_batch<'a>(
    windows: &'a SegmentationWindows<'_>,
    next: usize,
    plan: BatchPlan,
    zeros: &'a [f32],
) -> Result<Vec<&'a [f32]>, SegmentationError> {
    let mut batch = (next..next + plan.useful())
        .map(|idx| windows.window(idx, "segmentation batch window"))
        .collect::<Result<Vec<_>, _>>()?;
    batch.resize(plan.model(), zeros);
    Ok(batch)
}
