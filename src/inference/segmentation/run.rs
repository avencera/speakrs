use crossbeam_channel::Sender;
use ndarray::Array2;
use ort::value::TensorRef;
use tracing::debug;

use super::{PRIMARY_BATCH_SIZE, SegmentationError, SegmentationModel};
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
        let windows = SegmentationWindows::collect(audio, self.window_samples, self.step_samples);
        let total_windows = windows.total_windows();
        if windows.is_empty() {
            return Ok(0);
        }

        let seg_start = std::time::Instant::now();
        let mut seg_infer_time = std::time::Duration::ZERO;
        let mut seg_batched = 0u32;
        let mut seg_single = 0u32;

        let has_batched = self.primary_batched_session.is_some();
        let zeros = vec![0.0f32; self.window_samples];

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
    pub fn run(&mut self, audio: &[f32]) -> Result<Vec<Array2<f32>>, ort::Error> {
        let windows = SegmentationWindows::collect(audio, self.window_samples, self.step_samples);
        let total_windows = windows.total_windows();
        let mut results = Vec::with_capacity(total_windows);
        let mut next_idx = 0;

        while next_idx < total_windows {
            let remaining = total_windows - next_idx;
            if remaining >= PRIMARY_BATCH_SIZE && self.primary_batched_session.is_some() {
                let batch: Vec<&[f32]> = (next_idx..next_idx + PRIMARY_BATCH_SIZE)
                    .map(|idx| windows.window(idx, "segmentation run batch window"))
                    .collect::<Result<_, _>>()
                    .map_err(|error| ort::Error::new(error.to_string()))?;
                results.extend(self.run_batch(&batch)?);
                next_idx += PRIMARY_BATCH_SIZE;
                continue;
            }

            let window = windows
                .window(next_idx, "segmentation run tail window")
                .map_err(|error| ort::Error::new(error.to_string()))?;
            results.push(self.run_window(window)?);
            next_idx += 1;
        }

        Ok(results)
    }

    fn run_window(&mut self, window: &[f32]) -> Result<Array2<f32>, ort::Error> {
        #[cfg(feature = "coreml")]
        if let Some(ref native) = self.native_session {
            return Self::run_native_single(
                native,
                window,
                &mut self.input_buffer,
                &self.cached_single_input_shape,
            );
        }

        self.input_buffer.fill(0.0);
        self.input_buffer
            .slice_mut(ndarray::s![0, 0, ..window.len()])
            .assign(&ndarray::ArrayView1::from(window));
        let input_tensor = TensorRef::from_array_view(self.input_buffer.view())?;

        let outputs = self.session.run(ort::inputs![input_tensor])?;
        let (shape, data) = outputs[0].try_extract_tensor::<f32>()?;

        let frames = shape[1] as usize;
        let classes = shape[2] as usize;

        Array2::from_shape_vec((frames, classes), data.to_vec())
            .map_err(|error| ort::Error::new(format!("segmentation window output shape: {error}")))
    }

    fn run_batch(&mut self, windows: &[&[f32]]) -> Result<Vec<Array2<f32>>, ort::Error> {
        #[cfg(feature = "coreml")]
        if let Some(ref native) = self.native_batched_session {
            return Self::run_native_batch(
                native,
                windows,
                &mut self.primary_batch_input_buffer,
                &self.cached_batch_input_shape,
            );
        }

        self.primary_batch_input_buffer.fill(0.0);
        for (batch_idx, window) in windows.iter().enumerate() {
            self.primary_batch_input_buffer
                .slice_mut(ndarray::s![batch_idx, 0, ..window.len()])
                .assign(&ndarray::ArrayView1::from(*window));
        }
        let input_tensor = TensorRef::from_array_view(self.primary_batch_input_buffer.view())?;

        let outputs = self
            .primary_batched_session
            .as_mut()
            .ok_or_else(|| ort::Error::new("missing primary batched segmentation session"))?
            .run(ort::inputs![input_tensor])?;
        let (shape, data) = outputs[0].try_extract_tensor::<f32>()?;

        let batch = shape[0] as usize;
        let frames = shape[1] as usize;
        let classes = shape[2] as usize;
        let stride = frames * classes;

        (0..batch)
            .map(|batch_idx| {
                let start = batch_idx * stride;
                Array2::from_shape_vec((frames, classes), data[start..start + stride].to_vec())
                    .map_err(|error| {
                        ort::Error::new(format!("segmentation batch output shape: {error}"))
                    })
            })
            .collect::<Result<Vec<_>, _>>()
    }
}
