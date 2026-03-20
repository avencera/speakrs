use crossbeam_channel::Sender;
use ndarray::Array2;
use ort::session::Session;
use ort::value::TensorRef;

#[cfg(feature = "coreml")]
use crate::inference::coreml::{
    CachedInputShape, CoreMlModel, GpuPrecision, SharedCoreMlModel, coreml_model_path,
    coreml_w8a16_model_path,
};
use crate::inference::{ExecutionMode, with_execution_mode};
#[cfg(feature = "coreml")]
use objc2_core_ml::MLComputeUnits;

#[derive(Debug, thiserror::Error)]
pub enum SegmentationError {
    #[error(transparent)]
    Ort(#[from] ort::Error),
    #[error("receiver disconnected")]
    Disconnected(#[from] crossbeam_channel::SendError<Array2<f32>>),
}

const PRIMARY_BATCH_SIZE: usize = 64;

#[cfg(feature = "coreml")]
type SegParallelResult = Result<Vec<Vec<(usize, Array2<f32>)>>, SegmentationError>;

pub struct SegmentationModel {
    model_path: String,
    mode: ExecutionMode,
    session: Session,
    primary_batched_session: Option<Session>,
    #[cfg(feature = "coreml")]
    native_session: Option<SharedCoreMlModel>,
    #[cfg(feature = "coreml")]
    native_batched_session: Option<SharedCoreMlModel>,
    #[cfg(feature = "coreml")]
    cached_single_input_shape: CachedInputShape,
    #[cfg(feature = "coreml")]
    cached_batch_input_shape: CachedInputShape,
    input_buffer: ndarray::Array3<f32>,
    primary_batch_input_buffer: ndarray::Array3<f32>,
    window_samples: usize,
    step_samples: usize,
    sample_rate: usize,
}

// SAFETY: SegmentationModel is only used from one thread at a time via &mut self.
// The non-Send fields (CachedInputShape) contain Objective-C objects that are safe
// to move between threads when not accessed concurrently.
// SharedCoreMlModel is already Send + Sync
#[cfg(feature = "coreml")]
unsafe impl Send for SegmentationModel {}

impl SegmentationModel {
    /// Load a segmentation-3.0 ONNX model
    pub fn new(model_path: &str, step_duration: f32) -> Result<Self, ort::Error> {
        Self::with_mode(model_path, step_duration, ExecutionMode::Cpu)
    }

    /// Load a segmentation-3.0 ONNX model with the requested execution mode
    pub fn with_mode(
        model_path: &str,
        step_duration: f32,
        mode: ExecutionMode,
    ) -> Result<Self, ort::Error> {
        let sample_rate = 16000;
        let window_duration = 10.0;
        let window_samples = (window_duration * sample_rate as f32) as usize;
        let step_samples = (step_duration * sample_rate as f32) as usize;

        Ok(Self {
            model_path: model_path.to_owned(),
            mode,
            session: Self::build_session(model_path, mode)?,
            primary_batched_session: batched_model_path(model_path, PRIMARY_BATCH_SIZE)
                .filter(|path| path.exists())
                .map(|path| Self::build_session(path.to_str().unwrap(), mode))
                .transpose()?,
            #[cfg(feature = "coreml")]
            native_session: Self::load_native_coreml(model_path, mode),
            #[cfg(feature = "coreml")]
            native_batched_session: Self::load_native_coreml_batched(model_path, mode),
            #[cfg(feature = "coreml")]
            cached_single_input_shape: CachedInputShape::new("input", &[1, 1, window_samples]),
            #[cfg(feature = "coreml")]
            cached_batch_input_shape: CachedInputShape::new(
                "input",
                &[PRIMARY_BATCH_SIZE, 1, window_samples],
            ),
            input_buffer: ndarray::Array3::zeros((1, 1, window_samples)),
            primary_batch_input_buffer: ndarray::Array3::zeros((
                PRIMARY_BATCH_SIZE,
                1,
                window_samples,
            )),
            window_samples,
            step_samples,
            sample_rate,
        })
    }

    fn build_session(model_path: &str, mode: ExecutionMode) -> Result<Session, ort::Error> {
        let builder = Session::builder()?
            .with_independent_thread_pool()?
            .with_intra_threads(Self::available_threads().min(6))?
            .with_inter_threads(1)?
            .with_memory_pattern(true)?;
        let mut builder = with_execution_mode(builder, mode)?;
        builder.commit_from_file(model_path)
    }

    fn available_threads() -> usize {
        std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1)
    }

    pub fn sample_rate(&self) -> usize {
        self.sample_rate
    }

    pub fn window_samples(&self) -> usize {
        self.window_samples
    }

    pub fn step_samples(&self) -> usize {
        self.step_samples
    }

    pub fn step_seconds(&self) -> f64 {
        self.step_samples as f64 / self.sample_rate as f64
    }

    pub fn mode(&self) -> ExecutionMode {
        self.mode
    }

    #[expect(dead_code)]
    pub(crate) fn model_path(&self) -> &str {
        &self.model_path
    }

    /// Run segmentation with N parallel workers, each loading their own CoreML model.
    /// Workers use CPUOnly compute units to avoid GPU contention with embedding.
    /// Results are sent through `tx` in chunk_idx order
    #[cfg(feature = "coreml")]
    pub fn run_streaming_parallel(
        &mut self,
        audio: &[f32],
        tx: Sender<Array2<f32>>,
        num_workers: usize,
    ) -> Result<usize, SegmentationError> {
        use objc2_core_ml::MLComputeUnits;

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

        // load N separate CPUOnly models for parallel execution
        let coreml_path = Self::resolve_coreml_path(&self.model_path, self.mode);
        let coreml_path = match coreml_path {
            Some(p) if p.exists() => p,
            _ => return self.run_streaming(audio, tx),
        };

        let worker_models: Vec<SharedCoreMlModel> = (0..num_workers)
            .filter_map(|_| {
                SharedCoreMlModel::load(
                    &coreml_path,
                    MLComputeUnits::CPUOnly,
                    "output",
                    GpuPrecision::Low,
                )
                .ok()
            })
            .collect();

        if worker_models.len() < 2 {
            return self.run_streaming(audio, tx);
        }

        let seg_start = std::time::Instant::now();
        let win_samples = self.window_samples;

        // divide windows into contiguous chunks for each worker
        let chunk_size = total_windows.div_ceil(worker_models.len());

        // workers process in parallel, collect results locally, then merge in order
        let all_results: SegParallelResult = std::thread::scope(|scope| {
            let mut handles = Vec::new();

            for (worker_idx, model) in worker_models.iter().enumerate() {
                let start = worker_idx * chunk_size;
                let end = (start + chunk_size).min(total_windows);
                if start >= total_windows {
                    break;
                }

                let offsets = &offsets;
                let padded = &padded;

                handles.push(scope.spawn(move || {
                    let cached_shape = CachedInputShape::new("input", &[1, 1, win_samples]);
                    let mut buffer = ndarray::Array3::<f32>::zeros((1, 1, win_samples));
                    let mut results = Vec::with_capacity(end - start);

                    for idx in start..end {
                        let window = if idx < offsets.len() {
                            &audio[offsets[idx]..offsets[idx] + win_samples]
                        } else {
                            padded.as_deref().unwrap()
                        };

                        buffer.fill(0.0);
                        buffer
                            .slice_mut(ndarray::s![0, 0, ..window.len()])
                            .assign(&ndarray::ArrayView1::from(window));
                        let input_data = buffer.as_slice().unwrap();

                        let (data, out_shape) = model
                            .predict_cached(&[(&cached_shape, input_data)])
                            .map_err(|e| ort::Error::new(e.to_string()))?;

                        let frames = out_shape[1];
                        let classes = out_shape[2];
                        results.push((
                            idx,
                            Array2::from_shape_vec((frames, classes), data).unwrap(),
                        ));
                    }

                    Ok::<Vec<(usize, Array2<f32>)>, SegmentationError>(results)
                }));
            }

            let mut all = Vec::new();
            for handle in handles {
                all.push(handle.join().unwrap()?);
            }
            Ok(all)
        });

        // merge sorted results and send in global order
        let mut merged: Vec<(usize, Array2<f32>)> = all_results?.into_iter().flatten().collect();
        merged.sort_by_key(|(idx, _)| *idx);

        for (_, result) in merged {
            tx.send(result)?;
        }

        let total_seg = seg_start.elapsed();
        tracing::info!(
            windows = total_windows,
            workers = worker_models.len(),
            seg_total_ms = total_seg.as_millis(),
            "Parallel segmentation complete"
        );

        Ok(total_windows)
    }

    pub fn reset_session(&mut self) -> Result<(), ort::Error> {
        self.session = Self::build_session(&self.model_path, self.mode)?;
        self.primary_batched_session = batched_model_path(&self.model_path, PRIMARY_BATCH_SIZE)
            .filter(|path| path.exists())
            .map(|path| Self::build_session(path.to_str().unwrap(), self.mode))
            .transpose()?;
        #[cfg(feature = "coreml")]
        {
            self.native_session = Self::load_native_coreml(&self.model_path, self.mode);
            self.native_batched_session =
                Self::load_native_coreml_batched(&self.model_path, self.mode);
        }
        Ok(())
    }

    /// Run segmentation on audio, streaming raw logits through a channel
    ///
    /// Same logic as `run()`, but sends each decoded window through `tx` as it's produced
    /// instead of collecting into a Vec. Returns total window count
    pub fn run_streaming(
        &mut self,
        audio: &[f32],
        tx: Sender<Array2<f32>>,
    ) -> Result<usize, SegmentationError> {
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
        let win_samples = self.window_samples;
        let window_at = |i: usize| -> &[f32] {
            if i < offsets.len() {
                &audio[offsets[i]..offsets[i] + win_samples]
            } else {
                padded.as_deref().unwrap()
            }
        };

        let seg_start = std::time::Instant::now();
        let mut seg_infer_time = std::time::Duration::ZERO;
        let mut seg_batched = 0u32;
        let mut seg_single = 0u32;

        let has_batched = self.primary_batched_session.is_some();
        let zeros = vec![0.0f32; win_samples];

        let mut next_idx = 0;
        while next_idx < total_windows {
            let remaining = total_windows - next_idx;

            if remaining >= PRIMARY_BATCH_SIZE && has_batched {
                let batch: Vec<&[f32]> = (next_idx..next_idx + PRIMARY_BATCH_SIZE)
                    .map(&window_at)
                    .collect();

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

            // pad remaining windows into a full batch to avoid single-window calls
            if remaining > 1 && has_batched {
                let mut batch: Vec<&[f32]> = (next_idx..total_windows).map(&window_at).collect();
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
            let result = self.run_window(window_at(next_idx))?;
            seg_infer_time += t.elapsed();
            seg_single += 1;
            tx.send(result)?;
            next_idx += 1;
        }

        let total_seg = seg_start.elapsed();
        tracing::info!(
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
    /// Returns Vec<Array2<f32>> where each element is [frames, 7] logits
    pub fn run(&mut self, audio: &[f32]) -> Result<Vec<Array2<f32>>, ort::Error> {
        let mut offsets = Vec::new();
        let mut offset = 0;

        while offset + self.window_samples <= audio.len() {
            offsets.push(offset);
            offset += self.step_samples;
        }

        // handle last partial window by zero-padding
        let padded = if offset < audio.len() && audio.len() > self.window_samples {
            let mut p = vec![0.0f32; self.window_samples];
            let remaining = audio.len() - offset;
            p[..remaining].copy_from_slice(&audio[offset..]);
            Some(p)
        } else {
            None
        };

        let total_windows = offsets.len() + padded.is_some() as usize;
        let mut results = Vec::with_capacity(total_windows);
        let mut next_idx = 0;

        while next_idx < total_windows {
            let remaining = total_windows - next_idx;
            if remaining >= PRIMARY_BATCH_SIZE && self.primary_batched_session.is_some() {
                let batch: Vec<&[f32]> = (next_idx..next_idx + PRIMARY_BATCH_SIZE)
                    .map(|i| {
                        if i < offsets.len() {
                            &audio[offsets[i]..offsets[i] + self.window_samples]
                        } else {
                            padded.as_deref().unwrap()
                        }
                    })
                    .collect();
                results.extend(self.run_batch(&batch)?);
                next_idx += PRIMARY_BATCH_SIZE;
                continue;
            }

            let window = if next_idx < offsets.len() {
                &audio[offsets[next_idx]..offsets[next_idx] + self.window_samples]
            } else {
                padded.as_deref().unwrap()
            };
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

        Ok(Array2::from_shape_vec((frames, classes), data.to_vec()).unwrap())
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
            .unwrap()
            .run(ort::inputs![input_tensor])?;
        let (shape, data) = outputs[0].try_extract_tensor::<f32>()?;

        let batch = shape[0] as usize;
        let frames = shape[1] as usize;
        let classes = shape[2] as usize;
        let stride = frames * classes;

        Ok((0..batch)
            .map(|batch_idx| {
                let start = batch_idx * stride;
                let mut arr = Array2::<f32>::zeros((frames, classes));
                arr.as_slice_mut()
                    .unwrap()
                    .copy_from_slice(&data[start..start + stride]);
                arr
            })
            .collect())
    }

    #[cfg(feature = "coreml")]
    fn resolve_coreml_path(model_path: &str, mode: ExecutionMode) -> Option<std::path::PathBuf> {
        match mode {
            ExecutionMode::CoreMlFast => Some(coreml_w8a16_model_path(model_path)),
            ExecutionMode::CoreMl => Some(coreml_model_path(model_path)),
            _ => None,
        }
    }

    #[cfg(feature = "coreml")]
    fn compute_units_for_mode(_mode: ExecutionMode) -> MLComputeUnits {
        // segmentation is LSTM-based, always best on CPU+GPU regardless of mode
        CoreMlModel::default_compute_units()
    }

    #[cfg(feature = "coreml")]
    fn load_native_coreml(model_path: &str, mode: ExecutionMode) -> Option<SharedCoreMlModel> {
        let coreml_path = Self::resolve_coreml_path(model_path, mode)?;
        if !coreml_path.exists() {
            eprintln!(
                "warning: native CoreML model not found at {}, falling back to ORT CPU",
                coreml_path.display()
            );
            return None;
        }
        match SharedCoreMlModel::load(
            &coreml_path,
            Self::compute_units_for_mode(mode),
            "output",
            GpuPrecision::Low,
        ) {
            Ok(model) => Some(model),
            Err(e) => {
                eprintln!("warning: failed to load native CoreML segmentation: {e}");
                None
            }
        }
    }

    #[cfg(feature = "coreml")]
    fn load_native_coreml_batched(
        model_path: &str,
        mode: ExecutionMode,
    ) -> Option<SharedCoreMlModel> {
        if !matches!(mode, ExecutionMode::CoreMl | ExecutionMode::CoreMlFast) {
            return None;
        }
        let batched_onnx = batched_model_path(model_path, PRIMARY_BATCH_SIZE)?;
        let onnx_str = batched_onnx.to_str().unwrap();
        let resolve = if mode == ExecutionMode::CoreMlFast {
            coreml_w8a16_model_path
        } else {
            coreml_model_path
        };
        let coreml_path = resolve(onnx_str);
        if !coreml_path.exists() {
            return None;
        }
        match SharedCoreMlModel::load(
            &coreml_path,
            Self::compute_units_for_mode(mode),
            "output",
            GpuPrecision::Low,
        ) {
            Ok(model) => Some(model),
            Err(e) => {
                eprintln!("warning: failed to load native CoreML batched segmentation: {e}");
                None
            }
        }
    }

    #[cfg(feature = "coreml")]
    fn run_native_single(
        native: &SharedCoreMlModel,
        window: &[f32],
        buffer: &mut ndarray::Array3<f32>,
        cached_shape: &CachedInputShape,
    ) -> Result<Array2<f32>, ort::Error> {
        buffer.fill(0.0);
        buffer
            .slice_mut(ndarray::s![0, 0, ..window.len()])
            .assign(&ndarray::ArrayView1::from(window));
        let input_data = buffer.as_slice().unwrap();

        let (data, out_shape) = native
            .predict_cached(&[(cached_shape, input_data)])
            .map_err(|e| ort::Error::new(e.to_string()))?;

        let frames = out_shape[1];
        let classes = out_shape[2];
        Ok(Array2::from_shape_vec((frames, classes), data).unwrap())
    }

    #[cfg(feature = "coreml")]
    fn run_native_batch(
        native: &SharedCoreMlModel,
        windows: &[&[f32]],
        buffer: &mut ndarray::Array3<f32>,
        cached_shape: &CachedInputShape,
    ) -> Result<Vec<Array2<f32>>, ort::Error> {
        buffer.fill(0.0);
        for (batch_idx, window) in windows.iter().enumerate() {
            buffer
                .slice_mut(ndarray::s![batch_idx, 0, ..window.len()])
                .assign(&ndarray::ArrayView1::from(*window));
        }
        let input_data = buffer.as_slice().unwrap();

        let (data, out_shape) = native
            .predict_cached(&[(cached_shape, input_data)])
            .map_err(|e| ort::Error::new(e.to_string()))?;

        let batch = out_shape[0];
        let frames = out_shape[1];
        let classes = out_shape[2];

        Ok((0..batch)
            .map(|batch_idx| {
                let start = batch_idx * frames * classes;
                let end = start + frames * classes;
                Array2::from_shape_vec((frames, classes), data[start..end].to_vec()).unwrap()
            })
            .collect())
    }
}

fn batched_model_path(model_path: &str, batch_size: usize) -> Option<std::path::PathBuf> {
    let path = std::path::Path::new(model_path);
    let file_name = path.file_name()?.to_str()?;
    let stem = file_name.strip_suffix(".onnx")?;
    Some(path.with_file_name(format!("{stem}-b{batch_size}.onnx")))
}
