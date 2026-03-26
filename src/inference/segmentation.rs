use std::path::{Path, PathBuf};

use crossbeam_channel::Sender;
use ndarray::Array2;
use ort::session::Session;
use ort::value::TensorRef;
use tracing::debug;

#[cfg(feature = "coreml")]
use crate::inference::coreml::{
    CachedInputShape, CoreMlModel, GpuPrecision, SharedCoreMlModel, coreml_model_path,
    coreml_w8a16_model_path,
};
use crate::inference::{ExecutionMode, ModelLoadError, ensure_ort_ready, with_execution_mode};
#[cfg(feature = "coreml")]
use objc2_core_ml::MLComputeUnits;
#[cfg(feature = "coreml")]
use tracing::{info, trace};

/// Errors that can occur during segmentation inference
#[derive(Debug, thiserror::Error)]
pub enum SegmentationError {
    /// ONNX Runtime error
    #[error(transparent)]
    Ort(#[from] ort::Error),
    /// Streaming channel was closed before all windows were sent
    #[error("receiver disconnected")]
    Disconnected(#[from] crossbeam_channel::SendError<Array2<f32>>),
}

// seg models exported with EnumeratedShapes for batch 1-32 and b64
const PRIMARY_BATCH_SIZE: usize = 32;
#[cfg(feature = "coreml")]
const LARGE_BATCH_SIZE: usize = 64;

/// Sliding-window segmentation model (pyannote segmentation-3.0)
pub struct SegmentationModel {
    model_path: PathBuf,
    mode: ExecutionMode,
    session: Session,
    primary_batched_session: Option<Session>,
    #[cfg(feature = "coreml")]
    native_session: Option<SharedCoreMlModel>,
    #[cfg(feature = "coreml")]
    native_batched_session: Option<SharedCoreMlModel>,
    #[cfg(feature = "coreml")]
    native_large_batched_session: Option<SharedCoreMlModel>,
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

// SAFETY: SegmentationModel is only used from one thread at a time via &mut self
// SAFETY: the non-Send fields contain Objective-C objects that are only moved, not shared
// SAFETY: SharedCoreMlModel is already Send + Sync
#[cfg(feature = "coreml")]
unsafe impl Send for SegmentationModel {}

impl SegmentationModel {
    /// Load a segmentation-3.0 ONNX model
    pub fn new(model_path: impl AsRef<Path>, step_duration: f32) -> Result<Self, ModelLoadError> {
        Self::with_mode(model_path, step_duration, ExecutionMode::Cpu)
    }

    /// Load a segmentation-3.0 ONNX model with the requested execution mode
    pub fn with_mode(
        model_path: impl AsRef<Path>,
        step_duration: f32,
        mode: ExecutionMode,
    ) -> Result<Self, ModelLoadError> {
        mode.validate()?;
        ensure_ort_ready()?;

        let model_path = model_path.as_ref();
        let sample_rate = 16000;
        let window_duration = 10.0;
        let window_samples = (window_duration * sample_rate as f32) as usize;
        let step_samples = (step_duration * sample_rate as f32) as usize;

        macro_rules! timed {
            ($expr:expr) => {{
                let start = std::time::Instant::now();
                let value = $expr;
                (value, start.elapsed())
            }};
        }

        let (session, session_elapsed) = timed!(Self::build_session(model_path, mode)?);
        let (primary_batched_session, primary_batched_elapsed) = timed!(
            batched_model_path(model_path, PRIMARY_BATCH_SIZE)
                .filter(|path| path.exists())
                .map(|path| Self::build_session(&path, mode))
                .transpose()?
        );
        #[cfg(feature = "coreml")]
        let (native_session, native_session_elapsed) =
            timed!(Self::load_native_coreml(model_path, mode));
        #[cfg(feature = "coreml")]
        let (native_batched_session, native_batched_elapsed) =
            timed!(Self::load_native_coreml_batched(model_path, mode));
        #[cfg(feature = "coreml")]
        let (native_large_batched_session, native_large_batched_elapsed) =
            timed!(Self::load_native_coreml_large_batched(model_path, mode));

        #[cfg(feature = "coreml")]
        {
            let total_ms = (session_elapsed
                + primary_batched_elapsed
                + native_session_elapsed
                + native_batched_elapsed
                + native_large_batched_elapsed)
                .as_millis();
            tracing::trace!(
                ort_single_ms = session_elapsed.as_millis(),
                ort_batched_ms = primary_batched_elapsed.as_millis(),
                native_single_ms = native_session_elapsed.as_millis(),
                native_b32_ms = native_batched_elapsed.as_millis(),
                native_b64_ms = native_large_batched_elapsed.as_millis(),
                total_ms,
                "Segmentation model init",
            );
        }
        #[cfg(not(feature = "coreml"))]
        {
            let total_ms = (session_elapsed + primary_batched_elapsed).as_millis();
            tracing::trace!(
                ort_single_ms = session_elapsed.as_millis(),
                ort_batched_ms = primary_batched_elapsed.as_millis(),
                total_ms,
                "Segmentation model init",
            );
        }

        Ok(Self {
            model_path: model_path.to_path_buf(),
            mode,
            session,
            primary_batched_session,
            #[cfg(feature = "coreml")]
            native_session,
            #[cfg(feature = "coreml")]
            native_batched_session,
            #[cfg(feature = "coreml")]
            native_large_batched_session,
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

    fn build_session(model_path: &Path, mode: ExecutionMode) -> Result<Session, ort::Error> {
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

    /// Audio sample rate in Hz (16000)
    pub fn sample_rate(&self) -> usize {
        self.sample_rate
    }

    /// Number of audio samples per sliding window
    pub fn window_samples(&self) -> usize {
        self.window_samples
    }

    /// Number of audio samples the window advances each step
    pub fn step_samples(&self) -> usize {
        self.step_samples
    }

    /// Step size in seconds
    pub fn step_seconds(&self) -> f64 {
        self.step_samples as f64 / self.sample_rate as f64
    }

    /// Execution mode this model was loaded with
    pub fn mode(&self) -> ExecutionMode {
        self.mode
    }

    /// Run segmentation with N parallel workers, each with a fresh CoreML model.
    /// Workers use CPUOnly compute units to avoid GPU contention with embedding.
    /// Results are sent through `tx` in chunk_idx order
    #[cfg(feature = "coreml")]
    pub fn run_streaming_parallel(
        &mut self,
        audio: &[f32],
        tx: Sender<Array2<f32>>,
        num_workers: usize,
        warm_start_target_windows: Option<usize>,
    ) -> Result<usize, SegmentationError> {
        use std::sync::{
            Arc,
            atomic::{AtomicU64, AtomicUsize, Ordering},
        };

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
        let use_warm_start_b32 = batch_size == LARGE_BATCH_SIZE
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
                model: &'a SharedCoreMlModel,
            }

            let mut batch_tasks: Vec<BatchTask<'_>> = Vec::new();
            if use_warm_start_b32 {
                let small_model = self.native_batched_session.as_ref().unwrap();
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
                    let end = (start + LARGE_BATCH_SIZE).min(total_windows);
                    batch_tasks.push(BatchTask {
                        batch_idx,
                        start,
                        end,
                        batch_capacity: LARGE_BATCH_SIZE,
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

            // use std::thread::scope so the merger can stream batches in order
            // while rayon computes future batches out of order
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
                                        padded.as_deref().unwrap()
                                    };
                                    let dst = b * win_samples;
                                    batch_buf[dst..dst + window.len()].copy_from_slice(window);
                                }

                                let batch_start = std::time::Instant::now();
                                let (data, out_shape) = task
                                    .model
                                    .predict_cached(&[(&*cached_batch, batch_buf.as_slice())])
                                    .map_err(|e| {
                                        SegmentationError::Ort(ort::Error::new(e.to_string()))
                                    })
                                    .unwrap();
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
                                    results.push(
                                        Array2::from_shape_vec(
                                            (frames, classes),
                                            data[s..s + stride].to_vec(),
                                        )
                                        .unwrap(),
                                    );
                                }
                                let _ = batch_tx.send((task.batch_idx, results));
                            }
                        });
                    }
                });

                drop(batch_tx);
                merge_handle.join().unwrap()?;
                Ok::<(), SegmentationError>(())
            })?;
        } else {
            // keep the worker-sharded path when only the single-window native model is available
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

                rayon::scope(|rscope| {
                    let seg_predict_us = &seg_predict_us;
                    let seg_single_calls = &seg_single_calls;
                    for (worker_idx, worker_tx) in worker_txs.into_iter().enumerate() {
                        let model = shared_model;
                        let start = worker_idx * chunk_size;
                        let end = (start + chunk_size).min(total_windows);
                        let offsets = &offsets;
                        let padded = &padded;

                        rscope.spawn(move |_| {
                            let cached_shape = CachedInputShape::new("input", &[1, 1, win_samples]);
                            let mut buffer = ndarray::Array3::<f32>::zeros((1, 1, win_samples));

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

                                let predict_start = std::time::Instant::now();
                                let (data, out_shape) = model
                                    .predict_cached(&[(&cached_shape, input_data)])
                                    .map_err(|e| {
                                        SegmentationError::Ort(ort::Error::new(e.to_string()))
                                    })
                                    .unwrap();
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
                                let _ = worker_tx
                                    .send(Array2::from_shape_vec((frames, classes), data).unwrap());
                            }
                        });
                    }
                });

                merge_handle.join().unwrap()?;
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

    /// Reload all ORT and native CoreML sessions from disk
    pub fn reset_session(&mut self) -> Result<(), ort::Error> {
        self.session = Self::build_session(&self.model_path, self.mode)?;
        self.primary_batched_session = batched_model_path(&self.model_path, PRIMARY_BATCH_SIZE)
            .filter(|path| path.exists())
            .map(|path| Self::build_session(&path, self.mode))
            .transpose()?;
        #[cfg(feature = "coreml")]
        {
            self.native_session = Self::load_native_coreml(&self.model_path, self.mode);
            self.native_batched_session =
                Self::load_native_coreml_batched(&self.model_path, self.mode);
            self.native_large_batched_session =
                Self::load_native_coreml_large_batched(&self.model_path, self.mode);
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
    /// Returns `Vec<Array2<f32>>` where each element is \[frames, 7\] logits
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
    fn select_parallel_native_model(
        &self,
        total_windows: usize,
    ) -> Option<(&SharedCoreMlModel, usize)> {
        let min_batch_windows = PRIMARY_BATCH_SIZE * 6;
        if total_windows < min_batch_windows {
            return self.native_session.as_ref().map(|model| (model, 1));
        }

        self.native_large_batched_session
            .as_ref()
            .map(|model| (model, LARGE_BATCH_SIZE))
            .or_else(|| {
                self.native_batched_session
                    .as_ref()
                    .map(|model| (model, PRIMARY_BATCH_SIZE))
            })
            .or_else(|| self.native_session.as_ref().map(|model| (model, 1)))
    }

    #[cfg(feature = "coreml")]
    fn resolve_coreml_path(model_path: &Path, mode: ExecutionMode) -> Option<PathBuf> {
        match mode {
            ExecutionMode::CoreMlFast => Some(coreml_w8a16_model_path(model_path)),
            ExecutionMode::CoreMl => Some(coreml_model_path(model_path)),
            _ => None,
        }
    }

    #[cfg(feature = "coreml")]
    fn compute_units_for_mode(_mode: ExecutionMode) -> MLComputeUnits {
        CoreMlModel::default_compute_units()
    }

    #[cfg(feature = "coreml")]
    fn resolve_batched_coreml_path(
        model_path: &Path,
        mode: ExecutionMode,
        batch_size: usize,
    ) -> Option<PathBuf> {
        if !matches!(mode, ExecutionMode::CoreMl | ExecutionMode::CoreMlFast) {
            return None;
        }

        let batched_onnx = batched_model_path(model_path, batch_size)?;
        Self::resolve_coreml_path(&batched_onnx, mode)
    }

    #[cfg(feature = "coreml")]
    fn load_native_coreml_model(
        coreml_path: &std::path::Path,
        mode: ExecutionMode,
        missing_message: &str,
        load_error_message: &str,
    ) -> Option<SharedCoreMlModel> {
        if !coreml_path.exists() {
            if !missing_message.is_empty() {
                tracing::warn!(path = %coreml_path.display(), "{missing_message}");
            }
            return None;
        }

        match SharedCoreMlModel::load(
            coreml_path,
            Self::compute_units_for_mode(mode),
            "output",
            GpuPrecision::Low,
        ) {
            Ok(model) => Some(model),
            Err(err) => {
                tracing::warn!("{load_error_message}: {err}");
                None
            }
        }
    }

    #[cfg(feature = "coreml")]
    fn load_native_coreml(model_path: &Path, mode: ExecutionMode) -> Option<SharedCoreMlModel> {
        let coreml_path = Self::resolve_coreml_path(model_path, mode)?;
        Self::load_native_coreml_model(
            &coreml_path,
            mode,
            "Native CoreML segmentation model not found, falling back to ORT CPU",
            "Failed to load native CoreML segmentation",
        )
    }

    #[cfg(feature = "coreml")]
    fn load_native_coreml_batched(
        model_path: &Path,
        mode: ExecutionMode,
    ) -> Option<SharedCoreMlModel> {
        let coreml_path = Self::resolve_batched_coreml_path(model_path, mode, PRIMARY_BATCH_SIZE)?;
        Self::load_native_coreml_model(
            &coreml_path,
            mode,
            "",
            "Failed to load native CoreML batched segmentation",
        )
    }

    #[cfg(feature = "coreml")]
    fn load_native_coreml_large_batched(
        model_path: &Path,
        mode: ExecutionMode,
    ) -> Option<SharedCoreMlModel> {
        let coreml_path = Self::resolve_batched_coreml_path(model_path, mode, LARGE_BATCH_SIZE)?;
        let model = Self::load_native_coreml_model(
            &coreml_path,
            mode,
            "",
            "Failed to load b64 segmentation",
        )?;
        info!("Loaded b64 segmentation model");
        Some(model)
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

fn batched_model_path(model_path: &Path, batch_size: usize) -> Option<PathBuf> {
    let path = model_path;
    let file_name = path.file_name()?.to_str()?;
    let stem = file_name.strip_suffix(".onnx")?;
    Some(path.with_file_name(format!("{stem}-b{batch_size}.onnx")))
}
