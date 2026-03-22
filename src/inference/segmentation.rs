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

// seg models exported with EnumeratedShapes for batch 1-32 and b64
const PRIMARY_BATCH_SIZE: usize = 32;
const LARGE_BATCH_SIZE: usize = 64;

#[cfg(feature = "coreml")]
#[expect(dead_code)]
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
            native_large_batched_session: Self::load_native_coreml_large_batched(model_path, mode),
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

    pub(crate) fn model_path(&self) -> &str {
        &self.model_path
    }

    #[cfg(feature = "coreml")]
    pub(crate) fn shared_seg_model(&self) -> Option<&SharedCoreMlModel> {
        self.native_session.as_ref()
    }

    /// Load a 30s chunk segmenter model (SpeakerKit-style: internal sliding_windows)
    #[cfg(feature = "coreml")]
    pub fn load_chunk_segmenter(model_path: &str) -> Option<SharedCoreMlModel> {
        let path = std::path::Path::new(model_path).with_file_name("speakerkit-segmenter.mlmodelc");
        if !path.exists() {
            return None;
        }
        SharedCoreMlModel::load(
            &path,
            CoreMlModel::default_compute_units(),
            "speaker_ids",
            GpuPrecision::Low,
        )
        .ok()
    }

    /// Run 30s chunk segmenter: one call → 21 raw logit windows (1s step)
    /// `step_subsample`: take every Nth window to match desired step
    #[cfg(feature = "coreml")]
    pub fn run_chunk_segmenter(
        model: &SharedCoreMlModel,
        audio: &[f32],
        step_subsample: usize,
    ) -> Result<Vec<Array2<f32>>, ort::Error> {
        let chunk_samples = 480_000usize;
        let window_samples = 160_000usize;
        let model_step = 16_000usize; // 1s step internal to model

        let mut buffer = vec![0.0f32; chunk_samples];
        let copy_len = audio.len().min(chunk_samples);
        buffer[..copy_len].copy_from_slice(&audio[..copy_len]);

        let shape = CachedInputShape::new("waveform", &[chunk_samples]);
        let (data, out_shape) = model
            .predict_cached(&[(&shape, &buffer)])
            .map_err(|e| ort::Error::new(e.to_string()))?;

        let num_windows = out_shape[0];
        let frames = out_shape[1];
        let classes = out_shape[2];
        let stride = frames * classes;

        // only keep windows that correspond to actual audio (not zero-padding)
        let valid_windows = if audio.len() >= window_samples {
            ((audio.len() - window_samples) / model_step + 1).min(num_windows)
        } else {
            1.min(num_windows)
        };

        let mut results = Vec::new();
        for win_idx in 0..valid_windows {
            if win_idx % step_subsample != 0 {
                continue;
            }
            let start = win_idx * stride;
            let window_data = &data[start..start + stride];
            results.push(Array2::from_shape_vec((frames, classes), window_data.to_vec()).unwrap());
        }

        Ok(results)
    }

    /// Load the SpeakerKit multi-output segmenter for the experiment pipeline
    #[cfg(feature = "coreml")]
    pub(crate) fn load_speakerkit_segmenter(model_path: &str) -> Option<SharedCoreMlModel> {
        let path = std::path::Path::new(model_path).with_file_name("speakerkit-segmenter.mlmodelc");
        if !path.exists() {
            return None;
        }
        // SpeakerKit uses cpuOnly for segmentation
        SharedCoreMlModel::load_multi_output(
            &path,
            objc2_core_ml::MLComputeUnits::CPUOnly,
            GpuPrecision::Low,
        )
        .ok()
    }

    /// Diagnostic: dump raw model output values for first chunk
    #[cfg(feature = "coreml")]
    pub(crate) fn diagnose_speakerkit_segmenter(model_path: &str, audio: &[f32]) {
        let chunk_samples = 480_000usize;
        if audio.len() < chunk_samples {
            return;
        }
        let model = match Self::load_speakerkit_segmenter(model_path) {
            Some(m) => m,
            None => return,
        };

        // use audio from 30min mark (known to have continuous speech)
        let start = 30 * 60 * 16000usize;
        if start + chunk_samples > audio.len() {
            return;
        }

        let shape = CachedInputShape::new("waveform", &[chunk_samples]);
        let mut buffer = vec![0.0f32; chunk_samples];
        buffer.copy_from_slice(&audio[start..start + chunk_samples]);

        let outputs = model
            .predict_cached_multi(&[(&shape, &buffer)], &["speaker_probs", "speaker_activity"])
            .unwrap();
        let (ids_data, ids_shape) = &outputs[0];
        let (act_data, _) = &outputs[1];

        // raw value analysis for speaker_ids
        let mut unique_vals: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();
        for &v in ids_data.iter() {
            unique_vals.insert(v.to_bits());
        }
        let unique_f32: Vec<f32> = unique_vals
            .iter()
            .map(|&bits| f32::from_bits(bits))
            .collect();

        // per-window analysis: count nonzero AND print actual values for window 5
        let frames = ids_shape[1];
        let spks = ids_shape[2];
        let stride = frames * spks;
        let mut per_win_nonzero: Vec<usize> = Vec::new();
        for w in 0..ids_shape[0] {
            let start = w * stride;
            let nz = ids_data[start..start + stride]
                .iter()
                .filter(|&&v| v != 0.0)
                .count();
            per_win_nonzero.push(nz);
        }

        // window 5 first 30 values (frames 0-9, all 3 speakers)
        let w5 = 5 * stride;
        let w5_vals: Vec<f32> = ids_data[w5..w5 + 30].to_vec();

        // speaker_activity for all windows
        let all_act: Vec<f32> = act_data.to_vec();

        tracing::info!(
            unique_values = ?unique_f32,
            per_win_nonzero = ?per_win_nonzero,
            w5_first30 = ?w5_vals,
            all_activity = ?all_act,
            "SpeakerKit segmenter diagnosis"
        );
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
    ) -> Result<usize, SegmentationError> {
        use std::sync::atomic::{AtomicU64, Ordering};

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

        // choose model and batch size based on window count:
        // - large files (200+ windows): batched model with parallel workers
        // - short files: individual native calls (2.5ms each vs 120ms per batch)
        let min_batch_windows = PRIMARY_BATCH_SIZE * 6; // ~192
        let (shared_model, batch_size) = if total_windows >= min_batch_windows {
            if let Some(ref m) = self.native_large_batched_session {
                (m, LARGE_BATCH_SIZE)
            } else if let Some(ref m) = self.native_batched_session {
                (m, PRIMARY_BATCH_SIZE)
            } else if let Some(ref m) = self.native_session {
                (m, 1)
            } else {
                return self.run_streaming(audio, tx);
            }
        } else if let Some(ref m) = self.native_session {
            // short files: individual native calls, no batching
            (m, 1)
        } else {
            return self.run_streaming(audio, tx);
        };

        let seg_start = std::time::Instant::now();
        let win_samples = self.window_samples;
        let profile_batches = std::env::var_os("SPEAKRS_PROFILE_SEG_BATCHES").is_some();
        let seg_predict_us = AtomicU64::new(0);
        let seg_batched_calls = AtomicU64::new(0);
        let seg_batched_windows = AtomicU64::new(0);
        let seg_single_calls = AtomicU64::new(0);
        let use_warm_start_b32 = batch_size == LARGE_BATCH_SIZE
            && total_windows < 1024
            && total_windows > PRIMARY_BATCH_SIZE
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

                for _ in 0..2 {
                    if start >= total_windows {
                        break;
                    }
                    let end = (start + PRIMARY_BATCH_SIZE).min(total_windows);
                    batch_tasks.push(BatchTask {
                        batch_idx,
                        start,
                        end,
                        batch_capacity: PRIMARY_BATCH_SIZE,
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

                    for task in batch_tasks {
                        let BatchTask {
                            batch_idx,
                            start,
                            end,
                            batch_capacity,
                            model,
                        } = task;
                        let offsets = &offsets;
                        let padded = &padded;
                        let batch_tx = batch_tx.clone();

                        rscope.spawn(move |_| {
                            let actual_batch = end - start;
                            let cached_batch =
                                CachedInputShape::new("input", &[batch_capacity, 1, win_samples]);
                            let mut batch_buf = vec![0.0f32; batch_capacity * win_samples];

                            batch_buf.fill(0.0);
                            for (b, widx) in (start..end).enumerate() {
                                let window = if widx < offsets.len() {
                                    &audio[offsets[widx]..offsets[widx] + win_samples]
                                } else {
                                    padded.as_deref().unwrap()
                                };
                                let dst = b * win_samples;
                                batch_buf[dst..dst + window.len()].copy_from_slice(window);
                            }

                            let batch_start = std::time::Instant::now();
                            let (data, out_shape) = model
                                .predict_cached(&[(&cached_batch, &batch_buf)])
                                .map_err(|e| SegmentationError::Ort(ort::Error::new(e.to_string())))
                                .unwrap();
                            let batch_us = batch_start.elapsed().as_micros() as u64;
                            seg_predict_us.fetch_add(batch_us, Ordering::Relaxed);
                            seg_batched_calls.fetch_add(1, Ordering::Relaxed);
                            seg_batched_windows.fetch_add(actual_batch as u64, Ordering::Relaxed);
                            if profile_batches {
                                tracing::info!(
                                    batch_idx,
                                    batch_capacity,
                                    batch_size = actual_batch,
                                    batch_ms = batch_us / 1000,
                                    "Seg batch profile"
                                );
                            }

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
                            let _ = batch_tx.send((batch_idx, results));
                        });
                    }
                });

                drop(batch_tx);
                merge_handle.join().unwrap()?;
                Ok::<(), SegmentationError>(())
            })?;
        } else {
            // single: keep the old worker-sharded path because this code path is
            // only used when no batched CoreML segmenter is available
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
                                if profile_batches {
                                    tracing::info!(
                                        worker_idx,
                                        batch_size = 1,
                                        batch_ms = predict_us / 1000,
                                        "Seg batch profile"
                                    );
                                }

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
        tracing::info!(
            windows = total_windows,
            workers = num_workers,
            seg_warm_start_b32 = use_warm_start_b32,
            seg_batched_calls = seg_batched_calls.load(Ordering::Relaxed),
            seg_batched_windows = seg_batched_windows.load(Ordering::Relaxed),
            seg_single_calls = seg_single_calls.load(Ordering::Relaxed),
            seg_predict_ms_sum = seg_predict_ms,
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
    fn load_native_coreml_large_batched(
        model_path: &str,
        mode: ExecutionMode,
    ) -> Option<SharedCoreMlModel> {
        if !matches!(mode, ExecutionMode::CoreMl | ExecutionMode::CoreMlFast) {
            return None;
        }
        let batched_onnx = batched_model_path(model_path, LARGE_BATCH_SIZE)?;
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
            Ok(model) => {
                tracing::info!("Loaded b64 segmentation model");
                Some(model)
            }
            Err(e) => {
                eprintln!("warning: failed to load b64 segmentation: {e}");
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

/// Output from SpeakerKit's multi-output segmenter (one 30s chunk)
#[cfg(feature = "coreml")]
pub(crate) struct SpeakerKitSegOutput {
    /// Binary speaker masks per window: 21 × [589, 3]
    pub speaker_ids: Vec<Array2<f32>>,
    /// Per-window speaker activity summary: [21, 3]
    pub speaker_activity: Array2<f32>,
    /// Per-window overlap flags: [21, 589]
    pub overlapped_activity: Array2<f32>,
}

/// Run SpeakerKit's multi-output segmenter on a 30s chunk
#[cfg(feature = "coreml")]
pub(crate) fn run_speakerkit_segmenter(
    model: &SharedCoreMlModel,
    audio: &[f32],
) -> Result<SpeakerKitSegOutput, SegmentationError> {
    use crate::inference::coreml::CachedInputShape;

    let chunk_samples = 480_000usize;
    let window_samples = 160_000usize;
    let model_step = 16_000usize;

    let mut buffer = vec![0.0f32; chunk_samples];
    let copy_len = audio.len().min(chunk_samples);
    buffer[..copy_len].copy_from_slice(&audio[..copy_len]);

    let shape = CachedInputShape::new("waveform", &[chunk_samples]);
    let outputs = model
        .predict_cached_multi(
            &[(&shape, &buffer)],
            &[
                "speaker_probs",
                "speaker_activity",
                "overlapped_speaker_activity",
            ],
        )
        .map_err(|e| SegmentationError::Ort(ort::Error::new(e.to_string())))?;

    // use speaker_probs (continuous) instead of speaker_ids (sparse binary)
    let (ids_data, ids_shape) = &outputs[0]; // [21, 589, 3]
    let (act_data, act_shape) = &outputs[1]; // [21, 3]
    let (overlap_data, overlap_shape) = &outputs[2]; // [21, 589]

    let num_windows = ids_shape[0];
    let frames = ids_shape[1];
    let speakers = ids_shape[2];

    // only keep windows that correspond to actual audio
    let valid_windows = if audio.len() >= window_samples {
        ((audio.len() - window_samples) / model_step + 1).min(num_windows)
    } else {
        1.min(num_windows)
    };

    let ids_stride = frames * speakers;
    let speaker_ids: Vec<Array2<f32>> = (0..valid_windows)
        .map(|w| {
            let start = w * ids_stride;
            Array2::from_shape_vec(
                (frames, speakers),
                ids_data[start..start + ids_stride].to_vec(),
            )
            .unwrap()
        })
        .collect();

    let speaker_activity = Array2::from_shape_vec(
        (valid_windows, act_shape[1]),
        act_data[..valid_windows * act_shape[1]].to_vec(),
    )
    .unwrap();

    let overlapped_activity = Array2::from_shape_vec(
        (valid_windows, overlap_shape[1]),
        overlap_data[..valid_windows * overlap_shape[1]].to_vec(),
    )
    .unwrap();

    Ok(SpeakerKitSegOutput {
        speaker_ids,
        speaker_activity,
        overlapped_activity,
    })
}

fn batched_model_path(model_path: &str, batch_size: usize) -> Option<std::path::PathBuf> {
    let path = std::path::Path::new(model_path);
    let file_name = path.file_name()?.to_str()?;
    let stem = file_name.strip_suffix(".onnx")?;
    Some(path.with_file_name(format!("{stem}-b{batch_size}.onnx")))
}
