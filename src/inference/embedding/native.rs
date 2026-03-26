#![cfg(feature = "coreml")]

use std::sync::Arc;

use ndarray::Array2;
use objc2_core_ml::MLComputeUnits;

use super::*;

impl EmbeddingModel {
    pub(super) fn ensure_native_fbank_loaded(&mut self) -> Option<&Arc<SharedCoreMlModel>> {
        if self.native_fbank_session.is_none() {
            let start = std::time::Instant::now();
            self.native_fbank_session =
                Self::load_native_fbank(&self.model_path, self.mode, 1).map(Arc::new);
            if self.native_fbank_session.is_some() {
                tracing::trace!(
                    ms = start.elapsed().as_millis(),
                    "Lazy loaded native fbank 10s"
                );
            }
        }
        self.native_fbank_session.as_ref()
    }

    pub(super) fn ensure_native_fbank_batched_loaded(&mut self) -> Option<&SharedCoreMlModel> {
        if self.native_fbank_batched_session.is_none() {
            let start = std::time::Instant::now();
            self.native_fbank_batched_session =
                Self::load_native_fbank(&self.model_path, self.mode, PRIMARY_BATCH_SIZE);
            if self.native_fbank_batched_session.is_some() {
                tracing::trace!(
                    ms = start.elapsed().as_millis(),
                    "Lazy loaded native fbank b64"
                );
            }
        }
        self.native_fbank_batched_session.as_ref()
    }

    pub(super) fn ensure_native_fbank_30s_loaded(&mut self) -> Option<&Arc<SharedCoreMlModel>> {
        if self.native_fbank_30s_session.is_none() {
            let start = std::time::Instant::now();
            self.native_fbank_30s_session =
                Self::load_native_fbank_30s(&self.model_path, self.mode).map(Arc::new);
            if self.native_fbank_30s_session.is_some() {
                tracing::trace!(
                    ms = start.elapsed().as_millis(),
                    "Lazy loaded native fbank 30s"
                );
            }
        }
        self.native_fbank_30s_session.as_ref()
    }

    pub(crate) fn prepare_chunk_resources(&mut self) -> Option<ChunkResourceBundle> {
        let capacity = self.chunk_window_capacity()?;
        self.ensure_chunk_session_loaded(capacity);

        if self.native_chunk_sessions.is_empty() {
            return None;
        }

        let sessions = self
            .native_chunk_sessions
            .iter()
            .map(|s| ChunkSessionInfo {
                model: Arc::clone(&s.model),
                cached_fbank_shape: Arc::clone(&s.cached_fbank_shape),
                cached_masks_shape: Arc::clone(&s.cached_masks_shape),
                num_windows: s.num_windows,
                fbank_frames: s.fbank_frames,
                num_masks: s.num_masks,
            })
            .collect();

        let _ = self.ensure_native_fbank_30s_loaded();
        let fbank_30s = self.native_fbank_30s_session.as_ref().map(Arc::clone);

        let _ = self.ensure_native_fbank_loaded();
        let fbank_10s = self.native_fbank_session.as_ref().map(Arc::clone);

        Some(ChunkResourceBundle {
            sessions,
            fbank_30s,
            fbank_10s,
        })
    }

    pub(super) fn ensure_native_multi_mask_loaded(&mut self) -> Option<&SharedCoreMlModel> {
        if self.native_multi_mask_session.is_none() {
            let start = std::time::Instant::now();
            self.native_multi_mask_session =
                Self::load_native_multi_mask(&self.model_path, self.mode);
            if self.native_multi_mask_session.is_some() {
                tracing::trace!(
                    ms = start.elapsed().as_millis(),
                    "Lazy loaded native multi mask"
                );
            }
        }
        self.native_multi_mask_session.as_ref()
    }

    pub(super) fn ensure_native_tail_loaded(&mut self) -> Option<&mut CoreMlModel> {
        if self.native_tail_session.is_none() {
            let start = std::time::Instant::now();
            self.native_tail_session = Self::load_native_tail(&self.model_path, self.mode, 1);
            if self.native_tail_session.is_some() {
                tracing::trace!(ms = start.elapsed().as_millis(), "Lazy loaded native tail");
            }
        }
        self.native_tail_session.as_mut()
    }

    pub(super) fn ensure_native_tail_batched_loaded(&mut self) -> Option<&mut CoreMlModel> {
        if self.native_tail_batched_session.is_none() {
            let start = std::time::Instant::now();
            self.native_tail_batched_session =
                Self::load_native_tail(&self.model_path, self.mode, CHUNK_SPEAKER_BATCH_SIZE);
            if self.native_tail_batched_session.is_some() {
                tracing::trace!(
                    ms = start.elapsed().as_millis(),
                    "Lazy loaded native tail b32"
                );
            }
        }
        self.native_tail_batched_session.as_mut()
    }

    pub(super) fn ensure_native_tail_primary_batched_loaded(&mut self) -> Option<&mut CoreMlModel> {
        if self.native_tail_primary_batched_session.is_none() {
            let start = std::time::Instant::now();
            self.native_tail_primary_batched_session =
                Self::load_native_tail(&self.model_path, self.mode, PRIMARY_BATCH_SIZE);
            if self.native_tail_primary_batched_session.is_some() {
                tracing::trace!(
                    ms = start.elapsed().as_millis(),
                    "Lazy loaded native tail b64"
                );
            }
        }
        self.native_tail_primary_batched_session.as_mut()
    }

    pub(crate) fn chunk_window_capacity(&self) -> Option<usize> {
        self.native_chunk_specs.last().map(|spec| spec.num_windows)
    }

    fn ensure_chunk_session_loaded(&mut self, num_windows: usize) -> bool {
        let Some(spec) = self
            .native_chunk_specs
            .iter()
            .find(|spec| spec.num_windows >= num_windows)
            .cloned()
        else {
            return false;
        };

        if self
            .native_chunk_sessions
            .iter()
            .any(|session| session.num_windows == spec.num_windows)
        {
            return true;
        }

        let start = std::time::Instant::now();
        match Self::load_chunk_session(&spec, self.native_chunk_compute_units) {
            Ok(session) => {
                tracing::trace!(
                    num_windows = spec.num_windows,
                    ms = start.elapsed().as_millis(),
                    "Lazy loaded chunk embedding",
                );
                self.native_chunk_sessions.push(session);
                self.native_chunk_sessions
                    .sort_by_key(|session| session.num_windows);
                true
            }
            Err(err) => {
                tracing::warn!(
                    num_windows = spec.num_windows,
                    "Failed to lazy load chunk embedding: {err}",
                );
                false
            }
        }
    }

    fn load_native_tail(
        model_path: &Path,
        mode: ExecutionMode,
        batch_size: usize,
    ) -> Option<CoreMlModel> {
        let compute_units = match mode {
            ExecutionMode::CoreMl | ExecutionMode::CoreMlFast => {
                CoreMlModel::default_compute_units()
            }
            _ => return None,
        };
        let tail_onnx = split_tail_model_path(model_path, batch_size);
        let coreml_path = fp32_coreml_path(&tail_onnx);
        if !coreml_path.exists() {
            if batch_size == 1 {
                tracing::warn!(
                    path = %coreml_path.display(),
                    "Native CoreML tail model not found, falling back to ORT CPU",
                );
            }
            return None;
        }
        match CoreMlModel::load(&coreml_path, compute_units, "output", GpuPrecision::Low) {
            Ok(model) => Some(model),
            Err(e) => {
                tracing::warn!(batch_size, "Failed to load native CoreML tail: {e}");
                None
            }
        }
    }

    pub(super) fn has_native_tail_model(
        model_path: &Path,
        mode: ExecutionMode,
        batch_size: usize,
    ) -> bool {
        match mode {
            ExecutionMode::CoreMl | ExecutionMode::CoreMlFast => {}
            _ => return false,
        }
        let tail_onnx = split_tail_model_path(model_path, batch_size);
        fp32_coreml_path(&tail_onnx).exists()
    }

    fn load_native_fbank(
        model_path: &Path,
        mode: ExecutionMode,
        batch_size: usize,
    ) -> Option<SharedCoreMlModel> {
        if !matches!(mode, ExecutionMode::CoreMl | ExecutionMode::CoreMlFast) {
            return None;
        }
        let fbank_onnx = if batch_size == 1 {
            split_fbank_model_path(model_path)
        } else {
            split_fbank_batched_model_path(model_path)
        };
        let coreml_path = fp32_coreml_path(&fbank_onnx);
        if !coreml_path.exists() {
            return None;
        }
        match SharedCoreMlModel::load(
            &coreml_path,
            CoreMlModel::default_compute_units(),
            "output",
            GpuPrecision::Low,
        ) {
            Ok(model) => Some(model),
            Err(e) => {
                tracing::warn!(batch_size, "Failed to load native CoreML fbank: {e}");
                None
            }
        }
    }

    pub(super) fn has_native_fbank_model(
        model_path: &Path,
        mode: ExecutionMode,
        batch_size: usize,
    ) -> bool {
        if !matches!(mode, ExecutionMode::CoreMl | ExecutionMode::CoreMlFast) {
            return false;
        }
        let fbank_onnx = if batch_size == 1 {
            split_fbank_model_path(model_path)
        } else {
            split_fbank_batched_model_path(model_path)
        };
        fp32_coreml_path(&fbank_onnx).exists()
    }

    fn load_native_fbank_30s(model_path: &Path, mode: ExecutionMode) -> Option<SharedCoreMlModel> {
        if !matches!(mode, ExecutionMode::CoreMl | ExecutionMode::CoreMlFast) {
            return None;
        }
        let coreml_path = model_path.with_file_name("wespeaker-fbank-30s.mlmodelc");
        if !coreml_path.exists() {
            return None;
        }
        match SharedCoreMlModel::load(
            &coreml_path,
            MLComputeUnits::CPUAndNeuralEngine,
            "output",
            GpuPrecision::Low,
        ) {
            Ok(model) => {
                tracing::info!("Loaded 30s fbank model (CPUAndNeuralEngine)");
                Some(model)
            }
            Err(e) => {
                tracing::warn!("Failed to load 30s fbank model: {e}");
                None
            }
        }
    }

    /// Compute fbank for up to 30s of audio in one call
    pub fn compute_chunk_fbank_30s(
        &mut self,
        audio: &[f32],
    ) -> Option<Result<Array2<f32>, ort::Error>> {
        if audio.len() > 480_000 {
            return None;
        }
        let _ = self.ensure_native_fbank_30s_loaded();
        let native = self.native_fbank_30s_session.as_ref()?;
        let mut buffer = vec![0.0f32; 480_000];
        buffer[..audio.len()].copy_from_slice(audio);
        let result = native
            .predict_cached(&[(&self.cached_fbank_30s_shape, &buffer)])
            .map_err(|e| ort::Error::new(e.to_string()));
        Some(result.map(|(data, out_shape)| {
            let frames = out_shape[1];
            let features = out_shape[2];
            array2_from_shape_vec(frames, features, data, "native 30s fbank output")
                .expect("native 30s fbank output shape is validated by model output")
        }))
    }

    fn load_native_multi_mask(model_path: &Path, mode: ExecutionMode) -> Option<SharedCoreMlModel> {
        if !matches!(mode, ExecutionMode::CoreMl | ExecutionMode::CoreMlFast) {
            return None;
        }
        let onnx_path = model_path.with_file_name("wespeaker-multimask-tail-b32.onnx");
        let coreml_path = fp32_coreml_path(&onnx_path);
        if !coreml_path.exists() {
            return None;
        }
        match SharedCoreMlModel::load(
            &coreml_path,
            CoreMlModel::default_compute_units(),
            "output",
            GpuPrecision::Low,
        ) {
            Ok(model) => Some(model),
            Err(e) => {
                tracing::warn!("Failed to load native CoreML multi-mask: {e}");
                None
            }
        }
    }

    pub(super) fn has_native_multi_mask_model(model_path: &Path, mode: ExecutionMode) -> bool {
        if !matches!(mode, ExecutionMode::CoreMl | ExecutionMode::CoreMlFast) {
            return false;
        }
        let onnx_path = model_path.with_file_name("wespeaker-multimask-tail-b32.onnx");
        fp32_coreml_path(&onnx_path).exists()
    }

    fn chunk_session_config(mode: ExecutionMode) -> &'static [(usize, usize, usize, usize)] {
        match mode {
            ExecutionMode::CoreMlFast => &[
                (25, 11, 3000, 33),
                (25, 16, 4000, 48),
                (25, 21, 5000, 63),
                (25, 26, 6000, 78),
                (25, 36, 8000, 108),
                (25, 46, 10000, 138),
                (25, 56, 12000, 168),
            ],
            _ => &[
                (12, 22, 3016, 66),
                (12, 37, 4456, 111),
                (12, 53, 5992, 159),
                (12, 84, 8968, 252),
                (12, 116, 12040, 348),
            ],
        }
    }

    pub(super) fn chunk_session_specs(
        model_path: &Path,
        mode: ExecutionMode,
    ) -> Vec<ChunkSessionSpec> {
        if !matches!(mode, ExecutionMode::CoreMl | ExecutionMode::CoreMlFast) {
            return Vec::new();
        }

        Self::chunk_session_config(mode)
            .iter()
            .filter_map(|&(step_resnet, num_windows, fbank_frames, num_masks)| {
                let stem = format!("wespeaker-chunk-emb-s{step_resnet}-w{num_windows}");
                let w8a16_path = model_path.with_file_name(format!("{stem}-w8a16.mlmodelc"));
                let fp32_path = model_path.with_file_name(format!("{stem}.mlmodelc"));

                let coreml_path = if fp32_path.exists() {
                    fp32_path
                } else if w8a16_path.exists() {
                    w8a16_path
                } else {
                    return None;
                };

                Some(ChunkSessionSpec {
                    coreml_path,
                    num_windows,
                    fbank_frames,
                    num_masks,
                })
            })
            .collect()
    }

    fn load_chunk_session(
        spec: &ChunkSessionSpec,
        compute_units: MLComputeUnits,
    ) -> Result<ChunkEmbeddingSession, crate::inference::coreml::CoreMlError> {
        let model = SharedCoreMlModel::load(
            &spec.coreml_path,
            compute_units,
            "output",
            GpuPrecision::Low,
        )?;
        Ok(ChunkEmbeddingSession {
            model: Arc::new(model),
            num_windows: spec.num_windows,
            fbank_frames: spec.fbank_frames,
            num_masks: spec.num_masks,
            cached_fbank_shape: Arc::new(CachedInputShape::new(
                "fbank",
                &[1, spec.fbank_frames, FBANK_FEATURES],
            )),
            cached_masks_shape: Arc::new(CachedInputShape::new(
                "masks",
                &[spec.num_masks, MASK_FRAMES],
            )),
        })
    }

    pub(crate) fn chunk_session_for_windows(
        &mut self,
        num_windows: usize,
    ) -> Option<&ChunkEmbeddingSession> {
        if !self.ensure_chunk_session_loaded(num_windows) {
            return None;
        }
        self.native_chunk_sessions
            .iter()
            .find(|s| s.num_windows >= num_windows)
    }

    pub(crate) fn embed_chunk_session(
        session: &ChunkEmbeddingSession,
        full_fbank: &[f32],
        masks: &[f32],
    ) -> Result<Array2<f32>, ort::Error> {
        let (data, _) = session
            .model
            .predict_cached(&[
                (&session.cached_fbank_shape, full_fbank),
                (&session.cached_masks_shape, masks),
            ])
            .map_err(|e| ort::Error::new(e.to_string()))?;
        let num_masks = session.num_masks;
        array2_from_shape_vec(num_masks, 256, data, "chunk embedding session output")
    }
}
