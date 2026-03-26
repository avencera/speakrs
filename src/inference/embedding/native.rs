#![cfg(feature = "coreml")]

use std::sync::Arc;

use ndarray::Array2;

use crate::inference::coreml::{CoreMlModel, SharedCoreMlModel};

use super::{
    CHUNK_SPEAKER_BATCH_SIZE, ChunkEmbeddingSession, ChunkResourceBundle, ChunkSessionInfo,
    EmbeddingModel, PRIMARY_BATCH_SIZE, array2_from_shape_vec,
};

mod loaders;

impl EmbeddingModel {
    pub(super) fn ensure_native_fbank_loaded(&mut self) -> Option<&Arc<SharedCoreMlModel>> {
        if self.coreml.native_fbank_session.is_none() {
            let start = std::time::Instant::now();
            self.coreml.native_fbank_session =
                Self::load_native_fbank(&self.meta.model_path, self.meta.mode, 1).map(Arc::new);
            if self.coreml.native_fbank_session.is_some() {
                tracing::trace!(
                    ms = start.elapsed().as_millis(),
                    "Lazy loaded native fbank 10s"
                );
            }
        }
        self.coreml.native_fbank_session.as_ref()
    }

    pub(super) fn ensure_native_fbank_batched_loaded(&mut self) -> Option<&SharedCoreMlModel> {
        if self.coreml.native_fbank_batched_session.is_none() {
            let start = std::time::Instant::now();
            self.coreml.native_fbank_batched_session =
                Self::load_native_fbank(&self.meta.model_path, self.meta.mode, PRIMARY_BATCH_SIZE);
            if self.coreml.native_fbank_batched_session.is_some() {
                tracing::trace!(
                    ms = start.elapsed().as_millis(),
                    "Lazy loaded native fbank b64"
                );
            }
        }
        self.coreml.native_fbank_batched_session.as_ref()
    }

    pub(super) fn ensure_native_fbank_30s_loaded(&mut self) -> Option<&Arc<SharedCoreMlModel>> {
        if self.coreml.native_fbank_30s_session.is_none() {
            let start = std::time::Instant::now();
            self.coreml.native_fbank_30s_session =
                Self::load_native_fbank_30s(&self.meta.model_path, self.meta.mode).map(Arc::new);
            if self.coreml.native_fbank_30s_session.is_some() {
                tracing::trace!(
                    ms = start.elapsed().as_millis(),
                    "Lazy loaded native fbank 30s"
                );
            }
        }
        self.coreml.native_fbank_30s_session.as_ref()
    }

    pub(crate) fn prepare_chunk_resources(&mut self) -> Option<ChunkResourceBundle> {
        let capacity = self.chunk_window_capacity()?;
        self.ensure_chunk_session_loaded(capacity);

        if self.coreml.native_chunk_sessions.is_empty() {
            return None;
        }

        let sessions = self
            .coreml
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
        let fbank_30s = self
            .coreml
            .native_fbank_30s_session
            .as_ref()
            .map(Arc::clone);

        let _ = self.ensure_native_fbank_loaded();
        let fbank_10s = self.coreml.native_fbank_session.as_ref().map(Arc::clone);

        Some(ChunkResourceBundle {
            sessions,
            fbank_30s,
            fbank_10s,
        })
    }

    pub(super) fn ensure_native_multi_mask_loaded(&mut self) -> Option<&SharedCoreMlModel> {
        if self.coreml.native_multi_mask_session.is_none() {
            let start = std::time::Instant::now();
            self.coreml.native_multi_mask_session =
                Self::load_native_multi_mask(&self.meta.model_path, self.meta.mode);
            if self.coreml.native_multi_mask_session.is_some() {
                tracing::trace!(
                    ms = start.elapsed().as_millis(),
                    "Lazy loaded native multi mask"
                );
            }
        }
        self.coreml.native_multi_mask_session.as_ref()
    }

    pub(super) fn ensure_native_tail_loaded(&mut self) -> Option<&mut CoreMlModel> {
        if self.coreml.native_tail_session.is_none() {
            let start = std::time::Instant::now();
            self.coreml.native_tail_session =
                Self::load_native_tail(&self.meta.model_path, self.meta.mode, 1);
            if self.coreml.native_tail_session.is_some() {
                tracing::trace!(ms = start.elapsed().as_millis(), "Lazy loaded native tail");
            }
        }
        self.coreml.native_tail_session.as_mut()
    }

    pub(super) fn ensure_native_tail_batched_loaded(&mut self) -> Option<&mut CoreMlModel> {
        if self.coreml.native_tail_batched_session.is_none() {
            let start = std::time::Instant::now();
            self.coreml.native_tail_batched_session = Self::load_native_tail(
                &self.meta.model_path,
                self.meta.mode,
                CHUNK_SPEAKER_BATCH_SIZE,
            );
            if self.coreml.native_tail_batched_session.is_some() {
                tracing::trace!(
                    ms = start.elapsed().as_millis(),
                    "Lazy loaded native tail b32"
                );
            }
        }
        self.coreml.native_tail_batched_session.as_mut()
    }

    pub(super) fn ensure_native_tail_primary_batched_loaded(&mut self) -> Option<&mut CoreMlModel> {
        if self.coreml.native_tail_primary_batched_session.is_none() {
            let start = std::time::Instant::now();
            self.coreml.native_tail_primary_batched_session =
                Self::load_native_tail(&self.meta.model_path, self.meta.mode, PRIMARY_BATCH_SIZE);
            if self.coreml.native_tail_primary_batched_session.is_some() {
                tracing::trace!(
                    ms = start.elapsed().as_millis(),
                    "Lazy loaded native tail b64"
                );
            }
        }
        self.coreml.native_tail_primary_batched_session.as_mut()
    }

    pub(crate) fn chunk_window_capacity(&self) -> Option<usize> {
        self.coreml
            .native_chunk_specs
            .last()
            .map(|spec| spec.num_windows)
    }

    fn ensure_chunk_session_loaded(&mut self, num_windows: usize) -> bool {
        let Some(spec) = self
            .coreml
            .native_chunk_specs
            .iter()
            .find(|spec| spec.num_windows >= num_windows)
            .cloned()
        else {
            return false;
        };

        if self
            .coreml
            .native_chunk_sessions
            .iter()
            .any(|session| session.num_windows == spec.num_windows)
        {
            return true;
        }

        let start = std::time::Instant::now();
        match Self::load_chunk_session(&spec, self.coreml.native_chunk_compute_units) {
            Ok(session) => {
                tracing::trace!(
                    num_windows = spec.num_windows,
                    ms = start.elapsed().as_millis(),
                    "Lazy loaded chunk embedding",
                );
                self.coreml.native_chunk_sessions.push(session);
                self.coreml
                    .native_chunk_sessions
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

    /// Compute fbank for up to 30s of audio in one call
    pub fn compute_chunk_fbank_30s(
        &mut self,
        audio: &[f32],
    ) -> Option<Result<Array2<f32>, ort::Error>> {
        if audio.len() > 480_000 {
            return None;
        }
        let _ = self.ensure_native_fbank_30s_loaded();
        let native = self.coreml.native_fbank_30s_session.as_ref()?;
        let mut buffer = vec![0.0f32; 480_000];
        buffer[..audio.len()].copy_from_slice(audio);
        let result = native
            .predict_cached(&[(&self.coreml.cached_fbank_30s_shape, &buffer)])
            .map_err(|e| ort::Error::new(e.to_string()));
        Some(result.and_then(|(data, out_shape)| {
            array2_from_shape_vec(out_shape[1], out_shape[2], data, "native 30s fbank output")
        }))
    }

    pub(crate) fn chunk_session_for_windows(
        &mut self,
        num_windows: usize,
    ) -> Option<&ChunkEmbeddingSession> {
        if !self.ensure_chunk_session_loaded(num_windows) {
            return None;
        }
        self.coreml
            .native_chunk_sessions
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
