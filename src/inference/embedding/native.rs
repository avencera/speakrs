#![cfg(feature = "coreml")]

use std::sync::Arc;

use ndarray::Array2;

use crate::inference::coreml::{CoreMlModel, GpuPrecision, SharedCoreMlModel};

use super::plan::LazySession;
use super::{
    ChunkEmbeddingSession, ChunkResourceBundle, ChunkSessionInfo, EmbeddingModel,
    array2_from_shape_vec, embedding_batch, fbank_hw_from_shape,
};

mod loaders;

impl EmbeddingModel {
    pub(super) fn ensure_native_fbank_loaded(
        &mut self,
    ) -> Result<Option<&Arc<SharedCoreMlModel>>, ort::Error> {
        load_lazy_shared(
            &mut self.coreml.native_fbank_session,
            CoreMlModel::default_compute_units(),
            "Lazy loaded native fbank 10s",
        )
    }

    pub(super) fn ensure_native_fbank_batched_loaded(
        &mut self,
    ) -> Result<Option<&SharedCoreMlModel>, ort::Error> {
        load_lazy(
            &mut self.coreml.native_fbank_batched_session,
            CoreMlModel::default_compute_units(),
            "Lazy loaded native fbank b64",
        )
    }

    pub(super) fn ensure_native_fbank_30s_loaded(
        &mut self,
    ) -> Result<Option<&Arc<SharedCoreMlModel>>, ort::Error> {
        load_lazy_shared(
            &mut self.coreml.native_fbank_30s_session,
            objc2_core_ml::MLComputeUnits::CPUAndNeuralEngine,
            "Lazy loaded native fbank 30s",
        )
    }

    pub(crate) fn prepare_chunk_resources(
        &mut self,
    ) -> Result<Option<ChunkResourceBundle>, ort::Error> {
        let Some(capacity) = self.chunk_window_capacity() else {
            return Ok(None);
        };
        self.ensure_chunk_session_loaded(capacity)?;

        let sessions: Vec<ChunkSessionInfo> = self
            .coreml
            .native_chunk_sessions
            .iter()
            .filter_map(|session| {
                session.as_ref().map(|loaded| ChunkSessionInfo {
                    model: Arc::clone(&loaded.model),
                    cached_fbank_shape: Arc::clone(&loaded.cached_fbank_shape),
                    cached_masks_shape: Arc::clone(&loaded.cached_masks_shape),
                    num_windows: loaded.num_windows,
                    fbank_frames: loaded.fbank_frames,
                    num_masks: loaded.num_masks,
                })
            })
            .collect();
        if sessions.is_empty() {
            return Ok(None);
        }

        self.ensure_native_fbank_30s_loaded()?;
        let fbank_30s = self
            .coreml
            .native_fbank_30s_session
            .as_ref()
            .map(Arc::clone);

        self.ensure_native_fbank_loaded()?;
        let fbank_10s = self.coreml.native_fbank_session.as_ref().map(Arc::clone);

        Ok(Some(ChunkResourceBundle {
            sessions,
            fbank_30s,
            fbank_10s,
        }))
    }

    pub(super) fn ensure_native_multi_mask_loaded(
        &mut self,
    ) -> Result<Option<&SharedCoreMlModel>, ort::Error> {
        let units = self.coreml.native_embedding_compute_units;
        load_lazy(
            &mut self.coreml.native_multi_mask_session,
            units,
            "Lazy loaded native multi mask",
        )
    }

    pub(super) fn ensure_native_tail_loaded(
        &mut self,
    ) -> Result<Option<&mut CoreMlModel>, ort::Error> {
        let units = self.coreml.native_embedding_compute_units;
        load_lazy_exclusive(
            &mut self.coreml.native_tail_session,
            units,
            "Lazy loaded native tail",
        )
    }

    pub(super) fn ensure_native_tail_batched_loaded(
        &mut self,
    ) -> Result<Option<&mut CoreMlModel>, ort::Error> {
        let units = self.coreml.native_embedding_compute_units;
        load_lazy_exclusive(
            &mut self.coreml.native_tail_batched_session,
            units,
            "Lazy loaded native tail b32",
        )
    }

    pub(super) fn ensure_native_tail_primary_batched_loaded(
        &mut self,
    ) -> Result<Option<&mut CoreMlModel>, ort::Error> {
        let units = self.coreml.native_embedding_compute_units;
        load_lazy_exclusive(
            &mut self.coreml.native_tail_primary_batched_session,
            units,
            "Lazy loaded native tail b64",
        )
    }

    pub(crate) fn chunk_window_capacity(&self) -> Option<usize> {
        self.coreml
            .native_chunk_sessions
            .last()
            .and_then(|session| match session {
                LazySession::Unloaded(spec) => Some(spec.num_windows),
                LazySession::Loaded(loaded) => Some(loaded.num_windows),
                LazySession::Unavailable => None,
            })
    }

    fn ensure_chunk_session_loaded(&mut self, num_windows: usize) -> Result<bool, ort::Error> {
        let idx = self
            .coreml
            .native_chunk_sessions
            .iter()
            .position(|session| match session {
                LazySession::Unloaded(spec) => spec.num_windows >= num_windows,
                LazySession::Loaded(loaded) => loaded.num_windows >= num_windows,
                LazySession::Unavailable => false,
            });
        let Some(idx) = idx else {
            return Ok(false);
        };
        if matches!(
            self.coreml.native_chunk_sessions[idx],
            LazySession::Loaded(_)
        ) {
            return Ok(true);
        }
        let units = self.coreml.native_embedding_compute_units;
        let start = std::time::Instant::now();
        let loaded = self.coreml.native_chunk_sessions[idx].load_mut(|spec| {
            Self::load_chunk_session(spec, units)
                .map_err(|error| ort::Error::new(error.to_string()))
        })?;
        if let Some(session) = loaded {
            tracing::trace!(
                num_windows = session.num_windows,
                ms = start.elapsed().as_millis(),
                "Lazy loaded chunk embedding",
            );
        }
        Ok(true)
    }

    /// Compute fbank for up to 30s of audio in one call
    pub fn compute_chunk_fbank_30s(
        &mut self,
        audio: &[f32],
    ) -> Result<Option<Array2<f32>>, ort::Error> {
        if audio.len() > 480_000 {
            return Ok(None);
        }
        self.ensure_native_fbank_30s_loaded()?;
        let Some(native) = self.coreml.native_fbank_30s_session.as_ref() else {
            return Ok(None);
        };
        let mut buffer = vec![0.0f32; 480_000];
        buffer[..audio.len()].copy_from_slice(audio);
        let result = native
            .predict_cached(&[(&self.coreml.cached_fbank_30s_shape, &buffer)])
            .map_err(|e| ort::Error::new(e.to_string()));
        result
            .and_then(|tensor| {
                let (frames, features) =
                    fbank_hw_from_shape(tensor.layout().dims(), "native 30s fbank output")?;
                array2_from_shape_vec(
                    frames,
                    features,
                    tensor.into_data(),
                    "native 30s fbank output",
                )
            })
            .map(Some)
    }

    pub(crate) fn chunk_session_for_windows(
        &mut self,
        num_windows: usize,
    ) -> Result<Option<&ChunkEmbeddingSession>, ort::Error> {
        if !self.ensure_chunk_session_loaded(num_windows)? {
            return Ok(None);
        }
        Ok(self
            .coreml
            .native_chunk_sessions
            .iter()
            .find_map(|session| {
                session
                    .as_ref()
                    .filter(|loaded| loaded.num_windows >= num_windows)
            }))
    }

    pub(crate) fn embed_chunk_session(
        session: &ChunkEmbeddingSession,
        full_fbank: &[f32],
        masks: &[f32],
    ) -> Result<Array2<f32>, ort::Error> {
        let tensor = session
            .model
            .predict_cached(&[
                (&session.cached_fbank_shape, full_fbank),
                (&session.cached_masks_shape, masks),
            ])
            .map_err(|e| ort::Error::new(e.to_string()))?;
        embedding_batch(
            &tensor.into_data(),
            session.num_masks,
            "chunk embedding session output",
        )
    }
}

fn load_lazy<'a>(
    slot: &'a mut LazySession<std::path::PathBuf, SharedCoreMlModel>,
    compute_units: objc2_core_ml::MLComputeUnits,
    message: &'static str,
) -> Result<Option<&'a SharedCoreMlModel>, ort::Error> {
    let start = std::time::Instant::now();
    let was_unloaded = matches!(slot, LazySession::Unloaded(_));
    slot.load_mut(|path| {
        SharedCoreMlModel::load(path, compute_units, "output", GpuPrecision::Low)
            .map_err(|error| ort::Error::new(error.to_string()))
    })?;
    if was_unloaded && slot.as_ref().is_some() {
        tracing::trace!(ms = start.elapsed().as_millis(), message);
    }
    Ok(slot.as_ref())
}

fn load_lazy_shared<'a>(
    slot: &'a mut LazySession<std::path::PathBuf, Arc<SharedCoreMlModel>>,
    compute_units: objc2_core_ml::MLComputeUnits,
    message: &'static str,
) -> Result<Option<&'a Arc<SharedCoreMlModel>>, ort::Error> {
    let start = std::time::Instant::now();
    let was_unloaded = matches!(slot, LazySession::Unloaded(_));
    slot.load_mut(|path| {
        SharedCoreMlModel::load(path, compute_units, "output", GpuPrecision::Low)
            .map(Arc::new)
            .map_err(|error| ort::Error::new(error.to_string()))
    })?;
    if was_unloaded && slot.as_ref().is_some() {
        tracing::trace!(ms = start.elapsed().as_millis(), message);
    }
    Ok(slot.as_ref())
}

fn load_lazy_exclusive<'a>(
    slot: &'a mut LazySession<std::path::PathBuf, CoreMlModel>,
    compute_units: objc2_core_ml::MLComputeUnits,
    message: &'static str,
) -> Result<Option<&'a mut CoreMlModel>, ort::Error> {
    let start = std::time::Instant::now();
    let was_unloaded = matches!(slot, LazySession::Unloaded(_));
    let loaded = slot.load_mut(|path| {
        CoreMlModel::load(path, compute_units, "output", GpuPrecision::Low)
            .map_err(|error| ort::Error::new(error.to_string()))
    })?;
    if was_unloaded && loaded.is_some() {
        tracing::trace!(ms = start.elapsed().as_millis(), message);
    }
    Ok(loaded)
}
