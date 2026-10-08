use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use ndarray::Array2;
use objc2_core_ml::MLComputeUnits;

use crate::inference::coreml::{
    CachedInputShape, CoreMlError, CoreMlModel, GpuPrecision, SharedCoreMlModel,
};
use crate::inference::{ExecutionMode, InferenceError, ModelLoadError};
use crate::pipeline::RuntimeConfig;

use super::buffers::EmbeddingBuffers;
use super::tensor::{array2_from_shape_vec, embedding_batch_from_coreml, fbank_hw_from_shape};
use super::{
    ChunkEmbeddingSession, ChunkResourceBundle, ChunkSessionInfo, ChunkSessionSpec,
    FBANK_BATCH_SIZE, FBANK_FEATURES, FBANK_FRAMES, MASK_FRAMES, MULTI_MASK_BATCH_SIZE,
    NUM_SPEAKERS, PRIMARY_BATCH_SIZE,
};

mod loaders;
mod run;

use loaders::{CoreMlEmbeddingAssets, load_chunk_session};

/// Model loaded from its spec on first use
struct Lazy<Spec, T> {
    spec: Spec,
    value: Option<T>,
}

impl<Spec, T> Lazy<Spec, T> {
    fn new(spec: Spec) -> Self {
        Self { spec, value: None }
    }

    fn spec(&self) -> &Spec {
        &self.spec
    }

    fn loaded(&self) -> Option<&T> {
        self.value.as_ref()
    }

    fn get_or_load<E>(&mut self, load: impl FnOnce(&Spec) -> Result<T, E>) -> Result<&mut T, E> {
        let value = match self.value.take() {
            Some(value) => value,
            None => load(&self.spec)?,
        };
        Ok(self.value.insert(value))
    }
}

type LazyModel<T> = Lazy<PathBuf, T>;

/// Native CoreML embedding models plus private staging for one model handle
///
/// Bundles required by every CoreML embedding path are plain lazy slots. The batch-64 tail and
/// the 30-second filterbank are optional because some inventories and experiment layouts omit
/// them
pub(super) struct CoreMlEmbedding {
    fbank: LazyModel<Arc<SharedCoreMlModel>>,
    fbank_batched: LazyModel<SharedCoreMlModel>,
    fbank_30s: Option<LazyModel<Arc<SharedCoreMlModel>>>,
    tail: LazyModel<CoreMlModel>,
    tail_batched: LazyModel<CoreMlModel>,
    tail_primary_batched: Option<LazyModel<CoreMlModel>>,
    multi_mask: LazyModel<SharedCoreMlModel>,
    chunk_sessions: Vec<Lazy<ChunkSessionSpec, ChunkEmbeddingSession>>,
    embedding_compute_units: MLComputeUnits,
    cached_fbank_single_shape: CachedInputShape,
    cached_fbank_batch_shape: CachedInputShape,
    cached_fbank_30s_shape: CachedInputShape,
    cached_tail_fbank_shape: CachedInputShape,
    cached_tail_weights_shape: CachedInputShape,
    cached_multi_mask_fbank_shape: CachedInputShape,
    cached_multi_mask_masks_shape: CachedInputShape,
    buffers: EmbeddingBuffers,
}

impl CoreMlEmbedding {
    pub(super) fn load(
        model_path: &Path,
        mode: ExecutionMode,
        config: &RuntimeConfig,
    ) -> Result<Self, ModelLoadError> {
        let assets = CoreMlEmbeddingAssets::resolve(model_path, mode, config)?;
        tracing::trace!(
            chunk_sessions = assets.chunk_sessions.len(),
            has_tail_b64 = assets.tail_primary_batched.is_some(),
            has_fbank_30s = assets.fbank_30s.is_some(),
            "Embedding model init",
        );

        Ok(Self {
            fbank: Lazy::new(assets.fbank),
            fbank_batched: Lazy::new(assets.fbank_batched),
            fbank_30s: assets.fbank_30s.map(Lazy::new),
            tail: Lazy::new(assets.tail),
            tail_batched: Lazy::new(assets.tail_batched),
            tail_primary_batched: assets.tail_primary_batched.map(Lazy::new),
            multi_mask: Lazy::new(assets.multi_mask),
            chunk_sessions: assets.chunk_sessions.into_iter().map(Lazy::new).collect(),
            embedding_compute_units: config
                .coreml_embedding_compute_units()
                .to_ml_compute_units(),
            cached_fbank_single_shape: CachedInputShape::new("waveform", &[1, 1, 160_000]),
            cached_fbank_batch_shape: CachedInputShape::new(
                "waveform",
                &[FBANK_BATCH_SIZE, 1, 160_000],
            ),
            cached_fbank_30s_shape: CachedInputShape::new("waveform", &[1, 1, 480_000]),
            cached_tail_fbank_shape: CachedInputShape::new(
                "fbank",
                &[PRIMARY_BATCH_SIZE, FBANK_FRAMES, FBANK_FEATURES],
            ),
            cached_tail_weights_shape: CachedInputShape::new(
                "weights",
                &[PRIMARY_BATCH_SIZE, MASK_FRAMES],
            ),
            cached_multi_mask_fbank_shape: CachedInputShape::new(
                "fbank",
                &[MULTI_MASK_BATCH_SIZE, FBANK_FRAMES, FBANK_FEATURES],
            ),
            cached_multi_mask_masks_shape: CachedInputShape::new(
                "masks",
                &[MULTI_MASK_BATCH_SIZE * NUM_SPEAKERS, MASK_FRAMES],
            ),
            buffers: EmbeddingBuffers::fresh(),
        })
    }

    /// No fused CoreML model exists, so masked batches run one window at a time
    pub(super) fn primary_batch_size(&self) -> usize {
        1
    }

    /// The filterbank and single tail bundles are required at load
    pub(super) fn prefers_chunk_embedding_path(&self) -> bool {
        true
    }

    /// The batch-32 multi-mask bundle is required at load
    pub(super) fn prefers_multi_mask_path(&self) -> bool {
        true
    }

    pub(super) fn multi_mask_batch_size(&self) -> usize {
        MULTI_MASK_BATCH_SIZE
    }

    /// The per-chunk speaker-batch tail bundle is required at load
    pub(super) fn has_batched_tail(&self) -> bool {
        true
    }

    pub(super) fn split_primary_batch_size(&self) -> usize {
        if self.tail_primary_batched.is_some() {
            PRIMARY_BATCH_SIZE
        } else {
            0
        }
    }

    pub(super) fn chunk_window_capacity(&self) -> Option<usize> {
        self.chunk_sessions
            .last()
            .map(|session| session.spec().num_windows)
    }

    pub(super) fn prepare_chunk_resources(
        &mut self,
    ) -> Result<Option<ChunkResourceBundle>, InferenceError> {
        let Some(capacity) = self.chunk_window_capacity() else {
            return Ok(None);
        };
        self.ensure_chunk_session_loaded(capacity)?;

        let sessions: Vec<ChunkSessionInfo> = self
            .chunk_sessions
            .iter()
            .filter_map(|session| {
                session.loaded().map(|loaded| ChunkSessionInfo {
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

        let fbank_30s = match self.fbank_30s.as_mut() {
            Some(slot) => Some(Arc::clone(load_fbank_30s(slot)?)),
            None => None,
        };
        let fbank_10s = Arc::clone(load_fbank(&mut self.fbank)?);

        Ok(Some(ChunkResourceBundle {
            sessions,
            fbank_30s,
            fbank_10s: Some(fbank_10s),
        }))
    }

    fn ensure_chunk_session_loaded(&mut self, num_windows: usize) -> Result<bool, InferenceError> {
        let Some(slot) = self
            .chunk_sessions
            .iter_mut()
            .find(|session| session.spec().num_windows >= num_windows)
        else {
            return Ok(false);
        };
        if slot.loaded().is_some() {
            return Ok(true);
        }

        let units = self.embedding_compute_units;
        let start = Instant::now();
        let session = slot.get_or_load(|spec| load_chunk_session(spec, units))?;
        tracing::trace!(
            num_windows = session.num_windows,
            ms = start.elapsed().as_millis(),
            "Lazy loaded chunk embedding",
        );
        Ok(true)
    }

    /// Compute fbank for up to 30s of audio in one call
    pub(super) fn compute_chunk_fbank_30s(
        &mut self,
        audio: &[f32],
    ) -> Result<Option<Array2<f32>>, InferenceError> {
        if audio.len() > 480_000 {
            return Ok(None);
        }
        let Some(slot) = self.fbank_30s.as_mut() else {
            return Ok(None);
        };
        let native = load_fbank_30s(slot)?;

        let mut buffer = vec![0.0f32; 480_000];
        buffer[..audio.len()].copy_from_slice(audio);
        let tensor = native.predict_cached(&[(&self.cached_fbank_30s_shape, &buffer)])?;
        let (frames, features) =
            fbank_hw_from_shape(tensor.layout().dims(), "native 30s fbank output")?;
        array2_from_shape_vec(
            frames,
            features,
            tensor.into_data(),
            "native 30s fbank output",
        )
        .map(Some)
    }

    pub(super) fn chunk_session_for_windows(
        &mut self,
        num_windows: usize,
    ) -> Result<Option<&ChunkEmbeddingSession>, InferenceError> {
        if !self.ensure_chunk_session_loaded(num_windows)? {
            return Ok(None);
        }
        Ok(self.chunk_sessions.iter().find_map(|session| {
            session
                .loaded()
                .filter(|loaded| loaded.num_windows >= num_windows)
        }))
    }

    pub(super) fn embed_chunk_session(
        session: &ChunkEmbeddingSession,
        full_fbank: &[f32],
        masks: &[f32],
    ) -> Result<Array2<f32>, InferenceError> {
        let tensor = session.model.predict_cached(&[
            (&session.cached_fbank_shape, full_fbank),
            (&session.cached_masks_shape, masks),
        ])?;
        embedding_batch_from_coreml(
            tensor,
            session.num_masks,
            session.num_masks,
            "chunk embedding session output",
        )
    }
}

fn load_traced<'a, T>(
    slot: &'a mut LazyModel<T>,
    message: &'static str,
    load: impl FnOnce(&Path) -> Result<T, CoreMlError>,
) -> Result<&'a mut T, InferenceError> {
    let start = Instant::now();
    let was_unloaded = slot.loaded().is_none();
    let model = slot.get_or_load(|path| load(path))?;
    if was_unloaded {
        tracing::trace!(ms = start.elapsed().as_millis(), message);
    }
    Ok(model)
}

fn load_shared(
    path: &Path,
    compute_units: MLComputeUnits,
) -> Result<SharedCoreMlModel, CoreMlError> {
    SharedCoreMlModel::load(path, compute_units, "output", GpuPrecision::Low)
}

fn load_exclusive(path: &Path, compute_units: MLComputeUnits) -> Result<CoreMlModel, CoreMlError> {
    CoreMlModel::load(path, compute_units, "output", GpuPrecision::Low)
}

fn load_fbank(
    slot: &mut LazyModel<Arc<SharedCoreMlModel>>,
) -> Result<&Arc<SharedCoreMlModel>, InferenceError> {
    load_traced(slot, "Lazy loaded native fbank 10s", |path| {
        load_shared(path, CoreMlModel::default_compute_units()).map(Arc::new)
    })
    .map(|model| &*model)
}

fn load_fbank_batched(
    slot: &mut LazyModel<SharedCoreMlModel>,
) -> Result<&SharedCoreMlModel, InferenceError> {
    load_traced(slot, "Lazy loaded native fbank b64", |path| {
        load_shared(path, CoreMlModel::default_compute_units())
    })
    .map(|model| &*model)
}

fn load_fbank_30s(
    slot: &mut LazyModel<Arc<SharedCoreMlModel>>,
) -> Result<&Arc<SharedCoreMlModel>, InferenceError> {
    load_traced(slot, "Lazy loaded native fbank 30s", |path| {
        load_shared(path, MLComputeUnits::CPUAndNeuralEngine).map(Arc::new)
    })
    .map(|model| &*model)
}

fn load_multi_mask(
    slot: &mut LazyModel<SharedCoreMlModel>,
    compute_units: MLComputeUnits,
) -> Result<&SharedCoreMlModel, InferenceError> {
    load_traced(slot, "Lazy loaded native multi mask", |path| {
        load_shared(path, compute_units)
    })
    .map(|model| &*model)
}

fn load_tail<'a>(
    slot: &'a mut LazyModel<CoreMlModel>,
    compute_units: MLComputeUnits,
    message: &'static str,
) -> Result<&'a mut CoreMlModel, InferenceError> {
    load_traced(slot, message, |path| load_exclusive(path, compute_units))
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::CoreMlEmbedding;
    use crate::inference::coreml::CoreMlError;
    use crate::inference::embedding::EmbeddingMeta;
    use crate::inference::{ExecutionMode, InferenceError};
    use crate::pipeline::RuntimeConfig;

    #[test]
    fn invalid_bundle_fails_on_first_use_with_typed_coreml_error() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("speakrs-emb-lazy-invalid-{unique}"));
        let bundles = [
            "wespeaker-fbank.mlmodelc",
            "wespeaker-fbank-b32.mlmodelc",
            "wespeaker-fbank-30s.mlmodelc",
            "wespeaker-voxceleb-resnet34-tail.mlmodelc",
            "wespeaker-voxceleb-resnet34-tail-b3.mlmodelc",
            "wespeaker-multimask-tail-b32.mlmodelc",
            "wespeaker-chunk-emb-p1s-w21.mlmodelc",
            "wespeaker-chunk-emb-p1s-w36.mlmodelc",
            "wespeaker-chunk-emb-p1s-w51.mlmodelc",
            "wespeaker-chunk-emb-p1s-w81.mlmodelc",
            "wespeaker-chunk-emb-p1s-w111.mlmodelc",
        ];
        for bundle in bundles {
            let bundle = dir.join(bundle);
            fs::create_dir_all(bundle.join("weights")).unwrap();
            fs::write(bundle.join("model.mil"), b"invalid").unwrap();
            fs::write(bundle.join("coremldata.bin"), b"invalid").unwrap();
            fs::write(bundle.join("weights/weight.bin"), b"invalid").unwrap();
        }
        let model_path: PathBuf = dir.join("wespeaker-voxceleb-resnet34.onnx");

        let mut backend = CoreMlEmbedding::load(
            &model_path,
            ExecutionMode::CoreMl,
            &RuntimeConfig::default(),
        )
        .unwrap();
        let meta = EmbeddingMeta {
            sample_rate: 16_000,
            window_samples: 160_000,
            mask_frames: 589,
            min_num_samples: 400,
        };
        let error = backend
            .compute_chunk_fbank(&meta, &[0.0; 16_000])
            .unwrap_err();

        assert!(
            matches!(error, InferenceError::CoreMl(CoreMlError::LoadFailed(_))),
            "unexpected error: {error}"
        );
        let _ = fs::remove_dir_all(dir);
    }
}
