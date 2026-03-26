use std::path::Path;

use super::*;

pub(super) struct LoadedOrtSessions {
    session: Session,
    primary_batched_session: Option<Session>,
    split_fbank_session: Option<Session>,
    split_fbank_batched_session: Option<Session>,
    split_tail_session: Option<Session>,
    split_tail_batched_session: Option<Session>,
    split_primary_tail_batched_session: Option<Session>,
    multi_mask_session: Option<Session>,
    multi_mask_batched_session: Option<Session>,
}

#[cfg(feature = "coreml")]
pub(super) struct LoadedCoreMlState {
    native_tail_session: Option<CoreMlModel>,
    native_tail_batched_session: Option<CoreMlModel>,
    native_tail_primary_batched_session: Option<CoreMlModel>,
    native_fbank_session: Option<Arc<SharedCoreMlModel>>,
    native_fbank_batched_session: Option<SharedCoreMlModel>,
    native_fbank_30s_session: Option<Arc<SharedCoreMlModel>>,
    native_multi_mask_session: Option<SharedCoreMlModel>,
    native_chunk_compute_units: MLComputeUnits,
    native_chunk_specs: Vec<ChunkSessionSpec>,
    native_chunk_sessions: Vec<ChunkEmbeddingSession>,
}

pub(super) struct LoadedSessions {
    ort: LoadedOrtSessions,
    #[cfg(feature = "coreml")]
    coreml: LoadedCoreMlState,
}

impl LoadedSessions {
    pub(super) fn load(
        model_path: &Path,
        mode: ExecutionMode,
        config: &crate::pipeline::RuntimeConfig,
    ) -> Result<Self, ModelLoadError> {
        let split_fbank_path = split_fbank_model_path(model_path);
        let split_fbank_batched_path = split_fbank_batched_model_path(model_path);
        let split_tail_path = split_tail_model_path(model_path, 1);
        let split_tail_batched_path = split_tail_model_path(model_path, CHUNK_SPEAKER_BATCH_SIZE);
        let split_primary_tail_batched_path = split_tail_model_path(model_path, PRIMARY_BATCH_SIZE);
        #[cfg(feature = "coreml")]
        let native_chunk_compute_units = config.chunk_emb_compute_units.to_ml_compute_units();
        #[cfg(not(feature = "coreml"))]
        let _ = config;
        let use_split_backend = EmbeddingModel::split_backend_available(model_path);

        macro_rules! timed {
            ($expr:expr) => {{
                let start = std::time::Instant::now();
                let value = $expr;
                (value, start.elapsed())
            }};
        }

        let (session, session_elapsed) = timed!(EmbeddingModel::build_session(
            model_path,
            EmbeddingModel::single_execution_mode(mode)
        )?);
        let (primary_batched_session, primary_batched_elapsed) = timed!(
            batched_model_path(model_path, PRIMARY_BATCH_SIZE)
                .filter(|path| path.exists())
                .map(|path| EmbeddingModel::build_batched_session(&path, mode))
                .transpose()?
        );
        let (split_fbank_session, split_fbank_elapsed) = timed!(
            use_split_backend
                .then(|| EmbeddingModel::build_fbank_session(&split_fbank_path, ExecutionMode::Cpu))
                .transpose()?
        );
        let (split_fbank_batched_session, split_fbank_batched_elapsed) = timed!(
            use_split_backend
                .then_some(split_fbank_batched_path)
                .filter(|path| path.exists())
                .map(|path: std::path::PathBuf| {
                    EmbeddingModel::build_fbank_session(path.as_path(), ExecutionMode::Cpu)
                })
                .transpose()?
        );
        let (split_tail_session, split_tail_elapsed) = timed!(
            use_split_backend
                .then(|| EmbeddingModel::build_session(&split_tail_path, mode))
                .transpose()?
        );
        let (split_tail_batched_session, split_tail_batched_elapsed) = timed!(
            use_split_backend
                .then_some(split_tail_batched_path)
                .filter(|path| path.exists())
                .map(|path: std::path::PathBuf| EmbeddingModel::build_session(path.as_path(), mode))
                .transpose()?
        );
        let (split_primary_tail_batched_session, split_primary_tail_batched_elapsed) = timed!(
            use_split_backend
                .then_some(split_primary_tail_batched_path)
                .filter(|path| path.exists())
                .map(|path: std::path::PathBuf| EmbeddingModel::build_session(path.as_path(), mode))
                .transpose()?
        );
        #[cfg(feature = "coreml")]
        let (native_tail_session, native_tail_elapsed) = (None, std::time::Duration::ZERO);
        #[cfg(feature = "coreml")]
        let (native_tail_batched_session, native_tail_batched_elapsed) =
            timed!(Option::<CoreMlModel>::None);
        #[cfg(feature = "coreml")]
        let (native_tail_primary_batched_session, native_tail_primary_batched_elapsed) =
            (None, std::time::Duration::ZERO);
        #[cfg(feature = "coreml")]
        let (native_fbank_session, native_fbank_elapsed) = (None, std::time::Duration::ZERO);
        #[cfg(feature = "coreml")]
        let (native_fbank_batched_session, native_fbank_batched_elapsed) =
            timed!(Option::<SharedCoreMlModel>::None);
        #[cfg(feature = "coreml")]
        let (native_fbank_30s_session, native_fbank_30s_elapsed) =
            (None, std::time::Duration::ZERO);
        #[cfg(feature = "coreml")]
        let (native_multi_mask_session, native_multi_mask_elapsed) =
            (None, std::time::Duration::ZERO);
        #[cfg(feature = "coreml")]
        let (native_chunk_specs, native_chunk_specs_elapsed) =
            timed!(EmbeddingModel::chunk_session_specs(model_path, mode));
        #[cfg(feature = "coreml")]
        let (native_chunk_sessions, native_chunk_sessions_elapsed) =
            (Vec::new(), std::time::Duration::ZERO);
        let (multi_mask_session, multi_mask_elapsed) = timed!(
            multi_mask_model_path(model_path, 1)
                .filter(|path| path.exists())
                .map(|path| EmbeddingModel::build_session(&path, mode))
                .transpose()?
        );
        let (multi_mask_batched_session, multi_mask_batched_elapsed) = timed!(
            multi_mask_model_path(model_path, PRIMARY_BATCH_SIZE)
                .filter(|path| path.exists())
                .map(|path| EmbeddingModel::build_session(&path, mode))
                .transpose()?
        );

        #[cfg(feature = "coreml")]
        {
            let total_ms = (session_elapsed
                + primary_batched_elapsed
                + split_fbank_elapsed
                + split_fbank_batched_elapsed
                + split_tail_elapsed
                + split_tail_batched_elapsed
                + split_primary_tail_batched_elapsed
                + native_tail_elapsed
                + native_tail_batched_elapsed
                + native_tail_primary_batched_elapsed
                + native_fbank_elapsed
                + native_fbank_batched_elapsed
                + native_fbank_30s_elapsed
                + native_multi_mask_elapsed
                + native_chunk_specs_elapsed
                + native_chunk_sessions_elapsed
                + multi_mask_elapsed
                + multi_mask_batched_elapsed)
                .as_millis();
            tracing::trace!(
                ort_single_ms = session_elapsed.as_millis(),
                ort_b64_ms = primary_batched_elapsed.as_millis(),
                split_fbank_ms = split_fbank_elapsed.as_millis(),
                split_fbank_b64_ms = split_fbank_batched_elapsed.as_millis(),
                split_tail_ms = split_tail_elapsed.as_millis(),
                split_tail_b32_ms = split_tail_batched_elapsed.as_millis(),
                split_tail_b64_ms = split_primary_tail_batched_elapsed.as_millis(),
                native_tail_ms = native_tail_elapsed.as_millis(),
                native_tail_b32_ms = native_tail_batched_elapsed.as_millis(),
                native_tail_b64_ms = native_tail_primary_batched_elapsed.as_millis(),
                native_fbank_ms = native_fbank_elapsed.as_millis(),
                native_fbank_b64_ms = native_fbank_batched_elapsed.as_millis(),
                native_fbank_30s_ms = native_fbank_30s_elapsed.as_millis(),
                native_multi_mask_ms = native_multi_mask_elapsed.as_millis(),
                native_chunk_spec_ms = native_chunk_specs_elapsed.as_millis(),
                native_chunk_ms = native_chunk_sessions_elapsed.as_millis(),
                ort_multi_mask_ms = multi_mask_elapsed.as_millis(),
                ort_multi_mask_b64_ms = multi_mask_batched_elapsed.as_millis(),
                total_ms,
                "Embedding model init",
            );
        }
        #[cfg(not(feature = "coreml"))]
        {
            let total_ms = (session_elapsed
                + primary_batched_elapsed
                + split_fbank_elapsed
                + split_fbank_batched_elapsed
                + split_tail_elapsed
                + split_tail_batched_elapsed
                + split_primary_tail_batched_elapsed
                + multi_mask_elapsed
                + multi_mask_batched_elapsed)
                .as_millis();
            tracing::trace!(
                ort_single_ms = session_elapsed.as_millis(),
                ort_b64_ms = primary_batched_elapsed.as_millis(),
                split_fbank_ms = split_fbank_elapsed.as_millis(),
                split_fbank_b64_ms = split_fbank_batched_elapsed.as_millis(),
                split_tail_ms = split_tail_elapsed.as_millis(),
                split_tail_b32_ms = split_tail_batched_elapsed.as_millis(),
                split_tail_b64_ms = split_primary_tail_batched_elapsed.as_millis(),
                ort_multi_mask_ms = multi_mask_elapsed.as_millis(),
                ort_multi_mask_b64_ms = multi_mask_batched_elapsed.as_millis(),
                total_ms,
                "Embedding model init",
            );
        }

        let ort = LoadedOrtSessions {
            session,
            primary_batched_session,
            split_fbank_session,
            split_fbank_batched_session,
            split_tail_session,
            split_tail_batched_session,
            split_primary_tail_batched_session,
            multi_mask_session,
            multi_mask_batched_session,
        };
        #[cfg(feature = "coreml")]
        let coreml = LoadedCoreMlState {
            native_tail_session,
            native_tail_batched_session,
            native_tail_primary_batched_session,
            native_fbank_session,
            native_fbank_batched_session,
            native_fbank_30s_session,
            native_multi_mask_session,
            native_chunk_compute_units,
            native_chunk_specs,
            native_chunk_sessions,
        };

        Ok(Self {
            ort,
            #[cfg(feature = "coreml")]
            coreml,
        })
    }

    pub(super) fn into_model(
        self,
        model_path: &Path,
        mode: ExecutionMode,
    ) -> Result<EmbeddingModel, ModelLoadError> {
        let metadata_path = model_path.with_extension("min_num_samples.txt");

        Ok(EmbeddingModel {
            meta: EmbeddingMeta {
                model_path: model_path.to_path_buf(),
                mode,
                sample_rate: 16_000,
                window_samples: 160_000,
                mask_frames: 589,
                min_num_samples: read_min_num_samples(&metadata_path).unwrap_or(400),
            },
            ort: OrtEmbeddingState {
                session: self.ort.session,
                primary_batched_session: self.ort.primary_batched_session,
                split_fbank_session: self.ort.split_fbank_session,
                split_fbank_batched_session: self.ort.split_fbank_batched_session,
                split_tail_session: self.ort.split_tail_session,
                split_tail_batched_session: self.ort.split_tail_batched_session,
                split_primary_tail_batched_session: self.ort.split_primary_tail_batched_session,
                multi_mask_session: self.ort.multi_mask_session,
                multi_mask_batched_session: self.ort.multi_mask_batched_session,
                primary_batch_run_options: batched_model_path(model_path, PRIMARY_BATCH_SIZE)
                    .filter(|path| path.exists())
                    .map(|_| {
                        let mut opts = preallocated_run_options(
                            PRIMARY_BATCH_SIZE,
                            256,
                            "primary batched embedding output",
                        )?;
                        let _ = opts.disable_device_sync();
                        Ok::<RunOptions<HasSelectedOutputs>, ort::Error>(opts)
                    })
                    .transpose()?,
            },
            #[cfg(feature = "coreml")]
            coreml: CoreMlEmbeddingState {
                native_tail_session: self.coreml.native_tail_session,
                native_tail_batched_session: self.coreml.native_tail_batched_session,
                native_tail_primary_batched_session: self
                    .coreml
                    .native_tail_primary_batched_session,
                native_fbank_session: self.coreml.native_fbank_session,
                native_fbank_batched_session: self.coreml.native_fbank_batched_session,
                native_fbank_30s_session: self.coreml.native_fbank_30s_session,
                cached_fbank_30s_shape: CachedInputShape::new("waveform", &[1, 1, 480_000]),
                native_multi_mask_session: self.coreml.native_multi_mask_session,
                native_chunk_compute_units: self.coreml.native_chunk_compute_units,
                native_chunk_specs: self.coreml.native_chunk_specs,
                native_chunk_sessions: self.coreml.native_chunk_sessions,
                cached_tail_fbank_shape: CachedInputShape::new(
                    "fbank",
                    &[PRIMARY_BATCH_SIZE, FBANK_FRAMES, FBANK_FEATURES],
                ),
                cached_tail_weights_shape: CachedInputShape::new(
                    "weights",
                    &[PRIMARY_BATCH_SIZE, MASK_FRAMES],
                ),
                cached_fbank_single_shape: CachedInputShape::new("waveform", &[1, 1, 160_000]),
                cached_fbank_batch_shape: CachedInputShape::new(
                    "waveform",
                    &[FBANK_BATCH_SIZE, 1, 160_000],
                ),
                cached_multi_mask_fbank_shape: CachedInputShape::new(
                    "fbank",
                    &[MULTI_MASK_BATCH_SIZE, FBANK_FRAMES, FBANK_FEATURES],
                ),
                cached_multi_mask_masks_shape: CachedInputShape::new(
                    "masks",
                    &[MULTI_MASK_BATCH_SIZE * NUM_SPEAKERS, MASK_FRAMES],
                ),
            },
            buffers: EmbeddingBuffers {
                multi_mask_fbank_buffer: Array3::zeros((
                    MULTI_MASK_BATCH_SIZE,
                    FBANK_FRAMES,
                    FBANK_FEATURES,
                )),
                multi_mask_masks_buffer: Array2::zeros((
                    MULTI_MASK_BATCH_SIZE * NUM_SPEAKERS,
                    MASK_FRAMES,
                )),
                waveform_buffer: Array3::zeros((1, 1, 160_000)),
                weights_buffer: Array2::zeros((1, 589)),
                primary_batch_waveform_buffer: Array3::zeros((PRIMARY_BATCH_SIZE, 1, 160_000)),
                primary_batch_weights_buffer: Array2::zeros((PRIMARY_BATCH_SIZE, 589)),
                split_waveform_buffer: Array3::zeros((1, 1, 160_000)),
                split_fbank_batch_buffer: Array3::zeros((FBANK_BATCH_SIZE, 1, 160_000)),
                split_feature_batch_buffer: Array3::zeros((
                    CHUNK_SPEAKER_BATCH_SIZE,
                    FBANK_FRAMES,
                    FBANK_FEATURES,
                )),
                split_weights_batch_buffer: Array2::zeros((CHUNK_SPEAKER_BATCH_SIZE, 589)),
                split_primary_feature_batch_buffer: Array3::zeros((
                    PRIMARY_BATCH_SIZE,
                    FBANK_FRAMES,
                    FBANK_FEATURES,
                )),
                split_primary_weights_batch_buffer: Array2::zeros((PRIMARY_BATCH_SIZE, 589)),
            },
        })
    }
}
