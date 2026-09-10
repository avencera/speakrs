use std::path::Path;

use ndarray::{Array2, Array3};
#[cfg(feature = "coreml")]
use objc2_core_ml::MLComputeUnits;
use ort::session::{HasSelectedOutputs, RunOptions, Session};

#[cfg(feature = "coreml")]
use crate::inference::coreml::CachedInputShape;
use crate::inference::{ExecutionMode, ModelLoadError};

#[cfg(feature = "coreml")]
use super::super::CoreMlEmbeddingState;
use super::super::plan::EmbeddingExecutionPlan;
#[cfg(feature = "coreml")]
use super::super::plan::LazySession;
use super::super::{
    CHUNK_SPEAKER_BATCH_SIZE, EmbeddingBuffers, EmbeddingMeta, EmbeddingModel, FBANK_BATCH_SIZE,
    FBANK_FEATURES, FBANK_FRAMES, MASK_FRAMES, MULTI_MASK_BATCH_SIZE, NUM_SPEAKERS,
    OrtEmbeddingState, PRIMARY_BATCH_SIZE, preallocated_run_options, read_min_num_samples,
};

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
    native_embedding_compute_units: MLComputeUnits,
}

pub(super) struct LoadedSessions {
    plan: EmbeddingExecutionPlan,
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
        #[cfg(feature = "coreml")]
        let native_embedding_compute_units = config
            .coreml_embedding_compute_units()
            .to_ml_compute_units();

        #[cfg(feature = "_metrics")]
        if let Some(experiment) = config.experiment {
            experiment
                .validate(mode)
                .map_err(|error| ModelLoadError::InvalidConfiguration {
                    message: error.to_string(),
                })?;
        }

        let plan = EmbeddingExecutionPlan::from_inventory(model_path, mode, config);
        let load_ort_split = plan.load_ort_split();

        #[cfg(feature = "coreml")]
        if matches!(mode, ExecutionMode::CoreMl | ExecutionMode::CoreMlFast) {
            EmbeddingModel::validate_native_coreml_assets(model_path, mode, config)?;
        }

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
            plan.fused
                .batched
                .as_ref()
                .map(|slot| EmbeddingModel::build_batched_session(slot.path(), mode))
                .transpose()?
        );
        let (split_fbank_session, split_fbank_elapsed) = timed!(load_optional_ort(
            load_ort_split,
            plan.split_fbank.single.as_ref(),
            |path| EmbeddingModel::build_fbank_session(path, ExecutionMode::Cpu),
        )?);
        let (split_fbank_batched_session, split_fbank_batched_elapsed) = timed!(load_optional_ort(
            load_ort_split,
            plan.split_fbank.batched.as_ref(),
            |path| EmbeddingModel::build_fbank_session(path, ExecutionMode::Cpu),
        )?);
        let (split_tail_session, split_tail_elapsed) = timed!(load_optional_ort(
            load_ort_split,
            plan.split_tail.single.as_ref(),
            |path| EmbeddingModel::build_session(path, mode),
        )?);
        let (split_tail_batched_session, split_tail_batched_elapsed) = timed!(load_optional_ort(
            load_ort_split,
            plan.split_tail.batched.as_ref(),
            |path| EmbeddingModel::build_session(path, mode),
        )?);
        let (split_primary_tail_batched_session, split_primary_tail_batched_elapsed) =
            timed!(load_optional_ort(
                load_ort_split,
                plan.split_tail.primary_batched.as_ref(),
                |path| EmbeddingModel::build_session(path, mode),
            )?);
        let (multi_mask_session, multi_mask_elapsed) = timed!(load_optional_ort(
            load_ort_split,
            plan.multi_mask.single.as_ref(),
            |path| EmbeddingModel::build_session(path, mode),
        )?);
        let (multi_mask_batched_session, multi_mask_batched_elapsed) = timed!(load_optional_ort(
            load_ort_split,
            plan.multi_mask.batched.as_ref(),
            |path| EmbeddingModel::build_session(path, mode),
        )?);

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
            split_tail_b3_ms = split_tail_batched_elapsed.as_millis(),
            split_tail_b64_ms = split_primary_tail_batched_elapsed.as_millis(),
            ort_multi_mask_ms = multi_mask_elapsed.as_millis(),
            ort_multi_mask_b64_ms = multi_mask_batched_elapsed.as_millis(),
            total_ms,
            "Embedding model init",
        );

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
            native_embedding_compute_units,
        };

        Ok(Self {
            plan,
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
                min_num_samples: read_min_num_samples(&metadata_path)?.get(),
            },
            plan: self.plan.clone(),
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
                primary_batch_run_options: self
                    .plan
                    .fused
                    .batched
                    .as_ref()
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
                native_tail_session: LazySession::from_slot(
                    self.plan.split_tail.native_single.as_ref(),
                ),
                native_tail_batched_session: LazySession::from_slot(
                    self.plan.split_tail.native_batched.as_ref(),
                ),
                native_tail_primary_batched_session: LazySession::from_slot(
                    self.plan.split_tail.native_primary_batched.as_ref(),
                ),
                native_fbank_session: LazySession::from_slot(
                    self.plan.split_fbank.native_single.as_ref(),
                ),
                native_fbank_batched_session: LazySession::from_slot(
                    self.plan.split_fbank.native_batched.as_ref(),
                ),
                native_fbank_30s_session: LazySession::from_slot(
                    self.plan.split_fbank.native_30s.as_ref(),
                ),
                cached_fbank_30s_shape: CachedInputShape::new("waveform", &[1, 1, 480_000]),
                native_multi_mask_session: LazySession::from_slot(
                    self.plan.multi_mask.native.as_ref(),
                ),
                native_embedding_compute_units: self.coreml.native_embedding_compute_units,
                native_chunk_sessions: self
                    .plan
                    .chunk_ladder
                    .iter()
                    .cloned()
                    .map(LazySession::Unloaded)
                    .collect(),
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

fn load_optional_ort<T, E>(
    enabled: bool,
    slot: Option<&super::super::plan::AssetSlot>,
    load: impl FnOnce(&Path) -> Result<T, E>,
) -> Result<Option<T>, E> {
    if !enabled {
        return Ok(None);
    }
    slot.map(|slot| load(slot.path())).transpose()
}
