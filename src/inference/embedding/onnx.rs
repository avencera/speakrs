use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use ndarray::{Array2, Array3};
use ort::session::{HasSelectedOutputs, RunOptions, Session};

#[cfg(not(feature = "coreml"))]
use crate::inference::InferenceError;
use crate::inference::{ModelLoadError, OrtProvider, SharedSession, ensure_ort_ready};
use crate::pipeline::{OrtThreadCount, RuntimeConfig};

use super::buffers::EmbeddingBuffers;
use super::tensor::preallocated_run_options;
use super::{EMBEDDING_WIDTH, PRIMARY_BATCH_SIZE};

mod fbank_pool;
mod plan;
mod run;

use fbank_pool::SharedFbankPool;
use plan::OrtEmbeddingPlan;

/// ONNX Runtime embedding sessions plus private staging for one model handle
pub(super) struct OrtEmbedding {
    sessions: OrtEmbeddingPlan<SharedSession>,
    fbank_pool: SharedFbankPool,
    // per-handle state carries a preallocated output tensor
    // do not share it across concurrent runs
    primary_batch_run_options: Option<RunOptions<HasSelectedOutputs>>,
    fused_buffers: FusedBuffers,
    buffers: EmbeddingBuffers,
}

/// Input staging for the fused waveform-to-embedding models
struct FusedBuffers {
    waveform_buffer: Array3<f32>,
    weights_buffer: Array2<f32>,
    primary_batch_waveform_buffer: Array3<f32>,
    primary_batch_weights_buffer: Array2<f32>,
}

impl FusedBuffers {
    fn fresh() -> Self {
        Self {
            waveform_buffer: Array3::zeros((1, 1, 160_000)),
            weights_buffer: Array2::zeros((1, 589)),
            primary_batch_waveform_buffer: Array3::zeros((PRIMARY_BATCH_SIZE, 1, 160_000)),
            primary_batch_weights_buffer: Array2::zeros((PRIMARY_BATCH_SIZE, 589)),
        }
    }
}

impl OrtEmbedding {
    pub(super) fn load(
        model_path: &Path,
        provider: OrtProvider,
        config: &RuntimeConfig,
    ) -> Result<Self, ModelLoadError> {
        ensure_ort_ready()?;

        let plan = OrtEmbeddingPlan::scan(model_path);
        let fbank_threads = config.fbank_threads;
        let load = |path: &Path| build_session(path, provider).map(SharedSession::new);
        let load_fbank = |path: &Path| {
            build_fbank_session(path, OrtProvider::Cpu, fbank_threads).map(SharedSession::new)
        };

        let (fused, fused_elapsed) = timed(|| load(&plan.fused))?;
        let (fused_batched, fused_batched_elapsed) =
            timed(|| plan.fused_batched.as_deref().map(load).transpose())?;
        let (fbank, fbank_elapsed) = timed(|| plan.fbank.as_deref().map(load_fbank).transpose())?;
        let (fbank_batched, fbank_batched_elapsed) =
            timed(|| plan.fbank_batched.as_deref().map(load_fbank).transpose())?;
        let (tail, tail_elapsed) = timed(|| plan.tail.as_deref().map(load).transpose())?;
        let (tail_batched, tail_batched_elapsed) =
            timed(|| plan.tail_batched.as_deref().map(load).transpose())?;
        let (tail_primary_batched, tail_primary_batched_elapsed) =
            timed(|| plan.tail_primary_batched.as_deref().map(load).transpose())?;
        let (multi_mask, multi_mask_elapsed) =
            timed(|| plan.multi_mask.as_deref().map(load).transpose())?;
        let (multi_mask_batched, multi_mask_batched_elapsed) =
            timed(|| plan.multi_mask_batched.as_deref().map(load).transpose())?;
        let (fbank_pool, fbank_pool_elapsed) = timed(|| load_fbank_pool(&plan, config))?;

        let sessions = OrtEmbeddingPlan {
            fused,
            fused_batched,
            fbank,
            fbank_batched,
            tail,
            tail_batched,
            tail_primary_batched,
            multi_mask,
            multi_mask_batched,
        };

        let total_ms = (fused_elapsed
            + fused_batched_elapsed
            + fbank_elapsed
            + fbank_batched_elapsed
            + tail_elapsed
            + tail_batched_elapsed
            + tail_primary_batched_elapsed
            + multi_mask_elapsed
            + multi_mask_batched_elapsed
            + fbank_pool_elapsed)
            .as_millis();
        tracing::trace!(
            ort_single_ms = fused_elapsed.as_millis(),
            ort_b64_ms = fused_batched_elapsed.as_millis(),
            split_fbank_ms = fbank_elapsed.as_millis(),
            split_fbank_b64_ms = fbank_batched_elapsed.as_millis(),
            split_tail_ms = tail_elapsed.as_millis(),
            split_tail_b3_ms = tail_batched_elapsed.as_millis(),
            split_tail_b64_ms = tail_primary_batched_elapsed.as_millis(),
            ort_multi_mask_ms = multi_mask_elapsed.as_millis(),
            ort_multi_mask_b64_ms = multi_mask_batched_elapsed.as_millis(),
            split_fbank_pool_ms = fbank_pool_elapsed.as_millis(),
            split_fbank_pool_size = fbank_pool.len(),
            total_ms,
            "Embedding model init",
        );

        let primary_batch_run_options =
            fresh_primary_run_options(sessions.fused_batched.is_some())?;
        Ok(Self {
            sessions,
            fbank_pool,
            primary_batch_run_options,
            fused_buffers: FusedBuffers::fresh(),
            buffers: EmbeddingBuffers::fresh(),
        })
    }

    /// Share sessions with a new handle that owns fresh scratch buffers and output state
    #[cfg(not(feature = "coreml"))]
    pub(super) fn clone_shared(&self) -> Result<Self, InferenceError> {
        Ok(Self {
            sessions: self.sessions.clone(),
            fbank_pool: self.fbank_pool.clone(),
            primary_batch_run_options: fresh_primary_run_options(
                self.sessions.fused_batched.is_some(),
            )?,
            fused_buffers: FusedBuffers::fresh(),
            buffers: EmbeddingBuffers::fresh(),
        })
    }

    pub(super) fn primary_batch_size(&self) -> usize {
        self.sessions.primary_batch_size()
    }

    pub(super) fn prefers_chunk_embedding_path(&self) -> bool {
        self.sessions.prefers_chunk_embedding_path()
    }

    pub(super) fn split_primary_batch_size(&self) -> usize {
        self.sessions.split_primary_batch_size()
    }

    pub(super) fn has_batched_fbank(&self) -> bool {
        self.sessions.has_batched_fbank()
    }

    pub(super) fn prefers_multi_mask_path(&self) -> bool {
        self.sessions.prefers_multi_mask_path()
    }

    pub(super) fn multi_mask_batch_size(&self) -> usize {
        self.sessions.multi_mask_batch_size()
    }

    pub(super) fn has_batched_tail(&self) -> bool {
        self.sessions.has_batched_tail()
    }
}

fn timed<T, E>(load: impl FnOnce() -> Result<T, E>) -> Result<(T, Duration), E> {
    let start = Instant::now();
    let value = load()?;
    Ok((value, start.elapsed()))
}

fn fresh_primary_run_options(
    has_primary_batched: bool,
) -> Result<Option<RunOptions<HasSelectedOutputs>>, ort::Error> {
    has_primary_batched
        .then(|| {
            let mut options = preallocated_run_options(PRIMARY_BATCH_SIZE, EMBEDDING_WIDTH)?;
            let _ = options.disable_device_sync();
            Ok(options)
        })
        .transpose()
}

fn load_fbank_pool(
    plan: &OrtEmbeddingPlan<PathBuf>,
    config: &RuntimeConfig,
) -> Result<SharedFbankPool, ort::Error> {
    let Some(path) = plan.fbank.as_deref() else {
        return Ok(SharedFbankPool::new(Vec::new()));
    };

    let sessions = config.fbank_pool.resolve(config.fbank_threads);
    tracing::debug!(sessions, "CPU filterbank session pool");
    (0..sessions)
        .map(|_| {
            build_fbank_session(path, OrtProvider::Cpu, config.fbank_threads)
                .map(SharedSession::new)
        })
        .collect::<Result<_, _>>()
        .map(SharedFbankPool::new)
}

fn build_session(model_path: &Path, provider: OrtProvider) -> Result<Session, ort::Error> {
    let builder = Session::builder()?
        .with_independent_thread_pool()?
        // Embedding inference dominates CPU-mode wall time. With
        // intra_threads(1) the whole pipeline runs at ~1-2x realtime on
        // Apple Silicon; letting ORT use up to 6 cores brings it to
        // ~8-9x realtime (measured on M-series, 5.7 min meeting audio)
        // with identical outputs.
        .with_intra_threads(
            std::thread::available_parallelism()
                .map(|n| n.get().min(6))
                .unwrap_or(1),
        )?
        .with_inter_threads(1)?
        .with_memory_pattern(true)?;
    let mut builder = provider.apply(builder)?;
    builder.commit_from_file(model_path)
}

fn build_fbank_session(
    model_path: &Path,
    provider: OrtProvider,
    threads: OrtThreadCount,
) -> Result<Session, ort::Error> {
    let builder = Session::builder()?
        .with_independent_thread_pool()?
        .with_intra_threads(threads.get())?
        .with_inter_threads(1)?
        .with_memory_pattern(true)?;
    let mut builder = provider.apply(builder)?;
    builder.commit_from_file(model_path)
}
