use std::path::PathBuf;

use crate::clustering::plda::PldaTransform;
use crate::inference::ExecutionMode;
use crate::inference::embedding::EmbeddingModel;
use crate::inference::segmentation::SegmentationModel;
use crate::models::ModelBundle;
use crate::powerset::PowersetMapping;

use super::OwnedDiarizationPipeline;
use super::config::{PipelineConfig, RuntimeConfig, segmentation_step_seconds};
use super::queued::QueuedDiarizationPipeline;
use super::types::PipelineError;

/// Builder for constructing diarization pipelines
///
/// # Examples
///
/// ```no_run
/// use speakrs::{ExecutionMode, PipelineBuilder};
///
/// // minimal
/// let mut pipeline = PipelineBuilder::from_pretrained(ExecutionMode::Cpu)?.build()?;
///
/// // with custom runtime config
/// # use speakrs::RuntimeConfig;
/// let mut pipeline = PipelineBuilder::from_pretrained(ExecutionMode::Cpu)?
///     .runtime(RuntimeConfig { chunk_emb_workers: 4, ..Default::default() })
///     .build()?;
///
/// // from local directory
/// let mut pipeline = PipelineBuilder::from_dir("./models", ExecutionMode::Cpu)
///     .build()?;
/// # Ok::<(), Box<dyn std::error::Error + Send + Sync>>(())
/// ```
pub struct PipelineBuilder {
    bundle: ModelBundle,
    mode: ExecutionMode,
    runtime: Option<RuntimeConfig>,
    pipeline: Option<PipelineConfig>,
}

impl PipelineBuilder {
    /// Start from a local models directory
    pub fn from_dir(models_dir: impl Into<PathBuf>, mode: ExecutionMode) -> Self {
        Self {
            bundle: ModelBundle::from_dir(models_dir),
            mode,
            runtime: None,
            pipeline: None,
        }
    }

    /// Start from a pre-resolved [`ModelBundle`](ModelBundle)
    pub fn from_bundle(bundle: ModelBundle, mode: ExecutionMode) -> Self {
        Self {
            bundle,
            mode,
            runtime: None,
            pipeline: None,
        }
    }

    /// Download models from HuggingFace and start building
    #[cfg(feature = "online")]
    pub fn from_pretrained(mode: ExecutionMode) -> Result<Self, PipelineError> {
        let bundle = ModelBundle::from_pretrained(mode)?;
        Ok(Self::from_bundle(bundle, mode))
    }

    /// Override runtime config (workers, compute units)
    pub fn runtime(mut self, config: RuntimeConfig) -> Self {
        self.runtime = Some(config);
        self
    }

    /// Override pipeline config (thresholds, clustering)
    pub fn pipeline(mut self, config: PipelineConfig) -> Self {
        self.pipeline = Some(config);
        self
    }

    /// Build the owned pipeline
    pub fn build(self) -> Result<OwnedDiarizationPipeline, PipelineError> {
        let pipeline = self
            .pipeline
            .unwrap_or_else(|| PipelineConfig::for_mode(self.mode));
        let runtime = self.runtime.unwrap_or_default();
        let step = segmentation_step_seconds(self.mode);

        let seg_model = SegmentationModel::with_mode(
            self.bundle.segmentation_path().to_str().unwrap(),
            step as f32,
            self.mode,
        )?;
        let emb_model = EmbeddingModel::with_mode_and_config(
            self.bundle.embedding_path().to_str().unwrap(),
            self.mode,
            &runtime,
        )?;
        let plda = PldaTransform::from_dir(self.bundle.plda_dir())?;

        Ok(OwnedDiarizationPipeline {
            seg_model,
            emb_model,
            plda,
            powerset: PowersetMapping::new(3, 2),
            default_config: pipeline,
        })
    }

    /// Build and immediately convert to a background-processing queue
    pub fn build_queued(self) -> Result<QueuedDiarizationPipeline, PipelineError> {
        let pipeline = self.build()?;
        pipeline
            .into_queued()
            .map_err(|e| PipelineError::Other(e.to_string()))
    }
}
