use std::path::Path;

use crate::inference::{ExecutionMode, InferenceBackend, ModelLoadError};
use crate::pipeline::RuntimeConfig;

#[cfg(feature = "coreml")]
use super::CoreMlEmbedding;
#[cfg(feature = "cpu")]
use super::CpuEmbedding;
#[cfg(feature = "_cuda")]
use super::CudaEmbedding;
#[cfg(feature = "migraphx")]
use super::OrtEmbedding;
use super::{EmbeddingBackend, EmbeddingMeta, EmbeddingModel, MASK_FRAMES, read_min_num_samples};
#[cfg(feature = "cpu")]
use crate::inference::cpu::assets::EmbeddingAssets;

impl EmbeddingModel {
    /// Load the WeSpeaker embedding model with the requested execution mode and runtime config
    ///
    /// Native CPU inference has no runtime tunables and ignores the config
    pub fn with_mode_and_config(
        model_path: impl AsRef<Path>,
        mode: ExecutionMode,
        #[cfg_attr(
            not(any(
                feature = "migraphx",
                feature = "coreml",
                feature = "_cuda",
                feature = "_metrics"
            )),
            allow(unused_variables)
        )]
        config: &RuntimeConfig,
    ) -> Result<Self, ModelLoadError> {
        let backend = mode.backend()?;

        #[cfg(feature = "_metrics")]
        if let Some(experiment) = config.experiment {
            experiment
                .validate(mode)
                .map_err(|error| ModelLoadError::InvalidConfiguration {
                    message: error.to_string(),
                })?;
        }

        let model_path = model_path.as_ref();
        #[cfg(any(feature = "migraphx", feature = "coreml", feature = "_cuda"))]
        let default_metadata = model_path.with_extension("min_num_samples.txt");
        let (backend, metadata_path) = match backend {
            #[cfg(feature = "cpu")]
            InferenceBackend::Cpu => {
                let assets = EmbeddingAssets::resolve(model_path)?;
                (
                    EmbeddingBackend::Cpu(Box::new(CpuEmbedding::load(&assets)?)),
                    assets.metadata().to_owned(),
                )
            }
            #[cfg(feature = "migraphx")]
            InferenceBackend::Ort(provider) => (
                EmbeddingBackend::Ort(Box::new(OrtEmbedding::load(model_path, provider, config)?)),
                default_metadata,
            ),
            #[cfg(feature = "coreml")]
            InferenceBackend::CoreMl => (
                EmbeddingBackend::CoreMl(Box::new(CoreMlEmbedding::load(
                    model_path, mode, config,
                )?)),
                default_metadata,
            ),
            #[cfg(feature = "_cuda")]
            InferenceBackend::Cuda => (
                EmbeddingBackend::Cuda(Box::new(CudaEmbedding::load(model_path, mode, config)?)),
                default_metadata,
            ),
        };

        Ok(Self {
            meta: EmbeddingMeta {
                sample_rate: 16_000,
                window_samples: 160_000,
                mask_frames: MASK_FRAMES,
                min_num_samples: read_min_num_samples(&metadata_path)?.get(),
            },
            backend,
        })
    }
}
