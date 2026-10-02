use std::path::Path;

use crate::inference::{ExecutionMode, InferenceBackend, ModelLoadError};
use crate::pipeline::RuntimeConfig;

#[cfg(feature = "coreml")]
use super::CoreMlEmbedding;
#[cfg(feature = "_ort")]
use super::OrtEmbedding;
use super::{EmbeddingBackend, EmbeddingMeta, EmbeddingModel, MASK_FRAMES, read_min_num_samples};

impl EmbeddingModel {
    /// Load the WeSpeaker embedding model with the requested execution mode and runtime config
    pub fn with_mode_and_config(
        model_path: impl AsRef<Path>,
        mode: ExecutionMode,
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
        let backend = match backend {
            #[cfg(feature = "_ort")]
            InferenceBackend::Ort(provider) => {
                EmbeddingBackend::Ort(Box::new(OrtEmbedding::load(model_path, provider, config)?))
            }
            #[cfg(feature = "coreml")]
            InferenceBackend::CoreMl => {
                EmbeddingBackend::CoreMl(Box::new(CoreMlEmbedding::load(model_path, mode, config)?))
            }
        };

        let metadata_path = model_path.with_extension("min_num_samples.txt");
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
