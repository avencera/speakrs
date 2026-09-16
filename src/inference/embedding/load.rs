use std::path::Path;

use crate::inference::{ExecutionMode, ModelLoadError, ensure_ort_ready};

use super::EmbeddingModel;

mod sessions;

use sessions::LoadedSessions;

impl EmbeddingModel {
    /// Load the WeSpeaker embedding model with the requested execution mode and runtime config
    pub fn with_mode_and_config(
        model_path: impl AsRef<Path>,
        mode: ExecutionMode,
        config: &crate::pipeline::RuntimeConfig,
    ) -> Result<Self, ModelLoadError> {
        mode.validate()?;
        ensure_ort_ready()?;

        let model_path = model_path.as_ref();
        LoadedSessions::load(model_path, mode, config)?.into_model(model_path, mode)
    }
}
