use std::path::Path;

use super::*;

mod sessions;

use sessions::LoadedSessions;

impl EmbeddingModel {
    pub(super) fn split_backend_available(model_path: &Path) -> bool {
        let split_fbank_path = split_fbank_model_path(model_path);
        let split_tail_path = split_tail_model_path(model_path, 1);
        let has_multi_mask = multi_mask_model_path(model_path, 1).is_some_and(|path| path.exists());

        split_fbank_path.exists() && (split_tail_path.exists() || has_multi_mask)
    }

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
