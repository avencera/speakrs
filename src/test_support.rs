use std::path::PathBuf;

pub(crate) fn model_fixture_dir() -> PathBuf {
    std::env::var_os("SPEAKRS_MODEL_FIXTURE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/models"))
}
