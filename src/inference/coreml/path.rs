use std::borrow::Cow;
use std::path::{Path, PathBuf};

fn coreml_stem(path: &Path) -> Cow<'_, str> {
    path.file_stem()
        .filter(|stem| !stem.is_empty())
        .unwrap_or_else(|| path.file_name().unwrap_or(path.as_os_str()))
        .to_string_lossy()
}

pub(crate) fn coreml_model_path(onnx_path: &Path) -> PathBuf {
    let stem = coreml_stem(onnx_path);
    onnx_path.with_file_name(format!("{stem}.mlmodelc"))
}

pub(crate) fn coreml_w8a16_model_path(onnx_path: &Path) -> PathBuf {
    let stem = coreml_stem(onnx_path);
    let w8a16_path = onnx_path.with_file_name(format!("{stem}-w8a16.mlmodelc"));
    if w8a16_path.exists() {
        w8a16_path
    } else {
        coreml_model_path(onnx_path)
    }
}
