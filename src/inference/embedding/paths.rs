use std::fs;
use std::path::{Path, PathBuf};

#[cfg(feature = "coreml")]
use crate::inference::coreml::coreml_model_path;

pub(super) fn batched_model_path(model_path: &Path, batch_size: usize) -> Option<PathBuf> {
    let file_name = model_path.file_name()?.to_str()?;
    let stem = file_name.strip_suffix(".onnx")?;
    Some(model_path.with_file_name(format!("{stem}-b{batch_size}.onnx")))
}

pub(super) fn split_fbank_model_path(model_path: &Path) -> PathBuf {
    model_path.with_file_name("wespeaker-fbank.onnx")
}

pub(super) fn split_fbank_batched_model_path(model_path: &Path) -> PathBuf {
    model_path.with_file_name("wespeaker-fbank-b32.onnx")
}

pub(super) fn split_tail_model_path(model_path: &Path, batch_size: usize) -> PathBuf {
    if batch_size == 1 {
        model_path.with_file_name("wespeaker-voxceleb-resnet34-tail.onnx")
    } else {
        model_path.with_file_name(format!(
            "wespeaker-voxceleb-resnet34-tail-b{batch_size}.onnx"
        ))
    }
}

pub(super) fn multi_mask_model_path(model_path: &Path, batch_size: usize) -> Option<PathBuf> {
    if batch_size == 1 {
        Some(model_path.with_file_name("wespeaker-multimask-tail.onnx"))
    } else {
        Some(model_path.with_file_name(format!("wespeaker-multimask-tail-b{batch_size}.onnx")))
    }
}

#[cfg(feature = "coreml")]
pub(super) fn fp32_coreml_path(model_path: &Path) -> PathBuf {
    coreml_model_path(model_path)
}

pub(super) fn read_min_num_samples(path: &Path) -> Option<usize> {
    fs::read_to_string(path).ok()?.trim().parse().ok()
}

pub(super) fn select_mask<'a>(
    mask: &'a [f32],
    clean_mask: Option<&'a [f32]>,
    num_samples: usize,
    min_num_samples: usize,
) -> &'a [f32] {
    let Some(clean_mask) = clean_mask else {
        return mask;
    };

    if clean_mask.len() != mask.len() || num_samples == 0 {
        return mask;
    }

    let min_mask_frames = (mask.len() * min_num_samples).div_ceil(num_samples) as f32;
    let clean_weight: f32 = clean_mask.iter().copied().sum();
    if clean_weight > min_mask_frames {
        clean_mask
    } else {
        mask
    }
}
