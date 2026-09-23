use std::fs;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

use crate::inference::ModelLoadError;

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

/// Positive minimum-sample count from embedding metadata
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MinEmbeddingSamples(NonZeroUsize);

impl MinEmbeddingSamples {
    pub(crate) fn new(value: usize) -> Result<Self, ModelLoadError> {
        NonZeroUsize::new(value)
            .map(Self)
            .ok_or_else(|| ModelLoadError::InvalidConfiguration {
                message: "minimum embedding samples must be greater than zero".to_owned(),
            })
    }

    pub(crate) fn get(self) -> usize {
        self.0.get()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EmbeddingMaskSource {
    Clean,
    Full,
}

#[must_use]
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum EmbeddingMaskEligibility<'a> {
    Eligible {
        mask: &'a [f32],
        source: EmbeddingMaskSource,
    },
    NoActivityAfterResize {
        source: EmbeddingMaskSource,
    },
}

impl<'a> EmbeddingMaskEligibility<'a> {
    pub(crate) const fn source(self) -> EmbeddingMaskSource {
        match self {
            Self::Eligible { source, .. } | Self::NoActivityAfterResize { source } => source,
        }
    }

    pub(crate) const fn mask(self) -> Option<&'a [f32]> {
        match self {
            Self::Eligible { mask, .. } => Some(mask),
            Self::NoActivityAfterResize { .. } => None,
        }
    }
}

pub(crate) fn read_min_num_samples(path: &Path) -> Result<MinEmbeddingSamples, ModelLoadError> {
    let text = fs::read_to_string(path).map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            ModelLoadError::InvalidConfiguration {
                message: format!("missing embedding metadata `{}`", path.display()),
            }
        } else {
            ModelLoadError::InvalidConfiguration {
                message: format!(
                    "unreadable embedding metadata `{}`: {error}",
                    path.display()
                ),
            }
        }
    })?;
    let parsed =
        text.trim()
            .parse::<usize>()
            .map_err(|_| ModelLoadError::InvalidConfiguration {
                message: format!("malformed embedding metadata `{}`", path.display()),
            })?;
    MinEmbeddingSamples::new(parsed).map_err(|_| ModelLoadError::InvalidConfiguration {
        message: format!(
            "embedding metadata `{}` must be greater than zero",
            path.display()
        ),
    })
}

pub(crate) fn select_mask<'a>(
    mask: &'a [f32],
    clean_mask: Option<&'a [f32]>,
    num_samples: usize,
    min_num_samples: usize,
    pooling_frames: usize,
) -> EmbeddingMaskEligibility<'a> {
    let (selected_mask, source) = match clean_mask {
        None => (mask, EmbeddingMaskSource::Full),
        Some(clean_mask) if clean_mask.len() == mask.len() && num_samples > 0 => {
            if super::should_use_clean_mask_slice(
                clean_mask,
                num_samples,
                min_num_samples,
                pooling_frames,
            ) {
                (clean_mask, EmbeddingMaskSource::Clean)
            } else {
                (mask, EmbeddingMaskSource::Full)
            }
        }
        Some(_) => (mask, EmbeddingMaskSource::Full),
    };

    if nearest_resize_has_activity(selected_mask, pooling_frames) {
        EmbeddingMaskEligibility::Eligible {
            mask: selected_mask,
            source,
        }
    } else {
        EmbeddingMaskEligibility::NoActivityAfterResize { source }
    }
}

/// Check activity with the same source indices used by PyTorch nearest interpolation
pub(crate) fn nearest_resize_has_activity(mask: &[f32], target_frames: usize) -> bool {
    nearest_resize_has_activity_by(mask.len(), target_frames, |index| mask[index] > 0.0)
}

pub(crate) fn nearest_resize_has_activity_by(
    mask_len: usize,
    target_frames: usize,
    is_active: impl Fn(usize) -> bool,
) -> bool {
    if mask_len == 0 || target_frames == 0 {
        return false;
    }

    // nearest uses floor(target * source / target_frames), as in the wrapper
    (0..target_frames).any(|target_index| {
        let source_index =
            ((target_index as u128) * (mask_len as u128) / (target_frames as u128)) as usize;
        is_active(source_index)
    })
}

#[cfg(test)]
mod tests {
    use super::{
        EmbeddingMaskEligibility, EmbeddingMaskSource, MinEmbeddingSamples,
        nearest_resize_has_activity, read_min_num_samples, select_mask,
    };
    use crate::inference::ModelLoadError;
    use std::fs;
    use std::path::PathBuf;

    fn scratch_file(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "speakrs-min-samples-{}-{}",
            std::process::id(),
            name
        ));
        let _ = fs::create_dir_all(&dir);
        dir.join("wespeaker-voxceleb-resnet34.min_num_samples.txt")
    }

    #[test]
    fn min_embedding_samples_rejects_zero() {
        let error = MinEmbeddingSamples::new(0).unwrap_err();
        match error {
            ModelLoadError::InvalidConfiguration { message } => {
                assert!(message.contains("greater than zero"));
            }
            other => panic!("unexpected error: {other}"),
        }
        assert_eq!(MinEmbeddingSamples::new(400).unwrap().get(), 400);
    }

    #[test]
    fn read_min_num_samples_accepts_valid_metadata() {
        let path = scratch_file("valid");
        fs::write(&path, "400\n").unwrap();
        assert_eq!(read_min_num_samples(&path).unwrap().get(), 400);
    }

    #[test]
    fn read_min_num_samples_rejects_missing_malformed_and_zero() {
        let missing = scratch_file("missing");
        let _ = fs::remove_file(&missing);
        let missing_error = read_min_num_samples(&missing).unwrap_err();
        assert!(
            missing_error
                .to_string()
                .contains("missing embedding metadata")
        );

        let malformed = scratch_file("malformed");
        fs::write(&malformed, "not-a-number\n").unwrap();
        let malformed_error = read_min_num_samples(&malformed).unwrap_err();
        assert!(
            malformed_error
                .to_string()
                .contains("malformed embedding metadata")
        );

        let zero = scratch_file("zero");
        fs::write(&zero, "0\n").unwrap();
        let zero_error = read_min_num_samples(&zero).unwrap_err();
        assert!(zero_error.to_string().contains("greater than zero"));
    }

    #[test]
    fn nearest_resize_rejects_sparse_399_frame_mask_and_accepts_a_sampled_frame() {
        let mut sparse = vec![0.0; 399];
        for index in [256, 261, 262, 268, 269, 270, 280, 393, 397, 398] {
            sparse[index] = 1.0;
        }
        assert_eq!(sparse.iter().sum::<f32>(), 10.0);
        assert!(!nearest_resize_has_activity(&sparse, 100));

        sparse[395] = 1.0;
        assert!(nearest_resize_has_activity(&sparse, 100));
    }

    #[test]
    fn mask_selection_reports_empty_resize_and_keeps_a_valid_control() {
        let mut full = vec![0.0; 399];
        for index in [256, 261, 262, 268, 269, 270, 280, 393, 397, 398] {
            full[index] = 1.0;
        }
        let mut clean = vec![0.0; 399];
        clean[393] = 1.0;
        clean[398] = 1.0;

        assert_eq!(
            select_mask(&full, Some(&clean), 128_000, 1, 100),
            EmbeddingMaskEligibility::NoActivityAfterResize {
                source: EmbeddingMaskSource::Full,
            }
        );

        full[395] = 1.0;
        assert_eq!(
            select_mask(&full, Some(&clean), 128_000, 1, 100),
            EmbeddingMaskEligibility::Eligible {
                mask: &full,
                source: EmbeddingMaskSource::Full,
            }
        );
    }
}
