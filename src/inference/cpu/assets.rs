use std::path::{Path, PathBuf};

use super::{CpuError, CpuModelFamily};
use crate::inference::{ExecutionMode, ModelLoadError};

/// Native PyanNet weights resolved from a supported segmentation selector
#[derive(Debug)]
pub(crate) struct SegmentationAssets(PathBuf);

/// Native WeSpeaker weights and canonical sample metadata from a supported selector
#[derive(Debug)]
pub(crate) struct EmbeddingAssets {
    weights: PathBuf,
    metadata: PathBuf,
}

impl SegmentationAssets {
    /// Resolves the supported segmentation selector without reading ONNX
    pub(crate) fn resolve(selector: &Path) -> Result<Self, ModelLoadError> {
        resolve(selector, CpuModelFamily::Segmentation).map(Self)
    }

    /// Returns the native weight file, never ONNX contents
    pub(crate) fn weights(&self) -> &Path {
        &self.0
    }
}

impl EmbeddingAssets {
    /// Resolves the supported embedding selector and canonical metadata
    pub(crate) fn resolve(selector: &Path) -> Result<Self, ModelLoadError> {
        let weights = resolve(selector, CpuModelFamily::Embedding)?;
        // native filenames select the same canonical metadata as the ONNX selector
        let metadata = selector.with_file_name("wespeaker-voxceleb-resnet34.min_num_samples.txt");
        Ok(Self { weights, metadata })
    }

    /// Returns the native weight file, never ONNX contents
    pub(crate) fn weights(&self) -> &Path {
        &self.weights
    }

    /// Returns the canonical minimum sample count file
    pub(crate) fn metadata(&self) -> &Path {
        &self.metadata
    }
}

fn resolve(selector: &Path, family: CpuModelFamily) -> Result<PathBuf, ModelLoadError> {
    let (canonical, native) = match family {
        CpuModelFamily::Segmentation => ("segmentation-3.0.onnx", "segmentation-3.0.safetensors"),
        CpuModelFamily::Embedding => (
            "wespeaker-voxceleb-resnet34.onnx",
            "wespeaker-multimask-tail.safetensors",
        ),
    };
    let name = selector.file_name().and_then(|name| name.to_str());
    if name != Some(canonical) && name != Some(native) {
        return Err(CpuError::UnsupportedModel {
            family,
            path: selector.to_owned(),
        }
        .into());
    }

    let weights = selector.with_file_name(native);
    if !weights.is_file() {
        return Err(ModelLoadError::MissingNativeAsset {
            mode: ExecutionMode::Cpu,
            path: weights,
        });
    }

    Ok(weights)
}

#[cfg(test)]
mod tests;
