use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use super::{EmbeddingAssets, SegmentationAssets};
use crate::inference::{CpuError, CpuModelFamily, ExecutionMode, ModelLoadError};

#[test]
fn unsupported_selector_precedes_missing_file_error() {
    assert!(matches!(
        EmbeddingAssets::resolve(Path::new("/does-not-exist/custom.onnx")),
        Err(ModelLoadError::Cpu(CpuError::UnsupportedModel {
            family: CpuModelFamily::Embedding,
            ..
        }))
    ));
    assert!(matches!(
        SegmentationAssets::resolve(Path::new("/does-not-exist/segmentation-3.0.onnx")),
        Err(ModelLoadError::MissingNativeAsset {
            mode: ExecutionMode::Cpu,
            ..
        })
    ));
    // a known native file of the other family is still an unsupported selector
    assert!(matches!(
        SegmentationAssets::resolve(Path::new(
            "/does-not-exist/wespeaker-multimask-tail.safetensors"
        )),
        Err(ModelLoadError::Cpu(CpuError::UnsupportedModel {
            family: CpuModelFamily::Segmentation,
            ..
        }))
    ));
}

#[test]
fn explicit_and_canonical_selectors_resolve_the_same_native_files() {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root =
        std::env::temp_dir().join(format!("speakrs-cpu-assets-{}-{nonce}", std::process::id()));
    std::fs::create_dir(&root).unwrap();
    let _cleanup = TestAssets(root.clone());
    for name in [
        "segmentation-3.0.safetensors",
        "wespeaker-multimask-tail.safetensors",
    ] {
        std::fs::write(root.join(name), []).unwrap();
    }

    let segmentation_weights = root.join("segmentation-3.0.safetensors");
    for selector in ["segmentation-3.0.onnx", "segmentation-3.0.safetensors"] {
        let assets = SegmentationAssets::resolve(&root.join(selector)).unwrap();
        assert_eq!(assets.weights(), segmentation_weights);
    }

    let embedding_weights = root.join("wespeaker-multimask-tail.safetensors");
    let metadata = root.join("wespeaker-voxceleb-resnet34.min_num_samples.txt");
    for selector in [
        "wespeaker-voxceleb-resnet34.onnx",
        "wespeaker-multimask-tail.safetensors",
    ] {
        let assets = EmbeddingAssets::resolve(&root.join(selector)).unwrap();
        assert_eq!(assets.weights(), embedding_weights);
        assert_eq!(assets.metadata(), metadata);
    }
}

struct TestAssets(PathBuf);

impl Drop for TestAssets {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
