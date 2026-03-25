use std::path::PathBuf;

use hf_hub::api::sync::{Api, ApiBuilder, ApiRepo};

use crate::inference::ExecutionMode;

const HF_REPO: &str = "avencera/speakrs-models";

/// Manages downloading and caching speakrs ONNX models from HuggingFace
pub struct ModelManager {
    repo: ApiRepo,
}

impl ModelManager {
    /// Create a manager using the default HuggingFace cache directory
    pub fn new() -> Result<Self, hf_hub::api::sync::ApiError> {
        let api = Api::new()?;
        let repo = api.model(HF_REPO.to_string());
        Ok(Self { repo })
    }

    /// Create a manager with a custom cache directory
    pub fn with_cache_dir(cache_dir: PathBuf) -> Result<Self, hf_hub::api::sync::ApiError> {
        let api = ApiBuilder::from_cache(hf_hub::Cache::new(cache_dir)).build()?;
        let repo = api.model(HF_REPO.to_string());
        Ok(Self { repo })
    }

    /// Download a single file, returns path to cached copy
    pub fn get(&self, filename: &str) -> Result<PathBuf, hf_hub::api::sync::ApiError> {
        self.repo.get(filename)
    }

    /// Ensure all files for a mode are downloaded, return base models dir
    pub fn ensure(&self, mode: ExecutionMode) -> Result<PathBuf, hf_hub::api::sync::ApiError> {
        let files = required_files(mode);
        for f in &files {
            self.repo.get(f)?;
        }
        // all files land in the same snapshot dir
        let first = self.repo.get(files[0])?;
        Ok(first.parent().unwrap().to_path_buf())
    }
}

const PLDA_FILES: &[&str] = &[
    "plda_lda.npy",
    "plda_tr.npy",
    "plda_mu.npy",
    "plda_psi.npy",
    "plda_mean1.npy",
    "plda_mean2.npy",
    "wespeaker-voxceleb-resnet34.min_num_samples.txt",
];

const ONNX_FILES: &[&str] = &[
    "segmentation-3.0.onnx",
    "wespeaker-voxceleb-resnet34.onnx",
    "wespeaker-voxceleb-resnet34.onnx.data",
];

fn mlmodelc_files(name: &str) -> Vec<&'static str> {
    // each .mlmodelc dir has these 4 files
    // we leak the strings since they're constructed at runtime
    let paths = [
        format!("{name}/model.mil"),
        format!("{name}/coremldata.bin"),
        format!("{name}/weights/weight.bin"),
        format!("{name}/analytics/coremldata.bin"),
    ];
    paths
        .into_iter()
        .map(|s| &*Box::leak(s.into_boxed_str()))
        .collect()
}

fn required_files(mode: ExecutionMode) -> Vec<&'static str> {
    let mut files: Vec<&str> = PLDA_FILES.to_vec();

    match mode {
        ExecutionMode::Cpu => {
            files.extend_from_slice(ONNX_FILES);
        }
        ExecutionMode::Cuda | ExecutionMode::CudaFast => {
            files.extend_from_slice(ONNX_FILES);
            // split models for multi-mask embedding (CPU fbank + GPU multi-mask)
            files.push("wespeaker-fbank.onnx");
            files.push("wespeaker-fbank-b32.onnx");
            files.push("wespeaker-multimask-tail.onnx");
            files.push("wespeaker-multimask-tail-b32.onnx");
            // batched seg/emb models
            files.push("segmentation-3.0-b32.onnx");
            files.push("wespeaker-voxceleb-resnet34-b64.onnx");
        }
        ExecutionMode::CoreMl | ExecutionMode::CoreMlFast => {
            // CoreML modes still need the ONNX segmentation model for the constructor
            files.push("segmentation-3.0.onnx");
            files.push("wespeaker-voxceleb-resnet34.onnx");
            files.push("wespeaker-voxceleb-resnet34.onnx.data");
            // b32 batched ONNX for segmentation
            files.push("segmentation-3.0-b32.onnx");
            // split ONNX models for embedding
            files.push("wespeaker-fbank.onnx");
            files.push("wespeaker-fbank-b32.onnx");
            files.push("wespeaker-voxceleb-resnet34-tail.onnx");
            files.push("wespeaker-voxceleb-resnet34-tail-b3.onnx");
            files.push("wespeaker-voxceleb-resnet34-tail-b32.onnx");
            // FP32 CoreML models
            files.extend(mlmodelc_files("segmentation-3.0-b32.mlmodelc"));
            files.extend(mlmodelc_files("segmentation-3.0.mlmodelc"));
            files.extend(mlmodelc_files("wespeaker-fbank-b32.mlmodelc"));
            files.extend(mlmodelc_files("wespeaker-fbank.mlmodelc"));
            files.extend(mlmodelc_files(
                "wespeaker-voxceleb-resnet34-tail-b3.mlmodelc",
            ));
            files.extend(mlmodelc_files(
                "wespeaker-voxceleb-resnet34-tail-b32.mlmodelc",
            ));
            files.extend(mlmodelc_files("wespeaker-voxceleb-resnet34-tail.mlmodelc"));
        }
    }

    files
}
