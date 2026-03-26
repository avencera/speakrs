use std::fs;
use std::path::{Path, PathBuf};

use speakrs::PipelineError;
use speakrs::inference::{DynamicRuntimeError, ModelLoadError, OrtRuntimeError};

pub fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join(name)
}

pub fn load_wav_samples(path: &Path) -> (Vec<f32>, u32) {
    let data = fs::read(path).unwrap();
    let sample_rate = u32::from_le_bytes(data[24..28].try_into().unwrap());
    let bits_per_sample = u16::from_le_bytes(data[34..36].try_into().unwrap());
    assert_eq!(bits_per_sample, 16);

    let mut pos = 12;
    while pos + 8 < data.len() {
        let chunk_id = &data[pos..pos + 4];
        let chunk_size = u32::from_le_bytes(data[pos + 4..pos + 8].try_into().unwrap()) as usize;
        if chunk_id == b"data" {
            let samples = data[pos + 8..pos + 8 + chunk_size]
                .chunks_exact(2)
                .map(|bytes| i16::from_le_bytes([bytes[0], bytes[1]]) as f32 / 32768.0)
                .collect();
            return (samples, sample_rate);
        }
        pos += 8 + chunk_size;
    }

    panic!("no data chunk found in WAV");
}

#[allow(dead_code)]
pub fn load_model_or_skip<T>(result: Result<T, ModelLoadError>) -> Option<T> {
    match result {
        Ok(value) => Some(value),
        Err(ModelLoadError::Runtime(OrtRuntimeError::Dynamic(DynamicRuntimeError::Missing {
            ..
        }))) if cfg!(feature = "load-dynamic") => {
            eprintln!("skipping model-loading test because ORT_DYLIB_PATH is not configured");
            None
        }
        Err(error) => panic!("failed to load model: {error}"),
    }
}

pub fn build_pipeline_or_skip<T>(result: Result<T, PipelineError>) -> Option<T> {
    match result {
        Ok(value) => Some(value),
        Err(PipelineError::ModelLoad(ModelLoadError::Runtime(OrtRuntimeError::Dynamic(
            DynamicRuntimeError::Missing { .. },
        )))) if cfg!(feature = "load-dynamic") => {
            eprintln!("skipping pipeline test because ORT_DYLIB_PATH is not configured");
            None
        }
        Err(error) => panic!("failed to build pipeline: {error}"),
    }
}
