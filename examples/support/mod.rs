use std::error::Error;
use std::ffi::OsStr;
use std::fs;
use std::path::Path;

use speakrs::inference::embedding::EmbeddingModel;
use speakrs::inference::segmentation::SegmentationModel;
use speakrs::pipeline::DiarizationPipeline;

pub type ExampleResult<T> = Result<T, Box<dyn Error + Send + Sync>>;

#[allow(dead_code)]
pub fn init_tracing() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "speakrs=info".parse().unwrap()),
        )
        .with_writer(std::io::stderr)
        .try_init();
}

#[allow(dead_code)]
pub fn file_id_from_path(path: &Path) -> String {
    path.file_stem()
        .and_then(OsStr::to_str)
        .unwrap_or("file1")
        .to_owned()
}

pub fn load_models(path: &Path) -> ExampleResult<(SegmentationModel, EmbeddingModel)> {
    let segmentation = SegmentationModel::new(
        path.join("segmentation-3.0.onnx").to_str().ok_or_else(|| {
            format!(
                "segmentation model path is not valid UTF-8: {}",
                path.join("segmentation-3.0.onnx").display()
            )
        })?,
        DiarizationPipeline::default_segmentation_step(),
    )?;
    let embedding = EmbeddingModel::new(
        path.join("wespeaker-voxceleb-resnet34.onnx")
            .to_str()
            .ok_or_else(|| {
                format!(
                    "embedding model path is not valid UTF-8: {}",
                    path.join("wespeaker-voxceleb-resnet34.onnx").display()
                )
            })?,
    )?;
    Ok((segmentation, embedding))
}

pub fn load_wav_samples(path: &Path) -> ExampleResult<Vec<f32>> {
    let data = fs::read(path)?;

    let channels = u16::from_le_bytes(data[22..24].try_into()?);
    let sample_rate = u32::from_le_bytes(data[24..28].try_into()?);
    let bits_per_sample = u16::from_le_bytes(data[34..36].try_into()?);

    if channels != 1 {
        return Err(format!("expected mono WAV, got {channels} channels").into());
    }
    if sample_rate != 16_000 {
        return Err(format!("expected 16kHz WAV, got {sample_rate}Hz").into());
    }
    if bits_per_sample != 16 {
        return Err(format!("expected 16-bit PCM WAV, got {bits_per_sample}-bit").into());
    }

    let mut pos = 12usize;
    while pos + 8 < data.len() {
        let chunk_id = &data[pos..pos + 4];
        let chunk_size = u32::from_le_bytes(data[pos + 4..pos + 8].try_into()?) as usize;

        if chunk_id == b"data" {
            let samples = data[pos + 8..pos + 8 + chunk_size]
                .chunks_exact(2)
                .map(|bytes| i16::from_le_bytes([bytes[0], bytes[1]]) as f32 / 32768.0)
                .collect();
            return Ok(samples);
        }

        pos += 8 + chunk_size;
    }

    Err("no data chunk found in WAV".into())
}
