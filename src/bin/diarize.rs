use std::fs;
use std::path::Path;

use speakrs::clustering::plda::PldaTransform;
use speakrs::inference::embedding::EmbeddingModel;
use speakrs::inference::segmentation::SegmentationModel;
use speakrs::pipeline::{SEGMENTATION_STEP_SECONDS, diarize};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: diarize <path/to/audio.wav>");
        std::process::exit(1);
    }

    let wav_path = &args[1];
    let models_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/models");

    let mut seg_model = SegmentationModel::new(
        models_dir.join("segmentation-3.0.onnx").to_str().unwrap(),
        SEGMENTATION_STEP_SECONDS as f32,
    )
    .expect("failed to load segmentation model");
    let mut emb_model = EmbeddingModel::new(
        models_dir
            .join("wespeaker-voxceleb-resnet34.onnx")
            .to_str()
            .unwrap(),
    )
    .expect("failed to load embedding model");
    let plda = PldaTransform::from_dir(&models_dir).expect("failed to load PLDA parameters");

    let (samples, sr) = load_wav_samples(wav_path);
    assert_eq!(sr, 16000, "expected 16kHz WAV, got {sr}Hz");

    let result = diarize(&mut seg_model, &mut emb_model, &plda, &samples, "file1")
        .expect("diarization failed");
    print!("{}", result.rttm);
}

fn load_wav_samples(path: &str) -> (Vec<f32>, u32) {
    let data = fs::read(path).expect("failed to read WAV file");

    let sample_rate = u32::from_le_bytes(data[24..28].try_into().unwrap());
    let bits_per_sample = u16::from_le_bytes(data[34..36].try_into().unwrap());
    assert_eq!(bits_per_sample, 16, "expected 16-bit PCM WAV");

    let mut pos = 12;
    while pos + 8 < data.len() {
        let chunk_id = &data[pos..pos + 4];
        let chunk_size = u32::from_le_bytes(data[pos + 4..pos + 8].try_into().unwrap()) as usize;
        if chunk_id == b"data" {
            let samples: Vec<f32> = data[pos + 8..pos + 8 + chunk_size]
                .chunks_exact(2)
                .map(|bytes| i16::from_le_bytes([bytes[0], bytes[1]]) as f32 / 32768.0)
                .collect();
            return (samples, sample_rate);
        }
        pos += 8 + chunk_size;
    }
    panic!("no data chunk found in WAV");
}
