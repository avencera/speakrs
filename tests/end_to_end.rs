use std::fs;
use std::path::{Path, PathBuf};

use ndarray::{Array2, Array3};
use ndarray_npy::ReadNpyExt;
use speakrs::clustering::plda::PldaTransform;
use speakrs::inference::embedding::EmbeddingModel;
use speakrs::inference::segmentation::SegmentationModel;
use speakrs::pipeline::{FRAME_STEP_SECONDS, SEGMENTATION_STEP_SECONDS, diarize};

#[cfg(feature = "coreml")]
use speakrs::inference::ExecutionMode;
#[cfg(feature = "coreml")]
use speakrs::metrics::{compute_der, parse_rttm};
#[cfg(feature = "coreml")]
use speakrs::pipeline::FAST_SEGMENTATION_STEP_SECONDS;
#[cfg(feature = "coreml")]
use std::time::{Duration, Instant};

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join(name)
}

fn load_wav_samples(path: &Path) -> (Vec<f32>, u32) {
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

#[test]
fn pipeline_fixture_shapes_are_available() {
    let seg: Array3<f32> =
        Array3::read_npy(fs::File::open(fixture_path("pipeline_segmentation_data.npy")).unwrap())
            .unwrap();
    assert_eq!(seg.shape(), &[18, 589, 3]);

    let counting: Array2<u8> = Array2::read_npy(
        fs::File::open(fixture_path("pipeline_speaker_counting_data.npy")).unwrap(),
    )
    .unwrap();
    assert!(counting.nrows() > 0);

    let embeddings: Array3<f32> =
        Array3::read_npy(fs::File::open(fixture_path("pipeline_embeddings_data.npy")).unwrap())
            .unwrap();
    assert_eq!(embeddings.shape()[0], 18);
    assert_eq!(embeddings.shape()[2], 256);
}

#[test]
fn segmentation_step_matches_pyannote_fixture() {
    assert_eq!(SEGMENTATION_STEP_SECONDS, 1.0);
    assert_eq!(FRAME_STEP_SECONDS, 0.016875);
}

#[test]
fn pipeline_runs_on_main_fixture_audio() {
    let models_dir = fixture_path("models");
    let mut seg_model = SegmentationModel::new(
        models_dir.join("segmentation-3.0.onnx").to_str().unwrap(),
        SEGMENTATION_STEP_SECONDS as f32,
    )
    .unwrap();
    let mut emb_model = EmbeddingModel::new(
        models_dir
            .join("wespeaker-voxceleb-resnet34.onnx")
            .to_str()
            .unwrap(),
    )
    .unwrap();
    let plda = PldaTransform::from_dir(&models_dir).unwrap();

    let (samples, sr) = load_wav_samples(&fixture_path("test.wav"));
    assert_eq!(sr, 16_000);

    let result = diarize(&mut seg_model, &mut emb_model, &plda, &samples, "fixture").unwrap();

    assert!(result.discrete_diarization.nrows() > 0);
    assert!(result.discrete_diarization.ncols() > 0);
    assert_eq!(
        result.speaker_count.len(),
        result.discrete_diarization.nrows()
    );
    assert!(
        result
            .discrete_diarization
            .iter()
            .all(|value| value.is_finite())
    );
    assert!(result.rttm.contains("SPEAKER fixture 1"));
}

#[cfg(feature = "coreml")]
const VOXCONVERSE_TEST_FILES: &[&str] = &[
    "hqyok", "tfvyr", "qrzjk", "qpylu", "szsyz", "gwtwd", "fxgvy", "whmpa", "rtvuw", "usbgm",
    "bkwns", "abjxc", "syiwe", "qppll", "cobal", "oenox", "bwzyf", "jiqvr", "jyirt", "hiyis",
    "plbbw", "vysqj", "sikkm", "wjhgf", "lknjp", "mevkw", "kctgl", "zfkap", "iqtde", "xiglo",
    "jsmbi", "qydmg", "akthc", "exymw", "kbkon", "wmori", "ysgbf", "atgpi", "qjgpl",
];

#[cfg(feature = "coreml")]
fn voxconverse_der(mode: ExecutionMode, step: f64) -> (Vec<(String, f64)>, Duration) {
    let models_dir = fixture_path("models");
    let mut seg_model = SegmentationModel::with_mode(
        models_dir.join("segmentation-3.0.onnx").to_str().unwrap(),
        step as f32,
        mode,
    )
    .unwrap();
    let mut emb_model = EmbeddingModel::with_mode(
        models_dir
            .join("wespeaker-voxceleb-resnet34.onnx")
            .to_str()
            .unwrap(),
        mode,
    )
    .unwrap();
    let plda = PldaTransform::from_dir(&models_dir).unwrap();

    let start = Instant::now();
    let mut results = Vec::new();
    for &name in VOXCONVERSE_TEST_FILES {
        let wav_path = fixture_path(&format!("datasets/voxconverse/wav/{name}.wav"));
        let rttm_path = fixture_path(&format!("datasets/voxconverse/rttm/{name}.rttm"));

        let file_start = Instant::now();
        let (samples, sr) = load_wav_samples(&wav_path);
        assert_eq!(sr, 16_000);

        let result = diarize(&mut seg_model, &mut emb_model, &plda, &samples, name).unwrap();

        let reference_rttm = fs::read_to_string(&rttm_path).unwrap();
        let reference = parse_rttm(&reference_rttm);
        let hypothesis = parse_rttm(&result.rttm);

        let der_result = compute_der(&reference, &hypothesis);
        let file_elapsed = file_start.elapsed();
        eprintln!(
            "{name}: DER={:.1}% (miss={:.1}% fa={:.1}% conf={:.1}%) ({:.1}s)",
            der_result.der() * 100.0,
            der_result.missed / der_result.total * 100.0,
            der_result.false_alarm / der_result.total * 100.0,
            der_result.confusion / der_result.total * 100.0,
            file_elapsed.as_secs_f64(),
        );
        results.push((name.to_string(), der_result.der()));
    }
    let elapsed = start.elapsed();
    (results, elapsed)
}

#[test]
#[cfg(feature = "coreml")]
fn der_coreml_fp32() {
    let (results, elapsed) = voxconverse_der(ExecutionMode::CoreMl, SEGMENTATION_STEP_SECONDS);
    let avg_der: f64 = results.iter().map(|(_, d)| d).sum::<f64>() / results.len() as f64;
    eprintln!(
        "CoreML FP32 avg DER: {:.1}%, total: {:.1}s",
        avg_der * 100.0,
        elapsed.as_secs_f64()
    );

    assert!(
        avg_der < 0.10,
        "CoreML FP32 avg DER {:.1}% exceeds 10%",
        avg_der * 100.0
    );
    for (name, der) in &results {
        assert!(
            *der < 0.45,
            "CoreML FP32 {name}: DER {:.1}% exceeds 45%",
            der * 100.0
        );
    }
}

#[test]
#[cfg(feature = "coreml")]
fn der_coreml_fast() {
    let (results, elapsed) =
        voxconverse_der(ExecutionMode::CoreMlFast, FAST_SEGMENTATION_STEP_SECONDS);
    let avg_der: f64 = results.iter().map(|(_, d)| d).sum::<f64>() / results.len() as f64;
    eprintln!(
        "CoreML Fast (FP32+2s) avg DER: {:.1}%, total: {:.1}s",
        avg_der * 100.0,
        elapsed.as_secs_f64()
    );

    assert!(
        avg_der < 0.12,
        "CoreML Fast avg DER {:.1}% exceeds 12%",
        avg_der * 100.0
    );
    for (name, der) in &results {
        assert!(
            *der < 0.45,
            "CoreML Fast {name}: DER {:.1}% exceeds 45%",
            der * 100.0
        );
    }
}

#[test]
fn pipeline_handles_short_audio_fixture() {
    let models_dir = fixture_path("models");
    let mut seg_model = SegmentationModel::new(
        models_dir.join("segmentation-3.0.onnx").to_str().unwrap(),
        SEGMENTATION_STEP_SECONDS as f32,
    )
    .unwrap();
    let mut emb_model = EmbeddingModel::new(
        models_dir
            .join("wespeaker-voxceleb-resnet34.onnx")
            .to_str()
            .unwrap(),
    )
    .unwrap();
    let plda = PldaTransform::from_dir(&models_dir).unwrap();

    let (samples, sr) = load_wav_samples(&fixture_path("test_short.wav"));
    assert_eq!(sr, 16_000);

    let result = diarize(&mut seg_model, &mut emb_model, &plda, &samples, "short").unwrap();

    assert!(
        result
            .discrete_diarization
            .iter()
            .all(|value| value.is_finite())
    );
    assert!(result.speaker_count.len() <= result.discrete_diarization.nrows());
}
