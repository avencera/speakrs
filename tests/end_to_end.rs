use std::fs;

use ndarray::{Array2, Array3};
use ndarray_npy::ReadNpyExt;
use speakrs::OwnedDiarizationPipeline;
use speakrs::inference::{EmbeddingModel, SegmentationModel};
use speakrs::pipeline::{DiarizationPipeline, FRAME_STEP_SECONDS, SEGMENTATION_STEP_SECONDS};

use speakrs::inference::ExecutionMode;
#[cfg(all(feature = "coreml", feature = "_metrics"))]
use speakrs::metrics::{compute_der, parse_rttm};
#[cfg(feature = "coreml")]
use speakrs::pipeline::FAST_SEGMENTATION_STEP_SECONDS;
#[cfg(feature = "coreml")]
use std::time::{Duration, Instant};

mod support;

use support::{build_pipeline_or_skip, fixture_path, load_model_or_skip, load_wav_samples};

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
    let Some(mut seg_model) = load_model_or_skip(SegmentationModel::new(
        models_dir.join("segmentation-3.0.onnx"),
        SEGMENTATION_STEP_SECONDS as f32,
    )) else {
        return;
    };
    let Some(mut emb_model) = load_model_or_skip(EmbeddingModel::new(
        models_dir.join("wespeaker-voxceleb-resnet34.onnx"),
    )) else {
        return;
    };
    let (samples, sr) = load_wav_samples(&fixture_path("test.wav"));
    assert_eq!(sr, 16_000);

    let mut pipeline =
        DiarizationPipeline::new(&mut seg_model, &mut emb_model, &models_dir).unwrap();
    let result = pipeline.run_with_file_id(&samples, "fixture").unwrap();

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
    assert!(result.rttm("fixture").contains("SPEAKER fixture 1"));
}

#[cfg(all(feature = "coreml", feature = "_metrics"))]
const VOXCONVERSE_TEST_FILES: &[&str] = &[
    "hqyok", "tfvyr", "qrzjk", "qpylu", "szsyz", "gwtwd", "fxgvy", "whmpa", "rtvuw", "usbgm",
    "bkwns", "abjxc", "syiwe", "qppll", "cobal", "oenox", "bwzyf", "jiqvr", "jyirt", "hiyis",
    "plbbw", "vysqj", "sikkm", "wjhgf", "lknjp", "mevkw", "kctgl", "zfkap", "iqtde", "xiglo",
    "jsmbi", "qydmg", "akthc", "exymw", "kbkon", "wmori", "ysgbf", "atgpi", "qjgpl",
];

#[cfg(all(feature = "coreml", feature = "_metrics"))]
fn voxconverse_der(mode: ExecutionMode, step: f64) -> Option<(Vec<(String, f64)>, Duration)> {
    let models_dir = fixture_path("models");
    let mut seg_model = load_model_or_skip(SegmentationModel::with_mode(
        models_dir.join("segmentation-3.0.onnx"),
        step as f32,
        mode,
    ))?;
    let mut emb_model = load_model_or_skip(EmbeddingModel::with_mode(
        models_dir.join("wespeaker-voxceleb-resnet34.onnx"),
        mode,
    ))?;
    let start = Instant::now();
    let mut results = Vec::new();
    let mut pipeline =
        DiarizationPipeline::new(&mut seg_model, &mut emb_model, &models_dir).unwrap();
    for &name in VOXCONVERSE_TEST_FILES {
        let wav_path = fixture_path(&format!("datasets/voxconverse/wav/{name}.wav"));
        let rttm_path = fixture_path(&format!("datasets/voxconverse/rttm/{name}.rttm"));

        let file_start = Instant::now();
        let (samples, sr) = load_wav_samples(&wav_path);
        assert_eq!(sr, 16_000);

        let result = pipeline.run_with_file_id(&samples, name).unwrap();

        let reference_rttm = fs::read_to_string(&rttm_path).unwrap();
        let reference = parse_rttm(&reference_rttm);
        let hypothesis = parse_rttm(&result.rttm(name));

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
    Some((results, elapsed))
}

#[test]
#[cfg(all(feature = "coreml", feature = "_metrics"))]
fn der_coreml_fp32() {
    let Some((results, elapsed)) =
        voxconverse_der(ExecutionMode::CoreMl, SEGMENTATION_STEP_SECONDS)
    else {
        return;
    };
    let avg_der: f64 = results.iter().map(|(_, d)| d).sum::<f64>() / results.len() as f64;
    eprintln!(
        "CoreML FP32 avg DER: {:.1}%, total: {:.1}s",
        avg_der * 100.0,
        elapsed.as_secs_f64()
    );

    assert!(
        avg_der < 0.105,
        "CoreML FP32 avg DER {:.1}% exceeds 10.5%",
        avg_der * 100.0
    );
    for (name, der) in &results {
        assert!(
            *der < 0.60,
            "CoreML FP32 {name}: DER {:.1}% exceeds 60%",
            der * 100.0
        );
    }
}

#[test]
#[cfg(all(feature = "coreml", feature = "_metrics"))]
fn der_coreml_fast() {
    let Some((results, elapsed)) =
        voxconverse_der(ExecutionMode::CoreMlFast, FAST_SEGMENTATION_STEP_SECONDS)
    else {
        return;
    };
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
    let Some(mut seg_model) = load_model_or_skip(SegmentationModel::new(
        models_dir.join("segmentation-3.0.onnx"),
        SEGMENTATION_STEP_SECONDS as f32,
    )) else {
        return;
    };
    let Some(mut emb_model) = load_model_or_skip(EmbeddingModel::new(
        models_dir.join("wespeaker-voxceleb-resnet34.onnx"),
    )) else {
        return;
    };
    let (samples, sr) = load_wav_samples(&fixture_path("test_short.wav"));
    assert_eq!(sr, 16_000);

    let mut pipeline =
        DiarizationPipeline::new(&mut seg_model, &mut emb_model, &models_dir).unwrap();
    let result = pipeline.run_with_file_id(&samples, "short").unwrap();

    assert!(
        result
            .discrete_diarization
            .iter()
            .all(|value| value.is_finite())
    );
    assert!(result.speaker_count.len() <= result.discrete_diarization.nrows());
}

#[test]
fn owned_pipeline_from_dir() {
    let models_dir = fixture_path("models");
    let Some(mut pipeline) = build_pipeline_or_skip(OwnedDiarizationPipeline::from_dir(
        &models_dir,
        ExecutionMode::Cpu,
    )) else {
        return;
    };

    let (samples, sr) = load_wav_samples(&fixture_path("test.wav"));
    assert_eq!(sr, 16_000);

    let result = pipeline.run(&samples).unwrap();

    assert!(result.discrete_diarization.nrows() > 0);
    assert!(result.discrete_diarization.ncols() > 0);
    assert!(
        result
            .discrete_diarization
            .iter()
            .all(|value| value.is_finite())
    );
    assert!(result.rttm("file1").contains("SPEAKER file1 1"));
}

/// Requires models deployed to HF (`cargo xtask models deploy`)
#[test]
#[ignore]
#[cfg(feature = "online")]
fn online_pipeline_downloads_and_runs() {
    let Some(mut pipeline) = build_pipeline_or_skip(OwnedDiarizationPipeline::from_pretrained(
        ExecutionMode::Cpu,
    )) else {
        return;
    };

    let (samples, sr) = load_wav_samples(&fixture_path("test.wav"));
    assert_eq!(sr, 16_000);

    let result = pipeline.run(&samples).unwrap();

    assert!(result.discrete_diarization.nrows() > 0);
    assert!(result.discrete_diarization.ncols() > 0);
    assert!(
        result
            .discrete_diarization
            .iter()
            .all(|value| value.is_finite())
    );
    assert!(result.rttm("file1").contains("SPEAKER file1 1"));
}
