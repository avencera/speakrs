//! Opt-in checks for `RuntimeConfig::coreml_segmentation_batch`
//!
//! They need the CoreML models from Hugging Face, so they are ignored by default:
//!
//! ```sh
//! cargo test --release --features coreml --test coreml_segmentation_batch -- --ignored
//! ```
#![cfg(all(target_os = "macos", feature = "online", feature = "coreml"))]

use speakrs::inference::{ExecutionMode, ModelLoadError, SegmentationModel};
use speakrs::pipeline::{COREML_SEGMENTATION_STEP_SECONDS, RuntimeConfig};
use speakrs::{CoreMlSegmentationBatch, ModelBundle, PipelineBuilder};

mod support;

use support::{fixture_path, load_wav_samples};

/// Long enough for the pipeline's batched segmentation path, which starts at 192 windows
const LONG_AUDIO_SECONDS: usize = 240;

const BATCHES: [CoreMlSegmentationBatch; 3] = [
    CoreMlSegmentationBatch::Single,
    CoreMlSegmentationBatch::Batch32,
    CoreMlSegmentationBatch::Batch64,
];

fn runtime(batch: CoreMlSegmentationBatch) -> RuntimeConfig {
    RuntimeConfig::default().with_coreml_segmentation_batch(batch)
}

/// Requires models deployed to HF (`cargo xtask models deploy`)
#[test]
#[ignore]
fn every_segmentation_batch_gives_the_same_diarization() {
    let (clip, sample_rate) = load_wav_samples(&fixture_path("test.wav"));
    assert_eq!(sample_rate, 16_000);
    let long_audio: Vec<f32> = clip
        .iter()
        .copied()
        .cycle()
        .take(LONG_AUDIO_SECONDS * 16_000)
        .collect();

    let rttms: Vec<String> = BATCHES
        .into_iter()
        .map(|batch| {
            let bundle = ModelBundle::from_pretrained(ExecutionMode::CoreMl).unwrap();
            let mut pipeline = PipelineBuilder::from_bundle(bundle, ExecutionMode::CoreMl)
                .runtime(runtime(batch))
                .build()
                .unwrap();
            pipeline.run(&long_audio).unwrap().rttm("long")
        })
        .collect();

    assert!(rttms[0].contains("SPEAKER long 1"));
    for (batch, rttm) in BATCHES.iter().zip(&rttms).skip(1) {
        assert_eq!(rttm, &rttms[0], "{batch:?} diarization differs from Single");
    }
}

/// Requires models deployed to HF (`cargo xtask models deploy`)
#[test]
#[ignore]
fn single_batch_loads_without_the_batched_bundles() {
    let bundle = ModelBundle::from_pretrained(ExecutionMode::CoreMl).unwrap();
    let models_dir = bundle.segmentation_path().parent().unwrap();

    // a directory with only the single-window bundle, as an app that ships batch 1 would have
    let single_only = tempfile::tempdir().unwrap();
    std::os::unix::fs::symlink(
        models_dir.join("segmentation-3.0.mlmodelc"),
        single_only.path().join("segmentation-3.0.mlmodelc"),
    )
    .unwrap();
    let model_path = single_only.path().join("segmentation-3.0.onnx");
    let load = |batch| {
        SegmentationModel::with_mode_and_config(
            &model_path,
            COREML_SEGMENTATION_STEP_SECONDS as f32,
            ExecutionMode::CoreMl,
            &runtime(batch),
        )
    };

    // more windows than one batch-32 prediction, so both sequential paths would batch them if
    // the single-window limit were ignored
    let (clip, _) = load_wav_samples(&fixture_path("test.wav"));
    let audio: Vec<f32> = clip.iter().copied().cycle().take(60 * 16_000).collect();
    let mut single = load(CoreMlSegmentationBatch::Single).unwrap();

    let collected = single.run(&audio).unwrap();
    // 60 s at a 1 s step: 51 full windows plus the zero-padded tail window
    assert_eq!(collected.len(), 52);

    let (tx, rx) = crossbeam_channel::unbounded();
    let streamed_windows = single.run_streaming(&audio, tx).unwrap();
    let streamed: Vec<_> = rx.into_iter().collect();
    assert_eq!(streamed_windows, collected.len());
    assert_eq!(streamed, collected);

    let error = match load(CoreMlSegmentationBatch::Batch32) {
        Ok(_) => panic!("Batch32 should need the batch-32 bundle"),
        Err(error) => error,
    };
    match error {
        ModelLoadError::MissingNativeAsset { path, .. } => {
            assert_eq!(
                path,
                single_only.path().join("segmentation-3.0-b32.mlmodelc")
            );
        }
        other => panic!("unexpected error: {other}"),
    }
}
