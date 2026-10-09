//! CPU batch output must match independent sequential runs byte for byte
#![cfg(feature = "cpu")]

use std::convert::Infallible;

use speakrs::{
    BatchInput, BatchStreamError, ExecutionMode, OwnedBatchInput, OwnedDiarizationPipeline,
};

mod support;

#[path = "../src/test_support.rs"]
mod model_fixtures;

use model_fixtures::model_fixture_dir;
use support::{fixture_path, load_wav_samples};

fn pipeline() -> OwnedDiarizationPipeline {
    OwnedDiarizationPipeline::from_dir(model_fixture_dir(), ExecutionMode::Cpu).unwrap()
}

#[test]
fn cpu_batches_match_sequential_bytes_and_input_order() {
    let (audio, sample_rate) = load_wav_samples(&fixture_path("test.wav"));
    assert_eq!(sample_rate, 16_000);
    let files = [
        ("third", audio[..320_001].to_vec()),
        ("first", audio[160_000..400_003].to_vec()),
        ("second", audio[..400_007].to_vec()),
    ];
    let mut sequential = pipeline();
    let expected: Vec<_> = files
        .iter()
        .map(|(file_id, audio)| {
            let output = sequential
                .run_with_file_id(audio, file_id)
                .unwrap()
                .rttm(file_id);
            assert!(!output.is_empty(), "fixture must contain speech");
            output
        })
        .collect();
    let inputs: Vec<_> = files
        .iter()
        .map(|(file_id, audio)| BatchInput { file_id, audio })
        .collect();
    let mut batch = pipeline();

    for _ in 0..2 {
        let outputs = batch.run_batch(&inputs).unwrap();
        assert_eq!(outputs.len(), files.len());
        for ((output, (file_id, _)), expected) in outputs.iter().zip(&files).zip(&expected) {
            assert_eq!(output.rttm(file_id).as_bytes(), expected.as_bytes());
        }

        let stream = files.iter().map(|(file_id, audio)| {
            Ok::<_, Infallible>(OwnedBatchInput {
                file_id: (*file_id).to_owned(),
                audio: audio.clone(),
            })
        });
        let outputs = batch.run_batch_stream(stream).unwrap();
        assert_eq!(outputs.len(), files.len());
        for ((output, (file_id, audio)), expected) in outputs.iter().zip(&files).zip(&expected) {
            assert_eq!(output.file_id, *file_id);
            assert_eq!(output.audio_secs, audio.len() as f64 / 16_000.0);
            assert_eq!(
                output.result.rttm(&output.file_id).as_bytes(),
                expected.as_bytes()
            );
        }
    }
}

#[test]
fn stream_failure_returns_no_partial_batch_and_pipeline_is_reusable() {
    let (audio, _) = load_wav_samples(&fixture_path("test.wav"));
    let audio = audio[..320_001].to_vec();
    let mut pipeline = pipeline();
    let expected = pipeline.run(&audio).unwrap().rttm("file1");
    let mut consumed = 0;
    let source = [
        Ok(OwnedBatchInput {
            file_id: "good".to_owned(),
            audio: audio.clone(),
        }),
        Err("decode failure"),
        Err("must not consume this file"),
    ]
    .into_iter()
    .inspect(|_| consumed += 1);
    let error = pipeline.run_batch_stream(source).err().unwrap();
    assert!(matches!(error, BatchStreamError::Input("decode failure")));
    assert_eq!(consumed, 2);
    assert_eq!(pipeline.run(&audio).unwrap().rttm("file1"), expected);
}
