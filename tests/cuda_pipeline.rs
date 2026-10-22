//! The native CUDA pipeline across the threads the pipeline really uses
//!
//! Models load on the test thread. Each run segments on a scoped worker thread that the
//! pipeline starts for that run while embedding stays on the calling thread; then the
//! pipeline moves to another thread, a `clone_shared` handle runs on another thread
//! next to it (not in CoreML builds, which have no `clone_shared`), and a queued
//! pipeline runs on its worker thread. Every run must give the same result.
//!
//! The CUDA weights, PLDA files and embedding metadata are read from
//! `SPEAKRS_CUDA_ASSETS`, else `/workspace/models-native`, else
//! `~/Library/Caches/speakrs-cuda-assets` (see `scripts/cuda/README.md`). The test skips
//! without a GPU or without those files; `SPEAKRS_REQUIRE_GPU=1` turns skips into failures
#![cfg(feature = "cuda")]

use std::path::PathBuf;
use std::thread;

use speakrs::inference::{CudaGraphs, CudaLstmAlgorithm, CudaMath, ExecutionMode, ModelLoadError};
use speakrs::pipeline::{
    BatchStreamError, DiarizationResult, OwnedBatchInput, OwnedDiarizationPipeline,
    PipelineBuilder, PipelineError, QueuedDiarizationRequest, RuntimeConfig,
};

mod support;

use support::{fixture_path, load_wav_samples};

fn required() -> bool {
    std::env::var_os("SPEAKRS_REQUIRE_GPU").is_some_and(|value| value != "0")
}

fn skip(test: &str, reason: &str) {
    assert!(!required(), "{test}: {reason}");
    eprintln!("skipping {test}: {reason}");
}

fn assets_dir() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")
        .map(|home| PathBuf::from(home).join("Library/Caches/speakrs-cuda-assets"));
    std::env::var_os("SPEAKRS_CUDA_ASSETS")
        .map(PathBuf::from)
        .into_iter()
        .chain([PathBuf::from("/workspace/models-native")])
        .chain(home)
        .find(|dir| dir.join("segmentation-3.0.safetensors").is_file())
}

fn pipeline(test: &str, mode: ExecutionMode) -> Option<OwnedDiarizationPipeline> {
    pipeline_with(test, mode, RuntimeConfig::default())
}

fn pipeline_with(
    test: &str,
    mode: ExecutionMode,
    runtime: RuntimeConfig,
) -> Option<OwnedDiarizationPipeline> {
    let Some(dir) = assets_dir() else {
        skip(test, "no CUDA assets (set SPEAKRS_CUDA_ASSETS)");
        return None;
    };

    let built =
        PipelineBuilder::from_dir(dir, mode).and_then(|builder| builder.runtime(runtime).build());
    match built {
        Ok(pipeline) => Some(pipeline),
        Err(PipelineError::ModelLoad(ModelLoadError::Cuda(error)))
            if error.is_device_unavailable() =>
        {
            skip(test, &format!("no usable NVIDIA GPU ({error})"));
            None
        }
        Err(error) => panic!("{test}: failed to build the CUDA pipeline: {error}"),
    }
}

fn turns(result: &DiarizationResult) -> String {
    result.rttm("test")
}

#[test]
fn cuda_pipeline_runs_on_every_pipeline_thread() {
    let test = "cuda_pipeline_runs_on_every_pipeline_thread";
    let (audio, _) = load_wav_samples(&fixture_path("test.wav"));

    for mode in [ExecutionMode::Cuda, ExecutionMode::CudaFast] {
        let Some(mut pipeline) = pipeline(test, mode) else {
            return;
        };

        // each run segments on a new scoped thread and embeds on this one
        let expected = turns(&pipeline.run(&audio).unwrap());
        assert!(!expected.is_empty(), "{mode}: no speaker turns");
        assert_eq!(turns(&pipeline.run(&audio).unwrap()), expected, "{mode}");

        // the pipeline moves to another thread and runs there
        let (mut pipeline, moved) = thread::scope(|scope| {
            scope
                .spawn(|| {
                    let mut pipeline = pipeline;
                    let result = turns(&pipeline.run(&audio).unwrap());
                    (pipeline, result)
                })
                .join()
                .unwrap()
        });
        assert_eq!(moved, expected, "{mode}: moved pipeline");

        // a handle with its own CUDA stream runs on another thread while the original
        // runs on this one
        #[cfg(not(feature = "coreml"))]
        {
            let shared = pipeline.clone_shared().unwrap();
            let (cloned, original) = thread::scope(|scope| {
                let cloned = scope.spawn(|| {
                    let mut shared = shared;
                    turns(&shared.run(&audio).unwrap())
                });
                let original = turns(&pipeline.run(&audio).unwrap());
                (cloned.join().unwrap(), original)
            });
            assert_eq!(cloned, expected, "{mode}: clone_shared handle");
            assert_eq!(original, expected, "{mode}: original next to clone_shared");
        }
        #[cfg(feature = "coreml")]
        let _ = &mut pipeline;

        // a queued pipeline runs on its worker thread and drops there
        let (sender, mut receiver) = pipeline.into_queued().unwrap();
        sender
            .try_push(QueuedDiarizationRequest::new("test", audio.clone()))
            .unwrap();
        drop(sender);
        let queued = receiver.recv().unwrap().result.unwrap();
        assert_eq!(turns(&queued), expected, "{mode}: queued pipeline");
    }
}

#[test]
fn cuda_pipeline_runs_with_every_runtime_option() {
    let test = "cuda_pipeline_runs_with_every_runtime_option";
    let (audio, _) = load_wav_samples(&fixture_path("test.wav"));
    let configs = [
        RuntimeConfig::default().with_cuda_lstm_algorithm(CudaLstmAlgorithm::PersistStaticSmallH),
        RuntimeConfig::default().with_cuda_lstm_algorithm(CudaLstmAlgorithm::PersistDynamic),
        RuntimeConfig::default()
            .with_cuda_segmentation_math(CudaMath::Fp32)
            .with_cuda_embedding_math(CudaMath::Fp32)
            .with_cuda_graphs(CudaGraphs::Disabled),
    ];

    for runtime in configs {
        let label = format!("{runtime:?}");
        let Some(mut pipeline) = pipeline_with(test, ExecutionMode::Cuda, runtime) else {
            return;
        };
        let result = pipeline.run(&audio).unwrap();
        assert!(!turns(&result).is_empty(), "{label}: no speaker turns");
    }
}

/// Batches segment the next file while the current one embeds; outputs must still match
/// independent runs byte for byte and keep input order, in both CUDA modes
#[test]
fn cuda_batch_stream_matches_sequential_runs() {
    let test = "cuda_batch_stream_matches_sequential_runs";
    let (audio, _) = load_wav_samples(&fixture_path("test.wav"));
    // a short file between longer ones, so one file's segmentation ends early
    let files = [
        ("third", audio[..320_001].to_vec()),
        ("short", audio[160_000..200_000].to_vec()),
        ("first", audio[160_000..400_003].to_vec()),
        ("second", audio[..400_007].to_vec()),
    ];
    for mode in [ExecutionMode::Cuda, ExecutionMode::CudaFast] {
        let Some(mut pipeline) = pipeline(test, mode) else {
            return;
        };
        let expected: Vec<_> = files
            .iter()
            .map(|(file_id, audio)| {
                pipeline
                    .run_with_file_id(audio, file_id)
                    .unwrap()
                    .rttm(file_id)
            })
            .collect();
        for round in 0..2 {
            let stream = files.iter().map(|(file_id, audio)| {
                Ok::<_, std::convert::Infallible>(OwnedBatchInput {
                    file_id: (*file_id).to_owned(),
                    audio: audio.clone(),
                })
            });
            let outputs = pipeline.run_batch_stream(stream).unwrap();
            assert_eq!(outputs.len(), files.len());
            for ((output, (file_id, _)), expected) in outputs.iter().zip(&files).zip(&expected) {
                assert_eq!(output.file_id, *file_id, "{mode} round {round}");
                assert_eq!(
                    output.result.rttm(file_id).as_bytes(),
                    expected.as_bytes(),
                    "{mode} round {round}: {file_id}"
                );
            }
        }
    }
}

/// An input error stops the stream after the file before it, without reading further
/// inputs, and the pipeline still runs afterwards
#[test]
fn cuda_stream_failure_returns_no_partial_batch() {
    let test = "cuda_stream_failure_returns_no_partial_batch";
    let (audio, _) = load_wav_samples(&fixture_path("test.wav"));
    let audio = audio[..320_001].to_vec();
    let Some(mut pipeline) = pipeline(test, ExecutionMode::Cuda) else {
        return;
    };
    let expected = turns(&pipeline.run(&audio).unwrap());
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
    assert_eq!(turns(&pipeline.run(&audio).unwrap()), expected);
}
