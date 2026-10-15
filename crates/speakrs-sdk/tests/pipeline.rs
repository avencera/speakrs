//! End-to-end checks against the CPU fixture models in `fixtures/models`

#![cfg(feature = "cpu")]

use std::cell::RefCell;
use std::path::PathBuf;
use std::time::Duration;

use speakrs_sdk::speakrs::ExecutionMode;
use speakrs_sdk::speakrs::pipeline::QueueConfig;
use speakrs_sdk::{
    CancelToken, ModelStore, PrepareModelsOptions, ProgressEvent, RunOptions, SdkErrorCategory,
    SdkPipeline,
};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures")
        .join(name)
}

fn cpu_pipeline() -> SdkPipeline {
    let mut options = PrepareModelsOptions::new(ExecutionMode::Cpu);
    options.model_dir = Some(fixture("models"));
    let prepared = ModelStore::new(std::env::temp_dir())
        .prepare(options, &|_| {})
        .unwrap();
    SdkPipeline::from_prepared(&prepared, ExecutionMode::Cpu, None, None).unwrap()
}

#[test]
fn batch_matches_single_file_runs_in_input_order() {
    let mut pipeline = cpu_pipeline();
    let files = [
        ("long", fixture("test.wav")),
        ("short", fixture("test_short.wav")),
    ];

    let singles: Vec<_> = files
        .iter()
        .map(|(file_id, path)| {
            pipeline
                .diarize_file(path, file_id, RunOptions::default())
                .unwrap()
        })
        .collect();
    let events = RefCell::new(Vec::new());
    let progress = |event| events.borrow_mut().push(event);
    let batch = pipeline
        .diarize_files(
            files
                .iter()
                .map(|(file_id, path)| (file_id.to_string(), path.clone()))
                .collect(),
            RunOptions {
                progress: Some(&progress),
                ..RunOptions::default()
            },
        )
        .unwrap();

    assert_eq!(batch.len(), 2);
    for (single, batched) in singles.iter().zip(&batch) {
        assert!(!single.segments.is_empty());
        assert_eq!(single.file_id, batched.file_id);
        assert_eq!(single.rttm(None, false), batched.rttm(None, false));
        assert_eq!(single.rttm(None, true), batched.rttm(None, true));
        assert!((single.duration - batched.duration).abs() < 1e-9);
    }
    assert!(singles[0].timing.is_some());
    assert!(batch[0].timing.is_none());
    assert_eq!(
        events.into_inner(),
        [
            ProgressEvent::BatchFile {
                index: 0,
                file_id: "long".into()
            },
            ProgressEvent::BatchFile {
                index: 1,
                file_id: "short".into()
            },
            ProgressEvent::Completed,
        ]
    );
}

#[test]
fn batch_reports_a_missing_file_and_cancellation() {
    let mut pipeline = cpu_pipeline();
    let missing = pipeline
        .diarize_files(
            vec![
                ("ok".into(), fixture("test_short.wav")),
                ("missing".into(), fixture("does-not-exist.wav")),
            ],
            RunOptions::default(),
        )
        .unwrap_err();
    assert_eq!(missing.category, SdkErrorCategory::AudioDecode);

    let token = CancelToken::new();
    token.cancel();
    let cancelled = pipeline
        .diarize_files(
            vec![("ok".into(), fixture("test_short.wav"))],
            RunOptions {
                cancel_token: Some(&token),
                ..RunOptions::default()
            },
        )
        .unwrap_err();
    assert_eq!(cancelled.category, SdkErrorCategory::Cancelled);

    // the pipeline stays usable after a failed batch
    let result = pipeline
        .diarize_file(&fixture("test_short.wav"), "again", RunOptions::default())
        .unwrap();
    assert!(!result.segments.is_empty());
}

#[test]
fn queue_returns_every_job_with_its_duration_then_closes() {
    let reference = cpu_pipeline()
        .diarize_file(&fixture("test_short.wav"), "path", RunOptions::default())
        .unwrap();
    let queue = cpu_pipeline()
        .into_queue(None, QueueConfig::new(4).unwrap())
        .unwrap();

    let path_job = queue
        .push_file("path".into(), &fixture("test_short.wav"))
        .unwrap();
    let silence_job = queue
        .push_samples("silence".into(), vec![0.0; 16_000])
        .unwrap();
    queue.close();

    let mut items = [
        queue.recv().unwrap(),
        queue
            .recv_timeout(Duration::from_secs(600))
            .unwrap()
            .expect("the second job finishes"),
    ];
    items.sort_by_key(|item| item.job_id);
    assert_eq!([items[0].job_id, items[1].job_id], [path_job, silence_job]);

    let path = items[0].result.as_ref().unwrap();
    assert_eq!(path.rttm(None, false), reference.rttm(None, false));
    assert!((path.duration - reference.duration).abs() < 1e-9);
    assert!((items[1].result.as_ref().unwrap().duration - 1.0).abs() < 1e-9);

    let closed = queue.recv().unwrap_err();
    assert_eq!(closed.code, speakrs_sdk::queue::QUEUE_CLOSED);
    let rejected = queue
        .push_samples("late".into(), vec![0.0; 16])
        .unwrap_err();
    assert_eq!(rejected.code, speakrs_sdk::queue::QUEUE_CLOSED);
}
