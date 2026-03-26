use std::collections::HashMap;
use std::thread;

use speakrs::inference::ExecutionMode;
use speakrs::pipeline::{
    OwnedDiarizationPipeline, PipelineBuilder, PipelineConfig, QueueError,
    QueuedDiarizationRequest, ReconstructMethod,
};

mod support;

use support::{build_pipeline_or_skip, fixture_path, load_wav_samples};

fn make_pipeline() -> Option<OwnedDiarizationPipeline> {
    build_pipeline_or_skip(
        PipelineBuilder::from_dir(fixture_path("models"), ExecutionMode::Cpu).build(),
    )
}

fn single_speaker_config() -> PipelineConfig {
    PipelineConfig {
        speaker_keep_threshold: f64::MAX,
        ..PipelineConfig::default()
    }
}

fn config_candidates() -> Vec<PipelineConfig> {
    vec![
        PipelineConfig {
            merge_gap: 10.0,
            ..PipelineConfig::default()
        },
        PipelineConfig {
            reconstruct_method: ReconstructMethod::Standard,
            ..PipelineConfig::default()
        },
        single_speaker_config(),
    ]
}

#[test]
fn queued_basic_round_trip() {
    let (samples, _) = load_wav_samples(&fixture_path("test.wav"));
    let Some((tx, mut rx)) = make_pipeline().map(|pipeline| pipeline.into_queued().unwrap()) else {
        return;
    };

    tx.push(QueuedDiarizationRequest::new("file_a", samples.clone()))
        .unwrap();
    tx.push(QueuedDiarizationRequest::new("file_b", samples))
        .unwrap();
    drop(tx);

    let r1 = rx.recv().unwrap();
    let r2 = rx.recv().unwrap();

    assert!(r1.result.is_ok());
    assert!(r2.result.is_ok());
    assert!(!r1.result.unwrap().segments.is_empty());
    assert!(!r2.result.unwrap().segments.is_empty());
    assert!(matches!(rx.recv(), Err(QueueError::Closed)));
}

#[test]
fn queued_shared_senders_across_threads() {
    let (samples, _) = load_wav_samples(&fixture_path("test.wav"));
    let Some((tx, mut rx)) = make_pipeline().map(|pipeline| pipeline.into_queued().unwrap()) else {
        return;
    };

    let handles: Vec<_> = ["thread_a", "thread_b", "thread_c"]
        .into_iter()
        .map(|file_id| {
            let tx = tx.clone();
            let samples = samples.clone();
            thread::spawn(move || {
                tx.push(QueuedDiarizationRequest::new(file_id, samples))
                    .unwrap();
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }
    drop(tx);

    let mut results = HashMap::new();
    for _ in 0..3 {
        let result = rx.recv().unwrap();
        results.insert(result.file_id.clone(), result);
    }

    assert!(results.contains_key("thread_a"));
    assert!(results.contains_key("thread_b"));
    assert!(results.contains_key("thread_c"));
    assert!(results.values().all(|result| result.result.is_ok()));
    assert!(matches!(rx.recv(), Err(QueueError::Closed)));
}

#[test]
fn queued_job_ids_are_monotonic_across_sender_clones() {
    let (samples, _) = load_wav_samples(&fixture_path("test.wav"));
    let Some((tx, mut rx)) = make_pipeline().map(|pipeline| pipeline.into_queued().unwrap()) else {
        return;
    };
    let tx_clone = tx.clone();

    let id0 = tx
        .push(QueuedDiarizationRequest::new("f0", samples.clone()))
        .unwrap();
    let id1 = tx_clone
        .push(QueuedDiarizationRequest::new("f1", samples.clone()))
        .unwrap();
    let id2 = tx
        .push(QueuedDiarizationRequest::new("f2", samples))
        .unwrap();

    assert_ne!(id0, id1);
    assert_ne!(id1, id2);
    assert_ne!(id0, id2);

    drop(tx_clone);
    drop(tx);

    for _ in 0..3 {
        let _ = rx.recv().unwrap();
    }
    assert!(matches!(rx.recv(), Err(QueueError::Closed)));
}

#[test]
fn queued_clean_shutdown() {
    let (samples, _) = load_wav_samples(&fixture_path("test.wav"));
    let Some((tx, mut rx)) = make_pipeline().map(|pipeline| pipeline.into_queued().unwrap()) else {
        return;
    };

    tx.push(QueuedDiarizationRequest::new("only", samples))
        .unwrap();
    drop(tx);

    let result = rx.recv().unwrap();
    assert!(result.result.is_ok());
    assert!(matches!(rx.recv(), Err(QueueError::Closed)));
}

#[test]
fn queued_drop_sender_unblocks_iteration() {
    let (samples, _) = load_wav_samples(&fixture_path("test.wav"));
    let Some((tx, rx)) = make_pipeline().map(|pipeline| pipeline.into_queued().unwrap()) else {
        return;
    };

    tx.push(QueuedDiarizationRequest::new("iter", samples))
        .unwrap();
    drop(tx);

    let results: Vec<_> = rx.into_iter().collect();
    assert_eq!(results.len(), 1);
    assert!(results[0].as_ref().unwrap().result.is_ok());
}

#[test]
fn queued_handles_short_and_normal_audio() {
    let (samples, _) = load_wav_samples(&fixture_path("test.wav"));
    let Some((tx, rx)) = make_pipeline().map(|pipeline| pipeline.into_queued().unwrap()) else {
        return;
    };

    tx.push(QueuedDiarizationRequest::new("normal", samples))
        .unwrap();
    tx.push(QueuedDiarizationRequest::new("short", vec![0.0; 100]))
        .unwrap();
    drop(tx);

    let mut results = HashMap::new();
    for result in rx {
        let result = result.unwrap();
        results.insert(result.file_id.clone(), result);
    }

    assert!(results["normal"].result.is_ok());
    assert!(results.contains_key("short"));
}

#[test]
fn queued_results_match_sync() {
    let (samples, _) = load_wav_samples(&fixture_path("test.wav"));

    let Some(mut pipeline) = make_pipeline() else {
        return;
    };
    let sync_result = pipeline.run_with_file_id(&samples, "compare").unwrap();

    let Some((tx, mut rx)) = make_pipeline().map(|pipeline| pipeline.into_queued().unwrap()) else {
        return;
    };
    tx.push(QueuedDiarizationRequest::new("compare", samples))
        .unwrap();
    drop(tx);

    let queued_result = rx.recv().unwrap().result.unwrap();
    assert!(matches!(rx.recv(), Err(QueueError::Closed)));

    assert_eq!(sync_result.segments, queued_result.segments);
}

#[test]
fn build_queued_preserves_custom_pipeline_config() {
    let (samples, _) = load_wav_samples(&fixture_path("test.wav"));

    let Some(mut default_pipeline) = make_pipeline() else {
        return;
    };
    let default_result = default_pipeline
        .run_with_file_id(&samples, "compare")
        .unwrap();

    let (custom_config, custom_result) = config_candidates()
        .into_iter()
        .find_map(|config| {
            let mut pipeline = build_pipeline_or_skip(
                PipelineBuilder::from_dir(fixture_path("models"), ExecutionMode::Cpu)
                    .pipeline(config.clone())
                    .build(),
            )?;
            let result = pipeline.run_with_file_id(&samples, "compare").unwrap();
            (result.segments != default_result.segments).then_some((config, result))
        })
        .expect("expected at least one custom pipeline config to change fixture output");

    let Some((tx, mut rx)) = build_pipeline_or_skip(
        PipelineBuilder::from_dir(fixture_path("models"), ExecutionMode::Cpu)
            .pipeline(custom_config)
            .build_queued(),
    ) else {
        return;
    };
    tx.push(QueuedDiarizationRequest::new("compare", samples))
        .unwrap();
    drop(tx);

    let queued_result = rx.recv().unwrap().result.unwrap();
    assert!(matches!(rx.recv(), Err(QueueError::Closed)));

    assert_eq!(custom_result.segments, queued_result.segments);
}
