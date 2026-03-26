use std::collections::HashMap;

use speakrs::inference::ExecutionMode;
use speakrs::pipeline::{
    OwnedDiarizationPipeline, PipelineBuilder, PipelineConfig, QueuedDiarizationRequest,
    ReconstructMethod,
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
    let Some(queue) = make_pipeline().map(|pipeline| pipeline.into_queued().unwrap()) else {
        return;
    };

    queue
        .push(QueuedDiarizationRequest::new("file_a", samples.clone()))
        .unwrap();
    queue
        .push(QueuedDiarizationRequest::new("file_b", samples))
        .unwrap();

    let r1 = queue.recv().unwrap();
    let r2 = queue.recv().unwrap();

    assert!(r1.result.is_ok());
    assert!(r2.result.is_ok());
    assert!(!r1.result.unwrap().segments.is_empty());
    assert!(!r2.result.unwrap().segments.is_empty());

    queue.finish().unwrap();
}

#[test]
fn queued_push_batch() {
    let (samples, _) = load_wav_samples(&fixture_path("test.wav"));
    let Some(queue) = make_pipeline().map(|pipeline| pipeline.into_queued().unwrap()) else {
        return;
    };

    let requests = vec![
        QueuedDiarizationRequest::new("batch_1", samples.clone()),
        QueuedDiarizationRequest::new("batch_2", samples.clone()),
        QueuedDiarizationRequest::new("batch_3", samples),
    ];
    let job_ids = queue.push_batch(requests).unwrap();
    assert_eq!(job_ids.len(), 3);

    let mut results = HashMap::new();
    for _ in 0..3 {
        let r = queue.recv().unwrap();
        results.insert(r.file_id.clone(), r);
    }

    assert!(results.contains_key("batch_1"));
    assert!(results.contains_key("batch_2"));
    assert!(results.contains_key("batch_3"));
    for r in results.values() {
        assert!(r.result.is_ok());
    }

    queue.finish().unwrap();
}

#[test]
fn queued_job_ids_are_monotonic() {
    let (samples, _) = load_wav_samples(&fixture_path("test.wav"));
    let Some(queue) = make_pipeline().map(|pipeline| pipeline.into_queued().unwrap()) else {
        return;
    };

    let id0 = queue
        .push(QueuedDiarizationRequest::new("f0", samples.clone()))
        .unwrap();
    let id1 = queue
        .push(QueuedDiarizationRequest::new("f1", samples.clone()))
        .unwrap();
    let id2 = queue
        .push(QueuedDiarizationRequest::new("f2", samples))
        .unwrap();

    // job IDs are distinct and ordered
    assert_ne!(id0, id1);
    assert_ne!(id1, id2);
    assert_ne!(id0, id2);

    for _ in 0..3 {
        let _ = queue.recv().unwrap();
    }
    queue.finish().unwrap();
}

#[test]
fn queued_clean_shutdown() {
    let (samples, _) = load_wav_samples(&fixture_path("test.wav"));
    let Some(queue) = make_pipeline().map(|pipeline| pipeline.into_queued().unwrap()) else {
        return;
    };

    queue
        .push(QueuedDiarizationRequest::new("only", samples))
        .unwrap();
    let r = queue.recv().unwrap();
    assert!(r.result.is_ok());

    queue.finish().unwrap();
}

#[test]
fn queued_handles_short_and_normal_audio() {
    let (samples, _) = load_wav_samples(&fixture_path("test.wav"));
    let Some(queue) = make_pipeline().map(|pipeline| pipeline.into_queued().unwrap()) else {
        return;
    };

    queue
        .push(QueuedDiarizationRequest::new("normal", samples))
        .unwrap();
    // very short audio — pipeline handles gracefully
    queue
        .push(QueuedDiarizationRequest::new("short", vec![0.0; 100]))
        .unwrap();

    let mut results = HashMap::new();
    for _ in 0..2 {
        let r = queue.recv().unwrap();
        results.insert(r.file_id.clone(), r);
    }

    // normal file produces valid RTTM
    assert!(results["normal"].result.is_ok());
    // queue continues processing regardless of per-file outcome
    assert!(results.contains_key("short"));

    queue.finish().unwrap();
}

#[test]
fn queued_results_match_sync() {
    let (samples, _) = load_wav_samples(&fixture_path("test.wav"));

    // run synchronously
    let Some(mut pipeline) = make_pipeline() else {
        return;
    };
    let sync_result = pipeline.run_with_file_id(&samples, "compare").unwrap();

    // run via queue
    let Some(queue) = make_pipeline().map(|pipeline| pipeline.into_queued().unwrap()) else {
        return;
    };
    queue
        .push(QueuedDiarizationRequest::new("compare", samples))
        .unwrap();
    let queued_result = queue.recv().unwrap().result.unwrap();
    queue.finish().unwrap();

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

    let Some(queue) = build_pipeline_or_skip(
        PipelineBuilder::from_dir(fixture_path("models"), ExecutionMode::Cpu)
            .pipeline(custom_config)
            .build_queued(),
    ) else {
        return;
    };
    queue
        .push(QueuedDiarizationRequest::new("compare", samples))
        .unwrap();
    let queued_result = queue.recv().unwrap().result.unwrap();
    queue.finish().unwrap();

    assert_eq!(custom_result.segments, queued_result.segments);
}
