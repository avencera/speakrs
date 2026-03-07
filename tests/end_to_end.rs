use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;

use ndarray::{Array2, Array3, s};
use ndarray_npy::ReadNpyExt;
use speakrs::aggregate::overlap_add;
use speakrs::binarize::{BinarizeConfig, binarize};
use speakrs::clustering::vbx::{VbxConfig, cluster};
use speakrs::inference::embedding::EmbeddingModel;
use speakrs::inference::segmentation::SegmentationModel;
use speakrs::powerset::PowersetMapping;
use speakrs::reconstruct::{count_speakers, make_exclusive, reconstruct};
use speakrs::segment::{merge_segments, to_rttm, to_segments};

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join(name)
}

fn parse_rttm(content: &str) -> Vec<(f64, f64, String)> {
    content
        .lines()
        .filter(|line| !line.is_empty())
        .map(|line| {
            let fields: Vec<&str> = line.split_whitespace().collect();
            let start: f64 = fields[3].parse().unwrap();
            let duration: f64 = fields[4].parse().unwrap();
            let speaker = fields[7].to_string();
            (start, duration, speaker)
        })
        .collect()
}

#[test]
fn test_pipeline_fixture_shapes() {
    // segmentation is [chunks, frames, speakers] — already soft-decoded
    let seg: Array3<f32> =
        Array3::read_npy(fs::File::open(fixture_path("pipeline_segmentation_data.npy")).unwrap())
            .unwrap();
    assert_eq!(seg.shape()[0], 18);
    assert_eq!(seg.shape()[1], 589);
    assert_eq!(seg.shape()[2], 3);

    let counting: Array2<u8> = Array2::read_npy(
        fs::File::open(fixture_path("pipeline_speaker_counting_data.npy")).unwrap(),
    )
    .unwrap();
    assert!(counting.nrows() > 0);

    // embeddings are [chunks, speakers_per_chunk, embedding_dim]
    let embeddings: Array3<f32> =
        Array3::read_npy(fs::File::open(fixture_path("pipeline_embeddings_data.npy")).unwrap())
            .unwrap();
    assert_eq!(embeddings.shape()[0], 18);
    assert_eq!(embeddings.shape()[2], 256);

    let diarization: Array2<f32> = Array2::read_npy(
        fs::File::open(fixture_path("pipeline_discrete_diarization_data.npy")).unwrap(),
    )
    .unwrap();
    assert!(diarization.nrows() > 0);
    assert!(diarization.ncols() > 0);
}

#[test]
fn test_expected_rttm_parseable() {
    let content = fs::read_to_string(fixture_path("expected.rttm")).unwrap();
    let segments = parse_rttm(&content);

    assert!(!segments.is_empty());
    for (start, duration, speaker) in &segments {
        assert!(*start >= 0.0, "start should be non-negative");
        assert!(*duration > 0.0, "duration should be positive");
        assert!(!speaker.is_empty(), "speaker should be non-empty");
    }
}

#[test]
fn test_pipeline_params_readable() {
    let content = fs::read_to_string(fixture_path("pipeline_params.json")).unwrap();
    let params: serde_json::Value = serde_json::from_str(&content).unwrap();

    let clustering = params["clustering"].as_str().unwrap();
    assert!(clustering.contains("0.07"), "expected Fa=0.07");
    assert!(clustering.contains("0.8"), "expected Fb=0.8");
}

#[test]
fn test_powerset_on_segmentation_fixture() {
    // the pipeline fixture is already soft-decoded [18, 589, 3]
    // use the raw powerset test fixture instead — shape is [1, 589, 7]
    let logits_3d: Array3<f32> =
        Array3::read_npy(fs::File::open(fixture_path("powerset_input_logits.npy")).unwrap())
            .unwrap();
    assert_eq!(logits_3d.shape()[2], 7, "expected 7 powerset columns");

    let logits = logits_3d.index_axis(ndarray::Axis(0), 0).to_owned();

    let pm = PowersetMapping::new(3, 2);
    let decoded = pm.soft_decode(&logits);

    assert_eq!(decoded.ncols(), 3);
    assert_eq!(decoded.nrows(), logits.nrows());
    // soft_decode exponentiates log-probs, so values can exceed 1.0
    for &val in decoded.iter() {
        assert!(
            val >= 0.0,
            "soft_decode values should be non-negative, got {val}"
        );
    }
}

#[test]
fn test_binarize_on_soft_decoded() {
    // use first chunk from the pipeline segmentation data
    let seg: Array3<f32> =
        Array3::read_npy(fs::File::open(fixture_path("pipeline_segmentation_data.npy")).unwrap())
            .unwrap();
    let first_chunk = seg.index_axis(ndarray::Axis(0), 0).to_owned();

    let config = BinarizeConfig::default();
    let binary = binarize(&first_chunk, &config);

    assert_eq!(binary.dim(), first_chunk.dim());
    for &val in binary.iter() {
        assert!(
            val == 0.0 || val == 1.0,
            "binarize output should be 0.0 or 1.0, got {val}"
        );
    }
}

fn seg_model_path() -> PathBuf {
    fixture_path("models/segmentation-3.0.onnx")
}

fn emb_model_path() -> PathBuf {
    fixture_path("models/wespeaker-voxceleb-resnet34.onnx")
}

fn run_diarization(
    seg_model: &mut SegmentationModel,
    emb_model: &mut EmbeddingModel,
    audio: &[f32],
) -> String {
    let pm = PowersetMapping::new(3, 2);
    let step_frames = 147;
    let warmup_frames = 0;

    // segmentation
    let windows = seg_model.run(audio).expect("segmentation failed");
    if windows.is_empty() {
        return String::new();
    }

    // powerset decode each window
    let decoded_windows: Vec<Array2<f32>> = windows.iter().map(|w| pm.soft_decode(w)).collect();

    let frames_per_window = decoded_windows[0].nrows();
    let num_speakers = decoded_windows[0].ncols();
    let mut windows_3d =
        Array3::<f32>::zeros((decoded_windows.len(), frames_per_window, num_speakers));
    for (i, w) in decoded_windows.iter().enumerate() {
        windows_3d.slice_mut(s![i, .., ..]).assign(w);
    }

    // aggregate
    let aggregated = overlap_add(&windows_3d, step_frames, warmup_frames);

    // binarize
    let config = BinarizeConfig::default();
    let binary = binarize(&aggregated, &config);

    let speaker_counts = count_speakers(&binary);

    // extract embeddings per active speaker region
    let sample_rate = seg_model.sample_rate();
    let frame_duration = seg_model.step_samples() as f64 / sample_rate as f64 / step_frames as f64;

    // simplified embedding: one embedding per local speaker across all frames
    let mut embeddings = Vec::new();
    for speaker_idx in 0..num_speakers {
        let active_frames: Vec<usize> = (0..binary.nrows())
            .filter(|&f| binary[[f, speaker_idx]] == 1.0)
            .collect();

        if active_frames.is_empty() {
            embeddings.push(ndarray::Array1::zeros(256));
            continue;
        }

        // take a representative segment from the middle of the active region
        let mid = active_frames[active_frames.len() / 2];
        let start_sample = (mid as f64 * frame_duration * sample_rate as f64) as usize;
        let end_sample = (start_sample + sample_rate).min(audio.len());

        if end_sample <= start_sample || end_sample - start_sample < sample_rate / 4 {
            embeddings.push(ndarray::Array1::zeros(256));
            continue;
        }

        let emb = emb_model
            .embed(&audio[start_sample..end_sample])
            .expect("embedding failed");
        embeddings.push(emb);
    }

    // cluster embeddings
    let emb_dim = 256;
    let mut emb_matrix = Array2::<f32>::zeros((num_speakers, emb_dim));
    for (i, emb) in embeddings.iter().enumerate() {
        emb_matrix.row_mut(i).assign(emb);
    }

    let vbx_config = VbxConfig {
        fa: 0.07,
        fb: 0.8,
        ..Default::default()
    };
    let cluster_result = cluster(&emb_matrix.view(), num_speakers, None, &vbx_config);

    // reconstruct
    let reconstructed = reconstruct(
        &aggregated,
        &cluster_result.labels,
        cluster_result.num_clusters,
        Some(&speaker_counts),
    );

    // binarize the reconstructed output
    let final_binary = binarize(&reconstructed, &config);

    let segments = to_segments(&final_binary, frame_duration);
    let merged = merge_segments(&segments, 0.5);
    to_rttm(&merged, "file1")
}

#[test]
fn test_full_pipeline_with_models() {
    let mut seg_model =
        SegmentationModel::new(seg_model_path().to_str().unwrap(), 2.5).expect("load seg model");
    let mut emb_model =
        EmbeddingModel::new(emb_model_path().to_str().unwrap()).expect("load emb model");

    let (samples, sr) = load_wav_samples(&fixture_path("test.wav"));
    assert_eq!(sr, 16000);

    let rttm = run_diarization(&mut seg_model, &mut emb_model, &samples);
    let parsed = parse_rttm(&rttm);

    assert!(!parsed.is_empty(), "should produce RTTM segments");

    let speakers = unique_speakers(&parsed);
    assert!(
        speakers.len() >= 1 && speakers.len() <= 4,
        "expected 1-4 speakers, got {}",
        speakers.len()
    );

    let total_speech: f64 = parsed.iter().map(|(_, d, _)| d).sum();
    let audio_duration = samples.len() as f64 / sr as f64;
    assert!(
        total_speech > audio_duration * 0.1,
        "at least 10% of audio should be speech"
    );
}

fn load_wav_samples(path: &PathBuf) -> (Vec<f32>, u32) {
    let data = fs::read(path).unwrap();

    // parse WAV header (PCM 16-bit mono)
    let sample_rate = u32::from_le_bytes(data[24..28].try_into().unwrap());
    let bits_per_sample = u16::from_le_bytes(data[34..36].try_into().unwrap());
    assert_eq!(bits_per_sample, 16);

    // find data chunk
    let mut pos = 12;
    while pos + 8 < data.len() {
        let chunk_id = &data[pos..pos + 4];
        let chunk_size = u32::from_le_bytes(data[pos + 4..pos + 8].try_into().unwrap()) as usize;
        if chunk_id == b"data" {
            let samples: Vec<f32> = data[pos + 8..pos + 8 + chunk_size]
                .chunks_exact(2)
                .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
                .collect();
            return (samples, sample_rate);
        }
        pos += 8 + chunk_size;
    }
    panic!("no data chunk found in WAV");
}

fn unique_speakers(rttm: &[(f64, f64, String)]) -> HashSet<String> {
    rttm.iter().map(|(_, _, s)| s.clone()).collect()
}

// ── Audio file loading tests ──

#[test]
fn test_3speaker_audio_loads() {
    let (samples, sr) = load_wav_samples(&fixture_path("test_3speakers.wav"));
    assert_eq!(sr, 16000);
    assert!(
        samples.len() > sr as usize * 10,
        "3-speaker audio should be > 10s"
    );
    assert!(
        samples.iter().any(|&s| s.abs() > 0.01),
        "audio should not be silent"
    );
}

#[test]
fn test_single_speaker_audio_loads() {
    let (samples, sr) = load_wav_samples(&fixture_path("test_single_speaker.wav"));
    assert_eq!(sr, 16000);
    assert!(samples.len() > sr as usize * 5, "monologue should be > 5s");
}

#[test]
fn test_short_clip_audio_loads() {
    let (samples, sr) = load_wav_samples(&fixture_path("test_short.wav"));
    assert_eq!(sr, 16000);
    let duration = samples.len() as f64 / sr as f64;
    assert!(
        duration < 10.0,
        "short clip should be < 10s, got {duration:.1}s"
    );
    assert!(duration > 1.0, "short clip should be > 1s");
}

// ── Post-processing pipeline with synthetic data for each scenario ──

#[test]
fn test_post_processing_two_speakers() {
    let num_speakers = 3;
    let step_frames = 147;

    // load real segmentation data as soft-decoded windows
    let seg: Array3<f32> =
        Array3::read_npy(fs::File::open(fixture_path("pipeline_segmentation_data.npy")).unwrap())
            .unwrap();

    let aggregated = overlap_add(&seg, step_frames, 0);
    assert!(aggregated.nrows() > 0);
    assert_eq!(aggregated.ncols(), num_speakers);

    let config = BinarizeConfig::default();
    let binary = binarize(&aggregated, &config);

    let speaker_counts = count_speakers(&binary);
    assert_eq!(speaker_counts.len(), aggregated.nrows());

    let segments = to_segments(&binary, 0.016875);
    assert!(
        !segments.is_empty(),
        "should produce segments from real data"
    );

    let merged = merge_segments(&segments, 0.1);
    assert!(!merged.is_empty());

    let rttm = to_rttm(&merged, "test");
    assert!(rttm.contains("SPEAKER test 1"));
}

#[test]
fn test_post_processing_three_speakers_synthetic() {
    // simulate 3 speakers with clear non-overlapping regions
    let frames = 600;
    let num_speakers = 3;
    let mut activations = Array2::<f32>::zeros((frames, num_speakers));

    // speaker 0: frames 0-199
    activations.slice_mut(s![0..200, 0]).fill(1.0);
    // speaker 1: frames 200-399
    activations.slice_mut(s![200..400, 1]).fill(1.0);
    // speaker 2: frames 400-599
    activations.slice_mut(s![400..600, 2]).fill(1.0);

    let frame_duration = 0.016875;
    let segments = to_segments(&activations, frame_duration);
    assert_eq!(
        segments.len(),
        3,
        "should produce 3 segments for 3 speakers"
    );

    let speakers: HashSet<_> = segments.iter().map(|s| s.speaker.clone()).collect();
    assert_eq!(speakers.len(), 3, "should have 3 distinct speaker labels");

    let rttm = to_rttm(&segments, "3speakers");
    let parsed = parse_rttm(&rttm);
    assert_eq!(parsed.len(), 3);

    // verify chronological ordering
    for pair in parsed.windows(2) {
        assert!(
            pair[0].0 <= pair[1].0,
            "segments should be sorted by start time"
        );
    }
}

#[test]
fn test_post_processing_single_speaker_synthetic() {
    // single speaker active across entire duration
    let frames = 500;
    let mut activations = Array2::<f32>::zeros((frames, 1));
    activations.fill(1.0);

    let frame_duration = 0.016875;
    let segments = to_segments(&activations, frame_duration);
    assert_eq!(segments.len(), 1, "single speaker should produce 1 segment");
    assert_eq!(segments[0].speaker, "SPEAKER_00");

    let expected_duration = frames as f64 * frame_duration;
    assert!(
        (segments[0].duration() - expected_duration).abs() < 0.001,
        "segment duration should match total frames"
    );
}

#[test]
fn test_post_processing_short_clip_synthetic() {
    // very few frames — tests edge cases in aggregation/binarization
    let frames = 50;
    let num_speakers = 2;
    let mut probs = Array2::<f32>::zeros((frames, num_speakers));

    probs.slice_mut(s![0..25, 0]).fill(0.9);
    probs.slice_mut(s![25..50, 1]).fill(0.9);

    let config = BinarizeConfig::default();
    let binary = binarize(&probs, &config);

    let segments = to_segments(&binary, 0.016875);
    assert_eq!(segments.len(), 2);

    let merged = merge_segments(&segments, 0.0);
    assert_eq!(merged.len(), 2, "no merging with zero gap tolerance");
}

#[test]
fn test_vbx_clustering_three_speakers() {
    // 3 well-separated clusters in 4D
    let mut features_vec = Vec::new();
    let mut init_labels = Vec::new();

    for _ in 0..10 {
        features_vec.extend_from_slice(&[10.0, 0.0, 0.0, 0.0]);
        init_labels.push(0);
    }
    for _ in 0..10 {
        features_vec.extend_from_slice(&[0.0, 10.0, 0.0, 0.0]);
        init_labels.push(1);
    }
    for _ in 0..10 {
        features_vec.extend_from_slice(&[0.0, 0.0, 10.0, 0.0]);
        init_labels.push(2);
    }

    let features = Array2::from_shape_vec((30, 4), features_vec).unwrap();
    let config = VbxConfig::default();
    let result = cluster(&features.view(), 3, Some(&init_labels), &config);

    assert_eq!(result.num_clusters, 3);

    // verify each group maps to same label
    assert!(result.labels[0..10].iter().all(|&l| l == result.labels[0]));
    assert!(
        result.labels[10..20]
            .iter()
            .all(|&l| l == result.labels[10])
    );
    assert!(
        result.labels[20..30]
            .iter()
            .all(|&l| l == result.labels[20])
    );

    // verify groups are distinct
    let unique: HashSet<_> = [result.labels[0], result.labels[10], result.labels[20]]
        .into_iter()
        .collect();
    assert_eq!(unique.len(), 3);
}

#[test]
fn test_reconstruct_three_speakers() {
    let frames = 100;
    let local_speakers = 3;
    let global_speakers = 3;

    let mut scores = Array2::<f32>::zeros((frames, local_speakers));
    scores.slice_mut(s![0..30, 0]).fill(0.9);
    scores.slice_mut(s![30..60, 1]).fill(0.8);
    scores.slice_mut(s![60..100, 2]).fill(0.95);

    // each local speaker maps to a different global speaker
    let labels = vec![2, 0, 1];
    let result = reconstruct(&scores, &labels, global_speakers, None);

    assert_eq!(result.dim(), (frames, global_speakers));

    // verify the remapping: local 0 → global 2, local 1 → global 0, local 2 → global 1
    assert!(result[[15, 2]] > 0.5);
    assert!(result[[45, 0]] > 0.5);
    assert!(result[[80, 1]] > 0.5);
}

#[test]
fn test_make_exclusive_with_overlap() {
    let frames = 50;
    let speakers = 3;
    let mut activations = Array2::<f32>::zeros((frames, speakers));

    // overlap region: speakers 0 and 1 both active
    activations.slice_mut(s![0..25, 0]).fill(0.8);
    activations.slice_mut(s![10..40, 1]).fill(0.9);
    activations.slice_mut(s![35..50, 2]).fill(0.7);

    make_exclusive(&mut activations);

    // in overlap region (10..25), speaker 1 should win (0.9 > 0.8)
    for f in 10..25 {
        assert_eq!(
            activations[[f, 0]],
            0.0,
            "speaker 0 should be zeroed in overlap"
        );
        assert!(activations[[f, 1]] > 0.0, "speaker 1 should win in overlap");
    }
}

#[test]
fn test_full_pipeline_post_processing_on_fixture() {
    // runs the complete post-processing chain on the real pipeline fixture data
    let seg: Array3<f32> =
        Array3::read_npy(fs::File::open(fixture_path("pipeline_segmentation_data.npy")).unwrap())
            .unwrap();

    let step_frames = 147;
    let aggregated = overlap_add(&seg, step_frames, 0);

    let config = BinarizeConfig {
        onset: 0.5,
        offset: 0.5,
        min_duration_on: 0,
        min_duration_off: 0,
        pad_onset: 0,
        pad_offset: 0,
    };
    let binary = binarize(&aggregated, &config);

    let speaker_counts = count_speakers(&binary);

    // simulate cluster labels (map local speakers to global)
    let cluster_labels = vec![0, 1, 0];
    let num_global = 2;
    let _reconstructed = reconstruct(
        &aggregated,
        &cluster_labels,
        num_global,
        Some(&speaker_counts),
    );

    let frame_duration = 0.016875;
    let segments = to_segments(&binary, frame_duration);
    let merged = merge_segments(&segments, 0.5);

    let rttm = to_rttm(&merged, "test_file");
    let parsed = parse_rttm(&rttm);

    assert!(!parsed.is_empty(), "pipeline should produce RTTM output");

    let speakers = unique_speakers(&parsed);
    assert!(
        speakers.len() <= 3,
        "3-speaker model shouldn't produce more than 3 speakers"
    );

    // verify total speech duration is reasonable (audio is ~26s)
    let total_speech: f64 = parsed.iter().map(|(_, d, _)| d).sum();
    assert!(total_speech > 1.0, "should have some speech");
    assert!(total_speech < 100.0, "total speech should be reasonable");
}

// ── Full pipeline tests with ONNX models ──

#[test]
fn test_full_pipeline_3speakers_with_models() {
    let mut seg_model =
        SegmentationModel::new(seg_model_path().to_str().unwrap(), 2.5).expect("load seg model");
    let mut emb_model =
        EmbeddingModel::new(emb_model_path().to_str().unwrap()).expect("load emb model");

    let (samples, sr) = load_wav_samples(&fixture_path("test_3speakers.wav"));
    assert_eq!(sr, 16000);

    let rttm = run_diarization(&mut seg_model, &mut emb_model, &samples);
    let parsed = parse_rttm(&rttm);

    assert!(!parsed.is_empty(), "should produce RTTM segments");

    let speakers = unique_speakers(&parsed);
    assert!(
        speakers.len() >= 2,
        "3-speaker audio should detect at least 2 speakers, got {}",
        speakers.len()
    );
}

#[test]
fn test_full_pipeline_single_speaker_with_models() {
    let mut seg_model =
        SegmentationModel::new(seg_model_path().to_str().unwrap(), 2.5).expect("load seg model");
    let mut emb_model =
        EmbeddingModel::new(emb_model_path().to_str().unwrap()).expect("load emb model");

    let (samples, sr) = load_wav_samples(&fixture_path("test_single_speaker.wav"));
    assert_eq!(sr, 16000);

    let rttm = run_diarization(&mut seg_model, &mut emb_model, &samples);
    let parsed = parse_rttm(&rttm);

    // single speaker monologue — should produce segments
    assert!(!parsed.is_empty(), "should produce RTTM segments");

    let speakers = unique_speakers(&parsed);
    assert!(
        speakers.len() <= 2,
        "single speaker audio shouldn't detect more than 2 speakers, got {}",
        speakers.len()
    );
}

#[test]
fn test_full_pipeline_short_clip_with_models() {
    let mut seg_model =
        SegmentationModel::new(seg_model_path().to_str().unwrap(), 2.5).expect("load seg model");
    let mut emb_model =
        EmbeddingModel::new(emb_model_path().to_str().unwrap()).expect("load emb model");

    let (samples, sr) = load_wav_samples(&fixture_path("test_short.wav"));
    assert_eq!(sr, 16000);

    let duration = samples.len() as f64 / sr as f64;
    assert!(
        duration < 10.0,
        "short clip should be < one segmentation window"
    );

    let rttm = run_diarization(&mut seg_model, &mut emb_model, &samples);

    // short clip may or may not produce segments depending on padding
    // main check: it shouldn't panic or error
    if !rttm.is_empty() {
        let parsed = parse_rttm(&rttm);
        for (start, dur, _) in &parsed {
            assert!(*start >= 0.0);
            assert!(*dur > 0.0);
        }
    }
}
