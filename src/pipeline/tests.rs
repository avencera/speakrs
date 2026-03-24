use ndarray::{Array1, Array2, Array3, array, s};
use ndarray_npy::ReadNpyExt;
use std::fs::File;
use std::path::{Path, PathBuf};

use super::*;
#[cfg(feature = "coreml")]
use crate::inference::ExecutionMode;
use crate::inference::embedding::SplitTailInput;

// --- test helpers ---

#[allow(dead_code)]
fn decode_windows(raw_windows: Vec<Array2<f32>>, powerset: &PowersetMapping) -> Array3<f32> {
    RawSegmentationWindows(raw_windows).decode(powerset).0
}

fn extract_embeddings(
    seg_model: &SegmentationModel,
    emb_model: &mut EmbeddingModel,
    audio: &[f32],
    segmentations: &Array3<f32>,
) -> Result<Array3<f32>, PipelineError> {
    let decoded_segmentations = DecodedSegmentations(segmentations.clone());
    let layout = ChunkLayout::new(
        seg_model.step_seconds(),
        seg_model.step_samples(),
        seg_model.window_samples(),
        decoded_segmentations.nchunks(),
    );
    let embedding_path = if emb_model.prefers_multi_mask_path()
        && emb_model.multi_mask_batch_size() > 0
    {
        EmbeddingPath::MultiMask
    } else if emb_model.prefers_chunk_embedding_path() && emb_model.split_primary_batch_size() > 0 {
        EmbeddingPath::Split
    } else {
        EmbeddingPath::Masked
    };
    decoded_segmentations
        .extract_embeddings(audio, emb_model, &layout, embedding_path)
        .map(|chunk_embeddings| chunk_embeddings.0)
}

fn filter_embeddings(
    segmentations: &Array3<f32>,
    embeddings: &Array3<f32>,
) -> (Array2<f32>, Vec<usize>, Vec<usize>) {
    let num_frames = segmentations.shape()[1] as f32;
    let mut filtered = Vec::new();
    let mut chunk_indices = Vec::new();
    let mut speaker_indices = Vec::new();

    for chunk_idx in 0..segmentations.shape()[0] {
        let single_active: Vec<bool> = segmentations
            .slice(s![chunk_idx, .., ..])
            .rows()
            .into_iter()
            .map(|row| (row.iter().copied().sum::<f32>() - 1.0).abs() < 1e-6)
            .collect();
        for speaker_idx in 0..segmentations.shape()[2] {
            let clean_frames = segmentations
                .slice(s![chunk_idx, .., speaker_idx])
                .iter()
                .zip(single_active.iter())
                .filter_map(|(value, is_single_active)| is_single_active.then_some(*value))
                .sum::<f32>();
            let embedding = embeddings.slice(s![chunk_idx, speaker_idx, ..]);
            let valid_embedding = embedding.iter().all(|value| value.is_finite());
            if valid_embedding && clean_frames >= 0.2 * num_frames {
                filtered.extend(embedding.iter());
                chunk_indices.push(chunk_idx);
                speaker_indices.push(speaker_idx);
            }
        }
    }

    let filtered_embeddings =
        Array2::from_shape_vec((chunk_indices.len(), embeddings.shape()[2]), filtered).unwrap();
    (filtered_embeddings, chunk_indices, speaker_indices)
}

#[allow(dead_code)]
fn chunk_audio<'a>(audio: &'a [f32], seg_model: &SegmentationModel, chunk_idx: usize) -> &'a [f32] {
    chunk_audio_raw(
        audio,
        seg_model.step_samples(),
        seg_model.window_samples(),
        chunk_idx,
    )
}

fn assign_embeddings(
    segmentations: &Array3<f32>,
    embeddings: &Array3<f32>,
    centroids: &Array2<f32>,
) -> Array2<i32> {
    super::clustering::assign_chunk_embeddings(
        &DecodedSegmentations(segmentations.clone()),
        &ChunkEmbeddings(embeddings.clone()),
        centroids,
    )
}

fn weighted_centroids(
    train_embeddings: &Array2<f32>,
    gamma: &Array2<f32>,
    kept_speakers: &[usize],
) -> Array2<f32> {
    super::clustering::weighted_centroids(train_embeddings, gamma, kept_speakers)
}

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join(name)
}

fn load_wav_samples(path: &Path) -> (Vec<f32>, u32) {
    let data = std::fs::read(path).unwrap();
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

// --- tests ---

#[test]
fn chunk_start_frames_match_pyannote_rounding() {
    assert_eq!(
        chunk_start_frames(4, SEGMENTATION_STEP_SECONDS),
        vec![0, 59, 119, 178]
    );
}

#[test]
fn total_output_frames_match_pyannote_aggregate_extent() {
    assert_eq!(total_output_frames(4, SEGMENTATION_STEP_SECONDS), 771);
}

#[test]
fn best_assignment_handles_more_speakers_than_clusters() {
    let scores = array![[0.9, 0.1], [0.8, 0.2], [0.1, 0.95]];
    let assignment = super::clustering::best_assignment(&scores, &[0, 1, 2], 2);
    assert_eq!(assignment.len(), 2);
    assert!(assignment.contains(&(0, 0)) || assignment.contains(&(1, 0)));
    assert!(assignment.contains(&(2, 1)));
}

#[test]
fn filter_embeddings_matches_python_fixture() {
    let segmentations: Array3<f32> =
        Array3::read_npy(File::open(fixture_path("pipeline_segmentation_data.npy")).unwrap())
            .unwrap();
    let embeddings: Array3<f32> =
        Array3::read_npy(File::open(fixture_path("pipeline_embeddings_data.npy")).unwrap())
            .unwrap();
    let expected_train_embeddings: Array2<f32> =
        Array2::read_npy(File::open(fixture_path("pipeline_train_embeddings.npy")).unwrap())
            .unwrap();
    let expected_chunk_idx: Array1<i64> =
        Array1::read_npy(File::open(fixture_path("pipeline_train_chunk_idx.npy")).unwrap())
            .unwrap();
    let expected_speaker_idx: Array1<i64> =
        Array1::read_npy(File::open(fixture_path("pipeline_train_speaker_idx.npy")).unwrap())
            .unwrap();

    let (train_embeddings, chunk_idx, speaker_idx) = filter_embeddings(&segmentations, &embeddings);

    assert_eq!(chunk_idx.len(), expected_chunk_idx.len());
    assert_eq!(speaker_idx.len(), expected_speaker_idx.len());
    for (lhs, rhs) in chunk_idx.iter().zip(expected_chunk_idx.iter()) {
        assert_eq!(*lhs as i64, *rhs);
    }
    for (lhs, rhs) in speaker_idx.iter().zip(expected_speaker_idx.iter()) {
        assert_eq!(*lhs as i64, *rhs);
    }
    for (lhs, rhs) in train_embeddings
        .iter()
        .zip(expected_train_embeddings.iter())
    {
        approx::assert_abs_diff_eq!(*lhs, *rhs, epsilon = 1e-5);
    }
}

#[test]
fn assign_embeddings_matches_python_fixture() {
    let segmentations: Array3<f32> =
        Array3::read_npy(File::open(fixture_path("pipeline_segmentation_data.npy")).unwrap())
            .unwrap();
    let embeddings: Array3<f32> =
        Array3::read_npy(File::open(fixture_path("pipeline_embeddings_data.npy")).unwrap())
            .unwrap();
    let train_embeddings: Array2<f32> =
        Array2::read_npy(File::open(fixture_path("pipeline_train_embeddings.npy")).unwrap())
            .unwrap();
    let gamma: Array2<f64> =
        Array2::read_npy(File::open(fixture_path("pipeline_vbx_gamma.npy")).unwrap()).unwrap();
    let pi: Array1<f64> =
        Array1::read_npy(File::open(fixture_path("pipeline_vbx_pi.npy")).unwrap()).unwrap();
    let expected: Array2<i8> =
        Array2::read_npy(File::open(fixture_path("pipeline_hard_clusters.npy")).unwrap()).unwrap();

    let kept_speakers: Vec<usize> = pi
        .iter()
        .enumerate()
        .filter_map(|(idx, weight)| (*weight > 1e-7).then_some(idx))
        .collect();
    let centroids = weighted_centroids(
        &train_embeddings,
        &gamma.mapv(|value| value as f32),
        &kept_speakers,
    );
    let mut hard_clusters = assign_embeddings(&segmentations, &embeddings, &centroids);
    mark_inactive_speakers(&segmentations, &mut hard_clusters);

    assert_eq!(hard_clusters.dim(), expected.dim());
    for (lhs, rhs) in hard_clusters.iter().zip(expected.iter()) {
        assert_eq!(*lhs as i8, *rhs);
    }
}

#[test]
fn extract_embeddings_matches_python_fixture() {
    let models_dir = fixture_path("models");
    let seg_model = SegmentationModel::new(
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
    let segmentations: Array3<f32> =
        Array3::read_npy(File::open(fixture_path("pipeline_segmentation_data.npy")).unwrap())
            .unwrap();
    let expected: Array3<f32> =
        Array3::read_npy(File::open(fixture_path("pipeline_embeddings_data.npy")).unwrap())
            .unwrap();
    let (audio, sample_rate) = load_wav_samples(&fixture_path("test.wav"));
    assert_eq!(sample_rate, 16_000);

    let embeddings =
        extract_embeddings(&seg_model, &mut emb_model, &audio, &segmentations).unwrap();

    for chunk_idx in 0..embeddings.shape()[0] {
        for speaker_idx in 0..embeddings.shape()[1] {
            for dim_idx in 0..embeddings.shape()[2] {
                let lhs = embeddings[[chunk_idx, speaker_idx, dim_idx]];
                let rhs = expected[[chunk_idx, speaker_idx, dim_idx]];
                if (lhs - rhs).abs() > 5e-3 || lhs.is_nan() != rhs.is_nan() {
                    panic!(
                        "chunk={chunk_idx} speaker={speaker_idx} dim={dim_idx} left={lhs} right={rhs}"
                    );
                }
            }
        }
    }
}

#[cfg(feature = "coreml")]
#[test]
fn fast_apple_segmentation_matches_python_fixture() {
    let models_dir = fixture_path("models");
    let mut seg_model = SegmentationModel::with_mode(
        models_dir.join("segmentation-3.0.onnx").to_str().unwrap(),
        SEGMENTATION_STEP_SECONDS as f32,
        ExecutionMode::CoreMl,
    )
    .unwrap();
    let expected: Array3<f32> =
        Array3::read_npy(File::open(fixture_path("pipeline_segmentation_data.npy")).unwrap())
            .unwrap();
    let powerset = PowersetMapping::new(3, 2);
    let (audio, sample_rate) = load_wav_samples(&fixture_path("test.wav"));
    assert_eq!(sample_rate, 16_000);

    let raw_windows = seg_model.run(&audio).unwrap();
    let segmentations = decode_windows(raw_windows, &powerset);

    for chunk_idx in 0..segmentations.shape()[0] {
        for frame_idx in 0..segmentations.shape()[1] {
            for speaker_idx in 0..segmentations.shape()[2] {
                let lhs = segmentations[[chunk_idx, frame_idx, speaker_idx]];
                let rhs = expected[[chunk_idx, frame_idx, speaker_idx]];
                if lhs != rhs {
                    panic!(
                        "chunk={chunk_idx} frame={frame_idx} speaker={speaker_idx} left={lhs} right={rhs}"
                    );
                }
            }
        }
    }
}

#[cfg(feature = "coreml")]
#[test]
fn fast_apple_embeddings_match_python_fixture() {
    let models_dir = fixture_path("models");
    let seg_model = SegmentationModel::new(
        models_dir.join("segmentation-3.0.onnx").to_str().unwrap(),
        SEGMENTATION_STEP_SECONDS as f32,
    )
    .unwrap();
    let mut emb_model = EmbeddingModel::with_mode(
        models_dir
            .join("wespeaker-voxceleb-resnet34.onnx")
            .to_str()
            .unwrap(),
        ExecutionMode::CoreMl,
    )
    .unwrap();
    let segmentations: Array3<f32> =
        Array3::read_npy(File::open(fixture_path("pipeline_segmentation_data.npy")).unwrap())
            .unwrap();
    let expected: Array3<f32> =
        Array3::read_npy(File::open(fixture_path("pipeline_embeddings_data.npy")).unwrap())
            .unwrap();
    let (audio, sample_rate) = load_wav_samples(&fixture_path("test.wav"));
    assert_eq!(sample_rate, 16_000);

    let embeddings =
        extract_embeddings(&seg_model, &mut emb_model, &audio, &segmentations).unwrap();

    for chunk_idx in 0..embeddings.shape()[0] {
        for speaker_idx in 0..embeddings.shape()[1] {
            for dim_idx in 0..embeddings.shape()[2] {
                let lhs = embeddings[[chunk_idx, speaker_idx, dim_idx]];
                let rhs = expected[[chunk_idx, speaker_idx, dim_idx]];
                if (lhs - rhs).abs() > 5e-3 || lhs.is_nan() != rhs.is_nan() {
                    panic!(
                        "chunk={chunk_idx} speaker={speaker_idx} dim={dim_idx} left={lhs} right={rhs}"
                    );
                }
            }
        }
    }
}

#[cfg(feature = "coreml")]
#[test]
fn fast_apple_split_primary_batch_matches_single_tail_path() {
    let models_dir = fixture_path("models");
    let seg_model = SegmentationModel::new(
        models_dir.join("segmentation-3.0.onnx").to_str().unwrap(),
        SEGMENTATION_STEP_SECONDS as f32,
    )
    .unwrap();
    let mut emb_model = EmbeddingModel::with_mode(
        models_dir
            .join("wespeaker-voxceleb-resnet34.onnx")
            .to_str()
            .unwrap(),
        ExecutionMode::CoreMl,
    )
    .unwrap();
    let segmentations: Array3<f32> =
        Array3::read_npy(File::open(fixture_path("pipeline_segmentation_data.npy")).unwrap())
            .unwrap();
    let (audio, sample_rate) = load_wav_samples(&fixture_path("test.wav"));
    assert_eq!(sample_rate, 16_000);

    let mut fbanks = Vec::new();
    let mut weights = Vec::new();
    let mut expected = Vec::new();

    'outer: for chunk_idx in 0..segmentations.shape()[0] {
        let chunk_audio = chunk_audio(&audio, &seg_model, chunk_idx);
        let chunk_segmentations = segmentations.slice(s![chunk_idx, .., ..]);
        let clean_masks = clean_masks(&chunk_segmentations);
        let fbank = emb_model.compute_chunk_fbank(chunk_audio).unwrap();

        for speaker_idx in 0..chunk_segmentations.ncols() {
            let mask = chunk_segmentations.column(speaker_idx).to_owned();
            let clean_mask = clean_masks.column(speaker_idx).to_owned();
            let used_mask = emb_model
                .select_chunk_mask(
                    mask.as_slice().unwrap(),
                    Some(clean_mask.as_slice().unwrap()),
                    chunk_audio.len(),
                )
                .to_vec();
            expected.push(
                emb_model
                    .embed_masked(
                        chunk_audio,
                        mask.as_slice().unwrap(),
                        Some(clean_mask.as_slice().unwrap()),
                    )
                    .unwrap(),
            );
            fbanks.push(fbank.clone());
            weights.push(used_mask);
            if fbanks.len() == emb_model.split_primary_batch_size() {
                break 'outer;
            }
        }
    }

    assert_eq!(fbanks.len(), emb_model.split_primary_batch_size());
    let batch_inputs: Vec<_> = fbanks
        .iter()
        .zip(weights.iter())
        .map(|(fbank, weights)| SplitTailInput {
            fbank,
            weights: weights.as_slice(),
        })
        .collect();
    let batched = emb_model.embed_tail_batch_inputs(&batch_inputs).unwrap();

    for (row_idx, expected_row) in expected.iter().enumerate() {
        for dim_idx in 0..expected_row.len() {
            let lhs = batched[[row_idx, dim_idx]];
            let rhs = expected_row[dim_idx];
            if (lhs - rhs).abs() > 5e-3 || lhs.is_nan() != rhs.is_nan() {
                panic!("row={row_idx} dim={dim_idx} left={lhs} right={rhs}");
            }
        }
    }
}

#[cfg(feature = "coreml")]
#[test]
fn fast_apple_single_embedding_matches_python_fixture() {
    let models_dir = fixture_path("models");
    let seg_model = SegmentationModel::new(
        models_dir.join("segmentation-3.0.onnx").to_str().unwrap(),
        SEGMENTATION_STEP_SECONDS as f32,
    )
    .unwrap();
    let mut emb_model = EmbeddingModel::with_mode(
        models_dir
            .join("wespeaker-voxceleb-resnet34.onnx")
            .to_str()
            .unwrap(),
        ExecutionMode::CoreMl,
    )
    .unwrap();
    let segmentations: Array3<f32> =
        Array3::read_npy(File::open(fixture_path("pipeline_segmentation_data.npy")).unwrap())
            .unwrap();
    let expected: Array3<f32> =
        Array3::read_npy(File::open(fixture_path("pipeline_embeddings_data.npy")).unwrap())
            .unwrap();
    let (audio, sample_rate) = load_wav_samples(&fixture_path("test.wav"));
    assert_eq!(sample_rate, 16_000);

    let chunk_idx = 0;
    let speaker_idx = 1;
    let chunk_segmentations = segmentations.slice(s![chunk_idx, .., ..]);
    let clean = clean_masks(&chunk_segmentations);
    let mask = chunk_segmentations.column(speaker_idx).to_vec();
    let clean_mask = clean.column(speaker_idx).to_vec();
    let embedding = emb_model
        .embed_masked(
            chunk_audio(&audio, &seg_model, chunk_idx),
            &mask,
            Some(&clean_mask),
        )
        .unwrap();

    for dim_idx in 0..embedding.len() {
        let lhs = embedding[dim_idx];
        let rhs = expected[[chunk_idx, speaker_idx, dim_idx]];
        if (lhs - rhs).abs() > 5e-4 || lhs.is_nan() != rhs.is_nan() {
            panic!("dim={dim_idx} left={lhs} right={rhs}");
        }
    }
}

#[test]
fn run_inference_only_plus_finish_matches_run_with_config() {
    let models_dir = fixture_path("models");
    let mut pipeline = OwnedDiarizationPipeline::from_dir(&models_dir, ExecutionMode::Cpu).unwrap();
    let (audio, sample_rate) = load_wav_samples(&fixture_path("test.wav"));
    assert_eq!(sample_rate, 16_000);

    let config = pipeline.pipeline_config();
    let combined = pipeline.run_with_config(&audio, "file1", &config).unwrap();

    let artifacts = pipeline.run_inference_only(&audio).unwrap();
    let split = pipeline
        .finish_post_inference(artifacts, "file1", &config)
        .unwrap();

    assert_eq!(combined.rttm, split.rttm);
}

#[cfg(feature = "coreml")]
#[test]
fn chunk_embedding_pipelined_vs_sequential_baseline() {
    let models_dir = fixture_path("models");
    let mut pipeline =
        OwnedDiarizationPipeline::from_dir(&models_dir, ExecutionMode::CoreMl).unwrap();

    // multi-chunk audio (triggers pipelined path)
    let (audio, sample_rate) = load_wav_samples(&fixture_path("test.wav"));
    assert_eq!(sample_rate, 16_000);

    // run full pipeline twice: chunk embedding path uses try_chunk_embedding
    // which internally picks pipelined vs sequential based on chunk count
    let result_a = pipeline.run(&audio).unwrap();
    let result_b = pipeline.run(&audio).unwrap();

    // both runs should produce identical RTTM
    assert_eq!(result_a.rttm, result_b.rttm);

    // also verify run_inference_only + finish_post_inference round-trips
    let config = pipeline.pipeline_config();
    let artifacts = pipeline.run_inference_only(&audio).unwrap();
    let result_split = pipeline
        .finish_post_inference(artifacts, "file1", &config)
        .unwrap();
    assert_eq!(result_a.rttm, result_split.rttm);
}
