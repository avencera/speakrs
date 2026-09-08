use ndarray::{Array1, Array2, Array3, array, s};
use ndarray_npy::ReadNpyExt;
use std::fs::File;
use std::path::{Path, PathBuf};

use super::*;
#[cfg(feature = "coreml")]
use crate::inference::ExecutionMode;
use crate::inference::{DynamicRuntimeError, ModelLoadError, OrtRuntimeError};

#[cfg(feature = "coreml")]
#[test]
fn coreml_uses_cuda_parity_segmentation_step() {
    assert_eq!(
        segmentation_step_seconds(ExecutionMode::CoreMl),
        segmentation_step_seconds(ExecutionMode::Cuda)
    );
}

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

fn models_dir() -> PathBuf {
    fixture_path("models")
}

fn load_fixture_array1<T>(name: &str) -> Array1<T>
where
    Array1<T>: ReadNpyExt,
{
    Array1::read_npy(File::open(fixture_path(name)).unwrap()).unwrap()
}

fn load_fixture_array2<T>(name: &str) -> Array2<T>
where
    Array2<T>: ReadNpyExt,
{
    Array2::read_npy(File::open(fixture_path(name)).unwrap()).unwrap()
}

fn load_fixture_array3<T>(name: &str) -> Array3<T>
where
    Array3<T>: ReadNpyExt,
{
    Array3::read_npy(File::open(fixture_path(name)).unwrap()).unwrap()
}

fn load_test_audio() -> (Vec<f32>, u32) {
    load_wav_samples(&fixture_path("test.wav"))
}

struct TestAudio {
    samples: Vec<f32>,
    sample_rate: u32,
}

impl TestAudio {
    fn load() -> Self {
        let (samples, sample_rate) = load_test_audio();
        Self {
            samples,
            sample_rate,
        }
    }

    fn samples(&self) -> &[f32] {
        &self.samples
    }

    fn assert_16khz(&self) {
        assert_eq!(self.sample_rate, 16_000);
    }
}

struct PipelineTestHarness {
    models_dir: PathBuf,
    audio: TestAudio,
}

impl PipelineTestHarness {
    fn load() -> Self {
        Self {
            models_dir: models_dir(),
            audio: TestAudio::load(),
        }
    }

    fn audio(&self) -> &[f32] {
        self.audio.assert_16khz();
        self.audio.samples()
    }

    fn models_dir(&self) -> &Path {
        &self.models_dir
    }

    fn segmentation_model_path(&self) -> PathBuf {
        self.models_dir.join("segmentation-3.0.onnx")
    }

    fn embedding_model_path(&self) -> PathBuf {
        self.models_dir.join("wespeaker-voxceleb-resnet34.onnx")
    }

    fn cpu_seg_model(&self) -> Option<SegmentationModel> {
        load_model_or_skip(SegmentationModel::new(
            self.segmentation_model_path(),
            SEGMENTATION_STEP_SECONDS as f32,
        ))
    }

    fn cpu_emb_model(&self) -> Option<EmbeddingModel> {
        load_model_or_skip(EmbeddingModel::new(self.embedding_model_path()))
    }

    fn cpu_pipeline(&self) -> Option<OwnedDiarizationPipeline> {
        build_pipeline_or_skip(
            PipelineBuilder::from_dir(self.models_dir(), ExecutionMode::Cpu).build(),
        )
    }

    #[cfg(feature = "coreml")]
    fn coreml_seg_model(&self) -> Option<SegmentationModel> {
        load_model_or_skip(SegmentationModel::with_mode(
            self.segmentation_model_path(),
            SEGMENTATION_STEP_SECONDS as f32,
            ExecutionMode::CoreMl,
        ))
    }

    #[cfg(feature = "coreml")]
    fn coreml_emb_model(&self) -> Option<EmbeddingModel> {
        load_model_or_skip(EmbeddingModel::with_mode(
            self.embedding_model_path(),
            ExecutionMode::CoreMl,
        ))
    }

    #[cfg(feature = "coreml")]
    fn coreml_pipeline(&self) -> Option<OwnedDiarizationPipeline> {
        build_pipeline_or_skip(
            PipelineBuilder::from_dir(self.models_dir(), ExecutionMode::CoreMl).build(),
        )
    }
}

fn custom_pipeline_config() -> PipelineConfig {
    PipelineConfig {
        merge_gap: 0.75,
        speaker_keep_threshold: 0.25,
        reconstruct_method: ReconstructMethod::Standard,
        ..PipelineConfig::default()
    }
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
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&bytes| i16::from_le_bytes(bytes) as f32 / 32768.0)
                .collect();
            return (samples, sample_rate);
        }
        pos += 8 + chunk_size;
    }

    panic!("no data chunk found in WAV");
}

fn load_model_or_skip<T>(result: Result<T, ModelLoadError>) -> Option<T> {
    match result {
        Ok(value) => Some(value),
        Err(ModelLoadError::Runtime(OrtRuntimeError::Dynamic(DynamicRuntimeError::Missing {
            ..
        }))) if cfg!(feature = "load-dynamic") => {
            eprintln!("skipping model-loading test because ORT_DYLIB_PATH is not configured");
            None
        }
        Err(error) => panic!("failed to load model: {error}"),
    }
}

fn build_pipeline_or_skip<T>(result: Result<T, PipelineError>) -> Option<T> {
    match result {
        Ok(value) => Some(value),
        Err(PipelineError::ModelLoad(ModelLoadError::Runtime(OrtRuntimeError::Dynamic(
            DynamicRuntimeError::Missing { .. },
        )))) if cfg!(feature = "load-dynamic") => {
            eprintln!("skipping pipeline test because ORT_DYLIB_PATH is not configured");
            None
        }
        Err(error) => panic!("failed to build pipeline: {error}"),
    }
}

fn assert_embedding_tensor_close(actual: &Array3<f32>, expected: &Array3<f32>, epsilon: f32) {
    let mut largest_difference = (0.0_f32, 0, 0, 0, 0.0_f32, 0.0_f32);
    for chunk_idx in 0..actual.shape()[0] {
        for speaker_idx in 0..actual.shape()[1] {
            for dim_idx in 0..actual.shape()[2] {
                let lhs = actual[[chunk_idx, speaker_idx, dim_idx]];
                let rhs = expected[[chunk_idx, speaker_idx, dim_idx]];
                assert_eq!(
                    lhs.is_nan(),
                    rhs.is_nan(),
                    "NaN mismatch at chunk={chunk_idx} speaker={speaker_idx} dim={dim_idx} left={lhs} right={rhs}"
                );
                let difference = (lhs - rhs).abs();
                if difference > largest_difference.0 {
                    largest_difference = (difference, chunk_idx, speaker_idx, dim_idx, lhs, rhs);
                }
            }
        }
    }
    assert!(
        largest_difference.0 <= epsilon,
        "largest difference={} at chunk={} speaker={} dim={} left={} right={} exceeded {epsilon}",
        largest_difference.0,
        largest_difference.1,
        largest_difference.2,
        largest_difference.3,
        largest_difference.4,
        largest_difference.5
    );
}

#[cfg(feature = "coreml")]
fn assert_embedding_tensor_similarity(
    actual: &Array3<f32>,
    expected: &Array3<f32>,
    maximum_absolute_difference: f32,
    minimum_cosine: f32,
) {
    let mut largest_difference = 0.0_f32;
    let mut lowest_cosine = 1.0_f32;

    for chunk_idx in 0..actual.shape()[0] {
        for speaker_idx in 0..actual.shape()[1] {
            let actual_row = actual.slice(s![chunk_idx, speaker_idx, ..]);
            let expected_row = expected.slice(s![chunk_idx, speaker_idx, ..]);
            assert_eq!(
                actual_row.iter().all(|value| value.is_nan()),
                expected_row.iter().all(|value| value.is_nan()),
                "NaN row mismatch at chunk={chunk_idx} speaker={speaker_idx}"
            );
            if expected_row.iter().any(|value| value.is_nan()) {
                continue;
            }

            for (&lhs, &rhs) in actual_row.iter().zip(expected_row.iter()) {
                largest_difference = largest_difference.max((lhs - rhs).abs());
            }
            let dot = actual_row
                .iter()
                .zip(expected_row.iter())
                .map(|(lhs, rhs)| lhs * rhs)
                .sum::<f32>();
            let actual_norm = actual_row
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                .sqrt();
            let expected_norm = expected_row
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                .sqrt();
            lowest_cosine = lowest_cosine.min(dot / (actual_norm * expected_norm));
        }
    }

    assert!(
        largest_difference <= maximum_absolute_difference,
        "largest absolute difference {largest_difference} exceeded {maximum_absolute_difference}"
    );
    assert!(
        lowest_cosine >= minimum_cosine,
        "lowest cosine {lowest_cosine} was below {minimum_cosine}"
    );
}

#[cfg(feature = "coreml")]
fn assert_segmentation_tensor_matches(actual: &Array3<f32>, expected: &Array3<f32>) {
    for chunk_idx in 0..actual.shape()[0] {
        for frame_idx in 0..actual.shape()[1] {
            for speaker_idx in 0..actual.shape()[2] {
                let lhs = actual[[chunk_idx, frame_idx, speaker_idx]];
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
    let segmentations: Array3<f32> = load_fixture_array3("pipeline_segmentation_data.npy");
    let embeddings: Array3<f32> = load_fixture_array3("pipeline_embeddings_data.npy");
    let expected_train_embeddings: Array2<f32> =
        load_fixture_array2("pipeline_train_embeddings.npy");
    let expected_chunk_idx: Array1<i64> = load_fixture_array1("pipeline_train_chunk_idx.npy");
    let expected_speaker_idx: Array1<i64> = load_fixture_array1("pipeline_train_speaker_idx.npy");

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
    let segmentations: Array3<f32> = load_fixture_array3("pipeline_segmentation_data.npy");
    let embeddings: Array3<f32> = load_fixture_array3("pipeline_embeddings_data.npy");
    let train_embeddings: Array2<f32> = load_fixture_array2("pipeline_train_embeddings.npy");
    let gamma: Array2<f64> = load_fixture_array2("pipeline_vbx_gamma.npy");
    let pi: Array1<f64> = load_fixture_array1("pipeline_vbx_pi.npy");
    let expected: Array2<i8> = load_fixture_array2("pipeline_hard_clusters.npy");

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
    let harness = PipelineTestHarness::load();
    let Some(seg_model) = harness.cpu_seg_model() else {
        return;
    };
    let Some(mut emb_model) = harness.cpu_emb_model() else {
        return;
    };
    let segmentations: Array3<f32> = load_fixture_array3("pipeline_segmentation_data.npy");
    let expected: Array3<f32> = load_fixture_array3("pipeline_embeddings_data.npy");
    let embeddings =
        extract_embeddings(&seg_model, &mut emb_model, harness.audio(), &segmentations).unwrap();

    assert_embedding_tensor_close(&embeddings, &expected, 5e-3);
}

#[cfg(feature = "coreml")]
#[test]
fn fast_apple_segmentation_matches_python_fixture() {
    let harness = PipelineTestHarness::load();
    let Some(mut seg_model) = harness.coreml_seg_model() else {
        return;
    };
    let expected: Array3<f32> = load_fixture_array3("pipeline_segmentation_data.npy");
    let powerset = PowersetMapping::new(3, 2);
    let raw_windows = seg_model.run(harness.audio()).unwrap();
    let segmentations = decode_windows(raw_windows, &powerset);

    assert_segmentation_tensor_matches(&segmentations, &expected);
}

#[cfg(feature = "coreml")]
#[test]
fn fast_apple_cpu_embeddings_match_python_fixture() {
    let harness = PipelineTestHarness::load();
    let Some(seg_model) = harness.cpu_seg_model() else {
        return;
    };
    let runtime = RuntimeConfig {
        chunk_emb_compute_units: crate::inference::CoreMlComputeUnits::CpuOnly,
        ..RuntimeConfig::default()
    };
    let Some(mut emb_model) = load_model_or_skip(EmbeddingModel::with_mode_and_config(
        harness.embedding_model_path(),
        ExecutionMode::CoreMl,
        &runtime,
    )) else {
        return;
    };
    let segmentations: Array3<f32> = load_fixture_array3("pipeline_segmentation_data.npy");
    let expected: Array3<f32> = load_fixture_array3("pipeline_embeddings_data.npy");
    let embeddings =
        extract_embeddings(&seg_model, &mut emb_model, harness.audio(), &segmentations).unwrap();

    assert_embedding_tensor_close(&embeddings, &expected, 5e-3);
}

#[cfg(feature = "coreml")]
#[test]
fn fast_apple_gpu_embeddings_stay_within_documented_fixture_bounds() {
    let harness = PipelineTestHarness::load();
    let Some(seg_model) = harness.cpu_seg_model() else {
        return;
    };
    let Some(mut emb_model) = harness.coreml_emb_model() else {
        return;
    };
    let segmentations: Array3<f32> = load_fixture_array3("pipeline_segmentation_data.npy");
    let expected: Array3<f32> = load_fixture_array3("pipeline_embeddings_data.npy");
    let embeddings =
        extract_embeddings(&seg_model, &mut emb_model, harness.audio(), &segmentations).unwrap();

    // gpu tail placement is faster but has a stable numerical gap from the CPU reference
    assert_embedding_tensor_similarity(&embeddings, &expected, 0.12, 0.94);
}

#[cfg(feature = "coreml")]
#[test]
fn fast_apple_split_primary_batch_matches_single_tail_path() {
    let harness = PipelineTestHarness::load();
    let Some(seg_model) = harness.cpu_seg_model() else {
        return;
    };
    let Some(mut emb_model) = harness.coreml_emb_model() else {
        return;
    };
    if emb_model.split_primary_batch_size() == 0 {
        eprintln!("skipping primary tail batch test because the compiled batch asset is missing");
        return;
    }
    let segmentations: Array3<f32> = load_fixture_array3("pipeline_segmentation_data.npy");
    let mut fbanks = Vec::new();
    let mut weights = Vec::new();
    let mut expected = Vec::new();

    'outer: for chunk_idx in 0..segmentations.shape()[0] {
        let chunk_audio = chunk_audio(harness.audio(), &seg_model, chunk_idx);
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
        .map(
            |(fbank, weights)| crate::inference::embedding::SplitTailInput {
                fbank,
                weights: weights.as_slice(),
            },
        )
        .collect();
    let batched = emb_model.embed_tail_batch_inputs(&batch_inputs).unwrap();

    let mut largest_difference = (0.0_f32, 0, 0, 0.0_f32, 0.0_f32);
    let mut lowest_cosine = (1.0_f32, 0);
    for (row_idx, expected_row) in expected.iter().enumerate() {
        let actual_row = batched.row(row_idx);
        for dim_idx in 0..expected_row.len() {
            let lhs = batched[[row_idx, dim_idx]];
            let rhs = expected_row[dim_idx];
            assert_eq!(
                lhs.is_nan(),
                rhs.is_nan(),
                "NaN mismatch at row={row_idx} dim={dim_idx} left={lhs} right={rhs}"
            );
            let difference = (lhs - rhs).abs();
            if difference > largest_difference.0 {
                largest_difference = (difference, row_idx, dim_idx, lhs, rhs);
            }
        }

        let dot = actual_row
            .iter()
            .zip(expected_row.iter())
            .map(|(lhs, rhs)| lhs * rhs)
            .sum::<f32>();
        let actual_norm = actual_row
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        let expected_norm = expected_row
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        let cosine = dot / (actual_norm * expected_norm);
        if cosine < lowest_cosine.0 {
            lowest_cosine = (cosine, row_idx);
        }
    }
    assert!(
        largest_difference.0 <= 5e-3,
        "largest difference={} at row={} dim={} left={} right={}; lowest cosine={} at row={}",
        largest_difference.0,
        largest_difference.1,
        largest_difference.2,
        largest_difference.3,
        largest_difference.4,
        lowest_cosine.0,
        lowest_cosine.1
    );
}

#[cfg(feature = "coreml")]
#[test]
fn fast_apple_single_embedding_matches_python_fixture() {
    let harness = PipelineTestHarness::load();
    let Some(seg_model) = harness.cpu_seg_model() else {
        return;
    };
    let Some(mut emb_model) = harness.coreml_emb_model() else {
        return;
    };
    let segmentations: Array3<f32> = load_fixture_array3("pipeline_segmentation_data.npy");
    let expected: Array3<f32> = load_fixture_array3("pipeline_embeddings_data.npy");
    let chunk_idx = 0;
    let speaker_idx = 1;
    let chunk_segmentations = segmentations.slice(s![chunk_idx, .., ..]);
    let clean = clean_masks(&chunk_segmentations);
    let mask = chunk_segmentations.column(speaker_idx).to_vec();
    let clean_mask = clean.column(speaker_idx).to_vec();
    let embedding = emb_model
        .embed_masked(
            chunk_audio(harness.audio(), &seg_model, chunk_idx),
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
    let harness = PipelineTestHarness::load();
    let Some(mut pipeline) = harness.cpu_pipeline() else {
        return;
    };
    let config = pipeline.pipeline_config();
    let combined = pipeline
        .run_with_config(harness.audio(), "file1", &config)
        .unwrap();

    let artifacts = pipeline.run_inference_only(harness.audio()).unwrap();
    let repeated = pipeline
        .finish_post_inference(artifacts.clone(), &config)
        .unwrap();
    let split = pipeline.finish_post_inference(artifacts, &config).unwrap();

    assert_eq!(combined.segments, split.segments);
    assert_eq!(repeated.segments, split.segments);
}

#[test]
fn pipeline_builder_applies_custom_default_config_to_build() {
    let harness = PipelineTestHarness::load();
    let expected = custom_pipeline_config();
    let Some(pipeline) = build_pipeline_or_skip(
        PipelineBuilder::from_dir(harness.models_dir(), ExecutionMode::Cpu)
            .pipeline(expected.clone())
            .build(),
    ) else {
        return;
    };

    let actual = pipeline.pipeline_config();
    assert_eq!(actual.merge_gap, expected.merge_gap);
    assert_eq!(
        actual.speaker_keep_threshold,
        expected.speaker_keep_threshold
    );
    assert_eq!(actual.reconstruct_method, expected.reconstruct_method);
}

#[test]
fn borrowed_pipeline_new_with_config_stores_custom_default_config() {
    let harness = PipelineTestHarness::load();
    let Some(mut seg_model) = harness.cpu_seg_model() else {
        return;
    };
    let Some(mut emb_model) = harness.cpu_emb_model() else {
        return;
    };
    let expected = custom_pipeline_config();
    let pipeline = DiarizationPipeline::new_with_config(
        &mut seg_model,
        &mut emb_model,
        harness.models_dir(),
        expected.clone(),
    )
    .unwrap();

    let actual = pipeline.pipeline_config();
    assert_eq!(actual.merge_gap, expected.merge_gap);
    assert_eq!(
        actual.speaker_keep_threshold,
        expected.speaker_keep_threshold
    );
    assert_eq!(actual.reconstruct_method, expected.reconstruct_method);
}

#[cfg(feature = "coreml")]
#[test]
fn chunk_embedding_pipelined_vs_sequential_baseline() {
    let harness = PipelineTestHarness::load();
    let Some(mut pipeline) = harness.coreml_pipeline() else {
        return;
    };

    // multi-chunk audio (triggers pipelined path)
    // run full pipeline twice: chunk embedding path uses try_chunk_embedding
    // which internally picks pipelined vs sequential based on chunk count
    let result_a = pipeline.run(harness.audio()).unwrap();
    let result_b = pipeline.run(harness.audio()).unwrap();

    // both runs should produce identical RTTM
    assert_eq!(result_a.segments, result_b.segments);

    // also verify run_inference_only + finish_post_inference round-trips
    let config = pipeline.pipeline_config();
    let artifacts = pipeline.run_inference_only(harness.audio()).unwrap();
    let result_split = pipeline.finish_post_inference(artifacts, &config).unwrap();
    assert_eq!(result_a.segments, result_split.segments);
}

#[cfg(feature = "coreml")]
#[test]
fn chunk_embedding_keeps_an_unaligned_padded_tail() {
    let harness = PipelineTestHarness::load();
    let Some(mut pipeline) = harness.coreml_pipeline() else {
        return;
    };
    let window_samples = pipeline.seg_model.window_samples();
    let step_samples = pipeline.seg_model.step_samples();
    let audio_len = window_samples + 2 * step_samples + 1;
    let artifacts = pipeline
        .run_inference_only(&harness.audio()[..audio_len])
        .unwrap();

    assert_eq!(artifacts.segmentations.nchunks(), 4);
    assert_eq!(artifacts.embeddings.0.shape()[0], 4);
}

#[cfg(feature = "coreml")]
#[test]
fn batch_chunk_embedding_keeps_unaligned_padded_tails() {
    let harness = PipelineTestHarness::load();
    let Some(mut pipeline) = harness.coreml_pipeline() else {
        return;
    };
    let window_samples = pipeline.seg_model.window_samples();
    let step_samples = pipeline.seg_model.step_samples();
    let audio_len = window_samples + 2 * step_samples + 1;
    let audio = &harness.audio()[..audio_len];
    let files = [
        BatchInput {
            audio,
            file_id: "padded-a",
        },
        BatchInput {
            audio,
            file_id: "padded-b",
        },
    ];
    let results = pipeline.run_batch(&files).unwrap();

    assert_eq!(results.len(), 2);
    assert!(
        results
            .iter()
            .all(|result| result.segmentations.nchunks() == 4)
    );
}
