use std::error::Error;
use std::path::Path;
use std::time::Instant;

use ndarray::{Array2, Array3, ArrayView2, s};

use crate::clustering::ahc::{AhcConfig, cluster as cluster_ahc};
use crate::clustering::plda::PldaTransform;
use crate::clustering::vbx::{VbxConfig, cluster_vbx};
#[cfg(any(feature = "online", feature = "coreml"))]
use crate::inference::ExecutionMode;
#[cfg(feature = "native-coreml")]
use crate::inference::embedding::FusedEmbeddingInput;
use crate::inference::embedding::{
    EmbeddingModel, MaskedEmbeddingInput, SplitTailInput, should_use_clean_mask,
};
use crate::inference::segmentation::SegmentationModel;
use crate::powerset::PowersetMapping;
use crate::reconstruct::{reconstruct, speaker_count};
use crate::segment::{merge_segments, to_rttm, to_segments};
use crate::utils::cosine_similarity;

type DynError = Box<dyn Error + Send + Sync + 'static>;

fn profiling_enabled() -> bool {
    std::env::var("SPEAKRS_PROFILE").is_ok()
}

pub const SEGMENTATION_WINDOW_SECONDS: f64 = 10.0;
pub const SEGMENTATION_STEP_SECONDS: f64 = 1.0;
pub const FAST_SEGMENTATION_STEP_SECONDS: f64 = 2.0;
pub const FRAME_DURATION_SECONDS: f64 = 0.0619375;
pub const FRAME_STEP_SECONDS: f64 = 0.016875;

struct PendingEmbedding<'a> {
    chunk_idx: usize,
    speaker_idx: usize,
    audio: &'a [f32],
    mask: Vec<f32>,
    clean_mask: Vec<f32>,
}

struct PendingSplitEmbedding {
    chunk_idx: usize,
    speaker_idx: usize,
    fbank_idx: usize,
    weights: Vec<f32>,
}

#[cfg(feature = "native-coreml")]
struct PendingFusedEmbedding<'a> {
    chunk_idx: usize,
    speaker_idx: usize,
    audio: &'a [f32],
    weights: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct DiarizationResult {
    pub segmentations: Array3<f32>,
    pub embeddings: Array3<f32>,
    pub speaker_count: Vec<usize>,
    pub hard_clusters: Array2<i32>,
    pub discrete_diarization: Array2<f32>,
    pub rttm: String,
}

/// Owned pipeline that manages its own model lifetimes
#[cfg(feature = "online")]
pub struct OwnedDiarizationPipeline {
    seg_model: SegmentationModel,
    emb_model: EmbeddingModel,
    plda: PldaTransform,
}

#[cfg(feature = "online")]
impl OwnedDiarizationPipeline {
    /// Download models from HuggingFace and build the pipeline
    pub fn from_pretrained(mode: crate::models::Mode) -> Result<Self, DynError> {
        let manager = crate::models::ModelManager::new()?;
        let models_dir = manager.ensure(mode)?;

        let execution_mode = match mode {
            crate::models::Mode::Cpu => ExecutionMode::Cpu,
            crate::models::Mode::CoreMl => ExecutionMode::CoreMl,
            crate::models::Mode::CoreMlLite => ExecutionMode::CoreMlLite,
            crate::models::Mode::Cuda => ExecutionMode::Cuda,
        };

        let step = match execution_mode {
            ExecutionMode::CoreMlLite => FAST_SEGMENTATION_STEP_SECONDS,
            _ => SEGMENTATION_STEP_SECONDS,
        };

        let seg_model = SegmentationModel::with_mode(
            models_dir.join("segmentation-3.0.onnx").to_str().unwrap(),
            step as f32,
            execution_mode,
        )?;
        let emb_model = EmbeddingModel::with_mode(
            models_dir
                .join("wespeaker-voxceleb-resnet34.onnx")
                .to_str()
                .unwrap(),
            execution_mode,
        )?;
        let plda = PldaTransform::from_dir(&models_dir)?;

        Ok(Self {
            seg_model,
            emb_model,
            plda,
        })
    }

    pub fn run(&mut self, audio: &[f32]) -> Result<DiarizationResult, DynError> {
        diarize(
            &mut self.seg_model,
            &mut self.emb_model,
            &self.plda,
            audio,
            "file1",
        )
    }

    pub fn segmentation_step(&self) -> f64 {
        self.seg_model.step_seconds()
    }
}

pub struct DiarizationPipeline<'a> {
    seg_model: &'a mut SegmentationModel,
    emb_model: &'a mut EmbeddingModel,
    plda: PldaTransform,
}

impl<'a> DiarizationPipeline<'a> {
    pub fn new(
        seg_model: &'a mut SegmentationModel,
        emb_model: &'a mut EmbeddingModel,
        models_dir: &Path,
    ) -> Result<Self, DynError> {
        Ok(Self {
            seg_model,
            emb_model,
            plda: PldaTransform::from_dir(models_dir)?,
        })
    }

    pub fn default_segmentation_step() -> f32 {
        SEGMENTATION_STEP_SECONDS as f32
    }

    pub fn run(&mut self, audio: &[f32]) -> Result<DiarizationResult, DynError> {
        diarize(self.seg_model, self.emb_model, &self.plda, audio, "file1")
    }

    pub fn segmentation_step(&self) -> f64 {
        self.seg_model.step_seconds()
    }
}

impl DiarizationResult {
    pub fn rttm(&self, file_id: &str) -> String {
        if file_id == "file1" {
            return self.rttm.clone();
        }

        self.rttm
            .replace("SPEAKER file1 1", &format!("SPEAKER {file_id} 1"))
    }
}

pub fn diarize(
    seg_model: &mut SegmentationModel,
    emb_model: &mut EmbeddingModel,
    plda: &PldaTransform,
    audio: &[f32],
    file_id: &str,
) -> Result<DiarizationResult, DynError> {
    let profile = profiling_enabled();
    let powerset = PowersetMapping::new(3, 2);
    let t0 = Instant::now();
    let raw_windows = seg_model.run(audio)?;
    if profile {
        eprintln!(
            "[profile] segmentation: {:.3}s ({} windows)",
            t0.elapsed().as_secs_f64(),
            raw_windows.len()
        );
    }
    if raw_windows.is_empty() {
        return Ok(DiarizationResult {
            segmentations: Array3::zeros((0, 0, 0)),
            embeddings: Array3::zeros((0, 0, 0)),
            speaker_count: Vec::new(),
            hard_clusters: Array2::zeros((0, 0)),
            discrete_diarization: Array2::zeros((0, 0)),
            rttm: String::new(),
        });
    }

    let step_seconds = seg_model.step_seconds();
    let segmentations = decode_windows(raw_windows, &powerset);
    let start_frames = chunk_start_frames(segmentations.shape()[0], step_seconds);
    let output_frames = total_output_frames(segmentations.shape()[0], step_seconds);
    let speaker_count = speaker_count(&segmentations, &start_frames, 0, output_frames);

    if speaker_count.iter().all(|count| *count == 0) {
        return Ok(DiarizationResult {
            segmentations,
            embeddings: Array3::zeros((0, 0, 0)),
            speaker_count,
            hard_clusters: Array2::zeros((0, 0)),
            discrete_diarization: Array2::zeros((0, 0)),
            rttm: String::new(),
        });
    }

    let t0 = Instant::now();
    let embeddings = extract_embeddings(seg_model, emb_model, audio, &segmentations)?;
    if profile {
        eprintln!(
            "[profile] embeddings: {:.3}s ({} chunks × {} speakers)",
            t0.elapsed().as_secs_f64(),
            segmentations.shape()[0],
            segmentations.shape()[2],
        );
    }
    let (train_embeddings, _train_chunk_idx, _train_speaker_idx) =
        filter_embeddings(&segmentations, &embeddings);

    let t0 = Instant::now();
    let hard_clusters = if train_embeddings.nrows() < 2 {
        let mut clusters =
            Array2::<i32>::zeros((segmentations.shape()[0], segmentations.shape()[2]));
        mark_inactive_speakers(&segmentations, &mut clusters);
        clusters
    } else {
        let ahc_labels = cluster_ahc(&train_embeddings.view(), AhcConfig::default());
        let plda_features = plda.transform(&train_embeddings.view(), 128);
        let (gamma, pi) = cluster_vbx(
            &ahc_labels,
            &plda_features.view(),
            &plda.phi().slice(s![..128]),
            &VbxConfig::default(),
        );
        let mut kept_speakers: Vec<usize> = pi
            .iter()
            .enumerate()
            .filter_map(|(idx, weight)| (*weight > 1e-7).then_some(idx))
            .collect();
        if kept_speakers.is_empty() && !pi.is_empty() {
            let best = pi
                .iter()
                .enumerate()
                .max_by(|lhs, rhs| lhs.1.partial_cmp(rhs.1).unwrap())
                .map(|(idx, _)| idx)
                .unwrap();
            kept_speakers.push(best);
        }
        let centroids = weighted_centroids(&train_embeddings, &gamma, &kept_speakers);
        let mut clusters = assign_embeddings(&segmentations, &embeddings, &centroids);
        mark_inactive_speakers(&segmentations, &mut clusters);
        clusters
    };

    if profile {
        eprintln!("[profile] clustering: {:.3}s", t0.elapsed().as_secs_f64());
    }

    let discrete_diarization = reconstruct(
        &segmentations,
        &hard_clusters,
        &speaker_count,
        &start_frames,
        0,
    );
    let segments = to_segments(
        &discrete_diarization,
        FRAME_STEP_SECONDS,
        FRAME_DURATION_SECONDS,
    );
    let segments = merge_segments(&segments, 0.0);
    let rttm = to_rttm(&segments, file_id);

    Ok(DiarizationResult {
        segmentations,
        embeddings,
        speaker_count,
        hard_clusters,
        discrete_diarization,
        rttm,
    })
}

fn decode_windows(raw_windows: Vec<Array2<f32>>, powerset: &PowersetMapping) -> Array3<f32> {
    let num_windows = raw_windows.len();
    let mut windows = raw_windows.into_iter();
    let first = powerset.hard_decode(&windows.next().unwrap());
    let mut stacked = Array3::<f32>::zeros((num_windows, first.nrows(), first.ncols()));
    stacked.slice_mut(s![0, .., ..]).assign(&first);

    for (window_idx, window) in windows.enumerate() {
        let decoded = powerset.hard_decode(&window);
        stacked
            .slice_mut(s![window_idx + 1, .., ..])
            .assign(&decoded);
    }
    stacked
}

fn extract_embeddings(
    seg_model: &SegmentationModel,
    emb_model: &mut EmbeddingModel,
    audio: &[f32],
    segmentations: &Array3<f32>,
) -> Result<Array3<f32>, DynError> {
    let num_chunks = segmentations.shape()[0];
    let num_speakers = segmentations.shape()[2];
    let mut embeddings = Array3::<f32>::from_elem((num_chunks, num_speakers, 256), f32::NAN);

    if emb_model.prefers_chunk_embedding_path() {
        if emb_model.split_primary_batch_size() > 0 {
            extract_split_embeddings(seg_model, emb_model, audio, segmentations, &mut embeddings)?;
        } else {
            for chunk_idx in 0..num_chunks {
                let chunk_audio = chunk_audio(audio, seg_model, chunk_idx);
                let chunk_segmentations = segmentations.slice(s![chunk_idx, .., ..]);
                let clean_masks = clean_masks(&chunk_segmentations);
                let chunk_embeddings = emb_model.embed_chunk_speakers(
                    chunk_audio,
                    chunk_segmentations,
                    &clean_masks,
                )?;
                embeddings
                    .slice_mut(s![chunk_idx, .., ..])
                    .assign(&chunk_embeddings);
            }
        }
        return Ok(embeddings);
    }

    let mut pending = Vec::with_capacity(emb_model.primary_batch_size());

    for chunk_idx in 0..num_chunks {
        let chunk_audio = chunk_audio(audio, seg_model, chunk_idx);
        let chunk_segmentations = segmentations.slice(s![chunk_idx, .., ..]);
        let clean_masks = clean_masks(&chunk_segmentations);

        for speaker_idx in 0..num_speakers {
            pending.push(PendingEmbedding {
                chunk_idx,
                speaker_idx,
                audio: chunk_audio,
                mask: chunk_segmentations.column(speaker_idx).to_vec(),
                clean_mask: clean_masks.column(speaker_idx).to_vec(),
            });
            if pending.len() == emb_model.primary_batch_size() {
                flush_embedding_batch(emb_model, &pending, &mut embeddings)?;
                pending.clear();
            }
        }
    }

    while !pending.is_empty() {
        let batch_len = emb_model.best_batch_len(pending.len());
        flush_embedding_batch(emb_model, &pending[..batch_len], &mut embeddings)?;
        pending.drain(..batch_len);
    }

    Ok(embeddings)
}

fn flush_embedding_batch(
    emb_model: &mut EmbeddingModel,
    pending: &[PendingEmbedding<'_>],
    embeddings: &mut Array3<f32>,
) -> Result<(), DynError> {
    let batch_inputs: Vec<_> = pending
        .iter()
        .map(|item| MaskedEmbeddingInput {
            audio: item.audio,
            mask: &item.mask,
            clean_mask: Some(&item.clean_mask),
        })
        .collect();
    let batch_embeddings = emb_model.embed_batch(&batch_inputs)?;

    for (batch_idx, item) in pending.iter().enumerate() {
        embeddings
            .slice_mut(s![item.chunk_idx, item.speaker_idx, ..])
            .assign(&batch_embeddings.row(batch_idx));
    }

    Ok(())
}

fn extract_split_embeddings(
    seg_model: &SegmentationModel,
    emb_model: &mut EmbeddingModel,
    audio: &[f32],
    segmentations: &Array3<f32>,
    embeddings: &mut Array3<f32>,
) -> Result<(), DynError> {
    // fused path: waveform+weights → embedding in one CoreML call, no separate fbank
    #[cfg(feature = "native-coreml")]
    if emb_model.has_fused_primary_batch() {
        return extract_fused_embeddings(seg_model, emb_model, audio, segmentations, embeddings);
    }

    let profile = profiling_enabled();
    let batch_size = emb_model.split_primary_batch_size();
    let num_chunks = segmentations.shape()[0];
    let num_speakers = segmentations.shape()[2];
    let min_num_samples = emb_model.min_num_samples();

    let t0 = Instant::now();
    let mut pending: Vec<PendingSplitEmbedding> = Vec::with_capacity(batch_size);
    let mut fbanks: Vec<Array2<f32>> = Vec::new();
    let mut tail_batches = 0usize;

    // stream: compute fbank per chunk, immediately enqueue speaker tail items
    for chunk_idx in 0..num_chunks {
        let chunk_audio = chunk_audio(audio, seg_model, chunk_idx);
        let fbank = emb_model.compute_chunk_fbank(chunk_audio)?;
        fbanks.push(fbank);
        let mut cur_fbank_idx = fbanks.len() - 1;

        let chunk_segmentations = segmentations.slice(s![chunk_idx, .., ..]);
        let clean = clean_masks(&chunk_segmentations);

        for speaker_idx in 0..num_speakers {
            let mask_col = chunk_segmentations.column(speaker_idx);
            let clean_col = clean.column(speaker_idx);
            let use_clean = should_use_clean_mask(
                &clean_col,
                mask_col.len(),
                chunk_audio.len(),
                min_num_samples,
            );
            let weights: Vec<f32> = if use_clean {
                clean_col.iter().copied().collect()
            } else {
                mask_col.iter().copied().collect()
            };
            pending.push(PendingSplitEmbedding {
                chunk_idx,
                speaker_idx,
                fbank_idx: cur_fbank_idx,
                weights,
            });
            if pending.len() == batch_size {
                flush_split_embedding_batch(emb_model, &pending, &fbanks, embeddings)?;
                tail_batches += 1;
                pending.clear();

                // keep current chunk's fbank if more speakers remain
                if speaker_idx + 1 < num_speakers {
                    let kept = fbanks.swap_remove(cur_fbank_idx);
                    fbanks.clear();
                    fbanks.push(kept);
                    cur_fbank_idx = 0;
                } else {
                    fbanks.clear();
                }
            }
        }
    }

    if !pending.is_empty() {
        flush_split_embedding_batch(emb_model, &pending, &fbanks, embeddings)?;
        tail_batches += 1;
    }
    if profile {
        eprintln!(
            "[profile]   fbank+tail: {:.3}s ({tail_batches} batches, {} items, streaming)",
            t0.elapsed().as_secs_f64(),
            num_chunks * num_speakers,
        );
    }

    Ok(())
}

fn flush_split_embedding_batch(
    emb_model: &mut EmbeddingModel,
    pending: &[PendingSplitEmbedding],
    fbanks: &[Array2<f32>],
    embeddings: &mut Array3<f32>,
) -> Result<(), DynError> {
    let batch_inputs: Vec<_> = pending
        .iter()
        .map(|item| SplitTailInput {
            fbank: &fbanks[item.fbank_idx],
            weights: &item.weights,
        })
        .collect();
    let batch_embeddings = emb_model.embed_tail_batch_inputs(&batch_inputs)?;

    for (batch_idx, item) in pending.iter().enumerate() {
        embeddings
            .slice_mut(s![item.chunk_idx, item.speaker_idx, ..])
            .assign(&batch_embeddings.row(batch_idx));
    }

    Ok(())
}

#[cfg(feature = "native-coreml")]
fn extract_fused_embeddings(
    seg_model: &SegmentationModel,
    emb_model: &mut EmbeddingModel,
    audio: &[f32],
    segmentations: &Array3<f32>,
    embeddings: &mut Array3<f32>,
) -> Result<(), DynError> {
    let profile = profiling_enabled();
    let batch_size = emb_model.split_primary_batch_size();
    let num_chunks = segmentations.shape()[0];
    let num_speakers = segmentations.shape()[2];

    let t_fused = Instant::now();
    let mut pending: Vec<PendingFusedEmbedding<'_>> = Vec::with_capacity(batch_size);
    let mut fused_batches = 0usize;

    let min_num_samples = emb_model.min_num_samples();

    for chunk_idx in 0..num_chunks {
        let chunk_audio = chunk_audio(audio, seg_model, chunk_idx);
        let chunk_segmentations = segmentations.slice(s![chunk_idx, .., ..]);
        let clean = clean_masks(&chunk_segmentations);

        for speaker_idx in 0..num_speakers {
            let mask_col = chunk_segmentations.column(speaker_idx);
            let clean_col = clean.column(speaker_idx);
            let use_clean = should_use_clean_mask(
                &clean_col,
                mask_col.len(),
                chunk_audio.len(),
                min_num_samples,
            );
            let weights: Vec<f32> = if use_clean {
                clean_col.iter().copied().collect()
            } else {
                mask_col.iter().copied().collect()
            };
            pending.push(PendingFusedEmbedding {
                chunk_idx,
                speaker_idx,
                audio: chunk_audio,
                weights,
            });
            if pending.len() == batch_size {
                flush_fused_embedding_batch(emb_model, &pending, embeddings)?;
                fused_batches += 1;
                pending.clear();
            }
        }
    }

    if !pending.is_empty() {
        flush_fused_embedding_batch(emb_model, &pending, embeddings)?;
        fused_batches += 1;
    }
    if profile {
        eprintln!(
            "[profile]   fused: {:.3}s ({fused_batches} batches, {} items)",
            t_fused.elapsed().as_secs_f64(),
            num_chunks * num_speakers,
        );
    }

    Ok(())
}

#[cfg(feature = "native-coreml")]
fn flush_fused_embedding_batch(
    emb_model: &mut EmbeddingModel,
    pending: &[PendingFusedEmbedding<'_>],
    embeddings: &mut Array3<f32>,
) -> Result<(), DynError> {
    let batch_inputs: Vec<_> = pending
        .iter()
        .map(|item| FusedEmbeddingInput {
            audio: item.audio,
            weights: &item.weights,
        })
        .collect();
    let batch_embeddings = emb_model.embed_fused_batch_inputs(&batch_inputs)?;

    for (batch_idx, item) in pending.iter().enumerate() {
        embeddings
            .slice_mut(s![item.chunk_idx, item.speaker_idx, ..])
            .assign(&batch_embeddings.row(batch_idx));
    }

    Ok(())
}

fn filter_embeddings(
    segmentations: &Array3<f32>,
    embeddings: &Array3<f32>,
) -> (Array2<f32>, Vec<usize>, Vec<usize>) {
    let num_frames = segmentations.shape()[1] as f32;
    let mut filtered = Vec::new();
    let mut chunk_idx = Vec::new();
    let mut speaker_idx = Vec::new();

    for chunk in 0..segmentations.shape()[0] {
        let single_active: Vec<bool> = segmentations
            .slice(s![chunk, .., ..])
            .rows()
            .into_iter()
            .map(|row| (row.iter().copied().sum::<f32>() - 1.0).abs() < 1e-6)
            .collect();
        for speaker in 0..segmentations.shape()[2] {
            let clean_frames = segmentations
                .slice(s![chunk, .., speaker])
                .iter()
                .zip(single_active.iter())
                .filter_map(|(value, single)| single.then_some(*value))
                .sum::<f32>();
            let embedding = embeddings.slice(s![chunk, speaker, ..]);
            let valid = embedding.iter().all(|value| value.is_finite());
            if valid && clean_frames >= 0.2 * num_frames {
                filtered.extend(embedding.iter());
                chunk_idx.push(chunk);
                speaker_idx.push(speaker);
            }
        }
    }

    let filtered_embeddings =
        Array2::from_shape_vec((chunk_idx.len(), embeddings.shape()[2]), filtered).unwrap();
    (filtered_embeddings, chunk_idx, speaker_idx)
}

fn weighted_centroids(
    train_embeddings: &Array2<f32>,
    gamma: &Array2<f32>,
    kept_speakers: &[usize],
) -> Array2<f32> {
    let mut centroids = Array2::<f32>::zeros((kept_speakers.len(), train_embeddings.ncols()));
    for (out_idx, &speaker_idx) in kept_speakers.iter().enumerate() {
        let weights = gamma.column(speaker_idx);
        let weight_sum = weights.sum().max(1e-8);
        for (row_idx, weight) in weights.iter().enumerate() {
            centroids
                .row_mut(out_idx)
                .scaled_add(*weight / weight_sum, &train_embeddings.row(row_idx));
        }
    }
    centroids
}

fn assign_embeddings(
    segmentations: &Array3<f32>,
    embeddings: &Array3<f32>,
    centroids: &Array2<f32>,
) -> Array2<i32> {
    let num_chunks = embeddings.shape()[0];
    let num_speakers = embeddings.shape()[1];
    let num_clusters = centroids.nrows();
    let mut labels = Array2::<i32>::from_elem((num_chunks, num_speakers), -2);

    for chunk_idx in 0..num_chunks {
        let mut active_local = Vec::new();
        let mut scores = Array2::<f32>::from_elem((num_speakers, num_clusters), f32::NEG_INFINITY);
        for speaker_idx in 0..num_speakers {
            let is_active = segmentations.slice(s![chunk_idx, .., speaker_idx]).sum() > 0.0;
            if !is_active {
                continue;
            }

            active_local.push(speaker_idx);
            let embedding = embeddings.slice(s![chunk_idx, speaker_idx, ..]);
            if embedding.iter().any(|value| !value.is_finite()) {
                continue;
            }

            for cluster_idx in 0..num_clusters {
                scores[[speaker_idx, cluster_idx]] =
                    1.0 + cosine_similarity(&embedding, &centroids.row(cluster_idx));
            }
        }

        let assignments = best_assignment(&scores, &active_local, num_clusters);
        for (speaker_idx, cluster_idx) in assignments {
            labels[[chunk_idx, speaker_idx]] = cluster_idx as i32;
        }
    }

    labels
}

fn best_assignment(
    scores: &Array2<f32>,
    active_local: &[usize],
    num_clusters: usize,
) -> Vec<(usize, usize)> {
    let target = active_local.len().min(num_clusters);
    let mut search = AssignmentSearch::new(scores, active_local, target, num_clusters);
    search.run(0, 0.0);
    search.best
}

struct AssignmentSearch<'a> {
    scores: &'a Array2<f32>,
    active_local: &'a [usize],
    target: usize,
    used_clusters: Vec<bool>,
    current: Vec<(usize, usize)>,
    best_score: f32,
    best: Vec<(usize, usize)>,
}

impl<'a> AssignmentSearch<'a> {
    fn new(
        scores: &'a Array2<f32>,
        active_local: &'a [usize],
        target: usize,
        num_clusters: usize,
    ) -> Self {
        Self {
            scores,
            active_local,
            target,
            used_clusters: vec![false; num_clusters],
            current: Vec::new(),
            best_score: f32::NEG_INFINITY,
            best: Vec::new(),
        }
    }

    fn run(&mut self, position: usize, current_score: f32) {
        if self.current.len() == self.target {
            if current_score > self.best_score {
                self.best_score = current_score;
                self.best = self.current.clone();
            }
            return;
        }

        if position == self.active_local.len() {
            return;
        }

        let remaining_local = self.active_local.len() - position;
        let remaining_needed = self.target - self.current.len();
        if remaining_local > remaining_needed {
            self.run(position + 1, current_score);
        }

        let speaker_idx = self.active_local[position];
        for cluster_idx in 0..self.used_clusters.len() {
            if self.used_clusters[cluster_idx] {
                continue;
            }

            self.used_clusters[cluster_idx] = true;
            self.current.push((speaker_idx, cluster_idx));
            self.run(
                position + 1,
                current_score + self.scores[[speaker_idx, cluster_idx]],
            );
            self.current.pop();
            self.used_clusters[cluster_idx] = false;
        }
    }
}

fn mark_inactive_speakers(segmentations: &Array3<f32>, hard_clusters: &mut Array2<i32>) {
    for chunk_idx in 0..segmentations.shape()[0] {
        for speaker_idx in 0..segmentations.shape()[2] {
            let active = segmentations.slice(s![chunk_idx, .., speaker_idx]).sum() > 0.0;
            if !active {
                hard_clusters[[chunk_idx, speaker_idx]] = -2;
            }
        }
    }
}

fn clean_masks(segmentations: &ArrayView2<f32>) -> Array2<f32> {
    let single_active: Vec<bool> = segmentations
        .rows()
        .into_iter()
        .map(|row| row.iter().copied().sum::<f32>() < 2.0)
        .collect();
    let mut clean = Array2::<f32>::zeros(segmentations.raw_dim());
    for (frame_idx, is_single_active) in single_active.iter().enumerate() {
        if !*is_single_active {
            continue;
        }

        clean
            .slice_mut(s![frame_idx, ..])
            .assign(&segmentations.slice(s![frame_idx, ..]));
    }
    clean
}

fn chunk_audio<'a>(audio: &'a [f32], seg_model: &SegmentationModel, chunk_idx: usize) -> &'a [f32] {
    let start = chunk_idx * seg_model.step_samples();
    let end = (start + seg_model.window_samples()).min(audio.len());
    if start < audio.len() {
        &audio[start..end]
    } else {
        &[]
    }
}

fn chunk_start_frames(num_chunks: usize, step_seconds: f64) -> Vec<usize> {
    (0..num_chunks)
        .map(|chunk_idx| {
            closest_frame(chunk_idx as f64 * step_seconds + 0.5 * FRAME_DURATION_SECONDS)
        })
        .collect()
}

fn total_output_frames(num_chunks: usize, step_seconds: f64) -> usize {
    if num_chunks == 0 {
        return 0;
    }

    closest_frame(
        SEGMENTATION_WINDOW_SECONDS
            + (num_chunks - 1) as f64 * step_seconds
            + 0.5 * FRAME_DURATION_SECONDS,
    ) + 1
}

fn closest_frame(timestamp: f64) -> usize {
    ((timestamp - 0.5 * FRAME_DURATION_SECONDS) / FRAME_STEP_SECONDS).round() as usize
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, Array2, Array3, array};
    use ndarray_npy::ReadNpyExt;
    use std::fs::File;
    use std::path::{Path, PathBuf};

    use super::*;
    #[cfg(feature = "coreml")]
    use crate::inference::ExecutionMode;

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
            let chunk_size =
                u32::from_le_bytes(data[pos + 4..pos + 8].try_into().unwrap()) as usize;
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
        let assignment = best_assignment(&scores, &[0, 1, 2], 2);
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

        let (train_embeddings, chunk_idx, speaker_idx) =
            filter_embeddings(&segmentations, &embeddings);

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
            Array2::read_npy(File::open(fixture_path("pipeline_hard_clusters.npy")).unwrap())
                .unwrap();

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
}
