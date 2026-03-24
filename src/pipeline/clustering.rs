use ndarray::{Array2, Array3, ArrayView2, s};
use tracing::{debug, trace};

use crate::clustering::ahc::cluster as cluster_ahc;
use crate::clustering::plda::PldaTransform;
use crate::clustering::vbx::cluster_vbx;
use crate::inference::embedding::should_use_clean_mask;
use crate::utils::cosine_similarity;

use super::config::*;
use super::types::*;

pub(super) struct TrainingEmbeddings(pub Array2<f32>);

impl ChunkEmbeddings {
    pub(super) fn training_set(&self, segmentations: &DecodedSegmentations) -> TrainingEmbeddings {
        let num_frames = segmentations.0.shape()[1] as f32;
        let mut filtered = Vec::new();
        let mut chunk_indices = Vec::new();

        for chunk_idx in 0..segmentations.0.shape()[0] {
            let single_active: Vec<bool> = segmentations
                .0
                .slice(s![chunk_idx, .., ..])
                .rows()
                .into_iter()
                .map(|row| (row.iter().copied().sum::<f32>() - 1.0).abs() < 1e-6)
                .collect();
            for speaker_idx in 0..segmentations.0.shape()[2] {
                let clean_frames = segmentations
                    .0
                    .slice(s![chunk_idx, .., speaker_idx])
                    .iter()
                    .zip(single_active.iter())
                    .filter_map(|(value, is_single_active)| is_single_active.then_some(*value))
                    .sum::<f32>();
                let embedding = self.0.slice(s![chunk_idx, speaker_idx, ..]);
                let valid_embedding = embedding.iter().all(|value| value.is_finite());
                if valid_embedding && clean_frames >= 0.2 * num_frames {
                    filtered.extend(embedding.iter());
                    chunk_indices.push(chunk_idx);
                }
            }
        }

        let row_count = chunk_indices.len();
        let filtered_embeddings =
            Array2::from_shape_vec((row_count, self.0.shape()[2]), filtered).unwrap();
        TrainingEmbeddings(filtered_embeddings)
    }
}

impl TrainingEmbeddings {
    pub(super) fn cluster(
        &self,
        segmentations: &DecodedSegmentations,
        embeddings: &ChunkEmbeddings,
        plda: &PldaTransform,
        config: &PipelineConfig,
    ) -> ChunkSpeakerClusters {
        if self.0.nrows() < 2 {
            let mut clusters =
                Array2::<i32>::zeros((segmentations.0.shape()[0], segmentations.0.shape()[2]));
            mark_inactive_speakers(&segmentations.0, &mut clusters);
            return ChunkSpeakerClusters(clusters);
        }

        let ahc_labels = cluster_ahc(&self.0.view(), config.ahc);
        debug!(
            rows = self.0.nrows(),
            cols = self.0.ncols(),
            "train_embeddings shape"
        );
        {
            let unique: std::collections::BTreeSet<_> = ahc_labels.iter().copied().collect();
            debug!(num_clusters = unique.len(), "AHC pre-clustering");
            for &cluster in &unique {
                let count = ahc_labels.iter().filter(|&&value| value == cluster).count();
                debug!(cluster, count, "AHC cluster size");
            }
        }

        let plda_features = plda.transform(&self.0.view(), 128);
        let phi = plda.phi();
        let (gamma, pi): (Array2<f32>, ndarray::Array1<f32>) = cluster_vbx(
            &ahc_labels,
            &plda_features.view(),
            &phi.slice(s![..128]),
            &config.vbx,
        );

        debug!(?pi, "VBx speaker priors");

        let mut kept_speakers: Vec<usize> = pi
            .iter()
            .enumerate()
            .filter_map(|(speaker_idx, weight)| {
                (*weight > config.speaker_keep_threshold as f32).then_some(speaker_idx)
            })
            .collect();
        if kept_speakers.is_empty() && !pi.is_empty() {
            let best_speaker = pi
                .iter()
                .enumerate()
                .max_by(|left, right| left.1.partial_cmp(right.1).unwrap())
                .map(|(speaker_idx, _)| speaker_idx)
                .unwrap();
            kept_speakers.push(best_speaker);
        }

        debug!(?kept_speakers, "VBx kept speakers");
        let centroids = weighted_centroids(&self.0, &gamma, &kept_speakers);
        for cluster_idx in 0..centroids.nrows() {
            let norm: f32 = centroids
                .row(cluster_idx)
                .dot(&centroids.row(cluster_idx))
                .sqrt();
            debug!(cluster = cluster_idx, norm, "centroid");
        }

        let mut clusters = assign_chunk_embeddings(segmentations, embeddings, &centroids);
        mark_inactive_speakers(&segmentations.0, &mut clusters);
        debug!(
            rows = clusters.nrows(),
            cols = clusters.ncols(),
            "hard_clusters shape"
        );

        ChunkSpeakerClusters(clusters)
    }
}

pub(super) fn weighted_centroids(
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

pub(super) fn assign_chunk_embeddings(
    segmentations: &DecodedSegmentations,
    embeddings: &ChunkEmbeddings,
    centroids: &Array2<f32>,
) -> Array2<i32> {
    let num_chunks = embeddings.0.shape()[0];
    let num_speakers = embeddings.0.shape()[1];
    let num_clusters = centroids.nrows();
    let mut labels = Array2::<i32>::from_elem((num_chunks, num_speakers), -2);

    for chunk_idx in 0..num_chunks {
        // compute similarity scores for all active speakers against all centroids
        let mut active_local = Vec::new();
        let mut scores = Array2::<f32>::from_elem((num_speakers, num_clusters), f32::NEG_INFINITY);
        for speaker_idx in 0..num_speakers {
            let is_active = segmentations.0.slice(s![chunk_idx, .., speaker_idx]).sum() > 0.0;
            if !is_active {
                continue;
            }

            active_local.push(speaker_idx);
            let embedding = embeddings.0.slice(s![chunk_idx, speaker_idx, ..]);
            if embedding.iter().any(|value| !value.is_finite()) {
                continue;
            }

            for cluster_idx in 0..num_clusters {
                scores[[speaker_idx, cluster_idx]] =
                    1.0 + cosine_similarity(&embedding, &centroids.row(cluster_idx));
            }
        }

        // mask inactive/invalid speakers to min - 1 instead of NEG_INFINITY,
        // matching pyannote's constrained_argmax masking behavior
        let finite_min = scores
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .fold(f32::INFINITY, f32::min);
        if finite_min.is_finite() {
            let mask_value = finite_min - 1.0;
            scores.mapv_inplace(|v| if v.is_finite() { v } else { mask_value });
        }

        let assignments = best_assignment(&scores, &active_local, num_clusters);
        if tracing::enabled!(tracing::Level::TRACE) {
            trace!(
                chunk = chunk_idx,
                ?active_local,
                ?assignments,
                "chunk assignment"
            );
            for speaker_idx in 0..num_speakers {
                let row: Vec<f32> = scores.row(speaker_idx).to_vec();
                trace!(chunk = chunk_idx, speaker = speaker_idx, ?row, "scores");
            }
        }
        for (speaker_idx, cluster_idx) in assignments {
            labels[[chunk_idx, speaker_idx]] = cluster_idx as i32;
        }
    }

    labels
}

pub(super) fn best_assignment(
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

pub(crate) fn mark_inactive_speakers(segmentations: &Array3<f32>, hard_clusters: &mut Array2<i32>) {
    for chunk_idx in 0..segmentations.shape()[0] {
        for speaker_idx in 0..segmentations.shape()[2] {
            let active = segmentations.slice(s![chunk_idx, .., speaker_idx]).sum() > 0.0;
            if !active {
                hard_clusters[[chunk_idx, speaker_idx]] = -2;
            }
        }
    }
}

pub(crate) fn clean_masks(segmentations: &ArrayView2<f32>) -> Array2<f32> {
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

/// Select speaker weights for embedding, returning None if speaker activity is below threshold
pub(crate) fn select_speaker_weights(
    seg_view: &ArrayView2<f32>,
    clean_masks: &Array2<f32>,
    speaker_idx: usize,
    audio_len: usize,
    min_num_samples: usize,
) -> Option<Vec<f32>> {
    let mask_col = seg_view.column(speaker_idx);
    let activity: f32 = mask_col.iter().sum();
    if activity < MIN_SPEAKER_ACTIVITY {
        return None;
    }

    let clean_col = clean_masks.column(speaker_idx);
    let use_clean = should_use_clean_mask(&clean_col, mask_col.len(), audio_len, min_num_samples);
    if use_clean {
        Some(clean_col.iter().copied().collect())
    } else {
        Some(mask_col.iter().copied().collect())
    }
}

/// Write the chosen speaker mask directly into a destination slice, avoiding
/// the intermediate Array2 and Vec allocations of clean_masks + select_speaker_weights
pub(crate) fn write_speaker_mask_to_slice(
    seg_view: &ArrayView2<f32>,
    speaker_idx: usize,
    audio_len: usize,
    min_num_samples: usize,
    dest: &mut [f32],
) -> bool {
    let mask_col = seg_view.column(speaker_idx);
    let activity: f32 = mask_col.iter().sum();
    if activity < MIN_SPEAKER_ACTIVITY {
        return false;
    }

    let nrows = seg_view.nrows();

    // compute clean column sum to decide which mask to use
    let mut clean_sum = 0.0f32;
    for row_idx in 0..nrows {
        let row_sum: f32 = seg_view.row(row_idx).iter().sum();
        if row_sum < 2.0 {
            clean_sum += seg_view[[row_idx, speaker_idx]];
        }
    }

    // inline should_use_clean_mask logic
    let use_clean = audio_len > 0 && {
        let min_mask_frames = (nrows * min_num_samples).div_ceil(audio_len) as f32;
        clean_sum > min_mask_frames
    };

    let copy_len = nrows.min(dest.len());
    if use_clean {
        for row_idx in 0..copy_len {
            let row_sum: f32 = seg_view.row(row_idx).iter().sum();
            dest[row_idx] = if row_sum < 2.0 {
                seg_view[[row_idx, speaker_idx]]
            } else {
                0.0
            };
        }
    } else {
        for row_idx in 0..copy_len {
            dest[row_idx] = mask_col[row_idx];
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify write_speaker_mask_to_slice matches clean_masks + select_speaker_weights
    fn assert_matches_original(seg: &Array2<f32>, audio_len: usize, min_num_samples: usize) {
        let clean = clean_masks(&seg.view());
        for speaker_idx in 0..seg.ncols() {
            let original = select_speaker_weights(
                &seg.view(),
                &clean,
                speaker_idx,
                audio_len,
                min_num_samples,
            );

            let mut dest = vec![0.0f32; seg.nrows()];
            let active = write_speaker_mask_to_slice(
                &seg.view(),
                speaker_idx,
                audio_len,
                min_num_samples,
                &mut dest,
            );

            match original {
                None => assert!(!active, "speaker {speaker_idx}: expected inactive"),
                Some(expected) => {
                    assert!(active, "speaker {speaker_idx}: expected active");
                    assert_eq!(
                        dest[..expected.len()],
                        expected[..],
                        "speaker {speaker_idx}: mask mismatch"
                    );
                }
            }
        }
    }

    #[test]
    fn single_active_speaker() {
        // speaker 0 active on all frames, speakers 1 and 2 inactive
        let mut seg = Array2::<f32>::zeros((20, 3));
        for i in 0..20 {
            seg[[i, 0]] = 1.0;
        }
        assert_matches_original(&seg, 160_000, 640);
    }

    #[test]
    fn inactive_speaker_below_threshold() {
        // speaker with activity sum < 10.0
        let mut seg = Array2::<f32>::zeros((20, 3));
        for i in 0..9 {
            seg[[i, 0]] = 1.0;
        }
        assert_matches_original(&seg, 160_000, 640);
    }

    #[test]
    fn overlapping_speakers_uses_clean_mask() {
        // frames where two speakers are active should be zeroed in clean mask
        let mut seg = Array2::<f32>::zeros((20, 3));
        for i in 0..20 {
            seg[[i, 0]] = 1.0;
        }
        // speaker 1 overlaps on frames 5-9
        for i in 5..10 {
            seg[[i, 1]] = 1.0;
        }
        // speaker 1 also active alone on 10-19
        for i in 10..20 {
            seg[[i, 1]] = 1.0;
        }
        assert_matches_original(&seg, 160_000, 640);
    }

    #[test]
    fn fallback_to_raw_mask_when_clean_too_sparse() {
        // clean mask has too few active frames, should fall back to raw
        let mut seg = Array2::<f32>::zeros((20, 3));
        for i in 0..20 {
            seg[[i, 0]] = 1.0;
            seg[[i, 1]] = 1.0; // overlap on all frames
        }
        // speaker 0 has zero clean frames but 20 raw frames
        assert_matches_original(&seg, 160_000, 640);
    }

    #[test]
    fn zero_audio_len() {
        let mut seg = Array2::<f32>::zeros((20, 3));
        for i in 0..20 {
            seg[[i, 0]] = 1.0;
        }
        assert_matches_original(&seg, 0, 640);
    }

    #[test]
    fn realistic_three_speaker_scenario() {
        // simulate a realistic window with three speakers
        let mut seg = Array2::<f32>::zeros((589, 3));
        // speaker 0: active frames 0-300
        for i in 0..300 {
            seg[[i, 0]] = 1.0;
        }
        // speaker 1: active frames 200-500 (overlaps with 0 on 200-300)
        for i in 200..500 {
            seg[[i, 1]] = 1.0;
        }
        // speaker 2: active frames 450-589 (overlaps with 1 on 450-500)
        for i in 450..589 {
            seg[[i, 2]] = 1.0;
        }
        assert_matches_original(&seg, 160_000, 640);
    }
}
