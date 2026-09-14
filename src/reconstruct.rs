use ndarray::{Array2, s};

use crate::pipeline::{
    ChunkSpeakerClusters, DecodedSegmentations, DiscreteDiarization, FrameActivations,
    PipelineGeometry, SpeakerCountTrack,
};

/// Invalid reconstruction inputs
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ReconstructError {
    /// Cluster rows do not match the number of segmentation chunks
    #[error("cluster rows {actual} do not match segmentation chunks {expected}")]
    ClusterRowsMismatch {
        /// Observed cluster row count
        actual: usize,
        /// Required cluster row count
        expected: usize,
    },
    /// Cluster columns do not match the number of local speakers
    #[error("cluster columns {actual} do not match local speakers {expected}")]
    ClusterColumnsMismatch {
        /// Observed cluster column count
        actual: usize,
        /// Required cluster column count
        expected: usize,
    },
    /// Geometry chunk count does not match the number of segmentation chunks
    #[error("geometry chunk count {actual} does not match segmentation chunks {expected}")]
    GeometryChunksMismatch {
        /// Observed geometry chunk count
        actual: usize,
        /// Required segmentation chunk count
        expected: usize,
    },
    /// Segmentation frame count does not match the local geometry grid
    #[error("segmentation frame count {actual} does not match geometry frames {expected}")]
    SegmentationFramesMismatch {
        /// Observed segmentation frame count
        actual: usize,
        /// Required local frame count
        expected: usize,
    },
    /// Geometry timing could not be materialized
    #[error("invalid geometry timing: {0}")]
    GeometryTiming(String),
}

pub struct Reconstructor<'a> {
    segmentations: &'a DecodedSegmentations,
    hard_clusters: &'a ChunkSpeakerClusters,
    geometry: &'a PipelineGeometry,
}

impl<'a> Reconstructor<'a> {
    pub fn new(
        segmentations: &'a DecodedSegmentations,
        hard_clusters: &'a ChunkSpeakerClusters,
        geometry: &'a PipelineGeometry,
    ) -> Result<Self, ReconstructError> {
        let num_chunks = segmentations.shape()[0];
        if hard_clusters.nrows() != num_chunks {
            return Err(ReconstructError::ClusterRowsMismatch {
                actual: hard_clusters.nrows(),
                expected: num_chunks,
            });
        }
        if hard_clusters.ncols() != segmentations.shape()[2] {
            return Err(ReconstructError::ClusterColumnsMismatch {
                actual: hard_clusters.ncols(),
                expected: segmentations.shape()[2],
            });
        }
        if geometry.chunk_count() != num_chunks {
            return Err(ReconstructError::GeometryChunksMismatch {
                actual: geometry.chunk_count(),
                expected: num_chunks,
            });
        }
        if num_chunks > 0 && segmentations.shape()[1] != geometry.frame_grid().frame_count as usize
        {
            return Err(ReconstructError::SegmentationFramesMismatch {
                actual: segmentations.shape()[1],
                expected: geometry.frame_grid().frame_count as usize,
            });
        }
        geometry
            .frame_timing()
            .map_err(|error| ReconstructError::GeometryTiming(error.to_string()))?;
        Ok(Self {
            segmentations,
            hard_clusters,
            geometry,
        })
    }

    pub(crate) fn frame_activations(&self, speaker_count: &SpeakerCountTrack) -> FrameActivations {
        let num_chunks = self.segmentations.shape()[0];
        let num_frames = self.segmentations.shape()[1];
        let num_clusters = self
            .hard_clusters
            .iter()
            .copied()
            .filter(|cluster| *cluster >= 0)
            .max()
            .map_or(0, |cluster| cluster as usize + 1);
        let mut activations = Array2::<f32>::zeros((self.geometry.output_frames(), num_clusters));

        for (chunk_idx, &start_frame) in self
            .geometry
            .start_frames()
            .iter()
            .enumerate()
            .take(num_chunks)
        {
            let chunk_labels = self.hard_clusters.row(chunk_idx);
            let chunk_segmentations = self.segmentations.slice(s![chunk_idx, .., ..]);
            let local_cluster_mapping = build_cluster_mapping(&chunk_labels, num_clusters);

            for (cluster_idx, local_indices) in local_cluster_mapping.iter().enumerate() {
                if local_indices.is_empty() {
                    continue;
                }

                for frame_idx in 0..num_frames {
                    let out_frame = start_frame + frame_idx;
                    if out_frame >= speaker_count.len() {
                        continue;
                    }

                    let mut score = 0.0f32;
                    for &local_idx in local_indices {
                        score = score.max(chunk_segmentations[[frame_idx, local_idx]]);
                    }
                    activations[[out_frame, cluster_idx]] += score;
                }
            }
        }

        let max_speakers_per_frame = speaker_count.iter().copied().max().unwrap_or(0);
        if activations.ncols() < max_speakers_per_frame {
            let mut padded = Array2::<f32>::zeros((activations.nrows(), max_speakers_per_frame));
            padded
                .slice_mut(s![.., ..activations.ncols()])
                .assign(&activations);
            activations = padded;
        }

        FrameActivations(activations)
    }

    pub fn reconstruct(&self, speaker_count: &SpeakerCountTrack) -> DiscreteDiarization {
        let activations = self.frame_activations(speaker_count);
        let mut discrete = Array2::<f32>::zeros(activations.raw_dim());
        for (frame_idx, &count) in speaker_count.iter().enumerate() {
            for speaker_idx in top_k_indices(&activations, frame_idx, count) {
                discrete[[frame_idx, speaker_idx]] = 1.0;
            }
        }
        DiscreteDiarization::with_timing(
            discrete,
            self.geometry
                .frame_timing()
                .expect("validated geometry timing"),
        )
    }

    pub fn reconstruct_smoothed(
        &self,
        speaker_count: &SpeakerCountTrack,
        epsilon: f32,
    ) -> DiscreteDiarization {
        let activations = self.frame_activations(speaker_count);
        let mut discrete = Array2::<f32>::zeros(activations.raw_dim());
        let mut previous_speakers: Vec<usize> = Vec::new();

        for (frame_idx, &count) in speaker_count.iter().enumerate() {
            let current_speakers =
                top_k_indices_smoothed(&activations, frame_idx, count, &previous_speakers, epsilon);
            for &speaker_idx in &current_speakers {
                discrete[[frame_idx, speaker_idx]] = 1.0;
            }
            previous_speakers = current_speakers;
        }

        DiscreteDiarization::with_timing(
            discrete,
            self.geometry
                .frame_timing()
                .expect("validated geometry timing"),
        )
    }
}

/// Zero out all but the highest-scoring speaker in each frame, making activations exclusive
pub fn make_exclusive(activations: &mut Array2<f32>) {
    for mut row in activations.rows_mut() {
        let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if max_val == 0.0 {
            continue;
        }

        let argmax = row
            .iter()
            .enumerate()
            .max_by(|(_, lhs), (_, rhs)| lhs.total_cmp(rhs))
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        for (column_idx, value) in row.iter_mut().enumerate() {
            if column_idx != argmax {
                *value = 0.0;
            }
        }
    }
}

fn build_cluster_mapping(
    chunk_labels: &ndarray::ArrayView1<i32>,
    num_clusters: usize,
) -> Vec<Vec<usize>> {
    let mut mapping = vec![Vec::new(); num_clusters];
    for (local_idx, &label) in chunk_labels.iter().enumerate() {
        if label >= 0 {
            mapping[label as usize].push(local_idx);
        }
    }
    mapping
}

fn top_k_indices(matrix: &Array2<f32>, frame_idx: usize, k: usize) -> Vec<usize> {
    let num_columns = matrix.ncols();
    if k >= num_columns {
        return (0..num_columns).collect();
    }

    let mut indexed: Vec<(usize, f32)> = (0..num_columns)
        .map(|column_idx| (column_idx, matrix[[frame_idx, column_idx]]))
        .collect();
    indexed.sort_by(|left, right| right.1.total_cmp(&left.1));

    indexed.into_iter().take(k).map(|(idx, _)| idx).collect()
}

fn top_k_indices_smoothed(
    matrix: &Array2<f32>,
    frame_idx: usize,
    k: usize,
    previous_speakers: &[usize],
    epsilon: f32,
) -> Vec<usize> {
    let num_columns = matrix.ncols();
    if k == 0 {
        return Vec::new();
    }
    if k >= num_columns {
        return (0..num_columns).collect();
    }

    let mut ranked: Vec<(usize, f32)> = (0..num_columns)
        .map(|column_idx| (column_idx, matrix[[frame_idx, column_idx]]))
        .collect();
    ranked.sort_by(|left, right| right.1.total_cmp(&left.1).then(left.0.cmp(&right.0)));

    let mut selected: Vec<(usize, f32)> = ranked.iter().copied().take(k).collect();
    let unselected_priors: Vec<(usize, f32)> = ranked
        .iter()
        .copied()
        .filter(|(speaker, _)| {
            previous_speakers.contains(speaker)
                && selected
                    .iter()
                    .all(|(selected_speaker, _)| selected_speaker != speaker)
        })
        .collect();

    for prior in unselected_priors {
        let Some(weakest_pos) = selected
            .iter()
            .enumerate()
            .rev()
            .find(|(_, (speaker, _))| !previous_speakers.contains(speaker))
            .map(|(pos, _)| pos)
        else {
            break;
        };
        let weak_score = selected[weakest_pos].1;
        if (weak_score - prior.1).abs() < epsilon {
            selected[weakest_pos] = prior;
        }
    }

    selected.sort_by(|left, right| right.1.total_cmp(&left.1).then(left.0.cmp(&right.0)));
    selected.into_iter().map(|(speaker, _)| speaker).collect()
}

pub(crate) fn aggregate_speaker_count(
    segmentations: &DecodedSegmentations,
    geometry: &PipelineGeometry,
) -> SpeakerCountTrack {
    let num_chunks = segmentations.shape()[0];
    if num_chunks == 0 {
        return SpeakerCountTrack(Vec::new());
    }

    let num_frames = segmentations.shape()[1];
    let mut numerator = vec![0.0f32; geometry.output_frames()];
    let mut denominator = vec![0.0f32; geometry.output_frames()];

    for (chunk_idx, &start_frame) in geometry.start_frames().iter().enumerate().take(num_chunks) {
        for frame_idx in 0..num_frames {
            let out_frame = start_frame + frame_idx;
            if out_frame >= geometry.output_frames() {
                continue;
            }

            numerator[out_frame] += segmentations
                .slice(s![chunk_idx, frame_idx, ..])
                .iter()
                .sum::<f32>();
            denominator[out_frame] += 1.0;
        }
    }

    SpeakerCountTrack(
        numerator
            .into_iter()
            .zip(denominator)
            .map(|(sum, weight)| {
                if weight == 0.0 {
                    0
                } else {
                    round_ties_even(sum / weight).max(0.0) as usize
                }
            })
            .collect(),
    )
}

fn round_ties_even(value: f32) -> f32 {
    let lower = value.floor();
    let fraction = value - lower;

    if fraction < 0.5 {
        return lower;
    }

    if fraction > 0.5 {
        return value.ceil();
    }

    if lower as i64 % 2 == 0 {
        lower
    } else {
        lower + 1.0
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Array2, array};

    use super::*;
    use crate::pipeline::{ChunkSpeakerClusters, DecodedSegmentations};

    #[test]
    fn speaker_count_rounds_overlap_added_sum() {
        let segmentations = DecodedSegmentations(array![
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0], [0.0, 1.0], [1.0, 0.0]],
        ]);
        let geometry = PipelineGeometry::from_legacy(16_000, 160_000, 270, 160_001).unwrap();
        let count = aggregate_speaker_count(&segmentations, &geometry);

        assert_eq!(&count[..4], &[1, 1, 1, 1]);
    }

    #[test]
    fn speaker_count_rounds_nearest_even_ties() {
        let mut values = ndarray::Array3::<f32>::zeros((1, 589, 2));
        values.slice_mut(s![0, 0, ..]).assign(&array![0.5, 0.0]);
        values.slice_mut(s![0, 1, ..]).assign(&array![1.5, 0.0]);
        let segmentations = DecodedSegmentations(values);
        let geometry = PipelineGeometry::from_legacy(16_000, 160_000, 270, 160_000).unwrap();

        let count = aggregate_speaker_count(&segmentations, &geometry);

        assert_eq!(&count[..2], &[0, 2]);
    }

    #[test]
    fn reconstruct_selects_top_k_per_frame() {
        let mut values = ndarray::Array3::<f32>::zeros((2, 589, 2));
        values.slice_mut(s![0, 0, ..]).assign(&array![1.0, 0.0]);
        values.slice_mut(s![0, 1, ..]).assign(&array![0.5, 0.5]);
        values.slice_mut(s![1, 0, ..]).assign(&array![0.0, 1.0]);
        values.slice_mut(s![1, 1, ..]).assign(&array![0.2, 0.8]);
        let segmentations = DecodedSegmentations(values);
        let hard_clusters = ChunkSpeakerClusters(array![[0, 1], [0, 1]]);
        let geometry = PipelineGeometry::from_legacy(16_000, 160_000, 270, 160_001).unwrap();
        let reconstructor = Reconstructor::new(&segmentations, &hard_clusters, &geometry).unwrap();
        let speaker_count = SpeakerCountTrack(vec![1; geometry.output_frames()]);

        let result = reconstructor.reconstruct(&speaker_count);

        let expected: Array2<f32> = array![[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]];
        assert_eq!(result.slice(s![..3, ..]), expected);
    }

    #[test]
    fn reconstructor_rejects_incomplete_inputs() {
        let segmentations =
            DecodedSegmentations(array![[[1.0, 0.0], [0.5, 0.5]], [[0.0, 1.0], [0.2, 0.8]]]);
        let hard_clusters = ChunkSpeakerClusters(array![[0, 1]]);
        let geometry = PipelineGeometry::from_legacy(16_000, 160_000, 160_000, 320_000).unwrap();
        let error = match Reconstructor::new(&segmentations, &hard_clusters, &geometry) {
            Ok(_) => panic!("expected incomplete reconstruction inputs to fail"),
            Err(error) => error,
        };
        assert_eq!(
            error,
            ReconstructError::ClusterRowsMismatch {
                actual: 1,
                expected: 2,
            }
        );
        assert_eq!(
            error.to_string(),
            "cluster rows 1 do not match segmentation chunks 2"
        );
    }

    #[test]
    fn reconstructor_rejects_mismatched_cluster_columns() {
        let segmentations = DecodedSegmentations(array![[[1.0, 0.0], [0.5, 0.5]]]);
        let hard_clusters = ChunkSpeakerClusters(array![[0]]);
        let geometry = PipelineGeometry::from_legacy(16_000, 160_000, 160_000, 160_000).unwrap();
        let error = match Reconstructor::new(&segmentations, &hard_clusters, &geometry) {
            Ok(_) => panic!("expected mismatched cluster columns to fail"),
            Err(error) => error,
        };

        assert_eq!(
            error,
            ReconstructError::ClusterColumnsMismatch {
                actual: 1,
                expected: 2,
            }
        );
    }

    #[test]
    fn reconstructor_rejects_mismatched_start_frames() {
        let segmentations = DecodedSegmentations(array![[[1.0, 0.0], [0.5, 0.5]]]);
        let hard_clusters = ChunkSpeakerClusters(array![[0, 1]]);
        let geometry = PipelineGeometry::from_legacy(16_000, 160_000, 160_000, 0).unwrap();
        let error = match Reconstructor::new(&segmentations, &hard_clusters, &geometry) {
            Ok(_) => panic!("expected mismatched start frames to fail"),
            Err(error) => error,
        };

        assert_eq!(
            error,
            ReconstructError::GeometryChunksMismatch {
                actual: 0,
                expected: 1,
            }
        );
    }

    #[test]
    fn chained_near_tie_smoothing_is_permutation_invariant() {
        let scores = [0.00_f32, 0.09, 0.18];
        let epsilon = 0.1;
        let k = 2;
        let previous = [0usize];
        let mut selected_sets = Vec::new();

        for permutation in [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ] {
            let mut matrix = Array2::<f32>::zeros((1, 3));
            for (column, &source) in permutation.iter().enumerate() {
                matrix[[0, column]] = scores[source];
            }
            let previous_columns: Vec<usize> = permutation
                .iter()
                .enumerate()
                .filter_map(|(column, &source)| (source == previous[0]).then_some(column))
                .collect();
            let selected = top_k_indices_smoothed(&matrix, 0, k, &previous_columns, epsilon);
            let mut original: Vec<usize> =
                selected.iter().map(|&column| permutation[column]).collect();
            original.sort_unstable();
            selected_sets.push(original);
        }

        for selected in &selected_sets {
            assert_eq!(selected, &selected_sets[0]);
        }
    }
}
