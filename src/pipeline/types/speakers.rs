use ndarray::Array2;

use super::DiarizationResult;
use crate::segment::{Segment, speaker_label};

/// One speaker's voice in a recording, for matching speakers across recordings
#[derive(Debug, Clone, PartialEq)]
pub struct SpeakerCentroid {
    /// The label this speaker's segments carry, such as `SPEAKER_00`
    pub label: String,
    /// Mean of the speaker's chunk embeddings, L2-normalised, so the cosine
    /// similarity of two centroids is their dot product
    pub embedding: Vec<f32>,
    /// Number of chunk embeddings averaged into `embedding`
    pub chunk_count: usize,
    /// Total duration of the speaker's `segments` in seconds, overlapped speech
    /// included
    pub speech_seconds: f64,
}

impl DiarizationResult {
    /// The centroid of each speaker cluster, in label order
    ///
    /// Clustering assigns each active chunk-speaker slot to a cluster, and cluster
    /// `k` becomes the speaker labeled `SPEAKER_k` in `segments` and
    /// `exclusive_segments`. A centroid averages the raw embeddings of the slots
    /// assigned to its cluster, skipping unassigned slots and non-finite embeddings,
    /// then L2-normalises the mean
    ///
    /// Every centroid's label appears in `segments`. A cluster whose segments
    /// activity cleanup removed has no centroid. A label in `segments` has no
    /// centroid when no finite embedding went into its cluster, as with the empty
    /// column reconstruction adds when a frame holds more speakers than clustering
    /// kept, or when its embeddings average to zero
    pub fn speaker_centroids(&self) -> Vec<SpeakerCentroid> {
        let width = self.embeddings.shape()[2];
        let num_clusters = self
            .hard_clusters
            .iter()
            .filter_map(|&cluster| usize::try_from(cluster).ok())
            .max()
            .map_or(0, |cluster| cluster + 1);

        // f64 sums keep hours of chunks from losing precision
        let mut sums = Array2::<f64>::zeros((num_clusters, width));
        let mut counts = vec![0usize; num_clusters];
        let chunks = self
            .hard_clusters
            .outer_iter()
            .zip(self.embeddings.outer_iter());
        for (chunk_clusters, chunk_embeddings) in chunks {
            for (&cluster, embedding) in chunk_clusters.iter().zip(chunk_embeddings.outer_iter()) {
                // negative ids mark inactive or unassigned slots
                let Ok(cluster) = usize::try_from(cluster) else {
                    continue;
                };
                if !embedding.iter().all(|value| value.is_finite()) {
                    continue;
                }

                sums.row_mut(cluster)
                    .zip_mut_with(&embedding, |sum, &value| *sum += f64::from(value));
                counts[cluster] += 1;
            }
        }

        counts
            .iter()
            .enumerate()
            .filter_map(|(cluster, &chunk_count)| {
                let sum = sums.row(cluster);
                let norm = sum.dot(&sum).sqrt();
                // a zero mean has no direction to compare
                if chunk_count == 0 || norm <= 0.0 {
                    return None;
                }

                // activity cleanup can remove every frame of a cluster, and a
                // speaker with no segments is not in the diarization
                let label = speaker_label(cluster);
                let mut speaker_segments = self
                    .segments
                    .iter()
                    .filter(|segment| segment.speaker == label)
                    .peekable();
                speaker_segments.peek()?;

                let speech_seconds = speaker_segments.map(Segment::duration).sum();
                Some(SpeakerCentroid {
                    embedding: sum.iter().map(|value| (value / norm) as f32).collect(),
                    label,
                    chunk_count,
                    speech_seconds,
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Array2, Array3, array};

    use crate::pipeline::{
        ChunkEmbeddings, ChunkSpeakerClusters, DecodedSegmentations, DiarizationResult,
        DiscreteDiarization, SpeakerCountTrack,
    };
    use crate::segment::Segment;

    use super::SpeakerCentroid;

    fn result(
        hard_clusters: Array2<i32>,
        embeddings: Array3<f32>,
        segments: Vec<Segment>,
    ) -> DiarizationResult {
        DiarizationResult {
            segmentations: DecodedSegmentations(Array3::zeros((0, 0, 0))),
            embeddings: ChunkEmbeddings(embeddings),
            speaker_count: SpeakerCountTrack(Vec::new()),
            hard_clusters: ChunkSpeakerClusters(hard_clusters),
            discrete_diarization: DiscreteDiarization(Array2::zeros((0, 0))),
            segments,
            exclusive_segments: Vec::new(),
        }
    }

    #[test]
    fn averages_assigned_finite_embeddings_per_cluster() {
        let nan = f32::NAN;
        // local speakers swap clusters between chunks, the third slot is inactive in
        // the first chunk, and a non-finite embedding in the second chunk is skipped
        let hard_clusters = array![[0, 1, -2], [1, 0, 0]];
        let embeddings = array![
            [[3.0, 0.0], [0.0, 2.0], [9.0, 9.0]],
            [[0.0, 4.0], [1.0, 4.0], [nan, 1.0]],
        ];
        // SPEAKER_02 has speech but no cluster, as a padded reconstruction column does
        let segments = vec![
            Segment::new(0.0, 2.0, "SPEAKER_00"),
            Segment::new(1.5, 2.5, "SPEAKER_01"),
            Segment::new(3.0, 3.5, "SPEAKER_00"),
            Segment::new(4.0, 5.0, "SPEAKER_02"),
        ];

        let centroids = result(hard_clusters, embeddings, segments).speaker_centroids();

        // SPEAKER_00 sums to (4, 4) and SPEAKER_01 to (0, 6)
        let diagonal = std::f32::consts::FRAC_1_SQRT_2;
        assert_eq!(
            centroids,
            [
                SpeakerCentroid {
                    label: "SPEAKER_00".to_owned(),
                    embedding: vec![diagonal, diagonal],
                    chunk_count: 2,
                    speech_seconds: 2.5,
                },
                SpeakerCentroid {
                    label: "SPEAKER_01".to_owned(),
                    embedding: vec![0.0, 1.0],
                    chunk_count: 2,
                    speech_seconds: 1.0,
                },
            ]
        );
    }

    #[test]
    fn skips_clusters_without_segments_or_direction() {
        // activity cleanup removed every frame of cluster 1, and cluster 2's
        // embeddings cancel out
        let hard_clusters = array![[0, 1, 2], [0, 1, 2]];
        let embeddings = array![
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]],
        ];
        let segments = vec![
            Segment::new(0.0, 1.0, "SPEAKER_00"),
            Segment::new(1.0, 2.0, "SPEAKER_02"),
        ];

        let centroids = result(hard_clusters, embeddings, segments).speaker_centroids();

        assert_eq!(
            centroids,
            [SpeakerCentroid {
                label: "SPEAKER_00".to_owned(),
                embedding: vec![1.0, 0.0],
                chunk_count: 2,
                speech_seconds: 1.0,
            }]
        );
    }

    #[test]
    fn no_speech_has_no_centroids() {
        let empty = result(
            Array2::zeros((0, 0)),
            Array3::zeros((3, 3, 256)),
            Vec::new(),
        );

        assert!(empty.speaker_centroids().is_empty());
    }
}
