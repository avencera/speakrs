use ndarray::Array2;
use tracing::debug;

use crate::binarize::ActivityCleanup;
use crate::clustering::plda::PldaTransform;
use crate::reconstruct::Reconstructor;
use crate::segment::merge_segments;

use super::config::{PipelineConfig, ReconstructMethod};
use super::types::{
    ChunkSpeakerClusters, DiarizationResult, DiscreteDiarization, InferenceArtifacts, PipelineError,
};

/// Run clustering and reconstruction on pre-computed inference artifacts
pub fn post_inference(
    inference_artifacts: InferenceArtifacts,
    config: &PipelineConfig,
    plda: &PldaTransform,
) -> Result<DiarizationResult, PipelineError> {
    let post_start = std::time::Instant::now();
    let InferenceArtifacts {
        layout,
        segmentations,
        embeddings,
        #[cfg(feature = "_metrics")]
            stage_timings: _,
    } = inference_artifacts;
    let speaker_count = segmentations.speaker_count(&layout);

    if speaker_count
        .iter()
        .all(|speaker_count| *speaker_count == 0)
    {
        return Ok(DiarizationResult {
            segmentations,
            embeddings,
            speaker_count,
            hard_clusters: ChunkSpeakerClusters(Array2::zeros((0, 0))),
            discrete_diarization: DiscreteDiarization(Array2::zeros((0, 0))),
            segments: Vec::new(),
        });
    }

    let training_embeddings =
        embeddings.training_set(&segmentations, config.effective_clean_frame_duration());
    let hard_clusters = training_embeddings.cluster(&segmentations, &embeddings, plda, config)?;

    let reconstructor = Reconstructor::new(&segmentations, &hard_clusters, &layout.start_frames)?;
    let discrete_diarization = match config.reconstruct_method {
        ReconstructMethod::Smoothed { epsilon } => {
            reconstructor.reconstruct_smoothed(&speaker_count, epsilon)
        }
        ReconstructMethod::Standard => reconstructor.reconstruct(&speaker_count),
    };

    let discrete_diarization = apply_activity_cleanup(discrete_diarization, config.activity);

    let segments = discrete_diarization.to_segments();
    let segments = merge_segments(&segments, config.merge_gap);

    debug!(
        post_inference_ms = post_start.elapsed().as_millis(),
        "Post-inference complete"
    );

    Ok(DiarizationResult {
        segmentations,
        embeddings,
        speaker_count,
        hard_clusters,
        discrete_diarization,
        segments,
    })
}

pub(super) fn apply_activity_cleanup(
    discrete: DiscreteDiarization,
    config: ActivityCleanup,
) -> DiscreteDiarization {
    if config.is_identity() {
        discrete
    } else {
        DiscreteDiarization(config.apply(&discrete))
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::apply_activity_cleanup;
    use crate::binarize::ActivityCleanup;
    use crate::pipeline::DiscreteDiarization;

    #[test]
    fn default_cleanup_is_identity() {
        assert!(ActivityCleanup::default().is_identity());
        let discrete = DiscreteDiarization(array![[0.0], [1.0], [0.0]]);
        let cleaned = apply_activity_cleanup(discrete.clone(), ActivityCleanup::default());
        assert_eq!(&*cleaned, &*discrete);
    }

    #[test]
    fn padding_only_cleanup_is_applied() {
        let config = ActivityCleanup::new(0, 0, 1, 1);
        assert!(!config.is_identity());
        let discrete = DiscreteDiarization(array![[0.0], [0.0], [1.0], [0.0], [0.0]]);
        let cleaned = apply_activity_cleanup(discrete, config);
        assert_eq!(&*cleaned, &array![[0.0], [1.0], [1.0], [1.0], [0.0]]);
    }
}
