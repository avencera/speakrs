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
        geometry,
        segmentations,
        embeddings,
        embedding_availability,
        embedding_receipt,
        #[cfg(feature = "_metrics")]
            stage_timings: _,
    } = inference_artifacts;
    let speaker_count = segmentations.speaker_count(&geometry);

    if speaker_count
        .iter()
        .all(|speaker_count| *speaker_count == 0)
    {
        let discrete_diarization =
            DiscreteDiarization::try_new(Array2::zeros((geometry.output_frames(), 0)), &geometry)?;
        return Ok(DiarizationResult {
            segmentations,
            embeddings,
            embedding_availability,
            embedding_receipt,
            speaker_count,
            hard_clusters: ChunkSpeakerClusters(Array2::zeros((0, 0))),
            discrete_diarization,
            segments: Vec::new(),
            geometry,
        });
    }

    let training_embeddings = embeddings.training_set(
        &segmentations,
        &geometry,
        config.effective_clean_frame_duration(),
    );
    let hard_clusters = training_embeddings.cluster(&segmentations, &embeddings, plda, config)?;

    let reconstructor = Reconstructor::new(&segmentations, &hard_clusters, &geometry)?;
    let discrete_diarization = match config.reconstruct_method {
        ReconstructMethod::Smoothed { epsilon } => {
            reconstructor.reconstruct_smoothed(&speaker_count, epsilon)
        }
        ReconstructMethod::Standard => reconstructor.reconstruct(&speaker_count),
    };

    let discrete_diarization = apply_activity_cleanup(discrete_diarization, config.activity)?;

    let segments = discrete_diarization.to_segments();
    let segments = merge_segments(&segments, config.merge_gap);

    debug!(
        post_inference_ms = post_start.elapsed().as_millis(),
        "Post-inference complete"
    );

    Ok(DiarizationResult {
        segmentations,
        embeddings,
        embedding_availability,
        embedding_receipt,
        speaker_count,
        hard_clusters,
        discrete_diarization,
        segments,
        geometry,
    })
}

pub(crate) fn apply_activity_cleanup(
    discrete: DiscreteDiarization,
    config: ActivityCleanup,
) -> Result<DiscreteDiarization, PipelineError> {
    if config.is_identity() {
        Ok(discrete)
    } else {
        discrete.map_activations(config.apply(&discrete))
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
        let geometry =
            crate::pipeline::PipelineGeometry::from_legacy(16_000, 160_000, 16_000, 48_000)
                .unwrap();
        let discrete = DiscreteDiarization::with_timing(
            array![[0.0], [1.0], [0.0]],
            geometry.frame_timing().unwrap(),
        );
        let cleaned = apply_activity_cleanup(discrete.clone(), ActivityCleanup::default()).unwrap();
        assert_eq!(&*cleaned, &*discrete);
        assert_eq!(cleaned.timing(), discrete.timing());
    }

    #[test]
    fn padding_only_cleanup_is_applied() {
        let config = ActivityCleanup::new(0, 0, 1, 1);
        assert!(!config.is_identity());
        let geometry =
            crate::pipeline::PipelineGeometry::from_legacy(16_000, 160_000, 16_000, 48_000)
                .unwrap();
        let discrete = DiscreteDiarization::with_timing(
            array![[0.0], [0.0], [1.0], [0.0], [0.0]],
            geometry.frame_timing().unwrap(),
        );
        let timing = discrete.timing();
        let cleaned = apply_activity_cleanup(discrete, config).unwrap();
        assert_eq!(&*cleaned, &array![[0.0], [1.0], [1.0], [1.0], [0.0]]);
        assert_eq!(cleaned.timing(), timing);
    }
}
