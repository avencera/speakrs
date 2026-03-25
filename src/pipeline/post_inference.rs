use ndarray::Array2;
use tracing::info;

use crate::binarize::binarize;
use crate::clustering::plda::PldaTransform;
use crate::reconstruct::Reconstructor;
use crate::segment::merge_segments;

use super::config::*;
use super::types::*;

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

    let training_embeddings = embeddings.training_set(&segmentations);
    let hard_clusters = training_embeddings.cluster(&segmentations, &embeddings, plda, config);

    let reconstructor =
        Reconstructor::with_clusters(&segmentations, &hard_clusters, &layout.start_frames, 0);
    let discrete_diarization = match config.reconstruct_method {
        ReconstructMethod::Smoothed { epsilon } => {
            reconstructor.reconstruct_smoothed(&speaker_count, epsilon)
        }
        ReconstructMethod::Standard => reconstructor.reconstruct(&speaker_count),
    };

    // apply min-duration filtering to remove single-frame speaker flickers
    let has_duration_filter =
        config.binarize.min_duration_on > 0 || config.binarize.min_duration_off > 0;
    let discrete_diarization = if has_duration_filter {
        DiscreteDiarization(binarize(&discrete_diarization, &config.binarize))
    } else {
        discrete_diarization
    };

    let segments = discrete_diarization.to_segments(FRAME_STEP_SECONDS, FRAME_DURATION_SECONDS);
    let segments = merge_segments(&segments, config.merge_gap);

    info!(
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
