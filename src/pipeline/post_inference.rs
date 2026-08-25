use ndarray::Array2;
use tracing::debug;

use crate::binarize::binarize;
use crate::clustering::plda::PldaTransform;
use crate::reconstruct::{Reconstructor, exclusive_from, resolve_exclusive_conflicts};
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
            exclusive_segments: Vec::new(),
        });
    }

    let training_embeddings = embeddings.training_set(&segmentations);
    let hard_clusters = training_embeddings.cluster(&segmentations, &embeddings, plda, config);

    let reconstructor =
        Reconstructor::with_clusters(&segmentations, &hard_clusters, &layout.start_frames, 0);
    // One activation pass feeds both reconstructions; the exclusive variant needs the
    // continuous scores, which a reconstruction has already flattened to 1.0.
    let activations = reconstructor.frame_activations(&speaker_count);
    let discrete_diarization = match config.reconstruct_method {
        ReconstructMethod::Smoothed { epsilon } => {
            reconstructor.reconstruct_smoothed_with(&activations, &speaker_count, epsilon)
        }
        ReconstructMethod::Standard => {
            reconstructor.reconstruct_with(&activations, &speaker_count)
        }
    };
    let exclusive_diarization = exclusive_from(&discrete_diarization, &activations);

    // apply min-duration filtering to remove single-frame speaker flickers
    let has_duration_filter =
        config.binarize.min_duration_on > 0 || config.binarize.min_duration_off > 0;
    let (discrete_diarization, exclusive_diarization) = if has_duration_filter {
        (
            DiscreteDiarization(binarize(&discrete_diarization, &config.binarize)),
            DiscreteDiarization(binarize(&exclusive_diarization, &config.binarize)),
        )
    } else {
        (discrete_diarization, exclusive_diarization)
    };
    // binarize runs per-speaker independently and can pad/extend two speakers' regions into the
    // same frame, undoing the exclusivity exclusive_from established — re-resolve any conflicts.
    let exclusive_diarization = resolve_exclusive_conflicts(&exclusive_diarization, &activations);

    let segments = discrete_diarization.to_segments();
    let segments = merge_segments(&segments, config.merge_gap);
    let exclusive_segments = exclusive_diarization.to_segments();
    let exclusive_segments = merge_segments(&exclusive_segments, config.merge_gap);

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
        exclusive_segments,
    })
}
