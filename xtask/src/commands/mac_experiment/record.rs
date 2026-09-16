use serde::{Deserialize, Serialize};

use crate::commands::benchmark::PerFileDerResult;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub(super) enum ExperimentRecord {
    Complete {
        #[serde(default)]
        audio_seconds: f64,
        #[serde(default)]
        wall_seconds: f64,
        #[serde(default)]
        rtfx: f64,
        #[serde(default)]
        worker_elapsed_seconds: f64,
        stage_timings: StageTimings,
        #[serde(default)]
        clustering: Option<ClusteringDiagnostics>,
        #[serde(default)]
        peak_rss_bytes: Option<u64>,
        der: Box<PerFileDerResult>,
        #[serde(default)]
        hypothesis_rttm: String,
        #[serde(default)]
        fallback_events: Vec<String>,
    },
    Failed {
        error: String,
        #[serde(default)]
        audio_seconds: f64,
        #[serde(default)]
        worker_elapsed_seconds: f64,
        #[serde(default)]
        inference_seconds: Option<f64>,
        #[serde(default)]
        peak_rss_bytes: Option<u64>,
    },
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(super) struct ClusteringDiagnostics {
    pub usable_training_embeddings: usize,
    #[serde(default)]
    pub clean_frame_seconds: f64,
    #[serde(flatten)]
    pub backend: Option<ClusteringBackendDiagnostics>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "backend", rename_all = "snake_case")]
pub(super) enum ClusteringBackendDiagnostics {
    GaussianVbx {
        fa: f64,
        fb: f64,
        max_iters: usize,
    },
    SphereVbxPf {
        fa: f64,
        fb: f64,
        max_iters: usize,
        responsibility_tolerance: f64,
        initialization: SphereInitializationDiagnostics,
        ahc_initialization: SphereAhcInitializationDiagnostics,
    },
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(super) enum SphereInitializationDiagnostics {
    Hard,
    Smoothed { scale: f64 },
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum SphereAhcInitializationDiagnostics {
    Cosine,
    PldaTransformed,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub(super) struct StageTimings {
    #[serde(default)]
    pub inference_seconds: f64,
    pub post_inference_seconds: f64,
    #[serde(default)]
    pub chunk_inference: Option<ChunkInferenceStageTimings>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub(super) struct ChunkInferenceStageTimings {
    pub segmentation_seconds: f64,
    pub embedding_seconds: f64,
    pub prediction_seconds: f64,
    pub filterbank_preparation_seconds: f64,
    pub mask_preparation_seconds: f64,
    pub total_seconds: f64,
    pub chunk_count: usize,
    pub pipelined: bool,
}

impl From<speakrs::pipeline::InferenceStageTimings> for ChunkInferenceStageTimings {
    fn from(value: speakrs::pipeline::InferenceStageTimings) -> Self {
        Self {
            segmentation_seconds: value.segmentation_seconds,
            embedding_seconds: value.embedding_seconds,
            prediction_seconds: value.prediction_seconds,
            filterbank_preparation_seconds: value.filterbank_preparation_seconds,
            mask_preparation_seconds: value.mask_preparation_seconds,
            total_seconds: value.total_seconds,
            chunk_count: value.chunk_count,
            pipelined: value.pipelined,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(super) struct RecordDocument {
    pub schema_version: u32,
    pub repetition: u32,
    pub file_index: usize,
    pub file_id: String,
    pub candidate_id: String,
    #[serde(flatten)]
    pub outcome: ExperimentRecord,
}
