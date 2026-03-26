use std::path::PathBuf;
use std::sync::Arc;

use crate::inference::coreml::{CachedInputShape, SharedCoreMlModel};

/// Chunk embedding model: runs ResNet once on full-audio fbank, gathers per-window features
pub(crate) struct ChunkEmbeddingSession {
    pub(crate) model: Arc<SharedCoreMlModel>,
    pub num_windows: usize,
    pub fbank_frames: usize,
    pub num_masks: usize,
    pub(crate) cached_fbank_shape: Arc<CachedInputShape>,
    pub(crate) cached_masks_shape: Arc<CachedInputShape>,
}

#[derive(Clone)]
pub(super) struct ChunkSessionSpec {
    pub coreml_path: PathBuf,
    pub num_windows: usize,
    pub fbank_frames: usize,
    pub num_masks: usize,
}

/// All resources needed for chunk embedding, returned by `prepare_chunk_resources`
pub(crate) struct ChunkResourceBundle {
    pub sessions: Vec<ChunkSessionInfo>,
    pub fbank_30s: Option<Arc<SharedCoreMlModel>>,
    pub fbank_10s: Option<Arc<SharedCoreMlModel>>,
}

pub(crate) struct ChunkSessionInfo {
    pub model: Arc<SharedCoreMlModel>,
    pub cached_fbank_shape: Arc<CachedInputShape>,
    pub cached_masks_shape: Arc<CachedInputShape>,
    pub num_windows: usize,
    pub fbank_frames: usize,
    pub num_masks: usize,
}
