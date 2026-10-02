use std::path::{Path, PathBuf};

use super::super::paths::{
    batched_model_path, multi_mask_model_path, split_fbank_batched_model_path,
    split_fbank_model_path, split_tail_model_path,
};
use super::super::{CHUNK_SPEAKER_BATCH_SIZE, MULTI_MASK_BATCH_SIZE, PRIMARY_BATCH_SIZE};

/// ONNX embedding models next to the fused model, one slot per session
///
/// `S` is a discovered path while scanning and a loaded session afterwards, so the routing
/// capabilities below read the same slots before and after load. Only the fused model is
/// required; every other slot is present only when its file exists
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct OrtEmbeddingPlan<S> {
    pub(super) fused: S,
    pub(super) fused_batched: Option<S>,
    pub(super) fbank: Option<S>,
    pub(super) fbank_batched: Option<S>,
    pub(super) tail: Option<S>,
    pub(super) tail_batched: Option<S>,
    pub(super) tail_primary_batched: Option<S>,
    pub(super) multi_mask: Option<S>,
    pub(super) multi_mask_batched: Option<S>,
}

impl OrtEmbeddingPlan<PathBuf> {
    /// Snapshot usable assets once. Path selection must not re-read the file system
    pub(super) fn scan(model_path: &Path) -> Self {
        Self {
            fused: model_path.to_path_buf(),
            fused_batched: batched_model_path(model_path, PRIMARY_BATCH_SIZE).and_then(existing),
            fbank: existing(split_fbank_model_path(model_path)),
            fbank_batched: existing(split_fbank_batched_model_path(model_path)),
            tail: existing(split_tail_model_path(model_path, 1)),
            tail_batched: existing(split_tail_model_path(model_path, CHUNK_SPEAKER_BATCH_SIZE)),
            tail_primary_batched: existing(split_tail_model_path(model_path, PRIMARY_BATCH_SIZE)),
            multi_mask: existing(multi_mask_model_path(model_path, 1)),
            multi_mask_batched: existing(multi_mask_model_path(model_path, MULTI_MASK_BATCH_SIZE)),
        }
    }
}

impl<S> OrtEmbeddingPlan<S> {
    pub(super) fn primary_batch_size(&self) -> usize {
        if self.fused_batched.is_some() {
            PRIMARY_BATCH_SIZE
        } else {
            1
        }
    }

    pub(super) fn prefers_chunk_embedding_path(&self) -> bool {
        self.fbank.is_some() && self.tail.is_some()
    }

    pub(super) fn split_primary_batch_size(&self) -> usize {
        if self.tail_primary_batched.is_some() {
            PRIMARY_BATCH_SIZE
        } else {
            0
        }
    }

    pub(super) fn has_batched_fbank(&self) -> bool {
        self.fbank_batched.is_some()
    }

    pub(super) fn prefers_multi_mask_path(&self) -> bool {
        self.multi_mask.is_some()
    }

    pub(super) fn multi_mask_batch_size(&self) -> usize {
        if self.multi_mask_batched.is_some() {
            MULTI_MASK_BATCH_SIZE
        } else if self.multi_mask.is_some() {
            1
        } else {
            0
        }
    }

    pub(super) fn has_batched_tail(&self) -> bool {
        self.tail_batched.is_some()
    }
}

fn existing(path: PathBuf) -> Option<PathBuf> {
    path.exists().then_some(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn scratch_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "speakrs-embedding-plan-{}-{}-{tag}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn touch(dir: &Path, name: &str) {
        fs::write(dir.join(name), []).unwrap();
    }

    fn plan_for(dir: &Path) -> OrtEmbeddingPlan<PathBuf> {
        OrtEmbeddingPlan::scan(&dir.join("wespeaker-voxceleb-resnet34.onnx"))
    }

    #[test]
    fn primary_only_inventory_disables_split_paths() {
        let dir = scratch_dir("primary");
        touch(&dir, "wespeaker-voxceleb-resnet34.onnx");
        let plan = plan_for(&dir);
        assert!(plan.fused.exists());
        assert!(!plan.prefers_chunk_embedding_path());
        assert!(!plan.prefers_multi_mask_path());
        assert_eq!(plan.split_primary_batch_size(), 0);
        assert_eq!(plan.multi_mask_batch_size(), 0);
        assert_eq!(plan.primary_batch_size(), 1);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn split_filterbank_and_tail_enable_chunk_path() {
        let dir = scratch_dir("split");
        touch(&dir, "wespeaker-voxceleb-resnet34.onnx");
        touch(&dir, "wespeaker-fbank.onnx");
        touch(&dir, "wespeaker-voxceleb-resnet34-tail.onnx");
        touch(&dir, "wespeaker-voxceleb-resnet34-tail-b64.onnx");
        let plan = plan_for(&dir);
        assert!(plan.prefers_chunk_embedding_path());
        assert!(!plan.prefers_multi_mask_path());
        assert_eq!(plan.split_primary_batch_size(), PRIMARY_BATCH_SIZE);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn filterbank_plus_multi_mask_without_tail_does_not_require_tail() {
        let dir = scratch_dir("fbank-multimask");
        touch(&dir, "wespeaker-voxceleb-resnet34.onnx");
        touch(&dir, "wespeaker-fbank.onnx");
        touch(&dir, "wespeaker-multimask-tail.onnx");
        let plan = plan_for(&dir);
        assert!(plan.fbank.is_some());
        assert!(plan.tail.is_none());
        assert!(plan.prefers_multi_mask_path());
        assert!(!plan.prefers_chunk_embedding_path());
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn complete_inventory_selects_independent_capabilities() {
        let dir = scratch_dir("cpu-complete");
        touch(&dir, "wespeaker-voxceleb-resnet34.onnx");
        touch(&dir, "wespeaker-voxceleb-resnet34-b64.onnx");
        touch(&dir, "wespeaker-fbank.onnx");
        touch(&dir, "wespeaker-fbank-b32.onnx");
        touch(&dir, "wespeaker-voxceleb-resnet34-tail.onnx");
        touch(&dir, "wespeaker-multimask-tail.onnx");
        touch(&dir, "wespeaker-multimask-tail-b32.onnx");
        let plan = plan_for(&dir);
        assert!(plan.fused_batched.is_some());
        assert_eq!(plan.primary_batch_size(), PRIMARY_BATCH_SIZE);
        assert!(plan.has_batched_fbank());
        assert!(plan.prefers_chunk_embedding_path());
        assert!(plan.prefers_multi_mask_path());
        assert_eq!(plan.multi_mask_batch_size(), MULTI_MASK_BATCH_SIZE);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn primary_sized_multi_mask_asset_is_ignored() {
        let dir = scratch_dir("cpu-primary-sized-multi-mask");
        touch(&dir, "wespeaker-voxceleb-resnet34.onnx");
        touch(&dir, "wespeaker-multimask-tail.onnx");
        touch(&dir, "wespeaker-multimask-tail-b64.onnx");
        let plan = plan_for(&dir);
        assert!(plan.multi_mask_batched.is_none());
        assert_eq!(plan.multi_mask_batch_size(), 1);
        let _ = fs::remove_dir_all(dir);
    }
}
