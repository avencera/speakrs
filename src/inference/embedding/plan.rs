use std::path::{Path, PathBuf};

use crate::inference::ExecutionMode;
use crate::pipeline::RuntimeConfig;

use super::{
    CHUNK_SPEAKER_BATCH_SIZE, PRIMARY_BATCH_SIZE, batched_model_path, multi_mask_model_path,
    split_fbank_batched_model_path, split_fbank_model_path, split_tail_model_path,
};
#[cfg(feature = "coreml")]
use super::{ChunkSessionSpec, EmbeddingModel, fp32_coreml_path};

/// Discovered on-disk asset used by [`EmbeddingExecutionPlan`]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AssetSlot {
    path: PathBuf,
}

impl AssetSlot {
    fn if_exists(path: PathBuf) -> Option<Self> {
        path.exists().then_some(Self { path })
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }
}

/// Primary fused embedding sessions
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FusedPlan {
    pub(crate) single: Option<AssetSlot>,
    pub(crate) batched: Option<AssetSlot>,
}

/// Split filterbank sessions
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SplitFilterbankPlan {
    pub(crate) single: Option<AssetSlot>,
    pub(crate) batched: Option<AssetSlot>,
    #[cfg(feature = "coreml")]
    pub(crate) native_single: Option<AssetSlot>,
    #[cfg(feature = "coreml")]
    pub(crate) native_batched: Option<AssetSlot>,
    #[cfg(feature = "coreml")]
    pub(crate) native_30s: Option<AssetSlot>,
}

/// Split embedding-tail sessions
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SplitTailPlan {
    pub(crate) single: Option<AssetSlot>,
    pub(crate) batched: Option<AssetSlot>,
    pub(crate) primary_batched: Option<AssetSlot>,
    #[cfg(feature = "coreml")]
    pub(crate) native_single: Option<AssetSlot>,
    #[cfg(feature = "coreml")]
    pub(crate) native_batched: Option<AssetSlot>,
    #[cfg(feature = "coreml")]
    pub(crate) native_primary_batched: Option<AssetSlot>,
}

/// Multi-mask tail sessions
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MultiMaskPlan {
    pub(crate) single: Option<AssetSlot>,
    pub(crate) batched: Option<AssetSlot>,
    #[cfg(feature = "coreml")]
    pub(crate) native: Option<AssetSlot>,
}

/// Backend capabilities resolved once from mode, runtime, and on-disk inventory
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct EmbeddingExecutionPlan {
    mode: ExecutionMode,
    pub(crate) fused: FusedPlan,
    pub(crate) split_fbank: SplitFilterbankPlan,
    pub(crate) split_tail: SplitTailPlan,
    pub(crate) multi_mask: MultiMaskPlan,
    #[cfg(feature = "coreml")]
    pub(crate) chunk_ladder: Vec<ChunkSessionSpec>,
}

impl EmbeddingExecutionPlan {
    /// Snapshot usable assets once. Path selection must not re-read the file system.
    pub(crate) fn from_inventory(
        model_path: &Path,
        mode: ExecutionMode,
        #[cfg_attr(not(feature = "coreml"), allow(unused_variables))] config: &RuntimeConfig,
    ) -> Self {
        let fused = FusedPlan {
            single: AssetSlot::if_exists(model_path.to_path_buf()),
            batched: batched_model_path(model_path, PRIMARY_BATCH_SIZE)
                .and_then(AssetSlot::if_exists),
        };
        let split_fbank = SplitFilterbankPlan {
            single: AssetSlot::if_exists(split_fbank_model_path(model_path)),
            batched: AssetSlot::if_exists(split_fbank_batched_model_path(model_path)),
            #[cfg(feature = "coreml")]
            native_single: native_slot(mode, fp32_coreml_path(&split_fbank_model_path(model_path))),
            #[cfg(feature = "coreml")]
            native_batched: native_slot(
                mode,
                fp32_coreml_path(&split_fbank_batched_model_path(model_path)),
            ),
            #[cfg(feature = "coreml")]
            native_30s: native_slot(
                mode,
                model_path.with_file_name("wespeaker-fbank-30s.mlmodelc"),
            ),
        };
        let split_tail = SplitTailPlan {
            single: AssetSlot::if_exists(split_tail_model_path(model_path, 1)),
            batched: AssetSlot::if_exists(split_tail_model_path(
                model_path,
                CHUNK_SPEAKER_BATCH_SIZE,
            )),
            primary_batched: AssetSlot::if_exists(split_tail_model_path(
                model_path,
                PRIMARY_BATCH_SIZE,
            )),
            #[cfg(feature = "coreml")]
            native_single: native_slot(
                mode,
                fp32_coreml_path(&split_tail_model_path(model_path, 1)),
            ),
            #[cfg(feature = "coreml")]
            native_batched: native_slot(
                mode,
                fp32_coreml_path(&split_tail_model_path(model_path, CHUNK_SPEAKER_BATCH_SIZE)),
            ),
            #[cfg(feature = "coreml")]
            native_primary_batched: native_slot(
                mode,
                fp32_coreml_path(&split_tail_model_path(model_path, PRIMARY_BATCH_SIZE)),
            ),
        };
        let multi_mask = MultiMaskPlan {
            single: multi_mask_model_path(model_path, 1).and_then(AssetSlot::if_exists),
            batched: multi_mask_model_path(model_path, PRIMARY_BATCH_SIZE)
                .and_then(AssetSlot::if_exists),
            #[cfg(feature = "coreml")]
            native: native_slot(
                mode,
                fp32_coreml_path(&model_path.with_file_name("wespeaker-multimask-tail-b32.onnx")),
            ),
        };
        #[cfg(feature = "coreml")]
        let chunk_ladder = if mode.is_coreml() {
            EmbeddingModel::chunk_session_specs(model_path, mode, config)
        } else {
            Vec::new()
        };

        Self {
            mode,
            fused,
            split_fbank,
            split_tail,
            multi_mask,
            #[cfg(feature = "coreml")]
            chunk_ladder,
        }
    }

    pub(crate) fn load_ort_split(&self) -> bool {
        !self.mode.is_coreml()
    }

    pub(crate) fn prefers_chunk_embedding_path(&self) -> bool {
        #[cfg(feature = "coreml")]
        if self.mode.is_coreml() {
            return self.split_fbank.native_single.is_some()
                && self.split_tail.native_single.is_some();
        }

        let ort_split = self.split_fbank.single.is_some() && self.split_tail.single.is_some();
        #[cfg(feature = "coreml")]
        let ort_split = ort_split || self.split_tail.native_single.is_some();
        ort_split
    }

    pub(crate) fn split_primary_batch_size(&self) -> usize {
        #[cfg(feature = "coreml")]
        if self.mode.is_coreml() {
            return usize::from(self.split_tail.native_primary_batched.is_some())
                * PRIMARY_BATCH_SIZE;
        }

        if self.split_tail.primary_batched.is_some() {
            return PRIMARY_BATCH_SIZE;
        }
        #[cfg(feature = "coreml")]
        if self.split_tail.native_primary_batched.is_some() {
            return PRIMARY_BATCH_SIZE;
        }
        0
    }

    pub(crate) fn has_batched_fbank(&self) -> bool {
        #[cfg(feature = "coreml")]
        if self.mode.is_coreml() {
            return self.split_fbank.native_batched.is_some();
        }

        let has = self.split_fbank.batched.is_some();
        #[cfg(feature = "coreml")]
        let has = has || self.split_fbank.native_batched.is_some();
        has
    }

    pub(crate) fn prefers_multi_mask_path(&self) -> bool {
        #[cfg(feature = "coreml")]
        if self.mode.is_coreml() {
            return self.multi_mask.native.is_some();
        }

        let has = self.multi_mask.single.is_some();
        #[cfg(feature = "coreml")]
        let has = has || self.multi_mask.native.is_some();
        has
    }

    pub(crate) fn multi_mask_batch_size(&self) -> usize {
        #[cfg(feature = "coreml")]
        if self.mode.is_coreml() {
            return usize::from(self.multi_mask.native.is_some()) * super::MULTI_MASK_BATCH_SIZE;
        }

        if self.multi_mask.batched.is_some() {
            super::MULTI_MASK_BATCH_SIZE
        } else if self.multi_mask.single.is_some() {
            1
        } else {
            0
        }
    }

    pub(crate) fn has_batched_tail(&self) -> bool {
        #[cfg(feature = "coreml")]
        if self.mode.is_coreml() {
            return self.split_tail.native_batched.is_some();
        }
        self.split_tail.batched.is_some()
    }
}

/// Lazy CoreML session permitted by the execution plan
#[cfg(feature = "coreml")]
#[derive(Debug)]
pub(crate) enum LazySession<Spec, Session> {
    Unavailable,
    Unloaded(Spec),
    Loaded(Session),
}

#[cfg(feature = "coreml")]
impl<Session> LazySession<PathBuf, Session> {
    pub(crate) fn from_slot(slot: Option<&AssetSlot>) -> Self {
        match slot {
            Some(slot) => Self::Unloaded(slot.path.clone()),
            None => Self::Unavailable,
        }
    }
}

#[cfg(feature = "coreml")]
impl<Spec, Session> LazySession<Spec, Session> {
    pub(crate) fn as_ref(&self) -> Option<&Session> {
        match self {
            Self::Loaded(session) => Some(session),
            Self::Unavailable | Self::Unloaded(_) => None,
        }
    }

    pub(crate) fn as_mut(&mut self) -> Option<&mut Session> {
        match self {
            Self::Loaded(session) => Some(session),
            Self::Unavailable | Self::Unloaded(_) => None,
        }
    }

    pub(crate) fn load_mut<E>(
        &mut self,
        load: impl FnOnce(&Spec) -> Result<Session, E>,
    ) -> Result<Option<&mut Session>, E> {
        if matches!(self, Self::Unavailable) {
            return Ok(None);
        }
        if matches!(self, Self::Loaded(_)) {
            return Ok(self.as_mut());
        }
        let spec = match std::mem::replace(self, Self::Unavailable) {
            Self::Unloaded(spec) => spec,
            other => {
                *self = other;
                return Ok(self.as_mut());
            }
        };
        *self = Self::Loaded(load(&spec)?);
        Ok(self.as_mut())
    }
}

#[cfg(feature = "coreml")]
fn native_slot(mode: ExecutionMode, path: PathBuf) -> Option<AssetSlot> {
    if !mode.is_coreml() {
        return None;
    }
    AssetSlot::if_exists(path)
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

    fn plan_for(dir: &Path, mode: ExecutionMode) -> EmbeddingExecutionPlan {
        EmbeddingExecutionPlan::from_inventory(
            &dir.join("wespeaker-voxceleb-resnet34.onnx"),
            mode,
            &RuntimeConfig::default(),
        )
    }

    #[test]
    fn primary_only_inventory_disables_split_paths() {
        let dir = scratch_dir("primary");
        touch(&dir, "wespeaker-voxceleb-resnet34.onnx");
        let plan = plan_for(&dir, ExecutionMode::Cpu);
        assert!(plan.fused.single.is_some());
        assert!(!plan.prefers_chunk_embedding_path());
        assert!(!plan.prefers_multi_mask_path());
        assert_eq!(plan.split_primary_batch_size(), 0);
        assert_eq!(plan.multi_mask_batch_size(), 0);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn split_filterbank_and_tail_enable_chunk_path() {
        let dir = scratch_dir("split");
        touch(&dir, "wespeaker-voxceleb-resnet34.onnx");
        touch(&dir, "wespeaker-fbank.onnx");
        touch(&dir, "wespeaker-voxceleb-resnet34-tail.onnx");
        touch(&dir, "wespeaker-voxceleb-resnet34-tail-b64.onnx");
        let plan = plan_for(&dir, ExecutionMode::Cpu);
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
        let plan = plan_for(&dir, ExecutionMode::Cpu);
        assert!(plan.split_fbank.single.is_some());
        assert!(plan.split_tail.single.is_none());
        assert!(plan.prefers_multi_mask_path());
        assert!(!plan.prefers_chunk_embedding_path());
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn cpu_mode_selects_independent_ort_capabilities() {
        let dir = scratch_dir("cpu-complete");
        touch(&dir, "wespeaker-voxceleb-resnet34.onnx");
        touch(&dir, "wespeaker-voxceleb-resnet34-b64.onnx");
        touch(&dir, "wespeaker-fbank.onnx");
        touch(&dir, "wespeaker-fbank-b32.onnx");
        touch(&dir, "wespeaker-voxceleb-resnet34-tail.onnx");
        touch(&dir, "wespeaker-multimask-tail.onnx");
        touch(&dir, "wespeaker-multimask-tail-b64.onnx");
        let plan = plan_for(&dir, ExecutionMode::Cpu);
        assert!(plan.fused.batched.is_some());
        assert!(plan.has_batched_fbank());
        assert!(plan.prefers_chunk_embedding_path());
        assert!(plan.prefers_multi_mask_path());
        assert_eq!(
            plan.multi_mask_batch_size(),
            super::super::MULTI_MASK_BATCH_SIZE
        );
        assert!(plan.load_ort_split());
        let _ = fs::remove_dir_all(dir);
    }

    #[cfg(feature = "coreml")]
    #[test]
    fn coreml_plan_uses_native_slots_and_skips_ort_split() {
        let dir = scratch_dir("coreml");
        touch(&dir, "wespeaker-voxceleb-resnet34.onnx");
        fs::create_dir_all(dir.join("wespeaker-fbank.mlmodelc")).unwrap();
        fs::create_dir_all(dir.join("wespeaker-voxceleb-resnet34-tail.mlmodelc")).unwrap();
        fs::create_dir_all(dir.join("wespeaker-multimask-tail-b32.mlmodelc")).unwrap();
        let plan = plan_for(&dir, ExecutionMode::CoreMl);
        assert!(plan.prefers_chunk_embedding_path());
        assert!(plan.prefers_multi_mask_path());
        assert!(!plan.load_ort_split());
        assert_eq!(
            plan.multi_mask_batch_size(),
            super::super::MULTI_MASK_BATCH_SIZE
        );
        let _ = fs::remove_dir_all(dir);
    }
}
