mod config;
pub use config::*;

mod types;
pub use types::*;

pub(crate) mod clustering;
#[cfg(test)]
use clustering::mark_inactive_speakers;
pub(crate) use clustering::{clean_masks, select_speaker_weights, write_speaker_mask_to_slice};

mod concurrent;
use concurrent::*;

mod post_inference;
pub use post_inference::post_inference;

#[cfg(feature = "coreml")]
mod chunk_embedding;

mod queued;
pub use queued::*;

#[cfg(test)]
mod tests;

use std::path::Path;

use ndarray::{Array2, Array3};
use tracing::{debug, info, trace};

use crate::clustering::plda::PldaTransform;
use crate::inference::ExecutionMode;
use crate::inference::embedding::EmbeddingModel;
use crate::inference::segmentation::SegmentationModel;
use crate::powerset::PowersetMapping;

/// Shared run/query methods for both owned and borrowed pipeline facades.
/// Both structs provide `runner()`, `mode`, and `seg_model` with compatible types
macro_rules! pipeline_run_methods {
    () => {
        /// Diarize audio using default config, file id defaults to "file1"
        pub fn run(&mut self, audio: &[f32]) -> Result<DiarizationResult, PipelineError> {
            self.run_with_file_id(audio, "file1")
        }

        /// Diarize audio with a custom file identifier for RTTM output
        pub fn run_with_file_id(
            &mut self,
            audio: &[f32],
            file_id: &str,
        ) -> Result<DiarizationResult, PipelineError> {
            self.run_with_config(audio, file_id, &PipelineConfig::for_mode(self.mode))
        }

        /// Diarize audio with a custom file identifier and pipeline config
        pub fn run_with_config(
            &mut self,
            audio: &[f32],
            file_id: &str,
            config: &PipelineConfig,
        ) -> Result<DiarizationResult, PipelineError> {
            self.runner().run(audio, file_id, config)
        }

        /// Run only inference (segmentation + embedding), returning intermediate artifacts
        pub fn run_inference_only(
            &mut self,
            audio: &[f32],
        ) -> Result<InferenceArtifacts, PipelineError> {
            self.runner().run_inference(audio)
        }

        /// Pipeline config for the current execution mode
        pub fn pipeline_config(&self) -> PipelineConfig {
            PipelineConfig::for_mode(self.mode)
        }

        /// Diarize a batch of files, keeping all hardware busy across file boundaries
        pub fn run_batch(
            &mut self,
            files: &[BatchInput<'_>],
        ) -> Result<Vec<DiarizationResult>, PipelineError> {
            self.run_batch_with_config(files, &PipelineConfig::for_mode(self.mode))
        }

        /// Diarize a batch of files with custom config
        pub fn run_batch_with_config(
            &mut self,
            files: &[BatchInput<'_>],
            config: &PipelineConfig,
        ) -> Result<Vec<DiarizationResult>, PipelineError> {
            self.runner().run_batch(files, config)
        }

        /// Segmentation step size in seconds for the current execution mode
        pub fn segmentation_step(&self) -> f64 {
            self.seg_model.step_seconds()
        }
    };
}

/// Owned pipeline that manages its own model lifetimes
pub struct OwnedDiarizationPipeline {
    seg_model: SegmentationModel,
    emb_model: EmbeddingModel,
    plda: PldaTransform,
    powerset: PowersetMapping,
    mode: ExecutionMode,
}

impl OwnedDiarizationPipeline {
    /// Download models from HuggingFace and build the pipeline
    #[cfg(feature = "online")]
    pub fn from_pretrained(mode: ExecutionMode) -> Result<Self, PipelineError> {
        Self::from_pretrained_with_config(mode, RuntimeConfig::default())
    }

    /// Download models from HuggingFace and build the pipeline with runtime config
    #[cfg(feature = "online")]
    pub fn from_pretrained_with_config(
        mode: ExecutionMode,
        config: RuntimeConfig,
    ) -> Result<Self, PipelineError> {
        let manager = crate::models::ModelManager::new()?;
        let models_dir = manager.ensure(mode)?;

        let step = match mode {
            ExecutionMode::CoreMlFast | ExecutionMode::CudaFast => FAST_SEGMENTATION_STEP_SECONDS,
            ExecutionMode::CoreMl => COREML_SEGMENTATION_STEP_SECONDS,
            ExecutionMode::Cuda => CUDA_SEGMENTATION_STEP_SECONDS,
            ExecutionMode::Cpu => SEGMENTATION_STEP_SECONDS,
        };

        let seg_model = SegmentationModel::with_mode(
            models_dir.join("segmentation-3.0.onnx").to_str().unwrap(),
            step as f32,
            mode,
        )?;
        let emb_model = EmbeddingModel::with_mode_and_config(
            models_dir
                .join("wespeaker-voxceleb-resnet34.onnx")
                .to_str()
                .unwrap(),
            mode,
            &config,
        )?;
        let plda = PldaTransform::from_dir(&models_dir)?;

        Ok(Self {
            seg_model,
            emb_model,
            plda,
            powerset: PowersetMapping::new(3, 2),
            mode,
        })
    }

    pipeline_run_methods!();

    /// Run post-inference (clustering + reconstruction) on pre-computed artifacts
    ///
    /// Does not need mutable model access, so it can run on a background
    /// thread while the next file's inference proceeds
    pub fn finish_post_inference(
        &self,
        artifacts: InferenceArtifacts,
        file_id: &str,
        config: &PipelineConfig,
    ) -> Result<DiarizationResult, PipelineError> {
        post_inference(artifacts, file_id, config, &self.plda)
    }

    /// Build the pipeline from a local models directory
    pub fn from_dir(models_dir: &Path, mode: ExecutionMode) -> Result<Self, PipelineError> {
        Self::from_dir_with_config(models_dir, mode, RuntimeConfig::default())
    }

    /// Build the pipeline from a local models directory with runtime config
    pub fn from_dir_with_config(
        models_dir: &Path,
        mode: ExecutionMode,
        config: RuntimeConfig,
    ) -> Result<Self, PipelineError> {
        let step = match mode {
            ExecutionMode::CoreMlFast | ExecutionMode::CudaFast => FAST_SEGMENTATION_STEP_SECONDS,
            ExecutionMode::CoreMl => COREML_SEGMENTATION_STEP_SECONDS,
            ExecutionMode::Cuda => CUDA_SEGMENTATION_STEP_SECONDS,
            ExecutionMode::Cpu => SEGMENTATION_STEP_SECONDS,
        };

        let seg_model = SegmentationModel::with_mode(
            models_dir.join("segmentation-3.0.onnx").to_str().unwrap(),
            step as f32,
            mode,
        )?;
        let emb_model = EmbeddingModel::with_mode_and_config(
            models_dir
                .join("wespeaker-voxceleb-resnet34.onnx")
                .to_str()
                .unwrap(),
            mode,
            &config,
        )?;
        let plda = PldaTransform::from_dir(models_dir)?;

        Ok(Self {
            seg_model,
            emb_model,
            plda,
            powerset: PowersetMapping::new(3, 2),
            mode,
        })
    }

    /// Convert into a background-processing queue
    pub fn into_queued(self) -> Result<QueuedDiarizationPipeline, QueueError> {
        let config = PipelineConfig::for_mode(self.mode);
        self.into_queued_with_config(config)
    }

    /// Convert into a background-processing queue with custom pipeline config
    pub fn into_queued_with_config(
        self,
        config: PipelineConfig,
    ) -> Result<QueuedDiarizationPipeline, QueueError> {
        QueuedDiarizationPipeline::new(self, config)
    }
}

/// Borrowed pipeline for when you manage model lifetimes yourself
pub struct DiarizationPipeline<'a> {
    seg_model: &'a mut SegmentationModel,
    emb_model: &'a mut EmbeddingModel,
    plda: PldaTransform,
    powerset: PowersetMapping,
    mode: ExecutionMode,
}

impl<'a> DiarizationPipeline<'a> {
    /// Build a pipeline from pre-loaded models and a PLDA parameters directory
    pub fn new(
        seg_model: &'a mut SegmentationModel,
        emb_model: &'a mut EmbeddingModel,
        models_dir: &Path,
    ) -> Result<Self, PipelineError> {
        Self::new_with_config(seg_model, emb_model, models_dir, RuntimeConfig::default())
    }

    /// Build a pipeline from pre-loaded models with custom runtime config
    pub fn new_with_config(
        seg_model: &'a mut SegmentationModel,
        emb_model: &'a mut EmbeddingModel,
        models_dir: &Path,
        _config: RuntimeConfig,
    ) -> Result<Self, PipelineError> {
        let mode = seg_model.mode();
        Ok(Self {
            seg_model,
            emb_model,
            plda: PldaTransform::from_dir(models_dir)?,
            powerset: PowersetMapping::new(3, 2),
            mode,
        })
    }

    /// Default segmentation step in seconds (CPU mode)
    pub fn default_segmentation_step() -> f32 {
        SEGMENTATION_STEP_SECONDS as f32
    }

    pipeline_run_methods!();
}

impl OwnedDiarizationPipeline {
    fn runner(&mut self) -> PipelineRunner<'_> {
        PipelineRunner {
            seg_model: &mut self.seg_model,
            emb_model: &mut self.emb_model,
            plda: &self.plda,
            powerset: &self.powerset,
        }
    }
}

impl<'a> DiarizationPipeline<'a> {
    fn runner(&mut self) -> PipelineRunner<'_> {
        PipelineRunner {
            seg_model: self.seg_model,
            emb_model: self.emb_model,
            plda: &self.plda,
            powerset: &self.powerset,
        }
    }
}

struct PipelineRunner<'a> {
    seg_model: &'a mut SegmentationModel,
    emb_model: &'a mut EmbeddingModel,
    plda: &'a PldaTransform,
    powerset: &'a PowersetMapping,
}

impl<'a> PipelineRunner<'a> {
    fn run(
        &mut self,
        audio: &[f32],
        file_id: &str,
        config: &PipelineConfig,
    ) -> Result<DiarizationResult, PipelineError> {
        let run_start = std::time::Instant::now();
        let inference_artifacts = self.run_inference(audio)?;
        let inference_ms = run_start.elapsed().as_millis();
        let post_start = std::time::Instant::now();
        let result = self.run_post_inference(inference_artifacts, file_id, config)?;
        let post_ms = post_start.elapsed().as_millis();
        let total_ms = run_start.elapsed().as_millis();
        let audio_secs = audio.len() as f64 / 16_000.0;
        trace!(
            %file_id,
            inference_ms,
            post_ms,
            total_ms,
            audio_secs,
            "Pipeline complete",
        );
        Ok(result)
    }

    fn inference_path(&self) -> InferencePath {
        if matches!(
            self.seg_model.mode(),
            ExecutionMode::CoreMl
                | ExecutionMode::CoreMlFast
                | ExecutionMode::Cuda
                | ExecutionMode::CudaFast
        ) {
            InferencePath::Concurrent
        } else {
            InferencePath::Sequential
        }
    }

    fn embedding_path(&self) -> EmbeddingPath {
        let path = if self.emb_model.prefers_multi_mask_path()
            && self.emb_model.multi_mask_batch_size() > 0
        {
            EmbeddingPath::MultiMask
        } else if self.emb_model.prefers_chunk_embedding_path()
            && self.emb_model.split_primary_batch_size() > 0
        {
            EmbeddingPath::Split
        } else {
            EmbeddingPath::Masked
        };
        debug!(?path, "Embedding path selected");
        path
    }

    fn run_batch(
        &mut self,
        files: &[BatchInput<'_>],
        config: &PipelineConfig,
    ) -> Result<Vec<DiarizationResult>, PipelineError> {
        if files.is_empty() {
            return Ok(Vec::new());
        }

        // try batch chunk embedding for CoreML concurrent path
        #[cfg(feature = "coreml")]
        if matches!(self.inference_path(), InferencePath::Concurrent)
            && let Some(results) = chunk_embedding::try_batch_chunk_embedding(
                self.seg_model,
                self.emb_model,
                self.powerset,
                self.plda,
                files,
                config,
            )?
        {
            return Ok(results);
        }

        // fallback: per-file sequential with post-inference overlap
        files
            .iter()
            .map(|f| self.run(f.audio, f.file_id, config))
            .collect()
    }

    fn run_inference(&mut self, audio: &[f32]) -> Result<InferenceArtifacts, PipelineError> {
        match self.inference_path() {
            InferencePath::Sequential => self.run_sequential_inference(audio),
            InferencePath::Concurrent => {
                #[cfg(feature = "coreml")]
                if let Some(result) = chunk_embedding::try_chunk_embedding(
                    self.seg_model,
                    self.emb_model,
                    self.powerset,
                    audio,
                )? {
                    return Ok(result);
                }
                self.run_concurrent_inference(audio)
            }
        }
    }

    fn run_sequential_inference(
        &mut self,
        audio: &[f32],
    ) -> Result<InferenceArtifacts, PipelineError> {
        let raw_windows = RawSegmentationWindows(self.seg_model.run(audio)?);
        info!(windows = raw_windows.0.len(), "Segmentation complete");

        let segmentations = raw_windows.decode(self.powerset);
        let layout = ChunkLayout::new(
            self.seg_model.step_seconds(),
            self.seg_model.step_samples(),
            self.seg_model.window_samples(),
            segmentations.nchunks(),
        );
        let embeddings = segmentations.extract_embeddings(
            audio,
            self.emb_model,
            &layout,
            self.embedding_path(),
        )?;

        info!(
            chunks = segmentations.nchunks(),
            speakers = segmentations.num_speakers(),
            "Embeddings complete"
        );

        Ok(InferenceArtifacts {
            layout,
            segmentations,
            embeddings,
        })
    }

    fn run_concurrent_inference(
        &mut self,
        audio: &[f32],
    ) -> Result<InferenceArtifacts, PipelineError> {
        let layout = ChunkLayout::without_frame_extent(
            self.seg_model.step_seconds(),
            self.seg_model.step_samples(),
            self.seg_model.window_samples(),
        );
        let concurrent_embedding_runner = ConcurrentEmbeddingRunner {
            powerset: self.powerset,
            audio,
            step_samples: layout.step_samples,
            window_samples: layout.window_samples,
            num_speakers: 3,
        };
        let embedding_path = self.embedding_path();
        let batch_size = match embedding_path {
            EmbeddingPath::MultiMask => self.emb_model.multi_mask_batch_size(),
            EmbeddingPath::Split => self.emb_model.split_primary_batch_size(),
            EmbeddingPath::Masked => self.emb_model.primary_batch_size(),
        };
        let min_num_samples = self.emb_model.min_num_samples();
        let (tx, rx) = crossbeam_channel::bounded::<Array2<f32>>(64);

        let inference_start = std::time::Instant::now();
        let use_parallel_seg = matches!(
            self.seg_model.mode(),
            ExecutionMode::CoreMl | ExecutionMode::CoreMlFast
        );
        let (segmentation_result, embedding_result) = std::thread::scope(|scope| {
            let segmentation_handle = if use_parallel_seg {
                #[cfg(feature = "coreml")]
                {
                    scope.spawn(|| self.seg_model.run_streaming_parallel(audio, tx, 4, None))
                }
                #[cfg(not(feature = "coreml"))]
                {
                    scope.spawn(|| self.seg_model.run_streaming(audio, tx))
                }
            } else {
                scope.spawn(|| self.seg_model.run_streaming(audio, tx))
            };

            let embedding_result = match embedding_path {
                EmbeddingPath::MultiMask => concurrent_embedding_runner.run_multi_mask(
                    rx,
                    self.emb_model,
                    batch_size,
                    min_num_samples,
                ),
                EmbeddingPath::Split => concurrent_embedding_runner.run_split(
                    rx,
                    self.emb_model,
                    batch_size,
                    min_num_samples,
                ),
                EmbeddingPath::Masked => {
                    concurrent_embedding_runner.run_masked(rx, self.emb_model, batch_size)
                }
            };

            let segmentation_result = segmentation_handle.join().unwrap();
            (segmentation_result, embedding_result)
        });
        let inference_elapsed = inference_start.elapsed();

        segmentation_result?;

        let concurrent_result = embedding_result?;
        if concurrent_result.is_empty() {
            return Ok(InferenceArtifacts {
                layout: layout.with_num_chunks(0),
                segmentations: DecodedSegmentations(Array3::zeros((0, 0, 0))),
                embeddings: ChunkEmbeddings(Array3::zeros((0, 0, 0))),
            });
        }

        let num_chunks = concurrent_result.num_chunks;
        let layout = layout.with_num_chunks(num_chunks);
        info!(
            chunks = concurrent_result.segmentations.shape()[0],
            speakers = concurrent_result.segmentations.shape()[2],
            inference_ms = inference_elapsed.as_millis(),
            embedding_path = ?embedding_path,
            "Concurrent seg+emb complete"
        );

        Ok(InferenceArtifacts {
            layout,
            segmentations: DecodedSegmentations(concurrent_result.segmentations),
            embeddings: ChunkEmbeddings(concurrent_result.embeddings),
        })
    }

    fn run_post_inference(
        &mut self,
        inference_artifacts: InferenceArtifacts,
        file_id: &str,
        config: &PipelineConfig,
    ) -> Result<DiarizationResult, PipelineError> {
        post_inference(inference_artifacts, file_id, config, self.plda)
    }
}
