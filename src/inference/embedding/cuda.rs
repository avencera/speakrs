//! Native CUDA embedding: GPU filterbank, then the ResNet34 multi-mask model, with the
//! features kept on the device between the two
//!
//! Every embedding path runs through the multi-mask model: a single-mask embedding is a
//! chunk whose other two speaker masks are zero, since each speaker row is pooled on its
//! own

use std::ops::Range;
use std::path::{Path, PathBuf};

use cudarc::driver::CudaView;
use ndarray::{Array1, Array2, ArrayView2, Axis};
use tracing::{debug, trace};

use crate::inference::cuda::implementation::policy::RecipeMode;
use crate::inference::cuda::{
    CudaError, CudaFbank, CudaGraphs, CudaMath, CudaRuntime, CudaSession, DeviceTensor,
    EMBEDDING_DIM, EmbeddingBatch, EmbeddingBatchClass, FBANK_FRAMES, FBANK_MEL_BINS,
    FBANK_WINDOW_SAMPLES, FbankBuffers, ResNetEmbedding, SPEAKERS_PER_CHUNK, SafetensorsFile,
};
use crate::inference::{ExecutionMode, InferenceError, ModelLoadError, TensorShapeError};
use crate::pipeline::RuntimeConfig;

use super::{
    EMBEDDING_WIDTH, EmbeddingMeta, FBANK_FEATURES, MASK_FRAMES, MULTI_MASK_BATCH_SIZE,
    NUM_SPEAKERS, SplitTailInput, prepare_weights, should_use_clean_mask,
};

/// The weights file CUDA modes load, next to the base embedding ONNX path
const WEIGHTS_FILE: &str = "wespeaker-multimask-tail.safetensors";

/// Values of one chunk's `[998, 80]` filterbank
const FBANK_LEN: usize = FBANK_FRAMES * FBANK_MEL_BINS;
/// Values of one chunk's three speaker masks
const CHUNK_MASKS_LEN: usize = SPEAKERS_PER_CHUNK * MASK_FRAMES;

const _: () = assert!(FBANK_FEATURES == FBANK_MEL_BINS && EMBEDDING_WIDTH == EMBEDDING_DIM);
const _: () = assert!(NUM_SPEAKERS == SPEAKERS_PER_CHUNK);

/// Native CUDA embedding plus private host staging for one model handle
pub(super) struct CudaEmbedding {
    session: CudaSession<EmbeddingState>,
    // the reload inputs are only read by `clone_shared`, which CoreML builds do not have
    #[cfg_attr(feature = "coreml", allow(dead_code))]
    weights: PathBuf,
    #[cfg_attr(feature = "coreml", allow(dead_code))]
    math: CudaMath,
    #[cfg_attr(feature = "coreml", allow(dead_code))]
    graphs: CudaGraphs,
    #[cfg_attr(feature = "coreml", allow(dead_code))]
    recipe_mode: RecipeMode,
    /// `[chunks * 3, 589]` speaker masks for the next batch
    masks: Vec<f32>,
    /// `[chunks, 998, 80]` host filterbanks for the next batch
    fbanks: Vec<f32>,
}

/// The device models and buffers; a session keeps them with the runtime they were
/// built on
struct EmbeddingState {
    fbank: CudaFbank,
    fbank_buffers: FbankBuffers,
    batches: Batches,
}

// SAFETY: `EmbeddingState` is not `Send` only because of cuDNN state (the convolution
// plans' descriptors and cudarc's cuDNN handle they share), the captured CUDA graphs
// and the pinned staging memory of `FbankBuffers`. All of it was created on the
// session's runtime and is reachable only through this session, which is neither
// `Clone` nor `Sync` and hands out no references to it, so moving the session moves
// every owner of that state at once and no two threads ever use it concurrently. cuDNN
// handles and descriptors, CUDA graphs and pinned host memory may be used from any host
// thread as long as calls are not concurrent, and `CudaSession::run` and its `Drop` make
// the runtime's context current on the thread that uses or frees them
unsafe impl Send for CudaSession<EmbeddingState> {}

/// The multi-mask model and its exact batch classes, each allocated on first use
struct Batches {
    model: ResNetEmbedding,
    graphs: bool,
    plans: [Option<EmbeddingBatch>; 5],
}

impl CudaEmbedding {
    pub(super) fn load(
        model_path: &Path,
        mode: ExecutionMode,
        config: &RuntimeConfig,
    ) -> Result<Self, ModelLoadError> {
        let weights = model_path.with_file_name(WEIGHTS_FILE);
        if !weights.is_file() {
            return Err(ModelLoadError::MissingCudaWeights {
                mode,
                path: weights,
            });
        }

        let math = config.cuda_embedding_math;
        let graphs = config.cuda_graphs;
        let recipe_mode = RecipeMode::new(config.cuda_segmentation_math, math);
        let session = open_session(&weights, math, graphs, recipe_mode)?;
        Ok(Self::with_session(
            session,
            weights,
            math,
            graphs,
            recipe_mode,
        ))
    }

    fn with_session(
        session: CudaSession<EmbeddingState>,
        weights: PathBuf,
        math: CudaMath,
        graphs: CudaGraphs,
        recipe_mode: RecipeMode,
    ) -> Self {
        Self {
            session,
            weights,
            math,
            graphs,
            recipe_mode,
            masks: Vec::new(),
            fbanks: Vec::new(),
        }
    }

    /// A handle with its own runtime, stream and device copy of the same weights
    #[cfg(not(feature = "coreml"))]
    pub(super) fn reload(&self) -> Result<Self, InferenceError> {
        let session = open_session(&self.weights, self.math, self.graphs, self.recipe_mode)?;
        Ok(Self::with_session(
            session,
            self.weights.clone(),
            self.math,
            self.graphs,
            self.recipe_mode,
        ))
    }

    /// No fused waveform-to-embedding batch model exists, so masked batches run one
    /// window at a time
    pub(super) fn primary_batch_size(&self) -> usize {
        1
    }

    /// Chunk speakers run as one multi-mask chunk on a downloaded filterbank
    pub(super) fn prefers_chunk_embedding_path(&self) -> bool {
        true
    }

    /// No batch-64 tail exists, so the pipeline's split path stays off
    pub(super) fn split_primary_batch_size(&self) -> usize {
        0
    }

    pub(super) fn prefers_multi_mask_path(&self) -> bool {
        true
    }

    pub(super) fn multi_mask_batch_size(&self) -> usize {
        MULTI_MASK_BATCH_SIZE
    }

    pub(super) fn has_batched_tail(&self) -> bool {
        true
    }

    pub(in crate::inference::embedding) fn embed_single(
        &mut self,
        meta: &EmbeddingMeta,
        audio: &[f32],
        weights: &[f32],
    ) -> Result<Array1<f32>, InferenceError> {
        stage_masks(&mut self.masks, [weights], meta.mask_frames);
        let rows = self.embed_audio(&[audio])?;
        first_row(rows)
    }

    /// Filterbanks and embeddings for up to [`MULTI_MASK_BATCH_SIZE`] audio chunks,
    /// with the filterbanks never leaving the device
    pub(in crate::inference::embedding) fn embed_multi_mask_audio_batch(
        &mut self,
        meta: &EmbeddingMeta,
        audios: &[&[f32]],
        masks: &[&[f32]],
    ) -> Result<Array2<f32>, InferenceError> {
        check_multi_mask_counts(audios.len(), masks.len())?;
        stage_masks(&mut self.masks, masks.iter().copied(), meta.mask_frames);
        let rows = self.embed_audio(audios)?;
        Ok(rows.slice_axis(Axis(0), (..masks.len()).into()).to_owned())
    }

    pub(in crate::inference::embedding) fn embed_multi_mask_batch(
        &mut self,
        meta: &EmbeddingMeta,
        fbanks: &[&Array2<f32>],
        masks: &[&[f32]],
    ) -> Result<Array2<f32>, InferenceError> {
        check_multi_mask_counts(fbanks.len(), masks.len())?;
        stage_fbanks(&mut self.fbanks, fbanks)?;
        stage_masks(&mut self.masks, masks.iter().copied(), meta.mask_frames);
        let rows = self.embed_fbanks()?;
        Ok(rows.slice_axis(Axis(0), (..masks.len()).into()).to_owned())
    }

    pub(in crate::inference::embedding) fn embed_tail_single(
        &mut self,
        meta: &EmbeddingMeta,
        fbank: &Array2<f32>,
        weights: &[f32],
    ) -> Result<Array1<f32>, InferenceError> {
        stage_fbanks(&mut self.fbanks, &[fbank])?;
        stage_masks(&mut self.masks, [weights], meta.mask_frames);
        let rows = self.embed_fbanks()?;
        first_row(rows)
    }

    pub(in crate::inference::embedding) fn embed_tail_batch(
        &mut self,
        meta: &EmbeddingMeta,
        fbank: &Array2<f32>,
        segmentations: &ArrayView2<'_, f32>,
        clean_masks: &Array2<f32>,
        num_samples: usize,
    ) -> Result<Array2<f32>, InferenceError> {
        let speakers = segmentations.ncols();
        if speakers > SPEAKERS_PER_CHUNK {
            return Err(InferenceError::BatchTooLarge {
                context: "cuda chunk speaker batch",
                rows: speakers,
                capacity: SPEAKERS_PER_CHUNK,
            });
        }

        let selected: Vec<Vec<f32>> = (0..speakers)
            .map(|speaker| {
                let mask = segmentations.column(speaker);
                let clean = clean_masks.column(speaker);
                let use_clean =
                    should_use_clean_mask(&clean, mask.len(), num_samples, meta.min_num_samples);
                let chosen = if use_clean { clean } else { mask };
                chosen.iter().copied().collect()
            })
            .collect();

        stage_fbanks(&mut self.fbanks, &[fbank])?;
        stage_masks(
            &mut self.masks,
            selected.iter().map(Vec::as_slice),
            meta.mask_frames,
        );
        let rows = self.embed_fbanks()?;
        Ok(rows.slice_axis(Axis(0), (..speakers).into()).to_owned())
    }

    pub(in crate::inference::embedding) fn embed_tail_batch_inputs(
        &mut self,
        meta: &EmbeddingMeta,
        inputs: &[SplitTailInput<'_>],
    ) -> Result<Array2<f32>, InferenceError> {
        let mut embeddings = Array2::<f32>::zeros((inputs.len(), EMBEDDING_WIDTH));
        for (row, input) in inputs.iter().enumerate() {
            let embedding = self.embed_tail_single(meta, input.fbank, input.weights)?;
            embeddings.row_mut(row).assign(&embedding);
        }

        Ok(embeddings)
    }

    pub(in crate::inference::embedding) fn compute_chunk_fbank(
        &mut self,
        meta: &EmbeddingMeta,
        audio: &[f32],
    ) -> Result<Array2<f32>, InferenceError> {
        let mut fbanks = self.compute_chunk_fbanks_batch(meta, &[audio])?;
        fbanks.pop().ok_or(InferenceError::MissingOutput {
            context: "cuda chunk fbank output",
        })
    }

    pub(in crate::inference::embedding) fn compute_chunk_fbanks_batch(
        &mut self,
        meta: &EmbeddingMeta,
        audios: &[&[f32]],
    ) -> Result<Vec<Array2<f32>>, InferenceError> {
        // the GPU filterbank pads and truncates to its own fixed window
        debug_assert_eq!(meta.window_samples, FBANK_WINDOW_SAMPLES);
        let mut results = Vec::with_capacity(audios.len());
        for batch in audios.chunks(MULTI_MASK_BATCH_SIZE) {
            let values = self.session.run(|runtime, state| {
                let features =
                    state
                        .fbank
                        .compute_host(runtime, batch, &mut state.fbank_buffers)?;
                Ok::<_, CudaError>(runtime.stream().clone_dtoh(&features)?)
            })?;

            for chunk in values.as_chunks::<FBANK_LEN>().0 {
                let fbank = Array2::from_shape_vec((FBANK_FRAMES, FBANK_MEL_BINS), chunk.to_vec())
                    .map_err(|source| InferenceError::OutputArray {
                        context: "cuda chunk fbank output",
                        source,
                    })?;
                results.push(fbank);
            }
        }

        Ok(results)
    }

    /// Embeds the staged masks for `audios`, computing their filterbanks on the device
    fn embed_audio(&mut self, audios: &[&[f32]]) -> Result<Array2<f32>, InferenceError> {
        let masks = &self.masks;
        let values = self.session.run(|runtime, state| {
            let EmbeddingState {
                fbank,
                fbank_buffers,
                batches,
            } = state;
            let features = fbank.compute_host(runtime, audios, fbank_buffers)?;
            batches.embed(runtime, audios.len(), masks, |runtime, rows, target| {
                let source = features.slice(rows.start * FBANK_LEN..rows.end * FBANK_LEN);
                copy_device(runtime, &source, target)
            })
        })?;

        embedding_rows(values)
    }

    /// Embeds the staged masks for the staged host filterbanks
    fn embed_fbanks(&mut self) -> Result<Array2<f32>, InferenceError> {
        let (fbanks, masks) = (&self.fbanks, &self.masks);
        let chunks = fbanks.len() / FBANK_LEN;
        let values = self.session.run(|runtime, state| {
            state
                .batches
                .embed(runtime, chunks, masks, |runtime, rows, target| {
                    let source = &fbanks[rows.start * FBANK_LEN..rows.end * FBANK_LEN];
                    target.copy_from_host(runtime.stream(), source)
                })
        })?;

        embedding_rows(values)
    }
}

impl Batches {
    /// Runs `chunks` chunks whose masks are in `masks`; `fill` writes the filterbanks
    /// of a range of chunks into a batch's input
    ///
    /// Partial batches use the largest exact class that fits each remaining range
    fn embed(
        &mut self,
        runtime: &CudaRuntime,
        chunks: usize,
        masks: &[f32],
        mut fill: impl FnMut(&CudaRuntime, Range<usize>, &mut DeviceTensor) -> Result<(), CudaError>,
    ) -> Result<Vec<f32>, CudaError> {
        let mut embeddings = Vec::with_capacity(chunks * SPEAKERS_PER_CHUNK * EMBEDDING_DIM);
        let mut start = 0;
        while let Some(class) = EmbeddingBatchClass::fitting(chunks - start) {
            let rows = start..start + class.chunks();
            let plan_start = std::time::Instant::now();
            let batch = self.batch(runtime, class)?;
            let plan_us = plan_start.elapsed().as_micros();
            let stage_start = std::time::Instant::now();
            fill(runtime, rows.clone(), batch.fbank_mut())?;
            let chunk_masks = &masks[rows.start * CHUNK_MASKS_LEN..rows.end * CHUNK_MASKS_LEN];
            batch
                .masks_mut()
                .copy_from_host(runtime.stream(), chunk_masks)?;
            let stage_us = stage_start.elapsed().as_micros();
            let launch_start = std::time::Instant::now();
            batch.forward(runtime)?;
            let launch_us = launch_start.elapsed().as_micros();
            let output_start = std::time::Instant::now();
            embeddings.extend(batch.download_output(runtime)?);
            trace!(
                target: "speakrs::timing",
                class = class.chunks(),
                plan_us,
                stage_us,
                launch_us,
                output_wait_us = output_start.elapsed().as_micros(),
                "CUDA embedding batch timing"
            );
            start = rows.end;
        }

        Ok(embeddings)
    }

    /// The batch for `chunks` chunks, planned and (with graphs on) captured on first use
    fn batch(
        &mut self,
        runtime: &CudaRuntime,
        class: EmbeddingBatchClass,
    ) -> Result<&mut EmbeddingBatch, CudaError> {
        let slot = &mut self.plans[class.slot()];
        if let Some(batch) = slot {
            return Ok(batch);
        }

        let mut batch = self.model.batch(runtime, class.chunks())?;
        if self.graphs {
            batch.capture_graph(runtime)?;
        }
        Ok(slot.insert(batch))
    }
}

fn open_session(
    weights: &Path,
    math: CudaMath,
    graphs: CudaGraphs,
    recipe_mode: RecipeMode,
) -> Result<CudaSession<EmbeddingState>, CudaError> {
    let file = SafetensorsFile::open(weights)?;
    CudaSession::new(
        CudaRuntime::new(0)?.with_recipe_mode(recipe_mode),
        |runtime| {
            // the filterbank stays FP32 whatever the embedding precision: TF32 moves
            // low-energy log-mel bins by up to 2.7 for a 0.16 ms gain
            let fbank = CudaFbank::new(runtime, CudaMath::Fp32)?;
            let model = ResNetEmbedding::load(runtime, &file, math)?;
            debug!(
                capability = %runtime.compute_capability(),
                ptx_limit = %runtime.ptx_tier(),
                fbank_tier = %fbank.tier(),
                embedding_tier = %model.kernel_tier(),
                ?math,
                ?graphs,
                "Loaded CUDA embedding"
            );

            Ok(EmbeddingState {
                fbank_buffers: fbank.buffers(runtime, MULTI_MASK_BATCH_SIZE)?,
                fbank,
                batches: Batches {
                    model,
                    graphs: graphs.enabled(),
                    plans: std::array::from_fn(|_| None),
                },
            })
        },
    )
}

/// Copies device filterbank values into a batch's input, which must have their length
fn copy_device(
    runtime: &CudaRuntime,
    source: &CudaView<'_, f32>,
    target: &mut DeviceTensor,
) -> Result<(), CudaError> {
    if source.len() != target.len() {
        return Err(CudaError::BufferLength {
            context: "cuda embedding filterbank copy",
            expected: target.len(),
            actual: source.len(),
        });
    }

    runtime.stream().memcpy_dtod(source, target.data_mut())?;
    Ok(())
}

/// Stages `[chunks, 998, 80]` host filterbanks, zero padding shorter ones
fn stage_fbanks(staging: &mut Vec<f32>, fbanks: &[&Array2<f32>]) -> Result<(), InferenceError> {
    staging.clear();
    staging.resize(fbanks.len() * FBANK_LEN, 0.0);
    for (row, fbank) in staging
        .as_chunks_mut::<FBANK_LEN>()
        .0
        .iter_mut()
        .zip(fbanks)
    {
        if fbank.nrows() > FBANK_FRAMES || fbank.ncols() != FBANK_MEL_BINS {
            return Err(TensorShapeError::ShapeMismatch {
                context: "cuda embedding filterbank",
                expected: vec![FBANK_FRAMES, FBANK_MEL_BINS],
                actual: fbank.shape().to_vec(),
            }
            .into());
        }

        for (target, source) in row
            .as_chunks_mut::<FBANK_MEL_BINS>()
            .0
            .iter_mut()
            .zip(fbank.rows())
        {
            for (target, source) in target.iter_mut().zip(source) {
                *target = *source;
            }
        }
    }

    Ok(())
}

/// Stages one mask row per item, padded to whole chunks of three speakers with zero
/// masks, each row truncated or zero padded to `mask_frames`
fn stage_masks<'a>(
    staging: &mut Vec<f32>,
    masks: impl IntoIterator<Item = &'a [f32]>,
    mask_frames: usize,
) {
    let masks: Vec<&[f32]> = masks.into_iter().collect();
    let rows = masks.len().div_ceil(SPEAKERS_PER_CHUNK) * SPEAKERS_PER_CHUNK;
    staging.clear();
    staging.resize(rows * mask_frames, 0.0);

    let Ok(mut view) = ndarray::ArrayViewMut2::from_shape((rows, mask_frames), staging) else {
        return;
    };
    for (row, mask) in masks.iter().enumerate() {
        prepare_weights(row, mask, mask_frames, &mut view);
    }
}

fn check_multi_mask_counts(chunks: usize, masks: usize) -> Result<(), InferenceError> {
    if chunks > MULTI_MASK_BATCH_SIZE {
        return Err(InferenceError::BatchTooLarge {
            context: "cuda multi-mask batch",
            rows: chunks,
            capacity: MULTI_MASK_BATCH_SIZE,
        });
    }

    let expected = chunks * NUM_SPEAKERS;
    if masks != expected {
        return Err(InferenceError::MaskCountMismatch {
            fbanks: chunks,
            expected,
            actual: masks,
        });
    }

    Ok(())
}

fn embedding_rows(values: Vec<f32>) -> Result<Array2<f32>, InferenceError> {
    let rows = values.len() / EMBEDDING_DIM;
    Array2::from_shape_vec((rows, EMBEDDING_DIM), values).map_err(|source| {
        InferenceError::OutputArray {
            context: "cuda embedding output",
            source,
        }
    })
}

fn first_row(rows: Array2<f32>) -> Result<Array1<f32>, InferenceError> {
    if rows.nrows() == 0 {
        return Err(InferenceError::MissingOutput {
            context: "cuda embedding output",
        });
    }

    Ok(rows.row(0).to_owned())
}

#[cfg(test)]
mod tests;
