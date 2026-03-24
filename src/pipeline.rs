use std::ops::Deref;
use std::path::Path;

use ndarray::{Array2, Array3, ArrayView2, s};
use tracing::{debug, info, trace};

use crate::binarize::{BinarizeConfig, binarize};
use crate::clustering::ahc::{AhcConfig, cluster as cluster_ahc};
use crate::clustering::plda::PldaTransform;
use crate::clustering::vbx::{VbxConfig, cluster_vbx};
#[cfg(feature = "coreml")]
use crate::inference::CoreMlComputeUnits;
use crate::inference::ExecutionMode;
use crate::inference::embedding::{
    EmbeddingModel, MaskedEmbeddingInput, SplitTailInput, should_use_clean_mask,
};
use crate::inference::segmentation::SegmentationModel;
use crate::powerset::PowersetMapping;
use crate::reconstruct::Reconstructor;
use crate::segment::{merge_segments, to_rttm, to_segments};
use crate::utils::cosine_similarity;

#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    #[error("{0}")]
    Ort(String),
    #[error("{0}")]
    Plda(#[from] crate::clustering::plda::PldaError),
    #[cfg(feature = "online")]
    #[error("{0}")]
    HfHub(#[from] hf_hub::api::sync::ApiError),
    #[error("{0}")]
    Other(String),
}

impl From<ort::Error> for PipelineError {
    fn from(e: ort::Error) -> Self {
        Self::Ort(e.to_string())
    }
}

impl From<crate::inference::segmentation::SegmentationError> for PipelineError {
    fn from(e: crate::inference::segmentation::SegmentationError) -> Self {
        Self::Other(e.to_string())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ReconstructMethod {
    /// Standard top-K selection (pyannote-compatible)
    Standard,
    /// Temporal smoothing — when scores are within epsilon, prefer previous frame's speaker
    Smoothed { epsilon: f32 },
}

#[derive(Debug, Clone)]
pub struct PipelineConfig {
    pub binarize: BinarizeConfig,
    pub ahc: AhcConfig,
    pub vbx: VbxConfig,
    pub merge_gap: f64,
    pub speaker_keep_threshold: f64,
    pub reconstruct_method: ReconstructMethod,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            binarize: BinarizeConfig::default(),
            ahc: AhcConfig::default(),
            vbx: VbxConfig::default(),
            merge_gap: 0.0,
            speaker_keep_threshold: 1e-7,
            reconstruct_method: ReconstructMethod::Smoothed { epsilon: 0.1 },
        }
    }
}

impl PipelineConfig {
    /// Mode-specific defaults. CoreMLFast uses min-duration filtering to remove
    /// single-frame speaker flickers caused by the larger step size
    pub fn for_mode(mode: ExecutionMode) -> Self {
        match mode {
            ExecutionMode::CoreMlFast | ExecutionMode::CudaFast => Self {
                binarize: BinarizeConfig {
                    min_duration_on: 3,
                    min_duration_off: 3,
                    ..BinarizeConfig::default()
                },
                // fast modes: 3 VBx iters avoids posterior over-fitting,
                // improves DER on 2s step embeddings
                vbx: VbxConfig {
                    max_iters: 3,
                    ..VbxConfig::default()
                },
                ..Self::default()
            },
            _ => Self::default(),
        }
    }
}

/// Runtime configuration for the diarization pipeline
///
/// Controls execution parameters that don't affect correctness
/// but influence performance characteristics
#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    /// Number of parallel chunk embedding workers (default: 1)
    pub chunk_emb_workers: usize,
    /// CoreML compute units for chunk embedding (CoreML modes only)
    #[cfg(feature = "coreml")]
    pub chunk_emb_compute_units: CoreMlComputeUnits,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            chunk_emb_workers: 1,
            #[cfg(feature = "coreml")]
            chunk_emb_compute_units: CoreMlComputeUnits::All,
        }
    }
}

pub const SEGMENTATION_WINDOW_SECONDS: f64 = 10.0;
pub const SEGMENTATION_STEP_SECONDS: f64 = 1.0;
// aligned to 8-frame ResNet stride: 96 fbank frames / 8 = 12 ResNet frames
// closest aligned step below 1.0s, enables chunk embedding
pub const COREML_SEGMENTATION_STEP_SECONDS: f64 = 0.96;
pub const CUDA_SEGMENTATION_STEP_SECONDS: f64 = 1.0;
pub const FAST_SEGMENTATION_STEP_SECONDS: f64 = 2.0;
pub const FRAME_DURATION_SECONDS: f64 = 0.0619375;
pub const FRAME_STEP_SECONDS: f64 = 0.016875;

/// Minimum speaker activity (sum of weights) to run embedding inference.
/// Speakers below this threshold are skipped — their NaN embedding is filtered out later
const MIN_SPEAKER_ACTIVITY: f32 = 10.0;

struct SpeakerEmbedding {
    chunk_idx: usize,
    speaker_idx: usize,
    embedding: Vec<f32>,
}

struct PendingEmbedding<'a> {
    chunk_idx: usize,
    speaker_idx: usize,
    audio: &'a [f32],
    mask: Vec<f32>,
    clean_mask: Vec<f32>,
}

struct PendingSplitEmbedding {
    chunk_idx: usize,
    speaker_idx: usize,
    fbank_idx: usize,
    weights: Vec<f32>,
}

struct ConcurrentEmbeddingResult {
    decoded_windows: Vec<Array2<f32>>,
    embeddings: Vec<SpeakerEmbedding>,
    num_speakers: usize,
}

impl ConcurrentEmbeddingResult {
    fn is_empty(&self) -> bool {
        self.decoded_windows.is_empty()
    }

    fn into_arrays(self) -> (Array3<f32>, Array3<f32>) {
        let num_chunks = self.decoded_windows.len();
        let num_frames = self.decoded_windows[0].nrows();

        let mut segmentations = Array3::<f32>::zeros((num_chunks, num_frames, self.num_speakers));
        for (i, w) in self.decoded_windows.iter().enumerate() {
            segmentations.slice_mut(s![i, .., ..]).assign(w);
        }

        let mut embeddings =
            Array3::<f32>::from_elem((num_chunks, self.num_speakers, 256), f32::NAN);
        for emb in &self.embeddings {
            embeddings
                .slice_mut(s![emb.chunk_idx, emb.speaker_idx, ..])
                .assign(&ndarray::ArrayView1::from(emb.embedding.as_slice()));
        }

        (segmentations, embeddings)
    }
}

#[derive(Debug, Clone)]
pub struct DecodedSegmentations(pub Array3<f32>);

impl Deref for DecodedSegmentations {
    type Target = Array3<f32>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Debug, Clone)]
pub struct ChunkEmbeddings(pub Array3<f32>);

impl Deref for ChunkEmbeddings {
    type Target = Array3<f32>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Debug, Clone)]
pub struct SpeakerCountTrack(pub Vec<usize>);

impl Deref for SpeakerCountTrack {
    type Target = Vec<usize>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Debug, Clone)]
pub struct ChunkSpeakerClusters(pub Array2<i32>);

impl Deref for ChunkSpeakerClusters {
    type Target = Array2<i32>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Debug, Clone)]
pub struct DiscreteDiarization(pub Array2<f32>);

impl Deref for DiscreteDiarization {
    type Target = Array2<f32>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Debug, Clone)]
pub(crate) struct FrameActivations(pub(crate) Array2<f32>);

impl Deref for FrameActivations {
    type Target = Array2<f32>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

struct RawSegmentationWindows(Vec<Array2<f32>>);

struct TrainingEmbeddings(Array2<f32>);

struct ChunkLayout {
    step_seconds: f64,
    step_samples: usize,
    window_samples: usize,
    start_frames: Vec<usize>,
    output_frames: usize,
}

pub struct InferenceArtifacts {
    layout: ChunkLayout,
    segmentations: DecodedSegmentations,
    embeddings: ChunkEmbeddings,
}

#[cfg(feature = "coreml")]
#[derive(Clone, Copy)]
struct SharedModelPtr(usize);

#[cfg(feature = "coreml")]
#[derive(Clone, Copy)]
struct SharedShapePtr(usize);

#[cfg(feature = "coreml")]
#[derive(Clone, Copy)]
struct ChunkSessionPtr {
    cached_fbank_shape: SharedShapePtr,
    cached_masks_shape: SharedShapePtr,
    model: SharedModelPtr,
}

#[cfg(feature = "coreml")]
#[derive(Clone)]
struct ChunkEmbeddingResources {
    chunk_sessions: Vec<ChunkSessionPtr>,
    chunk_lookup: Vec<(usize, usize, usize)>,
    fbank_30s_ptr: Option<SharedModelPtr>,
    fbank_10s_ptr: Option<SharedModelPtr>,
    ane_session: Option<ChunkSessionPtr>,
}

#[cfg(feature = "coreml")]
struct DecodedChunk {
    global_start: usize,
    decoded_chunk: Vec<Array2<f32>>,
    precomputed_fbank: Option<Vec<f32>>,
}

#[cfg(feature = "coreml")]
struct PreparedChunk {
    global_start: usize,
    decoded_chunk: Vec<Array2<f32>>,
    fbank: Vec<f32>,
    masks: Vec<f32>,
    active: Vec<(usize, usize)>,
    num_masks: usize,
}

#[cfg(feature = "coreml")]
struct EmbeddedChunk {
    global_start: usize,
    decoded_chunk: Vec<Array2<f32>>,
    data: Vec<f32>,
    active: Vec<(usize, usize)>,
    num_masks: usize,
    predict_us: u64,
}

#[cfg(feature = "coreml")]
#[derive(Default)]
struct PrepStats {
    chunks: u32,
    fbank_us: u64,
}

/// Prep a decoded chunk: compute fbank features and speaker masks on CPU
#[cfg(feature = "coreml")]
fn prep_decoded_chunk(
    decoded: &DecodedChunk,
    audio: &[f32],
    step_samples: usize,
    window_samples: usize,
    num_speakers: usize,
    min_num_samples: usize,
    largest_fbank_frames: usize,
    largest_num_masks: usize,
    max_active: usize,
    fbank_30s_ptr: Option<SharedModelPtr>,
    fbank_10s_ptr: Option<SharedModelPtr>,
    fbank_30s_buf: &mut [f32],
    waveform_10s_buf: &mut [f32],
    fbank_30s_shape: &crate::inference::coreml::CachedInputShape,
    fbank_10s_shape: &crate::inference::coreml::CachedInputShape,
) -> Result<PreparedChunk, PipelineError> {
    let global_start = decoded.global_start;
    let wins = decoded.decoded_chunk.len();
    let chunk_audio_start = global_start * step_samples;
    let chunk_audio_len = window_samples + (wins - 1) * step_samples;
    let chunk_audio_end = (chunk_audio_start + chunk_audio_len).min(audio.len());
    let chunk_audio = &audio[chunk_audio_start..chunk_audio_end];

    let mut fbank = vec![0.0f32; largest_fbank_frames * 80];

    if chunk_audio.len() <= 480_000 {
        if let Some(SharedModelPtr(ptr)) = fbank_30s_ptr {
            let fbank_model = unsafe {
                &*(ptr as *const crate::inference::coreml::SharedCoreMlModel)
            };
            fbank_30s_buf[..chunk_audio.len()].copy_from_slice(chunk_audio);
            fbank_30s_buf[chunk_audio.len()..].fill(0.0);
            let (data, out_shape) = fbank_model
                .predict_cached(&[(fbank_30s_shape, &*fbank_30s_buf)])
                .map_err(|e| PipelineError::Other(e.to_string()))?;
            let copy_frames = out_shape[1].min(largest_fbank_frames);
            for r in 0..copy_frames {
                let off = r * 80;
                fbank[off..off + 80].copy_from_slice(&data[off..off + 80]);
            }
        }
    } else if let Some(SharedModelPtr(ptr)) = fbank_10s_ptr {
        let fbank_model = unsafe {
            &*(ptr as *const crate::inference::coreml::SharedCoreMlModel)
        };
        let mut fb_off = 0usize;
        let mut au_off = 0usize;
        while fb_off < largest_fbank_frames && au_off < chunk_audio.len() {
            let seg_end = (au_off + window_samples).min(chunk_audio.len());
            let seg_len = seg_end - au_off;
            waveform_10s_buf[..seg_len].copy_from_slice(&chunk_audio[au_off..seg_end]);
            if seg_len < window_samples {
                waveform_10s_buf[seg_len..].fill(0.0);
            }
            let (data, out_shape) = fbank_model
                .predict_cached(&[(fbank_10s_shape, &*waveform_10s_buf)])
                .map_err(|e| PipelineError::Other(e.to_string()))?;
            let copy = out_shape[1].min(largest_fbank_frames - fb_off);
            for r in 0..copy {
                let src = r * 80;
                let dst = (fb_off + r) * 80;
                fbank[dst..dst + 80].copy_from_slice(&data[src..src + 80]);
            }
            fb_off += 998;
            au_off += window_samples;
        }
    }

    let mut masks = vec![0.0f32; largest_num_masks * 589];
    let mut active: Vec<(usize, usize)> = Vec::with_capacity(max_active);
    for (local, dec) in decoded.decoded_chunk.iter().enumerate() {
        let global_idx = global_start + local;
        let win_audio = chunk_audio_raw(audio, step_samples, window_samples, global_idx);
        let clean = clean_masks(&dec.view());
        for speaker_idx in 0..num_speakers {
            let mask_idx = local * num_speakers + speaker_idx;
            if mask_idx >= largest_num_masks {
                break;
            }
            if let Some(weights) = select_speaker_weights(
                &dec.view(),
                &clean,
                speaker_idx,
                win_audio.len(),
                min_num_samples,
            ) {
                let dst = mask_idx * 589;
                let cl = weights.len().min(589);
                masks[dst..dst + cl].copy_from_slice(&weights[..cl]);
                active.push((local, speaker_idx));
            }
        }
    }

    Ok(PreparedChunk {
        global_start,
        decoded_chunk: decoded.decoded_chunk.clone(),
        fbank,
        masks,
        active,
        num_masks: largest_num_masks,
    })
}

#[cfg(feature = "coreml")]
fn build_chunk_artifacts(
    step_seconds: f64,
    step_samples: usize,
    window_samples: usize,
    num_speakers: usize,
    decoded_all: Vec<Array2<f32>>,
    embeddings_vec: Vec<SpeakerEmbedding>,
) -> Option<InferenceArtifacts> {
    let num_chunks = decoded_all.len();
    if num_chunks == 0 {
        return None;
    }

    let num_frames = decoded_all[0].nrows();
    let mut segmentations = Array3::<f32>::zeros((num_chunks, num_frames, num_speakers));
    for (i, window) in decoded_all.iter().enumerate() {
        segmentations.slice_mut(s![i, .., ..]).assign(window);
    }

    let mut embeddings = Array3::<f32>::from_elem((num_chunks, num_speakers, 256), f32::NAN);
    for emb in &embeddings_vec {
        embeddings
            .slice_mut(s![emb.chunk_idx, emb.speaker_idx, ..])
            .assign(&ndarray::ArrayView1::from(emb.embedding.as_slice()));
    }

    Some(InferenceArtifacts {
        layout: ChunkLayout::new(step_seconds, step_samples, window_samples, num_chunks),
        segmentations: DecodedSegmentations(segmentations),
        embeddings: ChunkEmbeddings(embeddings),
    })
}

#[derive(Clone, Copy)]
enum InferencePath {
    Sequential,
    Concurrent,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum EmbeddingPath {
    Masked,
    Split,
    MultiMask,
}

struct ConcurrentEmbeddingRunner<'a> {
    powerset: &'a PowersetMapping,
    audio: &'a [f32],
    step_samples: usize,
    window_samples: usize,
    num_speakers: usize,
}

impl<'a> ConcurrentEmbeddingRunner<'a> {
    fn decode_chunk<'chunk>(
        &self,
        raw_window: &Array2<f32>,
        decoded_windows: &'chunk mut Vec<Array2<f32>>,
        chunk_idx: usize,
    ) -> (ArrayView2<'chunk, f32>, &'a [f32], Array2<f32>) {
        decoded_windows.push(self.powerset.hard_decode(raw_window));
        let segmentation_view = decoded_windows.last().unwrap().view();
        let chunk_audio = chunk_audio_raw(
            self.audio,
            self.step_samples,
            self.window_samples,
            chunk_idx,
        );
        let clean_masks = clean_masks(&segmentation_view);
        (segmentation_view, chunk_audio, clean_masks)
    }

    fn run_split(
        &self,
        receiver: crossbeam_channel::Receiver<Array2<f32>>,
        embedding_model: &mut EmbeddingModel,
        batch_size: usize,
        min_num_samples: usize,
    ) -> Result<ConcurrentEmbeddingResult, PipelineError> {
        let mut decoded_windows: Vec<Array2<f32>> = Vec::new();
        let mut embeddings: Vec<SpeakerEmbedding> = Vec::new();
        let mut pending: Vec<PendingSplitEmbedding> = Vec::with_capacity(batch_size);
        let mut fbanks: Vec<Array2<f32>> = Vec::new();
        let mut chunk_idx = 0usize;

        for raw_window in receiver {
            let (segmentation_view, chunk_audio, clean_masks) =
                self.decode_chunk(&raw_window, &mut decoded_windows, chunk_idx);
            let fbank = embedding_model.compute_chunk_fbank(chunk_audio)?;
            fbanks.push(fbank);
            let mut current_fbank_idx = fbanks.len() - 1;

            for speaker_idx in 0..self.num_speakers {
                let Some(weights) = select_speaker_weights(
                    &segmentation_view,
                    &clean_masks,
                    speaker_idx,
                    chunk_audio.len(),
                    min_num_samples,
                ) else {
                    continue;
                };
                pending.push(PendingSplitEmbedding {
                    chunk_idx,
                    speaker_idx,
                    fbank_idx: current_fbank_idx,
                    weights,
                });
                if pending.len() == batch_size {
                    self.flush_split_pending(embedding_model, &pending, &fbanks, &mut embeddings)?;
                    pending.clear();

                    // keep the current chunk fbank alive if later speakers in this chunk still need it
                    if speaker_idx + 1 < self.num_speakers {
                        let kept_fbank = fbanks.swap_remove(current_fbank_idx);
                        fbanks.clear();
                        fbanks.push(kept_fbank);
                        current_fbank_idx = 0;
                    } else {
                        fbanks.clear();
                    }
                }
            }
            chunk_idx += 1;
        }

        if !pending.is_empty() {
            self.flush_split_pending(embedding_model, &pending, &fbanks, &mut embeddings)?;
        }

        Ok(ConcurrentEmbeddingResult {
            decoded_windows,
            embeddings,
            num_speakers: self.num_speakers,
        })
    }

    fn run_multi_mask(
        &self,
        receiver: crossbeam_channel::Receiver<Array2<f32>>,
        embedding_model: &mut EmbeddingModel,
        batch_size: usize,
        min_num_samples: usize,
    ) -> Result<ConcurrentEmbeddingResult, PipelineError> {
        let mut decoded_windows: Vec<Array2<f32>> = Vec::new();
        let mut embeddings: Vec<SpeakerEmbedding> = Vec::new();
        // buffer audio slices instead of pre-computed fbanks — fbanks are computed
        // in a single batched call per flush, avoiding redundant per-window computation
        let mut audio_buffer: Vec<&[f32]> = Vec::with_capacity(batch_size);
        let mut masks_buffer: Vec<Vec<f32>> = Vec::with_capacity(batch_size * self.num_speakers);
        let mut chunk_indices: Vec<usize> = Vec::with_capacity(batch_size);
        let mut chunk_idx = 0usize;

        let mut total_recv_wait_us = 0u64;
        let mut total_decode_us = 0u64;
        let mut total_fbank_us = 0u64;
        let mut total_gpu_predict_us = 0u64;
        let mut flush_count = 0u32;

        loop {
            let recv_start = std::time::Instant::now();
            let raw_window = match receiver.recv() {
                Ok(w) => w,
                Err(_) => break,
            };
            total_recv_wait_us += recv_start.elapsed().as_micros() as u64;

            let decode_start = std::time::Instant::now();
            let (segmentation_view, chunk_audio, clean_masks) =
                self.decode_chunk(&raw_window, &mut decoded_windows, chunk_idx);
            audio_buffer.push(chunk_audio);
            chunk_indices.push(chunk_idx);

            for speaker_idx in 0..self.num_speakers {
                let Some(weights) = select_speaker_weights(
                    &segmentation_view,
                    &clean_masks,
                    speaker_idx,
                    chunk_audio.len(),
                    min_num_samples,
                ) else {
                    masks_buffer.push(vec![0.0; 589]);
                    continue;
                };
                masks_buffer.push(weights);
            }
            total_decode_us += decode_start.elapsed().as_micros() as u64;

            if audio_buffer.len() == batch_size {
                let (fbank_us, gpu_us) = self.flush_multi_mask_timed(
                    embedding_model,
                    &audio_buffer,
                    &masks_buffer,
                    &chunk_indices,
                    &mut embeddings,
                )?;
                total_fbank_us += fbank_us;
                total_gpu_predict_us += gpu_us;
                flush_count += 1;
                audio_buffer.clear();
                masks_buffer.clear();
                chunk_indices.clear();
            }
            chunk_idx += 1;
        }

        if !audio_buffer.is_empty() {
            let (fbank_us, gpu_us) = self.flush_multi_mask_timed(
                embedding_model,
                &audio_buffer,
                &masks_buffer,
                &chunk_indices,
                &mut embeddings,
            )?;
            total_fbank_us += fbank_us;
            total_gpu_predict_us += gpu_us;
            flush_count += 1;
        }

        trace!(
            flushes = flush_count,
            chunks = chunk_idx,
            recv_wait_ms = total_recv_wait_us / 1000,
            decode_ms = total_decode_us / 1000,
            fbank_ms = total_fbank_us / 1000,
            gpu_predict_ms = total_gpu_predict_us / 1000,
            "Multi-mask embedding timing"
        );

        Ok(ConcurrentEmbeddingResult {
            decoded_windows,
            embeddings,
            num_speakers: self.num_speakers,
        })
    }

    fn flush_multi_mask_timed(
        &self,
        embedding_model: &mut EmbeddingModel,
        audio_slices: &[&[f32]],
        masks: &[Vec<f32>],
        chunk_indices: &[usize],
        embeddings: &mut Vec<SpeakerEmbedding>,
    ) -> Result<(u64, u64), PipelineError> {
        let fbank_start = std::time::Instant::now();
        let fbanks = embedding_model.compute_chunk_fbanks_batch(audio_slices)?;
        let fbank_us = fbank_start.elapsed().as_micros() as u64;

        let fbank_refs: Vec<_> = fbanks.iter().collect();
        let mask_refs: Vec<_> = masks.iter().map(|m| m.as_slice()).collect();

        let predict_start = std::time::Instant::now();
        let batch_embeddings = embedding_model.embed_multi_mask_batch(&fbank_refs, &mask_refs)?;
        let predict_us = predict_start.elapsed().as_micros() as u64;

        for (fbank_idx, &chunk_idx) in chunk_indices.iter().enumerate() {
            for speaker_idx in 0..self.num_speakers {
                let mask_idx = fbank_idx * self.num_speakers + speaker_idx;
                let is_active = masks[mask_idx].iter().any(|&v| v > 0.0);
                if !is_active {
                    continue;
                }
                embeddings.push(SpeakerEmbedding {
                    chunk_idx,
                    speaker_idx,
                    embedding: batch_embeddings.row(mask_idx).to_vec(),
                });
            }
        }

        Ok((fbank_us, predict_us))
    }

    fn flush_split_pending(
        &self,
        embedding_model: &mut EmbeddingModel,
        pending: &[PendingSplitEmbedding],
        fbanks: &[Array2<f32>],
        embeddings: &mut Vec<SpeakerEmbedding>,
    ) -> Result<(), PipelineError> {
        let batch_inputs: Vec<_> = pending
            .iter()
            .map(|item| SplitTailInput {
                fbank: &fbanks[item.fbank_idx],
                weights: &item.weights,
            })
            .collect();
        let batch_embeddings = embedding_model.embed_tail_batch_inputs(&batch_inputs)?;
        for (batch_idx, item) in pending.iter().enumerate() {
            embeddings.push(SpeakerEmbedding {
                chunk_idx: item.chunk_idx,
                speaker_idx: item.speaker_idx,
                embedding: batch_embeddings.row(batch_idx).to_vec(),
            });
        }
        Ok(())
    }

    fn run_masked(
        &self,
        receiver: crossbeam_channel::Receiver<Array2<f32>>,
        embedding_model: &mut EmbeddingModel,
        batch_size: usize,
    ) -> Result<ConcurrentEmbeddingResult, PipelineError> {
        let mut decoded_windows: Vec<Array2<f32>> = Vec::new();
        let mut embeddings: Vec<SpeakerEmbedding> = Vec::new();
        let mut pending: Vec<PendingEmbedding<'_>> = Vec::with_capacity(batch_size);
        let mut chunk_idx = 0usize;
        let mut emb_calls = 0u32;
        let mut emb_batched = 0u32;
        let mut emb_single = 0u32;
        let mut total_speakers = 0u32;
        let mut skipped_speakers = 0u32;
        let mut channel_wait = std::time::Duration::ZERO;
        let mut decode_time = std::time::Duration::ZERO;
        let mut embed_time = std::time::Duration::ZERO;
        let emb_start = std::time::Instant::now();

        loop {
            let recv_start = std::time::Instant::now();
            let raw_window = match receiver.recv() {
                Ok(w) => w,
                Err(_) => break,
            };
            channel_wait += recv_start.elapsed();

            let decode_start = std::time::Instant::now();
            let (segmentation_view, chunk_audio, clean_masks) =
                self.decode_chunk(&raw_window, &mut decoded_windows, chunk_idx);

            for speaker_idx in 0..self.num_speakers {
                total_speakers += 1;
                let mask_col = segmentation_view.column(speaker_idx);
                let activity: f32 = mask_col.iter().sum();
                if activity < MIN_SPEAKER_ACTIVITY {
                    skipped_speakers += 1;
                    continue;
                }

                pending.push(PendingEmbedding {
                    chunk_idx,
                    speaker_idx,
                    audio: chunk_audio,
                    mask: mask_col.to_vec(),
                    clean_mask: clean_masks.column(speaker_idx).to_vec(),
                });
                if pending.len() == batch_size {
                    decode_time += decode_start.elapsed();
                    let flush_start = std::time::Instant::now();
                    self.flush_masked_pending(embedding_model, &pending, &mut embeddings)?;
                    embed_time += flush_start.elapsed();
                    emb_calls += 1;
                    emb_batched += 1;
                    pending.clear();
                    // restart decode timer for remaining speakers in this chunk
                    // (minor: just let it accumulate)
                }
            }
            decode_time += decode_start.elapsed();
            chunk_idx += 1;
        }

        while !pending.is_empty() {
            let batch_len = embedding_model.best_batch_len(pending.len());
            let flush_start = std::time::Instant::now();
            self.flush_masked_pending(embedding_model, &pending[..batch_len], &mut embeddings)?;
            embed_time += flush_start.elapsed();
            emb_calls += 1;
            emb_single += 1;
            pending.drain(..batch_len);
        }

        let total_emb = emb_start.elapsed();
        info!(
            chunks = chunk_idx,
            total_speakers,
            skipped_speakers,
            active_speakers = total_speakers - skipped_speakers,
            emb_calls,
            emb_batched,
            emb_single,
            channel_wait_ms = channel_wait.as_millis(),
            decode_ms = decode_time.as_millis(),
            embed_ms = embed_time.as_millis(),
            total_emb_ms = total_emb.as_millis(),
            "Embedding thread profile"
        );

        Ok(ConcurrentEmbeddingResult {
            decoded_windows,
            embeddings,
            num_speakers: self.num_speakers,
        })
    }

    fn flush_masked_pending(
        &self,
        embedding_model: &mut EmbeddingModel,
        pending: &[PendingEmbedding<'_>],
        embeddings: &mut Vec<SpeakerEmbedding>,
    ) -> Result<(), PipelineError> {
        let batch_inputs: Vec<_> = pending
            .iter()
            .map(|item| MaskedEmbeddingInput {
                audio: item.audio,
                mask: &item.mask,
                clean_mask: Some(&item.clean_mask),
            })
            .collect();
        let batch_embeddings = embedding_model.embed_batch(&batch_inputs)?;
        for (batch_idx, item) in pending.iter().enumerate() {
            embeddings.push(SpeakerEmbedding {
                chunk_idx: item.chunk_idx,
                speaker_idx: item.speaker_idx,
                embedding: batch_embeddings.row(batch_idx).to_vec(),
            });
        }
        Ok(())
    }
}

/// Select speaker weights for embedding, returning None if speaker activity is below threshold
fn select_speaker_weights(
    seg_view: &ArrayView2<f32>,
    clean_masks: &Array2<f32>,
    speaker_idx: usize,
    audio_len: usize,
    min_num_samples: usize,
) -> Option<Vec<f32>> {
    let mask_col = seg_view.column(speaker_idx);
    let activity: f32 = mask_col.iter().sum();
    if activity < MIN_SPEAKER_ACTIVITY {
        return None;
    }

    let clean_col = clean_masks.column(speaker_idx);
    let use_clean = should_use_clean_mask(&clean_col, mask_col.len(), audio_len, min_num_samples);
    if use_clean {
        Some(clean_col.iter().copied().collect())
    } else {
        Some(mask_col.iter().copied().collect())
    }
}

#[derive(Debug, Clone)]
pub struct DiarizationResult {
    pub segmentations: DecodedSegmentations,
    pub embeddings: ChunkEmbeddings,
    pub speaker_count: SpeakerCountTrack,
    pub hard_clusters: ChunkSpeakerClusters,
    pub discrete_diarization: DiscreteDiarization,
    pub rttm: String,
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

    pub fn run(&mut self, audio: &[f32]) -> Result<DiarizationResult, PipelineError> {
        self.run_with_file_id(audio, "file1")
    }

    pub fn run_with_file_id(
        &mut self,
        audio: &[f32],
        file_id: &str,
    ) -> Result<DiarizationResult, PipelineError> {
        self.run_with_config(audio, file_id, &PipelineConfig::for_mode(self.mode))
    }

    pub fn run_with_config(
        &mut self,
        audio: &[f32],
        file_id: &str,
        config: &PipelineConfig,
    ) -> Result<DiarizationResult, PipelineError> {
        self.runner().run(audio, file_id, config)
    }

    /// Run only inference (segmentation + embedding), returning intermediate artifacts
    ///
    /// Use with `finish_post_inference` to enable multi-file overlap:
    /// post-inference for file N runs on a background thread while
    /// inference for file N+1 starts
    pub fn run_inference_only(
        &mut self,
        audio: &[f32],
    ) -> Result<InferenceArtifacts, PipelineError> {
        self.runner().run_inference(audio)
    }

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

    /// Pipeline config for the current execution mode
    pub fn pipeline_config(&self) -> PipelineConfig {
        PipelineConfig::for_mode(self.mode)
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

    pub fn segmentation_step(&self) -> f64 {
        self.seg_model.step_seconds()
    }
}

pub struct DiarizationPipeline<'a> {
    seg_model: &'a mut SegmentationModel,
    emb_model: &'a mut EmbeddingModel,
    plda: PldaTransform,
    powerset: PowersetMapping,
    mode: ExecutionMode,
}

impl<'a> DiarizationPipeline<'a> {
    pub fn new(
        seg_model: &'a mut SegmentationModel,
        emb_model: &'a mut EmbeddingModel,
        models_dir: &Path,
    ) -> Result<Self, PipelineError> {
        Self::new_with_config(seg_model, emb_model, models_dir, RuntimeConfig::default())
    }

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

    pub fn default_segmentation_step() -> f32 {
        SEGMENTATION_STEP_SECONDS as f32
    }

    pub fn run(&mut self, audio: &[f32]) -> Result<DiarizationResult, PipelineError> {
        self.run_with_file_id(audio, "file1")
    }

    pub fn run_with_file_id(
        &mut self,
        audio: &[f32],
        file_id: &str,
    ) -> Result<DiarizationResult, PipelineError> {
        self.run_with_config(audio, file_id, &PipelineConfig::for_mode(self.mode))
    }

    pub fn run_with_config(
        &mut self,
        audio: &[f32],
        file_id: &str,
        config: &PipelineConfig,
    ) -> Result<DiarizationResult, PipelineError> {
        self.runner().run(audio, file_id, config)
    }

    pub fn segmentation_step(&self) -> f64 {
        self.seg_model.step_seconds()
    }

    /// Run only inference, returning intermediate artifacts for deferred post-processing
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
}

impl DiarizationResult {
    pub fn rttm(&self, file_id: &str) -> String {
        if file_id == "file1" {
            return self.rttm.clone();
        }

        self.rttm
            .replace("SPEAKER file1 1", &format!("SPEAKER {file_id} 1"))
    }
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

    fn run_inference(&mut self, audio: &[f32]) -> Result<InferenceArtifacts, PipelineError> {
        match self.inference_path() {
            InferencePath::Sequential => self.run_sequential_inference(audio),
            InferencePath::Concurrent => {
                #[cfg(feature = "coreml")]
                if let Some(result) = self.try_chunk_embedding(audio)? {
                    return Ok(result);
                }
                self.run_concurrent_inference(audio)
            }
        }
    }

    /// Number of parallel segmentation workers for CoreML
    #[cfg(feature = "coreml")]
    fn seg_worker_count() -> usize {
        std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(4)
            .min(8)
    }

    #[cfg(feature = "coreml")]
    fn chunk_embedding_resources(&mut self) -> Option<ChunkEmbeddingResources> {
        // only load the largest chunk session (w56 for CoreMlFast)
        let capacity = self.emb_model.chunk_window_capacity()?;
        self.emb_model.ensure_chunk_session_loaded_pub(capacity);
        let chunk_sessions: Vec<ChunkSessionPtr> = self
            .emb_model
            .chunk_session_refs()
            .into_iter()
            .map(|session| ChunkSessionPtr {
                cached_fbank_shape: SharedShapePtr(session.cached_fbank_shape as *const _ as usize),
                cached_masks_shape: SharedShapePtr(session.cached_masks_shape as *const _ as usize),
                model: SharedModelPtr(session.model as *const _ as usize),
            })
            .collect();
        let chunk_lookup = self
            .emb_model
            .chunk_session_refs()
            .into_iter()
            .map(|session| (session.num_windows, session.fbank_frames, session.num_masks))
            .collect();

        // ANE disabled: ResNet34 is ~6x slower on ANE than GPU even with
        // async prediction API (tested sync and async, 1-4 workers, FP32 and W8A16).
        // The model likely exceeds ANE SRAM causing memory bandwidth bottleneck
        let ane_session: Option<ChunkSessionPtr> = None;

        Some(ChunkEmbeddingResources {
            chunk_sessions,
            chunk_lookup,
            fbank_30s_ptr: self
                .emb_model
                .fbank_30s_refs()
                .map(|(model, _)| SharedModelPtr(model as *const _ as usize)),
            fbank_10s_ptr: self
                .emb_model
                .fbank_10s_ref()
                .map(|model| SharedModelPtr(model as *const _ as usize)),
            ane_session,
        })
    }

    /// Pipelined chunk embedding: seg (CPUOnly) and emb (GPU) run on separate
    /// threads processing 30s chunks. While emb processes chunk N, seg produces
    /// chunk N+1. Seg uses run_streaming_parallel (CPUOnly workers), emb uses
    /// chunk embedding models (GPU). Different hardware = no contention
    #[cfg(feature = "coreml")]
    fn try_chunk_embedding(
        &mut self,
        audio: &[f32],
    ) -> Result<Option<InferenceArtifacts>, PipelineError> {
        let step_samples = self.seg_model.step_samples();
        let window_samples = self.seg_model.window_samples();
        if audio.len() < window_samples {
            return Ok(None);
        }

        let Some(chunk_win_capacity) = self.emb_model.chunk_window_capacity() else {
            return Ok(None);
        };

        let inference_start = std::time::Instant::now();
        let num_speakers = 3;
        let powerset = self.powerset;
        let min_num_samples = self.emb_model.min_num_samples();
        let step_seconds = self.seg_model.step_seconds();
        use crate::inference::coreml::CachedInputShape;

        // pipelining threshold: only for files with 3+ chunks
        let total_windows = audio.len().saturating_sub(window_samples) / step_samples + 1;
        let est_chunks = total_windows.div_ceil(chunk_win_capacity);
        let use_pipelined = est_chunks >= 2;
        let seg_warm_start_windows = chunk_win_capacity * 2;
        let chunk_resources = use_pipelined
            .then(|| self.chunk_embedding_resources())
            .flatten();

        let (seg_tx, seg_rx) = crossbeam_channel::bounded::<Array2<f32>>(100);
        let (chunk_tx, chunk_rx) = crossbeam_channel::bounded::<DecodedChunk>(100);
        let chunk_rx_ref = &chunk_rx;
        let (prep_tx, prep_rx) = crossbeam_channel::bounded::<PreparedChunk>(4);
        let prep_rx_ref = &prep_rx;
        let (emb_tx, emb_rx) = crossbeam_channel::bounded::<EmbeddedChunk>(8);

        let seg_model = &mut *self.seg_model;
        let emb_model = &mut *self.emb_model;

        let result: Result<Option<InferenceArtifacts>, PipelineError> = std::thread::scope(
            |scope| {
                // seg thread: batched CPU-only segmentation feeding the bridge
                let seg_start = std::time::Instant::now();
                let seg_handle =
                    scope.spawn(move || -> Result<std::time::Duration, PipelineError> {
                        seg_model.run_streaming_parallel(
                            audio,
                            seg_tx,
                            Self::seg_worker_count(),
                            Some(seg_warm_start_windows),
                        )?;
                        Ok(seg_start.elapsed())
                    });

                // bridge thread: groups raw windows into decoded chunks
                let bridge_handle = scope.spawn(move || {
                    let mut group: Vec<Array2<f32>> = Vec::with_capacity(chunk_win_capacity);
                    let mut global_start = 0usize;

                    for raw_window in &seg_rx {
                        group.push(powerset.hard_decode(&raw_window));

                        if group.len() == chunk_win_capacity {
                            if chunk_tx
                                .send(DecodedChunk {
                                    global_start,
                                    decoded_chunk: std::mem::take(&mut group),
                                    precomputed_fbank: None,
                                })
                                .is_err()
                            {
                                break;
                            }
                            global_start += chunk_win_capacity;
                            group = Vec::with_capacity(chunk_win_capacity);
                        }
                    }
                    if !group.is_empty() {
                        let _ = chunk_tx.send(DecodedChunk {
                            global_start,
                            decoded_chunk: group,
                            precomputed_fbank: None,
                        });
                    }
                });

                // emb processing: heterogeneous compute (2+ chunks) or sequential fallback
                let emb_start = std::time::Instant::now();
                let mut decoded_all: Vec<Array2<f32>> = Vec::new();
                let mut embeddings_vec: Vec<SpeakerEmbedding> = Vec::new();

                // summary timing vars filled by whichever branch runs
                let summary_gpu_predict_us;
                let summary_prep_fbank_us;
                let summary_prep_mask_us;

                if use_pipelined {
                    let Some(ChunkEmbeddingResources {
                        chunk_sessions,
                        chunk_lookup,
                        fbank_30s_ptr,
                        fbank_10s_ptr,
                        ane_session,
                    }) = chunk_resources.clone()
                    else {
                        return Ok(None);
                    };

                    // use only the largest chunk session (w56 for CoreMlFast)
                    let largest_session = *chunk_sessions.last().unwrap();
                    let largest_fbank_frames = chunk_lookup.last().unwrap().1;
                    let largest_num_masks = chunk_lookup.last().unwrap().2;
                    let max_active = chunk_win_capacity * num_speakers;

                    // CPU prep threads: pull decoded chunks, prep fbank+masks, feed prepared queue
                    let num_prep_threads = 2usize;
                    let mut prep_handles = Vec::with_capacity(num_prep_threads);
                    for _prep_id in 0..num_prep_threads {
                        let prep_tx = prep_tx.clone();
                        prep_handles.push(scope.spawn(
                            move || -> Result<PrepStats, PipelineError> {
                                let fbank_30s_shape =
                                    CachedInputShape::new("waveform", &[1, 1, 480_000]);
                                let fbank_10s_shape =
                                    CachedInputShape::new("waveform", &[1, 1, window_samples]);
                                let mut fbank_30s_buf = vec![0.0f32; 480_000];
                                let mut waveform_10s_buf = vec![0.0f32; window_samples];
                                let mut stats = PrepStats::default();

                                while let Ok(decoded) = chunk_rx_ref.recv() {
                                    let chunk_audio_start = decoded.global_start * step_samples;
                                    if chunk_audio_start + window_samples > audio.len() {
                                        continue;
                                    }
                                    let prep_start = std::time::Instant::now();
                                    let prepared = prep_decoded_chunk(
                                        &decoded, audio, step_samples, window_samples,
                                        num_speakers, min_num_samples,
                                        largest_fbank_frames, largest_num_masks, max_active,
                                        fbank_30s_ptr, fbank_10s_ptr,
                                        &mut fbank_30s_buf, &mut waveform_10s_buf,
                                        &fbank_30s_shape, &fbank_10s_shape,
                                    )?;
                                    stats.fbank_us +=
                                        prep_start.elapsed().as_micros() as u64;
                                    stats.chunks += 1;
                                    if prep_tx.send(prepared).is_err() {
                                        break;
                                    }
                                }
                                Ok(stats)
                            },
                        ));
                    }
                    drop(prep_tx);

                    // GPU worker: work-stealing — tries prepared queue first,
                    // falls back to decoded queue and does prep+predict itself
                    let gpu_emb_tx = emb_tx.clone();
                    let gpu_handle = scope.spawn(move || -> Result<(u64, u32, u64), PipelineError> {
                        let model = unsafe {
                            &*(largest_session.model.0
                                as *const crate::inference::coreml::SharedCoreMlModel)
                        };
                        let fbank_shape = unsafe {
                            &*(largest_session.cached_fbank_shape.0
                                as *const crate::inference::coreml::CachedInputShape)
                        };
                        let masks_shape = unsafe {
                            &*(largest_session.cached_masks_shape.0
                                as *const crate::inference::coreml::CachedInputShape)
                        };
                        let fbank_30s_shape =
                            CachedInputShape::new("waveform", &[1, 1, 480_000]);
                        let fbank_10s_shape =
                            CachedInputShape::new("waveform", &[1, 1, window_samples]);
                        let mut fbank_30s_buf = vec![0.0f32; 480_000];
                        let mut waveform_10s_buf = vec![0.0f32; window_samples];
                        let mut total_predict_us = 0u64;
                        let mut total_prep_us = 0u64;
                        let mut chunk_num = 0u32;
                        let mut decoded_done = false;

                        loop {
                            // try prepared queue first (non-blocking)
                            let prepared = match prep_rx_ref.try_recv() {
                                Ok(p) => p,
                                Err(crossbeam_channel::TryRecvError::Empty) => {
                                    // nothing prepared yet — steal a decoded chunk and prep it ourselves
                                    if decoded_done {
                                        // no more decoded chunks, block on prepared queue
                                        match prep_rx_ref.recv() {
                                            Ok(p) => p,
                                            Err(_) => break,
                                        }
                                    } else {
                                        match chunk_rx_ref.try_recv() {
                                            Ok(decoded) => {
                                                let prep_start = std::time::Instant::now();
                                                let p = prep_decoded_chunk(
                                                    &decoded, audio, step_samples, window_samples,
                                                    num_speakers, min_num_samples,
                                                    largest_fbank_frames, largest_num_masks, max_active,
                                                    fbank_30s_ptr, fbank_10s_ptr,
                                                    &mut fbank_30s_buf, &mut waveform_10s_buf,
                                                    &fbank_30s_shape, &fbank_10s_shape,
                                                )?;
                                                total_prep_us += prep_start.elapsed().as_micros() as u64;
                                                p
                                            }
                                            Err(crossbeam_channel::TryRecvError::Empty) => {
                                                // both empty, block on either
                                                crossbeam_channel::select! {
                                                    recv(prep_rx_ref) -> msg => match msg {
                                                        Ok(p) => p,
                                                        Err(_) => break,
                                                    },
                                                    recv(chunk_rx_ref) -> msg => match msg {
                                                        Ok(decoded) => {
                                                            let prep_start = std::time::Instant::now();
                                                            let p = prep_decoded_chunk(
                                                                &decoded, audio, step_samples, window_samples,
                                                                num_speakers, min_num_samples,
                                                                largest_fbank_frames, largest_num_masks, max_active,
                                                                fbank_30s_ptr, fbank_10s_ptr,
                                                                &mut fbank_30s_buf, &mut waveform_10s_buf,
                                                                &fbank_30s_shape, &fbank_10s_shape,
                                                            )?;
                                                            total_prep_us += prep_start.elapsed().as_micros() as u64;
                                                            p
                                                        }
                                                        Err(_) => {
                                                            decoded_done = true;
                                                            match prep_rx_ref.recv() {
                                                                Ok(p) => p,
                                                                Err(_) => break,
                                                            }
                                                        }
                                                    },
                                                }
                                            }
                                            Err(crossbeam_channel::TryRecvError::Disconnected) => {
                                                decoded_done = true;
                                                match prep_rx_ref.recv() {
                                                    Ok(p) => p,
                                                    Err(_) => break,
                                                }
                                            }
                                        }
                                    }
                                }
                                Err(crossbeam_channel::TryRecvError::Disconnected) => break,
                            };

                            let predict_start = std::time::Instant::now();
                            let (data, _) = model
                                .predict_cached(&[
                                    (fbank_shape, &prepared.fbank),
                                    (masks_shape, &prepared.masks),
                                ])
                                .map_err(|e| PipelineError::Other(e.to_string()))?;
                            let predict_us =
                                predict_start.elapsed().as_micros() as u64;
                            total_predict_us += predict_us;

                            trace!(
                                chunk_num,
                                chunk_start = prepared.global_start,
                                predict_ms = predict_us / 1000,
                                "GPU chunk"
                            );
                            chunk_num += 1;

                            if gpu_emb_tx
                                .send(EmbeddedChunk {
                                    global_start: prepared.global_start,
                                    decoded_chunk: prepared.decoded_chunk,
                                    data,
                                    active: prepared.active,
                                    num_masks: prepared.num_masks,
                                    predict_us,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        Ok((total_predict_us, chunk_num, total_prep_us))
                    });

                    // ANE predict workers: pull from prepared queue alongside GPU
                    // ANE supports true concurrency (unlike GPU which serializes),
                    // so N workers can run in parallel on different ANE cores
                    let ane_num_workers = if ane_session.is_some() { 4usize } else { 0 };
                    let mut ane_handles = Vec::with_capacity(ane_num_workers);
                    if let Some(ane_sess) = ane_session {
                        for _worker_id in 0..ane_num_workers {
                            let ane_emb_tx = emb_tx.clone();
                            ane_handles.push(scope.spawn(move || -> Result<(u64, u32), PipelineError> {
                                let model = unsafe {
                                    &*(ane_sess.model.0
                                        as *const crate::inference::coreml::SharedCoreMlModel)
                                };
                                let fbank_shape = unsafe {
                                    &*(ane_sess.cached_fbank_shape.0
                                        as *const crate::inference::coreml::CachedInputShape)
                                };
                                let masks_shape = unsafe {
                                    &*(ane_sess.cached_masks_shape.0
                                        as *const crate::inference::coreml::CachedInputShape)
                                };
                                let mut total_predict_us = 0u64;
                                let mut chunks = 0u32;

                                while let Ok(prepared) = prep_rx_ref.recv() {
                                    let predict_start = std::time::Instant::now();
                                    let (data, _) = model
                                        .predict_async(&[
                                            (fbank_shape, &prepared.fbank),
                                            (masks_shape, &prepared.masks),
                                        ])
                                        .map_err(|e| PipelineError::Other(e.to_string()))?;
                                    let predict_us = predict_start.elapsed().as_micros() as u64;
                                    total_predict_us += predict_us;
                                    chunks += 1;

                                    if ane_emb_tx
                                        .send(EmbeddedChunk {
                                            global_start: prepared.global_start,
                                            decoded_chunk: prepared.decoded_chunk,
                                            data,
                                            active: prepared.active,
                                            num_masks: prepared.num_masks,
                                            predict_us,
                                        })
                                        .is_err()
                                    {
                                        break;
                                    }
                                }
                                Ok((total_predict_us, chunks))
                            }));
                        }
                    }
                    drop(emb_tx);

                    // collect results (out-of-order from workers)
                    let mut decoded_slots: Vec<Option<Array2<f32>>> =
                        std::iter::repeat_with(|| None)
                            .take(total_windows + chunk_win_capacity)
                            .collect();
                    let mut total_predict_us = 0u64;
                    let mut total_chunks = 0u32;

                    while let Ok(embedded) = emb_rx.recv() {
                        total_predict_us += embedded.predict_us;

                        let batch_emb =
                            Array2::from_shape_vec((embedded.num_masks, 256), embedded.data)
                                .unwrap();

                        for &(local, speaker_idx) in &embedded.active {
                            let mask_idx = local * num_speakers + speaker_idx;
                            embeddings_vec.push(SpeakerEmbedding {
                                chunk_idx: embedded.global_start + local,
                                speaker_idx,
                                embedding: batch_emb.row(mask_idx).to_vec(),
                            });
                        }

                        for (local, decoded) in embedded.decoded_chunk.into_iter().enumerate()
                        {
                            let slot = embedded.global_start + local;
                            if slot < decoded_slots.len() {
                                decoded_slots[slot] = Some(decoded);
                            }
                        }

                        total_chunks += 1;
                    }

                    // join workers
                    let (gpu_predict_us, gpu_chunks, gpu_self_prep_us) =
                        gpu_handle.join().unwrap()?;

                    let mut ane_predict_us = 0u64;
                    let mut ane_chunks = 0u32;
                    for handle in ane_handles {
                        let (predict_us, chunks) = handle.join().unwrap()?;
                        ane_predict_us += predict_us;
                        ane_chunks += chunks;
                    }

                    let mut total_prep_fbank_us = 0u64;
                    for handle in prep_handles {
                        let stats = handle.join().unwrap()?;
                        total_prep_fbank_us += stats.fbank_us;
                    }

                    // reconstruct ordered decoded_all from slots, remapping
                    // embedding chunk_idx from absolute window position to sequential index
                    let mut abs_to_seq: Vec<usize> = vec![0; decoded_slots.len()];
                    let mut seq_idx = 0usize;
                    for (abs_idx, slot) in decoded_slots.iter().enumerate() {
                        if slot.is_some() {
                            abs_to_seq[abs_idx] = seq_idx;
                            seq_idx += 1;
                        }
                    }
                    decoded_all = decoded_slots.into_iter().flatten().collect();
                    for emb in &mut embeddings_vec {
                        emb.chunk_idx = abs_to_seq[emb.chunk_idx];
                    }

                    debug!(
                        total_chunks,
                        gpu_chunks,
                        ane_chunks,
                        ane_workers = ane_num_workers,
                        gpu_predict_ms = gpu_predict_us / 1000,
                        gpu_self_prep_ms = gpu_self_prep_us / 1000,
                        ane_predict_ms = ane_predict_us / 1000,
                        cpu_prep_ms = total_prep_fbank_us / 1000,
                        predict_ms = total_predict_us / 1000,
                        emb_wall_ms = emb_start.elapsed().as_millis(),
                        "Work-stealing breakdown"
                    );
                    summary_gpu_predict_us = total_predict_us;
                    summary_prep_fbank_us = total_prep_fbank_us + gpu_self_prep_us;
                    summary_prep_mask_us = 0;
                } else {
                    // fallback: sequential fbank + masks + embed (single chunk)
                    let mut seq_fbank_us = 0u64;
                    let mut seq_mask_us = 0u64;
                    let mut seq_predict_us = 0u64;
                    let mut seq_chunks = 0u32;

                    for DecodedChunk {
                        global_start,
                        decoded_chunk,
                        precomputed_fbank,
                    } in &chunk_rx
                    {
                        let wins = decoded_chunk.len();
                        let session = emb_model.chunk_session_for_windows(wins).unwrap();
                        let sess_fbank_frames = session.fbank_frames;
                        let sess_num_masks = session.num_masks;
                        let _ = session;

                        let chunk_audio_start = global_start * step_samples;
                        if chunk_audio_start + window_samples > audio.len() {
                            continue;
                        }
                        let chunk_audio_len = window_samples + (wins - 1) * step_samples;
                        let chunk_audio_end =
                            (chunk_audio_start + chunk_audio_len).min(audio.len());
                        let chunk_audio = &audio[chunk_audio_start..chunk_audio_end];

                        let mut fbank = vec![0.0f32; sess_fbank_frames * 80];
                        let fbank_start = std::time::Instant::now();

                        #[cfg(feature = "coreml")]
                        if let Some(ref pre_data) = precomputed_fbank {
                            let copy_len = pre_data.len().min(fbank.len());
                            fbank[..copy_len].copy_from_slice(&pre_data[..copy_len]);
                        } else if let Some(result) = emb_model.compute_chunk_fbank_30s(chunk_audio)
                        {
                            let full_fbank = result?;
                            let copy_frames = full_fbank.nrows().min(sess_fbank_frames);
                            for r in 0..copy_frames {
                                let dst = r * 80;
                                fbank[dst..dst + 80]
                                    .copy_from_slice(full_fbank.row(r).as_slice().unwrap());
                            }
                        } else {
                            let mut fb_off = 0usize;
                            let mut au_off = 0usize;
                            while fb_off < sess_fbank_frames && au_off < chunk_audio.len() {
                                let seg_end = (au_off + window_samples).min(chunk_audio.len());
                                let seg_fbank =
                                    emb_model.compute_chunk_fbank(&chunk_audio[au_off..seg_end])?;
                                let copy = seg_fbank.nrows().min(sess_fbank_frames - fb_off);
                                for r in 0..copy {
                                    let dst = (fb_off + r) * 80;
                                    fbank[dst..dst + 80]
                                        .copy_from_slice(seg_fbank.row(r).as_slice().unwrap());
                                }
                                fb_off += 998;
                                au_off += window_samples;
                            }
                        }
                        #[cfg(not(feature = "coreml"))]
                        {
                            let mut fb_off = 0usize;
                            let mut au_off = 0usize;
                            while fb_off < sess_fbank_frames && au_off < chunk_audio.len() {
                                let seg_end = (au_off + window_samples).min(chunk_audio.len());
                                let seg_fbank =
                                    emb_model.compute_chunk_fbank(&chunk_audio[au_off..seg_end])?;
                                let copy = seg_fbank.nrows().min(sess_fbank_frames - fb_off);
                                for r in 0..copy {
                                    let dst = (fb_off + r) * 80;
                                    fbank[dst..dst + 80]
                                        .copy_from_slice(seg_fbank.row(r).as_slice().unwrap());
                                }
                                fb_off += 998;
                                au_off += window_samples;
                            }
                        }

                        seq_fbank_us += fbank_start.elapsed().as_micros() as u64;

                        // masks
                        let mask_start = std::time::Instant::now();
                        let mut masks = vec![0.0f32; sess_num_masks * 589];
                        let mut active: Vec<(usize, usize)> = Vec::new();
                        for (local, decoded) in decoded_chunk.iter().enumerate() {
                            let global_idx = global_start + local;
                            let win_audio =
                                chunk_audio_raw(audio, step_samples, window_samples, global_idx);
                            let clean = clean_masks(&decoded.view());
                            for speaker_idx in 0..num_speakers {
                                let mask_idx = local * num_speakers + speaker_idx;
                                if mask_idx >= sess_num_masks {
                                    break;
                                }
                                if let Some(weights) = select_speaker_weights(
                                    &decoded.view(),
                                    &clean,
                                    speaker_idx,
                                    win_audio.len(),
                                    min_num_samples,
                                ) {
                                    let dst = mask_idx * 589;
                                    let cl = weights.len().min(589);
                                    masks[dst..dst + cl].copy_from_slice(&weights[..cl]);
                                    active.push((local, speaker_idx));
                                }
                            }
                        }

                        seq_mask_us += mask_start.elapsed().as_micros() as u64;

                        let predict_start = std::time::Instant::now();
                        let session = emb_model.chunk_session_for_windows(wins).unwrap();
                        let batch_emb =
                            EmbeddingModel::embed_chunk_session(session, &fbank, &masks)?;
                        seq_predict_us += predict_start.elapsed().as_micros() as u64;
                        seq_chunks += 1;

                        for &(local, speaker_idx) in &active {
                            let mask_idx = local * num_speakers + speaker_idx;
                            embeddings_vec.push(SpeakerEmbedding {
                                chunk_idx: global_start + local,
                                speaker_idx,
                                embedding: batch_emb.row(mask_idx).to_vec(),
                            });
                        }

                        decoded_all.extend(decoded_chunk);
                    }
                    trace!(
                        seq_chunks,
                        fbank_ms = seq_fbank_us / 1000,
                        mask_ms = seq_mask_us / 1000,
                        predict_ms = seq_predict_us / 1000,
                        wall_ms = emb_start.elapsed().as_millis(),
                        "EMB sequential",
                    );
                    summary_gpu_predict_us = seq_predict_us;
                    summary_prep_fbank_us = seq_fbank_us;
                    summary_prep_mask_us = seq_mask_us;
                }

                let emb_elapsed = emb_start.elapsed();

                let seg_thread_elapsed = seg_handle.join().unwrap()?;
                bridge_handle.join().unwrap();
                trace!(
                    seg_thread_ms = seg_thread_elapsed.as_millis(),
                    seg_wall_ms = seg_start.elapsed().as_millis(),
                    "SEG timing"
                );

                let num_chunks = decoded_all.len();
                let Some(artifacts) = build_chunk_artifacts(
                    step_seconds,
                    step_samples,
                    window_samples,
                    num_speakers,
                    decoded_all,
                    embeddings_vec,
                ) else {
                    return Ok(None);
                };

                let inference_elapsed = inference_start.elapsed();
                let audio_secs = audio.len() as f64 / 16_000.0;
                info!(
                    chunks = num_chunks,
                    chunk_capacity = chunk_win_capacity,
                    pipelined = use_pipelined,
                    seg_ms = seg_thread_elapsed.as_millis(),
                    emb_ms = emb_elapsed.as_millis(),
                    predict_ms = summary_gpu_predict_us / 1000,
                    prep_fbank_ms = summary_prep_fbank_us / 1000,
                    prep_mask_ms = summary_prep_mask_us / 1000,
                    total_ms = inference_elapsed.as_millis(),
                    audio_secs = audio_secs as u64,
                    "Chunk embedding complete"
                );

                Ok(Some(artifacts))
            },
        );

        result
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

        let num_chunks = concurrent_result.decoded_windows.len();
        let (segmentations, embeddings) = concurrent_result.into_arrays();
        let layout = layout.with_num_chunks(num_chunks);
        info!(
            chunks = segmentations.shape()[0],
            speakers = segmentations.shape()[2],
            inference_ms = inference_elapsed.as_millis(),
            embedding_path = ?embedding_path,
            "Concurrent seg+emb complete"
        );

        Ok(InferenceArtifacts {
            layout,
            segmentations: DecodedSegmentations(segmentations),
            embeddings: ChunkEmbeddings(embeddings),
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

/// Post-inference processing: clustering + reconstruction + RTTM generation
///
/// Standalone function that only needs &PldaTransform (no model access),
/// enabling multi-file overlap where post-inference runs on a background
/// thread while the next file's inference starts
pub fn post_inference(
    inference_artifacts: InferenceArtifacts,
    file_id: &str,
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
            rttm: String::new(),
        });
    }

    let training_embeddings = embeddings.training_set(&segmentations);
    let hard_clusters =
        training_embeddings.cluster(&segmentations, &embeddings, plda, config);

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
    let rttm = to_rttm(&segments, file_id);

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
        rttm,
    })
}

impl ChunkLayout {
    fn new(
        step_seconds: f64,
        step_samples: usize,
        window_samples: usize,
        num_chunks: usize,
    ) -> Self {
        Self {
            step_seconds,
            step_samples,
            window_samples,
            start_frames: chunk_start_frames(num_chunks, step_seconds),
            output_frames: total_output_frames(num_chunks, step_seconds),
        }
    }

    fn without_frame_extent(step_seconds: f64, step_samples: usize, window_samples: usize) -> Self {
        Self::new(step_seconds, step_samples, window_samples, 0)
    }

    fn with_num_chunks(mut self, num_chunks: usize) -> Self {
        self.start_frames = chunk_start_frames(num_chunks, self.step_seconds);
        self.output_frames = total_output_frames(num_chunks, self.step_seconds);
        self
    }

    fn chunk_audio<'a>(&self, audio: &'a [f32], chunk_idx: usize) -> &'a [f32] {
        chunk_audio_raw(audio, self.step_samples, self.window_samples, chunk_idx)
    }
}

impl RawSegmentationWindows {
    fn decode(self, powerset: &PowersetMapping) -> DecodedSegmentations {
        let num_windows = self.0.len();
        if num_windows == 0 {
            return DecodedSegmentations(Array3::zeros((0, 0, 0)));
        }

        let mut windows = self.0.into_iter();
        let first = powerset.hard_decode(&windows.next().unwrap());
        let mut stacked = Array3::<f32>::zeros((num_windows, first.nrows(), first.ncols()));
        stacked.slice_mut(s![0, .., ..]).assign(&first);

        for (window_idx, window) in windows.enumerate() {
            let decoded = powerset.hard_decode(&window);
            stacked
                .slice_mut(s![window_idx + 1, .., ..])
                .assign(&decoded);
        }

        DecodedSegmentations(stacked)
    }
}

impl DecodedSegmentations {
    fn nchunks(&self) -> usize {
        self.0.shape()[0]
    }

    fn num_speakers(&self) -> usize {
        if self.0.ndim() < 3 {
            return 0;
        }
        self.0.shape()[2]
    }

    fn speaker_count(&self, layout: &ChunkLayout) -> SpeakerCountTrack {
        let reconstructor = Reconstructor::new(self, &layout.start_frames, 0);
        reconstructor.speaker_count(layout.output_frames)
    }

    fn extract_embeddings(
        &self,
        audio: &[f32],
        emb_model: &mut EmbeddingModel,
        layout: &ChunkLayout,
        embedding_path: EmbeddingPath,
    ) -> Result<ChunkEmbeddings, PipelineError> {
        let num_chunks = self.0.shape()[0];
        let num_speakers = self.0.shape()[2];
        let mut embeddings = Array3::<f32>::from_elem((num_chunks, num_speakers, 256), f32::NAN);

        match embedding_path {
            EmbeddingPath::MultiMask => {
                self.extract_multi_mask_embeddings(audio, emb_model, layout, &mut embeddings)?
            }
            EmbeddingPath::Split => {
                self.extract_split_embeddings(audio, emb_model, layout, &mut embeddings)?
            }
            EmbeddingPath::Masked => {
                if emb_model.prefers_chunk_embedding_path() {
                    for chunk_idx in 0..num_chunks {
                        let chunk_audio = layout.chunk_audio(audio, chunk_idx);
                        let chunk_segmentations = self.0.slice(s![chunk_idx, .., ..]);
                        let clean_masks = clean_masks(&chunk_segmentations);
                        let chunk_embeddings = emb_model.embed_chunk_speakers(
                            chunk_audio,
                            chunk_segmentations,
                            &clean_masks,
                        )?;
                        embeddings
                            .slice_mut(s![chunk_idx, .., ..])
                            .assign(&chunk_embeddings);
                    }
                } else {
                    self.extract_masked_embeddings(audio, emb_model, layout, &mut embeddings)?;
                }
            }
        }

        Ok(ChunkEmbeddings(embeddings))
    }

    fn extract_masked_embeddings(
        &self,
        audio: &[f32],
        emb_model: &mut EmbeddingModel,
        layout: &ChunkLayout,
        embeddings: &mut Array3<f32>,
    ) -> Result<(), PipelineError> {
        let mut pending = Vec::with_capacity(emb_model.primary_batch_size());

        for chunk_idx in 0..self.0.shape()[0] {
            let chunk_audio = layout.chunk_audio(audio, chunk_idx);
            let chunk_segmentations = self.0.slice(s![chunk_idx, .., ..]);
            let clean_masks = clean_masks(&chunk_segmentations);

            for speaker_idx in 0..self.0.shape()[2] {
                let mask = chunk_segmentations.column(speaker_idx);
                let activity: f32 = mask.iter().sum();
                if activity < MIN_SPEAKER_ACTIVITY {
                    continue;
                }

                pending.push(PendingEmbedding {
                    chunk_idx,
                    speaker_idx,
                    audio: chunk_audio,
                    mask: mask.to_vec(),
                    clean_mask: clean_masks.column(speaker_idx).to_vec(),
                });
                if pending.len() == emb_model.primary_batch_size() {
                    flush_embedding_batch(emb_model, &pending, embeddings)?;
                    pending.clear();
                }
            }
        }

        while !pending.is_empty() {
            let batch_len = emb_model.best_batch_len(pending.len());
            flush_embedding_batch(emb_model, &pending[..batch_len], embeddings)?;
            pending.drain(..batch_len);
        }

        Ok(())
    }

    fn extract_split_embeddings(
        &self,
        audio: &[f32],
        emb_model: &mut EmbeddingModel,
        layout: &ChunkLayout,
        embeddings: &mut Array3<f32>,
    ) -> Result<(), PipelineError> {
        let batch_size = emb_model.split_primary_batch_size();
        let num_speakers = self.0.shape()[2];
        let min_num_samples = emb_model.min_num_samples();

        let mut pending: Vec<PendingSplitEmbedding> = Vec::with_capacity(batch_size);
        let mut fbanks: Vec<Array2<f32>> = Vec::new();
        let mut tail_batches = 0usize;
        let mut active_items = 0usize;

        for chunk_idx in 0..self.0.shape()[0] {
            let chunk_audio = layout.chunk_audio(audio, chunk_idx);
            let fbank = emb_model.compute_chunk_fbank(chunk_audio)?;
            fbanks.push(fbank);
            let mut current_fbank_idx = fbanks.len() - 1;

            let chunk_segmentations = self.0.slice(s![chunk_idx, .., ..]);
            let clean_masks = clean_masks(&chunk_segmentations);

            for speaker_idx in 0..num_speakers {
                let Some(weights) = select_speaker_weights(
                    &chunk_segmentations,
                    &clean_masks,
                    speaker_idx,
                    chunk_audio.len(),
                    min_num_samples,
                ) else {
                    continue;
                };
                active_items += 1;
                pending.push(PendingSplitEmbedding {
                    chunk_idx,
                    speaker_idx,
                    fbank_idx: current_fbank_idx,
                    weights,
                });
                if pending.len() == batch_size {
                    flush_split_embedding_batch(emb_model, &pending, &fbanks, embeddings)?;
                    tail_batches += 1;
                    pending.clear();

                    if speaker_idx + 1 < num_speakers {
                        let kept_fbank = fbanks.swap_remove(current_fbank_idx);
                        fbanks.clear();
                        fbanks.push(kept_fbank);
                        current_fbank_idx = 0;
                    } else {
                        fbanks.clear();
                    }
                }
            }
        }

        if !pending.is_empty() {
            flush_split_embedding_batch(emb_model, &pending, &fbanks, embeddings)?;
            tail_batches += 1;
        }

        info!(
            batches = tail_batches,
            active_items,
            total_items = self.0.shape()[0] * num_speakers,
            "Split embeddings complete (fbank+tail streaming)"
        );

        Ok(())
    }

    fn extract_multi_mask_embeddings(
        &self,
        audio: &[f32],
        emb_model: &mut EmbeddingModel,
        layout: &ChunkLayout,
        embeddings: &mut Array3<f32>,
    ) -> Result<(), PipelineError> {
        let batch_size = emb_model.multi_mask_batch_size();
        let num_speakers = self.0.shape()[2];
        let num_chunks = self.0.shape()[0];
        let min_num_samples = emb_model.min_num_samples();

        let mut fbank_buffer: Vec<Array2<f32>> = Vec::with_capacity(batch_size);
        let mut masks_buffer: Vec<Vec<f32>> = Vec::with_capacity(batch_size * num_speakers);
        let mut chunk_indices: Vec<usize> = Vec::with_capacity(batch_size);
        let mut batches = 0usize;

        for chunk_idx in 0..num_chunks {
            let chunk_audio = layout.chunk_audio(audio, chunk_idx);
            let fbank = emb_model.compute_chunk_fbank(chunk_audio)?;
            fbank_buffer.push(fbank);
            chunk_indices.push(chunk_idx);

            let chunk_segmentations = self.0.slice(s![chunk_idx, .., ..]);
            let clean_masks_arr = clean_masks(&chunk_segmentations);

            for speaker_idx in 0..num_speakers {
                let Some(weights) = select_speaker_weights(
                    &chunk_segmentations,
                    &clean_masks_arr,
                    speaker_idx,
                    chunk_audio.len(),
                    min_num_samples,
                ) else {
                    masks_buffer.push(vec![0.0; 589]);
                    continue;
                };
                masks_buffer.push(weights);
            }

            if fbank_buffer.len() == batch_size {
                flush_multi_mask_batch(
                    emb_model,
                    &fbank_buffer,
                    &masks_buffer,
                    &chunk_indices,
                    num_speakers,
                    embeddings,
                )?;
                batches += 1;
                fbank_buffer.clear();
                masks_buffer.clear();
                chunk_indices.clear();
            }
        }

        if !fbank_buffer.is_empty() {
            flush_multi_mask_batch(
                emb_model,
                &fbank_buffer,
                &masks_buffer,
                &chunk_indices,
                num_speakers,
                embeddings,
            )?;
            batches += 1;
        }

        info!(
            batches,
            total_chunks = num_chunks,
            "Multi-mask embeddings complete"
        );

        Ok(())
    }
}

#[cfg(test)]
#[allow(dead_code)]
fn decode_windows(raw_windows: Vec<Array2<f32>>, powerset: &PowersetMapping) -> Array3<f32> {
    RawSegmentationWindows(raw_windows).decode(powerset).0
}

#[cfg(test)]
fn extract_embeddings(
    seg_model: &SegmentationModel,
    emb_model: &mut EmbeddingModel,
    audio: &[f32],
    segmentations: &Array3<f32>,
) -> Result<Array3<f32>, PipelineError> {
    let decoded_segmentations = DecodedSegmentations(segmentations.clone());
    let layout = ChunkLayout::new(
        seg_model.step_seconds(),
        seg_model.step_samples(),
        seg_model.window_samples(),
        decoded_segmentations.nchunks(),
    );
    let embedding_path = if emb_model.prefers_multi_mask_path()
        && emb_model.multi_mask_batch_size() > 0
    {
        EmbeddingPath::MultiMask
    } else if emb_model.prefers_chunk_embedding_path() && emb_model.split_primary_batch_size() > 0 {
        EmbeddingPath::Split
    } else {
        EmbeddingPath::Masked
    };
    decoded_segmentations
        .extract_embeddings(audio, emb_model, &layout, embedding_path)
        .map(|chunk_embeddings| chunk_embeddings.0)
}

#[cfg(test)]
fn filter_embeddings(
    segmentations: &Array3<f32>,
    embeddings: &Array3<f32>,
) -> (Array2<f32>, Vec<usize>, Vec<usize>) {
    let num_frames = segmentations.shape()[1] as f32;
    let mut filtered = Vec::new();
    let mut chunk_indices = Vec::new();
    let mut speaker_indices = Vec::new();

    for chunk_idx in 0..segmentations.shape()[0] {
        let single_active: Vec<bool> = segmentations
            .slice(s![chunk_idx, .., ..])
            .rows()
            .into_iter()
            .map(|row| (row.iter().copied().sum::<f32>() - 1.0).abs() < 1e-6)
            .collect();
        for speaker_idx in 0..segmentations.shape()[2] {
            let clean_frames = segmentations
                .slice(s![chunk_idx, .., speaker_idx])
                .iter()
                .zip(single_active.iter())
                .filter_map(|(value, is_single_active)| is_single_active.then_some(*value))
                .sum::<f32>();
            let embedding = embeddings.slice(s![chunk_idx, speaker_idx, ..]);
            let valid_embedding = embedding.iter().all(|value| value.is_finite());
            if valid_embedding && clean_frames >= 0.2 * num_frames {
                filtered.extend(embedding.iter());
                chunk_indices.push(chunk_idx);
                speaker_indices.push(speaker_idx);
            }
        }
    }

    let filtered_embeddings =
        Array2::from_shape_vec((chunk_indices.len(), embeddings.shape()[2]), filtered).unwrap();
    (filtered_embeddings, chunk_indices, speaker_indices)
}

fn flush_embedding_batch(
    emb_model: &mut EmbeddingModel,
    pending: &[PendingEmbedding<'_>],
    embeddings: &mut Array3<f32>,
) -> Result<(), PipelineError> {
    let batch_inputs: Vec<_> = pending
        .iter()
        .map(|item| MaskedEmbeddingInput {
            audio: item.audio,
            mask: &item.mask,
            clean_mask: Some(&item.clean_mask),
        })
        .collect();
    let batch_embeddings = emb_model.embed_batch(&batch_inputs)?;

    for (batch_idx, item) in pending.iter().enumerate() {
        embeddings
            .slice_mut(s![item.chunk_idx, item.speaker_idx, ..])
            .assign(&batch_embeddings.row(batch_idx));
    }

    Ok(())
}

fn flush_split_embedding_batch(
    emb_model: &mut EmbeddingModel,
    pending: &[PendingSplitEmbedding],
    fbanks: &[Array2<f32>],
    embeddings: &mut Array3<f32>,
) -> Result<(), PipelineError> {
    let batch_inputs: Vec<_> = pending
        .iter()
        .map(|item| SplitTailInput {
            fbank: &fbanks[item.fbank_idx],
            weights: &item.weights,
        })
        .collect();
    let batch_embeddings = emb_model.embed_tail_batch_inputs(&batch_inputs)?;

    for (batch_idx, item) in pending.iter().enumerate() {
        embeddings
            .slice_mut(s![item.chunk_idx, item.speaker_idx, ..])
            .assign(&batch_embeddings.row(batch_idx));
    }

    Ok(())
}

fn flush_multi_mask_batch(
    emb_model: &mut EmbeddingModel,
    fbanks: &[Array2<f32>],
    masks: &[Vec<f32>],
    chunk_indices: &[usize],
    num_speakers: usize,
    embeddings: &mut Array3<f32>,
) -> Result<(), PipelineError> {
    let fbank_refs: Vec<_> = fbanks.iter().collect();
    let mask_refs: Vec<_> = masks.iter().map(|m| m.as_slice()).collect();
    let batch_embeddings = emb_model.embed_multi_mask_batch(&fbank_refs, &mask_refs)?;

    for (fbank_idx, &chunk_idx) in chunk_indices.iter().enumerate() {
        for speaker_idx in 0..num_speakers {
            let mask_idx = fbank_idx * num_speakers + speaker_idx;
            let is_active = masks[mask_idx].iter().any(|&v| v > 0.0);
            if !is_active {
                continue;
            }
            embeddings
                .slice_mut(s![chunk_idx, speaker_idx, ..])
                .assign(&batch_embeddings.row(mask_idx));
        }
    }

    Ok(())
}

impl ChunkEmbeddings {
    fn training_set(&self, segmentations: &DecodedSegmentations) -> TrainingEmbeddings {
        let num_frames = segmentations.0.shape()[1] as f32;
        let mut filtered = Vec::new();
        let mut chunk_indices = Vec::new();

        for chunk_idx in 0..segmentations.0.shape()[0] {
            let single_active: Vec<bool> = segmentations
                .0
                .slice(s![chunk_idx, .., ..])
                .rows()
                .into_iter()
                .map(|row| (row.iter().copied().sum::<f32>() - 1.0).abs() < 1e-6)
                .collect();
            for speaker_idx in 0..segmentations.0.shape()[2] {
                let clean_frames = segmentations
                    .0
                    .slice(s![chunk_idx, .., speaker_idx])
                    .iter()
                    .zip(single_active.iter())
                    .filter_map(|(value, is_single_active)| is_single_active.then_some(*value))
                    .sum::<f32>();
                let embedding = self.0.slice(s![chunk_idx, speaker_idx, ..]);
                let valid_embedding = embedding.iter().all(|value| value.is_finite());
                if valid_embedding && clean_frames >= 0.2 * num_frames {
                    filtered.extend(embedding.iter());
                    chunk_indices.push(chunk_idx);
                }
            }
        }

        let row_count = chunk_indices.len();
        let filtered_embeddings =
            Array2::from_shape_vec((row_count, self.0.shape()[2]), filtered).unwrap();
        TrainingEmbeddings(filtered_embeddings)
    }
}

impl TrainingEmbeddings {
    fn cluster(
        &self,
        segmentations: &DecodedSegmentations,
        embeddings: &ChunkEmbeddings,
        plda: &PldaTransform,
        config: &PipelineConfig,
    ) -> ChunkSpeakerClusters {
        if self.0.nrows() < 2 {
            let mut clusters =
                Array2::<i32>::zeros((segmentations.0.shape()[0], segmentations.0.shape()[2]));
            mark_inactive_speakers(&segmentations.0, &mut clusters);
            return ChunkSpeakerClusters(clusters);
        }

        let ahc_labels = cluster_ahc(&self.0.view(), config.ahc);
        debug!(
            rows = self.0.nrows(),
            cols = self.0.ncols(),
            "train_embeddings shape"
        );
        {
            let unique: std::collections::BTreeSet<_> = ahc_labels.iter().copied().collect();
            debug!(num_clusters = unique.len(), "AHC pre-clustering");
            for &cluster in &unique {
                let count = ahc_labels.iter().filter(|&&value| value == cluster).count();
                debug!(cluster, count, "AHC cluster size");
            }
        }

        let plda_features = plda.transform(&self.0.view(), 128);
        let phi = plda.phi();
        let (gamma, pi): (Array2<f32>, ndarray::Array1<f32>) = cluster_vbx(
            &ahc_labels,
            &plda_features.view(),
            &phi.slice(s![..128]),
            &config.vbx,
        );

        debug!(?pi, "VBx speaker priors");

        let mut kept_speakers: Vec<usize> = pi
            .iter()
            .enumerate()
            .filter_map(|(speaker_idx, weight)| {
                (*weight > config.speaker_keep_threshold as f32).then_some(speaker_idx)
            })
            .collect();
        if kept_speakers.is_empty() && !pi.is_empty() {
            let best_speaker = pi
                .iter()
                .enumerate()
                .max_by(|left, right| left.1.partial_cmp(right.1).unwrap())
                .map(|(speaker_idx, _)| speaker_idx)
                .unwrap();
            kept_speakers.push(best_speaker);
        }

        debug!(?kept_speakers, "VBx kept speakers");
        let centroids = weighted_centroids(&self.0, &gamma, &kept_speakers);
        for cluster_idx in 0..centroids.nrows() {
            let norm: f32 = centroids
                .row(cluster_idx)
                .dot(&centroids.row(cluster_idx))
                .sqrt();
            debug!(cluster = cluster_idx, norm, "centroid");
        }

        let mut clusters = assign_chunk_embeddings(segmentations, embeddings, &centroids);
        mark_inactive_speakers(&segmentations.0, &mut clusters);
        debug!(
            rows = clusters.nrows(),
            cols = clusters.ncols(),
            "hard_clusters shape"
        );

        ChunkSpeakerClusters(clusters)
    }
}

impl DiscreteDiarization {
    pub fn to_segments(
        &self,
        frame_step_seconds: f64,
        frame_duration_seconds: f64,
    ) -> Vec<crate::segment::Segment> {
        to_segments(&self.0, frame_step_seconds, frame_duration_seconds)
    }
}

fn weighted_centroids(
    train_embeddings: &Array2<f32>,
    gamma: &Array2<f32>,
    kept_speakers: &[usize],
) -> Array2<f32> {
    let mut centroids = Array2::<f32>::zeros((kept_speakers.len(), train_embeddings.ncols()));
    for (out_idx, &speaker_idx) in kept_speakers.iter().enumerate() {
        let weights = gamma.column(speaker_idx);
        let weight_sum = weights.sum().max(1e-8);
        for (row_idx, weight) in weights.iter().enumerate() {
            centroids
                .row_mut(out_idx)
                .scaled_add(*weight / weight_sum, &train_embeddings.row(row_idx));
        }
    }
    centroids
}

fn assign_chunk_embeddings(
    segmentations: &DecodedSegmentations,
    embeddings: &ChunkEmbeddings,
    centroids: &Array2<f32>,
) -> Array2<i32> {
    let num_chunks = embeddings.0.shape()[0];
    let num_speakers = embeddings.0.shape()[1];
    let num_clusters = centroids.nrows();
    let mut labels = Array2::<i32>::from_elem((num_chunks, num_speakers), -2);

    for chunk_idx in 0..num_chunks {
        // compute similarity scores for all active speakers against all centroids
        let mut active_local = Vec::new();
        let mut scores = Array2::<f32>::from_elem((num_speakers, num_clusters), f32::NEG_INFINITY);
        for speaker_idx in 0..num_speakers {
            let is_active = segmentations.0.slice(s![chunk_idx, .., speaker_idx]).sum() > 0.0;
            if !is_active {
                continue;
            }

            active_local.push(speaker_idx);
            let embedding = embeddings.0.slice(s![chunk_idx, speaker_idx, ..]);
            if embedding.iter().any(|value| !value.is_finite()) {
                continue;
            }

            for cluster_idx in 0..num_clusters {
                scores[[speaker_idx, cluster_idx]] =
                    1.0 + cosine_similarity(&embedding, &centroids.row(cluster_idx));
            }
        }

        // mask inactive/invalid speakers to min - 1 instead of NEG_INFINITY,
        // matching pyannote's constrained_argmax masking behavior
        let finite_min = scores
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .fold(f32::INFINITY, f32::min);
        if finite_min.is_finite() {
            let mask_value = finite_min - 1.0;
            scores.mapv_inplace(|v| if v.is_finite() { v } else { mask_value });
        }

        let assignments = best_assignment(&scores, &active_local, num_clusters);
        if tracing::enabled!(tracing::Level::TRACE) {
            trace!(
                chunk = chunk_idx,
                ?active_local,
                ?assignments,
                "chunk assignment"
            );
            for speaker_idx in 0..num_speakers {
                let row: Vec<f32> = scores.row(speaker_idx).to_vec();
                trace!(chunk = chunk_idx, speaker = speaker_idx, ?row, "scores");
            }
        }
        for (speaker_idx, cluster_idx) in assignments {
            labels[[chunk_idx, speaker_idx]] = cluster_idx as i32;
        }
    }

    labels
}

#[cfg(test)]
fn assign_embeddings(
    segmentations: &Array3<f32>,
    embeddings: &Array3<f32>,
    centroids: &Array2<f32>,
) -> Array2<i32> {
    assign_chunk_embeddings(
        &DecodedSegmentations(segmentations.clone()),
        &ChunkEmbeddings(embeddings.clone()),
        centroids,
    )
}

fn best_assignment(
    scores: &Array2<f32>,
    active_local: &[usize],
    num_clusters: usize,
) -> Vec<(usize, usize)> {
    let target = active_local.len().min(num_clusters);
    let mut search = AssignmentSearch::new(scores, active_local, target, num_clusters);
    search.run(0, 0.0);
    search.best
}

struct AssignmentSearch<'a> {
    scores: &'a Array2<f32>,
    active_local: &'a [usize],
    target: usize,
    used_clusters: Vec<bool>,
    current: Vec<(usize, usize)>,
    best_score: f32,
    best: Vec<(usize, usize)>,
}

impl<'a> AssignmentSearch<'a> {
    fn new(
        scores: &'a Array2<f32>,
        active_local: &'a [usize],
        target: usize,
        num_clusters: usize,
    ) -> Self {
        Self {
            scores,
            active_local,
            target,
            used_clusters: vec![false; num_clusters],
            current: Vec::new(),
            best_score: f32::NEG_INFINITY,
            best: Vec::new(),
        }
    }

    fn run(&mut self, position: usize, current_score: f32) {
        if self.current.len() == self.target {
            if current_score > self.best_score {
                self.best_score = current_score;
                self.best = self.current.clone();
            }
            return;
        }

        if position == self.active_local.len() {
            return;
        }

        let remaining_local = self.active_local.len() - position;
        let remaining_needed = self.target - self.current.len();
        if remaining_local > remaining_needed {
            self.run(position + 1, current_score);
        }

        let speaker_idx = self.active_local[position];
        for cluster_idx in 0..self.used_clusters.len() {
            if self.used_clusters[cluster_idx] {
                continue;
            }

            self.used_clusters[cluster_idx] = true;
            self.current.push((speaker_idx, cluster_idx));
            self.run(
                position + 1,
                current_score + self.scores[[speaker_idx, cluster_idx]],
            );
            self.current.pop();
            self.used_clusters[cluster_idx] = false;
        }
    }
}

fn mark_inactive_speakers(segmentations: &Array3<f32>, hard_clusters: &mut Array2<i32>) {
    for chunk_idx in 0..segmentations.shape()[0] {
        for speaker_idx in 0..segmentations.shape()[2] {
            let active = segmentations.slice(s![chunk_idx, .., speaker_idx]).sum() > 0.0;
            if !active {
                hard_clusters[[chunk_idx, speaker_idx]] = -2;
            }
        }
    }
}

fn clean_masks(segmentations: &ArrayView2<f32>) -> Array2<f32> {
    let single_active: Vec<bool> = segmentations
        .rows()
        .into_iter()
        .map(|row| row.iter().copied().sum::<f32>() < 2.0)
        .collect();
    let mut clean = Array2::<f32>::zeros(segmentations.raw_dim());
    for (frame_idx, is_single_active) in single_active.iter().enumerate() {
        if !*is_single_active {
            continue;
        }

        clean
            .slice_mut(s![frame_idx, ..])
            .assign(&segmentations.slice(s![frame_idx, ..]));
    }
    clean
}

#[cfg(test)]
#[allow(dead_code)]
fn chunk_audio<'a>(audio: &'a [f32], seg_model: &SegmentationModel, chunk_idx: usize) -> &'a [f32] {
    chunk_audio_raw(
        audio,
        seg_model.step_samples(),
        seg_model.window_samples(),
        chunk_idx,
    )
}

fn chunk_audio_raw(
    audio: &[f32],
    step_samples: usize,
    window_samples: usize,
    chunk_idx: usize,
) -> &[f32] {
    let start = chunk_idx * step_samples;
    let end = (start + window_samples).min(audio.len());
    if start < audio.len() {
        &audio[start..end]
    } else {
        &[]
    }
}

fn chunk_start_frames(num_chunks: usize, step_seconds: f64) -> Vec<usize> {
    (0..num_chunks)
        .map(|chunk_idx| {
            closest_frame(chunk_idx as f64 * step_seconds + 0.5 * FRAME_DURATION_SECONDS)
        })
        .collect()
}

fn total_output_frames(num_chunks: usize, step_seconds: f64) -> usize {
    if num_chunks == 0 {
        return 0;
    }

    closest_frame(
        SEGMENTATION_WINDOW_SECONDS
            + (num_chunks - 1) as f64 * step_seconds
            + 0.5 * FRAME_DURATION_SECONDS,
    ) + 1
}

fn closest_frame(timestamp: f64) -> usize {
    ((timestamp - 0.5 * FRAME_DURATION_SECONDS) / FRAME_STEP_SECONDS).round() as usize
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, Array2, Array3, array};
    use ndarray_npy::ReadNpyExt;
    use std::fs::File;
    use std::path::{Path, PathBuf};

    use super::*;
    #[cfg(feature = "coreml")]
    use crate::inference::ExecutionMode;

    fn fixture_path(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures")
            .join(name)
    }

    fn load_wav_samples(path: &Path) -> (Vec<f32>, u32) {
        let data = std::fs::read(path).unwrap();
        let sample_rate = u32::from_le_bytes(data[24..28].try_into().unwrap());
        let bits_per_sample = u16::from_le_bytes(data[34..36].try_into().unwrap());
        assert_eq!(bits_per_sample, 16);

        let mut pos = 12;
        while pos + 8 < data.len() {
            let chunk_id = &data[pos..pos + 4];
            let chunk_size =
                u32::from_le_bytes(data[pos + 4..pos + 8].try_into().unwrap()) as usize;
            if chunk_id == b"data" {
                let samples = data[pos + 8..pos + 8 + chunk_size]
                    .chunks_exact(2)
                    .map(|bytes| i16::from_le_bytes([bytes[0], bytes[1]]) as f32 / 32768.0)
                    .collect();
                return (samples, sample_rate);
            }
            pos += 8 + chunk_size;
        }

        panic!("no data chunk found in WAV");
    }

    #[test]
    fn chunk_start_frames_match_pyannote_rounding() {
        assert_eq!(
            chunk_start_frames(4, SEGMENTATION_STEP_SECONDS),
            vec![0, 59, 119, 178]
        );
    }

    #[test]
    fn total_output_frames_match_pyannote_aggregate_extent() {
        assert_eq!(total_output_frames(4, SEGMENTATION_STEP_SECONDS), 771);
    }

    #[test]
    fn best_assignment_handles_more_speakers_than_clusters() {
        let scores = array![[0.9, 0.1], [0.8, 0.2], [0.1, 0.95]];
        let assignment = best_assignment(&scores, &[0, 1, 2], 2);
        assert_eq!(assignment.len(), 2);
        assert!(assignment.contains(&(0, 0)) || assignment.contains(&(1, 0)));
        assert!(assignment.contains(&(2, 1)));
    }

    #[test]
    fn filter_embeddings_matches_python_fixture() {
        let segmentations: Array3<f32> =
            Array3::read_npy(File::open(fixture_path("pipeline_segmentation_data.npy")).unwrap())
                .unwrap();
        let embeddings: Array3<f32> =
            Array3::read_npy(File::open(fixture_path("pipeline_embeddings_data.npy")).unwrap())
                .unwrap();
        let expected_train_embeddings: Array2<f32> =
            Array2::read_npy(File::open(fixture_path("pipeline_train_embeddings.npy")).unwrap())
                .unwrap();
        let expected_chunk_idx: Array1<i64> =
            Array1::read_npy(File::open(fixture_path("pipeline_train_chunk_idx.npy")).unwrap())
                .unwrap();
        let expected_speaker_idx: Array1<i64> =
            Array1::read_npy(File::open(fixture_path("pipeline_train_speaker_idx.npy")).unwrap())
                .unwrap();

        let (train_embeddings, chunk_idx, speaker_idx) =
            filter_embeddings(&segmentations, &embeddings);

        assert_eq!(chunk_idx.len(), expected_chunk_idx.len());
        assert_eq!(speaker_idx.len(), expected_speaker_idx.len());
        for (lhs, rhs) in chunk_idx.iter().zip(expected_chunk_idx.iter()) {
            assert_eq!(*lhs as i64, *rhs);
        }
        for (lhs, rhs) in speaker_idx.iter().zip(expected_speaker_idx.iter()) {
            assert_eq!(*lhs as i64, *rhs);
        }
        for (lhs, rhs) in train_embeddings
            .iter()
            .zip(expected_train_embeddings.iter())
        {
            approx::assert_abs_diff_eq!(*lhs, *rhs, epsilon = 1e-5);
        }
    }

    #[test]
    fn assign_embeddings_matches_python_fixture() {
        let segmentations: Array3<f32> =
            Array3::read_npy(File::open(fixture_path("pipeline_segmentation_data.npy")).unwrap())
                .unwrap();
        let embeddings: Array3<f32> =
            Array3::read_npy(File::open(fixture_path("pipeline_embeddings_data.npy")).unwrap())
                .unwrap();
        let train_embeddings: Array2<f32> =
            Array2::read_npy(File::open(fixture_path("pipeline_train_embeddings.npy")).unwrap())
                .unwrap();
        let gamma: Array2<f64> =
            Array2::read_npy(File::open(fixture_path("pipeline_vbx_gamma.npy")).unwrap()).unwrap();
        let pi: Array1<f64> =
            Array1::read_npy(File::open(fixture_path("pipeline_vbx_pi.npy")).unwrap()).unwrap();
        let expected: Array2<i8> =
            Array2::read_npy(File::open(fixture_path("pipeline_hard_clusters.npy")).unwrap())
                .unwrap();

        let kept_speakers: Vec<usize> = pi
            .iter()
            .enumerate()
            .filter_map(|(idx, weight)| (*weight > 1e-7).then_some(idx))
            .collect();
        let centroids = weighted_centroids(
            &train_embeddings,
            &gamma.mapv(|value| value as f32),
            &kept_speakers,
        );
        let mut hard_clusters = assign_embeddings(&segmentations, &embeddings, &centroids);
        mark_inactive_speakers(&segmentations, &mut hard_clusters);

        assert_eq!(hard_clusters.dim(), expected.dim());
        for (lhs, rhs) in hard_clusters.iter().zip(expected.iter()) {
            assert_eq!(*lhs as i8, *rhs);
        }
    }

    #[test]
    fn extract_embeddings_matches_python_fixture() {
        let models_dir = fixture_path("models");
        let seg_model = SegmentationModel::new(
            models_dir.join("segmentation-3.0.onnx").to_str().unwrap(),
            SEGMENTATION_STEP_SECONDS as f32,
        )
        .unwrap();
        let mut emb_model = EmbeddingModel::new(
            models_dir
                .join("wespeaker-voxceleb-resnet34.onnx")
                .to_str()
                .unwrap(),
        )
        .unwrap();
        let segmentations: Array3<f32> =
            Array3::read_npy(File::open(fixture_path("pipeline_segmentation_data.npy")).unwrap())
                .unwrap();
        let expected: Array3<f32> =
            Array3::read_npy(File::open(fixture_path("pipeline_embeddings_data.npy")).unwrap())
                .unwrap();
        let (audio, sample_rate) = load_wav_samples(&fixture_path("test.wav"));
        assert_eq!(sample_rate, 16_000);

        let embeddings =
            extract_embeddings(&seg_model, &mut emb_model, &audio, &segmentations).unwrap();

        for chunk_idx in 0..embeddings.shape()[0] {
            for speaker_idx in 0..embeddings.shape()[1] {
                for dim_idx in 0..embeddings.shape()[2] {
                    let lhs = embeddings[[chunk_idx, speaker_idx, dim_idx]];
                    let rhs = expected[[chunk_idx, speaker_idx, dim_idx]];
                    if (lhs - rhs).abs() > 5e-3 || lhs.is_nan() != rhs.is_nan() {
                        panic!(
                            "chunk={chunk_idx} speaker={speaker_idx} dim={dim_idx} left={lhs} right={rhs}"
                        );
                    }
                }
            }
        }
    }

    #[cfg(feature = "coreml")]
    #[test]
    fn fast_apple_segmentation_matches_python_fixture() {
        let models_dir = fixture_path("models");
        let mut seg_model = SegmentationModel::with_mode(
            models_dir.join("segmentation-3.0.onnx").to_str().unwrap(),
            SEGMENTATION_STEP_SECONDS as f32,
            ExecutionMode::CoreMl,
        )
        .unwrap();
        let expected: Array3<f32> =
            Array3::read_npy(File::open(fixture_path("pipeline_segmentation_data.npy")).unwrap())
                .unwrap();
        let powerset = PowersetMapping::new(3, 2);
        let (audio, sample_rate) = load_wav_samples(&fixture_path("test.wav"));
        assert_eq!(sample_rate, 16_000);

        let raw_windows = seg_model.run(&audio).unwrap();
        let segmentations = decode_windows(raw_windows, &powerset);

        for chunk_idx in 0..segmentations.shape()[0] {
            for frame_idx in 0..segmentations.shape()[1] {
                for speaker_idx in 0..segmentations.shape()[2] {
                    let lhs = segmentations[[chunk_idx, frame_idx, speaker_idx]];
                    let rhs = expected[[chunk_idx, frame_idx, speaker_idx]];
                    if lhs != rhs {
                        panic!(
                            "chunk={chunk_idx} frame={frame_idx} speaker={speaker_idx} left={lhs} right={rhs}"
                        );
                    }
                }
            }
        }
    }

    #[cfg(feature = "coreml")]
    #[test]
    fn fast_apple_embeddings_match_python_fixture() {
        let models_dir = fixture_path("models");
        let seg_model = SegmentationModel::new(
            models_dir.join("segmentation-3.0.onnx").to_str().unwrap(),
            SEGMENTATION_STEP_SECONDS as f32,
        )
        .unwrap();
        let mut emb_model = EmbeddingModel::with_mode(
            models_dir
                .join("wespeaker-voxceleb-resnet34.onnx")
                .to_str()
                .unwrap(),
            ExecutionMode::CoreMl,
        )
        .unwrap();
        let segmentations: Array3<f32> =
            Array3::read_npy(File::open(fixture_path("pipeline_segmentation_data.npy")).unwrap())
                .unwrap();
        let expected: Array3<f32> =
            Array3::read_npy(File::open(fixture_path("pipeline_embeddings_data.npy")).unwrap())
                .unwrap();
        let (audio, sample_rate) = load_wav_samples(&fixture_path("test.wav"));
        assert_eq!(sample_rate, 16_000);

        let embeddings =
            extract_embeddings(&seg_model, &mut emb_model, &audio, &segmentations).unwrap();

        for chunk_idx in 0..embeddings.shape()[0] {
            for speaker_idx in 0..embeddings.shape()[1] {
                for dim_idx in 0..embeddings.shape()[2] {
                    let lhs = embeddings[[chunk_idx, speaker_idx, dim_idx]];
                    let rhs = expected[[chunk_idx, speaker_idx, dim_idx]];
                    if (lhs - rhs).abs() > 5e-3 || lhs.is_nan() != rhs.is_nan() {
                        panic!(
                            "chunk={chunk_idx} speaker={speaker_idx} dim={dim_idx} left={lhs} right={rhs}"
                        );
                    }
                }
            }
        }
    }

    #[cfg(feature = "coreml")]
    #[test]
    fn fast_apple_split_primary_batch_matches_single_tail_path() {
        let models_dir = fixture_path("models");
        let seg_model = SegmentationModel::new(
            models_dir.join("segmentation-3.0.onnx").to_str().unwrap(),
            SEGMENTATION_STEP_SECONDS as f32,
        )
        .unwrap();
        let mut emb_model = EmbeddingModel::with_mode(
            models_dir
                .join("wespeaker-voxceleb-resnet34.onnx")
                .to_str()
                .unwrap(),
            ExecutionMode::CoreMl,
        )
        .unwrap();
        let segmentations: Array3<f32> =
            Array3::read_npy(File::open(fixture_path("pipeline_segmentation_data.npy")).unwrap())
                .unwrap();
        let (audio, sample_rate) = load_wav_samples(&fixture_path("test.wav"));
        assert_eq!(sample_rate, 16_000);

        let mut fbanks = Vec::new();
        let mut weights = Vec::new();
        let mut expected = Vec::new();

        'outer: for chunk_idx in 0..segmentations.shape()[0] {
            let chunk_audio = chunk_audio(&audio, &seg_model, chunk_idx);
            let chunk_segmentations = segmentations.slice(s![chunk_idx, .., ..]);
            let clean_masks = clean_masks(&chunk_segmentations);
            let fbank = emb_model.compute_chunk_fbank(chunk_audio).unwrap();

            for speaker_idx in 0..chunk_segmentations.ncols() {
                let mask = chunk_segmentations.column(speaker_idx).to_owned();
                let clean_mask = clean_masks.column(speaker_idx).to_owned();
                let used_mask = emb_model
                    .select_chunk_mask(
                        mask.as_slice().unwrap(),
                        Some(clean_mask.as_slice().unwrap()),
                        chunk_audio.len(),
                    )
                    .to_vec();
                expected.push(
                    emb_model
                        .embed_masked(
                            chunk_audio,
                            mask.as_slice().unwrap(),
                            Some(clean_mask.as_slice().unwrap()),
                        )
                        .unwrap(),
                );
                fbanks.push(fbank.clone());
                weights.push(used_mask);
                if fbanks.len() == emb_model.split_primary_batch_size() {
                    break 'outer;
                }
            }
        }

        assert_eq!(fbanks.len(), emb_model.split_primary_batch_size());
        let batch_inputs: Vec<_> = fbanks
            .iter()
            .zip(weights.iter())
            .map(|(fbank, weights)| SplitTailInput {
                fbank,
                weights: weights.as_slice(),
            })
            .collect();
        let batched = emb_model.embed_tail_batch_inputs(&batch_inputs).unwrap();

        for (row_idx, expected_row) in expected.iter().enumerate() {
            for dim_idx in 0..expected_row.len() {
                let lhs = batched[[row_idx, dim_idx]];
                let rhs = expected_row[dim_idx];
                if (lhs - rhs).abs() > 5e-3 || lhs.is_nan() != rhs.is_nan() {
                    panic!("row={row_idx} dim={dim_idx} left={lhs} right={rhs}");
                }
            }
        }
    }

    #[cfg(feature = "coreml")]
    #[test]
    fn fast_apple_single_embedding_matches_python_fixture() {
        let models_dir = fixture_path("models");
        let seg_model = SegmentationModel::new(
            models_dir.join("segmentation-3.0.onnx").to_str().unwrap(),
            SEGMENTATION_STEP_SECONDS as f32,
        )
        .unwrap();
        let mut emb_model = EmbeddingModel::with_mode(
            models_dir
                .join("wespeaker-voxceleb-resnet34.onnx")
                .to_str()
                .unwrap(),
            ExecutionMode::CoreMl,
        )
        .unwrap();
        let segmentations: Array3<f32> =
            Array3::read_npy(File::open(fixture_path("pipeline_segmentation_data.npy")).unwrap())
                .unwrap();
        let expected: Array3<f32> =
            Array3::read_npy(File::open(fixture_path("pipeline_embeddings_data.npy")).unwrap())
                .unwrap();
        let (audio, sample_rate) = load_wav_samples(&fixture_path("test.wav"));
        assert_eq!(sample_rate, 16_000);

        let chunk_idx = 0;
        let speaker_idx = 1;
        let chunk_segmentations = segmentations.slice(s![chunk_idx, .., ..]);
        let clean = clean_masks(&chunk_segmentations);
        let mask = chunk_segmentations.column(speaker_idx).to_vec();
        let clean_mask = clean.column(speaker_idx).to_vec();
        let embedding = emb_model
            .embed_masked(
                chunk_audio(&audio, &seg_model, chunk_idx),
                &mask,
                Some(&clean_mask),
            )
            .unwrap();

        for dim_idx in 0..embedding.len() {
            let lhs = embedding[dim_idx];
            let rhs = expected[[chunk_idx, speaker_idx, dim_idx]];
            if (lhs - rhs).abs() > 5e-4 || lhs.is_nan() != rhs.is_nan() {
                panic!("dim={dim_idx} left={lhs} right={rhs}");
            }
        }
    }
}
