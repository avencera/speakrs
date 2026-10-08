//! Native CPU embedding facade adapter

use super::{EMBEDDING_WIDTH, EmbeddingMeta, SplitTailInput, should_use_clean_mask};
use crate::inference::cpu::{
    assets::EmbeddingAssets,
    embedding::{CpuResNet34, ResNetWorkspace},
    fbank::{CpuFbank, FbankWorkspace},
    workers::CpuWorkers,
};
use crate::inference::native_model::{
    WeightsFile,
    fbank::{FBANK_FRAMES, FBANK_MEL_BINS as FBANK_FEATURES, FBANK_WINDOW_SAMPLES},
};

use crate::inference::{CpuError, InferenceError, ModelLoadError, TensorShapeError};
use ndarray::{Array1, Array2, ArrayView2};

const NUM_SPEAKERS: usize = 3;

/// Shared immutable model owners with private fbank and trunk scratch
pub(super) struct CpuEmbedding {
    model: CpuResNet34,
    fbank: CpuFbank,
    workers: CpuWorkers<EmbeddingWorker>,
}

struct EmbeddingWorker {
    model: ResNetWorkspace,
    fbank: FbankWorkspace,
    features: Array2<f32>,
}

impl EmbeddingWorker {
    fn new(model: &CpuResNet34, fbank: &CpuFbank) -> Self {
        Self {
            model: model.workspace(),
            fbank: fbank.workspace(),
            features: Array2::zeros((FBANK_FRAMES, FBANK_FEATURES)),
        }
    }
}

impl Clone for CpuEmbedding {
    fn clone(&self) -> Self {
        Self {
            model: self.model.clone(),
            fbank: self.fbank.clone(),
            workers: CpuWorkers::new(EmbeddingWorker::new(&self.model, &self.fbank)),
        }
    }
}

impl CpuEmbedding {
    /// Loads the resolved native weights
    pub(super) fn load(assets: &EmbeddingAssets) -> Result<Self, ModelLoadError> {
        let file = WeightsFile::open(assets.weights()).map_err(CpuError::from)?;
        let model = CpuResNet34::load(&file)?;
        let fbank = CpuFbank::new();
        Ok(Self {
            workers: CpuWorkers::new(EmbeddingWorker::new(&model, &fbank)),
            model,
            fbank,
        })
    }

    /// Primary single-window capacity
    pub(super) fn primary_batch_size(&self) -> usize {
        1
    }
    /// No fused split-primary session
    pub(super) fn split_primary_batch_size(&self) -> usize {
        0
    }
    /// Uses the native filterbank and tail path
    pub(super) fn prefers_chunk_embedding_path(&self) -> bool {
        true
    }
    /// Supports three masks per useful chunk
    pub(super) fn prefers_multi_mask_path(&self) -> bool {
        true
    }
    /// Maximum chunks per multi-mask facade batch
    pub(super) fn multi_mask_batch_size(&self) -> usize {
        8
    }
    /// Pools multiple speakers after one trunk call
    pub(super) fn has_batched_tail(&self) -> bool {
        true
    }

    /// Computes a masked embedding from one useful audio window
    pub(super) fn embed_single(
        &mut self,
        meta: &EmbeddingMeta,
        audio: &[f32],
        weights: &[f32],
    ) -> Result<Array1<f32>, InferenceError> {
        check_window(meta)?;
        let worker = self.workers.first();
        self.fbank
            .compute_into(audio, worker.features.view_mut(), &mut worker.fbank)?;
        let rows = self
            .model
            .forward(worker.features.view(), &[weights], &mut worker.model)?;
        Ok(rows.row(0).to_owned())
    }

    /// Computes all three masks per chunk in chunk-major, speaker-minor order
    pub(super) fn embed_multi_mask_batch(
        &mut self,
        _meta: &EmbeddingMeta,
        fbanks: &[&Array2<f32>],
        masks: &[&[f32]],
    ) -> Result<Array2<f32>, InferenceError> {
        self.check_counts(fbanks.len(), masks.len())?;
        // validate the entire request before the first model call
        for fbank in fbanks {
            check_fbank(fbank.view())?;
        }
        let jobs: Vec<_> = fbanks
            .iter()
            .zip(masks.as_chunks::<NUM_SPEAKERS>().0.iter())
            .collect();
        let rows = self.workers.map(
            &jobs,
            || EmbeddingWorker::new(&self.model, &self.fbank),
            |worker, (fbank, masks)| self.model.forward(fbank.view(), *masks, &mut worker.model),
        )?;
        Ok(join_rows(rows, masks.len()))
    }

    /// Computes one filterbank and one trunk pass per useful audio chunk
    pub(super) fn embed_multi_mask_audio_batch(
        &mut self,
        meta: &EmbeddingMeta,
        audios: &[&[f32]],
        masks: &[&[f32]],
    ) -> Result<Array2<f32>, InferenceError> {
        self.check_counts(audios.len(), masks.len())?;
        check_window(meta)?;
        let jobs: Vec<_> = audios
            .iter()
            .zip(masks.as_chunks::<NUM_SPEAKERS>().0.iter())
            .collect();
        let rows = self.workers.map(
            &jobs,
            || EmbeddingWorker::new(&self.model, &self.fbank),
            |worker, (audio, masks)| {
                self.fbank
                    .compute_into(audio, worker.features.view_mut(), &mut worker.fbank)?;
                self.model
                    .forward(worker.features.view(), *masks, &mut worker.model)
            },
        )?;
        Ok(join_rows(rows, masks.len()))
    }

    /// Runs one tail mask without normalization of the output vector
    pub(super) fn embed_tail_single(
        &mut self,
        _meta: &EmbeddingMeta,
        fbank: &Array2<f32>,
        weights: &[f32],
    ) -> Result<Array1<f32>, InferenceError> {
        let rows = self
            .model
            .forward(fbank.view(), &[weights], &mut self.workers.first().model)?;
        Ok(rows.row(0).to_owned())
    }

    /// Selects masks with the existing facade rule and runs the trunk once
    pub(super) fn embed_tail_batch(
        &mut self,
        meta: &EmbeddingMeta,
        fbank: &Array2<f32>,
        segmentations: &ArrayView2<'_, f32>,
        clean_masks: &Array2<f32>,
        num_samples: usize,
    ) -> Result<Array2<f32>, InferenceError> {
        if segmentations.dim() != clean_masks.dim() {
            return Err(TensorShapeError::ShapeMismatch {
                context: "CPU chunk clean masks",
                expected: segmentations.shape().to_vec(),
                actual: clean_masks.shape().to_vec(),
            }
            .into());
        }
        if segmentations.ncols() > NUM_SPEAKERS {
            return Err(InferenceError::BatchTooLarge {
                context: "CPU chunk speakers",
                rows: segmentations.ncols(),
                capacity: NUM_SPEAKERS,
            });
        }
        check_fbank(fbank.view())?;
        let selected: Vec<Vec<f32>> = (0..segmentations.ncols())
            .map(|speaker| {
                let mask = segmentations.column(speaker);
                let clean = clean_masks.column(speaker);
                let selected =
                    if should_use_clean_mask(&clean, mask.len(), num_samples, meta.min_num_samples)
                    {
                        clean
                    } else {
                        mask
                    };
                selected.iter().copied().collect()
            })
            .collect();
        let masks: Vec<&[f32]> = selected.iter().map(Vec::as_slice).collect();
        self.model
            .forward(fbank.view(), &masks, &mut self.workers.first().model)
    }

    /// Runs arbitrary split-tail inputs in their original order
    pub(super) fn embed_tail_batch_inputs(
        &mut self,
        _meta: &EmbeddingMeta,
        inputs: &[SplitTailInput<'_>],
    ) -> Result<Array2<f32>, InferenceError> {
        for input in inputs {
            check_fbank(input.fbank.view())?;
        }
        let rows = self.workers.map(
            inputs,
            || EmbeddingWorker::new(&self.model, &self.fbank),
            |worker, input| {
                self.model
                    .forward(input.fbank.view(), &[input.weights], &mut worker.model)
            },
        )?;
        Ok(join_rows(rows, inputs.len()))
    }

    /// Pads or truncates audio to the deployed filterbank window
    pub(super) fn compute_chunk_fbank(
        &mut self,
        meta: &EmbeddingMeta,
        audio: &[f32],
    ) -> Result<Array2<f32>, InferenceError> {
        check_window(meta)?;
        let worker = self.workers.first();
        self.fbank
            .compute_into(audio, worker.features.view_mut(), &mut worker.fbank)?;
        Ok(worker.features.clone())
    }

    /// Computes arbitrary useful filterbank batches without fake rows
    pub(super) fn compute_chunk_fbanks_batch(
        &mut self,
        meta: &EmbeddingMeta,
        audios: &[&[f32]],
    ) -> Result<Vec<Array2<f32>>, InferenceError> {
        check_window(meta)?;
        self.workers.map(
            audios,
            || EmbeddingWorker::new(&self.model, &self.fbank),
            |worker, audio| {
                self.fbank
                    .compute_into(audio, worker.features.view_mut(), &mut worker.fbank)?;
                Ok(worker.features.clone())
            },
        )
    }

    fn check_counts(&self, chunks: usize, masks: usize) -> Result<(), InferenceError> {
        if chunks > self.multi_mask_batch_size() {
            return Err(InferenceError::BatchTooLarge {
                context: "CPU multi-mask chunks",
                rows: chunks,
                capacity: self.multi_mask_batch_size(),
            });
        }
        if masks != chunks * NUM_SPEAKERS {
            return Err(InferenceError::MaskCountMismatch {
                fbanks: chunks,
                expected: chunks * NUM_SPEAKERS,
                actual: masks,
            });
        }
        Ok(())
    }
}

fn check_window(meta: &EmbeddingMeta) -> Result<(), InferenceError> {
    if meta.window_samples == FBANK_WINDOW_SAMPLES {
        return Ok(());
    }
    Err(TensorShapeError::LengthMismatch {
        context: "CPU embedding window",
        expected: FBANK_WINDOW_SAMPLES,
        actual: meta.window_samples,
    }
    .into())
}

fn join_rows(chunks: Vec<Array2<f32>>, row_count: usize) -> Array2<f32> {
    let mut output = Array2::zeros((row_count, EMBEDDING_WIDTH));
    let mut offset = 0;
    for rows in chunks {
        let end = offset + rows.nrows();
        output.slice_mut(ndarray::s![offset..end, ..]).assign(&rows);
        offset = end;
    }
    output
}

fn check_fbank(fbank: ArrayView2<'_, f32>) -> Result<(), InferenceError> {
    if fbank.nrows() <= FBANK_FRAMES && fbank.ncols() == FBANK_FEATURES {
        return Ok(());
    }
    Err(TensorShapeError::ShapeMismatch {
        context: "CPU tail filterbank",
        expected: vec![FBANK_FRAMES, FBANK_FEATURES],
        actual: fbank.shape().to_vec(),
    }
    .into())
}

#[cfg(test)]
mod tests;
