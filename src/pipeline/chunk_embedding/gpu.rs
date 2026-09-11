use std::sync::Arc;

use crossbeam_channel::{Receiver, Sender};

use crate::inference::coreml::{CachedInputShape, SharedCoreMlModel};

use super::prep::{ChunkJob, PrepScratch, audio_for};
use super::{EmbeddingModel, PipelineError, backend_error, invariant_error};

impl ChunkEmbeddingResources {
    pub(super) fn largest_session(&self) -> Result<&ChunkSessionDescriptor, PipelineError> {
        self.sessions
            .last()
            .ok_or_else(|| invariant_error("missing chunk embedding session"))
    }
}

pub(super) fn chunk_embedding_resources(
    emb_model: &mut EmbeddingModel,
) -> Result<Option<ChunkEmbeddingResources>, PipelineError> {
    let Some(bundle) = emb_model.prepare_chunk_resources()? else {
        return Ok(None);
    };

    let sessions = bundle
        .sessions
        .iter()
        .map(|session| ChunkSessionDescriptor {
            handle: ChunkSessionHandle {
                cached_fbank_shape: Arc::clone(&session.cached_fbank_shape),
                cached_masks_shape: Arc::clone(&session.cached_masks_shape),
                model: Arc::clone(&session.model),
            },
            num_windows: session.num_windows,
            fbank_frames: session.fbank_frames,
            num_masks: session.num_masks,
        })
        .collect();

    Ok(Some(ChunkEmbeddingResources {
        sessions,
        fbank_30s: bundle.fbank_30s,
        fbank_10s: bundle.fbank_10s,
    }))
}

pub(super) struct GpuWorker {
    pub(super) model: Arc<SharedCoreMlModel>,
    pub(super) fbank_shape: Arc<CachedInputShape>,
    pub(super) masks_shape: Arc<CachedInputShape>,
    pub(super) prep: super::ChunkPrep,
    pub(super) scratch: PrepScratch,
}

impl GpuWorker {
    fn next_prepared(
        &mut self,
        audios: &[&[f32]],
        prep_rx: &Receiver<PreparedChunk>,
        job_rx: &Receiver<ChunkJob>,
        decoded_done: &mut bool,
        total_prep_us: &mut u64,
    ) -> Result<Option<PreparedChunk>, PipelineError> {
        match prep_rx.try_recv() {
            Ok(prepared) => return Ok(Some(prepared)),
            Err(crossbeam_channel::TryRecvError::Disconnected) => return Ok(None),
            Err(crossbeam_channel::TryRecvError::Empty) => {}
        }

        if *decoded_done {
            return match prep_rx.recv() {
                Ok(prepared) => Ok(Some(prepared)),
                Err(_) => Ok(None),
            };
        }

        match job_rx.try_recv() {
            Ok(job) => self.prep_job(audios, job, total_prep_us).map(Some),
            Err(crossbeam_channel::TryRecvError::Empty) => crossbeam_channel::select! {
                recv(prep_rx) -> message => match message {
                    Ok(prepared) => Ok(Some(prepared)),
                    Err(_) => Ok(None),
                },
                recv(job_rx) -> message => match message {
                    Ok(job) => self.prep_job(audios, job, total_prep_us).map(Some),
                    Err(_) => {
                        *decoded_done = true;
                        match prep_rx.recv() {
                            Ok(prepared) => Ok(Some(prepared)),
                            Err(_) => Ok(None),
                        }
                    }
                },
            },
            Err(crossbeam_channel::TryRecvError::Disconnected) => {
                *decoded_done = true;
                match prep_rx.recv() {
                    Ok(prepared) => Ok(Some(prepared)),
                    Err(_) => Ok(None),
                }
            }
        }
    }

    fn prep_job(
        &mut self,
        audios: &[&[f32]],
        job: ChunkJob,
        total_prep_us: &mut u64,
    ) -> Result<PreparedChunk, PipelineError> {
        let audio = audio_for(audios, job.file_index)?;
        let prep_start = std::time::Instant::now();
        let prepared = self.prep.prep(job, audio, &mut self.scratch)?;
        *total_prep_us += prep_start.elapsed().as_micros() as u64;
        Ok(prepared)
    }

    fn predict(&self, prepared: &PreparedChunk) -> Result<(Vec<f32>, u64), PipelineError> {
        let predict_start = std::time::Instant::now();
        let tensor = self
            .model
            .predict_cached(&[
                (&*self.fbank_shape, &prepared.fbank),
                (&*self.masks_shape, &prepared.masks),
            ])
            .map_err(|error| backend_error("chunk embedding prediction failed", error))?;
        Ok((
            tensor.into_data(),
            predict_start.elapsed().as_micros() as u64,
        ))
    }

    pub(super) fn run(
        mut self,
        audios: &[&[f32]],
        prep_rx: Receiver<PreparedChunk>,
        job_rx: Receiver<ChunkJob>,
        emb_tx: Sender<EmbeddedChunk>,
    ) -> Result<GpuStats, PipelineError> {
        let mut total_predict_us = 0u64;
        let mut total_prep_us = 0u64;
        let mut chunk_num = 0u32;
        let mut decoded_done = false;

        loop {
            let Some(prepared) = self.next_prepared(
                audios,
                &prep_rx,
                &job_rx,
                &mut decoded_done,
                &mut total_prep_us,
            )?
            else {
                break;
            };

            let (data, predict_us) = self.predict(&prepared)?;
            total_predict_us += predict_us;
            chunk_num += 1;

            if emb_tx
                .send(EmbeddedChunk {
                    file_index: prepared.file_index,
                    window_start: prepared.window_start,
                    decoded: prepared.decoded,
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

        Ok(GpuStats {
            predict_us: total_predict_us,
            chunks: chunk_num,
            self_prep_us: total_prep_us,
        })
    }
}

#[derive(Clone)]
pub(super) struct ChunkSessionHandle {
    pub(super) cached_fbank_shape: Arc<CachedInputShape>,
    pub(super) cached_masks_shape: Arc<CachedInputShape>,
    pub(super) model: Arc<SharedCoreMlModel>,
}

#[derive(Clone)]
pub(super) struct ChunkSessionDescriptor {
    pub(super) handle: ChunkSessionHandle,
    pub(super) num_windows: usize,
    pub(super) fbank_frames: usize,
    pub(super) num_masks: usize,
}

#[derive(Clone)]
pub(super) struct ChunkEmbeddingResources {
    pub(super) sessions: Vec<ChunkSessionDescriptor>,
    pub(super) fbank_30s: Option<Arc<SharedCoreMlModel>>,
    pub(super) fbank_10s: Option<Arc<SharedCoreMlModel>>,
}

pub(super) struct PreparedChunk {
    pub(super) file_index: usize,
    pub(super) window_start: usize,
    pub(super) decoded: Vec<ndarray::Array2<f32>>,
    pub(super) fbank: Vec<f32>,
    pub(super) masks: Vec<f32>,
    pub(super) active: Vec<(usize, usize)>,
    pub(super) num_masks: usize,
}

pub(super) struct EmbeddedChunk {
    pub(super) file_index: usize,
    pub(super) window_start: usize,
    pub(super) decoded: Vec<ndarray::Array2<f32>>,
    pub(super) data: Vec<f32>,
    pub(super) active: Vec<(usize, usize)>,
    pub(super) num_masks: usize,
    pub(super) predict_us: u64,
}

pub(super) struct GpuStats {
    pub(super) predict_us: u64,
    pub(super) chunks: u32,
    pub(super) self_prep_us: u64,
}
