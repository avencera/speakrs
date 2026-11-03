use std::sync::Arc;

use crossbeam_channel::{Receiver, Sender};

use crate::inference::coreml::{CachedInputShape, SharedCoreMlModel};
use crate::inference::geometry::CoreMlTensor;
use crate::pipeline::config::ChunkFbankNormalizationScope;

use super::fbank_plan::{ChunkFbankRequest, FbankCall, FbankModel, THIRTY_SECOND_SAMPLES};
use super::gpu::PreparedChunk;
use super::{
    PipelineError, backend_error, chunk_audio_raw, invariant_error, write_speaker_mask_to_slice,
};
use crate::inference::embedding::FBANK_FEATURES;

fn copy_fbank_output(
    tensor: CoreMlTensor,
    context: &'static str,
    fbank: &mut [f32],
    call: FbankCall,
) -> Result<(), PipelineError> {
    let (batch, frames, features) = tensor
        .try_rank3(context)
        .map_err(|error| backend_error(context, error))?;
    if batch != 1 {
        return Err(backend_error(
            context,
            format!("expected batch size 1, got {batch}"),
        ));
    }
    if features != FBANK_FEATURES {
        return Err(backend_error(
            context,
            format!("expected {FBANK_FEATURES} filterbank features, got {features}"),
        ));
    }
    if call.skip_frames + call.frames > frames {
        return Err(backend_error(
            context,
            format!(
                "expected at least {} filterbank frames, got {frames}",
                call.skip_frames + call.frames
            ),
        ));
    }

    let source = &tensor.into_data()
        [call.skip_frames * FBANK_FEATURES..(call.skip_frames + call.frames) * FBANK_FEATURES];
    let output_offset = call.dest_frame * FBANK_FEATURES;
    fbank[output_offset..output_offset + source.len()].copy_from_slice(source);
    Ok(())
}

/// Copy `audio` into a fixed-size model input and zero the rest
fn stage_input(buffer: &mut [f32], audio: &[f32]) {
    buffer[..audio.len()].copy_from_slice(audio);
    buffer[audio.len()..].fill(0.0);
}

impl PrepScratch {
    pub(super) fn new(window_samples: usize) -> Self {
        Self {
            fbank_30s_buf: vec![0.0f32; THIRTY_SECOND_SAMPLES],
            waveform_10s_buf: vec![0.0f32; window_samples],
            fbank_30s_shape: CachedInputShape::new("waveform", &[1, 1, THIRTY_SECOND_SAMPLES]),
            fbank_10s_shape: CachedInputShape::new("waveform", &[1, 1, window_samples]),
        }
    }
}

impl ChunkPrep {
    fn compute_chunk_fbank(
        &self,
        global_start: usize,
        audio: &[f32],
        scratch: &mut PrepScratch,
    ) -> Result<Vec<f32>, PipelineError> {
        let request = ChunkFbankRequest {
            chunk_start: global_start * self.step_samples,
            frame_capacity: self.largest_fbank_frames,
            audio_len: audio.len(),
            ten_second_samples: self.window_samples,
            allow_thirty_second: self.fbank_normalization_scope.uses_chunk_scope()
                && self.fbank_30s.is_some(),
        };
        let mut fbank = vec![0.0f32; self.largest_fbank_frames * FBANK_FEATURES];

        for call in request.plan() {
            match call.model {
                FbankModel::ThirtySecond => {
                    let model = self
                        .fbank_30s
                        .as_deref()
                        .ok_or_else(|| invariant_error("chunk fbank plan needs the 30s model"))?;
                    stage_input(
                        &mut scratch.fbank_30s_buf,
                        call.audio(audio, THIRTY_SECOND_SAMPLES),
                    );
                    let tensor = model
                        .predict_cached(&[(&scratch.fbank_30s_shape, &*scratch.fbank_30s_buf)])
                        .map_err(|error| {
                            backend_error("chunk fbank 30s prediction failed", error)
                        })?;
                    copy_fbank_output(tensor, "chunk fbank 30s output", &mut fbank, call)?;
                }
                FbankModel::TenSecond => {
                    let model = self.fbank_10s.as_deref().ok_or_else(|| {
                        invariant_error("no compatible chunk fbank model is available")
                    })?;
                    stage_input(
                        &mut scratch.waveform_10s_buf,
                        call.audio(audio, self.window_samples),
                    );
                    let tensor = model
                        .predict_cached(&[(&scratch.fbank_10s_shape, &*scratch.waveform_10s_buf)])
                        .map_err(|error| {
                            backend_error("chunk fbank 10s prediction failed", error)
                        })?;
                    copy_fbank_output(tensor, "chunk fbank 10s output", &mut fbank, call)?;
                }
            }
        }

        Ok(fbank)
    }

    pub(super) fn prep(
        &self,
        job: ChunkJob,
        audio: &[f32],
        scratch: &mut PrepScratch,
    ) -> Result<PreparedChunk, PipelineError> {
        let fbank = self.compute_chunk_fbank(job.window_start, audio, scratch)?;
        let (masks, active) = SpeakerMaskLayout {
            step_samples: self.step_samples,
            window_samples: self.window_samples,
            num_speakers: self.num_speakers,
            min_num_samples: self.min_num_samples,
            num_masks: self.largest_num_masks,
            max_active: self.max_active,
        }
        .collect(job.window_start, &job.decoded, audio);

        Ok(PreparedChunk {
            file_index: job.file_index,
            window_start: job.window_start,
            decoded: job.decoded,
            fbank,
            masks,
            active,
            num_masks: self.largest_num_masks,
        })
    }
}

pub(super) struct SpeakerMaskLayout {
    pub(super) step_samples: usize,
    pub(super) window_samples: usize,
    pub(super) num_speakers: usize,
    pub(super) min_num_samples: usize,
    pub(super) num_masks: usize,
    pub(super) max_active: usize,
}

impl SpeakerMaskLayout {
    pub(super) fn collect(
        self,
        window_start: usize,
        decoded: &[ndarray::Array2<f32>],
        audio: &[f32],
    ) -> (Vec<f32>, Vec<(usize, usize)>) {
        let mut masks = vec![0.0f32; self.num_masks * 589];
        let mut active = Vec::with_capacity(self.max_active);

        for (local_idx, decoded) in decoded.iter().enumerate() {
            let global_idx = window_start + local_idx;
            let win_audio =
                chunk_audio_raw(audio, self.step_samples, self.window_samples, global_idx);
            for speaker_idx in 0..self.num_speakers {
                let mask_idx = local_idx * self.num_speakers + speaker_idx;
                if mask_idx >= self.num_masks {
                    break;
                }
                let dest = &mut masks[mask_idx * 589..mask_idx * 589 + 589];
                if write_speaker_mask_to_slice(
                    &decoded.view(),
                    speaker_idx,
                    win_audio.len(),
                    self.min_num_samples,
                    dest,
                ) {
                    active.push((local_idx, speaker_idx));
                }
            }
        }

        (masks, active)
    }
}

pub(super) fn audio_for<'a>(
    audios: &'a [&[f32]],
    file_index: usize,
) -> Result<&'a [f32], PipelineError> {
    audios.get(file_index).copied().ok_or_else(|| {
        super::invariant_error(format!(
            "chunk job file index {file_index} is outside audio table length {}",
            audios.len()
        ))
    })
}

pub(super) struct PrepWorker {
    pub(super) prep: ChunkPrep,
    pub(super) scratch: PrepScratch,
}

impl PrepWorker {
    pub(super) fn run(
        mut self,
        audios: &[&[f32]],
        job_rx: Receiver<ChunkJob>,
        prep_tx: Sender<PreparedChunk>,
    ) -> Result<PrepStats, PipelineError> {
        let mut stats = PrepStats::default();

        while let Ok(job) = job_rx.recv() {
            let audio = audio_for(audios, job.file_index)?;
            let chunk_audio_start = job.window_start * self.prep.step_samples;
            debug_assert!(chunk_audio_start < audio.len());
            let prep_start = std::time::Instant::now();
            let prepared = self.prep.prep(job, audio, &mut self.scratch)?;
            stats.fbank_us += prep_start.elapsed().as_micros() as u64;
            stats.chunks += 1;
            if prep_tx.send(prepared).is_err() {
                break;
            }
        }
        Ok(stats)
    }
}

#[derive(Clone)]
pub(super) struct ChunkPrep {
    pub(super) step_samples: usize,
    pub(super) window_samples: usize,
    pub(super) num_speakers: usize,
    pub(super) min_num_samples: usize,
    pub(super) largest_fbank_frames: usize,
    pub(super) largest_num_masks: usize,
    pub(super) max_active: usize,
    pub(super) fbank_30s: Option<Arc<SharedCoreMlModel>>,
    pub(super) fbank_10s: Option<Arc<SharedCoreMlModel>>,
    pub(super) fbank_normalization_scope: ChunkFbankNormalizationScope,
}

pub(super) struct PrepScratch {
    pub(super) fbank_30s_buf: Vec<f32>,
    pub(super) waveform_10s_buf: Vec<f32>,
    pub(super) fbank_30s_shape: CachedInputShape,
    pub(super) fbank_10s_shape: CachedInputShape,
}

#[derive(Default)]
pub(super) struct PrepStats {
    pub(super) chunks: u32,
    pub(super) fbank_us: u64,
}

/// One non-empty group of decoded windows. Single-file work uses file index zero.
pub(super) struct ChunkJob {
    pub(super) file_index: usize,
    pub(super) window_start: usize,
    pub(super) decoded: Vec<ndarray::Array2<f32>>,
}

impl ChunkJob {
    pub(super) fn new(
        file_index: usize,
        window_start: usize,
        decoded: Vec<ndarray::Array2<f32>>,
    ) -> Result<Self, PipelineError> {
        if decoded.is_empty() {
            return Err(super::invariant_error(
                "chunk job decoded windows must be non-empty",
            ));
        }
        Ok(Self {
            file_index,
            window_start,
            decoded,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{ChunkJob, audio_for};
    use ndarray::Array2;

    #[test]
    fn chunk_job_rejects_empty_decoded_windows() {
        assert!(ChunkJob::new(0, 0, Vec::new()).is_err());
        assert!(ChunkJob::new(0, 0, vec![Array2::zeros((2, 3))]).is_ok());
    }

    #[test]
    fn audio_table_rejects_invalid_file_index() {
        let audio: &[f32] = &[0.0; 8];
        let audios = [audio];
        assert!(audio_for(&audios, 0).is_ok());
        assert!(audio_for(&audios, 1).is_err());
    }
}
