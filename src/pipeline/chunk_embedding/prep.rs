use std::sync::Arc;

use crossbeam_channel::{Receiver, Sender};

use crate::inference::coreml::{CachedInputShape, SharedCoreMlModel};

use super::gpu::PreparedChunk;
use super::{
    FBANK_SEGMENT_SAMPLES, PipelineError, backend_error, chunk_audio_raw,
    write_speaker_mask_to_slice,
};
use crate::inference::embedding::FBANK_FRAMES;

impl PrepScratch {
    pub(super) fn new(window_samples: usize) -> Self {
        Self {
            fbank_30s_buf: vec![0.0f32; 480_000],
            waveform_10s_buf: vec![0.0f32; window_samples],
            fbank_30s_shape: CachedInputShape::new("waveform", &[1, 1, 480_000]),
            fbank_10s_shape: CachedInputShape::new("waveform", &[1, 1, window_samples]),
        }
    }
}

impl ChunkPrep {
    fn compute_chunk_fbank(
        &self,
        global_start: usize,
        num_windows: usize,
        audio: &[f32],
        scratch: &mut PrepScratch,
    ) -> Result<Vec<f32>, PipelineError> {
        let chunk_audio_start = global_start * self.step_samples;
        let chunk_audio_len = self.window_samples + (num_windows - 1) * self.step_samples;
        let chunk_audio_end = (chunk_audio_start + chunk_audio_len).min(audio.len());
        let chunk_audio = &audio[chunk_audio_start..chunk_audio_end];

        let mut fbank = vec![0.0f32; self.largest_fbank_frames * 80];

        if chunk_audio.len() <= 480_000 && self.fbank_normalization_scope.uses_chunk_scope() {
            if let Some(fbank_model) = &self.fbank_30s {
                scratch.fbank_30s_buf[..chunk_audio.len()].copy_from_slice(chunk_audio);
                scratch.fbank_30s_buf[chunk_audio.len()..].fill(0.0);
                let tensor = fbank_model
                    .predict_cached(&[(&scratch.fbank_30s_shape, &*scratch.fbank_30s_buf)])
                    .map_err(|error| backend_error("chunk fbank 30s prediction failed", error))?;
                let (_, frames, _) = tensor
                    .try_rank3("chunk fbank 30s output")
                    .map_err(|error| backend_error("chunk fbank 30s output", error))?;
                let data = tensor.into_data();
                let copy_frames = frames.min(self.largest_fbank_frames);
                for row_idx in 0..copy_frames {
                    let offset = row_idx * 80;
                    fbank[offset..offset + 80].copy_from_slice(&data[offset..offset + 80]);
                }
            }
        } else if let Some(fbank_model) = &self.fbank_10s {
            let mut fbank_offset = 0usize;
            let mut audio_offset = 0usize;
            while fbank_offset < self.largest_fbank_frames && audio_offset < chunk_audio.len() {
                let segment_end = (audio_offset + self.window_samples).min(chunk_audio.len());
                let segment_len = segment_end - audio_offset;
                scratch.waveform_10s_buf[..segment_len]
                    .copy_from_slice(&chunk_audio[audio_offset..segment_end]);
                if segment_len < self.window_samples {
                    scratch.waveform_10s_buf[segment_len..].fill(0.0);
                }
                let tensor = fbank_model
                    .predict_cached(&[(&scratch.fbank_10s_shape, &*scratch.waveform_10s_buf)])
                    .map_err(|error| backend_error("chunk fbank 10s prediction failed", error))?;
                let (_, frames, _) = tensor
                    .try_rank3("chunk fbank 10s output")
                    .map_err(|error| backend_error("chunk fbank 10s output", error))?;
                let data = tensor.into_data();
                let copy = frames.min(self.largest_fbank_frames - fbank_offset);
                for row_idx in 0..copy {
                    let src = row_idx * 80;
                    let dst = (fbank_offset + row_idx) * 80;
                    fbank[dst..dst + 80].copy_from_slice(&data[src..src + 80]);
                }
                fbank_offset += FBANK_FRAMES;
                audio_offset += FBANK_SEGMENT_SAMPLES;
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
        let fbank =
            self.compute_chunk_fbank(job.window_start, job.decoded.len(), audio, scratch)?;
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
    pub(super) fbank_normalization_scope: crate::pipeline::config::ChunkFbankNormalizationScope,
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
