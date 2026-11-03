//! Filterbank model calls that build one chunk's stitched filterbank
//!
//! The CoreML filterbank models subtract each mel bin's mean over every frame of
//! their fixed-length input, so zero-padded input frames (log energy near
//! `ln(f32::EPSILON)`) shift every real frame of that call. The plan therefore
//! only gives a model real audio: a call that would run past the end of the file
//! starts earlier on the frame grid and skips its leading frames, and a call that
//! runs past the end of the chunk keeps reading the file

use crate::inference::embedding::{FBANK_FRAMES, FBANK_HOP_SAMPLES};

/// Samples one filterbank frame reads
const FRAME_SAMPLES: usize = 400;
/// Input samples of the 30 second filterbank model
pub(super) const THIRTY_SECOND_SAMPLES: usize = 480_000;
/// Output frames of the 30 second filterbank model
const THIRTY_SECOND_FRAMES: usize = (THIRTY_SECOND_SAMPLES - FRAME_SAMPLES) / FBANK_HOP_SAMPLES + 1;

/// Filterbank model a call runs
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum FbankModel {
    ThirtySecond,
    TenSecond,
}

/// One filterbank model call and the chunk frames it fills
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct FbankCall {
    pub(super) model: FbankModel,
    /// First file sample of the model input
    pub(super) audio_start: usize,
    /// Leading output frames the chunk does not use
    pub(super) skip_frames: usize,
    /// First chunk frame this call fills
    pub(super) dest_frame: usize,
    /// Number of chunk frames this call fills
    pub(super) frames: usize,
}

impl FbankCall {
    /// File samples of the model input, shorter than the model input only for audio
    /// shorter than one input
    pub(super) fn audio<'a>(&self, audio: &'a [f32], input_samples: usize) -> &'a [f32] {
        let end = (self.audio_start + input_samples).min(audio.len());
        &audio[self.audio_start..end]
    }
}

/// Where a chunk starts and how much filterbank it needs
#[derive(Clone, Copy, Debug)]
pub(super) struct ChunkFbankRequest {
    /// First file sample of the chunk, on the frame grid of the whole file
    pub(super) chunk_start: usize,
    /// Filterbank frames the chunk model reads
    pub(super) frame_capacity: usize,
    /// Samples in the whole file
    pub(super) audio_len: usize,
    /// Samples of one 10 second model input
    pub(super) ten_second_samples: usize,
    /// Whether a 30 second model may cover the chunk in one call
    pub(super) allow_thirty_second: bool,
}

impl ChunkFbankRequest {
    /// Model calls that fill every chunk frame backed by file audio; frames past the
    /// end of the file stay zero
    pub(super) fn plan(self) -> Vec<FbankCall> {
        let frames_needed = self.frames_needed();
        if frames_needed == 0 {
            return Vec::new();
        }

        if self.allow_thirty_second
            && let Some(call) = self.thirty_second_call(frames_needed)
        {
            return vec![call];
        }

        let mut calls = Vec::new();
        let mut dest_frame = 0;
        while dest_frame < frames_needed {
            let (audio_start, skip_frames) =
                self.aligned_start(dest_frame, self.ten_second_samples, FBANK_FRAMES);
            let frames = (FBANK_FRAMES - skip_frames).min(frames_needed - dest_frame);
            calls.push(FbankCall {
                model: FbankModel::TenSecond,
                audio_start,
                skip_frames,
                dest_frame,
                frames,
            });
            dest_frame += frames;
        }

        calls
    }

    /// Chunk frames whose samples all lie inside the file
    fn frames_needed(self) -> usize {
        let available = self
            .audio_len
            .checked_sub(self.chunk_start + FRAME_SAMPLES)
            .map_or(0, |tail| tail / FBANK_HOP_SAMPLES + 1);
        available.min(self.frame_capacity)
    }

    /// One 30 second call, only when 30 seconds of file audio can cover the chunk
    fn thirty_second_call(self, frames_needed: usize) -> Option<FbankCall> {
        if self.audio_len < THIRTY_SECOND_SAMPLES {
            return None;
        }

        let (audio_start, skip_frames) =
            self.aligned_start(0, THIRTY_SECOND_SAMPLES, THIRTY_SECOND_FRAMES);
        (skip_frames + frames_needed <= THIRTY_SECOND_FRAMES).then_some(FbankCall {
            model: FbankModel::ThirtySecond,
            audio_start,
            skip_frames,
            dest_frame: 0,
            frames: frames_needed,
        })
    }

    /// Input start for the call that fills `dest_frame`, moved back on the frame grid
    /// until the input ends inside the file. Never moves past the call's last output
    /// frame or the start of the file, so audio shorter than one input still pads
    fn aligned_start(
        self,
        dest_frame: usize,
        input_samples: usize,
        output_frames: usize,
    ) -> (usize, usize) {
        let nominal = self.chunk_start + dest_frame * FBANK_HOP_SAMPLES;
        let overrun = (nominal + input_samples).saturating_sub(self.audio_len);
        let skip_frames = overrun
            .div_ceil(FBANK_HOP_SAMPLES)
            .min(output_frames - 1)
            .min(nominal / FBANK_HOP_SAMPLES);
        (nominal - skip_frames * FBANK_HOP_SAMPLES, skip_frames)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEN_SECONDS: usize = 160_000;

    fn request(chunk_start: usize, frame_capacity: usize, audio_len: usize) -> ChunkFbankRequest {
        ChunkFbankRequest {
            chunk_start,
            frame_capacity,
            audio_len,
            ten_second_samples: TEN_SECONDS,
            allow_thirty_second: false,
        }
    }

    /// Every call reads only file audio, except the unread tail after its last frame,
    /// and the calls fill each needed chunk frame exactly once with the frame that
    /// starts at the same file sample
    fn assert_covers(request: ChunkFbankRequest, calls: &[FbankCall]) {
        let mut next = 0;
        for call in calls {
            let (input, outputs) = match call.model {
                FbankModel::TenSecond => (request.ten_second_samples, FBANK_FRAMES),
                FbankModel::ThirtySecond => (THIRTY_SECOND_SAMPLES, THIRTY_SECOND_FRAMES),
            };
            let last_read = call.audio_start
                + (call.skip_frames + call.frames - 1) * FBANK_HOP_SAMPLES
                + FRAME_SAMPLES;
            assert!(
                last_read <= request.audio_len,
                "{call:?} reads past the file"
            );
            if request.audio_len >= input {
                let unread = input - ((outputs - 1) * FBANK_HOP_SAMPLES + FRAME_SAMPLES);
                assert!(
                    call.audio_start + input <= request.audio_len + unread,
                    "{call:?} pads frames it normalizes over"
                );
            }
            assert!(call.skip_frames + call.frames <= outputs);
            assert_eq!(call.dest_frame, next);
            assert_eq!(
                call.audio_start + call.skip_frames * FBANK_HOP_SAMPLES,
                request.chunk_start + call.dest_frame * FBANK_HOP_SAMPLES
            );
            next += call.frames;
        }
        assert_eq!(next, request.frames_needed());
    }

    #[test]
    fn middle_chunk_reads_past_the_chunk_instead_of_padding() {
        // a 111-window one-second chunk needs 12000 frames, and the file continues
        let request = request(10 * 16_000, 12_000, 300 * 16_000);
        let calls = request.plan();

        assert_covers(request, &calls);
        assert_eq!(calls.len(), 13);
        assert!(calls.iter().all(|call| call.skip_frames == 0));
    }

    #[test]
    fn final_chunk_moves_its_last_call_back_inside_the_file() {
        // 79 seconds of audio left: the eighth call would end 0.84 s past the file
        let request = request(221 * 16_000, 12_000, 300 * 16_000);
        let calls = request.plan();

        assert_covers(request, &calls);
        assert_eq!(request.frames_needed(), 7_898);
        let last = calls.last().unwrap();
        assert_eq!(last.audio_start + TEN_SECONDS, 300 * 16_000);
        assert!(last.skip_frames > 0);
    }

    #[test]
    fn file_end_off_the_frame_grid_stays_inside_the_file() {
        for audio_len in [44 * 16_000 + 1, 44 * 16_000 + 79, 44 * 16_000 + 81, 704_399] {
            let request = request(0, 4_500, audio_len);
            assert_covers(request, &request.plan());
        }
    }

    #[test]
    fn thirty_second_call_covers_a_short_final_chunk_with_earlier_audio() {
        let mut request = request(290 * 16_000, 12_000, 303 * 16_000);
        request.allow_thirty_second = true;
        let calls = request.plan();

        assert_covers(request, &calls);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].model, FbankModel::ThirtySecond);
        assert_eq!(calls[0].audio_start, 273 * 16_000);
    }

    #[test]
    fn short_file_uses_ten_second_calls_instead_of_a_padded_thirty_second_call() {
        let mut request = request(0, 2_500, 25 * 16_000);
        request.allow_thirty_second = true;
        let calls = request.plan();

        assert_covers(request, &calls);
        assert!(calls.iter().all(|call| call.model == FbankModel::TenSecond));
    }

    #[test]
    fn audio_shorter_than_one_input_pads_once() {
        let request = request(0, 1_000, 9 * 16_000);
        let calls = request.plan();

        assert_covers(request, &calls);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].audio_start, 0);
    }
}
