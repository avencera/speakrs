use crate::pipeline::{FRAME_DURATION_SECONDS, FRAME_STEP_SECONDS, SEGMENTATION_WINDOW_SECONDS};

pub(in crate::pipeline) struct ChunkLayout {
    pub step_seconds: f64,
    pub step_samples: usize,
    pub window_samples: usize,
    pub start_frames: Vec<usize>,
    pub output_frames: usize,
}

impl ChunkLayout {
    pub(in crate::pipeline) fn new(
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

    pub(in crate::pipeline) fn without_frame_extent(
        step_seconds: f64,
        step_samples: usize,
        window_samples: usize,
    ) -> Self {
        Self::new(step_seconds, step_samples, window_samples, 0)
    }

    pub(in crate::pipeline) fn with_num_chunks(mut self, num_chunks: usize) -> Self {
        self.start_frames = chunk_start_frames(num_chunks, self.step_seconds);
        self.output_frames = total_output_frames(num_chunks, self.step_seconds);
        self
    }

    pub(in crate::pipeline) fn chunk_audio<'a>(
        &self,
        audio: &'a [f32],
        chunk_idx: usize,
    ) -> &'a [f32] {
        chunk_audio_raw(audio, self.step_samples, self.window_samples, chunk_idx)
    }
}

pub(crate) fn chunk_audio_raw(
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

pub(in crate::pipeline) fn chunk_start_frames(num_chunks: usize, step_seconds: f64) -> Vec<usize> {
    (0..num_chunks)
        .map(|chunk_idx| {
            closest_frame(chunk_idx as f64 * step_seconds + 0.5 * FRAME_DURATION_SECONDS)
        })
        .collect()
}

pub(in crate::pipeline) fn total_output_frames(num_chunks: usize, step_seconds: f64) -> usize {
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
