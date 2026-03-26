#[cfg(feature = "coreml")]
use ndarray::{Array2, Array3};

use super::SegmentationError;

pub(super) struct SegmentationWindows<'a> {
    audio: &'a [f32],
    offsets: Vec<usize>,
    padded: Option<Vec<f32>>,
    window_samples: usize,
}

impl<'a> SegmentationWindows<'a> {
    pub(super) fn collect(audio: &'a [f32], window_samples: usize, step_samples: usize) -> Self {
        let mut offsets = Vec::new();
        let mut offset = 0;
        while offset + window_samples <= audio.len() {
            offsets.push(offset);
            offset += step_samples;
        }

        let padded = if offset < audio.len() && audio.len() > window_samples {
            let mut padded = vec![0.0f32; window_samples];
            let remaining = audio.len() - offset;
            padded[..remaining].copy_from_slice(&audio[offset..]);
            Some(padded)
        } else {
            None
        };

        Self {
            audio,
            offsets,
            padded,
            window_samples,
        }
    }

    pub(super) fn total_windows(&self) -> usize {
        self.offsets.len() + self.padded.is_some() as usize
    }

    pub(super) fn is_empty(&self) -> bool {
        self.total_windows() == 0
    }

    pub(super) fn window(
        &self,
        idx: usize,
        context: &'static str,
    ) -> Result<&[f32], SegmentationError> {
        if idx < self.offsets.len() {
            let start = self.offsets[idx];
            return Ok(&self.audio[start..start + self.window_samples]);
        }
        if idx == self.offsets.len() {
            return padded_window(&self.padded, context);
        }

        Err(SegmentationError::Invariant {
            context,
            message: format!(
                "window index {idx} exceeded total window count {}",
                self.total_windows()
            ),
        })
    }
}

#[cfg(feature = "coreml")]
pub(super) fn array3_slice<'a>(
    buffer: &'a Array3<f32>,
    context: &'static str,
) -> Result<&'a [f32], SegmentationError> {
    buffer
        .as_slice()
        .ok_or_else(|| SegmentationError::Invariant {
            context,
            message: "input buffer was not contiguous".to_owned(),
        })
}

pub(super) fn padded_window<'a>(
    padded: &'a Option<Vec<f32>>,
    context: &'static str,
) -> Result<&'a [f32], SegmentationError> {
    padded
        .as_deref()
        .ok_or_else(|| SegmentationError::Invariant {
            context,
            message: "missing padded window".to_owned(),
        })
}

#[cfg(feature = "coreml")]
pub(super) fn segmentation_array(
    frames: usize,
    classes: usize,
    data: Vec<f32>,
    context: &'static str,
) -> Result<Array2<f32>, SegmentationError> {
    Array2::from_shape_vec((frames, classes), data).map_err(|error| SegmentationError::Invariant {
        context,
        message: format!("invalid segmentation output shape: {error}"),
    })
}

#[cfg(feature = "coreml")]
pub(super) fn segmentation_array_from_slice(
    frames: usize,
    classes: usize,
    data: &[f32],
    context: &'static str,
) -> Result<Array2<f32>, SegmentationError> {
    segmentation_array(frames, classes, data.to_vec(), context)
}

#[cfg(feature = "coreml")]
pub(super) fn worker_panic(worker: &'static str) -> SegmentationError {
    SegmentationError::WorkerPanic {
        worker: worker.to_owned(),
    }
}
