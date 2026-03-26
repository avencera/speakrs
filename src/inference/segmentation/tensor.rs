#[cfg(feature = "coreml")]
use ndarray::{Array2, Array3};

use super::SegmentationError;

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
