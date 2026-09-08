use std::num::NonZeroUsize;

#[cfg(feature = "coreml")]
use ndarray::{Array2, Array3};

use super::SegmentationError;

pub(super) type OutputShape3 = (usize, usize, usize);

/// Non-zero sliding-window length and step used by segmentation
#[derive(Clone, Copy, Debug)]
pub(crate) struct WindowSpec {
    window_samples: NonZeroUsize,
    step_samples: NonZeroUsize,
}

impl WindowSpec {
    pub(crate) fn new(window_samples: usize, step_samples: usize) -> Option<Self> {
        Some(Self {
            window_samples: NonZeroUsize::new(window_samples)?,
            step_samples: NonZeroUsize::new(step_samples)?,
        })
    }

    pub(crate) fn window_samples(self) -> usize {
        self.window_samples.get()
    }

    pub(crate) fn step_samples(self) -> usize {
        self.step_samples.get()
    }
}

pub(super) struct SegmentationWindows<'a> {
    audio: &'a [f32],
    offsets: Vec<usize>,
    padded: Option<Vec<f32>>,
    window_samples: usize,
}

/// Count sliding windows the same way [`SegmentationWindows::collect`] emits them
pub(crate) fn segmentation_window_count(audio_len: usize, spec: WindowSpec) -> usize {
    if audio_len == 0 {
        return 0;
    }

    let window_samples = spec.window_samples();
    if audio_len <= window_samples {
        return 1;
    }

    let step_samples = spec.step_samples();
    let full_windows = (audio_len - window_samples) / step_samples + 1;
    let offset_after_full = full_windows * step_samples;
    let has_tail = offset_after_full < audio_len;
    full_windows + has_tail as usize
}

impl<'a> SegmentationWindows<'a> {
    pub(super) fn collect(audio: &'a [f32], spec: WindowSpec) -> Self {
        let window_samples = spec.window_samples();
        let step_samples = spec.step_samples();
        if !audio.is_empty() && audio.len() < window_samples {
            let mut padded = vec![0.0f32; window_samples];
            padded[..audio.len()].copy_from_slice(audio);
            return Self {
                audio,
                offsets: Vec::new(),
                padded: Some(padded),
                window_samples,
            };
        }

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

pub(super) fn first_output<T>(
    outputs: impl IntoIterator<Item = T>,
    context: &'static str,
) -> Result<T, SegmentationError> {
    outputs
        .into_iter()
        .next()
        .ok_or_else(|| SegmentationError::MalformedOutput {
            context,
            message: "missing output tensor".to_owned(),
        })
}

pub(super) fn output_shape3(
    shape: &ort::value::Shape,
    context: &'static str,
) -> Result<OutputShape3, SegmentationError> {
    let [batch, frames, classes]: [i64; 3] =
        shape
            .as_ref()
            .try_into()
            .map_err(|_| SegmentationError::MalformedOutput {
                context,
                message: format!("expected rank 3 output, got shape {shape}"),
            })?;

    let dims = [batch, frames, classes];
    if dims.iter().any(|dim| *dim < 0) {
        return Err(SegmentationError::MalformedOutput {
            context,
            message: format!("expected non-negative output dimensions, got shape {shape}"),
        });
    }

    Ok((batch as usize, frames as usize, classes as usize))
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

#[cfg(test)]
mod tests {
    use super::{
        SegmentationWindows, WindowSpec, first_output, output_shape3, segmentation_window_count,
    };

    const WINDOW: usize = 160_000;

    fn window_spec(window_samples: usize, step_samples: usize) -> WindowSpec {
        WindowSpec::new(window_samples, step_samples).expect("non-zero window spec")
    }

    #[test]
    fn window_count_keeps_the_padded_tail_at_control_lengths() {
        let spec_16k = window_spec(WINDOW, 16_000);
        let spec_16640 = window_spec(WINDOW, 16_640);
        assert_eq!(segmentation_window_count(WINDOW, spec_16k), 1);
        assert_eq!(segmentation_window_count(30 * 16_000, spec_16k), 22);
        assert_eq!(segmentation_window_count(120 * 16_000, spec_16k), 112);
        assert_eq!(segmentation_window_count(30 * 16_000, spec_16640), 21);
    }

    #[test]
    fn window_count_matches_collected_full_and_padded_windows() {
        let spec = window_spec(WINDOW, 16_640);
        for audio_samples in [
            WINDOW - 1,
            WINDOW,
            WINDOW + 1,
            30 * 16_000,
            120 * 16_000 + 731,
        ] {
            let audio = vec![0.0; audio_samples];
            let windows = SegmentationWindows::collect(&audio, spec);

            assert_eq!(
                segmentation_window_count(audio_samples, spec),
                windows.total_windows()
            );
        }
    }

    #[test]
    fn first_output_reports_missing_tensor() {
        let error = first_output(Vec::<()>::new(), "segmentation test").unwrap_err();

        assert_eq!(
            error.to_string(),
            "segmentation test: missing output tensor"
        );
    }

    #[test]
    fn output_shape3_reports_low_rank_tensor() {
        let shape = ort::value::Shape::from([10_i64, 3]);
        let error = output_shape3(&shape, "segmentation test").unwrap_err();

        assert_eq!(
            error.to_string(),
            "segmentation test: expected rank 3 output, got shape [10, 3]"
        );
    }

    #[test]
    fn nonempty_recording_shorter_than_one_window_emits_one_padded_window() {
        let audio = vec![0.5_f32; 8];
        let spec = window_spec(16, 8);
        let windows = SegmentationWindows::collect(&audio, spec);
        assert_eq!(windows.total_windows(), 1);
        assert_eq!(segmentation_window_count(audio.len(), spec), 1);
        let window = windows.window(0, "short recording").expect("window");
        assert_eq!(window.len(), 16);
        assert_eq!(&window[..8], audio.as_slice());
        assert_eq!(&window[8..], &[0.0_f32; 8]);
    }

    #[test]
    fn empty_recording_emits_no_window() {
        let spec = window_spec(16, 8);
        let windows = SegmentationWindows::collect(&[], spec);
        assert!(windows.is_empty());
        assert_eq!(segmentation_window_count(0, spec), 0);
    }

    #[test]
    fn window_count_matches_collected_windows() {
        let spec = window_spec(16, 8);
        for audio_len in [0, 1, 8, 15, 16, 17, 23, 24, 31, 32, 40] {
            let audio = vec![1.0_f32; audio_len];
            let windows = SegmentationWindows::collect(&audio, spec);
            assert_eq!(
                segmentation_window_count(audio_len, spec),
                windows.total_windows(),
                "window count drifted from collect at audio_len={audio_len}"
            );
        }
    }
}
