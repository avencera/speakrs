use std::num::NonZeroUsize;

#[cfg(feature = "coreml")]
use ndarray::{Array2, Array3};

use super::SegmentationError;

#[cfg(feature = "migraphx")]
pub(super) type OutputShape3 = (usize, usize, usize);

/// Non-zero sliding-window length and step used by segmentation
#[derive(Clone, Copy, Debug)]
pub(crate) struct WindowSpec {
    window_samples: NonZeroUsize,
    step_samples: NonZeroUsize,
}

/// Invalid window or step duration before a model is loaded
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct InvalidWindowGeometry {
    pub message: String,
}

impl WindowSpec {
    pub(crate) fn new(window_samples: usize, step_samples: usize) -> Option<Self> {
        Some(Self {
            window_samples: NonZeroUsize::new(window_samples)?,
            step_samples: NonZeroUsize::new(step_samples)?,
        })
    }

    /// Convert durations in seconds to a checked window spec before session load
    pub(crate) fn from_seconds(
        window_duration: f32,
        step_duration: f32,
        sample_rate: usize,
    ) -> Result<Self, InvalidWindowGeometry> {
        if sample_rate == 0 {
            return Err(InvalidWindowGeometry {
                message: "sample rate must be greater than zero".to_owned(),
            });
        }
        let window_samples = duration_to_samples(window_duration, sample_rate, "window")?;
        let step_samples = duration_to_samples(step_duration, sample_rate, "step")?;
        Ok(Self {
            window_samples,
            step_samples,
        })
    }

    pub(crate) fn window_samples(self) -> usize {
        self.window_samples.get()
    }

    pub(crate) fn step_samples(self) -> usize {
        self.step_samples.get()
    }
}

fn duration_to_samples(
    duration: f32,
    sample_rate: usize,
    name: &str,
) -> Result<NonZeroUsize, InvalidWindowGeometry> {
    if !duration.is_finite() || duration <= 0.0 {
        return Err(InvalidWindowGeometry {
            message: format!(
                "{name} duration must be finite and greater than zero, got {duration}"
            ),
        });
    }
    let samples = duration * sample_rate as f32;
    if !samples.is_finite() || samples < 1.0 {
        return Err(InvalidWindowGeometry {
            message: format!(
                "{name} duration {duration} rounds below one sample at {sample_rate} Hz"
            ),
        });
    }
    if samples >= usize::MAX as f32 {
        return Err(InvalidWindowGeometry {
            message: format!(
                "{name} duration {duration} exceeds the maximum usize sample count at {sample_rate} Hz"
            ),
        });
    }
    NonZeroUsize::new(samples as usize).ok_or_else(|| InvalidWindowGeometry {
        message: format!("{name} duration {duration} produced zero samples"),
    })
}

pub(super) struct SegmentationWindows<'a> {
    audio: &'a [f32],
    offsets: Vec<usize>,
    padded: Option<Vec<f32>>,
    /// where the zero padded last window starts in `audio`
    #[cfg(feature = "_cuda")]
    tail_offset: usize,
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
    let has_tail = full_windows
        .checked_mul(step_samples)
        .is_some_and(|offset_after_full| offset_after_full < audio_len);
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
                #[cfg(feature = "_cuda")]
                tail_offset: 0,
                window_samples,
            };
        }

        let mut offsets = Vec::new();
        let mut offset = 0;
        while audio.len() >= window_samples && offset <= audio.len() - window_samples {
            offsets.push(offset);
            let Some(next_offset) = offset.checked_add(step_samples) else {
                offset = audio.len();
                break;
            };
            offset = next_offset;
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
            #[cfg(feature = "_cuda")]
            tail_offset: offset,
            window_samples,
        }
    }

    /// The audio that windows `next..next + useful` cover and where each starts in it,
    /// then `model - useful` starts at its end, which stand for zero windows
    ///
    /// Cutting `span[start..start + window]`, clipped and zero padded, gives exactly
    /// [`Self::window`], including the padded last window
    #[cfg(feature = "_cuda")]
    pub(super) fn span(&self, next: usize, useful: usize, model: usize) -> (&'a [f32], Vec<usize>) {
        let start_of = |idx: usize| self.offsets.get(idx).copied().unwrap_or(self.tail_offset);
        let first = start_of(next).min(self.audio.len());
        let last = start_of(next + useful.max(1) - 1);
        let end = (last + self.window_samples)
            .min(self.audio.len())
            .max(first);
        let span = &self.audio[first..end];
        let mut starts: Vec<usize> = (next..next + useful)
            .map(|idx| start_of(idx) - first)
            .collect();
        starts.resize(model, span.len());
        (span, starts)
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

#[cfg(feature = "migraphx")]
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

#[cfg(feature = "migraphx")]
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
    use super::{SegmentationWindows, WindowSpec, segmentation_window_count};
    #[cfg(feature = "migraphx")]
    use super::{first_output, output_shape3};

    #[test]
    fn from_seconds_rejects_non_positive_and_non_finite_steps() {
        for step in [0.0, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let error = WindowSpec::from_seconds(10.0, step, 16_000).unwrap_err();
            assert!(
                error.message.contains("step duration"),
                "unexpected message for step={step}: {}",
                error.message
            );
        }
        let too_small = WindowSpec::from_seconds(10.0, 1.0 / 32_000.0, 16_000).unwrap_err();
        assert!(too_small.message.contains("rounds below one sample"));
        let spec = WindowSpec::from_seconds(10.0, 1.0, 16_000).unwrap();
        assert_eq!(spec.window_samples(), 160_000);
        assert_eq!(spec.step_samples(), 16_000);
    }

    #[test]
    fn from_seconds_rejects_finite_sample_counts_above_usize() {
        let error = WindowSpec::from_seconds(10.0, 2.0e15, 16_000).unwrap_err();

        assert!(error.message.contains("maximum usize sample count"));
    }

    const WINDOW: usize = 160_000;

    fn window_spec(window_samples: usize, step_samples: usize) -> WindowSpec {
        WindowSpec::new(window_samples, step_samples).expect("non-zero window spec")
    }

    #[cfg(feature = "migraphx")]
    #[test]
    fn first_output_reports_missing_tensor() {
        let error = first_output(Vec::<()>::new(), "segmentation test").unwrap_err();

        assert_eq!(
            error.to_string(),
            "segmentation test: missing output tensor"
        );
    }

    #[cfg(feature = "migraphx")]
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
    fn window_counts_match_expected_boundary_results() {
        let small = [
            (0, 0),
            (1, 1),
            (8, 1),
            (15, 1),
            (16, 1),
            (17, 2),
            (23, 2),
            (24, 3),
            (31, 3),
            (32, 4),
            (40, 5),
        ];
        let standard = [(WINDOW, 1), (30 * 16_000, 22), (120 * 16_000, 112)];
        let accelerated = [
            (WINDOW - 1, 1),
            (WINDOW, 1),
            (WINDOW + 1, 2),
            (30 * 16_000, 21),
            (120 * 16_000 + 731, 107),
        ];

        for (spec, cases) in [
            (window_spec(16, 8), small.as_slice()),
            (window_spec(WINDOW, 16_000), standard.as_slice()),
            (window_spec(WINDOW, 16_640), accelerated.as_slice()),
        ] {
            for &(samples, expected) in cases {
                let audio = vec![1.0_f32; samples];
                let windows = SegmentationWindows::collect(&audio, spec);
                assert_eq!(
                    segmentation_window_count(samples, spec),
                    expected,
                    "count at samples={samples}, step={}",
                    spec.step_samples()
                );
                assert_eq!(
                    windows.total_windows(),
                    expected,
                    "collected windows at samples={samples}, step={}",
                    spec.step_samples()
                );
            }
        }
    }

    #[test]
    fn collect_handles_large_step_without_overflowing_window_bound() {
        let spec = window_spec(8, usize::MAX);
        let audio = vec![1.0_f32; 9];

        let windows = SegmentationWindows::collect(&audio, spec);

        assert_eq!(windows.total_windows(), 1);
        assert_eq!(windows.window(0, "overflowing step").unwrap(), &audio[..8]);
    }

    #[test]
    fn window_count_handles_unrepresentable_tail_offset() {
        let spec = window_spec(1, usize::MAX - 1);

        assert_eq!(segmentation_window_count(usize::MAX, spec), 2);
    }

    /// The CUDA backend cuts `span[start..start + window]`, clipped and zero padded;
    /// every batch, including the padded last window and zero pad rows, must come out
    /// as the windows the other backends upload
    #[cfg(feature = "_cuda")]
    #[test]
    fn span_cuts_the_same_windows_as_window() {
        let window = 100;
        for len in [40, 100, 1_000, 1_005, 1_037] {
            let audio: Vec<f32> = (0..len).map(|sample| sample as f32 + 1.0).collect();
            let windows = SegmentationWindows::collect(&audio, window_spec(window, 10));
            let total = windows.total_windows();
            let zeros = vec![0.0; window];
            for (next, useful) in [
                (0, 1),
                (0, total),
                (total - 1, 1),
                (total / 2, total - total / 2),
            ] {
                let model = useful + 3;
                let (span, starts) = windows.span(next, useful, model);
                let cut: Vec<Vec<f32>> = starts
                    .iter()
                    .map(|&start| {
                        let mut row =
                            span[start.min(span.len())..(start + window).min(span.len())].to_vec();
                        row.resize(window, 0.0);
                        row
                    })
                    .collect();
                let mut expected: Vec<Vec<f32>> = (next..next + useful)
                    .map(|idx| windows.window(idx, "test").unwrap().to_vec())
                    .collect();
                expected.resize(model, zeros.clone());
                assert_eq!(cut, expected, "len {len} next {next} useful {useful}");
            }
        }
    }
}
