use ndarray::{Array1, Array2, Array3, s};

#[derive(Debug, Clone, Copy)]
pub struct AggregateOptions {
    pub use_hamming: bool,
    pub skip_average: bool,
    pub missing: f32,
    pub warmup_left: usize,
    pub warmup_right: usize,
    pub output_frames: Option<usize>,
}

impl Default for AggregateOptions {
    fn default() -> Self {
        Self {
            use_hamming: false,
            skip_average: false,
            missing: 0.0,
            warmup_left: 0,
            warmup_right: 0,
            output_frames: None,
        }
    }
}

pub struct Aggregate<'a>(&'a Array3<f32>);

impl<'a> Aggregate<'a> {
    pub fn new(windows: &'a Array3<f32>) -> Self {
        Self(windows)
    }

    pub fn overlap_add(&self, step_frames: usize, warmup_frames: usize) -> Array2<f32> {
        let start_frames: Vec<usize> = (0..self.0.shape()[0])
            .map(|window_idx| window_idx * step_frames)
            .collect();
        self.aggregate_at(
            &start_frames,
            AggregateOptions {
                use_hamming: true,
                warmup_left: warmup_frames,
                warmup_right: warmup_frames,
                ..Default::default()
            },
        )
    }

    pub fn aggregate_at(&self, start_frames: &[usize], options: AggregateOptions) -> Array2<f32> {
        let num_windows = self.0.shape()[0];
        let frames_per_window = self.0.shape()[1];
        let num_speakers = self.0.shape()[2];

        if num_windows == 0 {
            return Array2::<f32>::zeros((0, num_speakers));
        }

        assert_eq!(
            num_windows,
            start_frames.len(),
            "start_frames length must match number of windows"
        );

        let total_frames = options
            .output_frames
            .unwrap_or(start_frames[num_windows - 1] + frames_per_window);
        assert!(
            total_frames >= start_frames[num_windows - 1] + frames_per_window,
            "output_frames must cover the last window"
        );

        let weights = build_weights(frames_per_window, options);
        let mut numerator = Array2::<f32>::zeros((total_frames, num_speakers));
        let mut denominator =
            (!options.skip_average).then(|| Array2::<f32>::zeros((total_frames, num_speakers)));
        let mut missing_mask =
            (options.missing != 0.0).then(|| Array2::<f32>::zeros((total_frames, num_speakers)));

        for (window_idx, start_frame) in start_frames.iter().copied().enumerate().take(num_windows)
        {
            for frame_offset in 0..frames_per_window {
                let output_frame = start_frame + frame_offset;
                let weight = weights[frame_offset];
                let window_slice = self.0.slice(s![window_idx, frame_offset, ..]);

                for speaker_idx in 0..num_speakers {
                    let value = window_slice[speaker_idx];
                    if value.is_nan() {
                        continue;
                    }

                    numerator[[output_frame, speaker_idx]] += weight * value;
                    if let Some(denominator) = denominator.as_mut() {
                        denominator[[output_frame, speaker_idx]] += weight;
                    }
                    if let Some(missing_mask) = missing_mask.as_mut() {
                        missing_mask[[output_frame, speaker_idx]] = 1.0;
                    }
                }
            }
        }

        if let Some(denominator) = denominator.as_ref() {
            for frame_idx in 0..total_frames {
                for speaker_idx in 0..num_speakers {
                    let weight = denominator[[frame_idx, speaker_idx]];
                    if weight > 0.0 {
                        numerator[[frame_idx, speaker_idx]] /= weight;
                    }
                }
            }
        }

        if let Some(missing_mask) = missing_mask.as_ref() {
            for frame_idx in 0..total_frames {
                for speaker_idx in 0..num_speakers {
                    if missing_mask[[frame_idx, speaker_idx]] == 0.0 {
                        numerator[[frame_idx, speaker_idx]] = options.missing;
                    }
                }
            }
        }

        numerator
    }
}

fn hamming_window(len: usize) -> Array1<f32> {
    if len <= 1 {
        return Array1::ones(len);
    }

    let n_minus_1 = (len - 1) as f32;
    Array1::from_shape_fn(len, |index| {
        0.54 - 0.46 * (2.0 * std::f32::consts::PI * index as f32 / n_minus_1).cos()
    })
}

fn build_weights(frames_per_window: usize, options: AggregateOptions) -> Array1<f32> {
    let mut weights = if options.use_hamming {
        hamming_window(frames_per_window)
    } else {
        Array1::ones(frames_per_window)
    };
    for frame_idx in 0..options.warmup_left.min(frames_per_window) {
        weights[frame_idx] = 0.0;
    }
    for frame_idx in 0..options.warmup_right.min(frames_per_window) {
        weights[frames_per_window - 1 - frame_idx] = 0.0;
    }
    weights
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use ndarray::{Array3, array};

    use super::*;

    #[test]
    fn hamming_window_symmetry() {
        let weights = hamming_window(11);

        for index in 0..11 / 2 {
            assert_relative_eq!(weights[index], weights[11 - 1 - index], epsilon = 1e-6);
        }
    }

    #[test]
    fn hamming_window_endpoints() {
        let weights = hamming_window(11);
        assert_relative_eq!(weights[0], 0.08, epsilon = 1e-6);
        assert_relative_eq!(weights[10], 0.08, epsilon = 1e-6);
    }

    #[test]
    fn hamming_window_center_peak() {
        let weights = hamming_window(11);
        assert_relative_eq!(weights[5], 1.0, epsilon = 1e-6);
    }

    #[test]
    fn hamming_window_len_one() {
        let weights = hamming_window(1);
        assert_eq!(weights.len(), 1);
        assert_relative_eq!(weights[0], 1.0);
    }

    #[test]
    fn single_window_passthrough() {
        let data = array![[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]];
        let result = Aggregate::new(&data).overlap_add(1, 0);

        assert_eq!(result.shape(), &[3, 2]);
        for frame_idx in 0..3 {
            for speaker_idx in 0..2 {
                assert_relative_eq!(
                    result[[frame_idx, speaker_idx]],
                    data[[0, frame_idx, speaker_idx]],
                    epsilon = 1e-6
                );
            }
        }
    }

    #[test]
    fn two_windows_overlap_blend() {
        let frames_per_window = 4;
        let step = 2;
        let mut windows = Array3::<f32>::zeros((2, frames_per_window, 1));
        for frame_idx in 0..frames_per_window {
            windows[[0, frame_idx, 0]] = 1.0;
            windows[[1, frame_idx, 0]] = 3.0;
        }

        let result = Aggregate::new(&windows).overlap_add(step, 0);
        let total_frames = step + frames_per_window;
        let hamming = hamming_window(frames_per_window);
        assert_eq!(result.shape(), &[total_frames, 1]);

        for frame_idx in 0..2 {
            let output_frame = step + frame_idx;
            let first_weight = hamming[step + frame_idx];
            let second_weight = hamming[frame_idx];
            let expected =
                (first_weight * 1.0 + second_weight * 3.0) / (first_weight + second_weight);
            assert_relative_eq!(result[[output_frame, 0]], expected, epsilon = 1e-6);
        }
    }

    #[test]
    fn warmup_zeros_edges() {
        let frames_per_window = 6;
        let warmup = 2;
        let mut windows = Array3::<f32>::zeros((1, frames_per_window, 1));
        for frame_idx in 0..frames_per_window {
            windows[[0, frame_idx, 0]] = 10.0;
        }

        let result = Aggregate::new(&windows).overlap_add(1, warmup);

        for frame_idx in 0..warmup {
            assert_relative_eq!(result[[frame_idx, 0]], 0.0, epsilon = 1e-6);
        }
        for frame_idx in 0..warmup {
            assert_relative_eq!(
                result[[frames_per_window - 1 - frame_idx, 0]],
                0.0,
                epsilon = 1e-6
            );
        }
        for frame_idx in warmup..frames_per_window - warmup {
            assert_relative_eq!(result[[frame_idx, 0]], 10.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn aggregate_without_hamming_and_average_sums_windows() {
        let windows = array![[[1.0], [2.0]], [[3.0], [4.0]]];
        let result = Aggregate::new(&windows).aggregate_at(
            &[0, 1],
            AggregateOptions {
                skip_average: true,
                ..Default::default()
            },
        );

        assert_eq!(result[[0, 0]], 1.0);
        assert_eq!(result[[1, 0]], 5.0);
        assert_eq!(result[[2, 0]], 4.0);
    }

    #[test]
    fn aggregate_at_respects_custom_offsets() {
        let windows = array![[[1.0]], [[2.0]], [[3.0]]];
        let result = Aggregate::new(&windows).aggregate_at(&[0, 2, 3], AggregateOptions::default());

        assert_eq!(result.nrows(), 4);
        assert_eq!(result[[0, 0]], 1.0);
        assert_eq!(result[[1, 0]], 0.0);
        assert_eq!(result[[2, 0]], 2.0);
        assert_eq!(result[[3, 0]], 3.0);
    }

    #[test]
    fn aggregate_at_respects_explicit_output_frames() {
        let windows = array![[[1.0], [2.0]], [[3.0], [4.0]]];
        let result = Aggregate::new(&windows).aggregate_at(
            &[0, 1],
            AggregateOptions {
                output_frames: Some(4),
                ..Default::default()
            },
        );

        assert_eq!(result.nrows(), 4);
        assert_eq!(result[[3, 0]], 0.0);
    }
}
