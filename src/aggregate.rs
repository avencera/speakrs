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

pub fn hamming_window(len: usize) -> Array1<f32> {
    if len <= 1 {
        return Array1::ones(len);
    }

    let n_minus_1 = (len - 1) as f32;
    Array1::from_shape_fn(len, |n| {
        0.54 - 0.46 * (2.0 * std::f32::consts::PI * n as f32 / n_minus_1).cos()
    })
}

pub fn overlap_add(windows: &Array3<f32>, step_frames: usize, warmup_frames: usize) -> Array2<f32> {
    let start_frames: Vec<usize> = (0..windows.shape()[0]).map(|i| i * step_frames).collect();
    aggregate_at(
        windows,
        &start_frames,
        AggregateOptions {
            use_hamming: true,
            warmup_left: warmup_frames,
            warmup_right: warmup_frames,
            ..Default::default()
        },
    )
}

pub fn aggregate_at(
    windows: &Array3<f32>,
    start_frames: &[usize],
    options: AggregateOptions,
) -> Array2<f32> {
    let num_windows = windows.shape()[0];
    let frames_per_window = windows.shape()[1];
    let num_speakers = windows.shape()[2];

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

    let mut weights = if options.use_hamming {
        hamming_window(frames_per_window)
    } else {
        Array1::ones(frames_per_window)
    };
    for i in 0..options.warmup_left.min(frames_per_window) {
        weights[i] = 0.0;
    }
    for i in 0..options.warmup_right.min(frames_per_window) {
        weights[frames_per_window - 1 - i] = 0.0;
    }

    let mut numerator = Array2::<f32>::zeros((total_frames, num_speakers));
    let mut denominator =
        (!options.skip_average).then(|| Array2::<f32>::zeros((total_frames, num_speakers)));
    let mut mask =
        (options.missing != 0.0).then(|| Array2::<f32>::zeros((total_frames, num_speakers)));

    for (i, start_frame) in start_frames.iter().copied().enumerate().take(num_windows) {
        for j in 0..frames_per_window {
            let out_frame = start_frame + j;
            let w = weights[j];

            let window_slice = windows.slice(s![i, j, ..]);
            for s in 0..num_speakers {
                let value = window_slice[s];
                if value.is_nan() {
                    continue;
                }

                numerator[[out_frame, s]] += w * value;
                if let Some(denominator) = denominator.as_mut() {
                    denominator[[out_frame, s]] += w;
                }
                if let Some(mask) = mask.as_mut() {
                    mask[[out_frame, s]] = 1.0;
                }
            }
        }
    }

    if let Some(denominator) = denominator.as_ref() {
        for frame in 0..total_frames {
            for s in 0..num_speakers {
                let weight = denominator[[frame, s]];
                if weight > 0.0 {
                    numerator[[frame, s]] /= weight;
                }
            }
        }
    }

    if let Some(mask) = mask.as_ref() {
        for frame in 0..total_frames {
            for s in 0..num_speakers {
                if mask[[frame, s]] == 0.0 {
                    numerator[[frame, s]] = options.missing;
                }
            }
        }
    }

    numerator
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn hamming_window_symmetry() {
        let len = 11;
        let w = hamming_window(len);

        for i in 0..len / 2 {
            assert_relative_eq!(w[i], w[len - 1 - i], epsilon = 1e-6);
        }
    }

    #[test]
    fn hamming_window_endpoints() {
        let w = hamming_window(11);
        assert_relative_eq!(w[0], 0.08, epsilon = 1e-6);
        assert_relative_eq!(w[10], 0.08, epsilon = 1e-6);
    }

    #[test]
    fn hamming_window_center_peak() {
        let w = hamming_window(11);
        assert_relative_eq!(w[5], 1.0, epsilon = 1e-6);
    }

    #[test]
    fn hamming_window_len_one() {
        let w = hamming_window(1);
        assert_eq!(w.len(), 1);
        assert_relative_eq!(w[0], 1.0);
    }

    #[test]
    fn single_window_passthrough() {
        let data = array![[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]];
        let result = overlap_add(&data, 1, 0);

        assert_eq!(result.shape(), &[3, 2]);
        // single window: numerator = h*x, denominator = h, so result = x
        for j in 0..3 {
            for s in 0..2 {
                assert_relative_eq!(result[[j, s]], data[[0, j, s]], epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn two_windows_overlap_blend() {
        let frames_per_window = 4;
        let step = 2;
        let num_speakers = 1;

        let mut windows = Array3::<f32>::zeros((2, frames_per_window, num_speakers));
        for j in 0..frames_per_window {
            windows[[0, j, 0]] = 1.0;
            windows[[1, j, 0]] = 3.0;
        }

        let result = overlap_add(&windows, step, 0);
        let total = step + frames_per_window;
        assert_eq!(result.shape(), &[total, num_speakers]);

        let h = hamming_window(frames_per_window);

        // overlap region: frames 2 and 3
        for j in 0..2 {
            let frame = step + j;
            let h0 = h[step + j]; // contribution from window 0
            let h1 = h[j]; // contribution from window 1
            let expected = (h0 * 1.0 + h1 * 3.0) / (h0 + h1);
            assert_relative_eq!(result[[frame, 0]], expected, epsilon = 1e-6);
        }
    }

    #[test]
    fn warmup_zeros_edges() {
        let frames_per_window = 6;
        let warmup = 2;
        let num_speakers = 1;

        let mut windows = Array3::<f32>::zeros((1, frames_per_window, num_speakers));
        for j in 0..frames_per_window {
            windows[[0, j, 0]] = 10.0;
        }

        let result = overlap_add(&windows, 1, warmup);

        // first and last warmup_frames should be 0 because hamming is zeroed there
        for i in 0..warmup {
            assert_relative_eq!(result[[i, 0]], 0.0, epsilon = 1e-6);
        }
        for i in 0..warmup {
            assert_relative_eq!(result[[frames_per_window - 1 - i, 0]], 0.0, epsilon = 1e-6);
        }

        // middle frames should be the original value
        for i in warmup..frames_per_window - warmup {
            assert_relative_eq!(result[[i, 0]], 10.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn aggregate_without_hamming_and_average_sums_windows() {
        let windows = array![[[1.0], [2.0]], [[3.0], [4.0]]];
        let result = aggregate_at(
            &windows,
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
        let result = aggregate_at(&windows, &[0, 2, 3], AggregateOptions::default());

        assert_eq!(result.nrows(), 4);
        assert_eq!(result[[0, 0]], 1.0);
        assert_eq!(result[[1, 0]], 0.0);
        assert_eq!(result[[2, 0]], 2.0);
        assert_eq!(result[[3, 0]], 3.0);
    }

    #[test]
    fn aggregate_at_respects_explicit_output_frames() {
        let windows = array![[[1.0], [2.0]], [[3.0], [4.0]]];
        let result = aggregate_at(
            &windows,
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
