use ndarray::{Array1, Array2, Array3, s};

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
    let num_windows = windows.shape()[0];
    let frames_per_window = windows.shape()[1];
    let num_speakers = windows.shape()[2];

    let total_frames = (num_windows - 1) * step_frames + frames_per_window;

    let mut hamming = hamming_window(frames_per_window);
    // zero out warmup region at edges so boundary frames contribute nothing
    for i in 0..warmup_frames.min(frames_per_window) {
        hamming[i] = 0.0;
    }
    for i in 0..warmup_frames.min(frames_per_window) {
        hamming[frames_per_window - 1 - i] = 0.0;
    }

    let mut numerator = Array2::<f32>::zeros((total_frames, num_speakers));
    let mut denominator = Array1::<f32>::zeros(total_frames);

    for i in 0..num_windows {
        for j in 0..frames_per_window {
            let out_frame = i * step_frames + j;
            let w = hamming[j];
            denominator[out_frame] += w;

            let window_slice = windows.slice(s![i, j, ..]);
            for s in 0..num_speakers {
                numerator[[out_frame, s]] += w * window_slice[s];
            }
        }
    }

    for frame in 0..total_frames {
        if denominator[frame] > 0.0 {
            for s in 0..num_speakers {
                numerator[[frame, s]] /= denominator[frame];
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
        let total = (2 - 1) * step + frames_per_window;
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
}
