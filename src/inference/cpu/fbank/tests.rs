use ndarray::{Array2, s};
use rustfft::{FftPlanner, num_complex::Complex64};

use super::{
    CpuFbank, ENERGY_FLOOR, FBANK_FRAME_LENGTH, FBANK_FRAME_SHIFT, FBANK_FRAMES, FBANK_MEL_BINS,
    FBANK_WINDOW_SAMPLES, FFT_SIZE, WAVEFORM_SCALE, mel_filters,
};

impl CpuFbank {
    pub(crate) fn compute(
        &self,
        audio: &[f32],
        workspace: &mut super::FbankWorkspace,
    ) -> Result<Array2<f32>, crate::inference::InferenceError> {
        let mut output = Array2::zeros((FBANK_FRAMES, FBANK_MEL_BINS));
        self.compute_into(audio, output.view_mut(), workspace)?;
        Ok(output)
    }
}

#[test]
fn zero_waveform_and_reused_scratch_preserve_actual_features() {
    let fbank = CpuFbank::new();
    let mut workspace = fbank.workspace();
    let pointer = workspace.signal.fft.as_ptr();
    let mut audio = vec![0.0; 400];
    audio[177] = 0.5;
    let impulse = fbank.compute(&audio, &mut workspace).unwrap();
    assert!(impulse.iter().all(|value| value.is_finite()));
    assert!(impulse.iter().any(|value| value.abs() > 1.0));
    let zero = fbank.compute(&[], &mut workspace).unwrap();
    assert_eq!(zero, Array2::<f32>::zeros((998, 80)));
    let mut storage = Array2::from_elem((998, 160), 99.0);
    fbank
        .compute_into(&audio, storage.slice_mut(s![.., ..;2]), &mut workspace)
        .unwrap();
    assert_eq!(storage.slice(s![.., ..;2]), impulse);
    assert!(
        storage
            .slice(s![.., 1..;2])
            .iter()
            .all(|value| *value == 99.0)
    );
    assert_eq!(workspace.signal.fft.as_ptr(), pointer);
    let mut bad = Array2::from_elem((1, 80), 17.0);
    assert!(
        fbank
            .compute_into(&audio, bad.view_mut(), &mut workspace)
            .is_err()
    );
    assert_eq!(bad, Array2::from_elem((1, 80), 17.0));
    audio.resize(FBANK_WINDOW_SAMPLES + 20, 0.25);
    let truncated = fbank
        .compute(&audio[..FBANK_WINDOW_SAMPLES], &mut workspace)
        .unwrap();
    assert_eq!(fbank.compute(&audio, &mut workspace).unwrap(), truncated);
}

#[test]
fn planned_fft_has_unnormalized_scale_and_forward_bin_order() {
    let fbank = CpuFbank::new();
    let mut workspace = fbank.workspace();
    workspace.signal.fft[1] = Complex64::new(1.0, 0.0);
    fbank
        .0
        .fft
        .process_with_scratch(&mut workspace.signal.fft, &mut workspace.signal.scratch);
    for (bin, value) in workspace.signal.fft.iter().enumerate() {
        let angle = -std::f64::consts::TAU * bin as f64 / FFT_SIZE as f64;
        assert!((value.re - angle.cos()).abs() < 2e-6);
        assert!((value.im - angle.sin()).abs() < 2e-6);
        assert!((value.norm_sqr() - 1.0).abs() < 2e-6);
    }
    workspace.signal.fft.fill(Complex64::new(2.0, 0.0));
    fbank
        .0
        .fft
        .process_with_scratch(&mut workspace.signal.fft, &mut workspace.signal.scratch);
    assert_eq!(workspace.signal.fft[0], Complex64::new(1024.0, 0.0));
    assert!(
        workspace.signal.fft[1..]
            .iter()
            .all(|value| value.norm_sqr() < 1e-10)
    );
}

fn f64_reference(audio: &[f32]) -> Array2<f32> {
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(FFT_SIZE);
    let window: Vec<f64> = (0..FBANK_FRAME_LENGTH)
        .map(|index| 0.54 - 0.46 * (std::f64::consts::TAU * index as f64 / 399.0).cos())
        .collect();
    let mel = mel_filters();
    let mut spectrum = vec![Complex64::default(); FFT_SIZE];
    let mut scratch = vec![Complex64::default(); fft.get_inplace_scratch_len()];
    let mut output = Array2::<f64>::zeros((FBANK_FRAMES, FBANK_MEL_BINS));
    for frame in 0..FBANK_FRAMES {
        let samples: Vec<f64> = (0..FBANK_FRAME_LENGTH)
            .map(|index| {
                f64::from(
                    audio
                        .get(frame * FBANK_FRAME_SHIFT + index)
                        .copied()
                        .unwrap_or(0.0),
                ) * f64::from(WAVEFORM_SCALE)
            })
            .collect();
        let mean = samples.iter().sum::<f64>() / FBANK_FRAME_LENGTH as f64;
        for index in 0..FBANK_FRAME_LENGTH {
            let current = samples[index] - mean;
            let previous = samples[index.saturating_sub(1)] - mean;
            spectrum[index] = Complex64::new((current - 0.97 * previous) * window[index], 0.0);
        }
        spectrum[FBANK_FRAME_LENGTH..].fill(Complex64::default());
        fft.process_with_scratch(&mut spectrum, &mut scratch);
        for filter in 0..FBANK_MEL_BINS {
            let energy: f64 = spectrum[..257]
                .iter()
                .enumerate()
                .map(|(bin, value)| {
                    value.norm_sqr() * f64::from(mel[bin * FBANK_MEL_BINS + filter])
                })
                .sum();
            output[[frame, filter]] = energy.max(f64::from(ENERGY_FLOOR)).ln();
        }
    }
    for mut column in output.columns_mut() {
        let mean = column.sum() / FBANK_FRAMES as f64;
        column.mapv_inplace(|value| value - mean);
    }
    output.mapv(|value| value as f32)
}

fn max_error(actual: &Array2<f32>, expected: &Array2<f32>) -> f32 {
    assert_eq!(actual.dim(), expected.dim());
    assert!(actual.iter().all(|value| value.is_finite()));
    actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max)
}

#[test]
fn full_partial_short_waveforms_match_independent_f64_recipe() {
    let audio: Vec<f32> = (0..FBANK_WINDOW_SAMPLES)
        .map(|index| {
            let noise = (index as u32).wrapping_mul(2_654_435_761) >> 8;
            let noise = noise as f32 / 16_777_216.0 - 0.5;
            let tone = (std::f32::consts::TAU * 440.0 * index as f32 / 16_000.0).sin();
            0.03 * noise + 0.25 * tone
        })
        .collect();
    let fbank = CpuFbank::new();
    let mut workspace = fbank.workspace();
    for samples in [160_000, 144_247, 52_623] {
        let input = &audio[..samples];
        let actual = fbank.compute(input, &mut workspace).unwrap();
        let expected = f64_reference(input);
        let error = max_error(&actual, &expected);
        assert!(error <= 1e-3, "samples={samples} error={error}");
        assert!(actual.iter().any(|value| value.abs() > 1.0));
    }
}
