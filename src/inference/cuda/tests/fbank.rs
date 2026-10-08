//! Parity tests and a benchmark for the native CUDA fbank front end
//!
//! The references are the ONNX Runtime CPU tensors of `wespeaker-fbank.onnx` and
//! `wespeaker-fbank-b32.onnx` that the references task wrote. They are found through
//! `SPEAKRS_CUDA_REF`, then `/workspace/ref`, then
//! `~/Library/Caches/speakrs-cuda-ref`. Tests skip with a message when the GPU or the
//! references are missing; `SPEAKRS_REQUIRE_GPU=1` turns both skips into failures
//!
//! Besides ONNX Runtime, every result is compared with an f64 evaluation of the same
//! recipe. ONNX Runtime itself is an f32 computation, and on bins whose energy is
//! about 1e-10 of their frame's energy its log-mel values are off by up to 1.4e-3
//! from the f64 result; the GPU path has errors of the same kind but not the same
//! values, so its distance to ONNX Runtime can exceed either error alone
use std::path::Path;
use std::time::Instant;

use cudarc::driver::CudaView;
use cudarc::driver::sys::CUevent_flags;

use super::super::fbank::{FbankConstants, MelProjection};
use super::super::{
    CudaError, CudaFbank, CudaMath, CudaRuntime, FBANK_FRAMES, FBANK_MEL_BINS,
    FBANK_WINDOW_SAMPLES, SafetensorsFile,
};
use super::{reference_dir, runtime};

const FRAME_LENGTH: usize = 400;
const FRAME_SHIFT: usize = 160;
const SPECTRUM_BINS: usize = 257;
const FEATURES: usize = FBANK_FRAMES * FBANK_MEL_BINS;

/// Batch-1 reference cases: a full window, the zero-padded last window of `test.wav`
/// and the mostly zero-padded `test_short.wav`
const B1_CASES: [&str; 3] = [
    "test_first_b1",
    "test_last_partial_b1",
    "test_short_partial_b1",
];

/// Valid samples of `test_short.wav` in its zero-padded window
const SHORT_VALID_SAMPLES: usize = 52_623;

/// Largest log-mel difference to ONNX Runtime accepted in FP32
const FP32_MAX_ABS_VS_ORT: f64 = 2e-3;
/// Largest log-mel difference to the f64 evaluation accepted in FP32
const FP32_MAX_ABS_VS_EXACT: f64 = 1e-3;

const MODES: [(MelProjection, CudaMath); 4] = [
    (MelProjection::Sparse, CudaMath::Fp32),
    (MelProjection::Gemm, CudaMath::Fp32),
    (MelProjection::Sparse, CudaMath::Tf32),
    (MelProjection::Gemm, CudaMath::Tf32),
];

fn open(dir: &Path, model: &str, file: &str) -> SafetensorsFile {
    let path = dir.join(model).join(format!("{file}.safetensors"));
    SafetensorsFile::open(&path).unwrap_or_else(|error| panic!("{error}"))
}

/// Differences between a result and a reference, on log-mel values
#[derive(Debug, Clone, Copy, Default)]
struct Diff {
    max_abs: f64,
    mean_abs: f64,
    /// `|a - b| / max(|b|, 1)`: CMN values cross zero, so plain relative error is
    /// meaningless near it
    max_rel: f64,
}

impl Diff {
    fn new(actual: &[f32], expected: impl IntoIterator<Item = f64>) -> Self {
        let mut diff = Self::default();
        let mut count = 0_usize;
        for (&actual, expected) in actual.iter().zip(expected) {
            let error = (f64::from(actual) - expected).abs();
            assert!(error.is_finite(), "non-finite fbank value {actual}");
            diff.max_abs = diff.max_abs.max(error);
            diff.max_rel = diff.max_rel.max(error / expected.abs().max(1.0));
            diff.mean_abs += error;
            count += 1;
        }
        assert_eq!(count, actual.len(), "reference length");

        diff.mean_abs /= count as f64;
        diff
    }
}

impl std::fmt::Display for Diff {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "max abs {:.3e}, max rel {:.3e}, mean abs {:.3e}",
            self.max_abs, self.max_rel, self.mean_abs
        )
    }
}

/// The fbank recipe in f64 with a direct DFT, as the ground truth for both the GPU and
/// ONNX Runtime
struct ExactFbank {
    window: Vec<f64>,
    cos: Vec<f64>,
    sin: Vec<f64>,
    mel: Vec<f64>,
}

impl ExactFbank {
    /// Uses the ONNX model's own mel filters, so the comparison includes the error of
    /// the host-computed filters
    fn new(onnx_mel: &[f32]) -> Self {
        let angle = |bin: usize, sample: usize| {
            std::f64::consts::TAU * ((bin * sample) % 512) as f64 / 512.0
        };
        let table = |f: fn(f64) -> f64| -> Vec<f64> {
            (0..SPECTRUM_BINS)
                .flat_map(|bin| (0..FRAME_LENGTH).map(move |sample| f(angle(bin, sample))))
                .collect()
        };

        Self {
            window: (0..FRAME_LENGTH)
                .map(|n| 0.54 - 0.46 * (std::f64::consts::TAU * n as f64 / 399.0).cos())
                .collect(),
            cos: table(f64::cos),
            sin: table(f64::sin),
            mel: onnx_mel.iter().copied().map(f64::from).collect(),
        }
    }

    /// `[FBANK_FRAMES, FBANK_MEL_BINS]` features of one padded waveform row
    fn features(&self, row: &[f32]) -> Vec<f64> {
        let mut log_mel = vec![0.0; FEATURES];
        let mut frame = [0.0_f64; FRAME_LENGTH];
        let mut power = [0.0_f64; SPECTRUM_BINS];

        for (index, out) in log_mel
            .as_chunks_mut::<FBANK_MEL_BINS>()
            .0
            .iter_mut()
            .enumerate()
        {
            let samples = &row[index * FRAME_SHIFT..index * FRAME_SHIFT + FRAME_LENGTH];
            for (value, &sample) in frame.iter_mut().zip(samples) {
                *value = f64::from(sample) * 32_768.0;
            }

            let mean = frame.iter().sum::<f64>() / FRAME_LENGTH as f64;
            let centered: Vec<f64> = frame.iter().map(|value| value - mean).collect();
            for (j, value) in frame.iter_mut().enumerate() {
                let previous = centered[j.saturating_sub(1)];
                *value = (centered[j] - 0.97 * previous) * self.window[j];
            }

            for (bin, power) in power.iter_mut().enumerate() {
                let basis = bin * FRAME_LENGTH..(bin + 1) * FRAME_LENGTH;
                let real: f64 = frame
                    .iter()
                    .zip(&self.cos[basis.clone()])
                    .map(|(x, c)| x * c)
                    .sum();
                let imaginary: f64 = frame.iter().zip(&self.sin[basis]).map(|(x, s)| x * s).sum();
                *power = real * real + imaginary * imaginary;
            }

            for (mel, out) in out.iter_mut().enumerate() {
                let energy: f64 = power
                    .iter()
                    .enumerate()
                    .map(|(bin, power)| power * self.mel[bin * FBANK_MEL_BINS + mel])
                    .sum();
                *out = energy.max(f64::from(f32::EPSILON)).ln();
            }
        }

        for mel in 0..FBANK_MEL_BINS {
            let mean = (0..FBANK_FRAMES)
                .map(|frame| log_mel[frame * FBANK_MEL_BINS + mel])
                .sum::<f64>()
                / FBANK_FRAMES as f64;
            for frame in 0..FBANK_FRAMES {
                log_mel[frame * FBANK_MEL_BINS + mel] -= mean;
            }
        }

        log_mel
    }
}

/// Runs the GPU front end on host rows and downloads `[rows, FRAMES, MEL_BINS]`
fn gpu_features(
    runtime: &CudaRuntime,
    projection: MelProjection,
    math: CudaMath,
    rows: &[&[f32]],
) -> Result<Vec<f32>, CudaError> {
    let fbank = CudaFbank::with_projection(runtime, math, projection)?;
    let mut buffers = fbank.buffers(runtime, rows.len())?;
    let features = fbank.compute_host(runtime, rows, &mut buffers)?;
    Ok(runtime.stream().clone_dtoh(&features)?)
}

fn mode_name(projection: MelProjection, math: CudaMath) -> String {
    format!("{projection:?}/{math:?}")
}

/// The host-computed window and mel filters against the ONNX initializers; needs only
/// the reference files, no GPU
#[test]
fn constants_match_onnx_initializers() -> Result<(), CudaError> {
    let Some(dir) = reference_dir("constants_match_onnx_initializers", "wespeaker-fbank") else {
        return Ok(());
    };
    let weights = open(&dir, "wespeaker-fbank", "wespeaker-fbank");
    let constants = FbankConstants::new();

    let window = weights.read_f32("view", &[1, 1, FRAME_LENGTH])?;
    let window_error = Diff::new(constants.window(), window.iter().copied().map(f64::from));
    let mel = weights.read_f32("mel", &[SPECTRUM_BINS, FBANK_MEL_BINS])?;
    let mel_error = Diff::new(constants.mel(), mel.iter().copied().map(f64::from));
    let mel_mismatches = constants
        .mel()
        .iter()
        .zip(&mel)
        .filter(|(ours, onnx)| ours != onnx)
        .count();
    eprintln!("window vs ONNX `view`: {window_error}");
    eprintln!(
        "mel vs ONNX `mel`: {mel_error}, {mel_mismatches} of {} entries differ",
        mel.len()
    );

    assert!(window_error.max_abs <= 1e-6);
    assert!(mel_error.max_abs <= 1e-5);
    let zero_pattern_matches = constants
        .mel()
        .iter()
        .zip(&mel)
        .all(|(&ours, &onnx)| (ours == 0.0) == (onnx == 0.0));
    assert!(zero_pattern_matches, "mel filters cover different bins");
    Ok(())
}

/// Batch-1 parity for every mode on the three single-window references
#[test]
fn fbank_b1_matches_reference() -> Result<(), CudaError> {
    let test = "fbank_b1_matches_reference";
    let Some(dir) = reference_dir(test, "wespeaker-fbank") else {
        return Ok(());
    };
    let Some(runtime) = runtime(test) else {
        return Ok(());
    };
    let weights = open(&dir, "wespeaker-fbank", "wespeaker-fbank");
    let exact = ExactFbank::new(&weights.read_f32("mel", &[SPECTRUM_BINS, FBANK_MEL_BINS])?);

    for case in B1_CASES {
        let tensors = open(&dir, "wespeaker-fbank", case);
        let waveform = tensors.read_f32("input/waveform", &[1, 1, FBANK_WINDOW_SAMPLES])?;
        let onnx = tensors.read_f32("tensor/fbank", &[1, FBANK_FRAMES, FBANK_MEL_BINS])?;
        let truth = exact.features(&waveform);
        eprintln!(
            "{case}: ORT vs exact {}",
            Diff::new(&onnx, truth.iter().copied())
        );

        for (projection, math) in MODES {
            let gpu = gpu_features(&runtime, projection, math, &[&waveform])?;
            let vs_onnx = Diff::new(&gpu, onnx.iter().copied().map(f64::from));
            let vs_exact = Diff::new(&gpu, truth.iter().copied());
            let mode = mode_name(projection, math);
            eprintln!("{case} {mode}: vs ORT {vs_onnx}; vs exact {vs_exact}");

            if math == CudaMath::Fp32 {
                assert!(
                    vs_onnx.max_abs <= FP32_MAX_ABS_VS_ORT,
                    "{case} {mode} vs ORT"
                );
                assert!(
                    vs_exact.max_abs <= FP32_MAX_ABS_VS_EXACT,
                    "{case} {mode} vs exact"
                );
            }
        }
    }

    Ok(())
}

/// Batch-32 parity: 18 windows of `test.wav`, `test_short.wav`, then the first 13
/// windows again, which must give bit-identical rows
#[test]
fn fbank_b32_matches_reference() -> Result<(), CudaError> {
    let test = "fbank_b32_matches_reference";
    let Some(dir) = reference_dir(test, "wespeaker-fbank") else {
        return Ok(());
    };
    let Some(runtime) = runtime(test) else {
        return Ok(());
    };
    let weights = open(&dir, "wespeaker-fbank-b32", "wespeaker-fbank-b32");
    let exact = ExactFbank::new(&weights.read_f32("mel", &[SPECTRUM_BINS, FBANK_MEL_BINS])?);
    let tensors = open(&dir, "wespeaker-fbank-b32", "test_and_short_b32");
    let waveform = tensors.read_f32("input/waveform", &[32, 1, FBANK_WINDOW_SAMPLES])?;
    let onnx = tensors.read_f32("tensor/fbank", &[32, FBANK_FRAMES, FBANK_MEL_BINS])?;
    drop(tensors);

    let rows: Vec<&[f32]> = waveform
        .as_chunks::<FBANK_WINDOW_SAMPLES>()
        .0
        .iter()
        .map(|row| row.as_slice())
        .collect();
    // the first full window, the zero-padded last window and the short file
    let exact_rows = [0_usize, 17, 18];
    let truth: Vec<Vec<f64>> = exact_rows
        .iter()
        .map(|&row| exact.features(rows[row]))
        .collect();

    for (projection, math) in MODES {
        let gpu = gpu_features(&runtime, projection, math, &rows)?;
        let mode = mode_name(projection, math);
        let vs_onnx = Diff::new(&gpu, onnx.iter().copied().map(f64::from));
        eprintln!("b32 {mode}: vs ORT {vs_onnx}");

        for (&row, truth) in exact_rows.iter().zip(&truth) {
            let features = &gpu[row * FEATURES..(row + 1) * FEATURES];
            eprintln!(
                "b32 {mode} row {row}: vs exact {}",
                Diff::new(features, truth.iter().copied())
            );
        }

        let (first, repeated) = gpu.split_at(19 * FEATURES);
        assert_eq!(
            &first[..13 * FEATURES],
            repeated,
            "{mode}: batch position changed a row"
        );
        if math == CudaMath::Fp32 {
            assert!(vs_onnx.max_abs <= FP32_MAX_ABS_VS_ORT, "b32 {mode} vs ORT");
        }
    }

    Ok(())
}

/// Host padding and truncation, partial batches and device input all give the same
/// features as a padded host row
#[test]
fn fbank_input_paths_agree() -> Result<(), CudaError> {
    let test = "fbank_input_paths_agree";
    let Some(dir) = reference_dir(test, "wespeaker-fbank") else {
        return Ok(());
    };
    let Some(runtime) = runtime(test) else {
        return Ok(());
    };
    let tensors = open(&dir, "wespeaker-fbank", "test_short_partial_b1");
    let padded = tensors.read_f32("input/waveform", &[1, 1, FBANK_WINDOW_SAMPLES])?;
    assert!(
        padded[SHORT_VALID_SAMPLES..]
            .iter()
            .all(|&sample| sample == 0.0)
    );

    let stream = runtime.stream();
    let fbank = CudaFbank::new(&runtime, CudaMath::Fp32)?;
    let mut buffers = fbank.buffers(&runtime, 4)?;
    let download = |view: CudaView<'_, f32>| stream.clone_dtoh(&view);

    let expected = download(fbank.compute_host(&runtime, &[&padded], &mut buffers)?)?;

    let short =
        download(fbank.compute_host(&runtime, &[&padded[..SHORT_VALID_SAMPLES]], &mut buffers)?)?;
    assert_eq!(short, expected, "zero padding a short row");

    let mut long = padded.clone();
    long.extend(std::iter::repeat_n(0.5, 1_000));
    let truncated = download(fbank.compute_host(&runtime, &[&long, &padded], &mut buffers)?)?;
    assert_eq!(&truncated[..FEATURES], expected, "truncating a long row");
    assert_eq!(
        &truncated[FEATURES..],
        expected,
        "second row of a partial batch"
    );

    let device = stream.clone_htod(&padded)?;
    let from_device = download(fbank.compute_device(&runtime, &device.slice(..), &mut buffers)?)?;
    assert_eq!(from_device, expected, "device input");

    let too_many = vec![padded.as_slice(); 5];
    assert!(matches!(
        fbank.compute_host(&runtime, &too_many, &mut buffers),
        Err(CudaError::BufferLength { .. })
    ));
    Ok(())
}

/// Batch-32 throughput, separating the upload, the GPU computation and the download
///
/// The upload is GPU-timed from an idle stream, so it includes filling the pinned
/// staging buffer on the host, which is also reported on its own
///
/// Run on the GPU box with the benchmark lock held:
/// `flock /workspace/gpu-bench.lock cargo test --release --features cuda --lib fbank_b32_benchmark -- --ignored --nocapture`
#[test]
#[ignore = "benchmark; run explicitly with the GPU lock held"]
fn fbank_b32_benchmark() -> Result<(), CudaError> {
    const WARMUP: usize = 10;
    const RUNS: usize = 50;
    let test = "fbank_b32_benchmark";
    let Some(dir) = reference_dir(test, "wespeaker-fbank") else {
        return Ok(());
    };
    let Some(runtime) = runtime(test) else {
        return Ok(());
    };
    let tensors = open(&dir, "wespeaker-fbank-b32", "test_and_short_b32");
    let waveform = tensors.read_f32("input/waveform", &[32, 1, FBANK_WINDOW_SAMPLES])?;
    drop(tensors);
    let rows: Vec<&[f32]> = waveform
        .as_chunks::<FBANK_WINDOW_SAMPLES>()
        .0
        .iter()
        .map(|row| row.as_slice())
        .collect();

    let context = runtime.context();
    let stream = runtime.stream();
    let timing = || context.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT));
    let median = |values: &mut Vec<f64>| {
        values.sort_by(f64::total_cmp);
        values[values.len() / 2]
    };
    // SAFETY: the download target is fully overwritten by each copy before it is read
    let mut host_out = unsafe { context.alloc_pinned_with_flags::<f32>(32 * FEATURES, 0) }?;

    for (projection, math) in MODES {
        let fbank = CudaFbank::with_projection(&runtime, math, projection)?;
        let mut buffers = fbank.buffers(&runtime, 32)?;
        let mut stage_ms = Vec::with_capacity(RUNS);
        let mut upload_ms = Vec::with_capacity(RUNS);
        let mut compute_ms = Vec::with_capacity(RUNS);
        let mut download_ms = Vec::with_capacity(RUNS);
        let mut wall_ms = Vec::with_capacity(RUNS);

        for run in 0..WARMUP + RUNS {
            let started = Instant::now();
            let start = timing()?;
            start.record(stream)?;
            let staging = Instant::now();
            buffers.upload(&runtime, &rows)?;
            let staged = staging.elapsed().as_secs_f64() * 1e3;
            let uploaded = timing()?;
            uploaded.record(stream)?;
            let features = fbank.compute_uploaded(&runtime, &mut buffers)?;
            let computed = timing()?;
            computed.record(stream)?;
            stream.memcpy_dtoh(&features, &mut host_out)?;
            let downloaded = timing()?;
            downloaded.record(stream)?;
            stream.synchronize()?;
            let wall = started.elapsed().as_secs_f64() * 1e3;

            if run >= WARMUP {
                stage_ms.push(staged);
                upload_ms.push(f64::from(start.elapsed_ms(&uploaded)?));
                compute_ms.push(f64::from(uploaded.elapsed_ms(&computed)?));
                download_ms.push(f64::from(computed.elapsed_ms(&downloaded)?));
                wall_ms.push(wall);
            }
        }

        // back-to-back batches on device-resident input: steady-state GPU throughput
        let start = timing()?;
        start.record(stream)?;
        for _ in 0..RUNS {
            fbank.compute_uploaded(&runtime, &mut buffers)?;
        }
        let end = timing()?;
        end.record(stream)?;
        let streamed = f64::from(start.elapsed_ms(&end)?) / RUNS as f64;

        eprintln!(
            "bench b32 {} ({}): median upload {:.3} ms (host staging {:.3} ms of it), \
             compute {:.3} ms, download {:.3} ms, end-to-end wall {:.3} ms; \
             back-to-back compute {:.3} ms/batch ({:.0} windows/s)",
            mode_name(projection, math),
            fbank.tier(),
            median(&mut upload_ms),
            median(&mut stage_ms),
            median(&mut compute_ms),
            median(&mut download_ms),
            median(&mut wall_ms),
            streamed,
            32.0 * 1e3 / streamed,
        );
    }

    Ok(())
}
