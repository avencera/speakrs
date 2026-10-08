//! GPU smoke tests for the native CUDA runtime
//!
//! Each test skips with a message when the machine has no NVIDIA driver or GPU. Set
//! `SPEAKRS_REQUIRE_GPU=1` to turn a skip into a failure, so a GPU host proves that
//! every test really ran
use std::path::PathBuf;

use super::super::probe::ProbeKernels;
use super::super::{
    ComputeCapability, Conv2d, ConvPlanner, CudaError, CudaMath, CudaRuntime, DeviceTensor,
    PtxTier, SafetensorsFile, Sgemm,
};
use super::{runtime, runtime_with_tier};

/// Deterministic values in [-1, 1) with full FP32 mantissas, so a TF32 path (10-bit
/// mantissa) would show errors around 1e-3 instead of FP32's 1e-7
fn values(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

/// Largest absolute difference and the reference magnitude it is judged against
fn max_error(actual: &[f32], expected: &[f64]) -> (f64, f64) {
    assert_eq!(actual.len(), expected.len());
    let error = actual
        .iter()
        .zip(expected)
        .map(|(&a, &e)| (f64::from(a) - e).abs())
        .fold(0.0, f64::max);
    let scale = expected.iter().map(|e| e.abs()).fold(1.0, f64::max);
    (error, scale)
}

#[test]
fn probe_kernel_matches_cpu() -> Result<(), CudaError> {
    let Some(runtime) = runtime("probe_kernel_matches_cpu") else {
        return Ok(());
    };
    let probe = ProbeKernels::load(&runtime)?;
    let (actual, expected) = run_probe(&runtime, &probe)?;

    let (error, _) = max_error(&actual, &expected);
    eprintln!(
        "probe_scale_add ({}): {} elements, max abs error {error:e}",
        probe.tier(),
        actual.len()
    );
    assert_eq!(error, 0.0);
    Ok(())
}

/// Runs the probe on a fixed input; returns the GPU output and the CPU reference
fn run_probe(
    runtime: &CudaRuntime,
    probe: &ProbeKernels,
) -> Result<(Vec<f32>, Vec<f64>), CudaError> {
    let stream = runtime.stream();

    // not a multiple of the block size, so the tail guard runs
    let len = 10_007;
    let alpha = 1.75_f32;
    let x = values(len, 1);
    let y = values(len, 2);
    let x_dev = DeviceTensor::upload(stream, &x, &[len])?;
    let y_dev = DeviceTensor::upload(stream, &y, &[len])?;
    let mut out = DeviceTensor::<f32>::zeros(stream, &[len])?;

    probe.scale_add(runtime, alpha, x_dev.data(), y_dev.data(), out.data_mut())?;
    let actual = out.download(stream)?;

    // cuda-oxide contracts `alpha * x + y` into one FMA, like nvcc's default
    let expected = x
        .iter()
        .zip(&y)
        .map(|(&x, &y)| f64::from(alpha.mul_add(x, y)))
        .collect();
    Ok((actual, expected))
}

/// The probe ships an sm80 variant behind `cuda-sm80`: automatic selection must load
/// the highest compiled-in tier the card supports, forcing the baseline must load
/// sm75, and both must give identical results
#[test]
fn probe_tier_dispatch_matches_forced_baseline() -> Result<(), CudaError> {
    let test = "probe_tier_dispatch_matches_forced_baseline";
    let Some(auto) = runtime_with_tier(test, None) else {
        return Ok(());
    };
    let Some(forced) = runtime_with_tier(test, Some(PtxTier::Sm75)) else {
        return Ok(());
    };

    let capability = auto.compute_capability();
    let expected_tier = if cfg!(feature = "cuda-sm80") && capability >= ComputeCapability::new(8, 0)
    {
        PtxTier::Sm80
    } else {
        PtxTier::Sm75
    };

    let auto_probe = ProbeKernels::load(&auto)?;
    let forced_probe = ProbeKernels::load(&forced)?;
    eprintln!(
        "compute capability {capability}: automatic tier {} (limit {}), forced tier {}",
        auto_probe.tier(),
        auto.ptx_tier(),
        forced_probe.tier()
    );
    assert_eq!(auto_probe.tier(), expected_tier);
    assert_eq!(forced_probe.tier(), PtxTier::Sm75);

    let (auto_out, expected) = run_probe(&auto, &auto_probe)?;
    let (forced_out, _) = run_probe(&forced, &forced_probe)?;
    assert_eq!(auto_out, forced_out, "tiers disagree");
    assert_eq!(max_error(&auto_out, &expected).0, 0.0);
    Ok(())
}

/// Tier selection is pure, so it runs without a GPU
#[test]
fn ptx_tier_selection_follows_capability_and_build() {
    let turing = ComputeCapability::new(7, 5);
    let ampere = ComputeCapability::new(8, 6);
    let blackwell = ComputeCapability::new(12, 0);

    assert!(matches!(
        PtxTier::select(ComputeCapability::new(7, 0), None),
        Err(CudaError::UnsupportedDevice { .. })
    ));
    assert_eq!(PtxTier::select(turing, None).ok(), Some(PtxTier::Sm75));
    assert_eq!(
        PtxTier::select(blackwell, Some(PtxTier::Sm75)).ok(),
        Some(PtxTier::Sm75)
    );

    let automatic = PtxTier::select(ampere, None).ok();
    if cfg!(feature = "cuda-sm80") {
        assert_eq!(automatic, Some(PtxTier::Sm80));
        assert!(matches!(
            PtxTier::select(turing, Some(PtxTier::Sm80)),
            Err(CudaError::PtxTierAboveDevice { .. })
        ));
    } else {
        assert_eq!(automatic, Some(PtxTier::Sm75));
        assert!(matches!(
            PtxTier::select(blackwell, Some(PtxTier::Sm80)),
            Err(CudaError::TierNotCompiledIn {
                tier: PtxTier::Sm80,
                device,
                feature: "cuda-sm80",
            }) if device == blackwell
        ));
    }

    assert_eq!("sm80".parse::<PtxTier>().ok(), Some(PtxTier::Sm80));
    assert!(matches!(
        "sm_80".parse::<PtxTier>(),
        Err(CudaError::InvalidPtxTier { .. })
    ));
}

#[test]
fn sgemm_matches_cpu_in_both_math_modes() -> Result<(), CudaError> {
    let Some(runtime) = runtime("sgemm_matches_cpu_in_both_math_modes") else {
        return Ok(());
    };
    let stream = runtime.stream();
    let (m, n, k) = (67, 45, 129);
    let transposes = [(false, false), (false, true), (true, false), (true, true)];

    for (math, (a_transposed, b_transposed)) in [CudaMath::Fp32, CudaMath::Tf32]
        .into_iter()
        .flat_map(|math| transposes.map(|transpose| (math, transpose)))
    {
        let a = values(m * k, 3);
        let b = values(k * n, 4);
        let c0 = values(m * n, 5);
        let spec = Sgemm {
            a_transposed,
            b_transposed,
            alpha: 0.5,
            beta: -2.0,
            math,
            ..Sgemm::new(m, n, k)
        };

        let a_at = |i: usize, p: usize| {
            if a_transposed {
                a[p * m + i]
            } else {
                a[i * k + p]
            }
        };
        let b_at = |p: usize, j: usize| {
            if b_transposed {
                b[j * k + p]
            } else {
                b[p * n + j]
            }
        };
        let expected: Vec<f64> = (0..m * n)
            .map(|index| {
                let (i, j) = (index / n, index % n);
                let dot: f64 = (0..k)
                    .map(|p| f64::from(a_at(i, p)) * f64::from(b_at(p, j)))
                    .sum();
                f64::from(spec.alpha) * dot + f64::from(spec.beta) * f64::from(c0[index])
            })
            .collect();

        let a_dev = DeviceTensor::upload(stream, &a, &[a.len()])?;
        let b_dev = DeviceTensor::upload(stream, &b, &[b.len()])?;
        let mut c_dev = DeviceTensor::upload(stream, &c0, &[m, n])?;
        runtime.prepare_library(super::super::CudaLibrary::Cublas)?;
        runtime.sgemm(spec, a_dev.data(), b_dev.data(), c_dev.data_mut())?;
        let actual = c_dev.download(stream)?;

        let (error, scale) = max_error(&actual, &expected);
        let relative = error / scale;
        eprintln!(
            "sgemm {math:?} {m}x{n}x{k} a_t={a_transposed} b_t={b_transposed}: max abs error {error:e}, relative {relative:e}"
        );
        assert!(
            relative < tolerance(math),
            "sgemm {math:?} relative error {relative:e} too large"
        );
    }

    Ok(())
}

/// Relative error bound per math mode: FP32 accumulation over these sizes stays near
/// 1e-6, while TF32's 10-bit mantissa gives errors near 1e-3
fn tolerance(math: CudaMath) -> f64 {
    match math {
        CudaMath::Tf32 => 1e-2,
        _ => 1e-5,
    }
}

#[test]
fn conv2d_matches_cpu_in_both_math_modes() -> Result<(), CudaError> {
    let Some(runtime) = runtime("conv2d_matches_cpu_in_both_math_modes") else {
        return Ok(());
    };
    let stream = runtime.stream();

    let cases = [
        Conv2d {
            batch: 2,
            in_channels: 3,
            out_channels: 8,
            input: [17, 23],
            kernel: [3, 3],
            padding: [1, 1],
            stride: [1, 1],
            dilation: [1, 1],
            math: CudaMath::Fp32,
        },
        Conv2d {
            batch: 3,
            in_channels: 16,
            out_channels: 32,
            input: [20, 15],
            kernel: [3, 3],
            padding: [1, 2],
            stride: [2, 1],
            dilation: [1, 2],
            math: CudaMath::Fp32,
        },
        // a ResNet-sized layer, large enough that cuDNN picks a tensor-core kernel
        // when TF32 is allowed
        Conv2d {
            batch: 2,
            in_channels: 32,
            out_channels: 32,
            input: [32, 48],
            kernel: [3, 3],
            padding: [1, 1],
            stride: [1, 1],
            dilation: [1, 1],
            math: CudaMath::Fp32,
        },
    ];

    let planner = ConvPlanner::new(&runtime)?;
    let modes = [CudaMath::Fp32, CudaMath::Tf32];
    for spec in modes
        .into_iter()
        .flat_map(|math| cases.map(|case| Conv2d { math, ..case }))
    {
        let x = values(spec.input_shape().iter().product(), 6);
        let w = values(spec.filter_shape().iter().product(), 7);
        let expected = conv2d_reference(&spec, &x, &w);

        let plan = planner.plan(spec)?;
        let x_dev = DeviceTensor::upload(stream, &x, &spec.input_shape())?;
        let w_dev = DeviceTensor::upload(stream, &w, &spec.filter_shape())?;
        let mut y_dev = DeviceTensor::<f32>::zeros(stream, &spec.output_shape())?;
        let mut workspace = stream.alloc_zeros::<u8>(plan.workspace_bytes().max(1))?;
        plan.forward(
            &mut workspace.as_view_mut(),
            &x_dev.data().as_view(),
            &w_dev.data().as_view(),
            &mut y_dev.data_mut().as_view_mut(),
        )?;
        let actual = y_dev.download(stream)?;

        let (error, scale) = max_error(&actual, &expected);
        let relative = error / scale;
        eprintln!(
            "conv2d {:?} {:?} -> {:?}: max abs error {error:e}, relative {relative:e}",
            spec.math,
            spec.input_shape(),
            spec.output_shape()
        );
        assert!(
            relative < tolerance(spec.math),
            "conv2d {:?} relative error {relative:e} too large",
            spec.math
        );
    }

    Ok(())
}

/// Direct NCHW cross-correlation in f64
fn conv2d_reference(spec: &Conv2d, x: &[f32], w: &[f32]) -> Vec<f64> {
    let [n, c, h, wd] = spec.input_shape();
    let [k, _, r, s] = spec.filter_shape();
    let [_, _, p, q] = spec.output_shape();
    let mut y = vec![0.0; n * k * p * q];

    for (index, out) in y.iter_mut().enumerate() {
        let (b, rest) = (index / (k * p * q), index % (k * p * q));
        let (o, rest) = (rest / (p * q), rest % (p * q));
        let (oy, ox) = (rest / q, rest % q);
        for ci in 0..c {
            for fy in 0..r {
                for fx in 0..s {
                    let iy = (oy * spec.stride[0] + fy * spec.dilation[0]) as isize
                        - spec.padding[0] as isize;
                    let ix = (ox * spec.stride[1] + fx * spec.dilation[1]) as isize
                        - spec.padding[1] as isize;
                    if iy < 0 || ix < 0 || iy >= h as isize || ix >= wd as isize {
                        continue;
                    }

                    let xv = x[((b * c + ci) * h + iy as usize) * wd + ix as usize];
                    let wv = w[((o * c + ci) * r + fy) * s + fx];
                    *out += f64::from(xv) * f64::from(wv);
                }
            }
        }
    }

    y
}

#[test]
fn safetensors_upload_checks_shape_and_dtype() -> Result<(), CudaError> {
    let Some(runtime) = runtime("safetensors_upload_checks_shape_and_dtype") else {
        return Ok(());
    };

    let weight = values(6, 8);
    let path = write_safetensors("upload", &weight);
    let file = SafetensorsFile::open(&path)?;
    assert_eq!(file.names(), ["conv.weight", "steps"]);
    assert_eq!(file.shape("conv.weight"), Some(&[2, 3][..]));

    let tensor = file.upload(&runtime, "conv.weight", &[2, 3])?;
    assert_eq!(tensor.shape(), [2, 3]);
    assert_eq!(tensor.download(runtime.stream())?, weight);

    assert!(matches!(
        file.upload(&runtime, "conv.weight", &[3, 2]),
        Err(CudaError::TensorShape { .. })
    ));
    assert!(matches!(
        file.upload(&runtime, "steps", &[1]),
        Err(CudaError::TensorDtype { .. })
    ));
    assert!(matches!(
        file.upload(&runtime, "missing", &[1]),
        Err(CudaError::MissingTensor { .. })
    ));

    std::fs::remove_file(path).ok();
    Ok(())
}

/// A two-tensor safetensors file: `conv.weight` (F32, 2x3) and `steps` (I64, 1)
fn write_safetensors(name: &str, weight: &[f32]) -> PathBuf {
    let header = r#"{"conv.weight":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]},"steps":{"dtype":"I64","shape":[1],"data_offsets":[24,32]}}"#;
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(header.as_bytes());
    for value in weight {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes.extend_from_slice(&7_i64.to_le_bytes());

    let path = std::env::temp_dir().join(format!(
        "speakrs-cuda-runtime-{name}-{}.safetensors",
        std::process::id()
    ));
    std::fs::write(&path, bytes).expect("write safetensors fixture");
    path
}

#[test]
fn cooperative_capacity_is_occupancy_times_sms() -> Result<(), CudaError> {
    let Some(runtime) = runtime("cooperative_capacity_is_occupancy_times_sms") else {
        return Ok(());
    };
    let kernels = runtime.load_kernels(super::super::KernelModule::Segmentation)?;
    let function = kernels.function("segmentation_bias_leaky")?;
    let sms = runtime.multiprocessor_count()?;
    let per_sm = function.occupancy_max_active_blocks_per_multiprocessor(256, 0, None)?;
    let capacity = runtime.cooperative_capacity(&function, 256, 0)?;
    eprintln!("cooperative capacity: {capacity} blocks = {per_sm} per SM x {sms} SMs");

    assert!(runtime.supports_cooperative_launch()?);
    assert!(capacity > 0);
    assert_eq!(capacity, per_sm as usize * sms);
    Ok(())
}

#[test]
fn concurrent_cooperative_capacity_fits_within_single_grid_capacity() -> Result<(), CudaError> {
    let Some(runtime) = runtime("concurrent_cooperative_capacity_fits_within_single_grid_capacity")
    else {
        return Ok(());
    };
    let kernels = runtime.load_kernels(super::super::KernelModule::Segmentation)?;
    let function = kernels.function("segmentation_bias_leaky")?;
    let capacity = runtime.cooperative_capacity(&function, 256, 0)?;
    let concurrent = runtime.concurrent_cooperative_capacity(&function, 256, 0)?;
    eprintln!("concurrent cooperative capacity: {concurrent:?} of {capacity} blocks");

    // an SM-count affinity can only lower the joint budget; without one, as on the
    // qualification box, it equals the single-grid capacity
    let concurrent = concurrent.expect("primary context reports its SM limit");
    assert!(concurrent > 0 && concurrent <= capacity);
    Ok(())
}
