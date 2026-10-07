//! The fused ResNet convolution candidate against the cuDNN fused call, on seeded
//! random tensors, so it runs on any GPU without the reference files

use super::super::candidate::{
    ConvCandidate, ConvInputs, ConvKernel, ConvLayerSpec, ConvOxide, ConvPin, ConvShape, Phases,
    PlanError,
};
use super::super::dnn::ConvPlanner;
use super::super::geometry::{Conv2d, Residual};
use super::super::{CudaError, CudaMath};
use super::runtime;

/// Synthetic shapes plan the candidate's implemented pin and require explicit
/// refusals, never production fallback
fn explicit_plan(
    runtime: &super::super::CudaRuntime,
    spec: ConvLayerSpec<'_>,
) -> Result<ConvOxide, CudaError> {
    let tier = runtime
        .load_kernels(super::super::KernelModule::Resnet)?
        .tier();
    let device = runtime.compute_capability();
    let (area, boundary, batch, math) = (
        "resnet",
        spec.name.to_owned(),
        spec.conv.batch,
        spec.conv.math,
    );
    let kernels =
        runtime.load_module(runtime.embedded_exact_request(super::super::KernelModule::Resnet)?)?;
    ConvOxide::implemented_pin(&spec)
        .and_then(|pin| ConvOxide::plan(runtime, &kernels, spec, pin))
        .map_err(|error| match error {
            PlanError::Cuda(error) => error,
            PlanError::DeviceUnsupported { reason } => CudaError::CandidateDeviceUnsupported {
                area,
                boundary,
                batch,
                math,
                tier,
                device,
                reason,
            },
            PlanError::WeightsOutOfContract { layer, fault } => {
                CudaError::CandidateWeightsOutOfContract {
                    area,
                    boundary,
                    batch,
                    math,
                    tier,
                    device,
                    layer,
                    fault,
                }
            }
            PlanError::Geometry(error) => CudaError::CandidateGeometry {
                area,
                boundary,
                batch,
                math,
                error,
            },
        })
}

/// Seeded values in `[-1, 1)` from a 64-bit LCG, so failures reproduce
fn values(seed: u64, len: usize) -> Vec<f32> {
    let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 40) as f32 / (1u64 << 23) as f32) - 1.0
        })
        .collect()
}

/// Every fused shape against `cudnnConvolutionBiasActivationForward`, with and without
/// a residual, in both math modes, at sizes that leave partial tiles in both spatial
/// directions over more than one batch item. Batch 3 runs the small-block kernels and
/// batch 40 the 256-thread ones. FP32 must match cuDNN FP32 closely. TF32 must be
/// no less accurate than cuDNN TF32 against the same-input cuDNN FP32 truth. A NaN
/// input must reach the output through the ReLU. Each plan packs the same fixed
/// weights used by the Library
#[test]
fn resnet_candidate_matches_cudnn_on_partial_tiles() -> Result<(), CudaError> {
    let Some(runtime) = runtime("resnet_candidate_matches_cudnn_on_partial_tiles") else {
        return Ok(());
    };
    let stream = runtime.stream();
    let planner = ConvPlanner::new(&runtime)?;
    let cases = [
        ("resnet.layer1.0.conv2", 32, 32, 1, [11, 70]),
        ("resnet.layer2.1.conv2", 64, 64, 1, [7, 131]),
        ("resnet.layer2.0.conv1", 32, 64, 2, [13, 133]),
    ];

    let runs = [3, 40]
        .into_iter()
        .flat_map(|batch| cases.map(|case| (batch, case)));
    for (seed, (batch, (name, cin, cout, stride, input))) in runs.enumerate() {
        let seed = seed as u64 * 16;
        let fp32 = Conv2d {
            batch,
            in_channels: cin,
            out_channels: cout,
            input,
            kernel: [3, 3],
            padding: [1, 1],
            stride: [stride; 2],
            dilation: [1, 1],
            math: CudaMath::Fp32,
        };
        let plan = planner.plan(fp32)?;
        let output_len = fp32.output_shape().iter().product();

        let weight = stream.clone_htod(&values(seed + 1, cout * cin * 9))?;
        let bias = stream.clone_htod(&values(seed + 2, cout))?;
        let mut x = values(seed + 3, batch * cin * input[0] * input[1]);
        let residual = stream.clone_htod(&values(seed + 4, output_len))?;
        let mut workspace = stream.alloc_zeros::<u8>(plan.workspace_bytes().max(1))?;
        let x_device = stream.clone_htod(&x)?;
        let z = residual.as_view();

        for add in [false, true] {
            let mut expected = stream.alloc_zeros::<f32>(output_len)?;
            let operand = if add {
                Residual::Add(&z)
            } else {
                Residual::None { scratch: &z }
            };
            plan.forward_bias_relu(
                &mut workspace.as_view_mut(),
                &x_device.as_view(),
                &weight.as_view(),
                &bias.as_view(),
                operand,
                &mut expected.as_view_mut(),
            )?;
            let expected = stream.clone_dtoh(&expected)?;
            let scale = expected
                .iter()
                .fold(1.0f32, |max, value| max.max(value.abs()));

            for math in [CudaMath::Fp32, CudaMath::Tf32] {
                let fused = explicit_plan(
                    &runtime,
                    ConvLayerSpec {
                        name,
                        conv: Conv2d { math, ..fp32 },
                        epilogue: if add {
                            super::super::candidate::Epilogue::BiasReluResidual
                        } else {
                            super::super::candidate::Epilogue::BiasRelu
                        },
                        weight: &weight,
                        bias: &bias,
                    },
                )?;
                let mut actual = stream.alloc_zeros::<f32>(output_len)?;
                fused.enqueue(
                    ConvInputs {
                        x: &x_device.as_view(),
                        residual: add.then_some(&z),
                        weight: &weight.as_view(),
                        bias: &bias.as_view(),
                    },
                    &mut actual.as_view_mut(),
                    &Phases::new(),
                    stream,
                )?;
                let actual = stream.clone_dtoh(&actual)?;
                let worst = expected
                    .iter()
                    .zip(&actual)
                    .map(|(e, a)| (e - a).abs())
                    .fold(0.0f32, f32::max);
                if math == CudaMath::Fp32 {
                    assert!(
                        worst <= 1e-5 * scale,
                        "{name} b{batch} residual={add} {math:?}: max difference {worst} at scale {scale}"
                    );
                    continue;
                }

                let tf32_plan = planner.plan(Conv2d { math, ..fp32 })?;
                let mut tf32_workspace =
                    stream.alloc_zeros::<u8>(tf32_plan.workspace_bytes().max(1))?;
                let mut library = stream.alloc_zeros::<f32>(output_len)?;
                let operand = if add {
                    Residual::Add(&z)
                } else {
                    Residual::None { scratch: &z }
                };
                tf32_plan.forward_bias_relu(
                    &mut tf32_workspace.as_view_mut(),
                    &x_device.as_view(),
                    &weight.as_view(),
                    &bias.as_view(),
                    operand,
                    &mut library.as_view_mut(),
                )?;
                let library = stream.clone_dtoh(&library)?;
                let candidate_error = truth_error(&actual, &expected);
                let library_error = truth_error(&library, &expected);
                // cuDNN may run an FP32 algorithm for a TF32 request, as for the strided
                // layer on the 4060 Ti; the FP32 kernel then only has to meet the FP32 bound
                let within = if library_error == (0.0, 0.0) {
                    candidate_error.0 <= 1e-5 * f64::from(scale)
                } else {
                    candidate_error.0 <= library_error.0 && candidate_error.1 <= library_error.1
                };
                assert!(
                    within,
                    "{name} b{batch} residual={add} TF32: candidate max-abs/L2 {candidate_error:?} exceeds Library {library_error:?} against FP32 truth"
                );

                let Some(kernel) = tensor_kernel(&runtime, cin, stride)? else {
                    continue;
                };
                let kernels = runtime.load_module(
                    runtime.embedded_exact_request(super::super::KernelModule::Resnet)?,
                )?;
                let tensor = ConvOxide::plan(
                    &runtime,
                    &kernels,
                    ConvLayerSpec {
                        name,
                        conv: Conv2d { math, ..fp32 },
                        epilogue: if add {
                            super::super::candidate::Epilogue::BiasReluResidual
                        } else {
                            super::super::candidate::Epilogue::BiasRelu
                        },
                        weight: &weight,
                        bias: &bias,
                    },
                    ConvPin::Kernel(kernel),
                )
                .map_err(|error| CudaError::Unsupported {
                    context: "tensor-core conv3x3 plan",
                    reason: error.to_string(),
                })?;
                let mut actual = stream.alloc_zeros::<f32>(output_len)?;
                tensor.enqueue(
                    ConvInputs {
                        x: &x_device.as_view(),
                        residual: add.then_some(&z),
                        weight: &weight.as_view(),
                        bias: &bias.as_view(),
                    },
                    &mut actual.as_view_mut(),
                    &Phases::new(),
                    stream,
                )?;
                let tensor_error = truth_error(&stream.clone_dtoh(&actual)?, &expected);
                eprintln!(
                    "RESNET_TC {name} b{batch} residual={add} {kernel:?}: max-abs/L2 {tensor_error:?}, Library TF32 {library_error:?}"
                );
                // one TF32 product per term, as cuDNN's TF32 path: gross mismatches only,
                // since cuDNN may run an FP32 algorithm for a TF32 request
                let floor = 2f64.powi(-10);
                assert!(
                    tensor_error.0 <= 10.0 * library_error.0 + floor * f64::from(scale)
                        && tensor_error.1 <= 10.0 * library_error.1 + floor,
                    "{name} b{batch} residual={add} {kernel:?}: max-abs/L2 {tensor_error:?} grossly exceeds Library TF32 {library_error:?}"
                );
            }
        }

        // a NaN at the first input element must reach the first output
        x[0] = f32::NAN;
        let x_device = stream.clone_htod(&x)?;
        let fused = explicit_plan(
            &runtime,
            ConvLayerSpec {
                name,
                conv: fp32,
                epilogue: super::super::candidate::Epilogue::BiasRelu,
                weight: &weight,
                bias: &bias,
            },
        )?;
        let mut actual = stream.alloc_zeros::<f32>(output_len)?;
        fused.enqueue(
            ConvInputs {
                x: &x_device.as_view(),
                residual: None,
                weight: &weight.as_view(),
                bias: &bias.as_view(),
            },
            &mut actual.as_view_mut(),
            &Phases::new(),
            stream,
        )?;
        assert!(
            stream.clone_dtoh(&actual)?[0].is_nan(),
            "{name}: ReLU must keep NaN"
        );
    }

    Ok(())
}

/// The TF32 tensor-core entry of a case where the loaded resnet module is the sm80 tier
/// or newer
fn tensor_kernel(
    runtime: &super::super::CudaRuntime,
    channels: usize,
    stride: usize,
) -> Result<Option<ConvKernel>, CudaError> {
    let tier = runtime
        .load_kernels(super::super::KernelModule::Resnet)?
        .tier();
    if tier < super::super::PtxTier::Sm80 {
        return Ok(None);
    }
    let shape = match (channels, stride) {
        (32, 1) => ConvShape::C32,
        (64, 1) => ConvShape::C64,
        _ => ConvShape::C32Stride2,
    };
    Ok(Some(shape.tensor_kernel()))
}

/// Max-abs and relative L2 errors have no tolerance floor; exact zero stays exact
fn truth_error(actual: &[f32], truth: &[f32]) -> (f64, f64) {
    assert_eq!(actual.len(), truth.len());
    let mut max_abs = 0.0f64;
    let mut squared_error = 0.0;
    let mut squared_truth = 0.0;
    for (&actual, &truth) in actual.iter().zip(truth) {
        assert!(actual.is_finite() && truth.is_finite());
        let actual = f64::from(actual);
        let truth = f64::from(truth);
        let difference = actual - truth;
        max_abs = max_abs.max(difference.abs());
        squared_error += difference * difference;
        squared_truth += truth * truth;
    }

    let relative_l2 = if squared_truth == 0.0 {
        if squared_error == 0.0 {
            0.0
        } else {
            f64::INFINITY
        }
    } else {
        (squared_error / squared_truth).sqrt()
    };
    (max_abs, relative_l2)
}

#[test]
fn tf32_truth_error_keeps_zero_exact_and_detects_each_error() {
    assert_eq!(truth_error(&[0.0], &[0.0]), (0.0, 0.0));
    assert_eq!(truth_error(&[1.0], &[0.0]), (1.0, f64::INFINITY));
    assert_eq!(truth_error(&[3.0, 0.0], &[2.0, 0.0]), (1.0, 0.5));
    assert!(std::panic::catch_unwind(|| truth_error(&[f32::NAN], &[1.0])).is_err());
}
