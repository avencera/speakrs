//! GPU parity tests for the native CUDA ResNet34 multi-mask embedding
//!
//! The references are ONNX Runtime CPU outputs and intermediates of
//! `wespeaker-multimask-tail.onnx` and its `-b32` variant, written by the CUDA
//! references task. They are read from `SPEAKRS_CUDA_REF`, else `/workspace/ref`,
//! else `~/Library/Caches/speakrs-cuda-ref`
//!
//! Each test skips with a message when the machine has no NVIDIA GPU or no
//! references. Set `SPEAKRS_REQUIRE_GPU=1` to turn a skip into a failure
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::time::Instant;

use super::super::embedding::EmbeddingTap;
use super::super::{
    CudaError, CudaMath, CudaRuntime, EMBEDDING_DIM, EmbeddingBatch, ResNetEmbedding,
    SafetensorsFile,
};
use super::{reference_dir, runtime};

const B1_CASES: [&str; 3] = [
    "test_first_b1",
    "test_last_partial_b1",
    "test_short_partial_b1",
];
const B32_CASE: &str = "test_and_short_b32";
const B1_MODEL: &str = "wespeaker-multimask-tail";
const B32_MODEL: &str = "wespeaker-multimask-tail-b32";

/// Minimum per-embedding cosine similarity against the FP32 reference
const FP32_MIN_COSINE: f64 = 0.99999;

/// TF32 only has to stay recognisably the same embedding here; whether its drift
/// is acceptable is decided by a DER run, not by this test
const TF32_MIN_COSINE: f64 = 0.999;

/// The ONNX tensor names of the three shortcut convolution outputs, in order
const SHORTCUT_TENSORS: [&str; 3] = ["getitem_27", "getitem_54", "getitem_93"];

/// A runtime on device 0 plus the reference directory, or `None` to skip
fn setup(test: &str) -> Option<(CudaRuntime, PathBuf)> {
    let runtime = runtime(test)?;
    let root = reference_dir(test, B1_MODEL)?;
    Some((runtime, root))
}

/// A reference safetensors file read one tensor at a time; the b32 file is 13 GB
struct ReferenceFile {
    file: File,
    data_start: u64,
    /// name to (dtype, shape, start, end)
    tensors: BTreeMap<String, (String, Vec<usize>, u64, u64)>,
}

impl ReferenceFile {
    fn open(path: &Path) -> Self {
        let mut file =
            File::open(path).unwrap_or_else(|error| panic!("open {}: {error}", path.display()));
        let mut len = [0; 8];
        file.read_exact(&mut len)
            .expect("safetensors header length");
        let header_len = u64::from_le_bytes(len);
        let mut header = vec![0; usize::try_from(header_len).expect("header length")];
        file.read_exact(&mut header).expect("safetensors header");
        let header: serde_json::Map<String, serde_json::Value> =
            serde_json::from_slice(&header).expect("safetensors header json");

        let tensors = header
            .into_iter()
            .filter(|(name, _)| name != "__metadata__")
            .map(|(name, info)| {
                let dtype = info["dtype"].as_str().expect("dtype").to_string();
                let shape = info["shape"]
                    .as_array()
                    .expect("shape")
                    .iter()
                    .map(|dim| dim.as_u64().expect("dim") as usize)
                    .collect();
                let offsets = info["data_offsets"].as_array().expect("offsets");
                let start = offsets[0].as_u64().expect("start");
                let end = offsets[1].as_u64().expect("end");
                (name, (dtype, shape, start, end))
            })
            .collect();

        Self {
            file,
            data_start: 8 + header_len,
            tensors,
        }
    }

    fn has(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    fn read(&mut self, name: &str) -> (Vec<f32>, Vec<usize>) {
        let (dtype, shape, start, end) = self
            .tensors
            .get(name)
            .unwrap_or_else(|| panic!("no reference tensor {name}"))
            .clone();
        assert_eq!(dtype, "F32", "reference tensor {name}");
        let mut bytes = vec![0; usize::try_from(end - start).expect("tensor size")];
        self.file
            .seek(SeekFrom::Start(self.data_start + start))
            .expect("seek reference");
        self.file.read_exact(&mut bytes).expect("read reference");
        let (chunks, _) = bytes.as_chunks::<4>();
        (
            chunks.iter().copied().map(f32::from_le_bytes).collect(),
            shape,
        )
    }
}

struct Case {
    fbank: Vec<f32>,
    masks: Vec<f32>,
    expected: Vec<f32>,
    chunks: usize,
}

fn load_case(root: &Path, model: &str, case: &str) -> Case {
    let mut file = ReferenceFile::open(&root.join(model).join(format!("{case}.safetensors")));
    let (fbank, fbank_shape) = file.read("input/fbank");
    let (masks, _) = file.read("input/masks");
    let (expected, _) = file.read("tensor/output");
    Case {
        fbank,
        masks,
        expected,
        chunks: fbank_shape[0],
    }
}

fn load_model(
    runtime: &CudaRuntime,
    root: &Path,
    model: &str,
    math: CudaMath,
) -> Result<ResNetEmbedding, CudaError> {
    let weights = SafetensorsFile::open(root.join(model).join(format!("{model}.safetensors")))?;
    ResNetEmbedding::load(runtime, &weights, math)
}

/// Worst per-row cosine similarity and the largest absolute difference
#[derive(Debug, Clone, Copy)]
struct Parity {
    min_cosine: f64,
    max_abs: f64,
}

fn parity(actual: &[f32], expected: &[f32], dim: usize) -> Parity {
    assert_eq!(actual.len(), expected.len());
    let min_cosine = actual
        .chunks(dim)
        .zip(expected.chunks(dim))
        .map(|(a, e)| {
            let dot: f64 = a
                .iter()
                .zip(e)
                .map(|(&x, &y)| f64::from(x) * f64::from(y))
                .sum();
            let na: f64 = a.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>().sqrt();
            let ne: f64 = e.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>().sqrt();
            dot / (na * ne)
        })
        .fold(f64::INFINITY, f64::min);
    let max_abs = max_abs_diff(actual, expected);
    Parity {
        min_cosine,
        max_abs,
    }
}

fn max_abs_diff(actual: &[f32], expected: &[f32]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .map(|(&a, &e)| (f64::from(a) - f64::from(e)).abs())
        .fold(0.0, f64::max)
}

fn min_cosine_bound(math: CudaMath) -> f64 {
    match math {
        CudaMath::Tf32 => TF32_MIN_COSINE,
        _ => FP32_MIN_COSINE,
    }
}

fn run_case(runtime: &CudaRuntime, batch: &mut EmbeddingBatch, case: &Case) -> Vec<f32> {
    batch
        .embed(runtime, &case.fbank, &case.masks)
        .expect("embedding forward")
}

#[test]
fn embedding_b1_matches_reference() -> Result<(), CudaError> {
    let Some((runtime, root)) = setup("embedding_b1_matches_reference") else {
        return Ok(());
    };

    for math in [CudaMath::Fp32, CudaMath::Tf32] {
        let model = load_model(&runtime, &root, B1_MODEL, math)?;
        let mut batch = model.batch(&runtime, 1)?;
        for name in B1_CASES {
            let case = load_case(&root, B1_MODEL, name);
            assert_eq!(case.chunks, 1);
            let actual = run_case(&runtime, &mut batch, &case);
            let result = parity(&actual, &case.expected, EMBEDDING_DIM);
            eprintln!(
                "b1 {name} {math:?}: min cosine {:.12} (1 - cos {:.2e}), max abs {:.3e}",
                result.min_cosine,
                1.0 - result.min_cosine,
                result.max_abs
            );
            assert!(
                result.min_cosine >= min_cosine_bound(math),
                "{name} {math:?}: cosine {} below bound",
                result.min_cosine
            );
        }
    }

    Ok(())
}

#[test]
fn embedding_b32_matches_reference() -> Result<(), CudaError> {
    let Some((runtime, root)) = setup("embedding_b32_matches_reference") else {
        return Ok(());
    };
    let case = load_case(&root, B32_MODEL, B32_CASE);
    assert_eq!(case.chunks, 32);

    for math in [CudaMath::Fp32, CudaMath::Tf32] {
        let model = load_model(&runtime, &root, B32_MODEL, math)?;
        let mut batch = model.batch(&runtime, case.chunks)?;
        let eager = run_case(&runtime, &mut batch, &case);
        let result = parity(&eager, &case.expected, EMBEDDING_DIM);
        eprintln!(
            "b32 {B32_CASE} {math:?}: min cosine {:.12} (1 - cos {:.2e}), max abs {:.3e} over {} rows",
            result.min_cosine,
            1.0 - result.min_cosine,
            result.max_abs,
            batch.rows()
        );
        assert!(
            result.min_cosine >= min_cosine_bound(math),
            "b32 {math:?}: cosine {} below bound",
            result.min_cosine
        );

        // a replayed graph must give the eager result bit for bit, and a second
        // eager pass must too, since algorithms are picked once per plan
        let again = run_case(&runtime, &mut batch, &case);
        assert_eq!(
            again, eager,
            "b32 {math:?}: eager forward is not deterministic"
        );
        batch.capture_graph(&runtime)?;
        assert!(batch.has_graph());
        let graphed = run_case(&runtime, &mut batch, &case);
        assert_eq!(
            graphed, eager,
            "b32 {math:?}: graph replay differs from eager"
        );
    }

    Ok(())
}

#[test]
fn embedding_rejects_wrong_input_lengths() -> Result<(), CudaError> {
    let Some((runtime, root)) = setup("embedding_rejects_wrong_input_lengths") else {
        return Ok(());
    };
    let model = load_model(&runtime, &root, B1_MODEL, CudaMath::Fp32)?;
    let mut batch = model.batch(&runtime, 1)?;

    let error = batch
        .embed(&runtime, &[0.0; 10], &[0.0; 3 * 589])
        .expect_err("short fbank must fail");
    assert!(matches!(error, CudaError::BufferLength { .. }), "{error}");
    Ok(())
}

/// Per-layer error against the reference intermediates, for finding the first
/// layer where error jumps; prints a table and fails only on gross mismatches
#[test]
fn embedding_layers_match_reference() -> Result<(), CudaError> {
    let Some((runtime, root)) = setup("embedding_layers_match_reference") else {
        return Ok(());
    };
    let name = B1_CASES[0];
    let mut reference =
        ReferenceFile::open(&root.join(B1_MODEL).join(format!("{name}.safetensors")));
    let case = load_case(&root, B1_MODEL, name);

    for math in [CudaMath::Fp32, CudaMath::Tf32] {
        let model = load_model(&runtime, &root, B1_MODEL, math)?;
        let mut batch = model.batch(&runtime, 1)?;
        let stream = runtime.stream().clone();
        batch.fbank_mut().copy_from_host(&stream, &case.fbank)?;
        batch.masks_mut().copy_from_host(&stream, &case.masks)?;

        let mut shortcuts = 0;
        let mut output_taps = 0;
        let mut rows = Vec::new();
        let mut compare = |tensor: String, actual: &[f32]| {
            let key = format!("tensor/{tensor}");
            assert!(reference.has(&key), "reference has no {key}");
            let (expected, _) = reference.read(&key);
            let scale = expected
                .iter()
                .map(|value| f64::from(value.abs()))
                .fold(0.0, f64::max);
            let error = max_abs_diff(actual, &expected);
            rows.push((tensor, error, scale));
        };
        batch.forward_with_taps(&runtime, &mut |tap, view| {
            let tensor = match tap {
                EmbeddingTap::Stem => "relu".to_string(),
                EmbeddingTap::Hidden { block } => format!("relu_{}", 2 * block + 1),
                EmbeddingTap::Block { block } => format!("relu_{}", 2 * block + 2),
                EmbeddingTap::Shortcut { .. } => {
                    shortcuts += 1;
                    SHORTCUT_TENSORS[shortcuts - 1].to_string()
                }
                EmbeddingTap::Pooled => "where_1".to_string(),
                EmbeddingTap::Output => {
                    output_taps += 1;
                    return Ok(());
                }
            };
            let actual = stream.clone_dtoh(view)?;
            compare(tensor, &actual);
            Ok(())
        })?;

        // downloading applies the FP16 range fallback before comparing the final output
        let output = batch.download_output(&runtime)?;
        compare("output".to_string(), &output);
        assert_eq!(output_taps, 1, "the raw output tap must fire once");

        eprintln!("layer errors {name} {math:?} (max abs, max |ref|, relative):");
        let mut previous = 0.0;
        for (tensor, error, scale) in &rows {
            let relative = error / scale.max(f64::MIN_POSITIVE);
            let jump = if relative > 10.0 * previous && previous > 0.0 {
                "  <- jump"
            } else {
                ""
            };
            eprintln!("  {tensor:>12}  {error:.3e}  {scale:.3e}  {relative:.3e}{jump}");
            previous = relative;
            let bound = if math == CudaMath::Fp32 { 1e-4 } else { 1e-1 };
            assert!(
                relative < bound,
                "{tensor} {math:?}: relative error {relative:.3e}"
            );
        }
        assert_eq!(rows.len(), 1 + 2 * 16 + 3 + 2, "every tap must fire once");
    }

    Ok(())
}

/// Latency, throughput and memory at batch 1 and 32; run on its own under the GPU
/// lock: `flock /workspace/gpu-bench.lock cargo test --release --features cuda --lib
/// embedding_benchmark -- --ignored --nocapture`
#[test]
#[ignore = "benchmark; run explicitly under the GPU lock"]
fn embedding_benchmark() -> Result<(), CudaError> {
    let Some((runtime, root)) = setup("embedding_benchmark") else {
        return Ok(());
    };
    for (model, case) in [(B1_MODEL, B1_CASES[0]), (B32_MODEL, B32_CASE)] {
        benchmark(&runtime, &root, model, case)?;
    }
    Ok(())
}

fn benchmark(
    runtime: &CudaRuntime,
    root: &Path,
    model_name: &str,
    case_name: &str,
) -> Result<(), CudaError> {
    let case = load_case(root, model_name, case_name);
    let warmup = 5;
    let runs = 30;
    let stream = runtime.stream().clone();

    for math in [CudaMath::Fp32, CudaMath::Tf32] {
        for graph in [false, true] {
            runtime.synchronize()?;
            let (free_before, _) = cudarc_mem_info()?;
            let model = load_model(runtime, root, model_name, math)?;
            let mut batch = model.batch(runtime, case.chunks)?;
            if graph {
                batch.capture_graph(runtime)?;
            }
            batch.forward(runtime)?;
            runtime.synchronize()?;
            let (free_after, total) = cudarc_mem_info()?;

            let mut h2d = Vec::new();
            let mut compute = Vec::new();
            let mut d2h = Vec::new();
            let mut output = Vec::new();
            for run in 0..warmup + runs {
                let start = Instant::now();
                batch.fbank_mut().copy_from_host(&stream, &case.fbank)?;
                batch.masks_mut().copy_from_host(&stream, &case.masks)?;
                runtime.synchronize()?;
                let copied = Instant::now();
                batch.forward(runtime)?;
                runtime.synchronize()?;
                let computed = Instant::now();
                output = batch.download_output(runtime)?;
                let downloaded = Instant::now();
                if run >= warmup {
                    h2d.push((copied - start).as_secs_f64() * 1e3);
                    compute.push((computed - copied).as_secs_f64() * 1e3);
                    d2h.push((downloaded - computed).as_secs_f64() * 1e3);
                }
            }

            let result = parity(&output, &case.expected, EMBEDDING_DIM);
            let compute_ms = median(&mut compute);
            eprintln!(
                "bench b{} {math:?} graph={graph}: compute median {compute_ms:.3} ms (min {:.3}), {:.1} chunks/s, \
                 h2d {:.3} ms, d2h {:.3} ms, device memory {:.0} MiB (batch buffers {:.0} MiB, workspace {:.0} MiB, {} plans, {:.0} MiB total), \
                 min cosine {:.12} (1 - cos {:.2e}), max abs {:.3e}",
                case.chunks,
                compute.iter().copied().fold(f64::INFINITY, f64::min),
                case.chunks as f64 / (compute_ms / 1e3),
                median(&mut h2d),
                median(&mut d2h),
                (free_before.saturating_sub(free_after)) as f64 / 1048576.0,
                batch.buffer_bytes() as f64 / 1048576.0,
                batch.workspace_bytes() as f64 / 1048576.0,
                batch.plan_count(),
                total as f64 / 1048576.0,
                result.min_cosine,
                1.0 - result.min_cosine,
                result.max_abs,
            );
        }
    }

    Ok(())
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn cudarc_mem_info() -> Result<(usize, usize), CudaError> {
    Ok(cudarc::driver::result::mem_get_info()?)
}

/// GPU time of each layer of one b32 forward pass, from CUDA events recorded at
/// every tap; `SPEAKRS_EMBEDDING_MATH=tf32` profiles TF32. Run under the GPU lock
#[test]
#[ignore = "profiling; run explicitly under the GPU lock"]
fn embedding_profile_b32() -> Result<(), CudaError> {
    use cudarc::driver::sys::CUevent_flags;

    let Some((runtime, root)) = setup("embedding_profile_b32") else {
        return Ok(());
    };
    let math = match std::env::var("SPEAKRS_EMBEDDING_MATH").as_deref() {
        Ok("tf32") => CudaMath::Tf32,
        _ => CudaMath::Fp32,
    };
    let case = load_case(&root, B32_MODEL, B32_CASE);
    let model = load_model(&runtime, &root, B32_MODEL, math)?;
    let mut batch = model.batch(&runtime, case.chunks)?;
    run_case(&runtime, &mut batch, &case);

    let stream = runtime.stream().clone();
    let timing = Some(CUevent_flags::CU_EVENT_DEFAULT);
    let start = stream.record_event(timing)?;
    let mut events = Vec::new();
    batch.forward_with_taps(&runtime, &mut |tap, _| {
        events.push((tap, stream.record_event(timing)?));
        Ok(())
    })?;
    runtime.synchronize()?;

    let mut previous = &start;
    let mut total = 0.0;
    eprintln!("b32 {math:?} per-tap GPU time (ms):");
    for (tap, event) in &events {
        let ms = previous.elapsed_ms(event)?;
        total += ms;
        eprintln!("  {tap:?}: {ms:.3}");
        previous = event;
    }
    eprintln!("  total {total:.3}");
    Ok(())
}

/// Times every cuDNN forward algorithm on each distinct trunk convolution at b32,
/// to compare cuDNN's heuristic pick with the fastest supported algorithm
#[test]
#[ignore = "exploration; run explicitly under the GPU lock"]
fn embedding_conv_algorithms_b32() -> Result<(), CudaError> {
    use cudarc::cudnn::ConvForward;
    use cudarc::cudnn::sys::{
        cudnnConvolutionFwdAlgo_t, cudnnConvolutionMode_t, cudnnTensorFormat_t,
    };
    use cudarc::driver::sys::CUevent_flags;

    let Some((runtime, _)) = setup("embedding_conv_algorithms_b32") else {
        return Ok(());
    };
    let stream = runtime.stream().clone();
    runtime.prepare_library(super::super::CudaLibrary::Cudnn)?;
    let dnn = runtime.dnn()?.clone();
    // (in, out, h, w, kernel, stride) of the 11 distinct trunk shapes
    let shapes = [
        (1, 32, 80, 998, 3, 1),
        (32, 32, 80, 998, 3, 1),
        (32, 64, 80, 998, 3, 2),
        (32, 64, 80, 998, 1, 2),
        (64, 64, 40, 499, 3, 1),
        (64, 128, 40, 499, 3, 2),
        (64, 128, 40, 499, 1, 2),
        (128, 128, 20, 250, 3, 1),
        (128, 256, 20, 250, 3, 2),
        (128, 256, 20, 250, 1, 2),
        (256, 256, 10, 125, 3, 1),
    ];
    let algos = [
        cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
        cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
        cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
        cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_DIRECT,
        cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_FFT,
        cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING,
        cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
        cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED,
    ];
    let batch = 32;
    let timing = Some(CUevent_flags::CU_EVENT_DEFAULT);

    for math in [CudaMath::Fp32, CudaMath::Tf32] {
        for &(cin, cout, h, w, k, s) in &shapes {
            let pad = k / 2;
            let oh = (h + 2 * pad - k) / s + 1;
            let ow = (w + 2 * pad - k) / s + 1;
            let dims = |v: [usize; 4]| v.map(|d| d as i32);
            let mut conv = dnn.create_conv2d::<f32>(
                [pad as i32; 2],
                [s as i32; 2],
                [1, 1],
                cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
            )?;
            conv.set_math_type(match math {
                CudaMath::Tf32 => cudarc::cudnn::sys::cudnnMathType_t::CUDNN_TENSOR_OP_MATH,
                _ => cudarc::cudnn::sys::cudnnMathType_t::CUDNN_FMA_MATH,
            })?;
            let nchw = cudnnTensorFormat_t::CUDNN_TENSOR_NCHW;
            let xd = dnn.create_4d_tensor::<f32>(nchw, dims([batch, cin, h, w]))?;
            let wd = dnn.create_4d_filter::<f32>(nchw, dims([cout, cin, k, k]))?;
            let yd = dnn.create_4d_tensor::<f32>(nchw, dims([batch, cout, oh, ow]))?;
            let x = stream.alloc_zeros::<f32>(batch * cin * h * w)?;
            let wt = stream.alloc_zeros::<f32>(cout * cin * k * k)?;
            let mut y = stream.alloc_zeros::<f32>(batch * cout * oh * ow)?;
            let forward = ConvForward {
                conv: &conv,
                x: &xd,
                w: &wd,
                y: &yd,
            };
            let heuristic = forward.pick_algorithm()?;

            let mut line = format!(
                "{math:?} {cin}->{cout} {h}x{w} k{k} s{s}: heuristic {:?}",
                heuristic as u32
            );
            for algo in algos {
                let Ok(bytes) = forward.get_workspace_size(algo) else {
                    continue;
                };
                let mut workspace = (bytes > 0)
                    .then(|| stream.alloc_zeros::<u8>(bytes))
                    .transpose()?;
                let mut run = |workspace: &mut Option<cudarc::driver::CudaSlice<u8>>| {
                    // SAFETY: x, wt and y were allocated with the descriptor shapes and
                    // the workspace has the size cuDNN reported for `algo`
                    unsafe {
                        forward.launch(algo, workspace.as_mut(), (1.0f32, 0.0f32), &x, &wt, &mut y)
                    }
                };
                if run(&mut workspace).is_err() {
                    continue;
                }
                let mut times = Vec::new();
                for _ in 0..7 {
                    let start = stream.record_event(timing)?;
                    run(&mut workspace)?;
                    let end = stream.record_event(timing)?;
                    end.synchronize()?;
                    times.push(f64::from(start.elapsed_ms(&end)?));
                }
                line += &format!(
                    " | {}: {:.3}ms ws {}MiB",
                    algo as u32,
                    median(&mut times),
                    bytes >> 20
                );
            }
            eprintln!("{line}");
        }
    }

    Ok(())
}

/// Times cuDNN's fused convolution, bias, residual and ReLU against the plain
/// convolution on the stride-1 3x3 trunk shapes at b32, to size what fusing the
/// epilogue into cuDNN would save
#[test]
#[ignore = "exploration; run explicitly under the GPU lock"]
fn embedding_fused_conv_b32() -> Result<(), CudaError> {
    use cudarc::cudnn::sys::{
        cudnnActivationMode_t, cudnnConvolutionFwdAlgo_t, cudnnConvolutionMode_t, cudnnMathType_t,
        cudnnNanPropagation_t, cudnnTensorFormat_t,
    };
    use cudarc::cudnn::{ConvBiasActivationForward, ConvForward};
    use cudarc::driver::sys::CUevent_flags;

    let Some((runtime, _)) = setup("embedding_fused_conv_b32") else {
        return Ok(());
    };
    let stream = runtime.stream().clone();
    runtime.prepare_library(super::super::CudaLibrary::Cudnn)?;
    let dnn = runtime.dnn()?.clone();
    let batch = 32;
    let timing = Some(CUevent_flags::CU_EVENT_DEFAULT);
    let shapes = [(32, 80, 998), (64, 40, 499), (128, 20, 250), (256, 10, 125)];
    let algos = [
        cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
        cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED,
    ];

    for (math, math_type) in [
        (CudaMath::Fp32, cudnnMathType_t::CUDNN_FMA_MATH),
        (CudaMath::Tf32, cudnnMathType_t::CUDNN_TENSOR_OP_MATH),
    ] {
        for &(c, h, w) in &shapes {
            let mut conv = dnn.create_conv2d::<f32>(
                [1, 1],
                [1, 1],
                [1, 1],
                cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
            )?;
            conv.set_math_type(math_type)?;
            let nchw = cudnnTensorFormat_t::CUDNN_TENSOR_NCHW;
            let dims = [batch as i32, c as i32, h as i32, w as i32];
            let xd = dnn.create_4d_tensor::<f32>(nchw, dims)?;
            let wd = dnn.create_4d_filter::<f32>(nchw, [c as i32, c as i32, 3, 3])?;
            let bd = dnn.create_4d_tensor::<f32>(nchw, [1, c as i32, 1, 1])?;
            let act = dnn.create_activation::<f32>(
                cudnnActivationMode_t::CUDNN_ACTIVATION_RELU,
                cudnnNanPropagation_t::CUDNN_PROPAGATE_NAN,
                0.0,
            )?;
            let len = batch * c * h * w;
            let x = stream.clone_htod(&values(len, 1))?;
            let z = stream.clone_htod(&values(len, 2))?;
            let wt = stream.clone_htod(&values(c * c * 9, 3))?;
            let bias = stream.clone_htod(&values(c, 4))?;
            let mut y = stream.alloc_zeros::<f32>(len)?;
            let plain = ConvForward {
                conv: &conv,
                x: &xd,
                w: &wd,
                y: &xd,
            };
            let fused = ConvBiasActivationForward {
                conv: &conv,
                act: &act,
                x: &xd,
                w: &wd,
                z: &xd,
                bias: &bd,
                y: &xd,
            };

            let mut line = format!("{math:?} {c}ch {h}x{w}:");
            for algo in algos {
                let Ok(bytes) = plain.get_workspace_size(algo) else {
                    continue;
                };
                let mut workspace = stream.alloc_zeros::<u8>(bytes.max(1))?;
                let mut time = |fuse: bool,
                                y: &mut cudarc::driver::CudaSlice<f32>|
                 -> Result<f64, CudaError> {
                    let mut times = Vec::new();
                    for _ in 0..7 {
                        let start = stream.record_event(timing)?;
                        // SAFETY: buffers match the descriptors and the workspace size
                        unsafe {
                            if fuse {
                                fused.launch(
                                    algo,
                                    Some(&mut workspace),
                                    (1.0, 1.0),
                                    &x,
                                    &wt,
                                    &z,
                                    &bias,
                                    y,
                                )?;
                            } else {
                                plain.launch(algo, Some(&mut workspace), (1.0, 0.0), &x, &wt, y)?;
                            }
                        }
                        let end = stream.record_event(timing)?;
                        end.synchronize()?;
                        times.push(f64::from(start.elapsed_ms(&end)?));
                    }
                    Ok(median(&mut times))
                };
                let plain_ms = time(false, &mut y)?;
                let fused_ms = time(true, &mut y)?;
                line += &format!(
                    " | algo {}: conv {plain_ms:.3} ms, fused conv+bias+z+relu {fused_ms:.3} ms",
                    algo as u32
                );
            }
            eprintln!("{line}");
        }
    }

    Ok(())
}

/// Deterministic values in [-1, 1)
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
