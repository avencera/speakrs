//! Graph timings and correctness for embedding batch classes, including driver-only builds

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::time::Instant;

use super::{CudaError, CudaMath, CudaRuntime, EMBEDDING_DIM, ResNetEmbedding, SafetensorsFile};

const B1_MODEL: &str = "wespeaker-multimask-tail";
const B32_MODEL: &str = "wespeaker-multimask-tail-b32";
const B32_CASE: &str = "test_and_short_b32";
const TF32_MIN_COSINE: f64 = 0.999;

pub(crate) fn setup(test: &str) -> Option<(CudaRuntime, PathBuf)> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::new(
            "speakrs::inference::cuda=info",
        ))
        .with_test_writer()
        .try_init();
    let required = std::env::var_os("SPEAKRS_REQUIRE_GPU").is_some_and(|value| value != "0");
    let runtime = match CudaRuntime::new(0) {
        Ok(runtime) => runtime,
        Err(error) if error.is_device_unavailable() && !required => {
            eprintln!("skipping {test}: {error}");
            return None;
        }
        Err(error) => panic!("{test}: {error}"),
    };
    let root = std::env::var_os("SPEAKRS_CUDA_REF")
        .map_or_else(|| PathBuf::from("/workspace/ref"), PathBuf::from);
    if !root.join(B32_MODEL).is_dir() {
        assert!(!required, "{test}: no reference tensors");
        eprintln!("skipping {test}: no reference tensors");
        return None;
    }
    Some((runtime, root))
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
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

pub(crate) struct Case {
    pub(crate) fbank: Vec<f32>,
    pub(crate) masks: Vec<f32>,
    pub(crate) expected: Vec<f32>,
    pub(crate) chunks: usize,
}

pub(crate) fn load_case(root: &Path, model: &str, case: &str) -> Case {
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

/// Worst per-row cosine similarity and the largest absolute difference
#[derive(Debug, Clone, Copy)]
pub(crate) struct Parity {
    pub(crate) min_cosine: f64,
    pub(crate) max_abs: f64,
}

pub(crate) fn parity(actual: &[f32], expected: &[f32], dim: usize) -> Parity {
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

/// Warm production graph timings and row parity for intermediate batch classes
#[test]
#[ignore = "measurement; run on a GPU box under the GPU lock"]
fn embedding_batch_classes() -> Result<(), CudaError> {
    let Some((runtime, root)) = setup("embedding_batch_classes") else {
        return Ok(());
    };
    let source = load_case(&root, B32_MODEL, B32_CASE);
    let weights = SafetensorsFile::open(std::env::var_os("TRUNK_WEIGHTS").map_or_else(
        || root.join(B1_MODEL).join(format!("{B1_MODEL}.safetensors")),
        PathBuf::from,
    ))?;
    let model = ResNetEmbedding::load(&runtime, &weights, CudaMath::Tf32)?;
    let stream = runtime.stream();
    let mut single = model.batch(&runtime, 1)?;
    single.capture_graph(&runtime)?;
    let mut serial = Vec::with_capacity(source.expected.len());
    for row in 0..source.chunks {
        let fbank_len = source.fbank.len() / source.chunks;
        let masks_len = source.masks.len() / source.chunks;
        single.fbank_mut().copy_from_host(
            stream,
            &source.fbank[row * fbank_len..(row + 1) * fbank_len],
        )?;
        single.masks_mut().copy_from_host(
            stream,
            &source.masks[row * masks_len..(row + 1) * masks_len],
        )?;
        single.forward(&runtime)?;
        serial.extend(single.download_output(&runtime)?);
    }
    drop(single);

    // reverse the class order on the second pass to expose order-dependent timing
    for (round, classes) in [[1, 4, 8, 16, 32], [32, 16, 8, 4, 1]]
        .into_iter()
        .enumerate()
    {
        for chunks in classes {
            let mut batch = model.batch(&runtime, chunks)?;
            batch.fbank_mut().copy_from_host(
                stream,
                &source.fbank[..source.fbank.len() * chunks / source.chunks],
            )?;
            batch.masks_mut().copy_from_host(
                stream,
                &source.masks[..source.masks.len() * chunks / source.chunks],
            )?;
            batch.capture_graph(&runtime)?;
            let warm_start = Instant::now();
            while warm_start.elapsed().as_millis() < 200 {
                batch.forward(&runtime)?;
                runtime.synchronize()?;
            }

            let mut gpu_ms = Vec::new();
            let mut wall_ms = Vec::new();
            for _ in 0..10 {
                let start = stream
                    .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))?;
                let wall_start = Instant::now();
                for _ in 0..5 {
                    batch.forward(&runtime)?;
                }
                let end = stream
                    .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))?;
                end.synchronize()?;
                gpu_ms.push(f64::from(start.elapsed_ms(&end)?) / 5.0);
                wall_ms.push(wall_start.elapsed().as_secs_f64() * 1000.0 / 5.0);
            }

            let output = batch.download_output(&runtime)?;
            let expected = &source.expected[..output.len()];
            let reference = parity(&output, expected, EMBEDDING_DIM);
            let serial_parity = parity(&output, &serial[..output.len()], EMBEDDING_DIM);
            assert!(reference.min_cosine >= TF32_MIN_COSINE);
            assert!(serial_parity.min_cosine >= TF32_MIN_COSINE);
            assert!(output.iter().all(|value| value.is_finite()));
            eprintln!(
                "BATCH_CLASS {}",
                serde_json::json!({
                    "round": round, "chunks": chunks, "gpu_ms": median(&mut gpu_ms),
                    "wall_ms": median(&mut wall_ms), "reference_cosine": reference.min_cosine,
                    "serial_cosine": serial_parity.min_cosine, "serial_max_abs": serial_parity.max_abs,
                    "bitwise_serial": output.iter().zip(&serial).all(|(a, b)| a.to_bits() == b.to_bits()),
                    "driver_only": super::driver_only(),
                })
            );
        }
    }
    Ok(())
}

/// Matched host and device measurements with cold and warm graph phases
#[test]
#[ignore = "whole-graph timing; run on a GPU box under the GPU lock"]
fn embedding_whole_warm_timing() -> Result<(), CudaError> {
    let Some((runtime, root)) = setup("embedding_whole_warm_timing") else {
        return Ok(());
    };
    let source = load_case(&root, B32_MODEL, B32_CASE);
    let weights = SafetensorsFile::open(std::env::var_os("TRUNK_WEIGHTS").map_or_else(
        || root.join(B1_MODEL).join(format!("{B1_MODEL}.safetensors")),
        PathBuf::from,
    ))?;
    let choice = std::env::var("SPEAKRS_EMBEDDING_TIMING_CHOICE")
        .unwrap_or_else(|_| "production".to_owned());
    #[cfg(not(feature = "_cuda-libraries"))]
    assert_eq!(choice, "production");

    let model = ResNetEmbedding::load(&runtime, &weights, CudaMath::Tf32)?;
    #[cfg(feature = "_cuda-libraries")]
    let model = {
        let mut model = model;
        match choice.as_str() {
            "production" => {}
            "explicit" => assert!(
                model.select_every_conv(super::implementation::Choice::Oxide(
                    super::implementation::Selection::Explicit,
                ))
            ),
            "library" => assert!(model.select_every_conv(super::implementation::Choice::Library)),
            _ => panic!("unknown timing choice {choice}"),
        }
        model
    };

    let stream = runtime.stream();
    for chunks in [32, 1] {
        let mut batch = model.batch(&runtime, chunks)?;
        #[cfg(feature = "_cuda-libraries")]
        let library_convs: Vec<String> = batch
            .library_convs()
            .into_iter()
            .map(str::to_owned)
            .collect();
        #[cfg(not(feature = "_cuda-libraries"))]
        let library_convs: Vec<String> = Vec::new();
        if choice == "library" {
            assert_eq!(library_convs.len(), 36);
        }
        batch
            .fbank_mut()
            .copy_from_host(stream, &source.fbank[..source.fbank.len() * chunks / 32])?;
        batch
            .masks_mut()
            .copy_from_host(stream, &source.masks[..source.masks.len() * chunks / 32])?;
        batch.capture_graph(&runtime)?;
        batch.forward(&runtime)?;
        let actual = batch.download_output(&runtime)?;
        assert!(actual.iter().all(|value| value.is_finite()));
        assert!(
            parity(&actual, &source.expected[..actual.len()], EMBEDDING_DIM).min_cosine
                >= TF32_MIN_COSINE
        );

        let cold_start = Instant::now();
        for _ in 0..20 {
            batch.forward(&runtime)?;
        }
        runtime.synchronize()?;
        let cold_host_ms = cold_start.elapsed().as_secs_f64() * 1000.0 / 20.0;
        let warm_start = Instant::now();
        while warm_start.elapsed().as_millis() < 300 {
            batch.forward(&runtime)?;
            runtime.synchronize()?;
        }

        let mut gpu_ms = Vec::new();
        let mut wall_ms = Vec::new();
        let epoch_start_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("epoch")
            .as_nanos();
        for _ in 0..10 {
            let start =
                stream.record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))?;
            let host_start = Instant::now();
            for _ in 0..10 {
                batch.forward(&runtime)?;
            }
            let end =
                stream.record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))?;
            end.synchronize()?;
            wall_ms.push(host_start.elapsed().as_secs_f64() * 1000.0 / 10.0);
            gpu_ms.push(f64::from(start.elapsed_ms(&end)?) / 10.0);
        }
        let epoch_end_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("epoch")
            .as_nanos();
        let gpu_median = median(&mut gpu_ms);
        let wall_median = median(&mut wall_ms);
        eprintln!(
            "WHOLE_WARM {}",
            serde_json::json!({
                "chunks": chunks, "math": "Tf32", "choice": choice,
                "driver_only": super::driver_only(), "library_convs": library_convs, "cold_host_ms": cold_host_ms,
                "gpu_ms": gpu_median, "host_ms": wall_median,
                "gpu_samples_ms": gpu_ms, "host_samples_ms": wall_ms,
                "epoch_start_ns": epoch_start_ns.to_string(), "epoch_end_ns": epoch_end_ns.to_string(),
                "warm_replays": 100,
            })
        );
    }
    Ok(())
}
