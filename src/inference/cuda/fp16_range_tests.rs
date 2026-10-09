//! The FP16 range guard on a GPU, in hybrid and driver-only builds
//!
//! FP16 trunk tiles scale both operands by 2^10, so an operand above
//! [`FP16_OPERAND_LIMIT`] saturates. These tests check that such operands never change
//! the embedding: weights above the limit keep their layer off FP16 tiles, and a batch
//! whose activations exceed it is recomputed without them.
//!
//! The runtimes use the production FP32 segmentation and TF32 embedding recipe mode, so
//! a Tesla T4 or RTX 4060 Ti selects FP16 tiles as the pipeline does. On a GPU without
//! an FP16 route the results are still checked, and each case reports that no layer ran
//! FP16 tiles. Each test skips without a GPU unless `SPEAKRS_REQUIRE_GPU` is set

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde_json::json;

use super::candidate::FP16_OPERAND_LIMIT;
use super::fbank::{FBANK_FRAMES, FBANK_MEL_BINS};
use super::implementation::policy::RecipeMode;
use super::{
    CudaError, CudaMath, CudaRuntime, EMBEDDING_DIM, EmbeddingBatch, ResNetEmbedding,
    SafetensorsFile,
};
use crate::inference::cuda::embedding::{MASK_FRAMES, SPEAKERS_PER_CHUNK};

/// Output channels and basic-block count of the four residual stages
const STAGES: [(usize, usize); 4] = [(32, 3), (64, 4), (128, 6), (256, 3)];
/// Frequency bins of the trunk output, after three stride-2 stages of the 80 bins
const OUTPUT_BINS: usize = 10;

/// A runtime on device 0 with the pipeline's recipe mode, or `None` to skip
fn runtime(test: &str) -> Option<CudaRuntime> {
    let required = std::env::var_os("SPEAKRS_REQUIRE_GPU").is_some_and(|value| value != "0");
    match CudaRuntime::new(0) {
        Ok(runtime) => {
            // a global subscriber on a host without libcuda would make other host tests
            // format driver errors, which loads the missing library and panics
            let _ = tracing_subscriber::fmt()
                .with_env_filter(tracing_subscriber::EnvFilter::new(
                    "speakrs::inference::cuda=info",
                ))
                .with_test_writer()
                .try_init();
            Some(runtime.with_recipe_mode(RecipeMode::new(CudaMath::Fp32, CudaMath::Tf32)))
        }
        Err(error) if error.is_device_unavailable() && !required => {
            eprintln!("skipping {test}: {error}");
            None
        }
        Err(error) => panic!("{test}: {error}"),
    }
}

/// Every tensor `ResNetEmbedding::load` reads, all zero until set
struct SyntheticModel(BTreeMap<String, (Vec<usize>, Vec<f32>)>);

impl SyntheticModel {
    fn zeros() -> Self {
        let mut tensors = BTreeMap::new();
        let mut conv = |prefix: &str, out_channels: usize, in_channels: usize, kernel: usize| {
            let shape = vec![out_channels, in_channels, kernel, kernel];
            let len = shape.iter().product();
            tensors.insert(format!("{prefix}.weight"), (shape, vec![0.0; len]));
            tensors.insert(
                format!("{prefix}.weight_bias"),
                (vec![out_channels], vec![0.0; out_channels]),
            );
        };
        conv("resnet.conv1", 32, 1, 3);
        let mut channels = 32;
        for (stage, (out_channels, count)) in STAGES.into_iter().enumerate() {
            for index in 0..count {
                let prefix = format!("resnet.layer{}.{index}", stage + 1);
                conv(&format!("{prefix}.conv1"), out_channels, channels, 3);
                conv(&format!("{prefix}.conv2"), out_channels, out_channels, 3);
                if index == 0 && stage > 0 {
                    conv(&format!("{prefix}.shortcut.0"), out_channels, channels, 1);
                }
                channels = out_channels;
            }
        }
        let pooled = 2 * channels * OUTPUT_BINS;
        tensors.insert(
            "resnet.seg_1.weight".into(),
            (
                vec![EMBEDDING_DIM, pooled],
                vec![0.0; EMBEDDING_DIM * pooled],
            ),
        );
        tensors.insert(
            "resnet.seg_1.bias".into(),
            (vec![EMBEDDING_DIM], vec![0.0; EMBEDDING_DIM]),
        );
        Self(tensors)
    }

    /// An identity path for channel 0 from the stem's bias to embedding element 0: every
    /// shortcut copies channel 0 and the head reads the mean of channel 0, bin 0. With
    /// every other weight zero, each block passes its input through its residual
    fn channel_zero_path(stem_bias: f32) -> Self {
        let mut model = Self::zeros();
        model.set("resnet.conv1.weight_bias", 0, stem_bias);
        for stage in 2..=4 {
            model.set(&format!("resnet.layer{stage}.0.shortcut.0.weight"), 0, 1.0);
        }
        model.set("resnet.seg_1.weight", 0, 1.0);
        model
    }

    /// Sets the centre tap from input channel 0 to output channel 0 of a 3x3 layer
    fn center(&mut self, layer: &str, value: f32) {
        self.set(&format!("{layer}.weight"), 4, value);
    }

    fn set(&mut self, name: &str, index: usize, value: f32) {
        let (_, values) = self.0.get_mut(name).expect("known tensor");
        values[index] = value;
    }

    fn write(&self, path: &Path) -> SafetensorsFile {
        let bytes: BTreeMap<&str, (Vec<usize>, Vec<u8>)> = self
            .0
            .iter()
            .map(|(name, (shape, values))| {
                let bytes = values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect();
                (name.as_str(), (shape.clone(), bytes))
            })
            .collect();
        let views = bytes.iter().map(|(name, (shape, bytes))| {
            let view =
                safetensors::tensor::TensorView::new(safetensors::Dtype::F32, shape.clone(), bytes)
                    .expect("consistent tensor");
            (*name, view)
        });
        let serialized = safetensors::serialize(views, None).expect("serialized model");
        std::fs::write(path, serialized).expect("written model");
        SafetensorsFile::open(path).expect("readable model")
    }
}

/// A scratch path unique to this process and case
fn scratch(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("speakrs-{}-{name}.safetensors", std::process::id()))
}

/// Writes inputs, runs the captured graph as production does, and downloads
fn embed(
    runtime: &CudaRuntime,
    batch: &mut EmbeddingBatch,
    fbank: &[f32],
) -> Result<Vec<f32>, CudaError> {
    let stream = runtime.stream();
    let rows = fbank.len() / (FBANK_FRAMES * FBANK_MEL_BINS) * SPEAKERS_PER_CHUNK;
    batch.fbank_mut().copy_from_host(stream, fbank)?;
    batch
        .masks_mut()
        .copy_from_host(stream, &vec![1.0; rows * MASK_FRAMES])?;
    batch.capture_graph(runtime)?;
    batch.forward(runtime)?;
    batch.download_output(runtime)
}

/// Cosine similarity in f64, 1 for two zero rows
fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| f64::from(x) * f64::from(y))
        .sum();
    let norm = |row: &[f32]| {
        row.iter()
            .map(|&x| f64::from(x).powi(2))
            .sum::<f64>()
            .sqrt()
    };
    let (norm_a, norm_b) = (norm(a), norm(b));
    if norm_a == 0.0 && norm_b == 0.0 {
        return 1.0;
    }
    dot / (norm_a * norm_b)
}

/// The review's counterexample through the whole model: an operand of 128 must give
/// the exact product, not the saturated `FP16_OPERAND_LIMIT`
///
/// Each case drives one constant through `resnet.layer1.0`, whose two 32-channel layers
/// run FP16 tiles on FP16 devices, and on to embedding element 0. All values are
/// exact in FP16, TF32 and FP32
#[test]
fn fp16_range_guard_keeps_saturating_operands_exact() -> Result<(), CudaError> {
    let Some(runtime) = runtime("fp16_range_guard_keeps_saturating_operands_exact") else {
        return Ok(());
    };
    struct Case {
        name: &'static str,
        stem: f32,
        conv1: f32,
        expected: f32,
        /// the value if the saturated operand were kept
        saturated: f32,
    }
    let limit = FP16_OPERAND_LIMIT;
    let cases = [
        // block output relu(conv2(relu(conv1(x))) + x), with conv2's centre tap 1
        Case {
            name: "in_range",
            stem: 16.0,
            conv1: 1.0,
            expected: 32.0,
            saturated: 32.0,
        },
        Case {
            name: "activation_above_limit",
            stem: 128.0,
            conv1: 1.0,
            expected: 256.0,
            saturated: limit + 128.0,
        },
        Case {
            name: "weight_above_limit",
            stem: 0.25,
            conv1: 128.0,
            expected: 32.25,
            saturated: limit * 0.25 + 0.25,
        },
    ];
    for case in cases {
        let mut model = SyntheticModel::channel_zero_path(case.stem);
        model.center("resnet.layer1.0.conv1", case.conv1);
        model.center("resnet.layer1.0.conv2", 1.0);
        let path = scratch(case.name);
        let weights = model.write(&path);
        let embedding = ResNetEmbedding::load(&runtime, &weights, CudaMath::Tf32)?;
        for chunks in [1, 32] {
            let mut batch = embedding.batch(&runtime, chunks)?;
            let fp16: Vec<String> = batch.fp16_convs().into_iter().map(str::to_owned).collect();
            let output = embed(
                &runtime,
                &mut batch,
                &vec![0.0; chunks * FBANK_FRAMES * FBANK_MEL_BINS],
            )?;
            let recomputed = batch.recomputed_without_fp16();
            eprintln!(
                "FP16_RANGE {}",
                json!({
                    "case": case.name, "chunks": chunks, "fp16_convs": fp16.len(),
                    "conv1_fp16": fp16.iter().any(|name| name == "resnet.layer1.0.conv1"),
                    "conv2_fp16": fp16.iter().any(|name| name == "resnet.layer1.0.conv2"),
                    "recomputed": recomputed, "value": output[0],
                    "saturated_value": case.saturated,
                })
            );
            for (row, embedding) in output.chunks(EMBEDDING_DIM).enumerate() {
                assert_eq!(
                    embedding[0], case.expected,
                    "{} b{chunks} row {row}",
                    case.name
                );
                assert!(
                    embedding[1..].iter().all(|&value| value == 0.0),
                    "{} b{chunks} row {row}",
                    case.name
                );
            }
            // the weight guard keeps the layer off FP16 tiles; nothing else saturates
            if case.name == "weight_above_limit" {
                assert!(!fp16.iter().any(|name| name == "resnet.layer1.0.conv1"));
            }
            let saturating = case.name == "activation_above_limit"
                && fp16.iter().any(|name| name == "resnet.layer1.0.conv1");
            assert_eq!(recomputed, saturating, "{} b{chunks}", case.name);
        }
        drop(weights);
        std::fs::remove_file(&path).expect("removed scratch model");
    }
    Ok(())
}

/// The shipped weights with an input scaled until FP16 activations saturate: the
/// guarded batch must return exactly what the plans without FP16 tiles compute
///
/// Reads the weights from `TRUNK_WEIGHTS`, else the reference model under
/// `SPEAKRS_CUDA_REF` or `/workspace/ref`
#[test]
#[ignore = "development check; run on a GPU box with the model weights under the GPU lock"]
fn fp16_range_guard_matches_the_non_fp16_path() -> Result<(), CudaError> {
    let test = "fp16_range_guard_matches_the_non_fp16_path";
    let Some(runtime) = runtime(test) else {
        return Ok(());
    };
    const MODEL: &str = "wespeaker-multimask-tail";
    let path = std::env::var_os("TRUNK_WEIGHTS").map_or_else(
        || {
            std::env::var_os("SPEAKRS_CUDA_REF")
                .map_or_else(|| PathBuf::from("/workspace/ref"), PathBuf::from)
                .join(MODEL)
                .join(format!("{MODEL}.safetensors"))
        },
        PathBuf::from,
    );
    let weights = SafetensorsFile::open(&path)?;
    let model = ResNetEmbedding::load(&runtime, &weights, CudaMath::Tf32)?;
    let mut saturated = 0;
    for chunks in [1, 32] {
        let mut reference = model.batch_without_fp16(&runtime, chunks)?;
        assert!(reference.fp16_convs().is_empty());
        let mut recomputed_any = false;
        let mut fp16 = 0;
        // the tuning benchmark's synthetic filterbank, about +-1, scaled up
        for scale in [1.0f32, 16.0, 256.0, 4096.0] {
            // a batch keeps its fallback plans once built, so each scale gets its own
            // batch to tell which inputs recomputed
            let mut guarded = model.batch(&runtime, chunks)?;
            fp16 = guarded.fp16_convs().len();
            let fbank: Vec<f32> = (0..chunks * FBANK_FRAMES * FBANK_MEL_BINS)
                .map(|index| ((index % 157) as f32 - 78.0) / 80.0 * scale)
                .collect();
            let output = embed(&runtime, &mut guarded, &fbank)?;
            let recomputed = guarded.recomputed_without_fp16();
            let expected = embed(&runtime, &mut reference, &fbank)?;
            let identical = output
                .iter()
                .zip(&expected)
                .all(|(a, b)| a.to_bits() == b.to_bits());
            let min_cosine = output
                .chunks(EMBEDDING_DIM)
                .zip(expected.chunks(EMBEDDING_DIM))
                .map(|(a, b)| cosine(a, b))
                .fold(f64::INFINITY, f64::min);
            eprintln!(
                "FP16_RANGE_FORCED {}",
                json!({
                    "chunks": chunks, "scale": scale, "fp16_convs": fp16,
                    "recomputed": recomputed, "identical": identical,
                    "min_cosine": min_cosine,
                    "finite": output.iter().all(|value| value.is_finite()),
                })
            );
            if scale == 1.0 {
                assert!(!recomputed, "b{chunks}: the unscaled input stays in range");
            }
            // a recomputed batch is exactly the non-FP16 path; an unsaturated one only
            // differs by FP16 rounding
            if recomputed {
                assert!(identical, "b{chunks} scale {scale}");
                saturated += 1;
                recomputed_any = true;
            } else {
                assert!(min_cosine > 0.99, "b{chunks} scale {scale}: {min_cosine}");
            }
        }
        if fp16 > 0 {
            assert!(recomputed_any, "b{chunks}: no scale saturated");
        }
    }
    eprintln!(
        "FP16_RANGE_FORCED_SUMMARY {}",
        json!({ "saturated_cases": saturated })
    );
    Ok(())
}
