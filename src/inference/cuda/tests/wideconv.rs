//! Development checks of the driver-only ResNet trunk: every trunk convolution on its
//! candidate kernel against cuDNN and a sampled f64 truth, then the whole embedding with
//! every convolution on a candidate against the all-cuDNN build
//!
//! Run on a GPU box with the reference tensors and the GPU lock held, for example
//! `flock /workspace/gpu-bench.lock cargo test --release --features cuda --lib
//! driver_trunk -- --ignored --nocapture`. Each case prints one `TRUNK` or
//! `TRUNK_FORWARD` JSON line. Environment filters: `TRUNK_BATCHES` (default
//! `1,7,32,33`), `TRUNK_MATHS`, `TRUNK_LAYERS`, `TRUNK_TIMING=1` for graph-timed
//! library and kernel medians at every batch, and `TRUNK_DEVICE=a100` to plan the
//! wideconv layers with the A100 selection on any sm80-capable GPU, `TRUNK_DEVICE=t4`
//! with the Turing coverage and selection on any GPU, `TRUNK_RESNET=legacy`
//! or `tensor` to force the ResNet FP32 or TF32 tensor-core kernels.
//! `TRUNK_CONFIG=<kernel>[:<partition>[:<first split cell>]]` forces one wideconv
//! configuration on every selected layer, with kernels `tc`, `fp32`, `sweep2`, `wtc1`,
//! `wtp1`, `wtc2`, `wtc3`, `bf16x3`, `h16`, `h16n` (narrow FP16 tiles), `spatial` or
//! `widestem` and partitions
//! `whole`, `two`, `four` or `eight`.
//! `TRUNK_B1_ONLY=1` builds every batch from the batch-1 reference and skips the
//! batch-32 embedding case.
//! `TRUNK_RESNET=sm80` uses the sm80 tier in both modes, with tensor kernels only
//! in TF32 mode, for a direct comparison with the legacy artifact. `TRUNK_WEIGHTS`
//! names the model weights when they are not beside the references

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::Arc;

use cudarc::driver::{CudaGraph, CudaSlice, CudaStream, LaunchConfig, PushKernelArg, sys};
use serde_json::json;

use super::super::candidate::{
    ConfigPin, ConvCandidate, ConvInputs, ConvKernel, ConvLayerSpec, ConvOxide, ConvPin,
    DriverCandidate, Epilogue, FP16_OPERAND_LIMIT, Fp16Policy, HalfIo, Phases, PlanError,
    WideconvAlgorithm, WideconvConfig, WideconvDevice, WideconvFp16Tiles, WideconvOxide,
    WideconvPartition, WideconvProducts, WideconvSplitCells, WideconvTensorKernel, f16_bits,
};
use super::super::dnn::ConvPlanner;
use super::super::geometry::{Conv2d, Residual};
use super::super::implementation::Choice;
use super::super::kernels::ModuleRequest;
use super::super::{
    ComputeCapability, CudaError, CudaMath, CudaRuntime, KernelModule, PtxTier, ResNetEmbedding,
    SafetensorsFile,
};
use super::{reference_dir, runtime};

const B1_MODEL: &str = "wespeaker-multimask-tail";
const B32_MODEL: &str = "wespeaker-multimask-tail-b32";
const B1_CASE: &str = "test_first_b1";
const B32_CASE: &str = "test_and_short_b32";

/// The ONNX outputs of the three shortcut convolutions, by block
const SHORTCUTS: [(usize, &str); 3] = [(3, "getitem_27"), (7, "getitem_54"), (13, "getitem_93")];

/// A reference safetensors file read one tensor at a time; the b32 file is 13 GB
struct References {
    file: File,
    start: u64,
    tensors: BTreeMap<String, (Vec<usize>, u64, u64)>,
}

impl References {
    fn open(path: &Path) -> Self {
        let mut file = File::open(path).unwrap_or_else(|error| panic!("{path:?}: {error}"));
        let mut len = [0; 8];
        file.read_exact(&mut len).expect("header length");
        let len = u64::from_le_bytes(len);
        let mut bytes = vec![0; len as usize];
        file.read_exact(&mut bytes).expect("header");
        let header: serde_json::Map<String, serde_json::Value> =
            serde_json::from_slice(&bytes).expect("reference metadata");
        let tensors = header
            .into_iter()
            .filter(|(name, info)| name != "__metadata__" && info["dtype"] == "F32")
            .map(|(name, info)| {
                let shape = info["shape"]
                    .as_array()
                    .expect("shape")
                    .iter()
                    .map(|v| v.as_u64().expect("dimension") as usize)
                    .collect();
                let offsets = info["data_offsets"].as_array().expect("offsets");
                let start = offsets[0].as_u64().expect("start");
                let end = offsets[1].as_u64().expect("end");
                (name, (shape, start, end))
            })
            .collect();
        Self {
            file,
            start: 8 + len,
            tensors,
        }
    }

    fn read(&mut self, name: &str) -> (Vec<f32>, Vec<usize>) {
        let (shape, start, end) = self.tensors.get(name).expect(name).clone();
        self.file
            .seek(SeekFrom::Start(self.start + start))
            .expect("tensor seek");
        let mut bytes = vec![0; (end - start) as usize];
        self.file.read_exact(&mut bytes).expect("tensor read");
        let (words, tail) = bytes.as_chunks::<4>();
        assert!(tail.is_empty());
        (
            words.iter().copied().map(f32::from_le_bytes).collect(),
            shape,
        )
    }
}

/// One trunk convolution and where its operands live in the reference tensors
#[derive(Debug, Clone)]
struct Layer {
    name: String,
    /// Basic block, 0 to 15; the stem has none
    block: Option<usize>,
    second: bool,
    shortcut: bool,
    ci: usize,
    co: usize,
    stride: usize,
}

impl Layer {
    fn kernel(&self) -> usize {
        if self.shortcut { 1 } else { 3 }
    }

    fn epilogue(&self) -> Epilogue {
        match (self.shortcut, self.second) {
            (true, _) => Epilogue::Bias,
            (false, true) => Epilogue::BiasReluResidual,
            (false, false) => Epilogue::BiasRelu,
        }
    }

    fn input(&self) -> String {
        match (self.block, self.second) {
            (None, _) => "tensor/unsqueeze".into(),
            (Some(block), true) => format!("tensor/relu_{}", 2 * block + 1),
            (Some(block), false) => block_input(block),
        }
    }

    fn residual(&self) -> Option<String> {
        let block = self.block.filter(|_| self.second)?;
        Some(
            SHORTCUTS
                .iter()
                .find(|(first, _)| *first == block)
                .map_or_else(|| block_input(block), |(_, name)| format!("tensor/{name}")),
        )
    }
}

fn block_input(block: usize) -> String {
    if block == 0 {
        "tensor/relu".into()
    } else {
        format!("tensor/relu_{}", 2 * block)
    }
}

/// The 36 trunk convolutions in forward order
fn layers() -> Vec<Layer> {
    let mut result = vec![Layer {
        name: "resnet.conv1".into(),
        block: None,
        second: false,
        shortcut: false,
        ci: 1,
        co: 32,
        stride: 1,
    }];
    let mut block = 0;
    let mut channels = 32;
    for (stage, (co, count)) in [(32, 3), (64, 4), (128, 6), (256, 3)]
        .into_iter()
        .enumerate()
    {
        for index in 0..count {
            let stride = if index == 0 && stage > 0 { 2 } else { 1 };
            let prefix = format!("resnet.layer{}.{index}", stage + 1);
            let layer = |suffix: &str, second, shortcut, ci, stride| Layer {
                name: format!("{prefix}.{suffix}"),
                block: Some(block),
                second,
                shortcut,
                ci,
                co,
                stride,
            };
            result.push(layer("conv1", false, false, channels, stride));
            result.push(layer("conv2", true, false, co, 1));
            if stride != 1 {
                result.push(layer("shortcut.0", false, true, channels, stride));
            }
            channels = co;
            block += 1;
        }
    }
    result
}

/// `TRUNK_WEIGHTS`, else the weights beside the b1 references
fn weights_path(root: &Path) -> std::path::PathBuf {
    std::env::var_os("TRUNK_WEIGHTS").map_or_else(
        || root.join(B1_MODEL).join(format!("{B1_MODEL}.safetensors")),
        Into::into,
    )
}

/// Whether `TRUNK_B1_ONLY` asks every batch to cycle the batch-1 reference item, for
/// boxes without the multi-gigabyte batch-32 reference; shapes and timing are unchanged
fn b1_only() -> bool {
    std::env::var("TRUNK_B1_ONLY").as_deref() == Ok("1")
}

fn selected<T: ToString>(key: &str, value: T) -> bool {
    std::env::var(key)
        .ok()
        .is_none_or(|items| items.split(',').any(|item| item == value.to_string()))
}

fn batches() -> Vec<usize> {
    std::env::var("TRUNK_BATCHES")
        .unwrap_or_else(|_| "1,7,32,33".into())
        .split(',')
        .map(|batch| batch.parse().expect("TRUNK_BATCHES"))
        .collect()
}

/// `batch` items cycled from the reference items, so stress batches see real
/// activations
fn items(values: &[f32], items: usize, batch: usize) -> Vec<f32> {
    let item = values.len() / items;
    (0..batch)
        .flat_map(|index| &values[index % items * item..][..item])
        .copied()
        .collect()
}

fn capture(
    stream: &Arc<CudaStream>,
    mut enqueue: impl FnMut() -> Result<(), CudaError>,
) -> Result<CudaGraph, CudaError> {
    stream.synchronize()?;
    stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
    let enqueued = enqueue();
    let captured = stream.end_capture(sys::CUgraphInstantiate_flags(0));
    enqueued?;
    Ok(captured?.expect("nonempty graph"))
}

/// Median milliseconds per replay over 10 samples of 10 replays, after 100 ms of
/// warm-up so the clock settles
fn timed(graph: &CudaGraph, stream: &CudaStream) -> Result<f64, CudaError> {
    let start = stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
    loop {
        for _ in 0..5 {
            graph.launch()?;
        }
        let end = stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        end.synchronize()?;
        if start.elapsed_ms(&end)? >= 100.0 {
            break;
        }
    }
    let mut samples = Vec::new();
    for _ in 0..10 {
        let start = stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        for _ in 0..10 {
            graph.launch()?;
        }
        let end = stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        end.synchronize()?;
        samples.push(f64::from(start.elapsed_ms(&end)?) / 10.0);
    }
    samples.sort_by(f64::total_cmp);
    Ok((samples[4] + samples[5]) / 2.0)
}

/// The f64 output at fixed corners and 1024 spread samples
fn truth(
    conv: Conv2d,
    x: &[f32],
    weight: &[f32],
    bias: &[f32],
    residual: Option<&[f32]>,
) -> Vec<(usize, f64)> {
    let [oh, ow] = conv.output();
    let [ih, iw] = conv.input;
    let plane = oh * ow;
    let len = conv.batch * conv.out_channels * plane;
    let mut indices = vec![0, ow - 1, (oh - 1) * ow, plane - 1, len - 1];
    indices.extend((0..1024).map(|i| (i * 1_000_003 + 17) % len));
    indices
        .into_iter()
        .map(|index| {
            let item = index / (conv.out_channels * plane);
            let co = index / plane % conv.out_channels;
            let oy = index % plane / ow;
            let ox = index % ow;
            let mut sum = 0.0f64;
            for ci in 0..conv.in_channels {
                for ky in 0..conv.kernel[0] {
                    let iy = (oy * conv.stride[0] + ky) as isize - conv.padding[0] as isize;
                    for kx in 0..conv.kernel[1] {
                        let ix = (ox * conv.stride[1] + kx) as isize - conv.padding[1] as isize;
                        if iy < 0 || ix < 0 || iy as usize >= ih || ix as usize >= iw {
                            continue;
                        }
                        let xi = (item * conv.in_channels + ci) * ih * iw
                            + iy as usize * iw
                            + ix as usize;
                        let wi = (co * conv.in_channels + ci) * conv.kernel[0] * conv.kernel[1]
                            + ky * conv.kernel[1]
                            + kx;
                        sum += f64::from(x[xi]) * f64::from(weight[wi]);
                    }
                }
            }
            if let Some(residual) = residual {
                sum += f64::from(residual[index]);
            }
            sum += f64::from(bias[co]);
            let value = if conv.kernel == [1, 1] {
                sum
            } else {
                sum.max(0.0)
            };
            (index, value)
        })
        .collect()
}

/// Max-abs and relative L2 error of `actual` at the sampled truth
fn error(actual: &[f32], truth: &[(usize, f64)]) -> (f64, f64) {
    let mut max = 0.0f64;
    let mut diff = 0.0f64;
    let mut norm = 0.0f64;
    for &(i, expected) in truth {
        assert!(actual[i].is_finite(), "non-finite output at {i}");
        let delta = f64::from(actual[i]) - expected;
        max = max.max(delta.abs());
        diff += delta * delta;
        norm += expected * expected;
    }
    (max, (diff / norm.max(f64::MIN_POSITIVE)).sqrt())
}

/// Max-abs and relative L2 difference over every element
fn difference(actual: &[f32], expected: &[f32]) -> (f64, f64) {
    assert_eq!(actual.len(), expected.len());
    let mut max = 0.0f64;
    let mut diff = 0.0f64;
    let mut norm = 0.0f64;
    for (&a, &e) in actual.iter().zip(expected) {
        let delta = f64::from(a) - f64::from(e);
        max = max.max(delta.abs());
        diff += delta * delta;
        norm += f64::from(e) * f64::from(e);
    }
    (max, (diff / norm.max(f64::MIN_POSITIVE)).sqrt())
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

/// The wideconv configuration `TRUNK_CONFIG` names
fn forced_config(text: &str) -> WideconvConfig {
    let mut parts = text.split(':');
    let algorithm = match parts.next().expect("TRUNK_CONFIG kernel") {
        "tc" => WideconvAlgorithm::TensorCore(WideconvTensorKernel::Tf32),
        "fp32" => WideconvAlgorithm::Winograd(WideconvProducts::Fp32),
        "sweep2" => WideconvAlgorithm::Winograd(WideconvProducts::Fp32Sweep2),
        "wtc1" => WideconvAlgorithm::Winograd(WideconvProducts::Tf32x1),
        "wtp1" => WideconvAlgorithm::Winograd(WideconvProducts::Tf32x1Staged),
        "wtc2" => WideconvAlgorithm::Winograd(WideconvProducts::Tf32x2),
        "wtc3" => WideconvAlgorithm::Winograd(WideconvProducts::Tf32x3),
        "bf16x3" => WideconvAlgorithm::Winograd(WideconvProducts::Bf16x3),
        "h16" => WideconvAlgorithm::Fp16(WideconvFp16Tiles::Wide),
        "h16n" => WideconvAlgorithm::Fp16(WideconvFp16Tiles::Narrow),
        "spatial" => WideconvAlgorithm::Spatial,
        "widestem" => WideconvAlgorithm::WideStem,
        other => panic!("unknown TRUNK_CONFIG kernel {other}"),
    };
    let partition = match parts.next().unwrap_or("whole") {
        "whole" => WideconvPartition::Whole,
        "two" => WideconvPartition::Two,
        "four" => WideconvPartition::Four,
        "eight" => WideconvPartition::Eight,
        other => panic!("unknown TRUNK_CONFIG partition {other}"),
    };
    let split_cells = parts.next().map_or(WideconvSplitCells::All, |cell| {
        WideconvSplitCells::From(cell.parse().expect("TRUNK_CONFIG first split cell"))
    });
    WideconvConfig {
        algorithm,
        partition,
        split_cells,
    }
}

#[test]
fn trunk_resnet_override_obeys_forced_ptx_jit() -> Result<(), CudaError> {
    use super::super::kernels::{ArtifactHash, LoadedArtifact};

    let area = KernelModule::Resnet;
    let device = ComputeCapability::new(8, 9);
    let tier = if area.variants().embedded(PtxTier::Sm80).is_some() {
        PtxTier::Sm80
    } else {
        PtxTier::Sm75
    };
    let embedded = area.variants().embedded(tier).expect("ResNet PTX");
    let cubin = embedded.cubin(device).expect("RTX 4090 cubin");
    for override_name in ["tensor", "sm80"] {
        for force_jit in [false, true] {
            let request = Candidate::resnet_override_request(
                Some(override_name),
                PtxTier::Sm80,
                device,
                |request| request.diagnostic_request(force_jit),
            )?
            .expect("trunk override selects an artifact");
            let artifact = if force_jit {
                LoadedArtifact::PtxJit {
                    sha256: ArtifactHash::of(embedded.text.as_bytes()),
                }
            } else {
                LoadedArtifact::Cubin {
                    arch: device,
                    sha256: ArtifactHash::of(cubin.bytes),
                }
            };
            assert_eq!(request.area(), area);
            assert_eq!(request.tier(), tier);
            assert_eq!(
                request.artifact(),
                artifact,
                "{override_name} JIT={force_jit}"
            );
        }
    }
    Ok(())
}

/// The candidate that owns a trunk layer
enum Candidate {
    Resnet(ConvOxide, ConvPin),
    Wide(WideconvOxide),
}

impl Candidate {
    /// Plans the area whose coverage names the layer, as driver-only routing does
    fn plan(runtime: &CudaRuntime, spec: ConvLayerSpec<'_>) -> Result<Self, PlanError> {
        let tier = |area| -> Result<_, CudaError> {
            runtime.load_module(runtime.embedded_exact_request(area)?)
        };
        let kernels = tier(KernelModule::Wideconv)?;
        // device-aware coverage, so Turing plans its 32- and 64-channel layers here as
        // routing does
        let turing = std::env::var("TRUNK_DEVICE").is_ok_and(|device| device == "t4");
        let capability = if turing {
            ComputeCapability::new(7, 5)
        } else {
            runtime.device().capability()
        };
        let coverage = WideconvOxide::coverage_on(kernels.tier(), capability, Fp16Policy::Allowed);
        if coverage.covers(spec.name, spec.conv.batch, spec.conv.math) {
            let forced = std::env::var("TRUNK_CONFIG")
                .ok()
                .map(|text| forced_config(&text));
            let plan = match std::env::var("TRUNK_DEVICE").ok().as_deref() {
                _ if forced.is_some() => {
                    let config = forced.expect("checked above");
                    WideconvOxide::with_config(runtime, &kernels, spec, config)?
                }
                Some("t4") => {
                    let device = WideconvDevice {
                        capability,
                        sms: 40,
                        tier: kernels.tier(),
                    };
                    let config = WideconvConfig::select(device, spec.conv, Fp16Policy::Allowed)?;
                    WideconvOxide::with_config(runtime, &kernels, spec, config)?
                }
                Some("a100") => {
                    let device = WideconvDevice {
                        capability: ComputeCapability::new(8, 0),
                        sms: 108,
                        tier: kernels.tier(),
                    };
                    let config = WideconvConfig::select(device, spec.conv, Fp16Policy::Allowed)?;
                    WideconvOxide::with_config(runtime, &kernels, spec, config)?
                }
                _ => {
                    let pin = WideconvOxide::implemented_pin(&spec)?;
                    WideconvOxide::plan(runtime, &kernels, spec, pin)?
                }
            };
            return Ok(Self::Wide(plan));
        }
        let resnet_override = std::env::var("TRUNK_RESNET").ok();
        let kernels = if let Some(request) = Self::resnet_override_request(
            resnet_override.as_deref(),
            runtime.ptx_tier(),
            runtime.device().capability(),
            |request| runtime.effective_request(request),
        )? {
            runtime.load_module(request)?
        } else {
            tier(KernelModule::Resnet)?
        };
        let pin = Self::resnet_pin(runtime, &spec, kernels.tier())?;
        Ok(Self::Resnet(
            ConvOxide::plan(runtime, &kernels, spec, pin)?,
            pin,
        ))
    }

    fn resnet_override_request(
        override_name: Option<&str>,
        tier: PtxTier,
        device: ComputeCapability,
        effective_request: impl FnOnce(ModuleRequest) -> Result<ModuleRequest, CudaError>,
    ) -> Result<Option<ModuleRequest>, CudaError> {
        if !matches!(override_name, Some("tensor" | "sm80" | "slim")) {
            return Ok(None);
        }

        // select the newest runnable tier independently of production bindings,
        // but retain the runtime's artifact policy for the diagnostic load
        let area = KernelModule::Resnet;
        let request = area.variants().driver_request(area, tier, device).ok_or(
            CudaError::AreaTierNotCompiledIn {
                area: area.name(),
                tier,
                device,
                feature: tier.feature(),
            },
        )?;
        effective_request(request).map(Some)
    }

    /// The driver-only pin on this device, or the one `TRUNK_RESNET` forces: `legacy`
    /// for the PR #36 rule, `tensor` for the layer's TF32 tensor-core entry, `slim` for
    /// the 64-channel 32-column tensor-core entry
    fn resnet_pin(
        runtime: &CudaRuntime,
        spec: &ConvLayerSpec<'_>,
        tier: PtxTier,
    ) -> Result<ConvPin, PlanError> {
        let conv = spec.conv;
        match std::env::var("TRUNK_RESNET").ok().as_deref() {
            Some("legacy") => ConvOxide::implemented_pin(spec),
            Some("tensor") | Some("sm80") if conv.math == CudaMath::Tf32 => {
                Ok(ConvPin::Kernel(match (conv.in_channels, conv.stride) {
                    (32, [1, 1]) => ConvKernel::C32Tensor,
                    (64, _) => ConvKernel::C64Tensor,
                    _ => ConvKernel::C32Stride2Tensor,
                }))
            }
            Some("slim") if conv.math == CudaMath::Tf32 && conv.in_channels == 64 => {
                Ok(ConvPin::Kernel(ConvKernel::C64TensorSlim))
            }
            _ => {
                let boundary = super::super::implementation::BoundaryId::named(spec.name);
                match ConvOxide::driver_pin(
                    boundary,
                    conv.batch,
                    conv.math,
                    runtime.device(),
                    tier,
                    Fp16Policy::Allowed,
                )? {
                    ConfigPin::Conv(pin) => Ok(pin),
                    other => Err(PlanError::DeviceUnsupported {
                        reason: format!("foreign pin {other:?}"),
                    }),
                }
            }
        }
    }

    /// Enqueues the plan; FP16 tiles set `range` nonzero when an activation saturates
    fn enqueue(
        &self,
        inputs: ConvInputs<'_, '_>,
        y: &mut cudarc::driver::CudaViewMut<'_, f32>,
        range: &mut cudarc::driver::CudaViewMut<'_, f32>,
        stream: &CudaStream,
    ) -> Result<(), CudaError> {
        match self {
            Self::Resnet(plan, _) => plan.enqueue(inputs, y, &Phases::new(), stream),
            Self::Wide(plan) => plan.enqueue_checked(inputs, y, range, &Phases::new(), stream),
        }
    }

    fn describe(&self) -> String {
        match self {
            Self::Resnet(_, pin) => format!("resnet {pin:?}"),
            Self::Wide(plan) => format!("wideconv {:?}", plan.config()),
        }
    }
}

/// cuDNN for one layer: the fused call, or the bare convolution and the shared bias
/// kernel for a shortcut, as the Library trunk runs them
struct Library {
    plan: super::super::dnn::ConvPlan,
    workspace: CudaSlice<u8>,
    scratch: CudaSlice<f32>,
    bias: cudarc::driver::CudaFunction,
}

impl Library {
    fn new(runtime: &CudaRuntime, conv: Conv2d, output_len: usize) -> Result<Self, CudaError> {
        let plan = ConvPlanner::new(runtime)?.plan(conv)?;
        let stream = runtime.stream();
        Ok(Self {
            workspace: stream.alloc_zeros::<u8>(plan.workspace_bytes().max(1))?,
            scratch: stream.alloc_zeros::<f32>(output_len)?,
            bias: runtime
                .load_kernels(KernelModule::Embedding)?
                .function("embedding_bias")?,
            plan,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn enqueue(
        &mut self,
        stream: &CudaStream,
        layer: &Layer,
        x: &CudaSlice<f32>,
        weight: &CudaSlice<f32>,
        bias: &CudaSlice<f32>,
        residual: Option<&CudaSlice<f32>>,
        out: &mut CudaSlice<f32>,
    ) -> Result<(), CudaError> {
        if !layer.shortcut {
            let residual_view = residual.map(|value| value.as_view());
            return self.plan.forward_bias_relu(
                &mut self.workspace.as_view_mut(),
                &x.as_view(),
                &weight.as_view(),
                &bias.as_view(),
                residual_view.as_ref().map_or(
                    Residual::None {
                        scratch: &self.scratch.as_view(),
                    },
                    Residual::Add,
                ),
                &mut out.as_view_mut(),
            );
        }
        self.plan.forward(
            &mut self.workspace.as_view_mut(),
            &x.as_view(),
            &weight.as_view(),
            &mut out.as_view_mut(),
        )?;
        let conv = *self.plan.spec();
        let co = conv.out_channels as u32;
        let plane = conv.output().iter().product::<usize>() as u32;
        let bias_len = bias.len() as u64;
        let output_len = out.len() as u64;
        let mut launch = stream.launch_builder(&self.bias);
        launch
            .arg(bias)
            .arg(&bias_len)
            .arg(&co)
            .arg(&plane)
            .arg(out)
            .arg(&output_len);
        // safety: the embedding bias ABI and launch cover the checked NCHW buffer
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (
                    plane.div_ceil(1024),
                    (conv.batch * conv.out_channels) as u32,
                    1,
                ),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }?;
        Ok(())
    }
}

/// Every trunk convolution on its candidate against cuDNN and the f64 truth, at the
/// model batches and two stress batches, in both math modes
#[test]
#[ignore = "development check; run on a GPU box with the references under the GPU lock"]
fn driver_trunk_layers_match_library() -> Result<(), CudaError> {
    let Some(runtime) = runtime("driver_trunk_layers_match_library") else {
        return Ok(());
    };
    let Some(root) = reference_dir("driver_trunk_layers_match_library", B1_MODEL) else {
        return Ok(());
    };
    // safety: this test uses one stream for every allocation, transfer, launch and replay
    unsafe { runtime.context().disable_event_tracking() };
    let weights = SafetensorsFile::open(weights_path(&root))?;
    let stream = runtime.stream();
    let timing = std::env::var_os("TRUNK_TIMING").is_some();
    let mut failures = Vec::new();
    for batch in batches() {
        // stress batches cycle the items of the b32 reference
        let (model, case, items_in) = if batch == 1 || b1_only() {
            (B1_MODEL, B1_CASE, 1)
        } else {
            (B32_MODEL, B32_CASE, 32)
        };
        let mut reference = References::open(&root.join(model).join(format!("{case}.safetensors")));
        for layer in layers() {
            if !selected("TRUNK_LAYERS", &layer.name) {
                continue;
            }
            let (xh, shape) = reference.read(&layer.input());
            let xh = items(&xh, items_in, batch);
            let rh = layer
                .residual()
                .map(|name| items(&reference.read(&name).0, items_in, batch));
            let kernel = layer.kernel();
            let wh = weights.read_f32(
                &format!("{}.weight", layer.name),
                &[layer.co, layer.ci, kernel, kernel],
            )?;
            let bh = weights.read_f32(&format!("{}.weight_bias", layer.name), &[layer.co])?;
            let xd = stream.clone_htod(&xh)?;
            let wd = stream.clone_htod(&wh)?;
            let bd = stream.clone_htod(&bh)?;
            let rd = rh.as_ref().map(|v| stream.clone_htod(v)).transpose()?;
            for math in [CudaMath::Fp32, CudaMath::Tf32] {
                if !selected("TRUNK_MATHS", format!("{math:?}")) {
                    continue;
                }
                let conv = Conv2d {
                    batch,
                    in_channels: layer.ci,
                    out_channels: layer.co,
                    input: [shape[2], shape[3]],
                    kernel: [kernel; 2],
                    padding: [kernel / 2; 2],
                    stride: [layer.stride; 2],
                    dilation: [1, 1],
                    math,
                };
                let output_len = conv.output_shape().iter().product();
                let spec = ConvLayerSpec {
                    name: &layer.name,
                    conv,
                    epilogue: layer.epilogue(),
                    weight: &wd,
                    bias: &bd,
                };
                let candidate =
                    Candidate::plan(&runtime, spec).map_err(|error| CudaError::Unsupported {
                        context: "driver trunk plan",
                        reason: format!("{} b{batch} {math:?}: {error}", layer.name),
                    })?;

                let mut library = Library::new(&runtime, conv, output_len)?;
                let mut lib_out = stream.alloc_zeros::<f32>(output_len)?;
                let mut enqueue_library =
                    || library.enqueue(stream, &layer, &xd, &wd, &bd, rd.as_ref(), &mut lib_out);
                enqueue_library()?;
                let lib_graph = capture(stream, &mut enqueue_library)?;
                let lib_values = stream.clone_dtoh(&lib_out)?;
                // cuMemcpyDtoHAsync into pageable memory has no completion guarantee
                stream.synchronize()?;

                let mut out = stream.alloc_zeros::<f32>(output_len)?;
                let mut range = stream.alloc_zeros::<f32>(1)?;
                let residual = rd.as_ref().map(|value| value.as_view());
                let mut enqueue = || {
                    candidate.enqueue(
                        ConvInputs {
                            x: &xd.as_view(),
                            residual: residual.as_ref(),
                            weight: &wd.as_view(),
                            bias: &bd.as_view(),
                        },
                        &mut out.as_view_mut(),
                        &mut range.as_view_mut(),
                        stream,
                    )
                };
                enqueue()?;
                let graph = capture(stream, &mut enqueue)?;
                let eager = stream.clone_dtoh(&out)?;
                // cuMemcpyDtoHAsync into pageable memory has no completion guarantee
                stream.synchronize()?;
                graph.launch()?;
                let replay = stream.clone_dtoh(&out)?;
                // cuMemcpyDtoHAsync into pageable memory has no completion guarantee
                stream.synchronize()?;
                let bitwise = bits(&eager) == bits(&replay);
                // the reference activations stay inside the FP16 operand range
                let range_words = stream.clone_dtoh(&range)?;
                // cuMemcpyDtoHAsync into pageable memory has no completion guarantee
                stream.synchronize()?;
                let saturated = range_words[0].to_bits() != 0;

                let truth = truth(conv, &xh, &wh, &bh, rh.as_deref());
                let lib_error = error(&lib_values, &truth);
                let kernel_error = error(&eager, &truth);
                let versus_library = difference(&eager, &lib_values);
                // gross mismatch only: within ten times the library's error, plus a
                // floor for layers where cuDNN is exact at the samples; cuDNN may pick
                // an FP32 algorithm for a TF32 request, so the TF32 floor is two TF32
                // unit roundoffs (2^-10), not a fraction of one
                let floor = if math == CudaMath::Tf32 {
                    2f64.powi(-10)
                } else {
                    1e-6
                };
                let pass = bitwise && !saturated && kernel_error.1 <= 10.0 * lib_error.1 + floor;
                let (library_ms, kernel_ms) = if timing {
                    (
                        Some(timed(&lib_graph, stream)?),
                        Some(timed(&graph, stream)?),
                    )
                } else {
                    (None, None)
                };
                eprintln!(
                    "TRUNK {}",
                    json!({
                        "layer": layer.name, "batch": batch, "math": format!("{math:?}"),
                        "tier": runtime.ptx_tier().to_string(), "plan": candidate.describe(),
                        "kernel_error": kernel_error, "library_error": lib_error,
                        "versus_library": versus_library, "bitwise": bitwise, "saturated": saturated,
                        "pass": pass,
                        "library_ms": library_ms, "kernel_ms": kernel_ms,
                    })
                );
                if !pass {
                    failures.push(format!("{} b{batch} {math:?}", layer.name));
                }
            }
        }
    }
    assert!(failures.is_empty(), "gross mismatches: {failures:?}");
    Ok(())
}

/// Minimum per-embedding cosine against the ONNX Runtime FP32 reference, as the
/// embedding parity tests require
fn min_cosine_bound(math: CudaMath) -> f64 {
    match math {
        CudaMath::Fp32 => 0.99999,
        CudaMath::Tf32 => 0.999,
    }
}

fn min_cosine(actual: &[f32], expected: &[f32]) -> Option<f64> {
    let dimension = super::super::EMBEDDING_DIM;
    if actual.is_empty()
        || actual.len() != expected.len()
        || !actual.len().is_multiple_of(dimension)
        || !actual.iter().chain(expected).all(|value| value.is_finite())
    {
        return None;
    }
    let mut minimum = 1.0f64;
    for (a, e) in actual
        .chunks_exact(dimension)
        .zip(expected.chunks_exact(dimension))
    {
        let dot: f64 = a
            .iter()
            .zip(e)
            .map(|(&a, &e)| f64::from(a) * f64::from(e))
            .sum();
        let norm = |values: &[f32]| {
            values
                .iter()
                .map(|&value| f64::from(value).powi(2))
                .sum::<f64>()
                .sqrt()
        };
        let cosine = dot / (norm(a) * norm(e));
        if !cosine.is_finite() {
            return None;
        }
        minimum = minimum.min(cosine);
    }
    Some(minimum)
}

#[test]
fn embedding_cosine_rejects_nonfinite_values_and_invalid_rows() {
    let reference = vec![1.0; 2 * super::super::EMBEDDING_DIM];
    assert_eq!(min_cosine(&reference, &reference), Some(1.0));
    assert_eq!(min_cosine(&[], &[]), None);
    assert_eq!(min_cosine(&reference[..256], &reference), None);
    assert_eq!(min_cosine(&reference[..257], &reference[..257]), None);
    assert_eq!(min_cosine(&vec![0.0; reference.len()], &reference), None);
    assert_eq!(min_cosine(&reference, &vec![0.0; reference.len()]), None);
    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut values = reference.clone();
        let last = values.len() - 1;
        values[last] = invalid;
        assert_eq!(min_cosine(&values, &reference), None);
        assert_eq!(min_cosine(&reference, &values), None);
    }
    let mut opposite = reference.clone();
    opposite[super::super::EMBEDDING_DIM..].fill(-1.0);
    assert_eq!(min_cosine(&opposite, &reference), Some(-1.0));
}

/// The whole embedding with every trunk convolution requested on a candidate, replayed
/// as a CUDA graph, against the same model with every convolution on cuDNN and against
/// the ONNX Runtime reference
///
/// Production and explicit plans share one artifact per area, including the pinned
/// ResNet PTX JIT binding on capability 12.0; no artifact override is needed
#[test]
#[ignore = "development check; run on a GPU box with the references under the GPU lock"]
fn driver_trunk_embedding_matches_library() -> Result<(), CudaError> {
    let Some(runtime) = runtime("driver_trunk_embedding_matches_library") else {
        return Ok(());
    };
    let Some(root) = reference_dir("driver_trunk_embedding_matches_library", B1_MODEL) else {
        return Ok(());
    };
    let weights = SafetensorsFile::open(weights_path(&root))?;
    let stream = runtime.stream().clone();
    let timing = std::env::var_os("TRUNK_TIMING").is_some();
    for (model, case) in [(B1_MODEL, B1_CASE), (B32_MODEL, B32_CASE)] {
        if model == B32_MODEL && b1_only() {
            continue;
        }
        let mut reference = References::open(&root.join(model).join(format!("{case}.safetensors")));
        let (fbank, fbank_shape) = reference.read("input/fbank");
        let (masks, _) = reference.read("input/masks");
        let (expected, _) = reference.read("tensor/output");
        let chunks = fbank_shape[0];
        for math in [CudaMath::Fp32, CudaMath::Tf32] {
            if !selected("TRUNK_MATHS", format!("{math:?}")) {
                continue;
            }
            let run = |choice| -> Result<_, CudaError> {
                let mut model = ResNetEmbedding::load(&runtime, &weights, math)?;
                assert!(model.select_every_conv(choice));
                let mut batch = model.batch(&runtime, chunks)?;
                let library: Vec<String> = batch
                    .library_convs()
                    .into_iter()
                    .map(str::to_owned)
                    .collect();
                batch.fbank_mut().copy_from_host(&stream, &fbank)?;
                batch.masks_mut().copy_from_host(&stream, &masks)?;
                batch.capture_graph(&runtime)?;
                batch.forward(&runtime)?;
                let output = batch.download_output(&runtime)?;
                let ms = if timing {
                    let start = std::time::Instant::now();
                    for _ in 0..20 {
                        batch.forward(&runtime)?;
                    }
                    runtime.synchronize()?;
                    Some(start.elapsed().as_secs_f64() * 1000.0 / 20.0)
                } else {
                    None
                };
                Ok((output, library, ms))
            };
            let (oxide, oxide_library, oxide_ms) = run(Choice::Oxide(
                super::super::implementation::Selection::Explicit,
            ))?;
            let (library, library_library, library_ms) = run(Choice::Library)?;
            assert_eq!(
                library_library.len(),
                36,
                "the control runs every conv on cuDNN"
            );
            let cosine = min_cosine(&oxide, &expected).expect(
                "candidate embedding must contain finite, complete rows and finite cosines",
            );
            let library_cosine = min_cosine(&library, &expected)
                .expect("library embedding must contain finite, complete rows and finite cosines");
            eprintln!(
                "TRUNK_FORWARD {}",
                json!({
                    "chunks": chunks, "math": format!("{math:?}"),
                    "tier": runtime.ptx_tier().to_string(),
                    "library_convs": oxide_library,
                    "versus_library": difference(&oxide, &library),
                    "versus_reference": difference(&oxide, &expected),
                    "library_versus_reference": difference(&library, &expected),
                    "min_cosine": cosine, "library_min_cosine": library_cosine,
                    "forward_ms": oxide_ms, "library_forward_ms": library_ms,
                })
            );
            assert!(
                cosine >= min_cosine_bound(math),
                "{case} {math:?}: cosine {cosine}"
            );
        }
    }
    Ok(())
}

/// The same-channel FP16 shapes whose tiles have [`HalfIo`] launches: channels, input
/// plane and tiles
const HALF_SHAPES: [(usize, [usize; 2], WideconvFp16Tiles); 6] = [
    (32, [80, 998], WideconvFp16Tiles::Wide),
    (64, [40, 499], WideconvFp16Tiles::Wide),
    (128, [20, 250], WideconvFp16Tiles::Wide),
    (128, [20, 250], WideconvFp16Tiles::Narrow),
    (256, [10, 125], WideconvFp16Tiles::Wide),
    (256, [10, 125], WideconvFp16Tiles::Narrow),
];

/// Seeded values uniform in `[-scale, scale)`, from SplitMix64
fn seeded(seed: u64, len: usize, scale: f32) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^= z >> 31;
            // 24 bits give an exact float in [0, 1)
            let unit = (z >> 40) as f32 / (1u64 << 24) as f32;
            (2.0 * unit - 1.0) * scale
        })
        .collect()
}

/// `value` as the FP16 tiles convert an activation: scaled by 2^10, saturated at the
/// largest finite FP16 value and rounded to nearest even
fn half_bits(value: f32) -> u16 {
    f16_bits((value * 1024.0).clamp(-65504.0, 65504.0))
}

/// The FP32 value of [`half_bits`], which a half residual adds back
fn half_value(value: f32) -> f32 {
    let bits = half_bits(value);
    let magnitude = i32::from((bits >> 10) & 0x1f);
    let mantissa = f32::from(bits & 0x3ff);
    // saturation keeps every half finite, and both scales are powers of two
    let scaled = if magnitude == 0 {
        mantissa * 2f32.powi(-24)
    } else {
        (1024.0 + mantissa) * 2f32.powi(magnitude - 25)
    };
    let sign = if bits & 0x8000 == 0 { 1.0 } else { -1.0 };
    sign * scaled / 1024.0
}

/// NCHW `values` as a half tensor, `[batch, channels / 8, h, w, 8]` halves, two to a word
fn half_tensor(values: &[f32], channels: usize, plane: usize) -> Vec<u16> {
    let mut halves = vec![0u16; values.len()];
    for (index, &value) in values.iter().enumerate() {
        let item = index / (channels * plane);
        let channel = index / plane % channels;
        let pixel = index % plane;
        let at = (item * channels + channel / 8 * 8) * plane + pixel * 8 + channel % 8;
        halves[at] = half_bits(value);
    }
    halves
}

/// The halves of a half tensor's words, low half first
fn halves(words: &[f32]) -> Vec<u16> {
    words
        .iter()
        .flat_map(|word| {
            let bits = word.to_bits();
            [bits as u16, (bits >> 16) as u16]
        })
        .collect()
}

/// Packs halves two to a word, as a half tensor's FP32 buffer holds them
fn half_words(halves: &[u16]) -> Vec<f32> {
    halves
        .as_chunks::<2>()
        .0
        .iter()
        .map(|&[low, high]| f32::from_bits(u32::from(low) | u32::from(high) << 16))
        .collect()
}

/// Asserts equal sequences, naming the first difference instead of printing both
fn assert_same<T: PartialEq + std::fmt::Debug>(context: &str, actual: &[T], expected: &[T]) {
    assert_eq!(actual.len(), expected.len(), "{context}: length");
    if let Some(index) = actual.iter().zip(expected).position(|(a, e)| a != e) {
        panic!(
            "{context}: element {index} is {:?}, expected {:?}",
            actual[index], expected[index]
        );
    }
}

/// One forced FP16 plan with its weights
struct HalfLayer {
    plan: WideconvOxide,
    weight: CudaSlice<f32>,
    /// FP32 elements of the output
    len: usize,
}

impl HalfLayer {
    /// Runs the plan once with the operands `io` names as half tensors: the output
    /// words, and whether the launch set the out-of-range word
    fn run(
        &self,
        stream: &Arc<CudaStream>,
        bias: &CudaSlice<f32>,
        x: &CudaSlice<f32>,
        residual: Option<&CudaSlice<f32>>,
        io: HalfIo,
    ) -> Result<(Vec<f32>, bool), CudaError> {
        let words = if io.output { self.len / 2 } else { self.len };
        let mut y = stream.alloc_zeros::<f32>(words)?;
        let mut range = stream.alloc_zeros::<f32>(1)?;
        let residual = residual.map(|value| value.as_view());
        self.plan.enqueue_half(
            ConvInputs {
                x: &x.as_view(),
                residual: residual.as_ref(),
                weight: &self.weight.as_view(),
                bias: &bias.as_view(),
            },
            &mut y.as_view_mut(),
            &mut range.as_view_mut(),
            io,
            &Phases::new(),
            stream,
        )?;
        let values = stream.clone_dtoh(&y)?;
        let range = stream.clone_dtoh(&range)?;
        // cuMemcpyDtoHAsync into pageable memory has no completion guarantee
        stream.synchronize()?;
        Ok((values, range[0].to_bits() != 0))
    }
}

/// Every FP16 plan with [`HalfIo`] launches, forced whatever this device selects, on a
/// residual block of seeded activations and weights: the hidden activation passed as
/// halves gives the FP32 pair's result bit for bit, a half residual gives the FP32
/// result with that residual rounded to FP16 bit for bit, half outputs are the FP32
/// outputs converted as the staging would, and a saturating half output sets the
/// out-of-range word
#[test]
fn forced_fp16_half_launches_match_fp32() -> Result<(), CudaError> {
    const TEST: &str = "forced_fp16_half_launches_match_fp32";
    let Some(runtime) = runtime(TEST) else {
        return Ok(());
    };
    let kernels = runtime.load_module(runtime.embedded_exact_request(KernelModule::Wideconv)?)?;
    let stream = runtime.stream();
    let fp32 = HalfIo::FP32;
    let io = |input, residual, output| HalfIo {
        input,
        residual,
        output,
    };
    for (shape, &(channels, input, tiles)) in HALF_SHAPES.iter().enumerate() {
        for batch in [2, 3] {
            let conv = Conv2d {
                batch,
                in_channels: channels,
                out_channels: channels,
                input,
                kernel: [3, 3],
                padding: [1, 1],
                stride: [1, 1],
                dilation: [1, 1],
                math: CudaMath::Tf32,
            };
            let len = conv.output_shape().iter().product::<usize>();
            let plane = input[0] * input[1];
            let name = format!("c{channels} {tiles:?} b{batch}");

            // activations of a few units, as the trunk's are, and weights that keep
            // each convolution's output variance near its input's, so nothing saturates
            let seed = 16 * shape as u64 + 4 * batch as u64;
            let scale = (3.0 / (9 * channels) as f32).sqrt();
            let xh = seeded(seed, len, 2.0);
            let b1h = seeded(seed + 1, channels, 0.1);
            let b2h = seeded(seed + 2, channels, 0.1);
            let xd = stream.clone_htod(&xh)?;
            let b1 = stream.clone_htod(&b1h)?;
            let b2 = stream.clone_htod(&b2h)?;
            let mut layers = Vec::new();
            for (offset, epilogue) in [(3, Epilogue::BiasRelu), (4, Epilogue::BiasReluResidual)] {
                let weight =
                    stream.clone_htod(&seeded(seed + offset, channels * channels * 9, scale))?;
                let bias = if offset == 3 { &b1 } else { &b2 };
                let spec = ConvLayerSpec {
                    name: &name,
                    conv,
                    epilogue,
                    weight: &weight,
                    bias,
                };
                let config = WideconvConfig {
                    algorithm: WideconvAlgorithm::Fp16(tiles),
                    partition: WideconvPartition::Whole,
                    split_cells: WideconvSplitCells::All,
                };
                // the FP16 tiles build on every tier, so no device may skip a shape
                let plan = WideconvOxide::with_config(&runtime, &kernels, spec, config).map_err(
                    |error| CudaError::Unsupported {
                        context: "forced FP16 half plan",
                        reason: format!("{name}: {error}"),
                    },
                )?;
                layers.push(HalfLayer { plan, weight, len });
            }

            let [conv1, conv2] = &layers[..] else {
                unreachable!("a residual block has two convolutions");
            };
            assert!(
                conv1.plan.has_half_io() && conv2.plan.has_half_io(),
                "{name}: no half launches"
            );

            // (a) the hidden activation as halves: conv1's half output is the FP32
            // output as the staging converts it, and conv2 reads it to the FP32 pair's
            // result bit for bit, from the FP32 and from the half block input
            let (hidden, saturated) = conv1.run(stream, &b1, &xd, None, fp32)?;
            assert!(!saturated, "{name}: FP32 conv1 set the range word");
            let hidden_halves = half_tensor(&hidden, channels, plane);
            let x_half = stream.clone_htod(&half_words(&half_tensor(&xh, channels, plane)))?;
            for (form, x, io) in [
                ("half out", &xd, io(false, false, true)),
                ("half in out", &x_half, io(true, false, true)),
            ] {
                let (words, saturated) = conv1.run(stream, &b1, x, None, io)?;
                assert!(!saturated, "{name}: conv1 {form} set the range word");
                assert_same(
                    &format!("{name}: conv1 {form}"),
                    &halves(&words),
                    &hidden_halves,
                );
            }
            let (from_half, _) = conv1.run(stream, &b1, &x_half, None, io(true, false, false))?;
            assert_same(
                &format!("{name}: conv1 half in"),
                &bits(&from_half),
                &bits(&hidden),
            );

            let hd = stream.clone_htod(&hidden)?;
            let hidden_half = stream.clone_htod(&half_words(&hidden_halves))?;
            let (output, _) = conv2.run(stream, &b2, &hd, Some(&xd), fp32)?;
            let (paired, saturated) =
                conv2.run(stream, &b2, &hidden_half, Some(&xd), io(true, false, false))?;
            assert!(!saturated, "{name}: conv2 half in set the range word");
            assert_same(
                &format!("{name}: conv2 half in"),
                &bits(&paired),
                &bits(&output),
            );
            let (words, _) =
                conv2.run(stream, &b2, &hidden_half, Some(&xd), io(true, false, true))?;
            assert_same(
                &format!("{name}: conv2 half in out"),
                &halves(&words),
                &half_tensor(&output, channels, plane),
            );

            // (b) the residual as halves adds exactly the FP16-rounded residual, so the
            // FP32 launch with that residual rounded on the host is the bit-exact bound
            let rounded: Vec<f32> = xh.iter().map(|&value| half_value(value)).collect();
            let rd = stream.clone_htod(&rounded)?;
            let (expected, _) = conv2.run(stream, &b2, &hd, Some(&rd), fp32)?;
            let (actual, saturated) = conv2.run(
                stream,
                &b2,
                &hidden_half,
                Some(&x_half),
                io(true, true, false),
            )?;
            assert!(!saturated, "{name}: conv2 half in res set the range word");
            assert_same(
                &format!("{name}: conv2 half in res"),
                &bits(&actual),
                &bits(&expected),
            );
            let (words, saturated) = conv2.run(
                stream,
                &b2,
                &hidden_half,
                Some(&x_half),
                io(true, true, true),
            )?;
            assert!(!saturated, "{name}: conv2 half all set the range word");
            assert_same(
                &format!("{name}: conv2 half all"),
                &halves(&words),
                &half_tensor(&expected, channels, plane),
            );

            // (c) a bias above the FP16 operand range saturates one output channel: the
            // FP32 output passes it through, and the half output clamps it to the largest
            // finite FP16 value and sets the range word
            let mut loud = b1h.clone();
            loud[0] = 2.0 * FP16_OPERAND_LIMIT;
            let loud = stream.clone_htod(&loud)?;
            let (unclamped, saturated) = conv1.run(stream, &loud, &xd, None, fp32)?;
            assert!(!saturated, "{name}: an FP32 output set the range word");
            let (words, saturated) = conv1.run(stream, &loud, &xd, None, io(false, false, true))?;
            assert!(
                saturated,
                "{name}: a saturating half output left the range word clear"
            );
            assert_same(
                &format!("{name}: saturated conv1 half out"),
                &halves(&words),
                &half_tensor(&unclamped, channels, plane),
            );

            eprintln!("FP16_HALF {name}: ok");
        }
    }
    Ok(())
}
