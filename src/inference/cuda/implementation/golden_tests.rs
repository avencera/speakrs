//! Golden legacy selection after replacing the ResNet artifact binding
//!
//! The expectations are literal and independent of the production table. Only
//! `observe` and `module_request` adapt the selection API under test

use super::{BoundaryId, Modules, PlanRequest, Selected, TokenEvidence};
use crate::inference::cuda::device::DeviceAttributes;
use crate::inference::cuda::device::test_support::Builder;
use crate::inference::cuda::kernels::{ArtifactHash, LoadedArtifact, ModuleRequest};
use crate::inference::cuda::{ComputeCapability, CudaError, CudaMath, KernelModule, PtxTier};

/// One physical device class the golden table covers
#[derive(Debug, Clone, Copy)]
struct Device {
    capability: ComputeCapability,
    multiprocessors: u32,
    name: &'static str,
}

const DEVICES: [Device; 5] = [
    Device {
        capability: ComputeCapability::new(12, 0),
        multiprocessors: 36,
        name: "NVIDIA GeForce RTX 5060 Ti",
    },
    Device {
        capability: ComputeCapability::new(12, 0),
        multiprocessors: 70,
        name: "NVIDIA GeForce RTX 5070 Ti",
    },
    Device {
        capability: ComputeCapability::new(8, 9),
        multiprocessors: 34,
        name: "NVIDIA GeForce RTX 4060 Ti",
    },
    Device {
        capability: ComputeCapability::new(8, 0),
        multiprocessors: 108,
        name: "NVIDIA A100-SXM4-80GB",
    },
    Device {
        capability: ComputeCapability::new(7, 5),
        multiprocessors: 40,
        name: "Tesla T4",
    },
];

const LSTM_RECORD: &str = "3badc1aec939b0e8f7312786d695bec6445de1dacb1f85e44124bf3ac20356f8";
const SINC_RECORD: &str = "a4d1a2692b78a814cd2da16f084801c3641bfd11c6d095d610f3a8182ad6f675";
const INTEGRATED_DER: &str = "8066268031afba058d93e305206d5b8225e40c6ab2ebb1052874d607e055646f";
const LSTM_PTX: &str = "72945743a3c1b915c05d8ea21b438dfd860fa9487401c447fb24fd48802916fa";
const SINC_PTX: &str = "967bc6893f80da84d8d4d288f2cf1ca3336ca09c4386495cab722beb0ac87247";

/// What production selection decided and which artifact it asked the loader for
#[derive(Debug, Clone, PartialEq, Eq)]
enum Outcome {
    Library,
    Oxide {
        record: String,
        der: String,
        tier: PtxTier,
        artifact: LoadedArtifact,
    },
}

/// Every boundary the models ask production selection about
fn model_boundaries() -> Vec<String> {
    // the same names and order as `Trunk::load`
    let mut boundaries = vec!["resnet.conv1".to_owned()];
    for (stage, blocks) in [(1, 3), (2, 4), (3, 6), (4, 3)] {
        for block in 0..blocks {
            let prefix = format!("resnet.layer{stage}.{block}");
            boundaries.push(format!("{prefix}.conv1"));
            boundaries.push(format!("{prefix}.conv2"));
            if block == 0 && stage > 1 {
                boundaries.push(format!("{prefix}.shortcut.0"));
            }
        }
    }
    boundaries.extend(
        [
            "sincnet.conv0.abs_pool",
            "sincnet.conv1",
            "sincnet.conv2",
            "lstm.stack",
        ]
        .map(str::to_owned),
    );
    boundaries
}

/// Model and workspace batch sizes, including every development stress batch
fn batches() -> impl Iterator<Item = usize> {
    (1..=66).chain([96, 128, 1024])
}

fn legacy(record: &'static str, ptx: &str) -> Outcome {
    Outcome::Oxide {
        record: record.to_owned(),
        der: INTEGRATED_DER.to_owned(),
        tier: PtxTier::Sm75,
        artifact: LoadedArtifact::PtxJit {
            sha256: ArtifactHash::from_hex(ptx),
        },
    }
}

/// The four remaining PR #36 tuples, written out without consulting any coverage declaration
fn expected(device: Device, boundary: &str, batch: usize, math: CudaMath) -> Outcome {
    if device.capability != ComputeCapability::new(12, 0) || ![1, 32].contains(&batch) {
        return Outcome::Library;
    }
    let fp32 = math == CudaMath::Fp32;
    let selected = match boundary {
        "lstm.stack" => fp32.then(|| legacy(LSTM_RECORD, LSTM_PTX)),
        "sincnet.conv0.abs_pool" => fp32.then(|| legacy(SINC_RECORD, SINC_PTX)),
        _ => None,
    };
    selected.unwrap_or(Outcome::Library)
}

impl Device {
    fn attributes(self) -> DeviceAttributes {
        Builder::new(self.capability)
            .multiprocessors(self.multiprocessors)
            .name(self.name)
            .build()
    }
}

/// A host-only runtime: cached attributes, the default tier limit and a scripted loader
struct Fixture {
    device: DeviceAttributes,
    limit: PtxTier,
    loader: Loader,
    loads: usize,
}

impl Modules for &mut Fixture {
    fn device(&self) -> &DeviceAttributes {
        &self.device
    }

    fn tier_limit(&self) -> PtxTier {
        self.limit
    }

    fn load(&mut self, request: ModuleRequest) -> Result<ModuleRequest, CudaError> {
        self.loads += 1;
        match self.loader {
            Loader::Exact => Ok(request),
            Loader::Refuses => Err(CudaError::ArtifactUnavailable {
                module: request.area().name(),
                artifact: request.artifact(),
            }),
        }
    }

    fn embedded_exact(&self, _area: KernelModule) -> Result<ModuleRequest, CudaError> {
        panic!("production never asks for the best embedded artifact")
    }
}

/// Run production selection with a loader that succeeds exactly as requested
fn observe(
    device: Device,
    boundary: &str,
    batch: usize,
    math: CudaMath,
    loader: Loader,
) -> Result<(Outcome, usize), CudaError> {
    let mut fixture = Fixture {
        device: device.attributes(),
        limit: PtxTier::select(device.capability, None)?,
        loader,
        loads: 0,
    };
    let boundary = BoundaryId::parse(boundary).expect("model boundaries are in the table");
    let selected = PlanRequest::Production.resolve(boundary, batch, math, &mut fixture)?;
    let outcome = match selected {
        Selected::Library => Outcome::Library,
        Selected::Oxide(token) => {
            let TokenEvidence::Production { accuracy, speed } = token.evidence else {
                panic!("production tokens carry production evidence")
            };
            // a PR #36 record proves accuracy and speed together
            assert_eq!(accuracy, speed.record);
            Outcome::Oxide {
                record: speed.record.to_string(),
                der: speed.integrated.to_string(),
                tier: token.target.module.tier(),
                artifact: token.target.module.artifact(),
            }
        }
    };
    Ok((outcome, fixture.loads))
}

/// The artifact a production module load requests before any driver work, if any
fn module_request(device: Device, area: KernelModule) -> Option<(PtxTier, LoadedArtifact)> {
    let limit = PtxTier::select(device.capability, None).unwrap();
    super::qualified_module(area, &device.attributes(), limit, area.variants())
        .unwrap()
        .map(|request| (request.tier(), request.artifact()))
}

#[derive(Debug, Clone, Copy)]
enum Loader {
    Exact,
    Refuses,
}

#[test]
fn golden_legacy_selection_is_unchanged() {
    let mut selected = Vec::new();
    for device in DEVICES {
        for boundary in model_boundaries() {
            for batch in batches() {
                for math in [CudaMath::Fp32, CudaMath::Tf32] {
                    let expected = expected(device, &boundary, batch, math);
                    let (actual, loads) =
                        observe(device, &boundary, batch, math, Loader::Exact).unwrap();
                    assert_eq!(
                        actual, expected,
                        "{} ({}, {} SMs) {boundary} b{batch} {math:?}",
                        device.name, device.capability, device.multiprocessors
                    );
                    // a Library tuple never loads a candidate module
                    assert_eq!(loads, usize::from(expected != Outcome::Library));
                    if expected == Outcome::Library {
                        continue;
                    }
                    selected.push((device.multiprocessors, boundary.clone(), batch, math));

                    // an artifact error does not imply a kernel capability refusal
                    assert!(matches!(
                        observe(device, &boundary, batch, math, Loader::Refuses),
                        Err(CudaError::ArtifactUnavailable { .. })
                    ));
                }
            }
            for math in [CudaMath::Fp32, CudaMath::Tf32] {
                assert!(observe(device, &boundary, 0, math, Loader::Exact).is_err());
            }
        }
    }
    // the same four legacy tuples on both cc 12.0 cards, none elsewhere
    assert_eq!(selected.len(), 2 * 4);
    for sms in [36, 70] {
        assert_eq!(selected.iter().filter(|row| row.0 == sms).count(), 4);
    }
}

#[test]
fn golden_legacy_module_requests_are_unchanged() {
    for device in DEVICES {
        for area in [
            KernelModule::Fbank,
            KernelModule::Embedding,
            KernelModule::Segmentation,
        ] {
            let ptx = match area {
                KernelModule::Fbank => include_str!("../ptx/fbank.sm75.ptx"),
                KernelModule::Embedding => include_str!("../ptx/embedding.sm75.ptx"),
                _ => include_str!("../ptx/segmentation.sm75.ptx"),
            };
            assert_eq!(
                module_request(device, area),
                Some((
                    PtxTier::Sm75,
                    LoadedArtifact::PtxJit {
                        sha256: ArtifactHash::of(ptx.as_bytes())
                    }
                )),
                "always-on {} on {}",
                area.name(),
                device.capability
            );
        }
        for (area, ptx) in [
            (KernelModule::Lstm, LSTM_PTX),
            (KernelModule::Sincnet, SINC_PTX),
        ] {
            let expected = (device.capability == ComputeCapability::new(12, 0)).then(|| {
                (
                    PtxTier::Sm75,
                    LoadedArtifact::PtxJit {
                        sha256: ArtifactHash::from_hex(ptx),
                    },
                )
            });
            assert_eq!(
                module_request(device, area),
                expected,
                "{} on {}",
                area.name(),
                device.capability
            );
        }
    }
}
