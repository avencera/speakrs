//! Golden production selection, frozen from the PR #36 table at 0cf5403
//!
//! The expectations are literal and independent of the production table. Only
//! `observe` and `module_request` adapt the selection API under test

use super::{AreaTarget, ArtifactRequest, PlanRequest, Selected, Target};
use crate::inference::cuda::kernels::{ArtifactHash, LoadedArtifact};
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

const RESNET_RECORD: &str = "8f8fa3e3c158771e354aad83f4e42fca6fac998aa192b39a966067a4b0035758";
const LSTM_RECORD: &str = "3badc1aec939b0e8f7312786d695bec6445de1dacb1f85e44124bf3ac20356f8";
const SINC_RECORD: &str = "a4d1a2692b78a814cd2da16f084801c3641bfd11c6d095d610f3a8182ad6f675";
const INTEGRATED_DER: &str = "8066268031afba058d93e305206d5b8225e40c6ab2ebb1052874d607e055646f";
const RESNET_PTX: &str = "dd6449c0129f9a03bf691c0338611b50b651ab714d5803caedea87de3c72b6b7";
const LSTM_PTX: &str = "72945743a3c1b915c05d8ea21b438dfd860fa9487401c447fb24fd48802916fa";
const SINC_PTX: &str = "967bc6893f80da84d8d4d288f2cf1ca3336ca09c4386495cab722beb0ac87247";

/// What production selection decided and which artifact it asked the loader for
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Outcome {
    Library,
    Oxide {
        record: &'static str,
        der: &'static str,
        tier: PtxTier,
        artifact: LoadedArtifact,
    },
}

/// Every boundary the models ask production selection about, with its owning area
fn model_boundaries() -> Vec<(KernelModule, String)> {
    // the same names and order as `Trunk::load`
    let mut boundaries = vec![(KernelModule::Resnet, "resnet.conv1".to_owned())];
    for (stage, blocks) in [(1, 3), (2, 4), (3, 6), (4, 3)] {
        for block in 0..blocks {
            let prefix = format!("resnet.layer{stage}.{block}");
            boundaries.push((KernelModule::Resnet, format!("{prefix}.conv1")));
            boundaries.push((KernelModule::Resnet, format!("{prefix}.conv2")));
            if block == 0 && stage > 1 {
                boundaries.push((KernelModule::Resnet, format!("{prefix}.shortcut.0")));
            }
        }
    }
    boundaries.extend([
        (KernelModule::Sincnet, "sincnet.conv0.abs_pool".to_owned()),
        (KernelModule::Segmentation, "sincnet.conv1".to_owned()),
        (KernelModule::Segmentation, "sincnet.conv2".to_owned()),
        (KernelModule::Lstm, "lstm.stack".to_owned()),
    ]);
    boundaries
}

/// Model and workspace batch sizes, including every harness stress batch
fn batches() -> impl Iterator<Item = usize> {
    (1..=66).chain([96, 128, 1024])
}

fn legacy(record: &'static str, ptx: &str) -> Outcome {
    Outcome::Oxide {
        record,
        der: INTEGRATED_DER,
        tier: PtxTier::Sm75,
        artifact: LoadedArtifact::PtxJit {
            sha256: ArtifactHash::from_hex(ptx),
        },
    }
}

/// The 52 PR #36 tuples, written out without consulting any coverage declaration
fn expected(device: Device, boundary: &str, batch: usize, math: CudaMath) -> Outcome {
    const C32: [&str; 6] = [
        "resnet.layer1.0.conv1",
        "resnet.layer1.0.conv2",
        "resnet.layer1.1.conv1",
        "resnet.layer1.1.conv2",
        "resnet.layer1.2.conv1",
        "resnet.layer1.2.conv2",
    ];
    const C64: [&str; 7] = [
        "resnet.layer2.0.conv2",
        "resnet.layer2.1.conv1",
        "resnet.layer2.1.conv2",
        "resnet.layer2.2.conv1",
        "resnet.layer2.2.conv2",
        "resnet.layer2.3.conv1",
        "resnet.layer2.3.conv2",
    ];
    if device.capability != ComputeCapability::new(12, 0) || ![1, 32].contains(&batch) {
        return Outcome::Library;
    }
    let fp32 = math == CudaMath::Fp32;
    let selected = match boundary {
        layer if C32.contains(&layer) => Some(legacy(RESNET_RECORD, RESNET_PTX)),
        "resnet.layer2.0.conv1" => {
            (batch == 32 || !fp32).then(|| legacy(RESNET_RECORD, RESNET_PTX))
        }
        layer if C64.contains(&layer) => {
            (batch == 32 || fp32).then(|| legacy(RESNET_RECORD, RESNET_PTX))
        }
        "lstm.stack" => fp32.then(|| legacy(LSTM_RECORD, LSTM_PTX)),
        "sincnet.conv0.abs_pool" => fp32.then(|| legacy(SINC_RECORD, SINC_PTX)),
        _ => None,
    };
    selected.unwrap_or(Outcome::Library)
}

/// Run production selection with a loader that succeeds exactly as requested
fn observe(
    device: Device,
    area: KernelModule,
    boundary: &str,
    batch: usize,
    math: CudaMath,
    loader: Loader,
) -> Result<(Outcome, usize), CudaError> {
    let limit = PtxTier::select(device.capability, None)?;
    let (tier, _) = area.variants().resolve(area, limit, device.capability)?;
    let location = AreaTarget {
        tier,
        device: device.capability,
    };
    let mut loads = 0;
    let selected =
        PlanRequest::Production.resolve(area, boundary, batch, math, location, |request| {
            loads += 1;
            let ArtifactRequest::Pinned(artifact) = request else {
                panic!("production requested a non-pinned artifact")
            };
            match loader {
                Loader::Exact => Ok(Target {
                    tier,
                    device: device.capability,
                    artifact,
                }),
                Loader::Refuses => Err(CudaError::ArtifactUnavailable {
                    module: area.name(),
                    artifact,
                }),
            }
        })?;
    let outcome = match selected {
        Selected::Library => Outcome::Library,
        Selected::Oxide(token) => Outcome::Oxide {
            record: token.record,
            der: token.der,
            tier: token.target.tier,
            artifact: token.target.artifact,
        },
        Selected::Mutant(_) => panic!("production cannot select a mutant"),
    };
    Ok((outcome, loads))
}

/// The artifact a production module load requests before any driver work, if any
fn module_request(device: Device, area: KernelModule) -> Option<(PtxTier, LoadedArtifact)> {
    let limit = PtxTier::select(device.capability, None).unwrap();
    let (tier, ptx) = area
        .variants()
        .resolve(area, limit, device.capability)
        .unwrap();
    let owner = super::artifact_owner(
        area,
        AreaTarget {
            tier,
            device: device.capability,
        },
    )?;
    let ArtifactRequest::Pinned(artifact) = owner.request(ptx) else {
        panic!("production module requests are pinned")
    };
    Some((tier, artifact))
}

#[derive(Debug, Clone, Copy)]
enum Loader {
    Exact,
    Refuses,
}

#[test]
fn golden_production_selection_is_unchanged() {
    let mut selected = Vec::new();
    for device in DEVICES {
        for (area, boundary) in model_boundaries() {
            for batch in batches() {
                for math in [CudaMath::Fp32, CudaMath::Tf32] {
                    let expected = expected(device, &boundary, batch, math);
                    let (actual, loads) =
                        observe(device, area, &boundary, batch, math, Loader::Exact).unwrap();
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

                    // production keeps today's Library fallback when the pinned load fails
                    let (refused, loads) =
                        observe(device, area, &boundary, batch, math, Loader::Refuses).unwrap();
                    assert_eq!((refused, loads), (Outcome::Library, 1));
                }
            }
            for math in [CudaMath::Fp32, CudaMath::Tf32] {
                assert!(observe(device, area, &boundary, 0, math, Loader::Exact).is_err());
            }
        }
    }
    // the same 52 tuples on both cc 12.0 cards, none elsewhere
    assert_eq!(selected.len(), 2 * 52);
    for sms in [36, 70] {
        assert_eq!(selected.iter().filter(|row| row.0 == sms).count(), 52);
    }
}

#[test]
fn golden_module_requests_are_unchanged() {
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
            (KernelModule::Resnet, RESNET_PTX),
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
