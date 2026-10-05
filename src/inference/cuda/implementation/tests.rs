//! Pin production to the triples accepted for integration, independent of declarations

use super::{PRODUCTION, Selected, Target, select};
use crate::inference::cuda::CudaMath;
use crate::inference::cuda::{ComputeCapability, PtxTier};

#[test]
fn production_selects_exactly_the_qualified_triples() {
    let c32 = [
        "resnet.layer1.0.conv1",
        "resnet.layer1.0.conv2",
        "resnet.layer1.1.conv1",
        "resnet.layer1.1.conv2",
        "resnet.layer1.2.conv1",
        "resnet.layer1.2.conv2",
        "resnet.layer2.0.conv1",
    ];
    let c64 = [
        "resnet.layer2.0.conv2",
        "resnet.layer2.1.conv1",
        "resnet.layer2.1.conv2",
        "resnet.layer2.2.conv1",
        "resnet.layer2.2.conv2",
        "resnet.layer2.3.conv1",
        "resnet.layer2.3.conv2",
    ];
    assert_eq!(
        crate::inference::cuda::candidate::QUALIFIED_BATCHES,
        [1, 7, 32, 33, 64]
    );
    let declared = PRODUCTION
        .iter()
        .flat_map(|coverage| coverage.coverage.entries())
        .flat_map(|entry| entry.layers.iter().copied());
    let layers = c32.into_iter().chain(c64).chain(declared).chain([
        "lstm.stack",
        "sincnet.conv0.abs_pool",
        "resnet.layer3.0.conv1",
        "resnet.conv1",
        "unknown",
    ]);
    for layer in layers {
        for batch in (1..=66).chain([128, usize::MAX]) {
            for math in [CudaMath::Fp32, CudaMath::Tf32] {
                for tier in PtxTier::ALL {
                    for device in [
                        ComputeCapability::new(7, 5),
                        ComputeCapability::new(8, 0),
                        ComputeCapability::new(8, 6),
                        ComputeCapability::new(8, 9),
                        ComputeCapability::new(9, 0),
                        ComputeCapability::new(10, 0),
                        ComputeCapability::new(12, 0),
                        ComputeCapability::new(12, 1),
                        ComputeCapability::new(13, 0),
                    ] {
                        let expected = [1, 32].contains(&batch)
                            && tier == PtxTier::Sm75
                            && device == ComputeCapability::new(12, 0)
                            && (c32.contains(&layer)
                                || c64.contains(&layer) && (batch != 1 || math == CudaMath::Fp32)
                                || ["lstm.stack", "sincnet.conv0.abs_pool"].contains(&layer)
                                    && math == CudaMath::Fp32);
                        let selected = select(layer, batch, math, Target { tier, device }).unwrap();
                        assert_eq!(
                            matches!(selected, Selected::Oxide(_)),
                            expected,
                            "{layer} b{batch} {math:?} {tier} {device}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn invalid_requests_cannot_make_tokens() {
    let target = Target {
        tier: PtxTier::Sm75,
        device: ComputeCapability::new(12, 0),
    };
    assert!(select("", 1, CudaMath::Fp32, target).is_err());
    assert!(select("lstm.stack", 0, CudaMath::Fp32, target).is_err());
}

#[test]
fn records_and_integrated_evidence_are_pinned() {
    let pins = [
        "8f8fa3e3c158771e354aad83f4e42fca6fac998aa192b39a966067a4b0035758",
        "3badc1aec939b0e8f7312786d695bec6445de1dacb1f85e44124bf3ac20356f8",
        "a4d1a2692b78a814cd2da16f084801c3641bfd11c6d095d610f3a8182ad6f675",
    ];
    for (entry, pin) in PRODUCTION.iter().zip(pins) {
        assert_eq!(entry.record, pin);
        assert_eq!(
            entry.der,
            "8066268031afba058d93e305206d5b8225e40c6ab2ebb1052874d607e055646f"
        );
    }
}
