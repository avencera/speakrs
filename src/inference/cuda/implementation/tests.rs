//! Pin production to the triples accepted for integration, independent of declarations

use super::{Choice, PRODUCTION, Selection, production};
use crate::inference::cuda::CudaMath;

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
        .flat_map(|coverage| coverage.entries())
        .flat_map(|entry| entry.layers.iter().copied());
    let layers = c32.into_iter().chain(c64).chain(declared).chain([
        "lstm.stack",
        "sincnet.conv0.abs_pool",
        "resnet.layer3.0.conv1",
        "resnet.conv1",
        "unknown",
    ]);
    for layer in layers {
        for batch in (0..=66).chain([128, usize::MAX]) {
            for math in [CudaMath::Fp32, CudaMath::Tf32] {
                let qualified_batch = [1, 7, 32, 33, 64].contains(&batch);
                let expected = qualified_batch
                    && (c32.contains(&layer)
                        || c64.contains(&layer) && (batch != 1 || math == CudaMath::Fp32)
                        || ["lstm.stack", "sincnet.conv0.abs_pool"].contains(&layer)
                            && math == CudaMath::Fp32);
                assert_eq!(
                    production(layer, batch, math) == Choice::Oxide(Selection::Production),
                    expected,
                    "{layer} b{batch} {math:?}"
                );
            }
        }
    }
}
