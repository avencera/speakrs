//! Scalar tuner alternatives retain the pipeline mode required by the plan

use super::{Area, Entry, Site};
use crate::inference::cuda::candidate::{ConfigPin, DriverCandidate};
use crate::inference::cuda::device::test_support::Builder;
use crate::inference::cuda::implementation::BoundaryId;
use crate::inference::cuda::{ComputeCapability, CudaMath, PtxTier};

#[test]
fn scalar_embedding_alternative_plans_the_requested_tf32_tuple() {
    let device = Builder::new(ComputeCapability::new(8, 9))
        .multiprocessors(34)
        .shared_optin_bytes(101376)
        .build();
    let pin = Area::tuning_fp32_pin(
        BoundaryId::named("resnet.seg_1"),
        32,
        CudaMath::Tf32,
        &device,
        PtxTier::Sm80,
    )
    .unwrap()
    .unwrap();
    let ConfigPin::Segdense(pin) = pin else {
        panic!("segdense pin")
    };
    assert_eq!(pin.entry(), Entry::EmbedB32);
    assert_eq!(pin.config().kernel, "spk_segdense_embed_b32");
    assert_eq!(pin.splits(), Some(34));
    pin.check(Site::Embedding, 32, CudaMath::Tf32, PtxTier::Sm80)
        .unwrap();
    assert!(
        pin.check(Site::Embedding, 32, CudaMath::Fp32, PtxTier::Sm80)
            .is_err()
    );
    assert!(
        pin.check(Site::Embedding, 1, CudaMath::Tf32, PtxTier::Sm80)
            .is_err()
    );
}
