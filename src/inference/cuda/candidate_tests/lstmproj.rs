use super::super::LstmProjection;
use crate::inference::cuda::{ComputeCapability, CudaMath, PtxTier};

use super::super::lstmproj::{ProjectionTarget, projection_rule};

const FRAMES: usize = 589;

fn target(major: u32, minor: u32, tier: PtxTier) -> ProjectionTarget {
    ProjectionTarget {
        capability: ComputeCapability::new(major, minor),
        shared_optin_bytes: 99 << 10,
        tier,
    }
}

fn rule(target: ProjectionTarget, batch: usize, math: CudaMath) -> LstmProjection {
    projection_rule(target, batch, batch * FRAMES, math)
}

#[test]
fn fp32_mode_never_uses_tensor_projections() {
    for tier in PtxTier::ALL {
        let device = target(8, 0, tier);
        assert_eq!(rule(device, 1, CudaMath::Fp32), LstmProjection::Small);
        assert_eq!(rule(device, 3, CudaMath::Fp32), LstmProjection::Small);
        assert_eq!(rule(device, 4, CudaMath::Fp32), LstmProjection::Large);
        assert_eq!(rule(device, 32, CudaMath::Fp32), LstmProjection::Large);
    }
}

#[test]
fn tf32_tensor_projections_need_the_sm80_tier_shared_memory_and_rows() {
    assert_eq!(
        rule(target(8, 0, PtxTier::Sm80), 32, CudaMath::Tf32),
        LstmProjection::Tensor
    );
    assert_eq!(
        rule(target(8, 6, PtxTier::Sm75), 32, CudaMath::Tf32),
        LstmProjection::Large
    );
    assert_eq!(
        rule(target(8, 0, PtxTier::Sm80), 1, CudaMath::Tf32),
        LstmProjection::Small
    );
    let small_shared = ProjectionTarget {
        shared_optin_bytes: 48 << 10,
        ..target(8, 6, PtxTier::Sm80)
    };
    assert_eq!(
        rule(small_shared, 32, CudaMath::Tf32),
        LstmProjection::Large
    );
}

#[test]
fn measured_tf32_accuracy_failures_use_fp32_projections() {
    for tier in [PtxTier::Sm80, PtxTier::Sm120] {
        for batch in [4, 7, 32, 33, 64] {
            assert_eq!(
                rule(target(12, 0, tier), batch, CudaMath::Tf32),
                LstmProjection::Large,
                "cc 12.0 {tier} b{batch}"
            );
        }
    }

    let ada = target(8, 9, PtxTier::Sm80);
    assert_eq!(rule(ada, 32, CudaMath::Tf32), LstmProjection::Large);
    assert_eq!(rule(ada, 33, CudaMath::Tf32), LstmProjection::Tensor);
    assert_eq!(rule(ada, 64, CudaMath::Tf32), LstmProjection::Tensor);
}
