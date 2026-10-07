use super::lstm::layout;

/// ONNX gate `[i, o, f, c]` row `g * 128 + u` must land in packed row `u * 4 + p`
/// for packed gate `p` in `[i, f, c, o]`, so each block's four gates of a unit
/// are adjacent and the kernel reads them in `[i, f, c, o]` order
#[test]
fn packing_groups_the_four_gates_of_each_unit() {
    let columns = 3;
    // value = direction * 1e6 + onnx row * 10 + column
    let source: Vec<f32> = (0..2)
        .flat_map(|direction| {
            (0..layout::GATE_COLUMNS).flat_map(move |row| {
                (0..columns).map(move |column| (direction * 1_000_000 + row * 10 + column) as f32)
            })
        })
        .collect();
    let packed = layout::pack_directions(&source, columns);
    let onnx = |gate: usize, unit: usize| gate * layout::HIDDEN + unit;
    for (packed_gate, onnx_gate) in [(0, 0), (1, 2), (2, 3), (3, 1)] {
        for direction in 0..2 {
            let unit = 77;
            let row = direction * layout::GATE_COLUMNS + unit * 4 + packed_gate;
            let expected = (direction * 1_000_000 + onnx(onnx_gate, unit) * 10 + 2) as f32;
            assert_eq!(packed[row * columns + 2], expected);
        }
    }

    let bias: Vec<f32> = (0..2 * 2 * layout::GATE_COLUMNS)
        .map(|i| i as f32)
        .collect();
    let packed = layout::pack_bias(&bias);
    let unit = 5;
    // reverse direction, packed forget gate = ONNX gate 2
    let row = 2 * layout::HIDDEN + unit;
    let expected = bias[2 * layout::GATE_COLUMNS + row] + bias[3 * layout::GATE_COLUMNS + row];
    assert_eq!(packed[layout::GATE_COLUMNS + unit * 4 + 1], expected);
}

#[test]
fn schedule_keeps_concurrent_blocks_resident() {
    // RTX 5070 Ti: 70 SMs, five blocks each; b32 fits in 8-row tiles
    let wide = layout::Schedule::new(32, 350, Some(350)).expect("fits");
    assert_eq!((wide.tile_rows, wide.tiles), (8, 4));
    assert!(wide.concurrent);
    assert_eq!(wide.launches().collect::<Vec<_>>(), [(0, 4)]);

    // b64 needs 16-row tiles there
    assert_eq!(
        layout::Schedule::new(64, 350, Some(350))
            .expect("fits")
            .tile_rows,
        16
    );

    // the whole pair does not fit, so split one direction at a time
    let split = layout::Schedule::new(65, 2 * layout::GROUPS + 1, Some(2 * layout::GROUPS + 1))
        .expect("fits");
    assert!(!split.concurrent);
    assert_eq!(split.tile_rows, layout::TILE_ROWS);
    assert_eq!(split.launches().collect::<Vec<_>>(), [(0, 2), (2, 1)]);

    let serial = layout::Schedule::new(64, layout::GROUPS, Some(layout::GROUPS)).expect("fits");
    assert!(!serial.concurrent);
    assert_eq!(serial.launches().count(), 2);

    assert!(layout::Schedule::new(1, layout::GROUPS - 1, Some(350)).is_err());
}

#[test]
fn unknown_or_reduced_context_budget_selects_sequential() {
    for budget in [None, Some(0), Some(layout::GROUPS), Some(127)] {
        let schedule = layout::Schedule::new(64, 420, budget).expect("single grid fits");
        assert!(!schedule.concurrent);
        assert_eq!(schedule.tile_rows, layout::TILE_ROWS);
        assert_eq!(schedule.launches().collect::<Vec<_>>(), [(0, 2)]);
    }

    let exact = layout::Schedule::new(64, 420, Some(128)).expect("joint budget fits exactly");
    assert!(exact.concurrent);
    assert_eq!(exact.tile_rows, layout::TILE_ROWS);
    let capped = layout::Schedule::new(64, 64, Some(420)).expect("single grid fits");
    assert!(!capped.concurrent);
}

#[test]
fn insufficient_cooperative_capacity_falls_back_only_in_library_allowed_production() {
    use crate::inference::cuda::device::test_support::Builder;
    use crate::inference::cuda::implementation::{BoundaryId, Selected, select};
    use crate::inference::cuda::kernels::{ArtifactHash, LoadedArtifact, ModuleRequest};
    use crate::inference::cuda::{ComputeCapability, CudaError, CudaMath, KernelModule, PtxTier};

    let schedule = || layout::Schedule::new(1, layout::GROUPS - 1, Some(350));
    let device = Builder::new(ComputeCapability::new(12, 0)).build();
    let module = ModuleRequest::new(
        KernelModule::Lstm,
        PtxTier::Sm75,
        LoadedArtifact::PtxJit {
            sha256: ArtifactHash::of(include_str!("ptx/lstm.sm75.ptx").as_bytes()),
        },
    );
    let Selected::Oxide(token) = select(
        BoundaryId::named("lstm.stack"),
        1,
        CudaMath::Fp32,
        &device,
        module,
    )
    .unwrap() else {
        panic!("qualified production token")
    };
    assert!(
        token
            .finish(KernelModule::Lstm, false, schedule())
            .unwrap()
            .is_none()
    );
    assert!(matches!(
        token.finish(KernelModule::Lstm, true, schedule()),
        Err(CudaError::CandidateDeviceUnsupported {
            area: "lstm",
            batch: 1,
            math: CudaMath::Fp32,
            tier: PtxTier::Sm75,
            ..
        })
    ));
}

#[test]
fn schedules_cover_every_tile_without_exceeding_residency() {
    // adapted from the supplied geometry test; this branch has one fixed geometry
    for batch in [1usize, 7, 32, 33, 64, 65] {
        for capacity in [
            layout::GROUPS - 1,
            layout::GROUPS,
            2 * layout::GROUPS,
            140,
            280,
        ] {
            for joint in [
                None,
                Some(0),
                Some(layout::GROUPS),
                Some(capacity),
                Some(420),
            ] {
                let result = layout::Schedule::new(batch, capacity, joint);
                if capacity < layout::GROUPS {
                    assert!(result.is_err());
                    continue;
                }
                let schedule = result.expect("one group fits");
                let mut covered = Vec::new();
                for (first, count) in schedule.launches() {
                    assert!(count * layout::GROUPS <= capacity);
                    covered.extend(first..first + count);
                }
                assert_eq!(
                    covered,
                    (0..batch.div_ceil(schedule.tile_rows)).collect::<Vec<_>>()
                );
                if schedule.concurrent {
                    let budget = joint
                        .expect("concurrency needs a known budget")
                        .min(capacity);
                    assert!(2 * schedule.tiles * layout::GROUPS <= budget);
                }
            }
        }
    }
}
