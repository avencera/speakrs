//! Host-side layout of the oxide LSTM stack: packed gate order and the tile schedule

use crate::inference::cuda::CudaError;

/// Hidden units per direction
pub(super) const HIDDEN: usize = 128;
/// Gates per hidden unit; packed order `[i, f, c, o]`, the cuDNN order
pub(super) const GATES: usize = 4;
/// Packed gate columns of one direction: column `unit * GATES + gate`
pub(super) const GATE_COLUMNS: usize = GATES * HIDDEN;
/// Blocks per batch tile, `GROUPS` in the kernel crate's `lstm` area
pub(super) const GROUPS: usize = 32;
/// Most windows per tile, `TILE_ROWS` in the kernel crate's `lstm` area
pub(super) const TILE_ROWS: usize = 32;
/// Hidden-state exchange words per batch tile and direction, `STATE_TILE` in the
/// kernel crate's `lstm` area: two step parities of `[TILE_ROWS, HIDDEN]`
pub(super) const STATE_TILE: usize = 2 * TILE_ROWS * HIDDEN;

/// ONNX LSTM gates are stored `[i, o, f, c]`; packed gate `g` is ONNX gate
/// `ONNX_GATE[g]`
const ONNX_GATE: [usize; GATES] = [0, 2, 3, 1];

/// The recurrence kernel, for errors
pub(super) const KERNEL: &str = "spk_lstm_recurrence";

/// How the batch tiles of one layer are launched
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Schedule {
    /// Windows per tile
    pub(super) tile_rows: usize,
    pub(super) tiles: usize,
    /// Tiles per cooperative launch
    pub(super) tiles_per_launch: usize,
    /// Run the reverse direction on the side stream, at the same time as the forward
    /// direction
    pub(super) concurrent: bool,
}

impl Schedule {
    /// Tile heights to try, smallest first; a smaller tile gives each SM more
    /// independent windows to switch between while one tile waits
    const TILE_HEIGHTS: [usize; 3] = [8, 16, TILE_ROWS];

    /// Picks the smallest tile height whose blocks for both directions fit this
    /// context together, else splits the largest tiles into sequential launches
    ///
    /// `capacity` limits one cooperative grid. `concurrent_capacity` is the joint
    /// block budget from this context's SM count, including MPS or execution affinity;
    /// an unknown budget must select sequential launches
    ///
    /// Multiple pipelines in one process or other spinning cooperative work on the
    /// same device can still contend on pre-Hopper GPUs, which can start partly
    /// resident grids. Persistent library RNN algorithms have the same class of risk
    pub(super) fn new(
        batch: usize,
        capacity: usize,
        concurrent_capacity: Option<usize>,
    ) -> Result<Self, CudaError> {
        let pair = 2 * GROUPS;
        let joint = concurrent_capacity.unwrap_or(0).min(capacity);
        for tile_rows in Self::TILE_HEIGHTS {
            let tiles = batch.div_ceil(tile_rows);
            if tiles * pair <= joint {
                return Ok(Self {
                    tile_rows,
                    tiles,
                    tiles_per_launch: tiles,
                    concurrent: true,
                });
            }
        }

        let tiles = batch.div_ceil(TILE_ROWS);
        if GROUPS <= capacity {
            return Ok(Self {
                tile_rows: TILE_ROWS,
                tiles,
                tiles_per_launch: capacity / GROUPS,
                concurrent: false,
            });
        }

        Err(CudaError::Unsupported {
            context: KERNEL,
            reason: format!(
                "needs {GROUPS} co-resident blocks, but the GPU holds {capacity} in a cooperative launch"
            ),
        })
    }

    /// `(first tile, tiles)` of each launch of one direction
    pub(super) fn launches(self) -> impl Iterator<Item = (usize, usize)> {
        (0..self.tiles)
            .step_by(self.tiles_per_launch)
            .map(move |first| (first, self.tiles_per_launch.min(self.tiles - first)))
    }
}

/// Reorders the gate rows of an ONNX `[2, 4 * 128, columns]` matrix, gates
/// `[i, o, f, c]`, to packed rows `unit * 4 + gate` with gates `[i, f, c, o]`
pub(super) fn pack_directions(source: &[f32], columns: usize) -> Vec<f32> {
    let direction_len = GATE_COLUMNS * columns;
    let mut packed = Vec::with_capacity(source.len());
    for direction in source.chunks_exact(direction_len) {
        for unit in 0..HIDDEN {
            for gate in ONNX_GATE {
                let row = gate * HIDDEN + unit;
                packed.extend_from_slice(&direction[row * columns..(row + 1) * columns]);
            }
        }
    }
    packed
}

/// Packs ONNX `B`, `[2, 8 * 128]` (input biases then recurrent biases per direction),
/// into one `Wb + Rb` per packed column
///
/// ONNX Runtime's CPU LSTM also adds the two biases before it uses them
pub(super) fn pack_bias(source: &[f32]) -> Vec<f32> {
    let mut packed = Vec::with_capacity(source.len() / 2);
    for direction in source.as_chunks::<{ 2 * GATE_COLUMNS }>().0 {
        let (input, recurrent) = direction.split_at(GATE_COLUMNS);
        for unit in 0..HIDDEN {
            for gate in ONNX_GATE {
                let row = gate * HIDDEN + unit;
                packed.push(input[row] + recurrent[row]);
            }
        }
    }
    packed
}

// the harness scan refuses `cfg(test)` in candidate files, so the tests name every
// item by path and the module is empty outside test builds
mod tests {

    /// ONNX gate `[i, o, f, c]` row `g * 128 + u` must land in packed row `u * 4 + p`
    /// for packed gate `p` in `[i, f, c, o]`, so each block's four gates of a unit
    /// are adjacent and the kernel reads them in `[i, f, c, o]` order
    #[test]
    fn packing_groups_the_four_gates_of_each_unit() {
        let columns = 3;
        // value = direction * 1e6 + onnx row * 10 + column
        let source: Vec<f32> = (0..2)
            .flat_map(|direction| {
                (0..super::GATE_COLUMNS).flat_map(move |row| {
                    (0..columns)
                        .map(move |column| (direction * 1_000_000 + row * 10 + column) as f32)
                })
            })
            .collect();
        let packed = super::pack_directions(&source, columns);
        let onnx = |gate: usize, unit: usize| gate * super::HIDDEN + unit;
        for (packed_gate, onnx_gate) in [(0, 0), (1, 2), (2, 3), (3, 1)] {
            for direction in 0..2 {
                let unit = 77;
                let row = direction * super::GATE_COLUMNS + unit * 4 + packed_gate;
                let expected = (direction * 1_000_000 + onnx(onnx_gate, unit) * 10 + 2) as f32;
                assert_eq!(packed[row * columns + 2], expected);
            }
        }

        let bias: Vec<f32> = (0..2 * 2 * super::GATE_COLUMNS).map(|i| i as f32).collect();
        let packed = super::pack_bias(&bias);
        let unit = 5;
        // reverse direction, packed forget gate = ONNX gate 2
        let row = 2 * super::HIDDEN + unit;
        let expected = bias[2 * super::GATE_COLUMNS + row] + bias[3 * super::GATE_COLUMNS + row];
        assert_eq!(packed[super::GATE_COLUMNS + unit * 4 + 1], expected);
    }

    #[test]
    fn schedule_keeps_concurrent_blocks_resident() {
        // RTX 5070 Ti: 70 SMs, five blocks each; b32 fits in 8-row tiles
        let wide = super::Schedule::new(32, 350, Some(350)).expect("fits");
        assert_eq!((wide.tile_rows, wide.tiles), (8, 4));
        assert!(wide.concurrent);
        assert_eq!(wide.launches().collect::<Vec<_>>(), [(0, 4)]);

        // b64 needs 16-row tiles there
        assert_eq!(
            super::Schedule::new(64, 350, Some(350))
                .expect("fits")
                .tile_rows,
            16
        );

        // the whole pair does not fit, so split one direction at a time
        let split = super::Schedule::new(65, 2 * super::GROUPS + 1, Some(2 * super::GROUPS + 1))
            .expect("fits");
        assert!(!split.concurrent);
        assert_eq!(split.tile_rows, super::TILE_ROWS);
        assert_eq!(split.launches().collect::<Vec<_>>(), [(0, 2), (2, 1)]);

        let serial = super::Schedule::new(64, super::GROUPS, Some(super::GROUPS)).expect("fits");
        assert!(!serial.concurrent);
        assert_eq!(serial.launches().count(), 2);

        assert!(super::Schedule::new(1, super::GROUPS - 1, Some(350)).is_err());
    }

    #[test]
    fn unknown_or_reduced_context_budget_selects_sequential() {
        for budget in [None, Some(0), Some(super::GROUPS), Some(127)] {
            let schedule = super::Schedule::new(64, 420, budget).expect("single grid fits");
            assert!(!schedule.concurrent);
            assert_eq!(schedule.tile_rows, super::TILE_ROWS);
            assert_eq!(schedule.launches().collect::<Vec<_>>(), [(0, 2)]);
        }

        let exact = super::Schedule::new(64, 420, Some(128)).expect("joint budget fits exactly");
        assert!(exact.concurrent);
        assert_eq!(exact.tile_rows, super::TILE_ROWS);
        let capped = super::Schedule::new(64, 64, Some(420)).expect("single grid fits");
        assert!(!capped.concurrent);
    }
}
