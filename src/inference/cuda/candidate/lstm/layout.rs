//! Host-side layout of the oxide LSTM stack: packed gate order and the tile schedule

use super::super::PlanError;

/// Hidden units per direction
pub(crate) const HIDDEN: usize = 128;
/// Gates per hidden unit; packed order `[i, f, c, o]`, the cuDNN order
pub(crate) const GATES: usize = 4;
/// Packed gate columns of one direction: column `unit * GATES + gate`
pub(crate) const GATE_COLUMNS: usize = GATES * HIDDEN;
/// Blocks per batch tile, `GROUPS` in the kernel crate's `lstm` area
pub(crate) const GROUPS: usize = 32;
/// Most windows per tile, `TILE_ROWS` in the kernel crate's `lstm` area
pub(crate) const TILE_ROWS: usize = 32;
/// Hidden-state exchange words per batch tile and direction, `STATE_TILE` in the
/// kernel crate's `lstm` area: two step parities of `[TILE_ROWS, HIDDEN]`
pub(crate) const STATE_TILE: usize = 2 * TILE_ROWS * HIDDEN;

/// ONNX LSTM gates are stored `[i, o, f, c]`; packed gate `g` is ONNX gate
/// `ONNX_GATE[g]`
const ONNX_GATE: [usize; GATES] = [0, 2, 3, 1];

/// The recurrence kernel, for errors
pub(crate) const KERNEL: &str = "spk_lstm_recurrence";

/// How the batch tiles of one layer are launched
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Schedule {
    /// Windows per tile
    pub(crate) tile_rows: usize,
    pub(crate) tiles: usize,
    /// Tiles per cooperative launch
    pub(crate) tiles_per_launch: usize,
    /// Run the reverse direction on the side stream, at the same time as the forward
    /// direction
    pub(crate) concurrent: bool,
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
    pub(crate) fn new(
        batch: usize,
        capacity: usize,
        concurrent_capacity: Option<usize>,
    ) -> Result<Self, PlanError> {
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

        Err(PlanError::DeviceUnsupported {
            reason: format!(
                "needs {GROUPS} co-resident blocks, but the GPU holds {capacity} in a cooperative launch"
            ),
        })
    }

    /// `(first tile, tiles)` of each launch of one direction
    pub(crate) fn launches(self) -> impl Iterator<Item = (usize, usize)> {
        (0..self.tiles)
            .step_by(self.tiles_per_launch)
            .map(move |first| (first, self.tiles_per_launch.min(self.tiles - first)))
    }
}

/// Reorders the gate rows of an ONNX `[2, 4 * 128, columns]` matrix, gates
/// `[i, o, f, c]`, to packed rows `unit * 4 + gate` with gates `[i, f, c, o]`
pub(crate) fn pack_directions(source: &[f32], columns: usize) -> Vec<f32> {
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
pub(crate) fn pack_bias(source: &[f32]) -> Vec<f32> {
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
