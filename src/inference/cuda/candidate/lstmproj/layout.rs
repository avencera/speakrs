//! Host-side layout of the library-free LSTM stack: block geometry, the padded
//! hidden-state exchange, aligned views and the projection weight layouts
//!
//! The packed gate order, bias packing and tile schedule are the `lstm` area's

use std::marker::PhantomData;
use std::ops::Range;

pub(crate) use super::super::lstm::layout::{
    GATE_COLUMNS, GATES, HIDDEN, KERNEL, ONNX_GATE, Schedule, TILE_ROWS, pack_bias, pack_directions,
};

/// Physical word ownership, shared with the kernel crate's `lstmproj` area
pub(crate) mod exchange;

pub(crate) use exchange::{PAD as STATE_PAD, TILE as STATE_TILE};

/// Physical words of the eight-window recurrence, shared with the kernel crate's
/// `lstmproj` area
pub(crate) mod tiled_exchange;

// check both physical maps in host builds, not only when the device crate compiles
const _: () = assert!(exchange::bulk_word(TILE_ROWS * HIDDEN - 1) < exchange::SLOT);
const _: () = assert!(
    exchange::single_copy(
        exchange::single_base(HIDDEN - 1),
        exchange::COPIES - 1,
        u32::MAX
    ) < exchange::SLOT
);

// every producer's line and the last unit of the last row stay inside one parity slot
const _: () = assert!(tiled_exchange::LINE == HIDDEN / TILED_GROUPS);
const _: () = assert!(tiled_exchange::word(TILED_ROWS - 1, HIDDEN - 1) < tiled_exchange::SLOT);

/// Hidden units per block, `UNITS` in the kernel crate's `lstmproj` area
const UNITS: usize = 8;
/// Threads per block, `THREADS` in the kernel crate's `lstmproj` area
pub(crate) const THREADS: u32 = 32 * UNITS as u32;
/// Blocks per batch tile, `GROUPS` in the kernel crate's `lstmproj` area
pub(crate) const GROUPS: usize = HIDDEN / UNITS;

/// The batched recurrence entry, `spk_lstm_recurrence_tiled` in the kernel crate's
/// `lstmproj` area
pub(crate) const TILED_KERNEL: &str = "spk_lstm_recurrence_tiled";
/// Blocks per eight-window tile of the batched recurrence, `tiled::GROUPS`
pub(crate) const TILED_GROUPS: usize = 8;
/// Threads per block of the batched recurrence, `tiled::THREADS`
pub(crate) const TILED_THREADS: u32 = 256;
/// Windows per tile of the batched recurrence, `tiled::TILE_ROWS`
pub(crate) const TILED_ROWS: usize = tiled_exchange::ROWS;

/// Dynamic shared bytes of the TF32 projection tile
pub(crate) const TENSOR_SHARED_BYTES: u32 = 61_440;

/// A subview whose active address is 2 MiB aligned, independent of allocator order
#[derive(Debug, Clone)]
pub(crate) struct AlignedSpan<T>(Range<usize>, PhantomData<T>);

impl<T> AlignedSpan<T> {
    const ALIGN_BYTES: usize = 2 * 1024 * 1024;

    /// Allocation size including slack for any naturally aligned element address
    pub(crate) const fn allocation_len(len: usize) -> usize {
        const { assert!(size_of::<T>() > 0 && Self::ALIGN_BYTES % size_of::<T>() == 0) };
        len + Self::ALIGN_BYTES / size_of::<T>() - 1
    }

    /// Active span inside the owning allocation at `address`, excluding alignment slack
    pub(crate) fn new(address: u64, len: usize) -> Self {
        const { assert!(size_of::<T>() > 0 && Self::ALIGN_BYTES % size_of::<T>() == 0) };
        let remainder = address as usize % Self::ALIGN_BYTES;
        let offset = (Self::ALIGN_BYTES - remainder) % Self::ALIGN_BYTES / size_of::<T>();
        Self(offset..offset + len, PhantomData)
    }

    /// The active subview's bounds in elements of the owning allocation
    pub(crate) fn range(&self) -> Range<usize> {
        self.0.clone()
    }
}

/// Cache-line-separated exchange directions, tiles and parity slots
#[derive(Debug, Clone, Copy)]
pub(crate) struct ExchangeLayout(usize);

impl ExchangeLayout {
    /// Direction stride including one cache-line gap after `tiles` padded tiles of
    /// `tile_words` words each
    pub(crate) const fn new(tiles: usize, tile_words: usize) -> Self {
        Self(tiles * tile_words + STATE_PAD)
    }

    /// Words to clear, including padding but excluding allocation alignment slack
    pub(crate) const fn words(self) -> usize {
        2 * self.0
    }

    /// Word offset of one direction within the aligned active subview
    pub(crate) const fn direction(self, direction: usize) -> usize {
        direction * self.0
    }
}

/// Input-weight layout consumed by a projection kernel
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ProjectionLayout {
    /// K-major weights for the FP32 SIMT tiles
    Simt,
    /// Gate-major weights, rounded to TF32, for the column-major matrix fragments
    Tensor,
}

/// Input width rounded to a complete projection staging tile
pub(crate) const fn padded_input(columns: usize) -> usize {
    columns.div_ceil(16) * 16
}

/// Packs both ONNX input matrices, `[2, 4 * 128, columns]` with gates `[i, o, f, c]`,
/// in the packed gate order and the projection's storage order
///
/// Zero padding completes the last K tile without moving bias addition ahead of
/// the recurrent sum
pub(crate) fn pack_input_directions(
    source: &[f32],
    columns: usize,
    layout: ProjectionLayout,
) -> Vec<f32> {
    let padded = padded_input(columns);
    let direction_len = GATE_COLUMNS * padded;
    let mut packed = vec![0.0; 2 * direction_len];
    for direction in 0..2 {
        for unit in 0..HIDDEN {
            for (gate, onnx_gate) in ONNX_GATE.into_iter().enumerate() {
                let column = unit * GATES + gate;
                let source_row = (direction * GATE_COLUMNS + onnx_gate * HIDDEN + unit) * columns;
                for k in 0..columns {
                    let value = source[source_row + k];
                    let (offset, value) = match layout {
                        ProjectionLayout::Simt => (k * GATE_COLUMNS + column, value),
                        ProjectionLayout::Tensor => (column * padded + k, round_tf32(value)),
                    };
                    packed[direction * direction_len + offset] = value;
                }
            }
        }
    }

    packed
}

/// Rounds an immutable weight to TF32 once, round to nearest with ties to even
pub(crate) fn round_tf32(value: f32) -> f32 {
    let bits = value.to_bits();
    if bits & 0x7f80_0000 == 0x7f80_0000 {
        return value;
    }

    f32::from_bits((bits + 0x0fff + ((bits >> 13) & 1)) & 0xffff_e000)
}
