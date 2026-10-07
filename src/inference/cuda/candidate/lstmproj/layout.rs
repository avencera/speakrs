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
mod exchange;

use exchange::{PAD as STATE_PAD, TILE as STATE_TILE};

// check both physical maps in host builds, not only when the device crate compiles
const _: () = assert!(exchange::bulk_word(TILE_ROWS * HIDDEN - 1) < exchange::SLOT);
const _: () = assert!(
    exchange::single_copy(
        exchange::single_base(HIDDEN - 1),
        exchange::COPIES - 1,
        u32::MAX
    ) < exchange::SLOT
);

/// Hidden units per block, `UNITS` in the kernel crate's `lstmproj` area
const UNITS: usize = 8;
/// Threads per block, `THREADS` in the kernel crate's `lstmproj` area
pub(crate) const THREADS: u32 = 32 * UNITS as u32;
/// Blocks per batch tile, `GROUPS` in the kernel crate's `lstmproj` area
pub(crate) const GROUPS: usize = HIDDEN / UNITS;

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
    /// Direction stride including one cache-line gap after its padded tiles
    pub(crate) const fn new(tiles: usize) -> Self {
        Self(tiles * STATE_TILE + STATE_PAD)
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

#[cfg(test)]
mod tests {
    use super::{
        AlignedSpan, ExchangeLayout, GATE_COLUMNS, GROUPS, HIDDEN, ProjectionLayout, STATE_TILE,
        Schedule, THREADS, TILE_ROWS, exchange, pack_directions, pack_input_directions,
        padded_input, round_tf32,
    };

    #[test]
    fn geometry_matches_the_kernel_area() {
        assert_eq!(THREADS, 256);
        assert_eq!(GROUPS, 16);
        assert_eq!(padded_input(60), 64);
        assert_eq!(padded_input(256), 256);
    }

    #[test]
    fn eight_unit_blocks_halve_the_blocks_per_tile_in_the_schedule() {
        // 16 blocks per tile let b64 run 8-row tiles concurrently where four-unit
        // blocks need 16-row tiles
        let wide = Schedule::for_groups(GROUPS, 64, 350, Some(350)).expect("fits");
        assert_eq!((wide.tile_rows, wide.tiles, wide.concurrent), (8, 8, true));
        let legacy = Schedule::new(64, 350, Some(350)).expect("fits");
        assert_eq!(legacy.tile_rows, 16);

        let serial = Schedule::for_groups(GROUPS, 65, GROUPS, None).expect("fits");
        assert_eq!(serial.tile_rows, TILE_ROWS);
        assert!(!serial.concurrent);
        assert_eq!(serial.launches().count(), 3);
        assert!(Schedule::for_groups(GROUPS, 1, GROUPS - 1, Some(350)).is_err());
    }

    #[test]
    fn simt_and_tensor_packing_transpose_the_packed_rows() {
        let columns = 60;
        let source: Vec<f32> = (0..2 * GATE_COLUMNS * columns)
            .map(|i| (i % 977) as f32 * 0.25 - 100.0)
            .collect();
        let rows = pack_directions(&source, columns);
        let simt = pack_input_directions(&source, columns, ProjectionLayout::Simt);
        let tensor = pack_input_directions(&source, columns, ProjectionLayout::Tensor);
        let padded = padded_input(columns);
        for direction in 0..2 {
            let base = direction * GATE_COLUMNS * padded;
            for column in [0, 1, 5, 130, GATE_COLUMNS - 1] {
                for k in 0..padded {
                    let expected = if k < columns {
                        rows[(direction * GATE_COLUMNS + column) * columns + k]
                    } else {
                        0.0
                    };
                    assert_eq!(simt[base + k * GATE_COLUMNS + column], expected);
                    assert_eq!(
                        tensor[base + column * padded + k],
                        round_tf32(expected),
                        "direction {direction} column {column} k {k}"
                    );
                }
            }
        }
    }

    #[test]
    fn tf32_rounding_keeps_ten_mantissa_bits_with_ties_to_even() {
        // 1 + 2^-11 is halfway between two TF32 values; the even neighbour is 1
        assert_eq!(round_tf32(1.0 + 2f32.powi(-11)), 1.0);
        // 1 + 3 * 2^-11 is halfway again; the even neighbour is 1 + 2^-9
        assert_eq!(round_tf32(1.0 + 3.0 * 2f32.powi(-11)), 1.0 + 2f32.powi(-9));
        assert_eq!(round_tf32(1.0 + 2f32.powi(-12)), 1.0);
        assert_eq!(round_tf32(-3.5), -3.5);
        assert!(round_tf32(f32::NAN).is_nan());
        assert_eq!(round_tf32(f32::INFINITY), f32::INFINITY);
        assert_eq!(round_tf32(1.0 + 2f32.powi(-12)).to_bits() & 0x1fff, 0);
    }

    #[test]
    fn aligned_span_starts_on_a_two_mebibyte_boundary_inside_its_allocation() {
        // allocations are naturally aligned for their element type
        let align = 2 * 1024 * 1024;
        for address in [0u64, 8, 256, align as u64 - 8, align as u64 + 512] {
            let len = 1000;
            let span = AlignedSpan::<u64>::new(address, len);
            let range = span.range();
            assert_eq!((address as usize + range.start * 8) % align, 0);
            assert_eq!(range.len(), len);
            assert!(range.end <= AlignedSpan::<u64>::allocation_len(len));
        }
    }

    #[test]
    fn exchange_words_stay_inside_their_tile_and_producer_regions() {
        let mut bulk = std::collections::BTreeSet::new();
        for logical in 0..32 * HIDDEN {
            let word = exchange::bulk_word(logical);
            assert!(word < exchange::SLOT);
            assert!(bulk.insert(word), "two hidden values share word {word}");
            // eight consecutive units of one row share one producer's line
            assert_eq!(
                word / exchange::LINE,
                exchange::bulk_word(logical / 8 * 8) / exchange::LINE
            );
        }

        for flag in [1, 2, 3, 590, u32::MAX] {
            let mut single = std::collections::BTreeSet::new();
            for unit in 0..HIDDEN {
                for copy in 0..exchange::COPIES {
                    let word = exchange::single_copy(exchange::single_base(unit), copy, flag);
                    assert!(word < exchange::SLOT);
                    // a producer group of eight units keeps its own 4 KiB region
                    assert_eq!(word / 512, unit / 8);
                    assert!(single.insert(word), "flag {flag}: copies collide at {word}");
                }
            }
        }

        let layout = ExchangeLayout::new(3);
        assert_eq!(layout.direction(1), 3 * STATE_TILE + exchange::PAD);
        assert_eq!(layout.words(), 2 * layout.direction(1));
    }
}
