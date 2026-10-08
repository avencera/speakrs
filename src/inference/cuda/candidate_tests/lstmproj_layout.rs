// both compilation targets must use the same physical ownership map
#[path = "../../../../crates/speakrs-cuda-kernels/src/lstmproj/exchange.rs"]
mod device_exchange;

use super::super::lstmproj::layout::{
    AlignedSpan, ExchangeLayout, GATE_COLUMNS, GROUPS, HIDDEN, ProjectionLayout, STATE_TILE,
    Schedule, THREADS, TILE_ROWS, exchange, pack_directions, pack_input_directions, padded_input,
    round_tf32,
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
    assert_eq!(exchange::TILE, device_exchange::TILE);
    assert_eq!(exchange::SLOT, device_exchange::SLOT);
    assert_eq!(exchange::COPIES, device_exchange::COPIES);
    let mut bulk = std::collections::BTreeSet::new();
    for logical in 0..32 * HIDDEN {
        let word = exchange::bulk_word(logical);
        assert_eq!(word, device_exchange::bulk_word(logical));
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
                assert_eq!(
                    word,
                    device_exchange::single_copy(device_exchange::single_base(unit), copy, flag)
                );
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
