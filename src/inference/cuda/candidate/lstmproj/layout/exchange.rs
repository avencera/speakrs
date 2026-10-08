//! Physical ownership of scalar flag/value words, shared with the host layout

/// Words in one exclusive 128-byte producer line
pub(super) const LINE: usize = 16;
/// Words between parity, tile, and direction regions
pub(super) const PAD: usize = LINE;
/// Words per padded hidden row
pub(super) const ROW: usize = 16 * LINE;
/// Words per parity, including its trailing line gap
pub(super) const SLOT: usize = 32 * ROW + PAD;
/// Words per tile, including its trailing line gap
pub(super) const TILE: usize = 2 * SLOT + PAD;
/// Read copies of the one-row vector; each copy serves four consumer blocks
pub(super) const COPIES: usize = 4;

/// Physical word for a multi-row tile's packed hidden value
#[inline(always)]
pub(super) const fn bulk_word(logical: usize) -> usize {
    let group = logical / 8;
    // alternate the payload's sector without sharing its line with another producer
    let half = (group ^ (logical / 128)) & 1;
    group * LINE + half * 8 + logical % 8
}

/// One producer's line colour inside its exclusive 4 KiB one-row region
#[inline(always)]
pub(super) const fn single_base(unit: usize) -> usize {
    let group = unit / 8;
    // multiplication by three in GF(16) permutes adjacent producer line colours
    let colour = group ^ ((group << 1) & 15) ^ ((group >> 3) * 3);
    let colour = colour | ((colour & 1) << 4);
    group * 512 + colour * LINE + unit % 8
}

/// Read-copy word inside the same producer region and step-parity slot
#[inline(always)]
pub(super) const fn single_copy(base: usize, copy: usize, flag: u32) -> usize {
    // same-parity steps visit every line colour; the odd multiplier permutes 0..32
    // all XOR bits are above the unit bits and below the producer-region bits
    let phase = (((flag >> 1).wrapping_mul(11) & 31) as usize) * LINE;
    base ^ (copy * 64) ^ phase
}
