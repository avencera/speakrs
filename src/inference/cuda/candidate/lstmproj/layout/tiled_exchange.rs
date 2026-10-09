//! Physical words of the eight-window recurrence's hidden-state exchange, shared with
//! the host layout
//!
//! A block owns sixteen adjacent units, so one window row of one producer is exactly
//! one 128-byte line. Rows, parity slots, tiles and directions follow each other with
//! line-sized gaps

/// Words in one producer's line: its sixteen units of one window
pub(crate) const LINE: usize = 16;
/// Words between parity, tile, and direction regions
pub(crate) const PAD: usize = LINE;
/// Words per window row: every unit of the direction
pub(crate) const ROW: usize = 128;
/// Windows per tile
pub(crate) const ROWS: usize = 8;
/// Words per parity, including its trailing line gap
pub(crate) const SLOT: usize = ROWS * ROW + PAD;
/// Words per tile, including its trailing line gap
pub(crate) const TILE: usize = 2 * SLOT + PAD;

/// Word of hidden unit `unit` of tile row `row` within a parity slot
#[inline(always)]
pub(crate) const fn word(row: usize, unit: usize) -> usize {
    row * ROW + unit
}
