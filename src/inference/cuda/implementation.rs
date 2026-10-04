//! Which implementation runs at each of the three qualified operator boundaries
//!
//! The boundaries are every eligible ResNet 3x3 convolution, the Sinc producer and the
//! complete four-layer bidirectional LSTM stack. Locked dispatch code in
//! `embedding/dispatch.rs` and `segmentation/dispatch.rs` reads the choice; the
//! candidate interface lives in `candidate.rs`

use super::CudaMath;
use super::candidate::{
    ConvCandidate, ConvOxide, Coverage, LstmCandidate, LstmOxide, SincCandidate, SincOxide,
};

/// The implementation selected for one boundary
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Choice {
    /// The current cuDNN path
    #[default]
    Library,
    /// The registered candidate, for the layer, batch and math triples its coverage
    /// declares; every other triple still runs the Library path
    Oxide,
    /// Planted faults exist only in the qualification test binary
    #[cfg(test)]
    Mutant(super::test_support::Mutant),
}

/// Qualified production coverage, shared with the harness candidate declarations
///
/// Selection includes the boundary, batch class and math mode. Sharing coverage
/// prevents production from selecting an unqualified triple
const PRODUCTION: &[Coverage] = &[
    ConvOxide::COVERAGE,
    LstmOxide::COVERAGE,
    SincOxide::COVERAGE,
];

/// The production choice for one boundary, batch class and math mode
pub(crate) fn production(boundary: &str, batch: usize, math: CudaMath) -> Choice {
    if PRODUCTION
        .iter()
        .any(|coverage| coverage.covers(boundary, batch, math))
    {
        Choice::Oxide
    } else {
        Choice::Library
    }
}

#[cfg(test)]
mod tests;
