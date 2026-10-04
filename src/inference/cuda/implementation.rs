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
    Oxide(Selection),
    /// Planted faults exist only in the qualification test binary
    #[cfg(test)]
    Mutant(super::test_support::Mutant),
}

/// Why the candidate was selected; explicit qualification cannot fall back
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Selection {
    /// Selected from the qualified production coverage
    Production,
    /// Selected by the qualification harness
    #[cfg(test)]
    Explicit,
}

impl Choice {
    /// Whether dispatch should plan a registered candidate
    pub(crate) fn is_candidate(self) -> bool {
        matches!(self, Self::Oxide(_))
    }
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
        Choice::Oxide(Selection::Production)
    } else {
        Choice::Library
    }
}

#[cfg(test)]
mod tests;
