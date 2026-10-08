//! Captured graph inspection for qualification tests

use super::{CapturedGraph, CudaGraph};

impl CapturedGraph {
    /// The captured graph, for the locked qualification driver's replays
    pub(crate) fn inner(&self) -> &CudaGraph {
        &self.0
    }
}
