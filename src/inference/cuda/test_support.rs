//! Implementation selection for direct CUDA development tests

use super::implementation::{Choice, Selection};
use super::{CudaError, CudaRuntime, CudaSegmentation};

/// Select the complete stack and invalidate any captured graph
pub(crate) fn select_lstm(
    model: &mut CudaSegmentation,
    runtime: &CudaRuntime,
    shape: [usize; 2],
    choice_name: &str,
) -> Result<(), CudaError> {
    let choice = match choice_name {
        "Library" => Choice::Library,
        "Oxide" => Choice::Oxide(Selection::Explicit),
        _ => panic!("unknown development implementation"),
    };
    model.select_lstm(runtime, shape, choice)
}
