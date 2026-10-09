//! Raw implemented selection for routing comparisons, without production policy

use super::select_from;
use crate::inference::cuda::implementation::{BoundaryId, Modules, Selected};
use crate::inference::cuda::{CudaError, CudaMath};

/// Select implemented coverage without applying unmeasured-device defaults
pub(in crate::inference::cuda::implementation) fn select(
    boundary: BoundaryId,
    batch: usize,
    math: CudaMath,
    mut modules: impl Modules,
) -> Result<Selected, CudaError> {
    select_from(
        &super::areas(),
        boundary,
        batch,
        math,
        &mut modules,
        super::Selection::DriverOnly,
    )
    .map(|selected| selected.expect("driver route returns a selection or an error"))
}
