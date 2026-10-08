//! Harness scopes compiled only in CUDA tests

use super::{CudaStream, Direction};

/// A locked harness sub-scope in the qualification binary; production opens none
pub(super) fn sub_scope(name: impl FnOnce() -> String) -> super::super::test_support::Scope {
    super::super::test_support::sub_scope(&name())
}

pub(super) fn projection_scope(
    stream: &CudaStream,
    layer: usize,
    direction: Direction,
) -> super::super::test_support::Scope {
    super::super::test_support::projection(stream, layer, direction)
}
