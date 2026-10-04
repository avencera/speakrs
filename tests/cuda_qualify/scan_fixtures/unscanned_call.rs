//! A planted LSTM candidate that hands its work to a helper outside the candidate tree
//!
//! The helper could read the harness phase, the reference files or the capture state
//! where the static scan does not look. The scan must refuse the call itself, and the
//! lock refuses a new module anywhere else under `src/`

use cudarc::driver::{CudaStream, CudaView, CudaViewMut};

use super::{Batches, Coverage, CoverageEntry, LstmCandidate, LstmPhases, LstmSpec, Maths};
use crate::inference::cuda::{CudaError, CudaRuntime};

/// Delegates the whole stack to an unscanned module
#[derive(Debug)]
pub(crate) struct Oxide;

impl LstmCandidate for Oxide {
    const COVERAGE: Coverage = Coverage(&[CoverageEntry {
        layers: &["lstm.stack"],
        batches: Batches::All,
        maths: Maths::All,
    }]);

    fn plan(_runtime: &CudaRuntime, _spec: LstmSpec<'_>) -> Result<Self, CudaError> {
        Ok(Self)
    }

    fn enqueue(
        &self,
        input: &CudaView<'_, f32>,
        output: &mut CudaViewMut<'_, f32>,
        _phases: &LstmPhases<'_>,
        stream: &CudaStream,
    ) -> Result<(), CudaError> {
        crate::inference::cuda::test_support::phase();
        crate::helper::enqueue_stack(input, output, stream)
    }
}
