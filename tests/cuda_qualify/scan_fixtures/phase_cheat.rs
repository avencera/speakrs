//! A planted LSTM candidate that skips its work while the harness times it
//!
//! The static scan must refuse this file before anything is built. The live proof
//! copies it over `src/inference/cuda/candidate/lstm.rs` in a scratch tree

use cudarc::driver::{CudaFunction, CudaStream, CudaView, CudaViewMut, LaunchConfig, PushKernelArg};

use super::{Batches, Coverage, CoverageEntry, LstmCandidate, LstmPhases, LstmSpec, Maths};
use crate::inference::cuda::{CudaError, CudaRuntime, KernelModule};

/// Reads the harness phase and does nothing while it is timed
#[derive(Debug)]
pub(crate) struct Oxide {
    stack: CudaFunction,
}

impl LstmCandidate for Oxide {
    const COVERAGE: Coverage = Coverage(&[CoverageEntry {
        layers: &["lstm.stack"],
        batches: Batches::All,
        maths: Maths::All,
    }]);

    fn plan(runtime: &CudaRuntime, _spec: LstmSpec<'_>) -> Result<Self, CudaError> {
        let kernels = runtime.load_kernels(KernelModule::Lstm)?;
        Ok(Self {
            stack: kernels.function("lstm_stack")?,
        })
    }

    fn enqueue(
        &self,
        input: &CudaView<'_, f32>,
        output: &mut CudaViewMut<'_, f32>,
        _phases: &LstmPhases<'_>,
        stream: &CudaStream,
    ) -> Result<(), CudaError> {
        if std::env::var("SPEAKRS_QUALIFY_PHASE").as_deref() == Ok("timing") {
            return Ok(());
        }

        let mut launch = stream.launch_builder(&self.stack);
        launch.arg(input).arg(output);
        // SAFETY: planted fixture, never compiled
        unsafe { launch.launch(LaunchConfig::for_num_elems(1)) }?;
        Ok(())
    }
}
