//! Complete dense plans shared by segmentation and embedding

use super::candidate::{DenseCandidate, DenseOxide, DenseSpec, Phases};
use super::implementation::{AreaTarget, LibraryNeed, Selected, plan_selection};
use super::{CudaError, CudaLibrary, CudaRuntime, KernelModule};
use cudarc::driver::{CudaSlice, CudaView, CudaViewMut};

/// One implementation owner for a dense operation and its epilogue
#[derive(Debug)]
pub(super) enum DensePlan {
    #[cfg(feature = "_cuda-libraries")]
    Library,
    Oxide(Box<DenseOxide>),
}

impl DensePlan {
    /// Library plans must finish lazy setup before graph capture
    pub(super) fn requires_warmup(&self) -> bool {
        match self {
            #[cfg(feature = "_cuda-libraries")]
            Self::Library => true,
            Self::Oxide(_) => false,
        }
    }

    pub(super) fn new(
        runtime: &CudaRuntime,
        spec: DenseSpec,
        rows: usize,
        weight: &CudaSlice<f32>,
        bias: &CudaSlice<f32>,
    ) -> Result<Self, CudaError> {
        let boundary = spec.site().boundary();
        let selected = plan_selection(
            runtime,
            boundary,
            spec.batch(),
            spec.math(),
            #[cfg(all(test, feature = "_cuda-libraries"))]
            None,
        )?;
        if let Selected::Oxide(token) = selected
            && let Some(plan) = token.dense(runtime, spec, rows, weight, bias)?
        {
            return Ok(Self::Oxide(Box::new(plan)));
        }
        LibraryNeed::new(
            boundary,
            spec.batch(),
            spec.math(),
            AreaTarget::for_area(runtime, KernelModule::Segdense)?,
            CudaLibrary::Cublas,
        )
        .prepare(runtime)?;
        #[cfg(feature = "_cuda-libraries")]
        return Ok(Self::Library);
        #[cfg(not(feature = "_cuda-libraries"))]
        Err(CudaError::LibraryUnavailable {
            library: CudaLibrary::Cublas,
        })
    }

    pub(super) fn enqueue_slice(
        &self,
        runtime: &CudaRuntime,
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
        library: impl FnOnce(&mut CudaSlice<f32>) -> Result<(), CudaError>,
    ) -> Result<(), CudaError> {
        #[cfg(not(feature = "_cuda-libraries"))]
        let _ = library;
        match self {
            #[cfg(feature = "_cuda-libraries")]
            Self::Library => library(output),
            Self::Oxide(plan) => plan.enqueue(
                &input.as_view(),
                &mut output.as_view_mut(),
                &Phases::new(),
                runtime,
            ),
        }
    }

    pub(super) fn enqueue(
        &self,
        runtime: &CudaRuntime,
        input: &CudaView<'_, f32>,
        output: &mut CudaViewMut<'_, f32>,
        library: impl FnOnce(&mut CudaViewMut<'_, f32>) -> Result<(), CudaError>,
    ) -> Result<(), CudaError> {
        #[cfg(not(feature = "_cuda-libraries"))]
        let _ = library;
        match self {
            #[cfg(feature = "_cuda-libraries")]
            Self::Library => library(output),
            Self::Oxide(plan) => plan.enqueue(input, output, &Phases::new(), runtime),
        }
    }
}
