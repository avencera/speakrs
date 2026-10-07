//! Library implementation of the filterbank producer interface and its shared consumer

use std::cell::RefCell;

use cudarc::driver::{CudaView, CudaViewMut};

use super::{CudaFbank, FBANK_MEL_BINS, FBANK_WINDOW_SAMPLES, FbankBuffers};
use crate::inference::cuda::candidate::{
    Coverage, FbankCandidate, FbankPin, FbankSpec, FiniteContract, InfinityContract, NanContract,
    Phases, PlanError, SignedZeroContract, SpecialValues,
};
use crate::inference::cuda::error::check_len;
use crate::inference::cuda::{CudaError, CudaRuntime};

/// A full Library producer plan, with scratch owned by this one batch plan
pub(crate) struct Library {
    front: CudaFbank,
    spec: FbankSpec,
    buffers: RefCell<FbankBuffers>,
}

impl FbankCandidate for Library {
    type Pin = ();
    const COVERAGE: Coverage = Coverage::NONE;
    const SPECIAL_VALUES: SpecialValues = SpecialValues {
        finite: FiniteContract::UnitWaveform,
        nan: NanContract::Unspecified,
        infinity: InfinityContract::Unspecified,
        signed_zero: SignedZeroContract::Positive,
    };

    fn implemented_pin(_spec: FbankSpec) -> Result<Self::Pin, PlanError> {
        Ok(())
    }

    fn plan(runtime: &CudaRuntime, spec: FbankSpec, (): Self::Pin) -> Result<Self, PlanError> {
        let front = CudaFbank::new(runtime, spec.math())?;
        let buffers = front.buffers(runtime, spec.batch())?;
        Ok(Self {
            front,
            spec,
            buffers: RefCell::new(buffers),
        })
    }

    fn enqueue(
        &self,
        waveform: &CudaView<'_, f32>,
        energies: &mut CudaViewMut<'_, f32>,
        _phases: &Phases,
        runtime: &CudaRuntime,
    ) -> Result<(), CudaError> {
        let batch = self.spec.batch();
        check_len(
            "fbank producer waveform",
            batch * FBANK_WINDOW_SAMPLES,
            waveform.len(),
        )?;
        check_len(
            "fbank producer energies",
            batch * super::FBANK_FRAMES * FBANK_MEL_BINS,
            energies.len(),
        )?;
        let mut buffers = self.buffers.borrow_mut();
        self.front.produce(
            runtime,
            waveform,
            batch,
            &mut buffers.work.producer,
            energies,
        )
    }
}

impl Library {
    /// Run the identical locked log/CMN consumer on energies from either producer
    pub(crate) fn consume(
        &self,
        runtime: &CudaRuntime,
        energies: &CudaView<'_, f32>,
        features: &mut CudaViewMut<'_, f32>,
    ) -> Result<(), CudaError> {
        check_len(
            "fbank consumer energies",
            self.spec.batch() * super::FBANK_FRAMES * FBANK_MEL_BINS,
            energies.len(),
        )?;
        self.front
            .log_cmn(runtime, self.spec.batch(), energies, features)
    }

    /// The original full Library path, independent of the replacement producer
    pub(crate) fn full(
        &self,
        runtime: &CudaRuntime,
        waveform: &CudaView<'_, f32>,
        features: &mut CudaViewMut<'_, f32>,
    ) -> Result<(), CudaError> {
        check_len(
            "fbank full-control waveform",
            self.spec.batch() * FBANK_WINDOW_SAMPLES,
            waveform.len(),
        )?;
        let mut buffers = self.buffers.borrow_mut();
        let work = &mut buffers.work;
        self.front.produce(
            runtime,
            waveform,
            self.spec.batch(),
            &mut work.producer,
            &mut work.energies.as_view_mut(),
        )?;
        self.front.log_cmn(
            runtime,
            self.spec.batch(),
            &work.energies.as_view(),
            features,
        )
    }
}

#[test]
fn producer_shape_uses_the_boundary_batch_set() {
    use crate::inference::cuda::CudaMath;
    for batch in 0..=64 {
        let spec = FbankSpec::new(batch, CudaMath::Fp32);
        assert_eq!(spec.is_ok(), (1..=32).contains(&batch));
        if let Ok(spec) = spec {
            assert_eq!(spec.batch(), batch);
            assert_eq!(spec.math(), CudaMath::Fp32);
            Library::implemented_pin(spec).expect("Library pin");
        }
    }
    assert_eq!(
        crate::inference::cuda::candidate::ConfigPin::Fbank(FbankPin::FftMelAccurate).area(),
        crate::inference::cuda::KernelModule::FbankDft
    );
    assert_ne!(
        crate::inference::cuda::KernelModule::FbankDft,
        crate::inference::cuda::KernelModule::Fbank
    );
    assert_eq!(Library::SPECIAL_VALUES.finite, FiniteContract::UnitWaveform);
    assert!(Library::COVERAGE.entries().is_empty());
}

#[test]
#[ignore = "requires the GPU lock and qualification environment"]
fn library_producer_and_full_control_are_identical() -> Result<(), CudaError> {
    use crate::inference::cuda::CudaMath;
    let runtime = CudaRuntime::new(0)?;
    let spec = FbankSpec::new(1, CudaMath::Fp32).expect("supported batch");
    Library::implemented_pin(spec).expect("pin");
    let plan = Library::plan(&runtime, spec, ()).map_err(|error| CudaError::Unsupported {
        context: "fbank control",
        reason: error.to_string(),
    })?;
    let audio: Vec<_> = (0..FBANK_WINDOW_SAMPLES)
        .map(|i| (i as f32 * 0.007).sin() * 0.2)
        .collect();
    let waveform = runtime.stream().clone_htod(&audio)?;
    let len = super::FBANK_FRAMES * FBANK_MEL_BINS;
    let mut energies = runtime.stream().alloc_zeros::<f32>(len)?;
    let mut features = runtime.stream().alloc_zeros::<f32>(len)?;
    let mut control = runtime.stream().alloc_zeros::<f32>(len)?;
    plan.enqueue(
        &waveform.as_view(),
        &mut energies.as_view_mut(),
        &Phases::new(),
        &runtime,
    )?;
    plan.consume(&runtime, &energies.as_view(), &mut features.as_view_mut())?;
    plan.full(&runtime, &waveform.as_view(), &mut control.as_view_mut())?;
    let actual = runtime.stream().clone_dtoh(&features)?;
    let expected = runtime.stream().clone_dtoh(&control)?;
    assert!(actual.iter().all(|x| x.is_finite()));
    assert_eq!(
        actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
    );
    Ok(())
}
