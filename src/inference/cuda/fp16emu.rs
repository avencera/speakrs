//! Private, off-by-default FP16 operand experiment

use std::path::PathBuf;

use cudarc::driver::{CudaFunction, CudaViewMut, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;

use super::{CudaError, CudaRuntime, KernelModule, PtxTier};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Operands,
    Stores,
}

#[derive(Debug)]
pub(super) struct Experiment {
    mode: Mode,
    directory: PathBuf,
}

impl Experiment {
    pub(super) fn from_env() -> Result<Option<Self>, CudaError> {
        Self::parse(
            std::env::var("SPEAKRS_FP16_EMU").ok().as_deref(),
            std::env::var_os("SPEAKRS_FP16_EMU_DIR").map(PathBuf::from),
        )
    }

    fn parse(mode: Option<&str>, directory: Option<PathBuf>) -> Result<Option<Self>, CudaError> {
        let mode = match mode {
            None => return Ok(None),
            Some("operands") => Mode::Operands,
            Some("stores") => Mode::Stores,
            Some(_) => return Err(invalid("SPEAKRS_FP16_EMU must be operands or stores")),
        };
        let directory = directory.ok_or_else(|| invalid("SPEAKRS_FP16_EMU_DIR is required"))?;
        Ok(Some(Self { mode, directory }))
    }

    pub(super) fn ptx(
        &self,
        area: KernelModule,
        tier: PtxTier,
    ) -> Result<Option<String>, CudaError> {
        if !matches!(area, KernelModule::Resnet | KernelModule::Wideconv) {
            return Ok(None);
        }
        if tier != PtxTier::Sm80 {
            return Err(invalid(
                "the FP16 operand experiment requires the sm80 trunk tier",
            ));
        }
        let path = self.directory.join(format!("{}.sm80.ptx", area.name()));
        let text = std::fs::read_to_string(path).map_err(|error| invalid(error.to_string()))?;
        if !text.contains("// SPEAKRS_FP16_EMU_SCALE=1024") {
            return Err(invalid(
                "the alternate PTX lacks the FP16 emulation scale marker",
            ));
        }
        Ok(Some(text))
    }

    pub(super) fn stores(&self) -> bool {
        self.mode == Mode::Stores
    }

    pub(super) fn store_rounder(
        &self,
        runtime: &CudaRuntime,
    ) -> Result<Option<StoreRounder>, CudaError> {
        if !self.stores() {
            return Ok(None);
        }
        let path = self.directory.join("probe.ptx");
        let text = std::fs::read_to_string(path).map_err(|error| invalid(error.to_string()))?;
        let module = runtime.context().load_module(Ptx::from_src(text))?;
        Ok(Some(StoreRounder(module.load_function("round_stores")?)))
    }
}

#[derive(Debug)]
pub(super) struct StoreRounder(CudaFunction);

impl StoreRounder {
    pub(super) fn round(
        &self,
        runtime: &CudaRuntime,
        values: &mut CudaViewMut<'_, f32>,
    ) -> Result<(), CudaError> {
        let elements = u32::try_from(values.len())
            .map_err(|_| invalid("conv1 stores exceed the diagnostic grid limit"))?;
        let len = u64::from(elements);
        let mut launch = runtime.stream().launch_builder(&self.0);
        launch.arg(values).arg(&len);
        // safety: one thread per element, with the diagnostic's pointer and length ABI
        unsafe { launch.launch(LaunchConfig::for_num_elems(elements)) }?;
        Ok(())
    }
}

fn invalid(reason: impl Into<String>) -> CudaError {
    CudaError::Unsupported {
        context: "FP16 operand experiment",
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::{Experiment, Mode};
    use std::path::PathBuf;

    #[test]
    fn experiment_is_off_without_a_mode() {
        assert!(
            Experiment::parse(None, Some(PathBuf::from("unused")))
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn experiment_requires_an_explicit_mode_and_artifact_directory() {
        assert!(Experiment::parse(Some("enabled"), Some(PathBuf::from("ptx"))).is_err());
        assert!(Experiment::parse(Some("operands"), None).is_err());
        let operands = Experiment::parse(Some("operands"), Some(PathBuf::from("ptx")))
            .unwrap()
            .unwrap();
        assert_eq!(operands.mode, Mode::Operands);
        assert!(!operands.stores());
        let stores = Experiment::parse(Some("stores"), Some(PathBuf::from("ptx")))
            .unwrap()
            .unwrap();
        assert!(stores.stores());
    }
}
