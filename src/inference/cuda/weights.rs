use std::path::Path;

use crate::inference::native_model::WeightsFile;

use super::{CudaError, CudaRuntime, DeviceTensor};

/// A safetensors weights file read into memory, uploaded one named tensor at a time
///
/// Uploads accept FP32 only; batch norm is folded into the exported weights
#[derive(Debug)]
pub struct SafetensorsFile(WeightsFile);

impl SafetensorsFile {
    /// Reads and validates the header of a safetensors file
    pub fn open(path: impl AsRef<Path>) -> Result<Self, CudaError> {
        WeightsFile::open(path).map(Self).map_err(Into::into)
    }

    /// Uploads tensor `name` after checking that it is FP32 with exactly `expected_shape`
    pub fn upload(
        &self,
        runtime: &CudaRuntime,
        name: &str,
        expected_shape: &[usize],
    ) -> Result<DeviceTensor<f32>, CudaError> {
        let host = self.read_f32(name, expected_shape)?;
        DeviceTensor::upload(runtime.stream(), &host, expected_shape)
    }

    /// Reads tensor `name` to the host after the same checks as [`Self::upload`]
    pub fn read_f32(&self, name: &str, expected_shape: &[usize]) -> Result<Vec<f32>, CudaError> {
        self.0.read_f32(name, expected_shape).map_err(Into::into)
    }

    /// The runtime-neutral decoder used by native model recipes
    pub(crate) fn host(&self) -> &WeightsFile {
        &self.0
    }
}

#[cfg(all(test, feature = "_cuda-libraries"))]
mod test_support;
