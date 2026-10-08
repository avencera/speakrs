use std::path::{Path, PathBuf};

use safetensors::tensor::Metadata;
use safetensors::{Dtype, SafeTensors};

use super::{CudaError, CudaRuntime, DeviceTensor};

/// Size of the little-endian header length that starts every safetensors file
const HEADER_LEN_BYTES: usize = 8;

/// A safetensors weights file read into memory, uploaded one named tensor at a time
///
/// Weights are FP32 only, matching the FP32-everywhere backend; batch norm is folded
/// into the exported weights, so there is no separate statistics format
#[derive(Debug)]
pub struct SafetensorsFile {
    path: PathBuf,
    bytes: Vec<u8>,
    data_start: usize,
    metadata: Metadata,
}

impl SafetensorsFile {
    /// Reads and validates the header of a safetensors file
    pub fn open(path: impl AsRef<Path>) -> Result<Self, CudaError> {
        let path = path.as_ref().to_path_buf();
        let bytes = std::fs::read(&path).map_err(|source| CudaError::WeightsIo {
            path: path.clone(),
            source,
        })?;

        let (header_len, metadata) =
            SafeTensors::read_metadata(&bytes).map_err(|source| CudaError::WeightsFormat {
                path: path.clone(),
                source,
            })?;

        Ok(Self {
            path,
            bytes,
            data_start: HEADER_LEN_BYTES + header_len,
            metadata,
        })
    }

    /// Tensor names in the file, sorted
    pub fn names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.metadata.tensors().into_keys().collect();
        names.sort();
        names
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
        let info = self
            .metadata
            .info(name)
            .ok_or_else(|| CudaError::MissingTensor {
                path: self.path.clone(),
                name: name.to_string(),
            })?;

        if info.dtype != Dtype::F32 {
            return Err(CudaError::TensorDtype {
                name: name.to_string(),
                dtype: info.dtype.to_string(),
            });
        }

        if info.shape != expected_shape {
            return Err(CudaError::TensorShape {
                name: name.to_string(),
                expected: expected_shape.to_vec(),
                actual: info.shape.clone(),
            });
        }

        // `read_metadata` already checked that every offset lies inside the file
        let (start, end) = info.data_offsets;
        let bytes = &self.bytes[self.data_start + start..self.data_start + end];
        let (chunks, _) = bytes.as_chunks::<4>();
        Ok(chunks.iter().copied().map(f32::from_le_bytes).collect())
    }
}

#[cfg(all(test, feature = "cuda", not(feature = "cuda-driver-only")))]
mod test_support;

#[cfg(all(test, feature = "cuda", not(feature = "cuda-driver-only")))]
pub(crate) use test_support::uniform;
