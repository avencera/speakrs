use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, DeviceRepr, ValidAsZeroBits};

use super::CudaError;
use super::error::{check_len, element_count};

/// A device buffer with a row-major shape
///
/// The shape is metadata for checks and indexing; the storage is one contiguous
/// [`CudaSlice`]. Allocate persistent tensors once per fixed batch shape and refill
/// them with [`Self::copy_from_host`] instead of reallocating per call
#[derive(Debug)]
pub struct DeviceTensor<T = f32> {
    data: CudaSlice<T>,
    shape: Vec<usize>,
}

impl<T: DeviceRepr + ValidAsZeroBits + Clone + Default + Unpin> DeviceTensor<T> {
    /// Allocates a zero-filled tensor on `stream`
    pub fn zeros(stream: &Arc<CudaStream>, shape: &[usize]) -> Result<Self, CudaError> {
        let len = element_count("device tensor", shape)?;
        Ok(Self {
            data: stream.alloc_zeros(len)?,
            shape: shape.to_vec(),
        })
    }

    /// Copies `host` to a new device tensor; `host` must hold exactly `shape` elements
    pub fn upload(
        stream: &Arc<CudaStream>,
        host: &[T],
        shape: &[usize],
    ) -> Result<Self, CudaError> {
        check_len(
            "device tensor upload",
            element_count("device tensor", shape)?,
            host.len(),
        )?;
        Ok(Self {
            data: stream.clone_htod(host)?,
            shape: shape.to_vec(),
        })
    }

    /// Overwrites the tensor with `host`, which must hold exactly [`Self::len`] elements
    pub fn copy_from_host(
        &mut self,
        stream: &Arc<CudaStream>,
        host: &[T],
    ) -> Result<(), CudaError> {
        check_len("device tensor copy", self.len(), host.len())?;
        stream.memcpy_htod(host, &mut self.data)?;
        Ok(())
    }

    /// Copies the tensor back to the host; this waits for `stream`
    pub fn download(&self, stream: &Arc<CudaStream>) -> Result<Vec<T>, CudaError> {
        Ok(stream.clone_dtoh(&self.data)?)
    }
}

impl<T> DeviceTensor<T> {
    #[cfg(all(test, feature = "cuda", not(feature = "cuda-driver-only")))]
    /// The row-major shape
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Number of elements
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    /// The device storage, for kernel arguments and library calls
    pub fn data(&self) -> &CudaSlice<T> {
        &self.data
    }

    /// The mutable device storage, for kernel arguments and library calls
    pub fn data_mut(&mut self) -> &mut CudaSlice<T> {
        &mut self.data
    }
}
