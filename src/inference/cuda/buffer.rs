use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, CudaView, DeviceRepr, ValidAsZeroBits};

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
    #[cfg(all(test, feature = "_cuda-libraries"))]
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

/// Writes rows of `window` samples cut out of a device span into `target`
///
/// Row `i` is `span[starts[i]..starts[i] + window]`, clipped at the end of `span` and
/// zero padded to `window`, so overlapping windows of one recording reach the device
/// as their shared audio. `target` must hold at least `starts.len() * window` values
pub fn unfold_windows(
    stream: &Arc<CudaStream>,
    span: &CudaView<'_, f32>,
    starts: &[usize],
    window: usize,
    target: &mut CudaSlice<f32>,
) -> Result<(), CudaError> {
    let rows = starts.len() * window;
    if rows > target.len() {
        return Err(CudaError::BufferLength {
            context: "unfolded windows",
            expected: target.len(),
            actual: rows,
        });
    }

    for (row, &start) in starts.iter().enumerate() {
        let copied = span.len().saturating_sub(start).min(window);
        let mut target = target.slice_mut(row * window..(row + 1) * window);
        if copied > 0 {
            stream.memcpy_dtod(
                &span.slice(start..start + copied),
                &mut target.slice_mut(..copied),
            )?;
        }
        if copied < window {
            stream.memset_zeros(&mut target.slice_mut(copied..))?;
        }
    }

    Ok(())
}
