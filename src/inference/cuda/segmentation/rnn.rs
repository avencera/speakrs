//! The four bidirectional LSTM layers as one cuDNN RNN (`cudnnRNNForward`, v8 API)
//!
//! ONNX Runtime's CUDA execution provider also runs ONNX `LSTM` through cuDNN, so this
//! keeps the reference numerics closest. cudarc has no safe RNN wrapper and keeps its
//! cuDNN handle private, so this module owns a second cuDNN handle on the runtime's
//! stream and wraps the raw descriptors

use std::ffi::c_void;
use std::mem::MaybeUninit;
use std::ptr;
use std::sync::Arc;

use cudarc::cudnn::sys::{
    self as dnn, cudnnDataType_t, cudnnDirectionMode_t, cudnnForwardMode_t, cudnnHandle_t,
    cudnnRNNAlgo_t, cudnnRNNBiasMode_t, cudnnRNNDataDescriptor_t, cudnnRNNDataLayout_t,
    cudnnRNNDescriptor_t, cudnnRNNInputMode_t, cudnnRNNMode_t, cudnnTensorDescriptor_t,
};
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut};

use super::super::error::{check_len, to_c_int};
use super::super::{CudaError, CudaLstmAlgorithm, CudaMath, CudaRuntime};
use super::shape::{FEATURES, HIDDEN, LSTM_LAYERS};
use super::weights::{LstmLayer, ONNX_GATE};

/// `CUDNN_RNN_PADDED_IO_ENABLED` from `cudnn_adv.h`, a macro bindgen does not export;
/// cuDNN only accepts the unpacked (padded) data layouts with it set
const PADDED_IO_ENABLED: u32 = 1;

/// The cuDNN algorithm behind each public choice
fn cudnn_algo(algo: CudaLstmAlgorithm) -> cudnnRNNAlgo_t {
    match algo {
        CudaLstmAlgorithm::Standard => cudnnRNNAlgo_t::CUDNN_RNN_ALGO_STANDARD,
        CudaLstmAlgorithm::PersistStaticSmallH => {
            cudnnRNNAlgo_t::CUDNN_RNN_ALGO_PERSIST_STATIC_SMALL_H
        }
        CudaLstmAlgorithm::PersistDynamic => cudnnRNNAlgo_t::CUDNN_RNN_ALGO_PERSIST_DYNAMIC,
    }
}

/// A cuDNN handle bound to the runtime's stream, which it keeps alive
#[derive(Debug)]
struct Handle {
    raw: cudnnHandle_t,
    _stream: Arc<CudaStream>,
}

impl Handle {
    fn new(runtime: &CudaRuntime) -> Result<Self, CudaError> {
        runtime.context().bind_to_thread()?;
        let mut raw = MaybeUninit::uninit();
        // SAFETY: cudnnCreate writes the handle on success
        let handle = unsafe {
            dnn::cudnnCreate(raw.as_mut_ptr()).result()?;
            Self {
                raw: raw.assume_init(),
                _stream: runtime.stream().clone(),
            }
        };

        // SAFETY: the handle holds the stream, so the stream outlives it
        unsafe { dnn::cudnnSetStream(handle.raw, runtime.stream().cu_stream() as _) }.result()?;
        Ok(handle)
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        // SAFETY: created by cudnnCreate and destroyed once; a failure here cannot be
        // reported and leaks nothing worse than the handle
        let _ = unsafe { dnn::cudnnDestroy(self.raw) };
    }
}

// SAFETY: a cuDNN handle may move between threads as long as calls are not
// concurrent; `Handle` is not `Sync`, so they never are
unsafe impl Send for Handle {}

#[derive(Debug)]
struct RnnDescriptor(cudnnRNNDescriptor_t);

impl RnnDescriptor {
    /// The four-layer bidirectional FP32 LSTM of the segmentation model
    fn new(math: CudaMath, algo: CudaLstmAlgorithm) -> Result<Self, CudaError> {
        let context = "cuDNN LSTM";
        let mut rnn = MaybeUninit::uninit();
        // SAFETY: cudnnCreateRNNDescriptor writes the descriptor on success
        let rnn = unsafe {
            dnn::cudnnCreateRNNDescriptor(rnn.as_mut_ptr()).result()?;
            Self(rnn.assume_init())
        };

        // SAFETY: a null dropout descriptor means no dropout, which inference needs
        unsafe {
            dnn::cudnnSetRNNDescriptor_v8(
                rnn.0,
                cudnn_algo(algo),
                cudnnRNNMode_t::CUDNN_LSTM,
                // ONNX LSTM adds separate input (Wb) and recurrent (Rb) biases
                cudnnRNNBiasMode_t::CUDNN_RNN_DOUBLE_BIAS,
                cudnnDirectionMode_t::CUDNN_BIDIRECTIONAL,
                cudnnRNNInputMode_t::CUDNN_LINEAR_INPUT,
                cudnnDataType_t::CUDNN_DATA_FLOAT,
                cudnnDataType_t::CUDNN_DATA_FLOAT,
                math.cudnn(),
                to_c_int(context, FEATURES)?,
                to_c_int(context, HIDDEN)?,
                to_c_int(context, HIDDEN)?,
                to_c_int(context, LSTM_LAYERS)?,
                ptr::null_mut(),
                PADDED_IO_ENABLED,
            )
        }
        .result()?;

        Ok(rnn)
    }

    fn weight_space_bytes(&self, handle: &Handle) -> Result<usize, CudaError> {
        let mut bytes = 0;
        // SAFETY: the handle and descriptor are valid; cuDNN writes the size
        unsafe { dnn::cudnnGetRNNWeightSpaceSize(handle.raw, self.0, &mut bytes) }.result()?;
        Ok(bytes)
    }
}

impl Drop for RnnDescriptor {
    fn drop(&mut self) {
        // SAFETY: created by cudnnCreateRNNDescriptor and destroyed once
        let _ = unsafe { dnn::cudnnDestroyRNNDescriptor(self.0) };
    }
}

// SAFETY: descriptors are plain host structures with no thread affinity
unsafe impl Send for RnnDescriptor {}

#[derive(Debug)]
struct DataDescriptor(cudnnRNNDataDescriptor_t);

impl DataDescriptor {
    /// A batch-major `[batch, seq_len, vector]` FP32 sequence with every sequence full
    fn new(batch: usize, seq_len: usize, vector: usize) -> Result<Self, CudaError> {
        let context = "cuDNN RNN data";
        let mut desc = MaybeUninit::uninit();
        // SAFETY: cudnnCreateRNNDataDescriptor writes the descriptor on success
        let desc = unsafe {
            dnn::cudnnCreateRNNDataDescriptor(desc.as_mut_ptr()).result()?;
            Self(desc.assume_init())
        };

        let lengths = vec![to_c_int(context, seq_len)?; batch];
        // SAFETY: `lengths` holds `batch` entries and is only read during the call
        unsafe {
            dnn::cudnnSetRNNDataDescriptor(
                desc.0,
                cudnnDataType_t::CUDNN_DATA_FLOAT,
                cudnnRNNDataLayout_t::CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED,
                to_c_int(context, seq_len)?,
                to_c_int(context, batch)?,
                to_c_int(context, vector)?,
                lengths.as_ptr(),
                ptr::null_mut(),
            )
        }
        .result()?;

        Ok(desc)
    }
}

impl Drop for DataDescriptor {
    fn drop(&mut self) {
        // SAFETY: created by cudnnCreateRNNDataDescriptor and destroyed once
        let _ = unsafe { dnn::cudnnDestroyRNNDataDescriptor(self.0) };
    }
}

// SAFETY: descriptors are plain host structures with no thread affinity
unsafe impl Send for DataDescriptor {}

#[derive(Debug)]
struct TensorDescriptor(cudnnTensorDescriptor_t);

impl TensorDescriptor {
    fn empty() -> Result<Self, CudaError> {
        let mut desc = MaybeUninit::uninit();
        // SAFETY: cudnnCreateTensorDescriptor writes the descriptor on success
        unsafe {
            dnn::cudnnCreateTensorDescriptor(desc.as_mut_ptr()).result()?;
            Ok(Self(desc.assume_init()))
        }
    }

    /// A packed FP32 tensor of `dims`
    fn packed(dims: [usize; 3]) -> Result<Self, CudaError> {
        let context = "cuDNN RNN state";
        let desc = Self::empty()?;
        let [outer, middle, inner] = dims;
        let dims = [
            to_c_int(context, outer)?,
            to_c_int(context, middle)?,
            to_c_int(context, inner)?,
        ];
        let strides = [dims[1] * dims[2], dims[2], 1];
        // SAFETY: both arrays hold three entries
        unsafe {
            dnn::cudnnSetTensorNdDescriptor(
                desc.0,
                cudnnDataType_t::CUDNN_DATA_FLOAT,
                3,
                dims.as_ptr(),
                strides.as_ptr(),
            )
        }
        .result()?;

        Ok(desc)
    }

    /// Element count of a descriptor cuDNN filled in
    fn element_count(&self) -> Result<usize, CudaError> {
        let mut data_type = cudnnDataType_t::CUDNN_DATA_FLOAT;
        let mut rank = 0;
        let mut dims = [0; 8];
        let mut strides = [0; 8];
        // SAFETY: the arrays hold the 8 dimensions requested
        unsafe {
            dnn::cudnnGetTensorNdDescriptor(
                self.0,
                8,
                &mut data_type,
                &mut rank,
                dims.as_mut_ptr(),
                strides.as_mut_ptr(),
            )
        }
        .result()?;

        let rank = usize::try_from(rank).unwrap_or(0).min(dims.len());
        Ok(dims[..rank]
            .iter()
            .map(|&dim| usize::try_from(dim).unwrap_or(0))
            .product())
    }
}

impl Drop for TensorDescriptor {
    fn drop(&mut self) {
        // SAFETY: created by cudnnCreateTensorDescriptor and destroyed once
        let _ = unsafe { dnn::cudnnDestroyTensorDescriptor(self.0) };
    }
}

// SAFETY: descriptors are plain host structures with no thread affinity
unsafe impl Send for TensorDescriptor {}

/// The four-layer bidirectional LSTM with its weights packed into cuDNN's weight
/// space; independent of the batch size
#[derive(Debug)]
pub(super) struct CudnnLstm {
    handle: Handle,
    rnn: RnnDescriptor,
    weights: CudaSlice<u8>,
    math: CudaMath,
    algo: CudaLstmAlgorithm,
}

/// Per-batch descriptors, sequence lengths and workspace for [`CudnnLstm::forward`]
#[derive(Debug)]
pub(super) struct LstmPlan {
    batch: usize,
    seq_len: usize,
    x: DataDescriptor,
    y: DataDescriptor,
    state: TensorDescriptor,
    seq_lengths: CudaSlice<i32>,
    workspace: Option<CudaSlice<u8>>,
    /// [`CudaLstmAlgorithm::PersistDynamic`] only: a descriptor built for this batch size
    rnn: Option<RnnDescriptor>,
}

impl LstmPlan {
    pub(super) fn batch(&self) -> usize {
        self.batch
    }

    pub(super) fn seq_len(&self) -> usize {
        self.seq_len
    }

    #[cfg(test)]
    pub(super) fn workspace_bytes(&self) -> usize {
        self.workspace.as_ref().map_or(0, CudaSlice::len)
    }
}

impl CudnnLstm {
    /// Builds the RNN descriptor in `math` precision and uploads the ONNX weights into
    /// cuDNN's layout
    pub(super) fn new(
        runtime: &CudaRuntime,
        layers: &[LstmLayer; LSTM_LAYERS],
        math: CudaMath,
        algo: CudaLstmAlgorithm,
    ) -> Result<Self, CudaError> {
        let handle = Handle::new(runtime)?;
        let rnn = RnnDescriptor::new(math, algo)?;
        let stream = runtime.stream();
        let mut weights = stream.alloc_zeros::<u8>(rnn.weight_space_bytes(&handle)?)?;
        let host = Self::pack(&handle, &rnn, &weights, layers, stream)?;
        stream.memcpy_htod(&host, &mut weights)?;

        Ok(Self {
            handle,
            rnn,
            weights,
            math,
            algo,
        })
    }

    /// Lays the ONNX weights out as cuDNN's weight space on the host
    ///
    /// cuDNN reports where each gate matrix and bias lives for a weight space at a
    /// given address; this asks with the real allocation and turns the addresses back
    /// into byte offsets. Every matrix is row-major `[hidden, input]`, the same as one
    /// gate block of the ONNX `W` and `R`
    fn pack(
        handle: &Handle,
        rnn: &RnnDescriptor,
        weights: &CudaSlice<u8>,
        layers: &[LstmLayer; LSTM_LAYERS],
        stream: &Arc<CudaStream>,
    ) -> Result<Vec<u8>, CudaError> {
        let context = "cuDNN LSTM weights";
        let weight_bytes = weights.len();
        let (base, _record) = weights.device_ptr(stream);
        let mut host = vec![0u8; weight_bytes];
        let matrix_desc = TensorDescriptor::empty()?;
        let bias_desc = TensorDescriptor::empty()?;

        for (layer_index, layer) in layers.iter().enumerate() {
            for direction in 0..2 {
                let pseudo_layer = to_c_int(context, 2 * layer_index + direction)?;
                // cuDNN IDs 0..4 are the input matrices and biases, 4..8 the recurrent
                for lin_id in 0..8 {
                    let recurrent = lin_id >= 4;
                    let gate = ONNX_GATE[lin_id % 4];
                    let mut matrix_addr: *mut c_void = ptr::null_mut();
                    let mut bias_addr: *mut c_void = ptr::null_mut();
                    // SAFETY: the descriptors are valid, and the weight space at `base`
                    // has the size cuDNN asked for
                    unsafe {
                        dnn::cudnnGetRNNWeightParams(
                            handle.raw,
                            rnn.0,
                            pseudo_layer,
                            weight_bytes,
                            base as *const c_void,
                            to_c_int(context, lin_id)?,
                            matrix_desc.0,
                            &mut matrix_addr,
                            bias_desc.0,
                            &mut bias_addr,
                        )
                    }
                    .result()?;

                    let (source, columns) = if recurrent {
                        (&layer.r, HIDDEN)
                    } else {
                        (&layer.w, layer.input)
                    };
                    let block = HIDDEN * columns;
                    let start = (direction * 4 + gate) * block;
                    check_len(context, block, matrix_desc.element_count()?)?;
                    write_at(&mut host, base, matrix_addr, &source[start..start + block])?;

                    // `B[dir]` is the 4 input-bias gates followed by the 4 recurrent ones
                    let bias_start =
                        direction * 8 * HIDDEN + (usize::from(recurrent) * 4 + gate) * HIDDEN;
                    check_len(context, HIDDEN, bias_desc.element_count()?)?;
                    write_at(
                        &mut host,
                        base,
                        bias_addr,
                        &layer.b[bias_start..bias_start + HIDDEN],
                    )?;
                }
            }
        }

        Ok(host)
    }

    /// Descriptors and workspace for a `[batch, seq_len, 60]` input
    pub(super) fn plan(
        &self,
        runtime: &CudaRuntime,
        batch: usize,
        seq_len: usize,
    ) -> Result<LstmPlan, CudaError> {
        let x = DataDescriptor::new(batch, seq_len, FEATURES)?;
        let y = DataDescriptor::new(batch, seq_len, 2 * HIDDEN)?;
        let state = TensorDescriptor::packed([2 * LSTM_LAYERS, batch, HIDDEN])?;
        let rnn = match self.algo {
            CudaLstmAlgorithm::PersistDynamic => Some(self.dynamic_descriptor(batch)?),
            CudaLstmAlgorithm::Standard | CudaLstmAlgorithm::PersistStaticSmallH => None,
        };
        let rnn_desc = rnn.as_ref().unwrap_or(&self.rnn);

        let mut workspace_bytes = 0;
        let mut reserve_bytes = 0;
        // SAFETY: valid handle and descriptors; cuDNN writes both sizes
        unsafe {
            dnn::cudnnGetRNNTempSpaceSizes(
                self.handle.raw,
                rnn_desc.0,
                cudnnForwardMode_t::CUDNN_FWD_MODE_INFERENCE,
                x.0,
                &mut workspace_bytes,
                &mut reserve_bytes,
            )
        }
        .result()?;

        let stream = runtime.stream();
        let lengths = vec![to_c_int("cuDNN LSTM sequence", seq_len)?; batch];
        let workspace = match workspace_bytes {
            0 => None,
            bytes => Some(stream.alloc_zeros::<u8>(bytes)?),
        };

        Ok(LstmPlan {
            batch,
            seq_len,
            x,
            y,
            state,
            seq_lengths: stream.clone_htod(&lengths)?,
            workspace,
            rnn,
        })
    }

    /// A [`CudaLstmAlgorithm::PersistDynamic`] descriptor compiled for `batch`
    ///
    /// The weight layout depends only on the network shape, which the check on the
    /// weight space size guards
    fn dynamic_descriptor(&self, batch: usize) -> Result<RnnDescriptor, CudaError> {
        let rnn = RnnDescriptor::new(self.math, self.algo)?;
        // SAFETY: valid handle and descriptor; cuDNN compiles kernels for this batch
        unsafe {
            dnn::cudnnBuildRNNDynamic(self.handle.raw, rnn.0, to_c_int("cuDNN LSTM batch", batch)?)
        }
        .result()?;
        check_len(
            "cuDNN LSTM weight space",
            self.weights.len(),
            rnn.weight_space_bytes(&self.handle)?,
        )?;
        Ok(rnn)
    }

    /// Runs all four layers: `x` is `[batch, seq_len, 60]`, `y` is
    /// `[batch, seq_len, 256]` with the forward direction in the first 128 features,
    /// the layout of the ONNX graph after its transpose and reshape
    ///
    /// The initial hidden and cell states are zero, as in the ONNX graph
    pub(super) fn forward<X, Y>(
        &self,
        runtime: &CudaRuntime,
        plan: &mut LstmPlan,
        x: &X,
        y: &mut Y,
    ) -> Result<(), CudaError>
    where
        X: DevicePtr<f32>,
        Y: DevicePtrMut<f32>,
    {
        #[cfg(test)]
        let _library = super::super::test_support::call("cudnn.rnn");
        let steps = plan.batch * plan.seq_len;
        check_len("cuDNN LSTM input", steps * FEATURES, x.len())?;
        check_len("cuDNN LSTM output", steps * 2 * HIDDEN, y.len())?;

        let stream = runtime.stream();
        let (x_ptr, _record_x) = x.device_ptr(stream);
        let (y_ptr, _record_y) = y.device_ptr_mut(stream);
        let (weights_ptr, _record_w) = self.weights.device_ptr(stream);
        let (lengths_ptr, _record_l) = plan.seq_lengths.device_ptr(stream);
        let workspace_bytes = plan.workspace.as_ref().map_or(0, CudaSlice::len);
        let (workspace_ptr, _record_ws) = plan
            .workspace
            .as_mut()
            .map(|workspace| workspace.device_ptr_mut(stream))
            .unzip();

        // SAFETY: the buffers hold exactly the shapes of the data descriptors, the
        // weight space and workspace have the sizes cuDNN asked for, and null states
        // mean zero initial states and no final-state output
        unsafe {
            dnn::cudnnRNNForward(
                self.handle.raw,
                plan.rnn.as_ref().unwrap_or(&self.rnn).0,
                cudnnForwardMode_t::CUDNN_FWD_MODE_INFERENCE,
                lengths_ptr as *const i32,
                plan.x.0,
                x_ptr as *const c_void,
                plan.y.0,
                y_ptr as *mut c_void,
                plan.state.0,
                ptr::null(),
                ptr::null_mut(),
                plan.state.0,
                ptr::null(),
                ptr::null_mut(),
                self.weights.len(),
                weights_ptr as *const c_void,
                workspace_bytes,
                workspace_ptr.map_or(ptr::null_mut(), |ptr| ptr as *mut c_void),
                0,
                ptr::null_mut(),
            )
        }
        .result()?;

        Ok(())
    }
}

/// Copies `values` into `host` at the byte offset of device address `addr` from `base`
fn write_at(
    host: &mut [u8],
    base: u64,
    addr: *mut c_void,
    values: &[f32],
) -> Result<(), CudaError> {
    let context = "cuDNN LSTM weight offset";
    let offset = (addr as u64)
        .checked_sub(base)
        .map(|offset| offset as usize);
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect();
    let target = offset.and_then(|offset| host.get_mut(offset..offset.checked_add(bytes.len())?));
    let Some(target) = target else {
        return Err(CudaError::BufferLength {
            context,
            expected: host.len(),
            actual: offset.unwrap_or(usize::MAX),
        });
    };

    target.copy_from_slice(&bytes);
    Ok(())
}
