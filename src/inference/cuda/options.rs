/// cuDNN RNN algorithm for the segmentation model's four bidirectional LSTM layers in
/// the CUDA modes
///
/// Every algorithm computes in the precision set by
/// [`RuntimeConfig::cuda_segmentation_math`](crate::pipeline::RuntimeConfig::cuda_segmentation_math).
/// They differ in speed and in summation order, so the persistent ones drift slightly
/// further from the ONNX references. The timings and errors below were measured in FP32
/// on an RTX 5070 Ti with CUDA graphs on
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum CudaLstmAlgorithm {
    /// One GEMM for the input projections of all steps, then a recurrent GEMM per step,
    /// as ONNX Runtime's CUDA execution provider runs it
    ///
    /// The closest to the references (logits within 6.5e-5); batch 32 takes 21 ms and a
    /// single window 6.6 ms
    Standard,
    /// Persistent kernels for small hidden sizes that keep the recurrent weights on chip;
    /// the default
    ///
    /// It matched FP32 Standard DER on all 216 VoxConverse-dev files and is the fastest
    /// choice for the batches that carry most windows
    ///
    /// Batch 32 takes 13.2 ms; logits stay within 2e-4 of the references with the same
    /// argmax on the test fixtures. A single window is slower than with
    /// [`Self::Standard`] (7.7 ms)
    #[default]
    PersistStaticSmallH,
    /// Persistent kernels that cuDNN compiles at run time with NVRTC for each batch size
    ///
    /// The fastest for a single window (3.6 ms) but by far the slowest for batch 32
    /// (85 ms), which the pipeline uses for most windows. It needs the NVRTC library;
    /// when NVRTC cannot be loaded, the model logs a warning and uses
    /// [`Self::Standard`] instead
    PersistDynamic,
}

/// Whether the CUDA modes record each fixed-shape forward pass as a CUDA graph and
/// replay it
///
/// A replay gives bit-identical results to running the same pass launch by launch,
/// and removes most of the launch overhead: a single segmentation window drops from
/// 27 ms to 6.6 ms on an RTX 5070 Ti
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum CudaGraphs {
    /// Record each batch size's forward pass on first use and replay it afterwards
    #[default]
    Enabled,
    /// Launch every kernel and library call on each forward pass
    Disabled,
}

impl CudaGraphs {
    pub(crate) const fn enabled(self) -> bool {
        matches!(self, Self::Enabled)
    }
}
