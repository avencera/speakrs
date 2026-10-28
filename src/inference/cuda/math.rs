//! speakrs's precision choice, re-exported from cudakit

/// Arithmetic precision of CUDA kernels, cuBLAS and cuDNN work on FP32 data
///
/// Segmentation and embedding choose it separately through
/// [`RuntimeConfig`](crate::pipeline::RuntimeConfig), which defaults to [`CudaMath::Fp32`]
/// for segmentation and [`CudaMath::Tf32`] for embedding. The filterbank always runs in FP32
///
/// [`CudaMath::Tf32`] on embedding may also use scaled FP16 operands for same-channel,
/// stride-1 trunk layers on Turing and newer, with FP32 accumulation. FP16 is excluded
/// for a layer with non-finite weights or weight magnitudes above 65504 / 1024, and if
/// an activation exceeds that range the complete embedding batch is recomputed without
/// FP16 tiles. [`CudaMath::Fp32`] never selects FP16 tiles and is the closest match to
/// the ONNX references; embedding drift can change PLDA/VBx clustering (adr/001)
pub use cudakit::Math as CudaMath;
