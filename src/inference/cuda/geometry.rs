//! Library-independent convolution geometry

use cudarc::driver::CudaView;

pub(crate) use cudakit::Conv2d;

/// What a fused convolution adds before its ReLU besides the bias
#[derive(Debug)]
pub(crate) enum Residual<'a, 'b> {
    /// Nothing. cuDNN still takes a `z` operand of the output's shape, scaled by
    /// zero; pass any buffer of that size holding finite values, since `0 * NaN`
    /// would still be NaN
    None {
        /// A finite buffer of the output's size
        #[cfg(feature = "_cuda-libraries")]
        scratch: &'a CudaView<'b, f32>,
    },
    /// A residual of the output's shape
    Add(&'a CudaView<'b, f32>),
}

#[cfg(feature = "_cuda-libraries")]
impl<'a, 'b> Residual<'a, 'b> {
    /// The same operand for cudakit's cuDNN planner
    pub(crate) fn library(&self) -> cudakit::dnn::Residual<'a, 'b> {
        match *self {
            Self::None { scratch } => cudakit::dnn::Residual::None { scratch },
            Self::Add(view) => cudakit::dnn::Residual::Add(view),
        }
    }
}
