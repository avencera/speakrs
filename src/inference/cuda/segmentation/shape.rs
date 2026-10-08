use super::super::CudaError;

pub(super) use crate::inference::native_model::segmentation::{
    CLASSES, CONV_KERNEL, FEATURES, HIDDEN, LEAKY_SLOPE, LINEAR, LSTM_LAYERS, NORM_EPSILON, POOL,
    SINC_CHANNELS, SINC_KERNEL, SINC_STRIDE, WINDOW_SAMPLES,
};

/// Activation lengths of one forward pass for a batch of equal-length windows
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SegmentationShape {
    /// Windows per batch
    pub batch: usize,
    /// Samples per window
    pub samples: usize,
    /// SincNet convolution output steps
    pub sinc: usize,
    /// Steps after the first max pool
    pub pool0: usize,
    /// Steps after the first learned convolution
    pub conv1: usize,
    /// Steps after the second max pool
    pub pool1: usize,
    /// Steps after the second learned convolution
    pub conv2: usize,
    /// Output frames, 589 for a 10 s window
    pub frames: usize,
}

impl SegmentationShape {
    /// Derives every length from the window length, as the ONNX graph does
    pub fn new(batch: usize, samples: usize) -> Result<Self, CudaError> {
        let too_short = CudaError::BufferLength {
            context: "segmentation window samples",
            expected: WINDOW_SAMPLES,
            actual: samples,
        };
        if batch == 0 {
            return Err(CudaError::BufferLength {
                context: "segmentation batch",
                expected: 1,
                actual: 0,
            });
        }

        let valid = |len: usize, kernel: usize| len.checked_sub(kernel).map(|rest| rest + 1);
        let shape = (|| {
            let sinc = samples.checked_sub(SINC_KERNEL)? / SINC_STRIDE + 1;
            let pool0 = sinc / POOL;
            let conv1 = valid(pool0, CONV_KERNEL)?;
            let pool1 = conv1 / POOL;
            let conv2 = valid(pool1, CONV_KERNEL)?;
            let frames = conv2 / POOL;
            (frames > 0).then_some(Self {
                batch,
                samples,
                sinc,
                pool0,
                conv1,
                pool1,
                conv2,
                frames,
            })
        })();

        shape.ok_or(too_short)
    }
}
