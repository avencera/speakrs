use super::super::CudaError;

/// Samples in one 10 s window at 16 kHz, the input length speakrs always uses
pub(super) const WINDOW_SAMPLES: usize = 160_000;
/// Powerset classes per frame (3 speakers, at most 2 active)
pub(super) const CLASSES: usize = 7;

/// SincNet filters: 40 sine and 40 cosine band-pass filters
pub(super) const SINC_CHANNELS: usize = 80;
pub(super) const SINC_KERNEL: usize = 251;
pub(super) const SINC_STRIDE: usize = 10;
/// Kernel and stride of every max pool
pub(super) const POOL: usize = 3;
/// Channels of the two learned convolutions and the LSTM input
pub(super) const FEATURES: usize = 60;
pub(super) const CONV_KERNEL: usize = 5;
/// LSTM hidden size per direction
pub(super) const HIDDEN: usize = 128;
pub(super) const LSTM_LAYERS: usize = 4;
/// `[in, out]` of the three linear layers
pub(super) const LINEAR: [[usize; 2]; 3] = [[2 * HIDDEN, 128], [128, 128], [128, CLASSES]];
/// LeakyReLU slope of every activation, `0.01` as stored in the ONNX graph
pub(super) const LEAKY_SLOPE: f32 = 0.01;
/// Instance normalization epsilon, `1e-5` as stored in the ONNX graph
pub(super) const NORM_EPSILON: f32 = 1e-5;

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
