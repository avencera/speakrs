use super::{NativeWeightsError, WeightsFile};

/// Samples in one 10 s window at 16 kHz, the input length speakrs always uses
pub(crate) const WINDOW_SAMPLES: usize = 160_000;
/// Powerset classes per frame (3 speakers, at most 2 active)
pub(crate) const CLASSES: usize = 7;

/// SincNet filters: 40 sine and 40 cosine band-pass filters
pub(crate) const SINC_CHANNELS: usize = 80;
/// Samples per learned Sinc filter
pub(crate) const SINC_KERNEL: usize = 251;
/// Samples between Sinc convolution outputs
pub(crate) const SINC_STRIDE: usize = 10;
/// Kernel and stride of every max pool
pub(crate) const POOL: usize = 3;
/// Channels of the two learned convolutions and the LSTM input
pub(crate) const FEATURES: usize = 60;
/// Samples per learned convolution kernel
pub(crate) const CONV_KERNEL: usize = 5;
/// LSTM hidden size per direction
pub(crate) const HIDDEN: usize = 128;
/// Bidirectional recurrent layers
pub(crate) const LSTM_LAYERS: usize = 4;
/// `[in, out]` of the three linear layers
pub(crate) const LINEAR: [[usize; 2]; 3] = [[2 * HIDDEN, 128], [128, 128], [128, CLASSES]];
/// LeakyReLU slope of every activation, `0.01` as stored in the ONNX graph
pub(crate) const LEAKY_SLOPE: f32 = 0.01;
/// Instance normalization epsilon, `1e-5` as stored in the ONNX graph
pub(crate) const NORM_EPSILON: f32 = 1e-5;

/// Half the SincNet filter length; the filters are symmetric around one center tap
const SINC_HALF: usize = SINC_KERNEL / 2;

/// Scale and shift of one normalization layer
#[derive(Debug, Clone)]
pub(crate) struct Affine {
    /// Learned scale per channel
    pub(crate) gamma: Vec<f32>,
    /// Learned shift per channel
    pub(crate) beta: Vec<f32>,
}

/// One bidirectional ONNX LSTM layer, both directions stacked on the first axis
#[derive(Debug, Clone)]
pub(crate) struct LstmLayer {
    /// Input size of this layer
    pub(crate) input: usize,
    /// `W`, `[2, 4 * hidden, input]`, gates `[i, o, f, c]`
    pub(crate) w: Vec<f32>,
    /// `R`, `[2, 4 * hidden, hidden]`
    pub(crate) r: Vec<f32>,
    /// `B`, `[2, 8 * hidden]`: input biases then recurrent biases
    pub(crate) b: Vec<f32>,
}

/// One linear layer as an ONNX `MatMul` weight `[in, out]` and its bias
#[derive(Debug, Clone)]
pub(crate) struct Linear {
    /// Contiguous weights in the exported tensor layout
    pub(crate) weight: Vec<f32>,
    /// Learned output bias
    pub(crate) bias: Vec<f32>,
}

/// Every weight of `segmentation-3.0.onnx` on the host, with the SincNet filters
/// already generated from their learned band edges
#[derive(Debug, Clone)]
pub(crate) struct PyanNetWeights {
    /// Waveform population normalization affine parameters
    pub(crate) wav_norm: Affine,
    /// `[80, 1, 251]`
    pub(crate) sinc_filters: Vec<f32>,
    /// `[60, 80, 5]`
    pub(crate) conv1: Linear,
    /// `[60, 60, 5]`
    pub(crate) conv2: Linear,
    /// Affine parameters after each convolution and pool
    pub(crate) norms: [Affine; 3],
    /// Bidirectional recurrent layers in graph order
    pub(crate) lstm: [LstmLayer; LSTM_LAYERS],
    /// Two hidden projections and the powerset classifier
    pub(crate) linear: [Linear; 3],
}

impl PyanNetWeights {
    /// Reads the weights of either export (`segmentation-3.0` or `-b32`)
    ///
    /// The named PyTorch parameters keep their names in both exports. The LSTM and
    /// `MatMul` weights only have generated names (`onnx::LSTM_783` in one export,
    /// `onnx::LSTM_814` in the other) whose numbers grow in graph order, so they are
    /// matched by prefix and numeric order and then checked by shape
    pub(crate) fn load(file: &WeightsFile) -> Result<Self, NativeWeightsError> {
        let read = |name: &str, shape: &[usize]| file.read_f32(name, shape);
        let affine = |prefix: &str, channels: usize| -> Result<Affine, NativeWeightsError> {
            Ok(Affine {
                gamma: read(&format!("{prefix}.weight"), &[channels])?,
                beta: read(&format!("{prefix}.bias"), &[channels])?,
            })
        };

        let generated = GeneratedNames::new(file);
        let lstm_names = generated.take("onnx::LSTM_", 3 * LSTM_LAYERS)?;
        let matmul_names = generated.take("onnx::MatMul_", 3)?;
        let half_n_name = generated.take("onnx::Div_", 1)?;

        let lstm = try_array(|layer| {
            // each layer's initializers are numbered B, W, R in the export
            let names = &lstm_names[3 * layer..3 * layer + 3];
            let input = if layer == 0 { FEATURES } else { 2 * HIDDEN };
            Ok::<_, NativeWeightsError>(LstmLayer {
                input,
                b: read(&names[0], &[2, 8 * HIDDEN])?,
                w: read(&names[1], &[2, 4 * HIDDEN, input])?,
                r: read(&names[2], &[2, 4 * HIDDEN, HIDDEN])?,
            })
        })?;

        let linear_bias = ["linear.0.bias", "linear.1.bias", "classifier.bias"];
        let linear = try_array(|index| {
            let [inputs, outputs] = LINEAR[index];
            Ok::<_, NativeWeightsError>(Linear {
                weight: read(&matmul_names[index], &[inputs, outputs])?,
                bias: read(linear_bias[index], &[outputs])?,
            })
        })?;

        let filterbank = SincFilterbank {
            low_hz: read(
                "sincnet.conv1d.0.filterbank.low_hz_",
                &[SINC_CHANNELS / 2, 1],
            )?,
            band_hz: read(
                "sincnet.conv1d.0.filterbank.band_hz_",
                &[SINC_CHANNELS / 2, 1],
            )?,
            window: read("sincnet.conv1d.0.filterbank.window_", &[SINC_HALF])?,
            n: read("sincnet.conv1d.0.filterbank.n_", &[1, SINC_HALF])?,
            half_n: read(&half_n_name[0], &[1, SINC_HALF])?,
        };

        Ok(Self {
            wav_norm: affine("sincnet.wav_norm1d", 1)?,
            sinc_filters: filterbank.filters(),
            conv1: Linear {
                weight: read(
                    "sincnet.conv1d.1.weight",
                    &[FEATURES, SINC_CHANNELS, CONV_KERNEL],
                )?,
                bias: read("sincnet.conv1d.1.bias", &[FEATURES])?,
            },
            conv2: Linear {
                weight: read(
                    "sincnet.conv1d.2.weight",
                    &[FEATURES, FEATURES, CONV_KERNEL],
                )?,
                bias: read("sincnet.conv1d.2.bias", &[FEATURES])?,
            },
            norms: [
                affine("sincnet.norm1d.0", SINC_CHANNELS)?,
                affine("sincnet.norm1d.1", FEATURES)?,
                affine("sincnet.norm1d.2", FEATURES)?,
            ],
            lstm,
            linear,
        })
    }
}

/// `std::array::from_fn` for a fallible element constructor
fn try_array<T, const N: usize>(
    mut element: impl FnMut(usize) -> Result<T, NativeWeightsError>,
) -> Result<[T; N], NativeWeightsError> {
    let elements = (0..N).map(&mut element).collect::<Result<Vec<T>, _>>()?;
    // the constructor visits exactly N indices, so the array length cannot differ
    Ok(elements
        .try_into()
        .unwrap_or_else(|_| unreachable!("exactly N elements")))
}

/// Generated initializer names, sorted by their numeric suffix
struct GeneratedNames(Vec<(String, u64)>);

impl GeneratedNames {
    fn new(file: &WeightsFile) -> Self {
        let mut names: Vec<(String, u64)> = file
            .names()
            .into_iter()
            .filter_map(|name| {
                let number = name.rsplit('_').next()?.parse().ok()?;
                name.starts_with("onnx::").then_some((name, number))
            })
            .collect();
        names.sort_by_key(|(_, number)| *number);
        Self(names)
    }

    /// The `count` names starting with `prefix`, in numeric order
    fn take(&self, prefix: &str, count: usize) -> Result<Vec<String>, NativeWeightsError> {
        let names: Vec<String> = self
            .0
            .iter()
            .filter(|(name, _)| name.starts_with(prefix))
            .map(|(name, _)| name.clone())
            .collect();
        if names.len() != count {
            return Err(NativeWeightsError::TensorShape {
                name: format!("{prefix}*"),
                expected: vec![count],
                actual: vec![names.len()],
            });
        }

        Ok(names)
    }
}

/// The learned SincNet band edges, from which the 80 convolution filters are built
struct SincFilterbank {
    low_hz: Vec<f32>,
    band_hz: Vec<f32>,
    /// Hamming half window, `[125]`
    window: Vec<f32>,
    /// `2π t / sample_rate` for the left half of the filter, `[1, 125]`
    n: Vec<f32>,
    /// `n / 2`, stored by the export as its own initializer
    half_n: Vec<f32>,
}

impl SincFilterbank {
    /// Lowest band edge in Hz
    const MIN_LOW_HZ: f32 = 50.0;
    /// Smallest band width in Hz
    const MIN_BAND_HZ: f32 = 50.0;
    /// Nyquist frequency at 16 kHz
    const MAX_HZ: f32 = 8000.0;

    /// The `[80, 1, 251]` filters: 40 sine (band-pass) filters, then 40 cosine filters
    ///
    /// Follows the exported graph op by op in FP32, so the result matches ONNX
    /// Runtime up to the rounding of `sin` and `cos`
    fn filters(&self) -> Vec<f32> {
        let bands = self.low_hz.len();
        let mut sine = Vec::with_capacity(bands * SINC_KERNEL);
        let mut cosine = Vec::with_capacity(bands * SINC_KERNEL);

        for band in 0..bands {
            let low = Self::MIN_LOW_HZ + self.low_hz[band].abs();
            let high = (low + Self::MIN_BAND_HZ + self.band_hz[band].abs())
                .clamp(Self::MIN_LOW_HZ, Self::MAX_HZ);
            let width = high - low;
            let scale = width * 2.0;

            let left: Vec<(f32, f32)> = (0..SINC_HALF)
                .map(|t| {
                    let f_low = low * self.n[t];
                    let f_high = high * self.n[t];
                    let sin = (f_high.sin() - f_low.sin()) / self.half_n[t] * self.window[t];
                    let cos = (f_low.cos() - f_high.cos()) / self.half_n[t] * self.window[t];
                    (sin, cos)
                })
                .collect();

            // [left, center, reversed left], divided by twice the band width
            sine.extend(left.iter().map(|&(sin, _)| sin / scale));
            sine.push(width * 2.0 / scale);
            sine.extend(left.iter().rev().map(|&(sin, _)| sin / scale));

            // the cosine filters are odd: [left, 0, -reversed left]
            cosine.extend(left.iter().map(|&(_, cos)| cos / scale));
            cosine.push(0.0 / scale);
            cosine.extend(left.iter().rev().map(|&(_, cos)| -cos / scale));
        }

        sine.extend(cosine);
        sine
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::super::weights::test_support::TestFile;
    use super::{GeneratedNames, NativeWeightsError, WeightsFile};

    #[test]
    fn generated_initializers_use_numeric_order_and_exact_counts() {
        let file = TestFile::new(
            &json!({
                "onnx::LSTM_100": {"dtype": "F32", "shape": [0], "data_offsets": [0, 0]},
                "onnx::LSTM_9": {"dtype": "F32", "shape": [0], "data_offsets": [0, 0]},
                "onnx::LSTM_10": {"dtype": "F32", "shape": [0], "data_offsets": [0, 0]},
                "onnx::MatMul_8": {"dtype": "F32", "shape": [0], "data_offsets": [0, 0]},
                "onnx::LSTM_text": {"dtype": "F32", "shape": [0], "data_offsets": [0, 0]}
            }),
            &[],
        );
        let weights = WeightsFile::open(&file.0).unwrap();
        let generated = GeneratedNames::new(&weights);
        assert_eq!(
            generated.take("onnx::LSTM_", 3).unwrap(),
            ["onnx::LSTM_9", "onnx::LSTM_10", "onnx::LSTM_100"]
        );
        assert!(matches!(generated.take("onnx::LSTM_", 4),
            Err(NativeWeightsError::TensorShape { name, expected, actual })
                if name == "onnx::LSTM_*" && expected == [4] && actual == [3]));
    }
}
