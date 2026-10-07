//! Independent textbook f64 definitions for the secret operator gate

use super::cpu;

#[path = "fbank_truth.rs"]
pub(super) mod fbank_truth;
use crate::inference::cuda::dnn::Conv2d;
use crate::inference::cuda::weights::uniform;
use std::collections::BTreeSet;

/// An output sample, with indices into the full NCHW or batch-major output
pub(crate) struct Sample {
    pub(crate) indices: Vec<usize>,
    pub(crate) values: Vec<f64>,
}

/// Stratified samples cover every row and channel, spatial edges and random positions
pub(crate) fn indices(shape: &[usize], state: &mut u64) -> Vec<usize> {
    let batch = shape[0];
    let channels = shape[1];
    let spatial: usize = shape[2..].iter().product();
    let total = batch * channels * spatial;
    let mut selected = BTreeSet::new();
    // cover rows and channels separately so the CPU reference stays bounded at large batches
    let mut pairs: Vec<_> = (0..batch)
        .map(|row| (row, (uniform(state) * channels as f32) as usize))
        .collect();
    pairs.extend((0..channels).map(|channel| ((uniform(state) * batch as f32) as usize, channel)));
    for (row, channel) in pairs {
        let base = (row * channels + channel) * spatial;
        for edge in [0, spatial - 1] {
            selected.insert(base + edge);
        }
        if shape.len() == 4 {
            let width = shape[3];
            for edge in [width - 1, spatial - width] {
                selected.insert(base + edge);
            }
        }
        selected.insert(base + (uniform(state) * spatial as f32) as usize);
    }
    while selected.len() < 4096.min(total) {
        selected.insert((uniform(state) * total as f32) as usize);
    }
    selected.into_iter().collect()
}

/// NCHW cross-correlation with zero padding, bias, optional residual, then ReLU
pub(crate) fn conv(
    spec: &Conv2d,
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    residual: Option<&[f32]>,
    indices: Vec<usize>,
) -> Sample {
    let [height, width] = spec.input;
    let [out_height, out_width] = spec.output();
    assert_eq!(spec.kernel, [3, 3]);
    cpu::evaluate(|mode| {
        let values = cpu::ordered_map(mode, indices.len(), |position| {
            let index = indices[position];
            let col = index % out_width;
            let row = index / out_width % out_height;
            let channel = index / (out_width * out_height) % spec.out_channels;
            let batch = index / (out_width * out_height * spec.out_channels);
            let mut sum = 0.0f64;
            for c in 0..spec.in_channels {
                for ky in 0..3 {
                    let y = (row * spec.stride[0] + ky * spec.dilation[0]) as isize
                        - spec.padding[0] as isize;
                    if !(0..height as isize).contains(&y) {
                        continue;
                    }
                    for kx in 0..3 {
                        let x = (col * spec.stride[1] + kx * spec.dilation[1]) as isize
                            - spec.padding[1] as isize;
                        if !(0..width as isize).contains(&x) {
                            continue;
                        }
                        let i = ((batch * spec.in_channels + c) * height + y as usize) * width
                            + x as usize;
                        let w = ((channel * spec.in_channels + c) * 3 + ky) * 3 + kx;
                        sum += f64::from(input[i]) * f64::from(weight[w]);
                    }
                }
            }
            sum += f64::from(bias[channel]);
            if let Some(residual) = residual {
                sum += f64::from(residual[index]);
            }
            sum.max(0.0)
        });
        Sample {
            indices: indices.clone(),
            values,
        }
    })
}

/// Valid 251-tap cross-correlation at stride 10, absolute value, then max-pool by 3
pub(crate) fn sinc(input: &[f32], filters: &[f32], indices: Vec<usize>) -> Sample {
    let samples = 160_000;
    let pooled = 5325;
    cpu::evaluate(|mode| {
        let values = cpu::ordered_map(mode, indices.len(), |position| {
            let index = indices[position];
            let position = index % pooled;
            let channel = index / pooled % 80;
            let batch = index / (pooled * 80);
            (0..3)
                .map(|offset| {
                    let start = batch * samples + (3 * position + offset) * 10;
                    (0..251)
                        .map(|tap| {
                            f64::from(input[start + tap]) * f64::from(filters[channel * 251 + tap])
                        })
                        .sum::<f64>()
                        .abs()
                })
                .fold(0.0, f64::max)
        });
        Sample {
            indices: indices.clone(),
            values,
        }
    })
}

/// One ONNX bidirectional layer, gates ordered i, o, f, c
pub(crate) struct Lstm<'a> {
    pub(crate) input: usize,
    pub(crate) w: &'a [f32],
    pub(crate) r: &'a [f32],
    pub(crate) b: &'a [f32],
}

/// Full recurrence of all layers, both directions and the entire sequence, for selected rows
pub(crate) fn lstm(input: &[f32], frames: usize, layers: &[Lstm<'_>], rows: &[usize]) -> Sample {
    cpu::evaluate(|mode| {
        let samples = cpu::ordered_map(mode, rows.len(), |position| {
            lstm_row(input, frames, layers, rows[position])
        });
        Sample {
            indices: samples
                .iter()
                .flat_map(|sample| sample.indices.iter().copied())
                .collect(),
            values: samples
                .into_iter()
                .flat_map(|sample| sample.values)
                .collect(),
        }
    })
}

fn lstm_row(input: &[f32], frames: usize, layers: &[Lstm<'_>], row: usize) -> Sample {
    let hidden = 128;
    let width = layers[0].input;
    let mut sequence: Vec<f64> = input[row * frames * width..(row + 1) * frames * width]
        .iter()
        .copied()
        .map(f64::from)
        .collect();
    for layer in layers {
        let mut output = vec![0.0; frames * hidden * 2];
        for direction in 0..2 {
            let mut h = vec![0.0; hidden];
            let mut c = vec![0.0; hidden];
            for step in 0..frames {
                let t = if direction == 0 {
                    step
                } else {
                    frames - 1 - step
                };
                let x = &sequence[t * layer.input..(t + 1) * layer.input];
                let mut gates = vec![0.0; hidden * 4];
                for (gate, value) in gates.iter_mut().enumerate() {
                    let base = direction * 4 * hidden + gate;
                    *value = f64::from(layer.b[direction * 8 * hidden + gate])
                        + f64::from(layer.b[direction * 8 * hidden + 4 * hidden + gate]);
                    for (k, &x) in x.iter().enumerate() {
                        *value += x * f64::from(layer.w[base * layer.input + k]);
                    }
                    for (k, &h) in h.iter().enumerate() {
                        *value += h * f64::from(layer.r[base * hidden + k]);
                    }
                }
                let sigmoid = |v: f64| 1.0 / (1.0 + (-v).exp());
                for k in 0..hidden {
                    c[k] = sigmoid(gates[2 * hidden + k]) * c[k]
                        + sigmoid(gates[k]) * gates[3 * hidden + k].tanh();
                    h[k] = sigmoid(gates[hidden + k]) * c[k].tanh();
                    output[t * 2 * hidden + direction * hidden + k] = h[k];
                }
            }
        }
        sequence = output;
    }
    Sample {
        indices: (row * frames * 2 * hidden..(row + 1) * frames * 2 * hidden).collect(),
        values: sequence,
    }
}

/// At least two distinct rows, or the full batch when only one row exists
pub(crate) fn batch_rows(batch: usize, state: &mut u64) -> Vec<usize> {
    let first = (uniform(state) * batch as f32) as usize;
    if batch == 1 {
        return vec![0];
    }
    let second = (first + 1 + (uniform(state) * (batch - 1) as f32) as usize) % batch;
    vec![first, second]
}

#[test]
fn sampling_covers_rows_channels_and_edges() {
    for batch in [1, 7, 32, 33, 64] {
        let shape = [batch, 512, 20, 148];
        let sample = indices(&shape, &mut 19);
        assert_eq!(sample.len(), 4096);
        for edge in [0, 147, 19 * 148, 20 * 148 - 1] {
            for row in 0..batch {
                assert!(
                    sample
                        .iter()
                        .any(|&i| i / (512 * 20 * 148) == row && i % (20 * 148) == edge)
                );
            }
            for channel in 0..512 {
                assert!(
                    sample
                        .iter()
                        .any(|&i| i / (20 * 148) % 512 == channel && i % (20 * 148) == edge)
                );
            }
        }
    }

    assert_eq!(indices(&[1, 1, 2, 2], &mut 19), [0, 1, 2, 3]);
    assert_ne!(batch_rows(7, &mut 1)[0], batch_rows(7, &mut 1)[1]);
}

#[test]
fn textbook_conv_padding_and_epilogue() {
    use crate::inference::cuda::CudaMath;
    let spec = Conv2d {
        batch: 1,
        in_channels: 1,
        out_channels: 1,
        input: [2, 2],
        kernel: [3, 3],
        padding: [1, 1],
        stride: [1, 1],
        dilation: [1, 1],
        math: CudaMath::Fp32,
    };
    let result = conv(
        &spec,
        &[1., 2., 3., 4.],
        &[1.; 9],
        &[-2.],
        Some(&[-9., 1., 2., 3.]),
        vec![0, 1, 2, 3],
    );
    assert_eq!(result.values, [0., 9., 10., 11.]);
}

/// Fixture outputs round through f32 reductions, so compare their rounding-scale errors
pub(crate) fn assert_rounding(truth: &Sample, expected: &[f32], l2_bound: f64, abs_bound: f64) {
    let mut num = 0.0;
    let mut den = 0.0;
    let mut max_abs = 0.0f64;
    for (&i, &value) in truth.indices.iter().zip(&truth.values) {
        let rounded = f64::from(value as f32);
        let error = rounded - f64::from(expected[i]);
        num += error * error;
        den += rounded * rounded;
        max_abs = max_abs.max(error.abs());
    }
    let l2 = (num / den).sqrt();
    println!("fixture rounding relative_l2={l2} max_abs={max_abs}");
    assert!(l2 <= l2_bound && max_abs <= abs_bound);
}

#[test]
#[ignore = "requires reference fixtures, CPU only"]
fn conv_f64_matches_fixture_rounding() -> Result<(), crate::inference::cuda::CudaError> {
    use crate::inference::cuda::{CudaMath, SafetensorsFile};
    let weights =
        SafetensorsFile::open("/workspace/models-native/wespeaker-multimask-tail.safetensors")?;
    let file =
        SafetensorsFile::open("/workspace/ref/wespeaker-multimask-tail/test_first_b1.safetensors")?;
    for block in 0..7 {
        for second in [false, true] {
            let layer = if block < 3 {
                format!("resnet.layer1.{block}")
            } else {
                format!("resnet.layer2.{}", block - 3)
            };
            let prefix = format!("{layer}.conv{}", if second { 2 } else { 1 });
            let input_name = if second {
                format!("tensor/relu_{}", 2 * block + 1)
            } else if block == 0 {
                "tensor/relu".into()
            } else {
                format!("tensor/relu_{}", 2 * block)
            };
            let output_name = format!("tensor/relu_{}", 2 * block + if second { 2 } else { 1 });
            let x = file.shape(&input_name).expect("input shape");
            let y = file.shape(&output_name).expect("output shape");
            let spec = Conv2d {
                batch: 1,
                in_channels: x[1],
                out_channels: y[1],
                input: [x[2], x[3]],
                kernel: [3, 3],
                padding: [1, 1],
                stride: if block == 3 && !second {
                    [2, 2]
                } else {
                    [1, 1]
                },
                dilation: [1, 1],
                math: CudaMath::Fp32,
            };
            let input = file.read_f32(&input_name, x)?;
            let expected = file.read_f32(&output_name, y)?;
            let weight = weights.read_f32(&format!("{prefix}.weight"), &spec.filter_shape())?;
            let bias = weights.read_f32(&format!("{prefix}.weight_bias"), &[y[1]])?;
            let residual = if second {
                let name = if block == 3 {
                    "tensor/getitem_27".into()
                } else if block == 0 {
                    "tensor/relu".into()
                } else {
                    format!("tensor/relu_{}", 2 * block)
                };
                Some(file.read_f32(&name, y)?)
            } else {
                None
            };
            let truth = conv(
                &spec,
                &input,
                &weight,
                &bias,
                residual.as_deref(),
                indices(y, &mut 73),
            );
            println!("{prefix}");
            assert_rounding(&truth, &expected, 2e-6, 5e-5);
        }
    }
    Ok(())
}

/// Exact Hamming coefficients used by the independent producer and stage truth
fn fbank_window() -> Vec<f64> {
    use std::f64::consts::TAU;
    (0..400)
        .map(|n| 0.54 - 0.46 * (TAU * n as f64 / 399.0).cos())
        .collect()
}

/// Direct DFT basis with the original reduced-angle definition
fn fbank_basis() -> Vec<Vec<(f64, f64)>> {
    use std::f64::consts::TAU;
    (0..257)
        .map(|bin| {
            (0..400)
                .map(|sample| {
                    let angle = TAU * ((bin * sample) % 512) as f64 / 512.0;
                    (angle.cos(), angle.sin())
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

/// Scale, remove the ordered frame mean, pre-emphasize and apply exact Hamming
fn fbank_frame(input: &[f32], offset: usize, window: &[f64]) -> Vec<f64> {
    let audio: Vec<_> = input[offset..offset + 400]
        .iter()
        .map(|&x| f64::from(x) * 32768.0)
        .collect();
    let mean = audio.iter().sum::<f64>() / 400.0;
    (0..400)
        .map(|n| ((audio[n] - mean) - 0.97 * (audio[n.saturating_sub(1)] - mean)) * window[n])
        .collect()
}

/// Direct f64 DFT and mel energy, with a fixed sample, bin and reduction order
pub(crate) fn fbank(input: &[f32], mel: &[f32], indices: Vec<usize>) -> Sample {
    assert_eq!(mel.len(), 257 * 80);
    let window = fbank_window();
    let basis = fbank_basis();
    cpu::evaluate(|mode| {
        let values = cpu::ordered_map(mode, indices.len(), |position| {
            let index = indices[position];
            let bin = index % 80;
            let frame = index / 80 % 998;
            let row = index / (998 * 80);
            let offset = row * 160_000 + frame * 160;
            let samples = fbank_frame(input, offset, &window);
            let mut energy = 0.0;
            for k in 0..257 {
                let weight = f64::from(mel[k * 80 + bin]);
                if weight == 0.0 {
                    continue;
                }
                let mut real = 0.0;
                let mut imaginary = 0.0;
                for (sample, &(cosine, sine)) in samples.iter().zip(&basis[k]) {
                    real += sample * cosine;
                    imaginary += sample * sine;
                }
                energy += (real * real + imaginary * imaginary) * weight;
            }
            energy
        });
        Sample {
            indices: indices.clone(),
            values,
        }
    })
}

#[test]
fn lstm_rows_keep_their_order_and_full_recurrence() {
    let w = vec![0.0; 2 * 4 * 128];
    let r = vec![0.0; 2 * 4 * 128 * 128];
    let mut b = vec![0.0; 2 * 8 * 128];
    for direction in 0..2 {
        b[direction * 8 * 128 + 3 * 128..direction * 8 * 128 + 4 * 128].fill(0.5);
    }
    let layer = Lstm {
        input: 1,
        w: &w,
        r: &r,
        b: &b,
    };
    let output = lstm(&[0.1, 0.2, 0.3, 0.4], 2, &[layer], &[1, 0]);
    assert_eq!(
        output.indices,
        (512..1024).chain(0..512).collect::<Vec<_>>()
    );
    assert_eq!(output.values.len(), 1024);
    let first = 0.5 * (0.5 * 0.5f64.tanh()).tanh();
    let second = 0.5 * (0.75 * 0.5f64.tanh()).tanh();
    for row in 0..2 {
        for unit in 0..128 {
            for (offset, expected) in [(0, first), (128, second), (256, second), (384, first)] {
                assert!((output.values[row * 512 + offset + unit] - expected).abs() < 1e-15);
            }
        }
    }
}

#[test]
fn fbank_sample_mapping_preserves_the_quadratic_energy_scale() {
    let row: Vec<_> = (0..160_000)
        .map(|index| (index as f32 * 0.04).sin() * 0.1)
        .collect();
    let audio: Vec<_> = row
        .iter()
        .copied()
        .chain(row.iter().map(|x| x * 2.0))
        .collect();
    let mel = crate::inference::cuda::fbank::FbankConstants::new();
    let sample = fbank(&audio, mel.mel(), vec![0, 79, 998 * 80, 2 * 998 * 80 - 1]);
    assert_eq!(sample.indices, [0, 79, 998 * 80, 2 * 998 * 80 - 1]);
    assert!(
        sample
            .values
            .iter()
            .all(|energy| energy.is_finite() && *energy > 0.0)
    );
    assert_eq!(sample.values[2], sample.values[0] * 4.0);
}
