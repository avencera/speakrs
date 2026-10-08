//! Native PyanNet with one useful frontend window in private scratch

mod lstm;

use super::{
    conv::{Conv1d, Conv1dSpec, ConvWorkspace},
    gemm::matmul,
};
use crate::inference::native_model::{
    WeightsFile,
    segmentation::{
        Affine, LEAKY_SLOPE, NORM_EPSILON, POOL, PyanNetWeights, SINC_STRIDE, WINDOW_SAMPLES,
    },
};
use crate::inference::{CpuError, InferenceError};
use lstm::{Lstm, LstmScratch, array};
use ndarray::{Array2, ArrayView2};
use std::sync::Arc;

struct PackedWeights {
    wav_norm: Affine,
    conv: [Conv1d; 3],
    norms: [Affine; 3],
    lstm: [Lstm; 4],
    linear: [(Array2<f32>, Vec<f32>); 3],
}

/// Immutable validated PyanNet parameters shared between handles
#[derive(Clone)]
pub(crate) struct CpuPyanNet(Arc<PackedWeights>);

/// Private reusable activation and recurrent state for one handle
pub(crate) struct PyanNetWorkspace {
    waveform: Array2<f32>,
    conv: [Array2<f32>; 3],
    pooled: [Array2<f32>; 3],
    recurrent: [Array2<f32>; 2],
    head: [Array2<f32>; 3],
    convolution: ConvWorkspace,
    lstm: LstmScratch,
}

impl CpuPyanNet {
    /// Validates the deployed host topology and packs all matrix operands
    pub(crate) fn load(file: &WeightsFile) -> Result<Self, CpuError> {
        let weights = PyanNetWeights::load(file)?;
        Self::pack(weights)
    }

    fn pack(weights: PyanNetWeights) -> Result<Self, CpuError> {
        let conv = [
            Conv1d::new(
                Conv1dSpec::new(1, 80, 251, SINC_STRIDE)?,
                weights.sinc_filters,
                None,
            )?,
            Conv1d::new(
                Conv1dSpec::new(80, 60, 5, 1)?,
                weights.conv1.weight,
                Some(weights.conv1.bias),
            )?,
            Conv1d::new(
                Conv1dSpec::new(60, 60, 5, 1)?,
                weights.conv2.weight,
                Some(weights.conv2.bias),
            )?,
        ];
        let [l0, l1, l2, l3] = weights.lstm;
        let lstm = [
            Lstm::new(l0)?,
            Lstm::new(l1)?,
            Lstm::new(l2)?,
            Lstm::new(l3)?,
        ];
        let pack_linear = |linear: crate::inference::native_model::segmentation::Linear| {
            let out = linear.bias.len();
            array((linear.weight.len() / out, out), linear.weight)
                .map(|weight| (weight, linear.bias))
        };
        let [h0, h1, h2] = weights.linear;
        let linear = [pack_linear(h0)?, pack_linear(h1)?, pack_linear(h2)?];
        Ok(Self(Arc::new(PackedWeights {
            wav_norm: weights.wav_norm,
            conv,
            norms: weights.norms,
            lstm,
            linear,
        })))
    }

    /// Creates unshared scratch with zero recurrent state
    pub(crate) fn workspace(&self) -> PyanNetWorkspace {
        PyanNetWorkspace {
            waveform: Array2::zeros((1, WINDOW_SAMPLES)),
            conv: [
                Array2::zeros((80, 15975)),
                Array2::zeros((60, 5321)),
                Array2::zeros((60, 1769)),
            ],
            pooled: [
                Array2::zeros((80, 5325)),
                Array2::zeros((60, 1773)),
                Array2::zeros((60, 589)),
            ],
            recurrent: [Array2::zeros((589, 256)), Array2::zeros((589, 256))],
            head: [
                Array2::zeros((589, 128)),
                Array2::zeros((589, 128)),
                Array2::zeros((589, 7)),
            ],
            convolution: ConvWorkspace::new(),
            lstm: LstmScratch::new(),
        }
    }

    /// Pads or truncates one window, resets recurrent state, and returns 589x7 log probabilities
    pub(crate) fn forward(
        &self,
        audio: &[f32],
        scratch: &mut PyanNetWorkspace,
    ) -> Result<Array2<f32>, InferenceError> {
        scratch.waveform.fill(0.0);
        for (target, source) in scratch.waveform.iter_mut().zip(audio) {
            *target = *source;
        }
        normalize(&mut scratch.waveform, &self.0.wav_norm, false);
        for stage in 0..3 {
            let input = if stage == 0 {
                scratch.waveform.view()
            } else {
                scratch.pooled[stage - 1].view()
            };
            self.0.conv[stage].forward(
                input,
                scratch.conv[stage].view_mut(),
                &mut scratch.convolution,
            )?;
            pool(
                scratch.conv[stage].view(),
                &mut scratch.pooled[stage],
                stage == 0,
            );
            normalize(&mut scratch.pooled[stage], &self.0.norms[stage], true);
        }
        self.0.lstm[0].forward(
            scratch.pooled[2].t(),
            &mut scratch.recurrent[0],
            &mut scratch.lstm,
        )?;
        for layer in 1..4 {
            let (left, right) = scratch.recurrent.split_at_mut(1);
            let (input, output) = if layer % 2 == 1 {
                (&left[0], &mut right[0])
            } else {
                (&right[0], &mut left[0])
            };
            self.0.lstm[layer].forward(input.view(), output, &mut scratch.lstm)?;
        }
        for stage in 0..3 {
            let (left, right) = scratch.head.split_at_mut(stage);
            let input = if stage == 0 {
                scratch.recurrent[1].view()
            } else {
                left[stage - 1].view()
            };
            let (weight, bias) = &self.0.linear[stage];
            matmul(input, weight.view(), right[0].view_mut(), 0.0)?;
            for mut row in right[0].rows_mut() {
                for (value, bias) in row.iter_mut().zip(bias) {
                    *value += bias;
                    if stage < 2 {
                        *value = leaky(*value);
                    }
                }
                if stage == 2 {
                    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let log_sum = row
                        .iter()
                        .map(|value| (*value - max).exp())
                        .sum::<f32>()
                        .ln();
                    row.mapv_inplace(|value| value - max - log_sum);
                }
            }
        }
        Ok(scratch.head[2].clone())
    }
}

fn pool(input: ArrayView2<'_, f32>, output: &mut Array2<f32>, absolute: bool) {
    for channel in 0..output.nrows() {
        for time in 0..output.ncols() {
            output[[channel, time]] = (0..POOL)
                .map(|offset| {
                    let value = input[[channel, time * POOL + offset]];
                    if absolute { value.abs() } else { value }
                })
                .fold(f32::NEG_INFINITY, f32::max);
        }
    }
}

fn normalize(values: &mut Array2<f32>, affine: &Affine, activation: bool) {
    for (channel, mut row) in values.rows_mut().into_iter().enumerate() {
        let mean = row.iter().map(|value| f64::from(*value)).sum::<f64>() / row.len() as f64;
        let variance = row
            .iter()
            .map(|value| (f64::from(*value) - mean).powi(2))
            .sum::<f64>()
            / row.len() as f64;
        let inverse = (variance as f32 + NORM_EPSILON).sqrt().recip();
        row.mapv_inplace(|value| {
            let value =
                (value - mean as f32) * inverse * affine.gamma[channel] + affine.beta[channel];
            if activation { leaky(value) } else { value }
        });
    }
}

fn leaky(value: f32) -> f32 {
    if value >= 0.0 {
        value
    } else {
        value * LEAKY_SLOPE
    }
}

#[cfg(test)]
mod tests;
