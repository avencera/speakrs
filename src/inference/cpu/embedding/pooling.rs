//! Deployed centered masked statistics, including degenerate NaN values

use crate::inference::{InferenceError, TensorShapeError};
use ndarray::{ArrayView2, ArrayViewMut1};

pub(super) fn pool(
    features: ArrayView2<'_, f32>,
    mask: &[f32],
    mut output: ArrayViewMut1<'_, f32>,
) -> Result<(), InferenceError> {
    let channels = features.nrows();
    let frames = features.ncols();
    if frames == 0 || output.len() != 2 * channels {
        return Err(TensorShapeError::ShapeMismatch {
            context: "CPU masked pooling",
            expected: vec![2 * channels, 1],
            actual: vec![output.len(), frames],
        }
        .into());
    }
    let weights: Vec<f32> = (0..frames)
        .map(|time| mask.get(time * 589 / frames).copied().unwrap_or(0.0))
        .collect();
    let sum = weights.iter().sum::<f32>();
    if sum <= 0.0 {
        for channel in 0..channels {
            output[channel] = 0.0;
            output[channels + channel] = 1e-5;
        }
        return Ok(());
    }
    // no epsilon is added: a single effective frame must keep the deployed NaN result
    let total = sum;
    let squared = weights.iter().map(|value| value * value).sum::<f32>();
    let denominator = total - squared / total;
    for channel in 0..channels {
        let row = features.row(channel);
        let mean = row
            .iter()
            .zip(&weights)
            .map(|(value, weight)| value * weight)
            .sum::<f32>()
            / total;
        let numerator = row
            .iter()
            .zip(&weights)
            .map(|(value, weight)| (value - mean).powi(2) * weight)
            .sum::<f32>();
        let variance = numerator / denominator;
        let variance = if variance < 1e-10 { 1e-10 } else { variance };
        output[channel] = mean;
        output[channels + channel] = variance.sqrt();
    }
    Ok(())
}

#[cfg(test)]
mod tests;
