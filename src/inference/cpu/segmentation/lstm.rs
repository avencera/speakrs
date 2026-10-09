//! Fixed bidirectional ONNX LSTM with IOFC gates

use ndarray::{Array2, ArrayView2, s};

use super::super::gemm::matmul;
use crate::inference::native_model::segmentation::{HIDDEN, LstmLayer};
use crate::inference::{InferenceError, TensorShapeError};

pub(super) struct Direction {
    w: Array2<f32>,
    r: Array2<f32>,
    bias: Vec<f32>,
}

pub(super) struct Lstm([Direction; 2]);

pub(super) struct LstmScratch {
    projection: Array2<f32>,
    hidden: Array2<f32>,
    cell: Vec<f32>,
    gates: Array2<f32>,
}

impl LstmScratch {
    pub(super) fn new() -> Self {
        Self {
            projection: Array2::zeros((589, 4 * HIDDEN)),
            hidden: Array2::zeros((1, HIDDEN)),
            cell: vec![0.0; HIDDEN],
            gates: Array2::zeros((1, 4 * HIDDEN)),
        }
    }
}

impl Lstm {
    pub(super) fn new(layer: LstmLayer) -> Result<Self, TensorShapeError> {
        for (actual, expected) in [
            (layer.w.len(), 2 * 4 * HIDDEN * layer.input),
            (layer.r.len(), 2 * 4 * HIDDEN * HIDDEN),
            (layer.b.len(), 2 * 8 * HIDDEN),
        ] {
            if actual != expected {
                return Err(TensorShapeError::LengthMismatch {
                    context: "CPU recurrent weights",
                    expected,
                    actual,
                });
            }
        }
        let direction = |direction: usize| -> Result<Direction, TensorShapeError> {
            let w_len = 4 * HIDDEN * layer.input;
            let r_len = 4 * HIDDEN * HIDDEN;
            let b = &layer.b[direction * 8 * HIDDEN..(direction + 1) * 8 * HIDDEN];
            Ok(Direction {
                w: array(
                    (4 * HIDDEN, layer.input),
                    layer.w[direction * w_len..(direction + 1) * w_len].to_vec(),
                )?
                .t()
                .as_standard_layout()
                .to_owned(),
                r: array(
                    (4 * HIDDEN, HIDDEN),
                    layer.r[direction * r_len..(direction + 1) * r_len].to_vec(),
                )?
                .t()
                .as_standard_layout()
                .to_owned(),
                bias: (0..4 * HIDDEN)
                    .map(|index| b[index] + b[index + 4 * HIDDEN])
                    .collect(),
            })
        };
        Ok(Self([direction(0)?, direction(1)?]))
    }

    pub(super) fn forward(
        &self,
        input: ArrayView2<'_, f32>,
        output: &mut Array2<f32>,
        scratch: &mut LstmScratch,
    ) -> Result<(), InferenceError> {
        let expected = [589, self.0[0].w.nrows()];
        if input.shape() != expected || output.dim() != (589, 2 * HIDDEN) {
            return Err(crate::inference::TensorShapeError::ShapeMismatch {
                context: "CPU recurrent input/output",
                expected: expected.to_vec(),
                actual: input.shape().to_vec(),
            }
            .into());
        }
        for (direction, weights) in self.0.iter().enumerate() {
            matmul(input, weights.w.view(), scratch.projection.view_mut(), 0.0)?;
            scratch.hidden.fill(0.0);
            scratch.cell.fill(0.0);
            for step in 0..input.nrows() {
                let time = if direction == 0 {
                    step
                } else {
                    input.nrows() - 1 - step
                };
                matmul(
                    scratch.hidden.view(),
                    weights.r.view(),
                    scratch.gates.view_mut(),
                    0.0,
                )?;
                for gate in 0..4 * HIDDEN {
                    scratch.gates[[0, gate]] +=
                        scratch.projection[[time, gate]] + weights.bias[gate];
                }
                for unit in 0..HIDDEN {
                    let (hidden, cell) = step_iofc(
                        [
                            scratch.gates[[0, unit]],
                            scratch.gates[[0, HIDDEN + unit]],
                            scratch.gates[[0, 2 * HIDDEN + unit]],
                            scratch.gates[[0, 3 * HIDDEN + unit]],
                        ],
                        scratch.cell[unit],
                    );
                    scratch.cell[unit] = cell;
                    scratch.hidden[[0, unit]] = hidden;
                }
                output
                    .slice_mut(s![time, direction * HIDDEN..(direction + 1) * HIDDEN])
                    .assign(&scratch.hidden.row(0));
            }
        }
        Ok(())
    }
}

fn step_iofc(gates: [f32; 4], cell: f32) -> (f32, f32) {
    let sigmoid = |value: f32| 1.0 / (1.0 + (-value).exp());
    let [i, o, f, c] = gates;
    let cell = sigmoid(f) * cell + sigmoid(i) * c.tanh();
    (sigmoid(o) * cell.tanh(), cell)
}

pub(super) fn array(
    shape: (usize, usize),
    values: Vec<f32>,
) -> Result<Array2<f32>, TensorShapeError> {
    let actual = values.len();
    Array2::from_shape_vec(shape, values).map_err(|_| TensorShapeError::ShapeMismatch {
        context: "CPU packed matrix",
        expected: vec![shape.0, shape.1],
        actual: vec![actual],
    })
}

#[cfg(test)]
mod tests;
