//! Bounded channels-first cross-correlation

use std::num::NonZeroUsize;

use ndarray::{Array2, ArrayView2, ArrayView3, ArrayViewMut2, ArrayViewMut3, Axis, s};

use super::gemm::matmul;
use crate::inference::{InferenceError, TensorShapeError};

const TILE: usize = 1024;
const CONTEXT: &str = "CPU convolution";

#[derive(Debug, Clone, Copy)]
struct Kernel {
    input: NonZeroUsize,
    output: NonZeroUsize,
    kernel: NonZeroUsize,
    stride: NonZeroUsize,
    rows: usize,
}

impl Kernel {
    fn new(
        input: usize,
        output: usize,
        kernel: usize,
        stride: usize,
        rank: usize,
    ) -> Result<Self, TensorShapeError> {
        let input = nonzero(input)?;
        let output = nonzero(output)?;
        let kernel = nonzero(kernel)?;
        let stride = nonzero(stride)?;
        let mut rows = input.get();
        for _ in 0..rank {
            rows = product(&[rows, kernel.get()])?;
        }
        product(&[rows, output.get()])?;
        product(&[rows, TILE])?;
        product(&[output.get(), TILE])?;
        Ok(Self {
            input,
            output,
            kernel,
            stride,
            rows,
        })
    }
}

fn nonzero(value: usize) -> Result<NonZeroUsize, TensorShapeError> {
    NonZeroUsize::new(value)
        .filter(|value| value.get() <= isize::MAX as usize)
        .ok_or(TensorShapeError::InvalidDimension {
            context: CONTEXT,
            dimension: i64::try_from(value).unwrap_or(i64::MAX),
        })
}

fn product(values: &[usize]) -> Result<usize, TensorShapeError> {
    values.iter().try_fold(1usize, |result, value| {
        result
            .checked_mul(*value)
            .filter(|result| *result <= isize::MAX as usize)
            .ok_or(TensorShapeError::Overflow { context: CONTEXT })
    })
}

fn length(actual: usize, expected: usize) -> Result<(), TensorShapeError> {
    if actual == expected {
        return Ok(());
    }
    Err(TensorShapeError::LengthMismatch {
        context: CONTEXT,
        expected,
        actual,
    })
}

fn shape(actual: &[usize], expected: &[usize]) -> Result<(), TensorShapeError> {
    if actual == expected {
        return Ok(());
    }
    Err(TensorShapeError::ShapeMismatch {
        context: CONTEXT,
        expected: expected.to_vec(),
        actual: actual.to_vec(),
    })
}

fn axis_len(input: usize, kernel: Kernel, padding: usize) -> Result<usize, TensorShapeError> {
    nonzero(input)?;
    let padded = padding
        .checked_mul(2)
        .and_then(|pad| input.checked_add(pad))
        .filter(|value| *value <= isize::MAX as usize)
        .ok_or(TensorShapeError::Overflow { context: CONTEXT })?;
    if padded < kernel.kernel.get() {
        return Err(TensorShapeError::AxisMismatch {
            context: CONTEXT,
            axis: 0,
            expected: kernel.kernel.get(),
            actual: padded,
        });
    }
    Ok((padded - kernel.kernel.get()) / kernel.stride.get() + 1)
}

/// Valid non-zero one-dimensional convolution geometry
#[derive(Debug, Clone, Copy)]
pub(crate) struct Conv1dSpec(Kernel);

impl Conv1dSpec {
    /// Checks geometry and bounded workspace sizes before packing weights
    pub(crate) fn new(
        in_channels: usize,
        out_channels: usize,
        kernel: usize,
        stride: usize,
    ) -> Result<Self, TensorShapeError> {
        Ok(Self(Kernel::new(
            in_channels,
            out_channels,
            kernel,
            stride,
            1,
        )?))
    }

    /// Checks the unpadded output length and total output size
    pub(crate) fn output_len(self, input_len: usize) -> Result<usize, TensorShapeError> {
        let len = axis_len(input_len, self.0, 0)?;
        product(&[self.0.input.get(), input_len])?;
        product(&[self.0.output.get(), len])?;
        Ok(len)
    }
}

/// Packed immutable one-dimensional cross-correlation weights
#[derive(Debug)]
pub(crate) struct Conv1d {
    spec: Conv1dSpec,
    weight: Array2<f32>,
    bias: Option<Vec<f32>>,
}

impl Conv1d {
    /// Validates and packs `[out,in,kernel]` weights in row-major order
    pub(crate) fn new(
        spec: Conv1dSpec,
        weight: Vec<f32>,
        bias: Option<Vec<f32>>,
    ) -> Result<Self, TensorShapeError> {
        let kernel = spec.0;
        length(weight.len(), kernel.output.get() * kernel.rows)?;
        if let Some(bias) = &bias {
            length(bias.len(), kernel.output.get())?;
        }
        let weight = Array2::from_shape_vec((kernel.output.get(), kernel.rows), weight)
            .map_err(|_| TensorShapeError::Overflow { context: CONTEXT })?;
        Ok(Self { spec, weight, bias })
    }

    /// Writes channels/time output using private bounded scratch
    pub(crate) fn forward(
        &self,
        input: ArrayView2<'_, f32>,
        mut output: ArrayViewMut2<'_, f32>,
        workspace: &mut ConvWorkspace,
    ) -> Result<(), InferenceError> {
        let kernel = self.spec.0;
        let out_len = self.spec.output_len(input.ncols())?;
        shape(input.shape(), &[kernel.input.get(), input.ncols()])?;
        shape(output.shape(), &[kernel.output.get(), out_len])?;
        workspace.prepare(kernel.rows, kernel.output.get());
        for start in (0..out_len).step_by(TILE) {
            let count = TILE.min(out_len - start);
            for channel in 0..kernel.input.get() {
                let source = input.row(channel);
                for offset in 0..kernel.kernel.get() {
                    let mut target = workspace
                        .columns
                        .row_mut(channel * kernel.kernel.get() + offset);
                    let first = start * kernel.stride.get() + offset;
                    if kernel.stride.get() == 1
                        && let Some(source) = source.as_slice()
                    {
                        target
                            .as_slice_mut()
                            .ok_or(InferenceError::NonContiguousBuffer { context: CONTEXT })?
                            [..count]
                            .copy_from_slice(&source[first..first + count]);
                    } else {
                        for index in 0..count {
                            target[index] = source[first + index * kernel.stride.get()];
                        }
                    }
                }
            }
            let columns = workspace.columns.slice(s![..kernel.rows, ..count]);
            let mut tile = output.slice_mut(s![.., start..start + count]);
            matmul(self.weight.view(), columns, tile.view_mut(), 0.0)?;
            if let Some(bias) = &self.bias {
                for (mut row, bias) in tile.axis_iter_mut(Axis(0)).zip(bias) {
                    row.mapv_inplace(|value| value + bias);
                }
            }
        }
        Ok(())
    }
}

/// Valid square two-dimensional convolution geometry
#[derive(Debug, Clone, Copy)]
pub(crate) struct Conv2dSpec {
    kernel: Kernel,
    padding: usize,
}

impl Conv2dSpec {
    /// Checks geometry, padding and bounded workspace sizes
    pub(crate) fn new(
        in_channels: usize,
        out_channels: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
    ) -> Result<Self, TensorShapeError> {
        product(&[padding, 2])?;
        Ok(Self {
            kernel: Kernel::new(in_channels, out_channels, kernel, stride, 2)?,
            padding,
        })
    }

    /// Checks height/width and input/output tensor products
    pub(crate) fn output_shape(
        self,
        height: usize,
        width: usize,
    ) -> Result<(usize, usize), TensorShapeError> {
        let output = (
            axis_len(height, self.kernel, self.padding)?,
            axis_len(width, self.kernel, self.padding)?,
        );
        product(&[self.kernel.input.get(), height, width])?;
        product(&[self.kernel.output.get(), output.0, output.1])?;
        Ok(output)
    }
}

/// Packed immutable two-dimensional cross-correlation weights and bias
#[derive(Debug)]
pub(crate) struct Conv2d {
    spec: Conv2dSpec,
    weight: Array2<f32>,
    bias: Vec<f32>,
}

impl Conv2d {
    /// Validates and packs `[out,in,kernel,kernel]` weights in row-major order
    pub(crate) fn new(
        spec: Conv2dSpec,
        weight: Vec<f32>,
        bias: Vec<f32>,
    ) -> Result<Self, TensorShapeError> {
        let kernel = spec.kernel;
        length(weight.len(), kernel.output.get() * kernel.rows)?;
        length(bias.len(), kernel.output.get())?;
        let weight = Array2::from_shape_vec((kernel.output.get(), kernel.rows), weight)
            .map_err(|_| TensorShapeError::Overflow { context: CONTEXT })?;
        Ok(Self { spec, weight, bias })
    }

    /// Writes channels/height/width output, with no activation or residual fusion
    pub(crate) fn forward(
        &self,
        input: ArrayView3<'_, f32>,
        mut output: ArrayViewMut3<'_, f32>,
        workspace: &mut ConvWorkspace,
    ) -> Result<(), InferenceError> {
        let kernel = self.spec.kernel;
        let (height, width) = self.spec.output_shape(input.shape()[1], input.shape()[2])?;
        shape(
            input.shape(),
            &[kernel.input.get(), input.shape()[1], input.shape()[2]],
        )?;
        shape(output.shape(), &[kernel.output.get(), height, width])?;
        workspace.prepare(kernel.rows, kernel.output.get());
        for start in (0..height * width).step_by(TILE) {
            let count = TILE.min(height * width - start);
            self.gather(input, width, start, count, workspace)?;
            // contiguous outputs use their native channel stride without a tile copy
            if let Some(flat) = output.as_slice_mut() {
                use ndarray::ShapeBuilder;
                let tile = ArrayViewMut2::from_shape(
                    (kernel.output.get(), count).strides((height * width, 1)),
                    &mut flat[start..],
                )
                .map_err(|source| InferenceError::OutputArray {
                    context: CONTEXT,
                    source,
                })?;
                matmul(
                    self.weight.view(),
                    workspace.columns.slice(s![..kernel.rows, ..count]),
                    tile,
                    0.0,
                )?;
                for (channel, bias) in self.bias.iter().enumerate() {
                    let first = channel * height * width + start;
                    for value in &mut flat[first..first + count] {
                        *value += bias;
                    }
                }
                continue;
            }

            let tile = workspace
                .result
                .slice_mut(s![..kernel.output.get(), ..count]);
            matmul(
                self.weight.view(),
                workspace.columns.slice(s![..kernel.rows, ..count]),
                tile,
                0.0,
            )?;
            for channel in 0..kernel.output.get() {
                for local in 0..count {
                    let position = start + local;
                    output[[channel, position / width, position % width]] =
                        workspace.result[[channel, local]] + self.bias[channel];
                }
            }
        }
        Ok(())
    }

    fn gather(
        &self,
        input: ArrayView3<'_, f32>,
        out_width: usize,
        start: usize,
        count: usize,
        workspace: &mut ConvWorkspace,
    ) -> Result<(), InferenceError> {
        let kernel = self.spec.kernel;
        let height = input.shape()[1];
        let width = input.shape()[2];
        let padding = self.spec.padding as isize;
        for channel in 0..kernel.input.get() {
            let plane = input.index_axis(Axis(0), channel);
            for ky in 0..kernel.kernel.get() {
                for kx in 0..kernel.kernel.get() {
                    let row = (channel * kernel.kernel.get() + ky) * kernel.kernel.get() + kx;
                    let mut target = workspace.columns.row_mut(row);
                    let target = &mut target
                        .as_slice_mut()
                        .ok_or(InferenceError::NonContiguousBuffer { context: CONTEXT })?[..count];
                    let mut local = 0;
                    while local < count {
                        let position = start + local;
                        let y =
                            (position / out_width * kernel.stride.get() + ky) as isize - padding;
                        let x =
                            (position % out_width * kernel.stride.get() + kx) as isize - padding;
                        let span = (out_width - position % out_width).min(count - local);
                        let destination = &mut target[local..local + span];
                        local += span;
                        if y < 0 || y >= height as isize {
                            destination.fill(0.0);
                            continue;
                        }
                        let source = plane.row(y as usize);
                        if kernel.stride.get() == 1
                            && let Some(source) = source.as_slice()
                        {
                            let left = (-x).max(0).min(span as isize) as usize;
                            let right = (width as isize - x).max(0).min(span as isize) as usize;
                            destination[..left].fill(0.0);
                            destination[right..].fill(0.0);
                            if right > left {
                                let first = (x + left as isize) as usize;
                                destination[left..right]
                                    .copy_from_slice(&source[first..first + right - left]);
                            }
                            continue;
                        }
                        for (index, value) in destination.iter_mut().enumerate() {
                            let sx = x + (index * kernel.stride.get()) as isize;
                            *value = if sx >= 0 && sx < width as isize {
                                source[sx as usize]
                            } else {
                                0.0
                            };
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

/// Reusable private im2col and strided-output tiles, each bounded to1024 columns
#[derive(Debug, Default)]
pub(crate) struct ConvWorkspace {
    columns: Array2<f32>,
    result: Array2<f32>,
}

impl ConvWorkspace {
    /// Creates empty scratch that grows only when a convolution needs more rows
    pub(crate) fn new() -> Self {
        Self::default()
    }

    fn prepare(&mut self, rows: usize, outputs: usize) {
        if self.columns.nrows() < rows {
            self.columns = Array2::zeros((rows, TILE));
        }
        if self.result.nrows() < outputs {
            self.result = Array2::zeros((outputs, TILE));
        }
    }
}

#[cfg(test)]
mod tests;
