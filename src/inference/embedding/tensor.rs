use ndarray::{Array1, Array2, Array3};
use ort::memory::Allocator;
use ort::session::{HasSelectedOutputs, OutputSelector, RunOptions};
use ort::value::Tensor;

pub(super) fn array1_slice<'a>(
    array: &'a Array1<f32>,
    context: &'static str,
) -> Result<&'a [f32], ort::Error> {
    array
        .as_slice()
        .ok_or_else(|| ort::Error::new(format!("{context}: mask buffer was not contiguous")))
}

pub(super) fn array2_from_shape_vec(
    rows: usize,
    cols: usize,
    data: Vec<f32>,
    context: &'static str,
) -> Result<Array2<f32>, ort::Error> {
    Array2::from_shape_vec((rows, cols), data)
        .map_err(|error| ort::Error::new(format!("{context}: invalid output shape: {error}")))
}

#[cfg(feature = "coreml")]
pub(super) fn array2_slice<'a>(
    array: &'a Array2<f32>,
    context: &'static str,
) -> Result<&'a [f32], ort::Error> {
    array
        .as_slice()
        .ok_or_else(|| ort::Error::new(format!("{context}: array buffer was not contiguous")))
}

pub(super) fn array2_slice_mut<'a>(
    array: &'a mut Array2<f32>,
    context: &'static str,
) -> Result<&'a mut [f32], ort::Error> {
    array
        .as_slice_mut()
        .ok_or_else(|| ort::Error::new(format!("{context}: array buffer was not contiguous")))
}

#[cfg(feature = "coreml")]
pub(super) fn array3_slice<'a>(
    array: &'a Array3<f32>,
    context: &'static str,
) -> Result<&'a [f32], ort::Error> {
    array
        .as_slice()
        .ok_or_else(|| ort::Error::new(format!("{context}: array buffer was not contiguous")))
}

pub(super) fn array3_slice_mut<'a>(
    array: &'a mut Array3<f32>,
    context: &'static str,
) -> Result<&'a mut [f32], ort::Error> {
    array
        .as_slice_mut()
        .ok_or_else(|| ort::Error::new(format!("{context}: array buffer was not contiguous")))
}

pub(super) fn preallocated_run_options(
    rows: usize,
    cols: usize,
    context: &'static str,
) -> Result<RunOptions<HasSelectedOutputs>, ort::Error> {
    let output = Tensor::<f32>::new(&Allocator::default(), [rows, cols]).map_err(|error| {
        ort::Error::new(format!(
            "{context}: failed to allocate output tensor: {error}"
        ))
    })?;
    RunOptions::new()
        .map_err(|error| {
            ort::Error::new(format!("{context}: failed to build run options: {error}"))
        })
        .map(|options| {
            options.with_outputs(OutputSelector::default().preallocate("output", output))
        })
}
