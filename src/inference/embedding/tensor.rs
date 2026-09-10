use ndarray::{Array1, Array2, Array3};

use super::{EMBEDDING_WIDTH, FBANK_FEATURES};
use crate::inference::geometry::{GeometryError, TensorLayout};
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

pub(super) fn embedding_vector(
    data: Vec<f32>,
    context: &'static str,
) -> Result<Array1<f32>, ort::Error> {
    if data.len() != EMBEDDING_WIDTH {
        return Err(ort::Error::new(format!(
            "{context}: expected embedding width {EMBEDDING_WIDTH}, got {}",
            data.len()
        )));
    }
    Ok(Array1::from_vec(data))
}

pub(super) fn embedding_batch(
    data: &[f32],
    rows: usize,
    context: &'static str,
) -> Result<Array2<f32>, ort::Error> {
    let expected = rows
        .checked_mul(EMBEDDING_WIDTH)
        .ok_or_else(|| ort::Error::new(format!("{context}: embedding batch size overflowed")))?;
    if data.len() < expected {
        return Err(ort::Error::new(format!(
            "{context}: expected at least {expected} values for {rows} embeddings of width {EMBEDDING_WIDTH}, got {}",
            data.len()
        )));
    }
    array2_from_shape_vec(rows, EMBEDDING_WIDTH, data[..expected].to_vec(), context)
}

pub(super) fn fbank_hw_from_shape(
    shape: &[usize],
    context: &'static str,
) -> Result<(usize, usize), ort::Error> {
    let layout = TensorLayout::from_dims(shape, context).map_err(GeometryError::into_ort)?;
    let (_, frames, features) = layout.try_rank3(context).map_err(GeometryError::into_ort)?;
    if features != FBANK_FEATURES {
        return Err(ort::Error::new(format!(
            "{context}: expected {FBANK_FEATURES} filterbank features, got {features}"
        )));
    }
    Ok((frames, features))
}

pub(super) fn fbank_hw_from_i64(
    shape: &[i64],
    context: &'static str,
) -> Result<(usize, usize), ort::Error> {
    if shape.iter().any(|dim| *dim < 0) {
        return Err(ort::Error::new(format!(
            "{context}: expected non-negative filterbank dimensions, got {shape:?}"
        )));
    }
    let dims: Vec<usize> = shape.iter().map(|dim| *dim as usize).collect();
    fbank_hw_from_shape(&dims, context)
}

pub(super) fn first_output<T>(
    outputs: impl IntoIterator<Item = T>,
    context: &'static str,
) -> Result<T, ort::Error> {
    outputs
        .into_iter()
        .next()
        .ok_or_else(|| ort::Error::new(format!("{context}: missing output tensor")))
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

#[cfg(test)]
mod tests {
    use super::{
        EMBEDDING_WIDTH, FBANK_FEATURES, embedding_batch, embedding_vector, fbank_hw_from_i64,
        fbank_hw_from_shape, first_output,
    };

    #[test]
    fn first_output_reports_missing_tensor() {
        let error = first_output(Vec::<()>::new(), "embedding test").unwrap_err();

        assert_eq!(error.to_string(), "embedding test: missing output tensor");
    }

    #[test]
    fn embedding_vector_rejects_short_and_long_output() {
        let short = embedding_vector(vec![0.0; EMBEDDING_WIDTH - 1], "masked embedding output")
            .unwrap_err();
        let long = embedding_vector(vec![0.0; EMBEDDING_WIDTH + 1], "masked embedding output")
            .unwrap_err();
        assert!(short.to_string().contains("expected embedding width 256"));
        assert!(long.to_string().contains("got 257"));
        assert_eq!(
            embedding_vector(vec![0.0; EMBEDDING_WIDTH], "masked embedding output")
                .unwrap()
                .len(),
            EMBEDDING_WIDTH
        );
    }

    #[test]
    fn embedding_batch_rejects_short_output() {
        let error = embedding_batch(&[0.0; 10], 2, "batched embedding output").unwrap_err();
        assert!(
            error
                .to_string()
                .contains("expected at least 512 values for 2 embeddings")
        );
        let batch = embedding_batch(&vec![1.0; 512], 2, "batched embedding output").unwrap();
        assert_eq!(batch.dim(), (2, 256));
    }

    #[test]
    fn fbank_rejects_wrong_rank_and_feature_count() {
        let rank = fbank_hw_from_shape(&[998, 80], "chunk fbank output").unwrap_err();
        assert!(rank.to_string().contains("expected rank 3"));
        let features = fbank_hw_from_shape(&[1, 998, 40], "chunk fbank output").unwrap_err();
        assert!(
            features
                .to_string()
                .contains(&format!("expected {FBANK_FEATURES} filterbank features"))
        );
        let negative = fbank_hw_from_i64(&[1, -1, 80], "chunk fbank output").unwrap_err();
        assert!(negative.to_string().contains("non-negative"));
        assert_eq!(
            fbank_hw_from_shape(&[1, 998, 80], "chunk fbank output").unwrap(),
            (998, 80)
        );
    }
}
