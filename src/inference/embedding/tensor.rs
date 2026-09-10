use ndarray::{Array1, Array2, Array3, s};

use super::{EMBEDDING_WIDTH, FBANK_FEATURES};
#[cfg(any(test, feature = "coreml"))]
use crate::inference::geometry::CoreMlTensor;
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

fn embedding_vector(
    layout: &TensorLayout,
    data: &[f32],
    context: &'static str,
) -> Result<Array1<f32>, ort::Error> {
    layout
        .try_rank(2, context)
        .map_err(GeometryError::into_ort)?;
    layout
        .try_exact_dims(&[1, EMBEDDING_WIDTH], context)
        .map_err(GeometryError::into_ort)?;
    crate::inference::geometry::require_exact_len(data.len(), layout.element_count(), context)
        .map_err(GeometryError::into_ort)?;

    Ok(Array1::from_vec(data.to_vec()))
}

pub(super) fn embedding_vector_from_ort(
    shape: &ort::value::Shape,
    data: &[f32],
    context: &'static str,
) -> Result<Array1<f32>, ort::Error> {
    let layout = TensorLayout::from_ort_shape(shape, context).map_err(GeometryError::into_ort)?;
    embedding_vector(&layout, data, context)
}

#[cfg(any(test, feature = "coreml"))]
pub(super) fn embedding_vector_from_coreml(
    tensor: CoreMlTensor,
    context: &'static str,
) -> Result<Array1<f32>, ort::Error> {
    let (layout, data) = tensor.into_parts();
    embedding_vector(&layout, &data, context)
}

/// Model output rows and useful rows selected by the caller
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct EmbeddingBatchGeometry {
    model_rows: usize,
    useful_rows: usize,
}

impl EmbeddingBatchGeometry {
    fn new(
        model_rows: usize,
        useful_rows: usize,
        context: &'static str,
    ) -> Result<Self, ort::Error> {
        if useful_rows > model_rows {
            return Err(ort::Error::new(format!(
                "{context}: useful rows {useful_rows} exceed model capacity {model_rows}"
            )));
        }
        model_rows.checked_mul(EMBEDDING_WIDTH).ok_or_else(|| {
            ort::Error::new(format!("{context}: embedding batch size overflowed"))
        })?;
        useful_rows.checked_mul(EMBEDDING_WIDTH).ok_or_else(|| {
            ort::Error::new(format!("{context}: useful embedding size overflowed"))
        })?;
        Ok(Self {
            model_rows,
            useful_rows,
        })
    }
}

pub(super) fn embedding_batch(
    layout: &TensorLayout,
    data: &[f32],
    model_rows: usize,
    useful_rows: usize,
    context: &'static str,
) -> Result<Array2<f32>, ort::Error> {
    let geometry = EmbeddingBatchGeometry::new(model_rows, useful_rows, context)?;
    layout
        .try_rank(2, context)
        .map_err(GeometryError::into_ort)?;
    layout
        .try_exact_dims(&[geometry.model_rows, EMBEDDING_WIDTH], context)
        .map_err(GeometryError::into_ort)?;
    crate::inference::geometry::require_exact_len(data.len(), layout.element_count(), context)
        .map_err(GeometryError::into_ort)?;

    let batch =
        array2_from_shape_vec(geometry.model_rows, EMBEDDING_WIDTH, data.to_vec(), context)?;
    Ok(batch.slice(s![0..geometry.useful_rows, ..]).to_owned())
}

pub(super) fn embedding_batch_from_ort(
    shape: &ort::value::Shape,
    data: &[f32],
    model_rows: usize,
    useful_rows: usize,
    context: &'static str,
) -> Result<Array2<f32>, ort::Error> {
    let layout = TensorLayout::from_ort_shape(shape, context).map_err(GeometryError::into_ort)?;
    embedding_batch(&layout, data, model_rows, useful_rows, context)
}

#[cfg(feature = "coreml")]
pub(super) fn embedding_batch_from_coreml(
    tensor: CoreMlTensor,
    model_rows: usize,
    useful_rows: usize,
    context: &'static str,
) -> Result<Array2<f32>, ort::Error> {
    let (layout, data) = tensor.into_parts();
    embedding_batch(&layout, &data, model_rows, useful_rows, context)
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
        EMBEDDING_WIDTH, FBANK_FEATURES, embedding_batch_from_ort, embedding_vector_from_coreml,
        embedding_vector_from_ort, fbank_hw_from_i64, fbank_hw_from_shape, first_output,
    };
    use crate::inference::geometry::CoreMlTensor;

    #[test]
    fn first_output_reports_missing_tensor() {
        let error = first_output(Vec::<()>::new(), "embedding test").unwrap_err();

        assert_eq!(error.to_string(), "embedding test: missing output tensor");
    }

    #[test]
    fn embedding_vector_from_ort_rejects_wrong_rank_and_width_with_matching_element_count() {
        let rank = ort::value::Shape::from([EMBEDDING_WIDTH as i64]);
        let rank_error = embedding_vector_from_ort(
            &rank,
            &vec![0.0; EMBEDDING_WIDTH],
            "single embedding output",
        )
        .unwrap_err();
        assert!(rank_error.to_string().contains("expected rank 2"));

        let width = ort::value::Shape::from([2_i64, (EMBEDDING_WIDTH / 2) as i64]);
        let width_error = embedding_vector_from_ort(
            &width,
            &vec![0.0; EMBEDDING_WIDTH],
            "single embedding output",
        )
        .unwrap_err();
        assert!(width_error.to_string().contains("expected shape [1, 256]"));
        assert!(width_error.to_string().contains("got [2, 128]"));
    }

    #[test]
    fn embedding_vector_from_ort_rejects_short_and_excess_output() {
        let shape = ort::value::Shape::from([1_i64, EMBEDDING_WIDTH as i64]);
        let short = embedding_vector_from_ort(
            &shape,
            &vec![0.0; EMBEDDING_WIDTH - 1],
            "single embedding output",
        )
        .unwrap_err();
        assert!(short.to_string().contains("expected 256 values, got 255"));

        let excess = embedding_vector_from_ort(
            &shape,
            &vec![0.0; EMBEDDING_WIDTH + 1],
            "single embedding output",
        )
        .unwrap_err();
        assert!(excess.to_string().contains("expected 256 values, got 257"));
    }

    #[test]
    fn embedding_vector_from_ort_accepts_the_model_output_shape() {
        let shape = ort::value::Shape::from([1_i64, EMBEDDING_WIDTH as i64]);
        let data: Vec<f32> = (0..EMBEDDING_WIDTH).map(|value| value as f32).collect();

        let vector = embedding_vector_from_ort(&shape, &data, "single embedding output").unwrap();

        assert_eq!(vector.len(), EMBEDDING_WIDTH);
        assert_eq!(vector[0], 0.0);
        assert_eq!(vector[EMBEDDING_WIDTH - 1], (EMBEDDING_WIDTH - 1) as f32);
    }

    #[test]
    fn embedding_vector_from_coreml_validates_retained_output_shape() {
        let valid = CoreMlTensor::try_from_decoded(
            vec![0.0; EMBEDDING_WIDTH],
            vec![1, EMBEDDING_WIDTH],
            "native tail output",
        )
        .unwrap();
        assert_eq!(
            embedding_vector_from_coreml(valid, "native tail output")
                .unwrap()
                .len(),
            EMBEDDING_WIDTH
        );

        let rank = CoreMlTensor::try_from_decoded(
            vec![0.0; EMBEDDING_WIDTH],
            vec![EMBEDDING_WIDTH],
            "native tail output",
        )
        .unwrap();
        assert!(
            embedding_vector_from_coreml(rank, "native tail output")
                .unwrap_err()
                .to_string()
                .contains("expected rank 2")
        );

        let width = CoreMlTensor::try_from_decoded(
            vec![0.0; EMBEDDING_WIDTH],
            vec![2, EMBEDDING_WIDTH / 2],
            "native tail output",
        )
        .unwrap();
        let width_error = embedding_vector_from_coreml(width, "native tail output").unwrap_err();
        assert!(width_error.to_string().contains("expected shape [1, 256]"));
        assert!(width_error.to_string().contains("got [2, 128]"));

        let short = CoreMlTensor::try_from_decoded(
            vec![0.0; EMBEDDING_WIDTH - 1],
            vec![1, EMBEDDING_WIDTH - 1],
            "native tail output",
        )
        .unwrap();
        let short_error = embedding_vector_from_coreml(short, "native tail output").unwrap_err();
        assert!(short_error.to_string().contains("expected shape [1, 256]"));
    }

    #[test]
    fn embedding_batch_rejects_short_and_excess_output() {
        let shape = ort::value::Shape::from([2_i64, EMBEDDING_WIDTH as i64]);
        let short =
            embedding_batch_from_ort(&shape, &vec![0.0; 511], 2, 2, "batched embedding output")
                .unwrap_err();
        assert!(short.to_string().contains("expected 512 values, got 511"));

        let excess =
            embedding_batch_from_ort(&shape, &vec![0.0; 513], 2, 2, "batched embedding output")
                .unwrap_err();
        assert!(excess.to_string().contains("expected 512 values, got 513"));
    }

    #[test]
    fn embedding_batch_rejects_wrong_rank_and_width_with_matching_element_count() {
        let rank = ort::value::Shape::from([1_i64, 2, EMBEDDING_WIDTH as i64]);
        let rank_error =
            embedding_batch_from_ort(&rank, &vec![0.0; 512], 2, 2, "batched embedding output")
                .unwrap_err();
        assert!(rank_error.to_string().contains("expected rank 2"));

        let width = ort::value::Shape::from([4_i64, 128]);
        let width_error =
            embedding_batch_from_ort(&width, &vec![0.0; 512], 2, 2, "batched embedding output")
                .unwrap_err();
        assert!(width_error.to_string().contains("expected shape [2, 256]"));
        assert!(width_error.to_string().contains("got [4, 128]"));
    }

    #[test]
    fn embedding_batch_selects_useful_rows_from_a_padded_model_output() {
        let shape = ort::value::Shape::from([4_i64, EMBEDDING_WIDTH as i64]);
        let data: Vec<f32> = (0..4 * EMBEDDING_WIDTH).map(|value| value as f32).collect();

        let batch =
            embedding_batch_from_ort(&shape, &data, 4, 2, "padded embedding output").unwrap();

        assert_eq!(batch.dim(), (2, EMBEDDING_WIDTH));
        assert_eq!(batch[[0, 0]], 0.0);
        assert_eq!(
            batch[[1, EMBEDDING_WIDTH - 1]],
            (2 * EMBEDDING_WIDTH - 1) as f32
        );
    }

    #[test]
    fn embedding_batch_rejects_useful_rows_above_capacity_and_overflow() {
        let shape = ort::value::Shape::from([2_i64, EMBEDDING_WIDTH as i64]);
        let too_many =
            embedding_batch_from_ort(&shape, &vec![0.0; 512], 2, 3, "padded embedding output")
                .unwrap_err();
        assert!(
            too_many
                .to_string()
                .contains("useful rows 3 exceed model capacity 2")
        );

        let overflow =
            embedding_batch_from_ort(&shape, &[], usize::MAX, 0, "padded embedding output")
                .unwrap_err();
        assert!(
            overflow
                .to_string()
                .contains("embedding batch size overflowed")
        );
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
