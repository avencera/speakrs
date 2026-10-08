use ndarray::{Array1, Array2, Array3};
#[cfg(feature = "migraphx")]
use ort::memory::Allocator;
#[cfg(feature = "migraphx")]
use ort::session::{HasSelectedOutputs, OutputSelector, RunOptions};
#[cfg(feature = "migraphx")]
use ort::value::Tensor;

use super::{EMBEDDING_WIDTH, FBANK_FEATURES};
#[cfg(any(test, feature = "coreml"))]
use crate::inference::geometry::CoreMlTensor;
use crate::inference::geometry::TensorLayout;
use crate::inference::{InferenceError, TensorShapeError};

pub(super) fn array2_from_shape_vec(
    rows: usize,
    cols: usize,
    data: Vec<f32>,
    context: &'static str,
) -> Result<Array2<f32>, InferenceError> {
    Array2::from_shape_vec((rows, cols), data)
        .map_err(|source| InferenceError::OutputArray { context, source })
}

#[cfg(feature = "coreml")]
pub(super) fn array2_slice<'a>(
    array: &'a Array2<f32>,
    context: &'static str,
) -> Result<&'a [f32], InferenceError> {
    array
        .as_slice()
        .ok_or(InferenceError::NonContiguousBuffer { context })
}

#[cfg(feature = "coreml")]
pub(super) fn array3_slice<'a>(
    array: &'a Array3<f32>,
    context: &'static str,
) -> Result<&'a [f32], InferenceError> {
    array
        .as_slice()
        .ok_or(InferenceError::NonContiguousBuffer { context })
}

pub(super) fn array3_slice_mut<'a>(
    array: &'a mut Array3<f32>,
    context: &'static str,
) -> Result<&'a mut [f32], InferenceError> {
    array
        .as_slice_mut()
        .ok_or(InferenceError::NonContiguousBuffer { context })
}

fn embedding_vector(
    layout: &TensorLayout,
    data: &[f32],
    context: &'static str,
) -> Result<Array1<f32>, InferenceError> {
    layout.try_rank(2, context)?;
    layout.try_exact_dims(&[1, EMBEDDING_WIDTH], context)?;
    crate::inference::geometry::require_exact_len(data.len(), layout.element_count(), context)?;

    Ok(Array1::from_vec(data.to_vec()))
}

#[cfg(feature = "migraphx")]
pub(super) fn embedding_vector_from_ort(
    shape: &ort::value::Shape,
    data: &[f32],
    context: &'static str,
) -> Result<Array1<f32>, InferenceError> {
    let layout = TensorLayout::from_ort_shape(shape, context)?;
    embedding_vector(&layout, data, context)
}

#[cfg(any(test, feature = "coreml"))]
pub(super) fn embedding_vector_from_coreml(
    tensor: CoreMlTensor,
    context: &'static str,
) -> Result<Array1<f32>, InferenceError> {
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
    ) -> Result<Self, InferenceError> {
        if useful_rows > model_rows {
            return Err(InferenceError::BatchTooLarge {
                context,
                rows: useful_rows,
                capacity: model_rows,
            });
        }
        model_rows
            .checked_mul(EMBEDDING_WIDTH)
            .ok_or(TensorShapeError::Overflow { context })?;
        useful_rows
            .checked_mul(EMBEDDING_WIDTH)
            .ok_or(TensorShapeError::Overflow { context })?;
        Ok(Self {
            model_rows,
            useful_rows,
        })
    }
}

fn embedding_batch(
    layout: &TensorLayout,
    data: &[f32],
    model_rows: usize,
    useful_rows: usize,
    context: &'static str,
) -> Result<Array2<f32>, InferenceError> {
    let geometry = EmbeddingBatchGeometry::new(model_rows, useful_rows, context)?;
    layout.try_rank(2, context)?;
    layout.try_exact_dims(&[geometry.model_rows, EMBEDDING_WIDTH], context)?;
    crate::inference::geometry::require_exact_len(data.len(), layout.element_count(), context)?;

    let useful_len = geometry.useful_rows * EMBEDDING_WIDTH;
    array2_from_shape_vec(
        geometry.useful_rows,
        EMBEDDING_WIDTH,
        data[..useful_len].to_vec(),
        context,
    )
}

#[cfg(feature = "migraphx")]
pub(super) fn embedding_batch_from_ort(
    shape: &ort::value::Shape,
    data: &[f32],
    model_rows: usize,
    useful_rows: usize,
    context: &'static str,
) -> Result<Array2<f32>, InferenceError> {
    let layout = TensorLayout::from_ort_shape(shape, context)?;
    embedding_batch(&layout, data, model_rows, useful_rows, context)
}

#[cfg(feature = "coreml")]
pub(super) fn embedding_batch_from_coreml(
    tensor: CoreMlTensor,
    model_rows: usize,
    useful_rows: usize,
    context: &'static str,
) -> Result<Array2<f32>, InferenceError> {
    let (layout, data) = tensor.into_parts();
    embedding_batch(&layout, &data, model_rows, useful_rows, context)
}

fn fbank_hw_from_layout(
    layout: &TensorLayout,
    context: &'static str,
) -> Result<(usize, usize), InferenceError> {
    let (_, frames, features) = layout.try_rank3(context)?;
    if features != FBANK_FEATURES {
        return Err(TensorShapeError::AxisMismatch {
            context,
            axis: 2,
            expected: FBANK_FEATURES,
            actual: features,
        }
        .into());
    }
    Ok((frames, features))
}

#[cfg(any(test, feature = "coreml"))]
pub(super) fn fbank_hw_from_shape(
    shape: &[usize],
    context: &'static str,
) -> Result<(usize, usize), InferenceError> {
    fbank_hw_from_layout(&TensorLayout::from_dims(shape, context)?, context)
}

#[cfg(feature = "migraphx")]
pub(super) fn fbank_hw_from_i64(
    shape: &[i64],
    context: &'static str,
) -> Result<(usize, usize), InferenceError> {
    fbank_hw_from_layout(&TensorLayout::from_ort_shape(shape, context)?, context)
}

/// Split a flat `[count, frames, features]` filterbank batch into one array per window
pub(super) fn push_fbank_batch_results(
    results: &mut Vec<Array2<f32>>,
    data: &[f32],
    frames: usize,
    features: usize,
    count: usize,
) -> Result<(), InferenceError> {
    let stride = frames * features;
    for idx in 0..count {
        let start = idx * stride;
        let batch = array2_from_shape_vec(
            frames,
            features,
            data[start..start + stride].to_vec(),
            "batched fbank output",
        )?;
        results.push(batch);
    }
    Ok(())
}

#[cfg(feature = "migraphx")]
pub(super) fn first_output<T>(
    outputs: impl IntoIterator<Item = T>,
    context: &'static str,
) -> Result<T, InferenceError> {
    outputs
        .into_iter()
        .next()
        .ok_or(InferenceError::MissingOutput { context })
}

#[cfg(feature = "migraphx")]
pub(super) fn preallocated_run_options(
    rows: usize,
    cols: usize,
) -> Result<RunOptions<HasSelectedOutputs>, ort::Error> {
    let output = Tensor::<f32>::new(&Allocator::default(), [rows, cols])?;
    RunOptions::new().map(|options| {
        options.with_outputs(OutputSelector::default().preallocate("output", output))
    })
}

#[cfg(test)]
mod tests {
    use super::{
        EMBEDDING_WIDTH, FBANK_FEATURES, embedding_vector_from_coreml, fbank_hw_from_shape,
    };
    #[cfg(feature = "migraphx")]
    use super::{
        embedding_batch_from_ort, embedding_vector_from_ort, fbank_hw_from_i64, first_output,
    };
    use crate::inference::geometry::CoreMlTensor;
    #[cfg(feature = "migraphx")]
    use crate::inference::{InferenceError, TensorShapeError};

    #[cfg(feature = "migraphx")]
    #[test]
    fn first_output_reports_missing_tensor() {
        let error = first_output(Vec::<()>::new(), "embedding test").unwrap_err();

        assert_eq!(error.to_string(), "embedding test: missing output tensor");
    }

    #[cfg(feature = "migraphx")]
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

    #[cfg(feature = "migraphx")]
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

    #[cfg(feature = "migraphx")]
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

    #[cfg(feature = "migraphx")]
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

    #[cfg(feature = "migraphx")]
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

    #[cfg(feature = "migraphx")]
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

    #[cfg(feature = "migraphx")]
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
        assert!(matches!(
            overflow,
            InferenceError::Shape(TensorShapeError::Overflow {
                context: "padded embedding output"
            })
        ));
    }

    #[test]
    fn fbank_rejects_wrong_rank_and_feature_count() {
        let rank = fbank_hw_from_shape(&[998, 80], "chunk fbank output").unwrap_err();
        assert!(rank.to_string().contains("expected rank 3"));
        let features = fbank_hw_from_shape(&[1, 998, 40], "chunk fbank output").unwrap_err();
        assert!(features.to_string().contains(&format!(
            "expected axis 2 to have length {FBANK_FEATURES}, got 40"
        )));
        #[cfg(feature = "migraphx")]
        {
            let negative = fbank_hw_from_i64(&[1, -1, 80], "chunk fbank output").unwrap_err();
            assert!(negative.to_string().contains("non-negative"));
        }
        assert_eq!(
            fbank_hw_from_shape(&[1, 998, 80], "chunk fbank output").unwrap(),
            (998, 80)
        );
    }
}
