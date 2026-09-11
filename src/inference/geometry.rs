//! Checked tensor layout used by inference adapters before native or ORT work.

use std::fmt;

/// Shape, length, rank, and CoreML output-type failures
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum GeometryError {
    /// Multiplying dimensions overflowed `usize`
    Overflow {
        /// Which bind or decode step failed
        context: &'static str,
    },
    /// A tensor dimension was negative or could not fit in `usize`
    InvalidDimension {
        /// Which bind or decode step failed
        context: &'static str,
        /// Observed dimension
        dimension: i64,
    },
    /// Flat buffer length did not match the shape product
    LengthMismatch {
        /// Which bind or decode step failed
        context: &'static str,
        /// Product of the declared dimensions
        expected: usize,
        /// Observed buffer length
        actual: usize,
    },
    /// Rank did not match the model contract
    RankMismatch {
        /// Which decode step failed
        context: &'static str,
        /// Required rank
        expected: usize,
        /// Observed rank
        actual: usize,
        /// Observed dimensions
        shape: Vec<usize>,
    },
    /// Tensor dimensions did not match the model contract
    ShapeMismatch {
        /// Which decode step failed
        context: &'static str,
        /// Required dimensions
        expected: Vec<usize>,
        /// Observed dimensions
        actual: Vec<usize>,
    },
    /// CoreML output was not Float16 or Float32
    #[cfg(any(test, feature = "coreml"))]
    UnsupportedDType {
        /// Which decode step failed
        context: &'static str,
        /// Reported CoreML data type
        dtype: &'static str,
    },
}

impl fmt::Display for GeometryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Overflow { context } => {
                write!(formatter, "{context}: tensor shape product overflowed")
            }
            Self::InvalidDimension { context, dimension } if *dimension < 0 => write!(
                formatter,
                "{context}: expected non-negative tensor dimensions, got {dimension}"
            ),
            Self::InvalidDimension { context, dimension } => write!(
                formatter,
                "{context}: tensor dimension {dimension} does not fit in usize"
            ),
            Self::LengthMismatch {
                context,
                expected,
                actual,
            } => write!(
                formatter,
                "{context}: expected {expected} values, got {actual}"
            ),
            Self::RankMismatch {
                context,
                expected,
                actual,
                shape,
            } => write!(
                formatter,
                "{context}: expected rank {expected}, got rank {actual} shape {shape:?}"
            ),
            Self::ShapeMismatch {
                context,
                expected,
                actual,
            } => write!(
                formatter,
                "{context}: expected shape {expected:?}, got {actual:?}"
            ),
            #[cfg(any(test, feature = "coreml"))]
            Self::UnsupportedDType { context, dtype } => write!(
                formatter,
                "{context}: CoreML output type {dtype} is not Float16 or Float32"
            ),
        }
    }
}

impl std::error::Error for GeometryError {}

impl GeometryError {
    pub(crate) fn into_ort(self) -> ort::Error {
        ort::Error::new(self.to_string())
    }
}

/// CoreML output types that adapters may decode
#[cfg(any(test, feature = "coreml"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CoreMlOutputDType {
    /// IEEE Float16 payload
    Float16,
    /// IEEE Float32 payload
    Float32,
}

/// Tags covering CoreML types the decoder must accept or reject
#[cfg(any(test, feature = "coreml"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CoreMlDTypeTag {
    /// IEEE Float16
    Float16,
    /// IEEE Float32
    Float32,
    /// IEEE Float64
    Float64,
    /// Signed 32-bit integer
    Int32,
    /// Signed 8-bit integer
    Int8,
    /// Any other CoreML type
    #[cfg(feature = "coreml")]
    Other,
}

#[cfg(any(test, feature = "coreml"))]
impl CoreMlOutputDType {
    /// Accept Float16 or Float32 and reject every other CoreML type
    pub(crate) fn try_from_tag(
        tag: CoreMlDTypeTag,
        context: &'static str,
    ) -> Result<Self, GeometryError> {
        match tag {
            CoreMlDTypeTag::Float16 => Ok(Self::Float16),
            CoreMlDTypeTag::Float32 => Ok(Self::Float32),
            CoreMlDTypeTag::Float64 => Err(GeometryError::UnsupportedDType {
                context,
                dtype: "Float64",
            }),
            CoreMlDTypeTag::Int32 => Err(GeometryError::UnsupportedDType {
                context,
                dtype: "Int32",
            }),
            CoreMlDTypeTag::Int8 => Err(GeometryError::UnsupportedDType {
                context,
                dtype: "Int8",
            }),
            #[cfg(feature = "coreml")]
            CoreMlDTypeTag::Other => Err(GeometryError::UnsupportedDType {
                context,
                dtype: "unsupported",
            }),
        }
    }
}

/// Declared dimensions plus their checked element count
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TensorLayout {
    dims: Vec<usize>,
    element_count: usize,
}

impl TensorLayout {
    /// Build a layout with a checked shape product
    pub(crate) fn from_dims(dims: &[usize], context: &'static str) -> Result<Self, GeometryError> {
        Ok(Self {
            dims: dims.to_vec(),
            element_count: checked_element_count(dims, context)?,
        })
    }

    /// Build a layout from an ORT shape after checking signed dimensions
    pub(crate) fn from_ort_shape(
        shape: &[i64],
        context: &'static str,
    ) -> Result<Self, GeometryError> {
        let dims = shape
            .iter()
            .copied()
            .map(|dimension| {
                usize::try_from(dimension)
                    .map_err(|_| GeometryError::InvalidDimension { context, dimension })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Self::from_dims(&dims, context)
    }

    pub(crate) fn element_count(&self) -> usize {
        self.element_count
    }

    #[cfg(feature = "coreml")]
    pub(crate) fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Require `data` to have exactly this layout's element count
    #[cfg(any(test, feature = "coreml"))]
    pub(crate) fn bind<'a>(&self, data: &'a [f32]) -> Result<&'a [f32], GeometryError> {
        require_exact_len(data.len(), self.element_count, "coreml input")?;
        Ok(data)
    }

    pub(crate) fn try_rank(
        &self,
        expected: usize,
        context: &'static str,
    ) -> Result<&[usize], GeometryError> {
        if self.dims.len() == expected {
            Ok(&self.dims)
        } else {
            Err(GeometryError::RankMismatch {
                context,
                expected,
                actual: self.dims.len(),
                shape: self.dims.clone(),
            })
        }
    }

    pub(crate) fn try_rank3(
        &self,
        context: &'static str,
    ) -> Result<(usize, usize, usize), GeometryError> {
        let dims = self.try_rank(3, context)?;
        Ok((dims[0], dims[1], dims[2]))
    }

    /// Require dimensions to match a model's exact output contract
    pub(crate) fn try_exact_dims(
        &self,
        expected: &[usize],
        context: &'static str,
    ) -> Result<(), GeometryError> {
        if self.dims == expected {
            Ok(())
        } else {
            Err(GeometryError::ShapeMismatch {
                context,
                expected: expected.to_vec(),
                actual: self.dims.clone(),
            })
        }
    }
}

/// Decoded CoreML tensor with checked layout
#[cfg(any(test, feature = "coreml"))]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CoreMlTensor {
    data: Vec<f32>,
    layout: TensorLayout,
}

#[cfg(any(test, feature = "coreml"))]
impl CoreMlTensor {
    /// Bind decoded values to a checked layout
    pub(crate) fn try_from_decoded(
        data: Vec<f32>,
        shape: Vec<usize>,
        context: &'static str,
    ) -> Result<Self, GeometryError> {
        let layout = TensorLayout::from_dims(&shape, context)?;
        require_exact_len(data.len(), layout.element_count(), context)?;
        Ok(Self { data, layout })
    }

    #[cfg(feature = "coreml")]
    pub(crate) fn layout(&self) -> &TensorLayout {
        &self.layout
    }

    pub(crate) fn try_rank3(
        &self,
        context: &'static str,
    ) -> Result<(usize, usize, usize), GeometryError> {
        self.layout.try_rank3(context)
    }

    #[cfg(feature = "coreml")]
    pub(crate) fn into_data(self) -> Vec<f32> {
        self.data
    }

    #[cfg(any(test, feature = "coreml"))]
    pub(crate) fn into_parts(self) -> (TensorLayout, Vec<f32>) {
        (self.layout, self.data)
    }

    #[cfg(feature = "coreml")]
    pub(crate) fn rank3_hw(
        self,
        context: &'static str,
    ) -> Result<(Vec<f32>, usize, usize), GeometryError> {
        let (_, frames, classes) = self.try_rank3(context)?;
        Ok((self.data, frames, classes))
    }
}

pub(crate) fn checked_element_count(
    dims: &[usize],
    context: &'static str,
) -> Result<usize, GeometryError> {
    let mut count = 1usize;
    for &dim in dims {
        count = count
            .checked_mul(dim)
            .ok_or(GeometryError::Overflow { context })?;
    }
    Ok(count)
}

pub(crate) fn require_exact_len(
    actual: usize,
    expected: usize,
    context: &'static str,
) -> Result<(), GeometryError> {
    if actual == expected {
        Ok(())
    } else {
        Err(GeometryError::LengthMismatch {
            context,
            expected,
            actual,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CoreMlDTypeTag, CoreMlOutputDType, CoreMlTensor, GeometryError, TensorLayout,
        checked_element_count,
    };

    #[test]
    fn bind_accepts_exact_input_length() {
        let layout = TensorLayout::from_dims(&[1, 1, 4], "coreml input").unwrap();
        let data = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(layout.bind(&data).unwrap(), &data);
    }

    #[test]
    fn bind_rejects_short_input() {
        let layout = TensorLayout::from_dims(&[1, 1, 4], "coreml input").unwrap();
        let error = layout.bind(&[1.0, 2.0, 3.0]).unwrap_err();
        assert_eq!(
            error,
            GeometryError::LengthMismatch {
                context: "coreml input",
                expected: 4,
                actual: 3,
            }
        );
    }

    #[test]
    fn bind_rejects_long_input() {
        let layout = TensorLayout::from_dims(&[2, 2], "coreml input").unwrap();
        let error = layout.bind(&[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap_err();
        assert_eq!(
            error,
            GeometryError::LengthMismatch {
                context: "coreml input",
                expected: 4,
                actual: 5,
            }
        );
    }

    #[test]
    fn shape_product_reports_overflow() {
        let error = checked_element_count(&[usize::MAX, 2], "coreml input").unwrap_err();
        assert_eq!(
            error,
            GeometryError::Overflow {
                context: "coreml input"
            }
        );
        assert!(TensorLayout::from_dims(&[usize::MAX, 3], "coreml input").is_err());
    }

    #[test]
    fn output_dtype_rejects_float64_int32_and_int8() {
        for (tag, dtype) in [
            (CoreMlDTypeTag::Float64, "Float64"),
            (CoreMlDTypeTag::Int32, "Int32"),
            (CoreMlDTypeTag::Int8, "Int8"),
        ] {
            let error = CoreMlOutputDType::try_from_tag(tag, "coreml output").unwrap_err();
            assert_eq!(
                error,
                GeometryError::UnsupportedDType {
                    context: "coreml output",
                    dtype,
                }
            );
        }
        assert_eq!(
            CoreMlOutputDType::try_from_tag(CoreMlDTypeTag::Float16, "coreml output").unwrap(),
            CoreMlOutputDType::Float16
        );
        assert_eq!(
            CoreMlOutputDType::try_from_tag(CoreMlDTypeTag::Float32, "coreml output").unwrap(),
            CoreMlOutputDType::Float32
        );
    }

    #[test]
    fn tensor_rejects_wrong_rank_and_element_count() {
        let rank0 = CoreMlTensor::try_from_decoded(vec![1.0], vec![], "coreml output").unwrap();
        assert!(matches!(
            rank0.try_rank3("coreml output"),
            Err(GeometryError::RankMismatch {
                expected: 3,
                actual: 0,
                ..
            })
        ));

        let rank2 =
            CoreMlTensor::try_from_decoded(vec![1.0, 2.0], vec![1, 2], "coreml output").unwrap();
        assert!(matches!(
            rank2.try_rank3("coreml output"),
            Err(GeometryError::RankMismatch {
                expected: 3,
                actual: 2,
                ..
            })
        ));

        let rank3 = CoreMlTensor::try_from_decoded(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![1, 2, 2],
            "coreml output",
        )
        .unwrap();
        assert_eq!(rank3.try_rank3("coreml output").unwrap(), (1, 2, 2));

        let error =
            CoreMlTensor::try_from_decoded(vec![1.0], vec![1, 2, 2], "coreml output").unwrap_err();
        assert_eq!(
            error,
            GeometryError::LengthMismatch {
                context: "coreml output",
                expected: 4,
                actual: 1,
            }
        );
    }
}
