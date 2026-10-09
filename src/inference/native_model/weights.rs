use std::path::{Path, PathBuf};

use safetensors::tensor::Metadata;
use safetensors::{Dtype, SafeTensorError, SafeTensors};

/// Size of the little-endian header length that starts every safetensors file
const HEADER_LEN_BYTES: usize = 8;

/// A validated safetensors file with backend-neutral host tensor decoding
///
/// Other dtypes may be present, but [`Self::read_f32`] accepts only FP32 tensors
#[derive(Debug)]
pub(crate) struct WeightsFile {
    path: PathBuf,
    bytes: Vec<u8>,
    data_start: usize,
    metadata: Metadata,
}

impl WeightsFile {
    /// Reads and validates the header of a safetensors file
    pub fn open(path: impl AsRef<Path>) -> Result<Self, NativeWeightsError> {
        let path = path.as_ref().to_path_buf();
        let bytes = std::fs::read(&path).map_err(|source| NativeWeightsError::WeightsIo {
            path: path.clone(),
            source,
        })?;

        let (header_len, metadata) = SafeTensors::read_metadata(&bytes).map_err(|source| {
            NativeWeightsError::WeightsFormat {
                path: path.clone(),
                source,
            }
        })?;

        Ok(Self {
            path,
            bytes,
            data_start: HEADER_LEN_BYTES + header_len,
            metadata,
        })
    }

    /// Tensor names in the file, sorted
    pub fn names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.metadata.tensors().into_keys().collect();
        names.sort();
        names
    }

    /// Decodes tensor `name` as little-endian FP32 after checking its dtype and shape
    pub fn read_f32(
        &self,
        name: &str,
        expected_shape: &[usize],
    ) -> Result<Vec<f32>, NativeWeightsError> {
        let info = self
            .metadata
            .info(name)
            .ok_or_else(|| NativeWeightsError::MissingTensor {
                path: self.path.clone(),
                name: name.to_string(),
            })?;

        if info.dtype != Dtype::F32 {
            return Err(NativeWeightsError::TensorDtype {
                name: name.to_string(),
                dtype: info.dtype.to_string(),
            });
        }

        if info.shape != expected_shape {
            return Err(NativeWeightsError::TensorShape {
                name: name.to_string(),
                expected: expected_shape.to_vec(),
                actual: info.shape.clone(),
            });
        }

        // `read_metadata` already checked that every offset lies inside the file
        let (start, end) = info.data_offsets;
        let bytes = &self.bytes[self.data_start + start..self.data_start + end];
        // metadata also validates the exact byte count for the dtype and shape
        let (chunks, remainder) = bytes.as_chunks::<4>();
        debug_assert!(remainder.is_empty());
        Ok(chunks.iter().copied().map(f32::from_le_bytes).collect())
    }
}

/// Errors from native host weight loading, without a runtime dependency
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum NativeWeightsError {
    /// A weights file could not be read
    #[error("reading weights `{path}`: {source}")]
    WeightsIo {
        /// The weights file
        path: PathBuf,
        /// The I/O error
        #[source]
        source: std::io::Error,
    },
    /// A weights file is not valid safetensors
    #[error("parsing weights `{path}`: {source}")]
    WeightsFormat {
        /// The weights file
        path: PathBuf,
        /// The safetensors error
        #[source]
        source: SafeTensorError,
    },
    /// A weights file lacks a required tensor
    #[error("weights `{path}` have no tensor `{name}`")]
    MissingTensor {
        /// The weights file
        path: PathBuf,
        /// The missing tensor name
        name: String,
    },
    /// A weight tensor is not stored as FP32
    #[error("tensor `{name}` is {dtype}, expected F32")]
    TensorDtype {
        /// The tensor name
        name: String,
        /// The stored dtype
        dtype: String,
    },
    /// A weight tensor has a different shape than the model expects
    #[error("tensor `{name}` has shape {actual:?}, expected {expected:?}")]
    TensorShape {
        /// The tensor name
        name: String,
        /// The shape the model expects
        expected: Vec<usize>,
        /// The stored shape
        actual: Vec<usize>,
    },
}

#[cfg(test)]
pub(super) mod test_support {
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};

    use serde_json::Value;

    use super::WeightsFile;

    impl WeightsFile {
        /// The file this was read from
        pub(crate) fn path(&self) -> &Path {
            &self.path
        }

        /// The stored shape of a tensor, if the file has it
        pub(crate) fn shape(&self, name: &str) -> Option<&[usize]> {
            self.metadata.info(name).map(|info| info.shape.as_slice())
        }
    }

    /// A temporary on-disk file removed when the test finishes
    pub(crate) struct TestFile(pub(crate) PathBuf);

    impl TestFile {
        pub(crate) fn new(header: &Value, data: &[u8]) -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let index = NEXT.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "speakrs-native-weights-{}-{index}.safetensors",
                std::process::id()
            ));
            let header = serde_json::to_vec(header).unwrap();
            let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
            bytes.extend(header);
            bytes.extend(data);
            std::fs::write(&path, bytes).unwrap();
            Self(path)
        }
    }

    impl Drop for TestFile {
        fn drop(&mut self) {
            std::fs::remove_file(&self.0).unwrap();
        }
    }
}

#[cfg(test)]
mod tests {
    use safetensors::SafeTensorError;
    use serde_json::json;

    use super::test_support::TestFile;
    use super::{NativeWeightsError, WeightsFile};

    #[test]
    fn decodes_little_endian_values_and_preserves_other_dtypes() {
        let bits = [0x3f80_0000_u32, 0x8000_0000, 0x7fc0_0123, 0xc020_0000];
        let mut data: Vec<u8> = bits.into_iter().flat_map(u32::to_le_bytes).collect();
        data.extend([1, 2]);
        let file = TestFile::new(
            &json!({
                "float": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]},
                "other": {"dtype": "U8", "shape": [2], "data_offsets": [16, 18]}
            }),
            &data,
        );
        let weights = WeightsFile::open(&file.0).unwrap();
        assert_eq!(weights.names(), ["float", "other"]);
        assert_eq!(weights.path(), file.0);
        assert_eq!(weights.shape("float"), Some([2, 2].as_slice()));
        let decoded = weights.read_f32("float", &[2, 2]).unwrap();
        assert_eq!(
            decoded.into_iter().map(f32::to_bits).collect::<Vec<_>>(),
            bits
        );
        let error = weights.read_f32("other", &[2]).unwrap_err();
        assert!(
            matches!(error, NativeWeightsError::TensorDtype { name, dtype }
            if name == "other" && dtype == "U8")
        );
    }

    #[test]
    fn rejects_missing_tensors_and_wrong_shapes() {
        let file = TestFile::new(
            &json!({"value": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}),
            &1.0_f32.to_le_bytes(),
        );
        let weights = WeightsFile::open(&file.0).unwrap();
        assert!(matches!(weights.read_f32("absent", &[1]),
            Err(NativeWeightsError::MissingTensor { path, name })
                if path == file.0 && name == "absent"));
        assert!(matches!(weights.read_f32("value", &[1, 1]),
            Err(NativeWeightsError::TensorShape { name, expected, actual })
                if name == "value" && expected == [1, 1] && actual == [1]));
    }

    #[test]
    fn rejects_invalid_offsets_counts_and_truncated_data() {
        for (shape, offsets, bytes) in [
            (vec![1], [4, 8], 8),
            (vec![1], [0, 3], 3),
            (vec![2], [0, 4], 4),
            (vec![1], [0, 4], 3),
            (vec![1], [0, 4], 5),
            (vec![usize::MAX, 2], [0, 4], 4),
        ] {
            let file = TestFile::new(
                &json!({"value": {"dtype": "F32", "shape": shape, "data_offsets": offsets}}),
                &vec![0; bytes],
            );
            assert!(
                matches!(
                    WeightsFile::open(&file.0),
                    Err(NativeWeightsError::WeightsFormat { .. })
                ),
                "shape={shape:?}, offsets={offsets:?}"
            );
        }

        let file = TestFile::new(&json!({}), &[]);
        std::fs::write(&file.0, [0, 1, 2]).unwrap();
        assert!(matches!(
            WeightsFile::open(&file.0),
            Err(NativeWeightsError::WeightsFormat {
                source: SafeTensorError::HeaderTooSmall,
                ..
            })
        ));
    }
}
