//! In-memory weights and tensor inspection for CUDA tests

use super::{Path, SafetensorsFile};

impl SafetensorsFile {
    /// A copy with every FP32 value scaled by `1 + relative * u`, `u` uniform in
    /// [-1, 1) from `seed`; it exists only in memory and is never written to a path
    pub fn perturbed(&self, seed: u64, relative: f32) -> Self {
        let mut state = seed;
        Self(self.0.map_f32(|value| {
            let scale = 1.0 + relative * (2.0 * uniform(&mut state) - 1.0);
            value * scale
        }))
    }

    /// The file this was read from
    pub fn path(&self) -> &Path {
        self.0.path()
    }

    /// The stored shape of a tensor, if the file has it
    pub fn shape(&self, name: &str) -> Option<&[usize]> {
        self.0.shape(name)
    }
}

/// SplitMix64 mapped to [0, 1), for in-process qualification inputs
pub(crate) fn uniform(state: &mut u64) -> f32 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^= z >> 31;
    (z >> 40) as f32 / (1u64 << 24) as f32
}

mod tests {
    use serde_json::json;

    use crate::inference::native_model::test_support::TestFile;

    use super::super::super::CudaError;
    use super::SafetensorsFile;

    #[test]
    fn adapter_keeps_cuda_weight_error_variants_and_payloads() {
        let file = TestFile::new(
            &json!({
                "float": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
                "other": {"dtype": "U8", "shape": [1], "data_offsets": [4, 5]}
            }),
            &[0, 0, 128, 63, 7],
        );
        let weights = SafetensorsFile::open(&file.0).unwrap();
        assert_eq!(weights.host().names(), ["float", "other"]);
        assert_eq!(weights.read_f32("float", &[1]).unwrap(), [1.0]);
        assert!(matches!(weights.read_f32("missing", &[1]),
            Err(CudaError::MissingTensor { path, name })
                if path == file.0 && name == "missing"));
        assert!(matches!(weights.read_f32("other", &[1]),
            Err(CudaError::TensorDtype { name, dtype })
                if name == "other" && dtype == "U8"));
        assert!(matches!(weights.read_f32("float", &[2]),
            Err(CudaError::TensorShape { name, expected, actual })
                if name == "float" && expected == [2] && actual == [1]));

        std::fs::write(&file.0, [0, 1, 2]).unwrap();
        assert!(matches!(SafetensorsFile::open(&file.0),
            Err(CudaError::WeightsFormat { path, .. }) if path == file.0));
        let absent = file.0.with_extension("absent");
        assert!(matches!(SafetensorsFile::open(&absent),
            Err(CudaError::WeightsIo { path, source })
                if path == absent && source.kind() == std::io::ErrorKind::NotFound));
    }
}
