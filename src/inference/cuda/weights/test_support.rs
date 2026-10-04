//! In-memory weights and tensor inspection for CUDA tests

use super::{Dtype, Path, PathBuf, SafetensorsFile};

impl SafetensorsFile {
    /// A copy with every FP32 value scaled by `1 + relative * u`, `u` uniform in
    /// [-1, 1) from `seed`; it exists only in memory and is never written to a path
    pub fn perturbed(&self, seed: u64, relative: f32) -> Self {
        let mut bytes = self.bytes.clone();
        let mut state = seed;
        for info in self.metadata.tensors().into_values() {
            if info.dtype != Dtype::F32 {
                continue;
            }

            let (start, end) = info.data_offsets;
            let data = &mut bytes[self.data_start + start..self.data_start + end];
            for chunk in data.as_chunks_mut::<4>().0 {
                let value = f32::from_le_bytes(*chunk);
                let scale = 1.0 + relative * (2.0 * uniform(&mut state) - 1.0);
                *chunk = (value * scale).to_le_bytes();
            }
        }

        Self {
            path: PathBuf::from("<in-memory perturbed weights>"),
            bytes,
            data_start: self.data_start,
            metadata: self.metadata.clone(),
        }
    }

    /// The file this was read from
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The stored shape of a tensor, if the file has it
    pub fn shape(&self, name: &str) -> Option<&[usize]> {
        self.metadata.info(name).map(|info| info.shape.as_slice())
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
