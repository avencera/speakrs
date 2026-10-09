//! In-memory weights and tensor inspection for CUDA tests

use super::{Path, SafetensorsFile};

impl SafetensorsFile {
    /// The file this was read from
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The stored shape of a tensor, if the file has it
    pub fn shape(&self, name: &str) -> Option<&[usize]> {
        self.metadata.info(name).map(|info| info.shape.as_slice())
    }
}
