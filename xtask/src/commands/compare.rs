use std::path::Path;

use color_eyre::eyre::Result;

use crate::compare_rttm::compare_rttm_files;

pub fn rttm(a: &Path, b: &Path) -> Result<()> {
    compare_rttm_files(a, b)
}
