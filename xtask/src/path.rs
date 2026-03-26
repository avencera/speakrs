use std::path::Path;

use color_eyre::eyre::{Result, eyre};

pub fn file_stem_string(path: &Path) -> Result<String> {
    let Some(stem) = path.file_stem() else {
        return Err(eyre!("path has no file stem: {}", path.display()));
    };

    Ok(stem.to_string_lossy().into_owned())
}
