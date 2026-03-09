use color_eyre::eyre::Result;

use crate::python::uv_run;

pub fn generate() -> Result<()> {
    uv_run(&["fixtures/generate.py"])
}
