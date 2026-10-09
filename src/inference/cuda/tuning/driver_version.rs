//! Driver identity from the driver, with file and CUDA API fallbacks

use std::ffi::{CStr, c_char, c_uint};

use libloading::Library;
use serde::{Deserialize, Serialize};

use super::CudaTuneError;
use crate::inference::cuda::CudaError;

/// Source tags prevent a CUDA API level from aliasing a driver release
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(
    tag = "source",
    content = "version",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub(super) enum DriverVersion {
    Nvml(String),
    Sysfs(String),
    Procfs(String),
    CudaApi(i32),
}

impl DriverVersion {
    pub(super) fn read() -> Result<Self, CudaTuneError> {
        Self::select(
            nvml_version,
            |path| std::fs::read_to_string(path),
            cuda_version,
        )
    }

    fn select(
        nvml: impl FnOnce() -> Option<String>,
        mut file: impl FnMut(&str) -> std::io::Result<String>,
        cuda: impl FnOnce() -> Result<i32, CudaTuneError>,
    ) -> Result<Self, CudaTuneError> {
        if let Some(version) = nvml().and_then(|text| first_line(&text)) {
            return Ok(Self::Nvml(version));
        }
        if let Some(version) = file("/sys/module/nvidia/version")
            .ok()
            .and_then(|text| first_line(&text))
        {
            return Ok(Self::Sysfs(version));
        }
        if let Some(version) = file("/proc/driver/nvidia/version")
            .ok()
            .and_then(|text| first_line(&text))
        {
            return Ok(Self::Procfs(version));
        }

        let version = cuda()?;
        if version <= 0 {
            return Err(CudaTuneError::Invalid(
                "CUDA driver API version is not positive".into(),
            ));
        }

        Ok(Self::CudaApi(version))
    }
}

fn first_line(text: &str) -> Option<String> {
    text.lines()
        .next()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(str::to_owned)
}

fn nvml_version() -> Option<String> {
    // safety: NVML library initializers have no caller preconditions
    let library = unsafe { Library::new("libnvidia-ml.so.1") }.ok()?;
    nvml_version_from_library(&library)
}

fn nvml_version_from_library(library: &Library) -> Option<String> {
    type Init = unsafe extern "C" fn() -> c_uint;
    type GetVersion = unsafe extern "C" fn(*mut c_char, c_uint) -> c_uint;
    type Shutdown = unsafe extern "C" fn() -> c_uint;

    // safety: signatures match NVML's C API; symbols cannot outlive the library
    let (init, get_version, shutdown) = unsafe {
        (
            library.get::<Init>(b"nvmlInit_v2\0").ok()?,
            library
                .get::<GetVersion>(b"nvmlSystemGetDriverVersion\0")
                .ok()?,
            library.get::<Shutdown>(b"nvmlShutdown\0").ok()?,
        )
    };
    // safety: all symbols were checked before initialization, and init takes no arguments
    if unsafe { init() } != 0 {
        return None;
    }

    // nvml specifies 80 bytes including the terminator; decode only inside this buffer
    let mut bytes = [0u8; 80];
    // safety: NVML is initialized and the writable buffer has the given length
    let status = unsafe { get_version(bytes.as_mut_ptr().cast(), bytes.len() as c_uint) };
    // safety: balance exactly this initialization, including a failed version query
    let _ = unsafe { shutdown() };
    if status != 0 {
        return None;
    }

    CStr::from_bytes_until_nul(&bytes)
        .ok()?
        .to_str()
        .ok()
        .map(str::to_owned)
}

fn cuda_version() -> Result<i32, CudaTuneError> {
    let mut version = 0;
    // safety: device-key callers already have a live CUDA context, and the output is valid
    unsafe { cudarc::driver::sys::cuDriverGetVersion(&mut version) }
        .result()
        .map_err(CudaError::from)?;
    Ok(version)
}

#[cfg(test)]
mod tests;
