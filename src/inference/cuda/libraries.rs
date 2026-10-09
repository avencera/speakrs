//! Per-runtime fallible library cells, requested only while a plan is built

use std::ffi::{CStr, OsStr};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError};

use libloading::Library;

use cudarc::cublas::{CudaBlas, result::CublasError};
use cudarc::cudnn::{Cudnn, CudnnError};
use cudarc::driver::CudaStream;

use super::{CudaError, CudaLibrary, CudaMath};

#[derive(Debug, Clone, Copy)]
enum Failure {
    Missing(CudaLibrary),
    Blas(CublasError),
    Dnn(CudnnError),
}

impl From<Failure> for CudaError {
    fn from(failure: Failure) -> Self {
        match failure {
            Failure::Missing(library) => Self::LibraryUnavailable { library },
            Failure::Blas(error) => Self::Cublas(error),
            Failure::Dnn(error) => Self::Cudnn(error),
        }
    }
}

#[derive(Debug)]
struct Blas {
    handle: CudaBlas,
    // keep mode selection and enqueue in one critical section
    math: Mutex<CudaMath>,
    // drop the handle before releasing the checked shared object
    _library: Library,
}

#[derive(Debug)]
struct Dnn {
    handle: Arc<Cudnn>,
    _library: Library,
}

/// Optional libraries drop before the runtime's stream and context
#[derive(Debug, Default)]
pub(super) struct Libraries {
    blas: OnceLock<Result<Blas, Failure>>,
    dnn: OnceLock<Result<Dnn, Failure>>,
    nvrtc: OnceLock<Result<Library, Failure>>,
}

impl Libraries {
    /// The policy check is deliberately before the cell lookup and every probe
    pub(super) fn prepare(
        &self,
        library: CudaLibrary,
        stream: &Arc<CudaStream>,
    ) -> Result<(), CudaError> {
        if super::driver_only() {
            return Err(CudaError::LibraryForbidden { library });
        }
        match library {
            CudaLibrary::Cublas => cached(self.blas.get_or_init(|| {
                let checked = present(library)?;
                let handle = CudaBlas::new(stream.clone()).map_err(Failure::Blas)?;
                set_math(&handle, CudaMath::Fp32).map_err(Failure::Blas)?;
                Ok(Blas {
                    handle,
                    math: Mutex::new(CudaMath::Fp32),
                    _library: checked,
                })
            }))
            .map(|_| ()),
            CudaLibrary::Cudnn => cached(self.dnn.get_or_init(|| {
                let checked = present(library)?;
                let handle = Cudnn::new(stream.clone()).map_err(Failure::Dnn)?;
                Ok(Dnn {
                    handle,
                    _library: checked,
                })
            }))
            .map(|_| ()),
            CudaLibrary::Nvrtc => cached(self.nvrtc.get_or_init(|| present(library))).map(|_| ()),
            CudaLibrary::Driver => Ok(()),
        }
    }

    /// Query the same loaded libraries and handle that execute Library plans
    pub(super) fn versions(
        &self,
        stream: &Arc<CudaStream>,
    ) -> Result<super::tuning::LibraryVersions, CudaError> {
        self.prepare(CudaLibrary::Cublas, stream)?;
        self.prepare(CudaLibrary::Cudnn, stream)?;
        let mut cublas = 0;
        // safety: the prepared handle is live and the version output is writable
        unsafe { cudarc::cublas::sys::cublasGetVersion_v2(*self.blas()?.handle(), &mut cublas) }
            .result()?;
        let cudnn = cudarc::cudnn::result::get_version();
        if cublas <= 0 || cudnn == 0 {
            return Err(super::tuning::invalid(
                "numerical-library versions must be positive",
            ));
        }
        Ok(super::tuning::LibraryVersions::Hybrid { cudnn, cublas })
    }

    /// Access an already prepared handle without loading during forward or capture
    pub(super) fn blas(&self) -> Result<&CudaBlas, CudaError> {
        let value = self
            .blas
            .get()
            .ok_or_else(|| unprepared(CudaLibrary::Cublas))?;
        cached(value).map(|blas| &blas.handle)
    }

    pub(super) fn dnn(&self) -> Result<&Arc<Cudnn>, CudaError> {
        cached(
            self.dnn
                .get()
                .ok_or_else(|| unprepared(CudaLibrary::Cudnn))?,
        )
        .map(|dnn| &dnn.handle)
    }

    pub(super) fn lock_blas(&self, math: CudaMath) -> Result<MutexGuard<'_, CudaMath>, CudaError> {
        let blas = cached(
            self.blas
                .get()
                .ok_or_else(|| unprepared(CudaLibrary::Cublas))?,
        )?;
        let mut current = blas.math.lock().unwrap_or_else(PoisonError::into_inner);
        if *current != math {
            set_math(&blas.handle, math)?;
            *current = math;
        }
        Ok(current)
    }
}

fn cached<T>(value: &Result<T, Failure>) -> Result<&T, CudaError> {
    value.as_ref().map_err(|error| (*error).into())
}

fn unprepared(library: CudaLibrary) -> CudaError {
    CudaError::Unsupported {
        context: "CUDA plan",
        reason: format!("{library} was not prepared before forward"),
    }
}

fn set_math(blas: &CudaBlas, math: CudaMath) -> Result<(), CublasError> {
    // SAFETY: the handle is live and the caller holds its mode lock
    unsafe { cudarc::cublas::sys::cublasSetMathMode(*blas.handle(), math.cublas()) }.result()
}

// these are the functions reached by speakrs and its cudarc wrappers, including
// handle/descriptors' destructors and development version queries
const CUBLAS_SYMBOLS: &[&CStr] = &[
    c"cublasCreate_v2",
    c"cublasDestroy_v2",
    c"cublasSetStream_v2",
    c"cublasSetMathMode",
    c"cublasSgemm_v2",
    c"cublasGetVersion_v2",
    c"cublasGetCudartVersion",
];

const CUDNN_SYMBOLS: &[&CStr] = &[
    c"cudnnCreate",
    c"cudnnDestroy",
    c"cudnnSetStream",
    c"cudnnGetVersion",
    c"cudnnCreateTensorDescriptor",
    c"cudnnDestroyTensorDescriptor",
    c"cudnnSetTensor4dDescriptor",
    c"cudnnSetTensorNdDescriptor",
    c"cudnnGetTensorNdDescriptor",
    c"cudnnCreateConvolutionDescriptor",
    c"cudnnDestroyConvolutionDescriptor",
    c"cudnnSetConvolution2dDescriptor",
    c"cudnnSetConvolutionMathType",
    c"cudnnCreateFilterDescriptor",
    c"cudnnDestroyFilterDescriptor",
    c"cudnnSetFilter4dDescriptor",
    c"cudnnGetConvolutionForwardAlgorithm_v7",
    c"cudnnGetConvolutionForwardWorkspaceSize",
    c"cudnnConvolutionForward",
    c"cudnnConvolutionBiasActivationForward",
    c"cudnnCreateActivationDescriptor",
    c"cudnnDestroyActivationDescriptor",
    c"cudnnSetActivationDescriptor",
    c"cudnnCreateRNNDescriptor",
    c"cudnnDestroyRNNDescriptor",
    c"cudnnSetRNNDescriptor_v8",
    c"cudnnCreateRNNDataDescriptor",
    c"cudnnDestroyRNNDataDescriptor",
    c"cudnnSetRNNDataDescriptor",
    c"cudnnGetRNNWeightSpaceSize",
    c"cudnnGetRNNWeightParams",
    c"cudnnGetRNNTempSpaceSizes",
    c"cudnnBuildRNNDynamic",
    c"cudnnRNNForward",
];

// PersistDynamic compiles device code inside cuDNN, not through a Rust wrapper
// preflight its compilation, diagnostic and artifact APIs before cuDNN builds it
const NVRTC_SYMBOLS: &[&CStr] = &[
    c"nvrtcVersion",
    c"nvrtcGetErrorString",
    c"nvrtcGetNumSupportedArchs",
    c"nvrtcGetSupportedArchs",
    c"nvrtcCreateProgram",
    c"nvrtcDestroyProgram",
    c"nvrtcCompileProgram",
    c"nvrtcGetProgramLogSize",
    c"nvrtcGetProgramLog",
    c"nvrtcGetPTXSize",
    c"nvrtcGetPTX",
    c"nvrtcGetCUBINSize",
    c"nvrtcGetCUBIN",
    c"nvrtcAddNameExpression",
    c"nvrtcGetLoweredName",
    c"nvrtcGetLTOIRSize",
    c"nvrtcGetLTOIR",
];

fn present(library: CudaLibrary) -> Result<Library, Failure> {
    let (name, symbols) = match library {
        CudaLibrary::Cublas => ("cublas", CUBLAS_SYMBOLS),
        CudaLibrary::Cudnn => ("cudnn", CUDNN_SYMBOLS),
        CudaLibrary::Nvrtc => ("nvrtc", NVRTC_SYMBOLS),
        CudaLibrary::Driver => ("cuda", &[][..]),
    };
    preflight(library, cudarc::get_lib_name_candidates(name), symbols)
}

fn preflight(
    library: CudaLibrary,
    choices: impl IntoIterator<Item = impl AsRef<OsStr>>,
    symbols: &[&CStr],
) -> Result<Library, Failure> {
    for choice in choices {
        // SAFETY: CUDA library initializers have no caller preconditions
        let Ok(checked) = (unsafe { Library::new(choice.as_ref()) }) else {
            continue;
        };
        // cudarc selects the first file that opens in this exact order; a missing
        // symbol in that file must fail, not fall through to a different version
        for symbol in symbols {
            // SAFETY: lookup only; the untyped address is never dereferenced or called
            unsafe { checked.get::<*const ()>(symbol.to_bytes_with_nul()) }
                .map_err(|_| Failure::Missing(library))?;
        }

        return Ok(checked);
    }

    Err(Failure::Missing(library))
}

#[cfg(test)]
mod tests {
    use std::ffi::CStr;
    use std::path::{Path, PathBuf};
    use std::process::Command;
    use std::sync::OnceLock;
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::{CUBLAS_SYMBOLS, CUDNN_SYMBOLS, NVRTC_SYMBOLS, cached, preflight};
    use crate::inference::cuda::{CudaError, CudaLibrary};

    struct StubDirectory(PathBuf);

    impl StubDirectory {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let sequence = NEXT.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "speakrs-library-preflight-{}-{sequence}",
                std::process::id()
            ));
            std::fs::create_dir(&path).unwrap();
            Self(path)
        }

        fn build(&self, index: usize, symbols: &[&CStr], omitted: Option<&CStr>) -> PathBuf {
            let source = self.0.join(format!("stub-{index}.c"));
            let output = self
                .0
                .join(format!("stub-{index}.{}", std::env::consts::DLL_EXTENSION));
            let exports = symbols
                .iter()
                .filter(|symbol| Some(**symbol) != omitted)
                .map(|symbol| format!("void {}(void) {{}}\n", symbol.to_str().unwrap()))
                .collect::<String>();
            std::fs::write(&source, exports).unwrap();
            let result = Command::new("cc")
                .args(["-shared", "-fPIC"])
                .arg(&source)
                .arg("-o")
                .arg(&output)
                .output()
                .unwrap();
            assert!(
                result.status.success(),
                "{}",
                String::from_utf8_lossy(&result.stderr)
            );
            output
        }
    }

    impl Drop for StubDirectory {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.0).unwrap();
        }
    }

    fn unavailable(error: CudaError, expected: CudaLibrary) {
        assert!(matches!(
            error,
            CudaError::LibraryUnavailable { library } if library == expected
        ));
    }

    #[test]
    fn every_required_symbol_is_checked_before_cudarc_can_run() {
        for (library, symbols) in [
            (CudaLibrary::Cublas, CUBLAS_SYMBOLS),
            (CudaLibrary::Cudnn, CUDNN_SYMBOLS),
            (CudaLibrary::Nvrtc, NVRTC_SYMBOLS),
        ] {
            let directory = StubDirectory::new();
            let complete = directory.build(0, symbols, None);
            let checked = preflight(library, [&complete], symbols).unwrap();
            // keep the checked object open after its name disappears
            std::fs::remove_file(&complete).unwrap();
            // SAFETY: the symbol is looked up, never called as a stub CUDA API
            assert!(unsafe { checked.get::<*const ()>(symbols[0].to_bytes_with_nul()) }.is_ok());
            for (index, omitted) in symbols.iter().enumerate() {
                let incomplete = directory.build(index + 1, symbols, Some(omitted));
                let error = preflight(library, [&incomplete], symbols).unwrap_err();
                unavailable(error.into(), library);
            }
        }
    }

    #[test]
    fn incomplete_first_library_does_not_fall_through_and_failure_is_cached() {
        let directory = StubDirectory::new();
        let incomplete = directory.build(0, CUBLAS_SYMBOLS, Some(CUBLAS_SYMBOLS[0]));
        let complete = directory.build(1, CUBLAS_SYMBOLS, None);
        let cell = OnceLock::new();
        let mut initializations = 0;
        for _ in 0..2 {
            let result = cell.get_or_init(|| {
                initializations += 1;
                preflight(
                    CudaLibrary::Cublas,
                    [&incomplete, &complete],
                    CUBLAS_SYMBOLS,
                )
            });
            unavailable(cached(result).unwrap_err(), CudaLibrary::Cublas);
        }

        assert_eq!(initializations, 1);
        assert!(preflight(CudaLibrary::Cublas, [&complete], CUBLAS_SYMBOLS).is_ok());
        let missing = directory.0.join("not-installed");
        unavailable(
            preflight(CudaLibrary::Nvrtc, [&missing], NVRTC_SYMBOLS)
                .unwrap_err()
                .into(),
            CudaLibrary::Nvrtc,
        );
    }

    #[test]
    fn direct_rnn_and_version_calls_have_preflight_symbols() {
        for (source, prefix, symbols) in
            [(include_str!("segmentation/rnn.rs"), "dnn::", CUDNN_SYMBOLS)]
        {
            for suffix in source.split(prefix).skip(1) {
                let name = suffix
                    .split(|ch: char| !ch.is_ascii_alphanumeric() && ch != '_')
                    .next()
                    .unwrap();
                if name.starts_with("cudnn") && suffix[name.len()..].starts_with('(') {
                    assert!(
                        symbols
                            .iter()
                            .any(|symbol| symbol.to_bytes() == name.as_bytes()),
                        "missing preflight for {name}"
                    );
                }
            }
        }
    }

    #[test]
    fn nonexistent_file_is_a_typed_error() {
        let directory = StubDirectory::new();
        let missing: &Path = &directory.0.join("missing-library");
        for library in [CudaLibrary::Cublas, CudaLibrary::Cudnn, CudaLibrary::Nvrtc] {
            unavailable(
                preflight(library, [missing], &[]).unwrap_err().into(),
                library,
            );
        }
    }
}
