//! One checked matrix multiplication boundary for native CPU models

use ndarray::{ArrayView2, ArrayViewMut2, linalg::general_mat_mul};

use crate::inference::{InferenceError, TensorShapeError};

/// Computes `output = a @ b + beta * output` without changing global thread settings
pub(crate) fn matmul(
    a: ArrayView2<'_, f32>,
    b: ArrayView2<'_, f32>,
    mut output: ArrayViewMut2<'_, f32>,
    beta: f32,
) -> Result<(), InferenceError> {
    if a.ncols() != b.nrows() || output.dim() != (a.nrows(), b.ncols()) {
        return Err(TensorShapeError::ShapeMismatch {
            context: "CPU matmul",
            expected: vec![a.ncols(), a.nrows(), b.ncols()],
            actual: vec![b.nrows(), output.nrows(), output.ncols()],
        }
        .into());
    }

    if output.is_empty() {
        return Ok(());
    }
    if a.ncols() == 0 {
        // beta zero must discard even a NaN in the old output
        output.mapv_inplace(|value| if beta == 0.0 { 0.0 } else { beta * value });
        return Ok(());
    }

    #[cfg(target_os = "macos")]
    if accelerate::multiply(a, b, output.view_mut(), beta) {
        return Ok(());
    }

    general_mat_mul(1.0, &a, &b, beta, &mut output);
    Ok(())
}

#[cfg(target_os = "macos")]
mod accelerate {
    use std::sync::OnceLock;

    use libloading::Library;
    use ndarray::{ArrayView2, ArrayViewMut2};

    type Sgemm = unsafe extern "C" fn(
        i32,
        i32,
        i32,
        i32,
        i32,
        i32,
        f32,
        *const f32,
        i32,
        *const f32,
        i32,
        f32,
        *mut f32,
        i32,
    );
    type GetThreading = unsafe extern "C" fn() -> u32;
    type SetThreading = unsafe extern "C" fn(u32) -> i32;

    struct Accelerate {
        // the process-lifetime owner keeps all resolved code pointers valid
        _library: Library,
        sgemm: Sgemm,
        threading: (GetThreading, SetThreading),
    }

    impl Accelerate {
        fn load() -> Option<Self> {
            // SAFETY: the system framework is trusted; LP64 CBLAS and thread API
            // signatures match the SDK cblas.h, cblas_new.h and thread_api.h
            unsafe {
                let library =
                    Library::new("/System/Library/Frameworks/Accelerate.framework/Accelerate")
                        .ok()?;
                let sgemm = library
                    .get::<Sgemm>(b"cblas_sgemm$NEWLAPACK\0")
                    .ok()
                    .map(|symbol| *symbol)?;
                let get = *library.get::<GetThreading>(b"BLASGetThreading\0").ok()?;
                let set = *library.get::<SetThreading>(b"BLASSetThreading\0").ok()?;
                let threading = (get, set);
                Some(Self {
                    _library: library,
                    sgemm,
                    threading,
                })
            }
        }
    }

    struct ThreadScope {
        previous: u32,
        set: SetThreading,
    }

    impl ThreadScope {
        fn single((get, set): (GetThreading, SetThreading)) -> Option<Self> {
            // SAFETY: these optional SDK functions affect only the calling thread
            unsafe {
                let previous = get();
                (set(1) == 0).then_some(Self { previous, set })
            }
        }
    }

    impl Drop for ThreadScope {
        fn drop(&mut self) {
            // SAFETY: this guard cannot escape the synchronous call; restore the
            // exact valid setting read from this same thread, with no panic in Drop
            unsafe {
                (self.set)(self.previous);
            }
        }
    }

    fn layout(view: ArrayView2<'_, f32>) -> Option<(i32, i32)> {
        let strides = view.strides();
        if strides[1] == 1 && strides[0] >= view.ncols().max(1) as isize {
            return Some((111, i32::try_from(strides[0]).ok()?));
        }
        if strides[0] == 1 && strides[1] >= view.nrows().max(1) as isize {
            return Some((112, i32::try_from(strides[1]).ok()?));
        }
        None
    }

    pub(super) fn multiply(
        a: ArrayView2<'_, f32>,
        b: ArrayView2<'_, f32>,
        mut output: ArrayViewMut2<'_, f32>,
        beta: f32,
    ) -> bool {
        // a dense transposed output is the reversed product, with no copy
        if output.strides()[1] != 1 && output.strides()[0] == 1 {
            return multiply(
                b.reversed_axes(),
                a.reversed_axes(),
                output.reversed_axes(),
                beta,
            );
        }
        let Some((ta, lda)) = layout(a) else {
            return false;
        };
        let Some((tb, ldb)) = layout(b) else {
            return false;
        };
        let Some((111, ldc)) = layout(output.view()) else {
            return false;
        };
        let (Ok(m), Ok(n), Ok(k)) = (
            i32::try_from(a.nrows()),
            i32::try_from(b.ncols()),
            i32::try_from(a.ncols()),
        ) else {
            return false;
        };
        static API: OnceLock<Option<Accelerate>> = OnceLock::new();
        let Some(api) = API.get_or_init(Accelerate::load) else {
            return false;
        };
        let Some(_scope) = ThreadScope::single(api.threading) else {
            return false;
        };
        // SAFETY: matmul checked logical dimensions first; layout accepts only
        // positive non-overlapping dense row/column strides and LP64 extents
        // ndarray's borrowed views contain every referenced element and the
        // exclusive output borrow excludes input/output aliasing
        unsafe {
            (api.sgemm)(
                101,
                ta,
                tb,
                m,
                n,
                k,
                1.0,
                a.as_ptr(),
                lda,
                b.as_ptr(),
                ldb,
                beta,
                output.as_mut_ptr(),
                ldc,
            );
        }
        true
    }

    #[cfg(test)]
    mod tests {
        use super::{Accelerate, ThreadScope};

        #[test]
        fn available_thread_scope_restores_setting_or_uses_rust_fallback() {
            let Some(api) = Accelerate::load() else {
                let input = ndarray::arr2(&[[1.0, 2.0], [3.0, 4.0]]);
                let mut output = ndarray::Array2::zeros((2, 2));
                super::super::matmul(input.view(), input.view(), output.view_mut(), 0.0).unwrap();
                assert_eq!(output, ndarray::arr2(&[[7.0, 10.0], [15.0, 22.0]]));
                return;
            };
            let (get, set) = api.threading;
            // SAFETY: thread-local SDK setting; the outer guard also restores it
            unsafe {
                let original = get();
                let _restore = ThreadScope {
                    previous: original,
                    set,
                };
                assert_eq!(set(0), 0);
                {
                    let _single = ThreadScope::single(api.threading).unwrap();
                    assert_eq!(get(), 1);
                }
                assert_eq!(get(), 0);
            }
        }
    }
}

#[cfg(test)]
mod tests;
