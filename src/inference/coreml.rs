use std::ffi::c_void;
use std::fmt;
use std::path::Path;
use std::ptr::NonNull;

use block2::RcBlock;
use objc2::rc::Retained;
use objc2::runtime::{AnyObject, ProtocolObject};
use objc2_core_ml::{MLComputeUnits, MLFeatureProvider, MLModel};
use objc2_foundation::{NSArray, NSCopying, NSError, NSMutableDictionary, NSNumber, NSString};

mod array;
mod path;
mod runtime;

use crate::inference::geometry::{CoreMlTensor, GeometryError, TensorLayout};
use array::{
    contiguous_strides, create_multi_array_cached_with_deallocator,
    create_multi_array_with_deallocator, extract_output, ns_number_array,
};

fn noop_deallocator() -> RcBlock<dyn Fn(NonNull<c_void>)> {
    RcBlock::new(|_ptr: NonNull<c_void>| {})
}
pub(crate) use path::{coreml_model_path, coreml_w8a16_model_path};
use runtime::{
    build_feature_provider, insert_input_feature, load_model, output_multi_array, predict_output,
};

#[derive(Debug, Clone, Copy)]
pub(crate) enum GpuPrecision {
    /// FP16 intermediate accumulations on GPU
    Low,
    /// Full FP32 accumulation on GPU
    #[expect(dead_code)]
    Full,
}

#[derive(Debug)]
pub(crate) enum CoreMlError {
    LoadFailed(String),
    PredictionFailed(String),
    OutputNotFound(String),
    ArrayCreationFailed(String),
    InvalidGeometry(GeometryError),
}

impl fmt::Display for CoreMlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LoadFailed(msg) => write!(f, "CoreML load failed: {msg}"),
            Self::PredictionFailed(msg) => write!(f, "CoreML prediction failed: {msg}"),
            Self::OutputNotFound(name) => write!(f, "CoreML output '{name}' not found"),
            Self::ArrayCreationFailed(msg) => write!(f, "CoreML array creation failed: {msg}"),
            Self::InvalidGeometry(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for CoreMlError {}

impl From<GeometryError> for CoreMlError {
    fn from(error: GeometryError) -> Self {
        Self::InvalidGeometry(error)
    }
}

/// Pre-computed NSArray<NSNumber> for shape and strides, avoiding per-call allocation
pub(crate) struct CachedInputShape {
    name: Retained<NSString>,
    ns_shape: Retained<NSArray<NSNumber>>,
    ns_strides: Retained<NSArray<NSNumber>>,
    total_elements: usize,
}

impl CachedInputShape {
    pub fn new(name: &str, shape: &[usize]) -> Self {
        Self::try_new(name, shape).expect("CoreML input shape product must fit in usize")
    }

    pub fn try_new(name: &str, shape: &[usize]) -> Result<Self, CoreMlError> {
        let layout = TensorLayout::from_dims(shape, "coreml input")?;
        let ns_shape = ns_number_array(shape);
        let ns_strides = ns_number_array(&contiguous_strides(shape));

        Ok(Self {
            name: NSString::from_str(name),
            ns_shape,
            ns_strides,
            total_elements: layout.element_count(),
        })
    }

    /// Bind a flat buffer to this cached shape before native array construction
    pub fn bind<'a>(&'a self, data: &'a [f32]) -> Result<BoundCoreMlInput<'a>, CoreMlError> {
        crate::inference::geometry::require_exact_len(
            data.len(),
            self.total_elements,
            "coreml input",
        )?;
        Ok(BoundCoreMlInput { cached: self, data })
    }
}

/// Checked CoreML input slice paired with its cached shape
pub(crate) struct BoundCoreMlInput<'a> {
    pub cached: &'a CachedInputShape,
    pub data: &'a [f32],
}

// SAFETY: CachedInputShape fields are immutable after construction and only accessed via &self
unsafe impl Send for CachedInputShape {}
// SAFETY: CachedInputShape fields are immutable after construction and only accessed via &self
unsafe impl Sync for CachedInputShape {}

pub(crate) struct CoreMlModel {
    model: Retained<MLModel>,
    output_name: String,
    output_key: Retained<NSString>,
    noop_deallocator: RcBlock<dyn Fn(NonNull<c_void>)>,
    input_dict: Retained<NSMutableDictionary<NSString, AnyObject>>,
}

// SAFETY: CoreMlModel is only used from one thread at a time via &mut self
// SAFETY: MLModel prediction is thread-safe per Apple docs, and the remaining fields are only
// SAFETY: accessed inside predict calls that require exclusive access
unsafe impl Send for CoreMlModel {}

impl CoreMlModel {
    /// Load a compiled .mlmodelc bundle
    pub fn load(
        path: &Path,
        compute_units: MLComputeUnits,
        output_name: &str,
        gpu_precision: GpuPrecision,
    ) -> Result<Self, CoreMlError> {
        Ok(Self {
            model: load_model(path, compute_units, gpu_precision)?,
            output_key: NSString::from_str(output_name),
            noop_deallocator: noop_deallocator(),
            input_dict: NSMutableDictionary::new(),
            output_name: output_name.to_owned(),
        })
    }

    /// Run prediction with named inputs
    ///
    /// Each input is (name, shape, flat_data). The data slice must outlive this call.
    pub fn predict(
        &mut self,
        inputs: &[(&str, &[usize], &[f32])],
    ) -> Result<CoreMlTensor, CoreMlError> {
        self.input_dict.removeAllObjects();

        for &(name, shape, data) in inputs {
            let layout = TensorLayout::from_dims(shape, "coreml input")?;
            let bound = layout.bind(data)?;
            let multi_array =
                create_multi_array_with_deallocator(bound, shape, &self.noop_deallocator)?;
            let key = NSString::from_str(name);
            let key_copy: &ProtocolObject<dyn NSCopying> = ProtocolObject::from_ref(&*key);
            insert_input_feature(&self.input_dict, key_copy, &multi_array);
        }

        let provider = build_feature_provider(&self.input_dict)?;
        let input_ref: &ProtocolObject<dyn MLFeatureProvider> =
            ProtocolObject::from_ref(&*provider);
        let output_array =
            predict_output(&self.model, input_ref, &self.output_key, &self.output_name)?;
        extract_output(&output_array)
    }

    /// Run prediction with cached shape and stride objects.
    pub fn predict_cached(
        &mut self,
        inputs: &[(&CachedInputShape, &[f32])],
    ) -> Result<CoreMlTensor, CoreMlError> {
        self.input_dict.removeAllObjects();

        for &(cached, data) in inputs {
            let bound = cached.bind(data)?;
            let multi_array = create_multi_array_cached_with_deallocator(
                bound.data,
                bound.cached,
                &self.noop_deallocator,
            )?;
            let key_copy: &ProtocolObject<dyn NSCopying> = ProtocolObject::from_ref(&*cached.name);
            insert_input_feature(&self.input_dict, key_copy, &multi_array);
        }

        let provider = build_feature_provider(&self.input_dict)?;
        let input_ref: &ProtocolObject<dyn MLFeatureProvider> =
            ProtocolObject::from_ref(&*provider);
        let output_array =
            predict_output(&self.model, input_ref, &self.output_key, &self.output_name)?;
        extract_output(&output_array)
    }

    /// Default compute units. CoreML decides placement per operation.
    pub fn default_compute_units() -> MLComputeUnits {
        MLComputeUnits::All
    }
}

/// Thread-safe CoreML model wrapper that can be shared across threads.
///
/// Unlike `CoreMlModel`, this allocates a fresh input dictionary per call, so
/// prediction can take `&self`.
pub(crate) struct SharedCoreMlModel {
    model: Retained<MLModel>,
    output_name: String,
    output_key: Retained<NSString>,
}

// SAFETY: MLModel predictionFromFeatures is documented as thread-safe by Apple
// SAFETY: all per-call mutable state is allocated fresh inside predict_cached
unsafe impl Send for SharedCoreMlModel {}
// SAFETY: MLModel predictionFromFeatures is documented as thread-safe by Apple
// SAFETY: all per-call mutable state is allocated fresh inside predict_cached
unsafe impl Sync for SharedCoreMlModel {}

impl SharedCoreMlModel {
    /// Load a compiled .mlmodelc bundle
    pub fn load(
        path: &Path,
        compute_units: MLComputeUnits,
        output_name: &str,
        gpu_precision: GpuPrecision,
    ) -> Result<Self, CoreMlError> {
        Ok(Self {
            model: load_model(path, compute_units, gpu_precision)?,
            output_key: NSString::from_str(output_name),
            output_name: output_name.to_owned(),
        })
    }

    /// Run prediction with cached shape and stride objects.
    pub fn predict_cached(
        &self,
        inputs: &[(&CachedInputShape, &[f32])],
    ) -> Result<CoreMlTensor, CoreMlError> {
        let deallocator = noop_deallocator();
        let input_dict: Retained<NSMutableDictionary<NSString, AnyObject>> =
            NSMutableDictionary::new();

        for &(cached, data) in inputs {
            let bound = cached.bind(data)?;
            let multi_array =
                create_multi_array_cached_with_deallocator(bound.data, bound.cached, &deallocator)?;
            let key_copy: &ProtocolObject<dyn NSCopying> = ProtocolObject::from_ref(&*cached.name);
            insert_input_feature(&input_dict, key_copy, &multi_array);
        }

        let provider = build_feature_provider(&input_dict)?;
        let input_ref: &ProtocolObject<dyn MLFeatureProvider> =
            ProtocolObject::from_ref(&*provider);
        let output_array =
            predict_output(&self.model, input_ref, &self.output_key, &self.output_name)?;
        extract_output(&output_array)
    }

    /// Async prediction: queues work on ANE and returns via callback
    ///
    /// Uses predictionFromFeatures:completionHandler: which lets CoreML
    /// pipeline multiple predictions onto ANE simultaneously. Critical for
    /// concurrent ANE workers -- sync prediction serializes while async
    /// lets the ANE queue depth (127) fill up
    #[expect(dead_code)]
    pub fn predict_async(
        &self,
        inputs: &[(&CachedInputShape, &[f32])],
    ) -> Result<CoreMlTensor, CoreMlError> {
        let deallocator = noop_deallocator();
        let input_dict: Retained<NSMutableDictionary<NSString, AnyObject>> =
            NSMutableDictionary::new();

        for &(cached, data) in inputs {
            let bound = cached.bind(data)?;
            let multi_array =
                create_multi_array_cached_with_deallocator(bound.data, bound.cached, &deallocator)?;
            let key_copy: &ProtocolObject<dyn NSCopying> = ProtocolObject::from_ref(&*cached.name);
            insert_input_feature(&input_dict, key_copy, &multi_array);
        }

        let provider = build_feature_provider(&input_dict)?;
        let input_ref: &ProtocolObject<dyn MLFeatureProvider> =
            ProtocolObject::from_ref(&*provider);

        // bridge the async callback to a blocking channel
        let (tx, rx) = std::sync::mpsc::sync_channel::<
            Result<Retained<ProtocolObject<dyn MLFeatureProvider>>, String>,
        >(1);

        let completion = block2::RcBlock::new(
            move |output: *mut ProtocolObject<dyn MLFeatureProvider>, error: *mut NSError| {
                if !error.is_null() {
                    // SAFETY: error is non-null in this branch and points to the NSError passed
                    // SAFETY: by CoreML for the duration of the callback invocation
                    let err_msg = unsafe { (*error).localizedDescription() }.to_string();
                    let _ = tx.send(Err(err_msg));
                } else if output.is_null() {
                    let _ = tx.send(Err("nil output with no error".to_owned()));
                } else {
                    // SAFETY: output is non-null in this branch and retain extends the lifetime so
                    // SAFETY: the returned feature provider survives after the callback returns
                    let Some(retained) = (unsafe { Retained::retain(output) }) else {
                        let _ = tx.send(Err("failed to retain CoreML output".to_owned()));
                        return;
                    };
                    let _ = tx.send(Ok(retained));
                }
            },
        );

        // SAFETY: input_ref and completion stay alive for the duration of the Objective-C call and
        // SAFETY: CoreML copies/retains the callback before invoking it asynchronously
        unsafe {
            self.model
                .predictionFromFeatures_completionHandler(input_ref, &completion);
        }

        // block until the callback fires
        let output = rx
            .recv()
            .map_err(|_| CoreMlError::PredictionFailed("channel closed".to_owned()))?
            .map_err(CoreMlError::PredictionFailed)?;

        let output_array = output_multi_array(&output, &self.output_key, &self.output_name)?;
        extract_output(&output_array)
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::Arc;

    use super::{CachedInputShape, CoreMlModel, GpuPrecision, SharedCoreMlModel};
    use crate::inference::geometry::GeometryError;

    #[test]
    fn bind_rejects_short_and_long_input_without_calling_coreml() {
        let cached = CachedInputShape::new("input", &[2, 2]);
        let short = match cached.bind(&[1.0, 2.0, 3.0]) {
            Ok(_) => panic!("expected short input to fail"),
            Err(error) => error,
        };
        let long = match cached.bind(&[1.0, 2.0, 3.0, 4.0, 5.0]) {
            Ok(_) => panic!("expected long input to fail"),
            Err(error) => error,
        };
        assert!(matches!(
            short,
            super::CoreMlError::InvalidGeometry(GeometryError::LengthMismatch {
                expected: 4,
                actual: 3,
                ..
            })
        ));
        assert!(matches!(
            long,
            super::CoreMlError::InvalidGeometry(GeometryError::LengthMismatch {
                expected: 4,
                actual: 5,
                ..
            })
        ));
        let exact = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(cached.bind(&exact).unwrap().data, &exact);
    }

    #[test]
    fn try_new_rejects_shape_product_overflow() {
        let error = match CachedInputShape::try_new("input", &[usize::MAX, 2]) {
            Ok(_) => panic!("expected overflow to fail"),
            Err(error) => error,
        };
        assert!(matches!(
            error,
            super::CoreMlError::InvalidGeometry(GeometryError::Overflow { .. })
        ));
    }

    fn fixture_model_path(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures")
            .join("models")
            .join(name)
    }

    #[test]
    fn shared_cached_prediction_handles_concurrent_calls_when_bundle_available() {
        let model_path = fixture_model_path("segmentation-3.0.mlmodelc");
        if !model_path.exists() {
            eprintln!(
                "skipping CoreML cached prediction stress test; missing {}",
                model_path.display()
            );
            return;
        }

        let model = Arc::new(
            SharedCoreMlModel::load(
                &model_path,
                CoreMlModel::default_compute_units(),
                "output",
                GpuPrecision::Low,
            )
            .unwrap(),
        );
        let cached_shape = Arc::new(CachedInputShape::new("input", &[1, 1, 160_000]));
        let input = Arc::new(vec![0.0_f32; 160_000]);

        let handles: Vec<_> = (0..4)
            .map(|_| {
                let model = Arc::clone(&model);
                let cached_shape = Arc::clone(&cached_shape);
                let input = Arc::clone(&input);
                std::thread::spawn(move || {
                    for _ in 0..4 {
                        let tensor = model
                            .predict_cached(&[(&*cached_shape, input.as_slice())])
                            .unwrap();
                        assert_eq!(tensor.layout().dims().len(), 3);
                        assert!(!tensor.into_data().is_empty());
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }
    }
}
