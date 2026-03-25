use std::ffi::c_void;
use std::fmt;
use std::path::Path;
use std::ptr::NonNull;

use block2::RcBlock;
use objc2::AnyThread;
use objc2::rc::Retained;
use objc2::runtime::{AnyObject, ProtocolObject};
use objc2_core_ml::{
    MLComputeUnits, MLDictionaryFeatureProvider, MLFeatureProvider, MLFeatureValue, MLModel,
    MLModelConfiguration, MLMultiArray, MLMultiArrayDataType,
};
use objc2_foundation::{
    NSArray, NSCopying, NSError, NSMutableDictionary, NSNumber, NSString, NSURL,
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
}

impl fmt::Display for CoreMlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LoadFailed(msg) => write!(f, "CoreML load failed: {msg}"),
            Self::PredictionFailed(msg) => write!(f, "CoreML prediction failed: {msg}"),
            Self::OutputNotFound(name) => write!(f, "CoreML output '{name}' not found"),
            Self::ArrayCreationFailed(msg) => write!(f, "CoreML array creation failed: {msg}"),
        }
    }
}

impl std::error::Error for CoreMlError {}

/// Pre-computed NSArray<NSNumber> for shape and strides, avoiding per-call allocation
pub(crate) struct CachedInputShape {
    name: Retained<NSString>,
    ns_shape: Retained<NSArray<NSNumber>>,
    ns_strides: Retained<NSArray<NSNumber>>,
    total_elements: usize,
}

impl CachedInputShape {
    pub fn new(name: &str, shape: &[usize]) -> Self {
        let ns_shape = ns_number_array(shape);
        let ns_strides = ns_number_array(&contiguous_strides(shape));

        let total_elements = shape.iter().product();

        Self {
            name: NSString::from_str(name),
            ns_shape,
            ns_strides,
            total_elements,
        }
    }
}

// SAFETY: CachedInputShape fields are immutable after construction and only accessed via &self
unsafe impl Send for CachedInputShape {}
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
        let path_str = NSString::from_str(&path.to_string_lossy());
        let url = NSURL::fileURLWithPath_isDirectory(&path_str, true);

        let config = unsafe { MLModelConfiguration::new() };
        unsafe { config.setComputeUnits(compute_units) };
        let low_precision = matches!(gpu_precision, GpuPrecision::Low);
        unsafe { config.setAllowLowPrecisionAccumulationOnGPU(low_precision) };

        let model = unsafe { MLModel::modelWithContentsOfURL_configuration_error(&url, &config) }
            .map_err(|e| CoreMlError::LoadFailed(format!("{e}")))?;

        Ok(Self {
            model,
            output_key: NSString::from_str(output_name),
            noop_deallocator: RcBlock::new(|_ptr: NonNull<c_void>| {}),
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
    ) -> Result<(Vec<f32>, Vec<usize>), CoreMlError> {
        self.input_dict.removeAllObjects();

        for &(name, shape, data) in inputs {
            let multi_array =
                create_multi_array_with_deallocator(data, shape, &self.noop_deallocator)?;
            let feature_value = unsafe { MLFeatureValue::featureValueWithMultiArray(&multi_array) };
            let key = NSString::from_str(name);
            let key_copy: &ProtocolObject<dyn NSCopying> = ProtocolObject::from_ref(&*key);
            let value_ref: &AnyObject =
                unsafe { &*((&*feature_value) as *const MLFeatureValue as *const AnyObject) };
            unsafe { self.input_dict.setObject_forKey(value_ref, key_copy) };
        }

        let provider = unsafe {
            MLDictionaryFeatureProvider::initWithDictionary_error(
                MLDictionaryFeatureProvider::alloc(),
                &self.input_dict,
            )
        }
        .map_err(|e| CoreMlError::PredictionFailed(format!("feature provider: {e}")))?;

        let input_ref: &ProtocolObject<dyn MLFeatureProvider> =
            ProtocolObject::from_ref(&*provider);
        let output = unsafe { self.model.predictionFromFeatures_error(input_ref) }
            .map_err(|e| CoreMlError::PredictionFailed(format!("{e}")))?;

        let output_value = unsafe { output.featureValueForName(&self.output_key) }
            .ok_or_else(|| CoreMlError::OutputNotFound(self.output_name.clone()))?;
        let output_array = unsafe { output_value.multiArrayValue() }
            .ok_or_else(|| CoreMlError::OutputNotFound(self.output_name.clone()))?;

        extract_output(&output_array)
    }

    /// Run prediction with pre-cached shape/strides objects (avoids per-call NSNumber allocation)
    pub fn predict_cached(
        &mut self,
        inputs: &[(&CachedInputShape, &[f32])],
    ) -> Result<(Vec<f32>, Vec<usize>), CoreMlError> {
        self.input_dict.removeAllObjects();

        for &(cached, data) in inputs {
            debug_assert_eq!(data.len(), cached.total_elements);
            let multi_array =
                create_multi_array_cached_with_deallocator(data, cached, &self.noop_deallocator)?;
            let feature_value = unsafe { MLFeatureValue::featureValueWithMultiArray(&multi_array) };
            let key_copy: &ProtocolObject<dyn NSCopying> = ProtocolObject::from_ref(&*cached.name);
            let value_ref: &AnyObject =
                unsafe { &*((&*feature_value) as *const MLFeatureValue as *const AnyObject) };
            unsafe { self.input_dict.setObject_forKey(value_ref, key_copy) };
        }

        let provider = unsafe {
            MLDictionaryFeatureProvider::initWithDictionary_error(
                MLDictionaryFeatureProvider::alloc(),
                &self.input_dict,
            )
        }
        .map_err(|e| CoreMlError::PredictionFailed(format!("feature provider: {e}")))?;

        let input_ref: &ProtocolObject<dyn MLFeatureProvider> =
            ProtocolObject::from_ref(&*provider);
        let output = unsafe { self.model.predictionFromFeatures_error(input_ref) }
            .map_err(|e| CoreMlError::PredictionFailed(format!("{e}")))?;

        let output_value = unsafe { output.featureValueForName(&self.output_key) }
            .ok_or_else(|| CoreMlError::OutputNotFound(self.output_name.clone()))?;
        let output_array = unsafe { output_value.multiArrayValue() }
            .ok_or_else(|| CoreMlError::OutputNotFound(self.output_name.clone()))?;

        extract_output(&output_array)
    }

    /// Default: All compute units — CoreML decides per-op placement
    pub fn default_compute_units() -> MLComputeUnits {
        MLComputeUnits::All
    }
}

fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    strides
}

fn ns_number_array(values: &[usize]) -> Retained<NSArray<NSNumber>> {
    let numbers: Vec<Retained<NSNumber>> = values
        .iter()
        .copied()
        .map(|value| NSNumber::new_isize(value as isize))
        .collect();

    NSArray::from_retained_slice(&numbers)
}

/// Create an MLMultiArray wrapping a data pointer with a shared no-op deallocator
fn create_multi_array_with_deallocator(
    data: &[f32],
    shape: &[usize],
    deallocator: &RcBlock<dyn Fn(NonNull<c_void>)>,
) -> Result<Retained<MLMultiArray>, CoreMlError> {
    let ns_shape = ns_number_array(shape);
    let ns_strides = ns_number_array(&contiguous_strides(shape));

    let ptr = NonNull::new(data.as_ptr() as *mut c_void)
        .ok_or_else(|| CoreMlError::ArrayCreationFailed("null data pointer".into()))?;

    #[allow(deprecated)]
    unsafe {
        MLMultiArray::initWithDataPointer_shape_dataType_strides_deallocator_error(
            MLMultiArray::alloc(),
            ptr,
            &ns_shape,
            MLMultiArrayDataType::Float32,
            &ns_strides,
            Some(deallocator),
        )
    }
    .map_err(|e| CoreMlError::ArrayCreationFailed(format!("{e}")))
}

/// Create an MLMultiArray using pre-cached shape/strides and a shared deallocator
fn create_multi_array_cached_with_deallocator(
    data: &[f32],
    cached: &CachedInputShape,
    deallocator: &RcBlock<dyn Fn(NonNull<c_void>)>,
) -> Result<Retained<MLMultiArray>, CoreMlError> {
    let ptr = NonNull::new(data.as_ptr() as *mut c_void)
        .ok_or_else(|| CoreMlError::ArrayCreationFailed("null data pointer".into()))?;

    #[allow(deprecated)]
    unsafe {
        MLMultiArray::initWithDataPointer_shape_dataType_strides_deallocator_error(
            MLMultiArray::alloc(),
            ptr,
            &cached.ns_shape,
            MLMultiArrayDataType::Float32,
            &cached.ns_strides,
            Some(deallocator),
        )
    }
    .map_err(|e| CoreMlError::ArrayCreationFailed(format!("{e}")))
}

/// Copy output MLMultiArray data into a Vec<f32> and return the shape.
/// Handles both FP32 and FP16 output data types (FP16 is auto-converted to FP32)
#[allow(deprecated)]
fn extract_output(array: &MLMultiArray) -> Result<(Vec<f32>, Vec<usize>), CoreMlError> {
    let count = unsafe { array.count() } as usize;
    let ptr = unsafe { array.dataPointer() };
    let dtype = unsafe { array.dataType() };

    let ns_shape = unsafe { array.shape() };
    let shape: Vec<usize> = (0..ns_shape.len())
        .map(|i| ns_shape.objectAtIndex(i).as_isize() as usize)
        .collect();

    let data = if dtype == MLMultiArrayDataType::Float16 {
        let fp16_data = unsafe { std::slice::from_raw_parts(ptr.as_ptr() as *const u16, count) };
        fp16_data.iter().copied().map(f16_to_f32).collect()
    } else {
        let fp32_data = unsafe { std::slice::from_raw_parts(ptr.as_ptr() as *const f32, count) };
        fp32_data.to_vec()
    };

    Ok((data, shape))
}

/// Convert IEEE 754 half-precision (FP16) bits to f32
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mant = (bits & 0x3ff) as u32;

    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign << 31);
        }
        // subnormal: normalize
        let mut e: i32 = exp as i32;
        let mut m = mant;
        while m & 0x400 == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x3ff;
        let f32_exp = ((127 - 15) + e + 1) as u32;
        return f32::from_bits((sign << 31) | (f32_exp << 23) | (m << 13));
    }

    if exp == 0x1f {
        return f32::from_bits((sign << 31) | (0xff_u32 << 23) | (mant << 13));
    }

    let f32_exp = exp - 15 + 127;
    f32::from_bits((sign << 31) | (f32_exp << 23) | (mant << 13))
}

/// Thread-safe CoreML model wrapper that can be shared across threads
///
/// Unlike CoreMlModel, this allocates a fresh input dictionary per call
/// So predict can take `&self`
/// Multiple threads can call predict concurrently on the same model instance
pub(crate) struct SharedCoreMlModel {
    model: Retained<MLModel>,
    output_name: String,
    output_key: Retained<NSString>,
}

// SAFETY: MLModel predictionFromFeatures is documented as thread-safe by Apple
// SAFETY: all per-call mutable state is allocated fresh inside predict_cached
unsafe impl Send for SharedCoreMlModel {}
unsafe impl Sync for SharedCoreMlModel {}

impl SharedCoreMlModel {
    /// Load a compiled .mlmodelc bundle
    pub fn load(
        path: &Path,
        compute_units: MLComputeUnits,
        output_name: &str,
        gpu_precision: GpuPrecision,
    ) -> Result<Self, CoreMlError> {
        let path_str = NSString::from_str(&path.to_string_lossy());
        let url = NSURL::fileURLWithPath_isDirectory(&path_str, true);

        let config = unsafe { MLModelConfiguration::new() };
        unsafe { config.setComputeUnits(compute_units) };
        let low_precision = matches!(gpu_precision, GpuPrecision::Low);
        unsafe { config.setAllowLowPrecisionAccumulationOnGPU(low_precision) };

        let model = unsafe { MLModel::modelWithContentsOfURL_configuration_error(&url, &config) }
            .map_err(|e| CoreMlError::LoadFailed(format!("{e}")))?;

        Ok(Self {
            model,
            output_key: NSString::from_str(output_name),
            output_name: output_name.to_owned(),
        })
    }

    /// Run prediction with pre-cached shape/strides objects
    ///
    /// Thread-safe: allocates fresh input dict per call
    pub fn predict_cached(
        &self,
        inputs: &[(&CachedInputShape, &[f32])],
    ) -> Result<(Vec<f32>, Vec<usize>), CoreMlError> {
        let deallocator = RcBlock::new(|_ptr: NonNull<c_void>| {});
        let input_dict: Retained<NSMutableDictionary<NSString, AnyObject>> =
            NSMutableDictionary::new();

        for &(cached, data) in inputs {
            debug_assert_eq!(data.len(), cached.total_elements);
            let multi_array =
                create_multi_array_cached_with_deallocator(data, cached, &deallocator)?;
            let feature_value = unsafe { MLFeatureValue::featureValueWithMultiArray(&multi_array) };
            let key_copy: &ProtocolObject<dyn NSCopying> = ProtocolObject::from_ref(&*cached.name);
            let value_ref: &AnyObject =
                unsafe { &*((&*feature_value) as *const MLFeatureValue as *const AnyObject) };
            unsafe { input_dict.setObject_forKey(value_ref, key_copy) };
        }

        let provider = unsafe {
            MLDictionaryFeatureProvider::initWithDictionary_error(
                MLDictionaryFeatureProvider::alloc(),
                &input_dict,
            )
        }
        .map_err(|e| CoreMlError::PredictionFailed(format!("feature provider: {e}")))?;

        let input_ref: &ProtocolObject<dyn MLFeatureProvider> =
            ProtocolObject::from_ref(&*provider);
        let output = unsafe { self.model.predictionFromFeatures_error(input_ref) }
            .map_err(|e| CoreMlError::PredictionFailed(format!("{e}")))?;

        let output_value = unsafe { output.featureValueForName(&self.output_key) }
            .ok_or_else(|| CoreMlError::OutputNotFound(self.output_name.clone()))?;
        let output_array = unsafe { output_value.multiArrayValue() }
            .ok_or_else(|| CoreMlError::OutputNotFound(self.output_name.clone()))?;

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
    ) -> Result<(Vec<f32>, Vec<usize>), CoreMlError> {
        let deallocator = RcBlock::new(|_ptr: NonNull<c_void>| {});
        let input_dict: Retained<NSMutableDictionary<NSString, AnyObject>> =
            NSMutableDictionary::new();

        for &(cached, data) in inputs {
            debug_assert_eq!(data.len(), cached.total_elements);
            let multi_array =
                create_multi_array_cached_with_deallocator(data, cached, &deallocator)?;
            let feature_value = unsafe { MLFeatureValue::featureValueWithMultiArray(&multi_array) };
            let key_copy: &ProtocolObject<dyn NSCopying> = ProtocolObject::from_ref(&*cached.name);
            let value_ref: &AnyObject =
                unsafe { &*((&*feature_value) as *const MLFeatureValue as *const AnyObject) };
            unsafe { input_dict.setObject_forKey(value_ref, key_copy) };
        }

        let provider = unsafe {
            MLDictionaryFeatureProvider::initWithDictionary_error(
                MLDictionaryFeatureProvider::alloc(),
                &input_dict,
            )
        }
        .map_err(|e| CoreMlError::PredictionFailed(format!("feature provider: {e}")))?;

        let input_ref: &ProtocolObject<dyn MLFeatureProvider> =
            ProtocolObject::from_ref(&*provider);

        // bridge the async callback to a blocking channel
        let (tx, rx) = std::sync::mpsc::sync_channel::<
            Result<Retained<ProtocolObject<dyn MLFeatureProvider>>, String>,
        >(1);

        let completion = block2::RcBlock::new(
            move |output: *mut ProtocolObject<dyn MLFeatureProvider>, error: *mut NSError| {
                if !error.is_null() {
                    let err_msg = unsafe { (*error).localizedDescription() }.to_string();
                    let _ = tx.send(Err(err_msg));
                } else if output.is_null() {
                    let _ = tx.send(Err("nil output with no error".to_owned()));
                } else {
                    // retain the output so it survives the callback
                    let retained = unsafe { Retained::retain(output) }.unwrap();
                    let _ = tx.send(Ok(retained));
                }
            },
        );

        unsafe {
            self.model
                .predictionFromFeatures_completionHandler(input_ref, &completion);
        }

        // block until the callback fires
        let output = rx
            .recv()
            .map_err(|_| CoreMlError::PredictionFailed("channel closed".to_owned()))?
            .map_err(CoreMlError::PredictionFailed)?;

        let output_value = unsafe { output.featureValueForName(&self.output_key) }
            .ok_or_else(|| CoreMlError::OutputNotFound(self.output_name.clone()))?;
        let output_array = unsafe { output_value.multiArrayValue() }
            .ok_or_else(|| CoreMlError::OutputNotFound(self.output_name.clone()))?;

        extract_output(&output_array)
    }
}

/// Resolve .mlmodelc path next to an .onnx file
pub(crate) fn coreml_model_path(onnx_path: &str) -> std::path::PathBuf {
    let path = Path::new(onnx_path);
    let stem = path.file_stem().unwrap().to_str().unwrap();
    path.with_file_name(format!("{stem}.mlmodelc"))
}

/// Resolve W8A16 .mlmodelc path, falling back to FP32 if not found
pub(crate) fn coreml_w8a16_model_path(onnx_path: &str) -> std::path::PathBuf {
    let path = Path::new(onnx_path);
    let stem = path.file_stem().unwrap().to_str().unwrap();
    let w8a16_path = path.with_file_name(format!("{stem}-w8a16.mlmodelc"));
    if w8a16_path.exists() {
        w8a16_path
    } else {
        coreml_model_path(onnx_path)
    }
}
