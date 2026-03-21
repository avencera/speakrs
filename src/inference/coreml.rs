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
use objc2_foundation::{NSArray, NSCopying, NSMutableDictionary, NSNumber, NSString, NSURL};

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
        let shape_nums: Vec<Retained<NSNumber>> = shape
            .iter()
            .map(|&d| NSNumber::new_isize(d as isize))
            .collect();
        let ns_shape = NSArray::from_retained_slice(&shape_nums);

        let mut strides = vec![1usize; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        let stride_nums: Vec<Retained<NSNumber>> = strides
            .iter()
            .map(|&s| NSNumber::new_isize(s as isize))
            .collect();
        let ns_strides = NSArray::from_retained_slice(&stride_nums);

        let total_elements = shape.iter().product();

        Self {
            name: NSString::from_str(name),
            ns_shape,
            ns_strides,
            total_elements,
        }
    }
}

pub(crate) struct CoreMlModel {
    model: Retained<MLModel>,
    output_name: String,
    output_key: Retained<NSString>,
    noop_deallocator: RcBlock<dyn Fn(NonNull<c_void>)>,
    input_dict: Retained<NSMutableDictionary<NSString, AnyObject>>,
}

// SAFETY: CoreMlModel is only used from one thread at a time (&mut self).
// MLModel.prediction is thread-safe per Apple docs, and the other fields
// (NSString, NSMutableDictionary, RcBlock) are only accessed during predict
// calls which require &mut self, preventing concurrent access
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

/// Create an MLMultiArray wrapping a data pointer with a shared no-op deallocator
fn create_multi_array_with_deallocator(
    data: &[f32],
    shape: &[usize],
    deallocator: &RcBlock<dyn Fn(NonNull<c_void>)>,
) -> Result<Retained<MLMultiArray>, CoreMlError> {
    let shape_nums: Vec<Retained<NSNumber>> = shape
        .iter()
        .map(|&d| NSNumber::new_isize(d as isize))
        .collect();
    let ns_shape = NSArray::from_retained_slice(&shape_nums);

    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    let stride_nums: Vec<Retained<NSNumber>> = strides
        .iter()
        .map(|&s| NSNumber::new_isize(s as isize))
        .collect();
    let ns_strides = NSArray::from_retained_slice(&stride_nums);

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
        fp16_data.iter().map(|&bits| f16_to_f32(bits)).collect()
    } else {
        let fp32_data = unsafe { std::slice::from_raw_parts(ptr.as_ptr() as *const f32, count) };
        fp32_data.to_vec()
    };

    Ok((data, shape))
}

/// Convert f32 to IEEE 754 half-precision (FP16) bits
fn f32_to_f16(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7fffff;

    if exp == 0xff {
        // inf/NaN
        return sign | 0x7c00 | ((mant >> 13) as u16 & 0x3ff);
    }

    let new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        return sign | 0x7c00; // overflow → inf
    }
    if new_exp <= 0 {
        if new_exp < -10 {
            return sign; // underflow → zero
        }
        let m = (mant | 0x800000) >> (1 - new_exp + 13);
        return sign | m as u16;
    }

    sign | ((new_exp as u16) << 10) | ((mant >> 13) as u16)
}

/// Create an FP16 MLMultiArray from f32 data (converts on the fly)
fn create_fp16_multi_array(
    data: &[f32],
    cached: &CachedInputShape,
    deallocator: &RcBlock<dyn Fn(NonNull<c_void>)>,
    fp16_buffer: &mut Vec<u16>,
) -> Result<Retained<MLMultiArray>, CoreMlError> {
    fp16_buffer.clear();
    fp16_buffer.extend(data.iter().map(|&v| f32_to_f16(v)));

    let ptr = NonNull::new(fp16_buffer.as_ptr() as *mut c_void)
        .ok_or_else(|| CoreMlError::ArrayCreationFailed("null data pointer".into()))?;

    #[allow(deprecated)]
    unsafe {
        MLMultiArray::initWithDataPointer_shape_dataType_strides_deallocator_error(
            MLMultiArray::alloc(),
            ptr,
            &cached.ns_shape,
            MLMultiArrayDataType::Float16,
            &cached.ns_strides,
            Some(deallocator),
        )
    }
    .map_err(|e| CoreMlError::ArrayCreationFailed(format!("{e}")))
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
/// Unlike CoreMlModel which reuses a cached NSMutableDictionary (requiring &mut self),
/// this allocates fresh input dicts per call, enabling &self predict methods.
/// Multiple threads can call predict concurrently on the same model instance
pub(crate) struct SharedCoreMlModel {
    model: Retained<MLModel>,
    output_name: String,
    output_key: Retained<NSString>,
    /// When true, this model was loaded without a specific output — use predict_cached_multi
    #[expect(dead_code)]
    multi_output: bool,
}

// SAFETY: MLModel.predictionFromFeatures is documented as thread-safe by Apple.
// All per-call mutable state (NSMutableDictionary, RcBlock deallocator) is allocated
// fresh in each predict call, preventing data races. Retained<NSString> is immutable
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
            multi_output: false,
        })
    }

    /// Load a model that returns multiple outputs (use predict_cached_multi)
    pub fn load_multi_output(
        path: &Path,
        compute_units: MLComputeUnits,
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
            output_key: NSString::from_str(""),
            output_name: String::new(),
            multi_output: true,
        })
    }

    /// Convert from an existing CoreMlModel, taking ownership of the MLModel
    #[expect(dead_code)]
    pub fn from_core_ml_model(model: CoreMlModel) -> Self {
        Self {
            model: model.model,
            output_key: model.output_key,
            output_name: model.output_name,
            multi_output: false,
        }
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

    /// Run prediction and extract multiple named outputs from a single inference call
    pub fn predict_cached_multi(
        &self,
        inputs: &[(&CachedInputShape, &[f32])],
        output_names: &[&str],
    ) -> Result<Vec<(Vec<f32>, Vec<usize>)>, CoreMlError> {
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

        let mut results = Vec::with_capacity(output_names.len());
        for &name in output_names {
            let ns_name = NSString::from_str(name);
            let output_value = unsafe { output.featureValueForName(&ns_name) }
                .ok_or_else(|| CoreMlError::OutputNotFound(name.to_owned()))?;
            let output_array = unsafe { output_value.multiArrayValue() }
                .ok_or_else(|| CoreMlError::OutputNotFound(name.to_owned()))?;
            results.push(extract_output(&output_array)?);
        }

        Ok(results)
    }

    /// Like predict_cached_multi but converts inputs to FP16 before feeding to CoreML
    pub fn predict_cached_multi_fp16(
        &self,
        inputs: &[(&CachedInputShape, &[f32])],
        output_names: &[&str],
    ) -> Result<Vec<(Vec<f32>, Vec<usize>)>, CoreMlError> {
        let deallocator = RcBlock::new(|_ptr: NonNull<c_void>| {});
        let input_dict: Retained<NSMutableDictionary<NSString, AnyObject>> =
            NSMutableDictionary::new();

        // keep FP16 buffers alive until after prediction
        let mut fp16_buffers: Vec<Vec<u16>> = Vec::with_capacity(inputs.len());

        for &(cached, data) in inputs {
            debug_assert_eq!(data.len(), cached.total_elements);
            fp16_buffers.push(Vec::with_capacity(data.len()));
            let buf = fp16_buffers.last_mut().unwrap();
            let multi_array = create_fp16_multi_array(data, cached, &deallocator, buf)?;
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

        let mut results = Vec::with_capacity(output_names.len());
        for &name in output_names {
            let ns_name = NSString::from_str(name);
            let output_value = unsafe { output.featureValueForName(&ns_name) }
                .ok_or_else(|| CoreMlError::OutputNotFound(name.to_owned()))?;
            let output_array = unsafe { output_value.multiArrayValue() }
                .ok_or_else(|| CoreMlError::OutputNotFound(name.to_owned()))?;
            results.push(extract_output(&output_array)?);
        }

        Ok(results)
    }

    /// Async prediction via CoreML's GCD dispatch — does NOT block the calling thread.
    /// Submits prediction to CoreML's internal dispatch queue and waits via channel.
    /// This allows multiple concurrent callers on the same model to pipeline
    /// GPU/ANE work, matching SpeakerKit's asyncPrediction pattern
    #[expect(dead_code)]
    pub fn predict_cached_async(
        &self,
        inputs: &[(&CachedInputShape, &[f32])],
    ) -> Result<(Vec<f32>, Vec<usize>), CoreMlError> {
        // allocate owned input data that lives until completion handler fires
        let owned_inputs: Vec<(Vec<f32>, &CachedInputShape)> = inputs
            .iter()
            .map(|&(cached, data)| (data.to_vec(), cached))
            .collect();

        let deallocator = RcBlock::new(|_ptr: NonNull<c_void>| {});
        let input_dict: Retained<NSMutableDictionary<NSString, AnyObject>> =
            NSMutableDictionary::new();

        for (data, cached) in &owned_inputs {
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

        // channel to receive result from GCD completion handler
        let (result_tx, result_rx) =
            crossbeam_channel::bounded::<Result<(Vec<f32>, Vec<usize>), CoreMlError>>(1);

        let output_name = self.output_name.clone();
        let output_key = self.output_key.clone();

        let completion = RcBlock::new(
            move |output_ptr: *mut ProtocolObject<dyn MLFeatureProvider>,
                  error_ptr: *mut objc2_foundation::NSError| {
                let result = if !error_ptr.is_null() {
                    let error = unsafe { &*error_ptr };
                    Err(CoreMlError::PredictionFailed(format!("{error}")))
                } else if output_ptr.is_null() {
                    Err(CoreMlError::PredictionFailed("null output".into()))
                } else {
                    let output = unsafe { &*output_ptr };
                    let output_value = unsafe { output.featureValueForName(&output_key) };
                    match output_value {
                        Some(val) => {
                            let output_array = unsafe { val.multiArrayValue() };
                            match output_array {
                                Some(arr) => extract_output(&arr),
                                None => Err(CoreMlError::OutputNotFound(output_name.clone())),
                            }
                        }
                        None => Err(CoreMlError::OutputNotFound(output_name.clone())),
                    }
                };
                let _ = result_tx.send(result);
            },
        );

        // submit to CoreML's GCD dispatch queue — returns immediately
        unsafe {
            self.model
                .predictionFromFeatures_completionHandler(input_ref, &completion);
        }

        // wait for completion (thread is free for other work while waiting)
        result_rx
            .recv()
            .map_err(|_| CoreMlError::PredictionFailed("channel disconnected".into()))?
    }

    /// Submit prediction to CoreML's GCD queue, return a Send receiver.
    /// All ObjC objects are created and dropped synchronously. Only the
    /// oneshot Receiver crosses the await point, making the Future Send
    #[cfg(feature = "tokio")]
    pub fn submit_prediction(
        &self,
        inputs: &[(Vec<f32>, CachedInputShapeRef)],
    ) -> Result<
        tokio::sync::oneshot::Receiver<Result<(Vec<f32>, Vec<usize>), CoreMlError>>,
        CoreMlError,
    > {
        let deallocator = RcBlock::new(|_ptr: NonNull<c_void>| {});
        let input_dict: Retained<NSMutableDictionary<NSString, AnyObject>> =
            NSMutableDictionary::new();

        for (data, cached) in inputs {
            debug_assert_eq!(data.len(), cached.total_elements);
            let multi_array =
                create_multi_array_cached_with_deallocator_ref(data, cached, &deallocator)?;
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

        let (result_tx, result_rx) =
            tokio::sync::oneshot::channel::<Result<(Vec<f32>, Vec<usize>), CoreMlError>>();

        let output_name = self.output_name.clone();
        let output_key = self.output_key.clone();
        let result_tx = std::sync::Mutex::new(Some(result_tx));

        let completion = RcBlock::new(
            move |output_ptr: *mut ProtocolObject<dyn MLFeatureProvider>,
                  error_ptr: *mut objc2_foundation::NSError| {
                let result = if !error_ptr.is_null() {
                    let error = unsafe { &*error_ptr };
                    Err(CoreMlError::PredictionFailed(format!("{error}")))
                } else if output_ptr.is_null() {
                    Err(CoreMlError::PredictionFailed("null output".into()))
                } else {
                    let output = unsafe { &*output_ptr };
                    let output_value = unsafe { output.featureValueForName(&output_key) };
                    match output_value {
                        Some(val) => {
                            let output_array = unsafe { val.multiArrayValue() };
                            match output_array {
                                Some(arr) => extract_output(&arr),
                                None => Err(CoreMlError::OutputNotFound(output_name.clone())),
                            }
                        }
                        None => Err(CoreMlError::OutputNotFound(output_name.clone())),
                    }
                };
                if let Some(tx) = result_tx.lock().unwrap().take() {
                    let _ = tx.send(result);
                }
            },
        );

        unsafe {
            self.model
                .predictionFromFeatures_completionHandler(input_ref, &completion);
        }

        // ObjC objects (input_dict, provider, completion, deallocator) drop here
        // CoreML retains the completion block internally
        Ok(result_rx)
    }
}

/// Owned copy of CachedInputShape data for async predictions (must be 'static)
#[cfg(feature = "tokio")]
#[derive(Clone)]
pub(crate) struct CachedInputShapeRef {
    pub name: Retained<NSString>,
    pub ns_shape: Retained<NSArray<NSNumber>>,
    pub ns_strides: Retained<NSArray<NSNumber>>,
    pub total_elements: usize,
}

// SAFETY: CachedInputShapeRef holds only immutable NSString and NSArray<NSNumber>,
// both documented as thread-safe by Apple. The Retained handles are reference-counted
// and never mutated after construction
#[cfg(feature = "tokio")]
unsafe impl Send for CachedInputShapeRef {}
#[cfg(feature = "tokio")]
unsafe impl Sync for CachedInputShapeRef {}

#[cfg(feature = "tokio")]
impl CachedInputShapeRef {
    /// Build directly from a name and shape, mirroring CachedInputShape::new
    pub fn from_shape(name: &str, shape: &[usize]) -> Self {
        let shape_nums: Vec<Retained<NSNumber>> = shape
            .iter()
            .map(|&d| NSNumber::new_isize(d as isize))
            .collect();
        let ns_shape = NSArray::from_retained_slice(&shape_nums);

        let mut strides = vec![1usize; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        let stride_nums: Vec<Retained<NSNumber>> = strides
            .iter()
            .map(|&s| NSNumber::new_isize(s as isize))
            .collect();
        let ns_strides = NSArray::from_retained_slice(&stride_nums);

        let total_elements = shape.iter().product();

        Self {
            name: NSString::from_str(name),
            ns_shape,
            ns_strides,
            total_elements,
        }
    }

    #[expect(dead_code)]
    pub fn from_cached(cached: &CachedInputShape) -> Self {
        Self {
            name: cached.name.clone(),
            ns_shape: cached.ns_shape.clone(),
            ns_strides: cached.ns_strides.clone(),
            total_elements: cached.total_elements,
        }
    }
}

#[cfg(feature = "tokio")]
fn create_multi_array_cached_with_deallocator_ref(
    data: &[f32],
    cached: &CachedInputShapeRef,
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
