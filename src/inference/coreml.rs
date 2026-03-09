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
}

impl CoreMlModel {
    /// Load a compiled .mlmodelc bundle
    pub fn load(
        path: &Path,
        compute_units: MLComputeUnits,
        output_name: &str,
    ) -> Result<Self, CoreMlError> {
        let path_str = NSString::from_str(&path.to_string_lossy());
        let url = NSURL::fileURLWithPath_isDirectory(&path_str, true);

        let config = unsafe { MLModelConfiguration::new() };
        unsafe { config.setComputeUnits(compute_units) };
        unsafe { config.setAllowLowPrecisionAccumulationOnGPU(true) };

        let model = unsafe { MLModel::modelWithContentsOfURL_configuration_error(&url, &config) }
            .map_err(|e| CoreMlError::LoadFailed(format!("{e}")))?;

        Ok(Self {
            model,
            output_name: output_name.to_owned(),
        })
    }

    /// Run prediction with named inputs
    ///
    /// Each input is (name, shape, flat_data). The data slice must outlive this call.
    pub fn predict(
        &self,
        inputs: &[(&str, &[usize], &[f32])],
    ) -> Result<(Vec<f32>, Vec<usize>), CoreMlError> {
        let dict: Retained<NSMutableDictionary<NSString, AnyObject>> = NSMutableDictionary::new();

        for &(name, shape, data) in inputs {
            let multi_array = create_multi_array(data, shape)?;
            let feature_value = unsafe { MLFeatureValue::featureValueWithMultiArray(&multi_array) };
            let key = NSString::from_str(name);
            let key_copy: &ProtocolObject<dyn NSCopying> = ProtocolObject::from_ref(&*key);
            // SAFETY: MLFeatureValue is an NSObject subclass, valid as AnyObject
            let value_ref: &AnyObject =
                unsafe { &*((&*feature_value) as *const MLFeatureValue as *const AnyObject) };
            unsafe { dict.setObject_forKey(value_ref, key_copy) };
        }

        let provider = unsafe {
            MLDictionaryFeatureProvider::initWithDictionary_error(
                MLDictionaryFeatureProvider::alloc(),
                &dict,
            )
        }
        .map_err(|e| CoreMlError::PredictionFailed(format!("feature provider: {e}")))?;

        let input_ref: &ProtocolObject<dyn MLFeatureProvider> =
            ProtocolObject::from_ref(&*provider);
        let output = unsafe { self.model.predictionFromFeatures_error(input_ref) }
            .map_err(|e| CoreMlError::PredictionFailed(format!("{e}")))?;

        let output_key = NSString::from_str(&self.output_name);
        let output_value = unsafe { output.featureValueForName(&output_key) }
            .ok_or_else(|| CoreMlError::OutputNotFound(self.output_name.clone()))?;
        let output_array = unsafe { output_value.multiArrayValue() }
            .ok_or_else(|| CoreMlError::OutputNotFound(self.output_name.clone()))?;

        extract_output(&output_array)
    }

    /// Run prediction with pre-cached shape/strides objects (avoids per-call NSNumber allocation)
    pub fn predict_cached(
        &self,
        inputs: &[(&CachedInputShape, &[f32])],
    ) -> Result<(Vec<f32>, Vec<usize>), CoreMlError> {
        let dict: Retained<NSMutableDictionary<NSString, AnyObject>> = NSMutableDictionary::new();

        for &(cached, data) in inputs {
            debug_assert_eq!(data.len(), cached.total_elements);
            let multi_array = create_multi_array_cached(data, cached)?;
            let feature_value = unsafe { MLFeatureValue::featureValueWithMultiArray(&multi_array) };
            let key_copy: &ProtocolObject<dyn NSCopying> = ProtocolObject::from_ref(&*cached.name);
            let value_ref: &AnyObject =
                unsafe { &*((&*feature_value) as *const MLFeatureValue as *const AnyObject) };
            unsafe { dict.setObject_forKey(value_ref, key_copy) };
        }

        let provider = unsafe {
            MLDictionaryFeatureProvider::initWithDictionary_error(
                MLDictionaryFeatureProvider::alloc(),
                &dict,
            )
        }
        .map_err(|e| CoreMlError::PredictionFailed(format!("feature provider: {e}")))?;

        let input_ref: &ProtocolObject<dyn MLFeatureProvider> =
            ProtocolObject::from_ref(&*provider);
        let output = unsafe { self.model.predictionFromFeatures_error(input_ref) }
            .map_err(|e| CoreMlError::PredictionFailed(format!("{e}")))?;

        let output_key = NSString::from_str(&self.output_name);
        let output_value = unsafe { output.featureValueForName(&output_key) }
            .ok_or_else(|| CoreMlError::OutputNotFound(self.output_name.clone()))?;
        let output_array = unsafe { output_value.multiArrayValue() }
            .ok_or_else(|| CoreMlError::OutputNotFound(self.output_name.clone()))?;

        extract_output(&output_array)
    }

    /// Default: CPU+GPU (FP32 models can't use ANE)
    /// Set SPEAKRS_COREML_ANE=1 to opt into CPUAndNeuralEngine (only useful with FP16 models)
    pub fn default_compute_units() -> MLComputeUnits {
        if std::env::var("SPEAKRS_COREML_ANE").is_ok() {
            MLComputeUnits(3) // CPUAndNeuralEngine
        } else {
            MLComputeUnits::CPUAndGPU
        }
    }

    /// Compute units for FP16 models: CPU+GPU+ANE
    pub fn fp16_compute_units() -> MLComputeUnits {
        MLComputeUnits(3) // CPUAndNeuralEngine
    }
}

/// Create an MLMultiArray wrapping a data pointer (zero-copy for the input side)
fn create_multi_array(
    data: &[f32],
    shape: &[usize],
) -> Result<Retained<MLMultiArray>, CoreMlError> {
    let shape_nums: Vec<Retained<NSNumber>> = shape
        .iter()
        .map(|&d| NSNumber::new_isize(d as isize))
        .collect();
    let ns_shape = NSArray::from_retained_slice(&shape_nums);

    // row-major strides
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

    // no-op deallocator — Rust owns the buffer and it outlives this synchronous call
    let deallocator = RcBlock::new(|_ptr: NonNull<c_void>| {});

    #[allow(deprecated)]
    unsafe {
        MLMultiArray::initWithDataPointer_shape_dataType_strides_deallocator_error(
            MLMultiArray::alloc(),
            ptr,
            &ns_shape,
            MLMultiArrayDataType::Float32,
            &ns_strides,
            Some(&deallocator),
        )
    }
    .map_err(|e| CoreMlError::ArrayCreationFailed(format!("{e}")))
}

/// Create an MLMultiArray using pre-cached shape and strides (zero-copy input)
fn create_multi_array_cached(
    data: &[f32],
    cached: &CachedInputShape,
) -> Result<Retained<MLMultiArray>, CoreMlError> {
    let ptr = NonNull::new(data.as_ptr() as *mut c_void)
        .ok_or_else(|| CoreMlError::ArrayCreationFailed("null data pointer".into()))?;

    let deallocator = RcBlock::new(|_ptr: NonNull<c_void>| {});

    #[allow(deprecated)]
    unsafe {
        MLMultiArray::initWithDataPointer_shape_dataType_strides_deallocator_error(
            MLMultiArray::alloc(),
            ptr,
            &cached.ns_shape,
            MLMultiArrayDataType::Float32,
            &cached.ns_strides,
            Some(&deallocator),
        )
    }
    .map_err(|e| CoreMlError::ArrayCreationFailed(format!("{e}")))
}

/// Copy output MLMultiArray data into a Vec<f32> and return the shape
#[allow(deprecated)]
fn extract_output(array: &MLMultiArray) -> Result<(Vec<f32>, Vec<usize>), CoreMlError> {
    let count = unsafe { array.count() } as usize;
    let ptr = unsafe { array.dataPointer() };
    let data = unsafe { std::slice::from_raw_parts(ptr.as_ptr() as *const f32, count) };

    let ns_shape = unsafe { array.shape() };
    let shape: Vec<usize> = (0..ns_shape.len())
        .map(|i| ns_shape.objectAtIndex(i).as_isize() as usize)
        .collect();

    Ok((data.to_vec(), shape))
}

/// Resolve .mlmodelc path next to an .onnx file
pub(crate) fn coreml_model_path(onnx_path: &str) -> std::path::PathBuf {
    let path = Path::new(onnx_path);
    let stem = path.file_stem().unwrap().to_str().unwrap();
    path.with_file_name(format!("{stem}.mlmodelc"))
}

/// Resolve FP16 .mlmodelc path (-f16 suffix) next to an .onnx file
pub(crate) fn coreml_model_path_f16(onnx_path: &str) -> std::path::PathBuf {
    let path = Path::new(onnx_path);
    let stem = path.file_stem().unwrap().to_str().unwrap();
    path.with_file_name(format!("{stem}-f16.mlmodelc"))
}
