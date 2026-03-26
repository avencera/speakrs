use std::path::Path;

use objc2::AnyThread;
use objc2::rc::Retained;
use objc2::runtime::{AnyObject, ProtocolObject};
use objc2_core_ml::{
    MLComputeUnits, MLDictionaryFeatureProvider, MLFeatureProvider, MLFeatureValue, MLModel,
    MLModelConfiguration, MLMultiArray,
};
use objc2_foundation::{NSCopying, NSMutableDictionary, NSString, NSURL};

use super::{CoreMlError, GpuPrecision};

pub(super) fn load_model(
    path: &Path,
    compute_units: MLComputeUnits,
    gpu_precision: GpuPrecision,
) -> Result<Retained<MLModel>, CoreMlError> {
    let path_str = NSString::from_str(&path.to_string_lossy());
    let url = NSURL::fileURLWithPath_isDirectory(&path_str, true);
    let low_precision = matches!(gpu_precision, GpuPrecision::Low);

    // SAFETY: objc2 marks CoreML object construction as unsafe, but the URL and configuration
    // SAFETY: objects are valid for the duration of this call and are only used synchronously here
    unsafe {
        let config = MLModelConfiguration::new();
        config.setComputeUnits(compute_units);
        config.setAllowLowPrecisionAccumulationOnGPU(low_precision);
        MLModel::modelWithContentsOfURL_configuration_error(&url, &config)
    }
    .map_err(|e| CoreMlError::LoadFailed(format!("{e}")))
}

pub(super) fn insert_input_feature(
    input_dict: &NSMutableDictionary<NSString, AnyObject>,
    key_copy: &ProtocolObject<dyn NSCopying>,
    multi_array: &MLMultiArray,
) {
    // SAFETY: multi_array is a live CoreML object for this prediction call, and setObject retains
    // SAFETY: the inserted feature value before the temporary Retained<MLFeatureValue> is dropped
    unsafe {
        let feature_value = MLFeatureValue::featureValueWithMultiArray(multi_array);
        input_dict.setObject_forKey(feature_value_as_any_object(&feature_value), key_copy);
    }
}

pub(super) fn build_feature_provider(
    input_dict: &NSMutableDictionary<NSString, AnyObject>,
) -> Result<Retained<MLDictionaryFeatureProvider>, CoreMlError> {
    // SAFETY: input_dict only contains NSString keys and MLFeatureValue-backed Objective-C objects
    unsafe {
        MLDictionaryFeatureProvider::initWithDictionary_error(
            MLDictionaryFeatureProvider::alloc(),
            input_dict,
        )
    }
    .map_err(|e| CoreMlError::PredictionFailed(format!("feature provider: {e}")))
}

pub(super) fn predict_output(
    model: &MLModel,
    input_ref: &ProtocolObject<dyn MLFeatureProvider>,
    output_key: &NSString,
    output_name: &str,
) -> Result<Retained<MLMultiArray>, CoreMlError> {
    // SAFETY: input_ref is a live feature provider constructed from valid CoreML objects and the
    // SAFETY: returned provider remains retained for all subsequent output lookups in this function
    let output = unsafe { model.predictionFromFeatures_error(input_ref) }
        .map_err(|e| CoreMlError::PredictionFailed(format!("{e}")))?;
    output_multi_array(&output, output_key, output_name)
}

pub(super) fn output_multi_array(
    output: &ProtocolObject<dyn MLFeatureProvider>,
    output_key: &NSString,
    output_name: &str,
) -> Result<Retained<MLMultiArray>, CoreMlError> {
    // SAFETY: output is a retained CoreML feature provider produced by a successful prediction call
    let output_value = unsafe { output.featureValueForName(output_key) }
        .ok_or_else(|| CoreMlError::OutputNotFound(output_name.to_owned()))?;
    // SAFETY: output_key names the declared tensor output for this model and CoreML keeps the array
    // SAFETY: alive as long as the owning feature provider is retained in this function
    unsafe { output_value.multiArrayValue() }
        .ok_or_else(|| CoreMlError::OutputNotFound(output_name.to_owned()))
}

fn feature_value_as_any_object(feature_value: &MLFeatureValue) -> &AnyObject {
    // SAFETY: MLFeatureValue is an Objective-C object, so it has the same pointer representation as
    // SAFETY: AnyObject and can be passed to NSDictionary APIs that erase the concrete class type
    unsafe { &*(feature_value as *const MLFeatureValue).cast::<AnyObject>() }
}
