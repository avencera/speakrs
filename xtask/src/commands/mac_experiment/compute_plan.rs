use std::path::{Path, PathBuf};
use std::time::Duration;

use block2::RcBlock;
use color_eyre::eyre::{Context, Result, ensure};
use objc2::{ClassType, runtime::ProtocolObject};
use objc2_core_ml::{
    MLCPUComputeDevice, MLComputeDeviceProtocol, MLComputePlan, MLComputePlanDeviceUsage,
    MLGPUComputeDevice, MLModelConfiguration, MLModelStructureProgramBlock,
    MLModelStructureProgramOperation, MLNeuralEngineComputeDevice,
};
use objc2_foundation::{NSError, NSObjectProtocol, NSString, NSURL};
use serde::Serialize;

use super::ValidatedExperiment;

#[derive(Serialize)]
struct ComputePlanReport {
    schema_version: u32,
    generated_at: String,
    models: Vec<ModelComputePlan>,
}

#[derive(Serialize)]
struct ModelComputePlan {
    model: PathBuf,
    error: Option<String>,
    operations: Vec<OperationPlacement>,
}

#[derive(Serialize)]
struct OperationPlacement {
    path: String,
    operator: String,
    estimated_weight: Option<f64>,
    preferred_device: Option<String>,
    supported_devices: Vec<String>,
}

pub(super) fn write_report(experiment: &ValidatedExperiment, output: &Path) -> Result<()> {
    let models = experiment
        .profiled_model_paths()
        .into_iter()
        .map(|model| match inspect_model(&model) {
            Ok(operations) => ModelComputePlan {
                model,
                error: None,
                operations,
            },
            Err(error) => ModelComputePlan {
                model,
                error: Some(format!("{error:#}")),
                operations: Vec::new(),
            },
        })
        .collect();
    let report = ComputePlanReport {
        schema_version: 1,
        generated_at: chrono::Utc::now().to_rfc3339(),
        models,
    };
    let mut bytes = serde_json::to_vec_pretty(&report)?;
    bytes.push(b'\n');
    super::store::atomic_write(output, &bytes)
}

fn inspect_model(path: &Path) -> Result<Vec<OperationPlacement>> {
    let path_text = NSString::from_str(path.to_string_lossy().as_ref());
    let url = NSURL::fileURLWithPath(&path_text);
    // safety: Core ML creates an owned configuration with no caller-provided pointer
    let configuration = unsafe { MLModelConfiguration::new() };
    // safety: the owned configuration is valid for this Objective-C property setter
    unsafe { configuration.setComputeUnits(objc2_core_ml::MLComputeUnits::All) };
    let (sender, receiver) = std::sync::mpsc::sync_channel(1);
    let completion = RcBlock::new(move |plan: *mut MLComputePlan, error: *mut NSError| {
        let result = if !error.is_null() {
            // safety: the callback supplied a non-null NSError pointer for this branch
            let message = unsafe { (*error).localizedDescription() }.to_string();
            Err(message)
        } else if plan.is_null() {
            Err("Core ML returned no compute plan and no error".to_owned())
        } else {
            // safety: the callback supplied a non-null MLComputePlan pointer for this branch
            inspect_loaded_plan(unsafe { &*plan })
        };
        let _ = sender.send(result);
    });

    // safety: the URL, configuration, and retained completion block remain live through callback
    unsafe {
        MLComputePlan::loadContentsOfURL_configuration_completionHandler(
            &url,
            &configuration,
            &completion,
        )
    };
    let operations = receiver
        .recv_timeout(Duration::from_secs(180))
        .wrap_err_with(|| format!("compute-plan load timed out for {}", path.display()))?
        .map_err(color_eyre::eyre::Report::msg)?;
    ensure!(
        !operations.is_empty(),
        "compute plan has no ML Program operations for {}",
        path.display()
    );
    Ok(operations)
}

fn inspect_loaded_plan(
    plan: &MLComputePlan,
) -> std::result::Result<Vec<OperationPlacement>, String> {
    // safety: `plan` is a live Core ML compute-plan object from the completion callback
    let structure = unsafe { plan.modelStructure() };
    // safety: `structure` is retained for this call and the method returns an owned result
    let program =
        unsafe { structure.program() }.ok_or_else(|| "model is not an ML Program".to_owned())?;
    // safety: `program` is live and the method returns an owned dictionary
    let functions = unsafe { program.functions() };
    let main_name = NSString::from_str("main");
    let main = functions
        .objectForKey(&main_name)
        .ok_or_else(|| "ML Program has no main function".to_owned())?;
    // safety: `main` is retained by `functions` and the method returns an owned block
    let block = unsafe { main.block() };
    let mut operations = Vec::new();
    append_block_operations(plan, &block, "main", &mut operations);
    Ok(operations)
}

fn append_block_operations(
    plan: &MLComputePlan,
    block: &MLModelStructureProgramBlock,
    block_path: &str,
    output: &mut Vec<OperationPlacement>,
) {
    // safety: `block` is live for the traversal and the method returns an owned array
    let operations = unsafe { block.operations() };
    for index in 0..operations.len() {
        let operation = operations.objectAtIndex(index);
        let path = format!("{block_path}/{index}");
        output.push(operation_placement(plan, &operation, path.clone()));
        // safety: `operation` is retained by `operations` and the method returns an owned array
        let blocks = unsafe { operation.blocks() };
        for nested_index in 0..blocks.len() {
            append_block_operations(
                plan,
                &blocks.objectAtIndex(nested_index),
                &format!("{path}/block-{nested_index}"),
                output,
            );
        }
    }
}

fn operation_placement(
    plan: &MLComputePlan,
    operation: &MLModelStructureProgramOperation,
    path: String,
) -> OperationPlacement {
    // safety: `plan` and `operation` come from the same live model structure
    let cost = unsafe { plan.estimatedCostOfMLProgramOperation(operation) };
    // safety: `plan` and `operation` come from the same live model structure
    let usage = unsafe { plan.computeDeviceUsageForMLProgramOperation(operation) };
    let (preferred_device, supported_devices) =
        usage.as_deref().map(device_usage).unwrap_or_default();
    // safety: `operation` is live and the method returns an owned string
    let operator = unsafe { operation.operatorName() }.to_string();
    let estimated_weight = cost.as_deref().map(|cost| {
        // safety: `cost` is retained by the local owned result
        unsafe { cost.weight() }
    });
    OperationPlacement {
        path,
        operator,
        estimated_weight,
        preferred_device,
        supported_devices,
    }
}

fn device_usage(usage: &MLComputePlanDeviceUsage) -> (Option<String>, Vec<String>) {
    // safety: `usage` is a live owned result from the compute plan
    let preferred = unsafe { usage.preferredComputeDevice() };
    // safety: `usage` is live and the method returns an owned device array
    let supported = unsafe { usage.supportedComputeDevices() };
    let supported = (0..supported.len())
        .map(|index| device_name(&supported.objectAtIndex(index)))
        .collect();
    (Some(device_name(&preferred)), supported)
}

fn device_name(device: &ProtocolObject<dyn MLComputeDeviceProtocol>) -> String {
    if device.isKindOfClass(MLCPUComputeDevice::class()) {
        "cpu".to_owned()
    } else if device.isKindOfClass(MLGPUComputeDevice::class()) {
        "gpu".to_owned()
    } else if device.isKindOfClass(MLNeuralEngineComputeDevice::class()) {
        "neural_engine".to_owned()
    } else {
        "unknown".to_owned()
    }
}
