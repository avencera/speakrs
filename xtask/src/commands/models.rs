use std::process::Command;

use color_eyre::eyre::Result;

use crate::cmd::{project_root, run_cmd};
use crate::python::{uv_run, uv_run_project};

pub fn export() -> Result<()> {
    uv_run(&["scripts/export_models.py", "fixtures/models"])?;

    if cfg!(target_os = "macos") {
        export_coreml()?;
    }

    Ok(())
}

pub fn export_coreml() -> Result<()> {
    uv_run_project(
        "scripts/native_coreml",
        "scripts/native_coreml/convert_coreml.py",
        &["--output-dir", "fixtures/models"],
    )
}

pub fn compare_coreml() -> Result<()> {
    uv_run_project(
        "scripts/native_coreml",
        "scripts/native_coreml/compare_coreml.py",
        &["--output-dir", "fixtures/models"],
    )
}

pub fn deploy() -> Result<()> {
    let root = project_root();

    let includes = [
        "plda_*.npy",
        "wespeaker-voxceleb-resnet34.min_num_samples.txt",
        "segmentation-3.0.onnx",
        "segmentation-3.0-b32.onnx",
        "wespeaker-voxceleb-resnet34.onnx",
        "wespeaker-voxceleb-resnet34.onnx.data",
        "wespeaker-fbank.onnx",
        "wespeaker-fbank-b32.onnx",
        "wespeaker-voxceleb-resnet34-tail.onnx",
        "wespeaker-voxceleb-resnet34-tail-b3.onnx",
        "wespeaker-voxceleb-resnet34-tail-b32.onnx",
        "segmentation-3.0.mlmodelc/**",
        "segmentation-3.0-b32.mlmodelc/**",
        "wespeaker-fbank.mlmodelc/**",
        "wespeaker-fbank-b32.mlmodelc/**",
        "wespeaker-voxceleb-resnet34-tail*.mlmodelc/**",
        "wespeaker-voxceleb-resnet34-fused*.mlmodelc/**",
    ];

    let mut cmd = Command::new("hf");
    cmd.args(["upload", "avencera/speakrs-models", "fixtures/models", "."]);
    for pattern in includes {
        cmd.args(["--include", pattern]);
    }
    cmd.current_dir(&root);

    run_cmd(&mut cmd)
}
