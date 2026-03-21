#!/usr/bin/env python3
"""Export a batch-64 segmentation CoreML model"""

from pathlib import Path
import coremltools as ct
import numpy as np
import torch
from common import (
    SEGMENTATION_SAMPLES,
    load_pipeline,
    patch_sincnet_encoder_for_tracing,
    save_model_artifacts,
    coreml_packages_dir,
)


def deployment_target():
    return ct.target.macOS13


def quantize_w8a16(mlmodel):
    from coremltools.optimize.coreml import (
        OpLinearQuantizerConfig,
        OptimizationConfig,
        linear_quantize_weights,
    )

    config = OptimizationConfig(
        global_config=OpLinearQuantizerConfig(
            mode="linear_symmetric", weight_threshold=512
        )
    )
    return linear_quantize_weights(mlmodel, config)


def main():
    output_dir = Path("fixtures/models")
    pipeline = load_pipeline()

    segmentation_model = pipeline._segmentation.model
    segmentation_model.eval()
    patch_sincnet_encoder_for_tracing(segmentation_model)

    # trace with batch=64
    example = torch.zeros(64, 1, SEGMENTATION_SAMPLES, dtype=torch.float32)
    with torch.inference_mode():
        traced = segmentation_model.to_torchscript(
            example_inputs=example, method="trace"
        )

    # fixed batch=64 (no enumeration — avoids LSTM shape compatibility issues)
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(
                name="input",
                shape=(64, 1, SEGMENTATION_SAMPLES),
                dtype=np.float32,
            )
        ],
        outputs=[ct.TensorType(name="output", dtype=np.float32)],
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
        minimum_deployment_target=deployment_target(),
        compute_precision=ct.precision.FLOAT32,
    )

    stem = "segmentation-3.0-b64"
    compiled = [output_dir / f"{stem}.mlmodelc"]
    pkg = coreml_packages_dir(output_dir) / f"{stem}.mlpackage"

    print(f"Saving {stem} (FP32)...")
    save_model_artifacts(mlmodel, pkg, compiled)

    # W8A16
    mlmodel_w8a16 = quantize_w8a16(mlmodel)
    w8a16_compiled = [output_dir / f"{stem}-w8a16.mlmodelc"]
    w8a16_pkg = coreml_packages_dir(output_dir) / f"{stem}-w8a16.mlpackage"
    print(f"Saving {stem} (W8A16)...")
    save_model_artifacts(mlmodel_w8a16, w8a16_pkg, w8a16_compiled)

    print("Done!")


if __name__ == "__main__":
    main()
