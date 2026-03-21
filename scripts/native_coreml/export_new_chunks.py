#!/usr/bin/env python3
"""Export only the new w86 and w112 chunk embedding models"""

from pathlib import Path
from common import (
    build_chunk_embedding_wrapper,
    coreml_packages_dir,
    load_pipeline,
    save_model_artifacts,
    CHUNK_STEM,
    FBANK_FEATURES,
    SEGMENTATION_FRAMES,
)
import coremltools as ct
import numpy as np
import torch


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

    # only the new large configs
    new_configs = [
        (86, 18000, 258, 25),  # ~172s
        (112, 23200, 336, 25),  # ~224s
    ]

    for num_windows, fbank_frames, num_masks, step_resnet in new_configs:
        print(
            f"\n=== Exporting w{num_windows} (fbank={fbank_frames}, masks={num_masks}) ==="
        )
        wrapper = build_chunk_embedding_wrapper(pipeline, num_windows, step_resnet)

        dummy_fbank = torch.zeros(1, fbank_frames, FBANK_FEATURES)
        dummy_masks = torch.zeros(num_masks, SEGMENTATION_FRAMES)

        exported = torch.export.export(
            wrapper, (dummy_fbank, dummy_masks), strict=False
        )
        exported = exported.run_decompositions({})

        stem = f"{CHUNK_STEM}-s{step_resnet}-w{num_windows}"
        compiled_paths = [output_dir / f"{stem}.mlmodelc"]

        mlmodel = ct.convert(
            exported,
            convert_to="mlprogram",
            inputs=[
                ct.TensorType(
                    name="fbank",
                    shape=(1, fbank_frames, FBANK_FEATURES),
                    dtype=np.float32,
                ),
                ct.TensorType(
                    name="masks",
                    shape=(num_masks, SEGMENTATION_FRAMES),
                    dtype=np.float32,
                ),
            ],
            outputs=[ct.TensorType(name="output", dtype=np.float32)],
            compute_units=ct.ComputeUnit.CPU_AND_GPU,
            minimum_deployment_target=deployment_target(),
            compute_precision=ct.precision.FLOAT32,
        )

        print(f"Saving {stem} (FP32)...")
        pkg_path = coreml_packages_dir(output_dir) / f"{stem}.mlpackage"
        save_model_artifacts(mlmodel, pkg_path, compiled_paths)

        # W8A16 variant
        mlmodel_w8a16 = quantize_w8a16(mlmodel)
        print(f"Saving {stem} (W8A16)...")
        w8a16_compiled = [output_dir / f"{stem}-w8a16.mlmodelc"]
        w8a16_pkg = coreml_packages_dir(output_dir) / f"{stem}-w8a16.mlpackage"
        save_model_artifacts(mlmodel_w8a16, w8a16_pkg, w8a16_compiled)

        print(f"Done: {stem}")


if __name__ == "__main__":
    main()
