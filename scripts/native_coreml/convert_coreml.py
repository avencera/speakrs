#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import coremltools as ct
import numpy as np
import torch

from common import (
    FBANK_BATCH_SIZES,
    FBANK_BATCHED_STEM,
    FBANK_FEATURES,
    FBANK_FRAMES,
    FBANK_STEM,
    FUSED_B32_STEM,
    FUSED_B3_STEM,
    FUSED_BATCH_SIZES,
    FUSED_STEM,
    MULTI_MASK_B32_STEM,
    MULTI_MASK_BATCH_SIZES,
    MULTI_MASK_STEM,
    NUM_SPEAKERS,
    SEGMENTATION_BATCHED_STEM,
    SEGMENTATION_BATCH_SIZES,
    SEGMENTATION_FRAMES,
    SEGMENTATION_SAMPLES,
    SEGMENTATION_STEM,
    TAIL_B32_STEM,
    TAIL_B3_STEM,
    TAIL_BATCH_SIZES,
    TAIL_STEM,
    CHUNK_CONFIGS_DEFAULT,
    CHUNK_CONFIGS_FAST,
    CHUNK_STEM,
    build_chunk_embedding_wrapper,
    build_fbank_wrapper,
    coreml_packages_dir,
    build_fused_wrapper,
    build_multi_mask_wrapper,
    build_tail_wrapper,
    chunk_embedding_package_path,
    fbank_package_path,
    fused_package_path,
    load_pipeline,
    multi_mask_package_path,
    patch_sincnet_encoder_for_tracing,
    save_model_artifacts,
    segmentation_package_path,
    tail_package_path,
)


def deployment_target() -> object:
    return getattr(ct.target, "macOS15", ct.target.iOS18)


def legacy_deployment_target() -> object:
    return getattr(ct.target, "macOS14", ct.target.iOS17)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert speakrs native CoreML models from pyannote community-1",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("fixtures/models"),
        help="Directory where compiled .mlmodelc bundles should be written",
    )
    return parser.parse_args()


def _f16_package_path(package_path: Path) -> Path:
    return package_path.with_name(package_path.stem + "-f16" + package_path.suffix)


def _f16_compiled_paths(compiled_paths: list[Path]) -> list[Path]:
    return [p.with_name(p.stem + "-f16" + p.suffix) for p in compiled_paths]


def _w8a16_package_path(package_path: Path) -> Path:
    return package_path.with_name(package_path.stem + "-w8a16" + package_path.suffix)


def _w8a16_compiled_paths(compiled_paths: list[Path]) -> list[Path]:
    return [p.with_name(p.stem + "-w8a16" + p.suffix) for p in compiled_paths]


def quantize_w8a16(mlmodel: ct.models.MLModel) -> ct.models.MLModel:
    """Quantize model weights to 8-bit with 16-bit activations (W8A16)"""
    from coremltools.optimize.coreml import (
        OpLinearQuantizerConfig,
        OptimizationConfig,
        linear_quantize_weights,
    )

    config = OptimizationConfig(
        global_config=OpLinearQuantizerConfig(
            mode="linear_symmetric",
            dtype="int8",
        )
    )
    return linear_quantize_weights(mlmodel, config)


def export_segmentation(pipeline: Any, output_dir: Path) -> None:
    segmentation_model = pipeline._segmentation.model
    segmentation_model.eval()
    patch_sincnet_encoder_for_tracing(segmentation_model)

    example = torch.zeros(
        32,
        1,
        SEGMENTATION_SAMPLES,
        dtype=torch.float32,
    )
    with torch.inference_mode():
        traced = segmentation_model.to_torchscript(
            example_inputs=example,
            method="trace",
        )

    seg_compiled_paths = [
        output_dir / f"{SEGMENTATION_STEM}.mlmodelc",
        output_dir / f"{SEGMENTATION_BATCHED_STEM}.mlmodelc",
    ]

    # FP32 — CPU+GPU optimized
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(
                name="input",
                shape=ct.EnumeratedShapes(
                    shapes=[
                        (batch_size, 1, SEGMENTATION_SAMPLES)
                        for batch_size in SEGMENTATION_BATCH_SIZES
                    ],
                    default=(32, 1, SEGMENTATION_SAMPLES),
                ),
                dtype=np.float32,
            )
        ],
        outputs=[ct.TensorType(name="output", dtype=np.float32)],
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
        minimum_deployment_target=deployment_target(),
        compute_precision=ct.precision.FLOAT32,
    )

    print("Saving segmentation CoreML artifacts (FP32)...")
    save_model_artifacts(
        mlmodel,
        segmentation_package_path(output_dir),
        seg_compiled_paths,
    )

    # FP16 — ANE-eligible
    mlmodel_f16 = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(
                name="input",
                shape=ct.EnumeratedShapes(
                    shapes=[
                        (batch_size, 1, SEGMENTATION_SAMPLES)
                        for batch_size in SEGMENTATION_BATCH_SIZES
                    ],
                    default=(32, 1, SEGMENTATION_SAMPLES),
                ),
                dtype=np.float32,
            )
        ],
        outputs=[ct.TensorType(name="output", dtype=np.float32)],
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=deployment_target(),
        compute_precision=ct.precision.FLOAT16,
    )

    print("Saving segmentation CoreML artifacts (FP16)...")
    save_model_artifacts(
        mlmodel_f16,
        _f16_package_path(segmentation_package_path(output_dir)),
        _f16_compiled_paths(seg_compiled_paths),
    )

    # W8A16 — quantized weights for faster CPU inference
    mlmodel_w8a16 = quantize_w8a16(mlmodel)
    print("Saving segmentation CoreML artifacts (W8A16)...")
    save_model_artifacts(
        mlmodel_w8a16,
        _w8a16_package_path(segmentation_package_path(output_dir)),
        _w8a16_compiled_paths(seg_compiled_paths),
    )


def export_tail(pipeline: Any, output_dir: Path) -> None:
    fbank_wrapper = build_fbank_wrapper()
    tail_wrapper = build_tail_wrapper(pipeline)

    dummy_fbank = fbank_wrapper(torch.randn(1, 1, SEGMENTATION_SAMPLES))
    dummy_weights = torch.ones(1, SEGMENTATION_FRAMES)

    with torch.inference_mode():
        traced = torch.jit.trace(tail_wrapper, (dummy_fbank, dummy_weights))

    tail_compiled_paths = [
        output_dir / f"{TAIL_STEM}.mlmodelc",
        output_dir / f"{TAIL_B3_STEM}.mlmodelc",
        output_dir / f"{TAIL_B32_STEM}.mlmodelc",
    ]

    # FP32 — CPU+GPU optimized, EnumeratedShapes for better GPU kernels
    # (multiple EnumeratedShapes inputs require iOS 18+ / macOS 15+)
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(
                name="fbank",
                shape=ct.EnumeratedShapes(
                    shapes=[
                        (bs, FBANK_FRAMES, FBANK_FEATURES) for bs in TAIL_BATCH_SIZES
                    ],
                    default=(32, FBANK_FRAMES, FBANK_FEATURES),
                ),
                dtype=np.float32,
            ),
            ct.TensorType(
                name="weights",
                shape=ct.EnumeratedShapes(
                    shapes=[(bs, SEGMENTATION_FRAMES) for bs in TAIL_BATCH_SIZES],
                    default=(32, SEGMENTATION_FRAMES),
                ),
                dtype=np.float32,
            ),
        ],
        outputs=[ct.TensorType(name="output", dtype=np.float32)],
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
        minimum_deployment_target=deployment_target(),
        compute_precision=ct.precision.FLOAT32,
    )

    print("Saving embedding tail CoreML artifacts (FP32)...")
    save_model_artifacts(
        mlmodel,
        tail_package_path(output_dir),
        tail_compiled_paths,
    )

    # FP16 — ANE-eligible
    mlmodel_f16 = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(
                name="fbank",
                shape=ct.EnumeratedShapes(
                    shapes=[
                        (bs, FBANK_FRAMES, FBANK_FEATURES) for bs in TAIL_BATCH_SIZES
                    ],
                    default=(32, FBANK_FRAMES, FBANK_FEATURES),
                ),
                dtype=np.float32,
            ),
            ct.TensorType(
                name="weights",
                shape=ct.EnumeratedShapes(
                    shapes=[(bs, SEGMENTATION_FRAMES) for bs in TAIL_BATCH_SIZES],
                    default=(32, SEGMENTATION_FRAMES),
                ),
                dtype=np.float32,
            ),
        ],
        outputs=[ct.TensorType(name="output", dtype=np.float32)],
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=deployment_target(),
        compute_precision=ct.precision.FLOAT16,
    )

    print("Saving embedding tail CoreML artifacts (FP16)...")
    save_model_artifacts(
        mlmodel_f16,
        _f16_package_path(tail_package_path(output_dir)),
        _f16_compiled_paths(tail_compiled_paths),
    )

    # W8A16 — quantized weights for faster inference
    mlmodel_w8a16 = quantize_w8a16(mlmodel)
    print("Saving embedding tail CoreML artifacts (W8A16)...")
    save_model_artifacts(
        mlmodel_w8a16,
        _w8a16_package_path(tail_package_path(output_dir)),
        _w8a16_compiled_paths(tail_compiled_paths),
    )


def export_fbank(output_dir: Path) -> None:
    fbank_wrapper = build_fbank_wrapper()
    dummy_waveform = torch.randn(32, 1, SEGMENTATION_SAMPLES)

    with torch.inference_mode():
        traced = torch.jit.trace(fbank_wrapper, dummy_waveform)

    fbank_compiled_paths = [
        output_dir / f"{FBANK_STEM}.mlmodelc",
        output_dir / f"{FBANK_BATCHED_STEM}.mlmodelc",
    ]

    # FP32 — CPU+GPU
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(
                name="waveform",
                shape=ct.EnumeratedShapes(
                    shapes=[(bs, 1, SEGMENTATION_SAMPLES) for bs in FBANK_BATCH_SIZES],
                    default=(32, 1, SEGMENTATION_SAMPLES),
                ),
                dtype=np.float32,
            )
        ],
        outputs=[ct.TensorType(name="output", dtype=np.float32)],
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
        minimum_deployment_target=deployment_target(),
        compute_precision=ct.precision.FLOAT32,
    )

    print("Saving fbank CoreML artifacts (FP32)...")
    save_model_artifacts(
        mlmodel,
        fbank_package_path(output_dir),
        fbank_compiled_paths,
    )

    # FP16
    mlmodel_f16 = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(
                name="waveform",
                shape=ct.EnumeratedShapes(
                    shapes=[(bs, 1, SEGMENTATION_SAMPLES) for bs in FBANK_BATCH_SIZES],
                    default=(32, 1, SEGMENTATION_SAMPLES),
                ),
                dtype=np.float32,
            )
        ],
        outputs=[ct.TensorType(name="output", dtype=np.float32)],
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
        minimum_deployment_target=deployment_target(),
        compute_precision=ct.precision.FLOAT16,
    )

    print("Saving fbank CoreML artifacts (FP16)...")
    save_model_artifacts(
        mlmodel_f16,
        _f16_package_path(fbank_package_path(output_dir)),
        _f16_compiled_paths(fbank_compiled_paths),
    )


def export_fused_embedding(pipeline: Any, output_dir: Path) -> None:
    fused_wrapper = build_fused_wrapper(pipeline)
    dummy_waveform = torch.randn(32, 1, SEGMENTATION_SAMPLES)
    dummy_weights = torch.ones(32, SEGMENTATION_FRAMES)

    with torch.inference_mode():
        traced = torch.jit.trace(fused_wrapper, (dummy_waveform, dummy_weights))

    fused_compiled_paths = [
        output_dir / f"{FUSED_STEM}.mlmodelc",
        output_dir / f"{FUSED_B3_STEM}.mlmodelc",
        output_dir / f"{FUSED_B32_STEM}.mlmodelc",
    ]

    # FP32 — CPU+GPU
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(
                name="waveform",
                shape=ct.EnumeratedShapes(
                    shapes=[(bs, 1, SEGMENTATION_SAMPLES) for bs in FUSED_BATCH_SIZES],
                    default=(32, 1, SEGMENTATION_SAMPLES),
                ),
                dtype=np.float32,
            ),
            ct.TensorType(
                name="weights",
                shape=ct.EnumeratedShapes(
                    shapes=[(bs, SEGMENTATION_FRAMES) for bs in FUSED_BATCH_SIZES],
                    default=(32, SEGMENTATION_FRAMES),
                ),
                dtype=np.float32,
            ),
        ],
        outputs=[ct.TensorType(name="output", dtype=np.float32)],
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
        minimum_deployment_target=deployment_target(),
        compute_precision=ct.precision.FLOAT32,
    )

    print("Saving fused embedding CoreML artifacts (FP32)...")
    save_model_artifacts(
        mlmodel,
        fused_package_path(output_dir),
        fused_compiled_paths,
    )

    # FP16
    mlmodel_f16 = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(
                name="waveform",
                shape=ct.EnumeratedShapes(
                    shapes=[(bs, 1, SEGMENTATION_SAMPLES) for bs in FUSED_BATCH_SIZES],
                    default=(32, 1, SEGMENTATION_SAMPLES),
                ),
                dtype=np.float32,
            ),
            ct.TensorType(
                name="weights",
                shape=ct.EnumeratedShapes(
                    shapes=[(bs, SEGMENTATION_FRAMES) for bs in FUSED_BATCH_SIZES],
                    default=(32, SEGMENTATION_FRAMES),
                ),
                dtype=np.float32,
            ),
        ],
        outputs=[ct.TensorType(name="output", dtype=np.float32)],
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
        minimum_deployment_target=deployment_target(),
        compute_precision=ct.precision.FLOAT16,
    )

    print("Saving fused embedding CoreML artifacts (FP16)...")
    save_model_artifacts(
        mlmodel_f16,
        _f16_package_path(fused_package_path(output_dir)),
        _f16_compiled_paths(fused_compiled_paths),
    )


def export_multi_mask_tail(pipeline: Any, output_dir: Path) -> None:
    multi_mask_wrapper = build_multi_mask_wrapper(pipeline)
    dummy_fbank = torch.zeros(32, FBANK_FRAMES, FBANK_FEATURES)
    dummy_masks = torch.ones(32 * NUM_SPEAKERS, SEGMENTATION_FRAMES)

    with torch.inference_mode():
        traced = torch.jit.trace(multi_mask_wrapper, (dummy_fbank, dummy_masks))

    compiled_paths = [
        output_dir / f"{MULTI_MASK_STEM}.mlmodelc",
        output_dir / f"{MULTI_MASK_B32_STEM}.mlmodelc",
    ]

    # fbank and masks dimensions are decoupled: fbank batch = B, masks batch = B*3
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(
                name="fbank",
                shape=ct.EnumeratedShapes(
                    shapes=[
                        (bs, FBANK_FRAMES, FBANK_FEATURES)
                        for bs in MULTI_MASK_BATCH_SIZES
                    ],
                    default=(32, FBANK_FRAMES, FBANK_FEATURES),
                ),
                dtype=np.float32,
            ),
            ct.TensorType(
                name="masks",
                shape=ct.EnumeratedShapes(
                    shapes=[
                        (bs * NUM_SPEAKERS, SEGMENTATION_FRAMES)
                        for bs in MULTI_MASK_BATCH_SIZES
                    ],
                    default=(32 * NUM_SPEAKERS, SEGMENTATION_FRAMES),
                ),
                dtype=np.float32,
            ),
        ],
        outputs=[ct.TensorType(name="output", dtype=np.float32)],
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
        minimum_deployment_target=deployment_target(),
        compute_precision=ct.precision.FLOAT32,
    )

    print("Saving multi-mask tail CoreML artifacts (FP32)...")
    save_model_artifacts(
        mlmodel,
        multi_mask_package_path(output_dir),
        compiled_paths,
    )

    # W8A16 — quantized weights for faster inference
    mlmodel_w8a16 = quantize_w8a16(mlmodel)
    print("Saving multi-mask tail CoreML artifacts (W8A16)...")
    save_model_artifacts(
        mlmodel_w8a16,
        _w8a16_package_path(multi_mask_package_path(output_dir)),
        _w8a16_compiled_paths(compiled_paths),
    )


def export_chunk_embedding(pipeline: Any, output_dir: Path) -> None:
    all_configs = CHUNK_CONFIGS_FAST + CHUNK_CONFIGS_DEFAULT

    for num_windows, fbank_frames, num_masks, step_resnet in all_configs:
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

        print(f"Saving chunk embedding s{step_resnet}-w{num_windows} (FP32)...")
        pkg_path = coreml_packages_dir(output_dir) / f"{stem}.mlpackage"
        save_model_artifacts(mlmodel, pkg_path, compiled_paths)

        # W8A16 variant
        mlmodel_w8a16 = quantize_w8a16(mlmodel)
        print(f"Saving chunk embedding s{step_resnet}-w{num_windows} (W8A16)...")
        w8a16_compiled = [output_dir / f"{stem}-w8a16.mlmodelc"]
        w8a16_pkg = coreml_packages_dir(output_dir) / f"{stem}-w8a16.mlpackage"
        save_model_artifacts(mlmodel_w8a16, w8a16_pkg, w8a16_compiled)


def main() -> None:
    args = parse_args()
    pipeline = load_pipeline()
    export_segmentation(pipeline, args.output_dir)
    export_tail(pipeline, args.output_dir)
    export_fbank(args.output_dir)
    export_fused_embedding(pipeline, args.output_dir)
    export_multi_mask_tail(pipeline, args.output_dir)
    export_chunk_embedding(pipeline, args.output_dir)
    print("CoreML conversion complete")


if __name__ == "__main__":
    main()
