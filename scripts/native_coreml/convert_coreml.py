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
    SEGMENTATION_BATCHED_STEM,
    SEGMENTATION_BATCH_SIZES,
    SEGMENTATION_FRAMES,
    SEGMENTATION_SAMPLES,
    SEGMENTATION_STEM,
    TAIL_B32_STEM,
    TAIL_B3_STEM,
    TAIL_BATCH_SIZES,
    TAIL_STEM,
    build_fbank_wrapper,
    build_fused_wrapper,
    build_tail_wrapper,
    fbank_package_path,
    fused_package_path,
    load_pipeline,
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


def main() -> None:
    args = parse_args()
    pipeline = load_pipeline()
    export_segmentation(pipeline, args.output_dir)
    export_tail(pipeline, args.output_dir)
    export_fbank(args.output_dir)
    export_fused_embedding(pipeline, args.output_dir)
    print("CoreML conversion complete")


if __name__ == "__main__":
    main()
