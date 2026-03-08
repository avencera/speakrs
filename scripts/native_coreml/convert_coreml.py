#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import coremltools as ct
import numpy as np
import torch

from common import (
    FBANK_FEATURES,
    FBANK_FRAMES,
    SEGMENTATION_BATCHED_STEM,
    SEGMENTATION_BATCH_SIZES,
    SEGMENTATION_FRAMES,
    SEGMENTATION_SAMPLES,
    SEGMENTATION_STEM,
    TAIL_B32_STEM,
    TAIL_B3_STEM,
    TAIL_STEM,
    build_fbank_wrapper,
    build_tail_wrapper,
    load_pipeline,
    patch_sincnet_encoder_for_tracing,
    save_model_artifacts,
    segmentation_package_path,
    tail_package_path,
)


def deployment_target() -> object:
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
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=deployment_target(),
        compute_precision=ct.precision.FLOAT32,
    )

    print("Saving segmentation CoreML artifacts...")
    save_model_artifacts(
        mlmodel,
        segmentation_package_path(output_dir),
        [
            output_dir / f"{SEGMENTATION_STEM}.mlmodelc",
            output_dir / f"{SEGMENTATION_BATCHED_STEM}.mlmodelc",
        ],
    )


def export_tail(pipeline: Any, output_dir: Path) -> None:
    fbank_wrapper = build_fbank_wrapper()
    tail_wrapper = build_tail_wrapper(pipeline)

    dummy_fbank = fbank_wrapper(torch.randn(1, 1, SEGMENTATION_SAMPLES))
    dummy_weights = torch.ones(1, SEGMENTATION_FRAMES)

    with torch.inference_mode():
        traced = torch.jit.trace(tail_wrapper, (dummy_fbank, dummy_weights))

    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(
                name="fbank",
                shape=(ct.RangeDim(1, 32), FBANK_FRAMES, FBANK_FEATURES),
                dtype=np.float32,
            ),
            ct.TensorType(
                name="weights",
                shape=(ct.RangeDim(1, 32), SEGMENTATION_FRAMES),
                dtype=np.float32,
            ),
        ],
        outputs=[ct.TensorType(name="output", dtype=np.float32)],
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=deployment_target(),
        compute_precision=ct.precision.FLOAT32,
    )

    print("Saving embedding tail CoreML artifacts...")
    save_model_artifacts(
        mlmodel,
        tail_package_path(output_dir),
        [
            output_dir / f"{TAIL_STEM}.mlmodelc",
            output_dir / f"{TAIL_B3_STEM}.mlmodelc",
            output_dir / f"{TAIL_B32_STEM}.mlmodelc",
        ],
    )


def main() -> None:
    args = parse_args()
    pipeline = load_pipeline()
    export_segmentation(pipeline, args.output_dir)
    export_tail(pipeline, args.output_dir)
    print("CoreML conversion complete")


if __name__ == "__main__":
    main()
