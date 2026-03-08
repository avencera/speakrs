#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import coremltools as ct
import numpy as np
import torch

from common import (
    SEGMENTATION_FRAMES,
    SEGMENTATION_SAMPLES,
    TAIL_BATCH_SIZES,
    build_fbank_wrapper,
    build_tail_wrapper,
    load_pipeline,
    segmentation_package_path,
    tail_package_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare PyTorch and CoreML outputs for the native CoreML converter",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("fixtures/models"),
        help="Directory containing the generated CoreML artifacts",
    )
    parser.add_argument(
        "--segmentation-atol",
        type=float,
        default=1e-4,
        help="Absolute tolerance for segmentation parity",
    )
    parser.add_argument(
        "--tail-atol",
        type=float,
        default=1.5e-2,
        help="Absolute tolerance for tail parity",
    )
    return parser.parse_args()


def compare_segmentation(
    pipeline: Any,
    output_dir: Path,
    atol: float,
) -> None:
    model = pipeline._segmentation.model
    model.eval()
    coreml_model = ct.models.MLModel(
        str(segmentation_package_path(output_dir)),
        compute_units=ct.ComputeUnit.ALL,
    )

    for batch_size in (1, 32):
        inputs = torch.randn(batch_size, 1, SEGMENTATION_SAMPLES, dtype=torch.float32)
        with torch.inference_mode():
            torch_output = model(inputs).detach().cpu().numpy()
        coreml_output = np.asarray(
            coreml_model.predict({"input": inputs.numpy()})["output"],
            dtype=np.float32,
        )
        max_abs = float(np.max(np.abs(torch_output - coreml_output)))
        print(f"segmentation batch={batch_size} max_abs={max_abs:.6e}")
        if max_abs > atol:
            raise SystemExit(
                f"segmentation parity failed for batch={batch_size}: {max_abs:.6e} > {atol:.6e}"
            )


def compare_tail(
    pipeline: Any,
    output_dir: Path,
    atol: float,
) -> None:
    fbank_wrapper = build_fbank_wrapper()
    tail_wrapper = build_tail_wrapper(pipeline)
    coreml_model = ct.models.MLModel(
        str(tail_package_path(output_dir)),
        compute_units=ct.ComputeUnit.ALL,
    )

    for batch_size in TAIL_BATCH_SIZES:
        waveform = torch.randn(batch_size, 1, SEGMENTATION_SAMPLES, dtype=torch.float32)
        weights = torch.rand(batch_size, SEGMENTATION_FRAMES, dtype=torch.float32)
        with torch.inference_mode():
            fbank = fbank_wrapper(waveform)
            torch_output = tail_wrapper(fbank, weights).detach().cpu().numpy()
        coreml_output = np.asarray(
            coreml_model.predict(
                {
                    "fbank": fbank.detach().cpu().numpy(),
                    "weights": weights.detach().cpu().numpy(),
                }
            )["output"],
            dtype=np.float32,
        )
        max_abs = float(np.max(np.abs(torch_output - coreml_output)))
        print(f"tail batch={batch_size} max_abs={max_abs:.6e}")
        if max_abs > atol:
            raise SystemExit(
                f"tail parity failed for batch={batch_size}: {max_abs:.6e} > {atol:.6e}"
            )


def main() -> None:
    args = parse_args()
    np.random.seed(0)
    torch.manual_seed(0)
    pipeline = load_pipeline()
    compare_segmentation(pipeline, args.output_dir, args.segmentation_atol)
    compare_tail(pipeline, args.output_dir, args.tail_atol)
    print("CoreML parity checks passed")


if __name__ == "__main__":
    main()
