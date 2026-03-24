# /// script
# requires-python = ">=3.10"
# dependencies = ["onnx", "onnxruntime"]
# ///
"""Quantize ONNX segmentation models to INT8 (dynamic quantization)"""

import sys
from pathlib import Path

import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic


def convert(input_path: str) -> str:
    p = Path(input_path)
    output_path = str(p.with_stem(p.stem + "-int8"))
    print(f"Quantizing {input_path} -> {output_path}")
    quantize_dynamic(
        input_path,
        output_path,
        weight_type=QuantType.QInt8,
    )
    return output_path


if __name__ == "__main__":
    for path in sys.argv[1:]:
        convert(path)
