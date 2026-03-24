# /// script
# requires-python = ">=3.10"
# dependencies = ["onnx", "onnxconverter-common"]
# ///
"""Convert ONNX models to FP16 with FP32 I/O"""

import sys
import onnx
from onnxconverter_common import float16


def convert(input_path: str, output_path: str) -> None:
    model = onnx.load(input_path)
    model_fp16 = float16.convert_float_to_float16(
        model,
        keep_io_types=True,
        min_positive_val=1e-7,
        max_finite_val=1e4,
    )
    onnx.save(model_fp16, output_path)


if __name__ == "__main__":
    for path in sys.argv[1:]:
        out = path.replace(".onnx", "-fp16.onnx")
        print(f"Converting {path} -> {out}")
        convert(path, out)
