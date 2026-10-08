# /// script
# requires-python = ">=3.12"
# dependencies = ["onnx", "numpy", "safetensors"]
# ///
"""Export the native CUDA backend's weights, or the pinned graphs, initializers, and manifests.

`--runtime-assets DIR` writes the files the CUDA modes load, plus the PLDA and embedding
metadata files they share with the other modes. Without it the script writes the reference
weights and graph manifests that the parity tests and reference tensors use.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[2]
CACHE = Path.home() / "Library/Caches/speakrs-cuda-ref"
MODEL_STEMS = ("segmentation-3.0", "wespeaker-fbank", "wespeaker-multimask-tail")
# ONNX model each CUDA runtime asset is exported from, and the asset file name
RUNTIME_ASSETS = {
    "segmentation-3.0": "segmentation-3.0.safetensors",
    "wespeaker-multimask-tail": "wespeaker-multimask-tail.safetensors",
}
# files the CUDA modes share with every other mode
SHARED_FILES = (
    "plda_lda.npy",
    "plda_tr.npy",
    "plda_mu.npy",
    "plda_psi.npy",
    "plda_mean1.npy",
    "plda_mean2.npy",
    "wespeaker-voxceleb-resnet34.min_num_samples.txt",
)
DTYPES = {
    "float32": "F32",
    "float64": "F64",
    "float16": "F16",
    "int64": "I64",
    "int32": "I32",
    "int16": "I16",
    "int8": "I8",
    "uint64": "U64",
    "uint32": "U32",
    "uint16": "U16",
    "uint8": "U8",
    "bool": "BOOL",
}


def write_json(path: Path, value: Any) -> None:
    """Write stable JSON without timestamps or machine-specific paths."""
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def sha256(path: Path) -> str:
    """Hash a file without loading the file into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_tensors(path: Path, tensors: dict[str, np.ndarray]) -> None:
    """Write and validate safetensors without a second full tensor-data copy."""
    # batch-32 tail activations exceed 12 GiB, so stream buffers in stable key order
    header = {}
    offset = 0
    for name, value in sorted(tensors.items()):
        if not value.flags.c_contiguous or not value.dtype.isnative:
            raise ValueError(f"non-contiguous or non-native tensor: {name}")
        header[name] = {
            "dtype": DTYPES[value.dtype.name],
            "shape": list(value.shape),
            "data_offsets": [offset, offset + value.nbytes],
        }
        offset += value.nbytes

    encoded = json.dumps(header, separators=(",", ":"), sort_keys=True).encode()
    encoded += b" " * (-len(encoded) % 8)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(struct.pack("<Q", len(encoded)))
        stream.write(encoded)
        for name in sorted(tensors):
            stream.write(memoryview(tensors[name].reshape(-1)).cast("B"))

    with safe_open(temporary, framework="numpy") as saved:
        if sorted(saved.keys()) != sorted(tensors):
            raise ValueError("safetensors key validation failed")
        for name, value in tensors.items():
            if saved.get_slice(name).get_shape() != list(value.shape):
                raise ValueError(f"safetensors shape validation failed: {name}")

    temporary.replace(path)


def revision() -> str:
    """Read the model revision from the Rust asset owner."""
    import re

    match = re.search(
        r'const HF_REVISION: &str = "([a-f0-9]+)";',
        (ROOT / "src/models.rs").read_text(),
    )
    if match is None:
        raise ValueError("HF_REVISION was not found in src/models.rs")
    return match[1]


def ensure_model(stem: str, output: Path) -> Path:
    """Copy a fixture model or download it anonymously at the pinned revision."""
    destination = output / "models" / f"{stem}.onnx"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        return destination
    fixture = ROOT / "fixtures/models" / destination.name
    if fixture.is_file():
        shutil.copyfile(fixture, destination)
        return destination
    url = f"https://huggingface.co/avencera/speakrs-models/resolve/{revision()}/{destination.name}"
    temporary = destination.with_suffix(".download")
    print(f"download {destination.name}", flush=True)
    urllib.request.urlretrieve(url, temporary)
    onnx.checker.check_model(str(temporary))
    temporary.replace(destination)
    return destination


def tensor_info(value: onnx.ValueInfoProto) -> dict[str, Any]:
    """Describe a graph tensor, keeping symbolic or unknown dimensions."""
    tensor = value.type.tensor_type
    shape = None
    if tensor.HasField("shape"):
        shape = [
            dim.dim_value if dim.HasField("dim_value") else dim.dim_param or None
            for dim in tensor.shape.dim
        ]
    return {
        "name": value.name,
        "dtype": TensorProto.DataType.Name(tensor.elem_type),
        "shape": shape,
    }


def initializer_info(value: onnx.TensorProto) -> dict[str, Any]:
    """Describe native initializer types and any lossy FP32 conversions."""
    array = numpy_helper.to_array(value)
    converted = array.astype(np.float32)
    exact = np.array_equal(array, converted.astype(array.dtype))
    result = {
        "name": value.name,
        "shape": list(array.shape),
        "source_dtype": str(array.dtype),
        "stored_dtype": "float32",
        "bytes": converted.nbytes,
        "conversion_exact": bool(exact),
    }
    if array.dtype.kind in "biu":
        # shape and index constants must be restored to their native integer type
        result["native_values"] = array.tolist()
    return result


def graph_info(
    graph: onnx.GraphProto, inherited: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Describe nodes and nested graphs in their checked topological order."""
    known = dict(inherited or {})
    known.update(
        {
            v.name: tensor_info(v)
            for v in [*graph.input, *graph.value_info, *graph.output]
        }
    )
    initializers = [initializer_info(value) for value in graph.initializer]
    known.update({item["name"]: item for item in initializers})
    nodes = []
    for index, node in enumerate(graph.node):
        attributes = {}
        for attribute in node.attribute:
            value = helper.get_attribute_value(attribute)
            if attribute.type == AttributeProto.GRAPH:
                value = graph_info(value, known)
            elif attribute.type == AttributeProto.TENSOR:
                array = numpy_helper.to_array(value)
                value = {
                    "dtype": str(array.dtype),
                    "shape": list(array.shape),
                    "values": array.tolist(),
                }
            elif isinstance(value, bytes):
                value = value.decode()
            attributes[attribute.name] = value
        nodes.append(
            {
                "index": index,
                "name": node.name or f"node_{index}",
                "op_type": node.op_type,
                "domain": node.domain,
                "inputs": [
                    known.get(name, {"name": name, "shape": None, "dtype": None})
                    for name in node.input
                ],
                "outputs": [
                    known.get(name, {"name": name, "shape": None, "dtype": None})
                    for name in node.output
                ],
                "initializers": [
                    name
                    for name in node.input
                    if known.get(name, {}).get("stored_dtype")
                ],
                "attributes": attributes,
            }
        )
    return {
        "name": graph.name,
        "inputs": [tensor_info(value) for value in graph.input],
        "outputs": [tensor_info(value) for value in graph.output],
        "initializers": initializers,
        "nodes": nodes,
    }


def fold_batch_norm(
    model: onnx.ModelProto, tensors: dict[str, np.ndarray]
) -> list[dict[str, Any]]:
    """Add FP32 Conv-BatchNormalization folds without changing original weights."""
    producers = {output: node for node in model.graph.node for output in node.output}
    folds = []
    for node in model.graph.node:
        if node.op_type != "BatchNormalization":
            continue
        conv = producers.get(node.input[0])
        if conv is None or conv.op_type != "Conv":
            raise ValueError(f"batch normalization is not after Conv: {node.name}")
        names = [conv.input[1], *node.input[1:5]]
        if not all(name in tensors for name in names):
            raise ValueError(
                f"batch normalization parameters are not initializers: {node.name}"
            )
        attributes = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        if attributes.get("training_mode", 0):
            raise ValueError("training-mode batch normalization cannot be folded")
        weight, gamma, beta, mean, variance = (tensors[name] for name in names)
        bias = (
            tensors[conv.input[2]]
            if len(conv.input) > 2 and conv.input[2]
            else np.zeros_like(mean)
        )
        epsilon = attributes.get("epsilon", 1e-5)
        scale = gamma / np.sqrt(variance + np.float32(epsilon))
        base = node.name or node.output[0]
        folded_weight = f"{base}.folded_weight"
        folded_bias = f"{base}.folded_bias"
        if folded_weight in tensors or folded_bias in tensors:
            raise ValueError(f"folded name collision: {base}")
        tensors[folded_weight] = weight * scale.reshape(
            (-1,) + (1,) * (weight.ndim - 1)
        )
        tensors[folded_bias] = (bias - mean) * scale + beta
        folds.append(
            {
                "conv": conv.name,
                "batch_norm": node.name,
                "epsilon": epsilon,
                "weight": folded_weight,
                "bias": folded_bias,
                "formula": "s=gamma/sqrt(variance+epsilon); W'=W*s; b'=(b-mean)*s+beta",
            }
        )
    return folds


def architecture(stem: str, graph: dict[str, Any]) -> dict[str, Any]:
    """Summarize these three graph families and list layer details."""
    nodes = graph["nodes"]
    if stem.startswith("segmentation"):
        summary = (
            "PyanNet: waveform instance normalization; learned SincNet bandpass filters "
            "(80 channels, kernel 251, stride 10); Conv1d 80->60->60 (kernel 5, stride 1); "
            "three max pools (kernel/stride 3), instance normalization and LeakyReLU; "
            "four bidirectional LSTMs (128 hidden units per direction); "
            "linear 256->128->128->7 with LeakyReLU; LogSoftmax powerset scores, 589 frames"
        )
        batch_norm = "No BatchNormalization nodes; input-dependent InstanceNormalization cannot be folded"
    elif stem.startswith("wespeaker-fbank"):
        summary = (
            "16 kHz Kaldi-style fbank: waveform*32768; 400-sample frames, 160-sample hop, "
            "998 frames, frame mean removal, replicate-edge preemphasis 0.97, symmetric "
            "Hamming window, zero pad to 512, one-sided DFT, squared magnitude, "
            "80 mel bins (20 Hz to Nyquist), log with float32 epsilon floor, temporal CMN"
        )
        batch_norm = "No batch normalization or convolution"
    else:
        summary = (
            "WeSpeaker ResNet34: NCHW input [B,1,80,998], stem 32 channels; residual stages "
            "[3,4,6,3] at [32,64,128,256] channels with spatial stride 2 at stage transitions; "
            "36 Conv nodes including three 1x1 shortcuts; trunk [B,256,10,125]; "
            "flatten to [B,2560,125] and repeat each chunk for three speakers; nearest resize "
            "589->125 masks; weighted mean and unbiased weighted standard deviation; "
            "zero masks produce [mean=0,std=1e-5]; Gemm 5120->256, no output normalization"
        )
        batch_norm = (
            "Already folded: no BatchNormalization nodes; each Conv has a weight and "
            "weight_bias initializer; scripts/export_models.py exports an eval-mode ResNet"
        )
    layers = [
        node
        for node in nodes
        if node["op_type"]
        in {
            "Conv",
            "LSTM",
            "MatMul",
            "Gemm",
            "MaxPool",
            "InstanceNormalization",
            "BatchNormalization",
        }
    ]
    pooling_start = next(
        (i for i, n in enumerate(nodes) if n["op_type"] == "Resize"), None
    )
    pooling = [] if pooling_start is None else nodes[pooling_start - 5 :]
    return {
        "summary": summary,
        "batch_normalization": batch_norm,
        "layers": layers,
        "multi_mask_pooling_nodes": pooling,
    }


def add_observed_shapes(graph: dict[str, Any], reference: dict[str, Any]) -> None:
    """Add concrete ten-second shapes from a completed reference index."""
    main = {
        item["onnx_name"]: item
        for item in reference["tensors"].values()
        if item.get("graph_scope", "main") == "main"
    }
    for node in graph["nodes"]:
        for value in [*node["inputs"], *node["outputs"]]:
            if value["name"] in main:
                value["observed_shape"] = main[value["name"]]["shape"]
                value["observed_dtype"] = main[value["name"]]["dtype"]
        for attribute in node["attributes"].values():
            if not isinstance(attribute, dict) or "nodes" not in attribute:
                continue
            for child in attribute["nodes"]:
                for value in [*child["inputs"], *child["outputs"]]:
                    scoped = [
                        item
                        for item in reference["tensors"].values()
                        if item.get("graph_scope", "").startswith(node["name"] + "/")
                        and item["onnx_name"] == value["name"]
                    ]
                    observed = scoped[0] if scoped else main.get(value["name"])
                    if observed:
                        value["observed_shape"] = observed["shape"]
                        value["observed_dtype"] = observed["dtype"]


def export(stem: str, output: Path) -> dict[str, Any]:
    """Export one model and return its size and hash summary."""
    path = ensure_model(stem, output)
    model = onnx.load(path)
    onnx.checker.check_model(model)
    inferred = onnx.shape_inference.infer_shapes(model, data_prop=True)
    graph = graph_info(inferred.graph)
    tensors = {
        value.name: np.array(numpy_helper.to_array(value), dtype=np.float32, copy=True)
        for value in model.graph.initializer
    }
    folds = fold_batch_norm(model, tensors)
    directory = output / stem
    directory.mkdir(parents=True, exist_ok=True)
    reference_indices = sorted(directory.glob("*.index.json"))
    shape_source = None
    if reference_indices:
        reference = json.loads(reference_indices[0].read_text())
        if reference["source_model_sha256"] != sha256(path):
            raise ValueError(f"reference index uses different model bytes: {stem}")
        add_observed_shapes(graph, reference)
        shape_source = reference_indices[0].name
    weights = directory / f"{stem}.safetensors"
    save_tensors(weights, tensors)
    manifest = {
        "schema_version": 1,
        "model": stem,
        "hf_revision": revision(),
        "source_sha256": sha256(path),
        "opsets": {
            item.domain or "ai.onnx": item.version for item in model.opset_import
        },
        "graph": graph,
        "observed_shapes_source": shape_source,
        "architecture": architecture(stem, graph),
        "batch_norm_folds": folds,
        "op_counts": dict(Counter(node.op_type for node in model.graph.node)),
        "weight_tensor_count": len(tensors),
        "weight_data_bytes": sum(value.nbytes for value in tensors.values()),
        "weights_sha256": sha256(weights),
        "notes": [
            "Original ONNX initializer names are stable safetensors keys; all stored as FP32",
            "Native integer values are in the manifest; FP32 index constants are not executable ONNX inputs",
            "Constant-node tensors remain in attributes; generated SincNet filters are reference intermediates",
            "Nested If graphs are included in attributes; node order follows the checked ONNX graph",
        ],
    }
    write_json(directory / f"{stem}.manifest.json", manifest)
    result = {
        key: manifest[key]
        for key in [
            "model",
            "weight_tensor_count",
            "weight_data_bytes",
            "weights_sha256",
        ]
    }
    print(json.dumps(result), flush=True)
    return result


def runtime_tensors(stem: str, model: onnx.ModelProto) -> dict[str, np.ndarray]:
    """Select the FP32 initializers the Rust loader reads, keyed by their ONNX names."""
    if any(node.op_type == "BatchNormalization" for node in model.graph.node):
        # the ResNet loader expects folded `<conv>.weight` and `<conv>.weight_bias`
        raise ValueError(f"{stem} has unfolded batch normalization")
    tensors = {}
    for value in model.graph.initializer:
        array = numpy_helper.to_array(value)
        if array.dtype != np.float32:
            continue
        # the multi-mask tail also stores shape and pooling constants; only the
        # ResNet weights are model parameters
        if stem == "wespeaker-multimask-tail" and not value.name.startswith("resnet."):
            continue
        tensors[value.name] = np.ascontiguousarray(array)
    return tensors


def ensure_shared_file(name: str, output: Path) -> None:
    """Copy a PLDA or metadata file from the fixtures, or download it anonymously."""
    destination = output / name
    if destination.is_file():
        return
    fixture = ROOT / "fixtures/models" / name
    if fixture.is_file():
        shutil.copyfile(fixture, destination)
        return
    url = f"https://huggingface.co/avencera/speakrs-models/resolve/{revision()}/{name}"
    temporary = destination.with_suffix(destination.suffix + ".download")
    print(f"download {name}", flush=True)
    urllib.request.urlretrieve(url, temporary)
    temporary.replace(destination)


def export_runtime_assets(output: Path, cache: Path) -> None:
    """Write every file the CUDA modes load into `output`."""
    output.mkdir(parents=True, exist_ok=True)
    for stem, file_name in RUNTIME_ASSETS.items():
        model = onnx.load(ensure_model(stem, cache))
        onnx.checker.check_model(model)
        path = output / file_name
        save_tensors(path, runtime_tensors(stem, model))
        print(json.dumps({"asset": file_name, "sha256": sha256(path)}), flush=True)
    for name in SHARED_FILES:
        ensure_shared_file(name, output)


def main() -> None:
    """Export the CUDA runtime assets, or all reference variants or one selected model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=CACHE)
    parser.add_argument(
        "--model",
        choices=[stem + suffix for stem in MODEL_STEMS for suffix in ("", "-b32")],
    )
    parser.add_argument(
        "--runtime-assets",
        type=Path,
        metavar="DIR",
        help="write the CUDA runtime assets to DIR; ONNX sources are cached in --output",
    )
    args = parser.parse_args()
    if args.runtime_assets is not None:
        export_runtime_assets(args.runtime_assets, args.output)
        return
    stems = (
        [args.model]
        if args.model
        else [stem + suffix for stem in MODEL_STEMS for suffix in ("", "-b32")]
    )
    for stem in stems:
        export(stem, args.output)


if __name__ == "__main__":
    main()
