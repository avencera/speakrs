# /// script
# requires-python = ">=3.12"
# dependencies = ["onnx", "onnxruntime", "numpy", "safetensors", "soundfile"]
# ///
"""Make per-node ORT CPU parity tensors from the Rust CUDA input rules."""

from __future__ import annotations

import argparse
import copy
import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import soundfile as sf
from onnx import AttributeProto, helper, numpy_helper

from export_weights import (
    CACHE,
    MODEL_STEMS,
    ROOT,
    ensure_model,
    revision,
    save_tensors,
    sha256,
    write_json,
)

WINDOW = 160_000
STEP = 16_000
MASK_FRAMES = 589
# the public ORT configuration classes are dynamic C-extension re-exports
ORT_CONFIG: Any = ort
POWERS = np.array(
    [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]],
    dtype=np.float32,
)


@dataclass(frozen=True)
class Window:
    """A Rust-compatible waveform window and its unpadded source length."""

    fixture: str
    index: int
    offset: int
    valid_samples: int
    waveform: np.ndarray

    def source(self) -> dict[str, Any]:
        """Return stable source data for a reference batch row."""
        return {
            "fixture": self.fixture,
            "window_index": self.index,
            "offset_samples": self.offset,
            "valid_samples": self.valid_samples,
            "zero_padding_samples": WINDOW - self.valid_samples,
        }


def windows(path: Path) -> list[Window]:
    """Match SegmentationWindows::collect, including its one final partial window."""
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    if rate != 16_000 or audio.shape[1] != 1:
        raise ValueError(f"expected mono 16 kHz fixture: {path}")
    audio = audio[:, 0]
    offsets = []
    offset = 0
    while offset + WINDOW <= len(audio):
        offsets.append(offset)
        offset += STEP
    if offset < len(audio) and (len(audio) < WINDOW or len(audio) > WINDOW):
        offsets.append(offset)
    result = []
    for index, offset in enumerate(offsets):
        valid = min(WINDOW, len(audio) - offset)
        padded = np.zeros((1, WINDOW), dtype=np.float32)
        padded[0, :valid] = audio[offset : offset + valid]
        result.append(Window(path.name, index, offset, valid, padded))
    return result


def session(model: Path | bytes, optimized: bool) -> ort.InferenceSession:
    """Use deterministic CPU sessions with one thread and no GPU providers."""
    options = ORT_CONFIG.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ORT_CONFIG.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = (
        ORT_CONFIG.GraphOptimizationLevel.ORT_ENABLE_ALL
        if optimized
        else ORT_CONFIG.GraphOptimizationLevel.ORT_DISABLE_ALL
    )
    options.use_deterministic_compute = True
    options.log_severity_level = 3
    return ort.InferenceSession(
        str(model) if isinstance(model, Path) else model,
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )


def run_tensors(
    runner: ort.InferenceSession, inputs: dict[str, np.ndarray]
) -> list[np.ndarray]:
    """Validate the ORT output boundary instead of assuming every value is a tensor."""
    values = runner.run(None, inputs)
    tensors = []
    for value in values:
        if not isinstance(value, np.ndarray):
            raise ValueError(f"non-tensor ORT output: {type(value).__name__}")
        tensors.append(value)
    return tensors


def masks(
    logits: np.ndarray, rows: list[Window], minimum: int
) -> tuple[np.ndarray, list[list[str]]]:
    """Match powerset hard decode and select_speaker_weights, including last-tie wins."""
    if logits.shape != (len(rows), MASK_FRAMES, 7):
        raise ValueError(f"unexpected segmentation layout: {logits.shape}")
    # Rust max_by returns the last equal maximum, unlike numpy's first maximum
    classes = 6 - np.argmax(logits[..., ::-1], axis=-1)
    decoded = POWERS[classes]
    clean = decoded * (decoded.sum(axis=2, keepdims=True) < 2)
    selected = np.zeros((len(rows), 3, MASK_FRAMES), dtype=np.float32)
    choices = []
    for batch_index, row in enumerate(rows):
        threshold = (MASK_FRAMES * minimum + row.valid_samples - 1) // row.valid_samples
        speakers = []
        for speaker in range(3):
            full = decoded[batch_index, :, speaker]
            single = clean[batch_index, :, speaker]
            choice = "inactive"
            if full.sum() >= 10:
                choice = "clean" if single.sum() > threshold else "full"
                selected[batch_index, speaker] = single if choice == "clean" else full
            speakers.append(choice)
        choices.append(speakers)
    return selected.reshape(len(rows) * 3, MASK_FRAMES), choices


def expose(model: onnx.ModelProto, inputs: dict[str, np.ndarray]) -> onnx.ModelProto:
    """Add all nonempty top-level node outputs while keeping each native dtype."""
    result = copy.deepcopy(model)
    for value in result.graph.input:
        shape = value.type.tensor_type.shape
        shape.ClearField("dim")
        for dimension in inputs[value.name].shape:
            shape.dim.add().dim_value = dimension
    result = onnx.shape_inference.infer_shapes(result, data_prop=True)
    known = {
        value.name: value
        for value in [
            *result.graph.input,
            *result.graph.value_info,
            *result.graph.output,
        ]
    }
    outputs = {value.name for value in result.graph.output}
    for node in result.graph.node:
        for name in node.output:
            if not name or name in outputs:
                continue
            if name not in known or not known[name].type.tensor_type.elem_type:
                raise ValueError(
                    f"shape inference did not give the output type: {node.name}:{name}"
                )
            result.graph.output.append(known[name])
            outputs.add(name)
    # ONNX's checker requires output ranks; ORT permits unknown ranks after If
    # exact runtime ranks are validated in the safetensors index
    return result


def branch_tensors(
    model: onnx.ModelProto,
    available: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, Any], list[dict[str, Any]]]:
    """Expose every executed If-branch output through a separate CPU graph."""
    tensors = {}
    index = {}
    branches = []
    initializers = {
        value.name: numpy_helper.to_array(value) for value in model.graph.initializer
    }
    for node in model.graph.node:
        if node.op_type != "If":
            continue
        condition = bool(available[node.input[0]].item())
        chosen = "then_branch" if condition else "else_branch"
        attributes = {
            a.name: a for a in node.attribute if a.type == AttributeProto.GRAPH
        }
        branch = copy.deepcopy(attributes[chosen].g)
        produced = {name for child in branch.node for name in child.output}
        local = {value.name for value in branch.initializer}
        captures = sorted(
            {
                name
                for child in branch.node
                for name in child.input
                if name and name not in produced and name not in local
            }
        )
        feeds = {
            name: available[name] if name in available else initializers[name]
            for name in captures
        }
        branch.ClearField("input")
        for name, value in feeds.items():
            branch.input.append(
                helper.make_tensor_value_info(
                    name,
                    helper.np_dtype_to_tensor_dtype(value.dtype),
                    list(value.shape),
                )
            )
        branch_model = copy.deepcopy(model)
        branch_model.graph.CopyFrom(branch)
        exposed = expose(branch_model, feeds)
        runner = session(exposed.SerializeToString(), optimized=False)
        values = dict(
            zip(
                [v.name for v in exposed.graph.output],
                run_tensors(runner, feeds),
                strict=True,
            )
        )
        for child_index, child in enumerate(branch.node):
            for name in child.output:
                if not name:
                    continue
                key = f"branch/{node.name}/{chosen}/{name}"
                tensors[key] = values[name]
                index[key] = {
                    "role": "branch_intermediate",
                    "node": child.name,
                    "node_index": child_index,
                    "op_type": child.op_type,
                    "onnx_name": name,
                    "graph_scope": f"{node.name}/{chosen}",
                }
        branches.append(
            {
                "node": node.name,
                "condition": condition,
                "executed": chosen,
                "not_executed": "else_branch" if condition else "then_branch",
            }
        )
    return tensors, index, branches


def capture(
    stem: str,
    case: str,
    inputs: dict[str, np.ndarray],
    sources: list[Window],
    details: dict[str, Any],
    output: Path,
    verify: bool,
) -> dict[str, Any]:
    """Save all node outputs and compare final values with the unmodified graph."""
    path = ensure_model(stem, output)
    original = onnx.load(path)
    exposed = expose(original, inputs)
    final_names = {value.name for value in original.graph.output}
    print(f"run {stem}/{case}", flush=True)
    runner = session(exposed.SerializeToString(), optimized=False)
    values = run_tensors(runner, inputs)
    available = dict(
        zip([value.name for value in exposed.graph.output], values, strict=True)
    )
    tensors = {f"input/{name}": value for name, value in inputs.items()}
    index: dict[str, dict[str, Any]] = {
        f"input/{name}": {"role": "input", "onnx_name": name, "node": None}
        for name in inputs
    }
    for node_index, node in enumerate(original.graph.node):
        for name in node.output:
            if not name:
                continue
            key = f"tensor/{name}"
            tensors[key] = available[name]
            index[key] = {
                "role": "output" if name in final_names else "intermediate",
                "onnx_name": name,
                "node": node.name or f"node_{node_index}",
                "node_index": node_index,
                "op_type": node.op_type,
                "graph_scope": "main",
            }
    nested, nested_index, branches = branch_tensors(original, available | inputs)
    tensors.update(nested)
    index.update(nested_index)
    del runner
    gc.collect()
    optimized = session(path, optimized=True)
    expected = run_tensors(optimized, inputs)
    final = {}
    for name, reference in zip(
        [value.name for value in original.graph.output], expected, strict=True
    ):
        actual = available[name]
        difference = float(np.max(np.abs(reference - actual)))
        if not np.allclose(reference, actual, rtol=1e-4, atol=1e-4):
            raise ValueError(
                f"exposing outputs changed final values: {stem}/{case}/{name}: {difference}"
            )
        final[name] = {
            "min": float(actual.min()),
            "max": float(actual.max()),
            "max_abs_diff_optimized": difference,
            "shape": list(actual.shape),
        }
    del optimized, expected
    for key, value in tensors.items():
        if value.dtype.kind == "f" and not np.isfinite(value).all():
            raise ValueError(f"non-finite parity tensor: {stem}/{case}/{key}")
        index[key].update(
            {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "bytes": value.nbytes,
            }
        )
    directory = output / stem
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{case}.safetensors"
    saved_path = (
        destination.with_suffix(".repeat.safetensors") if verify else destination
    )
    if verify and not destination.is_file():
        raise ValueError(f"missing first-run reference: {destination}")
    save_tensors(saved_path, tensors)
    digest = sha256(saved_path)
    if verify:
        first_digest = sha256(destination)
        saved_path.unlink()
        if digest != first_digest:
            raise ValueError(f"byte determinism failed: {stem}/{case}")
    report = {
        "schema_version": 1,
        "model": stem,
        "case": case,
        "hf_revision": revision(),
        "source_model_sha256": sha256(path),
        "safetensors_sha256": digest,
        "tensor_count": len(tensors),
        "tensor_data_bytes": sum(value.nbytes for value in tensors.values()),
        "file_bytes": destination.stat().st_size,
        "source_rows": [row.source() for row in sources],
        "fixture_sha256": {
            row.fixture: sha256(ROOT / "fixtures" / row.fixture) for row in sources
        },
        "cpu_runtime": {
            "onnxruntime": ort.__version__,
            "onnx": onnx.__version__,
            "numpy": np.__version__,
            "threads": 1,
            "provider": "CPUExecutionProvider",
            "optimizations": "disabled for node capture; all enabled for final-value comparison",
        },
        "preparation": details,
        "executed_branches": branches,
        "final_outputs": final,
        "tensors": index,
    }
    index_path = directory / f"{case}.index.json"
    if verify:
        if json.loads(index_path.read_text()) != report:
            raise ValueError(f"index determinism failed: {stem}/{case}")
    else:
        write_json(index_path, report)
    result = {
        key: report[key]
        for key in [
            "model",
            "case",
            "tensor_count",
            "tensor_data_bytes",
            "safetensors_sha256",
            "final_outputs",
        ]
    }
    print(json.dumps(result), flush=True)
    return result


def cases() -> dict[str, list[Window]]:
    """Cover both fixtures, the last partial window, and a full batch of 32."""
    long = windows(ROOT / "fixtures/test.wav")
    short = windows(ROOT / "fixtures/test_short.wav")
    if len(long) < 2 or len(short) != 1:
        raise ValueError("fixture window counts changed")
    combined = long + short
    return {
        "test_first_b1": [long[0]],
        "test_last_partial_b1": [long[-1]],
        "test_short_partial_b1": short,
        "test_and_short_b32": [combined[index % len(combined)] for index in range(32)],
    }


def self_test() -> None:
    """Check padding, PCM scaling, mask decisions, folding, and file serialization."""
    import tempfile
    import unittest

    from safetensors.numpy import load_file

    from export_weights import fold_batch_norm

    class PreparationTests(unittest.TestCase):
        def test_window_boundaries(self) -> None:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "waveform.wav"
                for length, offsets in [
                    (0, []),
                    (WINDOW - 1, [0]),
                    (WINDOW, [0]),
                    (WINDOW + 1, [0, STEP]),
                    (WINDOW + STEP, [0, STEP, 2 * STEP]),
                ]:
                    sf.write(
                        path,
                        np.full(length, 0.25, dtype=np.float32),
                        16_000,
                        subtype="PCM_16",
                    )
                    actual = windows(path)
                    self.assertEqual([row.offset for row in actual], offsets)
                    for row in actual:
                        self.assertTrue(
                            np.all(row.waveform[0, : row.valid_samples] == 0.25)
                        )
                        self.assertTrue(
                            np.all(row.waveform[0, row.valid_samples :] == 0)
                        )

        def test_fixture_scaling(self) -> None:
            for name in ["test.wav", "test_short.wav"]:
                path = ROOT / "fixtures" / name
                integer, _ = sf.read(path, dtype="int16")
                row = windows(path)[0]
                np.testing.assert_array_equal(
                    row.waveform[0, : row.valid_samples],
                    integer[: row.valid_samples].astype(np.float32) / 32768,
                )

        def test_masks_and_ties(self) -> None:
            row = Window(
                "test.wav", 0, 0, WINDOW, np.zeros((1, WINDOW), dtype=np.float32)
            )
            logits = np.full((1, MASK_FRAMES, 7), -100, dtype=np.float32)
            logits[:, :, 1:3] = 0
            selected, choices = masks(logits, [row], 400)
            self.assertEqual(choices, [["inactive", "clean", "inactive"]])
            self.assertTrue(np.all(selected[1] == 1))
            logits.fill(-100)
            logits[:, :, 0] = 0
            logits[:, :12, 0] = -100
            logits[:, :12, 1] = 0
            selected, choices = masks(logits, [row], 3200)
            self.assertEqual(choices[0][0], "full")
            logits[:, 12, 0] = -100
            logits[:, 12, 1] = 0
            _, choices = masks(logits, [row], 3200)
            self.assertEqual(choices[0][0], "clean")
            logits[:, 9:, 0] = 0
            logits[:, 9:, 1] = -100
            selected, choices = masks(logits, [row], 400)
            self.assertEqual(choices[0][0], "inactive")
            self.assertTrue(np.all(selected[0] == 0))

        def test_streaming_format(self) -> None:
            tensors = {
                "scalar": np.array(1.25, dtype=np.float32),
                "boolean": np.array([True, False]),
                "empty": np.zeros((0, 3), dtype=np.int64),
                "index": np.array([np.iinfo(np.int64).max], dtype=np.int64),
            }
            with tempfile.TemporaryDirectory() as directory:
                first = Path(directory) / "first.safetensors"
                second = Path(directory) / "second.safetensors"
                save_tensors(first, tensors)
                save_tensors(second, dict(reversed(list(tensors.items()))))
                self.assertEqual(first.read_bytes(), second.read_bytes())
                for name, actual in load_file(first).items():
                    np.testing.assert_array_equal(actual, tensors[name])
                    self.assertEqual(actual.dtype, tensors[name].dtype)

        def test_batch_norm_fold(self) -> None:
            tensors = {
                "weight": np.arange(6, dtype=np.float32).reshape(2, 1, 3) / 10,
                "bias": np.array([0.1, -0.2], dtype=np.float32),
                "gamma": np.array([0.5, 1.5], dtype=np.float32),
                "beta": np.array([0.2, 0.3], dtype=np.float32),
                "mean": np.array([-0.2, 0.1], dtype=np.float32),
                "variance": np.array([0.3, 0.6], dtype=np.float32),
            }
            nodes = [
                helper.make_node(
                    "Conv", ["input", "weight", "bias"], ["conv"], name="conv"
                ),
                helper.make_node(
                    "BatchNormalization",
                    ["conv", "gamma", "beta", "mean", "variance"],
                    ["output"],
                    name="bn",
                    epsilon=1e-4,
                ),
            ]
            graph = helper.make_graph(
                nodes,
                "fold",
                [
                    helper.make_tensor_value_info(
                        "input", onnx.TensorProto.FLOAT, [1, 1, 10]
                    )
                ],
                [
                    helper.make_tensor_value_info(
                        "output", onnx.TensorProto.FLOAT, [1, 2, 8]
                    )
                ],
                [
                    numpy_helper.from_array(value, name)
                    for name, value in tensors.items()
                ],
            )
            model = helper.make_model(
                graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10
            )
            folds = fold_batch_norm(model, tensors)
            folded = copy.deepcopy(model)
            folded.graph.ClearField("node")
            folded.graph.node.append(
                helper.make_node(
                    "Conv", ["input", folds[0]["weight"], folds[0]["bias"]], ["output"]
                )
            )
            folded.graph.ClearField("initializer")
            folded.graph.initializer.extend(
                [
                    numpy_helper.from_array(tensors[name], name)
                    for name in [folds[0]["weight"], folds[0]["bias"]]
                ]
            )
            inputs = {"input": np.arange(10, dtype=np.float32).reshape(1, 1, 10) / 10}
            original_output = run_tensors(
                session(model.SerializeToString(), False), inputs
            )[0]
            folded_output = run_tensors(
                session(folded.SerializeToString(), False), inputs
            )[0]
            np.testing.assert_allclose(
                original_output, folded_output, rtol=1e-6, atol=1e-6
            )

    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(PreparationTests)
    )
    if not result.wasSuccessful():
        raise SystemExit(1)


def main() -> None:
    """Make or verify the complete reference set in a fresh Python process."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=CACHE)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="rerun and require identical tensor bytes and indices",
    )
    parser.add_argument("--family", choices=MODEL_STEMS)
    parser.add_argument("--case", choices=list(cases()))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    minimum = int(
        (
            ROOT / "fixtures/models/wespeaker-voxceleb-resnet34.min_num_samples.txt"
        ).read_text()
    )
    summaries = []
    for case, rows in cases().items():
        if args.case and case != args.case:
            continue
        suffix = "-b32" if len(rows) == 32 else ""
        waveform = np.stack([row.waveform for row in rows])
        details = {
            "sample_rate": 16_000,
            "window_samples": WINDOW,
            "step_samples": STEP,
            "waveform_dtype": "float32 (PCM16/32768)",
            "batch_rule": "batch 32 repeats the complete fixture-window sequence in order; no unused batch slots",
            "routing_note": "Rust uses batch-1 for partial ORT batches; full 32 rows use the -b32 graph",
            "padding_rule": "waveform tails are zero padded; audio_len for mask selection excludes padding",
            "min_num_samples": minimum,
            "mask_activity_threshold": 10,
            "mask_layout": "[B*3,589], chunk-major then speaker-major",
            "clean_mask_rule": "remove overlapping frames; use clean only if sum > ceil(589*min_num_samples/audio_len)",
        }
        if args.family is None or args.family == MODEL_STEMS[0]:
            summaries.append(
                capture(
                    MODEL_STEMS[0] + suffix,
                    case,
                    {"input": waveform},
                    rows,
                    details,
                    args.output,
                    args.verify,
                )
            )
        if args.family is None or args.family == MODEL_STEMS[1]:
            summaries.append(
                capture(
                    MODEL_STEMS[1] + suffix,
                    case,
                    {"waveform": waveform},
                    rows,
                    details,
                    args.output,
                    args.verify,
                )
            )
        if args.family is not None and args.family != MODEL_STEMS[2]:
            continue
        seg = session(
            ensure_model(MODEL_STEMS[0] + suffix, args.output), optimized=True
        )
        logits = run_tensors(seg, {"input": waveform})[0]
        chosen_masks, choices = masks(logits, rows, minimum)
        fbank = session(
            ensure_model(MODEL_STEMS[1] + suffix, args.output), optimized=True
        )
        features = run_tensors(fbank, {"waveform": waveform})[0]
        del seg, fbank
        summaries.append(
            capture(
                MODEL_STEMS[2] + suffix,
                case,
                {"fbank": features, "masks": chosen_masks},
                rows,
                details
                | {
                    "speaker_mask_choices": choices,
                    "upstream": "unmodified ORT CPU graphs with all optimizations",
                },
                args.output,
                args.verify,
            )
        )
        gc.collect()
    summary_path = args.output / (
        "determinism.json" if args.verify else "reference-summary.json"
    )
    write_json(summary_path, {"byte_identical": args.verify, "cases": summaries})


if __name__ == "__main__":
    main()
