"""Unit tests for the fixed-shape WeSpeaker exporter helpers."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, cast
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def load_exporter() -> ModuleType:
    path = Path(__file__).with_name("export_models.py")
    spec = importlib.util.spec_from_file_location("export_models", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EXPORTER = load_exporter()


class FakeResNet(nn.Module):
    """Small deterministic ResNet-shaped test double."""

    def __init__(self) -> None:
        super().__init__()
        self.two_emb_layer = False
        self.seg_1 = nn.Linear(16, 256, bias=False)

    def forward_frames(self, fbank: torch.Tensor) -> torch.Tensor:
        pooled = F.adaptive_avg_pool1d(fbank.transpose(1, 2)[:, :8], 100)
        return pooled.reshape(fbank.size(0), 2, 4, 100)


class FakeSource(nn.Module):
    """Source model that uses the same admitted frontend and StatsPool path."""

    def __init__(self) -> None:
        super().__init__()
        self.resnet = FakeResNet()
        self.frontend = EXPORTER.FbankWrapper(self)
        self.tail = EXPORTER.FixedEmbeddingTailWrapper(self)

    def forward(
        self, waveforms: torch.Tensor, weights: torch.Tensor | None = None
    ) -> torch.Tensor:
        if weights is None:
            raise AssertionError("the fixed wrapper always supplies weights")
        frames = self.resnet.forward_frames(self.frontend(waveforms))
        frames = frames.reshape(frames.size(0), 8, frames.size(3))
        return self.resnet.seg_1(self.tail.pool(frames, weights))


class FakeOrtValue:
    def __init__(self, name: str) -> None:
        self.name = name


class FakeOrtSession:
    def __init__(
        self,
        result: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        calls: list[tuple[np.ndarray, np.ndarray]],
    ) -> None:
        self.result = result
        self.calls = calls

    def get_inputs(self) -> list[FakeOrtValue]:
        return [FakeOrtValue("waveform"), FakeOrtValue("weights")]

    def get_outputs(self) -> list[FakeOrtValue]:
        return [FakeOrtValue("output")]

    def run(
        self,
        output_names: list[str],
        inputs: dict[str, np.ndarray],
    ) -> list[np.ndarray]:
        del output_names
        waveform = inputs["waveform"]
        weights = inputs["weights"]
        self.calls.append((waveform.copy(), weights.copy()))
        result = self.result(torch.from_numpy(waveform), torch.from_numpy(weights))
        return [result.detach().cpu().numpy()]


def fake_onnxruntime_module(session: FakeOrtSession) -> ModuleType:
    module = ModuleType("onnxruntime")
    setattr(module, "InferenceSession", lambda *_args, **_kwargs: session)
    return module


class ExportModelsTests(unittest.TestCase):
    def test_fixed_window_has_798_fbank_and_100_resnet_frames(self) -> None:
        source = FakeSource().eval()
        waveform = torch.zeros((1, 1, EXPORTER.FIXED_WINDOW_SAMPLES))
        fbank = source.frontend(waveform)
        frames = source.resnet.forward_frames(fbank)

        self.assertEqual(EXPORTER.fbank_frame_count(128_000), 798)
        self.assertEqual(fbank.shape, (1, 798, 80))
        self.assertEqual(EXPORTER.resnet_frame_count(798), 100)
        self.assertEqual(frames.shape[-1], 100)

    def test_nearest_resize_matches_pytorch_reference(self) -> None:
        weights = torch.arange(399, dtype=torch.float32).reshape(1, 399)
        expected = F.interpolate(
            weights.unsqueeze(1), size=100, mode="nearest"
        ).squeeze(1)

        actual = EXPORTER.resize_mask_weights(weights, 100)

        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        expected_indices = torch.floor(torch.arange(100) * 399 / 100)
        torch.testing.assert_close(actual[0], expected_indices, rtol=0.0, atol=0.0)
        self.assertEqual(actual.shape, (1, 100))

    def test_fixed_wrapper_matches_source_probe_masks(self) -> None:
        source = FakeSource().eval()
        wrapper = EXPORTER.FixedEmbeddingWrapper(source).eval()

        EXPORTER.compare_fixed_wrapper_parity(source, wrapper, 400)

    def test_onnx_runtime_parity_uses_fixed_probe_cases(self) -> None:
        source = FakeSource().eval()
        wrapper = EXPORTER.FixedEmbeddingWrapper(source).eval()
        calls: list[tuple[np.ndarray, np.ndarray]] = []
        session = FakeOrtSession(
            lambda waveform, weights: wrapper(waveform, weights), calls
        )

        with patch.dict(sys.modules, {"onnxruntime": fake_onnxruntime_module(session)}):
            EXPORTER.compare_fixed_onnx_parity(Path("staged.onnx"), source, 400)

        expected_waveform = EXPORTER.fixed_parity_waveform().numpy()
        expected_masks = [
            weights.numpy() for _, weights in EXPORTER.fixed_parity_masks(400)
        ]
        self.assertEqual(len(calls), len(expected_masks))
        for (waveform, weights), expected_mask in zip(calls, expected_masks):
            np.testing.assert_array_equal(waveform, expected_waveform)
            np.testing.assert_array_equal(weights, expected_mask)

    def test_sidecar_hash_and_strict_content(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model_path = Path(temporary) / EXPORTER.FIXED_EMBEDDING_FILENAME
            model_bytes = b"deterministic onnx bytes"
            model_path.write_bytes(model_bytes)

            sidecar_path = EXPORTER.embedding_sidecar_path(model_path)
            sidecar_path.write_bytes(
                EXPORTER.build_embedding_sidecar(model_path, 400).json_bytes()
            )
            payload = json.loads(sidecar_path.read_text())

        self.assertEqual(
            set(payload),
            {
                "schema_version",
                "onnx_sha256",
                "sample_rate",
                "window_samples",
                "mask_frames",
                "embedding_width",
                "frontend",
                "pooling",
                "interpolation",
                "resnet_frames",
                "precision",
                "min_num_samples",
            },
        )
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(
            payload["onnx_sha256"], hashlib.sha256(model_bytes).hexdigest()
        )
        self.assertEqual(payload["sample_rate"], 16_000)
        self.assertEqual(payload["window_samples"], 128_000)
        self.assertEqual(payload["mask_frames"], 399)
        self.assertEqual(payload["embedding_width"], 256)
        self.assertEqual(payload["frontend"], "wespeaker_fbank_v1")
        self.assertEqual(payload["pooling"], "masked_stats_pool_v1")
        self.assertEqual(payload["interpolation"], "nearest")
        self.assertEqual(payload["resnet_frames"], 100)
        self.assertEqual(payload["precision"], "float32")
        self.assertEqual(payload["min_num_samples"], 400)

    def test_missing_threshold_is_rejected(self) -> None:
        class MissingEmbedding:
            pass

        class Pipeline:
            _embedding = MissingEmbedding()

        with self.assertRaises(EXPORTER.FixedEmbeddingExportError):
            EXPORTER.admitted_min_num_samples(Pipeline())

    def test_publish_interruption_cannot_admit_mismatched_pair(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_path = root / EXPORTER.FIXED_EMBEDDING_FILENAME
            sidecar_path = EXPORTER.embedding_sidecar_path(output_path)
            output_path.write_bytes(b"old model")
            sidecar_path.write_bytes(
                EXPORTER.build_embedding_sidecar(output_path, 400).json_bytes()
            )

            staging_dir = root / "staging"
            staging_dir.mkdir()
            staging_model = staging_dir / output_path.name
            staging_model.write_bytes(b"new model")
            staging_sidecar = staging_dir / sidecar_path.name
            staging_sidecar.write_bytes(
                EXPORTER.build_embedding_sidecar(staging_model, 400).json_bytes()
            )

            replace = EXPORTER.os.replace

            def interrupt_model_publish(
                source: str | os.PathLike[str], destination: str | os.PathLike[str]
            ) -> None:
                if Path(source) == staging_model:
                    raise OSError("intentional model publish interruption")
                replace(source, destination)

            with patch.object(
                EXPORTER.os, "replace", side_effect=interrupt_model_publish
            ):
                with self.assertRaises(OSError):
                    EXPORTER._publish_fixed_embedding_pair(
                        staging_model,
                        staging_sidecar,
                        output_path,
                        sidecar_path,
                        400,
                    )

            with self.assertRaises(EXPORTER.FixedEmbeddingExportError):
                EXPORTER._validate_published_sidecar(output_path, sidecar_path, 400)

    def test_failed_parity_does_not_publish(self) -> None:
        source = FakeSource().eval()
        with tempfile.TemporaryDirectory() as temporary:
            model_path = Path(temporary) / EXPORTER.FIXED_EMBEDDING_FILENAME
            sidecar_path = EXPORTER.embedding_sidecar_path(model_path)

            def fail_parity(*_args: object, **_kwargs: object) -> None:
                raise EXPORTER.FixedEmbeddingExportError("intentional parity failure")

            with patch.object(EXPORTER, "compare_fixed_wrapper_parity", fail_parity):
                with self.assertRaises(EXPORTER.FixedEmbeddingExportError):
                    EXPORTER.export_fixed_embedding_model(source, temporary, 400)

            self.assertFalse(model_path.exists())
            self.assertFalse(sidecar_path.exists())

    def test_failed_onnx_runtime_parity_does_not_publish(self) -> None:
        source = FakeSource().eval()
        calls: list[tuple[np.ndarray, np.ndarray]] = []

        def mismatched_result(
            waveform: torch.Tensor, weights: torch.Tensor
        ) -> torch.Tensor:
            with torch.inference_mode():
                return source(waveform, weights=weights) + 1.0

        session = FakeOrtSession(mismatched_result, calls)

        def fake_export(*args: Any, **_kwargs: Any) -> None:
            staged_path = Path(cast(str | os.PathLike[str], args[2]))
            staged_path.write_bytes(b"staged fixed wrapper")

        with tempfile.TemporaryDirectory() as temporary:
            model_path = Path(temporary) / EXPORTER.FIXED_EMBEDDING_FILENAME
            sidecar_path = EXPORTER.embedding_sidecar_path(model_path)
            with (
                patch.dict(
                    sys.modules, {"onnxruntime": fake_onnxruntime_module(session)}
                ),
                patch.object(EXPORTER.torch.onnx, "export", fake_export),
                patch.object(EXPORTER, "_validate_exported_onnx"),
            ):
                with self.assertRaises(EXPORTER.FixedEmbeddingExportError):
                    EXPORTER.export_fixed_embedding_model(source, temporary, 400)

            self.assertFalse(model_path.exists())
            self.assertFalse(sidecar_path.exists())


if __name__ == "__main__":
    unittest.main()
