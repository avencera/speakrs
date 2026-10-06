"""Keep candidate variant selection distinct from the Library tier limit."""

import copy
import importlib
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/cuda/qualify"))
qualify = importlib.import_module("qualify")


class TargetTier(unittest.TestCase):
    def process(self):
        process: dict = {
            "phase": "timing",
            "device_sm": "12.0",
            "device": {
                "name": "RTX test",
                "compute_capability": "12.0",
                "sm_count": 36,
                "l2_bytes": 33554432,
                "driver_version": "570.0",
                "driver_api_version": 12080,
                "cuda_version": 12080,
                "cudnn_version": 90000,
                "cublas_version": 120800,
            },
            "loaded_modules": [
                {"area": "resnet", "tier": "sm80"},
                {"area": "fbank", "tier": "sm75"},
                {"area": "segmentation", "tier": "sm75"},
                {"area": "qualify", "tier": "sm75"},
            ],
            "observed_sm_clock": {"samples": 4, "min_mhz": 2400, "max_mhz": 2700},
        }
        process["loaded_modules"] = [
            dict(module, artifact={"kind": "PtxJit", "sha256": "a" * 64})
            for module in process["loaded_modules"]
        ]
        return process

    def test_candidate_tier_is_exact_but_other_areas_retain_lower_tiers(self):
        process = self.process()
        evidence = qualify.validate_target(process, "sm80", candidate_area="resnet")
        self.assertEqual(
            evidence["loaded_tiers"],
            {
                "resnet": "sm80",
                "fbank": "sm75",
                "segmentation": "sm75",
                "qualify": "sm75",
            },
        )
        wrong = copy.deepcopy(process)
        wrong["loaded_modules"][0]["tier"] = "sm75"
        with self.assertRaisesRegex(qualify.Rejected, "differs from requested"):
            qualify.validate_target(wrong, "sm80", candidate_area="resnet")

    def test_library_control_has_no_forced_candidate_area(self):
        process = self.process()
        process["loaded_modules"][0]["tier"] = "sm75"
        evidence = qualify.validate_target(process, "sm80", candidate_area=None)
        self.assertEqual(evidence["loaded_tiers"]["resnet"], "sm75")

    def test_library_limit_still_requires_capable_hardware(self):
        process = self.process()
        process["device_sm"] = process["device"]["compute_capability"] = "7.5"
        with self.assertRaisesRegex(qualify.Rejected, "cannot execute requested"):
            qualify.validate_target(process, "sm80", candidate_area=None)
