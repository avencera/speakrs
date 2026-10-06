"""Recomputable noise evaluation must not hide weak speedups or hard failures."""

import copy
import importlib
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/cuda/qualify"))
verdict = importlib.import_module("verdict")


class Verdict(unittest.TestCase):
    def timing(self, speedup=1.2, stage=False):
        key = "fp32/first/b1/" + ("stage" if stage else "lstm.stack")
        bound = 0.003 if stage else 0.01
        return {
            "id": key,
            "medians_ms": [1.0, 1.0 / speedup, 1.02, 1.02 / speedup],
            "speedups": [speedup, speedup],
            "library_process_spread_fraction": 0.02,
            "spread_bound": bound,
        }

    def blocked(self, timing):
        return {
            "check": "speed:" + timing["id"],
            "passed": False,
            "blocked": True,
            "reason": "speed: blocked: Library process spread 0.020000 exceeds bound 0.01",
        }

    def record(self, timing, extra=()):
        return {
            "schema": 3,
            "implementation": "Oxide",
            "status": "blocked",
            "accepts_replacement": False,
            "tiers": {
                "sm75": {
                    "status": "blocked",
                    "checks": [self.blocked(timing), *extra],
                    "timing": [timing],
                    "coverage_declared": {
                        "triples": [
                            ["lstm.stack", 1, "fp32"],
                            ["lstm.stack", 32, "fp32"],
                        ]
                    },
                }
            },
        }

    def test_noise_rule_accepts_from_own_data_and_keeps_raw_block(self):
        timing = self.timing()
        record = self.record(timing)
        result = verdict.evaluate_record(record, "sm75")
        self.assertTrue(result["accepted"])
        self.assertEqual(result["noise_rule"][0]["threshold"], 1.06)
        self.assertFalse(record["accepts_replacement"])
        self.assertFalse(record["tiers"]["sm75"]["checks"][0]["passed"])
        self.assertEqual(len(result["accepted_tuples"]), 2)

    def test_below_threshold_excludes_exact_production_tuple(self):
        timing = self.timing(1.059)
        result = verdict.evaluate_record(self.record(timing), "sm75")
        self.assertFalse(result["accepted"])
        self.assertFalse(result["noise_rule"][0]["accepted"])
        self.assertEqual(result["accepted_tuples"], [["lstm.stack", 32, "fp32"]])
        self.assertEqual(result["excluded_tuples"], [["lstm.stack", 1, "fp32"]])

    def test_hard_failure_plus_noise_can_never_accept(self):
        hard = {
            "check": "layer:fp32/lstm.stack",
            "passed": False,
            "reason": "layer parity",
        }
        result = verdict.evaluate_record(self.record(self.timing(), [hard]), "sm75")
        self.assertFalse(result["accepted"])
        self.assertFalse(result["accepted_tuples"])
        self.assertTrue(result["noise_rule"][0]["accepted"])

    def test_stored_stage_noise_rule_but_never_new_paired_stage(self):
        record = self.record(self.timing(stage=True))
        self.assertTrue(verdict.evaluate_record(record, "sm75")["accepted"])
        record["schema"] = 4
        result = verdict.evaluate_record(record, "sm75")
        self.assertFalse(result["accepted"])
        self.assertIn("paired ABBA", result["unresolved"][0]["reason"])
        paired = {
            "check": "paired_stage:fp32/first/b1/stage",
            "passed": False,
            "blocked": True,
            "reason": "paired stage: CI cannot resolve saving",
        }
        result = verdict.evaluate_checks(
            [paired], [self.timing(stage=True)], allow_stage=False
        )
        self.assertFalse(result["accepted"])

    def test_tampered_ratios_spread_bound_and_other_block_never_pass(self):
        original = self.timing()
        for key, value in [
            ("speedups", [2.0, 2.0]),
            ("spread_bound", 0.001),
            ("library_process_spread_fraction", 0.015),
        ]:
            timing = {**original, key: value}
            with self.subTest(key=key), self.assertRaises(verdict.Rejected):
                verdict.noise_timing(self.blocked(timing), timing, allow_stage=False)
        check = {**self.blocked(original), "reason": "speed: blocked: missing samples"}
        self.assertFalse(
            verdict.evaluate_checks([check], [original], allow_stage=False)["accepted"]
        )
        check = {**self.blocked(original), "check": "sanitizer:memcheck"}
        self.assertFalse(
            verdict.evaluate_checks([check], [original], allow_stage=False)["accepted"]
        )

    def test_parent_only_failure_cannot_be_hidden(self):
        record = self.record(self.timing())
        record["checks"] = [
            {"check": "static_scan", "passed": False, "reason": "refused"}
        ]
        result = verdict.evaluate_record(record, "sm75")
        self.assertFalse(result["accepted_tuples"])


if __name__ == "__main__":
    unittest.main()
