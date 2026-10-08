"""The filterbank producer has its own boundary domain and evidence geometry."""

import importlib
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/cuda/qualify"))
qualify = importlib.import_module("qualify")
records = importlib.import_module("records")
verdict = importlib.import_module("verdict")


class FbankDomain(unittest.TestCase):
    def test_domain_expansion_covers_every_valid_batch_only(self):
        coverage = qualify.parse_coverage(
            {"entries": [{"layers": "all", "batches": "all", "maths": "all"}]},
            "fbankdft",
        )
        self.assertEqual(coverage.batches, frozenset(range(1, 33)))
        self.assertEqual(len(coverage.triples), 64)
        self.assertTrue(coverage.declared("fbank.dft", 31, "tf32"))
        self.assertFalse(coverage.declared("fbank.dft", 33, "tf32"))
        for batch in (0, 33, 64):
            with self.assertRaisesRegex(qualify.Rejected, "cannot test"):
                qualify.parse_coverage(
                    {
                        "entries": [
                            {"layers": "all", "batches": [batch], "maths": "all"}
                        ]
                    },
                    "fbankdft",
                )
        self.assertEqual(qualify.sanitizer_batches(coverage), list(range(1, 33)))

    def test_record_domain_keeps_inner_batches_and_rejects_stress_claims(self):
        entries = [{"layers": ["fbank.dft"], "batches": "all", "maths": "all"}]
        expected = {
            ("fbank.dft", batch, mode)
            for batch in range(1, 33)
            for mode in ("fp32", "tf32")
        }
        self.assertEqual(records.triples({"entries": entries}), expected)
        self.assertEqual(records.tested_declaration({"entries": entries}), expected)
        self.assertEqual(
            records.summary_tuples([list(row) for row in sorted(expected)]), expected
        )
        for layer, batch in (("fbank.dft", 33), ("lstm.stack", 2), ("lstm.stack", 7)):
            with self.assertRaises(records.Rejected):
                records.triples(
                    {
                        "entries": [
                            {"layers": [layer], "batches": [batch], "maths": ["fp32"]}
                        ]
                    }
                )
        child: dict = {
            "target": "fbankdft",
            "implementation": "Oxide",
            "status": "passed",
            "coverage_declared": {"triples": [list(row) for row in sorted(expected)]},
            "coverage": {
                "tier": "sm75",
                "math": ["fp32", "tf32"],
                "cases": [list(case) for case in qualify.target_cases("fbankdft")],
            },
            "phases_run": records.CURRENT_PHASES,
        }
        raw: dict = {
            "schema": 5,
            "target": "fbankdft",
            "implementation": "Oxide",
            "status": "passed",
            "tiers": {"sm75": child},
        }
        names, timing = records.tuple_requirements(raw, child)
        self.assertIn("speed:fp32/mixed/b2/fbank.dft", names)
        self.assertIn("tf32/mixed/b31/stage", timing)
        names |= records.required_checks("fbankdft")
        child["checks"] = [{"check": name, "passed": True} for name in sorted(names)]
        child["timing"] = [{"id": name} for name in sorted(timing)]
        raw["checks"] = [
            {"check": "sm75/" + name, "passed": True} for name in sorted(names)
        ]
        child["accepted_tuples"] = [list(row) for row in sorted(expected)]
        records.complete_collection(raw)
        self.assertEqual(
            verdict.evaluate_record(raw, "sm75")["accepted_tuples"],
            child["accepted_tuples"],
        )
        child["coverage"]["cases"] = [list(case) for case in records.TEST_CASES]
        with self.assertRaisesRegex(records.Rejected, "incomplete tier collection"):
            records.complete_collection(raw)

    def test_case_evidence_uses_the_producer_and_consumer_boundaries(self):
        numeric = qualify.expected_ids("fbankdft", "numeric", "fp32")
        self.assertEqual(len(numeric), 140)
        self.assertIn("fp32/mixed/b31/fbank.dft/switched", numeric)
        self.assertIn("fp32/last/b1/stage", numeric)
        self.assertNotIn("fp32/mixed/b33/fbank.dft", numeric)
        self.assertEqual(len(qualify.expected_ids("fbankdft", "timing", "tf32")), 70)
        self.assertEqual(qualify.target_cases("sincnet"), qualify.CASES)
        self.assertEqual(qualify.target_batches("lstm"), qualify.BATCHES)

    def test_draw_bytes_are_bound_to_energy_layout_and_boundary_domain(self):
        self.assertEqual(
            qualify.draw_length("tf32/mixed/b31/stage/switched", "fbank.dft"),
            31 * 998 * 80,
        )
        for case, layer in (
            ("tf32/mixed/b33/stage", "fbank.dft"),
            ("tf32/mixed/b31/stage", "lstm.stack"),
            ("tf32/mixed/b0/stage", "fbank.dft"),
        ):
            with self.assertRaises(qualify.Rejected):
                qualify.draw_length(case, layer)
        coverage = qualify.Coverage.product(("fbank.dft",), (2, 31), ("tf32",))
        self.assertEqual(
            qualify.band_layers(coverage, "Oxide"), "2:fbank.dft;31:fbank.dft"
        )


class ShortTrace(unittest.TestCase):
    def test_retains_eager_trace_only_when_side_streams_need_it(self):
        import json
        import tempfile
        from unittest.mock import patch

        for streams in ([], [7]):
            with tempfile.TemporaryDirectory() as directory:
                modes = []

                def command(argv, env, log, steps):
                    if argv[:2] == ["nsys", "export"]:
                        Path(argv[argv.index("-o") + 1]).write_bytes(b"trace")
                    return 0

                def gpu(argv, env, log, steps):
                    modes.append(env["SPEAKRS_QUALIFY_SHORT_TRACE"])
                    Path(env["SPEAKRS_QUALIFY_OUTPUT"]).write_text(
                        json.dumps({"side_streams": streams})
                    )
                    return 0

                result = {}
                with (
                    patch.object(qualify, "command", side_effect=command),
                    patch.object(qualify, "gpu_command", side_effect=gpu),
                    patch.object(qualify, "validate_target"),
                ):
                    path = qualify.profile(
                        result,
                        Path("driver"),
                        {
                            "SPEAKRS_CUDA_PTX_TIER": "sm75",
                            "SPEAKRS_QUALIFY_TARGET": "sincnet",
                        },
                        Path(directory),
                        [],
                        "Library",
                        "sincnet",
                    )
                self.assertEqual(modes, ["1", "0"] if streams else ["1"])
                self.assertEqual(
                    path.name, "profile-eager.sqlite" if streams else "profile.sqlite"
                )
                self.assertNotIn("short_profile", result)
                self.assertEqual(
                    bool(result["profile_traces"][0].get("retained_for_eager")),
                    bool(streams),
                )
                self.assertEqual(
                    [item["label"] for item in result["profile_traces"]],
                    ["profile", "profile-eager"] if streams else ["profile"],
                )
                with self.assertRaisesRegex(qualify.Rejected, "not checked"):
                    qualify.profile_trace_receipts(result)

    def test_first_use_fault_requires_the_existing_profile_reason(self):
        gate = qualify.MUTANT_GATES["FirstUseFallback"]
        self.assertTrue(
            gate.matches(
                {
                    "check": "sm75/profile",
                    "reason": "profile: forbidden library kernels in a candidate scope",
                }
            )
        )
        self.assertFalse(
            gate.matches(
                {
                    "check": "sm75/profile:graph_nodes",
                    "reason": "forbidden library kernels",
                }
            )
        )
        self.assertEqual(
            qualify.MUTANT_PHASES["FirstUseFallback"], ("numeric", "profile")
        )


class PhaseTimes(unittest.TestCase):
    def test_wall_time_table_keeps_cpu_subsets_without_double_counting(self):
        evidence = qualify.phase_wall_times(
            [
                {"phase": "build", "wall_seconds": 3.0},
                {
                    "phase": "numeric",
                    "wall_seconds": 11.0,
                    "cpu_work_wall_seconds": {"f64": 5.0, "tf32_draws": 2.0},
                },
                {"phase": "timing", "wall_seconds": 7.0},
                {"phase": "paired", "wall_seconds": 13.0},
                {"phase": "profile", "wall_seconds": 17.0},
                {"phase": "sanitize", "wall_seconds": 19.0},
                {"phase": "filter_proof", "wall_seconds": 23.0},
            ]
        )
        self.assertEqual(
            evidence["seconds"],
            {
                "build": 3.0,
                "cpu_truth": 5.0,
                "cpu_draws": 2.0,
                "numeric": 11.0,
                "timing": 7.0,
                "paired": 13.0,
                "profile": 17.0,
                "sanitizer": 42.0,
            },
        )
        self.assertTrue(evidence["numeric_includes_cpu_work"])
        self.assertTrue(evidence["cpu_work_excludes_lock_wait"])


class StageTruthRegression(unittest.TestCase):
    def binding(self, key="fp32/first/b1/stage"):
        import hashlib
        import struct

        mode, case, batch, *_ = key.split("/")
        count = int(batch[1:])
        source = case
        if key.endswith("/switched"):
            source = "first" if case == "short" else "short"
        truth = [1.0, -1.0] * (count * 998 * 40)
        return {
            "case": key,
            "evaluation": "complete-deterministic",
            "dtype": "f64",
            "definition": qualify.FBANK_STAGE_DEFINITION,
            "constants": {
                "sha256": "c" * 64,
                "mel_sha256": "d" * 64,
                "window": "exact-f64-Hamming",
                "mel": "built-in-f32-converted-to-f64",
                "scale": 32768.0,
                "preemphasis": 0.97,
                "energy_floor": 2.0**-23,
                "window_f32_max_abs": 2e-8,
            },
            "input_shape": [count, 160_000],
            "shape": [count, 998, 80],
            "fixture_rows": list(range(count))
            if source == "mixed"
            else [{"first": 0, "last": 17, "short": 18}[source]] * count,
            "input_length": count * 160_000,
            "length": len(truth),
            "input_sha256": "e" * 64,
            "truth_sha256": hashlib.sha256(
                struct.pack(f"<{len(truth)}d", *truth)
            ).hexdigest(),
        }

    def row(self, offset=0.0, key="fp32/first/b1/stage"):
        import hashlib
        import math
        import struct

        binding = self.binding(key)
        truth = [1.0, -1.0] * (binding["length"] // 2)
        actual = [value + offset for value in truth]
        error = sum((a - b) ** 2 for a, b in zip(actual, truth))
        norm = sum(value**2 for value in truth)
        metrics = {
            "relative_l2": math.sqrt(error / norm),
            "max_abs": max(abs(a - b) for a, b in zip(actual, truth)),
            "minimum_cosine": 1 / math.sqrt(1 + offset**2),
            "mean_cosine": 1 / math.sqrt(1 + offset**2),
            "argmax_flips": 0,
            "sha256": hashlib.sha256(
                struct.pack(f"<{len(actual)}f", *actual)
            ).hexdigest(),
            "elements": len(actual),
        }
        return {
            "id": key,
            "first": dict(metrics),
            "second": dict(metrics),
            "truth": dict(metrics),
            "truth_sha256": binding["truth_sha256"],
            "stage_truth": binding,
            "bitwise_equal": True,
            "declared": True,
            "fixture_diagnostic": {
                "reference": "ORT FP32 tensor/fbank",
                "acceptance_bound": False,
                "max_abs": abs(offset - 0.5),
            },
        }

    def test_exact_candidate_passes_and_ort_closer_but_worse_candidate_fails(self):
        control = self.row(0.125)
        exact = self.row()
        worse = self.row(0.25)
        self.assertLess(
            worse["fixture_diagnostic"]["max_abs"],
            control["fixture_diagnostic"]["max_abs"],
        )
        for candidate, passed in ((exact, True), (worse, False)):
            result = {"target": "fbankdft", "checks": []}
            qualify.stage_check(result, candidate["id"], candidate, control)
            self.assertEqual(result["checks"][0]["passed"], passed)
            if not passed:
                self.assertEqual(
                    result["checks"][0]["reason"], "segmentation stage parity: logits"
                )
        self.assertEqual(exact["first"]["max_abs"], 0.0)

    def test_missing_mismatched_or_incomplete_f64_identity_fails_closed(self):
        import copy

        control = self.row(0.125)
        for fault in (
            "missing",
            "truth_hash",
            "input_hash",
            "constants",
            "length",
            "shape",
            "seed",
            "definition",
            "fixture_rows",
            "metrics_length",
            "metrics_owner",
            "shape_type",
        ):
            candidate = copy.deepcopy(control)
            binding = candidate["stage_truth"]
            if fault == "missing":
                candidate.pop("stage_truth")
            elif fault == "truth_hash":
                candidate["truth_sha256"] = "f" * 64
            elif fault == "input_hash":
                binding["input_sha256"] = "f" * 64
            elif fault == "constants":
                binding["constants"]["sha256"] = "f" * 64
            elif fault == "length":
                binding["length"] -= 1
            elif fault == "shape":
                binding["shape"] = [1, 80, 998]
            elif fault == "seed":
                binding["seed"] = 11
            elif fault == "definition":
                binding["definition"] = "ORT"
            elif fault == "fixture_rows":
                binding["fixture_rows"] = [17]
            elif fault == "shape_type":
                binding["shape"][0] = True
            elif fault == "metrics_length":
                candidate["second"]["elements"] -= 1
            else:
                candidate["truth"]["max_abs"] = 0.0
            with self.subTest(fault=fault):
                result = {"target": "fbankdft", "checks": []}
                qualify.stage_check(result, candidate["id"], candidate, control)
                self.assertFalse(result["checks"][0]["passed"])
                self.assertIn("fbank stage truth", result["checks"][0]["reason"])

    def process(self, row, *, verify=False):
        section = {"work": "stage_f64", "locked": False, **row["stage_truth"]}
        return {
            "target": "fbankdft",
            "rows": [row],
            "gpu_lock": {
                "path": qualify.GPU_LOCK,
                "owner": "child",
                "gpu_sections": 2,
                "cpu_sections": [section],
                "cpu_mode": "verify" if verify else "parallel",
                "cpu_byte_identity": [
                    {
                        "serial_parallel_equal": True,
                        "bytes": 8 + row["stage_truth"]["length"] * 8,
                        "sha256": "a" * 64,
                        "binding": dict(section),
                    }
                ]
                if verify
                else [],
            },
        }

    def test_stage_owner_and_full_byte_proof_fail_closed(self):
        import copy

        original = self.process(self.row(), verify=True)
        qualify.validate_gpu_ownership(original, "numeric")
        for fault in (
            "missing",
            "locked",
            "parent",
            "case",
            "hash",
            "target",
            "duplicate",
            "proof_missing",
            "proof_length",
            "proof_owner",
            "secret_missing",
            "draw_missing",
            "duplicate_row",
            "proof_extra",
            "cpu_mode",
        ):
            process = copy.deepcopy(original)
            lock = process["gpu_lock"]
            section = lock["cpu_sections"][0]
            if fault == "missing":
                lock["cpu_sections"] = []
                lock["gpu_sections"] = 1
            elif fault == "locked":
                section["locked"] = True
            elif fault == "parent":
                lock["owner"] = "parent"
            elif fault == "case":
                section["case"] = "fp32/last/b1/stage"
            elif fault == "hash":
                section["truth_sha256"] = "b" * 64
            elif fault == "target":
                process["target"] = "resnet"
            elif fault == "duplicate":
                lock["cpu_sections"].append(dict(section))
                lock["gpu_sections"] += 1
            elif fault == "proof_missing":
                lock["cpu_byte_identity"] = []
            elif fault == "proof_length":
                lock["cpu_byte_identity"][0]["bytes"] -= 8
            elif fault == "proof_extra":
                lock["cpu_byte_identity"].append(
                    copy.deepcopy(lock["cpu_byte_identity"][0])
                )
            elif fault == "cpu_mode":
                lock["cpu_mode"] = "unknown"
            elif fault == "proof_owner":
                lock["cpu_byte_identity"][0]["binding"]["case"] = "wrong"
            elif fault == "duplicate_row":
                process["rows"].append(copy.deepcopy(process["rows"][0]))
            elif fault == "secret_missing":
                process["rows"].append(
                    {"id": "fp32/secret/b1/fbank.dft", "secret": True}
                )
            else:
                band = self.band(self.row(key="tf32/first/b1/stage"))
                process["rows"] = [self.row(key="tf32/first/b1/stage"), band]
                lock["cpu_sections"] = [
                    {"work": "stage_f64", "locked": False, **band["stage_truth"]}
                ]
                lock["cpu_mode"] = "parallel"
            with self.subTest(fault=fault), self.assertRaises(qualify.Rejected):
                qualify.validate_gpu_ownership(process, "numeric")

    def band(self, row):
        import copy

        return {
            "id": row["id"] + "/band",
            "layers": ["fbank.dft"],
            "seeds": [11, 23, 37, 41, 53, 67, 79, 97],
            "metrics": [dict(row["first"]) for _ in range(8)],
            "truth_draws": [dict(row["first"]) for _ in range(8)],
            "truth_sha256": row["truth_sha256"],
            "stage_truth": copy.deepcopy(row["stage_truth"]),
        }

    def test_tf32_uses_the_same_f64_identity_and_fixed_draws(self):
        import copy

        control = self.row(0.125, "tf32/first/b1/stage")
        band = self.band(control)
        coverage = qualify.Coverage.product(["fbank.dft"], [1], ["tf32"])
        for fault in (None, "worse", "input", "seed", "draw"):
            candidate, draws = copy.deepcopy(control), copy.deepcopy(band)
            if fault == "worse":
                candidate = self.row(0.25, control["id"])
            elif fault == "input":
                draws["stage_truth"]["input_sha256"] = "f" * 64
            elif fault == "seed":
                draws["seeds"][0] = 12
            elif fault == "draw":
                draws["truth_draws"].pop()
            result = {"target": "fbankdft", "implementation": "Oxide", "checks": []}
            qualify.numeric(result, [control, draws], [candidate], coverage)
            checked = next(
                row
                for row in result["checks"]
                if row["check"].startswith("stage_truth:")
            )
            self.assertEqual(checked["passed"], fault is None)
            if fault in (None, "worse"):
                self.assertEqual(
                    checked["evidence"]["truth"],
                    "independent CPU f64, complete stage, same input",
                )


if __name__ == "__main__":
    unittest.main()
