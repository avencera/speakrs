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
                self.assertEqual("short_profile" in result, bool(streams))

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


if __name__ == "__main__":
    unittest.main()
