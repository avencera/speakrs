"""CPU regressions for qualification binding and the Library-noise acceptance rule"""

import copy
import importlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/cuda/qualify"))
qualified = importlib.import_module("qualified")


def completed(record):
    """Give synthetic records the completion evidence required of real records"""
    required = [
        "ptx:loaded_bytes",
        "ptx:loaded_bytes/stable",
        "ptx:shared_initialization",
        "determinism:fixed_reduction_order",
        "profile",
        "profile:graph_nodes",
        "profile:captured_library_calls",
        *(f"sanitizer:control/{fault}" for fault in ["oob", "race", "uninit"]),
        *(f"sanitizer:Oxide/{tool}" for tool in ["memcheck", "racecheck", "initcheck"]),
    ]
    record["checks"] = []
    for tier, child in record["tiers"].items():
        existing = {c["check"] for c in child["checks"]}
        child["checks"].extend(
            {"check": name, "passed": True} for name in required if name not in existing
        )
        failed = [c for c in child["checks"] if not c["passed"]]
        child["status"] = "blocked" if failed else "passed"
        child["reason"] = (
            f"{len(failed)} checks cannot be decided: {[c['check'] for c in failed[:6]]}"
            if failed
            else "all required checks passed"
        )
        child["phases_run"] = ["numeric", "timing", "profile", "sanitize"]
        child["coverage"] = {"tier": tier}
        record["checks"].extend(
            {**c, "check": f"{tier}/{c['check']}"} for c in child["checks"]
        )
    return record


class Manifest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        names = {
            "src/inference/cuda/implementation.rs",
            "src/inference/cuda/candidate.rs",
            qualified.MANIFEST,
        }
        for area, host in qualified.production(ROOT).items():
            for hashes in qualified.files(ROOT, area, host).values():
                names.update(hashes)
        for name in names:
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)

    def change(self, name, old, new):
        path = self.root / name
        text = path.read_text()
        self.assertIn(old, text)
        path.write_text(text.replace(old, new, 1))

    def test_pass(self):
        qualified.check(self.root)

    def test_changed_ptx(self):
        path = self.root / "src/inference/cuda/ptx/lstm.sm75.ptx"
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaisesRegex(
            qualified.QualificationError, "qualification file differs.*lstm.sm75.ptx"
        ):
            qualified.check(self.root)

    def test_changed_candidate_source(self):
        path = self.root / "src/inference/cuda/candidate/sinc.rs"
        path.write_text(path.read_text() + "\n// changed host source\n")
        with self.assertRaisesRegex(
            qualified.QualificationError,
            "qualification file differs.*candidate/sinc.rs",
        ):
            qualified.check(self.root)

    def test_changed_shared_candidate_execution(self):
        self.change(
            "src/inference/cuda/candidate.rs",
            "self.runtime.sgemm(spec, a, weight, c)",
            "Ok(())",
        )
        with self.assertRaisesRegex(
            qualified.QualificationError, "qualification file differs.*candidate.rs"
        ):
            qualified.check(self.root)

    def test_changed_kernel_source(self):
        path = self.root / "crates/speakrs-cuda-kernels/src/lstm.rs"
        path.write_text(path.read_text() + "\n// changed kernel source\n")
        with self.assertRaisesRegex(
            qualified.QualificationError,
            "qualification file differs.*kernels/src/lstm.rs",
        ):
            qualified.check(self.root)

    def test_missing_coverage_entry(self):
        path = self.root / qualified.MANIFEST
        manifest = json.loads(path.read_text())
        del manifest["areas"]["lstm"]
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(
            qualified.QualificationError, "PRODUCTION coverage"
        ):
            qualified.check(self.root)

    def test_changed_coverage_even_if_source_hash_is_updated(self):
        name = "src/inference/cuda/candidate/lstm.rs"
        self.change(name, "Batches::All", "Batches::Only(&[1])")
        path = self.root / qualified.MANIFEST
        manifest = json.loads(path.read_text())
        manifest["areas"]["lstm"]["files"]["host_sources"][name] = qualified.digest(
            self.root / name
        )
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(
            qualified.QualificationError, "coverage differs from code"
        ):
            qualified.check(self.root)

    def test_changed_qualified_batches(self):
        self.change(
            "src/inference/cuda/candidate.rs",
            "[1, 7, 32, 33, 64]",
            "[1, 7, 32, 33, 65]",
        )
        with self.assertRaises(qualified.QualificationError):
            qualified.check(self.root)

    def test_unknown_production_entry_fails_closed(self):
        self.change(
            "src/inference/cuda/implementation.rs",
            "ConvOxide::COVERAGE",
            "UnqualifiedOxide::COVERAGE",
        )
        with self.assertRaisesRegex(qualified.QualificationError, "unknown PRODUCTION"):
            qualified.check(self.root)

    def test_new_source_module_requires_acceptance(self):
        path = self.root / "src/inference/cuda/candidate/lstm/new.rs"
        path.write_text("// new source\n")
        with self.assertRaisesRegex(
            qualified.QualificationError, "qualification file differs.*new.rs"
        ):
            qualified.check(self.root)

    def test_new_ptx_tier_requires_record_evidence(self):
        old = self.root / "src/inference/cuda/ptx/sincnet.sm75.ptx"
        shutil.copyfile(old, old.with_name("sincnet.sm80.ptx"))
        with self.assertRaises(qualified.QualificationError):
            qualified.accept(self.root, self.record_path())

    def record(self):
        area = "sincnet"
        hashes = qualified.files(self.root, area, "sinc")["ptx"]
        path, sha256 = next(iter(hashes.items()))
        check = {"check": "profile", "passed": True}
        child = {
            "target": area,
            "implementation": "Oxide",
            "status": "passed",
            "checks": [check],
            "coverage_declared": {
                "triples": qualified.coverage(self.root, "sinc")["triples"]
            },
            "loaded_ptx": {
                "modules": [
                    {"area": area, "path": path, "sha256": sha256, "tier": "sm75"}
                ]
            },
        }
        return completed(
            {
                "schema": 3,
                "target": area,
                "implementation": "Oxide",
                "status": "passed",
                "accepts_replacement": True,
                "lock_digest": "a" * 64,
                "checks": [{**check, "check": "sm75/profile"}],
                "tiers": {"sm75": child},
            }
        )

    def record_path(self, record=None):
        path = self.root / "record.json"
        path.write_text(json.dumps(self.record() if record is None else record))
        return path

    def test_accept_records_exact_hash_and_verdict(self):
        path = self.record_path()
        entry = qualified.accept(self.root, path)
        self.assertEqual(
            entry["record"], {"name": path.name, "sha256": qualified.digest(path)}
        )
        self.assertEqual(entry["verdict_basis"], {"rule": "accepts_replacement"})
        qualified.check(self.root)

    def test_accept_refuses_ptx_mismatch(self):
        record = self.record()
        record["tiers"]["sm75"]["loaded_ptx"]["modules"][0]["sha256"] = "0" * 64
        before = (self.root / qualified.MANIFEST).read_bytes()
        with self.assertRaisesRegex(qualified.QualificationError, "PTX differs"):
            qualified.accept(self.root, self.record_path(record))
        self.assertEqual((self.root / qualified.MANIFEST).read_bytes(), before)

    def test_accept_refuses_coverage_mismatch(self):
        record = self.record()
        record["tiers"]["sm75"]["coverage_declared"]["triples"].pop()
        with self.assertRaisesRegex(
            qualified.QualificationError, "record coverage differs"
        ):
            qualified.accept(self.root, self.record_path(record))

    def test_unsupported_coverage_syntax_fails_closed(self):
        self.change(
            "src/inference/cuda/candidate/sinc.rs",
            "Batches::All",
            "unrecognized_batches()",
        )
        with self.assertRaisesRegex(
            qualified.QualificationError, "unsupported or empty coverage"
        ):
            qualified.coverage(self.root, "sinc")


class NoiseRule(unittest.TestCase):
    def record(self, candidate_ms=0.9):
        check = {
            "check": "speed:fp32/first/b1/stage",
            "passed": False,
            "blocked": True,
            "reason": "speed: blocked: Library process spread 0.01 exceeds bound 0.003",
        }
        return completed(
            {
                "schema": 3,
                "implementation": "Oxide",
                "status": "blocked",
                "accepts_replacement": False,
                "checks": [{**check, "check": f"sm75/{check['check']}"}],
                "tiers": {
                    "sm75": {
                        "status": "blocked",
                        "checks": [check],
                        "timing": [
                            {
                                "id": "fp32/first/b1/stage",
                                "medians_ms": [1.0, candidate_ms, 1.01, candidate_ms],
                                "library_process_spread_fraction": 0.01,
                                "spread_bound": 0.003,
                                "measurable": False,
                            }
                        ],
                    }
                },
            }
        )

    def test_noise_accepts_with_table(self):
        basis = qualified.verdict(self.record())
        self.assertEqual(basis["rule"], "library_spread_noise")
        self.assertEqual(len(basis["blocked_cases"]), 1)
        self.assertGreater(basis["blocked_cases"][0]["smaller_pair_speedup"], 1.03)

    def test_smaller_pair_must_meet_margin(self):
        with self.assertRaisesRegex(
            qualified.QualificationError, "noise margin too small"
        ):
            qualified.verdict(self.record(0.99))

    def test_exact_margin_passes(self):
        qualified.verdict(self.record(1 / (1 + 3 * (1.01 - 1))))

    def test_hard_failures_and_non_speed_blocks_refused(self):
        for update in [
            {"blocked": False},
            {"check": "layer:parity"},
            {"reason": "speed: candidate slower"},
        ]:
            record = self.record()
            record["tiers"]["sm75"]["checks"][0].update(update)
            record["checks"][0] = {
                **record["tiers"]["sm75"]["checks"][0],
                "check": "sm75/" + record["tiers"]["sm75"]["checks"][0]["check"],
            }
            with (
                self.subTest(update=update),
                self.assertRaises(qualified.QualificationError),
            ):
                qualified.verdict(record)

    def test_contradictory_accepts_flag_refused(self):
        record = self.record()
        record.update(accepts_replacement=True, status="passed")
        with self.assertRaisesRegex(qualified.QualificationError, "contradicts checks"):
            qualified.verdict(record)

    def test_missing_or_inconsistent_timing_refused(self):
        row_changes = [
            {"medians_ms": [1, float("nan"), 1.01, 0.9]},
            {"medians_ms": [1, 0, 1.01, 0.9]},
            {"library_process_spread_fraction": 0.001},
            {"spread_bound": 0.02},
            {"measurable": True},
        ]
        for update in row_changes:
            record = copy.deepcopy(self.record())
            record["tiers"]["sm75"]["timing"][0].update(update)
            with (
                self.subTest(update=update),
                self.assertRaises(qualified.QualificationError),
            ):
                qualified.verdict(record)
        record = self.record()
        record["tiers"]["sm75"]["timing"] = []
        with self.assertRaisesRegex(
            qualified.QualificationError, "missing or duplicate timing"
        ):
            qualified.verdict(record)

    def test_partial_collection_after_speed_block_refused(self):
        record = self.record()
        child = record["tiers"]["sm75"]
        del child["coverage"]
        child["reason"] = "profile process did not complete"
        with self.assertRaisesRegex(
            qualified.QualificationError, "incomplete tier collection"
        ):
            qualified.verdict(record)

    def test_incomplete_sanitizer_driver_refused(self):
        record = self.record()
        record["tiers"]["sm75"]["reason"] = (
            "Compute Sanitizer drivers did not complete: ['Oxide/memcheck']"
        )
        with self.assertRaisesRegex(
            qualified.QualificationError, "incomplete or inconsistent tier verdict"
        ):
            qualified.verdict(record)

    def test_missing_required_check_refused(self):
        record = self.record()
        child = record["tiers"]["sm75"]
        child["checks"] = [
            c for c in child["checks"] if c["check"] != "profile:graph_nodes"
        ]
        record["checks"] = [
            c for c in record["checks"] if c["check"] != "sm75/profile:graph_nodes"
        ]
        with self.assertRaisesRegex(
            qualified.QualificationError, "missing required tier checks"
        ):
            qualified.verdict(record)

    def test_aggregate_cannot_hide_tier_failure(self):
        record = self.record()
        record["checks"] = [{"check": "sm75/speed:fp32/first/b1/stage", "passed": True}]
        with self.assertRaisesRegex(
            qualified.QualificationError, "tier checks missing"
        ):
            qualified.verdict(record)


if __name__ == "__main__":
    unittest.main()
