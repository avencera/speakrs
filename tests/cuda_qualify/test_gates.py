"""Gate regression checks. These do not substitute for GPU mutation proof."""

import copy
import importlib
import json
import math
import shutil
import sqlite3
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/cuda/qualify"))
trace = importlib.import_module("parse_trace")
gates = importlib.import_module("gates")
qualify = importlib.import_module("qualify")
snapshot = importlib.import_module("lock")
scan = importlib.import_module("scan")
verdict = importlib.import_module("verdict")

NONCE = "0123456789abcdef0123456789abcdef"


class Gates(unittest.TestCase):
    def test_exact_floor_and_nonfinite(self):
        self.assertEqual(
            gates.layer_parity([("a", gates.Error(0, 0), gates.Error(0, 0))])[
                "geometric_means"
            ],
            [1, 1],
        )
        for candidate in [
            gates.Error(1e-30, 0),
            gates.Error(0, 1e-30),
            gates.Error(math.nan, 0),
        ]:
            with self.assertRaises(gates.Rejected):
                gates.layer_parity([("a", candidate, gates.Error(0, 0))])
        with self.assertRaises(gates.Rejected):
            gates.error([math.nan], [1])
        with self.assertRaises(gates.Rejected):
            gates.error([1], [0])

    def test_individual_and_geometric_limits(self):
        for candidate in [
            gates.Error(1.100001, 1),
            gates.Error(1, 2.000001),
            gates.Error(1.01, 1.01),
        ]:
            with self.assertRaises(gates.Rejected):
                gates.layer_parity([("a", candidate, gates.Error(1, 1))])
        self.assertEqual(
            gates.layer_parity([("a", gates.Error(1, 1), gates.Error(1, 1))])[
                "geometric_means"
            ],
            [1, 1],
        )

    def test_parity_names_every_failing_case(self):
        good, bad = gates.Error(1, 1), gates.Error(2, 1)
        with self.assertRaises(gates.ParityRejected) as caught:
            gates.layer_parity(
                [("b1", bad, good), ("b32", good, good), ("b33", bad, good)]
            )
        self.assertEqual(caught.exception.failing, ["b1", "b33"])
        self.assertTrue(str(caught.exception).startswith("layer parity"))

    def test_stage_limits(self):
        gates.embedding_parity(0.999 - 1e-9, 0.999)
        with self.assertRaises(gates.Rejected):
            gates.embedding_parity(0.998, 0.999)
        gates.segmentation_parity(gates.Error(1, 1), gates.Error(1, 1), 0, 0)
        with self.assertRaises(gates.Rejected):
            gates.segmentation_parity(gates.Error(1, 1), gates.Error(1, 1), 1, 0)

    def test_bitwise_not_float_equality(self):
        gates.determinism(b"\0" * 4, b"\0" * 4)
        with self.assertRaises(gates.Rejected):
            gates.determinism(b"\0" * 4, b"\0\0\0\x80")

    def runs(self, values):
        return [
            gates.Timing(i + 1, name, 5, (value,) * 20)
            for i, (name, value) in enumerate(values)
        ]

    def test_candidate_must_beat_the_faster_library_process(self):
        quiet = [("Library", 1.0), ("c", 0.999), ("Library", 1.002), ("c", 0.998)]
        self.assertEqual(
            gates.speed(self.runs(quiet), "c", 0.003)["spread_bound"], 0.003
        )
        # both pairs win, but the candidate is slower than the faster Library process
        slower = [("Library", 1.0), ("c", 1.001), ("Library", 1.002), ("c", 0.999)]
        with self.assertRaisesRegex(gates.Rejected, "candidate slower"):
            gates.speed(self.runs(slower), "c", 0.003)

    def test_noisy_library_blocks_instead_of_passing(self):
        noisy = [("Library", 1.0), ("c", 0.5), ("Library", 1.02), ("c", 0.5)]
        with self.assertRaises(gates.Blocked):
            gates.speed(self.runs(noisy), "c", 0.003)
        gates.speed(self.runs(noisy), "c", 0.03)
        self.assertEqual(gates.spread_bound("fp32/first/b1/stage"), 0.003)
        self.assertEqual(gates.spread_bound("fp32/first/b1/lstm.stack"), 0.010)

    def paired_row(self, candidate=0.999, saving=0.003):
        return {
            "order": "ABBA",
            "warmup": 5,
            "stage_abba_ms": [[1.0, candidate, candidate, 1.0] for _ in range(256)],
            "operator_abba_ms": [
                [0.01, 0.01 - saving, 0.01 - saving, 0.01] for _ in range(256)
            ],
        }

    def test_stage_paired_guard_detects_regression_and_needs_pairs(self):
        row = self.paired_row()
        result = gates.stage_speed(row, False)
        self.assertGreater(result["one_sided95_lower"], 1.0)
        self.assertEqual(result["acceptance_path"], "primary")
        with self.assertRaisesRegex(gates.Rejected, "stage regression"):
            gates.stage_speed(self.paired_row(1.004), True)
        row["stage_abba_ms"].pop()
        with self.assertRaisesRegex(gates.Rejected, "insufficient"):
            gates.stage_speed(row, True)

    def test_tiny_operator_requires_resolving_its_saving_and_operator_gate(self):
        row = self.paired_row(1.0, 0.003)
        for i, block in enumerate(row["stage_abba_ms"]):
            block[1] = block[2] = 0.999 + (0.004 if (i // 32) % 2 else -0.004)
        result = gates.stage_speed(row, True)
        self.assertEqual(result["acceptance_path"], "margin")
        self.assertLess(result["one_sided95_lower"], 1)
        self.assertGreaterEqual(
            result["one_sided95_lower"], result["non_inferiority_ratio_threshold"]
        )
        with self.assertRaises(gates.Blocked):
            gates.stage_speed(row, False)
        row["operator_abba_ms"] = [[0.01, 0.00999, 0.00999, 0.01] for _ in range(256)]
        with self.assertRaisesRegex(gates.StageRejected, "non-inferiority margin"):
            gates.stage_speed(row, True)

    def test_grok_asymmetric_tail_does_not_fit_the_time_margin(self):
        # one 16-replay group at 2.5x Library time, the rest at 9ms: exact mean 10ms
        row = self.paired_row()
        row["stage_abba_ms"] = [[10, 25, 25, 10]] * 16 + [[10, 9, 9, 10]] * 240
        row["operator_abba_ms"] = [[2, 0.5, 0.5, 2]] * 256
        with self.assertRaisesRegex(
            gates.StageRejected, "non-inferiority margin"
        ) as caught:
            gates.stage_speed(row, True)
        evidence = caught.exception.evidence
        self.assertEqual(evidence["ratio"], 1)
        self.assertAlmostEqual(evidence["operator_saving_fraction"], 0.15)
        self.assertAlmostEqual(evidence["ci95_diagnostic"][0], 5 / 6)
        self.assertAlmostEqual(evidence["ci95_diagnostic"][1], 10 / 9)
        self.assertLess(evidence["one_sided95_lower"], 1 / 1.15)
        self.assertEqual(evidence["acceptance_path"], "margin")

    def test_bootstrap_does_not_split_measured_operator_strata(self):
        row = self.paired_row(1)
        row["operator_layers"] = ["one", "two", "three", "four"]
        row["operator_layer_by_block"] = [
            layer for layer in row["operator_layers"] for _ in range(64)
        ]
        result = gates.stage_speed(row, True)
        self.assertEqual(result["bootstrap_design"], "whole strata")
        self.assertEqual(result["bootstrap_block_abba"], 64)
        self.assertEqual(result["bootstrap_groups"], 4)

    def test_recorded_replay_fixture_with_one_scaled_group_fails_margin(self):
        # this fixture scales recorded data only; it is not a measured GPU mutant
        fixture = json.loads(
            (Path(__file__).parent / "stage_margin_fixture.json").read_text()
        )
        row = fixture["row"]
        start = fixture["group_start"]
        for block in row["stage_abba_ms"][start : start + fixture["group_blocks"]]:
            block[1] *= fixture["candidate_group_scale"]
            block[2] *= fixture["candidate_group_scale"]
        with self.assertRaisesRegex(
            gates.StageRejected, "non-inferiority margin"
        ) as caught:
            gates.stage_speed(row, True)
        evidence = caught.exception.evidence
        self.assertGreaterEqual(evidence["ratio"], 1)
        self.assertGreater(evidence["operator_saving_fraction"], 0)
        self.assertLess(
            evidence["one_sided95_lower"], evidence["non_inferiority_ratio_threshold"]
        )
        self.assertEqual(evidence["bootstrap_block_abba"], fixture["group_blocks"])
        self.assertEqual(evidence["acceptance_path"], "margin")

    def test_stratified_saving_is_the_sum_not_the_mean_and_needs_each_layer(self):
        row = self.paired_row(1.0)
        row["operator_layers"] = ["one", "two"]
        row["operator_layer_by_block"] = ["one"] * 128 + ["two"] * 128
        row["operator_abba_ms"] = [[0.02, 0.01, 0.01, 0.02]] * 128 + [
            [0.02, 0.019, 0.019, 0.02]
        ] * 128
        result = gates.stage_speed(row, True)
        self.assertAlmostEqual(result["operator_saving_ms"], 0.011)
        self.assertEqual(set(result["operator_saving_by_layer_ms"]), {"one", "two"})
        row["operator_layer_by_block"][0] = "two"
        with self.assertRaisesRegex(gates.Rejected, "operator strata"):
            gates.stage_speed(row, True)

    def test_tf32_truth_rejects_each_error_and_missing_draws(self):
        library = {"minimum_cosine": 0.9999, "relative_l2": 0.001, "max_abs": 0.01}
        gates.tf32_truth(library, library, [library] * 8)
        better = {"minimum_cosine": 1.0, "relative_l2": 0.0, "max_abs": 0.0}
        gates.tf32_truth(better, library, [library] * 8)
        for key, value in [
            ("minimum_cosine", 0.9998),
            ("relative_l2", 0.0011),
            ("max_abs", 0.011),
        ]:
            with (
                self.subTest(key=key),
                self.assertRaisesRegex(gates.Rejected, "less accurate"),
            ):
                gates.tf32_truth({**library, key: value}, library, [library] * 8)
        with self.assertRaisesRegex(gates.Rejected, "8 independent"):
            gates.tf32_truth(library, library, [library] * 7)

    def test_tf32_candidate_equal_to_library_passes_when_draws_are_more_accurate(self):
        library = {"minimum_cosine": 0.9998, "relative_l2": 0.002, "max_abs": 0.02}
        draw = {"minimum_cosine": 0.9999, "relative_l2": 0.001, "max_abs": 0.01}
        evidence = gates.tf32_truth(library, library, [draw] * 8)
        for key in library:
            component = evidence["bound_components"][key]
            self.assertEqual(component["candidate"], component["unperturbed_library"])
            self.assertEqual(component["maximum"], component["unperturbed_library"])
            self.assertEqual(len(component["draws"]), 8)
            self.assertLess(component["upper95_diagnostic"], component["maximum"])

    def test_tf32_slightly_worse_than_maximum_draw_fails_and_retains_components(self):
        library = {"minimum_cosine": 0.9999, "relative_l2": 0.001, "max_abs": 0.01}
        draw = {**library, "relative_l2": 0.002}
        candidate = {**draw, "relative_l2": 0.00200000001}
        with self.assertRaises(gates.TruthRejected) as raised:
            gates.tf32_truth(candidate, library, [library] * 7 + [draw])
        evidence = raised.exception.evidence
        self.assertEqual(evidence["error_limits"]["relative_l2"], 0.002)
        self.assertEqual(
            evidence["bound_components"]["relative_l2"]["draws"], [0.001] * 7 + [0.002]
        )
        result = {"checks": []}
        qualify.check(
            result,
            "stage_truth:case",
            lambda: gates.tf32_truth(candidate, library, [library] * 7 + [draw]),
        )
        self.assertFalse(result["checks"][0]["passed"])
        self.assertEqual(result["checks"][0]["evidence"], evidence)

    def test_stage_max_abs_only_failure_rejects_all_production_tuples(self):
        library = {"minimum_cosine": 0.9999, "relative_l2": 0.001, "max_abs": 0.01}
        draw = {**library, "max_abs": 0.02}
        candidate = {**library, "max_abs": math.nextafter(0.02, math.inf)}
        child: dict = {
            "checks": [{"check": "all_other_checks", "passed": True}],
            "coverage_declared": {"triples": [["boundary", 1, "tf32"]]},
            "accepted_tuples": [["boundary", 1, "tf32"]],
        }
        qualify.check(
            child,
            "stage_truth:tf32/first/b1/stage",
            lambda: gates.tf32_truth(candidate, library, [library] * 7 + [draw]),
        )
        failed: dict = child["checks"][-1]
        self.assertFalse(failed["passed"])
        evidence = failed["evidence"]
        assert isinstance(evidence, dict)
        limits: dict = evidence["error_limits"]
        self.assertEqual(limits["max_abs"], 0.02)
        qualify.finish_tier(child, [])
        self.assertEqual(child["status"], "rejected")
        self.assertEqual(
            child["verdict_evaluation"]["hard_failures"],
            ["stage_truth:tf32/first/b1/stage"],
        )
        # a forged passed label cannot turn the retained hard failure into coverage
        record = {
            "schema": 4,
            "implementation": "Oxide",
            "status": "passed",
            "checks": [{"check": "static_scan", "passed": True}],
            "tiers": {"sm75": {**child, "status": "passed"}},
        }
        decision = verdict.evaluate_record(record, "sm75")
        self.assertFalse(decision["accepted"])
        self.assertEqual(decision["accepted_tuples"], [])

    def test_speed_requires_fresh_processes_and_samples(self):
        runs = self.runs([("Library", 1), ("c", 0.99), ("Library", 1), ("c", 0.99)])
        for replacement in [
            gates.Timing(4, "c", 4, (1,) * 20),
            gates.Timing(4, "c", 5, (1,) * 19),
            gates.Timing(1, "c", 5, (1,) * 20),
        ]:
            with self.assertRaises(gates.Rejected):
                gates.speed([*runs[:3], replacement], "c", 0.003)

    def band_rows(self, drops):
        base = {"minimum_cosine": 0.9999, "mean_cosine": 0.99995}
        return [
            {
                "metrics": base,
                "band": [
                    {"minimum_cosine": base["minimum_cosine"] - drop}
                    for drop in [drop] * 8
                ],
            }
            for drop in drops
        ]

    def test_tf32_band_is_the_largest_perturbed_drop(self):
        band = gates.tf32_band(self.band_rows([1e-7, 3e-7]), True)
        self.assertAlmostEqual(band["cosine"], 3e-7)
        with self.assertRaisesRegex(gates.Rejected, "8 seeds"):
            rows = self.band_rows([1e-7])
            rows[0]["band"].pop()
            gates.tf32_band(rows, True)

    def test_tf32_segmentation_band_bounds_errors_and_flips(self):
        library = {"relative_l2": 0.01, "max_abs": 0.1, "argmax_flips": 2}
        rows = [
            {
                "metrics": library,
                "band": [{"relative_l2": 0.011, "max_abs": 0.12, "argmax_flips": 3}]
                * 8,
            }
        ]
        band = gates.tf32_band(rows, False)
        self.assertEqual(band["flips"], 1)
        self.assertEqual(band["total_flips"], 1)

    def test_completed_sanitizer_required(self):
        gates.sanitizer("memcheck", 0, "========= ERROR SUMMARY: 0 errors")
        gates.sanitizer(
            "racecheck", 0, "========= RACECHECK SUMMARY: 0 hazards displayed"
        )
        for code, log in [
            (0, ""),
            (0, "ERROR SUMMARY: 1 errors"),
            (86, "ERROR SUMMARY: 0 errors"),
            (0, "fatal permission\nERROR SUMMARY: 0 errors"),
        ]:
            with self.assertRaises(gates.Rejected):
                gates.sanitizer("initcheck", code, log)


class Lock(unittest.TestCase):
    def test_lock_add_edit_delete_scope_and_owner_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scripts/cuda/qualify").mkdir(parents=True)
            (root / "src/inference/cuda/candidate").mkdir(parents=True)
            (root / snapshot.SCOPE).write_text(
                json.dumps(
                    {
                        "schema": 1,
                        "directories": [
                            {"path": "scripts/cuda/qualify", "exclude": []},
                            {"path": "src/inference/cuda", "exclude": ["candidate"]},
                        ],
                        "files": ["Cargo.toml"],
                    }
                )
            )
            (root / "Cargo.toml").write_text("[package]\n")
            dispatch = root / "src/inference/cuda/dispatch.rs"
            dispatch.write_text("locked\n")
            candidate = root / "src/inference/cuda/candidate/conv.rs"
            candidate.write_text("candidate\n")
            lock = root / "scripts/cuda/qualify/LOCK"
            lock.write_bytes(snapshot.canonical(snapshot.inventory(root)))
            digest = snapshot.verify(root)
            self.assertEqual(snapshot.verify(root, digest), digest)
            with self.assertRaises(snapshot.LockError):
                snapshot.verify(root, "0" * 64)
            # candidate code is outside the lock
            candidate.write_text("changed candidate\n")
            self.assertEqual(snapshot.verify(root), digest)
            cache = root / "scripts/cuda/qualify/__pycache__"
            cache.mkdir()
            (cache / "ignored.pyc").write_bytes(b"cached bytes")
            self.assertEqual(snapshot.verify(root), digest)
            (cache / "bypass.py").write_text("executable source")
            with self.assertRaises(snapshot.LockError):
                snapshot.verify(root)
            (cache / "bypass.py").unlink()
            added = root / "src/inference/cuda/extra.rs"
            added.write_text("added")
            with self.assertRaises(snapshot.LockError):
                snapshot.verify(root)
            added.unlink()
            dispatch.write_text("edited")
            with self.assertRaises(snapshot.LockError):
                snapshot.verify(root)
            dispatch.unlink()
            with self.assertRaises(snapshot.LockError):
                snapshot.verify(root)

    @unittest.skipUnless(
        shutil.which("cargo") and (shutil.which("ruff") or shutil.which("uv")),
        "formatters are not installed",
    )
    def test_locked_files_are_formatter_stable(self):
        snapshot.formatter_stable(ROOT)

    def test_scope_covers_dispatch_seams_and_tier_features(self):
        files = snapshot.inventory(ROOT)
        for path in (
            "src/inference/cuda/candidate.rs",
            "src/inference/cuda/implementation.rs",
            "src/inference/cuda/embedding/dispatch.rs",
            "src/inference/cuda/segmentation/dispatch.rs",
            "src/inference/cuda/embedding.rs",
            "src/inference/cuda/segmentation.rs",
            "src/inference/cuda/kernels.rs",
            "src/inference/cuda/runtime.rs",
            "src/inference/cuda/test_support.rs",
            "Cargo.toml",
            ".cargo/config.toml",
        ):
            self.assertIn(path, files)
        self.assertFalse(
            any(path.startswith("src/inference/cuda/candidate/") for path in files)
        )
        self.assertFalse(
            any(
                path.startswith(
                    (
                        "src/inference/cuda/ptx/resnet.",
                        "src/inference/cuda/ptx/lstm.",
                        "src/inference/cuda/ptx/sincnet.",
                    )
                )
                for path in files
            )
        )
        for path in (
            "src/lib.rs",
            "src/inference/cuda/ptx/segmentation.sm75.ptx",
            "src/inference/cuda/tests.rs",
            "Cargo.lock",
        ):
            self.assertIn(path, files)


class Scan(unittest.TestCase):
    def test_phase_reading_candidate_is_refused(self):
        source = (ROOT / "tests/cuda_qualify/scan_fixtures/phase_cheat.rs").read_text()
        reasons = {
            finding.reason
            for finding in scan.scan_text("lstm.rs", source, scan.HOST_RULES)
        }
        self.assertIn(
            "environment, file, network, process, thread or clock API", reasons
        )
        self.assertIn("harness or reference path", reasons)

    def test_committed_placeholders_and_tree_pass(self):
        result = scan.scan(ROOT, None)
        self.assertEqual(result["findings"], [])
        self.assertIn("src/inference/cuda/candidate/lstm.rs", result["host_files"])

    def test_candidate_plans_cannot_reload_modules(self):
        relative = "src/inference/cuda/candidate/fixture.rs"
        good = "fn plan(runtime: &CudaRuntime, kernels: &LoadedKernels) { kernels.function(ENTRY)?; }"
        self.assertEqual(
            scan.scan_text(relative, good, scan.HOST_RULES, resolve_calls=True), []
        )
        for load in (
            "runtime.load_kernels(KernelModule::Lstm)",
            "runtime.load_module(request)",
            "load_artifact(request, bytes, loader)",
            "let loader = runtime.load_kernels; loader(KernelModule::Lstm)",
            "use driver::load_ptx as restore; restore(bytes)",
            "cuModuleLoadData(&mut module, bytes)",
        ):
            code = good.replace("kernels.function(ENTRY)?", load)
            with self.subTest(load=load):
                findings = scan.scan_text(
                    relative, code, scan.HOST_RULES, resolve_calls=True
                )
                self.assertTrue(
                    any(
                        "preloaded LoadedKernels" in finding.reason
                        for finding in findings
                    )
                )
        for name in ("conv", "lstm", "sinc"):
            path = f"src/inference/cuda/candidate/{name}.rs"
            self.assertEqual(
                scan.scan_text(
                    path, (ROOT / path).read_text(), scan.HOST_RULES, resolve_calls=True
                ),
                [],
            )

    def test_each_rule_refuses_its_construct(self):
        refused = {
            'let p = std::env::var("X");': "environment",
            "use std::{fs, io};": "file",
            'let s = env!("HOME");': "env!",
            "static mut COUNT: u32 = 0;": "static",
            "struct P(std::cell::Cell<u32>);": "interior-mutable",
            'test_support::range("x");': "harness call",
            "runtime.sgemm(spec, a, b, c);": "cuDNN or cuBLAS",
            'extern "C" { fn f(); }': "foreign code",
            "if stream.capture_status() {}": "capture",
            "#[cfg(test)] fn f() {}": "test-only",
            "let t = Instant::now();": "clock",
            'include_bytes!("x.ptx");': "foreign code",
        }
        for code, reason in refused.items():
            with self.subTest(code=code):
                findings = scan.scan_text("x.rs", code, scan.HOST_RULES)
                self.assertTrue(any(reason in f.reason for f in findings), findings)
        allowed = [
            'let d = env!("CARGO_MANIFEST_DIR");',
            "// std::env::var is mentioned only in a comment",
            'let q = \'"\'; let r = "text // not a comment";',
            "unsafe { launch.launch(config) }?;",
            'fn name(&self) -> &\'static str { "x" }',
            "struct S<T: 'static>(T);",
            "use std::sync::Arc;\nuse cudarc::driver::CudaStream;",
            "use std::{cell::Ref, sync::Arc};",
        ]
        for code in allowed:
            with self.subTest(code=code):
                self.assertEqual(scan.scan_text("x.rs", code, scan.HOST_RULES), [])

    def test_candidate_calls_resolve_to_the_tree_or_the_locked_interface(self):
        relative = "src/inference/cuda/candidate/lstm/kernel.rs"
        allowed = [
            "use super::{Coverage, LstmPhases};\nuse super::layout::pack;",
            "use crate::inference::cuda::{CudaError, CudaMath, CudaRuntime, KernelModule};",
            "use crate::inference::cuda::ComputeCapability as CC; let cc = CC::new(12, 0);",
            "use crate::inference::cuda::device::DeviceAttributes as Device; fn sm(d: &Device) { d.multiprocessors(); }",
            "use crate::inference::cuda::candidate::{ConfigPin, ConvPin, ConvKernel, FbankSpec}; let pin = ConfigPin::Conv(ConvPin::Kernel(ConvKernel::C64));",
            "use crate::inference::cuda::geometry::Conv2d;\nuse crate::inference::cuda::error::check_len;",
            'let e = crate::inference::cuda::CudaError::Unsupported { context: "x", reason: r };',
            "use cudarc::driver::{CudaStream, LaunchConfig, PushKernelArg};",
            "use cudarc::driver::sys::CUfunction_attribute;",
            "use std::sync::Arc;\nuse core::mem::size_of;",
            "let module = KernelModule::Lstm;",
        ]
        for code in allowed:
            with self.subTest(code=code):
                self.assertEqual(
                    scan.scan_text(relative, code, scan.HOST_RULES, resolve_calls=True),
                    [],
                )
        refused = [
            "use crate::inference::cuda::implementation::TupleProof;",
            "crate::inference::cuda::test_support::phase();",
            "use crate::inference::cuda::SafetensorsFile;\nSafetensorsFile::open(path);",
            "let f = crate::inference::cuda::weights::SafetensorsFile::open(p);",
            "use super::super::super::weights::SafetensorsFile;",
            "crate::helper::enqueue_conv(stream);",
            "use crate::inference::cuda::*;",
            "let r = crate::inference::cuda::CudaRuntime::new(0);",
            "use cudarc::driver::CudaContext;",
            "use cudarc::driver::sys::cuMemcpyDtoH_v2;",
            "use safetensors::SafeTensors;",
            "use std::sync::Mutex;",
            "let s = stream.capture_status();",
        ]
        for code in refused:
            with self.subTest(code=code):
                self.assertTrue(
                    scan.scan_text(relative, code, scan.HOST_RULES, resolve_calls=True)
                )

    def test_unscanned_call_fixture_is_refused(self):
        source = (
            ROOT / "tests/cuda_qualify/scan_fixtures/unscanned_call.rs"
        ).read_text()
        findings = scan.scan_text(
            "src/inference/cuda/candidate/lstm.rs",
            source,
            scan.HOST_RULES,
            resolve_calls=True,
        )
        self.assertTrue(
            any("outside the candidate tree" in finding.reason for finding in findings)
        )

    def test_unlocked_rust_outside_the_candidate_tree_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scripts/cuda/qualify").mkdir(parents=True)
            (root / "src/inference/cuda/candidate").mkdir(parents=True)
            (root / snapshot.SCOPE).write_text(
                json.dumps(
                    {
                        "schema": 1,
                        "directories": [
                            {
                                "path": "src",
                                "exclude": ["inference/cuda/candidate", "helper.rs"],
                            }
                        ],
                        "files": [],
                    }
                )
            )
            (root / "src/lib.rs").write_text("mod helper;")
            (root / "src/helper.rs").write_text("pub fn f() {}")
            (root / "src/inference/cuda/candidate/conv.rs").write_text(
                "pub struct Oxide;"
            )
            reasons = {str(finding) for finding in scan.scan(root, None)["findings"]}
            self.assertTrue(any("src/helper.rs" in reason for reason in reasons))
            self.assertFalse(any("candidate/conv.rs" in reason for reason in reasons))

    def test_device_rules_allow_shared_memory_only(self):
        ok = "static mut TILE: SharedArray<f32, 256> = SharedArray::UNINIT;"
        self.assertEqual(scan.scan_text("k.rs", ok, scan.DEVICE_RULES), [])
        for bad in ("static COUNTER: u32 = 0;", "static mut X: u32 = 0;"):
            self.assertTrue(scan.scan_text("k.rs", bad, scan.DEVICE_RULES), bad)

    def test_device_code_may_use_cuda_device_thread(self):
        allowed = [
            "use cuda_device::{DisjointSlice, kernel, thread, warp};\nlet i = thread::index_1d();",
            "use cuda_device::thread as t;\nt::sync_threads();",
            "fn f(x: &'static [f32]) {}",
        ]
        for code in allowed:
            with self.subTest(code=code):
                self.assertEqual(scan.scan_text("k.rs", code, scan.DEVICE_RULES), [])
        for code in ("use std::thread;", "use std::thread as t;\nt::spawn(f);"):
            with self.subTest(code=code):
                self.assertTrue(scan.scan_text("k.rs", code, scan.DEVICE_RULES))

    def test_crate_rules_refuse_code_that_runs_uncalled(self):
        code = (
            '#[used]\n#[unsafe(link_section = ".init_array")]\nstatic INIT: fn() = f;'
        )
        self.assertTrue(scan.scan_text("lib.rs", code, scan.CRATE_RULES))

    def test_build_scripts_and_foreign_cargo_config_are_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "tree"
            root.mkdir()
            self.assertEqual(scan.build_inputs(root, None), [])
            (root / "build.rs").write_text("fn main() {}")
            (Path(directory) / ".cargo").mkdir()
            (Path(directory) / ".cargo/config.toml").write_text("[build]")
            reasons = {finding.reason for finding in scan.build_inputs(root, None)}
            self.assertEqual(reasons, {"build script", "Cargo configuration"})


class Trace(unittest.TestCase):
    ALLOW = trace.AllowList(
        frozenset({"resnet_conv", "embedding_bias", "qualify_round"}),
        frozenset({"resnet_conv"}),
    )
    LAYER = "resnet.layer1.0.conv1"

    def build(self, path, launches, ranges, nonce=NONCE):
        """`launches`: (start, kernel name, stream, kind, copy kind)"""
        with closing(sqlite3.connect(path)) as c, c:
            c.executescript(
                "CREATE TABLE StringIds (id INTEGER, value TEXT);"
                "CREATE TABLE NVTX_EVENTS (start INTEGER, end INTEGER, globalTid INTEGER, endGlobalTid INTEGER, text TEXT, textId INTEGER);"
                "CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (start INTEGER, end INTEGER, globalTid INTEGER, correlationId INTEGER);"
                "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER, globalPid INTEGER, correlationId INTEGER, demangledName INTEGER, mangledName INTEGER, graphNodeId INTEGER, streamId INTEGER);"
                "CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (start INTEGER, end INTEGER, globalPid INTEGER, correlationId INTEGER, copyKind INTEGER, streamId INTEGER);"
            )
            for start, end, kind, name in ranges:
                c.execute(
                    "INSERT INTO NVTX_EVENTS VALUES(?,?,16777217,NULL,?,NULL)",
                    [start, end, f"qualify.{nonce}.{kind}.{name}"],
                )
            for correlation, (start, name, stream, kind, copy_kind) in enumerate(
                launches
            ):
                c.execute(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(?,?,16777217,?)",
                    [start, start + 1, correlation],
                )
                if kind == "copy":
                    c.execute(
                        "INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES(?,?,16777216,?,?,?)",
                        [
                            5000 + correlation,
                            5001 + correlation,
                            correlation,
                            copy_kind,
                            stream,
                        ],
                    )
                    continue
                c.execute(
                    "INSERT INTO StringIds VALUES(?, ?)", [100 + correlation, name]
                )
                c.execute(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,?,16777216,?,?,?,?,?)",
                    [
                        5000 + correlation,
                        5010 + correlation,
                        correlation,
                        100 + correlation,
                        100 + correlation,
                        77 if kind == "graph" else None,
                        stream,
                    ],
                )

    def good(self):
        ranges = [
            (10, 100, "window", "case"),
            (20, 40, "candidate", self.LAYER),
            (50, 70, "fixed", "embedding_bias"),
            (110, 200, "window", "library-case"),
            (120, 150, "library", self.LAYER),
            (125, 140, "call", "cudnn.conv"),
        ]
        launches = [
            (25, "resnet_conv", 7, "kernel", None),
            (55, "embedding_bias", 7, "kernel", None),
            (130, "sm80_xmma_fprop_cudnn", 7, "kernel", None),
        ]
        return launches, ranges

    def attribute(self, path, declared=frozenset({LAYER}), library=False):
        return trace.attribute(
            path, (self.LAYER,), NONCE, self.ALLOW, declared, library
        )

    def run_case(self, change):
        launches, ranges = self.good()
        change(launches, ranges)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.sqlite"
            self.build(path, launches, ranges)
            return self.attribute(path)

    def receipt(self, result: dict, index: int) -> dict:
        value = result["profile_traces"][index]["receipt"]
        if not isinstance(value, dict):
            self.fail("each produced trace must have a check receipt")
        return value

    def test_good_candidate_trace_passes(self):
        result = self.run_case(lambda launches, ranges: None)
        self.assertEqual(result["windows"], 2)

    def test_library_kernel_in_candidate_scope_is_forbidden(self):
        def change(launches, ranges):
            ranges.append((22, 30, "call", "cudnn.conv"))
            launches.append((24, "implicit_convolve_sgemm", 7, "kernel", None))

        with self.assertRaisesRegex(trace.Rejected, "forbidden library kernels"):
            self.run_case(change)

    def test_work_outside_scopes_inside_a_window_is_refused(self):
        def change(launches, ranges):
            launches.append((45, "embedding_bias", 7, "kernel", None))

        with self.assertRaisesRegex(
            trace.Rejected, "outside a candidate or library range"
        ):
            self.run_case(change)

    def test_unlisted_kernel_is_refused(self):
        def change(launches, ranges):
            launches.append((26, "qualify_unlisted", 7, "kernel", None))

        with self.assertRaisesRegex(trace.Rejected, "not on the loaded PTX allow-list"):
            self.run_case(change)

    def test_unlisted_kernel_in_a_candidate_plan_is_refused(self):
        def change(launches, ranges):
            ranges.append((1, 4, "plan", self.LAYER))
            launches.append((2, "warmup_cache_kernel", 7, "kernel", None))

        with self.assertRaisesRegex(trace.Rejected, "not on the loaded PTX allow-list"):
            self.run_case(change)

    def test_range_with_another_nonce_is_refused(self):
        launches, ranges = self.good()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.sqlite"
            self.build(path, launches, ranges, nonce="f" * 32)
            with self.assertRaisesRegex(trace.Rejected, "nonce"):
                self.attribute(path)

    def test_host_copy_in_candidate_scope_is_refused(self):
        def change(launches, ranges):
            launches.append((27, None, 7, "copy", 2))

        with self.assertRaisesRegex(trace.Rejected, "host transfer"):
            self.run_case(change)

    def test_candidate_kernel_on_a_library_path_is_refused(self):
        def change(launches, ranges):
            ranges.append((141, 149, "fixed", "resnet_conv"))
            launches.append((145, "resnet_conv", 7, "kernel", None))

        with self.assertRaisesRegex(
            trace.Rejected, "candidate kernel on a Library path"
        ):
            self.run_case(change)

    def test_side_stream_must_carry_only_candidate_work(self):
        def exclusive(launches, ranges):
            launches.append((28, "resnet_conv", 8, "kernel", None))

        self.run_case(exclusive)

        def shared(launches, ranges):
            launches.append((28, "resnet_conv", 8, "kernel", None))
            launches.append((3, "setup_kernel_outside_windows", 8, "kernel", None))

        with self.assertRaisesRegex(trace.Rejected, "another stream"):
            self.run_case(shared)

    def test_graph_nodes_never_pass(self):
        def change(launches, ranges):
            launches.append((29, "resnet_conv", 7, "graph", None))

        with self.assertRaisesRegex(trace.Rejected, "graph node"):
            self.run_case(change)

    def test_every_first_use_api_position_has_the_exact_profile_gate(self):
        for position in (
            "plan",
            "warmup/0",
            "eager",
            "capture",
            "first-replay",
            "fresh",
        ):
            launches, ranges = self.good()
            ranges.extend(
                [
                    (210, 300, "window", f"lifecycle/case/{position}"),
                    (220, 280, "candidate", self.LAYER),
                    (230, 240, "call", "cublas.m1.n1.k1"),
                ]
            )
            # capture may record no eager kernel event; the real API scope still belongs to the candidate
            if position != "capture":
                launches.append((235, "cublas_single", 7, "kernel", None))
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "trace.sqlite"
                self.build(path, launches, ranges)
                with self.assertRaises(trace.Rejected) as rejected:
                    self.attribute(path)
                self.assertTrue(
                    qualify.MUTANT_GATES["FirstUseFallbackCaptured"].matches(
                        {
                            "check": "profile",
                            "reason": str(rejected.exception),
                        }
                    ),
                    position,
                )

    def test_outside_window_work_is_not_silently_skipped(self):
        for kernel in ("cublas_single", "unknown_setup_kernel"):
            with self.subTest(kernel=kernel), self.assertRaises(trace.Rejected):
                self.run_case(
                    lambda launches, ranges: launches.append(
                        (3, kernel, 7, "kernel", None)
                    )
                )
        with self.assertRaisesRegex(
            trace.Rejected, "outside a checked lifecycle window"
        ):
            self.run_case(
                lambda launches, ranges: (
                    ranges.append((210, 250, "candidate", self.LAYER)),
                    launches.append((220, "resnet_conv", 7, "kernel", None)),
                )
            )

    def test_side_stream_does_not_hide_unread_captured_first_use_fault(self):
        with tempfile.TemporaryDirectory() as directory:
            eager = Path(directory) / "profile-eager.sqlite"
            short = Path(directory) / "profile.sqlite"
            launches, ranges = self.good()
            launches.append((28, "resnet_conv", 8, "kernel", None))
            self.build(eager, launches, ranges)
            ranges.extend(
                [
                    (210, 300, "window", "lifecycle/case/first-replay"),
                    (220, 280, "candidate", self.LAYER),
                    (230, 240, "call", "cublas.m1.n1.k1"),
                ]
            )
            launches.append((235, "cublas_single", 7, "kernel", None))
            self.build(short, launches, ranges)
            result: dict = {
                "profile_traces": [
                    {
                        "label": name,
                        "path": str(path),
                        "sha256": qualify.sha(path),
                        "receipt": None,
                    }
                    for name, path in (("profile", short), ("profile-eager", eager))
                ]
            }
            result["profile_traces"][0]["retained_for_eager"] = True
            self.assertNotIn("short_profile", result)
            with self.assertRaisesRegex(qualify.Rejected, "not checked"):
                qualify.profile_trace_receipts(result)
            with self.assertRaisesRegex(qualify.Rejected, "forbidden library kernels"):
                qualify.check_profile_traces(
                    result,
                    (self.LAYER,),
                    NONCE,
                    self.ALLOW,
                    frozenset({self.LAYER}),
                    False,
                )
            self.assertFalse(self.receipt(result, 0)["passed"])
            self.assertIs(result["short_profile"], result["profile_traces"][0])
            self.assertTrue(self.receipt(result, 1)["passed"])
            self.assertEqual(self.attribute(eager)["windows"], 2)

    def test_fresh_input_exposes_a_content_keyed_library_fallback(self):
        launches, ranges = self.good()
        ranges.extend(
            [
                (210, 300, "window", "lifecycle/secret/fp32/b1/fresh"),
                (220, 280, "candidate", self.LAYER),
            ]
        )
        pinned = (1.0, 2.0)
        fresh = (0.5, 3.0)
        calls = []

        def enqueue(values):
            if values != pinned:
                calls.append(values)
                ranges.append((230, 240, "call", "cublas.content_miss"))
                launches.append((235, "cublas_single", 7, "kernel", None))

        enqueue(pinned)
        self.assertEqual(calls, [])
        enqueue(fresh)
        self.assertEqual(calls, [fresh])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.sqlite"
            self.build(path, launches, ranges)
            with self.assertRaisesRegex(trace.Rejected, "forbidden library kernels"):
                self.attribute(path)

    def test_invalid_sqlite_has_a_failed_trace_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.sqlite"
            path.write_bytes(b"not sqlite")
            result: dict = {
                "profile_traces": [
                    {
                        "label": "bad",
                        "path": str(path),
                        "sha256": qualify.sha(path),
                        "receipt": None,
                    }
                ]
            }
            with self.assertRaisesRegex(qualify.Rejected, "invalid SQLite"):
                qualify.check_profile_traces(
                    result,
                    (self.LAYER,),
                    NONCE,
                    self.ALLOW,
                    frozenset({self.LAYER}),
                    False,
                )
            self.assertFalse(self.receipt(result, 0)["passed"])

    def test_fresh_coverage_requires_an_actual_correlated_kernel(self):
        name = f"lifecycle/secret/fp32/{self.LAYER}/b1/fresh"
        coverage = qualify.Coverage.product((self.LAYER,), (1,), ("fp32",))
        self.assertEqual(
            qualify.fresh_profile_windows("resnet", coverage, False), frozenset({name})
        )
        for present, launched in ((False, False), (True, False), (True, True)):
            launches, ranges = self.good()
            if present:
                ranges.extend(
                    [(210, 300, "window", name), (220, 280, "candidate", self.LAYER)]
                )
            if launched:
                launches.append((235, "resnet_conv", 7, "kernel", None))
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "trace.sqlite"
                self.build(path, launches, ranges)
                result: dict = {
                    "profile_traces": [
                        {
                            "label": "profile",
                            "path": str(path),
                            "sha256": qualify.sha(path),
                            "receipt": None,
                        }
                    ]
                }

                def check():
                    return qualify.check_profile_traces(
                        result,
                        (self.LAYER,),
                        NONCE,
                        self.ALLOW,
                        frozenset({self.LAYER}),
                        False,
                        fresh_windows=frozenset({name}),
                    )

                if launched:
                    self.assertEqual(check()["checked_traces"], ["profile"])
                else:
                    with self.assertRaisesRegex(
                        qualify.Rejected, "missing fresh-input kernel coverage"
                    ):
                        check()

    def test_a_receipt_cannot_accept_changed_or_removed_trace_bytes(self):
        for removed in (False, True):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "trace.sqlite"
                self.build(path, *self.good())
                result: dict = {
                    "profile_traces": [
                        {
                            "label": "profile",
                            "path": str(path),
                            "sha256": qualify.sha(path),
                            "receipt": None,
                        }
                    ]
                }
                self.assertEqual(
                    qualify.check_profile_traces(
                        result,
                        (self.LAYER,),
                        NONCE,
                        self.ALLOW,
                        frozenset({self.LAYER}),
                        False,
                    )["checked_traces"],
                    ["profile"],
                )
                if removed:
                    path.unlink()
                else:
                    path.write_bytes(b"changed after validation")
                with self.assertRaisesRegex(
                    qualify.Rejected, "file is missing" if removed else "bytes changed"
                ):
                    qualify.profile_trace_receipts(result)

    def test_lifecycle_enqueues_do_not_change_eager_multisets(self):
        launches, ranges = self.good()
        ranges.extend(
            [
                (210, 300, "window", "lifecycle/case/warmup/0"),
                (220, 280, "candidate", self.LAYER),
            ]
        )
        launches.append((235, "resnet_conv", 7, "kernel", None))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.sqlite"
            self.build(path, launches, ranges)
            self.assertEqual(self.attribute(path)["windows"], 3)
            self.assertEqual(
                dict(trace.window_kernels(path, NONCE, multiset=True)["case"]),
                {"resnet_conv": 1},
            )
            self.assertNotIn(
                "lifecycle/case/warmup/0", trace.window_kernels(path, NONCE)
            )

    def lstm_trace(
        self,
        path,
        drop_phase=None,
        overlap=False,
        projection_phase=True,
        library_projection=False,
    ):
        """A custom stack with an optional forbidden cuBLAS projection call."""
        layer = "lstm.stack"
        ranges = [(10, 1000, "window", "case"), (20, 900, "candidate", layer)]
        launches = []
        start = 30
        for name in trace.LSTM_PHASES:
            end = start + (60 if overlap and name == "input_proj.L0.reverse" else 40)
            if name != drop_phase:
                ranges.append((start, end, "phase", name))
            launches.append((start + 2, "resnet_conv", 7, "kernel", None))
            if name == "input_proj.L0.forward" and library_projection:
                inner = (start + 3, start + 30) if projection_phase else (905, 940)
                ranges.append((*inner, "projection", "L0.forward"))
                ranges.append(
                    (inner[0] + 1, inner[1] - 1, "call", "cublas.m589.n512.k60")
                )
                launches.append(
                    (inner[0] + 5, "ampere_sgemm_128x64_tn", 7, "kernel", None)
                )
            start += 50
        ranges.append((950, 990, "fixed", "embedding_bias"))
        launches.append((955, "embedding_bias", 7, "kernel", None))
        self.build(path, launches, ranges)
        return layer

    def test_lstm_candidate_requires_phases_and_zero_library_projections(self):
        stack = frozenset({"lstm.stack"})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "good.sqlite"
            layer = self.lstm_trace(path)
            trace.attribute(path, (layer,), NONCE, self.ALLOW, stack)
            path = Path(directory) / "library-projection.sqlite"
            self.lstm_trace(path, library_projection=True)
            with self.assertRaisesRegex(trace.Rejected, "forbidden library kernels"):
                trace.attribute(path, (layer,), NONCE, self.ALLOW, stack)
            cases = [
                ({"drop_phase": "recurrence.L3.reverse"}, "lacks locked phase scopes"),
                ({"overlap": True}, "overlapping LSTM phase scopes"),
                (
                    {"projection_phase": False, "library_projection": True},
                    "projection outside its input_proj phase",
                ),
            ]
            for options, reason in cases:
                with self.subTest(options=options):
                    path = Path(directory) / f"{reason}.sqlite"
                    self.lstm_trace(path, **options)
                    with self.assertRaisesRegex(trace.Rejected, reason):
                        trace.attribute(path, (layer,), NONCE, self.ALLOW, stack)

    def test_library_control_has_no_candidate_scopes(self):
        launches, ranges = self.good()
        ranges = [r for r in ranges if r[2] != "candidate"]
        launches = [item for item in launches if item[1] != "resnet_conv"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.sqlite"
            self.build(path, launches, ranges)
            evidence = self.attribute(path, frozenset(), library=True)
            self.assertTrue(
                any(scope["library_kernels"] for scope in evidence["scopes"])
            )

    def test_declared_boundary_without_candidate_kernels_is_refused(self):
        launches, ranges = self.good()
        ranges = [r for r in ranges if r[2] != "candidate"]
        launches = [item for item in launches if item[1] != "resnet_conv"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.sqlite"
            self.build(path, launches, ranges)
            with self.assertRaisesRegex(
                trace.Rejected, "declared boundary has no candidate"
            ):
                self.attribute(path)


class Tf32NegativeEvidence(unittest.TestCase):
    def fixture(self):
        truth = {"minimum_cosine": 0.9999, "relative_l2": 0.001, "max_abs": 0.01}
        metrics = {
            "minimum_cosine": 1.0,
            "mean_cosine": 1.0,
            "relative_l2": 0.0,
            "max_abs": 0.0,
            "sha256": "a" * 64,
        }
        row = {
            "id": "tf32/first/b1/stage",
            "first": dict(metrics),
            "second": dict(metrics),
            "bitwise_equal": True,
            "truth": dict(truth),
            "truth_sha256": "b" * 64,
        }
        band = {
            "id": row["id"] + "/band",
            "metrics": [dict(metrics) for _ in range(8)],
            "truth_sha256": row["truth_sha256"],
            "seeds": list(range(8)),
            "truth_draws": [dict(truth) for _ in range(8)],
        }
        return row, copy.deepcopy(row), band

    def evaluate(self, candidate, control, band, *, same_fp32=False):
        result = {"target": "resnet", "implementation": "Oxide", "checks": []}
        candidates, controls = [candidate], [control, band]
        if same_fp32:
            candidates.append({**candidate, "id": "fp32/first/b1/stage"})
            controls.append({**control, "id": "fp32/first/b1/stage"})
            self.assertEqual(
                candidates[0]["first"]["sha256"], candidates[1]["first"]["sha256"]
            )
        coverage = qualify.Coverage.product(
            ["resnet.layer1.0.conv1"], [1], ["fp32", "tf32"]
        )
        qualify.numeric(result, controls, candidates, coverage)
        return next(
            row
            for row in result["checks"]
            if row["check"] == "stage_truth:tf32/first/b1/stage"
        )

    def test_equal_fp32_and_tf32_outputs_do_not_prove_accuracy(self):
        candidate, control, band = self.fixture()
        candidate["truth"]["relative_l2"] = 0.002
        check = self.evaluate(candidate, control, band, same_fp32=True)
        self.assertFalse(check["passed"])
        self.assertIn("less accurate", check["reason"])
        self.assertEqual(
            check["evidence"]["bound_components"]["relative_l2"]["candidate"], 0.002
        )

    def test_fixture_accuracy_cannot_replace_same_input_truth(self):
        candidate, control, band = self.fixture()
        self.assertEqual(candidate["first"]["minimum_cosine"], 1.0)
        candidate.pop("truth")
        check = self.evaluate(candidate, control, band)
        self.assertFalse(check["passed"])
        self.assertIn("missing same-input", check["reason"])

    def test_invalid_truth_identity_metrics_and_draws_fail_closed(self):
        for fault in ("hash", "nan", "negative", "cosine", "seeds", "draws"):
            candidate, control, band = self.fixture()
            if fault == "hash":
                candidate["truth_sha256"] = "c" * 64
            elif fault == "nan":
                candidate["truth"]["max_abs"] = math.nan
            elif fault == "negative":
                candidate["truth"]["relative_l2"] = -0.1
            elif fault == "cosine":
                candidate["truth"]["minimum_cosine"] = 2.0
            elif fault == "seeds":
                band["seeds"] = [0] * 8
            else:
                band["truth_draws"].pop()
            with self.subTest(fault=fault):
                check = self.evaluate(candidate, control, band)
                self.assertFalse(check["passed"])
                self.assertTrue(check["reason"])


if __name__ == "__main__":
    unittest.main()
