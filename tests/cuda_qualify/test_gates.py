"""Gate regression checks. These do not substitute for GPU mutation proof."""

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
snapshot = importlib.import_module("lock")
scan = importlib.import_module("scan")

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

    def test_stage_guard_allows_noise_but_not_regression(self):
        within = [("Library", 1.0), ("c", 1.002), ("Library", 1.001), ("c", 0.997)]
        self.assertGreater(gates.stage_speed(self.runs(within), "c", 0.003)["ratio"], 0)
        slower = [("Library", 1.0), ("c", 1.004), ("Library", 1.0), ("c", 1.0)]
        with self.assertRaisesRegex(gates.Rejected, "stage candidate slower"):
            gates.stage_speed(self.runs(slower), "c", 0.003)
        offset = [("Library", 1.0), ("c", 1.002), ("Library", 1.0), ("c", 1.0)]
        with self.assertRaisesRegex(gates.Rejected, "slower on average"):
            gates.stage_speed(self.runs(offset), "c", 0.003)

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
        library = {"minimum_cosine": 0.9999, "mean_cosine": 0.99995}
        gates.tf32_stage_case({"minimum_cosine": 0.9999 - 2e-7}, library, band, True)
        with self.assertRaisesRegex(gates.Rejected, "exceeds band"):
            gates.tf32_stage_case(
                {"minimum_cosine": 0.9999 - 4e-7}, library, band, True
            )
        with self.assertRaisesRegex(gates.Rejected, "mean cosine"):
            gates.tf32_stage_aggregate([{"mean_cosine": 0.9999}], [library], band, True)
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
        gates.tf32_stage_case(
            {"relative_l2": 0.0105, "max_abs": 0.11, "argmax_flips": 3},
            library,
            band,
            False,
        )
        with self.assertRaises(gates.Rejected):
            gates.tf32_stage_case(
                {"relative_l2": 0.0105, "max_abs": 0.11, "argmax_flips": 4},
                library,
                band,
                False,
            )
        with self.assertRaisesRegex(gates.Rejected, "mean logits error"):
            gates.tf32_stage_aggregate(
                [{"relative_l2": 0.0105, "argmax_flips": 2}], [library], band, False
            )

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
            "use crate::inference::cuda::dnn::Conv2d;\nuse crate::inference::cuda::error::check_len;",
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
            (5, "setup_kernel_outside_windows", 9, "kernel", None),
            (25, "resnet_conv", 7, "kernel", None),
            (55, "embedding_bias", 7, "kernel", None),
            (130, "sm80_xmma_fprop_cudnn", 7, "kernel", None),
        ]
        return launches, ranges

    def attribute(
        self, path, declared=frozenset({LAYER}), library=False, shapes=frozenset()
    ):
        return trace.attribute(
            path, (self.LAYER,), NONCE, self.ALLOW, declared, shapes, library
        )

    def run_case(self, change):
        launches, ranges = self.good()
        change(launches, ranges)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.sqlite"
            self.build(path, launches, ranges)
            return self.attribute(path)

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

    def lstm_trace(self, path, drop_phase=None, overlap=False, projection_phase=True):
        """A candidate stack with all 16 phases; L0.forward projects through cuBLAS."""
        layer = "lstm.stack"
        ranges = [(10, 1000, "window", "case"), (20, 900, "candidate", layer)]
        launches = []
        start = 30
        for name in trace.LSTM_PHASES:
            end = start + (60 if overlap and name == "input_proj.L0.reverse" else 40)
            if name != drop_phase:
                ranges.append((start, end, "phase", name))
            launches.append((start + 2, "resnet_conv", 7, "kernel", None))
            if name == "input_proj.L0.forward":
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

    def test_lstm_candidate_needs_its_phases_and_helper(self):
        shapes = frozenset({(589, 512, 60)})
        stack = frozenset({"lstm.stack"})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "good.sqlite"
            layer = self.lstm_trace(path)
            trace.attribute(path, (layer,), NONCE, self.ALLOW, stack, shapes)
            with self.assertRaisesRegex(trace.Rejected, "forbidden library kernels"):
                trace.attribute(path, (layer,), NONCE, self.ALLOW, stack, frozenset())
            cases = [
                ({"drop_phase": "recurrence.L3.reverse"}, "lacks locked phase scopes"),
                ({"overlap": True}, "overlapping LSTM phase scopes"),
                (
                    {"projection_phase": False},
                    "projection outside its input_proj phase",
                ),
            ]
            for options, reason in cases:
                with self.subTest(options=options):
                    path = Path(directory) / f"{reason}.sqlite"
                    self.lstm_trace(path, **options)
                    with self.assertRaisesRegex(trace.Rejected, reason):
                        trace.attribute(
                            path, (layer,), NONCE, self.ALLOW, stack, shapes
                        )

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


if __name__ == "__main__":
    unittest.main()
