"""Keep every GPU process locked and every gate exact, without a GPU."""

import re
import importlib
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/cuda/qualify"))
qualify = importlib.import_module("qualify")


def failed(check, reason, **extra):
    return {"check": f"sm75/{check}", "passed": False, "reason": reason, **extra}


class Processes(unittest.TestCase):
    def test_control_build_cache_is_bound_to_source_bytes(self):
        first = qualify.control_target("a" * 64, "sm75")
        second = qualify.control_target("b" * 64, "sm75")
        self.assertNotEqual(first, second)
        self.assertNotEqual(first, qualify.control_target("a" * 64, "sm80"))
        with self.assertRaises(qualify.Rejected):
            qualify.control_target("../../other-task", "sm75")

    def collect(self, implementation, tier):
        result = {
            "target": "resnet",
            "implementation": implementation,
            "checks": [],
            "commands": [],
        }
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(qualify, "scan", return_value={"findings": []}),
            patch.object(qualify, "shipped_tiers", return_value=("sm75",)),
            patch.object(qualify, "collect_tier", side_effect=tier),
        ):
            qualify.collect("resnet", implementation, Path(directory), result)
        return result

    def test_incomplete_tier_can_never_accept_a_replacement(self):
        def incomplete(target, implementation, directory, child, tier):
            child.update(status="blocked", reason="required sanitizer not run")

        result = self.collect("Oxide", incomplete)
        self.assertEqual(result["status"], "blocked")
        self.assertFalse(result["accepts_replacement"])

    def test_static_scan_refuses_before_any_build(self):
        result = {
            "target": "lstm",
            "implementation": "Oxide",
            "checks": [],
            "commands": [],
        }
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(
                qualify, "scan", return_value={"findings": ["lstm.rs:1: environment"]}
            ),
            patch.object(qualify, "collect_tier") as tier,
        ):
            qualify.collect("lstm", "Oxide", Path(directory), result)
        tier.assert_not_called()
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["checks"][0]["check"], "static_scan")

    def test_projection_baseline_is_unfiltered_and_time_bounded(self):
        for tool in qualify.TOOLS:
            argv = qualify.sanitizer_argv(Path("driver"), tool)
            self.assertEqual(
                argv[argv.index("--force-synchronization-limit") + 1], "16"
            )
            self.assertEqual(argv[argv.index("--error-exitcode") + 1], "86")
            self.assertNotIn("--kernel-name", argv)
            self.assertNotIn("--kernel-name-exclude", argv)
            self.assertEqual(
                argv[:4], ["timeout", "--signal=TERM", "--kill-after=10s", "1200"]
            )
            self.assertNotIn("--launch-count", argv)

    def test_candidate_sanitizer_includes_exactly_the_loaded_entries(self):
        argv = qualify.sanitizer_argv(
            Path("driver"), "racecheck", ["b_kernel", "a_kernel"]
        )
        filters = [argv[i + 1] for i, arg in enumerate(argv) if arg == "--kernel-name"]
        self.assertEqual(sorted(filters), ["kne=a_kernel", "kne=b_kernel"])
        self.assertNotIn("--kernel-name-exclude", argv)
        with self.assertRaises(qualify.Rejected):
            qualify.sanitizer_argv(Path("driver"), "memcheck", [])
        with self.assertRaises(qualify.Rejected):
            qualify.sanitizer_argv(Path("driver"), "memcheck", ["a|.*"])

    def test_unlocked_gpu_process_is_rejected_before_start(self):
        with patch.object(qualify.subprocess, "run") as run:
            for argv in [
                [
                    str(qualify.BOX / "target/release/deps/speakrs-test"),
                    "--exact",
                    "test",
                ],
                ["compute-sanitizer", "--tool", "memcheck", "driver"],
                ["nsys", "profile", "driver"],
            ]:
                with self.assertRaisesRegex(qualify.Rejected, "shared GPU lock"):
                    qualify.command(argv, {}, Path("unused.log"), [])
            run.assert_not_called()

    def test_every_positive_control_needs_its_live_fault(self):
        faults = {
            "oob": "Invalid __global__ write of size 4 in qualify_oob\nERROR SUMMARY: 1 errors",
            "race": "Error: Potential WAW hazard detected in qualify_race\nRACECHECK SUMMARY: 1 hazards displayed",
            "uninit": "Uninitialized __global__ memory read of size 4 in qualify_round\nERROR SUMMARY: 1 errors",
        }
        for control, text in faults.items():
            with (
                self.subTest(control=control),
                tempfile.TemporaryDirectory() as directory,
            ):

                def run(argv, env, log, steps, text=text, control=control):
                    self.assertIn("--kernel-name", argv)
                    self.assertEqual(env["SPEAKRS_QUALIFY_CONTROL"], control)
                    log.write_text(text)
                    return 86

                with patch.object(qualify, "gpu_command", side_effect=run):
                    proof = qualify.filter_proof(
                        Path("driver"),
                        {},
                        Path(directory),
                        [],
                        control,
                        ["qualify_oob"],
                    )
                    self.assertTrue(proof["passed"])
                with (
                    patch.object(qualify, "gpu_command", return_value=0),
                    self.assertRaisesRegex(qualify.Rejected, "hid"),
                ):
                    qualify.filter_proof(
                        Path("driver"),
                        {},
                        Path(directory),
                        [],
                        control,
                        ["qualify_oob"],
                    )

    def test_timeout_split_covers_declared_layers_and_batches(self):
        coverage = qualify.Coverage.product(
            {"resnet.layer2.0.conv1"}, {32, 64}, {"fp32", "tf32"}
        )
        calls = []

        def run(argv, env, log, steps):
            layer = env.get("SPEAKRS_QUALIFY_SANITIZER_LAYER")
            calls.append((layer, env.get("SPEAKRS_QUALIFY_SANITIZER_BATCH")))
            self.assertEqual(env["SPEAKRS_QUALIFY_SANITIZE_BATCHES"], "32,64")
            if layer is None:
                log.write_text("timeout")
                return 124
            batch = int(env["SPEAKRS_QUALIFY_SANITIZER_BATCH"])
            rows = [
                {"id": f"{mode}/mixed/b{batch}/{layer}", "sanitized": True}
                for mode in ("fp32", "tf32")
            ]
            Path(env["SPEAKRS_QUALIFY_OUTPUT"]).write_text(
                qualify.json.dumps(
                    {
                        "target": "resnet",
                        "implementation": "Oxide",
                        "phase": "sanitize",
                        "tier": "sm75",
                        "rows": rows,
                    }
                )
            )
            log.write_text("1 passed; 0 failed\nRACECHECK SUMMARY: 0 hazards")
            return 0

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(qualify, "gpu_command", side_effect=run),
        ):
            env = {
                "SPEAKRS_QUALIFY_TARGET": "resnet",
                "SPEAKRS_QUALIFY_IMPL": "Oxide",
                "SPEAKRS_CUDA_PTX_TIER": "sm75",
            }
            record, text = qualify.candidate_sanitizer(
                Path("driver"),
                "racecheck",
                env,
                Path(directory),
                [],
                ["resnet_conv"],
                coverage,
            )
        self.assertTrue(record["split_after_timeout"])
        self.assertEqual(
            set(calls[1:]),
            {("resnet.layer2.0.conv1", "32"), ("resnet.layer2.0.conv1", "64")},
        )
        self.assertEqual(record["returncode"], 0)

    def test_mutants_are_caught_only_by_their_exact_check_and_reason(self):
        cases = [
            (
                "Fallback",
                failed(
                    "profile",
                    "profile: forbidden library kernels in a candidate scope (2)",
                ),
                True,
            ),
            # the host tracker alone is not the profile rule
            (
                "Fallback",
                failed(
                    "profile:captured_library_calls",
                    "profile: library calls from candidate scopes",
                ),
                False,
            ),
            (
                "Fallback",
                failed(
                    "profile", "profile: work outside a candidate or library range (1)"
                ),
                False,
            ),
            (
                "Slow",
                failed(
                    "speed:fp32/first/b1/stage",
                    "speed: candidate slower than the faster Library process",
                ),
                True,
            ),
            (
                "Slow",
                failed(
                    "speed:fp32/first/b1/stage", "speed: four fresh processes required"
                ),
                False,
            ),
            (
                "Slow",
                {
                    **failed(
                        "speed:fp32/first/b1/stage", "speed: blocked: candidate slower"
                    ),
                    "blocked": True,
                },
                False,
            ),
            (
                "Atomic",
                failed(
                    "determinism:fixed_reduction_order",
                    "determinism: floating-point atomic in launched custom entry qualify_atomic",
                ),
                True,
            ),
            (
                "Atomic",
                failed(
                    "determinism:fp32/first/b1/stage", "determinism: bitwise mismatch"
                ),
                False,
            ),
            (
                "Precision",
                failed("layer:fp32/lstm.stack", "layer parity: per-case error ratio"),
                True,
            ),
            ("Precision", failed("layer:fp32/lstm.stack", "no layer cases"), False),
            (
                "PhaseCheat",
                failed(
                    "timing_output:fp32/first/b1/lstm.stack",
                    "timing: final output differs from the numeric output",
                ),
                True,
            ),
            (
                "Unscoped",
                failed(
                    "profile", "profile: work outside a candidate or library range (2)"
                ),
                True,
            ),
            (
                "Unlisted",
                failed(
                    "profile", "profile: kernel is not on the loaded PTX allow-list (1)"
                ),
                True,
            ),
        ]
        for implementation, item, caught in cases:
            with self.subTest(
                implementation=implementation,
                check=item["check"],
                reason=item["reason"],
            ):
                self.assertEqual(
                    qualify.mutant_gate(implementation, [item])["caught"], caught
                )

    def test_each_mutant_runs_the_phase_its_gate_reads(self):
        needs = {
            "layer": "numeric",
            "profile": "profile",
            "determinism:fixed_reduction_order": "profile",
            "speed": "timing",
            "timing_output": "timing",
            "secret": "numeric",
            "ptx:shared_initialization": "numeric",
        }
        self.assertEqual(set(qualify.MUTANT_PHASES), set(qualify.MUTANTS))
        for mutant, gate in qualify.MUTANT_GATES.items():
            with self.subTest(mutant=mutant):
                phases = qualify.MUTANT_PHASES[mutant]
                self.assertIn("numeric", phases)
                self.assertIn(needs[gate.family], phases)

    def test_shape_and_tail_need_their_rows(self):
        cases = [f"fp32/{case}/b{batch}/lstm.stack" for case, batch in qualify.CASES]
        non_b32 = [case for case in cases if "/b32/" not in case]
        partial = [case for case in cases if qualify._batch(case) % 32]
        check = "layer:fp32/lstm.stack"
        reason = "layer parity: per-case error ratio"
        self.assertTrue(
            qualify.mutant_gate(
                "Shape", [failed(check, reason, cases=cases, failing_cases=non_b32)]
            )["caught"]
        )
        self.assertFalse(
            qualify.mutant_gate(
                "Shape", [failed(check, reason, cases=cases, failing_cases=cases)]
            )["caught"]
        )
        self.assertTrue(
            qualify.mutant_gate(
                "Tail", [failed(check, reason, cases=cases, failing_cases=partial)]
            )["caught"]
        )
        self.assertFalse(
            qualify.mutant_gate(
                "Tail", [failed(check, reason, cases=cases, failing_cases=partial[:1])]
            )["caught"]
        )

    def test_escaped_mutant_exits_nonzero(self):
        def tier(target, implementation, directory, child, tier):
            child.update(status="rejected")
            child["checks"].append(
                {
                    "check": "stage:fp32/first/b1/stage",
                    "passed": False,
                    "reason": "unrelated",
                }
            )

        result = self.collect("Slow", tier)
        self.assertEqual(result["status"], "escaped")
        self.assertEqual(qualify.EXIT_CODES[result["status"]], 4)
        self.assertFalse(result["accepts_replacement"])

    def test_every_driver_process_holds_the_lock(self):
        mode = "fp32"

        def run(argv, **kwargs):
            self.assertEqual(argv[:2], ["flock", qualify.GPU_LOCK])
            kwargs["stdout"].write(b"test result: ok. 1 passed; 0 failed\n")
            env = kwargs["env"]
            phase = env["SPEAKRS_QUALIFY_PHASE"]
            Path(env["SPEAKRS_QUALIFY_OUTPUT"]).write_text(
                qualify.json.dumps(
                    {
                        "implementation": env["SPEAKRS_QUALIFY_IMPL"],
                        "phase": phase,
                        "target": env["SPEAKRS_QUALIFY_TARGET"],
                        "mode": env.get("SPEAKRS_QUALIFY_MODE"),
                        "pid": 1,
                        "tier": "sm75",
                        "coverage": {"layers": [], "batches": [], "maths": []},
                        "rows": [
                            {"id": key, "cuda_graph": True}
                            for key in qualify.expected_ids("lstm", phase, mode)
                        ],
                    }
                )
            )
            return SimpleNamespace(returncode=0)

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(qualify.subprocess, "run", side_effect=run),
        ):
            steps = []
            env = {"SPEAKRS_QUALIFY_TARGET": "lstm", "SPEAKRS_CUDA_PTX_TIER": "sm75"}
            for phase in ("coverage", "numeric", "timing", "profile", "sanitize"):
                data = qualify.driver(
                    qualify.BOX / "target/release/driver",
                    env,
                    Path(directory),
                    steps,
                    "Library",
                    phase,
                    phase,
                    {"SPEAKRS_QUALIFY_MODE": mode},
                )
                self.assertEqual(data["phase"], phase)
            self.assertEqual(len(steps), 5)
            self.assertTrue(all(item["gpu_lock"] == qualify.GPU_LOCK for item in steps))

    def test_case_inventory_cannot_be_reduced(self):
        self.assertEqual(len(qualify.case_ids()), 16)
        self.assertEqual(len(qualify.expected_ids("resnet", "timing", "fp32")), 120)
        self.assertEqual(len(qualify.expected_ids("resnet", "numeric", "fp32")), 240)
        self.assertIn(
            "tf32/mixed/b64/stage/switched",
            qualify.expected_ids("lstm", "numeric", "tf32"),
        )

    def test_band_lists_declared_tf32_layers_per_batch(self):
        coverage = qualify.Coverage.product({"lstm.stack"}, {32, 64}, {"tf32"})
        self.assertEqual(
            qualify.band_layers(coverage, "Oxide"), "32:lstm.stack;64:lstm.stack"
        )
        self.assertEqual(qualify.band_layers(coverage, "Library"), "")

    def test_coverage_declarations_are_checked(self):
        coverage = qualify.parse_coverage(
            {
                "entries": [
                    {"layers": ["lstm.stack"], "batches": [32, 64], "maths": ["fp32"]}
                ]
            },
            "lstm",
        )
        self.assertTrue(coverage.declared("lstm.stack", 32, "fp32"))
        self.assertFalse(coverage.declared("lstm.stack", 32, "tf32"))
        self.assertFalse(coverage.declared("lstm.stack", 7, "fp32"))
        everything = qualify.parse_coverage(
            {"entries": [{"layers": "all", "batches": "all", "maths": "all"}]}, "resnet"
        )
        self.assertEqual(len(everything.layers), 14)
        for raw in (
            {"layers": ["lstm.stack"], "batches": [48], "maths": "all"},
            {"layers": ["resnet.layer3.0.conv1"], "batches": "all", "maths": "all"},
        ):
            with self.assertRaises(qualify.Rejected):
                qualify.parse_coverage({"entries": [raw]}, "lstm")
        self.assertFalse(qualify.parse_coverage({"entries": []}, "lstm").layers)

    def test_coverage_is_the_union_of_its_entries(self):
        c32 = ["resnet.layer1.0.conv1", "resnet.layer2.0.conv1"]
        c64 = ["resnet.layer2.1.conv1"]
        coverage = qualify.parse_coverage(
            {
                "entries": [
                    {"layers": c32, "batches": "all", "maths": "all"},
                    {"layers": c64, "batches": [7, 32, 33, 64], "maths": "all"},
                    {"layers": c64, "batches": [1], "maths": ["fp32"]},
                ]
            },
            "resnet",
        )
        self.assertTrue(coverage.declared(c64[0], 1, "fp32"))
        self.assertFalse(coverage.declared(c64[0], 1, "tf32"))
        self.assertTrue(coverage.declared(c64[0], 33, "tf32"))
        self.assertEqual(coverage.layers_at(1, "tf32"), sorted(c32))
        self.assertEqual(qualify.sanitizer_batches(coverage), [7, 33])

    def test_tiers_are_detected_from_candidate_area_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ptx = root / "src/inference/cuda/ptx"
            ptx.mkdir(parents=True)
            (ptx / "resnet.sm75.ptx").write_text("baseline")
            (ptx / "embedding.sm80.ptx").write_text("not the candidate area")
            self.assertEqual(qualify.shipped_tiers("resnet", root), ("sm75",))
            (ptx / "resnet.sm80.ptx").write_text("higher variant")
            self.assertEqual(qualify.shipped_tiers("resnet", root), ("sm75", "sm80"))
            (ptx / "resnet.sm999.ptx").write_text("unknown")
            with self.assertRaises(qualify.Rejected):
                qualify.shipped_tiers("resnet", root)

    def test_loaded_ptx_must_be_committed_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ptx = root / "src/inference/cuda/ptx"
            ptx.mkdir(parents=True)
            (root / "tests/cuda_qualify/device").mkdir(parents=True)
            path = ptx / "lstm.sm75.ptx"
            path.write_text(".visible .entry lstm_step(\n)")
            module = {
                "area": "lstm",
                "tier": "sm75",
                "sha256": qualify.sha(path),
                "entries": ["lstm_step"],
            }
            allow, _ = qualify.verify_modules([module], root)
            self.assertEqual(allow.candidate_entries, {"lstm_step"})
            with self.assertRaisesRegex(qualify.Rejected, "no committed file"):
                qualify.verify_modules([{**module, "sha256": "0" * 64}], root)
            with self.assertRaisesRegex(qualify.Rejected, "came from"):
                qualify.verify_modules([{**module, "area": "resnet"}], root)
            (ptx / "hidden").mkdir()
            (ptx / "hidden/extra.ptx").write_text(".entry gemm_x(")
            with self.assertRaisesRegex(qualify.Rejected, "outside the area layout"):
                qualify.verify_modules([module], root)

    def timing_runs(self, medians):
        return {
            mode: [
                {
                    "pid": i + 1,
                    "implementation": "Library" if i % 2 == 0 else "Oxide",
                    "rows": [
                        {
                            "id": key,
                            "warmup": 5,
                            "burst": 4,
                            "milliseconds": [medians[i]] * 20,
                        }
                        for key in qualify.expected_ids("lstm", "timing", mode)
                    ],
                }
                for i in range(4)
            ]
            for mode in qualify.MODES
        }

    def test_noisy_controls_block_and_undeclared_cases_are_not_timed(self):
        coverage = qualify.Coverage.product({"lstm.stack"}, {32}, {"fp32"})
        result: dict = {"target": "lstm", "checks": []}
        qualify.timing(
            result, self.timing_runs([1.0, 0.9, 1.02, 0.9]), "Oxide", coverage
        )
        gated = {item["check"]: item for item in result["checks"]}
        self.assertEqual(
            sorted(gated),
            ["speed:fp32/mixed/b32/lstm.stack", "speed:fp32/mixed/b32/stage"],
        )
        self.assertTrue(all(item.get("blocked") for item in gated.values()))
        self.assertEqual(result["measurability"]["stage"]["measurable"], 0)
        gates = {row["id"]: row["gate"] for row in result["timing"]}
        self.assertTrue(gates["fp32/first/b1/stage"].startswith("undeclared stage"))
        self.assertTrue(gates["fp32/mixed/b32/stage"].startswith("stage: regression"))

    def test_stage_within_its_bound_passes_while_the_operator_must_win(self):
        coverage = qualify.Coverage.product({"lstm.stack"}, {32}, {"fp32"})
        result: dict = {"target": "lstm", "checks": []}
        # the candidate is 0.1% slower: inside the stage bound, but slower on average
        qualify.timing(
            result, self.timing_runs([1.0, 1.001, 1.0, 1.001]), "Oxide", coverage
        )
        gated = {item["check"]: item for item in result["checks"]}
        self.assertIn(
            "slower on average", gated["speed:fp32/mixed/b32/stage"]["reason"]
        )
        self.assertFalse(gated["speed:fp32/mixed/b32/lstm.stack"]["passed"])
        result = {"target": "lstm", "checks": []}
        # inside the bound and faster on average: the stage passes
        qualify.timing(
            result, self.timing_runs([1.0, 1.002, 1.0, 0.997]), "Oxide", coverage
        )
        gated = {item["check"]: item for item in result["checks"]}
        self.assertTrue(gated["speed:fp32/mixed/b32/stage"]["passed"])

    def test_library_control_reports_measurability(self):
        coverage = qualify.Coverage(frozenset())
        result: dict = {"target": "lstm", "checks": []}
        runs = self.timing_runs([1.0, 1.0005, 1.001, 1.0])
        for runs_mode in runs.values():
            for run in runs_mode:
                run["implementation"] = "Library"
        qualify.timing(result, runs, "Library", coverage)
        self.assertFalse(result["checks"])
        self.assertEqual(
            result["measurability"]["stage"], {"measurable": 16, "cases": 16}
        )

    def test_timing_output_must_match_numeric_output_on_both_inputs(self):
        numeric = {
            "k": {"first": {"sha256": "a"}},
            "k/switched": {"first": {"sha256": "b"}},
        }
        run = {"rows": [{"id": "k", "output_sha256": ["a", "x"]}]}
        result: dict = {"checks": []}
        qualify.timing_outputs(result, [(run, numeric, True)], "Oxide")
        self.assertIn("final output differs", result["checks"][0]["reason"])
        result = {"checks": []}
        qualify.timing_outputs(result, [(run, numeric, False)], "Oxide")
        self.assertTrue(result["checks"][0]["blocked"])
        run["rows"][0]["output_sha256"] = ["a", "b"]
        result = {"checks": []}
        qualify.timing_outputs(result, [(run, numeric, True)], "Oxide")
        self.assertFalse(result["checks"])

    def test_secret_input_uses_same_input_f64_truth_without_a_floor(self):
        coverage = qualify.Coverage.product({"lstm.stack"}, {32}, {"fp32"})

        def row(relative_l2, library=1e-6, same=True):
            def measurement(error):
                return {"relative_l2": error, "max_abs": error, "finite": True}

            return {
                "id": "fp32/secret/b32/lstm.stack",
                "secret": True,
                "library": measurement(library),
                "truth": "f64",
                "eager": measurement(relative_l2),
                "replay": measurement(relative_l2),
                "replay_equals_eager": same,
            }

        for secret, reason in [
            (row(1e-6), None),
            (row(1.2e-6), "differs from Library on a fresh input"),
            (row(1e-6, library=2e-6), None),
            (row(0.0, library=0.0), "exactly zero"),
            (row(1e-15, library=0.0), "exactly zero"),
            (row(1e-6, same=False), "graph replay differs"),
            (row(float("nan")), "non-finite"),
            (row(-1e-6), "negative error"),
            (None, "missing"),
        ]:
            with self.subTest(reason=reason):
                result: dict = {"checks": []}
                qualify.secret_checks(result, [secret] if secret else [], coverage)
                item = result["checks"][0]
                self.assertEqual(item["passed"], reason is None)
                if reason:
                    self.assertIn(reason, item["reason"])

    def test_tf32_secret_accepts_an_answer_more_accurate_than_library(self):
        def measurement(error):
            return {"relative_l2": error, "max_abs": error, "finite": True}

        row = {
            "id": "tf32/secret/b32/lstm.stack",
            "secret": True,
            "library": measurement(1e-3),
            "truth": "f64",
            "eager": measurement(0.0),
            "replay": measurement(0.0),
            "replay_equals_eager": True,
        }
        result: dict = {"checks": []}
        qualify.secret_checks(
            result, [row], qualify.Coverage.product({"lstm.stack"}, {32}, {"tf32"})
        )
        self.assertTrue(result["checks"][0]["passed"])

    def test_fp_atomic_instruction_boundary(self):
        for mnemonic in (
            "atom.global.add.f32",
            "@%p1 red.global.add.f32",
            "atom.add.f32",
            "red.shared.add.f32",
        ):
            self.assertIsNotNone(re.search(qualify.FLOAT_ATOMIC, mnemonic))
        for mnemonic in ("ld.shared.v4.f32", "st.shared.f32", "ld.shared.v2.f32"):
            self.assertIsNone(re.search(qualify.FLOAT_ATOMIC, mnemonic))

    def module_process(self, mode, candidate=True, fixed_hash="a", candidate_hash="b"):
        modules = [{"area": "segmentation", "sha256": fixed_hash}]
        if candidate:
            modules.append({"area": "sincnet", "sha256": candidate_hash})
        return {
            "phase": "numeric",
            "mode": mode,
            "loaded_modules": modules,
            "rows": [{"id": f"{mode}/mixed/b32/sincnet.conv0.abs_pool"}],
        }

    def stable(self, *processes):
        coverage = qualify.Coverage.product({"sincnet.conv0.abs_pool"}, {32}, {"fp32"})
        return qualify.stable_modules(list(processes), coverage, "sincnet", "Oxide")

    def test_fixed_modules_must_match_in_every_process(self):
        with self.assertRaisesRegex(qualify.Rejected, "non-candidate PTX"):
            self.stable(
                self.module_process("fp32"),
                self.module_process("tf32", False, fixed_hash="c"),
            )

    def test_candidate_bytes_must_match_where_loaded(self):
        with self.assertRaisesRegex(qualify.Rejected, "candidate PTX bytes"):
            self.stable(
                self.module_process("fp32"),
                self.module_process("fp32", candidate_hash="c"),
            )
        self.stable(self.module_process("fp32"), self.module_process("fp32"))

    def test_declared_process_must_load_its_area(self):
        with self.assertRaisesRegex(qualify.Rejected, "did not load candidate area"):
            self.stable(self.module_process("fp32", False))

    def test_undeclared_process_must_load_no_candidate_area(self):
        with self.assertRaisesRegex(qualify.Rejected, "no declared triple"):
            self.stable(self.module_process("fp32"), self.module_process("tf32"))
        evidence = self.stable(
            self.module_process("fp32"), self.module_process("tf32", False)
        )
        self.assertEqual(evidence["processes"][1]["loaded_candidate_areas"], [])

    def test_captured_kernels_are_grouped_by_case(self):
        processes = [
            {
                "graph_evidence": [
                    {
                        "scope": "candidate",
                        "name": "lstm.stack",
                        "kernels": ["k1"],
                        "case": "c",
                    },
                    {
                        "scope": "library",
                        "name": "sincnet",
                        "kernels": ["cudnn"],
                        "case": "c",
                    },
                    {
                        "scope": "candidate",
                        "name": "lstm.stack",
                        "kernels": ["k2"],
                        "case": None,
                    },
                    {
                        "scope": "library",
                        "name": "lstm.stack",
                        "kernels": ["rnn"],
                        "case": "d",
                    },
                ]
            }
        ]
        self.assertEqual(qualify.graph_kernels(processes), {"c": {"k1"}, "d": set()})

    def test_direct_entry_requires_the_outside_tree_digest(self):
        with (
            patch.object(qualify.sys, "argv", ["qualify.py", "lstm", "Library"]),
            patch.dict(qualify.os.environ, {}, clear=True),
            patch.object(qualify, "verify") as verify,
            patch.object(qualify, "collect") as collect,
        ):
            self.assertEqual(qualify.main(), 2)
            verify.assert_not_called()
            collect.assert_not_called()

    def test_build_environment_drops_compiler_overrides(self):
        with patch.dict(
            qualify.os.environ,
            {
                "RUSTFLAGS": "-C x",
                "RUSTC_WRAPPER": "w",
                "CARGO_TARGET_DIR": "/tmp",
                "PATH": "/bin",
            },
            clear=True,
        ):
            env = qualify.clean_environment()
        self.assertEqual(env["PATH"], "/bin")
        for key in ("RUSTFLAGS", "RUSTC_WRAPPER", "CARGO_TARGET_DIR"):
            self.assertNotIn(key, env)


if __name__ == "__main__":
    unittest.main()
