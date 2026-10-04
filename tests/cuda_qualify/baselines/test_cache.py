"""A library cache needs matching identities and complete zero-error evidence."""

import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts/cuda/qualify"))
qualify = importlib.import_module("qualify")
freeze_baselines = importlib.import_module("freeze_baselines")


class BaselineCache(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        root_patch = patch.object(qualify, "ROOT", self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)
        resolve_patch = patch.object(
            qualify, "resolve", return_value=self.root / "source"
        )
        resolve_patch.start()
        self.addCleanup(resolve_patch.stop)
        self.directory = self.root / "tests/cuda_qualify/baselines"
        self.directory.mkdir(parents=True)
        self.path = self.directory / "lstm-sm75.json"
        self.result: dict = {
            "control": {
                "archive_sha256": "a" * 64,
                "driver_sha256": "d" * 64,
                "toolchain": "rustc 1.99.0",
            },
            "verified_inputs": {"input": "b" * 64},
            "sanitizer_fingerprint": {"tool": "c" * 64},
        }
        tools = []
        for tool in qualify.TOOLS:
            log = self.directory / f"{tool}.log"
            summary = (
                "RACECHECK SUMMARY: 0 hazards"
                if tool == "racecheck"
                else "ERROR SUMMARY: 0 errors"
            )
            markers = "\n".join(qualify.projection_markers())
            log.write_text(f"{markers}\n1 passed; 0 failed\n{summary}\n")
            tools.append(
                {
                    "tool": tool,
                    "log": log.name,
                    "sha256": qualify.sha(log),
                    "returncode": 0,
                    "scope": "LSTM input projections only",
                    "source_result_sha256": "e" * 64,
                    "source_lock_digest": "f" * 64,
                    "command": [
                        "flock",
                        qualify.GPU_LOCK,
                        *qualify.sanitizer_argv(Path("driver"), tool),
                    ],
                }
            )
        self.record: dict = {
            "target": "lstm",
            "tier": "sm75",
            "scope": "LSTM input projections only",
            "device_sm": "12.0",
            "control_archive_sha256": self.result["control"]["archive_sha256"],
            "control_toolchain": self.result["control"]["toolchain"],
            "verified_inputs": self.result["verified_inputs"],
            "tool_fingerprint": self.result["sanitizer_fingerprint"],
            "source_result_sha256": "e" * 64,
            "tools": tools,
        }
        self.save()

    def save(self):
        self.path.write_text(json.dumps(self.record))

    def load(self):
        return qualify.known_library_baseline("lstm", "sm75", self.result, "12.0")

    def test_completed_matching_record_is_reused(self):
        self.assertEqual(self.load(), self.record)
        for tool in qualify.TOOLS:
            self.assertIsNotNone(qualify.library_tool_record(self.record, tool))

    def test_changed_option_invalidates_only_its_tool(self):
        row = self.record["tools"][0]
        index = row["command"].index("--force-synchronization-limit") + 1
        row["command"][index] = "8"
        self.assertIsNone(qualify.library_tool_record(self.record, row["tool"]))
        self.assertIsNotNone(qualify.library_tool_record(self.record, "racecheck"))

    def test_cached_proof_keeps_its_original_command_and_provenance(self):
        row = json.loads(json.dumps(self.record["tools"][0]))
        row.update(
            implementation="ProjectionBaseline", source="owner-locked library baseline"
        )
        data = {**self.result, "device_sm": "12.0"}
        result = {"target": "lstm", "tiers": {"sm75": data}}
        source, command, provenance = freeze_baselines.evidence(
            row, result, "sm75", Path("unused"), self.directory
        )
        self.assertEqual(source, self.directory / row["log"])
        self.assertEqual(command["argv"], row["command"])
        self.assertEqual(provenance["source_result_sha256"], "e" * 64)
        row["command"].append("--launch-count")
        with self.assertRaisesRegex(qualify.Rejected, "differs from the owner record"):
            freeze_baselines.evidence(
                row, result, "sm75", Path("unused"), self.directory
            )

    def test_any_changed_identity_requires_fresh_library_checks(self):
        for field in (
            "target",
            "tier",
            "device_sm",
            "control_archive_sha256",
            "control_toolchain",
            "verified_inputs",
            "tool_fingerprint",
        ):
            with self.subTest(field=field):
                original = self.record[field]
                self.record[field] = "changed"
                self.save()
                self.assertIsNone(self.load())
                self.record[field] = original

    def test_missing_tool_or_changed_log_is_refused(self):
        removed = self.record["tools"].pop()
        self.save()
        with self.assertRaisesRegex(qualify.Rejected, "incomplete tool coverage"):
            self.load()
        self.record["tools"].append(removed)
        self.save()
        (self.directory / removed["log"]).write_text("changed")
        with self.assertRaisesRegex(qualify.Rejected, "log differs"):
            self.load()

    def test_full_library_baseline_is_not_a_projection_baseline(self):
        self.record["scope"] = "all kernels"
        self.save()
        self.assertIsNone(self.load())

    def test_timeout_wrapper_cannot_bypass_the_gpu_lock(self):
        argv = qualify.sanitizer_argv(Path("driver"), "memcheck")
        with patch.object(qualify.subprocess, "run") as run:
            with self.assertRaisesRegex(qualify.Rejected, "shared GPU lock"):
                qualify.command(argv, {}, Path("unused.log"), [])
            run.assert_not_called()

    def test_baseline_must_cover_every_projection_shape(self):
        row = self.record["tools"][0]
        log = self.directory / row["log"]
        text = log.read_text()
        log.write_text(text.replace(qualify.projection_markers()[-1], ""))
        row["sha256"] = qualify.sha(log)
        self.save()
        with self.assertRaisesRegex(qualify.Rejected, "omits a projection shape"):
            self.load()

    def test_findings_or_incomplete_driver_cannot_be_reused(self):
        row = self.record["tools"][0]
        log = self.directory / row["log"]
        for text in (
            "1 passed; 0 failed\nERROR SUMMARY: 1 errors",
            "ERROR SUMMARY: 0 errors",
        ):
            log.write_text(text)
            row["sha256"] = qualify.sha(log)
            self.save()
            with self.assertRaises(qualify.Rejected):
                self.load()


if __name__ == "__main__":
    unittest.main()
