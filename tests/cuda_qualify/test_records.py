"""Production provenance, target identity and immutable cache failures."""

import copy
import hashlib
import importlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/cuda/qualify"))
records = importlib.import_module("records")
qualify = importlib.import_module("qualify")


def complete_fixture(raw):
    """Populate complete synthetic collection markers without GPU measurements"""
    raw["checks"] = []
    for tier, child in raw["tiers"].items():
        if raw["schema"] != 3 and "numeric" in child:
            child["numeric"]["candidate"] = [
                {
                    "device": copy.deepcopy(child["device"]),
                    "loaded_libraries": copy.deepcopy(
                        child["numeric"]["library"][0]["loaded_libraries"]
                    ),
                    "mode": mode,
                    "configurations": [
                        {
                            "tuple": row,
                            "pin": (
                                {
                                    "kind": "Conv",
                                    "selection": "LegacyWaves",
                                    "shape": "C64",
                                }
                                if raw["target"] == "resnet"
                                else {"kind": "Lstm", "selection": "LegacyCooperative"}
                            ),
                        }
                        for row in child["coverage_declared"]["triples"]
                        if row[2] == mode
                    ],
                }
                for mode in ("fp32", "tf32")
                if any(row[2] == mode for row in child["coverage_declared"]["triples"])
            ]
        child["coverage"] = {
            "tier": tier,
            "math": ["fp32", "tf32"],
            "cases": [list(case) for case in records.TEST_CASES],
        }
        phases = records.LEGACY_PHASES if raw["schema"] == 3 else records.CURRENT_PHASES
        child["phases_run"] = list(phases)
        child["reason"] = (
            "all required checks passed"
            if raw["schema"] == 3
            else "all required checks passed under locked verdict evaluation"
        )
        required, timing = records.tuple_requirements(raw, child)
        required |= records.required_checks(raw["target"])
        names = {check["check"] for check in child["checks"]}
        child["checks"].extend(
            {"check": name, "passed": True} for name in sorted(required - names)
        )
        child["timing"] = [{"id": name} for name in sorted(timing)]
        raw["checks"].extend(
            {**check, "check": f"{tier}/{check['check']}"} for check in child["checks"]
        )
    return raw


class RecordsFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.cache = Path(self.temporary.name)
        self.env = patch.dict(os.environ, SPEAKRS_QUALIFY_CACHE=str(self.cache))
        self.env.start()
        self.addCleanup(self.env.stop)
        self.root = self.cache / "tree"
        for name in (
            "src/inference/cuda/candidate.rs",
            "src/inference/cuda/candidate/lstm.rs",
            "src/inference/cuda/candidate/lstm/layout.rs",
            "crates/speakrs-cuda-kernels/src/lstm.rs",
            "src/inference/cuda/ptx/lstm.manifest",
            "src/inference/cuda/ptx/lstm.sm75.ptx",
            "src/inference/cuda/ptx/lstm.sm75.sm_120.cubin",
        ):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(name)
        self.write_embed("sm75", ["cuda-sm75", "cuda-sm80", "cuda-sm90", "cuda-sm120"])
        self.files = records.shipped_files(self.root, "lstm")
        self.child: dict = {
            "status": "passed",
            "checks": [{"check": "layer:fp32/lstm.stack", "passed": True}],
            "requested_tier": "sm75",
            "device": {
                "compute_capability": "12.0",
                "name": "Fixture GPU",
                "sm_count": 36,
                "l2_bytes": 33554432,
                "driver_version": "580.0",
                "driver_api_version": 13000,
                "cuda_version": 12080,
                "cudnn_version": 90800,
                "cublas_version": 120804,
            },
            "accepted_tuples": [["lstm.stack", 1, "fp32"]],
            "coverage_declared": {"triples": [["lstm.stack", 1, "fp32"]]},
            "code_sha256": {
                name: value
                for group, hashes in self.files.items()
                for name, value in hashes.items()
            },
            "loaded_ptx": {
                "modules": [
                    {
                        "area": "lstm",
                        "path": "src/inference/cuda/ptx/lstm.sm75.ptx",
                        "tier": "sm75",
                        "sha256": self.files["ptx"][
                            "src/inference/cuda/ptx/lstm.sm75.ptx"
                        ],
                    }
                ]
            },
        }
        module = self.child["loaded_ptx"]["modules"][0]
        module.update(
            embedded_ptx_sha256=module["sha256"],
            artifact={"kind": "PtxJit", "sha256": module["sha256"]},
        )
        self.child["sanitizer_fingerprint"] = {
            "/lib/libcuda.so.580.0": "1" * 64,
            "/lib/libcudnn.so.9": "2" * 64,
            "/lib/libcublas.so.12": "3" * 64,
        }
        self.child["numeric"] = {
            "library": [
                {
                    "device": copy.deepcopy(self.child["device"]),
                    "loaded_libraries": copy.deepcopy(
                        self.child["sanitizer_fingerprint"]
                    ),
                }
            ],
            "candidate": [
                {
                    "device": copy.deepcopy(self.child["device"]),
                    "loaded_libraries": copy.deepcopy(
                        self.child["sanitizer_fingerprint"]
                    ),
                    "mode": "fp32",
                    "configurations": [
                        {
                            "tuple": ["lstm.stack", 1, "fp32"],
                            "pin": {"kind": "Lstm", "selection": "LegacyCooperative"},
                        }
                    ],
                }
            ],
        }
        self.child.update(
            target="lstm",
            implementation="Oxide",
            phases_run=list(records.CURRENT_PHASES),
            coverage={"tier": "sm75"},
            reason="all required checks passed under locked verdict evaluation",
        )
        self.child["checks"].extend(
            {"check": name, "passed": True}
            for name in sorted(records.required_checks("lstm"))
        )
        self.record: dict = {
            "schema": 5,
            "checks": [
                {**check, "check": f"sm75/{check['check']}"}
                for check in self.child["checks"]
            ],
            "implementation": "Oxide",
            "target": "lstm",
            "status": "passed",
            "accepts_replacement": True,
            "requested_tier": "sm75",
            "lock_digest": "a" * 64,
            "tiers": {"sm75": self.child},
        }
        complete_fixture(self.record)
        self.entry: dict = {
            "area": "lstm",
            "tier": "sm75",
            "artifact": copy.deepcopy(module["artifact"]),
            "devices": ["12.0"],
            "record": self.store(self.record),
            "der": self.store({"configuration": "integrated", "der": 0.123}),
            "coverage": {
                "entries": [
                    {"layers": ["lstm.stack"], "batches": [1], "maths": ["fp32"]}
                ]
            },
        }
        self.entry["configurations"] = copy.deepcopy(
            self.child["numeric"]["candidate"][0]["configurations"]
        )
        self.entry["candidate_coverage"] = copy.deepcopy(self.entry["coverage"])
        self.entry["speed_scope"] = {
            "kind": "Point",
            "capability": "12.0",
            "sm_count": 36,
            "device_name": "Fixture GPU",
        }

        self.entry["boundary_domain"] = [{"boundary": "lstm.stack", "batches": [1, 32]}]
        self.entry["models"] = {"embedding": "e" * 64, "segmentation": "f" * 64}
        self.entry["library_artifacts"] = {
            area: {"tier": "sm75", "artifact": {"kind": "PtxJit", "sha256": "c" * 64}}
            for area in ("fbank", "embedding", "segmentation")
        }
        self.entry["der_inventory"] = {
            "kind": "Locked",
            "inputs": {
                "manifest_sha256": "5" * 64,
                "reference_sha256": "6" * 64,
                "files": 2,
            },
            "pipeline_config_sha256": "7" * 64,
            "control_archive_sha256": "8" * 64,
        }
        self.entry["der"] = self.der_receipt(self.entry)

    def der_receipt(self, entries):
        if isinstance(entries, dict):
            entries = [entries]
        first = entries[0]
        domain = records.der_evidence.PlanDomain.parse(first["boundary_domain"])
        plans = []
        for mode in sorted(
            {row["tuple"][2] for entry in entries for row in entry["configurations"]}
        ):
            plan = {
                "device_scope": first["speed_scope"],
                "math": mode,
                "models": first["models"],
                "inputs": {
                    "manifest_sha256": "5" * 64,
                    "reference_sha256": "6" * 64,
                    "files": 2,
                },
                "pipeline_config_sha256": "7" * 64,
                "library_artifacts": first["library_artifacts"],
                "routes": records.der_evidence.routes(entries, domain, mode),
            }
            identity = records.der_evidence.digest(
                {
                    name: plan[name]
                    for name in (
                        "device_scope",
                        "math",
                        "models",
                        "inputs",
                        "pipeline_config_sha256",
                        "library_artifacts",
                    )
                }
            )
            baseline = {
                "routes": domain.library_routes(),
                "control_archive_sha256": "8" * 64,
                "metrics": {
                    "der": 1.0,
                    "output_sha256": "9" * 64,
                    "identity_sha256": identity,
                },
            }
            baseline["metrics"]["execution_sha256"] = (
                records.der_evidence.execution_identity(
                    identity, baseline["routes"], baseline["control_archive_sha256"]
                )
            )
            candidate = {
                "execution_sha256": records.der_evidence.execution_identity(
                    identity, plan["routes"]
                ),
                "der": 1.0,
                "output_sha256": "a" * 64,
                "identity_sha256": identity,
            }
            plan.update(
                baseline=baseline,
                candidate=candidate,
                verdict={
                    "passed": True,
                    "baseline_sha256": records.der_evidence.digest(baseline),
                    "candidate_sha256": records.der_evidence.digest(candidate),
                    "policy_sha256": "b" * 64,
                },
            )
            plans.append(plan)
        return self.store({"schema": 1, "plans": plans})

    def refresh_lock(self):
        path = self.root / records.ACCEPTANCE
        lock = self.root / "scripts/cuda/qualify/LOCK"
        lock.write_text(
            json.dumps(
                {
                    "schema": 1,
                    "files": {
                        records.ACCEPTANCE: hashlib.sha256(
                            path.read_bytes()
                        ).hexdigest()
                    },
                }
            )
        )

    def write_embed(self, tier, features):
        path = self.root / "src/inference/cuda/kernels.rs"
        previous = path.read_text() if path.exists() else ""
        path.write_text(
            previous
            + "tier_ptx!("
            + json.dumps(features)
            + ', "ptx/lstm.'
            + tier
            + '", [75, 80, 86, 89, 90, 120])\n'
        )

    def check(self, entries):
        return records.check_table_records(entries, self.root)

    def store(self, record):
        data = json.dumps(record).encode()
        digest = hashlib.sha256(data).hexdigest()
        path = self.cache / "records" / digest
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(data)
        return digest


class Records(RecordsFixture):
    def test_conflicting_artifacts_have_no_second_production_owner(self):
        conflicting = copy.deepcopy(self.entry)
        conflicting["artifact"] = {"kind": "Cubin", "arch": "12.0", "sha256": "b" * 64}
        with self.assertRaisesRegex(
            records.Rejected, "conflicting production artifact bindings"
        ):
            self.check([self.entry, conflicting])
        with self.assertRaisesRegex(
            records.Rejected, "conflicting production artifact bindings"
        ):
            records.check_table([self.entry, conflicting], self.root)

    def test_multiple_records_can_share_one_module_binding(self):
        other = {**self.entry, "record": "b" * 64}
        records.unique_artifact_owners([self.entry, other])
        self.assertEqual(len(self.check([self.entry, self.entry])["entries"]), 2)

    def test_different_point_scopes_can_load_different_modules(self):
        other = copy.deepcopy(self.entry)
        other["speed_scope"]["sm_count"] = 70
        other["artifact"] = {"kind": "PtxJit", "sha256": "b" * 64}
        records.unique_artifact_owners([self.entry, other])
        legacy = {
            **self.entry,
            "speed_scope": {"kind": "LegacyCapability", "capability": "12.0"},
        }
        with self.assertRaisesRegex(records.Rejected, "conflicting production"):
            records.unique_artifact_owners([legacy, other])
        other["artifact"] = self.entry["artifact"]
        other["tier"] = "sm80"
        with self.assertRaisesRegex(records.Rejected, "conflicting production"):
            records.unique_artifact_owners([legacy, other])

    def test_stale_control_environment_cannot_authorize_production(self):
        for field, value in (
            ("driver_version", "581.0"),
            ("cudnn_version", 90900),
            ("cublas_version", 120805),
        ):
            raw = copy.deepcopy(self.record)
            raw["tiers"]["sm75"]["numeric"]["library"][0]["device"][field] = value
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(records.Rejected, "stale comparison"),
            ):
                self.check([{**self.entry, "record": self.store(raw)}])

    def test_same_version_alternate_loaded_library_rejects_and_scan_is_not_evidence(
        self,
    ):
        for fault in ("missing", "alternate"):
            raw = copy.deepcopy(self.record)
            process = raw["tiers"]["sm75"]["numeric"]["library"][0]
            if fault == "missing":
                process.pop("loaded_libraries")
            else:
                process["loaded_libraries"].pop("/lib/libcudnn.so.9")
                process["loaded_libraries"]["/alternate/libcudnn.so.9"] = "4" * 64
            with (
                self.subTest(fault=fault),
                self.assertRaisesRegex(records.Rejected, "loaded"),
            ):
                self.check([{**self.entry, "record": self.store(raw)}])
        # sanitizer files are a separate tool policy, not the libraries comparisons loaded
        raw = copy.deepcopy(self.record)
        raw["tiers"]["sm75"]["sanitizer_fingerprint"]["/lib/libcudnn.so.9"] = "4" * 64
        self.assertEqual(
            self.check([{**self.entry, "record": self.store(raw)}])["entries"][0][
                "environment"
            ]["kind"],
            "LoadedLibraries",
        )

    def test_same_device_entries_require_one_control_fingerprint(self):
        raw = copy.deepcopy(self.record)
        for process in (
            raw["tiers"]["sm75"]["numeric"]["library"]
            + raw["tiers"]["sm75"]["numeric"]["candidate"]
        ):
            process["loaded_libraries"]["/lib/libcudnn.so.9"] = "4" * 64
        other = {**self.entry, "record": self.store(raw)}
        with self.assertRaisesRegex(records.Rejected, "different Library fingerprints"):
            self.check([self.entry, other])
        summary = records.derive_summaries(
            [self.entry], self.root, records=self.cache / "records"
        )
        other_summary = records.derive_summaries(
            [other], self.root, records=self.cache / "records"
        )
        summary["records"].update(other_summary["records"])
        path = self.root / records.ACCEPTANCE
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(records.canonical_summaries(summary))
        self.refresh_lock()
        with self.assertRaisesRegex(records.Rejected, "different Library fingerprints"):
            records.check_table([self.entry, other], self.root)

    def test_accuracy_and_speed_can_use_distinct_qualified_records(self):
        raw = {**self.record, "accuracy_snapshot": "separate process"}
        accuracy = self.store(raw)
        entry = {**self.entry, "accuracy_record": accuracy}
        checked = self.check([entry])
        self.assertEqual(
            {row["record"] for row in checked["entries"]},
            {accuracy, self.entry["record"]},
        )
        summary = records.write_summaries(
            [entry], self.root, records=self.cache / "records"
        )
        self.assertEqual(set(summary["records"]), {accuracy, self.entry["record"]})
        self.refresh_lock()
        self.assertEqual(
            records.check_table([entry], self.root)["verification"], "locked-summary"
        )
        self.assertEqual(
            records.check_table([entry], self.root, records=self.cache / "records")[
                "verification"
            ],
            "raw-records",
        )
        entry["accuracy_record"] = "0" * 64
        with self.assertRaisesRegex(records.Rejected, "unresolved"):
            self.check([entry])

    def test_swapping_a_legacy_wave_pin_for_a_fixed_entry_rejects_both_modes(self):
        area = "resnet"
        for old in self.files.values():
            for name in old:
                if name.endswith("candidate.rs"):
                    continue
                new = name.replace("candidate/lstm", "candidate/conv").replace(
                    "lstm", area
                )
                path = self.root / new
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(new)
        kernels = self.root / "src/inference/cuda/kernels.rs"
        kernels.write_text(kernels.read_text().replace("ptx/lstm", "ptx/resnet"))
        files = records.shipped_files(self.root, area)
        boundary = "resnet.layer2.1.conv1"
        triple = [boundary, 1, "fp32"]
        child = copy.deepcopy(self.child)
        child.update(
            target=area,
            coverage_declared={"triples": [triple]},
            accepted_tuples=[triple],
            code_sha256={
                name: pin for hashes in files.values() for name, pin in hashes.items()
            },
        )
        ptx_hash = files["ptx"]["src/inference/cuda/ptx/resnet.sm75.ptx"]
        artifact = {"kind": "PtxJit", "sha256": ptx_hash}
        child["loaded_ptx"]["modules"] = [
            {
                "area": area,
                "path": "src/inference/cuda/ptx/resnet.sm75.ptx",
                "tier": "sm75",
                "sha256": ptx_hash,
                "embedded_ptx_sha256": ptx_hash,
                "artifact": artifact,
            }
        ]
        raw = complete_fixture(
            {**self.record, "target": area, "tiers": {"sm75": child}}
        )
        coverage = {
            "entries": [{"layers": [boundary], "batches": [1], "maths": ["fp32"]}]
        }
        entry = {
            **self.entry,
            "area": area,
            "artifact": artifact,
            "record": self.store(raw),
            "coverage": coverage,
            "candidate_coverage": coverage,
            "configurations": [
                {
                    "tuple": triple,
                    "pin": {"kind": "Conv", "selection": "LegacyWaves", "shape": "C64"},
                }
            ],
        }
        entry["boundary_domain"] = [{"boundary": boundary, "batches": [1, 32]}]
        entry["der"] = self.der_receipt(entry)
        self.check([entry])
        records.write_summaries([entry], self.root, records=self.cache / "records")
        self.refresh_lock()
        records.check_table([entry], self.root)
        entry["configurations"][0]["pin"] = {
            "kind": "Conv",
            "selection": "Kernel",
            "entry": "C64",
        }
        with self.assertRaisesRegex(
            records.Rejected, "configuration differs from its record pin"
        ):
            self.check([entry])
        with self.assertRaisesRegex(
            records.Rejected, "configuration differs from its record pin"
        ):
            records.check_table([entry], self.root)
        with self.assertRaisesRegex(
            records.Rejected, "configuration differs from its record pin"
        ):
            records.check_table([entry], self.root, records=self.cache / "records")

    def test_configuration_receipts_are_required_in_records_and_exports(self):
        missing = copy.deepcopy(self.record)
        missing["tiers"]["sm75"]["numeric"]["candidate"][0].pop("configurations")
        with self.assertRaisesRegex(records.Rejected, "configuration receipts"):
            self.check([{**self.entry, "record": self.store(missing)}])
        entry = copy.deepcopy(self.entry)
        entry.pop("configurations")
        with self.assertRaisesRegex(records.Rejected, "configuration receipts"):
            self.check([entry])
        missing["tiers"]["sm75"]["numeric"]["candidate"][0]["configurations"] = []
        with self.assertRaisesRegex(records.Rejected, "missing planned configuration"):
            self.check([{**self.entry, "record": self.store(missing)}])

    def test_positive_and_four_required_negative_cases(self):
        self.assertFalse(self.check([self.entry])["entries"][0]["legacy"])
        for field, value, reason in [
            ("devices", ["8.0"], "device capability"),
            ("tier", "sm80", "tier not accepted"),
            ("record", "0" * 64, "unresolved"),
        ]:
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(records.Rejected, reason),
            ):
                self.check([{**self.entry, field: value}])
        outside = copy.deepcopy(self.entry)
        outside["coverage"]["entries"][0]["batches"] = [32]
        with self.assertRaisesRegex(records.Rejected, "tuples outside"):
            self.check([outside])

    def test_missing_or_failed_provenance_never_passes(self):
        for field in ("code_sha256", "accepted_tuples", "loaded_ptx"):
            child = copy.deepcopy(self.child)
            child.pop(field)
            record = {**self.record, "tiers": {"sm75": child}}
            with (
                self.subTest(field=field),
                self.assertRaises((records.Rejected, KeyError)),
            ):
                self.check([{**self.entry, "record": self.store(record)}])
        with self.assertRaisesRegex(records.Rejected, "unresolved"):
            self.check([{**self.entry, "der": "0" * 64}])
        path = records.record_path(self.entry["record"])
        path.write_text("changed")
        with self.assertRaisesRegex(records.Rejected, "hash mismatch"):
            self.check([self.entry])

    def test_all_bound_files_reject_edits_removals_and_additions(self):
        for group, hashes in self.files.items():
            for name in hashes:
                path = self.root / name
                original = path.read_bytes()
                for mutation in ("edit", "remove"):
                    with self.subTest(group=group, name=name, mutation=mutation):
                        if mutation == "edit":
                            path.write_bytes(original + b" drift")
                        else:
                            path.unlink()
                        with self.assertRaises(records.Rejected):
                            self.check([self.entry])
                        path.write_bytes(original)
        for name in (
            "src/inference/cuda/candidate/lstm/new.rs",
            "crates/speakrs-cuda-kernels/src/lstm/new.rs",
            "src/inference/cuda/ptx/lstm.extra.manifest",
            "src/inference/cuda/ptx/lstm.sm80.ptx",
        ):
            with self.subTest(name=name):
                path = self.root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("new unbound file")
                with self.assertRaises(records.Rejected):
                    self.check([self.entry])
                path.unlink()

    def test_invalid_bindings_and_missing_declaration_fail_closed(self):
        for field in ("code_sha256", "coverage_declared"):
            child = copy.deepcopy(self.child)
            child.pop(field)
            record = {**self.record, "tiers": {"sm75": child}}
            with self.subTest(field=field), self.assertRaises(records.Rejected):
                self.check([{**self.entry, "record": self.store(record)}])
        for field, value in (
            ("path", "../lstm.sm75.ptx"),
            ("sha256", "bad"),
            ("tier", "sm80"),
        ):
            child = copy.deepcopy(self.child)
            child["loaded_ptx"]["modules"][0][field] = value
            record = {**self.record, "tiers": {"sm75": child}}
            with self.subTest(field=field), self.assertRaises(records.Rejected):
                self.check([{**self.entry, "record": self.store(record)}])
        child = copy.deepcopy(self.child)
        child["code_sha256"]["src/inference/cuda/candidate/lstm.rs"] = "bad"
        with self.assertRaises(records.Rejected):
            self.check(
                [
                    {
                        **self.entry,
                        "record": self.store({**self.record, "tiers": {"sm75": child}}),
                    }
                ]
            )

    def test_candidate_coverage_drift_and_missing_export(self):
        entry = copy.deepcopy(self.entry)
        entry["candidate_coverage"]["entries"][0]["maths"] = ["tf32"]
        with self.assertRaisesRegex(records.Rejected, "candidate coverage differs"):
            self.check([entry])
        entry.pop("candidate_coverage")
        with self.assertRaisesRegex(records.Rejected, "missing Rust"):
            self.check([entry])

    def test_production_subset_and_legacy_stress_normalization(self):
        child = copy.deepcopy(self.child)
        child["coverage_declared"]["triples"].extend(
            [
                ["lstm.stack", batch, "fp32"]
                for batch in records.TEST_BATCHES
                if batch != 1
            ]
        )
        child["accepted_tuples"].append(["lstm.stack", 32, "fp32"])
        child["numeric"]["candidate"][0]["configurations"] = [
            {"tuple": row, "pin": {"kind": "Lstm", "selection": "LegacyCooperative"}}
            for row in child["coverage_declared"]["triples"]
        ]
        entry = copy.deepcopy(self.entry)
        entry["candidate_coverage"]["entries"][0]["batches"] = "all"
        entry["record"] = self.store(
            complete_fixture({**self.record, "tiers": {"sm75": child}})
        )
        self.check([entry])

    def test_legacy_binding_is_explicit_and_reports_evidence_gap(self):
        child = copy.deepcopy(self.child)
        child.pop("code_sha256")
        child["device_sm"] = "12.0"
        child["driver_sha256"] = "d" * 64
        child["phases_run"] = list(records.LEGACY_PHASES)
        child["reason"] = "all required checks passed"
        raw = complete_fixture({**self.record, "schema": 3, "tiers": {"sm75": child}})
        record_hash = self.store(raw)
        entry: dict = {
            **self.entry,
            "record": record_hash,
            "speed_scope": {"kind": "LegacyCapability", "capability": "12.0"},
        }
        entry["der"] = self.der_receipt(entry)
        binding = {
            "files": records.shipped_files(self.root, "lstm", legacy=True),
            "lock_digest": self.record["lock_digest"],
        }
        with (
            patch.dict(records.LEGACY, {record_hash: "lstm"}),
            patch.dict(records.configurations.LEGACY_AREAS, {record_hash: "lstm"}),
            patch.dict(records.LEGACY_BINDINGS, {record_hash: binding}),
        ):
            result = self.check([entry])["entries"][0]
            self.assertEqual(result["source_evidence_gap"], records.SOURCE_EVIDENCE_GAP)
            binding["files"]["host_sources"]["src/inference/cuda/candidate.rs"] = (
                "0" * 64
            )
            with self.assertRaisesRegex(records.Rejected, "qualification file differs"):
                self.check([entry])
        with (
            patch.dict(records.LEGACY, {record_hash: "lstm"}),
            patch.dict(records.configurations.LEGACY_AREAS, {record_hash: "lstm"}),
        ):
            with self.assertRaisesRegex(
                records.Rejected, "invalid legacy acceptance binding"
            ):
                self.check([entry])

    def test_speed_scope_must_match_record_evidence(self):
        self.check([self.entry])
        point = self.entry["speed_scope"]
        for scope in (
            None,
            {"kind": "LegacyCapability", "capability": "12.0"},
            {**point, "sm_count": 70},
            {**point, "device_name": "Other GPU"},
        ):
            with (
                self.subTest(scope=scope),
                self.assertRaisesRegex(records.Rejected, "speed scope"),
            ):
                self.check([{**self.entry, "speed_scope": scope}])

    def test_legacy_sources_and_coverage_remain_strict(self):
        child = copy.deepcopy(self.child)
        child.pop("code_sha256")
        child["device_sm"] = "12.0"
        child["driver_sha256"] = "d" * 64
        child["phases_run"] = list(records.LEGACY_PHASES)
        child["reason"] = "all required checks passed"
        record_hash = self.store(
            complete_fixture({**self.record, "schema": 3, "tiers": {"sm75": child}})
        )
        entry: dict = {
            **self.entry,
            "record": record_hash,
            "speed_scope": {"kind": "LegacyCapability", "capability": "12.0"},
        }
        entry["der"] = self.der_receipt(entry)
        binding = {
            "files": records.shipped_files(self.root, "lstm", legacy=True),
            "lock_digest": self.record["lock_digest"],
        }
        with (
            patch.dict(records.LEGACY, {record_hash: "lstm"}),
            patch.dict(records.configurations.LEGACY_AREAS, {record_hash: "lstm"}),
            patch.dict(records.LEGACY_BINDINGS, {record_hash: binding}),
        ):
            for hashes in binding["files"].values():
                for name in hashes:
                    path = self.root / name
                    original = path.read_bytes()
                    with self.subTest(name=name):
                        path.write_bytes(original + b" drift")
                        with self.assertRaisesRegex(
                            records.Rejected, "qualification file differs"
                        ):
                            self.check([entry])
                        path.write_bytes(original)
            drift = copy.deepcopy(entry)
            drift["candidate_coverage"]["entries"][0]["maths"] = ["tf32"]
            with self.assertRaisesRegex(records.Rejected, "candidate coverage differs"):
                self.check([drift])

    def test_incomplete_collection_and_invalid_check_outcomes_fail_closed(self):
        cases = []
        for field, value in (
            ("phases_run", ["numeric", "timing", "profile", "sanitize"]),
            ("coverage", {"tier": "sm80"}),
            ("target", "resnet"),
            ("implementation", "Library"),
            ("reason", "driver stopped after timing"),
            ("status", "blocked"),
        ):
            raw = copy.deepcopy(self.record)
            raw["tiers"]["sm75"][field] = value
            cases.append((field, raw))
        for mutation in (
            "missing",
            "duplicate",
            "nonboolean",
            "aggregate",
            "extra_aggregate",
        ):
            raw = copy.deepcopy(self.record)
            checks = raw["tiers"]["sm75"]["checks"]
            if mutation == "missing":
                checks.pop()
            elif mutation == "duplicate":
                checks.append(copy.deepcopy(checks[0]))
            elif mutation == "nonboolean":
                checks[0]["passed"] = 1
            elif mutation == "aggregate":
                raw["checks"][0]["evidence"] = {"changed": True}
            else:
                raw["checks"].append({"check": "sm75/extra", "passed": True})
            cases.append((mutation, raw))
        for name, raw in cases:
            with self.subTest(name=name), self.assertRaises(records.Rejected):
                self.check([{**self.entry, "record": self.store(raw)}])
        for schema in (2, 6, True):
            with (
                self.subTest(schema=schema),
                self.assertRaisesRegex(records.Rejected, "schema"),
            ):
                self.check(
                    [
                        {
                            **self.entry,
                            "record": self.store({**self.record, "schema": schema}),
                        }
                    ]
                )

    def test_missing_projection_baseline_and_parent_checks_fail_closed(self):
        raw = copy.deepcopy(self.record)
        raw["tiers"]["sm75"]["checks"] = [
            check
            for check in raw["tiers"]["sm75"]["checks"]
            if check["check"] != "sanitizer:ProjectionBaseline/memcheck"
        ]
        with self.assertRaisesRegex(records.Rejected, "missing required tier checks"):
            self.check([{**self.entry, "record": self.store(raw)}])
        raw = copy.deepcopy(self.record)
        raw["checks"].pop()
        with self.assertRaisesRegex(records.Rejected, "aggregate"):
            self.check([{**self.entry, "record": self.store(raw)}])

    def test_interrupted_legacy_verdict_cannot_pass_without_error_check(self):
        raw = copy.deepcopy(self.record)
        raw["schema"] = 3
        child = raw["tiers"]["sm75"]
        child["phases_run"] = list(records.LEGACY_PHASES)
        child["reason"] = "all required checks passed"
        complete_fixture(raw)
        records.complete_collection(raw)
        child["status"] = "blocked"
        child["reason"] = "collection stopped"
        with self.assertRaisesRegex(records.Rejected, "inconsistent tier verdict"):
            records.complete_collection(raw)

    def test_full_stress_coverage_is_bound_and_hashes_are_canonical(self):
        raw = copy.deepcopy(self.record)
        declared = [["lstm.stack", batch, "fp32"] for batch in records.TEST_BATCHES]
        raw["tiers"]["sm75"]["coverage_declared"]["triples"] = declared
        entry = copy.deepcopy(self.entry)
        entry["candidate_coverage"]["entries"][0]["batches"] = "all"
        entry["record"] = self.store(complete_fixture(raw))
        result = self.check([entry])["entries"][0]
        self.assertEqual(result["coverage_sha256"], result["recorded_coverage_sha256"])
        self.assertEqual(
            result["coverage_sha256"],
            records.coverage_digest(set(map(tuple, reversed(declared)))),
        )
        for batch in (7, 33, 64):
            drift = copy.deepcopy(entry)
            drift["candidate_coverage"]["entries"][0]["batches"] = [
                sample for sample in records.TEST_BATCHES if sample != batch
            ]
            with (
                self.subTest(batch=batch),
                self.assertRaisesRegex(records.Rejected, "candidate coverage differs"),
            ):
                self.check([drift])

    def test_missing_numeric_speed_and_paired_markers_fail_closed(self):
        markers = (
            "determinism:fp32/first/b1/lstm.stack",
            "determinism:fp32/first/b1/lstm.stack/switched",
            "layer:fp32/lstm.stack",
            "secret:fp32/secret/b1/lstm.stack",
            "speed:fp32/first/b1/lstm.stack",
            "stage:fp32/first/b1/stage/switched",
            "paired_stage:fp32/first/b1/stage",
            "paired_output:fp32/first/b1/stage",
        )
        for name in markers:
            raw = copy.deepcopy(self.record)
            raw["tiers"]["sm75"]["checks"] = [
                check
                for check in raw["tiers"]["sm75"]["checks"]
                if check["check"] != name
            ]
            raw["checks"] = [
                check for check in raw["checks"] if check["check"] != f"sm75/{name}"
            ]
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(
                    records.Rejected, "missing required tier checks"
                ),
            ):
                self.check([{**self.entry, "record": self.store(raw)}])

    def test_missing_duplicate_timing_and_coverage_cases_fail_closed(self):
        for mutation in ("missing_timing", "duplicate_timing", "cases", "math"):
            raw = copy.deepcopy(self.record)
            child = raw["tiers"]["sm75"]
            if mutation == "missing_timing":
                child["timing"].pop()
            elif mutation == "duplicate_timing":
                child["timing"].append(copy.deepcopy(child["timing"][0]))
            else:
                child["coverage"][mutation].pop()
            with self.subTest(mutation=mutation), self.assertRaises(records.Rejected):
                self.check([{**self.entry, "record": self.store(raw)}])

    def test_tf32_stage_checks_follow_record_schema(self):
        for schema, marker in (
            (3, "stage:tf32/first/b1/stage"),
            (4, "stage_truth:tf32/first/b1/stage"),
        ):
            raw = copy.deepcopy(self.record)
            raw["schema"] = schema
            raw["tiers"]["sm75"]["coverage_declared"]["triples"].append(
                ["lstm.stack", 1, "tf32"]
            )
            complete_fixture(raw)
            records.complete_collection(raw)
            raw["tiers"]["sm75"]["checks"] = [
                check
                for check in raw["tiers"]["sm75"]["checks"]
                if check["check"] != marker
            ]
            raw["checks"] = [
                check for check in raw["checks"] if check["check"] != f"sm75/{marker}"
            ]
            with (
                self.subTest(schema=schema),
                self.assertRaisesRegex(
                    records.Rejected, "missing required tier checks"
                ),
            ):
                records.complete_collection(raw)

    def test_stress_cases_do_not_grant_production_coverage(self):
        for batch in (7, 33, 64):
            outside = copy.deepcopy(self.entry)
            outside["coverage"]["entries"][0]["batches"] = [batch]
            with self.assertRaisesRegex(records.Rejected, "stress batch"):
                self.check([outside])

    def test_device_and_actual_area_tier_are_required(self):
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
                {
                    "area": "lstm",
                    "tier": "sm75",
                    "artifact": {"kind": "PtxJit", "sha256": "a" * 64},
                }
            ],
            "observed_sm_clock": {"samples": 4, "min_mhz": 2400, "max_mhz": 2700},
        }
        qualify.validate_target(process, "sm75", candidate_area="lstm")
        with self.assertRaisesRegex(records.Rejected, "differs from requested"):
            qualify.validate_target(process, "sm80", candidate_area="lstm")
        for key in process["device"]:
            missing = copy.deepcopy(process)
            del missing["device"][key]
            with (
                self.subTest(key=key),
                self.assertRaises(records.Rejected),
            ):
                qualify.validate_target(missing, "sm75", candidate_area="lstm")
        process.pop("observed_sm_clock")
        with self.assertRaisesRegex(records.Rejected, "observed SM clocks"):
            qualify.validate_target(process, "sm75", candidate_area="lstm")

    def test_probe_variants_bind_area_tier_bytes_and_entries(self):
        outputs = []
        for tier in ("sm75", "sm80"):
            path = self.root / f"src/inference/cuda/ptx/probe.{tier}.ptx"
            path.write_text(
                f".target {tier}\n.visible .entry fixture_probe() {{ ret; }}"
            )
            module = {
                "area": "probe",
                "tier": tier,
                "sha256": qualify.sha(path),
                "embedded_ptx_sha256": qualify.sha(path),
                "artifact": {"kind": "PtxJit", "sha256": qualify.sha(path)},
                "entries": qualify.ENTRY.findall(path.read_text()),
            }
            allow, evidence = qualify.verify_modules([module], self.root)
            self.assertTrue(allow.entries)
            self.assertEqual(evidence["modules"][0]["tier"], tier)
            outputs.append(module["entries"])
            wrong = "sm80" if tier == "sm75" else "sm75"
            with self.assertRaisesRegex(records.Rejected, "came from"):
                qualify.verify_modules([{**module, "tier": wrong}], self.root)
        self.assertEqual(*outputs)

    def test_multiset_proof_does_not_collapse_duplicate_launches(self):
        from collections import Counter

        process: dict = {
            "graph_evidence": [
                {"scope": "candidate", "case": "c", "kernels": ["k", "k"]}
            ]
        }
        with patch.object(
            qualify, "window_kernels", return_value={"c": Counter({"k": 2})}
        ):
            self.assertEqual(
                qualify.capture_multisets(
                    Path("trace"), [process], "nonce", required=1
                )["compared"],
                1,
            )
        with (
            patch.object(
                qualify, "window_kernels", return_value={"c": Counter({"k": 1})}
            ),
            self.assertRaisesRegex(records.Rejected, "capture != eager"),
        ):
            qualify.capture_multisets(Path("trace"), [process], "nonce", required=1)


class LegacyAmendments(unittest.TestCase):
    def test_later_pins_amend_earlier_ones_and_all_are_reported(self):
        path = "src/inference/cuda/candidate.rs"
        first = {
            "path": path,
            "acceptance_sha256": "a" * 64,
            "infrastructure_sha256": "b" * 64,
            "reason": "first",
        }
        second = {
            **first,
            "acceptance_sha256": "b" * 64,
            "infrastructure_sha256": "c" * 64,
        }
        unrelated = {
            **first,
            "acceptance_sha256": "d" * 64,
            "infrastructure_sha256": "e" * 64,
        }
        binding = {"files": {"host_sources": {path: "a" * 64}}}
        with patch.object(
            records, "INFRASTRUCTURE_AMENDMENTS", [first, second, unrelated]
        ):
            self.assertEqual(
                records.amended_legacy_files(binding),
                {"host_sources": {path: "c" * 64}},
            )
            self.assertEqual(records.legacy_amendments(binding), [first, second])
        # the archived binding keeps its acceptance hash
        self.assertEqual(binding["files"]["host_sources"][path], "a" * 64)
        # a pin listed before the one it amends never applies
        with patch.object(records, "INFRASTRUCTURE_AMENDMENTS", [second, first]):
            self.assertEqual(
                records.amended_legacy_files(binding),
                {"host_sources": {path: "b" * 64}},
            )


class ProductionLoad(RecordsFixture):
    def variant(self, tier):
        path = self.root / f"src/inference/cuda/ptx/lstm.{tier}.ptx"
        path.write_text(f"fixture {tier}")
        if tier in records.TIER_CAPABILITIES:
            self.write_embed(tier, ["cuda-" + tier])

    def test_minimum_feature_can_select_lower_than_default(self):
        self.variant("sm80")
        self.variant("sm90")
        for tier in ("sm75", "sm80", "sm90"):
            raw = copy.deepcopy(self.record)
            child = raw["tiers"]["sm75"]
            child["requested_tier"] = tier
            child["code_sha256"].update(records.shipped_files(self.root, "lstm")["ptx"])
            path = f"src/inference/cuda/ptx/lstm.{tier}.ptx"
            child["loaded_ptx"]["modules"][0].update(
                tier=tier, path=path, sha256=records.file_digest(self.root / path)
            )
            raw.update(requested_tier=tier, tiers={tier: child})
            complete_fixture(raw)
            module = child["loaded_ptx"]["modules"][0]
            module.update(
                embedded_ptx_sha256=module["sha256"],
                artifact={"kind": "PtxJit", "sha256": module["sha256"]},
            )
            entry = {
                **self.entry,
                "tier": tier,
                "artifact": module["artifact"],
                "record": self.store(raw),
            }
            entry["der"] = self.der_receipt(entry)
            with self.subTest(tier=tier):
                result = self.check([entry])["entries"][0]
                self.assertEqual(result["tier"], tier)
                self.assertEqual(result["device_capability"], "12.0")
                self.assertEqual(result["tuples"], [["lstm.stack", 1, "fp32"]])

    def test_forced_tier_has_no_possible_production_build_on_device(self):
        self.variant("sm90")
        with self.assertRaisesRegex(records.Rejected, "cannot realize"):
            records.production_load("lstm", "sm90", "8.0", self.root)

    def test_binding_tier_stays_loadable_beside_a_higher_variant(self):
        # the only build that embeds sm75 also embeds sm80, which the old
        # highest-variant loader would have chosen instead
        (self.root / "src/inference/cuda/kernels.rs").write_text("")
        self.write_embed("sm75", ["cuda-sm120"])
        (self.root / "src/inference/cuda/ptx/lstm.sm80.ptx").write_text("fixture sm80")
        self.write_embed("sm80", ["cuda-sm120"])
        records.production_load("lstm", "sm75", "12.0", self.root)
        records.production_load("lstm", "sm80", "12.0", self.root)
        # no build the device supports embeds the tier
        with self.assertRaisesRegex(records.Rejected, "cannot realize"):
            records.production_load("lstm", "sm80", "8.9", self.root)

    def test_malformed_device_and_unavailable_tiers_fail_closed(self):
        for capability in ("8", "8.0.0", "8.10", "-8.0", "NaN", "7.0"):
            with (
                self.subTest(capability=capability),
                self.assertRaises(records.Rejected),
            ):
                records.production_load("lstm", "sm75", capability, self.root)
        for tier in ("sm80", "sm86", "SM75"):
            with self.subTest(tier=tier), self.assertRaises(records.Rejected):
                records.production_load("lstm", tier, "12.0", self.root)
        self.variant("sm86")
        # an unembedded disk file cannot change production loadability
        records.production_load("lstm", "sm75", "12.0", self.root)

    def test_raw_gate_rejects_forced_tier_and_accepts_minimum_feature(self):
        self.variant("sm90")
        raw = copy.deepcopy(self.record)
        child = raw["tiers"]["sm75"]
        child["code_sha256"].update(records.shipped_files(self.root, "lstm")["ptx"])
        good = {**self.entry, "record": self.store(raw)}
        records.check_table_records([good], self.root)
        child["requested_tier"] = "sm90"
        child["device"]["compute_capability"] = "8.0"
        child["loaded_ptx"]["modules"][0].update(
            tier="sm90",
            path="src/inference/cuda/ptx/lstm.sm90.ptx",
            sha256=records.file_digest(
                self.root / "src/inference/cuda/ptx/lstm.sm90.ptx"
            ),
        )
        raw.update(requested_tier="sm90", tiers={"sm90": child})
        complete_fixture(raw)
        bad = {
            **self.entry,
            "tier": "sm90",
            "devices": ["8.0"],
            "record": self.store(raw),
        }
        with self.assertRaisesRegex(records.Rejected, "cannot realize"):
            records.check_table_records([bad], self.root)


class ArtifactEvidence(RecordsFixture):
    def cubin(self):
        module = self.child["loaded_ptx"]["modules"][0]
        path = self.root / "src/inference/cuda/ptx/lstm.sm75.sm_120.cubin"
        path.write_bytes(b"exact device cubin fixture")
        artifact = {
            "kind": "Cubin",
            "arch": "12.0",
            "sha256": records.file_digest(path),
            "ptx_sha256": module["sha256"],
            "ptxas_version": "fixture ptxas",
            "ptxas_flags": "-arch={arch} {input} -o {output}",
        }
        manifest = self.root / "src/inference/cuda/ptx/lstm.manifest"
        manifest.write_text(
            f"ptxas = {artifact['ptxas_version']}\nptxas-flags = {artifact['ptxas_flags']}\n[sm75]\nptx = {module['sha256']}\ncubin.sm_120 = {artifact['sha256']} {module['sha256']}\n"
        )
        self.child["code_sha256"]["src/inference/cuda/ptx/lstm.manifest"] = (
            records.file_digest(manifest)
        )
        self.child["code_sha256"][path.relative_to(self.root).as_posix()] = (
            records.file_digest(path)
        )
        module["artifact"] = artifact
        self.entry["artifact"] = {
            name: artifact[name] for name in ("kind", "arch", "sha256")
        }
        self.entry["record"] = self.store(self.record)
        self.entry["der"] = self.der_receipt(self.entry)
        return artifact

    def test_cubin_metadata_is_pinned_and_jit_does_not_match(self):
        artifact = self.cubin()
        evidence = self.check([self.entry])["entries"][0]
        self.assertEqual(evidence["artifact"], artifact)
        self.assertEqual(evidence["device"]["sm_count"], 36)
        self.assertEqual(evidence["device"]["l2_bytes"], 33554432)
        self.assertEqual(evidence["device"]["driver_version"], "580.0")
        jit = {"kind": "PtxJit", "sha256": artifact["ptx_sha256"]}
        with self.assertRaisesRegex(records.Rejected, "selected artifact differs"):
            self.check([{**self.entry, "artifact": jit}])
        for field, bad in (
            ("arch", "8.9"),
            ("sha256", "b" * 64),
            ("ptx_sha256", "b" * 64),
            ("ptxas_version", "other assembler"),
            ("ptxas_flags", "-different-flags"),
            ("extra", "untrusted"),
        ):
            raw = copy.deepcopy(self.record)
            raw["tiers"]["sm75"]["loaded_ptx"]["modules"][0]["artifact"][field] = bad
            with self.subTest(field=field), self.assertRaises(records.Rejected):
                self.check([{**self.entry, "record": self.store(raw)}])

    def test_jit_record_binds_every_candidate_cubin_and_manifest(self):
        self.check([self.entry])
        for name in (
            "src/inference/cuda/ptx/lstm.sm75.sm_120.cubin",
            "src/inference/cuda/ptx/lstm.manifest",
        ):
            path = self.root / name
            original = path.read_bytes()
            path.write_bytes(original + b" drift")
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(records.Rejected, "qualification file differs"),
            ):
                self.check([self.entry])
            path.write_bytes(original)
        for name in (
            "src/inference/cuda/ptx/lstm.sm75.sm_120.cubin",
            "src/inference/cuda/ptx/lstm.sm75.ptx",
            "src/inference/cuda/ptx/lstm.manifest",
        ):
            raw = copy.deepcopy(self.record)
            del raw["tiers"]["sm75"]["code_sha256"][name]
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(
                    records.Rejected,
                    "missing source or manifest hashes|missing candidate PTX code hashes",
                ),
            ):
                self.check([{**self.entry, "record": self.store(raw)}])

    def test_cubin_table_pin_is_checked_against_shipped_bytes(self):
        artifact = self.cubin()
        entries = [
            {**self.entry, "artifact": {**self.entry["artifact"], "sha256": "b" * 64}}
        ]
        with self.assertRaisesRegex(records.Rejected, "production pin differs"):
            self.check(entries)
        self.check([self.entry])
        records.write_summaries([self.entry], self.root, records=self.cache / "records")
        self.refresh_lock()
        with self.assertRaisesRegex(records.Rejected, "production pin differs"):
            records.check_table(entries, self.root)
        self.assertEqual(artifact["sha256"], self.entry["artifact"]["sha256"])

    def test_stale_embedded_ptx_and_missing_identity_cannot_accept(self):
        for value in (None, "b" * 64):
            raw = copy.deepcopy(self.record)
            raw["tiers"]["sm75"]["loaded_ptx"]["modules"][0]["embedded_ptx_sha256"] = (
                value
            )
            with self.subTest(value=value), self.assertRaises(records.Rejected):
                self.check([{**self.entry, "record": self.store(raw)}])
        raw = copy.deepcopy(self.record)
        raw["schema"] = 4
        with self.assertRaisesRegex(records.Rejected, "schema-5"):
            self.check([{**self.entry, "record": self.store(raw)}])

    def test_precise_device_and_driver_fields_fail_closed(self):
        for field, value in (
            ("name", ""),
            ("sm_count", True),
            ("sm_count", 0),
            ("l2_bytes", 0),
            ("l2_bytes", -1),
            ("driver_version", None),
            ("driver_api_version", 0),
        ):
            raw = copy.deepcopy(self.record)
            raw["tiers"]["sm75"]["device"][field] = value
            with (
                self.subTest(field=field, value=value),
                self.assertRaises(records.Rejected),
            ):
                self.check([{**self.entry, "record": self.store(raw)}])

    def test_ptx_on_disk_without_a_feature_mask_is_not_loadable(self):
        path = self.root / "src/inference/cuda/ptx/lstm.sm80.ptx"
        path.write_text("not embedded")
        with self.assertRaisesRegex(records.Rejected, "embed masks"):
            records.production_load("lstm", "sm80", "12.0", self.root)
        self.write_embed("sm80", ["cuda-sm75"])
        with self.assertRaisesRegex(records.Rejected, "cannot realize"):
            records.production_load("lstm", "sm80", "12.0", self.root)


class Summaries(RecordsFixture):
    def prepare(self, entries=None):
        entries = entries or [self.entry]
        result = records.write_summaries(
            entries, self.root, records=self.cache / "records"
        )
        self.refresh_lock()
        return result

    def test_offline_does_not_read_raw_cache_and_full_verification_matches(self):
        self.prepare()
        with patch.object(
            records,
            "load",
            side_effect=AssertionError("offline must not read raw records"),
        ):
            result = records.check_table([self.entry], self.root)
            self.assertEqual(result["verification"], "locked-summary")
        result = records.check_table([self.entry], self.root, self.cache / "records")
        self.assertEqual(result["verification"], "raw-records")

    def test_summary_tamper_fails_lock_and_refreshed_lock_fails_raw_derivation(self):
        self.prepare()
        path = self.root / records.ACCEPTANCE
        summary = json.loads(path.read_bytes())
        summary["records"][self.entry["record"]]["device_name"] = "Changed name"
        path.write_bytes(records.canonical_summaries(summary))
        with self.assertRaisesRegex(records.Rejected, "locked hash"):
            records.check_table([self.entry], self.root)
        self.refresh_lock()
        with self.assertRaisesRegex(
            records.Rejected, "precise device|raw-record derivation"
        ):
            records.check_table([self.entry], self.root, self.cache / "records")

    def test_missing_schema_and_canonical_summary_are_required(self):
        for mutation in ("schema", "missing", "extra", "canonical"):
            summary = self.prepare()
            path = self.root / records.ACCEPTANCE
            if mutation == "schema":
                summary["schema"] = 2
            elif mutation == "missing":
                summary["records"].clear()
            elif mutation == "extra":
                summary["records"]["0" * 64] = copy.deepcopy(
                    next(iter(summary["records"].values()))
                )
            if mutation == "canonical":
                path.write_bytes(json.dumps(summary).encode())
            else:
                path.write_bytes(records.canonical_summaries(summary))
            self.refresh_lock()
            with self.subTest(mutation=mutation), self.assertRaises(records.Rejected):
                records.check_table([self.entry], self.root)

    def test_invalid_summary_digests_and_missing_typed_fields_fail(self):
        for field, value in (
            ("coverage_sha256", "bad"),
            ("device_name", ""),
            ("raw_tier_reason", None),
            ("legacy", "false"),
        ):
            summary = self.prepare()
            summary["records"][self.entry["record"]][field] = value
            (self.root / records.ACCEPTANCE).write_bytes(
                records.canonical_summaries(summary)
            )
            self.refresh_lock()
            with self.subTest(field=field), self.assertRaises(records.Rejected):
                records.check_table([self.entry], self.root)
        summary = self.prepare()
        summary["records"][self.entry["record"]].pop("raw_tier_reason")
        (self.root / records.ACCEPTANCE).write_bytes(
            records.canonical_summaries(summary)
        )
        self.refresh_lock()
        with self.assertRaisesRegex(
            records.Rejected, "invalid acceptance record binding"
        ):
            records.check_table([self.entry], self.root)

    def test_offline_ptx_and_export_drift_fail(self):
        self.prepare()
        path = self.root / "src/inference/cuda/ptx/lstm.sm75.ptx"
        original = path.read_bytes()
        path.write_bytes(original + b" changed")
        with self.assertRaisesRegex(records.Rejected, "qualification files differ"):
            records.check_table([self.entry], self.root)
        path.write_bytes(original)
        entry = copy.deepcopy(self.entry)
        entry["candidate_coverage"]["entries"][0]["maths"] = ["tf32"]
        with self.assertRaisesRegex(records.Rejected, "candidate coverage differs"):
            records.check_table([entry], self.root)

    def test_offline_tier_device_area_der_and_subset_are_bound(self):
        self.prepare()
        for field, value in (
            ("tier", "sm80"),
            ("devices", ["8.0"]),
            ("area", "resnet"),
            ("der", "0" * 64),
        ):
            with self.subTest(field=field), self.assertRaises(records.Rejected):
                records.check_table([{**self.entry, field: value}], self.root)
        entry = copy.deepcopy(self.entry)
        entry["coverage"]["entries"][0]["batches"] = [32]
        with self.assertRaisesRegex(records.Rejected, "outside accepted"):
            records.check_table([entry], self.root)

    def test_complete_accepted_set_is_independent_of_table_subset(self):
        raw = copy.deepcopy(self.record)
        child = raw["tiers"]["sm75"]
        child["coverage_declared"]["triples"] = [
            ["lstm.stack", batch, "fp32"] for batch in records.TEST_BATCHES
        ]
        child["accepted_tuples"] = [["lstm.stack", batch, "fp32"] for batch in (1, 32)]
        pin = self.store(complete_fixture(raw))
        entry = copy.deepcopy(self.entry)
        entry["record"] = pin
        entry["candidate_coverage"]["entries"][0]["batches"] = "all"
        summary = self.prepare([entry])
        self.assertEqual(len(summary["records"][pin]["accepted_tuples"]), 2)
        changed = copy.deepcopy(entry)
        changed["coverage"]["entries"][0]["batches"] = [32]
        changed["configurations"][0]["tuple"][1] = 32
        with self.assertRaisesRegex(records.Rejected, "differ from whole plan"):
            records.check_table([changed], self.root, self.cache / "records")
        self.assertEqual(
            summary["records"][pin]["accepted_tuples"],
            [["lstm.stack", 1, "fp32"], ["lstm.stack", 32, "fp32"]],
        )

    def test_offline_gate_rejects_unrealizable_device_even_with_refreshed_lock(self):
        summary = self.prepare()
        summary["records"][self.entry["record"]]["device_capability"] = "7.0"
        (self.root / records.ACCEPTANCE).write_bytes(
            records.canonical_summaries(summary)
        )
        self.refresh_lock()
        entry = {**self.entry, "devices": ["7.0"]}
        with self.assertRaisesRegex(records.Rejected, "cannot realize"):
            records.check_table([entry], self.root)

    def test_explicit_records_directory_must_remain_outside_tree(self):
        self.prepare()
        with self.assertRaisesRegex(records.Rejected, "outside the source tree"):
            records.check_table([self.entry], self.root, self.root / "records")

    def test_explicit_record_directory_does_not_change_environment(self):
        before = os.environ.get("SPEAKRS_QUALIFY_CACHE")
        self.prepare()
        records.check_table([self.entry], self.root, self.cache / "records")
        self.assertEqual(os.environ.get("SPEAKRS_QUALIFY_CACHE"), before)


class WholePlanDer(RecordsFixture):
    def receipt(self):
        return records.load(self.entry["der"], self.root)

    def test_arbitrary_nonempty_json_and_each_missing_identity_are_rejected(self):
        with self.assertRaisesRegex(records.Rejected, "invalid evidence fields"):
            self.check([{**self.entry, "der": self.store({"anything": "nonempty"})}])
        raw = self.receipt()
        for field in raw["plans"][0]:
            bad = copy.deepcopy(raw)
            del bad["plans"][0][field]
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(records.Rejected, "invalid plan fields"),
            ):
                self.check([{**self.entry, "der": self.store(bad)}])

    def test_scope_math_models_input_library_and_plan_routes_are_bound(self):
        for fault in (
            "scope",
            "math",
            "models",
            "inputs",
            "pipeline",
            "library",
            "routes",
            "baseline",
            "verdict",
            "nonfinite",
        ):
            raw = self.receipt()
            plan = raw["plans"][0]
            if fault == "scope":
                plan["device_scope"]["sm_count"] = 70
            elif fault == "math":
                plan["math"] = "tf32"
            elif fault == "models":
                plan["models"]["embedding"] = "0" * 64
            elif fault == "inputs":
                plan["inputs"]["manifest_sha256"] = "0" * 64
            elif fault == "pipeline":
                plan["pipeline_config_sha256"] = "0" * 64
            elif fault == "library":
                plan["library_artifacts"]["fbank"]["artifact"]["sha256"] = "0" * 64
            elif fault == "routes":
                plan["routes"].pop()
            elif fault == "baseline":
                plan["baseline"]["routes"][0]["route"] = {"kind": "Candidate"}
            elif fault == "verdict":
                plan["candidate"]["output_sha256"] = "0" * 64
            else:
                plan["candidate"]["der"] = float("nan")
            with self.subTest(fault=fault), self.assertRaises(records.Rejected):
                self.check([{**self.entry, "der": self.store(raw)}])

    def test_forged_input_pins_fail_even_with_recomputed_receipt_identities(self):
        for field in (
            "manifest_sha256",
            "reference_sha256",
            "pipeline_config_sha256",
            "control_archive_sha256",
        ):
            raw = self.receipt()
            for plan in raw["plans"]:
                if field in ("manifest_sha256", "reference_sha256"):
                    plan["inputs"][field] = "0" * 64
                elif field == "pipeline_config_sha256":
                    plan[field] = "0" * 64
                else:
                    plan["baseline"][field] = "0" * 64
                identity = records.der_evidence.digest(
                    {
                        name: plan[name]
                        for name in (
                            "device_scope",
                            "math",
                            "models",
                            "inputs",
                            "pipeline_config_sha256",
                            "library_artifacts",
                        )
                    }
                )
                baseline = plan["baseline"]
                baseline["metrics"]["identity_sha256"] = identity
                baseline["metrics"]["execution_sha256"] = (
                    records.der_evidence.execution_identity(
                        identity, baseline["routes"], baseline["control_archive_sha256"]
                    )
                )
                plan["candidate"]["identity_sha256"] = identity
                plan["candidate"]["execution_sha256"] = (
                    records.der_evidence.execution_identity(identity, plan["routes"])
                )
                plan["verdict"]["baseline_sha256"] = records.der_evidence.digest(
                    baseline
                )
                plan["verdict"]["candidate_sha256"] = records.der_evidence.digest(
                    plan["candidate"]
                )
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(records.Rejected, "live locked inventory"),
            ):
                self.check([{**self.entry, "der": self.store(raw)}])

    def test_missing_live_der_source_cannot_accept_a_modern_receipt(self):
        for inventory in (
            None,
            {
                "kind": "Missing",
                "sources": ["DER audio content manifest with file count"],
            },
        ):
            entry = {**self.entry, "der_inventory": inventory}
            with (
                self.subTest(inventory=inventory),
                self.assertRaisesRegex(records.Rejected, "live locked"),
            ):
                self.check([entry])

    def test_route_change_cannot_reuse_scorer_metrics_or_verdict_in_either_mode(self):
        child = copy.deepcopy(self.child)
        both = [["lstm.stack", batch, "fp32"] for batch in (1, 32)]
        child["coverage_declared"]["triples"] = both
        child["accepted_tuples"] = both
        raw_record = complete_fixture({**self.record, "tiers": {"sm75": child}})
        original = copy.deepcopy(self.entry)
        original["record"] = self.store(raw_record)
        original["candidate_coverage"]["entries"][0]["batches"] = [1, 32]
        summary = records.write_summaries(
            [original], self.root, records=self.cache / "records"
        )
        raw_der = records.load(original["der"], self.root)
        old = copy.deepcopy(raw_der["plans"][0])
        changed = copy.deepcopy(original)
        changed["coverage"]["entries"][0]["batches"] = [32]
        changed["configurations"][0]["tuple"][1] = 32
        domain = records.der_evidence.PlanDomain.parse(changed["boundary_domain"])
        raw_der["plans"][0]["routes"] = records.der_evidence.routes(
            [changed], domain, "fp32"
        )
        changed["der"] = self.store(raw_der)
        for field in ("baseline", "candidate", "verdict"):
            self.assertEqual(old[field], raw_der["plans"][0][field])
        with self.assertRaisesRegex(records.Rejected, "scorer evidence"):
            self.check([changed])
        locked = summary["records"][original["record"]]
        locked["der"] = changed["der"]
        locked["der_evidence"] = {
            "kind": "WholePlan",
            "source_sha256": changed["der"],
            "evidence": raw_der,
        }
        (self.root / records.ACCEPTANCE).write_bytes(
            records.canonical_summaries(summary)
        )
        self.refresh_lock()
        for directory in (None, self.cache / "records"):
            with (
                self.subTest(raw=directory is not None),
                self.assertRaisesRegex(records.Rejected, "scorer evidence"),
            ):
                records.check_table([changed], self.root, directory)

    def test_an_area_receipt_cannot_authorize_a_combined_plan(self):
        original = self.receipt()
        first = copy.deepcopy(self.entry)
        other = copy.deepcopy(first)
        domain = [
            *first["boundary_domain"],
            {"boundary": "sincnet.conv0.abs_pool", "batches": [1, 32]},
        ]
        first["boundary_domain"] = other["boundary_domain"] = domain
        other.update(
            area="sincnet",
            configurations=[
                {
                    "tuple": ["sincnet.conv0.abs_pool", 1, "fp32"],
                    "pin": {"kind": "Sinc", "selection": "ConvAbsPool"},
                }
            ],
        )
        with self.assertRaisesRegex(records.Rejected, "differ from whole plan"):
            records.der_evidence.check(original, first["der"], [first, other])
        pin = self.der_receipt([first, other])
        raw = records.load(pin, self.root)
        accepted = records.der_evidence.check(raw, pin, [first, other])
        self.assertEqual(accepted["kind"], "WholePlan")
        self.assertEqual(
            sum(
                row["route"]["kind"] == "Candidate" for row in raw["plans"][0]["routes"]
            ),
            2,
        )
        other["der"] = "0" * 64
        with self.assertRaisesRegex(records.Rejected, "one whole execution plan"):
            records.der_evidence.groups([first, other])

    def test_locked_receipts_are_rebound_even_when_the_lock_is_refreshed(self):
        summary = records.write_summaries(
            [self.entry], self.root, records=self.cache / "records"
        )
        receipt = summary["records"][self.entry["record"]]["der_evidence"]
        receipt["evidence"]["plans"][0]["routes"].pop()
        (self.root / records.ACCEPTANCE).write_bytes(
            records.canonical_summaries(summary)
        )
        self.refresh_lock()
        with self.assertRaisesRegex(records.Rejected, "differ from whole plan"):
            records.check_table([self.entry], self.root)

    def test_legacy_der_mapping_cannot_authorize_a_new_record(self):
        with self.assertRaisesRegex(
            records.Rejected, "cannot authorize a new execution plan"
        ):
            records.der_evidence.check(
                {}, records.der_evidence.LEGACY_HASH, [self.entry]
            )


if __name__ == "__main__":
    unittest.main()
