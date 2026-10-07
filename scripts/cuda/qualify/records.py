"""Validate evaluated production entries against immutable, outside-tree records."""

import gzip
import hashlib
import json
import re
from pathlib import Path

import artifacts
import configurations
import der as der_evidence
import environment
from domains import MODEL, boundary, collection
from assets import cache_directory
from gates import Rejected
from lock import ROOT
from verdict import NOISE_REASON, evaluate_record, noise_timing

# legacy PR #36 results forced sm75 on RTX 5070 Ti (12.0), not on Turing
# sources: qualification/k1/REPORT-K1-requalify.md, k2/REPORT-final-lock.md,
# and k3/REPORT-K3-requalify.md in the archived native-CUDA records
LEGACY = {
    "8f8fa3e3c158771e354aad83f4e42fca6fac998aa192b39a966067a4b0035758": "resnet",
    "3badc1aec939b0e8f7312786d695bec6445de1dacb1f85e44124bf3ac20356f8": "lstm",
    "a4d1a2692b78a814cd2da16f084801c3641bfd11c6d095d610f3a8182ad6f675": "sincnet",
}


SOURCE_EVIDENCE_GAP = (
    "legacy record has no host-source, kernel-source, or PTX build-manifest hashes; "
    "area bindings were captured at PR36 acceptance; shared candidate.rs uses the "
    "explicit final structural baseline for PlanError, per-tier coverage, and batches; "
    "that structural change was not measured by the archived GPU qualification"
)
SHARED_SOURCE_AMENDMENT = {
    "path": "src/inference/cuda/candidate.rs",
    "acceptance_sha256": "132ac95289a11c642740370987f9090e6720fadf406c0b82e823f7b73f639427",
    "structural_baseline_sha256": "7ebccfee61ad72ad617addc5d7d32a78dc4895afbc4582383bdb2b1cc07fd2ee",
}
# device identity is recorded in the archived PR36 qualification reports
LEGACY_DEVICE_NAME = "NVIDIA GeForce RTX 5070 Ti"
AREA_HOST = {
    "resnet": "conv",
    "lstm": "lstm",
    "sincnet": "sinc",
    "fbankdft": "fbankdft",
}
# acceptance-time hashes migrated from QUALIFIED.json; this is not a writable manifest
LEGACY_BINDINGS: dict[str, dict] = {
    "3badc1aec939b0e8f7312786d695bec6445de1dacb1f85e44124bf3ac20356f8": {
        "files": {
            "host_sources": {
                "src/inference/cuda/candidate.rs": "7ebccfee61ad72ad617addc5d7d32a78dc4895afbc4582383bdb2b1cc07fd2ee",
                "src/inference/cuda/candidate/lstm.rs": "5306b541515451b80eeacc43bb922d68dec4601dcc9582292f4074ee91e0439a",
                "src/inference/cuda/candidate/lstm/layout.rs": "478adba61b4c5f2b21d22147b7f0e73768164831559f711a4d8653f04de77e8c",
            },
            "kernel_sources": {
                "crates/speakrs-cuda-kernels/src/lstm.rs": "f37dc36973e1a751926bd36c110d2ac4849a3b94b43950ace81a19b6d7b0724b"
            },
            "manifests": {
                "src/inference/cuda/ptx/lstm.manifest": "9346034eef96b3e54d679fcb2a721bf72437016c89505064276dcfb2af22f43b"
            },
            "ptx": {
                "src/inference/cuda/ptx/lstm.sm75.ptx": "72945743a3c1b915c05d8ea21b438dfd860fa9487401c447fb24fd48802916fa"
            },
        },
        "lock_digest": "4bb7818f209cad9a3d841e3a71e0d9abee46f419479aeb9bf07ceaec277dfdf9",
    },
    "8f8fa3e3c158771e354aad83f4e42fca6fac998aa192b39a966067a4b0035758": {
        "files": {
            "host_sources": {
                "src/inference/cuda/candidate.rs": "7ebccfee61ad72ad617addc5d7d32a78dc4895afbc4582383bdb2b1cc07fd2ee",
                "src/inference/cuda/candidate/conv.rs": "b4755e8bdf919edd39d2da5e632da7af5a52fd48f3852bd17968cc29be6310a8",
            },
            "kernel_sources": {
                "crates/speakrs-cuda-kernels/src/resnet.rs": "9f2b5fd4c5c88236aad9829c8ef48c1c7b1b55169a8d82228edbd622c8184122"
            },
            "manifests": {
                "src/inference/cuda/ptx/resnet.manifest": "0d9a2c625742b38f5bdb1fccb4a5465b45efb9c83830b1d1e5b51077ae574d1e"
            },
            "ptx": {
                "src/inference/cuda/ptx/resnet.sm75.ptx": "dd6449c0129f9a03bf691c0338611b50b651ab714d5803caedea87de3c72b6b7"
            },
        },
        "lock_digest": "368389d213c1c9c3ae32b0c9375d558efbd1d88f71eec063e67b8f14163bcb70",
    },
    "a4d1a2692b78a814cd2da16f084801c3641bfd11c6d095d610f3a8182ad6f675": {
        "files": {
            "host_sources": {
                "src/inference/cuda/candidate.rs": "7ebccfee61ad72ad617addc5d7d32a78dc4895afbc4582383bdb2b1cc07fd2ee",
                "src/inference/cuda/candidate/sinc.rs": "f0a95196f635a7fc25c45722f2df8f8c248d9d5e67ac2804b7cb45c0b4d91ded",
            },
            "kernel_sources": {
                "crates/speakrs-cuda-kernels/src/sincnet.rs": "600f1c3688e073e13b65a36b0fbe4917dde498218efba8c61fca912e6f9d6920"
            },
            "manifests": {
                "src/inference/cuda/ptx/sincnet.manifest": "9b2eb3c3db1d60ddadf364c1eea8ff79ec23e5d368cb13ea4037253412bf6f39"
            },
            "ptx": {
                "src/inference/cuda/ptx/sincnet.sm75.ptx": "967bc6893f80da84d8d4d288f2cf1ca3336ca09c4386495cab722beb0ac87247"
            },
        },
        "lock_digest": "368389d213c1c9c3ae32b0c9375d558efbd1d88f71eec063e67b8f14163bcb70",
    },
}


# these exact infrastructure substitutions do not claim a new GPU qualification
INFRASTRUCTURE_AMENDMENTS = [
    {
        "path": "src/inference/cuda/candidate.rs",
        "acceptance_sha256": "7ebccfee61ad72ad617addc5d7d32a78dc4895afbc4582383bdb2b1cc07fd2ee",
        "infrastructure_sha256": "f3eebb5a137ae963d0d8258b13f99d4aa01786467eb9e4b73af6a3e5e11a5343",
        "reason": "Kernel inventory exposes the existing host choices to the locked entry test; no algorithm changes",
    },
    {
        "path": "src/inference/cuda/candidate/lstm.rs",
        "acceptance_sha256": "5306b541515451b80eeacc43bb922d68dec4601dcc9582292f4074ee91e0439a",
        "infrastructure_sha256": "cb62ccec56e3549db8b25ecd5c33c8a63adfbfefe5153329fc69d401f1c6b731",
        "reason": "Kernel inventory exposes the existing host choices to the locked entry test; no algorithm changes",
    },
    {
        "path": "src/inference/cuda/candidate/conv.rs",
        "acceptance_sha256": "b4755e8bdf919edd39d2da5e632da7af5a52fd48f3852bd17968cc29be6310a8",
        "infrastructure_sha256": "0e9bc602a618dd1202df9e7cfc566461f6537c474bb2759e099c0c46165e8506",
        "reason": "Kernel inventory exposes the existing host choices to the locked entry test; no algorithm changes",
    },
    {
        "path": "src/inference/cuda/candidate/sinc.rs",
        "acceptance_sha256": "f0a95196f635a7fc25c45722f2df8f8c248d9d5e67ac2804b7cb45c0b4d91ded",
        "infrastructure_sha256": "96dac61ad53f015646b391a18febe024e5d2706c25075a9767caa8b5cd976d7e",
        "reason": "Kernel inventory exposes the existing host choices to the locked entry test; no algorithm changes",
    },
    {
        "path": "src/inference/cuda/ptx/lstm.manifest",
        "acceptance_sha256": "9346034eef96b3e54d679fcb2a721bf72437016c89505064276dcfb2af22f43b",
        "infrastructure_sha256": "09e6f3772953d81c57284baf88db917aea0a637e720b9a965e078ed0d634c3a4",
        "reason": "Exact-architecture cubin build pins extend the PTX manifest; PTX bytes are unchanged",
    },
    {
        "path": "src/inference/cuda/ptx/resnet.manifest",
        "acceptance_sha256": "0d9a2c625742b38f5bdb1fccb4a5465b45efb9c83830b1d1e5b51077ae574d1e",
        "infrastructure_sha256": "734d43a71f6958fce6cdac766723e4ffba6b32c584e2ebf5accfbdfafd019031",
        "reason": "Exact-architecture cubin build pins extend the PTX manifest; PTX bytes are unchanged",
    },
    {
        "path": "src/inference/cuda/ptx/sincnet.manifest",
        "acceptance_sha256": "9b2eb3c3db1d60ddadf364c1eea8ff79ec23e5d368cb13ea4037253412bf6f39",
        "infrastructure_sha256": "09a4a44ef930dee6fee351fe35fe6451cc316bf08da0bdf843bbb9f785531496",
        "reason": "Exact-architecture cubin build pins extend the PTX manifest; PTX bytes are unchanged",
    },
    {
        "path": "src/inference/cuda/candidate.rs",
        "acceptance_sha256": "f3eebb5a137ae963d0d8258b13f99d4aa01786467eb9e4b73af6a3e5e11a5343",
        "infrastructure_sha256": "abb8ecaef15c5771a4c058203e48d47b80845b55f6bd17672f348faf4e59e110",
        "reason": "Typed selection domain: config pins, special-value contracts and typed plan refusals; no algorithm changes",
    },
    {
        "path": "src/inference/cuda/candidate/conv.rs",
        "acceptance_sha256": "0e9bc602a618dd1202df9e7cfc566461f6537c474bb2759e099c0c46165e8506",
        "infrastructure_sha256": "fec622a1dcfa08cec6a313975a288eef70fc1c5d1e13e4736815dff8eca2253b",
        "reason": "Typed selection domain: plans build from the legacy wave-rule pin with the same tiling choice; no algorithm changes",
    },
    {
        "path": "src/inference/cuda/candidate/lstm.rs",
        "acceptance_sha256": "cb62ccec56e3549db8b25ecd5c33c8a63adfbfefe5153329fc69d401f1c6b731",
        "infrastructure_sha256": "8a52b696bc9a46059509cdeddd669e92921f3d960df2098860bd85ba0810d6f2",
        "reason": "Typed selection domain: plans build from the legacy cooperative pin; no algorithm changes",
    },
    {
        "path": "src/inference/cuda/candidate/sinc.rs",
        "acceptance_sha256": "96dac61ad53f015646b391a18febe024e5d2706c25075a9767caa8b5cd976d7e",
        "infrastructure_sha256": "bea5240a8503e375116533eac1237659182c109e78c34baf5e1c6be7b1652aa6",
        "reason": "Typed selection domain: plans build from the fixed producer pin; no algorithm changes",
    },
    {
        "path": "src/inference/cuda/candidate.rs",
        "acceptance_sha256": "abb8ecaef15c5771a4c058203e48d47b80845b55f6bd17672f348faf4e59e110",
        "infrastructure_sha256": "00206550965eee79ec7a5669d1f36b512056a8f9c5e0ddb17f1d7ab8cb61e39c",
        "reason": "Filterbank producer interface and compiled host candidate tests; existing candidate algorithms are unchanged",
    },
]


def apply_amendments(binding: dict) -> tuple[dict, list[dict]]:
    """Apply explicit infrastructure pins in order, so a later pin may amend an earlier one"""
    result = {group: dict(hashes) for group, hashes in binding["files"].items()}
    applied = []
    for amendment in INFRASTRUCTURE_AMENDMENTS:
        for hashes in result.values():
            if hashes.get(amendment["path"]) == amendment["acceptance_sha256"]:
                hashes[amendment["path"]] = amendment["infrastructure_sha256"]
                if amendment not in applied:
                    applied.append(amendment)
    return result, applied


def amended_legacy_files(binding: dict) -> dict:
    """Preserve original acceptance hashes and apply only explicit infrastructure pins"""
    return apply_amendments(binding)[0]


def legacy_amendments(binding: dict) -> list[dict]:
    """Expose every applied old/new pin without changing archived record provenance"""
    return apply_amendments(binding)[1]


def digest(value: str) -> str:
    """Require a canonical content hash, never a cache-relative path."""
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise Rejected("record: missing or invalid SHA-256")
    return value


def record_path(value: str, root: Path = ROOT, *, records: Path | None = None) -> Path:
    """Map a SHA-256 to the record namespace in the outside-tree cache."""
    directory = records if records is not None else cache_directory(root) / "records"
    resolved = directory.resolve()
    tree = root.resolve()
    if resolved == tree or tree in resolved.parents:
        raise Rejected("record: raw records directory must be outside the source tree")
    return directory / digest(value)


def load(value: str, root: Path = ROOT, *, records: Path | None = None) -> dict:
    """Verify raw bytes first; accept JSON or gzip-compressed JSON."""
    path = record_path(value, root, records=records)
    if path.is_symlink() or not path.is_file():
        raise Rejected(
            f"record: unresolved hash {value}; provide an outside-tree raw records directory"
        )
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != value:
        raise Rejected(f"record: cache hash mismatch {value}")
    if data.startswith(b"\x1f\x8b"):
        data = gzip.decompress(data)
    result = json.loads(data)
    if not isinstance(result, dict):
        raise Rejected("record: expected an object")
    return result


def triples(raw: dict) -> set[tuple[str, int, str]]:
    """Expand an evaluated model entry without consulting candidate coverage."""
    found = set()
    for entry in raw["entries"]:
        modes = ["fp32", "tf32"] if entry["maths"] == "all" else entry["maths"]
        for layer in entry["layers"]:
            batches = (
                boundary(layer).production
                if entry["batches"] == "all"
                else entry["batches"]
            )
            for batch in batches:
                for mode in modes:
                    if (
                        not isinstance(layer, str)
                        or type(batch) is not int
                        or batch not in boundary(layer).production
                        or mode not in ("fp32", "tf32")
                    ):
                        raise Rejected(
                            "table: invalid production tuple or stress batch"
                        )
                    found.add((layer, batch, mode))
    if not found:
        raise Rejected("table: empty production entry")
    return found


TEST_BATCHES = MODEL.tested
LEGACY_PHASES = ["numeric", "timing", "profile", "sanitize"]
CURRENT_PHASES = ["numeric", "timing", "paired", "profile", "sanitize"]


def required_checks(target: str) -> set[str]:
    """Identify collection markers that establish complete Oxide qualification"""
    names = {
        "ptx:loaded_bytes",
        "ptx:loaded_bytes/stable",
        "ptx:shared_initialization",
        "determinism:fixed_reduction_order",
        "profile",
        "profile:graph_nodes",
        "profile:captured_library_calls",
        *(f"sanitizer:control/{fault}" for fault in ("oob", "race", "uninit")),
        *(f"sanitizer:Oxide/{tool}" for tool in ("memcheck", "racecheck", "initcheck")),
    }
    if target == "lstm":
        names.update(
            f"sanitizer:ProjectionBaseline/{tool}"
            for tool in ("memcheck", "racecheck", "initcheck")
        )
    return names


TEST_CASES = MODEL.cases


def tuple_requirements(record: dict, child: dict) -> tuple[set[str], set[str]]:
    """Derive required numeric, timing, and paired markers from tested coverage"""
    declared = child.get("coverage_declared")
    if not isinstance(declared, dict) or "triples" not in declared:
        raise Rejected("table: missing recorded candidate coverage")
    triples = tested_declaration(declared)
    names, timing = set(), set()
    for layer, batch, mode in triples:
        names.add(f"layer:{mode}/{layer}")
        names.add(f"secret:{mode}/secret/b{batch}/{layer}")
        for case, sample in collection(record["target"]).cases:
            if sample != batch:
                continue
            key = f"{mode}/{case}/b{batch}/{layer}"
            names.update(
                (f"determinism:{key}", f"determinism:{key}/switched", f"speed:{key}")
            )
            timing.add(key)
    tf32 = False
    for case, batch in collection(record["target"]).cases:
        for mode in ("fp32", "tf32"):
            if not any(sample == batch and math == mode for _, sample, math in triples):
                continue
            key = f"{mode}/{case}/b{batch}/stage"
            names.update((f"determinism:{key}", f"determinism:{key}/switched"))
            timing.add(key)
            if mode == "fp32" or record["schema"] == 3:
                names.update((f"stage:{key}", f"stage:{key}/switched"))
            else:
                names.update((f"stage_truth:{key}", f"stage_truth:{key}/switched"))
                if record["target"] != "resnet":
                    names.update(
                        (f"stage:{key}/argmax", f"stage:{key}/switched/argmax")
                    )
            tf32 |= mode == "tf32"
            if record["schema"] == 3:
                names.add(f"speed:{key}")
            else:
                names.update((f"paired_output:{key}", f"paired_stage:{key}"))
    if tf32:
        if record["schema"] == 3:
            names.add("stage:tf32/aggregate")
        elif record["target"] != "resnet":
            names.add("stage:tf32/aggregate_argmax")
    return names, timing


def valid_checks(checks: object) -> list[dict]:
    """Require complete boolean outcomes and unique, named checks"""
    if not isinstance(checks, list) or not checks:
        raise Rejected("table: missing qualification checks")
    names: list[str] = []
    validated: list[dict] = []
    for raw_check in checks:
        if not isinstance(raw_check, dict):
            raise Rejected("table: invalid qualification check")
        check = dict(raw_check)
        if (
            type(check.get("passed")) is not bool
            or not isinstance(check.get("check"), str)
            or not check["check"]
        ):
            raise Rejected("table: invalid qualification check")
        name = check["check"]
        if not isinstance(name, str):
            raise Rejected("table: invalid qualification check")
        names.append(name)
        validated.append(check)
    if len(set(names)) != len(names):
        raise Rejected("table: duplicate qualification checks")
    return validated


def complete_collection(record: dict) -> None:
    """Reject interrupted or inconsistent collection before evaluating acceptance"""
    if (
        type(record.get("schema")) is not int
        or record["schema"] not in (3, 4, 5)
        or record.get("implementation") != "Oxide"
    ):
        raise Rejected(
            "table: not a supported schema for an Oxide qualification record"
        )
    if record.get("status") not in ("passed", "blocked"):
        raise Rejected("table: qualification record has an invalid status")
    checks = valid_checks(record.get("checks"))
    parent = {check["check"]: check for check in checks}
    tiers = record.get("tiers")
    if not isinstance(tiers, dict) or not tiers:
        raise Rejected("table: missing qualification tiers")
    prefixed = set()
    for tier, child in tiers.items():
        if (
            not isinstance(child, dict)
            or child.get("target") != record.get("target")
            or child.get("implementation") != "Oxide"
        ):
            raise Rejected(f"table: wrong tier target or implementation: {tier}")
        phases = LEGACY_PHASES if record["schema"] == 3 else CURRENT_PHASES
        coverage = child.get("coverage")
        if (
            child.get("phases_run") != phases
            or not isinstance(coverage, dict)
            or coverage.get("tier") != tier
            or coverage.get("cases")
            != [list(case) for case in collection(record["target"]).cases]
            or coverage.get("math") != ["fp32", "tf32"]
        ):
            raise Rejected(f"table: incomplete tier collection: {tier}")
        tier_checks = valid_checks(child.get("checks"))
        names = {check["check"] for check in tier_checks}
        tuple_checks, required_timing = tuple_requirements(record, child)
        missing = (required_checks(record["target"]) | tuple_checks) - names
        if missing:
            raise Rejected(
                f"table: missing required tier checks: {tier}: {sorted(missing)}"
            )
        timing = child.get("timing")
        if not isinstance(timing, list) or not all(
            isinstance(row, dict) and isinstance(row.get("id"), str) for row in timing
        ):
            raise Rejected(f"table: missing or invalid timing rows: {tier}")
        timing_ids = [row["id"] for row in timing]
        if len(set(timing_ids)) != len(timing_ids) or not required_timing <= set(
            timing_ids
        ):
            raise Rejected(f"table: missing or duplicate required timing rows: {tier}")
        failed = [check for check in tier_checks if not check["passed"]]
        if any(check.get("blocked") is not True for check in failed):
            raise Rejected(f"table: hard tier failure: {tier}")
        if child.get("status") not in ("passed", "blocked"):
            raise Rejected(f"table: invalid tier verdict: {tier}")
        if record["schema"] == 3:
            status = "blocked" if failed else "passed"
            reason = (
                f"{len(failed)} checks cannot be decided: {[check['check'] for check in failed[:6]]}"
                if failed
                else "all required checks passed"
            )
            if child.get("status") != status or child.get("reason") != reason:
                raise Rejected(
                    f"table: incomplete or inconsistent tier verdict: {tier}"
                )
        for check in tier_checks:
            name = f"{tier}/{check['check']}"
            prefixed.add(name)
            if parent.get(name) != {**check, "check": name}:
                raise Rejected(
                    "table: tier checks missing or inconsistent in aggregate"
                )
    aggregate = {name for name in parent if name.partition("/")[0] in tiers}
    if aggregate != prefixed:
        raise Rejected("table: aggregate has checks absent from tiers")


def complete_verdict(record: dict, tier: str, evaluation: dict) -> None:
    """Match the modern final collection verdict to the shared evaluator"""
    if record["schema"] == 3:
        return
    child = record["tiers"][tier]
    if evaluation["hard_failures"]:
        raise Rejected("table: record has hard failures")
    if evaluation["accepted"]:
        status = "passed"
        reason = "all required checks passed under locked verdict evaluation"
    else:
        status = "blocked"
        reason = f"{len(evaluation['unresolved'])} checks cannot be decided"
    if child.get("status") != status or child.get("reason") != reason:
        raise Rejected(f"table: incomplete or inconsistent tier verdict: {tier}")


def coverage_digest(rows: set[tuple[str, int, str]]) -> str:
    """Hash a canonical full tested declaration independently of raw JSON layout"""
    return hashlib.sha256(
        json.dumps(sorted(rows), separators=(",", ":")).encode()
    ).hexdigest()


def tested_declaration(raw: dict) -> set[tuple[str, int, str]]:
    """Normalize a candidate declaration to every harness test batch"""
    if "triples" in raw:
        rows = raw["triples"]
    else:
        rows = []
        for entry in raw["entries"]:
            modes = ["fp32", "tf32"] if entry["maths"] == "all" else entry["maths"]
            rows.extend(
                (layer, batch, mode)
                for layer in entry["layers"]
                for batch in (
                    boundary(layer).tested
                    if entry["batches"] == "all"
                    else entry["batches"]
                )
                for mode in modes
            )
    found = set()
    for row in rows:
        if (
            len(row) != 3
            or not isinstance(row[0], str)
            or type(row[1]) is not int
            or row[1] <= 0
            or row[2] not in ("fp32", "tf32")
        ):
            raise Rejected("table: invalid candidate declaration")
        if row[1] in boundary(row[0]).tested:
            found.add(tuple(row))
    if not found:
        raise Rejected("table: empty candidate declaration")
    return found


def file_digest(path: Path) -> str:
    """Hash a shipped regular file, excluding symbolic links."""
    if path.is_symlink() or not path.is_file():
        raise Rejected(f"table: not a regular qualification file: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


TIER_CAPABILITIES = {"sm75": (7, 5), "sm80": (8, 0), "sm90": (9, 0), "sm120": (12, 0)}


def production_load(area: str, tier: str, capability: str, root: Path = ROOT) -> None:
    """Require a supported feature build that can load the pinned tier on the device"""
    if area not in AREA_HOST or tier not in TIER_CAPABILITIES:
        raise Rejected("table: invalid production area or tier")
    artifacts.production_load(area, tier, capability, root)


def shipped_files(
    root: Path, area: str, *, legacy: bool = False
) -> dict[str, dict[str, str]]:
    """Collect candidate identity; legacy records predate shipped cubins."""
    host = AREA_HOST.get(area)
    if host is None:
        raise Rejected(f"table: unknown candidate area: {area}")
    ptx = root / "src/inference/cuda/ptx"
    candidates = root / "src/inference/cuda/candidate"
    kernels = root / "crates/speakrs-cuda-kernels/src"
    groups = {
        "ptx": sorted(ptx.glob(f"{area}.*.ptx")),
        "manifests": sorted(ptx.glob(f"{area}*.manifest")),
        "host_sources": [
            root / "src/inference/cuda/candidate.rs",
            candidates / f"{host}.rs",
            *sorted((candidates / host).rglob("*.rs")),
        ],
        "kernel_sources": [
            kernels / f"{area}.rs",
            *sorted((kernels / area).rglob("*.rs")),
        ],
    }
    if not legacy:
        groups["cubins"] = sorted(ptx.glob(f"{area}.*.cubin"))
    if not groups["ptx"] or ptx / f"{area}.manifest" not in groups["manifests"]:
        raise Rejected(f"table: missing candidate PTX or manifest: {area}")
    return {
        group: {path.relative_to(root).as_posix(): file_digest(path) for path in paths}
        for group, paths in groups.items()
    }


def check_binding(
    record_hash: str, record: dict, child: dict, tier: str, root: Path
) -> tuple[dict, str | None]:
    """Bind accepted evidence to all shipped files without trusting live baselines."""
    area = record["target"]
    legacy = record_hash in LEGACY
    current = shipped_files(root, area, legacy=legacy)
    modules = [
        module
        for module in child.get("loaded_ptx", {}).get("modules", [])
        if module.get("area") == area
    ]
    expected_path = f"src/inference/cuda/ptx/{area}.{tier}.ptx"
    if len(modules) != 1 or modules[0].get("path") != expected_path:
        raise Rejected("table: missing, duplicate, or invalid candidate PTX binding")
    module = modules[0]
    if module.get("tier") != tier:
        raise Rejected("table: loaded PTX tier mismatch")
    loaded = {expected_path: digest(module.get("sha256"))}
    legacy = record_hash in LEGACY
    if legacy:
        binding = LEGACY_BINDINGS.get(record_hash)
        if not binding or binding.get("lock_digest") != digest(record["lock_digest"]):
            raise Rejected("table: invalid legacy acceptance binding")
        expected = amended_legacy_files(binding)
        gap = SOURCE_EVIDENCE_GAP
    else:
        code = child.get("code_sha256", {})
        if not isinstance(code, dict) or not code:
            raise Rejected("table: missing source code hashes")
        for name, value in code.items():
            if (
                not isinstance(name, str)
                or Path(name).is_absolute()
                or ".." in Path(name).parts
            ):
                raise Rejected("table: invalid source binding path")
            digest(value)
        if any(name not in code for name in current["ptx"]):
            raise Rejected("table: missing candidate PTX code hashes")
        ptx_prefix = f"src/inference/cuda/ptx/{area}."
        expected = {
            "ptx": {
                name: value
                for name, value in code.items()
                if name.startswith(ptx_prefix) and name.endswith(".ptx")
            }
        }
        if (
            expected_path in expected["ptx"]
            and expected["ptx"][expected_path] != loaded[expected_path]
        ):
            raise Rejected("table: conflicting recorded PTX bindings")
        expected["ptx"].update(loaded)
        for group in ("host_sources", "kernel_sources", "manifests", "cubins"):
            # recorded additions and removals must remain visible, including nested modules
            names = set(current[group])
            if group == "host_sources":
                prefix = f"src/inference/cuda/candidate/{AREA_HOST[area]}/"
            elif group == "kernel_sources":
                prefix = f"crates/speakrs-cuda-kernels/src/{area}/"
            else:
                prefix = f"src/inference/cuda/ptx/{area}"
            names.update(
                name
                for name in code
                if name.startswith(prefix)
                and (group != "manifests" or name.endswith(".manifest"))
                and (group != "cubins" or name.endswith(".cubin"))
            )
            if any(name not in code for name in names):
                raise Rejected(f"table: missing source or manifest hashes: {group}")
            expected[group] = {name: code[name] for name in names}
        gap = None
    for group, hashes in current.items():
        recorded = expected.get(group, {})
        if not isinstance(recorded, dict):
            raise Rejected("table: invalid acceptance binding")
        for name in sorted(hashes.keys() | recorded.keys()):
            if hashes.get(name) != recorded.get(name):
                raise Rejected(f"table: qualification file differs: {name}")
    if loaded[expected_path] != current["ptx"].get(expected_path):
        raise Rejected("table: loaded candidate PTX differs from shipped files")
    return current, gap


def speed_scope(entry: dict, *, legacy: bool, capability: str, device: dict) -> None:
    """Bind the exported speed scope to the record's own device evidence.

    Legacy approval stays capability-wide; modern evidence covers only the measured
    card. Neither can be relabelled as the other.
    """
    if legacy:
        expected = {"kind": "LegacyCapability", "capability": capability}
    else:
        expected = {
            "kind": "Point",
            "capability": capability,
            "sm_count": device.get("sm_count"),
            "device_name": device.get("name"),
        }
    if entry.get("speed_scope") != expected:
        raise Rejected("table: speed scope differs from record device evidence")


def unique_artifact_owners(entries: list[dict]) -> None:
    """Allow repeated evidence for one binding, never conflicting overlapping loads"""
    owners: list[tuple[str, str, dict, tuple]] = []
    for entry in entries:
        area = entry.get("area")
        if area not in ("resnet", "lstm", "sincnet", "fbankdft"):
            raise Rejected("table: missing production artifact area")
        scope = entry.get("speed_scope")
        if not isinstance(scope, dict) or scope.get("kind") not in (
            "Point",
            "LegacyCapability",
        ):
            raise Rejected("table: missing production speed scope")
        identity = (entry.get("tier"), entry.get("artifact"))
        for device in entry["devices"]:
            for prior_area, prior_device, prior_scope, prior_identity in owners:
                if prior_area != area or prior_device != device:
                    continue
                points = scope["kind"] == prior_scope["kind"] == "Point"
                if points and (scope.get("sm_count"), scope.get("device_name")) != (
                    prior_scope.get("sm_count"),
                    prior_scope.get("device_name"),
                ):
                    continue
                if prior_identity != identity:
                    raise Rejected("table: conflicting production artifact bindings")
            owners.append((area, device, scope, identity))


def check_table_records(
    entries: list[dict], root: Path = ROOT, *, records: Path | None = None
) -> dict:
    """Check provenance and current Rust-exported per-tier candidate coverage.

    The caller must export coverage from the same source tree. Tests supply a
    fixture export; source text parsing is not an authoritative coverage model
    """
    if not entries:
        raise Rejected("table: no production entries")
    unique_artifact_owners(entries)
    evidence = []
    for entry in configurations.record_entries(entries):
        record = load(entry["record"], root, records=records)
        complete_collection(record)
        tier = entry["tier"]
        if tier not in record["tiers"]:
            raise Rejected("table: tier not accepted by record")
        if entry.get("area", record["target"]) != record["target"]:
            raise Rejected("table: candidate area differs from record")
        evaluations = {}
        for recorded_tier in record["tiers"]:
            evaluations[recorded_tier] = evaluate_record(record, recorded_tier)
            complete_verdict(record, recorded_tier, evaluations[recorded_tier])
        evaluation = evaluations[tier]
        child = record["tiers"][tier]
        if evaluation["hard_failures"]:
            raise Rejected(
                f"table: record has hard failures: {evaluation['hard_failures']}"
            )
        legacy = entry["record"] in LEGACY
        if legacy:
            if (
                record["target"] != LEGACY[entry["record"]]
                or tier != "sm75"
                or child.get("device_sm") != "12.0"
            ):
                raise Rejected("table: legacy target/tier/device mismatch")
            capability = "12.0"
            code = {"legacy_driver_sha256": digest(child["driver_sha256"])}
        else:
            if (
                child.get("requested_tier") != tier
                or record.get("requested_tier") != tier
            ):
                raise Rejected("table: requested tier mismatch")
            if record["schema"] != 5:
                raise Rejected(
                    "table: modern acceptance requires schema-5 loader evidence"
                )
            device = artifacts.device(child.get("device"))
            capability = device["compute_capability"]
            code = child.get("code_sha256", {})
        if not entry["devices"] or set(entry["devices"]) != {capability}:
            raise Rejected("table: device capability mismatch")
        production_load(record["target"], tier, capability, root)
        loaded = [
            module
            for module in child.get("loaded_ptx", {}).get("modules", [])
            if module.get("area") == record["target"]
        ]
        if len(loaded) != 1:
            raise Rejected("table: missing or duplicate loaded candidate artifact")
        if legacy:
            loaded_key = {"kind": "PtxJit", "sha256": digest(loaded[0].get("sha256"))}
            recorded_artifact = loaded_key
            recorded_device = {
                "name": LEGACY_DEVICE_NAME,
                "compute_capability": "12.0",
                "legacy_driver_sha256": digest(child["driver_sha256"]),
            }
        else:
            loaded_key = artifacts.module(loaded[0], root, device_capability=capability)
            recorded_artifact = loaded[0]["artifact"]
            recorded_device = device
        artifacts.shipped_key(
            entry["area"], tier, capability, entry.get("artifact"), root
        )
        if (
            artifacts.key(entry.get("artifact"), device_capability=capability)
            != loaded_key
        ):
            raise Rejected("table: selected artifact differs from qualified load")
        speed_scope(entry, legacy=legacy, capability=capability, device=recorded_device)
        selected = triples(entry["coverage"])
        accepted = evaluation["accepted_tuples"]
        outside = selected - {tuple(row) for row in accepted}
        if outside:
            raise Rejected(f"table: tuples outside accepted record: {sorted(outside)}")
        recorded_configs = configurations.recorded(
            entry["record"], child, entry["area"], is_legacy=legacy
        )
        configurations.check(entry, recorded_configs, selected)
        exported = entry.get("candidate_coverage")
        if exported is None:
            raise Rejected("table: missing Rust candidate coverage export")
        declared = child.get("coverage_declared", {}).get("triples")
        if declared is None:
            raise Rejected("table: missing recorded candidate coverage")
        current_declaration = tested_declaration(exported)
        recorded_declaration = tested_declaration({"triples": declared})
        if current_declaration != recorded_declaration:
            raise Rejected(
                "table: candidate coverage differs from accepted declaration"
            )
        files, gap = check_binding(entry["record"], record, child, tier, root)
        evidence.append(
            {
                "record": entry["record"],
                "area": record["target"],
                "device_name": LEGACY_DEVICE_NAME
                if legacy
                else child.get("device", {}).get("name"),
                "record_schema": record["schema"],
                "der": entry["der"],
                "tier": tier,
                "device_capability": capability,
                "artifact": recorded_artifact,
                "device": recorded_device,
                "configurations": recorded_configs,
                "environment": environment.collect(
                    child, legacy=legacy, recorded_device=recorded_device
                ),
                "legacy": legacy,
                "source_evidence_gap": gap,
                "shared_source_amendment": SHARED_SOURCE_AMENDMENT if legacy else None,
                "infrastructure_amendments": legacy_amendments(
                    LEGACY_BINDINGS[entry["record"]]
                )
                if legacy
                else [],
                "raw_tier_status": child["status"],
                "raw_tier_reason": child["reason"],
                "raw_record_reason": record.get("reason"),
                "verdict_evaluation": evaluation,
                "coverage_sha256": coverage_digest(current_declaration),
                "recorded_coverage_sha256": coverage_digest(recorded_declaration),
                "lock_digest": digest(record["lock_digest"]),
                "code_sha256": code,
                "files": files,
                "ptx_sha256": files["ptx"],
                "tuples": [list(row) for row in sorted(selected)],
            }
        )
    receipts = der_evidence.receipts(
        entries, lambda pin: load(pin, root, records=records)
    )
    for row in evidence:
        row["der_evidence"] = receipts[row["der"]]
    environment.consistent(evidence)
    return {"entries": evidence}


ACCEPTANCE = "scripts/cuda/qualify/ACCEPTANCE.json"
SUMMARY_FIELDS = {
    "record",
    "area",
    "device_name",
    "record_schema",
    "der",
    "der_evidence",
    "tier",
    "device_capability",
    "artifact",
    "device",
    "configurations",
    "environment",
    "legacy",
    "source_evidence_gap",
    "shared_source_amendment",
    "infrastructure_amendments",
    "raw_tier_status",
    "raw_tier_reason",
    "raw_record_reason",
    "verdict_evaluation",
    "coverage_sha256",
    "recorded_coverage_sha256",
    "lock_digest",
    "files",
    "accepted_tuples",
}


def canonical_summaries(value: dict) -> bytes:
    """Serialize acceptance evidence with stable ordering and finite JSON values"""
    return (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()


def derive_summaries(entries: list[dict], root: Path = ROOT, *, records: Path) -> dict:
    """Derive complete per-record acceptance evidence from immutable raw bytes"""
    checked = check_table_records(entries, root, records=records)
    summaries: dict[str, dict] = {}
    for evidence in checked["entries"]:
        summary = {
            name: value
            for name, value in evidence.items()
            if name not in ("tuples", "code_sha256", "ptx_sha256")
        }
        summary["accepted_tuples"] = evidence["verdict_evaluation"]["accepted_tuples"]
        pin = evidence["record"]
        if pin in summaries and summaries[pin] != summary:
            raise Rejected("table: conflicting entries for one acceptance record")
        summaries[pin] = summary
    return {"schema": 1, "records": summaries}


def write_summaries(entries: list[dict], root: Path = ROOT, *, records: Path) -> dict:
    """Write canonical acceptance evidence after complete raw-record verification"""
    result = derive_summaries(entries, root, records=records)
    path = root / ACCEPTANCE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_summaries(result))
    return result


def acceptance_summary(root: Path) -> tuple[dict, bytes]:
    """Read canonical acceptance evidence bound by the committed harness lock"""
    path = root / ACCEPTANCE
    raw = path.read_bytes() if path.is_file() and not path.is_symlink() else b""
    if not raw:
        raise Rejected("table: missing acceptance summary")
    lock_path = root / "scripts/cuda/qualify/LOCK"
    if lock_path.is_symlink() or not lock_path.is_file():
        raise Rejected("table: missing summary lock")
    snapshot = json.loads(lock_path.read_bytes())
    if (
        not isinstance(snapshot, dict)
        or type(snapshot.get("schema")) is not int
        or snapshot["schema"] != 1
    ):
        raise Rejected("table: invalid summary lock schema")
    files = snapshot.get("files")
    if (
        not isinstance(files, dict)
        or digest(files.get(ACCEPTANCE)) != hashlib.sha256(raw).hexdigest()
    ):
        raise Rejected("table: acceptance summary differs from locked hash")
    result = json.loads(raw)
    if (
        not isinstance(result, dict)
        or type(result.get("schema")) is not int
        or result["schema"] != 1
        or set(result) != {"schema", "records"}
        or not isinstance(result["records"], dict)
    ):
        raise Rejected("table: unsupported acceptance summary schema")
    if canonical_summaries(result) != raw:
        raise Rejected("table: acceptance summary is not canonical JSON")
    return result, raw


def summary_tuples(value: object) -> set[tuple[str, int, str]]:
    """Validate the exact production tuple representation stored in a summary"""
    if not isinstance(value, list) or not value:
        raise Rejected("table: missing accepted summary tuples")
    found = set()
    for row in value:
        if (
            not isinstance(row, list)
            or len(row) != 3
            or not isinstance(row[0], str)
            or not row[0]
            or type(row[1]) is not int
            or row[1] not in boundary(row[0]).production
            or row[2] not in ("fp32", "tf32")
        ):
            raise Rejected("table: invalid accepted summary tuple")
        found.add(tuple(row))
    if [list(row) for row in sorted(found)] != value:
        raise Rejected("table: summary tuples must be sorted and unique")
    return found


def check_table(
    entries: list[dict], root: Path = ROOT, records: Path | None = None
) -> dict:
    """Check locked acceptance summaries offline or against explicit raw records"""
    if not entries:
        raise Rejected("table: no production entries")
    unique_artifact_owners(entries)
    committed, raw = acceptance_summary(root)
    summaries = committed["records"]
    expanded = configurations.record_entries(entries)
    pins = {digest(entry["record"]) for entry in expanded}
    if set(summaries) != pins:
        raise Rejected("table: acceptance summaries differ from production record pins")
    evidence = []
    for entry in expanded:
        pin = entry["record"]
        summary = summaries[pin]
        if (
            not isinstance(summary, dict)
            or set(summary) != SUMMARY_FIELDS
            or summary.get("record") != pin
        ):
            raise Rejected("table: invalid acceptance record binding")
        if summary.get("raw_tier_status") not in (
            "passed",
            "blocked",
        ) or not isinstance(summary.get("raw_tier_reason"), str):
            raise Rejected("table: invalid summary raw verdict")
        if summary.get("raw_record_reason") is not None and not isinstance(
            summary["raw_record_reason"], str
        ):
            raise Rejected("table: invalid summary raw record reason")
        tier = entry["tier"]
        area = entry.get("area", summary.get("area"))
        if (
            summary.get("tier") != tier
            or summary.get("area") != area
            or area not in AREA_HOST
        ):
            raise Rejected("table: summary area or tier mismatch")
        capability = summary.get("device_capability")
        if (
            not isinstance(capability, str)
            or not entry["devices"]
            or set(entry["devices"]) != {capability}
        ):
            raise Rejected("table: summary device capability mismatch")
        if (
            not isinstance(summary.get("device_name"), str)
            or not summary["device_name"]
        ):
            raise Rejected("table: invalid summary device name")
        if digest(summary.get("der")) != digest(entry["der"]):
            raise Rejected("table: summary DER evidence mismatch")
        digest(summary.get("lock_digest"))
        if type(summary.get("legacy")) is not bool or summary["legacy"] != (
            pin in LEGACY
        ):
            raise Rejected("table: invalid summary legacy binding")
        if summary["legacy"]:
            if summary.get("configurations") != configurations.legacy(pin):
                raise Rejected(
                    "table: summary differs from explicit legacy configuration mapping"
                )
            binding = LEGACY_BINDINGS[pin]
            if (
                summary.get("files") != amended_legacy_files(binding)
                or summary.get("infrastructure_amendments")
                != legacy_amendments(binding)
                or summary["lock_digest"] != binding["lock_digest"]
                or summary.get("source_evidence_gap") != SOURCE_EVIDENCE_GAP
                or summary.get("shared_source_amendment") != SHARED_SOURCE_AMENDMENT
                or summary["device_name"] != LEGACY_DEVICE_NAME
                or LEGACY[pin] != area
                or tier != "sm75"
                or capability != "12.0"
            ):
                raise Rejected("table: invalid summary legacy acceptance binding")
        elif (
            summary.get("infrastructure_amendments") != []
            or summary.get("source_evidence_gap") is not None
            or summary.get("shared_source_amendment") is not None
        ):
            raise Rejected("table: invalid modern summary source evidence")
        bound = summary.get("files")
        if not isinstance(bound, dict) or set(bound) != {
            "ptx",
            "manifests",
            "host_sources",
            "kernel_sources",
            *([] if summary["legacy"] else ["cubins"]),
        }:
            raise Rejected("table: missing summary file bindings")
        for hashes in bound.values():
            if not isinstance(hashes, dict) or not hashes:
                raise Rejected("table: invalid summary file bindings")
            for name, value in hashes.items():
                if (
                    not isinstance(name, str)
                    or Path(name).is_absolute()
                    or ".." in Path(name).parts
                ):
                    raise Rejected("table: invalid summary file path")
                digest(value)
        production_load(area, tier, capability, root)
        current = shipped_files(root, area, legacy=summary["legacy"])
        if bound != current:
            raise Rejected("table: qualification files differ from acceptance summary")
        artifact = summary.get("artifact")
        if summary["legacy"]:
            expected_hash = LEGACY_BINDINGS[pin]["files"]["ptx"][
                f"src/inference/cuda/ptx/{area}.{tier}.ptx"
            ]
            loaded_key = {"kind": "PtxJit", "sha256": expected_hash}
            if not isinstance(summary.get("device"), dict):
                raise Rejected("table: missing explicit legacy device mapping")
            if artifact != loaded_key or summary["device"] != {
                "name": LEGACY_DEVICE_NAME,
                "compute_capability": "12.0",
                "legacy_driver_sha256": summary["device"].get("legacy_driver_sha256"),
            }:
                raise Rejected(
                    "table: invalid explicit legacy artifact or device mapping"
                )
            digest(summary["device"].get("legacy_driver_sha256"))
        else:
            precise = artifacts.device(summary.get("device"))
            if (
                precise["name"] != summary["device_name"]
                or precise["compute_capability"] != capability
            ):
                raise Rejected("table: inconsistent precise device evidence")
            ptx_hash = bound["ptx"].get(f"src/inference/cuda/ptx/{area}.{tier}.ptx")
            loaded_key = artifacts.module(
                {
                    "area": area,
                    "tier": tier,
                    "sha256": ptx_hash,
                    "embedded_ptx_sha256": ptx_hash,
                    "artifact": artifact,
                },
                root,
                device_capability=capability,
            )
        artifacts.shipped_key(
            entry["area"], tier, capability, entry.get("artifact"), root
        )
        if (
            artifacts.key(entry.get("artifact"), device_capability=capability)
            != loaded_key
        ):
            raise Rejected("table: selected artifact differs from qualified load")
        speed_scope(
            entry,
            legacy=summary["legacy"],
            capability=capability,
            device=summary["device"],
        )
        exported = entry.get("candidate_coverage")
        if not isinstance(exported, dict):
            raise Rejected("table: missing Rust candidate coverage export")
        coverage = digest(summary.get("coverage_sha256"))
        recorded = digest(summary.get("recorded_coverage_sha256"))
        if (
            coverage != recorded
            or coverage_digest(tested_declaration(exported)) != coverage
        ):
            raise Rejected("table: candidate coverage differs from acceptance summary")
        accepted = summary_tuples(summary.get("accepted_tuples"))
        evaluation = summary.get("verdict_evaluation")
        if (
            not isinstance(evaluation, dict)
            or evaluation.get("accepted_tuples") != summary["accepted_tuples"]
            or evaluation.get("hard_failures") != []
            or not isinstance(evaluation.get("noise_rule"), list)
            or type(evaluation.get("accepted")) is not bool
            or not isinstance(evaluation.get("unresolved"), list)
            or evaluation["accepted"] != (not evaluation["unresolved"])
        ):
            raise Rejected("table: invalid summary verdict evidence")
        schema = summary.get("record_schema")
        if (
            type(schema) is not int
            or schema not in (3, 5)
            or evaluation.get("stage_noise_rule_allowed") is not (schema == 3)
        ):
            raise Rejected("table: invalid summary record schema")
        for noise in evaluation["noise_rule"]:
            try:
                check = {
                    "check": noise["check"],
                    "passed": False,
                    "blocked": True,
                    "reason": NOISE_REASON,
                }
                timing = {
                    "id": noise["check"].removeprefix("speed:"),
                    "medians_ms": noise["medians_ms"],
                    "speedups": noise["pair_speedups"],
                    "library_process_spread_fraction": noise["library_spread_fraction"],
                    "spread_bound": noise["locked_bound"],
                }
                if noise_timing(check, timing, allow_stage=schema == 3) != noise:
                    raise Rejected(
                        "table: summary noise verdict differs from its inputs"
                    )
            except (KeyError, TypeError, ValueError) as error:
                raise Rejected(
                    f"table: invalid summary noise evidence: {error}"
                ) from error
        if not triples(entry["coverage"]) <= accepted:
            raise Rejected("table: tuples outside accepted summary")
        configurations.check(
            entry, summary.get("configurations"), triples(entry["coverage"])
        )
        evidence.append(summary)
    der_evidence.offline(entries, summaries)
    environment.consistent(evidence)
    if (
        records is not None
        and canonical_summaries(derive_summaries(entries, root, records=records)) != raw
    ):
        raise Rejected("table: acceptance summary differs from raw-record derivation")
    return {
        "entries": evidence,
        "verification": "raw-records" if records is not None else "locked-summary",
    }
