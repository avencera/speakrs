"""Validate integrated DER receipts against the whole selected CUDA execution plan."""

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import artifacts
import configurations
from domains import boundary
from gates import Rejected

LEGACY_HASH = "8066268031afba058d93e305206d5b8225e40c6ab2ebb1052874d607e055646f"
LEGACY_RECORDS = set(configurations.LEGACY_AREAS)
LEGACY_PLAN = json.loads((Path(__file__).with_name("LEGACY_DER.json")).read_bytes())
LEGACY_GAP = (
    "PR #36 integrated summary predates whole-plan receipts; it has no model or "
    "audio content hashes and no per-process environment query. Only its exact "
    "hash, original candidate configurations, capability-wide scope and archived "
    "cuda/cuda-fast A/B verdict are mapped. Removed tuple permissions stay removed."
)


def digest(value: object) -> str:
    """Bind finite canonical JSON, not presentation order or JSON number spellings"""
    try:
        raw = json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    except (TypeError, ValueError) as error:
        raise Rejected("DER: invalid canonical evidence") from error
    return hashlib.sha256(raw).hexdigest()


def fields(raw: object, names: set[str], context: str) -> dict:
    """Require a closed schema before any plan identity is used"""
    if not isinstance(raw, dict):
        raise Rejected(f"DER: missing {context}")
    value: dict = dict(raw)
    if set(value) != names:
        raise Rejected(f"DER: invalid {context} fields")
    return value


def scope(raw: object) -> dict:
    """Require the exact speed-evidence device scope, never a new range"""
    if not isinstance(raw, dict):
        raise Rejected("DER: missing device scope")
    value: dict = dict(raw)
    kind = value.get("kind")
    names = {"kind", "capability"}
    if kind == "Point":
        names |= {"sm_count", "device_name"}
        if (
            type(value.get("sm_count")) is not int
            or value["sm_count"] <= 0
            or not isinstance(value.get("device_name"), str)
            or not value["device_name"].strip()
        ):
            raise Rejected("DER: invalid point scope")
    elif kind != "LegacyCapability":
        raise Rejected("DER: invalid scope kind")
    value = fields(value, names, "device scope")
    artifacts.capability(value["capability"])
    return value


@dataclass(frozen=True)
class PlanDomain:
    """Complete typed-owner boundary domain, independent of candidate declarations"""

    tuples: tuple[tuple[str, int], ...]

    @classmethod
    def parse(cls, raw: object) -> "PlanDomain":
        if not isinstance(raw, list) or not raw:
            raise Rejected("DER: missing whole-plan boundary domain")
        result = []
        seen = set()
        for row in raw:
            value = fields(row, {"boundary", "batches"}, "boundary domain")
            name, batches = value["boundary"], value["batches"]
            if (
                not isinstance(name, str)
                or not name
                or name in seen
                or not isinstance(batches, list)
                or not batches
            ):
                raise Rejected("DER: invalid boundary domain")
            expected = list(boundary(name).production)
            if batches != expected or any(type(batch) is not int for batch in batches):
                raise Rejected("DER: invalid boundary batch domain")
            seen.add(name)
            result.extend((name, batch) for batch in batches)
        return cls(tuple(sorted(result)))

    def library_routes(self) -> list[dict]:
        """The independent baseline selects Library at every model boundary"""
        return [
            {"boundary": name, "batch": batch, "route": {"kind": "Library"}}
            for name, batch in self.tuples
        ]


def models(raw: object) -> dict:
    """Both model content identities are required for an integrated pipeline"""
    value = fields(raw, {"segmentation", "embedding"}, "models")
    return {name: artifacts.sha256(pin) for name, pin in value.items()}


def routes(entries: list[dict], domain: PlanDomain, mode: str) -> list[dict]:
    """Resolve every boundary, including Library routes, from evaluated tuple proofs"""
    selected = {}
    for entry in entries:
        for (name, batch, math_mode), pin in configurations.tuples(
            entry.get("configurations"), entry["area"]
        ).items():
            if math_mode != mode:
                continue
            key = (name, batch)
            identity = {
                "kind": "Candidate",
                "area": entry["area"],
                "tier": entry["tier"],
                "artifact": artifacts.key(
                    entry.get("artifact"),
                    device_capability=entry["speed_scope"]["capability"],
                ),
                "pin": pin,
            }
            if key in selected and selected[key] != identity:
                raise Rejected("DER: conflicting selected routes")
            selected[key] = identity
    if not selected.keys() <= set(domain.tuples):
        raise Rejected("DER: selected route outside model domain")
    return [
        {
            "boundary": name,
            "batch": batch,
            "route": selected.get((name, batch), {"kind": "Library"}),
        }
        for name, batch in domain.tuples
    ]


def metric(raw: object, context: str) -> dict:
    """Retain scorer outputs and their exact external evidence identity"""
    value = fields(raw, {"der", "output_sha256", "identity_sha256"}, context)
    if (
        type(value["der"]) not in (int, float)
        or not math.isfinite(value["der"])
        or value["der"] < 0
    ):
        raise Rejected(f"DER: invalid {context} score")
    artifacts.sha256(value["output_sha256"])
    artifacts.sha256(value["identity_sha256"])
    return value


def validate_plan(raw: object, entries: list[dict], mode: str) -> dict:
    """Bind model, input, configuration, baseline and verdict to one complete plan"""
    value = fields(
        raw,
        {
            "device_scope",
            "math",
            "models",
            "inputs",
            "pipeline_config_sha256",
            "library_artifacts",
            "routes",
            "baseline",
            "candidate",
            "verdict",
        },
        "plan",
    )
    first = entries[0]
    domain = PlanDomain.parse(first.get("boundary_domain"))
    if scope(value["device_scope"]) != first["speed_scope"] or value["math"] != mode:
        raise Rejected("DER: device scope or math differs from execution plan")
    if models(value["models"]) != models(first.get("models")):
        raise Rejected("DER: model identity differs from execution plan")
    inputs = fields(
        value["inputs"],
        {"manifest_sha256", "reference_sha256", "files"},
        "input identity",
    )
    artifacts.sha256(inputs["manifest_sha256"])
    artifacts.sha256(inputs["reference_sha256"])
    if type(inputs["files"]) is not int or inputs["files"] <= 0:
        raise Rejected("DER: invalid input file count")
    artifacts.sha256(value["pipeline_config_sha256"])
    library = fields(
        value["library_artifacts"],
        {"fbank", "embedding", "segmentation"},
        "Library artifacts",
    )
    for item in library.values():
        item = fields(item, {"tier", "artifact"}, "Library module")
        if not isinstance(item["tier"], str) or item["tier"] not in artifacts.TIERS:
            raise Rejected("DER: invalid Library artifact tier")
        artifacts.key(
            item["artifact"], device_capability=value["device_scope"]["capability"]
        )
    if library != first.get("library_artifacts"):
        raise Rejected("DER: Library-owned artifacts differ from execution plan")
    expected = routes(entries, domain, mode)
    if value["routes"] != expected:
        raise Rejected(
            "DER: selected routes, configurations or artifacts differ from whole plan"
        )
    identity = digest(
        {
            name: value[name]
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
    baseline = fields(
        value["baseline"], {"routes", "control_archive_sha256", "metrics"}, "baseline"
    )
    if baseline["routes"] != domain.library_routes():
        raise Rejected("DER: baseline is not the full Library execution plan")
    artifacts.sha256(baseline["control_archive_sha256"])
    base_metrics = metric(baseline["metrics"], "baseline metrics")
    candidate = metric(value["candidate"], "candidate metrics")
    if (
        base_metrics["identity_sha256"] != identity
        or candidate["identity_sha256"] != identity
    ):
        raise Rejected("DER: baseline and candidate do not share the plan snapshot")
    verdict = fields(
        value["verdict"],
        {"passed", "baseline_sha256", "candidate_sha256", "policy_sha256"},
        "verdict",
    )
    artifacts.sha256(verdict["policy_sha256"])
    if (
        verdict["passed"] is not True
        or verdict["baseline_sha256"] != digest(baseline)
        or verdict["candidate_sha256"] != digest(candidate)
    ):
        raise Rejected(
            "DER: verdict does not accept these baseline and candidate bytes"
        )
    return value


def legacy_receipt(entries: list[dict]) -> dict:
    """The exact archived integrated hash covers only original PR #36 identities"""
    expected_scope = {"kind": "LegacyCapability", "capability": "12.0"}
    for entry in entries:
        if (
            entry["record"] not in LEGACY_RECORDS
            or entry.get("accuracy_record", entry["record"]) not in LEGACY_RECORDS
            or entry["speed_scope"] != expected_scope
            or entry["tier"] != "sm75"
        ):
            raise Rejected("DER: legacy evidence cannot authorize a new execution plan")
        if (
            models(entry.get("models")) != LEGACY_PLAN["models"]
            or digest(entry.get("boundary_domain")) != LEGACY_PLAN["domain_sha256"]
            or entry.get("library_artifacts") != LEGACY_PLAN["library_artifacts"]
        ):
            raise Rejected(
                "DER: legacy model, domain or Library artifact identity changed"
            )
        configurations.check(
            entry,
            configurations.legacy(entry["record"]),
            set(configurations.tuples(entry["configurations"], entry["area"])),
        )
    return {
        "kind": "LegacyWholePlan",
        "source_sha256": LEGACY_HASH,
        "device_scope": expected_scope,
        "maths": ["fp32", "tf32"],
        "records": sorted(LEGACY_RECORDS),
        "evidence_gap": LEGACY_GAP,
        "plan_mapping": LEGACY_PLAN,
        "configurations": {
            pin: configurations.legacy(pin) for pin in sorted(LEGACY_RECORDS)
        },
    }


def check(raw: object, pin: str, entries: list[dict]) -> dict:
    """Check one integrated receipt against all device entries, not one area alone"""
    if pin == LEGACY_HASH:
        return legacy_receipt(entries)
    value = fields(raw, {"schema", "plans"}, "evidence")
    if (
        type(value["schema"]) is not int
        or value["schema"] != 1
        or not isinstance(value["plans"], list)
    ):
        raise Rejected("DER: invalid schema")
    needed = sorted(
        {row["tuple"][2] for entry in entries for row in entry["configurations"]}
    )
    plans = value["plans"]
    if len(plans) != len(needed):
        raise Rejected("DER: missing or duplicate math plans")
    checked = [
        validate_plan(plan, entries, mode)
        for plan, mode in zip(plans, needed, strict=True)
    ]
    return {
        "kind": "WholePlan",
        "source_sha256": pin,
        "evidence": {"schema": 1, "plans": checked},
    }


def groups(entries: list[dict]) -> list[list[dict]]:
    """An integrated evidence hash and domain cover every entry on the same scope"""
    grouped: dict[str, list[dict]] = {}
    for entry in entries:
        key = digest(scope(entry.get("speed_scope")))
        grouped.setdefault(key, []).append(entry)
    for group in grouped.values():
        first = group[0]
        if any(
            entry["der"] != first["der"]
            or entry.get("boundary_domain") != first.get("boundary_domain")
            or entry.get("models") != first.get("models")
            or entry.get("library_artifacts") != first.get("library_artifacts")
            for entry in group
        ):
            raise Rejected("DER: device entries do not share one whole execution plan")
    return list(grouped.values())


def receipts(entries: list[dict], load) -> dict[str, dict]:
    """Load immutable bytes once per integrated execution plan"""
    return {
        group[0]["der"]: check(load(group[0]["der"]), group[0]["der"], group)
        for group in groups(entries)
    }


def offline(entries: list[dict], summaries: dict) -> None:
    """Rebind owner-locked DER receipts to the current evaluated production plan"""
    for group in groups(entries):
        pin = group[0]["der"]
        recorded = [summaries[entry["record"]].get("der_evidence") for entry in group]
        if not recorded or any(receipt != recorded[0] for receipt in recorded):
            raise Rejected("DER: missing or inconsistent locked whole-plan receipt")
        receipt = recorded[0]
        if not isinstance(receipt, dict):
            raise Rejected("DER: missing locked receipt")
        actual = check(receipt.get("evidence"), pin, group)
        if actual != receipt:
            raise Rejected("DER: locked evidence differs from the execution plan")
