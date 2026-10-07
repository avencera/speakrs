"""Locked verdict evaluation, including recomputable Library-noise timing decisions."""

import math
from collections.abc import Sequence

from gates import Rejected, finite, spread_bound
from domains import boundary as batch_domain


NOISE_REASON = "speed: blocked: Library process spread "


def check_name(name: str) -> str:
    """Strip an optional tier prefix, never an arbitrary first path component."""
    prefix, _, rest = name.partition("/")
    return rest if prefix in ("sm75", "sm80", "sm90", "sm120") else name


def noise_timing(check: dict, timing: dict, *, allow_stage: bool) -> dict:
    """Evaluate only a raw timing check blocked solely by Library process spread.

    Validate stored ratios and spread against the four medians. The bound is the
    locked operator/stage constant, not an editable value supplied by a record.
    """
    name = check_name(check["check"])
    if check.get("passed") or not check.get("blocked") or not name.startswith("speed:"):
        raise Rejected("noise rule: not a noise-blocked timing check")
    if not check.get("reason", "").startswith(NOISE_REASON):
        raise Rejected(
            "noise rule: timing blocked for a reason other than Library noise"
        )
    key = name.removeprefix("speed:")
    if key.endswith("/stage") and not allow_stage:
        raise Rejected("noise rule: new stage records require the paired ABBA guard")
    if timing.get("id") != key:
        raise Rejected("noise rule: timing case does not match raw check")
    medians = timing["medians_ms"]
    pairs = timing["speedups"]
    spread = timing["library_process_spread_fraction"]
    bound = spread_bound(key)
    finite([*medians, *pairs, spread, timing["spread_bound"]])
    if len(medians) != 4 or len(pairs) != 2 or min(medians) <= 0:
        raise Rejected("noise rule: invalid A/B/A/B timing evidence")
    derived = [medians[0] / medians[1], medians[2] / medians[3]]
    derived_spread = abs(medians[0] - medians[2]) / min(medians[0], medians[2])
    if timing["spread_bound"] != bound or spread <= bound:
        raise Rejected("noise rule: invalid locked bound or no Library-noise block")
    if not all(
        math.isclose(a, b, rel_tol=1e-12, abs_tol=0)
        for a, b in zip(pairs, derived, strict=True)
    ) or not math.isclose(spread, derived_spread, rel_tol=1e-12, abs_tol=0):
        raise Rejected("noise rule: recorded ratios or spread disagree with medians")
    threshold = 1.0 + 3.0 * max(spread, bound)
    return {
        "check": name,
        "raw_passed": False,
        "raw_blocked": True,
        "medians_ms": medians,
        "pair_speedups": pairs,
        "library_spread_fraction": spread,
        "locked_bound": bound,
        "threshold": threshold,
        "min_pair_speedup": min(pairs),
        "accepted": min(pairs) >= threshold,
        "rule": "min_pair_speedup >= 1 + 3 * max(library_spread, locked_bound)",
    }


def evaluate_checks(
    checks: Sequence[dict], timing: Sequence[dict], *, allow_stage: bool
) -> dict:
    """Keep all raw outcomes and return a separate decision; hard failures never pass."""
    if not checks:
        raise Rejected("verdict: missing checks")
    details = {row["id"]: row for row in timing}
    if len(details) != len(timing):
        raise Rejected("verdict: duplicate timing case")
    noise = []
    unresolved = []
    hard = []
    for check in checks:
        if check.get("passed") is True:
            continue
        name = check_name(check["check"])
        if not check.get("blocked"):
            hard.append(name)
            continue
        row = details.get(name.removeprefix("speed:"))
        try:
            if row is None:
                raise Rejected("noise rule: missing matching timing data")
            evaluation = noise_timing(check, row, allow_stage=allow_stage)
            noise.append(evaluation)
            if not evaluation["accepted"]:
                unresolved.append(
                    {
                        "check": name,
                        "reason": "noise rule: smaller pair speedup below threshold",
                    }
                )
        except (Rejected, KeyError, TypeError, ValueError) as error:
            unresolved.append({"check": name, "reason": str(error)})
    return {
        "accepted": not hard and not unresolved,
        "hard_failures": hard,
        "unresolved": unresolved,
        "noise_rule": noise,
        "stage_noise_rule_allowed": allow_stage,
    }


def evaluate_record(record: dict, tier: str) -> dict:
    """Re-evaluate recorded checks and accepted tuples without a GPU or status bypass."""
    child = record.get("tiers", {}).get(tier)
    if child is None:
        raise Rejected("table: tier not accepted by record")
    if record.get("implementation") != "Oxide":
        raise Rejected(
            "table: Library controls and mutants cannot authorize production"
        )
    raw_status = record.get("status")
    if raw_status not in ("passed", "blocked") or child.get("status") not in (
        "passed",
        "blocked",
    ):
        raise Rejected("table: record has a rejecting verdict")
    allow_stage = record.get("schema", 3) < 4
    decision = evaluate_checks(
        child.get("checks", []), child.get("timing", []), allow_stage=allow_stage
    )
    # parent-only scan failures must not disappear when one tier is selected
    parent = [
        row
        for row in record.get("checks", [])
        if "/" not in row["check"].split(":", 1)[0]
    ]
    if any(row.get("passed") is not True for row in parent):
        decision["hard_failures"].extend(
            check_name(row["check"]) for row in parent if row.get("passed") is not True
        )
        decision["accepted"] = False
    declared = child.get("coverage_declared", {}).get(
        "triples", child.get("accepted_tuples", [])
    )
    accepted = {
        tuple(row) for row in declared if row[1] in batch_domain(row[0]).production
    }
    excluded = set()
    if decision["hard_failures"]:
        excluded = accepted.copy()
    else:
        for item in decision["unresolved"]:
            name = item["check"]
            if not name.startswith("speed:"):
                excluded |= accepted
                continue
            try:
                mode, _, batch, boundary = name.removeprefix("speed:").split("/", 3)
                batch = int(batch.removeprefix("b"))
            except ValueError:
                excluded |= accepted
                continue
            # stress failures remove the same layer/mode's production claim
            excluded |= {
                triple
                for triple in accepted
                if triple[2] == mode
                and (
                    batch not in batch_domain(triple[0]).production
                    or triple[1] == batch
                )
                and (boundary == "stage" or triple[0] == boundary)
            }
    accepted -= excluded
    if not allow_stage:
        # newly recorded acceptance cannot exceed its own final, pinned tuple list
        accepted &= {tuple(row) for row in child.get("accepted_tuples", [])}
    return {
        **decision,
        "raw_status": raw_status,
        "raw_accepts_replacement": record.get("accepts_replacement"),
        "accepted_tuples": [list(row) for row in sorted(accepted)],
        "excluded_tuples": [list(row) for row in sorted(excluded)],
    }
