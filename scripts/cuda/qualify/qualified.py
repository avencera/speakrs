"""Accept archived qualification records and check production qualification on CPU"""

import argparse
import gzip
import hashlib
import itertools
import json
import math
import re
import sys
from pathlib import Path

from scan import strip_comments

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = "scripts/cuda/qualify/QUALIFIED.json"
AREAS = {
    "ConvOxide": ("resnet", "conv"),
    "LstmOxide": ("lstm", "lstm"),
    "SincOxide": ("sincnet", "sinc"),
}
SOURCE_GAP = (
    "record has no host-source, kernel-source, or PTX build-manifest hashes; "
    "these hashes were captured at acceptance"
)


class QualificationError(ValueError):
    """The record or production files do not support qualification"""


def require(condition: bool, reason: str) -> None:
    """Refuse incomplete or inconsistent evidence"""
    if not condition:
        raise QualificationError(reason)


def digest(path: Path) -> str:
    """Hash the exact regular-file bytes that will ship"""
    require(path.is_file() and not path.is_symlink(), f"not a regular file: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256(value: str) -> bool:
    """Whether a value is a full SHA-256 digest"""
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def source(root: Path, name: str) -> str:
    """Read Rust declarations without comments"""
    digest(root / name)
    return strip_comments((root / name).read_text())


def declaration(text: str, name: str) -> str:
    """Read one constant initializer; unfamiliar syntax fails closed"""
    matches = re.findall(rf"\bconst\s+{name}\s*:[^=]+?=\s*(.*?);", text, re.S)
    require(len(matches) == 1, f"missing or duplicate Rust constant: {name}")
    return matches[0].strip()


def array(expression: str) -> list[str]:
    """Parse the literal arrays used by the coverage declarations"""
    match = re.fullmatch(r"&?\[(.*)\]", expression.strip(), re.S)
    require(match is not None, f"unsupported coverage array: {expression}")
    assert match is not None
    items = [item.strip() for item in match[1].split(",")]
    if items and not items[-1]:
        items.pop()
    require(all(items), "empty coverage array item")
    return items


def strings(expression: str, text: str) -> list[str]:
    """Resolve layer-name arrays and their local string constants"""
    if re.fullmatch(r"&[A-Z][A-Z0-9_]*", expression):
        return strings(declaration(text, expression[1:]), text)
    values = []
    for item in array(expression):
        if re.fullmatch(r"[A-Z][A-Z0-9_]*", item):
            item = declaration(text, item)
        require(
            re.fullmatch(r'"[a-z0-9_.]+"', item) is not None,
            f"unsupported layer: {item}",
        )
        values.append(json.loads(item))
    return values


def coverage(root: Path, host: str) -> dict:
    """Parse the code's declared coverage, preserving All versus Only.

    This deliberately accepts only the declaration grammar used by the candidate
    traits. A new expression needs a checker update, not a guessed interpretation
    """
    shared = source(root, "src/inference/cuda/candidate.rs")
    batches = array(declaration(shared, "QUALIFIED_BATCHES"))
    require(
        all(re.fullmatch(r"[1-9][0-9]*", b) for b in batches),
        "invalid qualified batches",
    )
    qualified_batches = [int(b) for b in batches]
    text = source(root, f"src/inference/cuda/candidate/{host}.rs")
    initializer = declaration(text, "COVERAGE")
    match = re.fullmatch(r"Coverage\(\s*&\[(.*)\]\s*\)", initializer, re.S)
    require(match is not None, "unsupported COVERAGE initializer")
    assert match is not None
    entries = []
    triples = set()
    entry_pattern = r"CoverageEntry\s*\{\s*layers:\s*(.*?),\s*batches:\s*(Batches::All|Batches::Only\(.*?\)),\s*maths:\s*(Maths::All|Maths::Only\(.*?\)),?\s*\}"
    remaining = match[1]
    for entry in re.finditer(entry_pattern, match[1], re.S):
        layers = strings(entry[1].strip(), text)
        batch_expr, math_expr = entry[2], entry[3]
        batch_only = (
            None
            if batch_expr == "Batches::All"
            else array(batch_expr[len("Batches::Only(") : -1])
        )
        require(
            batch_only is None
            or all(re.fullmatch(r"[1-9][0-9]*", b) for b in batch_only),
            "invalid Only batches",
        )
        selected_batches = (
            qualified_batches if batch_only is None else [int(b) for b in batch_only]
        )
        require(
            set(selected_batches) <= set(qualified_batches),
            "coverage exceeds harness batches",
        )
        modes = (
            ["fp32", "tf32"]
            if math_expr == "Maths::All"
            else array(math_expr[len("Maths::Only(") : -1])
        )
        if math_expr != "Maths::All":
            require(
                all(m in ["CudaMath::Fp32", "CudaMath::Tf32"] for m in modes),
                "unknown math mode",
            )
            modes = [m.removeprefix("CudaMath::").lower() for m in modes]
        require(
            bool(layers and selected_batches and modes), "empty production coverage"
        )
        entries.append(
            {
                "layers": layers,
                "batches": "All" if batch_only is None else selected_batches,
                "maths": "All" if math_expr == "Maths::All" else modes,
            }
        )
        triples.update(itertools.product(layers, selected_batches, modes))
        remaining = remaining.replace(entry[0], "", 1)
    require(
        not remaining.replace(",", "").strip() and bool(entries),
        "unsupported or empty coverage entries",
    )
    return {
        "qualified_batches": qualified_batches,
        "entries": entries,
        "triples": [list(t) for t in sorted(triples)],
    }


def production(root: Path) -> dict[str, str]:
    """Require a known candidate for every PRODUCTION entry"""
    text = source(root, "src/inference/cuda/implementation.rs")
    result = {}
    for item in array(declaration(text, "PRODUCTION")):
        match = re.fullmatch(r"([A-Za-z0-9_]+)::COVERAGE", item)
        require(
            match is not None and match[1] in AREAS,
            f"unknown PRODUCTION coverage: {item}",
        )
        assert match is not None
        area, host = AREAS[match[1]]
        require(area not in result, f"duplicate PRODUCTION coverage: {area}")
        result[area] = host
    require(bool(result), "empty PRODUCTION table")
    return result


def files(root: Path, area: str, host: str) -> dict[str, dict[str, str]]:
    """Collect all shipped tiers, manifests, and area source modules"""
    ptx = Path("src/inference/cuda/ptx")
    host_dir = Path("src/inference/cuda/candidate")
    kernel_dir = Path("crates/speakrs-cuda-kernels/src")
    groups = {
        "ptx": sorted((root / ptx).glob(f"{area}.*.ptx")),
        "manifests": [root / ptx / f"{area}.manifest"],
        "host_sources": [
            root / "src/inference/cuda/candidate.rs",
            root / host_dir / f"{host}.rs",
            *sorted((root / host_dir / host).rglob("*.rs")),
        ],
        "kernel_sources": [
            root / kernel_dir / f"{area}.rs",
            *sorted((root / kernel_dir / area).rglob("*.rs")),
        ],
    }
    require(bool(groups["ptx"]), f"no shipped PTX for {area}")
    return {
        group: {p.relative_to(root).as_posix(): digest(p) for p in paths}
        for group, paths in groups.items()
    }


def finite_positive(value: object) -> bool:
    """Reject nonfinite, nonnumeric, or nonpositive timing values"""
    return type(value) in (float, int) and math.isfinite(value) and value > 0


def completed_tier(tier: str, child: dict, target: str) -> None:
    """Reject interrupted collection, including unrecorded driver failures.

    collect() can catch an exception after timing and set a blocked status without
    adding a failed check. Only collect_tier()'s final coverage marker and exact
    finish_tier() verdict establish that collection reached its end
    """
    require(
        child.get("phases_run") == ["numeric", "timing", "profile", "sanitize"],
        f"incomplete qualification phases: {tier}",
    )
    require(
        child.get("coverage", {}).get("tier") == tier,
        f"incomplete tier collection: {tier}",
    )
    checks = child.get("checks", [])
    require(
        bool(checks) and all(type(c.get("passed")) is bool for c in checks),
        f"missing or invalid tier checks: {tier}",
    )
    names = {c["check"] for c in checks}
    required = {
        "ptx:loaded_bytes",
        "ptx:loaded_bytes/stable",
        "ptx:shared_initialization",
        "determinism:fixed_reduction_order",
        "profile",
        "profile:graph_nodes",
        "profile:captured_library_calls",
        *(f"sanitizer:control/{fault}" for fault in ["oob", "race", "uninit"]),
        *(f"sanitizer:Oxide/{tool}" for tool in ["memcheck", "racecheck", "initcheck"]),
    }
    if target == "lstm":
        required.update(
            f"sanitizer:ProjectionBaseline/{tool}"
            for tool in ["memcheck", "racecheck", "initcheck"]
        )
    require(
        required <= names,
        f"missing required tier checks: {tier}: {sorted(required - names)}",
    )
    failed = [c for c in checks if not c["passed"]]
    require(all(c.get("blocked") is True for c in failed), f"hard tier failure: {tier}")
    expected_status = "blocked" if failed else "passed"
    expected_reason = (
        f"{len(failed)} checks cannot be decided: {[c['check'] for c in failed[:6]]}"
        if failed
        else "all required checks passed"
    )
    require(
        child.get("status") == expected_status
        and child.get("reason") == expected_reason,
        f"incomplete or inconsistent tier verdict: {tier}",
    )


def verdict(record: dict) -> dict:
    """Accept a normal pass or a speed-only Library-noise block with a 3x margin"""
    require(
        record.get("schema") == 3 and record.get("implementation") == "Oxide",
        "not a schema-3 Oxide qualification record",
    )
    require(
        record.get("status") in ["passed", "blocked"],
        "qualification has a hard failure or invalid status",
    )
    checks = record.get("checks")
    tiers = record.get("tiers")
    require(
        isinstance(checks, list)
        and bool(checks)
        and isinstance(tiers, dict)
        and bool(tiers),
        "missing qualification checks or tiers",
    )
    for tier, child in tiers.items():
        completed_tier(tier, child, record.get("target"))
    tier_checks = [
        {**check, "check": f"{tier}/{check['check']}"}
        for tier, child in tiers.items()
        for check in child["checks"]
    ]
    require(
        all(c in checks for c in tier_checks), "tier checks missing from the aggregate"
    )
    require(
        len({c["check"] for c in checks}) == len(checks),
        "duplicate qualification checks",
    )
    require(all(type(c.get("passed")) is bool for c in checks), "invalid check verdict")
    failed = [check for check in checks if not check["passed"]]
    if record.get("accepts_replacement") is True:
        require(
            not failed and record["status"] == "passed",
            "accepts_replacement contradicts checks",
        )
        return {"rule": "accepts_replacement"}
    require(
        record.get("accepts_replacement") is False and bool(failed),
        "no acceptance verdict",
    )
    blocked = []
    for check in failed:
        name = check["check"]
        tier, _, speed = name.partition("/")
        require(
            check.get("blocked") is True
            and speed.startswith("speed:")
            and re.fullmatch(
                r"speed: blocked: Library process spread [0-9.]+ exceeds bound [0-9.]+",
                check.get("reason", ""),
            )
            is not None,
            f"not blocked only by Library spread: {name}",
        )
        rows = [
            row
            for row in tiers.get(tier, {}).get("timing", [])
            if row["id"] == speed.removeprefix("speed:")
        ]
        require(len(rows) == 1, f"missing or duplicate timing evidence: {name}")
        row = rows[0]
        medians = row.get("medians_ms", [])
        require(
            len(medians) == 4 and all(finite_positive(m) for m in medians),
            f"invalid medians: {name}",
        )
        spread = abs(medians[0] - medians[2]) / min(medians[0], medians[2])
        bound = 0.003 if row["id"].endswith("/stage") else 0.01
        require(
            row.get("measurable") is False
            and row.get("spread_bound") == bound
            and finite_positive(row.get("library_process_spread_fraction"))
            and math.isclose(
                spread, row["library_process_spread_fraction"], rel_tol=1e-9
            ),
            f"inconsistent Library spread evidence: {name}",
        )
        require(spread > bound, f"Library spread does not block: {name}")
        speedup = min(medians[0] / medians[1], medians[2] / medians[3])
        required = 1 + 3 * max(spread, bound)
        require(
            speedup >= required,
            f"noise margin too small: {name}: {speedup} < {required}",
        )
        blocked.append(
            {
                "check": name,
                "library_spread": spread,
                "spread_bound": bound,
                "smaller_pair_speedup": speedup,
                "required_speedup": required,
            }
        )
    return {"rule": "library_spread_noise", "blocked_cases": blocked}


def accept(root: Path, path: Path) -> dict:
    """Bind an accepted record to the current production files without GPU work"""
    raw = path.read_bytes()
    record = json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)
    basis = verdict(record)
    area = record.get("target")
    selected = production(root)
    require(area in selected, f"record area is not in PRODUCTION: {area}")
    declared = coverage(root, selected[area])
    recorded_files = files(root, area, selected[area])
    loaded = {}
    for tier, child in record["tiers"].items():
        require(
            child.get("target") == area and child.get("implementation") == "Oxide",
            f"wrong tier target: {tier}",
        )
        require(
            child.get("coverage_declared", {}).get("triples") == declared["triples"],
            f"record coverage differs from code: {area}/{tier}",
        )
        modules = [
            m
            for m in child.get("loaded_ptx", {}).get("modules", [])
            if m.get("area") == area
        ]
        require(
            len(modules) == 1,
            f"missing or duplicate loaded candidate PTX: {area}/{tier}",
        )
        module = modules[0]
        expected_path = f"src/inference/cuda/ptx/{area}.{tier}.ptx"
        require(
            module.get("path") == expected_path and module.get("tier") == tier,
            f"wrong loaded PTX path or tier: {area}/{tier}",
        )
        loaded[expected_path] = module.get("sha256")
    require(
        loaded == recorded_files["ptx"],
        f"loaded candidate PTX differs from shipped files: {area}",
    )
    lock_digest = record.get("lock_digest")
    require(sha256(lock_digest), "invalid record harness lock digest")
    entry = {
        "files": recorded_files,
        "coverage": declared,
        "harness_lock_digest": lock_digest,
        "record": {"name": path.name, "sha256": hashlib.sha256(raw).hexdigest()},
        "verdict_basis": basis,
        "source_evidence_gap": SOURCE_GAP,
    }
    manifest_path = root / MANIFEST
    manifest = (
        json.loads(manifest_path.read_text())
        if manifest_path.exists()
        else {"schema": 1, "areas": {}}
    )
    require(manifest.get("schema") == 1, "unsupported qualification manifest schema")
    manifest["areas"][area] = entry
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return entry


def check(root: Path) -> None:
    """Require an exact qualification entry for each production candidate area"""
    manifest = json.loads((root / MANIFEST).read_text())
    require(manifest.get("schema") == 1, "unsupported qualification manifest schema")
    selected = production(root)
    require(
        set(manifest["areas"]) == set(selected),
        "manifest entries differ from PRODUCTION coverage",
    )
    for area, host in selected.items():
        entry = manifest["areas"][area]
        current = files(root, area, host)
        for group, hashes in current.items():
            recorded = entry["files"].get(group, {})
            for name in sorted(hashes.keys() | recorded.keys()):
                require(
                    hashes.get(name) == recorded.get(name),
                    f"qualification file differs: {name}",
                )
        require(
            entry["coverage"] == coverage(root, host),
            f"qualification coverage differs from code: {area}",
        )
        require(
            sha256(entry["harness_lock_digest"]) and sha256(entry["record"]["sha256"]),
            f"invalid evidence digest: {area}",
        )
        require(
            Path(entry["record"]["name"]).name == entry["record"]["name"]
            and bool(entry["record"]["name"]),
            f"invalid record name: {area}",
        )
        require(
            entry.get("source_evidence_gap") == SOURCE_GAP,
            f"missing source evidence note: {area}",
        )
        basis = entry["verdict_basis"]
        require(
            basis.get("rule") in ["accepts_replacement", "library_spread_noise"],
            f"missing verdict basis: {area}",
        )
        if basis["rule"] == "library_spread_noise":
            require(bool(basis.get("blocked_cases")), f"missing noise table: {area}")
            for row in basis["blocked_cases"]:
                values = [
                    row[k]
                    for k in [
                        "library_spread",
                        "spread_bound",
                        "smaller_pair_speedup",
                        "required_speedup",
                    ]
                ]
                require(
                    all(finite_positive(v) for v in values)
                    and row["library_spread"] > row["spread_bound"]
                    and row["required_speedup"]
                    == 1 + 3 * max(row["library_spread"], row["spread_bound"])
                    and row["smaller_pair_speedup"] >= row["required_speedup"],
                    f"invalid noise verdict: {area}",
                )


def main() -> None:
    """Expose CPU-only check and record-acceptance commands"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check")
    accept_parser = commands.add_parser("accept")
    accept_parser.add_argument("record", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "accept":
            entry = accept(args.root, args.record)
            print(f"accepted {args.record.name}: {entry['verdict_basis']['rule']}")
            print(f"source evidence gap: {SOURCE_GAP}")
        else:
            check(args.root)
            print("production CUDA qualification manifest matches")
    except (QualificationError, OSError, ValueError, KeyError, TypeError) as error:
        print(f"qualification refused: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
