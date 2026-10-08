"""Retain the projection-only cuBLAS baseline before the final lock."""

import argparse
import json
import shutil
from pathlib import Path

from gates import TOOLS, Rejected, sanitizer
from lock import ROOT
from assets import resolve, retain
from qualify import GPU_LOCK, missing_projection_markers, sha


def evidence(
    row: dict, result: dict, tier: str, result_path: Path, destination: Path
) -> tuple[Path, dict, dict]:
    """A cached proof must match the existing owner record, without new claims."""
    if row.get("source") is None:
        source = result_path.with_suffix("") / tier / Path(row["path"]).name
        commands = [step for step in result["commands"] if step["log"] == row["path"]]
        if len(commands) != 1:
            raise Rejected("baseline command evidence missing")
        provenance = {
            "source_result_sha256": sha(result_path),
            "source_lock_digest": result["lock_digest"],
        }
        return source, commands[0], provenance
    if row["source"] != "owner-locked library baseline":
        raise Rejected("unknown cached proof source")
    prior = json.loads((destination / f"{result['target']}-{tier}.json").read_text())
    data = result["tiers"][tier]
    identity = {
        "control_archive_sha256": data["control"]["archive_sha256"],
        "control_toolchain": data["control"]["toolchain"],
        "device_sm": data["device_sm"],
        "verified_inputs": data["verified_inputs"],
        "tool_fingerprint": data["sanitizer_fingerprint"],
    }
    if any(prior.get(key) != value for key, value in identity.items()):
        raise Rejected("cached proof identity differs from the owner record")
    original = next(item for item in prior["tools"] if item["tool"] == row["tool"])
    if any(row.get(key) != value for key, value in original.items()):
        raise Rejected("cached proof differs from the owner record")
    source = destination / original["log"]
    provenance = {
        key: original[key] for key in ("source_result_sha256", "source_lock_digest")
    }
    return (
        source,
        {"argv": original["command"], "returncode": original["returncode"]},
        provenance,
    )


def freeze(result_path: Path) -> None:
    """Validate live evidence and copy its exact bytes into the protected harness."""
    result = json.loads(result_path.read_text())
    if result["implementation"] != "Library" or result["status"] != "passed":
        raise Rejected("baseline requires a completed Library qualification")
    target = result["target"]
    if target != "lstm":
        raise Rejected("unknown baseline target")
    destination = ROOT / "tests/cuda_qualify/baselines"
    destination.mkdir(exist_ok=True)
    for tier, data in result["tiers"].items():
        if tier not in ("sm75", "sm80", "sm90", "sm120"):
            raise Rejected("unknown baseline tier")
        if data["control"]["archive_sha256"] != sha(
            resolve("tests/cuda_qualify/control.tar.gz")
        ):
            raise Rejected("baseline control differs from the frozen Library")
        records = [
            row
            for row in data["sanitizer"]
            if row["implementation"] == "ProjectionBaseline"
        ]
        if len(records) != len(TOOLS) or {row["tool"] for row in records} != set(TOOLS):
            raise Rejected("baseline requires every sanitizer tool exactly once")
        retained = []
        for row in records:
            source, command, provenance = evidence(
                row, result, tier, result_path, destination
            )
            if (
                row["implementation"] != "ProjectionBaseline"
                or row["scope"] != "LSTM input projections only"
            ):
                raise Rejected("baseline must check only LSTM input projections")
            if sha(source) != row["sha256"]:
                raise Rejected("baseline log hash mismatch")
            text = source.read_text()
            sanitizer(row["tool"], row["returncode"], text)
            if "1 passed; 0 failed" not in text:
                raise Rejected("baseline driver did not finish")
            if missing_projection_markers(text):
                raise Rejected("baseline omitted an input projection shape")
            if command["argv"][:2] != ["flock", GPU_LOCK] or command["returncode"] != 0:
                raise Rejected("baseline command was not a successful locked run")
            if any(
                flag in command["argv"]
                for flag in ("--kernel-name", "--launch-count", "--launch-skip")
            ):
                raise Rejected("filtered Library baseline refused")
            name = f"{target}-{tier}-{row['tool']}.log"
            if source.resolve() != (destination / name).resolve():
                shutil.copyfile(source, destination / name)
            retained.append(
                {
                    "tool": row["tool"],
                    "log": name,
                    "sha256": row["sha256"],
                    "returncode": 0,
                    "command": command["argv"],
                    "scope": "LSTM input projections only",
                    **provenance,
                }
            )
        record = {
            "target": target,
            "tier": tier,
            "scope": "LSTM input projections only",
            "device_sm": data["device_sm"],
            "control_archive_sha256": data["control"]["archive_sha256"],
            "control_driver_sha256": data["control"]["driver_sha256"],
            "control_toolchain": data["control"]["toolchain"],
            "verified_inputs": data["verified_inputs"],
            "tool_fingerprint": data["sanitizer_fingerprint"],
            "source_result_sha256": sha(result_path),
            "source_lock_digest": result["lock_digest"],
            "tools": retained,
        }
        (destination / f"{target}-{tier}.json").write_text(
            json.dumps(record, indent=2) + "\n"
        )
    retain(
        f"tests/cuda_qualify/baselines/{target}-source-{sha(result_path)}.json",
        result_path.read_bytes(),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="+")
    for path in parser.parse_args().results:
        freeze(path)
