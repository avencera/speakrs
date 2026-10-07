"""Qualify one candidate implementation against a frozen Library control."""

import argparse
import hashlib
import json
import os
import re
import secrets
import sqlite3
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

# re-execution uses new bytecode storage for the direct entry point too
if __name__ == "__main__" and os.environ.get("SPEAKRS_QUALIFY_CACHE_CHILD") != "1":
    with tempfile.TemporaryDirectory(prefix="speakrs-qualify-cache-") as cache:
        child_env = dict(os.environ, SPEAKRS_QUALIFY_CACHE_CHILD="1")
        child = subprocess.run(
            [
                sys.executable,
                "-X",
                f"pycache_prefix={cache}",
                str(Path(__file__).resolve()),
                *sys.argv[1:],
            ],
            env=child_env,
            check=False,
        )
    raise SystemExit(child.returncode)

from gates import (
    TOOLS,
    Blocked,
    Error,
    ParityRejected,
    TruthRejected,
    StageRejected,
    StageBlocked,
    Rejected,
    Timing,
    embedding_parity,
    layer_parity,
    finite,
    ratio,
    sanitizer,
    segmentation_parity,
    speed,
    spread_bound,
    stage_speed,
    tf32_band,
    tf32_truth,
)
import artifacts
from lock import ROOT, LockError, inventory, verify
from assets import resolve
from records import check_table, shipped_files
from verdict import evaluate_checks, noise_timing
from domains import MODEL, collection
from parse_trace import AllowList, attribute, window_kernels
from ptx import shared_initialization
from scan import cargo_home, scan

WORKSPACE = Path("/workspace")
# each task runs from /workspace/<task>/tree with its own target and results
BOX = ROOT.parent
GPU_LOCK = "/workspace/gpu-bench.lock"
PROFILE_TEST = "inference::cuda::test_support::qualify::qualification_driver"
CASES = MODEL.cases
MODES = ("fp32", "tf32")
BATCHES = MODEL.tested
FRAMES = 589
PROJECTION_COLUMNS = (128, 256, 384, 512)
# the candidate PTX area of each target
AREAS = {
    "resnet": "resnet",
    "lstm": "lstm",
    "sincnet": "sincnet",
    "fbankdft": "fbankdft",
}


@dataclass(frozen=True)
class Gate:
    """The exact check a planted fault must fail, and the reason it must give."""

    family: str
    reason: str
    rows: str | None = None

    def matches(self, item: dict) -> bool:
        """Match the tier-stripped check name and the reason, never a prefix only."""
        name = item["check"].split("/", 1)[-1]
        if item.get("blocked") or self.reason not in item.get("reason", ""):
            return False
        if self.family in (
            "profile",
            "determinism:fixed_reduction_order",
            "ptx:shared_initialization",
        ):
            return name == self.family
        if not name.startswith(self.family + ":"):
            return False
        if self.rows is None:
            return True
        cases = item.get("cases", [])
        failing = set(item.get("failing_cases", []))
        if self.rows == "non_b32":
            non_b32 = {case for case in cases if "/b32/" not in case}
            return bool(non_b32) and non_b32 <= failing and not (failing - non_b32)
        partial = {case for case in cases if _batch(case) % 32}
        return bool(partial) and partial <= failing


def _batch(case: str) -> int:
    return int(case.split("/")[2].removeprefix("b"))


# each planted fault proves one gate; a mutant counts as caught only on that gate
MUTANT_GATES = {
    "Precision": Gate("layer", "layer parity"),
    "Shape": Gate("layer", "layer parity", "non_b32"),
    "Fallback": Gate("profile", "forbidden library kernels"),
    "FirstUseFallback": Gate("profile", "forbidden library kernels"),
    "Tail": Gate("layer", "layer parity", "partial"),
    "Atomic": Gate(
        "determinism:fixed_reduction_order",
        "floating-point atomic in launched custom entry",
    ),
    "StageSlow": Gate("paired_stage", "paired stage: stage regression"),
    "StageTail": Gate(
        "paired_stage", "paired stage: non-inferiority margin not established"
    ),
    "StageAccuracy": Gate("stage_truth", "candidate less accurate than Library TF32"),
    "Slow": Gate("speed", "candidate slower than the faster Library process"),
    "PhaseCheat": Gate("timing_output", "final output differs"),
    "Unscoped": Gate("profile", "outside a candidate or library range"),
    "Unlisted": Gate("profile", "not on the loaded PTX allow-list"),
    "Lookup": Gate("secret", "differs from Library on a fresh input"),
    "UninitShared": Gate("ptx:shared_initialization", "shared load"),
}
MUTANTS = tuple(MUTANT_GATES)
PHASES = ("numeric", "timing", "paired", "profile", "sanitize")
# a mutant proves only its intended check, so it runs the phases that check needs;
# Library controls and candidates run every phase
MUTANT_PHASES = {
    "StageSlow": ("numeric", "timing", "paired"),
    "StageTail": ("numeric", "timing", "paired"),
    "StageAccuracy": ("numeric",),
    "Precision": ("numeric",),
    "Shape": ("numeric",),
    "Tail": ("numeric",),
    "Fallback": ("numeric", "profile"),
    "FirstUseFallback": ("numeric", "profile"),
    "Atomic": ("numeric", "profile"),
    "Unscoped": ("numeric", "profile"),
    "Unlisted": ("numeric", "profile"),
    "Slow": ("numeric", "timing"),
    "PhaseCheat": ("numeric", "timing"),
    "Lookup": ("numeric",),
    "UninitShared": ("numeric",),
}
# the unmutated pinned control runs the same protocol but grants no acceptance
CONTROL_PHASES = {"StageTailControl": MUTANT_PHASES["StageTail"]}
EXIT_CODES = {"passed": 0, "rejected": 1, "blocked": 3, "escaped": 4}


def layers(target: str) -> tuple[str, ...]:
    """Return the fixed layer inventory, independent of candidate declarations."""
    if target == "resnet":
        return tuple(
            f"resnet.layer{stage}.{block}.conv{conv}"
            for stage, count in [(1, 3), (2, 4)]
            for block in range(count)
            for conv in (1, 2)
        )
    if target == "lstm":
        return ("lstm.stack",)
    if target == "sincnet":
        return ("sincnet.conv0.abs_pool",)
    if target == "fbankdft":
        return ("fbank.dft",)
    raise Rejected("unknown target")


def target_batches(target: str) -> tuple[int, ...]:
    """Use the batch domain owned by the selected boundary."""
    return collection(target).tested


def target_cases(target: str) -> tuple[tuple[str, int], ...]:
    """Retain the existing cases and select only valid filterbank batch plans."""
    return collection(target).cases


def case_ids(target: str = "resnet") -> list[str]:
    """Every math mode and case, in a fixed order."""
    return [
        f"{mode}/{case}/b{batch}"
        for mode in MODES
        for case, batch in target_cases(target)
    ]


def sha(path: Path) -> str:
    """Hash one retained evidence file without loading it all at once."""
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def command(argv: list[str], env: dict[str, str], path: Path, steps: list[dict]) -> int:
    """Record the exact argv, exit status and log bytes; never use a shell."""
    locked = argv[:2] == ["flock", GPU_LOCK]
    executable = Path(argv[4] if argv[0] == "timeout" else argv[0])
    touches_gpu = (
        executable.is_relative_to(BOX / "target")
        or executable.name == "compute-sanitizer"
        or (executable.name == "nsys" and "profile" in argv[1:])
    )
    child_owned = (
        not locked
        and executable.is_relative_to(BOX / "target")
        and argv[1:] == ["--exact", PROFILE_TEST, "--ignored", "--nocapture"]
        and env.get("SPEAKRS_QUALIFY_PHASE") == "numeric"
        and env.get("SPEAKRS_QUALIFY_LOCK_OWNER") == "child"
    )
    if touches_gpu and not (locked or child_owned):
        raise Rejected("GPU process refused: the shared GPU lock is required")
    start = time.monotonic()
    with path.open("wb") as log:
        process = subprocess.run(
            argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False
        )
    steps.append(
        {
            "argv": argv,
            "phase": env.get("SPEAKRS_QUALIFY_PHASE", "build"),
            "wall_seconds": time.monotonic() - start,
            "returncode": process.returncode,
            "log": str(path),
            "sha256": sha(path),
            "gpu_lock": GPU_LOCK if locked or child_owned else None,
            "gpu_lock_owner": "child" if child_owned else "parent" if locked else None,
        }
    )
    return process.returncode


def gpu_command(
    argv: list[str], env: dict[str, str], path: Path, steps: list[dict]
) -> int:
    """Hold the shared lock for the complete GPU process and its children."""
    child = dict(env, SPEAKRS_QUALIFY_LOCK_OWNER="parent")
    return command(["flock", GPU_LOCK, *argv], child, path, steps)


def clean_environment() -> dict[str, str]:
    """Drop variables that change how Cargo or rustc build the candidate."""
    refused = (
        "RUSTC_",
        "CARGO_BUILD_",
        "CARGO_TARGET_",
        "CARGO_PROFILE_",
        "CARGO_UNSTABLE_",
    )
    return {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(refused)
        and key
        not in (
            "RUSTC",
            "RUSTDOC",
            "RUSTUP_TOOLCHAIN",
            "CARGO",
            "RUSTFLAGS",
            "RUSTDOCFLAGS",
            "CARGO_ENCODED_RUSTFLAGS",
            "CARGO_INCREMENTAL",
            "SPEAKRS_QUALIFY_NVTX",
            "SPEAKRS_CUDA_FORCE_PTX_JIT",
        )
    } | {"CARGO_PROFILE_DEV_DEBUG": "0", "CARGO_PROFILE_TEST_DEBUG": "0"}


def build(
    env: dict[str, str],
    directory: Path,
    steps: list[dict],
    manifest: Path | None = None,
) -> Path:
    """Build only this task's release test binary, outside the GPU lock."""
    log = directory / ("build.jsonl" if manifest is None else "control-build.jsonl")
    args = [
        "cargo",
        "test",
        "--release",
        "-p",
        "speakrs",
        "--no-default-features",
        "--features",
        "cuda",
        "--lib",
        "--no-run",
        "--message-format=json",
    ]
    if manifest is not None:
        args.extend(["--manifest-path", str(manifest)])
    if command(args, env, log, steps):
        raise Rejected("qualification driver build failed")
    artifacts = []
    for line in log.read_text().splitlines():
        try:
            artifact = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            artifact.get("reason") == "compiler-artifact"
            and artifact.get("target", {}).get("name") == "speakrs"
            and artifact.get("profile", {}).get("test")
            and artifact.get("executable")
        ):
            artifacts.append(Path(artifact["executable"]))
    if (
        len(artifacts) != 1
        or not artifacts[0].is_file()
        or not artifacts[0].is_relative_to(BOX / "target")
    ):
        raise Rejected("missing or ambiguous task-owned qualification executable")
    return artifacts[0]


def toolchain(env: dict[str, str]) -> str:
    """The exact compiler that builds both binaries, as `rustc -vV` reports it."""
    process = subprocess.run(
        ["rustc", "-vV"], cwd=ROOT, env=env, capture_output=True, text=True, check=False
    )
    if process.returncode or not process.stdout.strip():
        raise Rejected("rustc -vV failed")
    return process.stdout.strip()


def build_library(env: dict[str, str], directory: Path, steps: list[dict]) -> None:
    """Build the non-test library from the same tree, so test-only code cannot pass."""
    args = [
        "cargo",
        "build",
        "--release",
        "-p",
        "speakrs",
        "--no-default-features",
        "--features",
        "cuda",
        "--lib",
    ]
    if command(args, env, directory / "build-library.log", steps):
        raise Rejected("non-test library build failed")


def control_target(archive_digest: str, tier: str) -> Path:
    """Separate control artifacts by source bytes, not zero archive timestamps."""
    if not re.fullmatch(r"[0-9a-f]{64}", archive_digest):
        raise Rejected("invalid control archive digest")
    return BOX / "target/control" / archive_digest / tier


def verify_inputs(target: str) -> dict[str, str]:
    """Require the owner's fixed model and reference bytes, before and after runs."""
    manifest = json.loads((ROOT / "tests/cuda_qualify/ASSETS.json").read_text())
    if manifest.get("schema") != 1:
        raise Rejected("unsupported input manifest")
    model = "wespeaker-multimask-tail" if target == "resnet" else "segmentation-3.0"
    required = {str(Path("/workspace/models-native") / f"{model}.safetensors")}
    for suffix, case in (
        ("", "test_first_b1"),
        ("", "test_last_partial_b1"),
        ("", "test_short_partial_b1"),
        ("-b32", "test_and_short_b32"),
    ):
        required.add(
            str(Path("/workspace/ref") / f"{model}{suffix}" / f"{case}.safetensors")
        )
    if target == "resnet":
        required.add(
            "/workspace/ref/wespeaker-fbank-b32/test_and_short_b32.safetensors"
        )
    if target == "fbankdft":
        required = {"/workspace/ref/wespeaker-fbank-b32/test_and_short_b32.safetensors"}
    files = manifest["files"]
    if not required <= files.keys():
        raise Rejected("input manifest omits a required asset")
    actual = {name: sha(Path(name)) for name in sorted(required)}
    if any(actual[name] != files[name] for name in required):
        raise Rejected("model or reference differs from the locked input manifest")
    return actual


@dataclass(frozen=True)
class Coverage:
    """The (layer, batch, math) triples an implementation declares: the union of its
    product entries."""

    triples: frozenset[tuple[str, int, str]]

    @classmethod
    def product(cls, layers, batches, modes) -> "Coverage":
        return cls(
            frozenset(
                (layer, batch, mode)
                for layer in layers
                for batch in batches
                for mode in modes
            )
        )

    @property
    def layers(self) -> frozenset[str]:
        return frozenset(layer for layer, _, _ in self.triples)

    @property
    def batches(self) -> frozenset[int]:
        return frozenset(batch for _, batch, _ in self.triples)

    def declared(self, layer: str, batch: int, mode: str) -> bool:
        return (layer, batch, mode) in self.triples

    def layers_at(self, batch: int, mode: str) -> list[str]:
        return sorted(
            layer for layer, at, math in self.triples if at == batch and math == mode
        )

    def record(self) -> dict:
        return {
            "layers": sorted(self.layers),
            "batches": sorted(self.batches),
            "triples": [list(triple) for triple in sorted(self.triples)],
        }


def parse_coverage(raw: dict, target: str) -> Coverage:
    """Refuse declarations the harness cannot qualify, such as an untested batch."""
    inventory = layers(target)
    triples: set[tuple[str, int, str]] = set()
    for entry in raw["entries"]:
        names = inventory if entry["layers"] == "all" else tuple(entry["layers"])
        batches = (
            target_batches(target)
            if entry["batches"] == "all"
            else tuple(entry["batches"])
        )
        modes = MODES if entry["maths"] == "all" else tuple(entry["maths"])
        if not set(names) <= set(inventory):
            raise Rejected(
                f"coverage names unknown layers: {sorted(set(names) - set(inventory))}"
            )
        if not set(batches) <= set(target_batches(target)):
            raise Rejected(
                f"coverage names batches the harness cannot test: {sorted(set(batches) - set(target_batches(target)))}"
            )
        if not set(modes) <= set(MODES):
            raise Rejected("coverage names an unknown math mode")
        triples |= Coverage.product(names, batches, modes).triples
    return Coverage(frozenset(triples))


def split_key(key: str) -> tuple[str, str, int, str, bool]:
    """mode, case, batch, boundary, switched"""
    switched = key.endswith("/switched")
    mode, case, batch, boundary = key.removesuffix("/switched").split("/", 3)
    return mode, case, int(batch.removeprefix("b")), boundary, switched


def expected_ids(target: str, phase: str, mode: str) -> set[str]:
    """The rows one per-mode process must produce, without trusting device output."""
    base = {
        f"{mode}/{case}/b{batch}/{layer}"
        for case, batch in target_cases(target)
        for layer in (*layers(target), "stage")
    }
    if phase == "numeric":
        return base | {f"{key}/switched" for key in base}
    return base


def draw_length(case: str, layer: str) -> int:
    """Bind draw lengths to the locked model geometry, not a reported CPU section."""
    match = re.fullmatch(
        r"tf32/(first|last|short|mixed)/b([1-9][0-9]*)/stage(/switched)?", case
    )
    if match is None:
        raise Rejected("GPU lock: invalid TF32 draw case")
    batch = int(match[2])
    if layer == "fbank.dft":
        if batch not in target_batches("fbankdft"):
            raise Rejected("GPU lock: invalid TF32 draw batch")
        return batch * 998 * 80
    if batch not in BATCHES:
        raise Rejected("GPU lock: invalid TF32 draw batch")
    if layer == "lstm.stack":
        return batch * FRAMES * 256
    if layer == "sincnet.conv0.abs_pool":
        return batch * 80 * ((160_000 - 251) // 10 + 1)
    conv = re.fullmatch(r"resnet.layer([12])\.([0-6])\.conv([12])", layer)
    if conv is None or int(conv[2]) >= (3 if conv[1] == "1" else 4):
        raise Rejected("GPU lock: invalid TF32 draw layer")
    return batch * (32 * 80 * 998 if conv[1] == "1" else 64 * 40 * 499)


def validate_gpu_ownership(process: dict, phase: str) -> None:
    """Require completed evidence from the selected GPU lock owner."""
    evidence = process.get("gpu_lock")
    owner = "child" if phase == "numeric" else "parent"
    if not isinstance(evidence, dict) or (
        evidence.get("path") != GPU_LOCK or evidence.get("owner") != owner
    ):
        raise Rejected("GPU lock: missing or inconsistent ownership evidence")
    sections = evidence.get("cpu_sections")
    if not isinstance(sections, list):
        raise Rejected("GPU lock: missing CPU ownership evidence")
    if owner == "parent" and sections:
        raise Rejected("GPU lock: parent-owned process ran CPU qualification work")
    for section in sections:
        if (
            not isinstance(section, dict)
            or section.get("work") not in ("f64", "tf32_draws")
            or section.get("locked") is not False
        ):
            raise Rejected("GPU lock: CPU qualification work held the shared lock")
        fields = (
            {"work", "locked", "case"}
            if section["work"] == "f64"
            else {"work", "locked", "case", "layer", "seed", "length"}
        )
        if set(section) != fields or (
            section["work"] == "f64"
            and (not isinstance(section["case"], str) or not section["case"])
        ):
            raise Rejected("GPU lock: missing or invalid CPU case evidence")
    count = evidence.get("gpu_sections")
    if type(count) is not int or count != len(sections) + 1:
        raise Rejected("GPU lock: incomplete GPU ownership evidence")
    if owner == "child":
        rows = process.get("rows", [])
        prepared = [section["case"] for section in sections if section["work"] == "f64"]
        if len(set(prepared)) != len(prepared):
            raise Rejected("GPU lock: duplicate prepared f64 case")
        emitted = [row.get("id") for row in rows if row.get("secret") is True]
        if any(not isinstance(case, str) or not case for case in emitted) or not set(
            emitted
        ) <= set(prepared):
            raise Rejected("GPU lock: missing unlocked f64 truth work")
        expected = Counter()
        for row in rows:
            if not row.get("id", "").endswith("/band") or not row.get("layers"):
                continue
            if row.get("seeds") != [11, 23, 37, 41, 53, 67, 79, 97]:
                raise Rejected("GPU lock: invalid TF32 draw seeds")
            case = row["id"].removesuffix("/band")
            for layer in row["layers"]:
                length = draw_length(case, layer)
                for seed in row["seeds"]:
                    expected[(case, layer, seed, length)] += 1
        actual = Counter()
        for section in sections:
            if section["work"] != "tf32_draws":
                continue
            if (
                not isinstance(section["case"], str)
                or not isinstance(section["layer"], str)
                or type(section["seed"]) is not int
                or type(section["length"]) is not int
            ):
                raise Rejected("GPU lock: invalid TF32 draw binding")
            actual[
                (section["case"], section["layer"], section["seed"], section["length"])
            ] += 1
        if actual != expected:
            raise Rejected("GPU lock: missing or mismatched unlocked TF32 draw work")


def driver(
    binary: Path,
    env: dict,
    directory: Path,
    steps: list,
    implementation: str,
    phase: str,
    label: str,
    extra: dict | None = None,
) -> dict:
    """Run a fresh driver with explicit GPU ownership and require its evidence."""
    output = directory / f"{label}.json"
    child = dict(
        env,
        SPEAKRS_QUALIFY_IMPL=implementation,
        SPEAKRS_QUALIFY_PHASE=phase,
        SPEAKRS_QUALIFY_OUTPUT=str(output),
        **(extra or {}),
    )
    log = directory / f"{label}.log"
    argv = [str(binary), "--exact", PROFILE_TEST, "--ignored", "--nocapture"]
    if phase == "numeric":
        child["SPEAKRS_QUALIFY_LOCK_OWNER"] = "child"
        code = command(argv, child, log, steps)
    else:
        code = gpu_command(argv, child, log, steps)
    if code:
        raise Rejected(f"driver failed: {label}")
    if "1 passed; 0 failed" not in log.read_text() or not output.is_file():
        raise Rejected(f"driver missing completed GPU evidence: {label}")
    data = json.loads(output.read_text())
    validate_gpu_ownership(data, phase)
    if phase == "numeric":
        steps[-1]["cpu_work_wall_seconds"] = data["gpu_lock"].get(
            "cpu_wall_seconds", {}
        )
        steps[-1]["cpu_byte_identity"] = data["gpu_lock"].get("cpu_byte_identity", [])
    if (
        data["implementation"] != implementation
        or data["phase"] != phase
        or data["target"] != env["SPEAKRS_QUALIFY_TARGET"]
    ):
        raise Rejected("driver identity mismatch")
    if phase == "coverage":
        return data
    mode = child.get("SPEAKRS_QUALIFY_MODE")
    if phase == "paired" and (mode not in MODES or data.get("mode") != mode):
        raise Rejected("driver math mode mismatch")
    if phase in ("numeric", "timing"):
        if mode not in MODES or data.get("mode") != mode:
            raise Rejected("driver math mode mismatch")
        ids = [
            row["id"]
            for row in data["rows"]
            if not row["id"].endswith("/band") and not row.get("secret")
        ]
        if len(ids) != len(set(ids)) or set(ids) != expected_ids(
            data["target"], phase, mode
        ):
            raise Rejected("driver missing or duplicate case coverage")
        if phase == "timing" and not all(row.get("cuda_graph") for row in data["rows"]):
            raise Rejected("graph replay timing is required")
    if data.get("tier") != env.get("SPEAKRS_CUDA_PTX_TIER"):
        raise Rejected("driver PTX tier mismatch")
    validate_target(
        data,
        env["SPEAKRS_CUDA_PTX_TIER"],
        candidate_area=AREAS[env["SPEAKRS_QUALIFY_TARGET"]]
        if implementation != "Library"
        else None,
    )
    data["evidence_sha256"] = sha(output)
    return data


def validate_target(process: dict, tier: str, *, candidate_area: str | None) -> dict:
    """Require the selected candidate tier and retain every area's loaded tier.

    The trusted caller identifies the candidate area. Library and glue areas use
    the production loader's tier limit, not a forced variant at that limit.
    """
    device = artifacts.device(process.get("device"))
    required = (
        "name",
        "compute_capability",
        "sm_count",
        "l2_bytes",
        "driver_api_version",
        "driver_version",
        "cuda_version",
        "cudnn_version",
        "cublas_version",
    )
    if any(not device.get(key) for key in required):
        raise Rejected("target: missing device or library version evidence")
    if device["compute_capability"] != process.get("device_sm"):
        raise Rejected("target: inconsistent compute capability")
    minimum = {"sm75": (7, 5), "sm80": (8, 0), "sm90": (9, 0), "sm120": (12, 0)}[tier]
    capability = artifacts.capability(device["compute_capability"])
    if capability < minimum:
        raise Rejected("target: device cannot execute requested tier")
    modules = process.get("loaded_modules", [])
    if not modules:
        raise Rejected("target: missing loaded area evidence")
    for module in modules:
        pin = module.get("artifact", {})
        if not isinstance(pin, dict):
            raise Rejected("artifact: missing loaded artifact")
        artifacts.key(
            {name: pin[name] for name in ("kind", "arch", "sha256") if name in pin},
            device_capability=device["compute_capability"],
        )
        if module["area"] == candidate_area and module["tier"] != tier:
            raise Rejected(
                f"target: loaded tier differs from requested tier: {module['area']} {module['tier']} != {tier}"
            )
    if process.get("phase") in ("timing", "paired"):
        clocks = process.get("observed_sm_clock", {})
        if (
            not clocks
            or clocks.get("samples", 0) < 1
            or not 0 < clocks.get("min_mhz", 0) <= clocks.get("max_mhz", 0)
        ):
            raise Rejected("target: missing observed SM clocks during timing")
    return {"device": device, "loaded_tiers": {m["area"]: m["tier"] for m in modules}}


def fail(error: Rejected):
    """Raise `error`; lets a recorded failure share the `check` bookkeeping."""
    raise error


def check(result: dict, name: str, operation) -> None:
    """Retain every failure; a later successful check cannot erase it."""
    try:
        evidence = operation()
        result["checks"].append({"check": name, "passed": True, "evidence": evidence})
    except StageBlocked as error:
        result["checks"].append(
            {
                "check": name,
                "passed": False,
                "blocked": True,
                "reason": str(error),
                "evidence": error.evidence,
            }
        )
    except Blocked as error:
        result["checks"].append(
            {"check": name, "passed": False, "blocked": True, "reason": str(error)}
        )
    except ParityRejected as error:
        result["checks"].append(
            {
                "check": name,
                "passed": False,
                "reason": str(error),
                "failing_cases": error.failing,
                "cases": error.cases,
            }
        )
    except (TruthRejected, StageRejected) as error:
        result["checks"].append(
            {
                "check": name,
                "passed": False,
                "reason": str(error),
                "evidence": error.evidence,
            }
        )
    except Rejected as error:
        result["checks"].append({"check": name, "passed": False, "reason": str(error)})


ENTRY = re.compile(r"\.entry\s+([A-Za-z_$][A-Za-z_0-9$]*)\s*\(")


def ptx_files(root: Path = ROOT) -> dict[str, Path]:
    """Every committed PTX file, by SHA-256, including any in subdirectories."""
    files = {}
    for directory in (
        root / "src/inference/cuda/ptx",
        root / "tests/cuda_qualify/device",
    ):
        for path in directory.rglob("*.ptx"):
            files[sha(path)] = path
    return files


def verify_modules(modules: list[dict], root: Path = ROOT) -> tuple[AllowList, dict]:
    """Bind the allow-list to the exact bytes the process loaded and to committed files.

    A loaded module must be byte-identical to a committed PTX file whose name matches
    its area, and the entries the process parsed must match the file. PTX outside the
    `<area>.<tier>.ptx` layout is refused.
    """
    ptx_dir = root / "src/inference/cuda/ptx"
    for path in ptx_dir.rglob("*"):
        if path.is_dir() or path.parent != ptx_dir:
            raise Rejected(f"PTX outside the area layout: {path.relative_to(root)}")
    files = ptx_files(root)
    entries, candidate, evidence = set(), set(), []
    for module in modules:
        path = files.get(module["sha256"])
        if path is None:
            raise Rejected(
                f"loaded PTX bytes match no committed file: {module['area']}"
            )
        names = ENTRY.findall(path.read_text())
        if names != module["entries"]:
            raise Rejected(f"loaded entries differ from {path.name}")
        artifacts.module(module, root)
        harness = path.parent.name == "device"
        if not harness and not path.name.startswith(
            f"{module['area']}.{module['tier']}."
        ):
            raise Rejected(f"loaded module {module['area']} came from {path.name}")
        entries.update(names)
        if module["area"] in AREAS.values():
            candidate.update(names)
        evidence.append({**module, "path": str(path.relative_to(root))})
    if not entries:
        raise Rejected("no loaded custom PTX entries")
    return AllowList(frozenset(entries), frozenset(candidate)), {"modules": evidence}


def sanitizer_fingerprint() -> dict[str, str]:
    """Bind known library findings to the exact installed tool and CUDA libraries."""
    paths = {Path("/usr/local/cuda-13.0/compute-sanitizer/compute-sanitizer")}
    for directory in (Path("/usr/local/cuda-13.0/compute-sanitizer"),):
        paths.update(
            path.resolve() for path in directory.rglob("*.so*") if path.is_file()
        )
    for directory in (
        Path("/usr/lib/x86_64-linux-gnu"),
        Path("/usr/local/cuda-13.0/targets/x86_64-linux/lib"),
        Path("/usr/local/cuda-12.8/targets/x86_64-linux/lib"),
    ):
        for prefix in ("libcuda", "libcublas", "libcudnn"):
            paths.update(
                path.resolve()
                for path in directory.glob(f"{prefix}*.so*")
                if path.is_file()
            )
    return {str(path): sha(path) for path in sorted(paths)}


def sanitizer_argv(
    binary: Path, tool: str, include: list[str] | None = None
) -> list[str]:
    """Bound each tool to 20 minutes after lock acquisition, without launch sampling.

    `include` is the allow-list of loaded entry names, matched exactly on the mangled
    name; None checks every kernel, which only the library projection baseline uses.
    """
    argv = [
        "/usr/local/cuda-13.0/bin/compute-sanitizer",
        "--tool",
        tool,
        # drain records at bounded intervals without changing timing or profile runs
        "--force-synchronization-limit",
        "16",
        "--error-exitcode",
        "86",
        str(binary),
        "--exact",
        PROFILE_TEST,
        "--ignored",
        "--nocapture",
    ]
    if tool == "racecheck":
        argv[1:1] = [
            "--racecheck-deadlock-timeout",
            "60000",
            "--racecheck-continue-on-deadlock",
            "no",
        ]
    if include is not None:
        if not include:
            raise Rejected("no loaded custom entries for sanitizer coverage")
        for name in sorted(include):
            if not re.fullmatch(r"[A-Za-z_$][A-Za-z_0-9$]*", name):
                raise Rejected(f"unexpected entry name {name}")
            argv[1:1] = ["--kernel-name", f"kne={name}"]
    return ["timeout", "--signal=TERM", "--kill-after=10s", "1200", *argv]


CONTROLS = {
    "oob": ("memcheck", "qualify_oob", "Invalid __global__ write"),
    "race": ("racecheck", "qualify_race", "hazard"),
    "uninit": ("initcheck", "qualify_round", "Uninitialized __global__ memory read"),
}


def filter_proof(
    binary: Path,
    env: dict[str, str],
    directory: Path,
    steps: list[dict],
    control: str,
    include: list[str],
) -> dict:
    """Require a planted fault to survive the exact candidate include filter."""
    tool, kernel, fault = CONTROLS[control]
    log = directory / f"filter-proof-{control}.log"
    child = dict(
        env,
        SPEAKRS_QUALIFY_IMPL="FilterProof",
        SPEAKRS_QUALIFY_PHASE="filter_proof",
        SPEAKRS_QUALIFY_CONTROL=control,
    )
    code = gpu_command(sanitizer_argv(binary, tool, include), child, log, steps)
    text = log.read_text()
    pattern = (
        r"RACECHECK SUMMARY: (\d+) hazards?"
        if tool == "racecheck"
        else r"ERROR SUMMARY: (\d+) errors?"
    )
    summaries = re.findall(pattern, text)
    if (
        code != 86
        or kernel not in text
        or fault not in text
        or not summaries
        or not any(int(value) > 0 for value in summaries)
    ):
        raise Rejected(f"sanitizer include filter hid the planted {control} fault")
    return {
        "tool": tool,
        "returncode": code,
        "path": str(log),
        "sha256": sha(log),
        "expected_fault": f"{kernel}: {fault}",
        "passed": True,
    }


def sanitizer_batches(coverage: Coverage) -> list[int]:
    """b7 and b33 when every batch is declared, else every declared batch."""
    if coverage.batches == frozenset(BATCHES):
        return [7, 33]
    return sorted(coverage.batches)


def candidate_sanitizer(
    binary: Path,
    tool: str,
    env: dict[str, str],
    directory: Path,
    steps: list[dict],
    include: list[str],
    coverage: Coverage,
) -> tuple[dict, str]:
    """Retry a bounded ResNet timeout with a complete, disjoint layer/batch partition."""
    impl = env["SPEAKRS_QUALIFY_IMPL"]
    batches = sanitizer_batches(coverage)
    attempts = []

    def run(
        layer: str | None = None, batch: int | None = None
    ) -> tuple[int, str, set[str]]:
        label = f"{impl}-{tool}" if layer is None else f"{impl}-{tool}-{layer}-b{batch}"
        log = directory / f"{label}.log"
        output = directory / f"{label}.json"
        child = dict(
            env,
            SPEAKRS_QUALIFY_OUTPUT=str(output),
            SPEAKRS_QUALIFY_SANITIZE_BATCHES=",".join(map(str, batches)),
        )
        child.pop("SPEAKRS_QUALIFY_SANITIZER_LAYER", None)
        child.pop("SPEAKRS_QUALIFY_SANITIZER_BATCH", None)
        if layer is not None:
            child.update(
                SPEAKRS_QUALIFY_SANITIZER_LAYER=layer,
                SPEAKRS_QUALIFY_SANITIZER_BATCH=str(batch),
            )
        code = gpu_command(sanitizer_argv(binary, tool, include), child, log, steps)
        text = log.read_text()
        ids = set()
        if code == 0:
            sanitizer(tool, code, text)
            if not output.is_file() or "1 passed; 0 failed" not in text:
                raise Rejected("sanitizer missing completed driver evidence")
            data = json.loads(output.read_text())
            if (
                data["target"] != env["SPEAKRS_QUALIFY_TARGET"]
                or data["implementation"] != impl
                or data["phase"] != "sanitize"
                or data["tier"] != env["SPEAKRS_CUDA_PTX_TIER"]
            ):
                raise Rejected("sanitizer driver identity mismatch")
            ids = {row["id"] for row in data["rows"]}
            expected = {
                f"{mode}/{'first' if rows == 1 else 'mixed'}/b{rows}/{name}"
                for mode in MODES
                for rows in ([batch] if batch is not None else batches)
                for name in ([layer] if layer is not None else layers(data["target"]))
            }
            if (
                ids != expected
                or len(data["rows"]) != len(expected)
                or not all(row["sanitized"] for row in data["rows"])
            ):
                raise Rejected("sanitizer missing or duplicate coverage")
        attempts.append(
            {
                "layer": layer,
                "batch": batch,
                "returncode": code,
                "path": str(log),
                "sha256": sha(log),
                "output_sha256": sha(output) if output.is_file() else None,
                "timeout_seconds": 1200,
            }
        )
        return code, text, ids

    code, text, ids = run()
    split = code in (124, 137) and env["SPEAKRS_QUALIFY_TARGET"] == "resnet"
    if split:
        texts, codes, covered = [], [], set()
        declared = [name for name in layers("resnet") if name in coverage.layers]
        for layer in declared:
            for batch in batches:
                child_code, child_text, child_ids = run(layer, batch)
                if covered & child_ids:
                    raise Rejected("duplicate sanitizer retry coverage")
                covered.update(child_ids)
                codes.append(child_code)
                texts.append(child_text)
        code = next((value for value in codes if value != 0), 0)
        text = "\n".join(texts)
        if code == 0 and len(covered) != len(declared) * len(batches) * len(MODES):
            raise Rejected("incomplete sanitizer retry coverage")
    return {
        "implementation": impl,
        "tool": tool,
        "returncode": code,
        "path": attempts[0]["path"],
        "sha256": attempts[0]["sha256"],
        "attempts": attempts,
        "split_after_timeout": split,
        "scope": "loaded custom PTX entries only (exact include filter)",
        "batches": batches,
        "timeout_seconds": 1200,
    }, text


def mutant_gate(implementation: str, failed: list[dict]) -> dict:
    """Name the failed checks that match the mutant's exact gate and reason."""
    gate = MUTANT_GATES[implementation]
    caught = [item["check"] for item in failed if gate.matches(item)]
    return {
        "intended_check": gate.family,
        "intended_reason": gate.reason,
        "intended_rows": gate.rows,
        "caught_by": caught,
        "caught": bool(caught),
    }


def library_tool_record(baseline: dict | None, tool: str) -> dict | None:
    """A changed option invalidates only that tool, not unrelated completed runs."""
    if baseline is None:
        return None
    record = next(row for row in baseline["tools"] if row["tool"] == tool)
    argv = list(record["command"])
    if argv[:2] != ["flock", GPU_LOCK] or "--exact" not in argv:
        raise Rejected("locked library baseline command is not a locked driver run")
    binary_index = argv.index("--exact") - 1
    argv[binary_index] = "<control>"
    expected = ["flock", GPU_LOCK, *sanitizer_argv(Path("<control>"), tool)]
    return record if argv == expected else None


def known_library_baseline(
    target: str, tier: str, result: dict, device_sm: str
) -> dict | None:
    """Reuse only owner-locked, completed library reports with identical inputs."""
    if target != "lstm":
        return None
    path = ROOT / "tests/cuda_qualify/baselines" / f"{target}-{tier}.json"
    if not path.exists():
        return None
    baseline = json.loads(path.read_text())
    identity = {
        "target": target,
        "tier": tier,
        "scope": "LSTM input projections only",
        "device_sm": device_sm,
        "control_archive_sha256": result["control"]["archive_sha256"],
        # the binary's bytes depend on its build path; the archive and the exact
        # toolchain determine what it computes
        "control_toolchain": result["control"]["toolchain"],
        "verified_inputs": result["verified_inputs"],
        "tool_fingerprint": result["sanitizer_fingerprint"],
    }
    if any(baseline.get(key) != value for key, value in identity.items()):
        return None
    resolve(
        f"tests/cuda_qualify/baselines/lstm-source-{baseline['source_result_sha256']}.json"
    )
    if len(baseline["tools"]) != len(TOOLS) or {
        record["tool"] for record in baseline["tools"]
    } != set(TOOLS):
        raise Rejected("locked library baseline has incomplete tool coverage")
    for record in baseline["tools"]:
        log = path.parent / record["log"]
        if sha(log) != record["sha256"]:
            raise Rejected("locked library baseline log differs from its record")
        text = log.read_text()
        sanitizer(record["tool"], record["returncode"], text)
        if "1 passed; 0 failed" not in text:
            raise Rejected("locked library baseline driver did not complete")
        if missing_projection_markers(text):
            raise Rejected("locked library baseline omits a projection shape")
    return baseline


def projection_markers() -> list[str]:
    """The log line the baseline prints for every shape the helper can issue."""
    return [
        f"projection baseline mode={mode} batch={batch} m={batch * FRAMES} n={n} k={k}"
        for mode in MODES
        for k in (60, 256)
        for batch in BATCHES
        for n in PROJECTION_COLUMNS
    ]


def missing_projection_markers(text: str) -> list[str]:
    return [marker for marker in projection_markers() if marker not in text]


def stage_check(result: dict, key: str, row: dict, control: dict) -> None:
    """The strict stage rule: FP32 always, TF32 when no candidate layer runs in TF32."""
    a, b = row["first"], control["first"]
    if result["target"] == "resnet":
        check(
            result,
            f"stage:{key}",
            lambda: embedding_parity(a["minimum_cosine"], b["minimum_cosine"]),
        )
        return
    check(
        result,
        f"stage:{key}",
        lambda: segmentation_parity(
            Error(a["relative_l2"], a["max_abs"]),
            Error(b["relative_l2"], b["max_abs"]),
            a["argmax_flips"],
            b["argmax_flips"],
        ),
    )


def numeric(
    result: dict, library: list[dict], candidate: list[dict], coverage: Coverage
) -> None:
    """Apply fixed gates to actual layer and stage measurements."""
    baseline = {row["id"]: row for row in library}
    bands = {
        row["id"].removesuffix("/band"): row["metrics"]
        for row in library
        if row["id"].endswith("/band")
    }
    grouped = defaultdict(list)
    band_rows = []
    embedding = result["target"] == "resnet"
    secrets = [row for row in candidate if row.get("secret")]
    for row in candidate:
        if row.get("secret"):
            continue
        key = row["id"]
        control = baseline[key]
        mode, _, batch, boundary, _ = split_key(key)

        def repeat(row=row):
            if (
                not row["bitwise_equal"]
                or row["first"]["sha256"] != row["second"]["sha256"]
            ):
                raise Rejected("determinism: bitwise mismatch")

        check(result, f"determinism:{key}", repeat)
        if boundary == "stage":
            if not coverage.layers_at(batch, mode):

                def stage_identity(row=row, control=control):
                    if row["first"]["sha256"] != control["first"]["sha256"]:
                        raise Rejected(
                            "undeclared stage differs from the frozen Library output"
                        )

                check(result, f"library_dispatch:{key}", stage_identity)
                if mode == "tf32" and result.get("implementation") == "Library":
                    band_rows.append((key, row, control))
            elif mode == "tf32":
                band_rows.append((key, row, control))
            else:
                stage_check(result, key, row, control)
            continue
        declared = coverage.declared(boundary, batch, mode)
        if bool(row["declared"]) != declared:
            check(
                result,
                f"coverage:{key}",
                lambda: fail(
                    Rejected("dispatch ran a different implementation than declared")
                ),
            )
        if declared:
            a, b = row["first"], control["first"]
            grouped[mode, boundary].append(
                (
                    key,
                    Error(a["relative_l2"], a["max_abs"]),
                    Error(b["relative_l2"], b["max_abs"]),
                )
            )
            continue

        def library_dispatch(row=row, control=control):
            if row["first"]["sha256"] != control["first"]["sha256"]:
                raise Rejected("undeclared pair differs from the frozen Library output")

        check(result, f"library_dispatch:{key}", library_dispatch)

    for (mode, layer), cases in sorted(grouped.items()):
        check(result, f"layer:{mode}/{layer}", lambda cases=cases: layer_parity(cases))

    secret_coverage = (
        Coverage.product(
            layers(result["target"]), target_batches(result["target"]), MODES
        )
        if result.get("implementation") == "Library"
        else coverage
    )
    secret_checks(result, secrets, secret_coverage)
    if not band_rows:
        return
    bands_by_case = {}
    for key, row, control in band_rows:
        band_row = baseline.get(f"{key}/band")

        def truth_gate(row=row, control=control, band_row=band_row):
            if (
                band_row is None
                or row.get("truth") is None
                or control.get("truth") is None
            ):
                raise Rejected(
                    "TF32 truth: missing same-input FP32 truth or perturbation draws"
                )
            truth_hash = row.get("truth_sha256")
            if (
                not truth_hash
                or truth_hash != control.get("truth_sha256")
                or truth_hash != band_row.get("truth_sha256")
            ):
                raise Rejected(
                    "TF32 truth: FP32 Library truth differs between processes"
                )
            seeds = band_row.get("seeds", [])
            if len(seeds) < 8 or len(set(seeds)) != len(seeds):
                raise Rejected("TF32 truth: independent perturbation seeds required")
            if len(band_row.get("truth_draws", [])) != len(seeds):
                raise Rejected("TF32 truth: missing perturbation draw")
            return tf32_truth(row["truth"], control["truth"], band_row["truth_draws"])

        check(result, f"stage_truth:{key}", truth_gate)
        if key in bands:
            bands_by_case[key] = tf32_band(
                [{"metrics": control["first"], "band": bands[key]}], embedding
            )
            if not embedding:

                def stage_flips(row=row, control=control, band=bands_by_case[key]):
                    if (
                        row["first"]["argmax_flips"]
                        > control["first"]["argmax_flips"] + band["flips"]
                    ):
                        raise Rejected(
                            "stage: argmax flips above Library plus measured band"
                        )

                check(result, f"stage:{key}/argmax", stage_flips)
    result["tf32_band"] = {
        "error_drift_diagnostic_only": True,
        "per_case": bands_by_case,
    }
    if not embedding:

        def total_flips():
            candidate = sum(row["first"]["argmax_flips"] for _, row, _ in band_rows)
            library = sum(
                control["first"]["argmax_flips"] for _, _, control in band_rows
            )
            band = sum(value.get("total_flips", 0) for value in bands_by_case.values())
            if candidate > library + band:
                raise Rejected("stage: total flips above Library plus measured band")

        check(result, "stage:tf32/aggregate_argmax", total_flips)


def secret_checks(result: dict, secrets: list[dict], coverage: Coverage) -> None:
    """Gate candidate and Library-in-mode errors on the same f64 reference sample."""
    found = {row["id"]: row for row in secrets}
    for layer, batch, mode in sorted(coverage.triples):
        key = f"{mode}/secret/b{batch}/{layer}"
        row = found.get(key)

        def gate(row=row):
            if row is None:
                raise Rejected("secret input: check missing for a declared triple")
            measurements = ("library", "eager", "replay")
            for name in measurements:
                difference = row[name]
                if not difference["finite"]:
                    raise Rejected(f"secret input: {name} output is not finite")
                values = [difference[field] for field in ("relative_l2", "max_abs")]
                finite(values)
                if min(values) < 0:
                    raise Rejected("secret input: negative error")
            if row.get("truth") != "f64":
                raise Rejected("secret input: independent f64 truth missing")
            floor = {
                field: row["library"][field] for field in ("relative_l2", "max_abs")
            }
            if any(value == 0 for value in floor.values()):
                raise Rejected(
                    "secret input: Library error against f64 is exactly zero; fail closed"
                )
            for name in ("eager", "replay"):
                difference = row[name]
                if (
                    difference["relative_l2"] > 1.10 * floor["relative_l2"]
                    or difference["max_abs"] > 2.00 * floor["max_abs"]
                ):
                    raise Rejected(
                        f"secret input: candidate {name} output differs from Library on a fresh input beyond the layer gate against f64 truth"
                    )
            if not row["replay_equals_eager"]:
                raise Rejected("secret input: graph replay differs from the eager run")
            return {
                **row,
                "floor": floor,
                "ratios": {
                    name: {
                        field: ratio(row[name][field], floor[field]) for field in floor
                    }
                    for name in ("eager", "replay")
                },
            }

        check(result, f"secret:{key}", gate)


def timing(
    result: dict, cases: dict[str, list[dict]], implementation: str, coverage: Coverage
) -> None:
    """Burst-timed A/B/A/B per math mode; noisy Library controls block a case.

    Declared operators get the strict 0% gate. Stages that run declared layers get the
    regression guard, with each case's mean speedup at least 1. Undeclared operators and stages get no speed
    gate; `numeric` requires their outputs to be bit-identical to the Library control.
    Each timing row records the basis of its decision in `gate`.
    """
    result["timing"] = []
    measurable = defaultdict(lambda: [0, 0])
    for mode in MODES:
        runs = cases[mode]
        maps = [{row["id"]: row for row in run["rows"]} for run in runs]
        for key in sorted(maps[0]):
            samples = [
                Timing(
                    run["pid"],
                    run["implementation"],
                    mapping[key]["warmup"],
                    tuple(mapping[key]["milliseconds"]),
                )
                for run, mapping in zip(runs, maps, strict=True)
            ]
            medians = [statistics.median(run.milliseconds) for run in samples]
            bound = spread_bound(key)
            library_spread = abs(medians[0] - medians[2]) / min(medians[0], medians[2])
            detail = {
                "id": key,
                "medians_ms": medians,
                "bursts": [mapping[key]["burst"] for mapping in maps],
                "speedups": [medians[0] / medians[1], medians[2] / medians[3]],
                "process_spread_fraction": (max(medians) - min(medians)) / min(medians),
                "library_process_spread_fraction": library_spread,
                "spread_bound": bound,
                "measurable": library_spread <= bound,
            }
            result["timing"].append(detail)
            kind = "stage" if key.endswith("/stage") else "operator"
            measurable[kind][0] += int(detail["measurable"])
            measurable[kind][1] += 1
            if implementation == "Library":
                if len({x.pid for x in samples}) != 4 or any(
                    x.warmup < 5 or len(x.milliseconds) < 20 or min(x.milliseconds) <= 0
                    for x in samples
                ):
                    raise Rejected("invalid Library control timing")
                continue
            _, _, batch, boundary, _ = split_key(key)
            if boundary != "stage":
                if not coverage.declared(boundary, batch, mode):
                    detail["gate"] = "undeclared operator: bitwise Library identity"
                    continue
                detail["gate"] = "operator: strict, faster Library median"
                check(
                    result,
                    f"speed:{key}",
                    lambda samples=samples, bound=bound: speed(
                        samples, implementation, bound
                    ),
                )
                continue
            if not coverage.layers_at(batch, mode):
                detail["gate"] = (
                    "undeclared stage: bitwise Library identity, no candidate kernels"
                )
                continue
            detail["gate"] = "stage: paired replay guard in same-process phase"
    result["measurability"] = {
        kind: {"measurable": count, "cases": total}
        for kind, (count, total) in measurable.items()
    }
    result["noise_floor_fraction"] = max(
        row["library_process_spread_fraction"] for row in result["timing"]
    )


def paired_checks(
    result: dict,
    process: dict,
    coverage: Coverage,
    library: dict,
    candidate: dict,
    implementation: str,
) -> None:
    """Require exact paired case coverage, numeric outputs and same-process saving."""
    mode = process["mode"]
    expected = {
        f"{mode}/{case}/b{batch}/stage"
        for case, batch in target_cases(result["target"])
        if coverage.layers_at(batch, mode) or implementation == "Library"
    }
    rows = {row["id"]: row for row in process["rows"]}
    if set(rows) != expected or len(rows) != len(process["rows"]):
        raise Rejected("paired stage: missing or duplicate case coverage")
    numeric = [{row["id"]: row for row in p["rows"]} for p in (library, candidate)]
    for key, row in rows.items():

        def outputs(row=row, key=key):
            for side in (0, 1):
                expected_hashes = [
                    numeric[side][key]["first"]["sha256"],
                    numeric[side][f"{key}/switched"]["first"]["sha256"],
                ]
                if row["output_sha256"][side] != expected_hashes:
                    raise Rejected(
                        "paired stage: final output differs from numeric evidence"
                    )
                if (
                    side == 1
                    and "stage_tail" in row
                    and row["stage_tail"].get("delayed_output_sha256")
                    != expected_hashes
                ):
                    raise Rejected(
                        "paired stage: delayed output differs from numeric evidence"
                    )
                for op in row["operator_outputs"]:
                    operator_key = (
                        op.get("id") or key.removesuffix("stage") + op["layer"]
                    )
                    hashes = [
                        numeric[side][operator_key]["first"]["sha256"],
                        numeric[side][f"{operator_key}/switched"]["first"]["sha256"],
                    ]
                    if op["output_sha256"][side] != hashes:
                        raise Rejected(
                            "paired stage: operator output differs from numeric evidence"
                        )

        _, _, batch, _, _ = split_key(key)
        expected_layers = (
            set(layers(result["target"]))
            if implementation == "Library"
            else set(coverage.layers_at(batch, mode))
        )
        observed_layers = [
            op.get("layer") or split_key(op["id"])[3] for op in row["operator_outputs"]
        ]
        if (
            len(observed_layers) != len(set(observed_layers))
            or set(observed_layers) != expected_layers
        ):
            raise Rejected("paired stage: missing or duplicate operator coverage")
        if "operator_layers" in row and set(row["operator_layers"]) != expected_layers:
            raise Rejected(
                "paired stage: measured operator strata differ from coverage"
            )
        check(result, f"paired_output:{key}", outputs)
        operator_keys = {
            f"speed:{key.removesuffix('stage')}{layer}"
            for layer in coverage.layers_at(batch, mode)
        }
        passed = {item["check"] for item in result["checks"] if item["passed"]}
        details = {detail["id"]: detail for detail in result.get("timing", [])}
        for item in result["checks"]:
            if item["check"] not in operator_keys or not item.get("blocked"):
                continue
            try:
                evaluation = noise_timing(
                    item,
                    details[item["check"].removeprefix("speed:")],
                    allow_stage=False,
                )
                if evaluation["accepted"]:
                    passed.add(item["check"])
            except (Rejected, KeyError, ValueError):
                pass
        operator_passed = bool(operator_keys) and operator_keys <= passed
        if implementation == "Library":
            # a sanity control validates the estimator but never claims a speed improvement
            check(result, f"paired_control:{key}", lambda row=row: paired_control(row))
        else:
            check(
                result,
                f"paired_stage:{key}",
                lambda row=row, ok=operator_passed: stage_speed(row, ok),
            )


def paired_control(row: dict) -> dict:
    """Check the full paired estimator contract without imposing a win on Library."""
    try:
        return stage_speed(row, True)
    except (StageBlocked, StageRejected) as error:
        return {
            **error.evidence,
            "control": True,
            "comparison": "Library vs Library",
        }


def timing_outputs(
    result: dict, runs: list[tuple[dict, dict, bool]], implementation: str
) -> None:
    """Each timing process's replays on both inputs must equal its numeric outputs.

    `runs` holds (timing process, numeric rows by id, candidate binary). A mismatch in
    the candidate binary fails the case; in the frozen control it blocks the run,
    because the control itself is then not reproducible.
    """
    for run, numeric_rows, is_candidate in runs:
        for row in run["rows"]:
            key = row["id"]
            expected = [
                numeric_rows[key]["first"]["sha256"],
                numeric_rows[f"{key}/switched"]["first"]["sha256"],
            ]
            if row["output_sha256"] == expected:
                continue
            if is_candidate and implementation != "Library":
                check(
                    result,
                    f"timing_output:{key}",
                    lambda: fail(
                        Rejected(
                            "timing: final output differs from the numeric output for the same input"
                        )
                    ),
                )
            else:
                check(
                    result,
                    f"timing_output:{key}/control",
                    lambda: fail(
                        Blocked(
                            "timing: Library final output differs between processes"
                        )
                    ),
                )


FLOAT_ATOMIC = r"(?<![\w.])(?:atom|red)\.[^;\n]*\.(?:f32|f64|f16|bf16|f16x2|bf16x2)\b"


def fixed_reduction_order(trace: Path, modules: dict) -> dict:
    """Reject launched custom FP atomic entries, even if two samples happen to agree."""
    with sqlite3.connect(f"file:{trace}?mode=ro", uri=True) as connection:
        launched = {
            row[0]
            for row in connection.execute(
                "SELECT DISTINCT s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName"
            )
        }
    inspected = []
    atomic = FLOAT_ATOMIC
    for module in modules["modules"]:
        path = ROOT / module["path"]
        source = path.read_text()
        markers = list(re.finditer(r"\.(entry|func)\b", source))
        functions = [
            source[
                marker.start() : markers[i + 1].start()
                if i + 1 < len(markers)
                else len(source)
            ]
            for i, marker in enumerate(markers)
            if marker.group(1) == "func"
        ]
        atomic_helper = any(re.search(atomic, body) for body in functions)
        entries = list(ENTRY.finditer(source))
        for i, entry in enumerate(entries):
            name = entry.group(1)
            if name not in launched:
                continue
            body = source[
                entry.start() : entries[i + 1].start()
                if i + 1 < len(entries)
                else len(source)
            ]
            if atomic_helper:
                raise Rejected(
                    f"determinism: floating-point atomic in custom PTX helper module {path.name}"
                )
            inspected.append(
                {"path": module["path"], "entry": name, "sha256": module["sha256"]}
            )
            if re.search(atomic, body):
                raise Rejected(
                    f"determinism: floating-point atomic in launched custom entry {name}"
                )
    if not inspected:
        raise Rejected("determinism: no custom PTX entry evidence")
    return {"inspected": inspected}


def profile(
    result: dict,
    binary: Path,
    env: dict,
    directory: Path,
    steps: list,
    implementation: str,
    target: str,
    *,
    test: str = PROFILE_TEST,
    label: str = "profile",
    eager_only: bool = False,
) -> Path:
    """Trace plan, warm-up, eager, capture and one replay with the locked shim."""
    result["profile_trace_mode"] = "graph"
    shim = directory / "nvtx.so"
    if command(
        [
            "g++",
            "-shared",
            "-fPIC",
            "-O2",
            "-I/usr/local/cuda-12.8/targets/x86_64-linux/include",
            str(ROOT / "scripts/cuda/qualify/nvtx.cpp"),
            "-o",
            str(shim),
            "-ldl",
        ],
        env,
        directory / "nvtx-build.log",
        steps,
    ):
        raise Rejected("NVTX shim build failed")
    prefix = directory / label
    profile_env = dict(
        env,
        SPEAKRS_QUALIFY_NVTX=str(shim),
        SPEAKRS_QUALIFY_IMPL=implementation,
        SPEAKRS_QUALIFY_PHASE="profile",
        SPEAKRS_QUALIFY_SHORT_TRACE="0" if eager_only else "1",
        SPEAKRS_QUALIFY_OUTPUT=str(directory / f"{label}-driver.json"),
    )
    if gpu_command(
        [
            "nsys",
            "profile",
            "--sample=none",
            "--cpuctxsw=none",
            "--trace=cuda,nvtx,cublas,cudnn",
            "--cuda-graph-trace=graph",
            "--force-overwrite=true",
            "-o",
            str(prefix),
            str(binary),
            "--exact",
            test,
            "--ignored",
            "--nocapture",
        ],
        profile_env,
        directory / f"{label}.log",
        steps,
    ):
        raise Rejected("blocked: nsys stage trace did not complete")
    exported = prefix.with_suffix(".sqlite")
    if command(
        [
            "nsys",
            "export",
            "--type",
            "sqlite",
            "--force-overwrite=true",
            "-o",
            str(exported),
            str(prefix.with_suffix(".nsys-rep")),
        ],
        dict(env, SPEAKRS_QUALIFY_PHASE="profile"),
        directory / f"{label}-export.log",
        steps,
    ):
        raise Rejected("blocked: nsys SQLite export failed")
    if test == PROFILE_TEST:
        process = json.loads((directory / f"{label}-driver.json").read_text())
        validate_target(
            process,
            env["SPEAKRS_CUDA_PTX_TIER"],
            candidate_area=AREAS[env["SPEAKRS_QUALIFY_TARGET"]]
            if implementation != "Library"
            else None,
        )
    result[label] = {"path": str(exported), "sha256": sha(exported)}
    if test == PROFILE_TEST and not eager_only and process.get("side_streams"):
        # replay correlations cannot recover per-node side-stream ownership
        # retain the original eager trace for the unchanged attribution check
        result["short_profile"] = result[label]
        retained = profile(
            result,
            binary,
            env,
            directory,
            steps,
            implementation,
            target,
            test=test,
            label=f"{label}-eager",
            eager_only=True,
        )
        result[label] = {"path": str(retained), "sha256": sha(retained)}
        result["profile_retained_checks"] = [
            "profile: stream attribution",
            "profile:graph_nodes: eager launch multisets",
        ]
        return retained
    return exported


CANDIDATE_AREAS = frozenset({"resnet", "lstm", "sincnet", "fbankdft"})


def stable_modules(
    processes: list[dict], coverage: Coverage, target: str, implementation: str
) -> dict:
    """Require stable bytes and coverage-exact candidate-area loading per process."""
    fixed = None
    candidate = {}
    evidence = []
    union = {}
    for process in processes:
        modules = {module["area"]: module for module in process["loaded_modules"]}
        if len(modules) != len(process["loaded_modules"]):
            raise Rejected("process loaded duplicate PTX areas")
        non_candidate = {
            area: module
            for area, module in modules.items()
            if area not in CANDIDATE_AREAS
        }
        if fixed is not None and fixed != non_candidate:
            raise Rejected("processes loaded different non-candidate PTX bytes")
        fixed = non_candidate
        loaded = set(modules) & CANDIDATE_AREAS
        for area in loaded:
            if area in candidate and candidate[area] != modules[area]:
                raise Rejected(
                    f"processes loaded different candidate PTX bytes for {area}"
                )
            candidate[area] = modules[area]
        # profile covers all cases; numeric and timing cover their recorded triples
        ran = (
            frozenset(
                t for t in coverage.triples if process.get("mode") in (None, t[2])
            )
            if process["phase"] in ("profile", "paired")
            else frozenset(
                (layer, batch, mode)
                for row in process["rows"]
                for mode, _, batch, layer, _ in [split_key(row["id"])]
                if (layer, batch, mode) in coverage.triples
            )
        )
        required = (
            {target}
            if implementation in ("Oxide", "StageTail", "StageTailControl") and ran
            else set()
        )
        if required - loaded:
            raise Rejected(
                f"process runs declared triples but did not load candidate area {target}"
            )
        if loaded - required:
            raise Rejected(
                f"process with no declared triple loaded candidate area: {sorted(loaded - required)}"
            )
        union.update(modules)
        evidence.append(
            {
                "pid": process.get("pid"),
                "phase": process["phase"],
                "mode": process.get("mode"),
                "loaded_candidate_areas": sorted(loaded),
                "loaded_areas": sorted(modules),
            }
        )
    return {"processes": evidence, "modules": [union[area] for area in sorted(union)]}


def graph_kernels(processes: list[dict]) -> dict[str, set[str]]:
    """Candidate kernels each labeled capture recorded, by case id."""
    found: dict[str, set[str]] = defaultdict(set)
    for process in processes:
        for capture in process.get("graph_evidence", []):
            if capture.get("case") is None:
                continue
            found[capture["case"]]
            if capture["scope"] == "candidate":
                found[capture["case"]].update(capture["kernels"])
    return dict(found)


def capture_multisets(
    trace: Path, processes: list[dict], nonce: str, *, required: int | None = None
) -> dict:
    """Compare each captured plan's multiset with correlated eager launches, not declarations."""
    eager = window_kernels(trace, nonce, multiset=True)
    compared = 0
    evidence = []
    for process in processes:
        found: dict[str, Counter] = defaultdict(Counter)
        for capture in process.get("graph_evidence", []):
            if capture.get("case") is not None:
                found[capture["case"]]
                if capture["scope"] == "candidate":
                    found[capture["case"]].update(capture["kernels"])
        for case, kernels in found.items():
            expected = eager.get(case, Counter())
            if kernels != expected:
                raise Rejected(
                    f"profile: capture != eager launch multiset for {case}: {dict(kernels)} != {dict(expected)}"
                )
            compared += 1
            evidence.append({"case": case, "kernel_multiset": dict(expected)})
    if required is not None and (
        compared != required or len(eager) != required or not all(eager.values())
    ):
        raise Rejected("profile: incomplete same-plan sequential capture evidence")
    return {"compared": compared, "eager": evidence}


def band_layers(coverage: Coverage, implementation: str) -> str:
    """`<batch>:<layers>;...` for every batch with declared TF32 layers."""
    if implementation == "Library":
        return ""
    return ";".join(
        f"{batch}:{','.join(coverage.layers_at(batch, 'tf32'))}"
        for batch in sorted(coverage.batches)
        if coverage.layers_at(batch, "tf32")
    )


def collect_tier(
    target: str, implementation: str, directory: Path, result: dict, tier: str
) -> None:
    """Collect all live checks, even when a planted fault already fails parity."""
    nonce = secrets.token_hex(16)
    env = dict(
        clean_environment(),
        CARGO_TARGET_DIR=str(BOX / "target" / tier),
        SPEAKRS_REQUIRE_GPU="1",
        SPEAKRS_CUDA_PTX_TIER=tier,
        SPEAKRS_QUALIFY_TARGET=target,
        SPEAKRS_QUALIFY_NONCE=nonce,
    )
    if target == "fbankdft" and implementation in ("StageTail", "StageTailControl"):
        raise Rejected(
            "fbankdft: StageTail requires an accepted production plan; none exists before the port"
        )
    # the paired fault fixture uses the archived JIT-qualified production plans
    if implementation in ("StageTail", "StageTailControl"):
        env["SPEAKRS_CUDA_FORCE_PTX_JIT"] = "1"
        result["artifact_policy"] = "LegacyPtxJitFixture"
    else:
        result["artifact_policy"] = "ExactCubinPreferred"
    result["requested_tier"] = tier
    result["code_sha256"] = {
        name: sha(ROOT / name)
        for name in inventory()
        if name.endswith((".rs", ".manifest")) and (ROOT / name).is_file()
    }
    result["code_sha256"].update(
        {
            str(path.relative_to(ROOT)): sha(path)
            for path in (ROOT / "src/inference/cuda/candidate").rglob("*.rs")
        }
    )
    result["code_sha256"].update(
        {
            str(path.relative_to(ROOT)): sha(path)
            for base in (
                ROOT / "crates/speakrs-cuda-kernels/src",
                ROOT / "src/inference/cuda/ptx",
            )
            for path in base.rglob("*")
            if path.is_file() and path.suffix in (".rs", ".ptx", ".manifest")
        }
    )
    # candidate artifacts sit outside the harness lock and must be pinned in full,
    # including unselected tiers and architectures, not just the loaded module
    candidate_manifest = ROOT / "src/inference/cuda/ptx" / f"{AREAS[target]}.manifest"
    if candidate_manifest.is_file() or implementation == "Oxide":
        for hashes in shipped_files(ROOT, AREAS[target]).values():
            result["code_sha256"].update(hashes)
    result["verified_inputs"] = verify_inputs(target)
    steps = result["commands"]
    binary = build(env, directory, steps)
    build_library(env, directory, steps)
    result["driver_sha256"] = sha(binary)
    control_source = directory / "control-source"
    control_source.mkdir()
    archive_path = resolve("tests/cuda_qualify/control.tar.gz")
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            name = Path(member.name)
            if not member.isfile() or name.is_absolute() or ".." in name.parts:
                raise Rejected("unsafe frozen control archive")
        archive.extractall(control_source, filter="data")
    archive_digest = sha(archive_path)
    control_env = dict(env, CARGO_TARGET_DIR=str(control_target(archive_digest, tier)))
    control = build(control_env, directory, steps, control_source / "Cargo.toml")
    result["control"] = {
        "archive_sha256": archive_digest,
        "driver_sha256": sha(control),
        "toolchain": toolchain(control_env),
    }

    raw = driver(binary, env, directory, steps, implementation, "coverage", "coverage")[
        "coverage"
    ]
    coverage = parse_coverage(raw, target)
    result["coverage_declared"] = coverage.record()
    if implementation == "Oxide" and not coverage.layers:
        raise Rejected(
            "coverage: the candidate declares no layer, batch and math triple"
        )

    numeric_library, numeric_candidate = {}, {}
    candidate_processes = []
    band_coverage = (
        Coverage.product(layers(target), target_batches(target), MODES)
        if implementation == "Library"
        else coverage
    )
    band = band_layers(band_coverage, "Oxide")
    for mode in MODES:
        extra = {"SPEAKRS_QUALIFY_MODE": mode}
        library = driver(
            control,
            env,
            directory,
            steps,
            "Library",
            "numeric",
            f"numeric-{mode}-library",
            {**extra, **({"SPEAKRS_QUALIFY_BAND_LAYERS": band} if band else {})},
        )
        candidate = driver(
            binary,
            env,
            directory,
            steps,
            implementation,
            "numeric",
            f"numeric-{mode}-candidate",
            extra,
        )
        if library["device"] != candidate["device"]:
            raise Rejected("target: Library and candidate device identity differs")
        result.setdefault("device_sm", library["device_sm"])
        result.setdefault("device", library["device"])
        numeric_library[mode] = library
        numeric_candidate[mode] = candidate
        candidate_processes.append(candidate)
    library_rows = [row for mode in MODES for row in numeric_library[mode]["rows"]]
    candidate_rows = [row for mode in MODES for row in numeric_candidate[mode]["rows"]]
    result["numeric"] = {
        "library": list(numeric_library.values()),
        "candidate": list(numeric_candidate.values()),
    }
    numeric(result, library_rows, candidate_rows, coverage)

    modules = list(
        {
            module["area"]: module
            for process in candidate_processes
            for module in process["loaded_modules"]
        }.values()
    )
    allow = None
    try:
        allow, module_evidence = verify_modules(modules)
        result["loaded_ptx"] = module_evidence
        result["checks"].append(
            {"check": "ptx:loaded_bytes", "passed": True, "evidence": module_evidence}
        )
    except Rejected as error:
        result["checks"].append(
            {"check": "ptx:loaded_bytes", "passed": False, "reason": str(error)}
        )
    if allow is not None:

        def shared_loads():
            failures = {}
            for module in result["loaded_ptx"]["modules"]:
                found = shared_initialization((ROOT / module["path"]).read_text())
                failures.update(
                    {f"{module['area']}:{name}": loads for name, loads in found.items()}
                )
            if failures:
                raise Rejected(
                    f"ptx: shared load without a dominating shared store or covering barrier: {sorted(failures)[:8]}"
                )
            return {"modules": len(result["loaded_ptx"]["modules"])}

        check(result, "ptx:shared_initialization", shared_loads)
    phases = CONTROL_PHASES.get(
        implementation, MUTANT_PHASES.get(implementation, PHASES)
    )
    result["phases_run"] = list(phases)
    if "timing" in phases:
        timing_cases = {}
        output_runs = []
        library_numeric_rows = {row["id"]: row for row in library_rows}
        candidate_numeric_rows = {row["id"]: row for row in candidate_rows}
        for mode in MODES:
            runs = []
            for i, impl in enumerate(
                ("Library", implementation, "Library", implementation)
            ):
                run = driver(
                    control if i % 2 == 0 else binary,
                    env,
                    directory,
                    steps,
                    impl,
                    "timing",
                    f"timing-{mode}-{i}-{impl}",
                    {"SPEAKRS_QUALIFY_MODE": mode},
                )
                runs.append(run)
                candidate_binary = i % 2 == 1
                output_runs.append(
                    (
                        run,
                        candidate_numeric_rows
                        if candidate_binary
                        else library_numeric_rows,
                        candidate_binary,
                    )
                )
                if candidate_binary:
                    candidate_processes.append(run)
            timing_cases[mode] = runs
        timing(result, timing_cases, implementation, coverage)
        timing_outputs(result, output_runs, implementation)
        result["timing_processes"] = [
            {k: v for k, v in run.items() if k != "rows"}
            for runs in timing_cases.values()
            for run in runs
        ]

    if "paired" in phases:
        result["paired_processes"] = []
        for mode in MODES:
            paired = driver(
                binary,
                env,
                directory,
                steps,
                implementation,
                "paired",
                f"paired-{mode}",
                {"SPEAKRS_QUALIFY_MODE": mode},
            )
            candidate_processes.append(paired)
            result["paired_processes"].append(paired)
            paired_checks(
                result,
                paired,
                coverage,
                numeric_library[mode],
                numeric_candidate[mode],
                implementation,
            )

    def captured_calls():
        violations = [
            v for p in candidate_processes for v in p["library_call_violations"]
        ]
        if violations:
            raise Rejected(
                f"profile: library calls from candidate scopes, eager or captured: {violations[:8]} ({len(violations)} calls)"
            )

    def graph_nodes():
        violations = [v for p in candidate_processes for v in p["graph_violations"]]
        if violations:
            raise Rejected(
                f"profile: captured graph nodes a candidate scope may not contain: {violations[:8]} ({len(violations)} nodes)"
            )
        return {
            "captured_scopes": sum(
                len(p["graph_evidence"]) for p in candidate_processes
            )
        }

    if "profile" not in phases:
        check(result, "profile:graph_nodes", graph_nodes)

    if "profile" in phases:
        exported = profile(
            result, binary, env, directory, steps, implementation, target
        )
        candidate_processes.append(
            json.loads((directory / "profile-driver.json").read_text())
        )
        if allow is not None:
            check(
                result,
                "determinism:fixed_reduction_order",
                lambda: fixed_reduction_order(exported, result["loaded_ptx"]),
            )
            declared = frozenset(layer for layer in coverage.layers)

            def trace(library_control: bool) -> dict:
                return attribute(
                    exported,
                    layers(target),
                    nonce,
                    allow,
                    declared,
                    library_control,
                )

            if implementation == "Library":
                try:
                    evidence = trace(True)
                except Rejected as error:
                    raise Rejected(
                        f"blocked: Library trace cannot establish attribution: {error}"
                    ) from error
                result["checks"].append(
                    {"check": "profile", "passed": True, "evidence": evidence}
                )
            else:
                check(result, "profile", lambda: trace(False))

        def graph_matches_profile():
            graphs = graph_kernels(candidate_processes)
            evidence = graph_nodes()
            eager = window_kernels(exported, nonce)
            differences = {
                case: {
                    "graph": sorted(kernels),
                    "eager": sorted(eager.get(case, set())),
                }
                for case, kernels in graphs.items()
                if kernels != eager.get(case, set())
            }
            if differences:
                raise Rejected(
                    f"profile: captured kernels differ from the eager profile: {dict(list(differences.items())[:4])}"
                )
            multisets = capture_multisets(exported, candidate_processes, nonce)
            return {**evidence, "compared_cases": len(graphs), "multisets": multisets}

        check(result, "profile:graph_nodes", graph_matches_profile)

    check(result, "profile:captured_library_calls", captured_calls)

    if "profile" in phases and implementation == "Oxide":
        regression = profile(
            result,
            binary,
            env,
            directory,
            steps,
            implementation,
            target,
            test="inference::cuda::test_support::qualify::sequential_capture_keeps_candidate_kernels",
            label="capture-regression",
        )
        regression_process = json.loads(
            (directory / "capture-regression-driver.json").read_text()
        )
        check(
            result,
            "profile:sequential_capture",
            lambda: capture_multisets(
                regression, [regression_process], nonce, required=7
            ),
        )

    check(
        result,
        "ptx:loaded_bytes/stable",
        lambda: stable_modules(candidate_processes, coverage, target, implementation),
    )

    result["sanitizer"] = []
    if os.environ.get("SPEAKRS_QUALIFY_DIAGNOSTICS_ONLY") == "1":
        for tool in TOOLS:
            result["checks"].append(
                {
                    "check": f"sanitizer:{tool}",
                    "passed": False,
                    "reason": "required tool not run: diagnostic collection cannot qualify a replacement",
                }
            )
        result.update(
            status="blocked",
            reason="diagnostic collection: required sanitizers were not run",
            diagnostics_only=True,
        )
        return
    policy_path = ROOT / "scripts/cuda/qualify/SANITIZER_POLICY.json"
    result["sanitizer_policy"] = json.loads(policy_path.read_text())
    result["sanitizer_policy_sha256"] = sha(policy_path)
    result["sanitizer_fingerprint"] = sanitizer_fingerprint()
    baseline = known_library_baseline(target, tier, result, result["device_sm"])
    blocked_tools = []
    include = sorted(allow.entries) if allow is not None else []
    # positive controls prove the include filter for Library controls and candidates
    if implementation not in (*MUTANTS, *CONTROL_PHASES) and include:
        for control_name in CONTROLS:
            check(
                result,
                f"sanitizer:control/{control_name}",
                lambda control_name=control_name: filter_proof(
                    binary, env, directory, steps, control_name, include
                ),
            )
    if target == "lstm" and implementation not in (*MUTANTS, *CONTROL_PHASES):
        for tool in TOOLS:
            record = library_tool_record(baseline, tool)
            if record is not None:
                result["sanitizer"].append(
                    {
                        **record,
                        "implementation": "ProjectionBaseline",
                        "source": "owner-locked library baseline",
                    }
                )
                result["checks"].append(
                    {
                        "check": f"sanitizer:ProjectionBaseline/{tool}",
                        "passed": True,
                        "evidence": record,
                    }
                )
                continue
            log = directory / f"ProjectionBaseline-{tool}.log"
            child = dict(
                env,
                SPEAKRS_QUALIFY_IMPL="Library",
                SPEAKRS_QUALIFY_PHASE="projection_baseline",
                SPEAKRS_QUALIFY_OUTPUT=str(
                    directory / f"ProjectionBaseline-{tool}.json"
                ),
                CUDA_MODULE_LOADING="LAZY",
            )
            returncode = gpu_command(sanitizer_argv(control, tool), child, log, steps)
            text = log.read_text()
            result["sanitizer"].append(
                {
                    "implementation": "ProjectionBaseline",
                    "tool": tool,
                    "returncode": returncode,
                    "path": str(log),
                    "sha256": sha(log),
                    "scope": "LSTM input projections only",
                    "timeout_seconds": 1200,
                }
            )

            def projection_check(tool=tool, code=returncode, text=text):
                sanitizer(tool, code, text)
                missing = missing_projection_markers(text)
                if missing:
                    raise Rejected(f"projection baseline omitted shapes: {missing[:4]}")

            check(result, f"sanitizer:ProjectionBaseline/{tool}", projection_check)
            if "1 passed; 0 failed" not in text:
                blocked_tools.append(
                    f"ProjectionBaseline/{tool}: exceeded 20 minutes"
                    if returncode in (124, 137)
                    else f"ProjectionBaseline/{tool}"
                )
    # mutants prove their intended gate; sanitizer completion adds no harness evidence
    if implementation not in ("Library", *MUTANTS, *CONTROL_PHASES) and include:
        for tool in TOOLS:
            child = dict(
                env,
                SPEAKRS_QUALIFY_IMPL=implementation,
                SPEAKRS_QUALIFY_PHASE="sanitize",
                CUDA_MODULE_LOADING="LAZY",
            )
            record, text = candidate_sanitizer(
                binary, tool, child, directory, steps, include, coverage
            )
            record["required"] = True
            result["sanitizer"].append(record)
            check(
                result,
                f"sanitizer:{implementation}/{tool}",
                lambda tool=tool, record=record, text=text: sanitizer(
                    tool, record["returncode"], text
                ),
            )
            if record["returncode"] in (124, 137) or "1 passed; 0 failed" not in text:
                blocked_tools.append(f"{implementation}/{tool}: incomplete bounded run")
    finish_tier(result, blocked_tools)
    clocks = [
        p["observed_sm_clock"]
        for p in candidate_processes
        if p.get("observed_sm_clock")
    ]
    if clocks:
        result["device"]["observed_sm_clock_range_mhz"] = [
            min(p["min_mhz"] for p in clocks),
            max(p["max_mhz"] for p in clocks),
        ]
    result["loaded_tiers"] = {
        m["area"]: m["tier"] for m in result.get("loaded_ptx", {}).get("modules", [])
    }
    result["accepted_tuples"] = (
        [
            list(t)
            for t in sorted(coverage.triples)
            if t[1] in collection(target).production
        ]
        if result["status"] == "passed" and implementation == "Oxide"
        else []
    )
    result["phase_wall_seconds"] = phase_wall_times(steps)
    result["coverage"] = {
        "math": list(MODES),
        "cases": target_cases(target),
        "tier": tier,
        "real_turing_hardware": "untested",
    }


def phase_wall_times(steps: list[dict]) -> dict:
    """Sum command wall times and expose unlocked CPU subsets without double counting."""
    totals = dict.fromkeys(
        (
            "build",
            "cpu_truth",
            "cpu_draws",
            "numeric",
            "timing",
            "paired",
            "profile",
            "sanitizer",
        ),
        0.0,
    )
    for step in steps:
        phase = step.get("phase", "build")
        phase = (
            "sanitizer"
            if phase in ("sanitize", "filter_proof", "projection_baseline")
            else phase
        )
        if phase in totals:
            totals[phase] += step.get("wall_seconds", 0.0)
        cpu = step.get("cpu_work_wall_seconds", {})
        totals["cpu_truth"] += cpu.get("f64", 0.0)
        totals["cpu_draws"] += cpu.get("tf32_draws", 0.0)
    return {
        "seconds": totals,
        "numeric_includes_cpu_work": True,
        "cpu_work_excludes_lock_wait": True,
    }


def finish_tier(result: dict, blocked_tools: list[str]) -> None:
    """Hard failures reject; incomplete or unmeasurable evidence blocks; else pass."""
    evaluation = evaluate_checks(
        result["checks"], result.get("timing", []), allow_stage=False
    )
    result["verdict_evaluation"] = evaluation
    if evaluation["hard_failures"]:
        result.update(
            status="rejected",
            reason=f"{len(evaluation['hard_failures'])} failed checks",
        )
    elif blocked_tools:
        result.update(
            status="blocked",
            reason=f"Compute Sanitizer drivers did not complete: {blocked_tools}",
        )
    elif not evaluation["accepted"]:
        result.update(
            status="blocked",
            reason=f"{len(evaluation['unresolved'])} checks cannot be decided",
        )
    else:
        result.update(
            status="passed",
            reason="all required checks passed under locked verdict evaluation",
        )


def shipped_tiers(target: str, root: Path = ROOT) -> tuple[str, ...]:
    """Detect the candidate area's PTX variants from files, not declarations."""
    area = AREAS[target]
    files = {
        path.name for path in (root / "src/inference/cuda/ptx").glob(f"{area}.sm*.ptx")
    }
    if f"{area}.sm75.ptx" not in files:
        raise Rejected("missing mandatory sm75 candidate PTX")
    known = {f"{area}.{tier}.ptx" for tier in ("sm75", "sm80", "sm90", "sm120")}
    if files - known:
        raise Rejected("unsupported shipped candidate tier")
    return tuple(
        tier
        for tier in ("sm75", "sm80", "sm90", "sm120")
        if f"{area}.{tier}.ptx" in files
    )


def collect(target: str, implementation: str, directory: Path, result: dict) -> None:
    """A pass requires the scan and the requested compiled tier to pass every check."""
    findings = scan(ROOT, cargo_home())
    result["static_scan"] = findings
    if findings["findings"]:
        result["checks"].append(
            {
                "check": "static_scan",
                "passed": False,
                "reason": f"static scan: {len(findings['findings'])} refused constructs: {findings['findings'][:8]}",
            }
        )
        result.update(
            status="rejected", reason="static scan refused the candidate code"
        )
        result["accepts_replacement"] = False
        return
    result["checks"].append(
        {"check": "static_scan", "passed": True, "evidence": findings}
    )
    result["tiers"] = {}
    for tier in (
        result.get("requested_tier", os.environ.get("SPEAKRS_CUDA_PTX_TIER", "sm75")),
    ):
        output = directory / tier
        output.mkdir()
        child = {
            "target": target,
            "implementation": implementation,
            "checks": [],
            "commands": result["commands"],
            "status": "blocked",
            "requested_tier": tier,
        }
        result["tiers"][tier] = child
        try:
            collect_tier(target, implementation, output, child, tier)
        except (Rejected, OSError, sqlite3.Error, KeyError, ValueError) as error:
            child.update(status="blocked", reason=str(error))
        finally:
            child.pop("commands")
            result["checks"].extend(
                {**item, "check": f"{tier}/{item['check']}"} for item in child["checks"]
            )
    result["phase_wall_seconds"] = phase_wall_times(result["commands"])
    for child in result["tiers"].values():
        child["phase_wall_seconds"] = result["phase_wall_seconds"]
    failed = [item for item in result["checks"] if not item["passed"]]
    hard = [item for item in failed if not item.get("blocked")]
    blocked = [
        child for child in result["tiers"].values() if child["status"] == "blocked"
    ]
    if hard:
        result["status"] = "rejected"
        result["reason"] = f"{len(hard)} failed checks"
    elif blocked:
        result["status"] = "blocked"
        result["reason"] = "; ".join(child.get("reason", "") for child in blocked)
    else:
        result["status"] = "passed"
        result["reason"] = "all required checks passed"
    if implementation in MUTANTS and result["status"] != "blocked":
        gate = mutant_gate(implementation, hard)
        result["mutant_gate"] = gate
        # an unrelated failure must not hide a mutant that escaped its intended check
        if not gate["caught"]:
            result.update(
                status="escaped",
                reason=f"mutant escaped its intended check {gate['intended_check']} ({gate['intended_reason']})",
            )
    result["accepts_replacement"] = (
        implementation == "Oxide" and result["status"] == "passed"
    )
    result["noise_floor_fraction"] = max(
        (child.get("noise_floor_fraction", 0.0) for child in result["tiers"].values()),
        default=None,
    )


def summary(result: dict) -> str:
    """Summarize retained evidence and every failed gate."""
    lines = [
        f"# CUDA qualification: {result['target']} / {result['implementation']}",
        "",
        f"Status: **{result['status']}**",
        "",
        f"Time: {result['time_central']}",
        "",
        f"Lock digest: `{result['lock_digest']}`",
        "",
        result.get("reason", ""),
        "",
        "Real Turing hardware: untested",
        "",
    ]
    for tier, child in result.get("tiers", {}).items():
        lines += [
            f"## {tier}",
            "",
            f"Declared coverage: `{json.dumps(child.get('coverage_declared'))}`",
            "",
            f"Measurability: `{json.dumps(child.get('measurability'))}`",
            "",
        ]
        if "tf32_band" in child:
            lines += [f"TF32 stage band: `{json.dumps(child['tf32_band'])}`", ""]
    if "mutant_gate" in result:
        gate = result["mutant_gate"]
        lines += [
            f"Mutant gate: `{gate['intended_check']}` with reason `{gate['intended_reason']}`, caught: {gate['caught']}",
            "",
        ]
    lines += ["## Failed checks", ""]
    lines.extend(
        f"- {x['check']}{' (blocked)' if x.get('blocked') else ''}: {x['reason']}"
        for x in result["checks"]
        if not x["passed"]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    """Refuse changed harnesses before GPU work and again before writing results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "target", nargs="?", choices=("resnet", "lstm", "sincnet", "fbankdft")
    )
    parser.add_argument("implementation", nargs="?")
    parser.add_argument("--tier", choices=("sm75", "sm80", "sm90", "sm120"))
    parser.add_argument("--check-table", action="store_true")
    parser.add_argument(
        "--cpu-mode",
        choices=("serial", "parallel", "verify"),
        default="parallel",
        help="CPU evidence policy; verify compares both modes on one in-memory snapshot",
    )
    parser.add_argument(
        "--records",
        type=Path,
        help="re-derive summaries from this outside-tree cache's records/<sha256>",
    )
    parser.add_argument(
        "--table", type=Path, help="check a scratch copy of the evaluated table"
    )
    args = parser.parse_args()
    if args.check_table:
        try:
            with tempfile.TemporaryDirectory(
                prefix="speakrs-production-table-"
            ) as output:
                path = args.table or Path(output) / "table.json"
                if args.table is None:
                    env = dict(
                        clean_environment(), SPEAKRS_QUALIFY_TABLE_OUTPUT=str(path)
                    )
                    run = subprocess.run(
                        [
                            "cargo",
                            "test",
                            "-p",
                            "speakrs",
                            "--no-default-features",
                            "--features",
                            "cuda",
                            "--lib",
                            "--",
                            "--exact",
                            "inference::cuda::implementation::tests::export_production_table",
                        ],
                        cwd=ROOT,
                        env=env,
                        check=False,
                    )
                    if run.returncode or not path.is_file():
                        raise Rejected("table: Rust export failed")
                print(
                    json.dumps(
                        check_table(
                            json.loads(path.read_text()),
                            records=args.records / "records" if args.records else None,
                        ),
                        indent=2,
                    )
                )
            return 0
        except (Rejected, LockError, OSError, KeyError, ValueError) as error:
            print(f"table rejected: {error}", file=sys.stderr)
            return 1
    if args.target is None or args.implementation is None or args.table or args.records:
        parser.error("target and implementation required, or --check-table")
    os.environ["SPEAKRS_QUALIFY_CPU_MODE"] = args.cpu_mode
    tier = args.tier or os.environ.get("SPEAKRS_CUDA_PTX_TIER", "sm75")
    if tier not in ("sm75", "sm80", "sm90", "sm120"):
        parser.error("unknown requested PTX tier")
    expected = os.environ.get("SPEAKRS_QUALIFY_OWNER_DIGEST")
    if not expected:
        print(
            "qualification refused: the owner's outside-tree digest is required",
            file=sys.stderr,
        )
        return 2
    try:
        digest = verify(expected=expected)
    except (LockError, OSError) as error:
        print(f"qualification refused: {error}", file=sys.stderr)
        return 2
    root = ROOT.resolve()
    if (
        sys.platform != "linux"
        or root.name != "tree"
        or root.parent.parent != WORKSPACE
    ):
        print(
            "qualification refused: run from /workspace/<task>/tree on the GPU box",
            file=sys.stderr,
        )
        return 2
    if args.implementation not in ("Library", "Oxide", *MUTANTS, *CONTROL_PHASES):
        print("qualification refused: unknown implementation name", file=sys.stderr)
        return 2
    if (
        args.implementation in ("StageTail", "StageTailControl")
        and args.target == "lstm"
    ):
        print(
            "qualification refused: LSTM StageTail awaits the Library-free candidate (phase 2c-2)",
            file=sys.stderr,
        )
        return 2
    now = datetime.now(UTC)
    stamp = now.strftime("%Y%m%dT%H%M%S.%fZ")
    prefix = f"qualify-{args.target}-{args.implementation}-{stamp}"
    directory = BOX / "results" / prefix
    directory.mkdir(parents=True, exist_ok=False)
    result = {
        "schema": 5,
        "requested_tier": tier,
        "target": args.target,
        "implementation": args.implementation,
        "lock_digest": digest,
        "tree": str(root),
        "time_central": now.astimezone(ZoneInfo("America/Chicago")).isoformat(),
        "status": "blocked",
        "accepts_replacement": False,
        "commands": [],
        "checks": [],
        "real_turing_hardware": "untested",
    }
    try:
        collect(args.target, args.implementation, directory, result)
    except (Rejected, OSError, sqlite3.Error, KeyError, ValueError) as error:
        result["reason"] = str(error)
    try:
        verify_inputs(args.target)
    except (Rejected, OSError, KeyError, ValueError) as error:
        result.update(
            reason=f"inputs changed or are invalid: {error}",
            status="rejected",
            accepts_replacement=False,
        )
    try:
        verify(expected=digest)
    except (LockError, OSError) as error:
        result.update(
            reason=f"harness changed during collection: {error}",
            status="rejected",
            accepts_replacement=False,
        )
    path = BOX / "results" / f"{prefix}.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    path.with_suffix(".md").write_text(summary(result))
    print(f"{result['status']}: {result.get('reason', '')}\n{path}")
    return EXIT_CODES[result["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
