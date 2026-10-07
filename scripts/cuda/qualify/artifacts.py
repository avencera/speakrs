"""Validate loader evidence against embedded-feature masks and committed bytes."""

import hashlib
import re
from pathlib import Path

from gates import Rejected

TIERS = {"sm75": (7, 5), "sm80": (8, 0), "sm90": (9, 0), "sm120": (12, 0)}


def sha256(value: object) -> str:
    """Require a canonical actual-byte identity"""
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise Rejected("artifact: invalid SHA-256")
    return value


def capability(value: object) -> tuple[int, int]:
    """Reject malformed capabilities before architecture or feature selection"""
    if not isinstance(value, str) or not re.fullmatch(r"[1-9][0-9]*\.[0-9]", value):
        raise Rejected("artifact: invalid device capability")
    major, minor = value.split(".")
    return int(major), int(minor)


def device(raw: object) -> dict:
    """Require precise physical device and installed driver evidence for new records"""
    if not isinstance(raw, dict):
        raise Rejected("target: missing device evidence")
    result = dict(raw)
    for field in ("name", "driver_version"):
        if not isinstance(result.get(field), str) or not result[field].strip():
            raise Rejected(f"target: missing device {field}")
    capability(result.get("compute_capability"))
    for field in ("sm_count", "l2_bytes", "driver_api_version"):
        if type(result.get(field)) is not int or result[field] <= 0:
            raise Rejected(f"target: missing positive device {field}")
    return result


def key(raw: object, *, device_capability: str) -> dict:
    """Validate the typed selection key without treating build metadata as a key"""
    if not isinstance(raw, dict):
        raise Rejected("artifact: missing loaded artifact")
    value = dict(raw)
    kind = value.get("kind")
    if kind == "PtxJit" and set(value) == {"kind", "sha256"}:
        return {"kind": kind, "sha256": sha256(value["sha256"])}
    if kind == "Cubin" and set(value) == {"kind", "arch", "sha256"}:
        if capability(value["arch"]) != capability(device_capability):
            raise Rejected("artifact: cubin architecture differs from exact device")
        return {"kind": kind, "arch": value["arch"], "sha256": sha256(value["sha256"])}
    raise Rejected("artifact: invalid selection key fields")


def file_hash(path: Path) -> str:
    """Hash only regular, non-symbolic shipped artifacts"""
    if path.is_symlink() or not path.is_file():
        raise Rejected(f"artifact: unavailable regular file: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module(raw: dict, root: Path, *, device_capability: str | None = None) -> dict:
    """Bind actual embedded PTX and successful cubin bytes to shipped build pins"""
    ptx_hash = sha256(raw.get("sha256"))
    if sha256(raw.get("embedded_ptx_sha256")) != ptx_hash:
        raise Rejected("artifact: embedded PTX identity differs from pinned PTX")
    artifact = raw.get("artifact")
    if not isinstance(artifact, dict):
        raise Rejected("artifact: missing loaded artifact")
    pin = dict(artifact)
    if pin.get("kind") == "PtxJit":
        result = key(pin, device_capability=device_capability or "1.0")
        if result["sha256"] != ptx_hash:
            raise Rejected("artifact: PTX JIT identity differs from embedded PTX")
        return result
    if set(pin) != {
        "kind",
        "arch",
        "sha256",
        "ptx_sha256",
        "ptxas_version",
        "ptxas_flags",
    }:
        raise Rejected("artifact: invalid cubin provenance fields")
    result = key(
        {name: pin[name] for name in ("kind", "arch", "sha256")},
        device_capability=device_capability or pin["arch"],
    )
    if sha256(pin["ptx_sha256"]) != ptx_hash:
        raise Rejected("artifact: cubin source PTX differs from embedded PTX")
    area, tier = raw.get("area"), raw.get("tier")
    if (
        area
        not in (
            "probe",
            "fbank",
            "embedding",
            "segmentation",
            "resnet",
            "lstm",
            "sincnet",
        )
        or tier not in TIERS
    ):
        raise Rejected("artifact: invalid cubin area or tier")
    arch = "sm_" + pin["arch"].replace(".", "")
    binary = root / f"src/inference/cuda/ptx/{area}.{tier}.{arch}.cubin"
    if file_hash(binary) != result["sha256"]:
        raise Rejected("artifact: loaded cubin differs from shipped bytes")
    manifest = root / f"src/inference/cuda/ptx/{area}.manifest"
    file_hash(manifest)
    header: dict[str, str] = {}
    sections: dict[str, dict[str, str]] = {}
    fields = header
    for line in manifest.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if re.fullmatch(r"\[sm[0-9]+\]", line):
            name = line[1:-1]
            if name in sections:
                raise Rejected("artifact: duplicate manifest tier")
            fields = sections.setdefault(name, {})
            continue
        name, separator, value = line.partition(" = ")
        if not separator or name in fields:
            raise Rejected("artifact: invalid or duplicate manifest field")
        fields[name] = value
    section = sections.get(tier, {})
    if (
        header.get("ptxas") != pin["ptxas_version"]
        or header.get("ptxas-flags") != pin["ptxas_flags"]
        or not pin["ptxas_version"]
        or not pin["ptxas_flags"]
        or section.get("ptx") != ptx_hash
        or section.get("cubin." + arch) != f"{result['sha256']} {ptx_hash}"
    ):
        raise Rejected("artifact: cubin build provenance differs from manifest")
    return result


EMBED = re.compile(
    r'tier_ptx!\(\s*\[(?P<features>[^\]]+)\]\s*,\s*"ptx/(?P<area>[a-z]+)\.(?P<tier>sm[0-9]+)"\s*,\s*\[(?P<arches>[0-9,\s]+)\]\s*\)',
    re.MULTILINE,
)


def production_load(area: str, tier: str, device_capability: str, root: Path) -> None:
    """Follow the checked AreaPtx feature masks, never directory-name availability.

    A production binding names its tier and loads exactly that variant, so some GPU
    feature build the device supports must embed the tier within its own limit. A
    higher embedded variant does not displace the binding's tier.
    """
    device_cc = capability(device_capability)
    path = root / "src/inference/cuda/kernels.rs"
    file_hash(path)
    # kernels.rs is a locked owner; xtask also checks this exact macro syntax
    source = re.sub(r"/\*.*?\*/|//[^\n]*", "", path.read_text(), flags=re.DOTALL)
    variants: dict[str, set[str]] = {}
    for match in EMBED.finditer(source):
        if match["area"] != area:
            continue
        variant = match["tier"]
        features = re.findall(r'"([^"]+)"', match["features"])
        if (
            variant not in TIERS
            or variant in variants
            or not features
            or len(set(features)) != len(features)
            or any(feature.removeprefix("cuda-") not in TIERS for feature in features)
            or re.sub(r'"[^"]+"|[,\s]', "", match["features"])
        ):
            raise Rejected("table: invalid production feature embed mask")
        variants[variant] = set(features)
        file_hash(root / f"src/inference/cuda/ptx/{area}.{variant}.ptx")
    if tier not in variants:
        raise Rejected(
            "table: pinned production PTX tier is unavailable in embed masks"
        )
    for build, minimum in TIERS.items():
        if minimum > device_cc:
            continue
        if "cuda-" + build in variants[tier] and TIERS[tier] <= minimum:
            return
    raise Rejected(
        "table: production load cannot realize the pinned tier on the device"
    )


def shipped_key(area: str, tier: str, capability: str, raw: object, root: Path) -> dict:
    """Check shipped cubin bytes; module and record bindings separately verify JIT PTX."""
    pinned = key(raw, device_capability=capability)
    if area not in ("resnet", "lstm", "sincnet") or tier not in TIERS:
        raise Rejected("artifact: invalid production area or tier")
    suffix = (
        "ptx"
        if pinned["kind"] == "PtxJit"
        else f"sm_{capability.replace('.', '')}.cubin"
    )
    path = root / "src/inference/cuda/ptx" / f"{area}.{tier}.{suffix}"
    if pinned["kind"] == "Cubin" and file_hash(path) != pinned["sha256"]:
        raise Rejected("artifact: production pin differs from shipped file")
    return pinned
