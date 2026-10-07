"""Bind installed Library comparisons to the device and software they used."""

from pathlib import PurePosixPath

from artifacts import capability, device, sha256
from gates import Rejected

VERSIONS = (
    "driver_version",
    "driver_api_version",
    "cuda_version",
    "cudnn_version",
    "cublas_version",
)
FAMILIES = ("libcuda.so", "libcudnn", "libcublas")


def versions(raw: object) -> dict:
    """Require driver and Library versions without clock or process-local attributes"""
    precise = device(raw)
    result = {name: precise.get(name) for name in VERSIONS}
    for name, value in result.items():
        valid = (
            isinstance(value, str) and bool(value.strip())
            if name == "driver_version"
            else type(value) is int and value > 0
        )
        if not valid:
            raise Rejected(f"table: missing environment {name}")
    return result


def libraries(raw: object) -> dict:
    """Keep exact installed driver, cuDNN and cuBLAS file identities"""
    if not isinstance(raw, dict):
        raise Rejected("table: missing installed Library fingerprint")
    result = {}
    for family in FAMILIES:
        found = {
            path: sha256(pin)
            for path, pin in raw.items()
            if isinstance(path, str) and PurePosixPath(path).name.startswith(family)
        }
        if not found:
            raise Rejected(f"table: missing installed fingerprint for {family}")
        result.update(found)
    return result


def loaded_libraries(raw: object) -> dict:
    """Require the actual API-provider files, not an installed-directory scan"""
    if not isinstance(raw, dict) or len(raw) != 3:
        raise Rejected("table: missing loaded Library fingerprint")
    value: dict = dict(raw)
    found = {}
    for family in ("libcuda.so", "libcudnn.so", "libcublas.so"):
        matches = [
            path
            for path in value
            if isinstance(path, str)
            and PurePosixPath(path).is_absolute()
            and PurePosixPath(path).name.startswith(family)
        ]
        if len(matches) != 1:
            raise Rejected(f"table: missing or ambiguous loaded provider for {family}")
        path = matches[0]
        found[path] = sha256(value[path])
    return found


def collect(child: dict, *, legacy: bool, recorded_device: dict) -> dict:
    """Check every numeric control and candidate against the record's environment"""
    identity = {
        "name": recorded_device["name"],
        "compute_capability": recorded_device["compute_capability"],
    }
    if legacy:
        # only exact hash-bound PR #36 records use this pre-version-query mapping
        return {
            "kind": "LegacyInstalledLibraries",
            "device": identity,
            "libraries": libraries(child.get("sanitizer_fingerprint")),
        }
    expected = versions(recorded_device)
    numeric = child.get("numeric", {})
    controls = numeric.get("library", [])
    candidates = numeric.get("candidate", [])
    if not controls or not candidates:
        raise Rejected("table: missing comparison control environment")
    processes = [*controls, *candidates]
    for phase in ("timing_processes", "paired_processes"):
        phase_processes = child.get(phase, [])
        if not isinstance(phase_processes, list):
            raise Rejected("table: invalid comparison process environments")
        processes.extend(phase_processes)
    fingerprint = loaded_libraries(controls[0].get("loaded_libraries"))
    for process in processes:
        precise = device(process.get("device"))
        if (
            versions(precise) != expected
            or {name: precise[name] for name in identity} != identity
            or precise["sm_count"] != recorded_device["sm_count"]
        ):
            raise Rejected("table: stale comparison control environment")
        if loaded_libraries(process.get("loaded_libraries")) != fingerprint:
            raise Rejected("table: comparison controls loaded different Library bytes")
    return {
        "kind": "LoadedLibraries",
        "device": identity,
        "versions": expected,
        "libraries": fingerprint,
    }


def validate(raw: object, *, legacy: bool, recorded_device: dict) -> dict:
    """Validate a locked receipt without accepting an opaque environment object"""
    if not isinstance(raw, dict):
        raise Rejected("table: missing environment receipt")
    value: dict = dict(raw)
    expected_keys = {"kind", "device", "libraries", *([] if legacy else ["versions"])}
    expected_kind = "LegacyInstalledLibraries" if legacy else "LoadedLibraries"
    identity = {name: recorded_device[name] for name in ("name", "compute_capability")}
    capability(identity["compute_capability"])
    if (
        set(value) != expected_keys
        or value.get("kind") != expected_kind
        or value.get("device") != identity
    ):
        raise Rejected("table: invalid environment receipt")
    if not legacy and value.get("versions") != versions(recorded_device):
        raise Rejected("table: stale environment receipt")
    parse = libraries if legacy else loaded_libraries
    if parse(value.get("libraries")) != value["libraries"]:
        raise Rejected("table: invalid installed Library fingerprint")
    return value


def consistent(entries: list[dict]) -> None:
    """Require entries for one device to share the comparison software fingerprint"""
    observed: dict[tuple[str, str], dict] = {}
    for entry in entries:
        receipt = validate(
            entry.get("environment"),
            legacy=entry["legacy"],
            recorded_device=entry["device"],
        )
        identity = receipt["device"]
        key = (identity["name"], identity["compute_capability"])
        prior = observed.get(key)
        if prior is not None and (
            prior["libraries"] != receipt["libraries"]
            or "versions" in prior
            and "versions" in receipt
            and prior["versions"] != receipt["versions"]
        ):
            raise Rejected("table: device entries have different Library fingerprints")
        if prior is None or "versions" in receipt:
            observed[key] = receipt
