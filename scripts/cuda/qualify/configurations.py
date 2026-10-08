"""Bind each qualified tuple to its complete, closed configuration identity."""

from artifacts import sha256
from gates import Rejected

# exact PR #36 records, not a mapping inferred from today's production table
LEGACY_AREAS = {
    "3badc1aec939b0e8f7312786d695bec6445de1dacb1f85e44124bf3ac20356f8": "lstm",
    "a4d1a2692b78a814cd2da16f084801c3641bfd11c6d095d610f3a8182ad6f675": "sincnet",
}
SHAPES = {"C32", "C64", "C32Stride2"}
ENTRIES = SHAPES | {"C64Small", "C32Stride2Small"}


def pin(raw: object, area: str) -> dict:
    """Reject missing fields, open-ended variants and foreign-area configurations"""
    if not isinstance(raw, dict):
        raise Rejected("table: missing configuration pin")
    value: dict = dict(raw)
    if area == "resnet":
        selection = value.get("selection")
        field, allowed = (
            ("shape", SHAPES) if selection == "LegacyWaves" else ("entry", ENTRIES)
        )
        if (
            selection not in ("LegacyWaves", "Kernel")
            or set(value) != {"kind", "selection", field}
            or value.get("kind") != "Conv"
            or not isinstance(value.get(field), str)
            or value.get(field) not in allowed
        ):
            raise Rejected("table: invalid convolution configuration pin")
        return value
    expected = {
        "lstm": {"kind": "Lstm", "selection": "LegacyCooperative"},
        "sincnet": {"kind": "Sinc", "selection": "ConvAbsPool"},
        "fbankdft": {"kind": "Fbank", "selection": "FftMelAccurate"},
    }.get(area)
    if expected is None or value != expected:
        raise Rejected("table: invalid candidate configuration pin")
    return value


def tuples(raw: object, area: str) -> dict[tuple[str, int, str], dict]:
    """Validate a receipt, rejecting duplicate tuples even when their pins agree"""
    if not isinstance(raw, list):
        raise Rejected("table: missing tuple configuration receipts")
    result = {}
    for row in raw:
        if not isinstance(row, dict) or set(row) != {"tuple", "pin"}:
            raise Rejected("table: invalid tuple configuration receipt")
        value: dict = dict(row)
        key = value["tuple"]
        if (
            not isinstance(key, list)
            or len(key) != 3
            or not isinstance(key[0], str)
            or not key[0]
            or type(key[1]) is not int
            or not 1 <= key[1] <= (32 if area == "fbankdft" else 64)
            or key[2] not in ("fp32", "tf32")
        ):
            raise Rejected("table: invalid configuration tuple")
        key = tuple(key)
        if key in result:
            raise Rejected("table: duplicate tuple configuration receipt")
        result[key] = pin(value["pin"], area)
    return result


def canonical(values: dict[tuple[str, int, str], dict]) -> list[dict]:
    """Use deterministic tuple order for immutable acceptance summaries"""
    return [{"tuple": list(key), "pin": value} for key, value in sorted(values.items())]


def legacy(record: str) -> list[dict]:
    """Map only the archived PR #36 tuples to their accepted selection rules"""
    area = LEGACY_AREAS.get(record)
    result = {}
    if area in ("lstm", "sincnet"):
        boundary = "lstm.stack" if area == "lstm" else "sincnet.conv0.abs_pool"
        identity = (
            {"kind": "Lstm", "selection": "LegacyCooperative"}
            if area == "lstm"
            else {"kind": "Sinc", "selection": "ConvAbsPool"}
        )
        result = {(boundary, batch, "fp32"): identity for batch in (1, 32)}
    else:
        raise Rejected("table: missing explicit legacy configuration mapping")
    return canonical(result)


def planned_for_area(raw: object, area: str) -> dict[tuple[str, int, str], dict]:
    """Validate all stage routes, then retain only this record's candidate area"""
    if not isinstance(raw, list):
        raise Rejected("table: missing tuple configuration receipts")
    result = {}
    seen = set()
    for row in raw:
        if not isinstance(row, dict):
            raise Rejected("table: invalid planned configuration")
        value: dict = dict(row)
        identity = value.get("pin")
        if not isinstance(identity, dict):
            raise Rejected("table: invalid planned configuration")
        identity: dict = dict(identity)
        kind = identity.get("kind")
        if not isinstance(kind, str):
            raise Rejected("table: missing planned configuration kind")
        owner = {
            "Conv": "resnet",
            "Lstm": "lstm",
            "Sinc": "sincnet",
            "Fbank": "fbankdft",
        }.get(kind)
        if owner is None:
            raise Rejected("table: invalid planned configuration owner")
        parsed = tuples([value], owner)
        for key, identity in parsed.items():
            if (owner, key) in seen:
                raise Rejected("table: duplicate planned configuration")
            seen.add((owner, key))
            if owner == area:
                result[key] = identity
    return result


def recorded(record: str, child: dict, area: str, *, is_legacy: bool) -> list[dict]:
    """Derive pins from successful plans in every retained numeric process"""
    if is_legacy:
        return legacy(record)
    processes = child.get("numeric", {}).get("candidate", [])
    if not processes:
        raise Rejected("table: missing recorded candidate configurations")
    declaration = {tuple(row) for row in child["coverage_declared"]["triples"]}
    result = {}
    for process in processes:
        mode = process.get("mode")
        if mode not in ("fp32", "tf32"):
            raise Rejected("table: missing configuration process math")
        planned = planned_for_area(process.get("configurations"), area)
        required = {key for key in declaration if key[2] == mode}
        if not required <= planned.keys():
            raise Rejected("table: missing planned configuration for declared tuple")
        for key in required:
            identity = planned[key]
            if key in result and result[key] != identity:
                raise Rejected("table: conflicting recorded configuration pins")
            result[key] = identity
    if result.keys() != declaration:
        raise Rejected("table: missing recorded configuration mode")
    return canonical(result)


def check(entry: dict, recorded: object, selected: set[tuple[str, int, str]]) -> None:
    """Require every production tuple to match the exact pin its record ran"""
    qualified = tuples(recorded, entry["area"])
    requested = tuples(entry.get("configurations"), entry["area"])
    if requested.keys() != selected or any(
        qualified.get(key) != value for key, value in requested.items()
    ):
        raise Rejected("table: production configuration differs from its record pin")


def record_entries(entries: list[dict]) -> list[dict]:
    """Check both immutable records when accuracy and speed evidence differ"""
    expanded = []
    for entry in entries:
        speed = sha256(entry.get("record"))
        accuracy = sha256(entry.get("accuracy_record", speed))
        expanded.append(entry)
        if accuracy != speed:
            expanded.append({**entry, "record": accuracy, "accuracy_record": accuracy})
    return expanded
