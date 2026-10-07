"""Attribute GPU work through CUDA launch correlation, never GPU timestamps.

The locked driver and dispatch push NVTX ranges named `qualify.<nonce>.<kind>.<name>`,
where the nonce is generated per run by the harness and is not readable by candidate
code. Kinds: `window` (between an input upload and its output download), `candidate`
(a candidate's enqueue for one declared pair), `library` (the Library path of a
boundary), `call` (a locked cuDNN or cuBLAS call site), `projection` (the locked LSTM
input-projection helper), `fixed` (a locked launch of a Library-owned kernel) and
`plan` (a candidate's plan creation, where kernels must be allow-listed too).

Inside a window every kernel, copy and memset must be launched inside one of those
scopes, on the qualification stream. Inside a candidate scope a kernel must be an
entry of the PTX bytes the process loaded. Candidates may make no Library calls,
including input projections. Library paths may not launch candidate kernels.
"""

import re
import sqlite3
from bisect import bisect_right
from collections import Counter, defaultdict
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path

from gates import Rejected

KINDS = (
    "window",
    "candidate",
    "library",
    "call",
    "projection",
    "fixed",
    "plan",
    "phase",
)
# every LSTM candidate stack opens these locked phase scopes exactly as named
LSTM_PHASES = tuple(
    f"{kind}.L{layer}.{direction}"
    for kind in ("input_proj", "recurrence")
    for layer in range(4)
    for direction in ("forward", "reverse")
)
# CUPTI copy kinds that stay on the device: device to device, peer to peer
DEVICE_COPIES = {8, 10}
LIBRARY_KERNEL = re.compile(r"cublas|cudnn|cufft|xmma|(?:^|_)s?gemm", re.IGNORECASE)


@dataclass(frozen=True)
class Range:
    """A closed, single-thread harness NVTX scope."""

    kind: str
    name: str
    start: int
    end: int
    tid: int


@dataclass(frozen=True)
class AllowList:
    """Entry names of the PTX bytes the candidate process loaded."""

    entries: frozenset[str]
    candidate_entries: frozenset[str]


@dataclass
class Scope:
    """Evidence gathered for one candidate or library scope name."""

    kernels: Counter = field(default_factory=Counter)
    library_kernels: Counter = field(default_factory=Counter)
    events: int = 0


class ContainmentIndex:
    """Index nested launch ranges without comparing every kernel to every range."""

    def __init__(self, ranges: list[Range], process: bool = False):
        groups = defaultdict(list)
        for index, scope in enumerate(ranges):
            key = scope.tid & ~0xFFFFFF if process else scope.tid
            groups[key].append((scope.start, scope.end, index))
        self.groups = {}
        for key, values in groups.items():
            values.sort()
            maximum = 0
            prefix = []
            for _, end, _ in values:
                maximum = max(maximum, end)
                prefix.append(maximum)
            self.groups[key] = ([start for start, _, _ in values], values, prefix)

    def contains(self, key: int, start: int, end: int) -> set[int]:
        """Return every enclosing interval, including a stack and its child."""
        if key not in self.groups:
            return set()
        starts, values, prefix = self.groups[key]
        position = bisect_right(starts, start) - 1
        found = set()
        while position >= 0 and prefix[position] >= end:
            _, stop, index = values[position]
            if end <= stop:
                found.add(index)
            position -= 1
        return found


def attribute(
    path: Path,
    required: tuple[str, ...],
    nonce: str,
    allow: AllowList,
    declared: frozenset[str] = frozenset(),
    library_control: bool = False,
) -> dict:
    """Apply the window, scope, allow-list and stream rules to an eager trace.

    `required` lists the boundary names; `declared` the ones a candidate runs in at
    least one case. A Library control may have no candidate scopes.
    """
    if not required or len(set(required)) != len(required):
        raise Rejected("profile: invalid layer inventory")
    if not re.fullmatch(r"[0-9a-f]{32}", nonce):
        raise Rejected("profile: invalid nonce")
    if not path.is_file():
        raise Rejected("profile: missing SQLite export")
    try:
        with closing(
            sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
        ) as connection:
            connection.row_factory = sqlite3.Row
            return _attribute(
                connection,
                required,
                nonce,
                allow,
                declared,
                library_control,
            )
    except (sqlite3.Error, IndexError, KeyError) as error:
        raise Rejected(f"profile: invalid SQLite trace: {error}") from error


def _columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in connection.execute(f'PRAGMA table_info("{table}")')}


def _ranges(connection, names: dict, nonce: str) -> list[Range]:
    pattern = re.compile(
        r"qualify\.([0-9a-f]{32})\.(" + "|".join(KINDS) + r")\.(.+)", re.DOTALL
    )
    ranges = []
    for row in connection.execute("SELECT * FROM NVTX_EVENTS"):
        text = row["text"] or names.get(row["textId"])
        if not text or not text.startswith("qualify."):
            continue
        match = pattern.fullmatch(text)
        if match is None or match[1] != nonce:
            raise Rejected(
                f"profile: harness range without this run's nonce: {text[:80]}"
            )
        if row["end"] is None or row["end"] <= row["start"] or row["globalTid"] is None:
            raise Rejected(f"profile: unclosed range {text}")
        if row["endGlobalTid"] not in (None, row["globalTid"]):
            raise Rejected(f"profile: cross-thread range {text}")
        ranges.append(
            Range(match[2], match[3], row["start"], row["end"], row["globalTid"])
        )
    return ranges


def _launches(connection, tables: set[str]) -> dict:
    launches = defaultdict(list)
    for table in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"):
        if table not in tables:
            continue
        for row in connection.execute(f"SELECT * FROM {table}"):
            if row["correlationId"] is None or row["globalTid"] is None:
                continue
            # Nsight stores globalPid as globalTid with its 24-bit thread field cleared
            pid = row["globalTid"] & ~0xFFFFFF
            launches[pid, row["correlationId"]].append(row)
    return launches


def _events(connection, tables: set[str], names: dict):
    """Kernels, copies and memsets with their kind, names and stream."""
    kernel_columns = _columns(connection, "CUPTI_ACTIVITY_KIND_KERNEL")
    for row in connection.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL"):
        demangled = names.get(row["demangledName"])
        mangled = (
            names.get(row["mangledName"]) if "mangledName" in kernel_columns else None
        )
        if not demangled or row["end"] <= row["start"]:
            raise Rejected("profile: unnamed or invalid kernel event")
        if row["graphNodeId"] not in (None, 0):
            raise Rejected(
                "profile: graph node has no direct per-layer launch range; use an eager trace"
            )
        yield row, "kernel", {demangled, mangled} - {None}, None
    for table, kind in (
        ("CUPTI_ACTIVITY_KIND_MEMCPY", "copy"),
        ("CUPTI_ACTIVITY_KIND_MEMSET", "memset"),
    ):
        if table not in tables:
            continue
        columns = _columns(connection, table)
        for row in connection.execute(f"SELECT * FROM {table}"):
            copy_kind = row["copyKind"] if "copyKind" in columns else None
            yield row, kind, set(), copy_kind


def _forbidden_calls(ranges: list[Range]) -> list[str]:
    """Capture records API calls, not eager launches; retain their real ownership."""
    candidates = [scope for scope in ranges if scope.kind == "candidate"]
    plans = [scope for scope in ranges if scope.kind == "plan"]
    libraries = [scope for scope in ranges if scope.kind == "library"]
    return [
        call.name
        for call in ranges
        if call.kind == "call"
        and (
            any(_inside(call, candidate) for candidate in candidates)
            or (
                any(_inside(call, plan) for plan in plans)
                and not any(_inside(call, library) for library in libraries)
            )
        )
    ]


def _attribute(
    connection: sqlite3.Connection,
    required: tuple[str, ...],
    nonce: str,
    allow: AllowList,
    declared: frozenset[str],
    library_control: bool,
) -> dict:
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    if not {"StringIds", "NVTX_EVENTS", "CUPTI_ACTIVITY_KIND_KERNEL"} <= tables:
        raise Rejected("profile: missing NVTX or kernel tables")
    names = dict(connection.execute("SELECT id, value FROM StringIds"))
    ranges = _ranges(connection, names, nonce)
    forbidden = _forbidden_calls(ranges)
    if forbidden:
        raise Rejected(
            f"profile: forbidden library kernels in a candidate scope: real API calls {forbidden[:8]}"
        )
    # a stage also runs the other boundaries of its model on their Library paths
    for range_ in ranges:
        if range_.kind == "candidate" and range_.name not in required:
            raise Rejected(
                f"profile: candidate scope for another boundary: {range_.name}"
            )
    windows = [index for index, scope in enumerate(ranges) if scope.kind == "window"]
    if not windows:
        raise Rejected("profile: no driver windows")
    scope_index = ContainmentIndex(ranges)
    process_index = ContainmentIndex([ranges[index] for index in windows], process=True)
    launches = _launches(connection, tables)
    if not launches:
        raise Rejected("profile: missing CUDA launch correlation")

    violations: dict[str, list[str]] = defaultdict(list)
    scopes: dict[tuple[str, str], Scope] = defaultdict(Scope)
    streams: Counter = Counter()
    candidate_streams = []
    # a side stream is valid only when every event on it is candidate work
    stream_events: Counter = Counter()
    stream_candidate_events: Counter = Counter()
    phase_events: Counter = Counter()
    window_events: Counter = Counter({ranges[index].name: 0 for index in windows})
    for row, kind, event_names, copy_kind in _events(connection, tables, names):
        linked = launches.get((row["globalPid"], row["correlationId"]), [])
        if not linked:
            raise Rejected(f"profile: {kind} has no CUDA launch correlation")
        owners = set()
        for launch in linked:
            owners.update(
                scope_index.contains(
                    launch["globalTid"], launch["start"], launch["end"]
                )
            )
        label = ", ".join(sorted(event_names)) or f"{kind} {copy_kind}"
        stream_events[row["streamId"]] += 1
        if any(ranges[index].kind in ("candidate", "plan") for index in owners):
            stream_candidate_events[row["streamId"]] += 1
        for index in owners:
            if ranges[index].kind == "window" and kind == "kernel":
                window_events[ranges[index].name] += 1
            if ranges[index].kind == "phase" and kind == "kernel":
                phase_events[index] += 1
        in_plan = any(ranges[index].kind == "plan" for index in owners)
        if in_plan and kind == "kernel" and not event_names <= allow.entries:
            violations["kernel is not on the loaded PTX allow-list"].append(label)
            continue
        in_window = any(ranges[index].kind == "window" for index in owners)
        if not in_window:
            if any(
                process_index.contains(
                    launch["globalTid"] & ~0xFFFFFF, launch["start"], launch["end"]
                )
                for launch in linked
            ):
                violations["cross-thread work during a window"].append(kind)
            if not owners:
                if kind == "kernel":
                    reason = (
                        "forbidden library kernels outside checked scopes"
                        if any(LIBRARY_KERNEL.search(name) for name in event_names)
                        else "work outside a candidate or library range"
                    )
                    violations[reason].append(label)
                # host setup transfers are not candidate execution
                continue
            if in_plan:
                continue
            if any(ranges[index].kind == "candidate" for index in owners):
                violations[
                    "candidate execution outside a checked lifecycle window"
                ].append(label)
            # locked Library front ends can run outside candidate windows; their
            # call/fixed ownership still has to pass the rules below

        held = [ranges[index] for index in owners if ranges[index].kind != "window"]
        if not held:
            violations["work outside a candidate or library range"].append(label)
            continue
        kinds = {scope.kind for scope in held}
        stream = row["streamId"]
        candidate = [scope for scope in held if scope.kind == "candidate"]
        library = [scope for scope in held if scope.kind == "library"]
        calls = [scope for scope in held if scope.kind == "call"]
        for scope in candidate + library:
            evidence = scopes[scope.kind, scope.name]
            evidence.events += 1
            if kind == "kernel":
                evidence.kernels[label] += 1
                if calls:
                    evidence.library_kernels[label] += 1
        # vendor libraries may use internal streams inside their own call scopes
        if candidate and not calls:
            candidate_streams.append((stream, label))
        elif not calls:
            streams[stream] += 1
        if candidate and calls:
            violations["forbidden library kernels in a candidate scope"].append(label)
            continue
        if candidate and kind == "kernel" and not event_names <= allow.entries:
            reason = (
                "forbidden library kernels in a candidate scope"
                if any(LIBRARY_KERNEL.search(name) for name in event_names)
                else "kernel is not on the loaded PTX allow-list"
            )
            violations[reason].append(label)
            continue
        if candidate and kind == "copy" and copy_kind not in DEVICE_COPIES:
            violations["host transfer inside a candidate scope"].append(label)
            continue
        if not candidate and library and event_names & allow.candidate_entries:
            violations["candidate kernel on a Library path"].append(label)
            continue
        if kinds == {"library"} and kind == "kernel":
            violations["work outside a candidate or library range"].append(label)
            continue
        if (
            not candidate
            and "fixed" in kinds
            and kind == "kernel"
            and not event_names <= allow.entries
        ):
            violations["kernel is not on the loaded PTX allow-list"].append(label)

    side_streams = {
        stream
        for stream, count in stream_events.items()
        if stream_candidate_events[stream] == count
    }
    if streams:
        qualification = streams.most_common(1)[0][0]
        for stream in streams:
            if stream != qualification:
                violations["work on another stream"].append(f"stream {stream}")
        for stream, label in candidate_streams:
            if stream != qualification and stream not in side_streams:
                violations["work on another stream"].append(label)
    _phase_rules(ranges, phase_events, required, library_control, violations)

    present = {
        name
        for (kind, name), evidence in scopes.items()
        if kind in ("candidate", "library") and evidence.kernels
    }
    missing = set(required) - present
    if missing:
        violations["boundary has no scope with correlated kernels"].extend(
            sorted(missing)
        )
    candidate_names = {name for kind, name in scopes if kind == "candidate"}
    candidate_ranges = {scope.name for scope in ranges if scope.kind == "candidate"}
    if library_control and candidate_ranges:
        violations["candidate scope in a Library control"].extend(
            sorted(candidate_ranges)
        )
    if not library_control and declared - candidate_names:
        violations["declared boundary has no candidate kernels"].extend(
            sorted(declared - candidate_names)
        )
    if candidate_ranges - candidate_names:
        violations["empty candidate scope"].extend(
            sorted(candidate_ranges - candidate_names)
        )
    if candidate_ranges - declared and not library_control:
        violations["candidate scope for an undeclared boundary"].extend(
            sorted(candidate_ranges - declared)
        )
    if library_control and not any(
        evidence.library_kernels for evidence in scopes.values()
    ):
        violations["library control exposed no library kernel"].append("none")

    if violations:
        summary = "; ".join(
            f"{reason} ({len(items)}): {sorted(set(items))[:6]}"
            for reason, items in sorted(violations.items())
        )
        raise Rejected(f"profile: {summary}")
    return {
        "scopes": [
            {
                "kind": kind,
                "name": name,
                "events": evidence.events,
                "kernels": dict(evidence.kernels.most_common(16)),
                "library_kernels": dict(evidence.library_kernels.most_common(16)),
            }
            for (kind, name), evidence in sorted(scopes.items())
        ],
        "windows": len(windows),
        "window_kernel_events": dict(sorted(window_events.items())),
        "attribution": "CPU launch correlation and same-thread nonce-scoped NVTX containment",
        "eager": True,
    }


def window_kernels(path: Path, nonce: str, *, multiset: bool = False) -> dict:
    """The candidate kernels each driver window launched, by window name.

    A kernel counts when a candidate scope and the window enclose its launch and no
    library call scope does; the mangled name is used when the trace has one, as the
    captured-graph check names kernels by their mangled entry name.
    """
    with closing(
        sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    ) as connection:
        connection.row_factory = sqlite3.Row
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        names = dict(connection.execute("SELECT id, value FROM StringIds"))
        ranges = _ranges(connection, names, nonce)
        index = ContainmentIndex(ranges)
        launches = _launches(connection, tables)
        columns = _columns(connection, "CUPTI_ACTIVITY_KIND_KERNEL")
        counts = defaultdict(Counter)
        found = defaultdict(set)
        for row in connection.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL"):
            owners = set()
            for launch in launches.get((row["globalPid"], row["correlationId"]), []):
                owners.update(
                    index.contains(launch["globalTid"], launch["start"], launch["end"])
                )
            kinds = {ranges[owner].kind for owner in owners}
            if "candidate" not in kinds or "call" in kinds:
                continue
            name = (
                names.get(row["mangledName"]) if "mangledName" in columns else None
            ) or names.get(row["demangledName"])
            for owner in owners:
                if (
                    ranges[owner].kind == "window"
                    and name
                    and not ranges[owner].name.startswith("lifecycle/")
                ):
                    if row["graphNodeId"] not in (None, 0):
                        raise Rejected(
                            "profile: graph node cannot count as an eager launch"
                        )
                    if multiset:
                        counts[ranges[owner].name][name] += 1
                    else:
                        found[ranges[owner].name].add(name)
        return dict(counts) if multiset else dict(found)


def _inside(inner: Range, outer: Range) -> bool:
    return (
        inner.tid == outer.tid and outer.start <= inner.start and inner.end <= outer.end
    )


def _phase_rules(
    ranges: list[Range],
    phase_events: Counter,
    required: tuple[str, ...],
    library_control: bool,
    violations: dict[str, list[str]],
) -> None:
    """Phase scopes sit inside a candidate scope; LSTM stacks open all 16 phases.

    The projection helper must run inside the `input_proj` phase of the same layer and
    direction. The 16 LSTM phases of one stack may not overlap and each must launch.
    """
    candidates = [scope for scope in ranges if scope.kind == "candidate"]
    phases = [
        (index, scope) for index, scope in enumerate(ranges) if scope.kind == "phase"
    ]
    for _, phase in phases:
        if not any(_inside(phase, candidate) for candidate in candidates):
            violations["phase scope outside a candidate scope"].append(phase.name)
    for projection in (scope for scope in ranges if scope.kind == "projection"):
        if not any(
            phase.name == f"input_proj.{projection.name}" and _inside(projection, phase)
            for _, phase in phases
        ):
            violations["projection outside its input_proj phase"].append(
                projection.name
            )
    if library_control or required != ("lstm.stack",):
        return
    for stack in candidates:
        inner = [
            (index, phase)
            for index, phase in phases
            if _inside(phase, stack) and phase.name in LSTM_PHASES
        ]
        missing = set(LSTM_PHASES) - {phase.name for _, phase in inner}
        if missing:
            violations["LSTM stack lacks locked phase scopes"].extend(sorted(missing))
        for position, (_, a) in enumerate(inner):
            for _, b in inner[position + 1 :]:
                if a.start < b.end and b.start < a.end:
                    violations["overlapping LSTM phase scopes"].append(
                        f"{a.name}/{b.name}"
                    )
        for index, phase in inner:
            if not phase_events[index]:
                violations["LSTM phase scope without a launch"].append(phase.name)
