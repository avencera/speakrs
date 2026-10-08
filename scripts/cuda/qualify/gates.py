"""Fixed qualification gates. Missing or non-finite evidence never passes."""

import math
import re
import statistics
import struct
from collections.abc import Sequence
from dataclasses import dataclass


class Rejected(ValueError):
    """A candidate does not meet the fixed qualification contract."""


class Blocked(Rejected):
    """The evidence cannot decide the check, so it is neither a pass nor a rejection."""


class ParityRejected(Rejected):
    """Layer parity failed; `failing` names the cases over a per-case limit."""

    def __init__(self, message: str, failing: list[str], cases: list[str]):
        super().__init__(message)
        self.failing = failing
        self.cases = cases


@dataclass(frozen=True)
class Error:
    """Reference-relative L2 error and maximum absolute error in FP64."""

    relative_l2: float
    max_abs: float


def finite(values: Sequence[float]) -> None:
    """Reject empty data and all non-finite boundary values."""
    if not values or any(not math.isfinite(value) for value in values):
        raise Rejected("empty or non-finite evidence")


def error(actual: Sequence[float], reference: Sequence[float]) -> Error:
    """Compute errors without an arbitrary absolute tolerance floor."""
    finite(actual)
    finite(reference)
    if len(actual) != len(reference):
        raise Rejected("tensor length mismatch")
    differences = [float(a) - float(r) for a, r in zip(actual, reference, strict=True)]
    numerator = math.hypot(*differences)
    denominator = math.hypot(*reference)
    if denominator == 0:
        if numerator != 0:
            raise Rejected("nonzero output against exact zero reference")
        return Error(0.0, 0.0)
    return Error(numerator / denominator, max(map(abs, differences)))


def ratio(candidate: float, library: float) -> float:
    """Treat zero library error as an exact equality requirement."""
    finite([candidate, library])
    if candidate < 0 or library < 0:
        raise Rejected("negative error")
    if library == 0:
        if candidate != 0:
            raise Rejected("candidate exceeds exact library error")
        return 1.0
    return candidate / library


def layer_parity(cases: Sequence[tuple[str, Error, Error]]) -> dict:
    """Enforce per-case limits and both aggregate error-ratio limits.

    Every case is evaluated before failing, so the failure names all cases over a
    per-case limit, not only the first.
    """
    if not cases:
        raise Rejected("no layer cases")
    l2, absolute, failing = [], [], []
    for case, candidate, library in cases:
        l2.append(ratio(candidate.relative_l2, library.relative_l2))
        absolute.append(ratio(candidate.max_abs, library.max_abs))
        if l2[-1] > 1.10 or absolute[-1] > 2.00:
            failing.append(case)
    if failing:
        raise ParityRejected(
            f"layer parity: per-case error ratio in {len(failing)} cases: {failing[:8]}",
            failing,
            [case for case, _, _ in cases],
        )
    means = [geometric_mean(values) for values in [l2, absolute]]
    if any(value > 1.00 for value in means):
        raise ParityRejected(
            "layer parity: geometric mean exceeds 1.00",
            [],
            [case for case, _, _ in cases],
        )
    return {
        "relative_l2_ratios": l2,
        "max_abs_ratios": absolute,
        "geometric_means": means,
    }


def geometric_mean(values: Sequence[float]) -> float:
    """Use logs for positive values and retain exact zeros."""
    finite(values)
    if any(value < 0 for value in values):
        raise Rejected("negative ratio")
    if 0 in values:
        return 0.0
    return math.exp(math.fsum(math.log(value) for value in values) / len(values))


def minimum_cosine(
    actual: Sequence[float], reference: Sequence[float], width: int
) -> float:
    """Compute the worst embedding cosine without accepting undefined rows."""
    finite(actual)
    finite(reference)
    if width <= 0 or len(actual) != len(reference) or len(actual) % width:
        raise Rejected("invalid embedding shape")
    cosines = []
    for start in range(0, len(actual), width):
        a, r = actual[start : start + width], reference[start : start + width]
        norms = math.hypot(*a) * math.hypot(*r)
        if norms == 0:
            raise Rejected("undefined embedding cosine")
        cosines.append(math.fsum(x * y for x, y in zip(a, r, strict=True)) / norms)
    return min(cosines)


def embedding_parity(candidate: float, library: float) -> None:
    """Keep the library minimum cosine, with only the specified 1e-9 margin."""
    finite([candidate, library])
    if (
        not -1 <= candidate <= 1.000000000000001
        or not -1 <= library <= 1.000000000000001
    ):
        raise Rejected("invalid cosine")
    if candidate < library - 1e-9:
        raise Rejected("embedding stage parity")


def flips(actual: Sequence[float], reference: Sequence[float], classes: int = 7) -> int:
    """Count frame argmax changes using first-index ties, as in production."""
    finite(actual)
    finite(reference)
    if classes <= 0 or len(actual) != len(reference) or len(actual) % classes:
        raise Rejected("invalid logits shape")
    return sum(
        max(range(classes), key=actual[start : start + classes].__getitem__)
        != max(range(classes), key=reference[start : start + classes].__getitem__)
        for start in range(0, len(actual), classes)
    )


def segmentation_parity(
    candidate: Error, library: Error, candidate_flips: int, library_flips: int
) -> None:
    """Bound both logits errors and the number of argmax changes."""
    if candidate_flips < 0 or library_flips < 0:
        raise Rejected("invalid argmax count")
    if (
        ratio(candidate.relative_l2, library.relative_l2) > 1.10
        or ratio(candidate.max_abs, library.max_abs) > 1.10
    ):
        raise Rejected("segmentation stage parity: logits")
    if candidate_flips > library_flips:
        raise Rejected("segmentation stage parity: argmax")


def determinism(first: bytes, second: bytes) -> None:
    """Compare raw F32 bytes, including signed zero, instead of float equality."""
    if not first or len(first) % 4 or len(first) != len(second):
        raise Rejected("invalid determinism tensors")
    finite([x[0] for x in struct.iter_unpack("<f", first)])
    finite([x[0] for x in struct.iter_unpack("<f", second)])
    if first != second:
        raise Rejected("determinism: bitwise mismatch")


@dataclass(frozen=True)
class Timing:
    """One fresh process with warm-up and burst-timed per-launch samples."""

    pid: int
    implementation: str
    warmup: int
    milliseconds: tuple[float, ...]


# largest Library-vs-Library process spread at which a case can show a 0% slowdown
STAGE_SPREAD_BOUND = 0.003
OPERATOR_SPREAD_BOUND = 0.010


def spread_bound(case: str) -> float:
    """Stages use the tighter bound; every other boundary is an operator."""
    return STAGE_SPREAD_BOUND if case.endswith("/stage") else OPERATOR_SPREAD_BOUND


def _speed_evidence(runs: Sequence[Timing], candidate: str, bound: float) -> dict:
    """Validate four fresh A/B/A/B processes; block when the Library pair is noisy."""
    if len(runs) != 4 or len({run.pid for run in runs}) != 4:
        raise Rejected("speed: four fresh processes required")
    expected = ["Library", candidate, "Library", candidate]
    for run, implementation in zip(runs, expected, strict=True):
        if (
            run.implementation != implementation
            or run.warmup < 5
            or len(run.milliseconds) < 20
        ):
            raise Rejected("speed: invalid A/B/A/B sample coverage")
        finite(run.milliseconds)
        if min(run.milliseconds) <= 0:
            raise Rejected("speed: non-positive timing")
    medians = [statistics.median(run.milliseconds) for run in runs]
    library = (medians[0], medians[2])
    spread = abs(library[0] - library[1]) / min(library)
    evidence = {
        "medians_ms": medians,
        "speedups": [medians[0] / medians[1], medians[2] / medians[3]],
        "library_process_spread_fraction": spread,
        "spread_bound": bound,
        "sample_spread_ms": [
            max(run.milliseconds) - min(run.milliseconds) for run in runs
        ],
    }
    if spread > bound:
        raise Blocked(
            f"speed: blocked: Library process spread {spread:.6f} exceeds bound {bound}"
        )
    return evidence


def speed(runs: Sequence[Timing], candidate: str, bound: float) -> dict:
    """The strict operator rule: no slowdown against the faster Library process.

    The case is blocked when the two Library processes disagree by more than
    `bound`, because then the measurement cannot show a 0% slowdown. Otherwise the
    candidate must win both A/B pairs, and each candidate median must be at most the
    faster Library median.
    """
    evidence = _speed_evidence(runs, candidate, bound)
    medians = evidence["medians_ms"]
    if any(value < 1.00 for value in evidence["speedups"]) or max(
        medians[1], medians[3]
    ) > min(medians[0], medians[2]):
        raise Rejected(
            "speed: candidate slower than the faster Library process or in an A/B pair"
        )
    return {**evidence, "gate": "operator: strict, faster Library median"}


def stage_speed(runs: Sequence[Timing], candidate: str, bound: float) -> dict:
    """The stage regression guard for a stage that runs declared candidate layers.

    A stage saving can sit at the stage noise, so each A/B pair only has to stay
    within `bound` of its Library process, and the mean of the two pair speedups must
    still be at least 1, so one case cannot be offset by another. The operator gate is
    the 0% test.
    """
    evidence = _speed_evidence(runs, candidate, bound)
    medians = evidence["medians_ms"]
    if medians[1] > medians[0] * (1 + bound) or medians[3] > medians[2] * (1 + bound):
        raise Rejected(
            f"speed: stage candidate slower than its Library process plus the {bound} stage bound"
        )
    if statistics.fmean(evidence["speedups"]) < 1.0:
        raise Rejected("speed: stage candidate slower on average in this case")
    return {
        **evidence,
        "ratio": statistics.fmean(evidence["speedups"]),
        "gate": "stage: regression guard within the stage bound",
    }


def tf32_band(rows: Sequence[dict], embedding: bool) -> dict:
    """Measure the TF32 stage noise band from seeded 1-ulp Library perturbations.

    Each row has the unperturbed Library `metrics` and the perturbed `band` metrics.
    The band is the largest drop in minimum cosine (embedding), or the largest
    increases in logits errors and flips (segmentation), plus the largest total flip
    increase of one seed across all cases.
    """
    if not rows or any(not row["band"] for row in rows):
        raise Rejected("stage band: missing noise-band evidence")
    seeds = len(rows[0]["band"])
    if seeds < 8 or any(len(row["band"]) != seeds for row in rows):
        raise Rejected("stage band: at least 8 seeds per case are required")
    if embedding:
        drops = [
            row["metrics"]["minimum_cosine"] - band["minimum_cosine"]
            for row in rows
            for band in row["band"]
        ]
        finite(drops)
        return {"cosine": max(0.0, *drops), "seeds": seeds, "cases": len(rows)}
    l2 = [
        band["relative_l2"] - row["metrics"]["relative_l2"]
        for row in rows
        for band in row["band"]
    ]
    absolute = [
        band["max_abs"] - row["metrics"]["max_abs"]
        for row in rows
        for band in row["band"]
    ]
    finite(l2 + absolute)
    flips = [
        band["argmax_flips"] - row["metrics"]["argmax_flips"]
        for row in rows
        for band in row["band"]
    ]
    total_flips = max(
        sum(
            max(0, row["band"][seed]["argmax_flips"] - row["metrics"]["argmax_flips"])
            for row in rows
        )
        for seed in range(seeds)
    )
    return {
        "relative_l2": max(0.0, *l2),
        "max_abs": max(0.0, *absolute),
        "flips": max(0, *flips),
        "total_flips": total_flips,
        "seeds": seeds,
        "cases": len(rows),
    }


def tf32_stage_case(
    candidate: dict, library: dict, band: dict, embedding: bool
) -> dict:
    """One TF32 stage case may change from the Library stage only within the band."""
    if embedding:
        drop = library["minimum_cosine"] - candidate["minimum_cosine"]
        finite([drop])
        if drop > band["cosine"]:
            raise Rejected(
                f"stage band: cosine drop {drop:.3e} exceeds band {band['cosine']:.3e}"
            )
        return {"cosine_drop": drop}
    changes = {
        "relative_l2": candidate["relative_l2"] - library["relative_l2"],
        "max_abs": candidate["max_abs"] - library["max_abs"],
        "flips": candidate["argmax_flips"] - library["argmax_flips"],
    }
    finite(list(changes.values()))
    for key, value in changes.items():
        if value > band[key]:
            raise Rejected(
                f"stage band: {key} increase {value} exceeds band {band[key]}"
            )
    return changes


def tf32_stage_aggregate(
    candidates: Sequence[dict], libraries: Sequence[dict], band: dict, embedding: bool
) -> dict:
    """Across every TF32 stage case the candidate may not be worse on average."""
    if not candidates or len(candidates) != len(libraries):
        raise Rejected("stage band: missing aggregate cases")
    if embedding:
        means = [
            statistics.fmean(row["mean_cosine"] for row in rows)
            for rows in (candidates, libraries)
        ]
        if means[0] < means[1]:
            raise Rejected("stage band: mean cosine below Library")
        return {"mean_cosine": means}
    errors = [
        statistics.fmean(row["relative_l2"] for row in rows)
        for rows in (candidates, libraries)
    ]
    flips = [
        sum(row["argmax_flips"] for row in rows) for rows in (candidates, libraries)
    ]
    if errors[0] > errors[1]:
        raise Rejected("stage band: mean logits error above Library")
    if flips[0] > flips[1] + band["total_flips"]:
        raise Rejected("stage band: total flips above Library plus the band")
    return {"mean_relative_l2": errors, "total_flips": flips}


TOOLS = ("memcheck", "racecheck", "initcheck")


def sanitizer(tool: str, returncode: int, log: str) -> None:
    """Require a completed zero-error report; do not subtract library counts."""
    if tool not in TOOLS or returncode != 0:
        raise Rejected("sanitizer: tool did not finish successfully")
    if re.search(
        r"(fatal|permission|no attachable process|no kernels were profiled)",
        log,
        re.IGNORECASE,
    ):
        raise Rejected("sanitizer: missing device evidence")
    pattern = (
        r"ERROR SUMMARY: (\d+) errors?"
        if tool != "racecheck"
        else r"RACECHECK SUMMARY: (\d+) hazards?"
    )
    summaries = re.findall(pattern, log)
    if not summaries or any(int(value) != 0 for value in summaries):
        raise Rejected(f"sanitizer: {tool} findings or missing summary")
