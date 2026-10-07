"""Own production and collection batch domains independently of candidate claims."""

from dataclasses import dataclass
import collection_paths

from gates import Rejected

COLLECTION_REGISTRY = vars(collection_paths)


@dataclass(frozen=True)
class BatchDomain:
    """Keep production authorization separate from diagnostic stress cases"""

    production: tuple[int, ...]
    tested: tuple[int, ...]
    cases: tuple[tuple[str, int], ...]


MODEL = BatchDomain(
    (1, 32),
    (1, 7, 32, 33, 64),
    (
        ("first", 1),
        ("last", 1),
        ("short", 1),
        ("mixed", 7),
        ("mixed", 32),
        ("mixed", 33),
        ("mixed", 64),
        ("short", 7),
    ),
)
FBANK = BatchDomain(
    tuple(range(1, 33)),
    tuple(range(1, 33)),
    (
        ("first", 1),
        ("last", 1),
        ("short", 1),
        *(("mixed", batch) for batch in range(2, 33)),
        ("short", 7),
    ),
)


def boundary(name: str) -> BatchDomain:
    """Select the same per-boundary production domain as the Rust owner"""
    return FBANK if name == "fbank.dft" else MODEL


def collection(target: str) -> BatchDomain:
    """Reject unknown targets rather than inventing collection coverage"""
    if target == "fbankdft":
        return FBANK
    if (
        target in ("resnet", "lstm", "sincnet")
        or target in COLLECTION_REGISTRY["LAYERS"]
    ):
        return MODEL
    raise Rejected("domain: unknown collection target")
