"""Static scan of candidate code and of the build inputs, before anything is built.

Candidate host code lives in src/inference/cuda/candidate/ and candidate kernels in
the resnet, lstm and sincnet areas of the kernel crate. Locked dispatch calls the
candidate, so candidate code must not observe or steer the measurement: no
environment, files, network, processes, threads, clocks or capture state; no state
that outlives a call; no harness, NVTX or library calls; no foreign code. Comments
are stripped first; string literals are scanned.
"""

import os
import re
from dataclasses import dataclass
from pathlib import Path

from lock import LockError, inventory

CANDIDATE_HOST = "src/inference/cuda/candidate"
KERNEL_CRATE = "crates/speakrs-cuda-kernels/src"
CANDIDATE_AREAS = ("resnet", "lstm", "sincnet", "fbankdft", "segdense", "wideconv")
# a static item declaration, not the `'static` lifetime
STATIC_ITEM = r"(?<!')\bstatic\s+(?:mut\s+)?[A-Za-z_][A-Za-z0-9_]*\s*:"

# std modules a candidate may not reach, however they are imported or renamed
FORBIDDEN_STD = ("env", "fs", "net", "process", "thread", "os", "io", "time")
STD_ROOTS = ("std", "core", "alloc")
# each rule is a pattern and the reason it is refused
HOST_RULES: tuple[tuple[str, str], ...] = (
    (
        r"\b(?:load_kernels|load_artifact|load_ptx|load_cubin|load_library|load_function|load_module|cuModuleLoad[A-Za-z_]*|cuLibraryLoad[A-Za-z_]*)\b",
        "candidate plans must use preloaded LoadedKernels",
    ),
    (r"\boption_env\s*!", "environment read"),
    (r"\benv\s*!\s*\(\s*\"(?!CARGO_)", "env! outside compile-time Cargo constants"),
    (
        r"\b(?:File|OpenOptions|TcpStream|TcpListener|UdpSocket|UnixStream|Command|Instant|SystemTime)\b",
        "file, network, process or clock API",
    ),
    (
        r"SPEAKRS_QUALIFY|/workspace|/proc/|/dev/|\.safetensors|\bref/",
        "harness or reference path",
    ),
    (STATIC_ITEM, "static item holding state"),
    (
        r"\b(?:OnceLock|OnceCell|LazyLock|LazyCell|lazy_static|once_cell|thread_local|Atomic[A-Za-z0-9]*|Cell|RefCell|UnsafeCell|Mutex|RwLock|Condvar)\b",
        "lazy, atomic or interior-mutable state",
    ),
    (r"\btest_support\b|nvtx|libloading|\bqualify_", "NVTX or harness call"),
    (
        r"(?i:cudnn|cublas)|\bCudaBlas\b|\.blas\s*\(|\.dnn\s*\(|\bsgemm\b|\bConvPlanner\b|\bConvPlan\b|forward_bias_relu|\block_blas\b",
        "direct cuDNN or cuBLAS call outside the projection helper",
    ),
    (
        r"\bextern\b|#\s*!?\s*\[\s*(?:link|used|no_mangle|export_name|global_allocator|path)\b|link_section|include(?:_bytes|_str)?\s*!|(?<![A-Za-z0-9_])(?:global_)?asm\s*!|\bload_module\b|\bPtx\b|nvrtc|\bsys\s*::\s*(?!(?:CUfunction_attribute|CUfunc_cache)(?:_enum)?\b)|\bcu[A-Z][A-Za-z]+\s*\(|dlopen|transmute",
        "foreign code or module loading",
    ),
    (
        r"capture|\bnew_stream\b|\bnew_event\b|\.(?:fork|join|wait|record)\s*\(|\brecord_event\b|\belapsed_ms\b|\bcontext\s*\(",
        "capture, stream, event or timing introspection outside SideStream",
    ),
    (
        r"cfg\s*!?\s*\([^)]*\b(?:test|debug_assertions)\b",
        "test-only or debug-only behavior",
    ),
)
# device code: cuda-oxide declares shared memory as `static mut NAME: SharedArray<..>`
DEVICE_RULES: tuple[tuple[str, str], ...] = (
    (r"SPEAKRS_QUALIFY|/workspace|/proc/|\.safetensors", "harness or reference path"),
    (
        r"(?<!')\bstatic\s+(?!mut\s+[A-Z_][A-Z0-9_]*\s*:\s*(?:SharedArray|DynamicSharedArray)\b)(?:mut\s+)?[A-Za-z_][A-Za-z0-9_]*\s*:",
        "static item other than shared memory",
    ),
    (r"\btest_support\b|nvtx|libloading", "NVTX or harness call"),
    (r"(?i:cudnn|cublas)", "library call"),
    (
        r"\bextern\b|link_section|#\s*\[\s*(?:used|no_mangle|export_name|path)\b|include(?:_bytes|_str)?\s*!",
        "foreign code or module loading",
    ),
)
# anywhere in the crate: code that runs without being called
CRATE_RULES: tuple[tuple[str, str], ...] = (
    (
        r"link_section|#\s*\[\s*used\b|no_mangle|export_name|global_allocator|init_array|\.ctors",
        "code that runs before or outside the driver",
    ),
)


@dataclass(frozen=True)
class Finding:
    """One refused construct."""

    path: str
    line: int
    reason: str
    text: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.reason}: {self.text}"


def strip_comments(source: str) -> str:
    """Blank `//` and `/* */` comments outside string literals, keeping line numbers."""
    out = []
    i = 0
    in_string = False
    while i < len(source):
        c = source[i]
        if in_string:
            out.append(c)
            if c == "\\" and i + 1 < len(source):
                out.append(source[i + 1])
                i += 2
                continue
            if c == '"':
                in_string = False
            i += 1
            continue
        if source.startswith("'\"'", i) or source.startswith("'\\\"'", i):
            # a char literal holding a quote does not open a string
            end = source.index("'", i + 1) + 1
            out.append(source[i:end])
            i = end
            continue
        if c == '"':
            in_string = True
            out.append(c)
            i += 1
            continue
        if source.startswith("//", i):
            end = source.find("\n", i)
            end = len(source) if end < 0 else end
            out.append(" " * (end - i))
            i = end
            continue
        if source.startswith("/*", i):
            end = source.find("*/", i + 2)
            end = len(source) if end < 0 else end + 2
            out.append("".join(ch if ch == "\n" else " " for ch in source[i:end]))
            i = end
            continue
        out.append(c)
        i += 1
    return "".join(out)


def use_tree(
    tree: str, prefix: tuple[str, ...] = ()
) -> list[tuple[tuple[str, ...], str]]:
    """Expand one `use` tree into (full path, local name) pairs.

    Handles groups, nesting, `self`, `as` renames and globs (local name `*`).
    """
    tree = tree.strip()
    if not tree:
        return []
    depth, split = 0, None
    for index, char in enumerate(tree):
        depth += char == "{"
        depth -= char == "}"
        if char == "{" and depth == 1:
            split = index
            break
    if split is not None:
        head = tuple(part for part in re.split(r"\s*::\s*", tree[:split]) if part)
        body = tree[split + 1 : tree.rindex("}")]
        items, depth, start = [], 0, 0
        for index, char in enumerate(body):
            depth += char == "{"
            depth -= char == "}"
            if char == "," and depth == 0:
                items.append(body[start:index])
                start = index + 1
        items.append(body[start:])
        return [pair for item in items for pair in use_tree(item, prefix + head)]
    alias = None
    renamed = re.fullmatch(r"(.+?)\s+as\s+([A-Za-z_][A-Za-z0-9_]*)", tree)
    if renamed:
        tree, alias = renamed[1], renamed[2]
    parts = tuple(part for part in re.split(r"\s*::\s*", tree) if part)
    if parts and parts[-1] == "self":
        parts = parts[:-1]
    path = prefix + parts
    if not path:
        return []
    return [(path, alias or path[-1])]


def imports(code: str) -> list[tuple[tuple[str, ...], str]]:
    """Every imported path with its local name, from `use` declarations."""
    pairs = []
    for match in re.finditer(r"\buse\s+([^;]+);", code):
        pairs.extend(use_tree(match[1]))
    return pairs


def forbidden_std(path: tuple[str, ...]) -> bool:
    """`std::env`, `::std::fs::File` and the like, or a glob or rename of a std root."""
    parts = tuple(part for part in path if part)
    if not parts or parts[0] not in STD_ROOTS:
        return False
    if len(parts) == 1 or parts[1] == "*":
        return True
    return parts[1] in FORBIDDEN_STD


def std_paths(path: str, code: str, source: str) -> list[Finding]:
    """Refuse forbidden std modules through imports, renames, groups and full paths."""
    reason = "environment, file, network, process, thread or clock API"
    findings = []
    lines = source.splitlines()

    def finding(offset: int) -> Finding:
        line = code.count("\n", 0, offset) + 1
        return Finding(path, line, reason, lines[line - 1].strip()[:160])

    aliases = {}
    for match in re.finditer(r"\buse\s+([^;]+);", code):
        for full, local in use_tree(match[1]):
            if forbidden_std(full):
                findings.append(finding(match.start()))
            elif local != "*":
                aliases[local] = full
    for match in re.finditer(
        r"(?<![A-Za-z0-9_:])(::)?\s*([A-Za-z_][A-Za-z0-9_]*)\s*::\s*([A-Za-z_*][A-Za-z0-9_]*)",
        code,
    ):
        head, tail = match[2], match[3]
        base = aliases.get(head, (head,))
        if forbidden_std((*base, tail)) or (
            base[0] in STD_ROOTS and len(base) > 1 and base[1] in FORBIDDEN_STD
        ):
            findings.append(finding(match.start()))
    return findings


CANDIDATE_MODULE = ("crate", "inference", "cuda", "candidate")
# locked items a candidate may name outside its own tree; `Name::*` items allow
# variants and associated items, the others only the name itself
CRATE_ITEMS = {
    ("crate", "inference", "cuda", "CudaError"): True,
    ("crate", "inference", "cuda", "CudaMath"): True,
    ("crate", "inference", "cuda", "KernelModule"): True,
    ("crate", "inference", "cuda", "PtxTier"): True,
    ("crate", "inference", "cuda", "ComputeCapability"): True,
    ("crate", "inference", "cuda", "device", "DeviceAttributes"): True,
    ("crate", "inference", "cuda", "CudaRuntime"): False,
    ("crate", "inference", "cuda", "LoadedKernels"): False,
    ("crate", "inference", "cuda", "dnn", "Conv2d"): True,
    ("crate", "inference", "cuda", "error", "check_len"): False,
    ("crate", "inference", "cuda", "error", "element_count"): False,
    ("crate", "inference", "cuda", "error", "to_c_int"): False,
}
# cudarc driver types the interface hands to candidates, and launch plumbing
CUDARC_ITEMS = {
    "CudaStream",
    "CudaView",
    "CudaViewMut",
    "CudaSlice",
    "CudaFunction",
    "LaunchConfig",
    "LaunchArgs",
    "PushKernelArg",
    "DeviceRepr",
    "ValidAsZeroBits",
    "DevicePtr",
    "DevicePtrMut",
    "DeviceSlice",
}
# function attribute enums, for `CudaFunction::set_attribute` (dynamic shared memory)
CUDARC_SYS_ITEMS = {
    "CUfunction_attribute",
    "CUfunction_attribute_enum",
    "CUfunc_cache",
    "CUfunc_cache_enum",
}
# std, core and alloc modules with no I/O, clock, thread or shared-state API
STD_MODULES = {
    "array",
    "borrow",
    "boxed",
    "clone",
    "cmp",
    "collections",
    "convert",
    "default",
    "f32",
    "f64",
    "fmt",
    "hash",
    "hint",
    "iter",
    "marker",
    "mem",
    "num",
    "ops",
    "option",
    "prelude",
    "primitive",
    "ptr",
    "result",
    "slice",
    "str",
    "string",
    "u16",
    "u32",
    "u64",
    "u8",
    "usize",
    "i32",
    "i64",
    "vec",
}
# the crates the speakrs manifest depends on; candidates may name only cudarc
KNOWN_CRATES = {
    "cudarc",
    "safetensors",
    "libloading",
    "ndarray",
    "tracing",
    "serde",
    "serde_json",
    "sha2",
    "ort",
    "hf_hub",
    "rayon",
    "thiserror",
    "kodama",
    "tempfile",
    "candle_core",
}


def module_path(relative: str) -> tuple[str, ...]:
    """`src/inference/cuda/candidate/lstm/layout.rs` is `crate::...::lstm::layout`."""
    parts = relative.removeprefix("src/").removesuffix(".rs").split("/")
    if parts[-1] in ("mod", "lib"):
        parts = parts[:-1]
    return ("crate", *parts)


def resolve(path: tuple[str, ...], module: tuple[str, ...]) -> tuple[str, ...]:
    """Make `self::` and `super::` paths absolute from the file's module."""
    current = list(module)
    parts = list(path)
    while parts and parts[0] in ("self", "super"):
        if parts.pop(0) == "super":
            current.pop()
    if parts and parts[0] == "crate":
        return tuple(parts)
    if path and path[0] in ("self", "super"):
        return (*current, *parts)
    return tuple(parts)


def allowed_path(path: tuple[str, ...]) -> str | None:
    """None when a resolved path is allowed, else the reason it is refused."""
    if not path:
        return None
    if path[0] == "crate":
        if path[: len(CANDIDATE_MODULE)] == CANDIDATE_MODULE:
            return None
        for item, members in CRATE_ITEMS.items():
            if path == item or (members and path[: len(item)] == item):
                return None
        return "call outside the candidate tree and its locked interface"
    if path[0] in STD_ROOTS:
        if len(path) == 1 or path[1] == "*":
            return "glob or rename of a std root"
        if path[1] == "sync" and path[2:3] in ((), ("Arc",)):
            return None
        if path[1] in STD_MODULES:
            return None
        return "environment, file, network, process, thread or clock API"
    if path[0] == "cudarc":
        if path[1:2] == ("driver",) and (
            len(path) == 2
            or path[2] in CUDARC_ITEMS
            or (path[2] == "sys" and path[3:4] and path[3] in CUDARC_SYS_ITEMS)
        ):
            return None
        return "cudarc API outside the candidate interface"
    if path[0] in KNOWN_CRATES:
        return "dependency API outside the candidate interface"
    return None


def candidate_paths(relative: str, code: str, source: str) -> list[Finding]:
    """Resolve every import and qualified path in a candidate host file.

    A candidate may reach only its own tree, the locked interface items above, the
    cudarc driver types the interface uses and std modules without I/O or state.
    """
    module = module_path(relative)
    lines = source.splitlines()
    findings = []

    def finding(offset: int, reason: str) -> None:
        line = code.count("\n", 0, offset) + 1
        findings.append(Finding(relative, line, reason, lines[line - 1].strip()[:160]))

    aliases: dict[str, tuple[str, ...]] = {}
    for match in re.finditer(r"\buse\s+([^;]+);", code):
        for full, local in use_tree(match[1]):
            absolute = resolve(full, module)
            if (
                local == "*"
                and absolute[:1] != ("crate",)
                or (
                    local == "*"
                    and absolute[: len(CANDIDATE_MODULE)] != CANDIDATE_MODULE
                )
            ):
                finding(match.start(), "glob import outside the candidate tree")
                continue
            reason = allowed_path(absolute)
            if reason:
                finding(match.start(), reason)
            elif local != "*":
                aliases[local] = absolute
    # imports were resolved above; blank them so their prefixes are not paths too
    body = re.sub(r"\buse\s+[^;]+;", lambda match: " " * len(match[0]), code)
    for match in re.finditer(
        r"(?<![A-Za-z0-9_:])(::\s*)?([A-Za-z_][A-Za-z0-9_]*(?:\s*::\s*[A-Za-z_][A-Za-z0-9_]*)+)",
        body,
    ):
        segments = tuple(part.strip() for part in match[2].split("::"))
        head = segments[0]
        if head in aliases:
            absolute = (*aliases[head], *segments[1:])
        else:
            absolute = resolve(segments, module)
        reason = allowed_path(absolute)
        if reason:
            finding(match.start(), reason)
    return findings


def scan_text(
    path: str, source: str, rules, resolve_std: bool = True, resolve_calls: bool = False
) -> list[Finding]:
    """Apply every rule to comment-free code, plus the std module and call rules."""
    code = strip_comments(source)
    findings = []
    for pattern, reason in rules:
        for match in re.finditer(pattern, code):
            line = code.count("\n", 0, match.start()) + 1
            text = source.splitlines()[line - 1].strip()[:160]
            findings.append(Finding(path, line, reason, text))
    if resolve_std:
        findings.extend(std_paths(path, code, source))
    if resolve_calls:
        findings.extend(candidate_paths(path, code, source))
    return findings


def candidate_files(root: Path) -> tuple[list[Path], list[Path]]:
    """Host and device candidate files; symlinks are refused by the caller."""
    host = sorted(
        path
        for path in (root / CANDIDATE_HOST).rglob("*")
        if path.is_file() or path.is_symlink()
    )
    device = []
    crate = root / KERNEL_CRATE
    for area in CANDIDATE_AREAS:
        device.extend(path for path in [crate / f"{area}.rs"] if path.exists())
        if (crate / area).exists():
            device.extend(sorted((crate / area).rglob("*")))
    return host, device


def build_inputs(root: Path, cargo_home: Path | None) -> list[Finding]:
    """Refuse build scripts and Cargo configuration that could change the build.

    The tree's own `.cargo/config.toml` is locked; any other configuration in the tree,
    its parents or the Cargo home is refused.
    """
    findings = []
    if (root / "build.rs").exists():
        findings.append(
            Finding("build.rs", 0, "build script", "a build script runs at build time")
        )
    candidates = [root / ".cargo/config"]
    for directory in root.parents:
        candidates += [directory / ".cargo/config", directory / ".cargo/config.toml"]
    if cargo_home is not None:
        candidates += [cargo_home / "config", cargo_home / "config.toml"]
    for path in candidates:
        if path.exists():
            findings.append(
                Finding(
                    str(path),
                    0,
                    "Cargo configuration",
                    "configuration can change flags or linkers",
                )
            )
    return findings


def scan(root: Path, cargo_home: Path | None = None) -> dict:
    """Scan candidate code, the crate and the build inputs; return files and findings."""
    findings: list[Finding] = []
    host, device = candidate_files(root)
    for path in host + device:
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            findings.append(
                Finding(relative, 0, "symlink", "candidate files must be regular files")
            )
            continue
        if path.is_dir():
            continue
        if path.suffix != ".rs":
            findings.append(Finding(relative, 0, "non-Rust candidate file", path.name))
            continue
        if path in host:
            findings.extend(
                scan_text(relative, path.read_text(), HOST_RULES, resolve_calls=True)
            )
        else:
            findings.extend(scan_text(relative, path.read_text(), DEVICE_RULES))
    try:
        locked = set(inventory(root))
    except LockError as error:
        locked = set()
        findings.append(
            Finding("scripts/cuda/qualify/SCOPE.json", 0, "lock scope", str(error))
        )
    for path in sorted((root / "src").rglob("*.rs")):
        relative = path.relative_to(root).as_posix()
        findings.extend(scan_text(relative, path.read_text(), CRATE_RULES, False))
        # every compiled file is either locked harness and production code or candidate
        # code under the host rules
        if relative not in locked and not relative.startswith(CANDIDATE_HOST + "/"):
            findings.append(
                Finding(
                    relative, 0, "unlocked Rust outside the candidate tree", path.name
                )
            )
    findings.extend(build_inputs(root, cargo_home))
    return {
        "host_files": [
            path.relative_to(root).as_posix() for path in host if path.is_file()
        ],
        "device_files": [
            path.relative_to(root).as_posix() for path in device if path.is_file()
        ],
        "findings": [str(finding) for finding in findings],
    }


def cargo_home() -> Path | None:
    """The Cargo home whose configuration applies to the build."""
    value = os.environ.get("CARGO_HOME")
    return Path(value) if value else Path.home() / ".cargo"
