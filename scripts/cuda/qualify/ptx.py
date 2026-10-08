"""Static checks on the PTX entries a candidate process loaded.

`shared_initialization` is a must-dataflow pass over each entry's control-flow graph:
every shared-memory load must be preceded, on every path from the entry, by a
shared-memory store (`st.shared`, `cp.async ... .shared`, `atom.shared`, `red.shared`
or `stmatrix`) or by a CTA barrier that some shared store in the entry can reach,
which is how one thread reads what another thread wrote. It is address-insensitive and
does not follow generic pointers converted from shared addresses, so it is a backstop
for the kernel audit, which still owns shared-memory initialization.
"""

import re
from dataclasses import dataclass, field

ENTRY = re.compile(r"\.entry\s+([A-Za-z_$][A-Za-z_0-9$]*)\s*\(")
LABEL = re.compile(r"^\s*([$A-Za-z_][$A-Za-z_0-9]*)\s*:\s*$")
BRANCH = re.compile(r"^\s*(@!?%\w+\s+)?bra(?:\.uni)?\s+([$A-Za-z_][$A-Za-z_0-9]*)\s*$")
EXIT = re.compile(r"^\s*(@!?%\w+\s+)?(?:ret|exit|trap)\b")
STORE = re.compile(
    r"\b(?:st(?:\.volatile|\.relaxed\.\w+|\.release\.\w+)?\.shared|cp\.async\S*\.shared|atom\S*\.shared|red\S*\.shared|stmatrix)\b"
)
BARRIER = re.compile(
    r"\b(?:bar\.sync|bar\.red|barrier\.sync|barrier\.cta\.sync|barrier\.red)\b"
)
LOAD = re.compile(
    r"\b(?:ld(?:\.volatile|\.relaxed\.\w+|\.acquire\.\w+)?\.shared|ldmatrix)\b"
)


@dataclass
class Block:
    """A straight-line run of instructions with its successors."""

    lines: list[str] = field(default_factory=list)
    successors: list[int] = field(default_factory=list)


def entries(source: str) -> dict[str, str]:
    """Each `.entry` body by name, from its header to the next entry or the end."""
    found = list(ENTRY.finditer(source))
    return {
        match[1]: source[
            match.start() : found[i + 1].start() if i + 1 < len(found) else len(source)
        ]
        for i, match in enumerate(found)
    }


def instructions(body: str) -> list[str]:
    """Instruction and label lines inside the entry braces, comments removed."""
    start = body.find("{")
    if start < 0:
        return []
    text = re.sub(r"//[^\n]*", "", body[start + 1 :])
    lines = []
    for raw in re.split(r";|\n", text):
        line = raw.strip()
        if not line or line in ("{", "}") or line.startswith("."):
            continue
        # a label can share a line with the next instruction
        while (
            match := re.match(r"^([$A-Za-z_][$A-Za-z_0-9]*)\s*:\s*(.*)$", line)
        ) and not line.startswith("@"):
            lines.append(f"{match[1]}:")
            line = match[2].strip()
        if line and line not in ("{", "}"):
            lines.append(line)
    return lines


def blocks(lines: list[str]) -> list[Block]:
    """Split at labels and after branches and exits, then link successors."""
    result = [Block()]
    labels: dict[str, int] = {}
    for line in lines:
        label = LABEL.match(line)
        if label:
            if result[-1].lines:
                result.append(Block())
            labels[label[1]] = len(result) - 1
            continue
        result[-1].lines.append(line)
        if BRANCH.match(line) or EXIT.match(line):
            result.append(Block())
    for index, block in enumerate(result):
        last = block.lines[-1] if block.lines else ""
        branch = BRANCH.match(last)
        exit_ = EXIT.match(last)
        if branch:
            if branch[2] in labels:
                block.successors.append(labels[branch[2]])
            if branch[1]:
                block.successors.append(index + 1)
        elif exit_ and not exit_[1]:
            pass
        else:
            block.successors.append(index + 1)
        block.successors = [s for s in block.successors if s < len(result)]
    return result


def uninitialized_shared_loads(body: str) -> list[str]:
    """Shared loads that some path reaches without a store or a covering barrier."""
    graph = blocks(instructions(body))
    predecessors: dict[int, list[int]] = {index: [] for index in range(len(graph))}
    for index, block in enumerate(graph):
        for successor in block.successors:
            predecessors[successor].append(index)

    # blocks some shared store can flow into, so a barrier there covers other threads
    reached: set[int] = set()
    frontier = [
        successor
        for index, block in enumerate(graph)
        if any(STORE.search(line) for line in block.lines)
        for successor in block.successors
    ]
    while frontier:
        index = frontier.pop()
        if index not in reached:
            reached.add(index)
            frontier.extend(graph[index].successors)

    def initializes(index: int, line: str, stored_here: bool) -> bool:
        if STORE.search(line):
            return True
        return bool(BARRIER.search(line)) and (stored_here or index in reached)

    def transfer(index: int, ready: bool) -> tuple[bool, list[str]]:
        stored_here = False
        loads = []
        for line in graph[index].lines:
            if initializes(index, line, stored_here):
                ready = True
            if STORE.search(line):
                stored_here = True
            elif LOAD.search(line) and not ready:
                loads.append(line)
        return ready, loads

    # must-analysis: start optimistic everywhere but the entry, then shrink
    ready_out = [True] * len(graph)
    changed = True
    while changed:
        changed = False
        for index in range(len(graph)):
            preds = predecessors[index]
            ready_in = index != 0 and bool(preds) and all(ready_out[p] for p in preds)
            out, _ = transfer(index, ready_in)
            if out != ready_out[index]:
                ready_out[index] = out
                changed = True

    findings = []
    for index in range(len(graph)):
        preds = predecessors[index]
        ready_in = index != 0 and bool(preds) and all(ready_out[p] for p in preds)
        findings.extend(transfer(index, ready_in)[1])
    return findings


def shared_initialization(source: str) -> dict[str, list[str]]:
    """Entries with a shared load that no shared store dominates, and those loads."""
    return {
        name: loads
        for name, body in entries(source).items()
        if (loads := uninitialized_shared_loads(body))
    }
