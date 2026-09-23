# SPDX-License-Identifier: MIT
"""The reply-contract paragraph of `02_model_runner_and_cost_backend.md`, kept
honest against the call sites it describes.

That paragraph is where a reader learns what a replacement runner has to return
and what happens when it does not. It closed with a rule -- "across a process
boundary a breached contract becomes a hang, not a traceback" -- that the tree
around it contradicts twice. `busy_loop` binds a runner's reply to a single
name, so the `None` the paragraph blamed for killing the worker could not have:
the unpack it describes is in the process that owns the worker, where a wrong
shape raises in the ordinary way. And "a hang" is asserted over all twelve
dispatched names when two of them have no caller waiting for a reply at all.

The corrected paragraph partitions the breach instead -- a reply that arrives
with the wrong shape, and a reply that never arrives -- and says for which names
the second one parks anybody.

**This guards the document, not the runner.** What a runner must return, and
what a raise becomes once it has left the worker, are asserted beside the module
that owns them. What is asserted here is that the prose still describes ATOM's
source: the names it enumerates and how many there are, how many of them are
waited on, which ones are not, and the site and arity it cites for the unpack.

That is the whole reach: the six values `audit` compares. Every other assertion
is outside it, including one that sits beside a compared name or number. Flip
where the unpack runs or whether the wait has a timeout, or cite other lines for
the dispatch, and nothing here fails; nor does putting the rule this paragraph
replaced back in place of the sentence that corrects it, with every compared
value left alone. The source facts behind the first two are checked below; that
the prose still makes them is not.

Nothing below writes down an answer. The waited/unwaited partition comes from
`RPC_SURFACE`, the unpack's line and arity are walked out of `engine_core.py`,
and the unboundedness of the wait is walked out of `async_proc.py`; the
paragraph's own numbers are parsed from its prose and compared. `audit` takes
the text and the surface as arguments, so a one-sided drift and a coordinated
move both run through the same code the passing test runs -- including a full
revert to the paragraph this replaces.

The derivation is an AST walk rather than an import: `model_runner.py` reaches
aiter's architecture probe, and these tests run with no driver.
"""

from __future__ import annotations

import ast
import hashlib
import re
from pathlib import Path

import pytest

from atom.compass.runner.overrides import RPC_SURFACE

TREE = Path(__file__).resolve().parents[2]
DOC = TREE / "atom/compass/design/02_model_runner_and_cost_backend.md"
DOCUMENT = DOC.read_text(encoding="utf-8")

HEADING = "### The RPC surface that must be honoured"
WAIT_CLAUSE = "have a caller that waits for the reply"
UNPACKED = "capture_cudagraph"

# A backtick-quoted bare identifier. It is applied only to the enumeration and
# to the sentence that partitions it; neither slice holds another backticked span.
_NAME = re.compile(r"`([a-z_][a-z0-9_]*)`")
_CITE = re.compile(r"`engine_core\.py:(\d+)`")
_ARITY = re.compile(r"unpacks ([a-z]+) values")
_SENTENCE = re.compile(r"[^.]*" + re.escape(WAIT_CLAUSE) + r"[^.]*\.")

# Spelling, not a claim: every number read through this map is compared against
# one derived from source.
NUMBER = {
    word: value
    for value, word in enumerate(
        ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight")
        + ("nine", "ten", "eleven", "twelve")
    )
}

# The paragraph as it stood at cddda00b50, so the guard can be shown refusing
# the defect it was written for and not merely a small drift from its fix.
REVERTED = """**Return contracts are load-bearing across a process boundary.** `engine_core` calls
`capture_cudagraph` with `wait_out=True` and unpacks three values; a stub that returned
`None` killed the worker on an unpacking error while the parent waited forever. **Across
a process boundary a breached contract becomes a hang, not a traceback.**"""


def _module(name: str) -> ast.Module:
    """One engine module, parsed. Importing it would need a driver."""
    return ast.parse((TREE / "atom/model_engine" / name).read_text(encoding="utf-8"))


def _unpacks(tree: ast.AST, name: str) -> list[tuple[int, int]]:
    """Every `a, b, c = <mgr>.call_func(name, ...)` under `tree`, with its arity."""
    return [
        (node.lineno, len(node.targets[0].elts))
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Tuple)
        and isinstance(node.value, ast.Call)
        and getattr(node.value.func, "attr", None) == "call_func"
        and node.value.args
        and getattr(node.value.args[0], "value", None) == name
    ]


def unpack_site(name: str) -> tuple[int, int]:
    """The site the paragraph cites, and how many values it unpacks.

    Scoped to `EngineCore` itself, which is the scope `RPC_SURFACE` declares:
    `DecodeEngineCore.__init__` unpacks the same reply a second time on the
    RapidServe path, which that table puts outside its own reach. The arity is
    checked across every site in the module, so the paragraph's "three values"
    does not depend on which of them it names.
    """
    module = _module("engine_core.py")
    assert (
        len({arity for _, arity in _unpacks(module, name)}) == 1
    ), f"{name} is unpacked into differing numbers of values in engine_core.py"
    scoped = _unpacks(
        next(
            n
            for n in module.body
            if isinstance(n, ast.ClassDef) and n.name == "EngineCore"
        ),
        name,
    )
    assert len(scoped) == 1, f"{name} is unpacked at {scoped} in EngineCore, wanted one"
    return scoped[0]


def stated(text: str) -> dict:
    """What the paragraph claims, parsed out of its own prose."""
    assert HEADING in text, f"{DOC.name} no longer carries {HEADING!r}"
    body = text.split(HEADING, 1)[1].split("\n### ", 1)[0]
    sentence = _SENTENCE.search(body)
    counted = (
        [NUMBER[w] for w in re.findall(r"[a-z]+", sentence.group()) if w in NUMBER]
        if sentence
        else []
    )
    return {
        "enumerated": _NAME.findall(
            body.split("must answer all of:", 1)[1].split(".", 1)[0]
        ),
        "waiters": counted[0] if counted else None,
        "total": counted[1] if len(counted) > 1 else None,
        "unwaited": _NAME.findall(sentence.group()) if sentence else [],
        "cited_line": int(m.group(1)) if (m := _CITE.search(body)) else None,
        "arity": NUMBER.get(m.group(1)) if (m := _ARITY.search(body)) else None,
    }


def audit(text: str, surface: dict[str, bool]) -> list[str]:
    """Every way the paragraph and the surface it describes can disagree."""
    claim = stated(text)
    line, arity = unpack_site(UNPACKED)
    checks = (
        ("the names it enumerates", sorted(claim["enumerated"]), sorted(surface)),
        ("how many there are", claim["total"], len(surface)),
        ("how many are waited on", claim["waiters"], sum(surface.values())),
        (
            "which are not waited on",
            sorted(claim["unwaited"]),
            sorted(n for n, waits in surface.items() if not waits),
        ),
        (f"the line it cites for the {UNPACKED} unpack", claim["cited_line"], line),
        (f"the values {UNPACKED} unpacks into", claim["arity"], arity),
    )
    return [
        f"{what}: the paragraph says {said!r}, the source says {derived!r}"
        for what, said, derived in checks
        if said != derived
    ]


def test_the_paragraph_as_it_stands_raises_no_complaint():
    """The silent direction, on the real document and the real surface."""
    assert audit(DOCUMENT, RPC_SURFACE) == []


def test_the_two_claims_the_paragraph_makes_without_a_number():
    """That the worker cannot break on an unpack, and that the wait is unbounded.

    `busy_loop` binds the resolved method's return to a single name, so there is
    no unpacking in the worker for a `None` reply to fail at; the sole unpack is
    in the module that constructs the manager, which is the parent. And
    `call_func` reads the output queue with no timeout, which is what makes a
    reply that never arrives a park rather than an error.
    """
    module = _module("async_proc.py")
    defs = {n.name: n for n in ast.walk(module) if isinstance(n, ast.FunctionDef)}
    dispatched = [
        node
        for node in ast.walk(defs["busy_loop"])
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and getattr(node.value.func, "id", None) == "func"
    ]
    assert len(dispatched) == 1 and isinstance(
        dispatched[0].targets[0], ast.Name
    ), "busy_loop no longer binds the resolved method's reply to a single name"
    reads = [
        node
        for node in ast.walk(defs["call_func"])
        if isinstance(node, ast.Call)
        and getattr(node.func, "attr", None) == "get"
        and getattr(getattr(node.func, "value", None), "attr", None) == "outputs_queue"
    ]
    assert (
        len(reads) == 1 and not reads[0].args and not reads[0].keywords
    ), "call_func's read of the output queue is no longer a single untimed get"
    assert any(
        isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "AsyncIOProcManager"
        for node in ast.walk(_module("engine_core.py"))
    ), "engine_core.py no longer constructs the manager, so it is not the parent"


def _revert(text: str) -> str:
    """The whole paragraph put back as it stood, inside the real document."""
    kept, replaced = text.split("**Return contracts are load-bearing", 1)
    return kept + REVERTED + "\n" + replaced.split("\n\n", 1)[1]


# One drift at a time, each a document edit, a surface edit, or both. The first
# is the whole paragraph reverted; then a name losing its waiting caller with
# the prose not told; then the cited unpack line ceasing to be the unpack line;
# then a dispatched name dropped from the list the paragraph counts. The last
# three move one claim alone: the total, the arity, and which names go unwaited
# with their count unchanged.
DRIFTS = (
    ("reverted", lambda text, surface: (_revert(text), surface)),
    ("surface", lambda text, surface: (text, dict(surface, dummy_execution=False))),
    ("citation", lambda t, s: (t.replace(f":{unpack_site(UNPACKED)[0]}`", ":1`"), s)),
    (
        "enumeration",
        lambda text, surface: (text.replace("`flush_pp_send`.", "."), surface),
    ),
    ("total", lambda t, s: (t.replace("the twelve names", "the eleven names"), s)),
    ("arity", lambda t, s: (t.replace("three values", "two values"), s)),
    ("unwaited", lambda t, s: (t, dict(s, exit=True, dummy_execution=False))),
)

# The checks each drift must raise and no others, so a check deleted from
# `audit` fails, by name, every drift that raises it.
TOTAL, CITE = "how many there are", f"the line it cites for the {UNPACKED} unpack"
WAITED, UNWAITED = "how many are waited on", "which are not waited on"
FIRES = {
    "reverted": {TOTAL, WAITED, UNWAITED, CITE},
    "surface": {WAITED, UNWAITED},
    "citation": {CITE},
    "enumeration": {"the names it enumerates"},
    "total": {TOTAL},
    "arity": {f"the values {UNPACKED} unpacks into"},
    "unwaited": {UNWAITED},
}


@pytest.mark.parametrize("drift", DRIFTS, ids=[name for name, _ in DRIFTS])
def test_the_guard_fires_when_one_side_moves_alone(drift):
    """The firing direction, each drift routed through the same `audit`."""
    fired = audit(*drift[1](DOCUMENT, RPC_SURFACE))
    assert {c.split(": the paragraph says ")[0] for c in fired} == FIRES[drift[0]]


def test_reverted_is_the_pre_fix_paragraph_byte_for_byte():
    """A paraphrase would still fire the guard; this keeps `REVERTED` the paragraph
    at the commit named above it. The digest is sha256 over that paragraph alone:
    the document split on blank lines, LF endings, no trailing newline, UTF-8. It
    pins history, so the only legitimate change is a new anchor commit."""
    digest = "36da9cc1ef3f9f1fa8bdf192023dc2e332d6c42dc5dce1ca318a5fb0f8c0a932"
    assert hashlib.sha256(REVERTED.encode("utf-8")).hexdigest() == digest


def test_the_guard_is_silent_when_the_paragraph_and_the_surface_move_together():
    """The control a drift guard characteristically lacks: re-partitioning one
    name on both sides is a deliberate change, and must raise no complaint."""
    moved = DOCUMENT.replace(
        "ten of\nthe twelve names above have a caller that waits for the reply, and"
        " `exit` and\n`process_kvconnector_output` do not.",
        "nine of\nthe twelve names above have a caller that waits for the reply, and"
        "\n`dummy_execution`, `exit` and `process_kvconnector_output` do not.",
    )
    assert moved != DOCUMENT, "the wait sentence is no longer worded as this edits it"
    assert audit(moved, dict(RPC_SURFACE, dummy_execution=False)) == []
