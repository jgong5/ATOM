# SPDX-License-Identifier: MIT
"""No clock call is made while holding a lock, read lexically over `atom/`.

A clock owner parked in `advance_to` or `next_event` waits for its own LP's
receiving threads to hand over what the grant released. If it holds a lock one
of them needs, the LP deadlocks. So a call to `advance_to`, `next_event`, or the
step-loop hooks ATOM calls (`clock.turn`, `clock.step_done`,
`clock.wait_output`) must not sit inside a `with` whose context expression is a
name bound, in the same file, to a `threading` Lock, RLock, Condition,
Semaphore or BoundedSemaphore.

What this cannot read it refuses by name rather than passing over: an
`.acquire()` in a function that makes a clock call, a `with` item in such a
function that resolves to no binding in the file, and a clock call reached
through an alias. A call made through any other helper function is out of
scope: the invariant is stated lexically.
"""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CA_CALLS = {"advance_to", "next_event"}
HOOKS = {"turn", "step_done", "wait_output"}
LOCKS = {"Lock", "RLock", "Condition", "Semaphore", "BoundedSemaphore"}


def _ca_name(node) -> str | None:
    if isinstance(node, ast.Name) and node.id in CA_CALLS:
        return node.id
    if isinstance(node, ast.Attribute):
        if node.attr in CA_CALLS:
            return node.attr
        if node.attr in HOOKS and getattr(node.value, "id", None) == "clock":
            return f"clock.{node.attr}"
    return None


def _key(node) -> str | None:
    return getattr(node, "id", None) or getattr(node, "attr", None)


def _is_lock(node) -> bool:
    return isinstance(node, ast.Call) and _key(node.func) in LOCKS


def check(path: str, source: str) -> tuple[list[str], list[str], list[str]]:
    """The clock call sites in `source`, the ones inside a lock, and the refusals."""
    tree = ast.parse(source)
    bound: dict[str, list] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                bound.setdefault(_key(target), []).append(node.value)

    def is_lock(expr) -> bool | None:
        """True or False when `expr` is read as a lock or not; None when it cannot be."""
        if isinstance(expr, ast.Call):
            return True if _is_lock(expr) else None
        values = bound.get(_key(expr))
        return None if not values else any(_is_lock(v) for v in values)

    sites, inside, refused = [], [], []
    called = set()

    def visit(node, withs):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            withs = []
            if any(_ca_name(c.func) for c in ast.walk(node) if isinstance(c, ast.Call)):
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Call) and _key(sub.func) == "acquire":
                        refused.append(
                            f"{path}:{sub.lineno}: acquire() in a function that asks for time"
                        )
                    if isinstance(sub, (ast.With, ast.AsyncWith)):
                        for item in sub.items:
                            if is_lock(item.context_expr) is None:
                                refused.append(
                                    f"{path}:{sub.lineno}: cannot classify with item "
                                    f"{ast.unparse(item.context_expr)}"
                                )
        if isinstance(node, (ast.With, ast.AsyncWith)):
            withs = withs + [node]
        if isinstance(node, ast.Call):
            name = _ca_name(node.func)
            if name:
                called.add(id(node.func))
                sites.append(f"{path}:{node.lineno}: {name}")
                for w in withs:
                    for item in w.items:
                        if is_lock(item.context_expr):
                            inside.append(
                                f"{path}:{node.lineno}: {name} inside "
                                f"with {ast.unparse(item.context_expr)} (line {w.lineno})"
                            )
        elif _ca_name(node) and id(node) not in called:
            refused.append(
                f"{path}:{node.lineno}: a clock call reached through an alias"
            )
        for child in ast.iter_child_nodes(node):
            visit(child, withs)

    visit(tree, [])
    return sites, inside, refused


def _check_tree():
    sites, inside, refused = [], [], []
    for p in sorted((ROOT / "atom").rglob("*.py")):
        rel = str(p.relative_to(ROOT))
        s, i, r = check(rel, p.read_text(encoding="utf-8"))
        sites += s
        inside += i
        refused += r
    return sites, inside, refused


def test_no_clock_call_in_atom_holds_a_lock():
    sites, inside, refused = _check_tree()
    print("clock call sites checked:\n" + "\n".join(sites))
    assert sites, "the guard found no clock call to check"
    assert not inside
    assert not refused


def _seeded(rel: str, anchor: str, seed: str):
    """Check `rel` with `seed` inserted on the line after `anchor`, indented one level deeper."""
    lines = (ROOT / rel).read_text(encoding="utf-8").splitlines()
    (k,) = [n for n, line in enumerate(lines) if line.strip() == anchor]
    indent = " " * (len(lines[k]) - len(lines[k].lstrip()) + 4)
    lines.insert(k + 1, indent + seed)
    return k + 2, check(rel, "\n".join(lines))


@pytest.mark.parametrize(
    "seed, name",
    [("rt.advance_to(1.0)", "advance_to"), ("clock.turn(self)", "clock.turn")],
)
def test_a_clock_call_inside_a_lock_is_named_by_file_and_line(seed, name):
    rel = "atom/model_engine/scheduler.py"
    line, (_, inside, _) = _seeded(rel, "with self._pending_lock:", seed)
    assert inside == [
        f"{rel}:{line}: {name} inside with self._pending_lock (line {line - 1})"
    ]


POLL = "def poll(self, timeout_ms: int | None = None) -> bool:"


def test_an_acquire_in_a_function_that_asks_for_time_is_refused():
    line, (_, _, refused) = _seeded(
        "atom/utils/clock.py", POLL, "self.rt.lock.acquire()"
    )
    assert refused == [
        f"atom/utils/clock.py:{line}: acquire() in a function that asks for time"
    ]


def test_a_with_item_bound_to_nothing_is_refused():
    line, (_, _, refused) = _seeded(
        "atom/utils/clock.py", POLL, "with self.rt.gate: pass"
    )
    assert refused == [
        f"atom/utils/clock.py:{line}: cannot classify with item self.rt.gate"
    ]


def test_a_clock_call_through_an_alias_is_refused():
    line, (_, _, refused) = _seeded(
        "atom/utils/clock.py", POLL, "ask = self.rt.next_event"
    )
    assert refused == [
        f"atom/utils/clock.py:{line}: a clock call reached through an alias"
    ]
