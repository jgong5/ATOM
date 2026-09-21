# SPDX-License-Identifier: MIT
"""A static pass for reads of a `set` in iteration order on the simulated path.

A `dict` hands its keys back in the order they went in. A `set` hands its
members back in the order its hash table happens to hold them, and that order is
not stable between processes: strings are hashed from a seed the interpreter
picks at start-up, so a set of request ids, block hashes or participant names
iterates differently in every run. Integers hash to themselves, which is worse
rather than better -- a set of small ints looks stable for as long as the ids
stay small and the table stays the size it was, and reorders the first time
neither holds.

The failure has nothing to point at. No exception, no warning, no log line:
two runs of one configuration, one diff, and no code change between them. So
this is a static check, and it is a check rather than a convention because the
mistake is invisible at the moment it is made.

**Membership is not iteration.** `x in s`, `len(s)`, `s.add(...)` and
`s.discard(...)` read nothing in order and are not reported. `sorted(s)` is not
reported either: it is the fix, and so is keeping the members in a `dict` whose
values are `None`.

**What it finds, and what it cannot.** A set is recognised where the module
itself says so -- a set literal or comprehension, a `set()` or `frozenset()`
call, a set operator between two of those or over a `dict` view, a name or
attribute assigned one of them, and a parameter or variable annotated as a set.
Everything outside the module's own text is invisible to it: a set returned by a
call with no annotation, one arriving in an unannotated parameter, one pulled
out of a list or a dict, and any subclass of `set` are all unrecognised, and the
iteration over them is not reported. It finds the forms this code is written in,
not every form Python allows.
"""

import argparse
import ast
import sys
from dataclasses import dataclass

# The enclosing-scope question and the walk over a tree are the same ones the
# clock-source pass answers, and the answers are taken from it rather than
# written twice.
from .clock_source import ClockSourceLint, _scope_at, _scopes

#: Builders that produce a set from anything.
SET_BUILDERS = ("set", "frozenset")

#: Annotations that declare one. The subscript is stripped before the name is
#: compared, so `set[str]` and `Set[LpId]` both land here.
SET_ANNOTATIONS = (
    "set",
    "frozenset",
    "Set",
    "FrozenSet",
    "AbstractSet",
    "MutableSet",
)

#: `dict` views are insertion-ordered and are **not** sets -- but a set operator
#: applied to one produces a real set, and that set is unordered.
VIEW_CALLS = ("keys", "items")

#: Calls whose result depends on the order their argument came out in. `sorted`
#: is deliberately absent: it is what a caller reaches for to fix one of these.
#: `sum` is here because float addition is not associative, so the order the
#: terms arrive in is the order the rounding happens in.
ORDERED_READS = (
    "list",
    "tuple",
    "iter",
    "reversed",
    "enumerate",
    "zip",
    "sum",
    "join",
)

#: The set operators. Applied to a set, or to a `dict` view, each produces a set.
SET_OPERATORS = (ast.BitOr, ast.BitAnd, ast.BitXor, ast.Sub)


@dataclass(frozen=True)
class SetIteration:
    """One ordered read of a set: where it is, what shape, and inside what."""

    path: str
    line: int
    form: str
    expression: str
    scope: str

    def __str__(self) -> str:
        return (
            f"{self.path}:{self.line}  {self.form} {self.expression}  in {self.scope}"
        )


class SetIterationLint:
    """Parses the simulated path and reports every ordered read of a set.

    `enabled=False` exists to show what the tree looks like with the check off,
    which is a tree that compiles, imports and runs.
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def scan_source(self, source: str, path: str) -> tuple[SetIteration, ...]:
        """Every ordered read of a set in one module's text, in line order."""
        if not self.enabled:
            return ()
        tree = ast.parse(source, filename=path)
        known = _set_valued_names(tree)
        scopes = _scopes(tree)
        found = {}
        for node in ast.walk(tree):
            for expression, form in _reads(node):
                if not _is_set(expression, known):
                    continue
                read = SetIteration(
                    path,
                    expression.lineno,
                    form,
                    ast.unparse(expression),
                    _scope_at(scopes, expression.lineno),
                )
                found[read] = None
        return tuple(
            sorted(found, key=lambda read: (read.line, read.form, read.expression))
        )

    def scan_modules(self, paths) -> tuple[SetIteration, ...]:
        """Every read in the modules named, in the order they were named."""
        found = []
        for path in paths:
            with open(path, encoding="utf-8") as handle:
                found.extend(self.scan_source(handle.read(), path))
        return tuple(found)

    @staticmethod
    def modules(root: str) -> tuple[str, ...]:
        """Every module under `root`, in a fixed order on every machine.

        The clock-source pass's walk, called rather than repeated, so the two
        checks are answering about the same files and a module that becomes
        invisible to one becomes invisible to both.
        """
        return ClockSourceLint.modules(root)

    def report(self, reads, scanned: int) -> str:
        """What the check prints. Clean is a line; dirty is the list and the fix."""
        if not self.enabled:
            return "set-iteration lint: disabled, so this tree makes no claim about its set order"
        if not reads:
            return f"set-iteration lint: clean over {scanned} module(s)"
        lines = [f"set-iteration lint: {len(reads)} ordered read(s) of a set:"]
        lines.extend(f"  {read}" for read in reads)
        lines.append(
            "A set hands its members back in hash order, and a string is hashed "
            "from a seed the interpreter picks per process, so two runs of one "
            "configuration read the same set in two orders and diverge with "
            "nothing to point at. Sort at the point of iteration, or hold the "
            "members in a dict whose values are None and iterate that."
        )
        return "\n".join(lines)

    def check(self, root: str) -> tuple[int, str]:
        """Scan `root` and return an exit code beside the report. Non-zero fails CI."""
        modules = self.modules(root)
        reads = self.scan_modules(modules)
        return (1 if reads else 0), self.report(reads, len(modules))


def _reads(node):
    """Every (expression, form) this node reads in the order it came out in."""
    if isinstance(node, (ast.For, ast.AsyncFor)):
        return ((node.iter, "for-loop over"),)
    if isinstance(node, ast.comprehension):
        return ((node.iter, "comprehension over"),)
    if isinstance(node, ast.Starred):
        return ((node.value, "unpacking of"),)
    if isinstance(node, ast.YieldFrom):
        return ((node.value, "yield from"),)
    if isinstance(node, ast.Assign) and any(
        isinstance(target, (ast.Tuple, ast.List)) for target in node.targets
    ):
        return ((node.value, "unpacking of"),)
    if isinstance(node, ast.Call):
        tail = _tail(node.func)
        if tail in ORDERED_READS:
            return tuple((argument, f"{tail}() over") for argument in node.args)
    return ()


def _tail(node) -> str | None:
    """The last segment of a name or attribute chain: `os.walk` -> `walk`."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _dotted(node) -> str | None:
    """`a.b.c` for an attribute or name chain, or None for anything else."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def _is_set(node, known) -> bool:
    """Whether this expression is a set, as far as the module's own text says."""
    if isinstance(node, (ast.Set, ast.SetComp)):
        return True
    if isinstance(node, ast.Call):
        return _tail(node.func) in SET_BUILDERS
    if isinstance(node, (ast.Name, ast.Attribute)):
        return _dotted(node) in known
    if isinstance(node, ast.BinOp) and isinstance(node.op, SET_OPERATORS):
        return any(
            _is_set(side, known) or _is_view(side) for side in (node.left, node.right)
        )
    if isinstance(node, ast.IfExp):
        return _is_set(node.body, known) or _is_set(node.orelse, known)
    return False


def _is_view(node) -> bool:
    """A `dict` view. Ordered on its own; a set the moment an operator touches it."""
    return isinstance(node, ast.Call) and _tail(node.func) in VIEW_CALLS


def _annotates_set(node) -> bool:
    """Whether an annotation declares a set. A string annotation is parsed first."""
    if node is None:
        return False
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        try:
            node = ast.parse(node.value, mode="eval").body
        except SyntaxError:
            return False
    if isinstance(node, ast.Subscript):
        node = node.value
    return _tail(node) in SET_ANNOTATIONS


def _set_valued_names(tree) -> dict[str, None]:
    """Every name and attribute this module says holds a set.

    Taken to a fixed point, because one name can be bound from another that a
    later line binds, and the walk order is not the source order.
    """
    known: dict[str, None] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.arg) and _annotates_set(node.annotation):
            known[node.arg] = None
    grew = True
    while grew:
        grew = False
        for node in ast.walk(tree):
            for name in _binds(node, known):
                if name is not None and name not in known:
                    known[name] = None
                    grew = True
    return known


def _binds(node, known):
    """Every name this statement binds to a set."""
    if isinstance(node, ast.Assign) and _is_set(node.value, known):
        return tuple(
            _dotted(target)
            for target in node.targets
            if isinstance(target, (ast.Name, ast.Attribute))
        )
    if (
        isinstance(node, ast.AnnAssign)
        and isinstance(node.target, (ast.Name, ast.Attribute))
        and (_annotates_set(node.annotation) or _is_set(node.value, known))
    ):
        return (_dotted(node.target),)
    if isinstance(node, ast.NamedExpr) and _is_set(node.value, known):
        return (_dotted(node.target),)
    return ()


def main(argv=None) -> int:
    """Run the check over the paths given, print the report, return the exit code."""
    parser = argparse.ArgumentParser(
        description="Find ordered reads of a set on the simulated path."
    )
    parser.add_argument("roots", nargs="+", help="files or directories to check")
    arguments = parser.parse_args(argv)
    lint = SetIterationLint()
    worst = 0
    for root in arguments.roots:
        code, report = lint.check(root)
        print(report)
        worst = max(worst, code)
    return worst


if __name__ == "__main__":
    sys.exit(main())
