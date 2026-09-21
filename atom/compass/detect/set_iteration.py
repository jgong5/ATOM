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
reported either: it is the fix.

**A report has to survive the fix it prescribes.** A name rebound to something
this pass cannot see as a set stops being one, so applying the remedy in place
-- `s = sorted(s)` -- ends the reports about `s` instead of continuing them.
That costs a miss: a genuine read earlier in the same scope goes unreported once
the name is rebound later. A check that fails the corrected code is a check
somebody switches off, so the miss is the side to be wrong on.

The miss is bounded to the one name and paid whether or not the rebinding
runs. `if flag: s = sorted(s)` silences every read of `s` in that scope,
because this is a pass over the text and not over the flow, and a branch that
is never taken is a binding all the same. Other sets in the same scope are
untouched, so the price is one name in one scope rather than the scope.

**Finding a read takes two recognitions, not one.** The expression has to be
recognised as a set, and the thing done to it has to be recognised as a read in
order. Either one missing is a miss, and the two fail for different reasons, so
they are listed separately below.

**What it recognises as a set.** A set literal or comprehension; a `set()` or
`frozenset()` call; a set operator or the method spelling of the same operator
applied to one of those, or to a `dict` view; a name or attribute assigned one
of them, in the scope that assigned it and the scopes inside that one; a
parameter or variable annotated as a set; a class or dataclass field annotated
as a set and read through `self`; and a call to a function this module itself
annotates as returning one.

**What it recognises as a read in order.** `for` and `async for`, a
comprehension or generator expression, tuple and starred unpacking, `yield
from`; `list`, `tuple`, `iter`, `reversed`, `enumerate`, `zip`, `sum`, `join`,
`map`, `filter` and `fromkeys`; `str`, `repr`, `format` and an f-string or `%`
interpolation, because a set's `repr` lists its members in iteration order;
`pop`, which takes a member out in hash order; and `min` or `max` given a `key`,
because ties there are resolved first-wins and first is whichever member the set
handed over first.

**What it does not find. This list is what has been tried, not an enumeration
of what Python allows.** It is short on both sides of the two recognitions.

A set it does not recognise: one returned by a call this module does not itself
annotate, or by one defined in another module; one arriving in an unannotated
parameter; one pulled out of a list, a dict or a tuple; any subclass of `set`;
one read through the class rather than through `self`, as `Step.members`; and a
name that is also bound, anywhere in the same scope, to something the pass
cannot see as a set -- by an assignment, an unpacking, a parameter, a `for` or
comprehension target, a `with ... as` or an import.

A read it does not recognise: `itertools.chain`, `functools.reduce`,
`heapq.nsmallest` and every other call not named above -- the list is the forms
this code is written in, not the forms that exist. `sorted(s, key=...)` belongs
on this side for a reason of its own rather than by omission: its ties keep the
set's order, so it is a real order dependence, and it is not reported because
`sorted` is the remedy this check prescribes.
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

#: The set operators. Applied to a set, or to a `dict` view, each produces a set.
SET_OPERATORS = (ast.BitOr, ast.BitAnd, ast.BitXor, ast.Sub)

#: The method spellings of those same operators, and `copy`. `s - t` and
#: `s.difference(t)` are one expression written two ways, so recognising one
#: and not the other would be two rules where the language has one.
SET_METHODS = (
    "union",
    "intersection",
    "difference",
    "symmetric_difference",
    "copy",
)

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
#: terms arrive in is the order the rounding happens in. `fromkeys` is here
#: because a dict built from a set holds the set's order and looks ordered
#: afterwards, and `str`, `repr` and `format` are here because a set's `repr`
#: lists its members in the order it holds them.
ORDERED_READS = (
    "list",
    "tuple",
    "iter",
    "reversed",
    "enumerate",
    "zip",
    "sum",
    "join",
    "map",
    "filter",
    "fromkeys",
    "str",
    "repr",
    "format",
)

#: Calls that read in order only when a `key` is given, and only in the form
#: that takes one iterable. Without a key the answer is decided by the members;
#: with one, ties are resolved first-wins, and first is whichever tied member
#: the set handed over first. `max(a, b, key=...)` is the other form: it
#: compares the arguments it is given and iterates neither of them.
KEYED_READS = ("min", "max")

#: Methods that take a member out in hash order. `pop` is a mutation as well,
#: which is why it is easy to file under the mutations that are not reported.
SET_DRAINS = ("pop",)


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
        for node, scope in _scoped(tree):
            for expression, form in _reads(node):
                if not _is_set(expression, known, scope):
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
            "nothing to point at. Sort at the point of iteration, or decide the "
            "order once from a sorted list and carry that: a dict keyed from "
            "sorted(s) holds an order, a dict keyed from s holds the hash table."
        )
        return "\n".join(lines)

    def check(self, root: str) -> tuple[int, str]:
        """Scan `root` and return an exit code beside the report. Non-zero fails CI."""
        modules = self.modules(root)
        reads = self.scan_modules(modules)
        return (1 if reads else 0), self.report(reads, len(modules))


def _scoped(node, scope=()):
    """Every node under `node`, each with the def and class path enclosing it.

    A name means different things in different scopes, and one flat table of
    names makes a short name bound in one function a set in every other one.
    """
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            inner = scope + (("def", child.name),)
        elif isinstance(child, ast.ClassDef):
            inner = scope + (("class", child.name),)
        else:
            inner = scope
        yield child, inner
        yield from _scoped(child, inner)


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
    if isinstance(node, ast.FormattedValue):
        return ((node.value, "formatted string of"),)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
        filled = node.right
        parts = filled.elts if isinstance(filled, ast.Tuple) else (filled,)
        return tuple((part, "formatted string of") for part in parts)
    if isinstance(node, ast.Assign) and any(
        isinstance(target, (ast.Tuple, ast.List)) for target in node.targets
    ):
        return ((node.value, "unpacking of"),)
    if isinstance(node, ast.Call):
        tail = _tail(node.func)
        if tail in SET_DRAINS and isinstance(node.func, ast.Attribute):
            return ((node.func.value, f"{tail}() from"),)
        if tail in ORDERED_READS:
            return tuple((argument, f"{tail}() over") for argument in node.args)
        if (
            tail in KEYED_READS
            and len(node.args) == 1
            and any(word.arg == "key" for word in node.keywords)
        ):
            return ((node.args[0], f"{tail}(key=...) over"),)
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


def _is_set(node, known, scope) -> bool:
    """Whether this expression is a set, as far as the module's own text says."""
    if isinstance(node, (ast.Set, ast.SetComp)):
        return True
    if isinstance(node, ast.Call):
        tail = _tail(node.func)
        if tail in SET_BUILDERS:
            return True
        if tail in SET_METHODS and isinstance(node.func, ast.Attribute):
            return _is_set(node.func.value, known, scope) or _is_view(node.func.value)
        return known.returns_a_set(scope, _dotted(node.func))
    if isinstance(node, (ast.Name, ast.Attribute)):
        return known.holds_a_set(scope, _dotted(node))
    if isinstance(node, ast.BinOp) and isinstance(node.op, SET_OPERATORS):
        return any(
            _is_set(side, known, scope) or _is_view(side)
            for side in (node.left, node.right)
        )
    if isinstance(node, ast.IfExp):
        return _is_set(node.body, known, scope) or _is_set(node.orelse, known, scope)
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


class _Names:
    """What the module says each name holds, in the scope that says it.

    Two things this has to get right that a flat table of names cannot. A name
    is only a set inside the scope that bound it and the scopes nested in that
    one, or `x` bound to a set in one function makes every other function's `x`
    a set. And a name bound anywhere in its scope to something that is not a
    recognised set is not claimed at all, or applying the fix in place leaves
    the report standing and the check reports its own remedy.
    """

    def __init__(self) -> None:
        self.sets: set[tuple[tuple, str]] = set()
        self.others: set[tuple[tuple, str]] = set()
        self.returns: set[tuple[tuple, str]] = set()

    def learn(self, scope, name: str, table) -> bool:
        """Write one binding down. True if this is the first time it is seen."""
        key = (_owner(scope, name), name)
        if key in table:
            return False
        table.add(key)
        return True

    def holds_a_set(self, scope, name) -> bool:
        """Whether a name read in `scope` holds a set here."""
        return self._look(scope, name, self.sets, self.others)

    def returns_a_set(self, scope, name) -> bool:
        """Whether a call made in `scope` reaches a function annotated to return one."""
        return self._look(scope, name, self.returns, frozenset())

    def _look(self, scope, name, yes, no) -> bool:
        if name is None:
            return False
        start = _owner(scope, name)
        levels = (start,) if name.startswith("self.") else _outward(start)
        for level in levels:
            if (level, name) in no:
                return False
            if (level, name) in yes:
                return True
        return False


def _owner(scope, name: str):
    """Where a name is written down. `self.x` belongs to its class, not its method."""
    if not name.startswith("self."):
        return scope
    for depth in range(len(scope), 0, -1):
        if scope[depth - 1][0] == "class":
            return scope[:depth]
    return scope


def _outward(scope):
    """The scopes a bare name is looked up in: this one, then out past the classes.

    A class body is skipped on the way out because a method does not see its
    class's names without going through `self`, and treating it as though it
    did makes a class attribute shadow a global of the same name.
    """
    levels = [scope[:depth] for depth in range(len(scope), -1, -1)]
    return levels[:1] + [
        level for level in levels[1:] if not (level and level[-1][0] == "class")
    ]


def _declared(scope, name):
    """The names one binding writes down.

    A name bound in a class body is also reachable as `self.<name>` from every
    method of that class, and an annotation-only field -- which is every field
    of a frozen dataclass -- is reachable no other way.
    """
    if name is None:
        return ()
    if scope and scope[-1][0] == "class" and "." not in name:
        return (name, f"self.{name}")
    return (name,)


def _set_valued_names(tree) -> _Names:
    """Every name this module says holds a set, and every name it says does not.

    The sets are taken to a fixed point first, because one name can be bound
    from another that a later line binds and the walk order is not the source
    order. What is *not* a set is collected afterwards, against the finished
    table, so a name waiting on a later binding is not written off on the way.
    The second pass is also where every binder that cannot bind a set lands,
    which is what stops an outer claim reaching a name that shadows it.
    """
    known = _Names()
    for node, scope in _scoped(tree):
        if isinstance(node, ast.arg) and _annotates_set(node.annotation):
            known.learn(scope, node.arg, known.sets)
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if _annotates_set(node.returns):
            for name in _declared(scope[:-1], node.name):
                known.learn(scope[:-1], name, known.returns)
    grew = True
    while grew:
        grew = False
        for node, scope in _scoped(tree):
            for name, holds_set in _binds(node, known, scope):
                if holds_set:
                    grew |= known.learn(scope, name, known.sets)
    for node, scope in _scoped(tree):
        for name, holds_set in _binds(node, known, scope):
            if not holds_set:
                known.learn(scope, name, known.others)
    return known


def _binds(node, known, scope):
    """Every name this statement binds, and whether it binds it to a set.

    Three of these can bind a set. The rest cannot bind one this pass can see,
    and they are here for the other half of the same rule: a name bound to
    something unrecognised is not claimed, so a binder that is missing from
    this list lets an outer claim through its own shadow and reports a name at
    a line that is not reading that set. An unannotated parameter, a `for` or
    comprehension target, a `with ... as`, an import and a tuple or list
    unpacking are all that shadow. `except ... as` is left out: the name is
    an exception, and nothing here iterates one.
    """
    if isinstance(node, ast.Assign):
        holds_set = _is_set(node.value, known, scope)
        bound = []
        for target in node.targets:
            if isinstance(target, (ast.Name, ast.Attribute)):
                bound.extend(
                    (name, holds_set) for name in _declared(scope, _dotted(target))
                )
            else:
                # An unpacking hands out members, not the container, so what
                # the value is says nothing about what the names hold.
                bound.extend(_shadows(scope, _bound_names(target)))
        return tuple(bound)
    if isinstance(node, ast.AnnAssign) and isinstance(
        node.target, (ast.Name, ast.Attribute)
    ):
        holds_set = _annotates_set(node.annotation) or _is_set(node.value, known, scope)
        return tuple(
            (name, holds_set) for name in _declared(scope, _dotted(node.target))
        )
    if isinstance(node, ast.NamedExpr):
        holds_set = _is_set(node.value, known, scope)
        return tuple(
            (name, holds_set) for name in _declared(scope, _dotted(node.target))
        )
    if isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
        return _shadows(scope, _bound_names(node.target))
    if isinstance(node, ast.arg):
        if _annotates_set(node.annotation):
            return ()
        return _shadows(scope, (node.arg,))
    if isinstance(node, ast.withitem):
        return _shadows(scope, _bound_names(node.optional_vars))
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return _shadows(
            scope,
            tuple(alias.asname or alias.name.split(".")[0] for alias in node.names),
        )
    return ()


def _bound_names(target):
    """Every name a binding target writes to, through tuples, lists and stars."""
    if isinstance(target, (ast.Name, ast.Attribute)):
        return (_dotted(target),)
    if isinstance(target, ast.Starred):
        return _bound_names(target.value)
    if isinstance(target, (ast.Tuple, ast.List)):
        return tuple(name for element in target.elts for name in _bound_names(element))
    return ()


def _shadows(scope, names):
    """Those names, each bound to something this pass cannot see as a set."""
    return tuple((name, False) for bound in names for name in _declared(scope, bound))


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
