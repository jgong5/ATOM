# SPDX-License-Identifier: MIT
"""A static pass for reads of a real clock on the simulated path.

Time on the simulated path comes from the run's own clock. A call to the
machine's clock returns real seconds, and real seconds put into a record of
modelled ones are not visibly different from the modelled ones -- the number
has the right magnitude, the table still adds up, and the only sign of trouble
is that two timestamps on one timeline disagree by a margin nobody is looking
for. So this is a static check: the read is a failure on the day it is written,
not on the day a number taken off it is argued about.

Reads are found by parsing rather than by matching text, because the forms that
matter are not all spelled the same. `from time import monotonic` followed by a
bare `monotonic()` is the same read as `time.monotonic()`, and a text search for
the second does not find the first. Imports and the aliases assigned from them
are resolved to the name they came from, and the call target is compared against
that -- so `now = time.monotonic` followed by `now()` is found as well, a form a
text search for the assignment *would* have found and an earlier parse of only
the imports did not.

**What this pass does not see.** Naming the blind spots is part of the check,
because a check whose limits are unstated gets read as a guarantee:

* a read reached through a string -- `getattr(time, "perf_counter")()`, or
  `importlib.import_module("time")` -- because the name is not in the syntax;
* `from time import *`, because the names it binds are in the module it imports
  and not in this one's text;
* a name bound to the result of a call rather than to a name, since only a
  dotted name on the right of an assignment is followed;
* scope: a name bound anywhere in the file is treated as bound everywhere in
  it, so a local alias in one function is resolved in another;
* **a clock handed around as a value rather than assigned to a name.** Only an
  import and a plain assignment are followed, so a parameter default
  (`def step(wall_clock=time.monotonic)`), a walrus, a tuple unpacking, a class
  attribute read back through the class, a dict entry read back by subscript
  and a `functools.partial` all carry the clock past this pass. They are one
  class -- the callable is bound somewhere the parser does not follow and
  called later -- and the parameter default is the form that appears in real
  code, which is also why a bare attribute load is not treated as a read.

Each of those is a read this pass would miss and a reviewer would not, which is
the trade the pass is worth making and not a reason to trust it alone. The list
is what has been tried against it, not an enumeration of what Python allows.

**Pacing a loop is the same mistake and only one form of it is listed.**
`asyncio.sleep` is flagged for that reason, and `time.sleep`,
`threading.Event.wait(timeout=)` and `queue.Queue.get(timeout=)` are the same
mistake spelled differently: each one turns a loop at a rate the modelled run
knows nothing about. They are one class and they are worth taking as a class,
against an inventory of the waits this code actually performs, rather than one
name at a time here.

**The allow-list is part of the check, not an escape from it.** A few reads are
supposed to stay real -- a failure detector that measured itself in simulated
seconds would stop whenever the thing it is watching stopped. Each entry names
the file and the reason it stays real, so the list is reviewable, and adding to
it is a decision somebody writes down rather than a silence.
"""

import argparse
import ast
import os
import sys
from dataclasses import dataclass

#: The calls that return real seconds. `asyncio.sleep` is here with the rest
#: because pacing a loop against the machine's clock is the same mistake as
#: reading it: the loop turns at a rate the modelled run does not know about.
#: The `_ns` forms are the same machine clocks in different units, and a number
#: divided by a billion after the fact is not distinguishable from one taken in
#: seconds, so they are listed beside them rather than under them.
#:
#: **Every entry is the fully qualified name, because the comparison is exact.**
#: A tail match would hand the whole list to any receiver whose attribute
#: happens to be spelled the same -- `self.time.monotonic()`, `row.datetime.now()`,
#: a `time` imported from somewhere that is not the standard library. The two
#: `datetime` entries are written out to `datetime.datetime` for the same
#: reason; `from datetime import datetime` resolves to exactly that, so the
#: short spelling costs nothing and the long one cannot be reached by accident.
CLOCK_READS = (
    "time.time",
    "time.time_ns",
    "time.monotonic",
    "time.monotonic_ns",
    "time.perf_counter",
    "time.perf_counter_ns",
    "time.clock_gettime",
    "time.clock_gettime_ns",
    "datetime.datetime.now",
    "datetime.datetime.utcnow",
    "asyncio.sleep",
)

#: Files whose real-clock reads are deliberate, and why. A path matches on whole
#: directory names ending at the entry, so the same entry works from any root
#: the check is run from and a directory that merely ends in the same letters
#: does not collect the exemption.
DEFAULT_ALLOW_LIST = {
    "atom/compass/detect/watchdog.py": (
        "the annotation watchdog measures how long a participant has been "
        "executing in real seconds; measured in simulated ones it would stop "
        "whenever the participant it is watching stopped"
    ),
}


@dataclass(frozen=True)
class ClockRead:
    """One call to a real clock: where it is, what it calls, and inside what."""

    path: str
    line: int
    call: str
    scope: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}  {self.call}  in {self.scope}"


class ClockSourceLint:
    """Parses the simulated path and reports every real-clock read it finds.

    `enabled=False` exists to show what the tree looks like with the check off,
    which is a tree that compiles, imports and runs.
    """

    def __init__(self, allow_list=DEFAULT_ALLOW_LIST, enabled: bool = True) -> None:
        self.allow_list = dict(allow_list)
        self.enabled = enabled

    def allowed(self, path: str) -> str | None:
        """The recorded reason this file may read a real clock, if it may.

        The match is on whole path segments. A suffix on the text alone would
        hand the exemption to any directory whose name happens to end in the
        first segment of an entry, which is a file nobody reviewed.
        """
        posix = path.replace(os.sep, "/")
        for entry in sorted(self.allow_list):
            if posix == entry or posix.endswith("/" + entry):
                return self.allow_list[entry]
        return None

    def scan_source(self, source: str, path: str) -> tuple[ClockRead, ...]:
        """Every real-clock read in one module's text, in line order.

        A call counts when its resolved name is exactly an entry in the list.
        Matching a tail instead would excuse nothing and accuse plenty: any
        receiver whose attribute is spelled like a module would collect the
        whole list, which is the same unanchored comparison the allow-list
        above had and the same answer.
        """
        if not self.enabled or self.allowed(path) is not None:
            return ()
        tree = ast.parse(source, filename=path)
        origins = _origins(tree)
        scopes = _scopes(tree)
        found = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            call = _resolve(_dotted(node.func), origins)
            if call is None:
                continue
            for read in CLOCK_READS:
                if call == read:
                    found.append(
                        ClockRead(
                            path, node.lineno, call, _scope_at(scopes, node.lineno)
                        )
                    )
                    break
        return tuple(sorted(found, key=lambda read: (read.path, read.line, read.call)))

    def scan_modules(self, paths) -> tuple[ClockRead, ...]:
        """Every read in the modules named, in the order they were named."""
        found = []
        for path in paths:
            with open(path, encoding="utf-8") as handle:
                found.extend(self.scan_source(handle.read(), path))
        return tuple(found)

    def scan_tree(self, root: str) -> tuple[ClockRead, ...]:
        """Every read under `root`, over every module in it, in path order."""
        return self.scan_modules(self.modules(root))

    @staticmethod
    def modules(root: str) -> tuple[str, ...]:
        """Every module under `root`, in a fixed order on every machine."""
        if os.path.isfile(root):
            return (root,)
        found = []
        for directory, subdirectories, names in os.walk(root):
            subdirectories[:] = sorted(
                name for name in subdirectories if name != "__pycache__"
            )
            found.extend(
                os.path.join(directory, name)
                for name in sorted(names)
                if name.endswith(".py")
            )
        return tuple(found)

    def report(self, reads, scanned: int, allow_listed: int = 0) -> str:
        """What the check prints. Clean is a line; dirty is the list and the fix.

        `allow_listed` is how many of the files in this scan were passed over,
        not how long the list is. The line reads as a statement about the scan,
        so it has to be one: a tree containing none of the listed files is a
        tree where nothing was exempted.
        """
        if not self.enabled:
            return "clock-source lint: disabled, so this tree makes no claim about its clock reads"
        if not reads:
            return (
                f"clock-source lint: clean over {scanned} module(s), "
                f"{allow_listed} file(s) allow-listed"
            )
        lines = [
            f"clock-source lint: {len(reads)} real-clock read(s) on the simulated path:"
        ]
        lines.extend(f"  {read}" for read in reads)
        lines.append(
            "A run charges simulated seconds for the work it models. A real read "
            "puts machine seconds into the same record, and the mixture is reported "
            "as one modelled number. Take the time from the run's clock, or add the "
            "file to the allow-list with the reason it stays real."
        )
        return "\n".join(lines)

    def check(self, root: str) -> tuple[int, str]:
        """Scan `root` and return an exit code beside the report. Non-zero fails CI."""
        modules = self.modules(root)
        reads = self.scan_modules(modules)
        allow_listed = sum(1 for path in modules if self.allowed(path) is not None)
        return (1 if reads else 0), self.report(reads, len(modules), allow_listed)


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


def _origins(tree) -> dict[str, str]:
    """Local name -> the dotted name it stands for: imports, and aliases of them.

    An alias is an assignment whose right-hand side is a dotted name, and it is
    recorded whole -- `self._clock = time.perf_counter` binds `self._clock`, not
    `self` -- so that the call it enables resolves the same way an import does.
    The first binding a name gets is the one kept: a name assigned twice says
    two things, and the check is not an interpreter.
    """
    origins: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                origins.setdefault(alias.asname or alias.name.split(".")[0], alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                origins.setdefault(
                    alias.asname or alias.name, f"{node.module}.{alias.name}"
                )
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            source = _dotted(node.value)
            if source is None:
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                name = _dotted(target)
                if name is not None and name != source:
                    origins.setdefault(name, source)
    return origins


def _resolve(call: str | None, origins: dict[str, str]) -> str | None:
    """Rewrite a call's head through the imports and aliases that introduced it.

    An alias can stand on an alias, and `ast.walk` does not visit the bindings
    in the order they were written, so the chain is followed rather than
    stepped once. Each name is expanded at most once along a chain: a name can
    legitimately stand for one that contains it -- `from datetime import
    datetime` binds `datetime` to `datetime.datetime` -- and expanding that a
    second time would grow the name instead of resolving it.
    """
    if call is None:
        return None
    expanded: dict[str, None] = {}
    while True:
        origin = origins.get(call)
        if origin is not None:
            if call in expanded:
                return call
            expanded[call] = None
        else:
            head, _, rest = call.partition(".")
            head_origin = origins.get(head)
            if head_origin is None or head in expanded:
                return call
            expanded[head] = None
            origin = f"{head_origin}.{rest}" if rest else head_origin
        if origin == call:
            return call
        call = origin


def _scopes(tree) -> tuple[tuple[int, int, str], ...]:
    """Every def and class as (first line, last line, name), outermost first."""
    return tuple(
        (node.lineno, node.end_lineno or node.lineno, node.name)
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    )


def _scope_at(scopes, line: int) -> str:
    """The innermost def or class containing `line`, or module level."""
    inner = [scope for scope in scopes if scope[0] <= line <= scope[1]]
    return max(inner)[2] if inner else "<module>"


def main(argv=None) -> int:
    """Run the check over the paths given, print the report, return the exit code."""
    parser = argparse.ArgumentParser(
        description="Find real-clock reads on the simulated path."
    )
    parser.add_argument("roots", nargs="+", help="files or directories to check")
    arguments = parser.parse_args(argv)
    lint = ClockSourceLint()
    worst = 0
    for root in arguments.roots:
        code, report = lint.check(root)
        print(report)
        worst = max(worst, code)
    return worst


if __name__ == "__main__":
    sys.exit(main())
