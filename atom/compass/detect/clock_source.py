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
the second does not find the first. Imports are resolved to the name they came
from and the call target is compared against that.

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
CLOCK_READS = (
    "time.time",
    "time.monotonic",
    "time.perf_counter",
    "datetime.now",
    "asyncio.sleep",
)

#: Files whose real-clock reads are deliberate, and why. A path matches by
#: suffix, so the same entry works from any root the check is run from.
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
        """The recorded reason this file may read a real clock, if it may."""
        posix = path.replace(os.sep, "/")
        for entry in sorted(self.allow_list):
            if posix.endswith(entry):
                return self.allow_list[entry]
        return None

    def scan_source(self, source: str, path: str) -> tuple[ClockRead, ...]:
        """Every real-clock read in one module's text, in line order."""
        if not self.enabled or self.allowed(path) is not None:
            return ()
        tree = ast.parse(source, filename=path)
        origins = _import_origins(tree)
        scopes = _scopes(tree)
        found = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            call = _resolve(_dotted(node.func), origins)
            if call is None:
                continue
            for read in CLOCK_READS:
                if call == read or call.endswith("." + read):
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

    def report(self, reads, scanned: int) -> str:
        """What the check prints. Clean is a line; dirty is the list and the fix."""
        if not self.enabled:
            return "clock-source lint: disabled, so this tree makes no claim about its clock reads"
        if not reads:
            return (
                f"clock-source lint: clean over {scanned} module(s), "
                f"{len(self.allow_list)} file(s) allow-listed"
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
        return (1 if reads else 0), self.report(reads, len(modules))


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


def _import_origins(tree) -> dict[str, str]:
    """Local name -> the dotted name it was imported from."""
    origins = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                origins[alias.asname or alias.name.split(".")[0]] = alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                origins[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return origins


def _resolve(call: str | None, origins: dict[str, str]) -> str | None:
    """Rewrite a call's head through the imports that introduced it."""
    if call is None:
        return None
    head, _, rest = call.partition(".")
    origin = origins.get(head)
    if origin is None:
        return call
    return f"{origin}.{rest}" if rest else origin


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
