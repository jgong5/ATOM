#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Accidental-tightness detector for compass refusal assertions.

Run as ``python tools/compass/detect_loose_refusals.py <tree root>``.

For every ``pytest.raises``/``pytest.warns`` ``match=`` needle under
``tests/compass/``, count the production raise templates under
``atom/compass/`` that the needle matches.  Two readings are printed:

  broad  -- the needle matches raise sites in >= 2 distinct places anywhere in
            production.  Over-counts: two refusals in unrelated modules are not
            alternatives, so no branch inversion can confuse them.
  sharp  -- the needle matches >= 2 raise sites inside ONE production function.

A template is read as its constant text runs: an interpolation ends one run and
starts the next, so a needle counts only when it lies inside a single run.  A
needle spanning an interpolated value is not an assertion on the sentence.

**This is a tool, not a gate and not a test.**  It is run by hand when a refusal
is added or an assertion is written, and it is wired into nothing.  Sharp is a
filter, not a verdict; what it flags is then classified by hand against:

    load-bearing when the two refusals are alternatives over one predicate on
    the same input, so a plausible branch inversion routes the test's own input
    to the other.

A count over the tree is an aggregate, and an aggregate is not a result without
the per-site lists printed beside it, which is why both are printed.

What this tool cannot see -- five, and a clean run means nothing without them
=============================================================================

1. **One template carrying two faults is invisible.**  A template counter can
   never flag it, because one template is never two -- and that single-template
   shape is the defect class this detector was built for.  A one-raise function
   cannot produce a within-function count at all, so when such a site is flagged
   it is flagged for some other reason.

2. **A structured assertion is invisible.**  A test that pins a refusal through
   its attributes -- comparing a list on the raised exception rather than
   matching its text -- is tight in a way no needle-matching pass can read, and
   is absent from this population entirely.

3. **Every string-bearing argument is read, not just the first.**  This is the
   one blind spot fixed here rather than reported.  A refusal raised as
   ``SpecRefusal(Rule.X, what, remedy)`` states nothing in its first argument, so
   an earlier reading of ``args[0]`` alone dropped the whole site before any
   needle was tried.  Measured at 92f1fdafe28817f864261f632d98cf5b052016dd: of
   the 121 ``raise <Exc>(...)`` sites with arguments under ``atom/compass/``, 19
   carry no text in the first -- 15% -- and 18 of those state their fault in a
   later argument, every one of them in ``atom/compass/spec/``.  A needle
   ambiguous only among those refusals read clean.  Runs are now collected from
   every positional and keyword argument of the raised call, each argument's runs
   kept separate, so a needle still cannot span an argument boundary.

4. **A non-constant needle cannot be read, and is reported rather than
   dropped.**  An f-string ``match=`` needle is no string constant, so its
   rendered text is unknown here and it leaves the population.  Every such
   needle is printed by name, because a broad flag that disappears when an
   assertion is rewritten as an f-string must not be read as a cleared one --
   the tool cannot score a remedy written in that form.  A needle held in a
   variable is caught by this too, and named.

5. **A needle behind a helper leaves the population with no word at all.**  A
   ``match=`` keyword is read only off a call written as ``raises`` or ``warns``,
   so an assertion wrapped in a project helper is not merely unreadable -- it is
   never seen, which is strictly worse than the case above, where the needle is
   at least named.  Measured at 92f1fdafe28817f864261f632d98cf5b052016dd: the 78
   ``match=`` occurrences under ``tests/compass/`` are 72 read, 1 named as
   unreadable, and 5 inside this tool's own fixtures and prose.  There is no
   helper-wrapped assertion today, so this is a forward gap, not a present hole.
"""

from __future__ import annotations

import ast
import re
import sys
from collections import defaultdict
from operator import itemgetter
from pathlib import Path

ASSERTION_HELPERS = {"raises", "warns"}
SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def literal_runs(node: ast.AST) -> list[str]:
    """The constant text runs of a message expression, interpolations excluded."""
    if isinstance(node, ast.Constant):
        return [node.value] if isinstance(node.value, str) else []
    if isinstance(node, ast.JoinedStr):
        runs = [""]
        for part in node.values:
            if isinstance(part, ast.Constant) and isinstance(part.value, str):
                runs[-1] += part.value
            else:
                runs.append("")
        return [text for text in runs if text]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return literal_runs(node.left) + literal_runs(node.right)
    return []


def raise_sites(source: str, label: str) -> tuple[list[dict], list[dict]]:
    """The ``raise Exc(...)`` sites in *source* that carry text, and those that do not.

    Text is read from every string-bearing argument of the raised call, keyword
    arguments included; runs from different arguments stay separate.  A refusal
    that passes arguments but states nothing readable in any of them goes into
    the second list under its own name, so half the population is not an
    aggregate the output gives no account of.
    """
    sites: list[dict] = []
    opaque: list[dict] = []

    def walk(node: ast.AST, scope: str) -> None:
        for child in ast.iter_child_nodes(node):
            inner = scope
            if isinstance(child, SCOPES):
                inner = f"{scope}.{child.name}".lstrip(".")
            if isinstance(child, ast.Raise) and isinstance(child.exc, ast.Call):
                args = [*child.exc.args, *(kw.value for kw in child.exc.keywords)]
                runs = [run for arg in args for run in literal_runs(arg)]
                where = {"label": label, "line": child.lineno}
                if runs:
                    sites.append({**where, "scope": scope, "runs": runs})
                elif args:
                    kinds = ", ".join(sorted({type(arg).__name__ for arg in args}))
                    opaque.append(
                        {**where, "why": f"no text in any argument ({kinds})"}
                    )
            walk(child, inner)

    walk(ast.parse(source), "")
    return sites, opaque


def match_needles(source: str, label: str) -> tuple[list[dict], list[dict]]:
    """The readable ``match=`` needles in *source*, and the unreadable ones.

    A needle that is not a string constant, or that is not a usable regex, goes
    into the second list under its own name rather than being discarded, because
    a flag that disappears when an assertion is rewritten is not a cleared one.
    """
    kept: list[dict] = []
    dropped: list[dict] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name not in ASSERTION_HELPERS:
            continue
        for keyword in node.keywords:
            if keyword.arg != "match":
                continue
            where = {"label": label, "line": node.lineno}
            value = keyword.value
            if not (isinstance(value, ast.Constant) and isinstance(value.value, str)):
                kind = type(value).__name__
                dropped.append({**where, "why": f"not a string constant ({kind})"})
                continue
            try:
                re.compile(value.value)
            except re.error as error:
                dropped.append({**where, "why": f"not a usable regex ({error})"})
                continue
            kept.append({**where, "needle": value.value})
    line = itemgetter("line")
    return sorted(kept, key=line), sorted(dropped, key=line)


def score(sites: list[dict], needles: list[dict]) -> tuple[list, list]:
    """Split *needles* into the broad flags and the sharp ones."""
    broad, sharp = [], []
    for entry in needles:
        pattern = re.compile(entry["needle"])
        hits = [s for s in sites if any(pattern.search(run) for run in s["runs"])]
        if len(hits) >= 2:
            broad.append((entry, hits))
        by_scope = defaultdict(list)
        for site in hits:
            by_scope[(site["label"], site["scope"])].append(site)
        groups = {key: hit for key, hit in by_scope.items() if len(hit) >= 2}
        if groups:
            sharp.append((entry, groups))
    return broad, sharp


def main(root: Path) -> int:
    sites: list[dict] = []
    opaque: list[dict] = []
    for path in sorted((root / "atom" / "compass").rglob("*.py")):
        found, blank = raise_sites(path.read_text(), str(path.relative_to(root)))
        sites += found
        opaque += blank
    needles: list[dict] = []
    dropped: list[dict] = []
    for path in sorted((root / "tests" / "compass").rglob("*.py")):
        kept, lost = match_needles(path.read_text(), str(path.relative_to(root)))
        needles += kept
        dropped += lost

    # An empty population prints zeroes that read exactly like a clean tree, in
    # a tool whose root is typed by hand -- the same silent-empty-collection
    # fault this pass exists to find. Say so instead of reporting a clean run.
    if not sites or not needles:
        print(
            f"REFUSED: {root} gave {len(sites)} production raise sites and "
            f"{len(needles)} needles. A tree this tool can read has both, under "
            f"atom/compass/ and tests/compass/; zero of either is a wrong root, "
            f"not a clean result.",
            file=sys.stderr,
        )
        return 2

    broad, sharp = score(sites, needles)

    print(
        f"production raise sites carrying text: {len(sites)}\n"
        f"production refusals stating no text:  {len(opaque)}\n"
        f"match= needles read:                  {len(needles)}\n"
        f"match= needles unreadable, dropped:   {len(dropped)}"
    )
    for entry in [*opaque, *dropped]:
        print(f"  ! {entry['label']}:{entry['line']}  {entry['why']}")

    print(f"\nBROAD  (needle matches >= 2 production raise sites): {len(broad)}")
    for entry, hits in broad:
        where = ", ".join(f"{s['label']}:{s['line']}" for s in hits)
        print(f"  {entry['label']}:{entry['line']}  {entry['needle']!r} -> {where}")

    print(f"\nSHARP  (>= 2 matching raise sites in ONE function): {len(sharp)}")
    for entry, groups in sharp:
        print(f"  {entry['label']}:{entry['line']}  {entry['needle']!r}")
        for (label, scope), group in groups.items():
            lines = ", ".join(str(s["line"]) for s in group)
            print(f"      {label}::{scope or '<module>'} at {lines}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <tree root>", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(main(Path(sys.argv[1])))
