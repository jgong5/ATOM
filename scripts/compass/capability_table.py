#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""The capability matrix of the scenario tests (`08` D120), from pytest's collection.

  capability_table.py [--log LOG] [--json] [PATH...]

Collects PATH, tests/compass by default, and prints a row for each form of
tests/compass/capabilities.py: for a supported one the highest level a case of
it reaches and its cases at each level, for a refused one its refusal test. A
collected case counts at the level its mark names; with --log, a pytest log at
-v or above (`gate_gpu.sh -vv`), only a case that log reports PASSED does.
--json prints the cases, the levels and the findings as JSON instead.

Exits 1, naming each, on a finding: a file that does not collect; a mark naming
a form the vocabulary does not hold, a refused form, or a level other than 0 or
1; a supported form with no level-0 case; a refused form whose refusal test does
not collect. A form whose entry names an open issue to add its first case, or
its refusal test, is exempt from the last two; once a case of it collects, or
its entry names its refusal test, naming the issue is itself a finding, so the
exemption ends with the case or test the issue adds.
"""

import argparse
import contextlib
import io
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
# The test files import it by this name, from the directory pytest puts first.
sys.path[:0] = [str(ROOT / "tests" / "compass"), str(ROOT)]
import capabilities

#: A test's own line at -v, and a failure's line in the short summary.
VERBOSE = re.compile(r"^(\S+::\S+) (PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)\b")
SUMMARY = re.compile(r"^(FAILED|ERROR) (\S+::\S+)")


class _Collector:
    def __init__(self) -> None:
        self.cases: list[dict] = []
        self.ids: list[str] = []
        self.errors: list[str] = []

    def pytest_collectreport(self, report) -> None:
        if report.failed:
            self.errors.append(report.nodeid)

    def pytest_collection_finish(self, session) -> None:
        for item in session.items:
            self.ids.append(item.nodeid)
            for mark in item.iter_markers(capabilities.MARK):
                args = [*mark.args, None, None]
                name = args[0] or mark.kwargs.get("name")
                level = mark.kwargs.get("level", 0 if args[1] is None else args[1])
                self.cases.append({"case": item.nodeid, "name": name, "level": level})


def collect(paths) -> _Collector:
    """What pytest collects of `paths` from the tree root, without running it."""
    collector = _Collector()
    args = ["--collect-only", "-q", "-p", "no:cacheprovider", "--rootdir", str(ROOT)]
    with contextlib.redirect_stdout(io.StringIO()):
        pytest.main([*args, *paths], plugins=[collector])
    return collector


def collects(test: str, ids) -> bool:
    """Whether `test`, ``path::function``, collects, under any parameters."""
    return any(i == test or i.startswith(test + "[") for i in ids)


def findings(cases, ids, errors=()) -> list[str]:
    """Each defect of the marks `cases` over the collected node-ids `ids`."""
    forms = capabilities.FORMS
    found = [f"{e}: does not collect, so its marks are unread" for e in errors]
    for c in cases:
        case, name, form = c["case"], c["name"], forms.get(c["name"])
        if form is None:
            found.append(
                f"{case}: marks {name!r}, a form capabilities.py does not hold"
            )
        elif form.refused:
            found.append(f"{case}: marks {name!r}, a refused form, which has no case")
        elif c["level"] not in capabilities.LEVELS:
            found.append(f"{case}: marks {name!r} at level {c['level']!r}, not 0 or 1")
    level0 = {c["name"] for c in cases if c["level"] == 0}
    marked = {c["name"] for c in cases}
    for name, form in forms.items():
        if form.issue:
            if form.test if form.refused else name in marked:
                what = "refusal test" if form.refused else "first case"
                found.append(
                    f"{name}: names {form.issue} to add its {what}, which it has; "
                    "drop the issue from its entry"
                )
            continue
        if form.refused and not collects(form.test, ids):
            found.append(
                f"{name}: refused, and its refusal test {form.test!r} does not collect"
            )
        elif not form.refused and name not in level0:
            found.append(f"{name}: supported, and no case of it collects at level 0")
    return found


def read_log(text: str) -> dict[str, bool]:
    """Each test a pytest log at -v reports, and whether only as PASSED."""
    seen: dict[str, set] = {}
    for line in text.splitlines():
        if m := VERBOSE.match(line):
            seen.setdefault(m[1], set()).add(m[2])
        elif m := SUMMARY.match(line):
            seen.setdefault(m[2], set()).add(m[1])
    return {test: statuses == {"PASSED"} for test, statuses in seen.items()}


def counted(cases, passed=None) -> list[dict]:
    """The cases that count, each of a supported form at a level: every one,
    or with `passed` (`read_log`) only one it reports PASSED."""
    forms = capabilities.FORMS
    return [
        c
        for c in cases
        if c["name"] in forms and not forms[c["name"]].refused
        if c["level"] in capabilities.LEVELS
        if passed is None or passed.get(c["case"], False)
    ]


def levels(cases) -> dict[str, int]:
    """Each form's highest level among `cases`."""
    top: dict[str, int] = {}
    for c in cases:
        top[c["name"]] = max(top.get(c["name"], -1), c["level"])
    return top


def rows(cases) -> list[str]:
    """The table over the cases that count."""
    top = levels(cases)
    out = [f"{'form':<20} {'level':<8} {'cases 0/1':<10} what"]
    for name, form in capabilities.FORMS.items():
        n = [
            sum(c["name"] == name and c["level"] == lv for c in cases) for lv in (0, 1)
        ]
        if form.refused:
            level, what = (
                "refused",
                form.test or f"its refusal test waits on {form.issue}",
            )
        else:
            level, what = str(top.get(name, "none")), form.what
            if name not in top and form.issue:
                what += f"; its first case waits on {form.issue}"
        out.append(f"{name:<20} {level:<8} {n[0]}/{n[1]:<8} {what}")
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("paths", nargs="*", default=[str(ROOT / "tests" / "compass")])
    parser.add_argument("--log", type=Path, help="a pytest log at -v or above")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    collected = collect(args.paths)
    found = findings(collected.cases, collected.ids, collected.errors)
    passed = read_log(args.log.read_text()) if args.log else None
    cases = counted(collected.cases, passed)
    if args.json:
        print(
            json.dumps(
                {"cases": collected.cases, "levels": levels(cases), "findings": found}
            )
        )
    else:
        how = (
            f"a case {args.log} reports PASSED" if args.log else "every collected case"
        )
        print(f"capability_table: levels count {how}")
        print("\n".join(rows(cases)))
        for f in found:
            print(f"capability_table: FINDING {f}")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
