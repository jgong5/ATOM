# SPDX-License-Identifier: MIT
"""The classified list of blocking calls on the serving path, kept honest.

A simulated run substitutes predicted durations for real work, so it has to
know every call that parks a thread on the real clock. The list of them is in
`atom/compass/clock/sync_sites.json`; deciding what each one is was a reading
job and cannot be re-derived. What *can* be re-derived is the set of call sites
that exist, and these tests assert the two agree -- so a blocking call added to
ATOM later fails here instead of quietly widening the blind spot.

Three failures are worth telling apart, and each has its own test:

* a call site the list does not classify, or a listed site that has gone away
  (`test_every_scanned_site_is_classified`) -- read it and classify it;
* a site whose recorded line has moved (`test_recorded_lines_match_the_tree`)
  -- update the line, the classification still holds;
* a pinned line of text that is no longer there
  (`test_anchor_lines_are_still_where_they_say`) -- the same, for the points
  that are not a call.

No driver, and no import of ATOM's serving modules: the scanner parses them.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from atom.compass.clock import sync_scan

TREE = sync_scan.repo_root_from_here()
INVENTORY = sync_scan.load_inventory()
CATEGORIES = set(INVENTORY["categories"])
README = Path(sync_scan.__file__).with_name("README.md")


@pytest.fixture(scope="module")
def scanned():
    return {site.id: site for site in sync_scan.scan(TREE)}


@pytest.fixture(scope="module")
def listed():
    return {row["id"]: row for row in INVENTORY["sites"]}


def test_every_scanned_site_is_classified(scanned, listed):
    """The two sets are equal, and the message says which way they differ."""
    missing = sorted(set(scanned) - set(listed))
    stale = sorted(set(listed) - set(scanned))
    assert not missing, (
        f"{len(missing)} call site(s) on the serving path are not classified. "
        "Read each one, decide what a simulated run does about it, and add a "
        f"row to {sync_scan.INVENTORY_PATH.name}: " + ", ".join(missing)
    )
    assert not stale, (
        f"{len(stale)} classified site(s) no longer exist in the tree; remove "
        "their rows: " + ", ".join(stale)
    )


def test_recorded_lines_match_the_tree(scanned, listed):
    moved = [
        f"{row['id']} recorded at line {row['line']}, found at {scanned[key].line}"
        for key, row in listed.items()
        if key in scanned and scanned[key].line != row["line"]
    ]
    assert not moved, "recorded lines are stale: " + "; ".join(moved)


def test_anchor_lines_are_still_where_they_say():
    wrong = []
    for row in INVENTORY["anchors"]:
        lines = (TREE / row["file"]).read_text(encoding="utf-8").splitlines()
        line = row["line"]
        if not (1 <= line <= len(lines)) or row["anchor"] not in lines[line - 1]:
            wrong.append(f"{row['file']}:{line} no longer holds {row['anchor']!r}")
    assert not wrong, "; ".join(wrong)


def test_every_row_carries_a_category_and_a_reason():
    for row in INVENTORY["sites"] + INVENTORY["anchors"]:
        key = row.get("id") or f"{row['file']}:{row['line']}"
        assert row["category"] in CATEGORIES, f"{key}: {row['category']}"
        assert len(row["why"]) > 20, f"{key}: the justification is too short"
        assert row["peer"] in {"none", "thread", "process", "deployment"}, key


def test_scanned_and_unscanned_roots_all_exist():
    for rel, _why in sync_scan.SCANNED_ROOTS + sync_scan.UNSCANNED_ROOTS:
        assert (TREE / rel).exists(), f"{rel} is named in the scanner but is gone"


def test_readme_counts_match_the_rows():
    """The count per category is stated once in prose and derived once here."""
    counts = sync_scan.category_counts(INVENTORY)
    stated = {
        m.group(1): int(m.group(2))
        for m in re.finditer(
            r"^\| (A|B|C1|C2|ignore|undecided) \| (\d+) \|",
            README.read_text(encoding="utf-8"),
            re.MULTILINE,
        )
    }
    assert stated == counts


# --- the shape rules, against the forms they are written to tell apart ------


def _first_call(src: str) -> ast.Call:
    return next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Call))


@pytest.mark.parametrize(
    "src,blocking",
    [
        ("q.get()", True),
        ("q.get(timeout=5)", True),
        ("q.get(False)", True),
        ('d.get("key")', False),
        ('d.get("key", None)', False),
    ],
)
def test_a_mapping_get_is_not_a_queue_get(src, blocking):
    assert sync_scan._no_positional(_first_call(src)) is blocking


@pytest.mark.parametrize(
    "src,blocking",
    [
        ("t.join()", True),
        ("t.join(5)", True),
        ("t.join(timeout=1)", True),
        ('",".join(parts)', False),
        ("os.path.join(a, b)", False),
    ],
)
def test_a_string_join_is_not_a_thread_join(src, blocking):
    assert sync_scan._join_form(_first_call(src)) is blocking


@pytest.mark.parametrize(
    "src,blocking",
    [
        ('mgr.call_func("forward", batch, wait_out=True)', True),
        ('mgr.call_func("process_kvconnector_output", meta)', False),
        ('mgr.call_func("x", wait_out=False)', False),
        ('mgr.call_func_with_aggregation("async_proc_aggregation")', True),
    ],
)
def test_a_worker_call_parks_only_when_it_asks_for_the_reply(src, blocking):
    assert sync_scan._worker_rpc_form(_first_call(src)) is blocking


def test_a_new_blocking_call_is_reported():
    """What the completeness test sees the day someone adds one."""
    src = "def handler(self):\n    item = self._inbox.get()\n    return item\n"
    visitor = sync_scan._CallVisitor("made/up.py", src)
    visitor.visit(ast.parse(src))
    assert [(s.symbol, s.call, s.shape) for s in visitor.sites] == [
        ("handler", "self._inbox.get", "queue_get")
    ]


def test_a_loop_that_calls_nothing_that_parks_is_reported_as_a_spin():
    spin = (
        "def f(self):\n"
        "    while self.pending:\n"
        "        if self.ready:\n"
        "            continue\n"
        "        break\n"
    )
    visitor = sync_scan._SpinVisitor("made/up.py", spin)
    visitor.visit(ast.parse(spin))
    assert [s.shape for s in visitor.sites] == ["spin_loop"]

    paced = (
        "import time\n"
        "def f(self):\n"
        "    while self.pending:\n"
        "        time.sleep(0.1)\n"
        "        continue\n"
    )
    visitor = sync_scan._SpinVisitor("made/up.py", paced)
    visitor.visit(ast.parse(paced))
    assert visitor.sites == []
