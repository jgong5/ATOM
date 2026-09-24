# SPDX-License-Identifier: MIT
"""The classified list of blocking calls on the serving path, kept honest.

A simulated run substitutes predicted durations for real work, so it has to
know every call that parks a thread on the real clock. The list of them is in
`atom/compass/audit/sync_sites.json`; deciding what each one is was a reading
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

The second message is only safe to act on if a row cannot silently change which
call it describes, which is what `test_inserting_a_call_above_another_leaves_
its_neighbour_alone` pins: an earlier identity keyed on the callee alone, so
inserting one `call_func("flush_pp_send", ...)` above a `call_func("forward",
...)` moved every later row onto the wrong site and reported only a moved line.

No driver, and no import of ATOM's serving modules: the scanner parses them.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from atom.compass.audit import sync_scan

TREE = sync_scan.repo_root_from_here()
INVENTORY = sync_scan.load_inventory()
CATEGORIES = set(INVENTORY["categories"])

# Every file that states the count per category. Each is parsed and compared
# against the rows, so a table cannot drift from the data it describes.
COUNT_TABLES = (
    Path(sync_scan.__file__).with_name("README.md"),
    TREE / "atom/compass/design/01_execution_and_time_model.md",
)


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


def test_calls_that_share_an_ordinal_are_answered_the_same_way(listed):
    """Two sites can share an ordinal only by being the same call, written the
    same way, in one function. The ordinal between them is positional, so the
    inventory must not use it to say two different things."""
    by_text: dict[tuple, set] = {}
    for row in listed.values():
        key = (row["file"], row["symbol"], row["expr"], row["shape"])
        by_text.setdefault(key, set()).add((row["category"], row["why"]))
    split = {k: v for k, v in by_text.items() if len(v) > 1}
    assert not split, (
        "identical calls in one function are classified differently, so their "
        "ordinals carry meaning they cannot keep across an edit: "
        + "; ".join(f"{k[0]}::{k[1]}::{k[2]}" for k in split)
    )


def test_scanned_and_unscanned_roots_all_exist():
    for rel, _why in sync_scan.SCANNED_ROOTS + sync_scan.UNSCANNED_ROOTS:
        assert (TREE / rel).exists(), f"{rel} is named in the scanner but is gone"


def test_no_scanned_file_is_also_declared_unscanned():
    """The two lists are at one granularity, so a file cannot be in both."""
    scanned_files = set(sync_scan.iter_scanned_files(TREE))
    for rel, _why in sync_scan.UNSCANNED_ROOTS:
        inside = {f for f in scanned_files if f == rel or f.startswith(rel)}
        assert not inside, f"{rel} is excluded but these are scanned: {sorted(inside)}"


def _stated_counts(text: str) -> dict[str, int]:
    """Read every markdown table that has a column headed `Count`."""
    counts: dict[str, int] = {}
    column = None
    for line in text.splitlines():
        if not line.startswith("|"):
            column = None
            continue
        cells = [c.strip().strip("*`") for c in line.strip().strip("|").split("|")]
        if "Count" in cells:
            column = cells.index("Count")
            continue
        if column is None or column >= len(cells):
            continue
        if cells[0] in CATEGORIES and cells[column].isdigit():
            counts[cells[0]] = int(cells[column])
    return counts


@pytest.mark.parametrize("path", COUNT_TABLES, ids=lambda p: p.name)
def test_every_stated_count_matches_the_rows(path):
    """The count per category is written down twice and derived once here."""
    stated = _stated_counts(path.read_text(encoding="utf-8"))
    assert stated, f"{path} states no counts; the parser or the table changed"
    assert stated == sync_scan.category_counts(INVENTORY), path


# --- the shape rules, against the forms they are written to tell apart ------


def _first_call(src: str) -> ast.Call:
    return next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Call))


def _scan_source(src: str):
    visitor = sync_scan._CallVisitor("made/up.py", src)
    visitor.visit(ast.parse(src))
    return sync_scan._number(visitor.sites)


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
    sites = _scan_source(src)
    assert [(s.symbol, s.call, s.shape) for s in sites] == [
        ("handler", "self._inbox.get", "queue_get")
    ]


def test_inserting_a_call_above_another_leaves_its_neighbour_alone():
    """The defect this identity scheme exists to prevent.

    Keyed on the callee alone, the two calls below are indistinguishable and
    the ordinal that separated them is positional -- so inserting the flush
    renamed the forward's row onto the flush and said only that a line moved.
    """
    forward = '    self.mgr.call_func("forward", batch, wait_out=True)\n'
    flush = '    self.mgr.call_func("flush_pp_send", wait_out=True)\n'
    before = {s.id for s in _scan_source("def step(self):\n" + forward)}
    after = {s.id for s in _scan_source("def step(self):\n" + flush + forward)}
    assert before < after, "the surviving call's identity changed under an insert"
    assert len(after - before) == 1


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
