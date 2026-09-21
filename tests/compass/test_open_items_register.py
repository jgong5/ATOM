# SPDX-License-Identifier: MIT
r"""The open-items register's own bookkeeping, counted rather than read.

The register is a table in `atom/compass/design/12_open_items.md`, and its
extent is restated three times -- the register file's own introduction, the
design front page's headline, and that page's index row. Nothing guarded any of
them, and in one day the same register was reported as 70, 73, 74, 75, 76, 80,
81 and 88 items by different readers of the same file.

The rule encoded here is the register's own: a row is a table row *in section 3
only* whose first cell is a T-number, tolerating the bold and strike-through
that an amended or a done row carries -- `^\| *~*\**T[0-9]+` over that section
and nothing else. Three other ways of counting, each of which produced one of
the wrong figures above, are pinned as cases below so the scoping cannot be
quietly relaxed:

* the same pattern over the whole file over-reads, because section 1's
  load-bearing-assumptions table restates five register ids as its own rows;
* a token grep over prose over-reads further, because prose here names ids
  allocated on branches that have not landed;
* a stated range reads contiguous when the ids are not, so one range can
  simultaneously claim rows that do not exist and drop a row that does.

What makes the rule trustworthy is that it reproduces an earlier generation of
the file from that generation's own sentence: before the register's most recent
row landed the file stated 86 rows, `T1-T80 and T82-T87` and 80 open, and the
parsers here return exactly those three figures from it.

Only the two files in `STATED_IN` must state these figures, and -- the other
half of the same rule -- no other design document may state them at all.
`16_execution_plan.md` is the reason: it restated a register count and went
stale twice in one day, and was then corrected to state none. Nothing here
re-requires it, but every design document including it is now read for a figure
it must not carry, so a fourth copy fails on arrival rather than years later.
What counts as such a figure is narrowed deliberately, in the comments on
`EXTENT` and `COUNT` below: a guard that refused a measurement table's row count
would be answered with an exemption, and an exemption list is the next hole.

And prose decomposition *is* in scope
here, unlike the synchronization inventory's guard which reads a `Count` column
and so cannot see a decomposition that disagrees with it: the open figure is
not derivable from the rows alone, because one row is closed without being
struck through, so the sentence naming which ids are done is the arithmetic and
is parsed and checked like any other figure.

No driver and no imports: the documents are parsed as text.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

DESIGN = Path(__file__).resolve().parents[2] / "atom" / "compass" / "design"
REGISTER = DESIGN / "12_open_items.md"
STATED_IN = (REGISTER, DESIGN / "README.md")

HEADING = re.compile(r"^## (\d+)\.")
ROW = re.compile(r"^\| *(~*)\**T(\d+)")
SPAN = re.compile(r"T\d+(?:[-–]T\d+)?(?:(?:,| and ) ?T\d+(?:[-–]T\d+)?)*")
TOTAL = re.compile(r"(\d+) (?:registered TODOs|rows)")
OPEN = re.compile(r"(?<![T\d.])(\d+) are open")
STRUCK = re.compile(r"((?:T\d+(?:, | and ))*T\d+) are struck through as done")
CLOSED = re.compile(r"T(\d+) was opened and closed")

DESIGN_DOCS = sorted(DESIGN.glob("*.md"))
ELSEWHERE = tuple(path for path in DESIGN_DOCS if path not in STATED_IN)

# An *extent* is a T-number range. It asserts the register holds every id
# between its endpoints, so it states how far the register runs and goes stale
# the moment the register grows. A list of ids is not an extent: it names the
# ids it names and claims nothing between them, which is how a topic document
# points at its own successors, and `15_parallelism_support.md` does exactly
# that today.
EXTENT = re.compile(r"T\d+ ?[-–] ?T\d+")

# A *count* is a number whose noun can mean nothing but the register. `rows`
# and `items` alone cannot qualify it, because the design counts rows of
# measurement tables and items of many other kinds; `TOTAL` above reads `rows`
# and is safe only because it runs over `STATED_IN`. The cost of that narrowing
# is stated with the cases it protects, in `LEGITIMATE`.
COUNT = re.compile(r"\d+ (?:registered (?:TODOs|items)|open items|TODOs|are open)")

# Quoted from the documents this scan reaches. Each is a number or a run of ids
# that a looser reading of either pattern would refuse, and each is legitimate:
# a measurement table's row count, successors named one by one, and a decision
# id sitting next to the word "open".
LEGITIMATE = (
    "| median over 121 rows | 9.51% | **9.03%** | |",
    "successors T83, T84, T85",
    "| T13 | Decide the connector's completion semantic | D6 open issue |",
)

# The register's introduction as it stood one generation back, quoted verbatim.
PREVIOUS = (
    "3. **TODO register** -- 86 rows, **T1–T80 and T82–T87**, per topic, "
    "of which **80 are open**: T10, T15, T22, T48 and T65 are struck through as "
    "done, and T77 was opened and closed by P0.1."
)

SAMPLE = """\
## 1. Load-bearing assumptions
| **T21** | an assumption that restates a register row |
Prose naming T99, allocated on a branch that has not landed.
## 3. TODO register
| T1 | an open row |
| ~~T2~~ | a row struck through as done |
| **T3** | a row in bold |
## 4. Cross-cutting issues
| T4 | a row that is not in the register |
"""


def register_rows(text: str, section: int | None = 3) -> list[tuple[int, bool]]:
    """Every T-numbered table row in one section, as `(id, struck through)`."""
    rows, at = [], None
    for line in text.splitlines():
        heading = HEADING.match(line)
        if heading:
            at = int(heading.group(1))
        row = ROW.match(line)
        if row and section in (None, at):
            rows.append((int(row.group(2)), len(row.group(1)) >= 2))
    return rows


def ids(span: str) -> set[int]:
    """The ids a range expression names, expanding each `Ta-Tb` it contains."""
    found: set[int] = set()
    for low, high in re.findall(r"T(\d+)(?:[-–]T(\d+))?", span):
        found |= set(range(int(low), int(high or low) + 1))
    return found


def flattened(path: Path) -> str:
    """One line, so a figure that wraps mid-sentence still reads as one."""
    return " ".join(path.read_text(encoding="utf-8").split())


def blocks(path: Path) -> list[tuple[int, str]]:
    """Each run of non-blank lines, flattened, with the line it opens on.

    Flattened per block rather than per file for the same reason `flattened` is,
    but a refusal about a document that should carry no figure has to say where
    the figure is, and a whole-file join has nowhere to point.
    """
    text = path.read_text(encoding="utf-8")
    return [
        (text[: found.start()].count("\n") + 1, " ".join(found.group().split()))
        for found in re.finditer(r"[^\n]+(?:\n[^\n]+)*", text)
    ]


@pytest.fixture(scope="module")
def rows() -> list[tuple[int, bool]]:
    return register_rows(REGISTER.read_text(encoding="utf-8"))


@pytest.mark.parametrize("path", STATED_IN, ids=lambda p: p.name)
def test_every_stated_extent_names_exactly_the_rows(path, rows):
    """A range that starts at T1 claims the whole register, so it must name
    every row and no id that has none. Any other span of two or more ids is a
    partial mention, and may only name ids that exist."""
    present = {number for number, _ in rows}
    spans = [span for span in SPAN.findall(flattened(path)) if len(ids(span)) > 1]
    assert spans, f"{path.name} states no register extent at all"
    for span in spans:
        named = ids(span)
        if min(named) == 1:
            assert named == present, (
                f"{path.name} states the register as {span!r}: "
                f"{sorted(named - present)} stated with no row, "
                f"{sorted(present - named)} in rows with no figure covering them"
            )
        else:
            # The `len(ids(span)) > 1` filter above is deliberate, and this is
            # the branch it decides. A span resolving to one id is dropped, so
            # prose keeps one free mention: the intro names an id allocated on
            # a branch that has not landed, and that has to stay writable. Two
            # or more read as a claim about a *run* of the register, which is
            # the form that went stale eight times, so a run must be rows. An
            # unlanded allocation is therefore nameable here one id at a time
            # and not as a range or a list.
            assert named <= present, (
                f"{path.name} names {sorted(named - present)} in {span!r} "
                "and no register row carries them"
            )


@pytest.mark.parametrize("path", STATED_IN, ids=lambda p: p.name)
def test_every_stated_count_matches_the_rows(path, rows):
    """Total, open, and the prose that decomposes one into the other."""
    text = flattened(path)
    struck = {number for number, is_struck in rows if is_struck}
    named_struck = ids(STRUCK.search(text).group(1))
    done = struck | {int(number) for number in CLOSED.findall(text)}
    stated_total = {int(number) for number in TOTAL.findall(text)}
    stated_open = {int(number) for number in OPEN.findall(text)}
    assert named_struck == struck, (
        f"{path.name} names {sorted(named_struck)} as struck through; the rows "
        f"struck through are {sorted(struck)}"
    )
    assert done > struck, (
        f"{path.name} names no row closed without a strike-through; the open "
        "figure is rows less struck less those, so the sentence carrying that "
        "last term has been reworded away and the arithmetic is now short"
    )
    assert stated_total and stated_open, f"{path.name} states no register count"
    assert stated_total == {len(rows)}, (
        f"{path.name} states {sorted(stated_total)} registered item(s); "
        f"section 3 holds {len(rows)} rows"
    )
    assert stated_open == {len(rows) - len(done)}, (
        f"{path.name} states {sorted(stated_open)} open; {len(rows)} rows less "
        f"the {len(done)} done ({sorted(done)}) is {len(rows) - len(done)}"
    )


def test_no_id_is_allocated_twice(rows):
    """Parallel branches have allocated one number two and three ways, so a
    merge can keep both rows and be wrong on a file that reads right."""
    numbers = [number for number, _ in rows]
    twice = sorted({number for number in numbers if numbers.count(number) > 1})
    assert not twice, f"more than one register row carries {twice}"


def test_some_rows_are_struck_through_and_not_every_row_is(rows):
    """The arithmetic over these rows is in `test_every_stated_count_matches_
    the_rows`; this pins only that the strike-through match still matches."""
    struck = {number for number, is_struck in rows if is_struck}
    assert struck, "no row is struck through; the strike-through match broke"
    assert struck < {number for number, _ in rows}


def test_the_assumptions_table_is_not_part_of_the_register():
    assert register_rows(SAMPLE) == [(1, False), (2, True), (3, False)]
    assert len(register_rows(SAMPLE, section=None)) == 5


def test_prose_naming_an_item_is_not_a_row():
    assert 99 not in {number for number, _ in register_rows(SAMPLE, section=None)}


def test_counting_the_whole_file_over_reads_by_restating_rows(rows):
    whole = register_rows(REGISTER.read_text(encoding="utf-8"), section=None)
    outside = {number for number, _ in whole} - {number for number, _ in rows}
    counts = (
        f"the whole file counts {len(whole)} T-rows against section 3's "
        f"{len(rows)} register rows"
    )
    assert len(whole) > len(rows), (
        f"{counts}; section 1 has stopped restating register ids, so the "
        "whole-file over-read this pins no longer happens"
    )
    assert not outside, (
        f"{counts}, and {sorted(outside)} appear in a T-row outside the "
        "register, so the over-read is extra items and not restated ones"
    )


@pytest.mark.parametrize(
    "span,present,agrees",
    [
        ("T1–T4", {1, 2, 3, 4}, True),
        ("T1–T4", {1, 2, 4}, False),
        ("T1–T2 and T4", {1, 2, 4}, True),
        ("T1–T80 and T82–T87", set(range(1, 88)), False),
    ],
)
def test_a_range_is_not_contiguous_unless_the_ids_are(span, present, agrees):
    assert (ids(span) == present) is agrees


def test_the_rule_reproduces_the_previous_generation():
    """The same parsers, over the register's own sentence one generation back,
    return that generation's own three figures."""
    named = ids(SPAN.findall(PREVIOUS)[0])
    done = ids(STRUCK.search(PREVIOUS).group(1))
    done |= {int(number) for number in CLOSED.findall(PREVIOUS)}
    assert int(TOTAL.search(PREVIOUS).group(1)) == len(named) == 86
    assert int(OPEN.search(PREVIOUS).group(1)) == len(named - done) == 80


@pytest.mark.parametrize("path", ELSEWHERE, ids=lambda p: p.name)
def test_no_other_design_document_states_a_register_figure(path):
    """The extent and the counts are stated in the two files named above and
    nowhere else. A third site is not caught by checking the sites already
    known, which is how one document restated a count and went stale twice in
    one day before anything noticed."""
    owners = " and ".join(owner.name for owner in STATED_IN)
    for number, text in blocks(path):
        for pattern, figure in ((EXTENT, "extent"), (COUNT, "count")):
            found = pattern.search(text)
            assert not found, (
                f"{path.name}:{number} states a register {figure}, "
                f"{found.group()!r}, in: "
                f"{text[max(0, found.start() - 70) : found.end() + 70]!r}. "
                f"Only {owners} may state the register's extent or its counts; "
                "a third copy goes stale unread. Link to the register, or name "
                "the ids as a list rather than as a range."
            )


@pytest.mark.parametrize("path", STATED_IN, ids=lambda p: p.name)
def test_the_two_patterns_still_read_the_documents_that_do_state_it(path):
    """Neither pattern may rot into matching nothing, which would pass every
    other document by default."""
    text = flattened(path)
    assert EXTENT.search(text), f"{path.name} states no register extent"
    assert COUNT.search(text), f"{path.name} states no register count"


def test_the_scan_reaches_every_design_document_but_those_two():
    """A renamed directory or a renamed stating file would empty the scan, and
    a parametrization over nothing passes."""
    assert set(STATED_IN) < set(DESIGN_DOCS), f"{DESIGN} does not hold both"
    assert len(ELSEWHERE) == len(DESIGN_DOCS) - len(STATED_IN) > 1


@pytest.mark.parametrize("phrase", LEGITIMATE)
def test_counting_something_other_than_the_register_is_not_a_figure(phrase):
    assert not EXTENT.search(phrase) and not COUNT.search(phrase)
