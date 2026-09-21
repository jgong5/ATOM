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
# that today. The dash class is every dash these documents write a range with:
# the front page writes `D0–D94`, `01` writes `D3-D5`, and an author restating
# the extent with the em dash their prose is full of has stated it just the same.
EXTENT = re.compile(r"T\d+ ?[-–—‒−] ?T\d+")

# A *count* is a number whose noun can mean nothing but the register. `rows`
# and `items` alone cannot qualify it, because the design counts rows of
# measurement tables and items of many other kinds; `TOTAL` above reads `rows`
# and is safe only because it runs over `STATED_IN`. The cost of that narrowing
# is stated with the cases it protects, in `LEGITIMATE`.
#
# The lookbehind is `OPEN`'s above, widened, and is here for the same reason: an
# id ends in digits, so without it a list of ids reads as a count -- `T83, T84
# and T85 are open` refused as `85 are open`. That refuses the list form this
# rule leaves legal, and prescribes as the remedy the list it just refused.
# `OPEN` excludes `T` alone because the two files it reads name nothing else;
# the documents read here name `D92`, `M1`, `TP4`, `W3` and `P0` as well, and
# `D4 and D6 are open` is the class the `D6 open issue` row below is pinned to
# protect. So the class is every letter: digits glued to one are an identifier.
# It is also `-`, `_` and `/`, because a digit glued to a separator is an
# identifier's tail just as surely: without them `TP2/4/8 are open` refused as
# `8 are open` and `tier-0 and tier-1 are open` as `1 are open`, while `T99 and
# T100 are open` was already legal. Both tokens are live text in six of the 18
# documents in `design/`, four of which this scan reaches, so neither refusal
# was invented vocabulary. What all three separators cost is a count with no
# space after one: `-81 are open`, `_81 are open` and `/81 are open` all pass
# now. Nothing writes a count that way -- a markdown bullet keeps its space and
# `blocks` flattens the line break to another, so `- 81 are open` still refuses
# -- but one of the three forms carries register content rather than a bare
# count. `T1-87 are open` is a range with its prefix written once, so it
# restates the register's extent, and after this widening nothing reads it:
# `EXTENT` wants `T` on both endpoints, and the narrower class had been refusing
# it only by accident, as `87 are open`. It stays latent because no document
# writes a prefix-once range -- every extent here writes `T` twice -- and
# because the same restatement spaced, `The register holds T1-87; 81 are
# open.`, still refuses.
#
# `N TODOs` and `N are open` reach wider than the register -- they refuse `The
# adapter still carries 4 TODOs` and `Of the five probes, 3 are open` -- and are
# kept that way. `TODOs` is the register's own noun and `are open` is how both
# stating sentences state the open figure, so narrowing either to exclude those
# would let the register's own phrasing through in a third document. Neither
# sentence occurs in `design/`; the remedy for a false refusal is to write the
# sentence another way, never an exemption.
COUNT = re.compile(
    r"(?<![A-Za-z\d./_-])\d+ (?:registered (?:TODOs|items)|open items|TODOs|are open)"
)

# Quoted from the documents this scan reaches. Each is a number or a run of ids
# that a looser reading of either pattern would refuse, and each is legitimate:
# a measurement table's row count, successors named one by one, and a decision
# id sitting next to the word "open".
LEGITIMATE = (
    "| median over 121 rows | 9.51% | **9.03%** | |",
    "successors T83, T84, T85",
    (
        "| T13 | Decide the simulated KV connector's completion semantic "
        "(MoRI-IO's last-status vs Mooncake's all-ranks) | doc 01 D6 open issue, "
        "surfaces here |"
    ),
)

# Not quoted: no document writes either sentence today. They are pinned because
# a list of ids is the form this rule leaves legal -- and the form its own
# refusal prescribes -- and the only thing keeping them legal is `COUNT`'s
# lookbehind, whose removal would otherwise still read 36 passed.
LISTED = (
    "Its successors T83, T84 and T85 are open.",
    "Of the validation gates, D4 and D6 are open.",
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


# The extent rule above lets prose name one id the register does not hold, so an
# allocation still on a branch can be mentioned before it lands. That threshold
# is one comparison in the filter above, and nothing asserted it: moving it to
# `> 2` or `> 3` left every test in this file passing. The two cases below pin
# it at exactly two by running that test over documents written here -- a single
# free mention and a pair -- rather than over a second copy of its rule, which
# would pin the copy and let the rule move. The stated extent below is three ids
# wide on purpose: at `> 2` a two-id extent is dropped by the same filter, and
# both cases would then die on the precondition instead of on the hatch.
FREE_MENTION = "The register runs T1–T3. {} allocated on a branch that has not landed."
LANDED = [(1, False), (2, False), (3, False)]


def test_one_unlanded_id_may_be_named_in_prose(tmp_path):
    path = tmp_path / "one_unlanded_id.md"
    path.write_text(FREE_MENTION.format("T99 is"), encoding="utf-8")
    test_every_stated_extent_names_exactly_the_rows(path, LANDED)


def test_two_unlanded_ids_refuse_and_the_refusal_names_them(tmp_path):
    path = tmp_path / "two_unlanded_ids.md"
    path.write_text(FREE_MENTION.format("T99 and T100 are"), encoding="utf-8")
    with pytest.raises(
        AssertionError,
        match=r"two_unlanded_ids\.md names \[99, 100\] in 'T99 and T100'",
    ):
        test_every_stated_extent_names_exactly_the_rows(path, LANDED)


def test_the_register_still_sanctions_naming_one_unlanded_id():
    """The sentence the two cases above hold in place. Reworded away, the rule
    would be enforcing a hatch its own document no longer offers."""
    assert "only one id at a time" in flattened(REGISTER), (
        f"{REGISTER.name} no longer sanctions naming an unlanded id one at a "
        "time, and the extent rule above still allows it"
    )


# The allowance above is bounded per *span*, and nothing bounds a document. Ten
# sentences each naming one unlanded id are ten free mentions and pass, while
# one sentence naming two refuses -- so the rule shapes a mention rather than
# rationing ids. `MANY_MENTIONS` is that document. Ten is the figure the
# register's own sentence names and is not otherwise load-bearing: any number
# above one shows the same thing. What is load-bearing is that each id sits in a
# sentence of its own -- joined by `,` or ` and ` the same ten ids read as one
# span and refuse, which is the case two above.
#
# `SPAN` joins with `-` and `–` where `EXTENT` above reads all five dashes these
# documents write, so a pair written with `—`, `‒` or `−` is two single mentions
# and passes here too -- appended to the real register, `T99–T100` refuses and
# `T99—T100` does not. Latent, not intended: widening the join class is
# behaviour, with its own reason to give.
MANY_MENTIONS = " ".join(FREE_MENTION.format(f"T{id_} is") for id_ in range(99, 109))


def test_many_unlanded_ids_pass_when_each_has_a_span_of_its_own(tmp_path):
    path = tmp_path / "ten_unlanded_ids.md"
    path.write_text(MANY_MENTIONS, encoding="utf-8")
    test_every_stated_extent_names_exactly_the_rows(path, LANDED)


def test_the_register_states_that_the_allowance_is_per_span():
    """What the case above passes is a limit a reader has to be told about, or
    they meet it by writing the document the case three above refuses."""
    assert "per span, not per document" in flattened(REGISTER), (
        f"{REGISTER.name} no longer states that the one-id allowance bounds a "
        "span and not a document, and the case above still passes ten of them"
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
    a parametrization over nothing passes. A document that leaves the glob
    without emptying it is the quieter failure: the glob is non-recursive and
    reads only `.md`, so moving one document into a subdirectory, or renaming
    it `.txt`, drops its case while the scan still looks healthy."""
    assert set(STATED_IN) < set(DESIGN_DOCS), f"{DESIGN} does not hold both"
    assert len(ELSEWHERE) == len(DESIGN_DOCS) - len(STATED_IN) > 1
    assert set(DESIGN_DOCS) == {path for path in DESIGN.rglob("*") if path.is_file()}, (
        f"{DESIGN} holds a document this scan does not read"
    )


@pytest.mark.parametrize("phrase", LEGITIMATE + LISTED)
def test_counting_something_other_than_the_register_is_not_a_figure(phrase):
    assert not EXTENT.search(phrase) and not COUNT.search(phrase)


# Quoted from the documents as tokens, not as sentences: `TP2/4/8` and `tier-0`
# are live text, and these are the sentences an author reaching for either would
# write. The first two read as counts until the separators joined `COUNT`'s
# lookbehind class and are the reason they did; the third was legal already and
# holds that half of the boundary. Narrowing the class back fails this by name.
GLUED = (
    "TP2/4/8 are open",
    "tier-0 and tier-1 are open",
    "T99 and T100 are open",
)


@pytest.mark.parametrize("phrase", GLUED)
def test_a_digit_glued_to_a_separator_is_part_of_an_identifier(phrase):
    assert not COUNT.search(phrase)


def test_a_range_is_an_extent_in_every_dash_these_documents_write():
    """The two stating documents write their range with an en dash, so that one
    dash and the plain hyphen are the only ones pinned by the documents
    themselves. The other three are pinned here, because narrowing the class
    back to `[-–]` refuses nothing that any test asserts."""
    for dash in "-–—‒−":
        assert EXTENT.search(f"T1{dash}T87"), f"{dash!r} does not read as a range"
