# SPDX-License-Identifier: MIT
"""The `kv_transfer_params` field table in `01_execution_and_time_model.md`,
kept honest against the connectors.

That table is what a reader builds a simulated producer's blob from. It was
written by reading the source once and then not re-read: it stood at twelve
fields while `moriio_connector.py` emitted thirteen, so anything written against
it would have shipped a blob without `do_remote_decode`. These tests re-derive
the shape and compare it against the table, so the next divergence fails here
instead of in whatever is built from the document.

**This guards the document, not the connector.** It makes no claim about what a
simulated connector emits -- that assertion belongs beside the connector and is
a different one. What it reads out of the document and compares with the source
is exactly this:

- each table row's field set and order, key count and cited line range;
- the one `mooncake_connector.py` range cited in the prose under the table, up
  to the next heading, and the full-transfer fallback that paragraph describes,
  run on the connector's own code;
- four count restatements in the table's section and the section before it,
  each read at a fixed anchor: "<N> fields and <N>", "exactly <once|N times>",
  "the <role> shape is the <role> shape plus <N>: <fields>." and
  "`<backend>` emits the <N>".

Everything else the prose says is outside this guard's reach, including a
sentence beside a compared count. A restatement reworded so that its anchor no
longer matches is not read, and nothing notices. Nor are "only the thirteen"
and "One of the four", which name no backend, so which count they mean is only
in the prose; nor "its twelve-field predecessor", which describes an earlier
revision rather than this tree. Which backend is push and which is pull is taken
from the table's backend column and is not checked against the source.

The derivation is an AST walk, not a regex over text and not an import: the
connectors pull in RDMA backends, and these tests run with no driver. Both the
document and the modules are read out of this checkout by path, so the tree
under test is the tree the test file lives in.
"""

from __future__ import annotations

import ast
import logging
import re
from pathlib import Path
from types import CodeType, FunctionType, SimpleNamespace

import pytest

TREE = Path(__file__).resolve().parents[2]
DOC = TREE / "atom/compass/design/01_execution_and_time_model.md"

# The attribute the scheduler reads back and the router relays
# (`scheduler.py:2719-2725`).
BLOB_ATTR = "kv_transfer_params_output"

CONNECTORS = {
    "moriio": TREE / "atom/kv_transfer/disaggregation/moriio/moriio_connector.py",
    "mooncake": TREE / "atom/kv_transfer/disaggregation/mooncake/mooncake_connector.py",
}

HEADER = "| Backend | Assignment | Keys | Field set, in source order |"

# A backtick-quoted bare identifier -- a field name. The citation columns are
# backticked too, but `moriio_connector.py:983-997` carries a dot and a colon,
# so it cannot be read as one.
_FIELD = re.compile(r"`([A-Za-z_][A-Za-z0-9_]*)`")
_CITE = re.compile(r"`([A-Za-z0-9_]+)\.py:(\d+)-(\d+)`")
_HEADING = re.compile(r"#{1,6} ")

# A count as the prose spells it: digits, or a word up to twenty.
_WORDS = (
    "zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|"
    "fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty"
)
_N = rf"\d+|{_WORDS}"


def _blob_site(path: Path) -> list[ast.Assign]:
    """Every `<something>.kv_transfer_params_output = ...` in one module.

    Deliberately narrow: plain `ast.Assign` to an `ast.Attribute`, which is the
    form both connectors use. A `setattr`, an `AnnAssign` or a walrus would be
    invisible to it, so it is narrower than the one-site claim it pins. Swept
    for those forms tree-wide on 92f1fdafe and none exists -- the only other
    writes to the name are `request.py:16` (a dataclass field default) and
    `sequence.py:260` (`= None` in `__init__`), and no dict literal anywhere
    else in the tree carries these keys. Widen the matcher, do not trust this
    comment, if that has to be re-established.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Attribute) and t.attr == BLOB_ATTR for t in node.targets
        )
    ]


def _derived_keys(assign: ast.Assign) -> list[str]:
    """The blob's keys, in source order, refusing anything not a string literal."""
    blob = assign.value
    assert isinstance(blob, ast.Dict), f"line {assign.lineno} is not a dict literal"
    literal = [
        k.value
        for k in blob.keys
        if isinstance(k, ast.Constant) and isinstance(k.value, str)
    ]
    assert len(literal) == len(blob.keys), (
        f"line {assign.lineno} has a key that is not a string literal, so the "
        "field set cannot be derived by reading it"
    )
    return literal


def _table_lines(text: str) -> tuple[list[str], int, list[int]]:
    """The document's lines, the index of the table's header, and every heading."""
    assert HEADER in text, (
        f"the blob table in {DOC.name} is gone or its header changed; "
        f"{HEADER!r} is what these tests locate it by"
    )
    lines = text.splitlines()
    top = next(i for i, ln in enumerate(lines) if HEADER in ln)
    return lines, top, [i for i, ln in enumerate(lines) if _HEADING.match(ln)]


def _split_table(text: str) -> tuple[list[str], str, int]:
    """The blob table's rows; the prose under it, up to the next heading; and
    the document line that prose starts on.

    The prose carries its own source citations, so it has to be kept separate
    from the rows: a search over both would find the table's citation first.
    It stops at the next heading because the sections after it cite the same
    module for other claims.
    """
    lines, top, headings = _table_lines(text)
    end = next(
        (i for i in range(top + 2, len(lines)) if not lines[i].startswith("|")),
        len(lines),
    )
    stop = next((i for i in headings if i > end), len(lines))
    return lines[top + 2 : end], "\n".join(lines[end:stop]), end + 1


def _restating_text(text: str) -> str:
    """The table's section and the one before it, on one line.

    The section before holds the bullet that sends a reader to the table, and it
    restates the counts too. Joined so that a count wrapped onto the next line
    still reads as one phrase.
    """
    lines, top, headings = _table_lines(text)
    start = [i for i in headings if i < top][-2]
    stop = next((i for i in headings if i > top), len(lines))
    return " ".join(" ".join(lines[start:stop]).split())


def _number(word: str) -> int:
    return int(word) if word.isdigit() else _WORDS.split("|").index(word.lower())


def _stated_totals(text: str) -> list[tuple[int, int]]:
    """Every "<N> fields and <N>" in *text*."""
    found = re.findall(rf"(?i)\b({_N}) fields and ({_N})\b", text)
    return [(_number(a), _number(b)) for a, b in found]


def _stated_sites(text: str) -> list[int]:
    """Every "exactly once", "exactly twice" or "exactly <N> times" in *text*."""
    found = re.findall(rf"(?i)\bexactly (once|twice|({_N}) times)\b", text)
    return [{"once": 1, "twice": 2}.get(w.lower()) or _number(n) for w, n in found]


def _stated_additions(text: str) -> list[tuple[str, str, int, list[str]]]:
    """Every "the <role> shape is the <role> shape plus <N>: <fields>." in *text*."""
    found = re.findall(
        rf"(?i)\bthe (\w+) shape is the (\w+) shape plus ({_N}):([^.]*)\.", text
    )
    return [
        (wide, narrow, _number(n), _FIELD.findall(f)) for wide, narrow, n, f in found
    ]


def _stated_emits(text: str) -> list[tuple[str, int]]:
    """Every "`<backend>` emits the <N>" in *text*."""
    found = re.findall(rf"(?i)`(\w+)` emits the ({_N})\b", text)
    return [(backend, _number(n)) for backend, n in found]


def _doc_rows(rows: list[str]) -> dict[str, dict]:
    """One parsed row per backend, keyed by the backend the row names."""
    parsed: dict[str, dict] = {}
    for line in rows:
        backend, cite, count, fields = (c.strip() for c in line.strip("|").split("|"))
        named = _CITE.search(cite)
        assert named, f"no `file.py:start-end` citation in the {backend} row"
        role = re.search(r"\((\w+)", backend)
        parsed[_FIELD.search(backend).group(1)] = {
            "role": role and role.group(1),
            "file": named.group(1) + ".py",
            "lines": (int(named.group(2)), int(named.group(3))),
            "stated_count": int(count),
            "fields": _FIELD.findall(fields),
        }
    return parsed


def assert_the_table_lists_what_the_backend_emits(backend: str, listed: list[str]):
    """The comparison this file exists to make, in one place.

    Both the positive test and the witness that shows it firing call this, so
    the witness watches this code rather than a copy of it.
    """
    derived = _derived_keys(_blob_site(CONNECTORS[backend])[0])
    assert not set(derived) - set(listed), (
        f"{backend} emits {sorted(set(derived) - set(listed))}, which the blob "
        f"table in {DOC.name} omits; the source wins -- add them to the table"
    )
    assert not set(listed) - set(derived), (
        f"the blob table in {DOC.name} lists {sorted(set(listed) - set(derived))} "
        f"for {backend}, which the source does not emit; remove them"
    )
    assert listed == derived, f"{backend}: listed in a different order than emitted"


@pytest.fixture(scope="module")
def stated():
    rows = _doc_rows(_split_table(DOC.read_text(encoding="utf-8"))[0])
    assert set(rows) == set(CONNECTORS), (
        f"the blob table names {sorted(rows)}; the connectors are "
        f"{sorted(CONNECTORS)}. Either a backend was added or the parser broke."
    )
    return rows


@pytest.fixture(scope="module")
def emitted():
    """Each backend's blob keys, in source order, from its one site."""
    return {b: _derived_keys(_blob_site(p)[0]) for b, p in CONNECTORS.items()}


@pytest.fixture(scope="module")
def restating():
    return _restating_text(DOC.read_text(encoding="utf-8"))


@pytest.mark.parametrize("backend", sorted(CONNECTORS))
def test_each_backend_builds_the_blob_at_exactly_one_site(backend):
    """A second site would mean the table describes one of two shapes."""
    sites = _blob_site(CONNECTORS[backend])
    assert len(sites) == 1, (
        f"{CONNECTORS[backend].name} assigns {BLOB_ATTR} at "
        f"{[s.lineno for s in sites]}; the table states one site per backend"
    )


@pytest.mark.parametrize("backend", sorted(CONNECTORS))
def test_the_table_lists_the_fields_the_backend_emits(backend, stated):
    """Set equality with the message naming which way it differs, then order,
    because the column says the fields are in source order."""
    assert_the_table_lists_what_the_backend_emits(backend, stated[backend]["fields"])


@pytest.mark.parametrize("backend", sorted(CONNECTORS))
def test_the_stated_key_count_matches_the_fields_beside_it(backend, stated):
    """The count is written twice in one row, and a reader quotes the number."""
    row = stated[backend]
    assert row["stated_count"] == len(row["fields"])


@pytest.mark.parametrize("backend", sorted(CONNECTORS))
def test_the_cited_lines_are_the_assignment_and_not_its_method(backend, stated):
    """The citation this replaces named the enclosing method's range instead."""
    row = stated[backend]
    assert row["file"] == CONNECTORS[backend].name
    site = _blob_site(CONNECTORS[backend])[0]
    assert row["lines"] == (site.lineno, site.end_lineno)


def test_the_cited_lines_hold_the_full_transfer_fallback():
    """The prose under the table claims `hash_block_size`'s absence leaves the
    transfer whole, and cites a range for it. Read the range the document
    states -- not a copy of it -- so falsifying the citation fails here and
    names the document line that needs editing. Only the prose up to the next
    heading is searched: the sections after it cite the same module for other
    claims, and a deleted citation would otherwise be replaced by one of those."""
    source = CONNECTORS["mooncake"]
    _, prose, first = _split_table(DOC.read_text(encoding="utf-8"))
    cited = [
        (at, int(start), int(end))
        for at, line in enumerate(prose.splitlines(), first)
        for name, start, end in _CITE.findall(line)
        if name + ".py" == source.name
    ]
    assert len(cited) == 1, (
        f"{DOC.name}:{first}-{first + prose.count(chr(10))}, the prose under the "
        f"blob table, cites `{source.name}:start-end` {len(cited)} times "
        f"{cited}; the full-transfer fallback is cited there exactly once"
    )
    at, start, end = cited[0]
    region = "\n".join(source.read_text(encoding="utf-8").splitlines()[start - 1 : end])
    assert "hash_block_size" in region and "num_computed_blocks" in region, (
        f"{DOC.name}:{at} cites {source.name}:{start}-{end} for the full-transfer "
        "fallback, but that range no longer reads one of the two names"
    )


def _consumer_offset(blob: dict, own: int) -> int:
    """The `num_computed_blocks` mooncake's scheduler sets on receiving *blob*.

    The module cannot be imported with no driver -- its `aiter` import asks
    `rocminfo` for the GPU arch -- so `update_state_after_alloc` is compiled out
    of the file's own AST and run on stand-ins for the scheduler and the
    sequence. It is the source on disk that runs, not a copy.
    """
    path = CONNECTORS["mooncake"]
    module = ast.parse(path.read_text(encoding="utf-8"))
    (method,) = [
        fn
        for cls in module.body
        if isinstance(cls, ast.ClassDef) and cls.name == "MooncakeConnectorScheduler"
        for fn in cls.body
        if isinstance(fn, ast.FunctionDef) and fn.name == "update_state_after_alloc"
    ]
    code = compile(ast.Module([method], []), str(path), "exec")
    (body,) = [c for c in code.co_consts if isinstance(c, CodeType)]
    run = FunctionType(body, {"logger": logging.getLogger(__name__)})
    scheduler = SimpleNamespace(
        is_producer=False,
        hash_block_size=own,
        _reqs_need_recv={},
        transfer_id_to_request_id={},
        request_id_to_transfer_id={},
    )
    seq = SimpleNamespace(
        id=0,
        kv_transfer_params={**blob, "do_remote_prefill": True},
        block_table=list(range(8)),
        has_per_req_cache=False,
        num_cached_tokens=4 * own,
    )
    run(scheduler, seq)
    return seq.kv_transfer_params["num_computed_blocks"]


def test_the_consumer_takes_the_full_transfer_unless_hash_block_size_matches(
    emitted,
):
    """The paragraph under the table says the push consumer falls back to a
    full transfer, `num_computed_blocks = 0`, whenever the producer's
    `hash_block_size` is absent or differs, so a blob carrying only moriio's
    fields never takes the incremental path. Each case is run through the
    connector's own method, with the matching size as the control that shows
    the incremental path is reachable at all."""
    own = 16
    mooncake = dict.fromkeys(emitted["mooncake"])
    assert _consumer_offset(dict.fromkeys(emitted["moriio"]), own) == 0
    assert _consumer_offset({**mooncake, "hash_block_size": 2 * own}, own) == 0
    assert _consumer_offset({**mooncake, "hash_block_size": own}, own) == 4


def test_each_reader_finds_its_restatement_in_literal_text():
    """The four readers, each run on literal text that holds its restatement,
    so that a reader broken into finding nothing fails here rather than
    passing on a document it no longer reads. This checks the readers, not
    the slice of the document they are given."""
    assert _stated_totals("Twelve fields and 17; four fields") == [(12, 17)]
    assert _stated_sites("exactly once, exactly twice, exactly 3 times") == [1, 2, 3]
    assert _stated_additions(
        "The pull shape is the push shape plus two: `a`, `b`."
    ) == [("pull", "push", 2, ["a", "b"])]
    assert _stated_emits("`moriio` emits the thirteen; `x` emits the 4") == [
        ("moriio", 13),
        ("x", 4),
    ]


def test_the_restated_totals_are_the_derived_ones(emitted, restating):
    """In "thirteen fields and seventeen", above the table, the two counts are
    the two backends' key counts."""
    derived = sorted(len(keys) for keys in emitted.values())
    for pair in _stated_totals(restating):
        assert sorted(pair) == derived, (
            f"{DOC.name} says {pair[0]} fields and {pair[1]}; the backends emit "
            f"{derived}"
        )


def test_the_restated_site_count_is_the_derived_one(restating):
    """In "exactly once", the count is the number of sites `_blob_site` finds
    in each module."""
    for stated in _stated_sites(restating):
        for backend, path in CONNECTORS.items():
            found = len(_blob_site(path))
            assert stated == found, (
                f"{DOC.name} says each connector assigns {BLOB_ATTR} {stated} "
                f"time(s); {backend} does at {found} site(s)"
            )


def test_the_restated_additions_are_the_derived_ones(stated, emitted, restating):
    """In "The push shape is the pull shape plus four: ...", the narrow shape is a
    subset of the wide one, and the count and the names are what the wide one
    emits beyond it."""
    by_role = {row["role"]: backend for backend, row in stated.items()}
    for wide, narrow, count, names in _stated_additions(restating):
        assert {wide, narrow} <= set(by_role), f"no table row is {wide}/{narrow}"
        more = emitted[by_role[wide]]
        less = emitted[by_role[narrow]]
        assert not set(less) - set(more), (
            f"{DOC.name} says the {wide} shape is the {narrow} shape plus more, "
            f"but the {narrow} shape has {sorted(set(less) - set(more))} and the "
            f"{wide} shape does not"
        )
        extra = [f for f in more if f not in less]
        assert (count, set(names)) == (len(extra), set(extra)), (
            f"{DOC.name} says the {wide} shape adds {count}: {names}; it adds "
            f"{len(extra)}: {extra}"
        )


def test_the_restated_count_a_backend_emits_is_the_derived_one(emitted, restating):
    """In "`moriio` emits the thirteen", the count is that backend's key count."""
    for backend, count in _stated_emits(restating):
        assert backend in emitted, f"{DOC.name} names no connector `{backend}`"
        assert count == len(emitted[backend]), (
            f"{DOC.name} says `{backend}` emits {count} fields; it emits "
            f"{len(emitted[backend])}"
        )


def test_the_guard_fires_on_the_defect_it_was_written_for(stated):
    """The twelve-field table this replaces, fed back through the guard itself:
    it must refuse, and must name `do_remote_decode` rather than report a
    count. Routed through the guard and matched against the omission
    message specifically, so deleting that assertion fails here too -- the
    order assertion below it refuses the same input for a different reason,
    and a looser match would be satisfied by it. The refusal is also checked
    to come from the helper's own code, so a witness given a copy of the
    comparison fails here instead of passing beside a neutered helper."""
    doctored = DOC.read_text(encoding="utf-8").replace("`do_remote_decode`, ", "", 1)
    listed = _doc_rows(_split_table(doctored)[0])["moriio"]["fields"]
    assert len(listed) == len(stated["moriio"]["fields"]) - 1
    with pytest.raises(AssertionError) as refused:
        assert_the_table_lists_what_the_backend_emits("moriio", listed)
    assert "moriio emits ['do_remote_decode']" in str(refused.value), (
        "the guard refused, but not with the omission message that names the "
        f"field the table is missing, so this witness is watching the wrong "
        f"assertion: {refused.value}"
    )
    raised_in = refused.traceback[-1].frame.code.raw
    assert raised_in is assert_the_table_lists_what_the_backend_emits.__code__, (
        f"the refusal was raised in {raised_in.co_name}, not in the helper the "
        "positive test calls, so this witness is not watching that helper"
    )
