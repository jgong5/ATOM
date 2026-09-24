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
a different one. What it asserts is that the prose a reader builds from still
describes the source: the field sets and their order, the key counts, the cited
line ranges, and the claim that each backend builds the blob at exactly one
site.

The derivation is an AST walk, not a regex over text and not an import: the
connectors pull in RDMA backends, and these tests run with no driver. Both the
document and the modules are read out of this checkout by path, so the tree
under test is the tree the test file lives in.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

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


def _blob_site(path: Path) -> list[ast.Assign]:
    """Every `<something>.kv_transfer_params_output = ...` in one module.

    Narrow on purpose: plain `ast.Assign` to an `ast.Attribute`, which is the
    form both connectors use. Every other write of the name is what
    `_writes_not_read` returns, and the one-site test refuses those, so a
    second site written another way fails there by name instead of passing
    unseen.
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


def _writes_not_read(path: Path) -> list[str]:
    """Every write of the name `_blob_site` does not read, by line and source.

    An annotated or augmented assignment, the name inside a tuple target, the
    name as a string anywhere in the module -- and any mention of `setattr`,
    `__setattr__`, `__dict__` or `vars` at all, since a name built at run time
    cannot be read and neither connector writes attributes that way.
    """
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    read = {
        id(t) for n in ast.walk(tree) if isinstance(n, ast.Assign) for t in n.targets
    }
    return [
        f"{path.name}:{n.lineno}: {source.splitlines()[n.lineno - 1].strip()}"
        for n in ast.walk(tree)
        if (
            isinstance(n, ast.Attribute)
            and n.attr == BLOB_ATTR
            and not isinstance(n.ctx, ast.Load)
            and id(n) not in read
        )
        or (isinstance(n, ast.Constant) and n.value == BLOB_ATTR)
        or getattr(n, "id", getattr(n, "attr", None))
        in ("setattr", "__setattr__", "__dict__", "vars")
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


def _split_table(text: str) -> tuple[list[str], str]:
    """The blob table's rows, and the prose that follows it, split apart.

    The prose carries its own source citations, so it has to be kept separate
    from the rows: a search over both would find the table's citation first.
    """
    assert HEADER in text, (
        f"the blob table in {DOC.name} is gone or its header changed; "
        f"{HEADER!r} is what these tests locate it by"
    )
    body = text.split(HEADER, 1)[1].splitlines()[2:]
    end = next((i for i, ln in enumerate(body) if not ln.startswith("|")), 0)
    return body[:end], "\n".join(body[end:])


def _doc_rows(rows: list[str]) -> dict[str, dict]:
    """One parsed row per backend, keyed by the backend the row names."""
    parsed: dict[str, dict] = {}
    for line in rows:
        backend, cite, count, fields = (c.strip() for c in line.strip("|").split("|"))
        named = _CITE.search(cite)
        assert named, f"no `file.py:start-end` citation in the {backend} row"
        parsed[_FIELD.search(backend).group(1)] = {
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


@pytest.mark.parametrize("backend", sorted(CONNECTORS))
def test_each_backend_builds_the_blob_at_exactly_one_site(backend):
    """A second site would mean the table describes one of two shapes."""
    unread = _writes_not_read(CONNECTORS[backend])
    assert not unread, f"{BLOB_ATTR} is written in a form not read here: {unread}"
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
    points at the document line that needs editing."""
    source = CONNECTORS["mooncake"]
    prose = _split_table(DOC.read_text(encoding="utf-8"))[1]
    cited = [c for c in _CITE.findall(prose) if c[0] + ".py" == source.name]
    assert cited, (
        f"the prose under the blob table in {DOC.name} no longer cites "
        f"`{source.name}:start-end`, so there is no range to check it against"
    )
    start, end = int(cited[0][1]), int(cited[0][2])
    region = "\n".join(source.read_text(encoding="utf-8").splitlines()[start - 1 : end])
    assert "hash_block_size" in region and "num_computed_blocks" in region, (
        f"{source.name}:{start}-{end} is cited for the full-transfer fallback "
        "but no longer reads one of the two names"
    )


def test_the_guard_fires_on_the_defect_it_was_written_for(stated):
    """The twelve-field table this replaces, fed back through the guard itself:
    it must refuse, and must name `do_remote_decode` rather than report a
    count. Routed through the guard and matched against the omission
    message specifically, so deleting that assertion fails here too -- the
    order assertion below it refuses the same input for a different reason,
    and a looser match would be satisfied by it."""
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
