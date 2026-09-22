# SPDX-License-Identifier: MIT
"""D6's `kv_transfer_params` table, kept honest against the connectors.

`01` D6 tells a reader what a simulated producer has to put in the blob the
router relays. That list was written by reading the source once and then not
re-read: it stood at twelve fields while `moriio_connector.py` emitted
thirteen, so anything built from the document would have shipped a blob without
`do_remote_decode`. These tests re-derive the shape and compare it against the
table, so the next divergence fails here instead of in whatever is written
against the document.

**This guards the document, not the connector.** It makes no claim about what a
simulated connector emits -- that assertion belongs beside the connector and is
a different one. What it asserts is that the prose a reader builds from still
describes the source: the field sets and their order, the key counts, the cited
line ranges, and the claim that each backend has exactly one blob site.

The derivation is an AST walk, not a regex over text and not an import: the
connectors pull in RDMA backends, and the point of the D6 tier is that it needs
no driver. Both the document and the modules are read out of this checkout by
path, so the tree under test is the tree the test file lives in.
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

# A backtick-quoted bare identifier -- a field name. The citation column is
# backticked too, but `moriio_connector.py:983-997` carries a dot and a colon,
# so it cannot be read as one.
_FIELD = re.compile(r"`([A-Za-z_][A-Za-z0-9_]*)`")
_CITE = re.compile(r"`([A-Za-z0-9_]+)\.py:(\d+)-(\d+)`")


def _blob_site(path: Path) -> list[ast.Assign]:
    """Every assignment to `<something>.kv_transfer_params_output` in a module."""
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


def _doc_rows(text: str) -> dict[str, dict]:
    """The D6 blob table, located by its header line, one row per backend."""
    rows: dict[str, dict] = {}
    body = text.split(HEADER, 1)[-1].splitlines()[2:]
    for line in body:
        if not line.startswith("|"):
            break
        backend, cite, count, fields = (c.strip() for c in line.strip("|").split("|"))
        named = _CITE.search(cite)
        assert named, f"no `file.py:start-end` citation in the {backend} row"
        rows[_FIELD.search(backend).group(1)] = {
            "file": named.group(1) + ".py",
            "lines": (int(named.group(2)), int(named.group(3))),
            "stated_count": int(count),
            "fields": _FIELD.findall(fields),
        }
    return rows


@pytest.fixture(scope="module")
def stated():
    assert HEADER in DOC.read_text(encoding="utf-8"), (
        f"D6's blob table is gone or its header changed; {HEADER!r} is what "
        "these tests locate it by"
    )
    rows = _doc_rows(DOC.read_text(encoding="utf-8"))
    assert set(rows) == set(CONNECTORS), (
        f"D6's blob table names {sorted(rows)}; the connectors are "
        f"{sorted(CONNECTORS)}. Either a backend was added or the parser broke."
    )
    return rows


@pytest.mark.parametrize("backend", sorted(CONNECTORS))
def test_each_backend_builds_the_blob_at_exactly_one_site(backend):
    """A second site would mean the table describes one of two shapes."""
    sites = _blob_site(CONNECTORS[backend])
    assert len(sites) == 1, (
        f"{CONNECTORS[backend].name} assigns {BLOB_ATTR} at "
        f"{[s.lineno for s in sites]}; D6 states one site per backend"
    )


@pytest.mark.parametrize("backend", sorted(CONNECTORS))
def test_the_table_lists_the_fields_the_backend_emits(backend, stated):
    """Set equality with the message naming which way it differs, then order,
    because the column says the fields are in source order."""
    derived = _derived_keys(_blob_site(CONNECTORS[backend])[0])
    listed = stated[backend]["fields"]
    assert not set(derived) - set(listed), (
        f"{backend} emits {sorted(set(derived) - set(listed))}, which D6's "
        "table omits; the source wins -- add them to the table"
    )
    assert not set(listed) - set(derived), (
        f"D6's table lists {sorted(set(listed) - set(derived))} for {backend}, "
        "which the source does not emit; remove them"
    )
    assert listed == derived, f"{backend}: listed in a different order"


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


def test_hash_block_size_is_load_bearing_where_the_table_says_it_is(stated):
    """The prose under the table claims `hash_block_size`'s absence forces a
    full transfer, and cites a range for it. Read that range: both names have to
    be in it, or the claim points at code that no longer makes it."""
    start, end = 389, 401
    lines = CONNECTORS["mooncake"].read_text(encoding="utf-8").splitlines()
    region = "\n".join(lines[start - 1 : end])
    assert "hash_block_size" in region and "num_computed_blocks" in region
    assert "hash_block_size" in stated["mooncake"]["fields"]
    assert "hash_block_size" not in stated["moriio"]["fields"]


def test_the_guard_reproduces_the_defect_it_was_written_for():
    """The twelve-field table this replaces, fed back through the parser: the
    comparison must fail, and must name `do_remote_decode` rather than a count."""
    doctored = DOC.read_text(encoding="utf-8").replace("`do_remote_decode`, ", "", 1)
    listed = _doc_rows(doctored)["moriio"]["fields"]
    derived = _derived_keys(_blob_site(CONNECTORS["moriio"])[0])
    assert len(listed) == 12
    assert set(derived) - set(listed) == {"do_remote_decode"}
