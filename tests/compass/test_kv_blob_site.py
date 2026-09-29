# SPDX-License-Identifier: MIT
"""Each disaggregation connector builds its `kv_transfer_params` blob at
exactly one site.

The derivation is an AST walk, not an import: the connectors pull in RDMA
backends, and these tests run with no driver. The modules are read out of this
checkout by path, so the tree under test is the tree the test file lives in.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

TREE = Path(__file__).resolve().parents[2]

# The attribute the scheduler reads back and the router relays.
BLOB_ATTR = "kv_transfer_params_output"

CONNECTORS = {
    "moriio": TREE / "atom/kv_transfer/disaggregation/moriio/moriio_connector.py",
    "mooncake": TREE / "atom/kv_transfer/disaggregation/mooncake/mooncake_connector.py",
}


def _blob_site(path: Path) -> list[ast.Assign]:
    """Every `<something>.kv_transfer_params_output = ...` in one module.

    Deliberately narrow: plain `ast.Assign` to an `ast.Attribute`, which is the
    form both connectors use. A `setattr`, an `AnnAssign` or a walrus would be
    invisible to it, so it is narrower than the one-site claim it pins.
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


@pytest.mark.parametrize("backend", sorted(CONNECTORS))
def test_each_backend_builds_the_blob_at_exactly_one_site(backend):
    """A second site would mean the backend emits one of two shapes."""
    sites = _blob_site(CONNECTORS[backend])
    assert len(sites) == 1, (
        f"{CONNECTORS[backend].name} assigns {BLOB_ATTR} at "
        f"{[s.lineno for s in sites]}; one site per backend is expected"
    )
