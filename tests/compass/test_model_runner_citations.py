# SPDX-License-Identifier: MIT
"""Every `model_runner.py` citation in the design documents names a symbol.

A citation is written `model_runner.py::Class.method`, and the symbol must be
defined in `atom/model_engine/model_runner.py` at this checkout. Any line number
cited into that file must fall inside the `ast` span (`def` to `end_lineno`) of
the nearest symbol cited before it in the same paragraph or table row. A line
number with no symbol before it fails. The line-number spellings recognised are
`model_runner.py:N`, `` `.../model_runner.py`:N ``, `model_runner.py` line N,
a bare `` `:N` ``, `(:N` or `# :N` whose nearest preceding file name is
`model_runner.py` or which follows a backticked name defined there, and a table
cell of line numbers right after a symbol's cell.

Symbols, not line numbers, keep a citation valid across edits to the file: a
line inserted above a cited function leaves a symbol-only citation green.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

TREE = Path(__file__).resolve().parents[2]
SOURCE = (TREE / "atom/model_engine/model_runner.py").read_text(encoding="utf-8")
DOCS = {
    p.name: p.read_text(encoding="utf-8")
    for p in sorted((TREE / "atom/compass/design").glob("*.md"))
}
CENSUS_DOC = "04_model_capture_and_cost_ir.md"

_NUMBERS = r"\d+(?:-\d+)?(?:, ?\d+(?:-\d+)?)*"
_FILE = r"[\w-]+\.(?:py|md|json|log|sh|yaml|txt)\b"
_TOKEN = re.compile(
    rf"model_runner\.py::(?P<symbol>[\w.]+)(?:` \| (?P<cells>{_NUMBERS}) \|)?"
    rf"|model_runner\.py`?(?::|,? lines? )(?P<direct>{_NUMBERS})"
    rf"|(?P<file>{_FILE})"
    rf"|(?:`(?P<beside>(?!{_FILE})[\w.]+)(?:\(\))?` \()?(?:`|\(|# ):(?P<bare>{_NUMBERS})"
)


def _spans(source: str) -> dict[str, tuple[int, int]]:
    spans = {}

    def walk(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                spans[prefix + child.name] = (child.lineno, child.end_lineno)
                walk(child, prefix + child.name + ".")

    walk(ast.parse(source), "")
    return spans


def _last(name: str | None) -> str | None:
    return name and name.rsplit(".", 1)[-1]


def check(docs: dict[str, str], source: str) -> list[str]:
    """One failure string per citation that names no symbol or leaves its span."""
    spans, failures = _spans(source), []
    names = {_last(name) for name in spans}
    for doc, text in docs.items():
        owner = None
        for segment in re.split(r"\n\s*\n|\n(?=\|)", text):
            symbol = None
            for m in _TOKEN.finditer(segment):
                if m["file"]:
                    owner = m["file"]
                    continue
                if not m["bare"]:
                    owner = "model_runner.py"
                elif owner != "model_runner.py" and _last(m["beside"]) not in names:
                    continue
                if m["symbol"]:
                    symbol = m["symbol"]
                    if symbol not in spans:
                        failures.append(f"{doc}: {symbol} is not defined")
                        continue
                cited = m["cells"] or m["direct"] or m["bare"]
                if not cited:
                    continue
                if symbol is None:
                    failures.append(f"{doc}: {m[0]!r} names no symbol")
                    continue
                lo, hi = spans.get(symbol, (0, 0))
                for part in re.split(r", ?", cited):
                    if not all(lo <= int(n) <= hi for n in part.split("-")):
                        failures.append(
                            f"{doc}: {part} is outside {symbol} ({lo}-{hi})"
                        )
    return failures


def test_every_citation_names_a_symbol_and_stays_in_its_span():
    assert check(DOCS, SOURCE) == []


def test_a_blank_line_above_a_cited_function_stays_green():
    shifted = SOURCE.replace("    def prepare_inputs(", "\n    def prepare_inputs(", 1)
    before, after = _spans(SOURCE), _spans(shifted)
    assert (
        after["ModelRunner.prepare_inputs"][0]
        == before["ModelRunner.prepare_inputs"][0] + 1
    )
    assert check(DOCS, shifted) == []


def test_a_line_moved_into_another_function_fails_by_name():
    docs = dict(DOCS)
    docs[CENSUS_DOC] = docs[CENSUS_DOC].replace(
        "prepare_sample` | 2564 |", "prepare_sample` | 2468 |"
    )
    lo, hi = _spans(SOURCE)["ModelRunner.prepare_sample"]
    assert check(docs, SOURCE) == [
        f"{CENSUS_DOC}: 2468 is outside ModelRunner.prepare_sample ({lo}-{hi})"
    ]


@pytest.mark.parametrize(
    "citation, token",
    [
        ("`model_runner.py:1234`", "'model_runner.py:1234'"),
        ("`atom/model_engine/model_runner.py`:1234", "'model_runner.py`:1234'"),
        ("`model_runner.py` line 1234", "'model_runner.py` line 1234'"),
        ("`model_runner.py` (`:1234`)", "'`:1234'"),
        ("`warmup_model` (`:1234`)", "'`warmup_model` (`:1234'"),
    ],
    ids=[
        "colon",
        "path-outside-backticks",
        "line-word",
        "bare-colon",
        "bare-beside-name",
    ],
)
def test_a_line_with_no_symbol_fails_by_name(citation, token):
    docs = {**DOCS, CENSUS_DOC: DOCS[CENSUS_DOC] + f"\n\nSee {citation}.\n"}
    assert check(docs, SOURCE) == [f"{CENSUS_DOC}: {token} names no symbol"]


def test_a_renamed_function_fails_by_name():
    renamed = SOURCE.replace("def prepare_sample(", "def prepare_sample_v2(", 1)
    assert check(DOCS, renamed) == [
        f"{CENSUS_DOC}: ModelRunner.prepare_sample is not defined"
    ]
