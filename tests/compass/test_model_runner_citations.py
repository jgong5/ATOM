# SPDX-License-Identifier: MIT
"""Every `model_runner.py` citation in the design documents names a symbol.

A citation is written `model_runner.py::Class.method`, and the symbol must be
defined in `atom/model_engine/model_runner.py` at this checkout. A line number
cited into that file, in any spelling `_TOKEN` recognises, must fall inside the
`ast` span (`def` to `end_lineno`) of the nearest symbol cited before it in the
same paragraph, bullet or table row; with no symbol there, it fails.

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
_FILE = r"[\w-]+\.(?:py|md)\b"
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


def check(docs: dict[str, str], source: str) -> list[str]:
    """One failure string per citation that names no symbol or leaves its span."""
    spans, failures = _spans(source), []
    names = {name.rsplit(".", 1)[-1] for name in spans}
    for doc, text in docs.items():
        owner = None
        for segment in re.split(r"\n\s*\n|\n(?=\||\s*[-*] )", text):
            symbol = None
            for m in _TOKEN.finditer(segment):
                if m["file"]:
                    owner = m["file"]
                    continue
                if not m["bare"]:
                    owner = "model_runner.py"
                elif owner != "model_runner.py" and (
                    (m["beside"] or "").rsplit(".", 1)[-1] not in names
                ):
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
    key = "ModelRunner.prepare_inputs"
    assert _spans(shifted)[key][0] == _spans(SOURCE)[key][0] + 1
    assert check(DOCS, shifted) == []


def test_a_line_moved_into_another_function_fails_by_name():
    census = (
        DOCS[CENSUS_DOC]
        .replace("prepare_sample` | 2564 |", "prepare_sample` | 2468 |")
        .replace("| 2468, 2479, 2481 |", "| 2468, 2479, 2564 |")
    )
    spans = _spans(SOURCE)
    assert check({**DOCS, CENSUS_DOC: census}, SOURCE) == [
        f"{CENSUS_DOC}: 2564 is outside ModelRunner.prepare_inputs (%d-%d)"
        % spans["ModelRunner.prepare_inputs"],
        f"{CENSUS_DOC}: 2468 is outside ModelRunner.prepare_sample (%d-%d)"
        % spans["ModelRunner.prepare_sample"],
    ]


NO_SYMBOL = {
    "colon": ("`model_runner.py:1234`", "'model_runner.py:1234'"),
    "path-outside-backticks": (
        "`atom/model_engine/model_runner.py`:1234",
        "'model_runner.py`:1234'",
    ),
    "line-word": ("`model_runner.py` line 1234", "'model_runner.py` line 1234'"),
    "bare-backtick": ("`model_runner.py` (`:1234`)", "'`:1234'"),
    "bare-paren": ("`model_runner.py` (:1234)", "'(:1234'"),
    "bare-comment": ("`model_runner.py` # :1234", "'# :1234'"),
    "bare-beside-name": ("`warmup_model` (`:1234`)", "'`warmup_model` (`:1234'"),
    "bare-beside-call": ("`warmup_model()` (`:1234`)", "'`warmup_model()` (`:1234'"),
    "later-row": (
        "\n| `model_runner.py::ModelRunner.forward` |\n| `model_runner.py:3300` |",
        "'model_runner.py:3300'",
    ),
    "later-bullet": (
        "\n- `model_runner.py::ModelRunner.forward`\n- `model_runner.py:3300`",
        "'model_runner.py:3300'",
    ),
}


@pytest.mark.parametrize("citation, token", NO_SYMBOL.values(), ids=NO_SYMBOL)
def test_a_line_with_no_symbol_fails_by_name(citation, token):
    docs = {**DOCS, CENSUS_DOC: DOCS[CENSUS_DOC] + f"\n\nSee {citation}.\n"}
    assert check(docs, SOURCE) == [f"{CENSUS_DOC}: {token} names no symbol"]


@pytest.mark.parametrize("owner", ["engine_core.py", "README.md"])
def test_a_bare_line_owned_by_another_file_is_not_checked(owner):
    tail = f"\n\n`model_runner.py::ModelRunner.forward` and `{owner}` (`:1234`).\n"
    assert check({**DOCS, CENSUS_DOC: DOCS[CENSUS_DOC] + tail}, SOURCE) == []


def test_a_renamed_function_fails_by_name():
    renamed = SOURCE.replace("def prepare_sample(", "def prepare_sample2(", 1)
    failure = f"{CENSUS_DOC}: ModelRunner.prepare_sample is not defined"
    assert check(DOCS, renamed) == [failure]
