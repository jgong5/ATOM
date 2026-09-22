# SPDX-License-Identifier: MIT
"""The tier-0 memory goals against the acceptance bounds they cite.

`10_analytic_laws.md` declares a tier-0 accuracy goal per quantity, and its
memory rows point at the project's acceptance table for their number. That table
has two memory buckets: **<=5%** for KV capacity / block count and **<=10%** for
each non-KV memory term, of which weights is one -- `03_memory_and_kv_model.md`
gates the non-KV terms together and lists weights first. So a tier-0 goal of
<=5% on weights is *tighter* than the acceptance bound, not the same as it.

An earlier wording denied that, grouping weights, KV capacity and block count
into one cell and calling the number "same as empirical". It was read the wrong
way in review once already: a measured weight-ratio error was checked against
<=5%, contradicting the acceptance table the same review had quoted correctly.

This is a drift guard, not a transcription. Nothing here states what either
bound is; both are read out of the documents and compared. It fires when

* the tier-0 goal and the acceptance bound differ and the row claims parity
  anyway, or they agree and the row denies it;
* either bound moves without the other -- including the acceptance bound moving
  in one of its two copies (`README.md`, `00_initial_prompt.md`) but not the
  other, and either citing row quoting a bound the acceptance table no longer
  states;
* the tier-0 KV goal stops matching the acceptance KV bound, which is the half
  of the old cell where the parity claim was true and is worth keeping.

It stays silent when every copy of a bound moves together, which is what a
deliberate re-gating looks like. That direction is pinned too, so the guard
cannot be satisfied by a checker that refuses every change; the firing direction
is pinned by re-introducing each drift into the real document text rather than
into a fixture that transcribes it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

DESIGN = Path(__file__).resolve().parents[2] / "atom" / "compass" / "design"
ACCEPTANCE = ("README.md", "00_initial_prompt.md")
GOALS = "10_analytic_laws.md"
TERMS = "03_memory_and_kv_model.md"
TERMS_HEADING = "The non-KV memory terms"
WEIGHTS_GOAL = "memoryweights"
KV_GOAL = "memorykvcapacityblockcount"
EXPECTED_ROW = "analyticmemoryweightskv"
WEIGHTS_BOUND = "eachnonkvmemoryterm"
KV_BOUND = "kvcapacityblockcount"
PARITY = "same as empirical"


def _key(line):
    """A table row's first cell, reduced so two documents' spellings of it agree."""
    return re.sub(r"[^a-z]", "", line.strip().strip("|").split("|")[0].lower())


def _percents(text):
    """Every percentage the text states, in the order written."""
    return [float(found) for found in re.findall(r"(\d+(?:\.\d+)?)\s*%", text)]


def _cells(text, key):
    """The cells of the one markdown row of `text` whose first cell reduces to `key`."""
    hits = [
        ln
        for ln in text.splitlines()
        if ln.lstrip().startswith("|") and _key(ln) == key
    ]
    assert len(hits) == 1, (
        f"{key!r} matched {len(hits)} rows of this document, wanted one"
    )
    return [cell.strip() for cell in hits[0].strip().strip("|").split("|")]


def _drift(text, key, old, new):
    """`text` with `old` -> `new` inside the row keyed by `key`: one drift, applied."""
    lines = text.splitlines()
    drifted = [line.replace(old, new) if _key(line) == key else line for line in lines]
    assert drifted != lines, f"{old!r} is not in row {key!r}, so no drift was applied"
    return "\n".join(drifted)


def read_bounds(acceptance, goals):
    """The bounds and the claims about them, all read out of the document text."""
    weights = _cells(goals, WEIGHTS_GOAL)
    return {
        "accepted_weights": [
            _percents(_cells(t, WEIGHTS_BOUND)[1]) for t in acceptance
        ],
        "accepted_kv": [_percents(_cells(t, KV_BOUND)[1]) for t in acceptance],
        "tier0_weights": _percents(weights[1]),
        "tier0_kv": _percents(_cells(goals, KV_GOAL)[1]),
        "weights_row": " ".join(weights[1:]),
        "expected_row": _cells(goals, EXPECTED_ROW)[2],
    }


def audit(bounds):
    """Every way the tier-0 weights goal and the accepted weights bound can disagree."""
    accepted, tier0 = bounds["accepted_weights"][0][0], bounds["tier0_weights"][0]
    complaints = []
    if len({tuple(copy) for copy in bounds["accepted_weights"]}) != 1:
        complaints.append(
            f"the acceptance table's copies disagree: {bounds['accepted_weights']}"
        )
    if bounds["tier0_kv"][0] != bounds["accepted_kv"][0][0]:
        complaints.append(
            f"the tier-0 KV goal {bounds['tier0_kv'][0]}% is no longer the accepted"
            f" {bounds['accepted_kv'][0][0]}%, yet its row still says {PARITY!r}"
        )
    if (tier0 == accepted) != (PARITY in bounds["weights_row"]):
        complaints.append(
            f"tier-0 weights goal {tier0}% against accepted bound {accepted}%, with"
            f" {PARITY!r} present in the row: {PARITY in bounds['weights_row']}"
        )
    if tier0 != accepted:
        complaints += [
            f"{row} does not quote the accepted weights bound {accepted}%"
            for row in ("weights_row", "expected_row")
            if accepted not in _percents(bounds[row])
        ]
    return complaints


@pytest.fixture(scope="module")
def documents():
    accepted = [(DESIGN / name).read_text(encoding="utf-8") for name in ACCEPTANCE]
    return accepted, (DESIGN / GOALS).read_text(encoding="utf-8")


def test_the_documents_as_they_stand_raise_no_complaint(documents):
    """The silent direction on the real pair: the two tables agree about weights."""
    assert audit(read_bounds(*documents)) == []


def test_the_tier0_weights_goal_is_tighter_and_the_kv_goal_is_the_same(documents):
    """The fact the correction turns on, derived rather than written down here."""
    bounds = read_bounds(*documents)
    assert bounds["tier0_weights"][0] < bounds["accepted_weights"][0][0]
    assert bounds["tier0_kv"][0] == bounds["accepted_kv"][0][0]


def test_weights_is_one_of_the_terms_the_acceptance_bound_calls_non_kv(documents):
    """The premise: the memory document gates those terms together and lists weights."""
    parts = (DESIGN / TERMS).read_text(encoding="utf-8").split("\n## ")
    section = next(p for p in parts if p.splitlines()[0].endswith(TERMS_HEADING))
    assert _percents(section)[0] == read_bounds(*documents)["accepted_weights"][0][0]
    assert re.search(r"^- \*\*Weights\*\*", section, re.MULTILINE)


DRIFTS = (
    ("goals", WEIGHTS_GOAL, "*tighter* than empirical's ≤10%", PARITY),
    ("goals", WEIGHTS_GOAL, "≤5%", "≤10%"),
    ("goals", EXPECTED_ROW, "10%", "12%"),
    ("acceptance", WEIGHTS_BOUND, "10%", "12%"),
)


@pytest.mark.parametrize("side,key,old,new", DRIFTS)
def test_the_guard_fires_when_one_side_moves_alone(documents, side, key, old, new):
    """The firing direction, one drift at a time, applied to the real document text."""
    acceptance, goals = documents
    if side == "goals":
        goals = _drift(goals, key, old, new)
    else:
        acceptance = [_drift(acceptance[0], key, old, new), acceptance[1]]
    assert audit(read_bounds(acceptance, goals)), "a lone drift left the guard silent"


def test_the_guard_is_silent_when_every_copy_of_a_bound_moves_together(documents):
    """A re-gating that moves the acceptance table and both citing rows stays green."""
    acceptance, goals = documents
    moved = [_drift(text, WEIGHTS_BOUND, "10%", "12%") for text in acceptance]
    for key in (WEIGHTS_GOAL, EXPECTED_ROW):
        goals = _drift(goals, key, "10%", "12%")
    assert audit(read_bounds(moved, goals)) == []
