# SPDX-License-Identifier: MIT
"""The tier-0 memory goals against the acceptance bounds they cite.

`10_analytic_laws.md` declares a tier-0 accuracy goal per quantity, and its
memory rows point at the project's acceptance table for their number. That table
has two memory buckets: a tight one for KV capacity / block count and a looser
one for each non-KV memory term, of which weights is one --
`03_memory_and_kv_model.md` gates the non-KV terms together and lists weights
first. So the tier-0 goal on weights, which is the KV bucket's number, is
*tighter* than the acceptance bound rather than the same as it.

An earlier wording denied that, grouping weights, KV capacity and block count
into one cell and calling the number "same as empirical". It was read the wrong
way in review once already: a measured weight-ratio error was checked against the
tighter bound, contradicting the acceptance table the same review had quoted
correctly.

**What states a bound and what does not.** The comparison states neither bound:
`read_bounds` and `audit` hold no numeric literal, so both bounds reach every
assertion through the documents and the guard has no copy to agree with itself
about. The drift cases below do quote the text in order to perturb it, but their
operands are *derived from the same reading*, so a re-gating that moves every
copy of a bound leaves this file green without editing it.

The guard fires when

* the tier-0 goal and the acceptance bound differ and the row claims parity
  anyway, or they agree and the row denies it;
* either bound moves without the other -- including a bound moving in one of the
  acceptance table's two copies (`README.md`, `00_initial_prompt.md`) but not the
  other, for the weights bound and for the KV bound alike, and either citing row
  quoting a bound the acceptance table no longer states;
* the tier-0 KV goal stops matching the acceptance KV bound, which is the half of
  the old cell where the parity claim was true and is worth keeping.

It stays silent when every copy of a bound moves together, which is what a
deliberate re-gating looks like. That direction is pinned too, so the guard
cannot be satisfied by a checker that refuses every change; the firing direction
is pinned by re-introducing each drift into the real document text rather than
into a fixture that transcribes it.

The acceptance bound on weights has **five** copies in this design directory:
`README.md`, `00_initial_prompt.md`, `03_memory_and_kv_model.md`, and the two
rows of `10_analytic_laws.md` that quote it. A re-gating has to move all five.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

DESIGN = Path(__file__).resolve().parents[2] / "atom" / "compass" / "design"
ACCEPTANCE = ("README.md", "00_initial_prompt.md")
GOALS = "10_analytic_laws.md"
TERMS = "03_memory_and_kv_model.md"
TERMS_GATE = "acceptance gate, individually"
TERMS_FIRST = "- **Weights**"
WEIGHTS_GOAL = "memoryweights"
KV_GOAL = "memorykvcapacityblockcount"
EXPECTED_ROW = "analyticmemoryweightskv"
WEIGHTS_BOUND = "eachnonkvmemoryterm"
KV_BOUND = "kvcapacityblockcount"
PARITY = "same as empirical"


def _key(line):
    """A table row's first cell, reduced so two documents' spellings of it agree."""
    return re.sub(r"[^a-z]", "", line.strip().strip("|").split("|")[0].lower())


def _pct(value):
    """A bound, spelled the way the documents spell it."""
    return f"{value:g}%"


def _percents(text):
    """Every percentage the text states, in the order written."""
    return [float(found) for found in re.findall(r"(\d+(?:\.\d+)?)\s*%", text)]


def _claims_parity(text):
    """Whether `text` asserts the parity phrase rather than denying it."""
    found = re.search(rf"(?P<lead>.{{0,16}}){PARITY}", text)
    return bool(found) and "not" not in found["lead"]


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


def _line(text, key):
    """The one line of `text` containing `key`; refuses rather than take the first."""
    hits = [line for line in text.splitlines() if key in line]
    assert len(hits) == 1, (
        f"{key!r} matched {len(hits)} lines of this document, wanted one"
    )
    return hits[0]


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
    """Every way the tier-0 memory goals and the accepted bounds can disagree."""
    accepted, tier0 = bounds["accepted_weights"][0][0], bounds["tier0_weights"][0]
    complaints = []
    for term in ("weights", "kv"):
        if len({tuple(copy) for copy in bounds[f"accepted_{term}"]}) != 1:
            complaints.append(
                f"the acceptance table's copies disagree on {term}:"
                f" {bounds[f'accepted_{term}']}"
            )
    if bounds["tier0_kv"][0] != bounds["accepted_kv"][0][0]:
        complaints.append(
            f"the tier-0 KV goal {bounds['tier0_kv'][0]}% is no longer the accepted"
            f" {bounds['accepted_kv'][0][0]}%, yet its row still says {PARITY!r}"
        )
    if (tier0 == accepted) != _claims_parity(bounds["weights_row"]):
        complaints.append(
            f"tier-0 weights goal {tier0}% against accepted bound {accepted}%, with"
            f" {PARITY!r} claimed by the row: {_claims_parity(bounds['weights_row'])}"
        )
    if tier0 != accepted:
        complaints += [
            f"{row} does not quote the accepted weights bound {accepted}%"
            for row in ("weights_row", "expected_row")
            if accepted not in _percents(bounds[row])
        ]
    return complaints


def drift_cases(bounds):
    """One drift each, with every operand derived from the bounds just read."""
    accepted, kv = bounds["accepted_weights"][0][0], bounds["accepted_kv"][0][0]
    goal, moved, kv_moved = bounds["tier0_weights"][0], _pct(accepted + 2), _pct(kv + 2)
    return (
        ("goals", 0, WEIGHTS_GOAL, "*tighter* than", "the same as"),
        ("goals", 0, WEIGHTS_GOAL, f"**≤{_pct(goal)}**", f"**≤{_pct(accepted)}**"),
        ("goals", 0, EXPECTED_ROW, _pct(accepted), moved),
        ("acceptance", 0, WEIGHTS_BOUND, _pct(accepted), moved),
        ("acceptance", 1, WEIGHTS_BOUND, _pct(accepted), moved),
        ("acceptance", 0, KV_BOUND, _pct(kv), kv_moved),
        ("acceptance", 1, KV_BOUND, _pct(kv), kv_moved),
    )


def _documents():
    accepted = [(DESIGN / name).read_text(encoding="utf-8") for name in ACCEPTANCE]
    return accepted, (DESIGN / GOALS).read_text(encoding="utf-8")


DRIFTS = drift_cases(read_bounds(*_documents()))


@pytest.fixture(scope="module")
def documents():
    return _documents()


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
    terms = (DESIGN / TERMS).read_text(encoding="utf-8")
    assert _percents(_line(terms, TERMS_GATE)) == [
        read_bounds(*documents)["accepted_weights"][0][0]
    ]
    assert _line(terms, TERMS_FIRST).startswith(TERMS_FIRST)


@pytest.mark.parametrize("side,copy,key,old,new", DRIFTS)
def test_the_guard_fires_when_one_side_moves_alone(
    documents, side, copy, key, old, new
):
    """The firing direction, one drift at a time, applied to the real document text."""
    acceptance, goals = documents
    if side == "goals":
        goals = _drift(goals, key, old, new)
    else:
        acceptance = [
            _drift(t, key, old, new) if i == copy else t
            for i, t in enumerate(acceptance)
        ]
    assert audit(read_bounds(acceptance, goals)), "a lone drift left the guard silent"


def test_the_guard_is_silent_when_every_copy_of_a_bound_moves_together(documents):
    """A re-gating that moves the acceptance table and both citing rows stays green."""
    acceptance, goals = documents
    accepted = read_bounds(acceptance, goals)["accepted_weights"][0][0]
    old, new = _pct(accepted), _pct(accepted + 2)
    moved = [_drift(text, WEIGHTS_BOUND, old, new) for text in acceptance]
    for key in (WEIGHTS_GOAL, EXPECTED_ROW):
        goals = _drift(goals, key, old, new)
    assert audit(read_bounds(moved, goals)) == []
