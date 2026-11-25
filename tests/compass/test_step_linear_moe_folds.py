# SPDX-License-Identifier: MIT
"""`moe_folds` (`COMPASS_FIT_MOE_FOLDS`): how a decode rung of several records
holds them out to pick its MoE column. By default each record is held out
whole, priced by a fit on the others alone; under ``covered`` each record's
group compositions are, each priced by a fit on every other cell, and each
form is graded only on the held-out steps inside the hull of the other cells'
steps over the largest rank context, the group's MoE tokens and its dummies
(`_covered`), the rung adopting by in-sample MAD where none is. Two records
whose MoE tokens or contexts do not overlap, held out whole, grade each column
on how it extrapolates from one record's to the other's.

Records are written as in `test_step_linear_moe_column`'s dummy-forward tests:
each rank's record from a law over both MoE columns, its idle ranks running
dummy forwards, every line labelled with the group's step.
"""

import json

import numpy as np
import pytest
from test_backend_step_linear_fit import _record

from atom.compass.backends import BatchView, RequestShape
from atom.compass.backends.step_linear_fit import (
    MOE_FOLDS_ENV,
    env_options,
    fit,
    label_records,
)
from atom.compass.backends.work import activated_experts
from atom.compass.validate.cost import cost_accuracy

DP, RUNG, ROUTING = 4, 16, (32, 2)
#: The law, over both MoE columns: their VIF over these steps is past
#: `MOE_BOTH_VIF`, so a rung picks one of them or none.
LAW = {
    "step": 0.015,
    "context": 2e-9,
    "graph_padding": 1e-10,
    "moe_tokens": 2e-4,
    "moe_experts": 4e-4,
}


def _price(view: BatchView) -> float:
    n = sum(r.context_tokens for r in view.decode)
    reached = view.moe_tokens - max(view.group_dummies - 1, 0)
    moe = (view.moe_tokens, activated_experts(*ROUTING, reached))
    return float(np.dot(list(LAW.values()), (1, n, view.graph_padding, *moe)))


def _groups(seed, widths, n=150):
    """`n` steps at `RUNG` over `DP` ranks, every other one with its last rank
    idle, running a dummy forward of one row: rank 0 the slowest, full with
    long rows, each other busy rank `widths` rows (low, high)."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        dummies = range(DP - i % 2, DP)
        contexts = []
        for r in range(DP):
            if r in dummies:
                contexts.append([1])
                continue
            width = RUNG if r == 0 else int(rng.integers(*widths))
            span = (150_000, 200_000) if r == 0 else (1, 40_000)
            contexts.append([int(rng.integers(*span)) for _ in range(width)])
        tokens = sum(map(len, contexts))
        views = [
            BatchView(
                tuple(RequestShape(1, c, True) for c in mine),
                capture_rung=RUNG,
                moe_rows=DP * RUNG,
                moe_tokens=tokens,
                dp_dummy=r in dummies,
                group_dummies=len(dummies),
            )
            for r, mine in enumerate(contexts)
        ]
        out.append((views, max(map(_price, views))))
    return out


def _records(path, groups):
    """Rank `r`'s record of `groups`, its labels stretched by `1 + 0.01 r`, a
    dummy rank's line written as ATOM writes a dummy forward's."""
    path.mkdir(parents=True)
    paths = []
    for r in range(DP):
        steps = [
            ([(1, row.context_tokens, True) for row in v[r].requests], RUNG, seconds)
            for v, seconds in groups
        ]
        stretch = dict.fromkeys(range(len(steps)), 1 + 0.01 * r)
        record = _record(path / f"dp{r}.jsonl", steps, stretch)
        lines = [json.loads(line) for line in record.read_text().splitlines()]
        for line, (v, _) in zip(lines, groups):
            line.update(dp_rank=r, moe_rows=v[r].moe_rows, moe_tokens=v[r].moe_tokens)
            if v[r].dp_dummy:
                line.update(is_dummy_run=True, batch=[])
        record.write_text("".join(json.dumps(line) + "\n" for line in lines))
        paths.append(record)
    return paths


def _law(records, folds=None):
    options = {"form": "base", "dp": DP, "moe_routing": ROUTING}
    if folds:
        options["moe_folds"] = folds
    return fit(label_records(records, DP)[0], options)[0]


def test_the_switch_reads_its_cells_and_refuses_others():
    assert env_options({MOE_FOLDS_ENV: "covered"}) == {"moe_folds": "covered"}
    assert env_options({MOE_FOLDS_ENV: "record"}) == {"moe_folds": "record"}
    assert env_options({}) == {}
    with pytest.raises(ValueError, match=MOE_FOLDS_ENV):
        env_options({MOE_FOLDS_ENV: "rung"})


def test_records_of_tokens_no_other_reaches_adopt_the_column_a_third_run_needs(
    tmp_path,
):
    """One record's groups gather 19-22 tokens, the other's 41-64. Held out
    whole, `moe_experts`, which saturates past the first, extrapolates nearer
    from one to the other. Under ``covered`` no cell's steps lie inside the
    others' hull: the rung adopts by in-sample MAD, `moe_tokens`, and a third
    run between them is priced with less than half the error."""
    records = _records(tmp_path / "low", _groups(0, (1, 3)))
    records += _records(tmp_path / "high", _groups(1, (12, 17)))
    _records(tmp_path / "middle", _groups(2, (4, 10)))
    by_record, by_composition = _law(records), _law(records, "covered")

    figures = by_record["decode"][str(RUNG)]["fit"]["moe_column"]
    assert figures["held_out_by"] == "record"
    assert set(figures["held_out"]["none"]["folds"]) == {"0", "1"}
    assert figures["adopted"] == "moe_experts"
    figures = by_composition["decode"][str(RUNG)]["fit"]["moe_column"]
    assert figures["fold_cover"] == {c: [75, 0] for c in ("0:0", "0:1", "1:0", "1:1")}
    assert figures["selected_by"] == "insample" and figures["fallback"] == "uncovered"
    assert figures["adopted"] == "moe_tokens"

    def third(law):
        return cost_accuracy(tmp_path / "middle", law)["graded"]["decode"]["on_totals"]

    assert abs(third(by_composition)) < 0.5 * abs(third(by_record))


def test_a_rung_of_one_record_is_held_out_by_composition_either_way(tmp_path):
    records = _records(tmp_path / "low", _groups(0, (1, 17)))
    assert json.dumps(_law(records, "covered")) == json.dumps(_law(records))


#: Cell B's D at rung 24 (#752): two records of one composition, every rank
#: busy, that do not overlap: a small one of short contexts and few group
#: tokens, a large one of long contexts and many. The rung's cost mostly
#: follows the tokens the group gathers, some the experts it reaches.
BUSY_LAW = {**LAW, "moe_tokens": 1e-4, "moe_experts": 8e-4}
#: (seed, steps, peers' rows, the slowest rank's context per row[, rung]).
SMALL = (1, 60, (1, 6), (20_000, 30_000))
LARGE = (2, 600, (12, 17), (300_000, 400_000))


def _busy_price(view: BatchView) -> float:
    n = sum(r.context_tokens for r in view.decode)
    moe = (view.moe_tokens, activated_experts(*ROUTING, view.moe_tokens))
    return float(np.dot(list(BUSY_LAW.values()), (1, n, view.graph_padding, *moe)))


def _busy_records(path, seed, n, widths, span, rung=RUNG):
    """`n` steps at `rung` over `DP` busy ranks, rank `i % DP` the slowest at
    `rung` rows of `span` context, each other `widths` rows of short ones,
    labelled the max price over the ranks scattered by 0.2%; written as
    `_records` writes, at each step's rung."""
    rng = np.random.default_rng(seed)
    groups = []
    for i in range(n):
        slow = i % DP
        rows = [rung if r == slow else int(rng.integers(*widths)) for r in range(DP)]
        spans = [span if r == slow else (1, 20_000) for r in range(DP)]
        contexts = [
            [int(rng.integers(*spans[r])) for _ in range(w)] for r, w in enumerate(rows)
        ]
        views = [
            BatchView(
                tuple(RequestShape(1, c, True) for c in mine),
                capture_rung=rung,
                moe_rows=DP * rung,
                moe_tokens=sum(rows),
            )
            for mine in contexts
        ]
        prices = [_busy_price(v) for v in views]
        assert int(np.argmax(prices)) == slow
        groups.append((views, max(prices) * (1 + rng.normal(0, 0.002))))
    path.mkdir(parents=True, exist_ok=True)
    paths = []
    for r in range(DP):
        steps = [
            ([(1, x.context_tokens, True) for x in v[r].requests], v[r].capture_rung, t)
            for v, t in groups
        ]
        stretch = dict.fromkeys(range(len(steps)), 1 + 0.01 * r)
        record = _record(path / f"dp{r}.jsonl", steps, stretch)
        lines = [json.loads(line) for line in record.read_text().splitlines()]
        for line, (v, _) in zip(lines, groups):
            line.update(dp_rank=r, moe_rows=v[r].moe_rows, moe_tokens=v[r].moe_tokens)
        record.write_text("".join(json.dumps(line) + "\n" for line in lines))
        paths.append(record)
    return paths


def _busy_law(tmp_path, cells, folds=None):
    """The law fitted on one record per cell, each record's steps one or more
    `_busy_records` parts, under D+'s MoE rule (`fallback`, both columns)."""
    records = []
    for i, cell in enumerate(cells):
        for j, part in enumerate(cell):
            records += _busy_records(tmp_path / f"r{i}" / f"p{j}", *part)
    options = {"form": "base", "dp": DP, "moe_routing": ROUTING}
    options |= {"moe_both": True, "moe_select": "fallback"}
    if folds:
        options["moe_folds"] = folds
    return fit(label_records(records, DP)[0], options)[0]


def test_records_of_contexts_no_other_reaches_adopt_by_in_sample_mad(tmp_path):
    """Held out whole, the large record's fold, fitted on the small record
    alone, decides: `moe_experts`, saturating, extrapolates least badly.
    Under ``covered`` neither covers a step of the other, and the rung adopts
    the column its fit on all the steps holds tightest."""
    cells = ((SMALL,), (LARGE,))
    off = _busy_law(tmp_path / "off", cells)["decode"][str(RUNG)]["fit"]["moe_column"]
    assert off["held_out_by"] == "record" and off["adopted"] == "moe_experts"
    error = {f: v["error"] for f, v in off["held_out"].items()}
    assert error["moe_experts"] < error["moe_tokens"] < error["none"]
    assert "fold_cover" not in off
    law = _busy_law(tmp_path / "on", cells, "covered")
    on = law["decode"][str(RUNG)]["fit"]["moe_column"]
    assert on["fold_cover"] == {"0:0": [SMALL[1], 0], "1:0": [LARGE[1], 0]}
    assert on["selected_by"] == "insample" and on["fallback"] == "uncovered"
    assert on["adopted"] in ("moe_tokens", "moe_tokens+moe_experts")
    mad = {f: v["mad"] for f, v in on["in_sample"].items()}
    assert mad[on["adopted"]] == min(mad.values()) < mad["moe_experts"]


def test_records_that_cover_each_other_grade_their_covered_steps(tmp_path):
    """Two records of one population and composition: nearly every held-out
    step is covered, each form is graded on those alone, and the rung picks
    as held out whole."""
    part = ((12, 17), (300_000, 400_000))
    cells = (((3, 300, *part),), ((4, 300, *part),))
    off = _busy_law(tmp_path / "off", cells)["decode"][str(RUNG)]["fit"]["moe_column"]
    law = _busy_law(tmp_path / "on", cells, "covered")
    on = law["decode"][str(RUNG)]["fit"]["moe_column"]
    assert on["held_out_by"] == "record:group_dummies"
    assert on["picked"] == off["picked"]
    for steps, covered in on["fold_cover"].values():
        assert covered >= 0.9 * steps
    for form, v in on["held_out"].items():
        assert v["error"] == pytest.approx(off["held_out"][form]["error"], abs=2e-3)
        assert set(v["covered_bands"]) <= set(v["bands"])


def test_every_rung_takes_the_rule_and_none_without_it(tmp_path):
    """Two rungs: each states its cover under ``covered``,
    and none without (the fit's vector hull, also named `hull`, once took a
    switch's place from the second rung on)."""
    cells = (
        (SMALL, (11, 120, (1, 4), (20_000, 30_000), 8)),
        (LARGE, (12, 300, (5, 9), (300_000, 400_000), 8)),
    )
    off = _busy_law(tmp_path / "off", cells)
    on = _busy_law(tmp_path / "on", cells, "covered")
    assert set(off["decode"]) == set(on["decode"]) == {"8", str(RUNG)}
    for vector in off["decode"].values():
        figures = vector["fit"]["moe_column"]
        assert "fold_cover" not in figures
        assert all("covered_bands" not in v for v in figures["held_out"].values())
    for vector in on["decode"].values():
        assert "fold_cover" in vector["fit"]["moe_column"]
