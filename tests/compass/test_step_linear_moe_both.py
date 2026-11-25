# SPDX-License-Identifier: MIT
"""`moe_both` (`COMPASS_FIT_MOE_BOTH`): a target decode rung may pick, beside
no MoE column, `moe_tokens` and `moe_experts`, the form carrying both
(`MOE_BOTH`). Under tensor-parallel MoE every rank runs the group's gathered
tokens: the experts' weights it reads stop growing once nearly every expert is
reached, while the rows the gather and the experts' GEMMs carry keep growing
with the tokens. `moe_experts` alone holds such a rung flat past saturation
and its fit drops the fullest steps as outliers; `moe_tokens` alone misses the
saturating start. The form is a candidate only where its two columns' VIF over
the rung's fitted steps is `MOE_BOTH_VIF` or less.

Records are written as in `test_step_linear_moe_column`: each rank's record
from a known per-rank law, every line labelled with the group's step.
"""

import json

import numpy as np
import pytest
from test_backend_step_linear_fit import _record
from test_step_linear_moe_column import BASE, DP

from atom.compass.backends import (
    BatchView,
    RequestShape,
    StepLinearBackend,
    StepLinearLaw,
)
from atom.compass.backends.step_linear_fit import (
    MOE_BOTH,
    MOE_BOTH_ENV,
    MOE_BOTH_VIF,
    env_options,
    fit,
    label_records,
)
from atom.compass.backends.work import activated_experts

#: Rung 16 over 4 ranks: 19 to 64 group tokens, past the knee where 32
#: experts top-2 are nearly all reached (21.5 of them at 19 tokens, 31.4 at 64).
RUNG = 16
ROUTING = (32, 2)
BOTH = {**BASE, "moe_tokens": 2e-4, "moe_experts": 4e-4}


def _view(contexts, moe_tokens):
    rows = tuple(RequestShape(1, n, True) for n in contexts)
    return BatchView(rows, capture_rung=RUNG, moe_rows=DP * RUNG, moe_tokens=moe_tokens)


def _records(path, groups):
    """Rank `r`'s record of `groups` (as `test_step_linear_moe_column`'s, at `RUNG`)."""
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
        record.write_text("".join(json.dumps(line) + "\n" for line in lines))
        paths.append(record)
    return paths


def _price(view, law=BOTH) -> float:
    n = sum(r.context_tokens for r in view.decode)
    moe = {
        "moe_tokens": view.moe_tokens,
        "moe_experts": activated_experts(*ROUTING, view.moe_tokens),
    }
    counts = (1, n, view.graph_padding, *(moe[f] for f in law if f in moe))
    return float(np.dot(list(law.values()), counts))


def _groups(seed, n, widths, law=BOTH):
    """`n` decode steps at `RUNG` over `DP` ranks, rank `i % DP` the slowest,
    full with long rows, each other rank `widths` rows (low, high)."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        slow = i % DP
        rows = [RUNG if r == slow else int(rng.integers(*widths)) for r in range(DP)]
        spans = [(150_000, 200_000) if r == slow else (1, 40_000) for r in range(DP)]
        contexts = [
            [int(rng.integers(*spans[r])) for _ in range(w)] for r, w in enumerate(rows)
        ]
        views = [_view(c, sum(rows)) for c in contexts]
        prices = [_price(v, law) for v in views]
        assert int(np.argmax(prices)) == slow
        out.append((views, max(prices)))
    return out


def _fitted(tmp_path, cells, both=True):
    """The law fitted on one record per cell of (seed, steps, widths)."""
    records = []
    tmp_path.mkdir(parents=True, exist_ok=True)
    for seed, n, widths in cells:
        records += _records(tmp_path / f"c{seed}", _groups(seed, n, widths))
    samples = label_records(records, DP)[0]
    options = {"form": "base", "dp": DP, "moe_routing": ROUTING}
    if both:
        options["moe_both"] = True
    return fit(samples, options)[0]


#: Three records: peers of 1, 4-8 and 12-16 rows: 19, 28-40 and 52-64 group tokens.
CELLS = ((0, 150, (1, 2)), (1, 150, (4, 9)), (2, 150, (12, 17)))


def test_the_switch_reads_one_and_refuses_other_values():
    assert env_options({MOE_BOTH_ENV: "1"}) == {"moe_both": True}
    assert env_options({MOE_BOTH_ENV: "0"}) == {}
    assert env_options({}) == {}
    with pytest.raises(ValueError, match=MOE_BOTH_ENV):
        env_options({MOE_BOTH_ENV: "2"})


def test_a_rung_whose_cost_follows_both_adopts_both_and_recovers_its_law(tmp_path):
    law = _fitted(tmp_path, CELLS)
    vector = law["decode"][str(RUNG)]
    figures = vector["fit"]["moe_column"]
    assert figures["adopted"] == figures["picked"] == MOE_BOTH
    assert max(figures["both_vif"].values()) <= MOE_BOTH_VIF
    assert set(vector["fit"]["vif"]) == {"moe_tokens", "moe_experts"}
    assert vector["coefficients"] == pytest.approx(BOTH, rel=1e-6)
    error = {f: v["error"] for f, v in figures["held_out"].items()}
    assert error[MOE_BOTH] < min(error["moe_tokens"], error["moe_experts"])
    # The law loads and its backend prices both terms, in MOE_COLUMNS order.
    parsed = StepLinearLaw(law)
    assert list(parsed.decode[RUNG].features) == list(BOTH)
    views, _seconds = _groups(9, 1, (12, 17))[0]
    cost = StepLinearBackend(parsed, DP).price(views[0])
    names = [t.name for t in cost.terms]
    assert names[-2:] == ["decode.moe_tokens", "decode.moe_experts"]
    assert sum(t.seconds for t in cost.terms) == pytest.approx(
        _price(views[0]), rel=1e-6
    )


def test_without_the_switch_the_fullest_steps_are_priced_low(tmp_path):
    """The single-column law the default rule adopts misses the cost the
    fullest group's tokens add: priced on held-out full steps, below them."""
    law = _fitted(tmp_path, CELLS, both=False)
    vector = law["decode"][str(RUNG)]
    assert vector["fit"]["moe_column"]["adopted"] in ("moe_tokens", "moe_experts")
    assert MOE_BOTH not in vector["fit"]["moe_column"]["held_out"]
    parsed = StepLinearBackend(StepLinearLaw(law), DP)
    both = StepLinearBackend(StepLinearLaw(_fitted(tmp_path / "b", CELLS)), DP)
    full = _groups(7, 40, (16, 17))
    err = [
        max(parsed.price(v).target_seconds for v in views) / s - 1 for views, s in full
    ]
    err_both = [
        max(both.price(v).target_seconds for v in views) / s - 1 for views, s in full
    ]
    assert abs(np.median(err_both)) < 1e-6 < abs(np.median(err))


def test_a_rung_whose_steps_cannot_separate_the_columns_keeps_one(tmp_path):
    """Every step at the same group tokens: the two columns are constant, as
    the intercept is; the form is no candidate, its VIF stated."""
    law = _fitted(tmp_path, ((0, 150, (16, 17)), (1, 150, (16, 17))))
    figures = law["decode"][str(RUNG)]["fit"]["moe_column"]
    assert "both_vif" in figures
    assert MOE_BOTH not in figures.get("held_out", figures.get("in_sample", {}))
    assert figures["adopted"] != MOE_BOTH
