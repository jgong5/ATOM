# SPDX-License-Identifier: MIT
"""A decode step's MoE reads the weights of the experts the data-parallel
group's real tokens reach (`moe_tokens`), not of every row its busiest rank pads
the gather to: a pad row reaches no expert. A rung-only law fitted where every
rank is full prices high a step at the same rung whose ranks are mostly idle;
a law fitted with `moe_routing` adds `moe_experts` and prices both.

Each rank's record is written from a known per-rank law with a `moe_experts`
slope, every line labelled with the group's step, as in `test_step_linear_dp`.
"""

import json

import numpy as np
import pytest
from test_backend_step_linear_fit import SPEC, _fit, _record, _steps

from atom.compass.backends import (
    BatchView,
    RequestShape,
    StepLinearBackend,
    StepLinearLaw,
)
from atom.compass.backends.step_linear_fit import (
    fit,
    label_records,
    main,
    mark_held_out,
)
from atom.compass.backends.work import activated_experts
from atom.compass.validate.cost import cost_accuracy

DP = 4
RUNG = 4
ROUTING = (64, 4)
LAW = {"step": 0.015, "context": 2e-9, "graph_padding": 1e-10, "moe_experts": 6e-5}


def _view(contexts, moe_tokens):
    rows = tuple(RequestShape(1, n, True) for n in contexts)
    return BatchView(rows, capture_rung=RUNG, moe_rows=DP * RUNG, moe_tokens=moe_tokens)


def _price(view: BatchView) -> float:
    n = sum(r.context_tokens for r in view.decode)
    reached = activated_experts(*ROUTING, view.moe_tokens)
    return float(np.dot(list(LAW.values()), (1, n, view.graph_padding, reached)))


def _groups(seed, n, full):
    """`n` decode steps at rung 4 over `DP` ranks, rank `i % DP` the slowest;
    every rank holding 3 or 4 rows when `full`, else the others 1 or 2."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        slow = i % DP
        low, high = (3, 5) if full else (1, 3)
        widths = [
            RUNG if r == slow else int(rng.integers(low, high)) for r in range(DP)
        ]
        spans = [(150_000, 200_000) if r == slow else (1, 40_000) for r in range(DP)]
        contexts = [
            [int(rng.integers(*spans[r])) for _ in range(w)]
            for r, w in enumerate(widths)
        ]
        views = [_view(c, sum(widths)) for c in contexts]
        prices = [_price(v) for v in views]
        assert int(np.argmax(prices)) == slow
        out.append((views, max(prices)))
    return out


def _records(path, groups):
    """Rank `r`'s record of `groups`, its labels stretched by `1 + 0.01 r`."""
    path.mkdir()
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


def _routed(records):
    samples = label_records(records, DP)[0]
    options = {"form": "base", "dp": DP, "moe_routing": ROUTING}
    return fit(samples, options)[0], samples, options


@pytest.fixture
def cells(tmp_path):
    """Two runs: one whose ranks are all full, one whose ranks are mostly idle."""
    full = _records(tmp_path / "full", _groups(0, 200, full=True))
    sparse = _records(tmp_path / "sparse", _groups(1, 200, full=False))
    return full, sparse


def test_the_fit_with_moe_routing_recovers_the_per_rank_law(cells):
    full, sparse = cells
    law = _routed(full + sparse)[0]
    assert law["moe_routing"] == {"experts": ROUTING[0], "top_k": ROUTING[1]}
    vector = law["decode"][str(RUNG)]
    assert vector["coefficients"] == pytest.approx(LAW, rel=1e-6)
    assert list(StepLinearLaw(law).decode[RUNG].features) == list(LAW)


def test_a_rung_law_fitted_on_full_ranks_prices_idle_ranks_high_and_the_routed_one_does_not(
    cells, tmp_path
):
    full, _ = cells
    errors = {}
    for name, law in (
        ("rung", _fit(full, dp=DP)),
        ("routed", _routed(full)[0]),
    ):
        errors[name] = cost_accuracy(tmp_path / "sparse", law)["graded"]["decode"]
    assert errors["rung"]["on_totals"] > 0.03
    assert errors["routed"]["on_totals"] == pytest.approx(0.0, abs=1e-6)


def test_the_backend_prices_a_step_from_its_group_tokens(cells):
    full, sparse = cells
    backend = StepLinearBackend(StepLinearLaw(_routed(full + sparse)[0]), DP)
    contexts = (180_000, 20_000, 30_000, 40_000)
    view, fuller = _view(contexts, 12), _view(contexts, 16)
    assert backend.price(view).seconds == pytest.approx(_price(view), rel=1e-6)
    gap = backend.price(fuller).seconds - backend.price(view).seconds
    reached = activated_experts(*ROUTING, 16) - activated_experts(*ROUTING, 12)
    assert gap == pytest.approx(reached * LAW["moe_experts"], rel=1e-6)


def test_on_one_rank_the_column_reads_the_steps_own_tokens(cells):
    full, sparse = cells
    backend = StepLinearBackend(StepLinearLaw(_routed(full + sparse)[0]))
    rows = tuple(RequestShape(1, n, True) for n in (180_000, 20_000, 30_000))
    terms = {t.name: t.seconds for t in backend.price(BatchView(rows, RUNG)).terms}
    reached = activated_experts(*ROUTING, 3)
    assert terms["decode.moe_experts"] == pytest.approx(reached * LAW["moe_experts"])


def test_a_law_with_the_column_and_no_routing_refuses(cells):
    full, sparse = cells
    law = _routed(full + sparse)[0]
    del law["moe_routing"]
    with pytest.raises(ValueError, match="states no moe_routing"):
        StepLinearLaw(law)


def test_the_held_out_hull_spans_the_column_and_validates_inside_it(cells):
    full, sparse = cells
    law, samples, options = _routed(full + sparse)
    mark_held_out(law, {"held_out": {}}, samples, options)
    band = law["decode"][str(RUNG)]["held_out"]["bands"][f"decode rung {RUNG}"]
    assert abs(band["p50"]) < 1e-6
    backend = StepLinearBackend(StepLinearLaw(law), DP)
    contexts = (170_000, 175_000, 180_000, 185_000)
    assert all(t.provenance.validated for t in backend.price(_view(contexts, 9)).terms)
    assert not any(
        t.provenance.validated for t in backend.price(_view(contexts, 60)).terms
    )


def test_the_shared_form_refuses_moe_routing(cells):
    full, sparse = cells
    samples = label_records(full + sparse, DP)[0]
    options = {"form": "shared", "dp": DP, "moe_routing": ROUTING}
    with pytest.raises(ValueError, match="shared form fits no moe_experts"):
        fit(samples, options)


def test_the_command_states_the_routing_it_fitted(tmp_path):
    records = [str(_record(tmp_path / f"dp0-{s}.jsonl", _steps(s))) for s in (6, 7)]
    out = tmp_path / "law.json"
    main([*records, "--out", str(out), *SPEC, "--moe-routing", "64,4"])
    law = json.loads(out.read_text())
    assert law["moe_routing"] == {"experts": 64, "top_k": 4}
    assert all("moe_experts" in v["coefficients"] for v in law["decode"].values())
    StepLinearLaw(law)
