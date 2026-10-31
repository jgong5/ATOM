# SPDX-License-Identifier: MIT
"""A decode step's MoE works on the real tokens the data-parallel group
gathers (`moe_tokens`), not on every row its busiest rank pads the gather to:
a pad row reaches no expert. A rung-only law fitted where every rank is full
prices high a step at the same rung whose ranks are mostly idle. A law fitted
with `moe_routing` lets each rung adopt one MoE column, the tokens `n` or the
experts they reach `X`: the form a record held out of the rest prices nearest,
weighted by steps, unless it takes some record out of the rung's bar.

Each rank's record is written from a known per-rank law with a MoE slope,
every line labelled with the group's step, as in `test_step_linear_dp`.
"""

import dataclasses
import json

import numpy as np
import pytest
from test_backend_step_linear_fit import SPEC, _fit, _record, _steps
from test_step_linear_draft import _mtp_record

from atom.compass.backends import (
    BatchView,
    CostRefused,
    RequestShape,
    StepLinearBackend,
    StepLinearLaw,
)
from atom.compass.backends.step_linear_fit import (
    _floor,
    _moe_column,
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
BASE = {"step": 0.015, "context": 2e-9, "graph_padding": 1e-10}
EXPERTS = {**BASE, "moe_experts": 6e-5}
TOKENS = {**BASE, "moe_tokens": 1.5e-4}


def _view(contexts, moe_tokens):
    rows = tuple(RequestShape(1, n, True) for n in contexts)
    return BatchView(rows, capture_rung=RUNG, moe_rows=DP * RUNG, moe_tokens=moe_tokens)


def _price(view: BatchView, law=EXPERTS) -> float:
    n = sum(r.context_tokens for r in view.decode)
    moe = {
        "moe_experts": activated_experts(*ROUTING, view.moe_tokens),
        "moe_tokens": view.moe_tokens,
    }
    counts = (1, n, view.graph_padding, *(moe[f] for f in law if f in moe))
    return float(np.dot(list(law.values()), counts))


FULL, SPARSE, PACKED = (3, 5), (1, 3), (RUNG, RUNG + 1)


def _groups(seed, n, others, law=EXPERTS, noise=0.0):
    """`n` decode steps at rung 4 over `DP` ranks, rank `i % DP` the slowest
    with 4 rows, each other rank `others`' rows: `FULL` 3 or 4, `SPARSE` 1 or
    2, `PACKED` 4, so that every step's group has the same tokens; each priced
    by `law`, scattered by up to `noise` either way."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        slow = i % DP
        low, high = others
        widths = [
            RUNG if r == slow else int(rng.integers(low, high)) for r in range(DP)
        ]
        spans = [(150_000, 200_000) if r == slow else (1, 40_000) for r in range(DP)]
        contexts = [
            [int(rng.integers(*spans[r])) for _ in range(w)]
            for r, w in enumerate(widths)
        ]
        views = [_view(c, sum(widths)) for c in contexts]
        prices = [_price(v, law) for v in views]
        assert int(np.argmax(prices)) == slow
        out.append((views, max(prices) * (1 + noise * rng.uniform(-1, 1))))
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


def _cells(tmp_path, law, steps=(200, 200), noise=0.0):
    """Two runs under `law`: one whose ranks are all full, one mostly idle."""
    full = _records(tmp_path / "full", _groups(0, steps[0], FULL, law, noise))
    sparse = _records(tmp_path / "sparse", _groups(1, steps[1], SPARSE, law, noise))
    return full, sparse


@pytest.fixture
def cells(tmp_path):
    return _cells(tmp_path, EXPERTS)


def test_a_rung_whose_cost_follows_the_experts_adopts_them(cells):
    full, sparse = cells
    law = _routed(full + sparse)[0]
    assert law["moe_routing"] == {"experts": ROUTING[0], "top_k": ROUTING[1]}
    vector = law["decode"][str(RUNG)]
    assert vector["coefficients"] == pytest.approx(EXPERTS, rel=1e-6)
    assert list(StepLinearLaw(law).decode[RUNG].features) == list(EXPERTS)
    figures = vector["fit"]["moe_column"]
    assert figures["adopted"] == figures["picked"] == "moe_experts"
    error = {f: v["error"] for f, v in figures["held_out"].items()}
    assert error["moe_experts"] < error["moe_tokens"] < error["none"]
    assert error["moe_experts"] == pytest.approx(0.0, abs=1e-6)


def test_a_rung_whose_cost_is_linear_in_the_tokens_adopts_them(tmp_path):
    full, sparse = _cells(tmp_path, TOKENS)
    law = _routed(full + sparse)[0]
    vector = law["decode"][str(RUNG)]
    assert vector["fit"]["moe_column"]["adopted"] == "moe_tokens"
    assert vector["coefficients"] == pytest.approx(TOKENS, rel=1e-6)
    # The tokens alone need no routing to price.
    del law["moe_routing"]
    backend = StepLinearBackend(StepLinearLaw(law), DP)
    view = _view((180_000, 20_000, 30_000, 40_000), 11)
    assert backend.price(view).seconds == pytest.approx(_price(view, TOKENS))


def _members(cells):
    """A rung's `fit` rows over `cells`, each (steps, (low, high) tokens, price
    of `n` tokens): its steps at tokens cycling over the span."""
    members = []
    for cell, (steps, (low, high), price) in enumerate(cells):
        for step in range(steps):
            n = low + step % (high - low + 1)
            members.append((cell, "8", (1, 0, 0), price(n), step, (), n))
    x = np.array([m[2] for m in members], dtype=np.float64)
    return members, x


def test_a_rung_whose_records_the_tokens_do_not_order_picks_no_column():
    """Within each record the cost rises with the tokens, across the two it
    does not: a column fitted on one record prices the other high."""
    rising = lambda low: lambda n: 0.010 + 1e-7 * (n - low)
    members, x = _members([(200, (10, 20), rising(10)), (200, (30, 40), rising(30))])
    chosen, figures = _moe_column(members, x, list(BASE), (384, 6))
    assert chosen is None
    assert figures["picked"] == "none" and figures["adopted"] is None
    error = {f: v["error"] for f, v in figures["held_out"].items()}
    assert error["none"] < min(error["moe_tokens"], error["moe_experts"])


def test_the_pick_weighs_each_held_out_record_by_its_steps():
    """Two large records the tokens price exactly, and a small one at the
    second's tokens that costs what the first does: the column misses the small
    one by more than no column does, and is picked on the steps."""
    line = lambda n: 0.010 + 5e-5 * n
    members, x = _members(
        [
            (400, (10, 20), line),
            (400, (30, 40), line),
            (20, (30, 40), lambda n: line(n - 20)),
        ]
    )
    figures = _moe_column(members, x, list(BASE), (384, 6))[1]
    held = figures["held_out"]
    for got in held.values():
        folds = got["folds"].values()
        weighted = sum(n * abs(p) for n, p in folds) / sum(n for n, _ in folds)
        assert got["error"] == pytest.approx(weighted)
    small = {f: abs(held[f]["folds"]["2"][1]) for f in held}
    assert small["moe_tokens"] > small["none"]
    assert figures["picked"] == figures["adopted"] == "moe_tokens"


def test_a_column_that_takes_a_record_out_of_the_bar_is_not_adopted():
    """The column prices the two large records far closer, but the small
    record, held out, leaves a bar of 4% either way that no column kept it in."""
    line = lambda n: 0.010 + 2e-5 * n
    members, x = _members(
        [
            (1000, (10, 20), line),
            (1000, (30, 40), line),
            (50, (50, 60), lambda n: line(25)),
        ]
    )
    reference = [-0.04, 0.04]
    chosen, figures = _moe_column(members, x, list(BASE), (384, 6), reference)
    held = figures["held_out"]
    assert figures["picked"] != "none"
    assert held[figures["picked"]]["error"] < held["none"]["error"]
    assert figures["bar"] == reference
    assert figures["left_bar"] == ["2"]
    assert abs(held["none"]["folds"]["2"][1]) <= 0.04
    assert chosen is None and figures["adopted"] is None


def test_a_rung_with_a_record_of_few_steps_adopts_its_column(tmp_path):
    """The cost follows the experts and one record holds 20 steps at the rung:
    held out of the other, it is priced as closely, and the column holds."""
    full, sparse = _cells(tmp_path, EXPERTS, steps=(200, 20))
    vector = _routed(full + sparse)[0]["decode"][str(RUNG)]
    figures = vector["fit"]["moe_column"]
    assert figures["adopted"] == "moe_experts"
    assert {c: f[0] for c, f in figures["held_out"]["none"]["folds"].items()} == {
        "0": 200,
        "1": 20,
    }
    assert vector["coefficients"] == pytest.approx(EXPERTS, rel=1e-6)


def test_a_rung_of_one_record_has_no_held_out_error_and_adopts_none():
    members, x = _members([(300, (10, 40), lambda n: 0.010 + 5e-5 * n)])
    chosen, figures = _moe_column(members, x, list(BASE), (384, 6))
    assert chosen is None and figures["picked"] is None
    assert all(v == {"error": None, "folds": {}} for v in figures["held_out"].values())


def test_the_floor_reads_each_records_steps_at_the_same_tokens():
    """Each step priced at the median of its record's steps at its tokens: a
    record's level is not mixed with another's, and a lone step is left out."""
    rows = [(0, 5, 1.0), (0, 5, 2.0), (0, 5, 3.0), (1, 5, 30.0), (1, 5, 30.0)]
    rows += [(0, 6, 9.0)]
    members = [(c, "4", (1, 0, 0), y, i, (), n) for i, (c, n, y) in enumerate(rows)]
    error = [1.0, 0.0, 2 / 3 - 1, 0.0, 0.0]
    assert _floor(members) == pytest.approx(np.percentile(error, (10, 90)).tolist())


@pytest.mark.parametrize("wide", [True, False])
def test_the_bar_is_the_wider_of_the_reference_band_and_the_floor(cells, wide):
    full, sparse = cells
    samples = label_records(full + sparse, DP)[0]
    band = 0.5 if wide else 1e-6
    held = {"bands": {f"decode rung {RUNG}": {"p10": -band, "p50": 0.0, "p90": band}}}
    reference = {"decode": {str(RUNG): {"held_out": held}}}
    options = {"form": "base", "dp": DP, "moe_routing": ROUTING}
    law = fit(samples, {**options, "reference": reference})[0]
    figures = law["decode"][str(RUNG)]["fit"]["moe_column"]
    assert figures["reference"] == [-band, band]
    floor = figures["floor"]
    assert floor[1] - floor[0] > 2e-6
    assert figures["bar"] == ([-band, band] if wide else floor)


def test_a_rung_law_fitted_on_full_ranks_prices_idle_ranks_high_and_the_routed_one_does_not(
    cells, tmp_path
):
    """Both laws are fitted on two runs of full ranks and price a run of idle
    ones, whose tokens no fitted step had."""
    full, _ = cells
    other = _records(tmp_path / "full2", _groups(2, 200, FULL))
    errors = {}
    for name, law in (
        ("rung", _fit(full + other, dp=DP)),
        ("routed", _routed(full + other)[0]),
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
    assert gap == pytest.approx(reached * EXPERTS["moe_experts"], rel=1e-6)


def test_on_one_rank_the_column_reads_the_steps_own_tokens(cells):
    full, sparse = cells
    backend = StepLinearBackend(StepLinearLaw(_routed(full + sparse)[0]))
    rows = tuple(RequestShape(1, n, True) for n in (180_000, 20_000, 30_000))
    terms = {t.name: t.seconds for t in backend.price(BatchView(rows, RUNG)).terms}
    reached = activated_experts(*ROUTING, 3)
    assert terms["decode.moe_experts"] == pytest.approx(
        reached * EXPERTS["moe_experts"]
    )


def test_a_data_parallel_step_stating_no_group_tokens_is_refused_not_priced_alone(
    cells,
):
    """A rank's own tokens are a fraction of its group's: the backend refuses
    a data-parallel step without `moe_tokens`, and the fit names it."""
    full, sparse = cells
    law, samples, options = _routed(full + sparse)
    backend = StepLinearBackend(StepLinearLaw(law), DP)
    contexts = (180_000, 20_000, 30_000, 40_000)
    with pytest.raises(CostRefused, match="no MoE group tokens on this step"):
        backend.price(_view(contexts, None))
    at = next(i for i, s in enumerate(samples) if not s.cut)
    view = dataclasses.replace(samples[at].view, moe_tokens=None)
    samples[at] = dataclasses.replace(samples[at], view=view)
    with pytest.raises(ValueError, match=f"step {samples[at].step} has no moe_tokens"):
        fit(samples, options)


def test_an_mtp_record_s_draft_law_takes_no_moe_column(tmp_path):
    records = [_mtp_record(tmp_path / f"dp0-{s}.jsonl", _steps(s)) for s in (3, 4)]
    law = fit(label_records(records)[0], {"moe_routing": ROUTING})[0]
    rungs = [v for k, v in law["decode"].items() if k.isdigit()]
    assert rungs and all("moe_column" in v["fit"] for v in rungs)
    assert law["draft"]["decode"]
    assert not any("moe_column" in v["fit"] for v in law["draft"]["decode"].values())
    assert "moe_routing" not in law["draft"]
    StepLinearLaw(law)


def test_a_law_with_the_experts_column_and_no_routing_refuses(cells):
    full, sparse = cells
    law = _routed(full + sparse)[0]
    del law["moe_routing"]
    with pytest.raises(ValueError, match="states no moe_routing"):
        StepLinearLaw(law)


def test_the_held_out_hull_spans_the_column_and_validates_inside_it(cells, tmp_path):
    """Three runs, so that each law fitted on the others holds two cells."""
    full, sparse = cells
    other = _records(tmp_path / "full2", _groups(2, 200, FULL))
    law, samples, options = _routed(full + sparse + other)
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
    with pytest.raises(ValueError, match="shared form fits no MoE column"):
        fit(samples, options)


def test_the_command_states_the_routing_it_fitted_and_the_reference_band(tmp_path):
    records = [str(_record(tmp_path / f"dp0-{s}.jsonl", _steps(s))) for s in (6, 7)]
    band = {"p10": -0.03, "p50": 0.0, "p90": 0.02}
    held = {"held_out": {"bands": {"decode rung 4": band}}}
    reference = tmp_path / "reference.json"
    reference.write_text(json.dumps({"decode": {"4": held}}))
    out = tmp_path / "law.json"
    main(
        [*records, "--out", str(out), *SPEC, "--moe-routing", "64,4"]
        + [
            "--reference",
            str(reference),
        ]
    )
    law = json.loads(out.read_text())
    assert law["moe_routing"] == {"experts": 64, "top_k": 4}
    figures = {k: law["decode"][k]["fit"]["moe_column"] for k in ("2", "4")}
    assert figures["4"]["reference"] == [-0.03, 0.02]
    assert figures["2"]["reference"] is None
    StepLinearLaw(law)


@pytest.mark.parametrize("value", ["64", "64,4,1", "4,64", "64,0", "64,x"])
def test_the_command_refuses_a_routing_other_than_experts_and_top_k(
    tmp_path, capsys, value
):
    out = tmp_path / "law.json"
    with pytest.raises(SystemExit):
        main(["dp0.jsonl", "--out", str(out), *SPEC, "--moe-routing", value])
    assert "argument --moe-routing" in capsys.readouterr().err
    assert not out.exists()


def test_a_rung_whose_steps_all_reach_the_same_experts_adopts_no_column(tmp_path):
    """Every rank packed on every step: one token count over the rung, which
    no column can price apart from the intercept."""
    runs = [_records(tmp_path / f"packed{s}", _groups(s, 100, PACKED)) for s in (2, 3)]
    law, samples, options = _routed(runs[0] + runs[1])
    vector = law["decode"][str(RUNG)]
    assert vector["fit"]["moe_column"]["adopted"] is None
    mark_held_out(law, {"held_out": {}}, samples, options)
    backend = StepLinearBackend(StepLinearLaw(law), DP)
    terms = backend.price(_view((170_000, 175_000, 180_000, 185_000), DP * RUNG)).terms
    assert all(t.provenance.validated for t in terms)
