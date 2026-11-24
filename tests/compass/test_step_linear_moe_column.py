# SPDX-License-Identifier: MIT
"""A decode step's MoE works on the real tokens the data-parallel group
gathers (`moe_tokens`), not on every row its busiest rank pads the gather to:
a pad row reaches no expert. A rung-only law fitted where every rank is full
prices high a step at the same rung whose ranks are mostly idle. A law fitted
with `moe_routing` lets each rung adopt one MoE column, the tokens `n` or the
experts they reach `X`: the form a record held out of the rest prices nearest,
weighted by steps. A rank with no work runs a dummy forward whose one row,
token 0 at position 0, reaches the gate like a real token, and every dummy's
row is the same: `X` counts a step's dummies as one token, where `n` counts
each.

Each rank's record is written from a known per-rank law with a MoE slope,
every line labelled with the group's step, as in `test_step_linear_dp`.
"""

import dataclasses
import json

import numpy as np
import pytest
from test_backend_step_linear_fit import SPEC, _fit, _record, _steps
from test_dp_step_max import Recording, dp2, prefill, runner  # noqa: F401  dp2
from test_step_linear_draft import _mtp_record

from atom.compass.backends import (
    BatchView,
    CostRefused,
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


def test_a_rung_whose_records_the_tokens_do_not_order_picks_no_column(tmp_path):
    """Within each record the cost rises with the tokens, across the two it
    does not: a column fitted on one record prices the other far off."""
    dearer = {**TOKENS, "step": TOKENS["step"] + 0.002}
    sparse = _records(tmp_path / "sparse", _groups(3, 200, SPARSE, dearer))
    full = _records(tmp_path / "full", _groups(4, 200, FULL, TOKENS))
    vector = _routed(sparse + full)[0]["decode"][str(RUNG)]
    figures = vector["fit"]["moe_column"]
    assert figures["picked"] == "none" and figures["adopted"] is None
    assert list(vector["coefficients"]) == list(BASE)
    error = {f: v["error"] for f, v in figures["held_out"].items()}
    assert error["none"] < min(error["moe_tokens"], error["moe_experts"])


WIDE = {"step": 0.015, "context": 2e-8, "graph_padding": 1e-10, "moe_experts": 2e-4}


def test_the_pick_weighs_each_held_out_record_by_its_steps(tmp_path):
    """Two large records the experts price exactly, and a small one at the
    first's tokens that costs 3% less: held out, the column misses the small
    one by more than no column does, and is adopted on the steps however far
    it moves that one record."""
    cheaper = {**WIDE, "step": WIDE["step"] - 0.0011}
    records = _records(tmp_path / "full", _groups(0, 200, FULL, WIDE))
    records += _records(tmp_path / "sparse", _groups(1, 200, SPARSE, WIDE))
    records += _records(tmp_path / "small", _groups(2, 20, FULL, cheaper))
    vector = _routed(records)[0]["decode"][str(RUNG)]
    figures = vector["fit"]["moe_column"]
    held = figures["held_out"]
    for got in held.values():
        folds = got["folds"].values()
        weighted = sum(n * abs(p) for n, p in folds) / sum(n for n, _ in folds)
        assert got["error"] == pytest.approx(weighted)
    small = {f: abs(held[f]["folds"]["2"][1]) for f in held}
    assert small["moe_experts"] > small["none"] + 0.02
    assert figures["picked"] == figures["adopted"] == "moe_experts"
    assert list(vector["coefficients"]) == list(WIDE)


SKEWED = {"step": 0.015, "context": 2e-8, "graph_padding": 3e-8, "moe_tokens": 4e-4}


def _wide_or_tall(seed, others, wide):
    """200 decode steps at rung 4 under `SKEWED`: rank 0 runs 4 rows near one
    context drawn from `wide`, rank 1 one row 1.2 to 2.2 times as long, which
    the rung pads, and on some steps a short one, the others `others`' short
    rows."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(200):
        n = rng.uniform(*wide)
        rows = [int(n * rng.uniform(0.9, 1.0)) for _ in range(RUNG)]
        tall = [int(n * rng.uniform(1.2, 2.2)), int(n * rng.uniform(0.05, 0.3))]
        contexts = [rows, tall[: 1 + int(rng.integers(0, 2))]]
        for _ in range(DP - 2):
            contexts.append(rng.integers(1, 1000, rng.integers(*others)).tolist())
        views = [_view(c, sum(map(len, contexts))) for c in contexts]
        out.append((views, max(_price(v, SKEWED) for v in views)))
    return out


@pytest.fixture
def skewed(tmp_path):
    groups = [
        _wide_or_tall(0, FULL, (110_000, 130_000)),
        _wide_or_tall(1, SPARSE, (80_000, 100_000)),
    ]
    paths = [
        _records(tmp_path / name, g) for name, g in zip(("full", "sparse"), groups)
    ]
    return groups, paths


def test_each_form_is_chosen_on_the_rows_it_prices_not_the_base_form_s(skewed):
    """The cost follows the tokens. The law without a column, fitted on both
    records, prices some steps costliest at another rank than the cost does:
    on the rows it picks, the experts column holds out nearer than the
    tokens. Each form fitted on its own rows, the tokens price every record
    exactly and are adopted."""
    groups, paths = skewed
    base = StepLinearBackend(StepLinearLaw(_fit(paths[0] + paths[1], dp=DP)), DP)
    moved = 0
    for views, _ in groups[0] + groups[1]:
        truth = np.argmax([_price(v, SKEWED) for v in views])
        moved += truth != np.argmax([base.price(v).seconds for v in views])
    assert moved

    vector = _routed(paths[0] + paths[1])[0]["decode"][str(RUNG)]
    figures = vector["fit"]["moe_column"]
    assert figures["picked"] == figures["adopted"] == "moe_tokens"
    error = {f: v["error"] for f, v in figures["held_out"].items()}
    assert error["moe_tokens"] == pytest.approx(0.0, abs=1e-6)
    assert error["moe_tokens"] < error["moe_experts"] < error["none"]
    assert vector["coefficients"] == pytest.approx(SKEWED, rel=1e-6)


def test_a_held_out_fold_prices_each_step_as_the_max_over_its_ranks(skewed):
    """A record's fold under no column is the law fitted on the other record
    alone, and its error on each held-out step is that law's costliest rank's
    price over the group's step."""
    groups, paths = skewed
    folds = _routed(paths[0] + paths[1])[0]["decode"][str(RUNG)]["fit"]
    folds = folds["moe_column"]["held_out"]["none"]["folds"]
    for held, other in ((0, 1), (1, 0)):
        backend = StepLinearBackend(StepLinearLaw(_fit(paths[other], dp=DP)), DP)
        error = [
            max(backend.price(v).seconds for v in views) / seconds - 1.0
            for views, seconds in groups[held]
        ]
        assert folds[str(held)] == pytest.approx([200, np.median(error)], rel=1e-9)


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


def test_a_rung_of_one_record_has_no_held_out_error_and_adopts_none(cells):
    full, _ = cells
    vector = _routed(full)[0]["decode"][str(RUNG)]
    figures = vector["fit"]["moe_column"]
    assert figures["picked"] is None and figures["adopted"] is None
    assert all(v == {"error": None, "folds": {}} for v in figures["held_out"].values())
    assert list(vector["coefficients"]) == list(BASE)


def test_a_routed_rung_fits_each_fold_of_the_form_it_adopts_once(cells, monkeypatch):
    """Both records hold enough steps at the rung for `rowmod` to hold each
    out without its column: it takes the experts column's folds the column
    was chosen on, and the law is the one it writes fitting them again."""
    from atom.compass.backends import step_linear_fit

    folds, fitted = step_linear_fit._max_affine_folds, []

    def counted(steps, vector, routing):
        fitted.append(tuple(vector["coefficients"]))
        return folds(steps, vector, routing)

    monkeypatch.setattr(step_linear_fit, "_max_affine_folds", counted)
    full, sparse = cells
    law = _routed(full + sparse)[0]
    assert law["decode"][str(RUNG)]["fit"]["moe_column"]["adopted"] == "moe_experts"
    assert fitted.count(tuple(EXPERTS)) == 1
    rowmod, fitted[:] = step_linear_fit._rowmod, []
    monkeypatch.setattr(step_linear_fit, "_rowmod", lambda *a: rowmod(*a[:-1], None))
    assert json.dumps(_routed(full + sparse)[0]) == json.dumps(law)
    assert fitted.count(tuple(EXPERTS)) == 2


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


def test_the_command_states_the_routing_it_fitted_and_takes_no_reference_law(
    tmp_path, capsys
):
    records = [str(_record(tmp_path / f"dp0-{s}.jsonl", _steps(s))) for s in (6, 7)]
    out = tmp_path / "law.json"
    argv = [*records, "--out", str(out), *SPEC, "--moe-routing", "64,4"]
    with pytest.raises(SystemExit):
        main([*argv, "--reference", str(tmp_path / "reference.json")])
    assert "unrecognized arguments: --reference" in capsys.readouterr().err
    main(argv)
    law = json.loads(out.read_text())
    assert law["moe_routing"] == {"experts": 64, "top_k": 4}
    assert all("moe_column" in law["decode"][k]["fit"] for k in ("2", "4"))
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


def _merged_price(view: BatchView, law=EXPERTS) -> float:
    """`_price` of `view` under `law`, a law over the experts alone, with its
    group's dummy forwards reaching one token's experts."""
    reached = view.moe_tokens - max(view.group_dummies - 1, 0)
    return _price(dataclasses.replace(view, moe_tokens=reached), law)


def _dummy_groups(seed, n, law=EXPERTS):
    """`n` decode steps at rung 4 over `DP` ranks: at step `i` the last
    `i % DP` ranks run a dummy forward, one row of context 1, and the others
    hold the same 4 rows. The fit picks a step's slowest rank on the rung's
    features without the MoE column, so ranks that differ only in those
    features keep that pick off the MoE term, which the dummies move. Each
    rank's view states the group's tokens, a dummy's one among them, and its
    dummies; each step is priced by `law` as `_merged_price` prices it."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        dummies = range(DP - i % DP, DP)
        rows = [int(rng.integers(150_000, 200_000)) for _ in range(RUNG)]
        contexts = [[1] if r in dummies else rows for r in range(DP)]
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
        out.append((views, max(_merged_price(v, law) for v in views)))
    return out


def _dummy_records(path, groups):
    """Rank `r`'s record of `groups`, every rank's label the group's step, a
    dummy rank's line written as ATOM writes a dummy forward's: no request,
    its one row."""
    path.mkdir()
    paths = []
    for r in range(DP):
        steps = [
            ([(1, row.context_tokens, True) for row in v[r].requests], RUNG, seconds)
            for v, seconds in groups
        ]
        record = _record(path / f"dp{r}.jsonl", steps)
        lines = [json.loads(line) for line in record.read_text().splitlines()]
        for line, (v, _) in zip(lines, groups):
            line.update(dp_rank=r, moe_rows=v[r].moe_rows, moe_tokens=v[r].moe_tokens)
            if v[r].dp_dummy:
                line.update(is_dummy_run=True, batch=[])
        record.write_text("".join(json.dumps(line) + "\n" for line in lines))
        paths.append(record)
    return paths


def test_a_group_s_dummy_forwards_reach_one_token_s_experts_in_the_fit(tmp_path):
    """Idle ranks run dummy forwards on most steps, from none to three a step:
    the routed fit reads each step's dummies off its ranks' lines and recovers
    the law whose experts count them as one token."""
    runs = [_dummy_records(tmp_path / f"run{s}", _dummy_groups(s, 200)) for s in (0, 1)]
    law, samples, _ = _routed(runs[0] + runs[1])
    assert {s.view.group_dummies for s in samples} == {0, 1, 2, 3}
    vector = law["decode"][str(RUNG)]
    figures = vector["fit"]["moe_column"]
    assert figures["adopted"] == figures["picked"] == "moe_experts"
    assert figures["held_out"]["moe_experts"]["error"] == pytest.approx(0, abs=1e-6)
    fitted = {f: vector["coefficients"][f] for f in EXPERTS}
    assert fitted == pytest.approx(EXPERTS, rel=1e-6)


def test_the_backend_counts_a_group_s_dummies_as_one_token_in_the_experts(cells):
    full, sparse = cells
    law = _routed(full + sparse)[0]
    backend = StepLinearBackend(StepLinearLaw(law), DP)
    slope = law["decode"][str(RUNG)]["coefficients"]["moe_experts"]
    view = _view((180_000, 20_000, 30_000, 40_000), 12)

    def experts(dummies):
        cost = backend.price(dataclasses.replace(view, group_dummies=dummies))
        return {t.name: t.seconds for t in cost.terms}["decode.moe_experts"]

    for dummies in (0, 1, 2, 3):
        reached = activated_experts(*ROUTING, 12 - max(dummies - 1, 0))
        assert experts(dummies) == pytest.approx(reached * slope, rel=1e-12)
    assert experts(1) == experts(0) > experts(2) > experts(3)


def test_the_tokens_column_counts_each_dummy_s_row(tmp_path):
    full, sparse = _cells(tmp_path, TOKENS)
    backend = StepLinearBackend(StepLinearLaw(_routed(full + sparse)[0]), DP)
    view = _view((180_000, 20_000, 30_000, 40_000), 11)

    def priced(dummies):
        return backend.price(dataclasses.replace(view, group_dummies=dummies)).seconds

    assert priced(3) == priced(0) == pytest.approx(_price(view, TOKENS))


def test_the_cost_check_prices_each_step_with_its_group_s_dummies(tmp_path):
    runs = [_dummy_records(tmp_path / f"run{s}", _dummy_groups(s, 200)) for s in (0, 1)]
    law = _routed(runs[0] + runs[1])[0]
    _dummy_records(tmp_path / "held", _dummy_groups(2, 200))
    graded = cost_accuracy(tmp_path / "held", law)["graded"]["decode"]
    assert graded["on_totals"] == pytest.approx(0.0, abs=1e-6)


def test_a_simulated_dp_step_states_its_group_s_dummy_forwards(dp2):  # noqa: F811
    """Rank 1 has no work on the first step and runs ATOM's dummy forward:
    both ranks price a view stating one dummy. On the second both work."""
    backends = [Recording(), Recording()]
    ranks = [runner(b, dp_size=2, dp_rank=r) for r, b in enumerate(backends)]
    dp2(
        lambda: [ranks[0].forward(prefill(40)) for _ in range(2)],
        lambda: [ranks[1].dummy_execution(), ranks[1].forward(prefill(8))],
    )
    assert [[v.group_dummies for v in b.views] for b in backends] == [[1, 0], [1, 0]]
    assert [b.views[0].dp_dummy for b in backends] == [False, True]
