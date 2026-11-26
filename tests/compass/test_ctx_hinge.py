# SPDX-License-Identifier: MIT
"""The step linear law's opt-in decode column `ctx_hinge` (#752): a decode
step's rows' context past a knot, Sum max(0, N_KV - K), the knot `K` stated by
the vector as `ctx_knot` and set for a fit by `COMPASS_FIT_CTX_HINGE`.

On #752's cell A (DeepSeek-V4.1, DPA4) the decode law priced rungs 1-2 0.2%
high below 32k tokens of mean row context, 0.4% low at 32k-131k, 0.8% low at
131k-524k and level from 524k: a context slope a knot may bend. The column is
counted by one `step_linear` function the fit and `StepLinearBackend` both
call, after `row_padding` and before the MoE column, `peer_tokens` and
`rowmod`; it is off unless asked for, and a fit with it off writes the law it
wrote before it.

Records are written from a known law on one rank, as in
`test_backend_step_linear_fit`. No device.
"""

import json

import numpy as np
import pytest
from test_backend_step_linear_fit import _record, _view
from test_calib_v1 import _dep

from atom.compass.backends import (
    BatchView,
    RequestShape,
    StepLinearBackend,
    StepLinearLaw,
)
from atom.compass.backends.step_linear import (
    DECODE,
    DECODE_EXTRA,
    EXTRAPOLATED,
    SOURCE,
    ctx_hinge,
    hull_axes,
    row_padding,
)
from atom.compass.backends.step_linear_fit import (
    CTX_HINGE_ENV,
    OPT_IN,
    _columns,
    env_options,
    fill_rungs,
    fit,
    label_records,
    mark_held_out,
)
from atom.compass.calib import columns, deploy

KNOT = 131_072
#: Each rung's law: its context slope bends past `KNOT`.
TRUTH = {
    "2": {"step": 0.021, "context": 1.0e-7, "graph_padding": 4e-8, "ctx_hinge": 2e-8},
    "4": {"step": 0.022, "context": 1.2e-7, "graph_padding": 3e-8, "ctx_hinge": 1.5e-8},
}


def _decode(contexts, rung, **fields):
    rows = tuple(RequestShape(1, n, True) for n in contexts)
    return BatchView(rows, capture_rung=rung, **fields)


def _anywhere(features: int) -> dict:
    """A hull every point is inside: one facet, `0 . x - 1 <= 0`."""
    return {"equal": {}, "facets": [[0.0] * features + [-1.0]]}


def _vector(coefficients: dict, **spec) -> dict:
    free = sum(1 for f in coefficients if f != "step")
    return {"coefficients": coefficients, "hull": _anywhere(free), **spec}


def _counts(rows, rung, knot=KNOT) -> dict:
    """Every column a decode law here reads, of `rows` at `rung` on one rank."""
    view = _view(rows, rung)
    return {
        "step": 1,
        "context": sum(r[1] for r in rows),
        "graph_padding": view.graph_padding,
        "row_padding": row_padding(view.decode),
        "ctx_hinge": ctx_hinge(view.decode, knot),
        "moe_tokens": len(rows),
    }


def _steps(seed, n=240, truth=TRUTH, rungs=(2, 4), top=600_000):
    """`n` decode steps over `rungs`, each of 1 to its rung's rows at 1k to
    `top` tokens of context, below and past `KNOT`, priced by `truth`."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        rung = rungs[i % len(rungs)]
        width = int(rng.integers(1, rung + 1))
        rows = [(1, int(rng.integers(1_000, top)), True) for _ in range(width)]
        x = _counts(rows, rung)
        out.append((rows, rung, sum(c * x[f] for f, c in truth[str(rung)].items())))
    return out


def _fit(records, **options):
    samples = label_records(records)[0]
    options = {"form": "base", "dp": 1, **options}
    return samples, options, fit(samples, options)


# -- the column -----------------------------------------------------------------


def test_ctx_hinge_is_the_rows_context_past_the_knot():
    view = _decode([100_000, KNOT, 200_000, 500_000], 4)
    assert ctx_hinge(view.decode, KNOT) == (200_000 - KNOT) + (500_000 - KNOT)
    assert ctx_hinge(_decode([1_000, KNOT], 2).decode, KNOT) == 0  # none past it
    assert ctx_hinge((), KNOT) == 0
    # A knot of 1 token counts every row's context but its first token.
    assert ctx_hinge(view.decode, 1) == sum(r.context_tokens - 1 for r in view.decode)


def test_ctx_hinge_follows_row_padding_and_rowmod_stays_last():
    assert DECODE_EXTRA == (
        "row_padding",
        "ctx_hinge",
        "moe_tokens",
        "moe_experts",
        "peer_tokens",
        "rowmod",
    )
    # A continuous hull axis, its variance inflation stated by the fit.
    features = (*DECODE, "ctx_hinge")
    assert features.index("ctx_hinge") in hull_axes(features)
    assert "ctx_hinge" in OPT_IN


# -- the switch -----------------------------------------------------------------


def test_the_switch_is_the_knot_in_tokens_and_unset_empty_or_0_is_off():
    assert CTX_HINGE_ENV == "COMPASS_FIT_CTX_HINGE"
    assert env_options({}) == {}
    assert env_options({CTX_HINGE_ENV: ""}) == {}
    assert env_options({CTX_HINGE_ENV: "0"}) == {}
    assert env_options({CTX_HINGE_ENV: "131072"}) == {"ctx_hinge": KNOT}
    both = {"COMPASS_FIT_ROW_PADDING": "1", CTX_HINGE_ENV: "65536"}
    assert env_options(both) == {"row_padding": True, "ctx_hinge": 65536}


@pytest.mark.parametrize("value", ["-1", "-131072", "1.5", "1e5", "128k", "true"])
def test_a_knot_that_is_not_a_positive_token_count_is_refused(value):
    with pytest.raises(ValueError, match=f"{CTX_HINGE_ENV}={value} is not"):
        env_options({CTX_HINGE_ENV: value})


def test_the_fit_option_takes_a_positive_knot_on_target_decode_alone():
    assert _columns({})["ctx_hinge"] is None
    assert _columns({"ctx_hinge": 0})["ctx_hinge"] is None
    assert _columns({"ctx_hinge": KNOT})["ctx_hinge"] == KNOT
    # The draft law fits none.
    assert _columns({"ctx_hinge": KNOT}, draft=True)["ctx_hinge"] is None
    for knot in (-1, 1.5, "131072", True):
        with pytest.raises(ValueError, match="ctx_hinge's knot"):
            _columns({"ctx_hinge": knot})
    with pytest.raises(ValueError, match="the shared form fits no ctx_hinge column"):
        _columns({"form": "shared", "ctx_hinge": KNOT})


# -- the backend ----------------------------------------------------------------


def _wide() -> dict:
    """A rung-4 decode law over every decode column but `moe_experts`, its
    coefficients stated out of order: the law orders them."""
    coefficients = {
        "rowmod": 2e-9,
        "peer_tokens": 4e-5,
        "moe_tokens": 1.5e-4,
        "ctx_hinge": 2e-8,
        "row_padding": 5e-8,
        "graph_padding": 3e-8,
        "context": 1.2e-7,
        "step": 0.021,
    }
    return {
        "provenance": "the test's",
        "decode": {"4": _vector(coefficients, ctx_knot=KNOT, rowmod=2)},
    }


def test_a_law_with_a_ctx_hinge_vector_round_trips_and_prices_the_term():
    law = StepLinearLaw(json.loads(json.dumps(_wide())))
    vector = law.decode[4]
    assert vector.features == (
        "step",
        "context",
        "graph_padding",
        "row_padding",
        "ctx_hinge",
        "moe_tokens",
        "peer_tokens",
        "rowmod",
    )
    assert vector.ctx_knot == KNOT
    backend = StepLinearBackend(law, 4)
    view = _decode([90_000, 140_000, 600_000], 4, moe_rows=16, moe_tokens=9)
    cost = backend.price(view)
    assert [t.name for t in cost.terms] == [f"decode.{f}" for f in vector.features]
    terms = {t.name: t.seconds for t in cost.terms}
    past = (140_000 - KNOT) + (600_000 - KNOT)
    assert terms["decode.ctx_hinge"] == pytest.approx(2e-8 * past)
    assert terms["decode.row_padding"] == pytest.approx(5e-8 * row_padding(view.decode))
    assert terms["decode.peer_tokens"] == pytest.approx(4e-5 * (9 - 3))
    # The same law without the column prices the step less the term alone.
    plain = _wide()
    del plain["decode"]["4"]["coefficients"]["ctx_hinge"]
    del plain["decode"]["4"]["ctx_knot"]
    plain["decode"]["4"]["hull"] = _anywhere(6)
    without = StepLinearBackend(StepLinearLaw(plain), 4).price(view)
    assert cost.seconds - without.seconds == pytest.approx(2e-8 * past)
    # Rows no further than the knot add none.
    below = _decode([90_000, KNOT], 4, moe_rows=16, moe_tokens=9)
    assert {t.name: t.seconds for t in backend.price(below).terms}[
        "decode.ctx_hinge"
    ] == 0.0


@pytest.mark.parametrize("knot", [None, 0, -1, 1.5, "131072", True])
def test_a_ctx_hinge_vector_states_its_knot_a_positive_token_count(knot):
    law = _wide()
    if knot is None:
        del law["decode"]["4"]["ctx_knot"]
    else:
        law["decode"]["4"]["ctx_knot"] = knot
    with pytest.raises(ValueError, match="ctx_knot"):
        StepLinearLaw(law)


# -- the fit --------------------------------------------------------------------


def test_ctx_hinge_is_fitted_when_asked_and_each_vector_states_the_knot(tmp_path):
    steps = _steps(0)
    _, _, (law, _) = _fit([_record(tmp_path / "dp0.jsonl", steps)], ctx_hinge=KNOT)
    for rung, truth in TRUTH.items():
        vector = law["decode"][rung]
        assert list(vector["coefficients"]) == list(truth)
        assert vector["coefficients"] == pytest.approx(truth, rel=1e-5)
        assert vector["ctx_knot"] == KNOT
        assert set(vector["fit"]["vif"]) == {"ctx_hinge"}
    loaded = StepLinearLaw(json.loads(json.dumps(law)))
    assert loaded.decode[2].features == (*DECODE, "ctx_hinge")
    backend = StepLinearBackend(loaded)
    for rows, rung, seconds in steps[:40]:
        cost = backend.price(_view(rows, rung))
        assert cost.seconds == pytest.approx(seconds, rel=1e-5)
        hinge = {t.name: t.seconds for t in cost.terms}["decode.ctx_hinge"]
        coefficient = TRUTH[str(rung)]["ctx_hinge"]
        assert hinge == pytest.approx(coefficient * _counts(rows, rung)["ctx_hinge"])
    # Past every fitted row the column's hull axis says so.
    far = backend.price(_view([(1, 900_000, True)] * 2, 2))
    assert {t.provenance.source for t in far.terms} == {EXTRAPOLATED}


def test_a_rung_whose_rows_never_pass_the_knot_holds_the_column_at_0(tmp_path):
    """Its coefficient is 0, named, its hull holds the column at 0, and a
    step past the knot prices extrapolated."""
    truth = {
        r: {f: c for f, c in v.items() if f != "ctx_hinge"} for r, v in TRUTH.items()
    }
    steps = _steps(1, truth=truth, top=KNOT)
    _, _, (law, _) = _fit([_record(tmp_path / "dp0.jsonl", steps)], ctx_hinge=KNOT)
    vector = law["decode"]["2"]
    assert vector["coefficients"]["ctx_hinge"] == 0.0
    assert "ctx_hinge" in vector["fit"]["zero"]
    assert vector["hull"]["equal"] == {"ctx_hinge": 0}
    backend = StepLinearBackend(StepLinearLaw(law))
    inside = backend.price(_view([(1, 50_000, True)], 2))
    assert {t.provenance.source for t in inside.terms} == {SOURCE}
    past = backend.price(_view([(1, 50_000, True), (1, KNOT + 1, True)], 2))
    assert {t.provenance.source for t in past.terms} == {EXTRAPOLATED}


def test_the_column_s_order_is_the_backend_s_beside_row_padding_and_a_moe_column(
    tmp_path,
):
    truth = {
        "4": {
            "step": 0.022,
            "context": 1.2e-7,
            "graph_padding": 3e-8,
            "row_padding": 5e-8,
            "ctx_hinge": 1.5e-8,
            "moe_tokens": 1.5e-3,
        }
    }
    steps = _steps(2, 200, truth, rungs=(4,))
    _, _, (law, _) = _fit(
        [_record(tmp_path / "dp0.jsonl", steps)],
        row_padding=True,
        ctx_hinge=KNOT,
        moe_routing=(64, 4),
        moe_select="insample",
    )
    vector = law["decode"]["4"]
    assert list(vector["coefficients"]) == list(truth["4"])
    assert vector["coefficients"] == pytest.approx(truth["4"], rel=1e-5)
    loaded = StepLinearLaw(law)
    assert list(loaded.decode[4].features) == list(truth["4"])
    backend = StepLinearBackend(loaded)
    for rows, rung, seconds in steps[:40]:
        assert backend.price(_view(rows, rung)).seconds == pytest.approx(
            seconds, rel=1e-5
        )


def test_held_out_folds_and_filled_rungs_carry_the_knot(tmp_path):
    records = [_record(tmp_path / f"dp0-{s}.jsonl", _steps(s)) for s in (3, 4)]
    samples, options, (law, report) = _fit(records, ctx_hinge=KNOT)
    mark_held_out(law, report, samples, options)
    for figures in law["figures"].values():
        assert figures["decode"]["steps"] > 0 and figures["decode"]["refused"] == 0
    for vector in law["decode"].values():
        assert vector["held_out"]["bands"]
    fill_rungs(law, [1, 2, 3, 4, 8])
    for rung in ("1", "3", "8"):
        assert law["decode"][rung]["ctx_knot"] == KNOT
    backend = StepLinearBackend(StepLinearLaw(law))
    rows = [(1, 300_000, True), (1, 20_000, True)]
    terms = {t.name: t.seconds for t in backend.price(_view(rows, 3)).terms}
    three = law["decode"]["3"]["coefficients"]["ctx_hinge"]
    assert terms["decode.ctx_hinge"] == pytest.approx(three * (300_000 - KNOT))


def test_the_shared_form_refuses_the_column(tmp_path):
    record = _record(tmp_path / "dp0.jsonl", _steps(5, 60))
    with pytest.raises(ValueError, match="the shared form fits no ctx_hinge column"):
        _fit([record], form="shared", ctx_hinge=KNOT)


def test_off_the_fit_writes_the_law_it_wrote_before_the_column(tmp_path):
    records = [_record(tmp_path / f"dp0-{s}.jsonl", _steps(s)) for s in (6, 7)]
    samples = label_records(records)[0]

    def written(**options) -> str:
        options = {"form": "base", "dp": 1, **options}
        law, report = fit(samples, options)
        mark_held_out(law, report, samples, options)
        fill_rungs(law, [1, 2, 3, 4, 8])
        return json.dumps([law, report])

    off = written()
    assert written(ctx_hinge=None) == off
    assert written(ctx_hinge=0) == off
    assert written(**env_options({CTX_HINGE_ENV: "0"})) == off
    for vector in json.loads(off)[0]["decode"].values():
        assert list(vector["coefficients"]) == list(DECODE)
        assert "ctx_knot" not in vector


# -- a design's columns ---------------------------------------------------------


def test_a_design_counts_ctx_hinge_as_the_backend_prices_it_at_its_config_s_knot(
    monkeypatch,
):
    """A config lists `ctx_hinge` beside its knot, `columns.ctx_knot`. Its
    deployment is `test_calib_v1`'s, whose small designs loosen #739's
    criteria (`deploy.LOOSEN_ENV`)."""
    monkeypatch.setenv(deploy.LOOSEN_ENV, "1")
    assert columns.backend_prices("decode", "ctx_hinge")
    assert columns.provisional("decode", ["step", "context", "ctx_hinge"]) == []
    view = BatchView(
        (RequestShape(2, 100_000, True), RequestShape(2, 300_000, True)),
        capture_rung=2,
        produces_output=True,
        dp_rank=0,
    )
    names = ["step", "context", "graph_padding", "ctx_hinge"]
    got = columns.values(view, "decode", names, _dep(), ctx_knot=KNOT)
    assert got == (1.0, 400_000.0, float(view.graph_padding), 300_000.0 - KNOT)
    with pytest.raises(ValueError, match="columns.ctx_knot"):
        columns.values(view, "decode", names, _dep())
    # A config listing no ctx_hinge counts as it did, whatever its knot.
    base = names[:3]
    assert columns.values(view, "decode", base, _dep()) == got[:3]
