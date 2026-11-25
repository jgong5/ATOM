# SPDX-License-Identifier: MIT
"""The step linear law's opt-in prefill columns `graph_tokens` and
`graph_query_cached` (#3720): `tokens` and `query_cached`, Sum N_Q.(N_KV - N_Q),
on a step that replays a prefill token graph, else 0.

On cell A (DeepSeek-V4.1, DPA4-EP4) the prefilling rank of a token-graph step
paid about 3.5-4 us more per token than an eager chunk, which a law with one
`tokens` slope priced 10-20% low; its dummy peers' pad rows then set the
group's price, 10-18% low at 150k-400k cached tokens. Both columns are counted
by one `step_linear` function that `fit` and `StepLinearBackend` both call;
off unless asked for, a fit without them writes the law it wrote before them.
Records are written from a known law. No device.
"""

import json

import numpy as np
import pytest
from test_backend_step_linear_fit import _engine_file, _record

from atom.compass.backends import (
    BatchView,
    RequestShape,
    StepLinearBackend,
    StepLinearLaw,
)
from atom.compass.backends.step_linear import (
    graph_columns,
    token_bucket,
    token_padding,
)
from atom.compass.backends.step_linear_fit import (
    GRAPH_QUERY_CACHED_ENV,
    GRAPH_TOKENS_ENV,
    env_options,
    fit,
    label_records,
)

BUCKETS = (256, 512, 1024, 2048, 4096)
TRUTH = {
    "step": 0.015,
    "tokens": 8e-6,
    "query_square": 2.5e-11,
    "query_cached": 1.4e-11,
    "padding": 1.2e-5,
    "graph_tokens": 3.5e-6,
    "graph_query_cached": 1e-11,
}


def _view(rows, peak=None, moe=None, **fields):
    shapes = tuple(RequestShape(q, n, False) for q, n in rows)
    return BatchView(shapes, moe_rows=moe, group_max_tokens=peak, **fields)


def test_the_columns_are_tokens_and_query_cached_on_a_token_graph_step():
    rows = [(3000, 303_000), (500, 40_500)]
    qc = 3000 * 300_000 + 500 * 40_000
    graphed = _view(rows)
    assert token_bucket(graphed, BUCKETS) == 4096
    assert graph_columns(graphed, BUCKETS) == (3500, qc)
    # Past the largest bucket, split for two-batch overlap, or refused a graph
    # by its attention backend, the step runs eagerly.
    assert graph_columns(_view([(5000, 305_000)]), BUCKETS) == (0, 0)
    assert graph_columns(_view(rows, tbo_tokens=(1750, 1750)), BUCKETS) == (0, 0)
    assert graph_columns(_view(rows, group_needs_eager=True), BUCKETS) == (0, 0)
    # On a data-parallel rank the group's busiest rank picks the bucket.
    assert graph_columns(_view(rows, peak=3500, moe=9000), BUCKETS) == (3500, qc)
    assert graph_columns(_view(rows, peak=6000, moe=9000), BUCKETS) == (0, 0)
    assert graph_columns(_view(rows, moe=9000), BUCKETS) is None
    # A prompt's first chunk has no cached tokens.
    assert graph_columns(_view([(3000, 3000)]), BUCKETS) == (3000, 0)


def _anywhere(features: int) -> dict:
    """A hull every point is inside: one facet, `0 . x - 1 <= 0`."""
    return {"equal": {}, "facets": [[0.0] * features + [-1.0]]}


def _law(prefill=TRUTH, buckets=BUCKETS) -> dict:
    vector = {"coefficients": prefill, "hull": _anywhere(len(prefill) - 1)}
    if buckets:
        vector["buckets"] = list(buckets)
    decode = {"step": 0.01, "context": 1e-8, "graph_padding": 1e-9}
    return {
        "provenance": "the test's",
        "prefill": vector,
        "decode": {"1": {"coefficients": decode, "hull": _anywhere(2)}},
    }


def test_the_backend_prices_the_columns_by_the_function_the_fit_counts_them_with():
    backend = StepLinearBackend(StepLinearLaw(_law()))
    for rows in ([(3000, 303_000)], [(5000, 305_000)], [(200, 200)]):
        view = _view(rows)
        terms = {t.name: t.seconds for t in backend.price(view).terms}
        tokens, qc = graph_columns(view, BUCKETS)
        assert terms["prefill.graph_tokens"] == pytest.approx(3.5e-6 * tokens)
        assert terms["prefill.graph_query_cached"] == pytest.approx(1e-11 * qc)
    graphed = {t.name: t.seconds for t in backend.price(_view([(3000, 303_000)])).terms}
    assert graphed["prefill.graph_tokens"] == pytest.approx(3.5e-6 * 3000)
    assert graphed["prefill.graph_query_cached"] == pytest.approx(1e-11 * 9e8)
    # A law may carry either column alone.
    alone = {k: v for k, v in TRUTH.items() if k != "graph_query_cached"}
    backend = StepLinearBackend(StepLinearLaw(_law(alone)))
    terms = {t.name for t in backend.price(_view([(3000, 303_000)])).terms}
    assert "prefill.graph_tokens" in terms
    assert "prefill.graph_query_cached" not in terms


def test_a_law_carrying_a_column_without_buckets_is_refused():
    with pytest.raises(ValueError, match="graph_tokens and graph_query_cached and no"):
        StepLinearLaw(
            _law({k: v for k, v in TRUTH.items() if k != "padding"}, buckets=None)
        )


def _price(q, n):
    view = _view([(q, n)])
    tokens, qc = graph_columns(view, BUCKETS)
    features = {
        "step": 1,
        "tokens": q,
        "query_square": q * q,
        "query_cached": q * (n - q),
        "padding": token_padding(view, BUCKETS),
        "graph_tokens": tokens,
        "graph_query_cached": qc,
    }
    return float(sum(c * features[f] for f, c in TRUTH.items()))


def _records(path, n=240, seed=0):
    """`n` one-request prefill steps on one rank, half under a token graph,
    each on a history of 0 to 400k tokens, labelled with `TRUTH`'s price."""
    rng = np.random.default_rng(seed)
    steps = []
    for i in range(n):
        q = int(rng.integers(100, 4096) if i % 2 else rng.integers(4200, 16_000))
        n_kv = q + int(rng.integers(0, 400_000)) * int(rng.integers(0, 2))
        steps.append(([(q, n_kv, False)], None, _price(q, n_kv)))
    path.mkdir()
    record = _record(path / "dp0.jsonl", steps)
    _engine_file(record, {"data_parallel_size": 1, "prefill_buckets": list(BUCKETS)})
    return [record]


def _fit(paths, **options):
    samples = label_records(paths, 1)[0]
    return fit(samples, {"form": "base", "dp": 1, "buckets": BUCKETS, **options})[0]


def test_the_columns_are_fitted_when_asked_and_off_by_default(tmp_path):
    paths = _records(tmp_path / "run")
    law = _fit(paths, graph_tokens=True, graph_query_cached=True)
    vector = law["prefill"]
    assert list(vector["coefficients"]) == list(TRUTH)
    assert vector["coefficients"] == pytest.approx(TRUTH, rel=1e-5)
    backend = StepLinearBackend(StepLinearLaw(law))
    for q, n in ((3000, 303_000), (6000, 306_000), (300, 300)):
        priced = backend.price(_view([(q, n)])).seconds
        assert priced == pytest.approx(_price(q, n), rel=1e-5)
    # Off, the vector keeps its columns and the law is the one the switch
    # left unset writes.
    off = _fit(paths)
    assert not {"graph_tokens", "graph_query_cached"} & set(
        off["prefill"]["coefficients"]
    )
    unset = {"graph_tokens": False, "graph_query_cached": False}
    assert json.dumps(_fit(paths, **unset)) == json.dumps(off)
    # Either alone adds its own column.
    one = _fit(paths, graph_tokens=True)["prefill"]["coefficients"]
    assert "graph_tokens" in one and "graph_query_cached" not in one
    # As every opt-in column does, each states its variance inflation factor.
    assert set(vector["fit"]["vif"]) == {"graph_tokens", "graph_query_cached"}
    assert "vif" not in off["prefill"]["fit"]


def test_a_column_on_records_stating_no_buckets_is_refused(tmp_path):
    paths = _records(tmp_path / "run", n=40)
    samples = label_records(paths, 1)[0]
    for column in ("graph_tokens", "graph_query_cached"):
        with pytest.raises(ValueError, match=f"{column} reads"):
            fit(samples, {"form": "base", "dp": 1, column: True})


def test_a_draft_law_takes_neither_column(tmp_path):
    paths = _records(tmp_path / "run", n=60)
    samples = label_records(paths, 1)[0]
    options = {"form": "base", "dp": 1, "buckets": BUCKETS}
    options |= {"graph_tokens": True, "graph_query_cached": True}
    draft = fit(samples, options, draft=True)[0]
    assert not {"graph_tokens", "graph_query_cached"} & set(
        draft["prefill"]["coefficients"]
    )


def test_the_switches_read_from_the_environment():
    assert GRAPH_TOKENS_ENV == "COMPASS_FIT_GRAPH_TOKENS"
    assert GRAPH_QUERY_CACHED_ENV == "COMPASS_FIT_GRAPH_QUERY_CACHED"
    both = {GRAPH_TOKENS_ENV: "1", GRAPH_QUERY_CACHED_ENV: "1"}
    assert env_options(both) == {"graph_tokens": True, "graph_query_cached": True}
    assert env_options({GRAPH_TOKENS_ENV: "1"}) == {"graph_tokens": True}
    assert env_options({GRAPH_QUERY_CACHED_ENV: "0"}) == {}
    for name in (GRAPH_TOKENS_ENV, GRAPH_QUERY_CACHED_ENV):
        with pytest.raises(ValueError, match=f"{name}=true is not 0 or 1"):
            env_options({name: "true"})
