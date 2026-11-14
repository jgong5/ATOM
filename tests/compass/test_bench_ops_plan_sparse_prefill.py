# SPDX-License-Identifier: MIT
"""`compass measure ops --plan` and `fit ops` over a prefill kernel whose
signature holds sizes its token count sets other than as T, as DeepSeek-V4's
sparse prefill attention's do: an index pointer of T + 1, and metadata index
lists whose lengths sum each token's window. The plan measures it on the
ladder of the steps' token counts, and `fit ops` reads its points as one
family that prices the token counts between them; where its time bends, the
plan measures the steps' token counts beside the bend, and where its rows
move it, each row count at the others' token counts.
The bench and graphs are `test_bench_ops_plan`'s stand-ins.
"""

import json

import pytest
from test_artifact_store import clean_checkout  # noqa: F401  autouse
from test_bench_ops import _graph, _leaf, _t
from test_bench_ops_plan import (  # noqa: F401  a fixture
    CONFIG,
    _bench,
    _graph_at,
    _plan,
    _records,
    _rows_bench,
    bench,
)

from atom.compass.backends.provenance import Refusal
from atom.compass.bench import ops
from atom.compass.capture.forward import decode, prefill
from atom.compass.fit import ops as fit_ops

SPARSE = "aiter.sparse_prefill"
I32 = "torch.int32"
#: The steps' token counts: 45 and 110 lie under 1.25 times the count before.
TOKENS = (40, 45, 60, 100, 110, 150)
#: The tokens each token's window reads back, itself included.
WINDOW = 4


def _window(tokens):
    """The KV entries the windows of `tokens` tokens read, summed."""
    return sum(min(t + 1, WINDOW) for t in range(tokens))


def _sparse_graph_at(store, config, shape, arch=None):
    """`_graph_at`'s graph, a prefill's a GEMM and a sparse attention."""
    if shape.phase == "decode":
        return _graph_at(store, config, shape, arch)
    t = shape.tokens
    gemm = _leaf("aiter.gemm", ("x", _t([t, 8], "activation")))
    sparse = _leaf(
        SPARSE,
        ("q", _t([t, 8], "activation")),
        ("indptr", _t([t + 1], "metadata", dtype=I32)),
        ("indices", _t([_window(t)], "metadata", dtype=I32)),
    )
    return _graph(gemm, sparse, layers=2)


def _sparse_bench(fn, config, todo, llc):
    """`_bench`, the sparse attention one kernel, 10 ns an entry it reads."""
    got = _bench(fn, config, todo, llc)
    for r in got:
        if r["op"] == SPARSE:
            r["kernels"] = ["s"]
            r["cold_ns"] = _window(ops._shape(r["shape"]).tokens) * 10.0
    return got


def _bent_bench(fn, config, todo, llc):
    """`_sparse_bench`, the sparse attention 30% slower at 100 tokens."""
    got = _sparse_bench(fn, config, todo, llc)
    for r in got:
        if r["op"] == SPARSE and r["shape"] == "prefill T=100 B=1":
            r["cold_ns"] *= 1.3
    return got


def _planned(bench, tmp_path, monkeypatch, stand_in, *steps):  # noqa: F811
    """`bench`'s plan over a decode step and `steps`, under `stand_in`."""
    monkeypatch.setattr(ops, "graph", _sparse_graph_at)
    monkeypatch.setattr(ops, "_child", stand_in)
    decoding = ([[1, 1000, True], [1, 3000, True]], 2)
    return _plan(bench, tmp_path, _records(tmp_path / "plan", decoding, *steps))


@pytest.fixture
def planned(bench, tmp_path, monkeypatch):  # noqa: F811
    steps = (([[t, t, False]], None) for t in TOKENS)
    return _planned(bench, tmp_path, monkeypatch, _sparse_bench, *steps)


def test_a_plan_measures_a_sparse_prefill_leaf_on_the_ladder_of_its_token_counts(
    planned,
):
    tokens = {ops._shape(r["shape"]).tokens for r in planned if r["op"] == SPARSE}

    assert sorted(tokens) == ops._ladder(TOKENS) == [40, 60, 100, 150]


def test_fit_ops_reads_a_sparse_prefill_leaf_s_points_as_one_family_over_t(planned):
    params = fit_ops.fit([r for r in planned if "refused" not in r])

    got = [
        (f["dims"][0], sorted(c[0] for c, *_ in f["points"]))
        for f in params["families"].values()
        if f["op"] == SPARSE
    ]
    assert got == [("T", [40, 60, 100, 150])]


def test_coverage_prices_a_sparse_prefill_leaf_between_its_ladder_s_token_counts(
    bench, planned, tmp_path  # noqa: F811
):
    graded = _records(
        tmp_path / "graded", ([[45, 45, False]], None), ([[110, 110, False]], None)
    )
    store, root, _, _ = bench

    report = ops.coverage(store, root, CONFIG, [graded], planned, "gfx942")

    assert report["by phase"] == {"prefill": {"steps": 2, "covered": 2}}


def test_a_plan_measures_a_bent_prefill_leaf_at_the_step_counts_beside_the_bend(
    bench, tmp_path, monkeypatch  # noqa: F811
):
    # 100 reads 30% off the line through 80 and 150, so `fit ops` declines the
    # gaps beside it: 110 is the one step's count inside them
    tokens = (40, 45, 60, 80, 100, 110, 150)
    steps = (([[t, t, False]], None) for t in tokens)
    records = _planned(bench, tmp_path, monkeypatch, _bent_bench, *steps)

    held = {ops._shape(r["shape"]).tokens for r in records if r["op"] == SPARSE}
    assert sorted(held) == [40, 60, 80, 100, 110, 150]
    graded = _records(
        tmp_path / "graded", ([[45, 45, False]], None), ([[110, 110, False]], None)
    )
    store, root, _, _ = bench

    report = ops.coverage(store, root, CONFIG, [graded], records, "gfx942")

    assert report["by phase"] == {"prefill": {"steps": 2, "covered": 2}}


def test_a_plan_fills_each_row_count_of_a_leaf_its_rows_move_at_the_others_counts(
    bench, tmp_path, monkeypatch  # noqa: F811
):
    # the GEMM does not depend on context, and its time grows with its rows,
    # which two rows' 100 tokens beside one row's read: one row's family takes
    # two rows' 60 tokens, so a step of one row between 40 and 60 is bracketed
    steps = [([[t, t, False]], None) for t in (40, 100, 150)]
    steps += [([[t // 2, t // 2, False]] * 2, None) for t in (60, 100)]
    records = _planned(bench, tmp_path, monkeypatch, _rows_bench, *steps)

    gemm = [r for r in records if r["op"] == "aiter.gemm"]
    at = {(ops._shape(r["shape"]).tokens, ops._shape(r["shape"]).seqs) for r in gemm}
    assert (60, 1) in at
    graded = _records(tmp_path / "graded", ([[50, 50, False]], None))
    store, root, _, _ = bench

    report = ops.coverage(store, root, CONFIG, [graded], records, "gfx942")

    assert report["by phase"] == {"prefill": {"steps": 1, "covered": 1}}


def _op(graph, name):
    """The leaf `name` of `graph`."""
    return next(
        op for op, _ in ops.leaves(graph["region"]).values() if op["op"] == name
    )


def test_a_price_list_fitted_before_prefill_leaves_read_t_refuses_them_by_name(
    planned,
):
    params = fit_ops.fit([r for r in planned if "refused" not in r])
    old = fit_ops.PriceList({k: v for k, v in params.items() if k != "tokens"})
    sparse = _op(_sparse_graph_at(None, None, prefill(45, 1)), SPARSE)
    gemm = _op(_graph_at(None, None, decode(2)), "aiter.gemm")
    asked = (sparse, ("prefill", 45, 1, None, 0, 0), None)
    decoded = (gemm, ("decode", 2, 2, None, 3000, 3000), None)

    assert not isinstance(fit_ops.PriceList(params).price(asked), Refusal)
    refused = old.price(asked)
    assert isinstance(refused, Refusal) and refused.reason.endswith("refit it")
    assert not isinstance(old.price(decoded), Refusal)


def test_a_masked_metadata_size_reads_no_rows_where_they_do_not_move_the_kind():
    def point(shape, n, context=None):
        x = {"shape": [n, 8], "stride": [8, 1], "dtype": "torch.float32"}
        idx = {"shape": [_window(n)], "stride": [1], "dtype": I32, "role": "metadata"}
        leaf = {"op": "aiter.k", "args": {"x": x | {"role": "activation"}, "i": idx}}
        record = {"shape": shape, "context": context, "op": "aiter.k"}
        return record | {"leaf": json.dumps(leaf), "cold_ns": n * 1e3, "kernels": ["k"]}

    # flat over a decode's context, measured at one row count per token count
    points = [point("decode B=2", 2, c) for c in (1000, 3000)]
    points += [point(f"prefill T={t} B={b}", t) for t, b in ((40, 1), (60, 2))]
    points += [point("prefill T=100 B=1", 100)]

    got = [
        f["dims"]
        for name, f in fit_ops.fit(points)["families"].items()
        if name.startswith('["prefill"')
    ]
    assert got == [["T"]]
