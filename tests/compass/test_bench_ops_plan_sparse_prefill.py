# SPDX-License-Identifier: MIT
"""`compass measure ops --plan` and `fit ops` over a prefill kernel whose
signature holds sizes its token count sets other than as T, as DeepSeek-V4's
sparse prefill attention's do: an index pointer of T + 1, and metadata index
lists whose lengths sum each token's window. The plan measures it on the
ladder of the steps' token counts, and `fit ops` reads its points as one
family that prices the token counts between them.
The bench and graphs are `test_bench_ops_plan`'s stand-ins.
"""

import pytest
from test_artifact_store import clean_checkout  # noqa: F401  autouse
from test_bench_ops import _graph, _leaf, _t
from test_bench_ops_plan import (  # noqa: F401  a fixture
    CONFIG,
    _bench,
    _graph_at,
    _plan,
    _records,
    bench,
)

from atom.compass.bench import ops
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


@pytest.fixture
def planned(bench, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setattr(ops, "graph", _sparse_graph_at)
    monkeypatch.setattr(ops, "_child", _sparse_bench)
    population = _records(
        tmp_path / "plan",
        ([[1, 1000, True], [1, 3000, True]], 2),
        *(([[t, t, False]], None) for t in TOKENS),
    )
    return _plan(bench, tmp_path, population)


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
