# SPDX-License-Identifier: MIT
"""`compass measure ops --plan` and `--coverage` over a prefill leaf whose
kernels switch at a fixed row length, its tokens and cached tokens summed, as
DeepSeek-V4's `indexer_score_topk` does: such a switch crosses each line of
the leaf's points at its own place, along T at a cached-token count and along
cached tokens at a T. The plan ends, and `fit ops` prices what it measured.
The bench and graphs are `test_bench_ops_plan`'s stand-ins.
"""

from itertools import pairwise

import pytest
from test_artifact_store import clean_checkout  # noqa: F401  autouse
from test_bench_ops_plan import (  # noqa: F401  a fixture
    CONFIG,
    _bench,
    _plan,
    _records,
    bench,
)

from atom.compass.bench import ops

#: The row lengths the stand-in attention switches kernels at.
SWITCHES = (2500, 6200)
#: Points a plan measures at most before the test calls it endless: one that
#: halves a switch along T at each cached count, and over cached tokens at
#: each T, opens new pairs on the other line every round.
ENDLESS = 2000


def _row_length_bench(fn, config, todo, llc):
    """`_bench` with prefill attention's kernel picked by its row's length,
    its tokens and cached tokens summed, and its time growing with it."""
    got = _bench(fn, config, todo, llc)
    for r in got:
        if r["op"] == "aiter.attn" and r["shape"].startswith("prefill"):
            shape = ops._shape(r["shape"])
            n = shape.tokens // shape.seqs + (r["context"] or 0)
            r["kernels"] = [f"p{sum(n >= k for k in SWITCHES)}"]
            r["cold_ns"] = n * 1e3
    _row_length_bench.points += len(got)
    assert _row_length_bench.points < ENDLESS, "the plan does not end"
    return got


@pytest.fixture
def planned(bench, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setattr(ops, "_child", _row_length_bench)
    _row_length_bench.points = 0
    plan = _records(
        tmp_path / "plan",
        ([[1, 1000, True], [1, 3000, True]], 2),
        ([[40, 40, False]], None),
        ([[100, 2100, False]], None),
        ([[150, 3150, False]], None),
        ([[400, 6400, False]], None),
        ([[1000, 9000, False]], None),
    )
    return _plan(bench, tmp_path, plan)


def test_a_plan_over_a_switch_at_a_fixed_row_length_ends_adjacent_at_each_t(planned):
    attn = {}
    for r in planned:
        if r["op"] == "aiter.attn" and r["shape"].startswith("prefill"):
            shape = ops._shape(r["shape"])
            attn.setdefault(shape.tokens, {})[r["context"] or 0] = r["kernels"]

    # The steps' token counts only: none halved where the switch moves.
    assert set(attn) == {40, 100, 150, 400, 1000}
    for tokens, line in attn.items():
        cached = sorted(line)
        # Each count holds the steps' ladder of cached tokens.
        assert {0, 2000, 3000, 6000, 8000} <= set(cached)
        # Each switch's two counts are adjacent along cached tokens, at its row length.
        switched = [(a, b) for a, b in pairwise(cached) if line[a] != line[b]]
        assert all(b - a == 1 for a, b in switched), (tokens, switched)
        assert [b + tokens for _, b in switched] == [
            k for k in SWITCHES if 0 < k - tokens <= 8000
        ]


def test_coverage_reads_a_switch_at_a_fixed_row_length_t_first_as_fit_ops_does(
    bench, planned, tmp_path  # noqa: F811
):
    graded = _records(
        tmp_path / "graded",
        ([[120, 2120, False]], None),  # between 100 and 150 at 2000 cached
        ([[120, 2220, False]], None),  # 2100 cached, a count 400 halved at alone
        ([[100, 2495, False]], None),  # at 100, just before its switch at 2400
        ([[120, 2505, False]], None),  # 2385 cached: 100 and 150 read either side
    )
    store, root, _, _ = bench

    report = ops.coverage(store, root, CONFIG, [graded], planned, "gfx942")

    assert report["by phase"] == {"prefill": {"steps": 4, "covered": 3}}
    assert report["left out, by leaf kind"] == {
        "aiter.attn": {"between two kernels": 1}
    }
