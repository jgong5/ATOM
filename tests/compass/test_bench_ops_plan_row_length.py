# SPDX-License-Identifier: MIT
"""`compass measure ops --plan` and `--coverage` over a prefill leaf whose
kernels switch at a fixed row length, its tokens and cached tokens summed, as
DeepSeek-V4's `indexer_score_topk` does: such a switch crosses each line of
the leaf's points at its own place, along T at a cached-token count and along
cached tokens at a T. Another of its kernels switches at a token count, as a
tile does. The plan ends, and `fit ops` prices what it measured.
The bench and graphs are `test_bench_ops_plan`'s stand-ins.
"""

import json
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

#: The row lengths the stand-in attention's first kernel switches at, and the
#: token count its second switches at, whatever its cached tokens.
SWITCHES, TILE = (2500, 6200), 120
#: The steps' token counts and cached tokens, as the plan ladders them.
TOKENS, CACHED = {40, 100, 150, 400, 1000}, {0, 2000, 3000, 6000, 8000}
#: Points a plan measures at most before the test calls it endless: one that
#: halves a switch along T at each cached count, and over cached tokens at
#: each T, opens new pairs on the other line every round.
ENDLESS = 2000


def _row_length_bench(fn, config, todo, llc):
    """`_bench` with prefill attention's kernels picked by its row's length,
    its tokens and cached tokens summed, and by its tokens, and its time
    growing with its row's length."""
    got = _bench(fn, config, todo, llc)
    for r in got:
        if r["op"] == "aiter.attn" and r["shape"].startswith("prefill"):
            shape = ops._shape(r["shape"])
            n = shape.tokens // shape.seqs + (r["context"] or 0)
            tile = f"t{int(shape.tokens >= TILE)}"
            r["kernels"] = [f"p{sum(n >= k for k in SWITCHES)}", tile]
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

    # Token counts halved only where T alone switches kernels, to 119 | 120 at
    # each of the steps' cached tokens.
    assert TOKENS | {119, 120} <= set(attn)
    assert all(t in TOKENS or 100 < t < 150 for t in attn)
    assert {(attn[119][c][1], attn[120][c][1]) for c in CACHED} == {("t0", "t1")}
    for tokens, line in attn.items():
        cached = sorted(line)
        # Each count holds the steps' ladder of cached tokens.
        assert CACHED <= set(cached)
        # Each switch's two counts are adjacent along cached tokens, at its row length.
        switched = [(a, b) for a, b in pairwise(cached) if line[a] != line[b]]
        assert all(b - a == 1 for a, b in switched), (tokens, switched)
        assert [b + tokens for _, b in switched] == [
            k for k in SWITCHES if 0 < k - tokens <= 8000
        ]
        # A step's count holds them alone: the fill takes them to no other step's.
        others = TOKENS - {tokens} if tokens in TOKENS else ()
        assert not [(t, b) for _, b in switched for t in others if b in attn[t]]


def test_coverage_reads_a_switch_at_a_fixed_row_length_t_first_as_fit_ops_does(
    bench, planned, tmp_path  # noqa: F811
):
    graded = _records(
        tmp_path / "graded",
        ([[140, 2140, False]], None),  # between 125 and 150 at 2000 cached
        ([[140, 2240, False]], None),  # 2100 cached, a count 400 halved at alone
        ([[150, 2495, False]], None),  # at 150, just before its switch at 2350
        ([[140, 2505, False]], None),  # 2365 cached: 125 and 150 read either side
    )
    store, root, _, _ = bench

    report = ops.coverage(store, root, CONFIG, [graded], planned, "gfx942")

    assert report["by phase"] == {"prefill": {"steps": 4, "covered": 3}}
    assert report["left out, by leaf kind"] == {
        "aiter.attn": {"between two kernels": 1}
    }


def test_two_token_counts_are_not_halved_at_a_count_halving_placed_on_their_lines():
    def point(tokens, context):
        leaf = json.dumps({"args": {"q": {"shape": [tokens, 8]}}, "op": "aiter.attn"})
        record = {"shape": f"prefill T={tokens} B=1", "context": context}
        record |= {"op": "aiter.attn", "leaf": leaf, "kernels": [f"t{tokens}"]}
        return record | {"cold_ns": 1e3 * (tokens + (context or 0))}

    # 100 and 200 launch other kernels at every cached count both hold.
    records = [point(t, c) for t in (100, 200) for c in (None, 500, 1000)]
    families = ops._families(records, ops._dependent(records))
    (base,) = {ops._cut(fam)[0] for fam in families}
    # Halving over cached tokens placed 500 at each, as it can at two token
    # counts whose switches lie apart.
    tried = {((base, t), 500) for t in (100, 200)}

    want = ops._densify(families, tried)

    assert {(shape.tokens, context) for shape, context in want} == {
        (150, None),
        (150, 1000),
    }
