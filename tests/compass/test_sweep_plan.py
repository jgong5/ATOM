# SPDX-License-Identifier: MIT
"""A JIT sweep's points, planned from a cell's engine files and workload, and
their coverage read off a step record's lines."""

import json

import pytest

from atom.compass.drive.sweep_plan import Point, covered, plan

SAMPLING = {"ignore_eos": True, "temperature": 0}
#: The server's ATOM_TBO_PREFILL_MIN_TOKENS.
TBO_MIN = 512


@pytest.fixture(autouse=True)
def tbo(monkeypatch):
    """ATOM's two-batch overlap rule as the server runs it: split at the token
    midpoint once a rank reaches `TBO_MIN`."""
    monkeypatch.setenv("ATOM_TBO_PREFILL_MIN_TOKENS", str(TBO_MIN))
    monkeypatch.delenv("ATOM_TBO_PREFILL_TOKEN_SPLIT", raising=False)


def _engine(**fields):
    return {
        "ladder": [1, 2, 4],
        "kv_cache_block_size": 16,
        "data_parallel_size": 2,
        "enable_dp_attention": True,
        "enable_tbo": False,
        "enable_prefix_caching": True,
        "max_num_seqs": 64,
        "max_num_batched_tokens": 512,
        "max_model_len": 4096,
        "num_kvcache_blocks": 100000,
    } | fields


def _write(directory, *engines):
    for rank, engine in enumerate(engines):
        (directory / f"dp{rank}.engine.json").write_text(json.dumps(engine))
    return directory


def _workload(longest):
    # The longest prompt is a subagent's, nested in its trace.
    trace = {"requests": [{"in": 40, "out": 9}, {"requests": [{"in": longest}]}]}
    return {"traces": [trace, {"requests": [{"in": 7}]}], "sampling": SAMPLING}


def _tokens(point):
    return sum(q for q, _ in point.rows)


def test_a_walks_every_count_once_on_the_rank_its_count_chooses(tmp_path):
    points = plan(_write(tmp_path, _engine(), _engine()), _workload(300))
    a = [p for p in points if p.split is None]
    assert [_tokens(p) for p in a] == list(range(1, 513))
    assert [(p.rank, p.step) for p in a[:4]] == [(0, 0), (1, 0), (0, 1), (1, 1)]
    # One row and no parked decode up to the longest prompt; past it, rows of
    # at most its length behind one parked decode.
    assert {(len(p.rows), p.held) for p in a[:300]} == {(1, 0)}
    assert {(len(p.rows), p.held) for p in a[300:]} == {(2, 1)}
    assert max(c for p in a for _, c in p.rows) == 300
    assert all(p.body == SAMPLING | {"max_tokens": 2} for p in points)
    # B's steps follow A's rounds, one point a rank.
    b = [p for p in points if p.split is not None]
    assert b[0].step == 256 and [p.rank for p in b[:4]] == [0, 1, 0, 1]
    # Without DP attention a rank's budget is its share of the group's.
    no_dpa = _engine(enable_dp_attention=False)
    points = plan(_write(tmp_path, no_dpa, no_dpa), _workload(300))
    assert max(_tokens(p) for p in points if p.split is None) == 256


def test_b_takes_every_offset_and_parked_count_and_splits_each_tbo_step(tmp_path):
    engine = _engine(enable_tbo=True, max_num_batched_tokens=1024)
    points = plan(_write(tmp_path, engine, engine), _workload(3000))
    two = [p for p in points if p.split is not None and len(p.rows) == 2]
    unsplit = [p for p in two if not p.split]
    # The second row's offset, the first row's length, mod 16, by parked decodes.
    pairs = {(p.rows[0][0] % 16, p.held) for p in unsplit}
    assert pairs == {(o, h) for o in range(16) for h in range(1, 17)}
    assert all(p.rows[0][0] < p.rows[1][0] for p in two)
    assert max(_tokens(p) for p in unsplit) < TBO_MIN
    steps = {}
    for p in two:
        steps.setdefault(p.step, []).append(p)
    assert {tuple(p.rank for p in s) for s in steps.values()} == {(0, 1)}
    # Each split step's first rank reaches the TBO minimum, the other does not.
    split = [s for s in steps.values() if s[0].split]
    assert len(split) == 128
    assert {tuple(_tokens(p) >= TBO_MIN for p in s) for s in split} == {(True, False)}
    # Too small a budget for a split step of two rows plans none.
    small = engine | {"max_num_batched_tokens": TBO_MIN + 16 * 17}
    points = plan(_write(tmp_path, small, small), _workload(3000))
    assert not any(p.split for p in points if len(p.rows) == 2)


def test_steps_split_by_atoms_rule_and_a_leaves_a_rank_idle_where_it_would_split(
    tmp_path, monkeypatch
):
    engine = _engine(enable_tbo=True, max_num_batched_tokens=1024)
    steps = {}
    for p in plan(_write(tmp_path, engine, engine), _workload(3000)):
        steps.setdefault(p.step, []).append(p)
    a = [s for s in steps.values() if s[0].split is None]
    b = [s for s in steps.values() if s[0].split is not None]
    # A round reaching the minimum would split, so from 511 and 512's round on
    # rank 1 has no row, which vetoes the split.
    assert [len(s) for s in a] == [2] * 255 + [1] * 514
    assert [_tokens(s[0]) for s in a[254:257]] == [509, 511, 512]
    # A B step splits when a rank reaches the minimum and every rank has two
    # prefill tokens to cut, a chunked row's step among them.
    tokens = [[_tokens(p) for p in s] for s in b]
    assert [s[0].split for s in b] == [
        len(t) == 2 and min(t) > 1 and max(t) >= TBO_MIN for t in tokens
    ]
    assert any(s[0].split for s in b if s[0].cached == ("chunk",))
    # Split on a request boundary, one row cannot split: A keeps every rank, and
    # only two-row steps split.
    monkeypatch.setenv("ATOM_TBO_PREFILL_TOKEN_SPLIT", "0")
    points = plan(tmp_path, _workload(3000))
    assert max(p.step for p in points if p.split is None) == 511
    assert {len(p.rows) for p in points if p.split} == {2}


def test_b_rows_on_a_context_take_each_query_size_over_chunk_cuts_and_prefixes(
    tmp_path,
):
    engine = _engine(data_parallel_size=1)
    points = plan(_write(tmp_path, engine), _workload(3000))
    contexts = {}
    for p in points:
        if p.cached[0]:
            assert (p.held, p.split, len(p.rows)) == (1, False, 1)
            ((q, c),) = p.rows
            contexts.setdefault(p.cached[0], set()).add((q, c - q))
    # Chunk cuts from the scheduler's alignment, 64 at 16-token blocks, below the
    # budget, then whole budgets, within the longest prompt.
    queries = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    cuts = [64, 128, 256, 512, 1024, 2048]
    want = {(q, c) for q in queries for c in cuts if q + c <= 3000}
    assert contexts["chunk"] == want
    # Prefixes from one KV block, where no chunk row already takes them.
    assert {c for _, c in contexts["prefix"]} == {16, 32}
    no_prefix = engine | {"enable_prefix_caching": False}
    points = plan(_write(tmp_path, no_prefix), _workload(3000))
    assert {p.cached for p in points} == {("",), ("", ""), ("chunk",)}


def test_no_point_leaves_the_engine_or_the_workload(tmp_path):
    engine = _engine(data_parallel_size=1, enable_prefix_caching=False, max_num_seqs=1)
    points = plan(_write(tmp_path, engine), _workload(100))
    # One sequence: A's counts up to the longest prompt, one row each, and no B.
    assert [p.rows for p in points] == [((t, t),) for t in range(1, 101)]
    # Two rows and 15 parked decodes fill 17 sequences.
    _write(tmp_path, engine | {"max_num_seqs": 17})
    assert max(p.held for p in plan(tmp_path, _workload(3000))) == 15
    # A prompt leaves room in the context for its two output tokens.
    _write(tmp_path, engine | {"max_num_seqs": 64, "max_model_len": 50})
    assert max(c for p in plan(tmp_path, _workload(100)) for _, c in p.rows) == 48
    # A drafter's tokens take KV blocks too: one block of 16 holds 14 prompt
    # tokens and the two output tokens, but only 9 with three drafts; a parked
    # decode needs more than one.
    one_block = engine | {"num_kvcache_blocks": 1}
    assert len(plan(_write(tmp_path, one_block), _workload(100))) == 14
    _write(tmp_path, one_block | {"speculative": ["mtp", 3]})
    assert len(plan(tmp_path, _workload(100))) == 9
    # The smallest rank's pool bounds every rank's points.
    two = plan(
        _write(
            tmp_path,
            engine | {"data_parallel_size": 2},
            one_block | {"data_parallel_size": 2},
        ),
        _workload(100),
    )
    assert max(_tokens(p) for p in two) == 14


def test_a_deployment_no_delayer_coalesces_plans_a_alone(tmp_path):
    points = plan(_write(tmp_path, _engine(), _engine()), _workload(300), True)
    assert len(points) == 512 and all(p.split is None for p in points)


@pytest.mark.parametrize(
    "second, refusal",
    [
        (
            {"enable_tbo": True},
            "dp0 and dp1 state two engines, which differ on enable_tbo",
        ),
        (None, "states no engine for DP rank 1 of 2"),
    ],
)
def test_ranks_that_state_two_engines_or_none_are_refused(tmp_path, second, refusal):
    engines = [_engine()] + ([_engine() | second] if second else [])
    with pytest.raises(ValueError, match=refusal):
        plan(_write(tmp_path, *engines), _workload(3000))


def test_an_engine_file_that_states_no_prefix_caching_is_refused(tmp_path):
    engine = _engine(data_parallel_size=1)
    del engine["enable_prefix_caching"]
    with pytest.raises(ValueError, match="dp0.engine.json states no enable_prefix"):
        plan(_write(tmp_path, engine), _workload(3000))


def test_a_line_covers_a_by_its_tokens_unsplit_and_b_by_its_rows_and_split():
    a = Point(1, ((8, 8),), ("",), 0, None, 0, {})
    b = Point(0, ((3, 3), (5, 5)), ("", ""), 2, False, 9, {})
    split = Point(1, ((3, 3), (5, 5)), ("", ""), 2, True, 10, {})
    line = {"rows": [[3, 3, False], [5, 5, False], [1, 9, True]], "is_dummy_run": False}
    # Any rank's unsplit step of 8 prefill tokens holds A's count 8.
    assert covered([a, b, split], {0: [line]}) == ([a, b], [split])
    tbo = line | {"tbo_tokens": [3, 5]}
    assert covered([a, b, split], {1: [tbo]}) == ([split], [a, b])
    swapped = line | {"rows": [[5, 5, False], [3, 3, False]]}
    dummy = line | {"is_dummy_run": True}
    assert covered([a, b, split], {0: [swapped, dummy]}) == ([a], [b, split])
