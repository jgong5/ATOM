# SPDX-License-Identifier: MIT
"""A JIT sweep's points, planned from a cell's engine files and workload, and
their coverage read off a step record.

The coverage tests send a plan's requests through ATOM's real `Scheduler` and
the simulated runner's `forward`, as `test_parity_record.py` drives them, and
read the record those write with `parity.read`.
"""

import itertools
import json
import queue

import pytest
from test_parity_record import _config, _Runner

from atom.compass.drive import sweep_plan
from atom.compass.drive.sweep_plan import Point, covered, plan
from atom.compass.parity import ENV, read
from atom.model_engine.scheduler import Scheduler
from atom.model_engine.sequence import Sequence
from atom.sampling_params import SamplingParams

pytestmark = pytest.mark.usefixtures("unpriced")

SAMPLING = {"ignore_eos": True, "temperature": 0}


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


def _by_rank(points):
    return {r: [p for p in points if p.rank == r] for r in {p.rank for p in points}}


def test_each_rank_runs_every_count_once_and_its_compositions_cover_each_axis(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(sweep_plan, "SPACING", 8)
    engine = _engine(enable_tbo=True)
    small = engine | {"num_kvcache_blocks": 9}
    points = plan(_write(tmp_path, engine, small), _workload(3000))
    ranks = _by_rank(points)
    assert sorted(ranks) == [0, 1]
    # dp1's 9 KV blocks hold only some steps, and it plans those alone.
    counts = [sum(q for q, _ in p.rows) for p in ranks[1]]
    assert counts == sorted(set(counts)) and 0 < len(counts) < 512
    for point in ranks[1]:
        assert sum(-(-(c + 2) // 16) for _, c in point.rows) + point.held <= 9
    mine = ranks[0]
    assert [sum(q for q, _ in p.rows) for p in mine] == list(range(1, 513))
    assert {p.held for p in mine} == set(range(16))
    fresh = [p for p in mine if p.cached == ("",) * len(p.rows)]
    # The second row's offset, the first row's length, takes every residue.
    assert {p.rows[0][0] % 16 for p in fresh if p.rows[0][0] <= 16} == set(range(16))
    # Under two-batch overlap every fresh step past one token has a request
    # boundary to split on.
    assert all(len(p.rows) == 2 for p in fresh[1:])
    contexts = {}
    for p in mine:
        if p.cached != ("",) * len(p.rows):
            ((q, c),) = p.rows
            contexts.setdefault(p.cached[0], []).append(c - q)
    assert sorted(contexts) == ["chunk", "prefix", "turn"]
    # Chunks of the whole budget, as deep as the longest prompt holds.
    assert {c % 512 for c in contexts["chunk"]} == {0}
    assert max(c for p in mine for _, c in p.rows) <= 3000
    assert max(contexts["chunk"]) == 5 * 512
    assert {c % 16 for c in contexts["prefix"] + contexts["turn"]} == {0}
    assert all(p.body == SAMPLING | {"max_tokens": 2} for p in points)


def test_no_point_leaves_the_engine_or_the_workload(tmp_path):
    engine = _engine(data_parallel_size=1, enable_prefix_caching=False, max_num_seqs=1)
    points = plan(_write(tmp_path, engine), _workload(100))
    # One sequence: one row, no held request; prompts no longer than 100.
    assert [p.rows for p in points[:3]] == [((1, 1),), ((2, 2),), ((3, 3),)]
    assert all(len(p.rows) == 1 and p.held == 0 for p in points)
    assert [sum(q for q, _ in p.rows) for p in points] == list(range(1, 101))
    # More sequences: a count past the longest prompt runs as several rows.
    _write(tmp_path, engine | {"max_num_seqs": 64})
    points = plan(tmp_path, _workload(100))
    assert len(points) == 512
    assert max(c for p in points for _, c in p.rows) == 100
    assert {len(p.rows) for p in points} == set(range(1, 7))
    # Without DP attention a rank's budget is its share of the group's.
    no_dpa = _engine(enable_dp_attention=False)
    assert len(plan(_write(tmp_path, no_dpa, no_dpa), _workload(3000))) == 2 * 256


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


def test_a_line_covers_a_point_by_its_rank_and_prefill_rows_in_order():
    a, b = (
        Point(0, ((3, 3), (5, 5)), ("", ""), 0, {}),
        Point(1, ((8, 40),), ("chunk",), 2, {}),
    )
    line = {"rows": [[3, 3, False], [5, 5, False], [1, 9, True]], "is_dummy_run": False}
    swapped = line | {"rows": [[5, 5, False], [3, 3, False]]}
    dummy = {"rows": [[8, 40, False]], "is_dummy_run": True}
    assert covered([a, b], {0: [line], 1: [dummy]}) == ([a], [b])
    assert covered([a, b], {0: [swapped], 1: [line]}) == ([], [a, b])


def _send(directory, monkeypatch, points=None):
    """Send each point's requests, as its fields say, to one rank's scheduler
    and simulated runner recording in `directory`. With no `points`, plan
    them once one request's steps have written the engine file."""
    monkeypatch.setenv(ENV, str(directory))
    config = _config(budget=64, enable_prefix_caching=True)
    config.max_num_seqs = 32
    scheduler, runner = Scheduler(config), _Runner(config)
    tokens = itertools.count(10)

    def fresh(n):
        return [next(tokens) for _ in range(n)]

    def add(prompts, max_tokens=2):
        params = SamplingParams(max_tokens=max_tokens, ignore_eos=True)
        sequences = [Sequence(p, 16, sampling_params=params) for p in prompts]
        for sequence in sequences:
            scheduler.add(sequence)
        return sequences

    def step():
        batch, seqs = scheduler.schedule()
        reply = runner.forward(batch)
        scheduler.postprocess(
            list(seqs.values()), reply, stream_output_queue=queue.Queue(), batch=batch
        )

    def run(prompts, max_tokens=2):
        sequences = add(prompts, max_tokens)
        while scheduler.has_unfinished_requests():
            step()
        return sequences

    if points is None:
        run([fresh(1)])
        points = plan(directory, _workload(300))
    for point in points:
        prompts = []
        for (q, c), how in zip(point.rows, point.cached):
            if how == "prefix":
                shared = fresh(c - q)
                run([shared], max_tokens=1)
                prompts.append(shared + fresh(q))
            elif how == "turn":
                (first,) = run([fresh(c - q)])
                tokens_so_far = [*first.prompt_token_ids, *first.completion_token_ids]
                prompts.append(tokens_so_far + fresh(q - 2))
            else:
                prompts.append(fresh(c))
        # Held: prefilled in a step of their own, decoded after the point's.
        if add([fresh(1) for _ in range(point.held)]):
            step()
        run(prompts)
    return points


def test_a_simulated_run_sent_a_plans_requests_covers_each_point(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep_plan, "SPACING", 8)
    points = _send(tmp_path / "all", monkeypatch)
    kinds = {how for p in points for how in p.cached}
    assert kinds == {"", "chunk", "prefix", "turn"}
    assert covered(points, read(tmp_path / "all")) == (points, [])

    # The same plan less one chunk's final step: that point, and only it, is missing.
    dropped = next(p for p in points if p.cached == ("chunk",))
    rest = [p for p in points if p is not dropped]
    _send(tmp_path / "less", monkeypatch, rest)
    assert covered(points, read(tmp_path / "less")) == (rest, [dropped])
