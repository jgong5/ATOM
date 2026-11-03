# SPDX-License-Identifier: MIT
"""Building a `BatchView` from a scheduled batch, and the rung it supplies.

The view is built as the runner builds it, `batch_view(batch,
forward_mode(batch, runner), runner)`. The batches are ATOM's own: a real
`Scheduler` runs with chunked prefill against a real block manager, and the run
is driven long enough that its steps include two-request prefill chunks, middle
chunks with history behind them, and decode. Nothing is hand-built except the
runner stand-in, which exists only to carry a capture ladder and an eager flag.

Two things deliberately not re-derived here, because a re-derivation that
agrees with itself proves nothing: the per-request attention sums are compared
against `Scheduler.compute_detailed_aggregates`, and the rung against the
ladder search inside `ForwardMode.decide`.
"""

from itertools import count
from types import SimpleNamespace
from typing import NamedTuple

import numpy as np
import pytest
from conftest import MockConfig

from atom.compass.backends import BatchView, RequestShape
from atom.compass.backends.shape import (
    sum_context,
    sum_query_context,
    sum_query_square,
)
from atom.compass.runner.overrides import NonAllocatingRunner, RunnerRefusal
from atom.compass.runner.projection import batch_view, forward_mode
from atom.model_engine.scheduler import ScheduledBatchOutput, Scheduler
from atom.model_engine.sequence import Sequence
from atom.sampling_params import SamplingParams

# The two switches that make the scheduler publish its attention aggregates.
# Both are plain attributes -- the public `profile_active`, and an env flag read
# once into `_detailed_annotation_enabled` -- so they are set directly. The
# start-profile RPC flips them too, and also starts a real torch profiler on the
# runner and writes trace files, which a run with no device has no business
# doing.
PUBLISHING = SimpleNamespace(profile_active=True, _detailed_annotation_enabled=True)

CHUNK = 256
PROMPT_LONG = 2048
PROMPT_SHORT = 512
STEPS = 20

# A ladder of the shape capture leaves behind: ascending, with the eager
# fallback 0 at the bottom. Ascending is what `ForwardMode.decide` searches.
LADDER = [0, 1, 2, 4, 8]

#: An attention builder that runs every decode from a graph, as ATOM's base does.
GRAPHABLE = SimpleNamespace(step_needs_eager=lambda batch: False)


class Step(NamedTuple):
    batch: object
    rows: tuple  # the view's rows, built before the step was postprocessed


def runner(ladder=LADDER, enforce_eager=False, dp_size=1):
    """The attributes a rung is read from, and nothing else."""
    return SimpleNamespace(
        capture_sizes_np=np.asarray(ladder, dtype=np.int32),
        enforce_eager=enforce_eager,
        running_tokens_per_dp_rank=np.zeros(dp_size, dtype=np.int32),
        tbo_decode_graphs=False,
        attn_metadata_builder=GRAPHABLE,
        config=SimpleNamespace(
            enable_tbo=False,
            parallel_config=SimpleNamespace(data_parallel_size=dp_size),
        ),
    )


def view(batch, r):
    return batch_view(batch, forward_mode(batch, r), r)


def replayed(batch, r):
    return view(batch, r).capture_rung


def _captured(declared, max_num_seqs, max_num_batched_tokens):
    """A `NonAllocatingRunner` before capture: declared sizes, `[0]` held."""

    class Captured(NonAllocatingRunner):
        enforce_eager = False
        capture_sizes_np = np.asarray([0], dtype=np.int32)
        running_tokens_per_dp_rank = np.zeros(1, dtype=np.int32)
        tbo_decode_graphs = False
        attn_metadata_builder = GRAPHABLE
        config = SimpleNamespace(
            capture_sizes=declared,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_tbo=False,
            enable_tbo_decode=False,
            parallel_config=SimpleNamespace(data_parallel_size=1),
        )

    return Captured()


def _scheduler(chunk=CHUNK, **overrides):
    """ATOM's own scheduler, chunked prefill against a real block manager."""
    Sequence.counter = count()
    return Scheduler(
        MockConfig(
            kv_cache_block_size=16,
            num_kvcache_blocks=1024,
            enable_chunked_prefill=True,
            max_num_seqs=8,
            max_num_batched_tokens=2 * chunk,
            long_prefill_token_threshold=chunk,
            max_model_len=4096,
            **overrides,
        )
    )


def _drive(steps=STEPS, chunk=CHUNK, **overrides):
    """Run ATOM's scheduler and keep every step it produced.

    Two prompts of different lengths, a token budget two chunks wide and a
    per-request chunk cap: both prompts share the first steps, the long one is
    then issued a chunk at a time on its own, and once it is through, every
    later step is a decode of both.
    """
    scheduler = _scheduler(chunk=chunk, **overrides)
    params = SamplingParams(max_tokens=200)
    for length in (PROMPT_LONG, PROMPT_SHORT):
        scheduler.add(Sequence(list(range(1, length + 1)), 16, sampling_params=params))
    trace = []
    for _ in range(steps):
        batch, seqs = scheduler.schedule()
        if not batch.req_ids:
            break
        Scheduler.compute_detailed_aggregates(PUBLISHING, batch, seqs)
        trace.append(Step(batch, view(batch, runner()).requests))
        scheduler.postprocess(
            list(seqs.values()),
            ScheduledBatchOutput(
                req_ids=list(batch.req_ids),
                token_ids=[(7,) for _ in batch.req_ids],
                num_rejected=None,
                num_bonus=None,
                draft_token_ids=None,
            ),
            batch=batch,
        )
    return trace


@pytest.fixture
def trace():
    return _drive()


def _decode(trace):
    for step in trace:
        if step.rows and all(row.decode for row in step.rows):
            return step
    raise AssertionError("the run produced no decode step")


class TestTheRun:
    """What the driven run contains, asserted before anything reads it.

    Empty, one-row and all-prefill are what a run that did nothing degrades to,
    and each would satisfy an assertion made over whatever steps remained. So
    the shape of the run is a test of its own, and the counts here are the ones
    every test below depends on.
    """

    def test_the_run_reached_every_kind_of_step(self, trace):
        multi = [s for s in trace if len(s.batch.req_ids) > 1]
        decode = [s for s in trace if s.batch.total_seqs_num_decode]
        middle = [
            s
            for s in trace
            if any(not r.decode and r.cached_tokens > 0 for r in s.rows)
        ]
        assert len(trace) == STEPS
        assert len(multi) >= 2
        assert len(decode) >= 2
        assert len(middle) >= 2

    def test_one_row_per_scheduled_request_on_every_step(self, trace):
        for step in trace:
            assert len(step.rows) == len(step.batch.req_ids)
            assert len(step.rows) == step.batch.total_seqs_num


class TestTheAttentionSums:
    """Against `Scheduler.compute_detailed_aggregates`, not a re-derivation.

    What the comparison catches is the row set and the counts each row reads
    off the batch: a dropped or duplicated row, or a query or context count
    taken from the wrong field. It cannot catch the `decode` flag, which none
    of the three sums reads; `TestTheRung` holds that, where a step with no
    decode row answers `None`.
    """

    def test_the_three_sums_equal_atom_s_own_on_every_step(self, trace):
        for step in trace:
            assert sum_query_square(step.rows) == step.batch.detailed_sqsq
            assert sum_query_context(step.rows) == step.batch.detailed_sqsk
            assert sum_context(step.rows) == step.batch.detailed_sk

    def test_the_sums_compared_are_neither_zero_nor_absent(self, trace):
        """Zero and `None` are what publishing off, and a run that scheduled
        nothing, produce. Without this the equality above passes on a run in
        which the scheduler published nothing and the projection built
        nothing."""
        for step in trace:
            published = (
                step.batch.detailed_sqsq,
                step.batch.detailed_sqsk,
                step.batch.detailed_sk,
            )
            assert None not in published
            assert min(published) > 0


class TestTheRung:
    """The width comes off the ladder, through the rule ATOM dispatches by."""

    def test_a_decode_step_takes_the_smallest_captured_width_that_holds_it(self, trace):
        step = _decode(trace)
        rung = replayed(step.batch, runner())
        assert rung in LADDER
        assert rung == min(g for g in LADDER if g >= len(step.rows))

    def test_the_rung_is_the_ladder_s_width_and_not_the_batch_s(self, trace):
        """A ladder with no rung at the batch's own width settles which it is.

        With the batch two requests wide and 2 taken out of the ladder, a rung
        read off the batch still answers 2 and a rung read off the ladder
        answers 4."""
        step = _decode(trace)
        assert len(step.rows) == 2
        assert replayed(step.batch, runner(ladder=[0, 1, 4, 8])) == 4

    def test_a_prefill_step_replays_no_graph(self, trace):
        prefill = [s for s in trace if any(not r.decode for r in s.rows)]
        assert len(prefill) >= 2
        for step in prefill:
            assert replayed(step.batch, runner()) is None

    def test_an_eager_runner_replays_no_graph(self, trace):
        step = _decode(trace)
        assert replayed(step.batch, runner(enforce_eager=True)) is None

    def test_a_batch_wider_than_the_widest_capture_replays_no_graph(self, trace):
        step = _decode(trace)
        assert len(step.rows) > 1
        assert replayed(step.batch, runner(ladder=[0, 1])) is None

    def test_the_eager_fallback_ladder_replays_no_graph(self, trace):
        """What a runner that captured nothing holds.

        `capture_sizes` is `[0]` from construction, and a runner that captures
        no graph never widens it, so every step of such a run comes back
        `None` here and the padding term is never charged."""
        for step in trace:
            assert replayed(step.batch, runner(ladder=[0])) is None

    @pytest.mark.parametrize(
        "ladder, wrong_rung",
        [([8, 4, 2, 1, 0], 8), ([0, 1, 8, 4, 2], 8)],
        ids=["descending", "one-pair-out-of-order"],
    )
    def test_a_ladder_that_is_not_ascending_is_refused(self, trace, ladder, wrong_rung):
        """`decide` resolves the rung by binary search, whose precondition is
        an ascending ladder, and this forwards whatever the runner holds.

        The second assertion is why refusing is the answer rather than a
        comment: the same widths out of order answer a rung four times the
        batch's own width, and `BatchView` accepts that -- a rung wider than
        its rows is the one direction rows cannot contradict, so the padding
        would be priced on a graph no capture ever recorded."""
        step = _decode(trace)
        assert sorted(ladder) == sorted(LADDER)
        with pytest.raises(RunnerRefusal, match="not ascending"):
            replayed(step.batch, runner(ladder=ladder))
        unchecked = BatchView(step.rows, capture_rung=wrong_rung)
        assert unchecked.capture_rung == wrong_rung > len(step.rows)

    @pytest.mark.parametrize(
        "declared, max_num_seqs, max_num_batched_tokens",
        [([8, 2, 4, 1], 3, 16384), ([1, 2, 4, 8], 512, 2)],
        ids=["sequence-bound", "token-bound"],
    )
    def test_a_runner_that_captured_nothing_replays_atom_s_ladder(
        self, trace, declared, max_num_seqs, max_num_batched_tokens
    ):
        """`capture_cudagraph` captures nothing and still leaves the ladder
        ATOM's capture would: declared, narrowed to the schedulable width,
        ascending. Left at the construction-time `[0]`, every decode step of a
        graphs-on run came back eager."""
        step = _decode(trace)
        assert len(step.rows) == 2
        r = _captured(declared, max_num_seqs, max_num_batched_tokens)
        assert r.capture_cudagraph() == (0.0, [], 0)
        assert r.capture_sizes_np.tolist() == [1, 2]
        assert replayed(step.batch, r) == 2

    def test_a_ladder_narrowed_to_nothing_is_refused(self):
        """ATOM's capture asserts on an empty narrowed ladder, so such a
        deployment never starts; pricing its decode steps as eager would
        report a run that cannot happen."""
        r = _captured([16, 32], 8, 16384)
        with pytest.raises(RunnerRefusal, match=r"\[16, 32\].*bound 8"):
            r.capture_cudagraph()

    @pytest.mark.parametrize(
        "config",
        [
            None,
            SimpleNamespace(),
            SimpleNamespace(parallel_config=SimpleNamespace()),
            SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_size=0)),
            SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_size=False)),
            SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_size=-1)),
        ],
        ids=[
            "no-config",
            "no-parallel-config",
            "no-data-parallel-size",
            "zero",
            "false",
            "negative",
        ],
    )
    def test_a_runner_that_states_no_parallel_size_is_refused(self, trace, config):
        """The module's only place a default could stand in for an answer.

        Each of the three lookups on the way to `data_parallel_size` is a
        place a differently shaped runner stops, and reading a stop as 1 rank
        prices a group's step as one rank's without saying so. Zero,
        `False` and a negative are not widths either, and each passes the
        `> 1` check in `forward_mode` as the single-rank case."""
        step = _decode(trace)
        bare = SimpleNamespace(
            capture_sizes_np=np.asarray(LADDER, dtype=np.int32),
            enforce_eager=False,
            config=config,
        )
        with pytest.raises(RunnerRefusal, match="states no"):
            replayed(step.batch, bare)


class TestTheWholeProjection:
    def test_a_decode_step_projects_its_rows_and_a_captured_width(self, trace):
        """The padding is the rung's rectangle, and a rung wider than the rows
        is what makes that visible: at a rung equal to the row count the two
        rectangles coincide, and a padding computed off the batch instead of
        off the graph would read the same."""
        step = _decode(trace)
        projected = view(step.batch, runner(ladder=[0, 1, 4, 8]))
        assert projected.requests == step.rows
        assert projected.capture_rung == 4
        widest = max(r.context_tokens for r in step.rows)
        assert projected.graph_padding == 4 * widest - sum_context(step.rows)
        assert projected.graph_padding > len(step.rows) * widest - sum_context(
            step.rows
        )

    def test_a_prefill_step_projects_its_rows_and_no_width(self, trace):
        step = trace[0]
        projected = view(step.batch, runner())
        assert projected.requests == step.rows
        assert projected.capture_rung is None
        assert projected.graph_padding == 0


class TestRowsTheShapesAreNotDefinedOver:
    """Refusals the projection leans on: a row whose shape sums mean nothing,
    and a step with nothing to price."""

    def test_a_row_reading_less_than_it_writes_is_refused(self):
        with pytest.raises(ValueError, match="fewer than"):
            RequestShape(100, 40, False)

    def test_an_empty_batch_has_no_shape_to_price(self):
        """The engine runs no forward for one, so one arriving here is a defect."""
        with pytest.raises(ValueError, match="no requests"):
            BatchView(())
