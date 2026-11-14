# SPDX-License-Identifier: MIT
"""A batch the bench replayed eagerly is measured so, and its prefill pays no
token-graph `padding` in a fit. ATOM runs a step eagerly when an attention
backend refuses a graph for it, as V4.1's does for rows its packed score
workspace cannot hold in one band, and the step then runs its own rows, not a
prefill bucket's.

The window runs on `test_bench_window`'s planted device, under the
`ForwardMode` ATOM settles for a one-rank prefill whose attention refuses a
graph or not; `measure steps` on `test_bench_measure`'s fake bench, replying
that each prefill ran eagerly.

No device.
"""

import dataclasses
import itertools
from types import SimpleNamespace

import numpy as np
import pytest
from test_bench_measure import Bench as FakeBench
from test_bench_measure import (
    by_view,
    run,  # noqa: F401  the run fixture
)
from test_bench_window import BUSY, MS, Planted, batch
from test_filler_tokens import model_dir  # noqa: F401  the model_dir fixture

from atom.compass.backends import step_linear_fit
from atom.compass.backends.step_linear import token_padding
from atom.compass.bench.window import BenchWindow
from atom.compass.fit.samples import load
from atom.compass.parity import StepRecording
from atom.utils import forward_context
from atom.utils.forward_context import ForwardContext, ForwardMode

#: One prefill token-graph bucket past every prefill here, so a prefill that
#: replayed its graph would pad to it.
BUCKETS = (8192,)


class Bench(BenchWindow, StepRecording, Planted):
    pass


def settled(refused: bool) -> ForwardMode:
    """The mode ATOM settles for a one-rank prefill of 300 tokens whose
    attention backend refuses a graph for it, or not."""
    prefill = SimpleNamespace(
        total_seqs_num=1,
        total_tokens_num=300,
        total_tokens_num_prefill=300,
        final_prefill_requests=lambda: 1,
    )
    return ForwardMode.decide(
        batch=prefill,
        dp_size=1,
        dp_group=None,
        enforce_eager=False,
        step_needs_eager=refused,
        capture_sizes=np.asarray([1, 2, 4], dtype=np.int32),
        is_block_drafter=False,
        tbo_on=False,
        local_tbo=(False, False, 0, 0),
        max_seqlen_q=300,
        running_tokens_per_dp_rank=np.zeros(1, dtype=np.int32),
    )


@pytest.mark.parametrize("refused", [False, True])
def test_the_bench_states_whether_the_mode_its_forwards_settled_runs_eagerly(
    monkeypatch, refused
):
    context = SimpleNamespace(forward_mode=settled(refused))
    monkeypatch.setattr(
        forward_context, "_forward_context", ForwardContext(context=context)
    )
    bench = Bench(itertools.repeat(MS), itertools.repeat(BUSY))

    result = bench.bench_step(batch())

    assert result["group_needs_eager"] is refused
    assert result["repeats"]


def test_a_prefill_the_bench_ran_eagerly_is_measured_so_and_pads_nothing_in_a_fit(
    run, monkeypatch  # noqa: F811
):
    reply = FakeBench.measure

    def eager(self, batch):
        prefill = batch.total_tokens_num_prefill > 0
        return reply(self, batch) | {"group_needs_eager": prefill}

    monkeypatch.setattr(FakeBench, "measure", eager)
    rec = str(run.record)

    store = run(rec, "--sample", "all", *run.named(rec))

    held = by_view(store).values()
    stated = {
        (all(decode for *_, decode in member["rows"]), member.get("group_needs_eager"))
        for _, member in held
    }
    assert stated == {(True, None), (False, True)}
    samples = load({}, [key for key, _ in held], store.root)[0]
    prefills = [s for s in samples if s.view.prefill]
    assert prefills and all(s.view.group_needs_eager for s in prefills)
    rows, _ = step_linear_fit._features(samples, "base", 1, buckets=BUCKETS)
    padding = [counts[-1] for key, counts, *_ in rows if key == "prefill"]
    assert padding == [0] * len(prefills)
    # Each would pad to the bucket had its replay run a graph.
    graphed = [dataclasses.replace(s.view, group_needs_eager=False) for s in prefills]
    assert all(token_padding(view, BUCKETS) > 0 for view in graphed)
