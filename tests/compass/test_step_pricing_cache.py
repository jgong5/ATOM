# SPDX-License-Identifier: MIT
"""A step's pricing reuses what earlier steps computed from the same inputs, and charges what it charges without.

`StepLinearBackend` keeps each term it priced by its feature's count and the
hulls' verdicts on the step's point, `HostDeviceRecurrence` each forward's
segment seconds by what they are priced from, and a cost line each term's
text; a DP group on shared memory runs its pricing collectives without the
tensors around them. Each is compared with the same steps priced with nothing
kept, or on gloo. ATOM's own scheduler drives the steps, and the simulated
runner's `forward` records and prices them.
"""

import json
import queue

import numpy as np
import pytest
import torch.distributed as dist
from test_dp_step_max import decode, dp2, runner  # noqa: F401  the dp2 fixture
from test_parity_record import BLOCK, _config, _Runner

from atom.compass.backends import step_linear
from atom.compass.backends.launch_features import LaunchFeatures
from atom.compass.backends.shape import BatchView, RequestShape
from atom.compass.backends.step_linear import StepLinearBackend, StepLinearLaw
from atom.compass.parity import ENV
from atom.compass.runner import projection, shm_collectives
from atom.compass.runner.overrides import (
    HOST_SEGMENTS,
    OUT_SEGMENT,
    HostDeviceRecurrence,
    install_cost_backend,
)
from atom.model_engine.scheduler import Scheduler
from atom.model_engine.sequence import Sequence
from atom.sampling_params import SamplingParams


def box(*highs):
    """The hull `0 <= x_i <= highs[i]` over as many features."""
    n = len(highs)
    low = [[-int(i == j) for j in range(n)] + [0] for i in range(n)]
    high = [[int(i == j) for j in range(n)] + [-h] for i, h in enumerate(highs)]
    return {"facets": low + high}


def triangle(limit):
    """Decode's (context, graph_padding) hull: both at least 0, their sum, the
    rung times the widest context, at most `limit`."""
    return {"facets": [[-1, 0, 0], [0, -1, 0], [1, 1, -limit]]}


def vectors(scale):
    """A prefill vector and a decode vector per rung, the eager one's included,
    at `scale`; rung 8's with a held-out hull. The driven streak's chunks and
    growing contexts leave each hull mid-way."""
    decode = {
        str(rung): {
            "coefficients": {
                "step": 4e-3 * scale,
                "context": 2e-7 * rung * scale,
                "graph_padding": 1e-8 * scale,
            },
            "hull": triangle(rung * 45),
        }
        for rung in (1, 2, 4, 8)
    }
    decode["8"]["held_out"] = {"hull": triangle(8 * 40)}
    decode["eager"] = {
        "coefficients": {
            "step": 4e-3 * scale,
            "context": 2e-7 * scale,
            "graph_padding": 0.0,
        },
        "hull": {"equal": {"graph_padding": 0}, "facets": [[-1, 0], [1, -45]]},
    }
    prefill = {
        "coefficients": {
            "step": 2e-3 * scale,
            "tokens": 3e-5 * scale,
            "query_square": 1e-9 * scale,
            "query_cached": 2e-9 * scale,
        },
        "hull": box(48, 2304, 48 * 400),
    }
    return {"prefill": prefill, "decode": decode}


LAW = {
    "provenance": "this module's law",
    **vectors(1.0),
    "draft": {
        "provenance": "this module's draft law",
        **vectors(0.25),
        "draft_kv": {"coefficients": {"step": 1e-4, "seqs": 2e-5}, "hull": box(4)},
    },
}
#: Each forward kind's host segments, on (1, tokens, seqs), each kind's apart.
HOST = {
    kind: {
        s: [1e-4 * (i + k), 2e-7 * k, 3e-6]
        for i, s in enumerate((*HOST_SEGMENTS, OUT_SEGMENT))
    }
    for k, kind in enumerate(("decode", "prefill", "middle"), 1)
}


class Deepest(LaunchFeatures):
    """A launch feature a forward's tokens and rows do not settle: the
    deepest row's context in whole 32-token spans, which a streak's decodes
    pass at batch sizes they met before."""

    features = ("deepest",)

    def predict(self, view, graphs):
        return {"deepest": float(max(r.context_tokens for r in view.requests) // 32)}


#: Each kind's `E` and `W_launch` adopt `Deepest`'s column.
COLUMNS = {
    kind: {"E": {"deepest": 1e-4}, "W_launch": {"deepest": 3e-4}}
    for kind in ("decode", "prefill", "middle")
}


def forget(backend, runner):
    """Empty what `backend`'s vectors, `runner`'s host and its cost lines kept."""
    laws = [backend.law]
    while laws:
        law = laws.pop()
        for value in vars(law).values():
            for v in value.values() if isinstance(value, dict) else [value]:
                if isinstance(v, step_linear._Vector):
                    v.kept.clear()
                    v.verdicts.clear()
                elif isinstance(v, StepLinearLaw):
                    laws.append(v)
    runner.compass_host._kept.clear()
    vars(runner).pop("_term_texts", None)


def drive(directory, monkeypatch, kept):
    """ATOM's scheduler over requests that decode past their hulls, each step
    recorded and priced by `LAW` with `HOST`'s host time and `COLUMNS`, with
    what earlier steps kept or with it emptied before each step: each step's
    reply, the cost table's lines, the runner's priced-steps mix and its host."""
    monkeypatch.setenv(ENV, str(directory))
    config = _config(0)
    scheduler, r, clock = Scheduler(config), _Runner(config), [0.0]
    backend = StepLinearBackend(StepLinearLaw(LAW))
    install_cost_backend(r, backend)
    r.compass_host = HostDeviceRecurrence(
        HOST, (), lambda: clock[0], columns=COLUMNS, launch=(Deepest(), None)
    )
    # The first prompt's middle chunk and its last are one budget each, so a
    # middle and a prefill forward run at the same tokens and rows.
    for i in range(16):
        prompt = list(range(5 + 1000 * i, 5 + 1000 * i + (128 if i == 0 else 24)))
        params = SamplingParams(max_tokens=48, ignore_eos=True)
        scheduler.add(Sequence(prompt, BLOCK, sampling_params=params))
    replies = []
    while scheduler.has_unfinished_requests():
        assert len(replies) < 400, "the driven sequence does not terminate"
        scheduled = scheduler.schedule()
        if scheduled is None or scheduled[0] is None or not scheduled[0].req_ids:
            continue
        batch, seqs = scheduled
        if not kept:
            forget(backend, r)
        reply = r.forward(batch)
        clock[0] += reply.predicted_s
        replies.append((reply.predicted_s, reply.out_s, reply.device_end))
        scheduler.postprocess(
            list(seqs.values()), reply, stream_output_queue=queue.Queue(), batch=batch
        )
    mix = r._priced
    summary = (
        mix.steps,
        mix.seconds,
        mix.extrapolated_steps,
        mix.seconds_by_method(),
        mix.steps_by_method(),
        mix.reasons(),
        dict(mix.treatments),
    )
    lines = (directory / "dp0.cost.jsonl").read_text().splitlines()
    return replies, lines, summary, r.compass_host


def test_a_recorded_streak_charges_alike_with_and_without_what_earlier_steps_kept(
    tmp_path, monkeypatch
):
    built = []
    term = step_linear._term
    monkeypatch.setattr(step_linear, "_term", lambda *a: built.append(a) or term(*a))

    replies, lines, mix, host = drive(tmp_path / "kept", monkeypatch, True)
    kept = len(built)
    fresh = drive(tmp_path / "fresh", monkeypatch, False)

    assert len(replies) > 100
    assert replies == fresh[0]
    assert lines == fresh[1]
    assert mix == fresh[2]
    # The streak leaves the hulls: some steps extrapolate, some do not, and
    # rung 8 steps lie on both sides of its held-out hull.
    assert 0 < mix[2] < mix[0]
    assert {validated for _, validated in mix[3]} == {False, True}
    # Every cost line is the JSON of its own object, as one `json.dumps` writes it.
    assert [json.dumps(json.loads(line)) for line in lines] == lines
    # What was kept was taken back: far fewer terms built, and the host's
    # segments priced once for each kind, tokens and rows it met.
    assert kept < (len(built) - kept) / 2
    assert len(host._kept) < len(replies) / 4


def test_a_term_is_taken_back_only_at_the_same_count_and_hull_verdicts():
    backend = StepLinearBackend(StepLinearLaw(LAW))

    def priced(*contexts):
        rows = tuple(RequestShape(1, c, True) for c in contexts)
        return {t.name: t for t in backend.price(BatchView(rows, capture_rung=2)).terms}

    inside, again = priced(20, 20), priced(20, 20)
    deeper = priced(21, 20)
    outside = priced(50, 50)  # 100 past the rung-2 triangle's 90

    assert all(again[n] is t for n, t in inside.items())
    assert deeper["decode.step"] is inside["decode.step"]
    assert deeper["decode.context"] is not inside["decode.context"]
    # The intercept's count is 1 on both sides of the hull, and its source is not.
    assert outside["decode.step"] is not inside["decode.step"]
    assert outside["decode.step"].provenance.source == step_linear.EXTRAPOLATED
    assert inside["decode.step"].provenance.source == step_linear.SOURCE


def test_a_dp_group_on_shared_memory_charges_what_it_charges_on_gloo(
    dp2, monkeypatch  # noqa: F811
):
    monkeypatch.setattr(dist, "all_reduce", dist.all_reduce)
    monkeypatch.setattr(dist, "all_gather", dist.all_gather)
    monkeypatch.setattr(shm_collectives, "_GROUPS", {})
    shm_collectives.install()
    taken = []
    gathered = shm_collectives.gathered

    def counted(*args):
        rows = gathered(*args)
        taken.append(rows is not None)
        return rows

    monkeypatch.setattr(shm_collectives, "gathered", counted)

    def rank(r):
        def priced():
            backend = StepLinearBackend(StepLinearLaw(LAW), 2)
            runner_, clock = runner(backend, dp_size=2, dp_rank=r), [0.0]
            runner_.compass_host = HostDeviceRecurrence(HOST, (), lambda: clock[0])
            replies = []
            for context in range(30 + 7 * r, 60 + 7 * r, 3):
                reply = runner_.forward(decode(context))
                clock[0] += reply.predicted_s
                replies.append(
                    (reply.predicted_s, reply.group_targets, reply.group_out)
                )
            return replies

        on_gloo = priced()
        assert shm_collectives.attach(projection.dp_group())
        return on_gloo, priced()

    (gloo0, shm0), (gloo1, shm1) = dp2(lambda: rank(0), lambda: rank(1))

    assert (shm0, shm1) == (gloo0, gloo1)
    assert all(a[1] == b[1] for a, b in zip(gloo0, gloo1))  # one group target
    # On gloo each rank's step made its dummy count, device end and target
    # calls; on shared memory it took all three off the segment.
    steps = len(gloo0)
    assert taken == [False] * 3 * 2 * steps + [True] * 3 * 2 * steps


@pytest.mark.parametrize(
    "rows, maxed",
    [
        ([[1.0, -2.0], [3.0, -1.0]], [3.0, -1.0]),
        ([[0.0, 1.0], [2.0, 1.0]], None),
        ([[float("nan"), 1.0], [2.0, 1.0]], None),
    ],
    ids=["nonzero", "a zero", "a NaN"],
)
def test_a_float_max_off_the_segment_is_redone_on_gloo_where_gloo_s_order_decides(
    rows, maxed
):
    assert shm_collectives.maxed(np.array(rows)) == maxed


def test_a_host_whose_laws_are_assigned_anew_prices_its_segments_by_them():
    """A record's grading assigns an idle forward's measured `E` as new laws
    (`parity.recurrence.priced`); the segments kept under the old ones go."""
    host = HostDeviceRecurrence(HOST, (), lambda: 0.0)
    before = host.prepare_end(0.0, "decode", (1, 4, 4))[1]["E"]
    host.laws = {**HOST, "decode": {**HOST["decode"], "E": [0.5, 0, 0]}}
    after = host.prepare_end(0.0, "decode", (1, 4, 4))[1]["E"]

    assert (before, after) == (pytest.approx(1e-4 + 8e-7 + 1.2e-5), 0.5)
