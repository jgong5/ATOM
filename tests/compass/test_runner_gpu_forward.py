# SPDX-License-Identifier: MIT
"""A simulated step's predicted device seconds in ATOM's GPU forward histograms.

With ATOM's device timer on, TP rank 0 of the Compass runner times each step
`forward` prices through ATOM's `GPUForwardMetrics`, built as `ModelRunner`
builds it, before `forward` replies: the step's `StepCost.target_seconds`, not
its charge, which adds the draft, nor its DP group's target; under DP with a
`HostDeviceRecurrence`, plus its wait from its own target end to the group's
last, which a real rank's device events span; and a prefill request's sum over
its chunks. Requests run through ATOM's own
`Scheduler`, whose `prefill_gpu_requests` names each prefill chunk, and the
runner's own `forward`, with `torch.cuda.Event` refusing, so a device event
fails the test. `compass run` starts its server with the timer on unless set.
"""

import json
from types import SimpleNamespace

import pytest
import torch
from conftest import MockConfig
from prometheus_client import CollectorRegistry
from test_dp_step_max import BLOCK, dp2, prefill, runner  # noqa: F401  dp2
from test_drive_run import (
    PLANTED,
    deployed,
    flags,
    plant,
    recorded,
    refused_summary,
    stub,
)
from test_step_grid_plan import SHORT

from atom.compass.backends.cost import CostTerm, StepCost
from atom.compass.backends.resolver import CostRefused, Refusal
from atom.compass.backends.step_stub import StepStubBackend
from atom.compass.drive import run as drive_run
from atom.compass.drive import serve
from atom.metrics.gpu import GPUForwardMetrics
from atom.model_engine.scheduler import Scheduler
from atom.model_engine.sequence import Sequence
from atom.sampling_params import SamplingParams

TIMER = "ATOM_ENABLE_METRICS_DEVICE_TIMER"
#: Prompts longer than the token budget: ATOM prefills the first in two chunks,
#: the second in three, the last of them beside the third prompt's one.
PROMPTS, BUDGET = (40, 70, 20), 32
#: The price calls refused: the second prompt's middle chunk, and a decode.
REFUSED = (3, 6)
#: The draft seconds `Kept` adds to each price it answers, far above the
#: stub's targets, so a step's charge and its DP group's target are not it.
DRAFT_S = 1.0
#: The planted server, which also writes the timer its environment holds to
#: the file `PLANTED_TIMER` names.
TIMED = (
    "import json, os\n"
    f'json.dump(os.environ.get("{TIMER}"), open(os.environ["PLANTED_TIMER"], "w"))\n'
) + PLANTED


class Kept(StepStubBackend):
    """The stub with a `draft_s` draft term beside its target terms, keeping
    each call's price, None for the calls it refuses."""

    def __init__(self, refuse=(), draft_s=DRAFT_S):
        super().__init__()
        self.refuse, self.draft_s, self.costs = set(refuse), draft_s, []

    def price(self, view):
        if len(self.costs) in self.refuse:
            self.costs.append(None)
            raise CostRefused(view, [Refusal("kept", "refused here")])
        cost = super().price(view)
        draft = CostTerm("draft", self.draft_s, cost.terms[0].provenance)
        self.costs.append(StepCost(cost.terms, cost.treatments, (draft,)))
        return self.costs[-1]


@pytest.fixture
def timer(monkeypatch):
    monkeypatch.setenv(TIMER, "1")

    def device_event(*args, **kwargs):
        raise AssertionError("a device event was created")

    monkeypatch.setattr(torch.cuda, "Event", device_event)


def timed(r):
    """`r` with the `GPUForwardMetrics` `ModelRunner` builds under the timer,
    each histogram's observations also kept in `r.seen`."""
    metrics = GPUForwardMetrics(
        lambda: torch.cuda.Event(enable_timing=True),
        dp_rank=r.config.parallel_config.data_parallel_rank,
        tp_rank=r.rank,
        registry=CollectorRegistry(),
    )
    r.gpu_forward_metrics, r.seen = metrics, {}
    for name in ("steps", "prefill_requests"):
        child, seen = getattr(metrics, name), r.seen.setdefault(name, [])

        def observe(value, observe=child.observe, seen=seen):
            seen.append(value)
            observe(value)

        child.observe = observe
    return r


def count(histogram):
    """`histogram`'s ``_count`` as a scrape reads it now, with no poll."""
    samples = next(iter(histogram.collect())).samples
    return next(s.value for s in samples if s.name.endswith("_count"))


def drive(r):
    """`PROMPTS` through ATOM's scheduler and `r.forward` until each finished:
    each step's batch and reply, and the steps `r` had timed when its
    `forward` returned, None when it times none."""
    scheduler = Scheduler(
        MockConfig(
            num_kvcache_blocks=64,
            kv_cache_block_size=BLOCK,
            max_model_len=256,
            max_num_batched_tokens=BUDGET,
            parallel_config=SimpleNamespace(data_parallel_rank=0),
        )
    )
    for n in PROMPTS:
        params = SamplingParams(max_tokens=4, ignore_eos=True)
        scheduler.add(Sequence(list(range(5, 5 + n)), BLOCK, sampling_params=params))
    steps = []
    while scheduler.has_unfinished_requests():
        assert len(steps) < 100, "the driven requests do not finish"
        batch, seqs = scheduler.schedule()
        reply = r.forward(batch)
        metrics = getattr(r, "gpu_forward_metrics", None)
        steps.append((batch, reply, metrics and count(metrics.steps)))
        scheduler.postprocess(list(seqs.values()), reply, batch=batch)
    return steps


def published(steps, costs):
    """Each request's prefill seconds, the sum of its chunks' target seconds,
    at its last chunk in batch order, when no chunk was refused. A batch's
    prefills follow its decodes."""
    chunks, last = {}, {}
    for k, ((batch, _, _), cost) in enumerate(zip(steps, costs)):
        for req in batch.req_ids[batch.total_seqs_num_decode :]:
            chunks.setdefault(req, []).append(cost)
            last[req] = k
    return [
        sum(c.target_seconds for c in chunks[req])
        for k, (batch, _, _) in enumerate(steps)
        for req in batch.req_ids[batch.total_seqs_num_decode :]
        if last[req] == k and None not in chunks[req]
    ]


@pytest.mark.parametrize("refuse", [(), REFUSED])
def test_rank_0_times_each_priced_step_and_each_prefill_before_it_replies(
    timer, refuse
):
    backend = Kept(refuse)
    r = timed(runner(backend))

    steps = drive(r)

    assert any(len(b.prefill_gpu_requests) > 1 for b, _, _ in steps)
    priced = [c is not None for c in backend.costs]
    assert [n for _, _, n in steps] == [sum(priced[: k + 1]) for k in range(len(steps))]
    targets = [c.target_seconds for c in backend.costs if c is not None]
    # Each priced step is charged its target plus the draft, as its reply says.
    charged = [x.predicted_s for (_, x, _), c in zip(steps, backend.costs) if c]
    assert charged == pytest.approx([t + DRAFT_S for t in targets], rel=1e-9)
    assert r.seen["steps"] == pytest.approx(targets, rel=1e-9)
    sums = published(steps, backend.costs)
    assert len(sums) == len(PROMPTS) - (1 if refuse else 0)
    assert r.seen["prefill_requests"] == pytest.approx(sums, rel=1e-9)


def test_with_the_timer_off_nothing_is_timed_and_every_reply_is_the_same(
    timer, monkeypatch
):
    on = drive(timed(runner(Kept())))
    monkeypatch.setenv(TIMER, "0")
    # `ModelRunner` builds no `GPUForwardMetrics` with the timer off.
    off = drive(runner(Kept()))

    assert all(not batch.prefill_gpu_requests for batch, _, _ in off)
    replies = [[(x.predicted_s, x.token_ids) for _, x, _ in s] for s in (on, off)]
    assert replies[0] == replies[1]


def test_another_tp_rank_and_a_dp_dummy_are_not_timed(timer, dp2):  # noqa: F811
    tp1 = timed(runner(Kept()))
    tp1.rank = 1
    backends = [Kept(), Kept(draft_s=2 * DRAFT_S)]
    busy, idle = (
        timed(runner(b, dp_size=2, dp_rank=r)) for r, b in enumerate(backends)
    )

    drive(tp1)
    reply, _ = dp2(lambda: busy.forward(prefill(40)), idle.dummy_execution)

    assert tp1.seen == idle.seen == {"steps": [], "prefill_requests": []}
    assert len(backends[1].costs) == 1  # the dummy priced, and not timed
    # The busy step leaves at its group's target, the idle dummy's charge.
    assert reply.predicted_s == pytest.approx(backends[1].costs[0].seconds)
    assert busy.seen["steps"] == [backends[0].costs[0].target_seconds]


@pytest.mark.parametrize("joined", [False, True])
@pytest.mark.parametrize("env, seen", [(None, "1"), ("0", "0")])
def test_compass_run_starts_its_server_with_the_timer_on_unless_set(
    tmp_path, monkeypatch, joined, env, seen
):
    summary = refused_summary(tmp_path / "authority", monkeypatch)
    if joined:
        kv = json.dumps({"kv_connector": "compass", "kv_role": "kv_producer"})
        argv = ["--join", str(deployed(tmp_path, monkeypatch, summary))]
        argv += ["--out", str(tmp_path / "prefill"), "--", "--model", str(tmp_path)]
        argv += ["--kv-transfer-config", kv]
    else:
        steps = recorded(tmp_path, monkeypatch, SHORT) / serve.STEPS
        plant(tmp_path, monkeypatch, steps, summary)
        argv = flags(tmp_path, "--step-stub", stub(tmp_path))
    (tmp_path / "planted.py").write_text(TIMED)
    monkeypatch.setenv("PLANTED_TIMER", str(tmp_path / "timer.json"))
    monkeypatch.delenv(TIMER, raising=False)
    if env is not None:
        monkeypatch.setenv(TIMER, env)

    drive_run.main(argv)

    assert json.loads((tmp_path / "timer.json").read_text()) == seen


def test_under_dp_a_host_paced_rank_times_its_wait_for_the_group_s_last_target(
    timer, dp2  # noqa: F811
):
    from atom.compass.runner.overrides import HostDeviceRecurrence

    backends = [Kept(), Kept()]
    ranks = [timed(runner(b, dp_size=2, dp_rank=r)) for r, b in enumerate(backends)]
    for r in ranks:
        r.compass_host = HostDeviceRecurrence({}, (), lambda: 0.0)

    dp2(lambda: ranks[0].forward(prefill(40)), lambda: ranks[1].forward(prefill(8)))

    big, small = (b.costs[0].target_seconds for b in backends)
    assert small < big
    # Both targets start at 0 s: the small rank's events span the big one's end.
    assert ranks[0].seen["steps"] == pytest.approx([big])
    assert ranks[1].seen["steps"] == pytest.approx([big])
