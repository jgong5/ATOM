# SPDX-License-Identifier: MIT
"""A simulated DeepSeek-V4.1-Flash DTP2 group schedules its prefill ticks as
ATOM's own code does.

DTP2 serves ``-tp 2 --enable-dp-attention``: two DP ranks of one forward each,
the whole 16384-token budget on each rank, KV blocks of 128 tokens, DSpark with
five drafts accepted at length 3.51, and the prefill delayer at
``ATOM_PREFILL_DECODE_INTERVAL=10`` with its cost gate and DP prefill cost
balancing. Each rank here is a thread holding ATOM's `Scheduler` and
`PrefillDelayer` at those settings, on gloo groups of their own, and a
`CompassModelRunner` on the meta device (`test_runner_prefill_buckets`), whose
capture states V4.1's prefill token-graph buckets. The loop is
`DPEngineCoreProc.busy_loop`'s: the DP state sync, `schedule`, then the
forward, or for an empty batch `dummy_execution` of the scheduler's
`dummy_prefill`. The LP clock moves, as the members settle at their next host
collective, to the latest of the group's step ends.
"""

import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from itertools import pairwise
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from conftest import MockConfig
from test_dp_step_max import Recording
from test_runner_prefill_buckets import CompassModelRunner

import atom.utils.distributed.utils as dist_utils
from atom.compass.backends.step_stub import Coefficients
from atom.compass.runner.overrides import install_cost_backend
from atom.compass.runner.step_output import DeferredTokenStream
from atom.model_engine.prefill_delayer import PrefillDelayer
from atom.model_engine.scheduler import Scheduler
from atom.model_engine.sequence import Sequence
from atom.sampling_params import SamplingParams
from atom.utils import clock
from atom.utils.graph_holders import release_registered_graphs
from tests.compass.held_ports import held_ports

BLOCK, BUDGET, DRAFTS, INTERVAL = 128, 16384, 5, 10
#: Length 3.51 over five drafts, as ATOM's config resolves it.
RATES = [1.0, 1.0, 0.51, 0.0, 0.0]
#: The buckets #3425's real DTP2 start logged as recorded.
DTP2_BUCKETS = (256, 512, 1024, 2048, 3072, 4096, 5120, 6144, 7168)
#: Seconds a step costs, drafts priced, as a step linear law would.
PRICES = Coefficients(
    prefill_step=1e-2,
    prefill_token=5e-5,
    prefill_query_square=0.0,
    prefill_query_cached=0.0,
    decode_step=8e-3,
    decode_request=1e-5,
    decode_context=0.0,
    decode_padding=0.0,
    collective_token_layer=0.0,
    draft_step=1e-3,
)
#: (arrival s, DP rank, prompt tokens, output tokens): both ranks' first prompts
#: together, then each rank's while its peer decodes, which the delayer holds
#: for alignment and then releases.
REQUESTS = (
    (0.10, 0, 3000, 120),
    (0.10, 1, 2500, 90),
    (0.40, 1, 1800, 60),
    (0.62, 0, 900, 80),
    (0.90, 1, 3500, 40),
    (1.30, 0, 1200, 30),
)


class _Runtime:
    """The LP clock both ranks read. It has no owner thread, so a host
    collective that `clock.settle_before_collectives` wraps settles nothing:
    the loop moves the clock itself."""

    owner = None

    def __init__(self):
        self.t = 0.0

    def read_clock(self):
        return self.t


def _spec():
    return SimpleNamespace(
        num_speculative_tokens=DRAFTS,
        method="dspark",
        use_dspark=lambda: True,
        synthetic_acceptance_rates=RATES,
    )


def _runner(rank):
    r = object.__new__(CompassModelRunner)
    r.config = MockConfig(
        pipeline_parallel_size=1,
        parallel_config=SimpleNamespace(data_parallel_size=2, data_parallel_rank=rank),
        max_num_batched_tokens=BUDGET,
        max_num_seqs=512,
        enable_dp_attention=True,
        prefill_context_parallel_size=1,
        disagg_is_decode=False,
        capture_sizes=[*range(1, 33), 48, 64, 128],
        speculative_config=_spec(),
    )
    r.rank, r.block_size, r.enforce_eager = 0, BLOCK, False
    r.capture_sizes_np = np.asarray([0], dtype=np.int32)
    r.running_tokens_per_dp_rank = np.zeros(2, dtype=np.int32)
    r.tbo_decode_graphs = False
    r.attn_metadata_builder = SimpleNamespace(
        step_needs_eager=lambda batch: False, supports_token_graph=True
    )
    r._token_stream = DeferredTokenStream(list(range(100, 200)), rates=RATES, seed=rank)
    install_cost_backend(r, Recording(PRICES))
    r.capture_cudagraph()
    return r


def _scheduler(runner, rank):
    return Scheduler(
        MockConfig(
            kv_cache_block_size=BLOCK,
            num_kvcache_blocks=4096,
            max_model_len=1 << 20,
            max_num_batched_tokens=BUDGET,
            max_num_seqs=512,
            enable_dp_attention=True,
            parallel_config=SimpleNamespace(
                data_parallel_size=2, data_parallel_rank=rank
            ),
            speculative_config=_spec(),
        ),
        prefill_token_graphs=runner.prefill_token_graphs(),
    )


def _prompt(n):
    return [int(t) for t in np.random.default_rng(n).integers(10, 50000, n)]


@pytest.fixture(scope="module")
def run():
    """Each rank's ticks: its clock at the decision, what it ran (``idle``
    where the group had no work), its delayer's prefill costs, cost gate and
    counters; and both runners."""
    rt = _Runtime()
    ticks, runners = {0: [], 1: []}, {}
    ends = [0.0, 0.0]
    barrier = threading.Barrier(2)
    local = threading.local()

    def rank_loop(rank, ports):
        local.group = dist_utils.stateless_init_torch_distributed_process_group(
            "127.0.0.1", ports[0], rank, 2, backend="gloo"
        )
        engine = dist_utils.stateless_init_torch_distributed_process_group(
            "127.0.0.1", ports[1], rank, 2, backend="gloo"
        )
        runners[rank] = runner = _runner(rank)
        scheduler = _scheduler(runner, rank)
        delayer = PrefillDelayer(
            2,
            engine,
            BUDGET,
            prefill_decode_interval=INTERVAL,
            exchange_prefill_costs=scheduler.cost_balance,
            cost_gate=True,
        )
        scheduler.set_prefill_delayer(delayer)
        mine = [r for r in REQUESTS if r[1] == rank]
        added, running = 0, False
        try:
            while True:
                barrier.wait()
                while added < len(mine) and mine[added][0] <= rt.t:
                    _, _, prompt, out = mine[added]
                    params = SamplingParams(max_tokens=out, ignore_eos=True)
                    scheduler.add(
                        Sequence(_prompt(prompt), BLOCK, sampling_params=params)
                    )
                    added += 1
                state = torch.tensor(
                    [int(not scheduler.is_finished()), int(added < len(mine))]
                )
                torch.distributed.all_reduce(
                    state, op=torch.distributed.ReduceOp.MAX, group=engine
                )
                busy, more = state.tolist()
                if not busy and not running:
                    delayer.notify_idle()
                    if not more:
                        return
                    ticks[rank].append({"at": rt.t, "kind": "idle"})
                    barrier.wait()
                    if rank == 0:
                        rt.t = min(r[0] for r in REQUESTS if r[0] > rt.t)
                    continue
                at = rt.t
                result = scheduler.schedule()
                batch = result[0] if result is not None else None
                if batch is not None and len(batch.req_ids):
                    reply = runner.forward(batch)
                    kind = "prefill" if batch.total_seqs_num_prefill else "decode"
                    if batch.total_seqs_num_prefill:
                        delayer.notify_prefill_executed()
                    scheduler.postprocess(list(result[1].values()), reply, batch=batch)
                else:
                    prefill = scheduler.dummy_prefill
                    reply = runner.dummy_execution(prefill)
                    kind = "dummy prefill" if prefill else "dummy decode"
                    if prefill and scheduler.running:
                        scheduler.postprocess((), reply, batch=None)
                ends[rank] = reply.predicted_s
                ticks[rank].append(
                    {
                        "at": at,
                        "kind": kind,
                        "costs": list(delayer.rank_prefill_costs),
                        "gate": (delayer._decode_tick_us, delayer._prefill_ps_per_cost),
                        "holds": (
                            delayer._stat_hold,
                            delayer._stat_hold_decode_interval,
                        ),
                    }
                )
                barrier.wait()
                if rank == 0:
                    rt.t += max(ends)
                running = bool(busy)
        finally:
            dist_utils.stateless_destroy_torch_distributed_process_group(local.group)
            dist_utils.stateless_destroy_torch_distributed_process_group(engine)

    with pytest.MonkeyPatch.context() as mp, held_ports(2) as ports:
        mp.setattr(dist_utils, "_get_default_timeout", lambda _: timedelta(seconds=60))
        mp.setenv("GLOO_SOCKET_IFNAME", "lo")
        mp.delenv("ATOM_TOKEN_GRAPH_SIZES", raising=False)
        mp.setitem(
            sys.modules,
            "aiter.dist.parallel_state",
            SimpleNamespace(
                get_dp_group=lambda: SimpleNamespace(cpu_group=local.group)
            ),
        )
        mp.setattr(clock, "_installed", rt)
        # Token-graph buckets are the process's: only this run's are stated.
        release_registered_graphs()
        try:
            with ThreadPoolExecutor(2) as pool:
                for f in [pool.submit(rank_loop, r, ports) for r in (0, 1)]:
                    f.result()
            yield SimpleNamespace(ticks=ticks, runners=runners)
        finally:
            release_registered_graphs()


def test_the_runner_states_the_buckets_a_real_dtp2_start_records(run):
    """DP attention keeps the whole batch budget on each rank, so every default
    bucket up to 16384 tokens is stated, as #3425's real start logged."""
    for runner in run.runners.values():
        assert runner.prefill_token_graphs() is True
    from atom.utils.region_graph import TokenGraph

    assert TokenGraph.captured_sizes("prefill") == DTP2_BUCKETS


def test_a_rank_without_prefill_keeps_its_peer_company_on_the_dummy_prefill(run):
    """On every tick some rank prefills, the other prefills too or runs the
    group's dummy prefill; it never decodes beside it."""
    pairs = list(zip(run.ticks[0], run.ticks[1]))
    group_prefills = [
        (a["kind"], b["kind"]) for a, b in pairs if "prefill" in (a["kind"], b["kind"])
    ]
    assert ("prefill", "dummy prefill") in group_prefills
    assert ("dummy prefill", "prefill") in group_prefills
    for kinds in group_prefills:
        assert set(kinds) <= {"prefill", "dummy prefill"}, kinds


def test_ten_decode_ticks_follow_each_prefill_while_the_group_decodes(run):
    """A prefill tick arms the interval on both ranks: while the group keeps
    decoding, no rank prefills in the ten ticks after it, and the delayer
    counts those holds."""
    pairs = list(zip(run.ticks[0], run.ticks[1]))
    prefill = [
        i
        for i, (a, b) in enumerate(pairs)
        if a["kind"] == "prefill" or b["kind"] == "prefill"
    ]
    for i, j in pairwise(prefill):
        between = pairs[i + 1 : j]
        if j > i + 1 and all("decode" in (a["kind"], b["kind"]) for a, b in between):
            assert j - i > INTERVAL, (i, j)
    for rank in (0, 1):
        assert run.ticks[rank][-1]["holds"][1] > 0
    assert run.ticks[0][-1]["holds"] == run.ticks[1][-1]["holds"]


def test_each_rank_s_cost_gate_times_the_group_s_steps_on_the_lp_clock(run):
    """Both ranks' gates hold the same step times, read on the LP clock: the
    decode tick lies among the group's decode steps, each the latest of its
    ranks' step ends, not among either rank's own."""
    at = [t["at"] for t in run.ticks[0]]
    kinds = list(
        zip((t["kind"] for t in run.ticks[0]), (t["kind"] for t in run.ticks[1]))
    )
    decodes = [
        (b - a) * 1e6
        for a, b, k in zip(at, at[1:], kinds)
        if "decode" in k and "prefill" not in k and "dummy prefill" not in k
    ]
    gates = [run.ticks[r][-1]["gate"] for r in (0, 1)]
    assert gates[0] == gates[1]
    decode_us, prefill_ps = gates[0]
    assert min(decodes) <= decode_us <= max(decodes)
    assert prefill_ps > 0


def test_every_rank_balances_on_the_same_prefill_costs(run):
    """The ranks' chunk costs ride the delayer's reduce: each tick both see
    one list, and a tick where both ranks prefill carries both costs."""
    both = 0
    for a, b in zip(run.ticks[0], run.ticks[1]):
        if a["kind"] == "idle":
            continue
        assert a["costs"] == b["costs"]
        both += all(c > 0 for c in a["costs"])
    assert both > 0
