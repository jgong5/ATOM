# SPDX-License-Identifier: MIT
"""The state slots a replayed batch's rows hold, for a model that keeps
per-request state under a drafter.

ATOM's block manager gives a drafting GDN request `pool_entries_per_req`'s
state entry of slots, 1 + K without ReplaySSM, as the workers' `get_num_blocks`
sizes it. Batches come from ATOM's own `Scheduler` serving a GDN model under an
MTP drafter of one draft, every draft accepted, as `test_bench_replay_drafter`
drives it; the bench starts on `test_bench_engine`'s workers; the ops stub runs
beside `test_bench_ops`'s stand-in runner.
"""

import queue
from types import SimpleNamespace

import pytest
import test_bench_engine
from test_bench_engine import manager  # noqa: F401  the manager fixture
from test_bench_ops import KEY, _graph, _leaf, _state, _t
from test_bench_ops import _Runner as OpsRunner
from test_bench_replay_drafter import spec
from test_filler_tokens import model_dir  # noqa: F401  the model_dir fixture
from test_parity_record import BLOCK, _config

from atom.compass import ir
from atom.compass.bench import ops, replay
from atom.compass.capture.forward import decode
from atom.compass.runner.step_output import DeferredTokenStream
from atom.model_engine.scheduler import ScheduledBatchOutput, Scheduler
from atom.model_engine.sequence import Sequence
from atom.sampling_params import SamplingParams
from atom.utils import forward_context as fctx

pytestmark = pytest.mark.usefixtures("unpriced")

#: A GDN model, and the state slots its request holds under MTP1: 1 + 1.
GDN, MTP1 = "qwen3_next", 2
PROMPTS = (23, 40, 90)


def gdn_config(model_dir, per_req=MTP1):  # noqa: F811
    """A GDN model's engine under MTP1, its state pool sized as the workers'
    `get_num_blocks` sizes it: `per_req` slots a request, for 8 requests."""
    built = _config(
        blocks=256,
        speculative_config=spec(1, False),
        pool_entries={"state": 8 * per_req},
        pool_entries_per_req={"state": per_req},
    )
    built.model, built.trust_remote_code = str(model_dir), False
    built.hf_config = SimpleNamespace(model_type=GDN)
    return built


def scheduled(built):
    """Each batch ATOM's scheduler builds to serve `PROMPTS` as a GDN model's
    requests under `built`, with its rows as a step record states them and its
    rows' state slots, read as it is built."""
    scheduler = Scheduler(built)
    stream = DeferredTokenStream([7], rates=[1.0])
    params = SamplingParams(max_tokens=8, ignore_eos=True)
    for i, n in enumerate(PROMPTS):
        tokens = list(range(5 + 1000 * i, 5 + 1000 * i + n))
        scheduler.add(Sequence(tokens, BLOCK, params, has_per_req_cache=True))
    out = []
    while scheduler.has_unfinished_requests():
        assert len(out) < 100, "the served requests do not finish"
        scheduled = scheduler.schedule()
        if scheduled is None or scheduled[0] is None:
            continue
        batch, seqs = scheduled
        prefill = batch.total_seqs_num_prefill
        rows = zip(batch.num_scheduled_tokens, batch.context_lens)
        line = {
            "rows": [[int(q), int(c), i >= prefill] for i, (q, c) in enumerate(rows)],
            "produces_output": batch.produces_output(),
        }
        out.append((batch, line, [list(s) for s in batch.state_slots]))
        scheduler.postprocess(
            list(seqs.values()),
            ScheduledBatchOutput(**stream.step(batch)),
            stream_output_queue=queue.Queue(),
            batch=batch,
        )
    return out


def test_a_replayed_verify_step_of_a_gdn_model_holds_the_slots_atoms_scheduler_gives_it(
    model_dir,  # noqa: F811
):
    """Every row of every batch holds as many slots as the scheduler gave its
    request, two under MTP1, each row's its own and inside the state pool."""
    built = gdn_config(model_dir)
    rule = replay.ReplayRule(built)

    steps = scheduled(built)

    for batch, line, given in steps:
        rebuilt = rule.batch(line)
        assert [len(s) for s in rebuilt.state_slots] == [len(s) for s in given]
        assert {len(s) for s in given} == {MTP1}
        held = [slot for s in rebuilt.state_slots for slot in s]
        assert len(set(held)) == len(held) and max(held) < 8 * MTP1
        assert rebuilt.state_slots_committed == [s[0] for s in rebuilt.state_slots]
    verify = [line for batch, line, _ in steps if batch.total_seqs_num_decode]
    widest = max(verify, key=lambda line: len(line["rows"]))
    assert rule.batch(widest).state_slots == [[0, 1], [2, 3], [4, 5]]


def test_a_model_whose_request_holds_one_slot_keeps_one_a_row_under_a_drafter(
    model_dir,  # noqa: F811
):
    """A model that states one slot a request, as DeepSeek-V4.1 does, or a
    GDN model under ReplaySSM, holds one a row, past the slot it is given."""
    rule = replay.ReplayRule(gdn_config(model_dir, per_req=1))
    line = {"rows": [[2, 9, True], [2, 30, True]], "produces_output": True}
    assert rule.batch(line, slot=3).state_slots == [[3], [4]]


class StateRunner(test_bench_engine.Runner):
    """`test_bench_engine`'s worker, sizing a GDN model's state under MTP1."""

    def get_num_blocks(self):
        reply = super().get_num_blocks()
        return reply | {"pool_entries_per_req": {"state": MTP1}}


def test_the_bench_carries_the_slots_a_request_holds_to_the_replay_rule(
    manager, monkeypatch, model_dir  # noqa: F811
):
    """The rule built on the bench's config, as `measure steps` builds it,
    gives each verify row the two slots the workers' reply states."""
    engine = test_bench_engine.engine
    monkeypatch.setattr(engine, "RUNNER", f"{__name__}.StateRunner")
    config = test_bench_engine._config(eager=True)
    engine.Bench(config, test_bench_engine.BLOCKS).exit()
    stated = gdn_config(model_dir)
    del stated.pool_entries_per_req
    vars(config).update(vars(stated) | vars(config))

    rule = replay.ReplayRule(config)

    line = {"rows": [[2, 9, True], [2, 30, True]], "produces_output": True}
    assert rule.batch(line).state_slots == [[0, 1], [2, 3]]


def test_each_ops_placement_holds_the_batch_s_slots_of_its_own(monkeypatch):
    """A state-only leaf over 8 slots, two rows each holding two: each
    placement takes 4 slots past the last one's, and its copies' 7
    placements take the 2 sets of 4 the slots hold again."""
    runner = OpsRunner()
    runner.pool_plan = SimpleNamespace(entries={"state": 8})
    runner.config.pool_entries_per_req = {"state": MTP1}
    runner.kv_held = _state((16,))
    monkeypatch.setattr(fctx._forward_kv_cache_context, "kv_cache_data", {})
    monkeypatch.setattr(ops.Stub, "_time", lambda *_: {"cold_ns": 1.0})
    state = _t([8, 16], "kv", stride=[64, 1])
    args = ("kv", _t([2, 4], "activation")), ("kv_state", state), ("score", state)
    leaf = _leaf("v4.states", *args, ("layer", ir.ContextRef(KEY)), key=KEY)
    found = ops.leaves(_graph(leaf, layers=1)["region"])
    placed = SimpleNamespace(block_size=16, batch=lambda view, b, s: (b, s))

    [record] = ops.Stub(runner, 1000).point(placed, decode(2), 20, found)

    # 32 bytes of kv and two states of 128 a call: 7 copies pass twice 1000,
    # on 7 placements whose state is the 2 slot sets 8 slots hold, read once.
    assert [slot for _, slot in runner.batches] == [0, 4] * 3 + [0]
    assert record["footprint_bytes"] == 7 * 32 + 2 * 256
