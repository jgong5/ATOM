"""Causal opening release through native cache probes and source readiness."""

import hashlib
import json

import pytest
from conftest import MockConfig

from atom.compass.config import CompassConfig
from atom.compass.core.loaded_input import load_json
from atom.compass.core.cache_policy import cache_on_policy
from atom.compass.prefix_workload import token_digest
from atom.compass.replay_plan import OpeningPlan, PROFILE, RESPONSE_DELIVERY, SCHEMA
from atom.compass.runtime.request_readiness import ReadyEvent, UnsupportedReadiness
from atom.model_engine.scheduler import ScheduledBatchOutput, Scheduler
from atom.model_engine.sequence import Sequence
from atom.model_engine.state_runtime import StateRuntime, StateTransfer
from atom.sampling_params import SamplingParams
from atom.utils.clock import VirtualClock, get_clock, set_clock


class SerialService:
    def __init__(self, source):
        self.delay, loaded = load_json(source, role="runtime.request_readiness.fixture")
        self.loaded_inputs = (loaded,)
        self.calls = []

    def resolve_closed_workload(self, requests):
        raise AssertionError("opening must not precharge a closed workload")

    def resolve_serial_release(self, request):
        self.calls.append(request)
        return ReadyEvent(request.arrived_at + self.delay["seconds"], 0)


def opening_fixture(tmp_path, gap):
    first = list(range(16400))
    later = first[:8192] + list(range(100000, 108212))
    rows = []
    for index, tokens in enumerate((first, later)):
        output = 3 if index == 0 else 1
        rows.append({"index": index, "source_path": f"/requests/{index}",
                     "arrival_s": gap if index else 0., "depends_on": [0] if index else [],
                     "input_tokens": len(tokens), "output_tokens": output,
                     "prompt_token_ids": tokens, "prompt_token_sha256": token_digest(tokens),
                     "body": {"model": "fixture", "messages": [{"role": "user", "content": "x"}],
                              "stream": True, "ignore_eos": True, "temperature": 0,
                              "max_completion_tokens": output}})
    payload = {"schema": SCHEMA, "profile": PROFILE, "clients": 1, "branches": [],
               "initial_cache": "acknowledged_empty", "time_scale": 1,
               "cache_policy": cache_on_policy(),
               "response_delivery": RESPONSE_DELIVERY, "requests": rows}
    path = tmp_path / "opening.json"
    path.write_text(json.dumps(payload))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    source = tmp_path / "service.json"
    source.write_text(json.dumps({"seconds": .25}))
    profile = tmp_path / "readiness.json"
    profile.write_text(json.dumps({
        "schema": "compass.request_readiness_profile/1",
        "resolver": f"{__name__}.SerialService", "options": {"source": str(source)},
        "source_law": "synthetic fixture", "support": {"kind": "test-only"},
        "origin_contract": {"declared_arrival_event": "causal release",
                            "writer_eligibility_event": "causal release",
                            "transition": "synthetic service duration"}}))
    return path, digest, profile, rows


@pytest.mark.parametrize("gap", [.5, 10.])
def test_source_and_response_release_precede_readiness_demand_and_allocation(tmp_path, gap):
    path, digest, profile, rows = opening_fixture(tmp_path, gap)
    old = get_clock()
    clock = VirtualClock(epoch=1000.)
    set_clock(clock)
    try:
        scheduler = Scheduler(MockConfig(
            kv_cache_block_size=16, num_kvcache_blocks=8192, max_num_batched_tokens=16384,
            max_num_seqs=32, max_model_len=262144, enable_prefix_caching=True,
            pool_entries={"state": 32}, state_checkpoint_interval_tokens=8192,
            state_checkpoint_demand=True,
            compass_config=CompassConfig(enabled=True, epoch=1000.,
                request_readiness_profile=str(profile), opening_plan=str(path), opening_plan_sha256=digest)),
            state_runtime=StateRuntime(transfer=StateTransfer.fork(1)))
        sequences = [Sequence(row["prompt_token_ids"], 16, has_per_req_cache=True,
                             sampling_params=SamplingParams(max_tokens=row["output_tokens"], ignore_eos=True))
                     for row in rows]
        for row, seq in zip(rows, sequences):
            seq.arrive_time = 1000. + row["arrival_s"]
            seq.compass_workload_index = row["index"]
            seq.compass_workload_size = 2
        first, later = sequences
        scheduler.extend([later, first])
        calendar = scheduler._release_calendar
        service = scheduler._request_readiness._provider
        bm = scheduler.block_manager
        while first in scheduler.waiting or first in scheduler.running:
            batch, selected = scheduler.schedule()
            assert batch.req_ids == [first.id]
            assert len(service.calls) == 1  # Source time can pass while output is unfinished.
            assert later.checkpoint_demand_pos == 0 and bm.demands_recorded == 0
            assert not later.block_table and not later.state_slots
            scheduler.postprocess(list(selected.values()), ScheduledBatchOutput(
                req_ids=batch.req_ids, token_ids=[(99,)], num_rejected=None,
                num_bonus=None, draft_token_ids=None), batch=batch)
            if first in scheduler.running:
                assert first.id not in calendar.completed
                clock.advance(1.)
        released_at = max(1000. + gap, first.finish_time)
        assert later.arrive_time == released_at
        assert service.calls[1].arrived_at == released_at
        assert len(service.calls) == 2
        assert calendar.completed[first.id]["native_engine_finished_at"] == first.finish_time
        assert calendar.completed[first.id]["response_delivery"]["kind"] == "assumed_zero"

        clock.advance(released_at + .24 - clock.time())
        before = (bm.pool_pressure(), dict(bm.state.hash_to_slot))
        assert not scheduler._can_admit_head_prefill()
        assert scheduler._waiting_new_token_count() == 0
        assert scheduler._oldest_waiting_prefill_age_ms() == 0
        assert (bm.pool_pressure(), dict(bm.state.hash_to_slot)) == before
        assert later.checkpoint_demand_pos == 0 and bm.demands_recorded == 0
        clock.advance(.01)
        assert scheduler._can_admit_head_prefill()
        assert bm.demands_recorded == 1
        batch, _ = scheduler.schedule()
        assert batch.req_ids == [later.id]
    finally:
        set_clock(old)


def test_opening_identity_and_scope_refuse_silent_changes(tmp_path):
    path, digest, _, _ = opening_fixture(tmp_path, 1.)
    plan = OpeningPlan.load(path, digest)
    exposed = plan.rows
    exposed[0]["prompt_token_ids"][0] = 999
    assert plan.rows[0]["prompt_token_ids"][0] == 0
    data = json.loads(path.read_text())
    data["requests"][1]["depends_on"] = []
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="digest"):
        OpeningPlan.load(path, digest)
    with pytest.raises(ValueError, match="first two"):
        OpeningPlan.load(path, hashlib.sha256(path.read_bytes()).hexdigest())


def test_unqualified_readiness_provider_cannot_use_the_serial_shortcut(tmp_path):
    path, digest, profile, _ = opening_fixture(tmp_path, 1.)
    from atom.compass.runtime.request_readiness import RequestReadiness
    service = RequestReadiness(str(profile))
    service._provider.resolve_serial_release = None
    with pytest.raises(UnsupportedReadiness, match="qualify serial"):
        service.register_serial_workload([])
