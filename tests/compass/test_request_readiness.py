"""Synthetic service events through native registration and scheduling seams."""

from dataclasses import FrozenInstanceError
import hashlib
import json
import os
import queue
import sys
from types import ModuleType

import pytest
from conftest import MockConfig

from atom.compass.config import CompassConfig
from atom.compass.core.loaded_input import load_json
from atom.compass.runtime.request_readiness import ReadyEvent, UnsupportedReadiness
from atom.model_engine.engine_utility import EngineUtilityHandler
from atom.model_engine.scheduler import ScheduledBatchOutput, Scheduler
from atom.sampling_params import SamplingParams
from atom.utils.clock import VirtualClock, WallClock, get_clock, set_clock


class SyntheticService:
    """Test-only ready events; no production source law is selected."""

    def __init__(self, table):
        self.rows, source = load_json(table, role="runtime.request_readiness.fixture")
        self.loaded_inputs = (source,)
        self.calls = 0

    def resolve_closed_workload(self, requests):
        self.calls += 1
        self.last_requests = requests
        events = {}
        for request in requests:
            key = str(request.workload_index)
            if key not in self.rows:
                raise UnsupportedReadiness("synthetic service has no coverage for request")
            row = self.rows[key]
            events[request.request_id] = ReadyEvent(
                request.arrived_at + row["delay"], row["order"])
        return events


class SyntheticSerialService(SyntheticService):
    """A synthetic one-writer fixture where offered order affects completion."""

    def resolve_closed_workload(self, requests):
        self.calls += 1
        available = float("-inf")
        events = {}
        for order, request in enumerate(requests):
            available = max(available, request.arrived_at) + self.rows[str(request.workload_index)]["delay"]
            events[request.request_id] = ReadyEvent(available, order)
        return events


@pytest.fixture
def virtual_clock():
    previous = get_clock()
    clock = VirtualClock(epoch=1000.)
    set_clock(clock)
    yield clock
    set_clock(previous)


@pytest.fixture
def profile(tmp_path, monkeypatch):
    module = ModuleType("synthetic_readiness_service")
    module.SyntheticService = SyntheticService
    module.SyntheticSerialService = SyntheticSerialService
    monkeypatch.setitem(sys.modules, module.__name__, module)

    def create(rows, resolver="SyntheticService"):
        table = tmp_path / "source_fixture.json"
        table.write_text(json.dumps(rows))
        path = tmp_path / "readiness.json"
        path.write_text(json.dumps({
            "schema": "compass.request_readiness_profile/1",
            "resolver": f"synthetic_readiness_service.{resolver}",
            "options": {"table": str(table)},
            "source_law": "synthetic-test-events-only",
            "support": {"kind": "synthetic fixture", "request_indices": list(rows)},
            "origin_contract": {
                "declared_arrival_event": "synthetic external arrival",
                "writer_eligibility_event": "synthetic writer eligibility",
                "transition": "fixture delay includes the entire represented transition",
            },
        }))
        return path, table
    return create


def scheduler_for(path, **overrides):
    return Scheduler(MockConfig(
        max_num_seqs=4, num_kvcache_blocks=64,
        compass_config=CompassConfig(enabled=True, epoch=1000.,
            request_readiness_profile=str(path)), **overrides))


def request(seq_factory, index, count, arrival=0.):
    seq = seq_factory([10] * 4, sampling_params=SamplingParams(max_tokens=1))
    seq.arrive_time = 1000. + arrival
    seq.compass_workload_index = index
    seq.compass_workload_size = count
    return seq


def finish(scheduler, batch, seqs):
    scheduler.postprocess(seqs.values(), ScheduledBatchOutput(
        req_ids=list(batch.req_ids), token_ids=[(99,)] * len(batch.req_ids),
        num_rejected=None, num_bonus=None, draft_token_ids=None), batch=batch)


def test_tied_arrivals_become_ready_separately_without_a_per_step_cap(
    virtual_clock, profile, seq_factory
):
    path, _ = profile({str(i): {"delay": (i + 1) / 10, "order": i} for i in range(4)})
    scheduler = scheduler_for(path)
    seqs = [request(seq_factory, i, 4) for i in range(4)]
    scheduler.extend(list(reversed(seqs)))
    batch, selected = scheduler.schedule()
    assert list(batch.req_ids) == [seqs[0].id]
    assert virtual_clock.elapsed == pytest.approx(.1)
    assert [seq.arrive_time for seq in seqs] == [1000.] * 4
    finish(scheduler, batch, selected)
    virtual_clock.advance(.4)
    batch, _ = scheduler.schedule()
    assert list(batch.req_ids) == [seq.id for seq in seqs[1:]]
    assert scheduler._request_readiness._provider.calls == 1


def test_registration_barrier_does_not_imply_visibility(
    virtual_clock, profile, seq_factory
):
    path, _ = profile({"0": {"delay": .3, "order": 1}, "1": {"delay": .1, "order": 0}})
    scheduler = scheduler_for(path)
    slow, fast = [request(seq_factory, i, 2) for i in range(2)]
    scheduler.add(slow)
    assert scheduler.schedule() is None
    assert virtual_clock.elapsed == 0
    assert scheduler._request_readiness.records is None
    scheduler.add(fast)
    batch, _ = scheduler.schedule()
    assert list(batch.req_ids) == [fast.id]
    assert virtual_clock.elapsed == pytest.approx(.1)


def test_initial_native_receipt_order_and_future_idle_events(
    virtual_clock, profile, seq_factory
):
    path, _ = profile({"0": {"delay": 8, "order": 2},
                       "1": {"delay": 1, "order": 1},
                       "2": {"delay": 2, "order": 0}})
    scheduler = scheduler_for(path, max_num_batched_tokens=4)
    first = request(seq_factory, 0, 3, arrival=0)
    second = request(seq_factory, 1, 3, arrival=3)
    third = request(seq_factory, 2, 3, arrival=2)
    scheduler.extend([first, second, third])
    actual = []
    for expected_time in [4, 4, 8]:
        batch, selected = scheduler.schedule()
        actual.extend(batch.req_ids)
        assert virtual_clock.elapsed == expected_time
        finish(scheduler, batch, selected)
    assert actual == [third.id, second.id, first.id]
    assert [first.arrive_time, second.arrive_time, third.arrive_time] == [1000, 1003, 1002]


def test_shuffled_http_receipt_does_not_set_synthetic_writer_order(
    virtual_clock, profile, seq_factory
):
    path, _ = profile({str(i): {"delay": 1} for i in range(3)}, "SyntheticSerialService")
    scheduler = scheduler_for(path)
    first = request(seq_factory, 0, 3)
    second = request(seq_factory, 1, 3)
    future = request(seq_factory, 2, 3, arrival=20)
    # The unpaced HTTP harness delivered the future request first, and also
    # reversed the tied logical arrivals. Neither fact is modeled writer order.
    scheduler.extend([future, second, first])
    for seq, ready in [(first, 1), (second, 2), (future, 21)]:
        batch, selected = scheduler.schedule()
        assert list(batch.req_ids) == [seq.id]
        assert virtual_clock.elapsed == ready
        finish(scheduler, batch, selected)
    assert scheduler._request_readiness._provider.calls == 1


def test_preemption_preserves_native_priority_and_never_recharges_ingress(
    virtual_clock, profile, seq_factory
):
    path, _ = profile({str(i): {"delay": i / 10, "order": i} for i in range(3)})
    scheduler = scheduler_for(path, max_num_batched_tokens=4)
    seqs = [request(seq_factory, i, 3) for i in range(3)]
    scheduler.extend(list(reversed(seqs)))
    virtual_clock.advance(1)
    assert list(scheduler.schedule()[0].req_ids) == [seqs[0].id]
    assert list(scheduler.schedule()[0].req_ids) == [seqs[1].id]
    records = scheduler._request_readiness.records
    for seq in seqs[:2]:
        scheduler.running.remove(seq)
        assert scheduler.preempt(seq)
    assert list(scheduler.schedule()[0].req_ids) == [seqs[1].id]
    assert scheduler._request_readiness.records is records
    assert scheduler._request_readiness._provider.calls == 1


def test_ready_running_work_prevents_an_idle_jump_to_future_readiness(
    virtual_clock, profile, seq_factory
):
    path, _ = profile({"0": {"delay": 0, "order": 0}, "1": {"delay": 10, "order": 1}})
    scheduler = scheduler_for(path)
    current, future = [request(seq_factory, i, 2) for i in range(2)]
    scheduler.extend([future, current])
    first, _ = scheduler.schedule()
    assert list(first.req_ids) == [current.id]
    second, _ = scheduler.schedule()
    assert list(second.req_ids) == [current.id]
    assert virtual_clock.elapsed == 0
    assert scheduler._declared_arrival_pending(future)


def test_real_clock_does_not_open_profile_or_change_fifo(profile, seq_factory):
    previous = get_clock()
    set_clock(WallClock())
    try:
        scheduler = scheduler_for("/profile/that/does/not/exist", max_num_batched_tokens=4)
        later = request(seq_factory, 1, 2, 20)
        earlier = request(seq_factory, 0, 2, 0)
        scheduler.extend([later, earlier])
        assert list(scheduler.schedule()[0].req_ids) == [later.id]
        assert scheduler._request_readiness is None
    finally:
        set_clock(previous)


def test_ready_records_and_declared_arrival_are_immutable(
    virtual_clock, profile, seq_factory
):
    path, _ = profile({"0": {"delay": 1, "order": 0}})
    scheduler = scheduler_for(path)
    seq = request(seq_factory, 0, 1)
    scheduler.add(seq)
    scheduler.schedule()
    readiness = scheduler._request_readiness
    with pytest.raises(TypeError):
        readiness.records[seq.id] = None
    with pytest.raises(FrozenInstanceError):
        readiness.record(seq).ready_at = 0
    with pytest.raises(UnsupportedReadiness, match="already resolved"):
        readiness.resolve_closed_workload([seq])
    seq.arrive_time += 1
    with pytest.raises(UnsupportedReadiness, match="arrival changed"):
        scheduler._schedulable_at(seq)


def test_ingress_descriptor_matches_native_single_request_sender(
    virtual_clock, profile, seq_factory
):
    from atom.model_engine.engine_core_mgr import CoreManager

    path, _ = profile({"0": {"delay": 1, "order": 0}})
    scheduler = scheduler_for(path)
    seq = request(seq_factory, 0, 1)
    sender = CoreManager.__new__(CoreManager)
    sender.label, sender.pp_size, sender.local_engine_count = "CPU sender", 1, 1
    payloads = []
    sender._send_request = lambda rank, payload: payloads.append((rank, payload))
    sender.add_request([seq])
    scheduler.add(seq)
    scheduler.schedule()
    offered, = scheduler._request_readiness._provider.last_requests
    assert offered.ingress.reconstructed_add_bytes == len(payloads[0][1])
    assert offered.ingress.frame_request_count == 1
    assert offered.ingress.token_typecode == "i" and offered.ingress.token_bytes == 16
    with pytest.raises(FrozenInstanceError):
        offered.ingress.token_bytes = 0


@pytest.mark.parametrize("name,value", [
    ("multimodal_data", {"image": "fixture"}), ("kv_transfer_params", {"remote": True}),
    ("stop_strings", ["stop"]), ("stop_token_sequences", [[1]]),
    ("return_logprobs", True), ("num_draft_tokens", 1), ("needs_independent_noise", True),
])
def test_expanded_ingress_layout_is_not_treated_as_plain_tokens(
    virtual_clock, profile, seq_factory, name, value
):
    path, _ = profile({"0": {"delay": 1, "order": 0}})
    scheduler = scheduler_for(path)
    seq = request(seq_factory, 0, 1)
    setattr(seq, name, value)
    scheduler.add(seq)
    with pytest.raises(UnsupportedReadiness, match="expanded request metadata"):
        scheduler.schedule()


def test_ready_time_cannot_contradict_native_receipt_order(
    virtual_clock, profile, seq_factory
):
    path, _ = profile({"0": {"delay": 3, "order": 0}, "1": {"delay": 2, "order": 1}})
    scheduler = scheduler_for(path)
    scheduler.extend([request(seq_factory, i, 2) for i in range(2)])
    with pytest.raises(UnsupportedReadiness, match="contradicts"):
        scheduler.schedule()


@pytest.mark.parametrize("rows,match", [
    ({}, "no coverage"),
    ({"0": {"delay": -1, "order": 0}}, "no earlier"),
    ({"0": {"delay": float("inf"), "order": 0}}, "finite"),
    ({"0": {"delay": 0, "order": -1}}, "nonnegative"),
])
def test_missing_or_invalid_events_fail_instead_of_zero_service(
    virtual_clock, profile, seq_factory, rows, match
):
    path, _ = profile(rows)
    scheduler = scheduler_for(path)
    scheduler.add(request(seq_factory, 0, 1))
    with pytest.raises(UnsupportedReadiness, match=match):
        scheduler.schedule()


def test_core_provenance_keeps_bytes_read_and_preserves_worker_manifest(
    virtual_clock, profile, seq_factory
):
    path, table = profile({"0": {"delay": 1, "order": 0}})
    expected = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [path, table]}
    scheduler = scheduler_for(path)
    path.write_text("changed after core load")
    table.write_text("changed after provider load")
    seq = request(seq_factory, 0, 1)
    scheduler.add(seq)
    scheduler.schedule()

    worker = {"inputs": [{"role": "oracle.synthetic", "sha256": "worker-only"}]}
    class Runner:
        def call_func(self, name, **kwargs):
            assert name == "compass_input_manifest" and kwargs == {"wait_out": True}
            return worker
    output = queue.Queue()
    EngineUtilityHandler(Runner(), output, scheduler=scheduler)._handle_get_compass_inputs({})
    kind, response = output.get_nowait()
    assert kind == "UTILITY_RESPONSE"
    result = response["result"]
    assert result["inputs"] == worker["inputs"] and "core_inputs" not in worker
    core = result["core_inputs"]
    assert {row["path"]: row["sha256"] for row in core["inputs"]} == expected
    assert core["reader"] == {"component": "EngineCore.Scheduler", "pid": os.getpid()}
    assert core["request_readiness"]["resolved_requests"] == 1


def test_old_admission_scalar_cannot_be_stacked():
    with pytest.raises(ValueError, match="mutually exclusive"):
        CompassConfig(request_readiness_profile="profile.json", admission_seconds=.001)


def test_origin_binding_is_required_before_loading_service(virtual_clock, profile):
    path, _ = profile({})
    data = json.loads(path.read_text())
    del data["origin_contract"]["transition"]
    path.write_text(json.dumps(data))
    with pytest.raises(UnsupportedReadiness, match="origin contract needs transition"):
        scheduler_for(path)


def test_incomplete_registration_retains_timeout_invalidation(
    virtual_clock, profile, seq_factory, monkeypatch
):
    import atom.model_engine.scheduler as scheduler_module
    path, _ = profile({"0": {"delay": 0, "order": 0}})
    scheduler = scheduler_for(path)
    scheduler.add(request(seq_factory, 0, 2))
    scheduler._arrival_barrier_since = 0
    scheduler.ARRIVAL_BARRIER_TIMEOUT_S = .5
    monkeypatch.setattr(scheduler_module._time, "monotonic", lambda: 1.)
    with pytest.raises(ValueError, match="incomplete arrival barrier"):
        scheduler.schedule()
    assert scheduler.arrival_barrier_timed_out == {"arrived": 1, "expected": 2, "timeout_s": .5}
    assert virtual_clock.elapsed == 0 and scheduler._request_readiness.records is None


@pytest.mark.parametrize("overrides", [
    {"tensor_parallel_size": 2}, {"pipeline_parallel_size": 2},
    {"data_parallel_size": 2}, {"scheduler_delay_factor": .1},
])
def test_unsupported_topology_or_policy_is_refused(virtual_clock, profile, overrides):
    path, _ = profile({})
    with pytest.raises(UnsupportedReadiness):
        scheduler_for(path, **overrides)


def test_prefill_delayer_cannot_be_enabled_later(virtual_clock, profile):
    path, _ = profile({})
    scheduler = scheduler_for(path)
    with pytest.raises(ValueError, match="prefill delayer"):
        scheduler.set_prefill_delayer(object())
