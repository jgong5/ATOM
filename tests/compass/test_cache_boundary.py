"""Warmup reset uses real scheduler pools and never rewrites cumulative counts."""

from collections import deque
from copy import deepcopy
import queue
from types import SimpleNamespace

import pytest
from conftest import MockConfig

from atom.compass.core import cache_boundary
from atom.compass.core.cache_policy import cache_on_policy
from atom.compass.runtime.predict import CompassPredictMixin
from atom.model_engine.engine_utility import EngineUtilityHandler
from atom.model_engine.scheduler import Scheduler
from atom.model_engine.sequence import Sequence
from atom.model_engine.state_runtime import StateRuntime, StateTransfer


def engine_with_cache(*, warm=True):
    runtime = StateRuntime(transfer=StateTransfer.fork(1, readable_midstep=False))
    config = MockConfig(
        kv_cache_block_size=16, num_kvcache_blocks=200, max_model_len=256,
        max_num_batched_tokens=128, max_num_seqs=4, enable_prefix_caching=True,
        pool_entries={"state": 8}, state_checkpoint_interval_tokens=8192,
        state_checkpoint_demand=True)
    scheduler = Scheduler(config, state_runtime=runtime)
    bm = scheduler.block_manager
    if warm:
        seq = Sequence(list(range(128)), 16, has_per_req_cache=True)
        bm.allocate(seq, bm.can_allocate(seq))
        bm.hash_blocks(seq, 112, next_forward_tokens=16)
        bm.deallocate(seq)
        assert bm.kv.num_indexed and bm.state.hash_to_slot
        scheduler.cache_stats.total_requests = 3
        scheduler.cache_stats.total_cached_tokens = 48
    calls = []

    def fence(name, *, wait_out):
        calls.append(name)
        assert wait_out and name == "compass_cache_barrier"
        if warm:
            assert bm.kv.num_indexed and bm.state.hash_to_slot
        return {"acknowledged": True, "kind": "modelled_no_device",
                "retained_output_requests": 0}

    engine = SimpleNamespace(
        scheduler=scheduler, state_runtime=runtime, input_queue=queue.Queue(),
        stream_output_queue=queue.Queue(), has_pending_kv_work=lambda: False,
        runner_mgr=SimpleNamespace(proc_num=2, call_func=fence))
    return engine, calls


def receipt(rank):
    return {"schema": cache_boundary.RESET_SCHEMA,
            "acknowledged": rank["acknowledged"], "ranks": [rank]}


def test_reset_clears_both_indexes_after_worker_fence_and_preserves_counters():
    engine, calls = engine_with_cache()
    before = cache_boundary.snapshot(engine)
    result = cache_boundary.reset(engine)
    assert calls == ["compass_cache_barrier"]
    assert result["acknowledged"] is True
    assert result["before"] == before
    assert result["after"]["indexes"] == {"kv": 0, "state": 0}
    assert result["after"]["pool_pressure"]["slots_held"] == 0
    assert result["after"]["counters"] == before["counters"]
    assert result["after"]["policy"] == cache_on_policy()
    assert cache_boundary.reset_receipt_errors(
        receipt(result), expected_worker_kind="modelled_no_device") == []


@pytest.mark.parametrize("warm", [False, True])
def test_modelled_measurement_requires_fresh_server_without_epoch_rebase(warm):
    from atom.utils.clock import VirtualClock, get_clock, set_clock

    engine, _ = engine_with_cache(warm=warm)
    previous = get_clock()
    set_clock(VirtualClock(epoch=100.0))
    try:
        result = cache_boundary.reset(engine)
        errors = cache_boundary.reset_receipt_errors(
            receipt(result), expected_worker_kind="modelled_no_device",
            require_fresh_modelled=True)
        assert bool(errors) is warm
    finally:
        set_clock(previous)


@pytest.mark.parametrize("busy", ["running", "waiting", "input", "deferred_reader",
                                  "state_owner", "kv_transfer", "pp_inflight"])
def test_reset_refuses_live_consumers_without_mutating_indexes(busy):
    engine, calls = engine_with_cache()
    bm = engine.scheduler.block_manager
    if busy in {"running", "waiting"}:
        getattr(engine.scheduler, busy).append(Sequence([1], 16))
    elif busy == "input":
        engine.input_queue.put_nowait([Sequence([1], 16)])
    elif busy == "deferred_reader":
        slot = next(iter(bm.state.hash_to_slot.values()))
        bm.state.claim(slot)
        bm.state.pin(slot, reader_is_next_batch=True)
    elif busy == "state_owner":
        bm.state.pop()
    elif busy == "kv_transfer":
        engine.has_pending_kv_work = lambda: True
    else:
        engine.scheduler._pp_inflight_token_block.add(1)
    before = cache_boundary.snapshot(engine)
    result = cache_boundary.reset(engine)
    assert result["acknowledged"] is False
    assert result["reasons"]
    assert calls == []
    assert cache_boundary.snapshot(engine) == before


def test_last_issued_reader_is_released_only_after_completion_fence():
    engine, calls = engine_with_cache()
    state = engine.scheduler.block_manager.state
    slot = next(iter(state.hash_to_slot.values()))
    state.claim(slot)
    state.pin(slot)
    original_fence = engine.runner_mgr.call_func

    def fence(*args, **kwargs):
        assert state.pin_count(slot) == 1
        return original_fence(*args, **kwargs)

    engine.runner_mgr.call_func = fence
    result = cache_boundary.reset(engine)
    assert result["acknowledged"] is True
    assert result["before"]["quiescence"]["state_readers"] == 1
    assert result["after"]["quiescence"]["state_readers"] == 0
    assert state.is_free(slot)


def test_new_input_during_worker_wait_refuses_the_clear():
    engine, calls = engine_with_cache()

    def fence(*args, **kwargs):
        engine.input_queue.put_nowait([Sequence([1], 16)])
        return {"acknowledged": True, "kind": "modelled_no_device"}

    engine.runner_mgr.call_func = fence
    result = cache_boundary.reset(engine)
    assert result["acknowledged"] is False
    assert result["after"]["indexes"] == result["before"]["indexes"]
    assert result["after"]["quiescence"]["input_batches"] == 1


def test_unacknowledged_worker_does_not_clear_indexes():
    engine, _ = engine_with_cache()
    engine.runner_mgr.call_func = lambda *a, **kw: {"acknowledged": False}
    before = cache_boundary.snapshot(engine)
    assert cache_boundary.reset(engine)["acknowledged"] is False
    assert cache_boundary.snapshot(engine) == before


@pytest.mark.parametrize("busy", [False, True])
def test_measurement_flush_preserves_cache_and_requires_quiescence(busy):
    engine, calls = engine_with_cache()
    native_fence = engine.runner_mgr.call_func

    def fence(*args, **kwargs):
        return {**native_fence(*args, **kwargs), "kind": "device_synchronize",
                "measurement_journal": {"pending_steps_before": 1,
                                        "drained_steps": 1, "pending_steps_after": 0}}

    engine.runner_mgr.call_func = fence
    if busy:
        engine.scheduler.waiting.append(Sequence([1], 16))
    before = cache_boundary.snapshot(engine)
    result = cache_boundary.flush_measurements(engine)
    assert result["acknowledged"] is (not busy)
    assert calls == ([] if busy else ["compass_cache_barrier"])
    assert cache_boundary.snapshot(engine) == before
    proof = {"schema": cache_boundary.FLUSH_SCHEMA,
             "acknowledged": result["acknowledged"], "ranks": [result]}
    assert bool(cache_boundary.flush_receipt_errors(proof)) is busy


def test_modelled_reset_drains_core_device_timeline_without_discarding_output():
    from atom.compass.runtime.timeline import ForwardTimeline
    from atom.utils.clock import VirtualClock, get_clock, set_clock

    engine, _ = engine_with_cache()
    previous_clock = get_clock()
    clock = VirtualClock(epoch=100.0)
    set_clock(clock)
    try:
        timeline = ForwardTimeline()
        marks = timeline.submit(clock.time(), 5.0, 0.1, 0.0, True)
        assert marks["host_returned_at"] == 100.0
        engine._compass_forward_timeline = timeline
        result = cache_boundary.reset(engine)
        assert result["acknowledged"]
        assert clock.time() == 105.0
        assert result["worker_barrier"]["core_timeline_advance_seconds"] == 5.0
        assert timeline.sample_ready == timeline.device_end == 105.0
    finally:
        set_clock(previous_clock)


def test_predict_boundary_is_gpu_free_and_preserves_deferred_output(monkeypatch):
    import torch

    def unexpected(*args, **kwargs):
        raise AssertionError("predict reset called a device API")

    monkeypatch.setattr(torch.cuda, "synchronize", unexpected)
    monkeypatch.setattr(torch.cuda, "_lazy_init", unexpected)
    deferred = [7, 8]
    runner = SimpleNamespace(_compass_config=SimpleNamespace(mode="predict"),
                             _pending=deque(), _deferred_output=deferred)
    result = CompassPredictMixin.compass_cache_barrier(runner)
    assert result["acknowledged"] and result["kind"] == "modelled_no_device"
    assert result["retained_output_requests"] == 2
    assert runner._deferred_output is deferred
    runner._pending.append(object())
    assert CompassPredictMixin.compass_cache_barrier(runner)["acknowledged"] is False


def test_native_utility_acknowledges_actual_scheduler_reset():
    engine, _ = engine_with_cache()
    output = queue.Queue()
    handler = EngineUtilityHandler(engine.runner_mgr, output,
                                   scheduler=engine.scheduler, engine=engine)
    handler._execute_utility_command("reset_compass_cache", {})
    kind, response = output.get_nowait()
    assert kind == "UTILITY_RESPONSE" and response["cmd"] == "reset_compass_cache"
    assert not cache_boundary.reset_receipt_errors(receipt(response["result"]))
    handler._execute_utility_command("get_compass_cache", {})
    assert output.get_nowait()[1]["result"]["indexes"] == {"kv": 0, "state": 0}


@pytest.mark.parametrize("break_proof", ["index", "reader", "worker", "counter",
                                         "rank", "schema", "kind"])
def test_receipt_checker_rejects_incomplete_or_contradictory_proof(break_proof):
    engine, _ = engine_with_cache()
    proof = deepcopy(receipt(cache_boundary.reset(engine)))
    rank = proof["ranks"][0]
    if break_proof == "index":
        rank["after"]["indexes"]["state"] = 1
    elif break_proof == "reader":
        rank["after"]["quiescence"]["state_readers"] = 1
    elif break_proof == "worker":
        rank["worker_barrier"]["workers_completed"] = 0
    elif break_proof == "counter":
        rank["after"]["counters"]["requests"] = 0
    elif break_proof == "rank":
        proof["ranks"] = []
    elif break_proof == "schema":
        proof["schema"] = "unknown"
    else:
        rank["worker_barrier"]["kind"] = "device_synchronize"
    assert cache_boundary.reset_receipt_errors(
        proof, expected_worker_kind="modelled_no_device")
