"""Native cache observations and a quiescent warmup-to-measurement boundary.

Counters are cumulative. A measurement subtracts the reset's ``after.counters``
from its final snapshot; clearing indexes must not look like cache eviction.
This module performs no device operations. Workers own their completion fence.
"""

SNAPSHOT_SCHEMA = "compass.cache_snapshot/1"
RESET_SCHEMA = "compass.cache_reset/1"
FLUSH_SCHEMA = "compass.measurement_flush/1"


def snapshot(engine, *, input_batches=None):
    import os
    from atom.compass.core.cache_policy import snapshot as policy_snapshot

    scheduler = engine.scheduler
    if scheduler is None:
        raise RuntimeError("this engine has no scheduler")
    bm = scheduler.block_manager
    state = bm.state
    pressure = bm.pool_pressure()
    cache_stats = (scheduler.cache_stats.get_statistics()
                   if scheduler.cache_stats is not None else {})
    counters = {**cache_stats, **bm.checkpoint_funnel(),
                "blocks_evicted": pressure["blocks_evicted"],
                "blocks_retired": pressure["blocks_retired"]}
    readers = sum(state._pinned.values())
    deferred_readers = sum(state._pinned.get(slot, 0) for slot in state._deferred)
    # A final issued reader may remain pinned until another schedule() pass.
    # Those pins are released only AFTER the worker fence. Free checkpoints
    # are cache ownership, not readers, and never prevent an idle reset.
    counts = {
        "running_requests": len(scheduler.running),
        "waiting_requests": len(scheduler.waiting),
        "rejected_requests": len(scheduler._rejected),
        "deferred_free_requests": len(scheduler.deferred_free_blocks),
        "input_batches": (engine.input_queue.qsize()
                          if input_batches is None else input_batches),
        "stream_outputs": engine.stream_output_queue.qsize(),
        "pp_inflight_requests": len(scheduler._pp_inflight_token_block),
        "state_deferred_readers": deferred_readers,
        "state_live_slots": pressure["slots_used"] - len(state._pinned),
        "state_relocations": len(state._relocations),
    }
    reasons = [f"{name}={value}" for name, value in counts.items() if value]
    if engine.has_pending_kv_work():
        reasons.append("pending KV transfer work")
    # These deployments have additional retained batches/PAGE writers. Refuse
    # until they have their own fence, rather than certify the TP-only seam.
    if scheduler.advance_on_schedule:
        reasons.append("pipeline-parallel reset is unsupported")
    if bm.paged_state_checkpoints is not None:
        reasons.append("PAGE checkpoint reset is unsupported")
    return {
        "reader": {"component": "EngineCore.Scheduler", "pid": os.getpid()},
        "scheduler_configuration": {
            "max_model_len": scheduler.max_model_len,
            "max_num_seqs": scheduler.max_num_seqs,
            "max_num_batched_tokens": scheduler.max_num_batched_tokens,
            "kv_cache_block_size": bm.block_size,
        },
        "policy": policy_snapshot(bm, engine.state_runtime, environment={}),
        "pool_pressure": pressure,
        "cache_statistics": cache_stats,
        "counters": counters,
        "indexes": {"kv": bm.kv.num_indexed, "state": len(state.hash_to_slot)},
        "quiescence": {"idle": not reasons, "reasons": reasons,
                       **counts, "state_readers": readers},
    }


def reset(engine):
    """Fence issued work, then atomically check queued input and clear indexes.

    EngineCore calls this on its scheduling thread. The queue mutex serializes
    the final check/clear with input ingestion; newly accepted requests belong
    after the boundary. A retained, already completed token output is preserved.
    """
    return _quiescent_fence(engine, clear_indexes=True)


def flush_measurements(engine):
    """Finish timed worker events after measurement, preserving cached content."""
    return _quiescent_fence(engine, clear_indexes=False)


def _quiescent_fence(engine, *, clear_indexes):
    from atom.utils.clock import get_clock

    before = snapshot(engine)
    result = {"acknowledged": False, "before": before}
    if not before["quiescence"]["idle"]:
        result["reasons"] = before["quiescence"]["reasons"]
        return result
    barrier = engine.runner_mgr.call_func("compass_cache_barrier", wait_out=True)
    if not isinstance(barrier, dict) or barrier.get("acknowledged") is not True:
        result["reasons"] = ["worker completion fence was not acknowledged"]
        result["worker_barrier"] = barrier
        return result
    # AsyncIOProc's all-worker barrier runs after each worker's fence and
    # before rank zero responds, so this count is a completed physical group.
    barrier = dict(barrier, workers_completed=engine.runner_mgr.proc_num)
    result["worker_barrier"] = barrier
    # A predicting worker has returned, but EngineCore may still model queued
    # work after host return. Drain that queue on its owning virtual clock,
    # outside measurement, without touching CUDA or dropping deferred output.
    timeline = getattr(engine, "_compass_forward_timeline", None)
    advance_seconds = 0.0
    if timeline is not None:
        clock = get_clock()
        if barrier.get("kind") != "modelled_no_device" or not hasattr(clock, "advance"):
            result["reasons"] = ["modelled device queue has no virtual-clock fence"]
            return result
        end = max(value for value in (clock.time(), timeline.device_end,
                                      timeline.staging_ready, timeline.sample_ready)
                  if value is not None)
        advance_seconds = end - clock.time()
        clock.advance(advance_seconds)
    barrier["core_timeline_drained"] = True
    barrier["core_timeline_advance_seconds"] = advance_seconds
    clock = get_clock()
    barrier["core_virtual_elapsed_seconds"] = (
        clock.perf_counter() if getattr(clock, "epoch", None) is not None else None)
    with engine.input_queue.mutex:
        current = snapshot(engine, input_batches=len(engine.input_queue.queue))
        if not current["quiescence"]["idle"]:
            result["after"] = current
            result["reasons"] = current["quiescence"]["reasons"]
            return result
        bm = engine.scheduler.block_manager
        bm.state.release_pins()
        if clear_indexes:
            bm.clear_cache()
        after = snapshot(engine, input_batches=len(engine.input_queue.queue))
    result["after"] = after
    result["counter_delta"] = {
        key: value - before["counters"][key]
        for key, value in after["counters"].items()}
    result["acknowledged"] = (
        after["quiescence"]["idle"] and after["quiescence"]["state_readers"] == 0
        and after["indexes"] == ({"kv": 0, "state": 0} if clear_indexes
                                 else before["indexes"])
        and not any(result["counter_delta"].values()))
    if not result["acknowledged"]:
        result["reasons"] = ["quiescent worker fence postconditions failed"]
    return result


def flush_receipt_errors(receipt):
    """Require a native all-worker fence and an empty timing-event queue."""
    if not isinstance(receipt, dict) or receipt.get("schema") != FLUSH_SCHEMA:
        return ["missing measurement flush receipt schema"]
    errors = []
    if receipt.get("acknowledged") is not True:
        errors.append("measurement flush was not acknowledged")
    ranks = receipt.get("ranks")
    if not isinstance(ranks, list) or not ranks:
        return errors + ["measurement flush has no rank receipts"]
    for i, rank in enumerate(ranks):
        if not isinstance(rank, dict):
            errors.append(f"rank {i} measurement flush is not an object")
            continue
        before, after, barrier = (rank.get(k) for k in
                                  ("before", "after", "worker_barrier"))
        if not all(isinstance(value, dict) for value in (before, after, barrier)):
            errors.append(f"rank {i} measurement flush lacks snapshots or worker proof")
            continue
        quiet = after.get("quiescence") or {}
        journal = barrier.get("measurement_journal") or {}
        workers = barrier.get("workers_completed")
        if (rank.get("acknowledged") is not True
                or quiet.get("idle") is not True or quiet.get("state_readers") != 0
                or barrier.get("acknowledged") is not True
                or barrier.get("kind") != "device_synchronize"
                or type(workers) is not int or workers < 1
                or barrier.get("core_timeline_drained") is not True
                or journal.get("pending_steps_after") != 0):
            errors.append(f"rank {i} measurement flush lacks completed native timing proof")
        if (after.get("indexes") != before.get("indexes")
                or not before.get("counters")
                or after.get("counters") != before.get("counters")):
            errors.append(f"rank {i} measurement flush changed cache indexes or counters")
    return errors


def reset_receipt_errors(receipt, *, expected_worker_kind=None,
                         require_fresh_modelled=False):
    """Validate the native proof shared by replay and acceptance readers."""
    if not isinstance(receipt, dict) or receipt.get("schema") != RESET_SCHEMA:
        return ["missing cache reset receipt schema"]
    errors = []
    if receipt.get("acknowledged") is not True:
        errors.append("cache reset was not acknowledged")
    ranks = receipt.get("ranks")
    if not isinstance(ranks, list) or not ranks:
        return errors + ["cache reset has no rank receipts"]
    for i, rank in enumerate(ranks):
        if not isinstance(rank, dict):
            errors.append(f"rank {i} cache reset receipt is not an object")
            continue
        before = rank.get("before")
        after = rank.get("after")
        barrier = rank.get("worker_barrier")
        if not all(isinstance(item, dict) for item in (before, after, barrier)):
            errors.append(f"rank {i} cache reset lacks snapshots or worker proof")
            continue
        quiet = after.get("quiescence")
        if not isinstance(quiet, dict):
            errors.append(f"rank {i} cache reset lacks quiescence proof")
            continue
        if rank.get("acknowledged") is not True:
            errors.append(f"rank {i} cache reset was not acknowledged")
        if (quiet.get("idle") is not True or quiet.get("reasons") != []
                or quiet.get("state_readers") != 0
                or quiet.get("state_deferred_readers") != 0):
            errors.append(f"rank {i} cache reset is not quiescent")
        if after.get("indexes") != {"kv": 0, "state": 0}:
            errors.append(f"rank {i} cache reset did not empty both indexes")
        workers = barrier.get("workers_completed")
        if (barrier.get("acknowledged") is not True or type(workers) is not int
                or workers <= 0 or barrier.get("kind") not in
                {"device_synchronize", "modelled_no_device"}
                or barrier.get("core_timeline_drained") is not True):
            errors.append(f"rank {i} cache reset has no worker completion proof")
        if expected_worker_kind is not None and barrier.get("kind") != expected_worker_kind:
            errors.append(f"rank {i} cache reset worker kind differs")
        before_counters = before.get("counters")
        after_counters = after.get("counters")
        delta = rank.get("counter_delta")
        if (not isinstance(before_counters, dict) or not before_counters
                or after_counters != before_counters
                or delta != {key: 0 for key in before_counters}):
            errors.append(f"rank {i} cache reset changed or omitted cumulative counters")
        if require_fresh_modelled and barrier.get("kind") == "modelled_no_device":
            # Resetting indexes does not rebase declared arrivals. A warmed
            # predictor could therefore measure against an old virtual epoch.
            if (not isinstance(before_counters, dict) or any(before_counters.values())
                    or before.get("indexes") != {"kv": 0, "state": 0}
                    or barrier.get("retained_output_requests") != 0
                    or barrier.get("core_timeline_advance_seconds") != 0
                    or barrier.get("core_virtual_elapsed_seconds") != 0):
                errors.append(f"rank {i} modelled cache boundary is not a fresh server")
    return errors
