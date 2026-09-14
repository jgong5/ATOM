"""Exercise native worker fences and the public cache boundary on CPU."""

import asyncio
import importlib
import json
import queue
import threading
from collections import deque
from types import SimpleNamespace

import pytest

from .test_cache_boundary import engine_with_cache


@pytest.fixture(scope="module")
def native():
    from atom.compass.replay import bootstrap

    try:
        if not bootstrap.state().get("installed"):
            bootstrap.install("gfx942:sramecc+:xnack-", source="test")
        return SimpleNamespace(
            model_runner=importlib.import_module("atom.model_engine.model_runner"),
            async_proc=importlib.import_module("atom.model_engine.async_proc"),
            llm=importlib.import_module("atom.model_engine.llm_engine"),
            api=importlib.import_module("atom.entrypoints.openai.api_server"))
    except ImportError as exc:
        pytest.skip(f"native optional dependencies are unavailable: {exc}")


def test_native_fence_synchronizes_device_and_retains_last_output(native, monkeypatch):
    calls = []
    monkeypatch.setattr(native.model_runner.torch.cuda, "synchronize",
                        lambda device: calls.append(device))
    pending_output = [object()]
    processor = SimpleNamespace(prev_batch=SimpleNamespace(req_ids=[3, 4]),
                                token_ids_cpu=pending_output)
    runner = SimpleNamespace(device="cuda:0", tokenID_processor=processor)
    result = native.model_runner.ModelRunner.compass_cache_barrier(runner)
    assert calls == ["cuda:0"]
    assert result["kind"] == "device_synchronize" and result["acknowledged"]
    assert result["retained_output_requests"] == 2
    assert processor.token_ids_cpu is pending_output
    assert processor.prev_batch.req_ids == [3, 4]


def test_native_final_timing_row_is_flushed_once_from_original_events(
        native, monkeypatch, tmp_path):
    from atom.compass.core.cost.base import StepShape
    from atom.compass.config import CompassConfig
    from atom.compass.runtime.runner import CompassModelRunner

    ready, synchronized, elapsed_reads = False, [], []

    class Event:
        def __init__(self, name, milliseconds=0.0):
            self.name, self.milliseconds = name, milliseconds

        def query(self):
            return ready

        def elapsed_time(self, end):
            assert ready
            elapsed_reads.append((self.name, end.name))
            return self.milliseconds

    def synchronize(device):
        nonlocal ready
        synchronized.append(device)
        ready = True

    monkeypatch.setattr(native.model_runner.torch.cuda, "synchronize", synchronize)
    runner = object.__new__(CompassModelRunner)
    runner.device, runner.rank = "cuda:0", 0
    runner.tokenID_processor = SimpleNamespace(prev_batch=SimpleNamespace(req_ids=[4]))
    path = tmp_path / "steps.jsonl"
    runner.config = SimpleNamespace(compass_config=CompassConfig(
        mode="measure", measure_out=str(path), measure_warmup_steps=0))
    runner._measured_by_kind, runner._measured_steps, runner._measure_fh = {}, 0, None
    runner._topology = lambda: {"tp": 1}
    shape = StepShape((1,), (40168,), topology={"tp": 1}, rank_coords={"tp": 0},
                      capture_bucket=1, compiled=True)
    runner._pending = deque([(shape, Event("outer_begin", 24.5), Event("outer_end"),
                              0.0008, [4], 123.0, {"kind": "decode"},
                              {"run_model": (Event("model_begin", 24.0), Event("model_end"))})])
    runner._drain_pending()
    assert len(runner._pending) == 1 and not path.exists()
    assert synchronized == elapsed_reads == []
    try:
        result = runner.compass_cache_barrier()
        assert synchronized == ["cuda:0"]
        assert result["measurement_journal"] == {
            "worker_rank": 0, "pending_steps_before": 1,
            "drained_steps": 1, "pending_steps_after": 0}
        first_bytes = path.read_bytes()
        rows = [json.loads(line) for line in first_bytes.splitlines()]
        assert len(rows) == 1
        assert rows[0]["seconds"] == 0.0245
        assert rows[0]["span_seconds"] == {"run_model": 0.024}
        assert rows[0]["started_at"] == 123.0
        assert rows[0]["context_lens"] == [40168] and rows[0]["req_ids"] == ["4"]
        assert elapsed_reads == [("model_begin", "model_end"), ("outer_begin", "outer_end")]
        second = runner.compass_cache_barrier()
        assert second["measurement_journal"]["drained_steps"] == 0
        assert path.read_bytes() == first_bytes and len(elapsed_reads) == 2
        assert runner.tokenID_processor.prev_batch.req_ids == [4]
    finally:
        if runner._measure_fh is not None:
            runner._measure_fh.close()


@pytest.mark.parametrize("failed_rank", [None, 0, 1])
def test_rank_zero_cannot_ack_before_every_worker_fence_completes(native, failed_rank):
    barrier = threading.Barrier(2)
    entered = [threading.Event(), threading.Event()]
    release_slow_worker = threading.Event()
    outputs = [queue.Queue(), queue.Queue()]
    threads = []
    failures = []
    for rank in range(2):
        def fence(rank=rank):
            entered[rank].set()
            if rank == 1:
                assert release_slow_worker.wait(5)
            return {"acknowledged": rank != failed_rank, "rank": rank}

        commands = iter([("compass_cache_barrier", ()), ("exit", ())])
        proc = SimpleNamespace(
            _BARRIER_FUNCS=native.async_proc.AsyncIOProc._BARRIER_FUNCS,
            _KV_FUNC_NAMES=native.async_proc.AsyncIOProc._KV_FUNC_NAMES,
            get_func=lambda commands=commands: next(commands),
            runners=[SimpleNamespace(compass_cache_barrier=fence, exit=lambda: None)],
            all_ranks_barrier=barrier, io_addrs=(None, "test" if rank == 0 else None),
            io_queues=(queue.Queue(), outputs[rank]), kv_queue=None, label="test")

        def run(proc=proc):
            try:
                native.async_proc.AsyncIOProc.busy_loop(proc)
            except BaseException as exc:
                failures.append(exc)

        thread = threading.Thread(target=run, daemon=True)
        threads.append(thread)
        thread.start()
    try:
        assert all(event.wait(5) for event in entered)
        assert outputs[0].empty()
    finally:
        release_slow_worker.set()
        for thread in threads:
            thread.join(timeout=5)
    assert all(not thread.is_alive() for thread in threads)
    if failed_rank is None:
        assert not failures
        assert outputs[0].get_nowait()["acknowledged"] is True
    else:
        assert len(failures) == 2
        assert outputs[0].empty()
        assert barrier.broken
    assert outputs[1].empty()


def test_http_reset_carries_native_ack_and_busy_refusal(native, monkeypatch):
    from atom.compass.core.cache_boundary import reset_receipt_errors
    from atom.model_engine.engine_utility import EngineUtilityHandler
    from atom.model_engine.sequence import Sequence

    core, _ = engine_with_cache()
    output = queue.Queue()
    handler = EngineUtilityHandler(core.runner_mgr, output,
                                   scheduler=core.scheduler, engine=core)

    def broadcast(cmd, timeout):
        handler._execute_utility_command(cmd, {})
        return [output.get_nowait()[1]]

    engine = SimpleNamespace(core_mgr=SimpleNamespace(
        broadcast_utility_command_sync=broadcast))
    engine.reset_compass_cache = lambda: native.llm.LLMEngine.reset_compass_cache(engine)
    engine.get_compass_cache = lambda: native.llm.LLMEngine.get_compass_cache(engine)
    engine.flush_compass_measurements = lambda: native.llm.LLMEngine.flush_compass_measurements(engine)
    monkeypatch.setattr(native.api, "engine", engine)
    snapshot = asyncio.run(native.api.compass_cache())
    assert snapshot["schema"] == "compass.cache_snapshot/1"
    assert snapshot["ranks"][0]["indexes"]["state"] > 0
    core.scheduler.waiting.append(Sequence([1], 16))
    assert asyncio.run(native.api.compass_measurements_flush()).status_code == 409
    refused = asyncio.run(native.api.compass_cache_reset())
    assert refused.status_code == 409
    assert json.loads(refused.body)["acknowledged"] is False
    core.scheduler.waiting.clear()
    original_fence = core.runner_mgr.call_func
    core.runner_mgr.call_func = lambda *a, **kw: {
        **original_fence(*a, **kw), "kind": "device_synchronize",
        "measurement_journal": {"pending_steps_before": 1,
                                "drained_steps": 1, "pending_steps_after": 0}}
    flushed = asyncio.run(native.api.compass_measurements_flush())
    assert flushed.status_code == 200
    proof = json.loads(flushed.body)
    from atom.compass.core.cache_boundary import flush_receipt_errors
    assert not flush_receipt_errors(proof)
    assert proof["ranks"][0]["after"]["indexes"] == snapshot["ranks"][0]["indexes"]
    response = asyncio.run(native.api.compass_cache_reset())
    assert response.status_code == 200
    assert not reset_receipt_errors(json.loads(response.body))
