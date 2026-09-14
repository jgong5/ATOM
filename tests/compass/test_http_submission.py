"""CPU transport contracts: responses cannot prevent later registrations."""

import asyncio
import importlib.util
import json
import pickle
import resource
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from aiohttp import web


spec = importlib.util.spec_from_file_location(
    "http_replay", Path(__file__).resolve().parents[2] / "scripts/compass/replay.py")
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


@asynccontextmanager
async def server(handler):
    app = web.Application(client_max_size=4 * 1024 * 1024)
    app.router.add_post("/v1/completions", handler)
    runner = web.AppRunner(app, handler_cancellation=True)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0, backlog=32768)
    await site.start()
    base = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
    try:
        yield base
    finally:
        await runner.cleanup()


def payloads(count, tokens=4, *, declared=True):
    return [json.dumps({
        "model": "m", "prompt": [1000] * tokens, "max_tokens": 2,
        "temperature": 0.0, "ignore_eos": True,
        **({"compass_arrival": i % 3 * 0.25,
            "compass_workload_size": count, "compass_workload_index": i}
           if declared else {}),
    }).encode() for i in range(count)]


def reply(body):
    return web.json_response({"usage": {
        "prompt_tokens": len(body["prompt"]),
        "completion_tokens": body["max_tokens"]}, "choices": [{"finish_reason": "length"}]})


@pytest.mark.parametrize("count,tokens", [(1025, 4), (3551, 4), (16913, 4), (32, 262144)])
def test_all_requests_register_before_any_response(count, tokens):
    async def scenario():
        received, all_received = {}, asyncio.Event()

        async def handle(request):
            body = await request.json()
            ordinal = body["compass_workload_index"]
            assert ordinal not in received
            assert body["compass_workload_size"] == count
            assert body["compass_arrival"] == ordinal % 3 * 0.25
            assert body["ignore_eos"] and len(body["prompt"]) == tokens
            received[ordinal] = body
            if len(received) == count:
                all_received.set()
            await all_received.wait()
            return reply(body)

        async with server(handle) as base:
            results, metadata = await asyncio.wait_for(replay._submit_requests(
                base, payloads(count, tokens), [i % 3 * 0.25 for i in range(count)],
                pace=False, timeout=45), timeout=50)
        assert sorted(received) == list(range(count))
        assert [r["index"] for r in results] == list(range(count))
        assert all(r["ok"] for r in results), [r for r in results if not r["ok"]][:3]
        assert metadata["requests_with_header_callback"] == count
        assert metadata["task_setup_seconds"] >= 0
        assert all(r["send_timing"]["body_chunk_callback_bytes"] > 0 for r in results)
        assert all(r["send_timing"]["headers_callback_at"] >= metadata["pacing_started_at"] for r in results)

    asyncio.run(scenario())


def test_paced_arrival_can_reach_server_while_earlier_response_is_held():
    async def scenario():
        times, both_received = [], asyncio.Event()

        async def handle(request):
            body = await request.json()
            assert not any(k.startswith("compass_") for k in body)
            times.append(asyncio.get_running_loop().time())
            if len(times) == 2:
                both_received.set()
            await both_received.wait()
            return reply(body)

        async with server(handle) as base:
            results, _ = await replay._submit_requests(
                base, payloads(2, declared=False), [0.0, 0.15], pace=True, timeout=2)
        assert all(r["ok"] for r in results)
        assert times[1] - times[0] >= 0.10
        assert results[0]["send_timing"]["finished_offset_s"] >= results[1]["send_timing"]["headers_callback_offset_s"]
    asyncio.run(scenario())


def test_http_failure_and_timeout_remain_incomplete_requests():
    async def scenario():
        async def handle(request):
            body = await request.json()
            if body["compass_workload_index"] == 0:
                return web.Response(status=503, text="unavailable")
            await asyncio.sleep(2)
            return reply(body)

        async with server(handle) as base:
            results, _ = await replay._submit_requests(
                base, payloads(2), [0.0, 0.0], pace=False, timeout=0.05)
        assert len(results) == 2 and not any(r["ok"] for r in results)
        assert "HTTP 503: unavailable" in results[0]["error"]
        assert "Timeout" in results[1]["error"]
        assert len(replay._incomplete(results, [{"output_tokens": 2}] * 2)["failed"]) == 2
    asyncio.run(scenario())


def test_cancellation_keeps_a_failed_result_for_every_row():
    async def scenario():
        arrived = asyncio.Event()

        async def handle(request):
            await request.read()
            arrived.set()
            await asyncio.Event().wait()

        async with server(handle) as base:
            task = asyncio.create_task(replay._submit_requests(
                base, payloads(3, declared=False), [0.0, 60.0, 120.0], pace=True, timeout=10))
            await arrived.wait()
            task.cancel()
            results, _ = await task
        assert len(results) == 3 and not any(r["ok"] for r in results)
        assert all("CancelledError" in r["error"] for r in results)
    asyncio.run(scenario())


def test_resource_shortage_refuses_before_server_preparation(tmp_path, monkeypatch):
    monkeypatch.setattr(replay, "_prepare", lambda *a: pytest.fail("must refuse before preparation"))
    monkeypatch.setattr(replay.resource, "getrlimit", lambda _kind: (128, 128))
    args = ["--port", "1", "--model", "m", "--num-requests", "1025",
            "--prepare", "1", "--out", str(tmp_path / "absent.json")]
    assert replay.main(args) == 3
    monkeypatch.setattr(replay.resource, "getrlimit", lambda _kind: (resource.RLIM_INFINITY,) * 2)
    assert replay.main(args + ["--client-memory-budget-mib", "1"]) == 3
    assert not (tmp_path / "absent.json").exists()


def test_prepared_bytes_refuse_as_a_whole_before_dispatch(monkeypatch):
    monkeypatch.setattr(replay, "_encoded_prompt", lambda *a: [1000] * 1000)
    with pytest.raises(ValueError, match="no requests from this phase were sent"):
        replay._encode_requests([{"input_tokens": 1000, "output_tokens": 2, "arrival_s": 0}],
                                "m", None, declared=True, byte_budget=100)


def test_received_rows_keep_native_singleton_add_bytes(seq_factory):
    from atom.compass.runtime.request_readiness import ingress_descriptor
    from atom.model_engine.engine_core_mgr import CoreManager
    from atom.sampling_params import SamplingParams

    frames = []
    sender = CoreManager.__new__(CoreManager)
    sender.label, sender.pp_size, sender.local_engine_count = "CPU HTTP canary", 1, 1
    sender._send_request = lambda rank, payload: frames.append((rank, payload))

    async def scenario():
        async def handle(request):
            body = await request.json()
            seq = seq_factory(body["prompt"], sampling_params=SamplingParams(
                max_tokens=body["max_tokens"], temperature=body["temperature"],
                ignore_eos=body["ignore_eos"]))
            seq.arrive_time = body["compass_arrival"]
            seq.compass_workload_size = body["compass_workload_size"]
            seq.compass_workload_index = body["compass_workload_index"]
            sender.add_request([seq])
            descriptor = ingress_descriptor(seq)
            assert descriptor.reconstructed_add_bytes == len(frames[-1][1])
            assert descriptor.frame_request_count == 1
            assert len(pickle.loads(frames[-1][1])[1]) == 1
            return reply(body)

        async with server(handle) as base:
            results, _ = await replay._submit_requests(
                base, payloads(4, 1024), [0.0] * 4, pace=False, timeout=2)
        assert all(r["ok"] for r in results)
        assert len(frames) == 4
    asyncio.run(scenario())


def test_cancelled_submission_writes_incomplete_cli_artifact(tmp_path, monkeypatch):
    original = replay._submit_requests

    async def cancel_submission(*args, **kwargs):
        task = asyncio.create_task(original(*args, **kwargs))
        await asyncio.sleep(0.05)
        task.cancel()
        return await task

    monkeypatch.setattr(replay, "_submit_requests", cancel_submission)
    trace, out = tmp_path / "trace.jsonl", tmp_path / "result.json"
    trace.write_text("\n".join(json.dumps({
        "arrival_s": at, "input_tokens": 4, "output_tokens": 2,
    }) for at in [0.0, 60.0, 120.0]))

    async def scenario():
        async def handle(request):
            return reply(await request.json())

        async with server(handle) as base:
            code = await asyncio.to_thread(replay.main, [
                "--port", base.rsplit(":", 1)[1], "--model", "m", "--trace", str(trace),
                "--pace", "--out", str(out), "--timeout", "2"])
        assert code == replay.INCOMPLETE_EXIT
        saved = json.loads(out.read_text())
        assert saved["run"]["complete"] is False
        assert saved["run"]["failed"] >= 2
        assert len(saved["results"]) == 3
    asyncio.run(scenario())


def test_peer_that_stops_reading_upload_has_deadline_and_incomplete_artifact(tmp_path, monkeypatch):
    # Exercise upload backpressure, not an unanswered read after a small upload.
    # The bare peer accepts TCP and immediately pauses reading. This is a
    # transport stress payload, not a model-serving/context-length claim.
    monkeypatch.setattr(replay, "_send", lambda *args: {})  # post-run records
    out = tmp_path / "stalled-upload.json"

    async def scenario():
        peers = []

        class StoppedReader(asyncio.Protocol):
            def connection_made(self, transport):
                peers.append(transport)
                transport.pause_reading()

        listener = await asyncio.get_running_loop().create_server(StoppedReader, "127.0.0.1", 0)
        try:
            port = listener.sockets[0].getsockname()[1]
            code = await asyncio.wait_for(asyncio.to_thread(replay.main, [
                "--port", str(port), "--model", "m", "--num-requests", "1",
                "--input-tokens", str(4 * 1024 * 1024), "--output-tokens", "2",
                "--timeout", "0.15", "--out", str(out)]), timeout=5)
        finally:
            for peer in peers:
                peer.close()
            listener.close()
            await listener.wait_closed()
        assert peers, "the client must have established a connection"
        assert code == replay.INCOMPLETE_EXIT
        saved = json.loads(out.read_text())
        assert saved["run"]["failed"] == 1 and not saved["run"]["complete"]
        row = saved["results"][0]
        assert "Timeout" in row["error"]
        timing = row["send_timing"]
        assert timing["body_chunk_callback_bytes"] > 0
        assert timing["finished_offset_s"] - timing["request_started_offset_s"] < 2
        assert "pre-write" in saved["run"]["submission"]["timing_meaning"]
    asyncio.run(scenario())


def test_network_deadline_does_not_include_the_pacing_wait():
    async def scenario():
        async def handle(request):
            return reply(await request.json())

        async with server(handle) as base:
            results, metadata = await replay._submit_requests(
                base, payloads(1, declared=False), [0.3], pace=True, timeout=0.2)
        assert results[0]["ok"]
        assert results[0]["send_timing"]["request_started_offset_s"] >= 0.3
        assert metadata["network_attempt_timeout_s"] == 0.2
    asyncio.run(scenario())
