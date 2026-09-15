"""Opening transport waits for EOF on real time and preregisters virtual rows."""

import asyncio
from contextlib import asynccontextmanager
import importlib.util
import json
from pathlib import Path

from aiohttp import web
import pytest


spec = importlib.util.spec_from_file_location(
    "opening_http_replay", Path(__file__).resolve().parents[2] / "scripts/compass/replay.py")
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


@asynccontextmanager
async def server(handler):
    app = web.Application()
    app.router.add_post("/v1/chat/completions", handler)
    runner = web.AppRunner(app, handler_cancellation=True)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    try:
        yield f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
    finally:
        await runner.cleanup()


def frames(index, count=2, done=True):
    finish = {"id": f"chat-{index}", "choices": [{"finish_reason": "length"}]}
    usage = {"id": f"chat-{index}", "choices": [],
             "usage": {"prompt_tokens": 4, "completion_tokens": count}}
    return (f"data: {json.dumps(finish)}\n\ndata: {json.dumps(usage)}\n\n"
            + ("data: [DONE]\n\n" if done else "")).encode()


@pytest.mark.parametrize("gap", [.01, .15])
def test_real_opening_obeys_source_time_and_complete_response_eof(gap):
    async def scenario():
        arrivals = []
        first_eof = False

        async def handle(request):
            nonlocal first_eof
            index = (await request.json())["index"]
            arrivals.append(index)
            if index == 1:
                assert first_eof
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            await response.write(frames(index))
            if index == 0:
                # DONE alone cannot release the next request while HTTP is open.
                await asyncio.sleep(.06)
                assert arrivals == [0]
                first_eof = True
            await response.write_eof()
            return response

        async with server(handle) as base:
            results, _ = await replay._submit_requests(
                base, [json.dumps({"index": i}).encode() for i in range(2)], [0., gap],
                pace=True, timeout=2, endpoint="/v1/chat/completions", streaming=True,
                response_gated=True, expected_outputs=[2, 2])
        assert all(row["ok"] for row in results)
        first, second = [row["send_timing"] for row in results]
        assert second["request_started_offset_s"] >= gap
        assert second["request_started_offset_s"] >= first["finished_offset_s"]
        assert first["response_eof_wall_time"] > first["sse_done_wall_time"]
        assert first["client_response_returned_wall_time"] >= first["response_eof_wall_time"]
    asyncio.run(scenario())


@pytest.mark.parametrize("count,done", [(1, True), (2, False)])
def test_incomplete_predecessor_cannot_release_the_second_request(count, done):
    async def scenario():
        received = []

        async def handle(request):
            index = (await request.json())["index"]
            received.append(index)
            return web.Response(body=frames(index, count, done), content_type="text/event-stream")

        async with server(handle) as base:
            results, _ = await replay._submit_requests(
                base, [json.dumps({"index": i}).encode() for i in range(2)], [0., 0.],
                pace=True, timeout=2, endpoint="/v1/chat/completions", streaming=True,
                response_gated=True, expected_outputs=[2, 2])
        assert received == [0]
        assert not any(row["ok"] for row in results)
        assert "predecessor" in results[1]["error"]
    asyncio.run(scenario())


def test_virtual_opening_registers_both_before_either_response():
    async def scenario():
        received = []
        registered = asyncio.Event()

        async def handle(request):
            index = (await request.json())["index"]
            received.append(index)
            if len(received) == 2:
                registered.set()
            await registered.wait()
            return web.Response(body=frames(index), content_type="text/event-stream")

        async with server(handle) as base:
            results, _ = await asyncio.wait_for(replay._submit_requests(
                base, [json.dumps({"index": i}).encode() for i in range(2)], [0., 21.437],
                pace=False, timeout=2, endpoint="/v1/chat/completions", streaming=True,
                expected_outputs=[2, 2]), timeout=3)
        assert sorted(received) == [0, 1]
        assert all(row["ok"] for row in results)
    asyncio.run(scenario())
