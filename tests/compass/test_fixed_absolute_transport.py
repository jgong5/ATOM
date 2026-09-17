"""Real aiohttp/SSE submission over a CPU fake endpoint, with finite release gates."""

import asyncio
from contextlib import asynccontextmanager
import importlib.util
import json
from pathlib import Path

import pytest
from aiohttp import web

from .test_fixed_absolute import bundle, plan_file


spec = importlib.util.spec_from_file_location(
    "fixed_absolute_transport_replay", Path(__file__).resolve().parents[2] / "scripts/compass/replay.py")
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


@asynccontextmanager
async def endpoint(delays, *, fail_first=False):
    seen, active = [], set()
    peak = [0]

    async def request(req):
        body = await req.json()
        index = int(body["messages"][0]["content"].split()[-1])
        seen.append(index)
        active.add(index)
        peak[0] = max(peak[0], len(active))
        await asyncio.sleep(delays.get(index, .001))
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(req)
        value = {"id": f"r{index}", "choices": [{"finish_reason": "length"}],
                 "usage": {"prompt_tokens": 2, "completion_tokens": 1 if fail_first and index == 0 else 2}}
        await response.write(("data: " + json.dumps(value) + "\n\ndata: [DONE]\n\n").encode())
        await response.write_eof()
        active.remove(index)
        return response

    app = web.Application()
    app.router.add_post("/v1/chat/completions", request)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}", seen, peak
    finally:
        await runner.cleanup()


async def submit(base, plan):
    return await replay._submit_requests(base, plan.encode_payloads(declared=False),
        [r["arrival_s"] for r in plan.rows], pace=True, timeout=2.,
        endpoint="/v1/chat/completions", streaming=True,
        expected_outputs=[2] * len(plan.rows), fixed_absolute_plan=plan)


@pytest.mark.parametrize("child_delay", [.005, .09])
def test_join_waits_for_eof_and_original_due_before_submitting_once(tmp_path, child_delay):
    _, _, plan = plan_file(tmp_path, bundle(root_times=(0., .06), child_times=((.01,),), join=True))

    async def run():
        async with endpoint({2: child_delay}) as (base, seen, _):
            results, receipt = await submit(base, plan)
        assert seen == [0, 2, 1]
        assert all(r["ok"] for r in results)
        parent, child = results[1]["send_timing"], results[2]["send_timing"]
        expected = max(.06, child["finished_offset_s"])
        assert parent["causal_release_offset_s"] == expected
        assert parent["request_started_offset_s"] >= expected
        assert child["response_eof_wall_time"] <= child["client_response_returned_wall_time"]
        assert receipt["fixed_absolute"]["all_leaves_returned"]

    asyncio.run(run())


def test_root_client_budget_does_not_cap_parallel_descendant_requests(tmp_path):
    _, _, plan = plan_file(tmp_path, bundle(root_times=(0., .001), child_times=((.001,), (.001,))))

    async def run():
        async with endpoint({1: .04, 2: .04, 3: .04}) as (base, seen, peak):
            results, receipt = await submit(base, plan)
        assert len(seen) == len(set(seen)) == 4
        assert peak[0] == 3 > receipt["fixed_absolute"]["clients"]
        assert all(r["ok"] for r in results)
        assert set(receipt["fixed_absolute"]["root_complete_offsets"]) == {"root0"}

    asyncio.run(run())


def test_simultaneous_future_roots_keep_frozen_tie_order_after_reverse_returns(tmp_path):
    _, _, plan = plan_file(tmp_path, bundle(root_times=(0., .06), child_times=(), clients=2))

    async def run():
        async with endpoint({0: .025, 2: .001}) as (base, seen, _):
            results, _ = await submit(base, plan)
        assert len(seen) == len(set(seen)) == 4
        assert results[1]["send_timing"]["causal_release_offset_s"] == .06
        assert results[3]["send_timing"]["causal_release_offset_s"] == .06
        assert results[1]["send_timing"]["request_started_offset_s"] <= results[3]["send_timing"]["request_started_offset_s"]

    asyncio.run(run())


def test_failed_prerequisite_preserves_all_missing_leaf_outcomes_and_stops_dispatch(tmp_path):
    _, _, plan = plan_file(tmp_path, bundle(root_times=(0., .1), child_times=((.1,),)))

    async def run():
        async with endpoint({}, fail_first=True) as (base, seen, _):
            results, receipt = await submit(base, plan)
        assert seen == [0]
        assert len(results) == 3 and all(not r["ok"] for r in results)
        assert receipt["fixed_absolute"]["all_leaves_returned"] is False

    asyncio.run(run())
