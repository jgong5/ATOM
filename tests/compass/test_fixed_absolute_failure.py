"""Failed fixed-absolute prerequisites must not leave a finite replay hanging."""

import asyncio
import json

from aiohttp import web
import pytest

from atom.model_engine.sequence import SequenceStatus
from .test_fixed_absolute import bundle, plan_file
from .test_fixed_absolute_scheduler import configured
from .test_fixed_absolute_transport import endpoint, replay


@pytest.mark.parametrize("failure", ["unschedulable", "aborted"])
def test_waiting_failure_records_incomplete_calendar_without_releasing_descendants(tmp_path, failure):
    with configured(tmp_path) as (scheduler, clock, sequences, _):
        if failure == "unschedulable":
            scheduler.max_model_len = 1
        else:
            sequences[0].status = SequenceStatus.ABORTED
        scheduler.extend(sequences)
        batch, selected = scheduler.schedule()
        assert not batch.req_ids and not selected
        assert scheduler.take_rejected() == [sequences[0]]
        evidence = scheduler._release_calendar.evidence()
        assert evidence.get("failed") is True
        assert evidence["complete"] is False
        assert evidence["failures"][0]["index"] == 0
        assert failure in evidence["failures"][0]["reason"]
        assert not evidence["completions"] and not evidence["root_completed_at"]
        assert all(not scheduler._release_calendar.is_released(seq) for seq in sequences[1:])
        before = clock.time()
        for seq in sequences[1:]:
            seq.status = SequenceStatus.ABORTED
        batch, selected = scheduler.schedule()
        assert not batch.req_ids and not selected
        assert {seq.id for seq in scheduler.take_rejected()} == {seq.id for seq in sequences[1:]}
        assert scheduler.is_finished() and clock.time() == before
        assert scheduler._release_calendar.evidence()["complete"] is False


@pytest.mark.parametrize("failure", ["http", "short_output"])
def test_virtual_failure_cancels_preregistered_waiters_before_http_timeout(tmp_path, failure):
    _, _, plan = plan_file(tmp_path, bundle(root_times=(0., .1), child_times=((.1,),)))

    async def run():
        seen, all_registered, release_waiters = set(), asyncio.Event(), asyncio.Event()

        async def request(req):
            index = int((await req.json())["messages"][0]["content"].split()[-1])
            seen.add(index)
            if len(seen) == len(plan.rows):
                all_registered.set()
            await all_registered.wait()
            if index:
                await release_waiters.wait()
                return web.Response(status=499)
            if failure == "http":
                return web.Response(status=422, text="unschedulable predecessor")
            value = {"choices": [{"finish_reason": "length"}],
                     "usage": {"prompt_tokens": 2, "completion_tokens": 0}}
            return web.Response(text="data: " + json.dumps(value) + "\n\ndata: [DONE]\n\n",
                                content_type="text/event-stream")

        app = web.Application()
        app.router.add_post("/v1/chat/completions", request)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        task = asyncio.create_task(replay._submit_requests(
            f"http://127.0.0.1:{port}", plan.encode_payloads(declared=True),
            [row["arrival_s"] for row in plan.rows], pace=False, timeout=30.,
            endpoint="/v1/chat/completions", streaming=True, expected_outputs=[2] * len(plan.rows),
            fixed_absolute_plan=plan))
        try:
            await asyncio.wait_for(all_registered.wait(), timeout=1.)
            done, _ = await asyncio.wait([task], timeout=.25)
            assert task in done, "failed virtual predecessor left descendants waiting for HTTP timeout"
            results, receipt = task.result()
            assert seen == {0, 1, 2} and len(results) == 3
            assert all(not row["ok"] for row in results)
            assert all("cancelled" in row["error"] for row in results[1:])
            assert receipt["fixed_absolute"]["all_leaves_returned"] is False
            assert receipt["fixed_absolute"]["failed_index"] == 0
            assert receipt["fixed_absolute"]["release_owner"] == "virtual_engine_calendar"
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            release_waiters.set()
            await runner.cleanup()

    asyncio.run(run())


def test_virtual_success_keeps_engine_release_ownership(tmp_path):
    _, _, plan = plan_file(tmp_path)

    async def run():
        async with endpoint({}) as (base, seen, _):
            result, receipt = await asyncio.wait_for(replay._submit_requests(
                base, plan.encode_payloads(declared=True), [r["arrival_s"] for r in plan.rows],
                pace=False, timeout=30., endpoint="/v1/chat/completions", streaming=True,
                expected_outputs=[2] * len(plan.rows), fixed_absolute_plan=plan), timeout=1.)
        assert len(seen) == len(plan.rows) and all(row["ok"] for row in result)
        assert all("causal_release_offset_s" not in row["send_timing"] for row in result)
        assert receipt["fixed_absolute"]["release_owner"] == "virtual_engine_calendar"
        assert receipt["fixed_absolute"]["root_complete_offsets"] is None
        assert receipt["fixed_absolute"]["all_leaves_returned"] is True
        assert receipt["fixed_absolute"]["failed_index"] is None

    asyncio.run(run())
