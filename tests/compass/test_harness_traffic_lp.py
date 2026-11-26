# SPDX-License-Identifier: MIT
"""compass_harness's traffic LP against the Clock Authority.

The trace tests serve a Clock Authority in this process with an idle engine
LP and a stand-in frontend LP. The frontend thread also stands in for the
aiperf worker: it reads the published stamps, reports each stream event it
sends and hands each final event's credit return to the router. aiperf's
strategy is a stand-in that schedules the next turn a fixed delay after a
return, on a ``ClockPacedLoopScheduler`` paced by the traffic LP. The hold
tests replace the authority with a scripted runtime, so a return can be put
on either side of its final event's release. The stop tests serve a run's
recording authority over TCP and stop it as a SIGTERM does. The shutdown
tests run aiperf's own system controller, as plugin discovery resolves it,
on the router's messages. Skips by name as ``test_harness_pacing.py`` does.
"""

import asyncio
import contextlib
import heapq
import json
import math
import os
import re
import shutil
import socket
import threading
import time
import uuid
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace

import pytest


def _missing() -> str | None:
    try:
        metadata.version("compass-harness")
        import compass_harness  # noqa: F401  checks aiperf against its pinned source
    except metadata.PackageNotFoundError as e:
        return f"{e.name} is not installed"
    except RuntimeError as e:
        return str(e)
    return None


if _why := _missing():
    pytest.skip(
        f"needs agentx-harness 56a0cf70 and compass-harness: {_why}",
        allow_module_level=True,
    )

from aiperf.plugin import plugins  # plugin discovery, as in aiperf

# isort: split
import aiperf
import zmq
import zmq.asyncio
from aiperf.common.enums import (
    CreditPhase,
    LifecycleState,
    make_result_producer_capability,
)
from aiperf.common.hooks import AIPerfHook
from aiperf.common.messages import Message, RegisterServiceCommand
from aiperf.common.mixins import HooksMixin
from aiperf.controller.result_join_coordinator import ResultJoinCoordinator
from aiperf.controller.system_controller import SystemController
from aiperf.credit.sticky_router import StickyCreditRouter
from aiperf.plugin.enums import PluginType, ServiceType
from compass_harness.controller import CompassSystemController
from compass_harness.router import CompassCreditRouter, send_rank
from compass_harness.scheduler import ClockPacedLoopScheduler
from compass_harness.strategy import ClockPhaseRunner
from compass_harness.timesource import ClockTimer
from compass_harness.traffic_lp import (
    HTTP,
    INTAKE_JITTER_BYTES,
    INTAKE_JITTER_SIGMA,
    PREFILL_URL_ENV,
    STREAM,
    SendAfterCancel,
    TrafficLP,
    UnansweredRequests,
)
from compass_harness.transport import addresses, credit_key

from atom.compass import clock_transport
from atom.compass import run as compass_run
from atom.compass.clock import (
    NER,
    ClockAbort,
    ClockAuthority,
    LpId,
    prefill_decode_table,
    single_engine_table,
)
from atom.compass.ingress import Ingress
from atom.compass.run import ENV as RUN_ENV
from atom.utils.clock import LPRuntime, Straggler
from compass_harness import ADDRESS_ENV, fingerprint, traffic_lp

#: The run file's table keys, as the deployment writes them.
RUN = {"admission_s": 0.54e-3, "ipc_s": 2.0**-10, "stream_s": 2.0**-9}
TABLE = single_engine_table(**RUN)
FRONTEND, ENGINE = LpId("frontend"), LpId("engine")
INF = math.inf
ADMIT = TABLE.lookahead(HTTP)
STREAM_S = TABLE.lookahead(STREAM)
#: Each request's stream events, as offsets from its admission; the last is final.
EVENTS = (0.5, 1.0)
DELAY = 0.25  # the stand-in strategy's gap between a return and the next turn


def _credit(num: int, conversation: str | None = None, turn: int = 0):
    """Credit `num`, as aiperf numbers them, sending `conversation`'s `turn`:
    by default trace `num`'s first."""
    return SimpleNamespace(
        id=num,
        phase=CreditPhase.PROFILING,
        phase_index=0,
        conversation_id=f"trace-{num}" if conversation is None else conversation,
        turn_index=turn,
        cache_bust_marker=None,
    )


def _key(num: int) -> tuple:
    return credit_key(CreditPhase.PROFILING, 0, num)


class Frontend(threading.Thread):
    """The frontend LP, and the worker that carries its stream to the traffic LP."""

    def __init__(self, endpoint, prefix, loop, held, drop=()) -> None:
        super().__init__(name="frontend", daemon=True)
        self.endpoint, self.loop, self.held, self.drop = endpoint, loop, held, drop
        self.stamps, self.reports = addresses(prefix)
        self.rt = None

    def run(self) -> None:
        self.rt = rt = LPRuntime(
            FRONTEND, TABLE, clock_transport.connect(FRONTEND, self.endpoint)
        )
        rt.start_run()
        ctx = zmq.Context()
        sub = ctx.socket(zmq.SUB)
        sub.connect(self.stamps)
        sub.subscribe(b"")
        push = ctx.socket(zmq.PUSH)
        push.connect(self.reports)
        keys, handled, due = {}, set(), []
        while True:
            for seq in sorted(rt.released[HTTP] - handled):
                while seq not in keys:
                    key, _, stamped = sub.recv_pyobj()
                    keys[stamped] = key
                handled.add(seq)
                for i, offset in enumerate(EVENTS):
                    final = i == len(EVENTS) - 1
                    if not (final and keys[seq] in self.drop):
                        heapq.heappush(due, (rt.now + offset, keys[seq], final))
            while due and due[0][0] <= rt.now:
                _, key, final = heapq.heappop(due)
                arrival, seq = rt.stamp_send(STREAM)
                push.send_pyobj((key, seq, arrival, final))
                if final:
                    ret = SimpleNamespace(credit=_credit(key[2]), error=None)
                    self.loop.call_soon_threadsafe(
                        asyncio.ensure_future, self.held("worker_0", ret)
                    )
            if rt.next_event(due[0][0] if due else INF) == INF:
                rt.close()
                ctx.destroy(linger=0)
                return


async def _trace(endpoint, prefix, monkeypatch, *, requests, drop=(), logged=None):
    """Run `requests` turns, each `DELAY` after the last one's return."""
    comms = SimpleNamespace(create_pub_client=lambda address: None)
    monkeypatch.setattr(
        StickyCreditRouter, "__init__", lambda s, **kw: setattr(s, "comms", comms)
    )
    sent = []
    monkeypatch.setattr(
        StickyCreditRouter, "send_credit", lambda self, credit: _record(sent, credit)
    )
    logged = [] if logged is None else logged
    monkeypatch.setattr(
        CompassCreditRouter, "exception", lambda self, msg: logged.append(msg)
    )
    router = CompassCreditRouter(run=None, service_id="timing_manager")
    router._workers = {"worker_0": None}
    traffic = router.traffic
    scheduler = ClockPacedLoopScheduler()
    times = {}

    async def issue(num):
        times[num] = {"sent": traffic.now()}
        await router.send_credit(_credit(num))

    async def on_return(worker_id, ret):
        num = ret.credit.id
        times[num]["returned"] = traffic.now()
        if num + 1 < requests:
            scheduler.schedule_later(DELAY, issue(num + 1))
        else:
            router.mark_credits_complete()

    router.set_return_callback(on_return)
    frontend = Frontend(
        endpoint, prefix, asyncio.get_running_loop(), router._on_return_callback, drop
    )
    frontend.start()
    await router.wait_for_workers(5)
    await issue(0)
    try:
        # Only the router stops the loop when the finish names an open request.
        await (asyncio.Event().wait() if drop else traffic.done)
    finally:
        frontend.join(10)
    return SimpleNamespace(
        traffic=traffic, frontend=frontend, times=times, sent=sent, logged=logged
    )


async def _record(sent, credit) -> None:
    sent.append(credit.id)


@pytest.fixture
def ca(tmp_path, monkeypatch):
    endpoint = f"inproc:test-traffic-lp-{uuid.uuid4().hex}"
    server = clock_transport.serve(ClockAuthority(TABLE), endpoint)
    clock_transport.connect(ENGINE, endpoint).send((NER, INF, [], INF))
    run_file = tmp_path / "run.json"
    run_file.write_text(json.dumps({"clock_endpoint": endpoint, "bound_s": 1e6, **RUN}))
    monkeypatch.setenv(RUN_ENV, str(run_file))
    monkeypatch.setenv(ADDRESS_ENV, str(tmp_path / "lp"))
    monkeypatch.setattr(ClockPacedLoopScheduler, "clock", None)
    try:
        yield SimpleNamespace(endpoint=endpoint, prefix=str(tmp_path / "lp"))
    finally:
        server.close()


def _run(coro):
    """Run `coro`, failing rather than hanging when the run never finishes."""
    return asyncio.run(asyncio.wait_for(coro, 30))


def test_a_three_request_trace_ends_by_the_finish(ca, monkeypatch):
    run = _run(_trace(ca.endpoint, ca.prefix, monkeypatch, requests=3))
    assert run.traffic.rt.now == INF and run.frontend.rt.now == INF
    assert run.sent == [0, 1, 2] and run.logged == []
    print("\nrequest  sent    first token  final   next turn")
    sent = 0.0
    for num in range(3):
        admitted = sent + ADMIT
        first = admitted + EVENTS[0] + STREAM_S
        final = admitted + EVENTS[-1] + STREAM_S
        assert run.times[num] == {"sent": sent, "returned": final}
        print(f"{num:7d} {sent:7.4f} {first:12.4f} {final:7.4f} {final + DELAY:10.4f}")
        sent = final + DELAY


def test_a_dropped_response_is_named_at_the_finish(ca, monkeypatch):
    logged = []
    named = re.escape(
        "the run finished with requests [('profiling', 0, 1)] sent but never given "
        "their final response"
    )
    with pytest.raises(RuntimeError, match="Event loop stopped before Future"):
        _run(
            _trace(
                ca.endpoint,
                ca.prefix,
                monkeypatch,
                requests=3,
                drop={_key(1)},
                logged=logged,
            )
        )
    # The router logged it and stopped the loop, which fails the timing manager.
    (message,) = logged
    assert re.search(named, message)


class IdleEngine(threading.Thread):
    """The engine LP with nothing to run: its only clock wait is a daemon
    deadline every second, as its metrics push is. It records each finite grant."""

    def __init__(self, endpoint) -> None:
        super().__init__(name="engine", daemon=True)
        self.endpoint, self.grants = endpoint, []

    def run(self) -> None:
        rt = LPRuntime(ENGINE, TABLE, clock_transport.connect(ENGINE, self.endpoint))
        while (g := rt.next_event(INF, rt.now + 1.0)) != INF:
            self.grants.append(g)


async def _wait_for_return(endpoint, prefix, monkeypatch, grace, sent):
    """Send request 0 if `sent`, and wait for its return as aiperf's phase
    runner does, with a `grace` deadline; then finish. A pacing timer due past
    the deadline is pending meanwhile and cancelled after it, as the phase's
    end cancels a turn scheduled past it."""
    traffic = TrafficLP.from_env()
    monkeypatch.setattr(ClockPacedLoopScheduler, "clock", traffic)
    returned = asyncio.Event()

    async def on_return(worker_id, ret):
        returned.set()

    loop = asyncio.get_running_loop()
    frontend = Frontend(endpoint, prefix, loop, traffic.hold(on_return))
    frontend.start()
    await traffic.subscribed(1)
    if sent:
        traffic.send(_key(0))
    pacing = ClockTimer(grace + 100.0, lambda: None)
    # The deadline wait reads nothing else of its runner.
    timed_out = await ClockPhaseRunner._wait_for_event_with_timeout(
        SimpleNamespace(info=lambda message: None),
        name="profiling phase credits returned",
        event=returned,
        timeout=grace,
        task_to_cancel=None,
    )
    pacing.cancel()
    at = traffic.now()
    traffic.finish()
    await traffic.done
    frontend.join(10)
    return timed_out, at


#: Request 0's final event reaches the traffic LP here: the client run's end.
END = ADMIT + EVENTS[-1] + STREAM_S


@pytest.mark.parametrize(
    ("grace", "sent", "waited", "end"),
    [
        (300.0, True, (False, END), END),
        (0.75, True, (True, 0.75), END),
        (5.0, False, (True, 5.0), 5.0),
    ],
    ids=["withdrawn", "fired", "idle"],
)
def test_a_grace_deadline_keeps_no_lp_running_past_the_last_request_s_end(
    grace, sent, waited, end, tmp_path, monkeypatch
):
    """aiperf waits for request 0's return with a grace deadline while the
    idle engine ticks every second. At 300 s the return comes first and aiperf
    withdraws the deadline: every LP is granted ``+inf`` at the request's end,
    with no tick granted past it. At 0.75 s the deadline falls inside the
    request and fires there. With no request sent, the deadline is the only
    work, and the clock runs to it. The pacing timer due past the deadline
    moves no LP past it in any case."""
    authority = ClockAuthority(TABLE)
    endpoint = f"inproc:test-traffic-lp-{uuid.uuid4().hex}"
    server = clock_transport.serve(authority, endpoint)
    run_file = tmp_path / "run.json"
    run_file.write_text(json.dumps({"clock_endpoint": endpoint, "bound_s": 1e6, **RUN}))
    monkeypatch.setenv(RUN_ENV, str(run_file))
    monkeypatch.setenv(ADDRESS_ENV, str(tmp_path / "lp"))
    engine = IdleEngine(endpoint)
    engine.start()
    prefix = str(tmp_path / "lp")
    try:
        got = _run(_wait_for_return(endpoint, prefix, monkeypatch, grace, sent))
    finally:
        engine.join(10)
        server.close()
    assert got == waited
    assert max(clock for _, clock in authority.final_clocks) == end
    assert engine.grants == [float(k) for k in range(1, math.floor(end) + 1)]


def test_a_client_bound_at_a_stop_names_its_open_request(tmp_path):
    """The frontend, the engine and the traffic LP each hold a TCP connection,
    as a deployment's processes do. The frontend takes request 0, then stops
    the run; the engine takes its ``+inf`` and leaves, ending its connection."""
    run = {**RUN, "bound_s": 100.0, "client_runs": None, "out_dir": str(tmp_path)}
    authority = compass_run._RecordingAuthority(run)
    server = clock_transport.serve(authority, "tcp://127.0.0.1:0")

    def runtime(lp, cls=LPRuntime):
        return cls(lp, TABLE, clock_transport.connect(lp, server.endpoint))

    frontend = runtime(FRONTEND, compass_run._StoppingClock)
    engine = runtime(ENGINE)
    engine.ask_next_event(INF, 50.0)
    frontend.ask_next_event(INF)

    def stop():
        frontend.take_grant()  # request 0's admission
        frontend.end_run()  # the stop, as the frontend's event loop makes it
        engine.take_grant()
        engine.conn.close()  # the engine's process leaves

    async def client():
        traffic = TrafficLP(runtime(compass_run.TRAFFIC), str(tmp_path / "lp"))
        traffic.send(_key(0))
        await traffic.done

    stopper = threading.Thread(target=stop, daemon=True)
    stopper.start()
    try:
        with pytest.raises(UnansweredRequests) as raised:
            _run(client())
    finally:
        stopper.join(10)
        server.close()
    assert str(raised.value) == (
        f"the run finished with requests [{_key(0)}] sent but never given their "
        "final response"
    )
    assert engine.now == INF and not authority.stopped_whole


def test_a_client_stopped_mid_stream_names_its_open_request(tmp_path):
    """The frontend admits request 0 and sends its first stream event, which
    the traffic LP reads; the stop comes before the frontend asks for time
    again, so the authority has not released that event."""
    run = {**RUN, "bound_s": 100.0, "out_dir": str(tmp_path)}
    server = clock_transport.serve(
        compass_run._RecordingAuthority(run), "tcp://127.0.0.1:0"
    )
    prefix, paced = str(tmp_path / "lp"), STREAM_S / 2

    def connect(lp):
        return clock_transport.connect(lp, server.endpoint)

    traffic = LPRuntime(compass_run.TRAFFIC, TABLE, connect(compass_run.TRAFFIC))
    engine = connect(ENGINE)  # held: a closed connection ends the run
    engine.send((NER, INF, [], 50.0))

    def stop():
        frontend = compass_run._StoppingClock(FRONTEND, TABLE, connect(FRONTEND))
        frontend.next_event(INF)  # request 0's admission
        arrival, seq = frontend.stamp_send(STREAM)
        with zmq.Context() as ctx, ctx.socket(zmq.PUSH) as push:
            push.connect(addresses(prefix)[1])
            push.send_pyobj((_key(0), seq, arrival, False))  # its worker's report
            while seq not in traffic.arrived[STREAM]:
                time.sleep(0.01)
        frontend.end_run()  # the stop, as the frontend's event loop makes it

    async def client():
        lp = TrafficLP(traffic, prefix)
        lp.send(_key(0))
        while traffic.now < paced:  # a pacing timer, as the scheduler keeps one
            await lp.wait(paced)
        while 0 not in traffic.arrived[STREAM]:  # the hold waits for this task
            await asyncio.sleep(0)
        await lp.done

    stopper = threading.Thread(target=stop, daemon=True)
    stopper.start()
    try:
        with pytest.raises(UnansweredRequests, match=re.escape(f"[{_key(0)}]")):
            _run(client())
    finally:
        stopper.join(10)
        server.close()


class ScriptedRuntime:
    """The traffic LP's runtime with its grants scripted instead of asked for.

    Each script entry ``(G, seqs)`` is the next grant and the stream seqs it
    releases; a clock call whose target or daemon deadline comes first is
    granted that.
    """

    def __init__(self, script, stream=STREAM) -> None:
        self.now, self.script, self.seq = 0.0, list(script), 0
        self.seqs = {}
        self.stream, self.released = stream, {stream: set()}
        self.calls = []

    def stamp_send(self, ch, at=None):
        # Each channel numbers its own sends, as `LPRuntime.stamp_send` does.
        seq = self.seqs.get(ch, 0)
        self.seqs[ch] = seq + 1
        self.seq = max(self.seq, seq + 1)
        return self.now + ADMIT if at is None else at, seq

    def check_arrival(self, ch, arrival, seq) -> None:
        pass

    def next_event(self, t, t_daemon=INF):
        t = min(t, t_daemon)
        self.calls.append(t)
        if self.script and self.script[0][0] <= t:
            self.now, seqs = self.script.pop(0)
            self.released[self.stream] |= seqs
        else:
            self.now = t
        return self.now

    def close(self) -> None:
        pass


@pytest.mark.parametrize("order", ["before", "after"])
def test_a_return_before_or_after_its_final_release_schedules_from_the_final(
    order, tmp_path, monkeypatch
):
    """Credit 0's final event, stream seq 0, arrives at 1.0; its return is
    passed to aiperf at 1.0 whichever side of the release it reaches the hold.
    Credit 9 stays open throughout, so only the wait for aiperf to block keeps
    the owner from asking for time before the next turn is scheduled."""

    async def main():
        rt = ScriptedRuntime([(1.0, {0})])
        traffic = TrafficLP(rt, str(tmp_path / "lp"))
        monkeypatch.setattr(ClockPacedLoopScheduler, "clock", traffic)
        scheduler = ClockPacedLoopScheduler()
        sends = {}

        async def issue(num):
            sends[num] = traffic.now()
            traffic.send(_key(num))

        async def on_return(worker_id, ret):
            scheduler.schedule_later(DELAY, issue(ret.credit.id + 1))

        held = traffic.hold(on_return)
        push = zmq.Context.instance().socket(zmq.PUSH)
        push.connect(addresses(str(tmp_path / "lp"))[1])
        traffic.send(_key(9))
        await issue(0)
        if order == "after":
            while rt.now < 1.0:  # the release
                await asyncio.sleep(0.001)
        push.send_pyobj((_key(0), 0, 1.0, True))
        if order == "after":
            while 0 not in traffic._reports:  # the owner holds for the return
                await asyncio.sleep(0.001)
            await asyncio.sleep(0.01)
        await held("worker_0", SimpleNamespace(credit=_credit(0), error=None))
        while 1 not in sends:
            await asyncio.sleep(0.001)
        push.close(linger=0)
        if traffic.done.done():
            traffic.done.exception()  # UnansweredRequests: 9 and 1 are open
        else:
            traffic.done.cancel()
        return sends, rt.calls

    sends, calls = _run(main())
    assert sends == {0: 0.0, 1: 1.0 + DELAY}
    # Nothing asks for time between the release and the return.
    assert calls[:2] == [INF, 1.0 + DELAY]


@pytest.mark.parametrize("order", ["before", "after"])
def test_an_errored_return_counts_its_released_final_as_returned(order, tmp_path):
    """aiohttp can raise after ``[DONE]``: the return carries an error although
    the final event was reported. Either side of the release, it is passed once
    and the run reaches the finish."""

    async def main():
        rt = ScriptedRuntime([(1.0, {0})])
        traffic = TrafficLP(rt, str(tmp_path / "lp"))
        passed = []

        async def on_return(worker_id, ret):
            passed.append(ret.credit.id)

        held = traffic.hold(on_return)
        push = zmq.Context.instance().socket(zmq.PUSH)
        push.connect(addresses(str(tmp_path / "lp"))[1])
        traffic.send(_key(0))
        push.send_pyobj((_key(0), 0, 1.0, True))
        if order == "after":
            while rt.now < 1.0 or 0 not in traffic._reports:
                await asyncio.sleep(0.001)
        error = SimpleNamespace(credit=_credit(0), error="ClientPayloadError")
        await held("worker_0", error)
        traffic.finish()
        await asyncio.wait_for(traffic.done, 5)
        push.close(linger=0)
        return passed, rt.calls

    passed, calls = _run(main())
    assert passed == [0] and calls == [INF, INF]


def test_a_request_failing_mid_stream_fails_the_hold_by_name(tmp_path, monkeypatch):
    """Credit 0's worker reported seq 0, then its request failed: the released
    final, seq 1, is never reported."""
    monkeypatch.setattr("compass_harness.traffic_lp.DIAG_S", 0.05)

    async def main():
        rt = ScriptedRuntime([(1.0, {0, 1})])
        traffic = TrafficLP(rt, str(tmp_path / "lp"))

        async def on_return(worker_id, ret):
            pass

        held = traffic.hold(on_return)
        push = zmq.Context.instance().socket(zmq.PUSH)
        push.connect(addresses(str(tmp_path / "lp"))[1])
        traffic.send(_key(0))
        push.send_pyobj((_key(0), 0, 1.0, False))
        await held("worker_0", SimpleNamespace(credit=_credit(0), error="TimeoutError"))
        traffic.finish()
        try:
            await asyncio.wait_for(traffic.done, 5)
        finally:
            push.close(linger=0)

    with pytest.raises(UnansweredRequests) as failed:
        _run(main())
    print(f"\n{failed.value}")
    assert str(failed.value) == (
        "requests [('profiling', 0, 0)] failed mid-stream: released stream seqs [1] "
        "were not reported after 0.05 wall seconds"
    )


def test_a_slow_report_with_no_errored_request_keeps_the_hold_waiting(
    tmp_path, monkeypatch
):
    """Credit 0's worker reports its released final, seq 0, six ``DIAG_S``
    periods late and its request did not error: the hold waits for it."""
    monkeypatch.setattr("compass_harness.traffic_lp.DIAG_S", 0.05)

    async def main():
        rt = ScriptedRuntime([(1.0, {0})])
        traffic = TrafficLP(rt, str(tmp_path / "lp"))
        passed = []

        async def on_return(worker_id, ret):
            passed.append(ret.error)

        held = traffic.hold(on_return)
        push = zmq.Context.instance().socket(zmq.PUSH)
        push.connect(addresses(str(tmp_path / "lp"))[1])
        traffic.send(_key(0))
        await asyncio.sleep(0.3)
        push.send_pyobj((_key(0), 0, 1.0, True))
        await held("worker_0", SimpleNamespace(credit=_credit(0), error=None))
        traffic.finish()
        try:
            await asyncio.wait_for(traffic.done, 5)
        finally:
            push.close(linger=0)
        return passed, rt.calls

    passed, calls = _run(main())
    assert passed == [None] and calls[-1] == INF


def test_a_grant_resolves_only_the_clock_waits_it_makes_due(tmp_path):
    """Waits for 2 and 5, with a grant at 1 that comes before the time asked,
    as an arrival's does: each wait resolves at the grant that reaches its
    time, and each clock call asks for the earliest one pending."""

    async def main():
        rt = ScriptedRuntime([(1.0, set())])
        traffic = TrafficLP(rt, str(tmp_path / "lp"))
        resolved = {}

        async def waiter(t):
            await traffic.wait(t)
            resolved[t] = traffic.now()

        await asyncio.gather(waiter(2.0), waiter(5.0))
        traffic.done.cancel()
        return resolved, rt.calls

    resolved, calls = _run(main())
    assert resolved == {2.0: 2.0, 5.0: 5.0}
    assert calls == [2.0, 2.0, 5.0]


def test_due_waits_resolve_in_the_order_asked_and_a_withdrawn_one_asks_nothing(
    tmp_path,
):
    """At 5, waits are asked for 4, then 3, then 2, the last withdrawn: the
    first two are due at once and resolve in the order asked, not by time.
    Then waits for 7 and 6, the 6 withdrawn: the clock is asked for 7 alone."""

    async def main():
        rt = ScriptedRuntime([])
        rt.now = 5.0
        traffic = TrafficLP(rt, str(tmp_path / "lp"))
        order = []
        first, second, dropped = traffic.wait(4.0), traffic.wait(3.0), traffic.wait(2.0)
        for name, fut in (("4.0", first), ("3.0", second)):
            fut.add_done_callback(lambda _, name=name: order.append(name))
        traffic.withdraw(dropped)
        await asyncio.wait_for(asyncio.gather(first, second), 5)
        later, sooner = traffic.wait(7.0), traffic.wait(6.0)
        traffic.withdraw(sooner)
        await asyncio.wait_for(later, 5)
        traffic.done.cancel()
        return order, rt.calls, dict(traffic._waiters)

    order, calls, left = _run(main())
    assert order == ["4.0", "3.0"] and calls == [7.0] and left == {}


def test_the_hold_s_look_finds_the_tasks_asyncio_all_tasks_finds():
    """`_look` reads the loop's tasks in one pass; what it returns is what the
    same look over `asyncio.all_tasks` returns: the ready tasks, those not
    done, and whether one it saw last time has ended."""
    from compass_harness.traffic_lp import _look

    def reference(alive):
        me = asyncio.current_task()
        tasks = {t for t in asyncio.all_tasks() if t is not me and not t.done()}
        ready = sorted(
            t.get_name() for t in tasks if t._fut_waiter is None or t._fut_waiter.done()
        )
        return ready, tasks, alive is None or any(t.done() for t in alive)

    async def main():
        gate = asyncio.Event()
        blocked = [asyncio.ensure_future(gate.wait()) for _ in range(50)]
        await asyncio.sleep(0)
        looks = []
        ready, alive, ended = _look(None)
        looks.append(((sorted(ready), alive, ended), reference(None)))
        blocked[3].cancel()  # ends at its next step
        fresh = asyncio.ensure_future(asyncio.sleep(0))  # ready, not yet run
        ready, now_alive, ended = _look(alive)
        looks.append(((sorted(ready), now_alive, ended), reference(alive)))
        await asyncio.sleep(0)
        ready, later, ended = _look(now_alive)
        looks.append(((sorted(ready), later, ended), reference(now_alive)))
        gate.set()
        await asyncio.gather(*blocked, fresh, return_exceptions=True)
        return looks

    looks = _run(main())
    assert [ours for ours, _ in looks] == [theirs for _, theirs in looks]
    assert looks[1][0][2] is False and looks[2][0][2] is True  # the cancel ended one


def test_a_task_ending_while_the_hold_waits_on_a_return_has_its_callbacks_run_first(
    tmp_path,
):
    """aiperf's return callback yields, so the hold waits for it, then starts a
    task that ends at its first step, as a turn the replay barrier holds does:
    that task's done callbacks run before the traffic LP asks for time again."""

    async def main():
        rt = ScriptedRuntime([(1.0, {0})])
        traffic = TrafficLP(rt, str(tmp_path / "lp"))
        calls_at_done = []

        async def ends_at_once():
            pass

        async def on_return(worker_id, ret):
            await asyncio.sleep(0)
            ended = asyncio.ensure_future(ends_at_once())
            ended.add_done_callback(lambda _: calls_at_done.append(len(rt.calls)))

        push = zmq.Context.instance().socket(zmq.PUSH)
        push.connect(addresses(str(tmp_path / "lp"))[1])
        traffic.send(_key(0))
        push.send_pyobj((_key(0), 0, 1.0, True))
        await traffic.hold(on_return)(
            "worker_0", SimpleNamespace(credit=_credit(0), error=None)
        )
        traffic.finish()
        try:
            await asyncio.wait_for(traffic.done, 5)
        finally:
            push.close(linger=0)
        return calls_at_done

    # The final event's grant answers the first clock call; the task ends in its hold.
    assert _run(main()) == [1]


def test_a_task_ending_while_the_owner_idles_has_its_callbacks_run_first(tmp_path):
    """The owner idles with no request open, no clock wait and no finish; a
    task sends a request and ends in one step, so its done callbacks queue
    behind the owner's wake: they run before the traffic LP asks for time."""

    async def main():
        rt = ScriptedRuntime([(1.0, {0})])
        traffic = TrafficLP(rt, str(tmp_path / "lp"))
        calls_at_done = []
        await asyncio.sleep(0.05)  # the owner settles, then idles

        async def sends():
            traffic.send(_key(0))

        sent = asyncio.ensure_future(sends())
        sent.add_done_callback(lambda _: calls_at_done.append(len(rt.calls)))
        await asyncio.sleep(0.05)
        traffic.done.cancel()
        return calls_at_done

    assert _run(main()) == [0]


class ScriptedConn:
    """A Clock Authority connection whose grants are scripted: grant ``k``
    releases stream seq ``k`` at ``k + 1``, then ``+inf`` ends the run. Each
    clock call records how many released stream seqs the runtime still holds."""

    def __init__(self, grants: int) -> None:
        self.rt, self.grants, self.k, self.held_at_call = None, grants, 0, []

    def send(self, msg) -> None:
        self.held_at_call.append(len(self.rt.released[STREAM]))

    def recv(self):
        if self.k == self.grants:
            return INF, {}
        self.k += 1
        return float(self.k), {STREAM: [(self.k - 1, float(self.k))]}


def test_a_pass_scans_only_the_open_stream_events(tmp_path):
    """Many requests, each with one final stream event, run through a real
    runtime's grants: at every clock call the runtime holds no released event
    the traffic LP has already handled."""
    n = 200

    async def main():
        conn = ScriptedConn(n)
        conn.rt = rt = LPRuntime(LpId("traffic"), TABLE, conn)
        traffic = TrafficLP(rt, str(tmp_path / "lp"))
        passed = []

        async def on_return(worker_id, ret):
            passed.append(ret.credit.id)

        held = traffic.hold(on_return)
        push = zmq.Context.instance().socket(zmq.PUSH)
        push.connect(addresses(str(tmp_path / "lp"))[1])
        for num in range(n):
            traffic.send(_key(num))
            await held("worker_0", SimpleNamespace(credit=_credit(num), error=None))
            push.send_pyobj((_key(num), num, float(num + 1), True))
        traffic.finish()
        try:
            await asyncio.wait_for(traffic.done, 10)
        finally:
            push.close(linger=0)
        return passed, conn.held_at_call, rt

    passed, held_at_call, rt = _run(main())
    assert passed == list(range(n)) and rt.now == INF
    assert held_at_call == [0] * (n + 1)


class ReportedDuringTheCall(ScriptedConn):
    """Grant ``k`` releases stream seq ``k - 1`` of request 0, the last one
    final; that event's report reaches the traffic LP's socket while the owner
    is blocked in the clock call, as a worker's report usually does. Each grant
    also puts a marker on the event loop, and each clock call records how many
    markers have run: the loop's turns since the first grant."""

    def __init__(self, grants: int) -> None:
        super().__init__(grants)
        self.push = self.pull = None
        self.turns, self.turns_at_call = [], []

    def send(self, msg) -> None:
        super().send(msg)
        self.turns_at_call.append(len(self.turns))

    def recv(self):
        if self.k < self.grants:
            seq = self.k
            last = seq == self.grants - 1
            self.push.send_pyobj((_key(0), seq, float(seq + 1), last))
            while not self.pull.get(zmq.EVENTS) & zmq.POLLIN:
                time.sleep(0.001)
            asyncio.get_running_loop().call_soon(self.turns.append, seq)
        return super().recv()


def test_a_report_in_the_socket_at_its_grant_counts_without_a_turn_of_the_event_loop(
    tmp_path, monkeypatch
):
    """The owner takes each report already in its socket and asks for its next
    time with no turn of the event loop between, while a pacing timer waits on
    the clock for a time past every grant; only the final event's grant, whose
    return aiperf must take, waits for one."""
    n = 20

    async def main():
        conn = ReportedDuringTheCall(n)
        conn.rt = rt = LPRuntime(LpId("traffic"), TABLE, conn)
        traffic = TrafficLP(rt, str(tmp_path / "lp"))
        monkeypatch.setattr(ClockPacedLoopScheduler, "clock", traffic)
        scheduler = ClockPacedLoopScheduler()
        scheduler.schedule_later(10.0 * n, _nothing())
        conn.pull = traffic._pull
        conn.push = zmq.Context.instance().socket(zmq.PUSH)
        conn.push.connect(addresses(str(tmp_path / "lp"))[1])
        passed = []

        async def on_return(worker_id, ret):
            passed.append(ret.credit.id)

        traffic.send(_key(0))
        await traffic.hold(on_return)(
            "worker_0", SimpleNamespace(credit=_credit(0), error=None)
        )
        traffic.finish()
        try:
            await asyncio.wait_for(traffic.done, 10)
        finally:
            conn.push.close(linger=0)
            scheduler.cancel_all()
        return passed, conn.turns_at_call

    passed, turns_at_call = _run(main())
    assert passed == [0]
    # Call k follows grant k; the loop first turns in the final event's hold,
    # after grant n.
    assert turns_at_call == [0] * n + [n]


def test_a_raising_return_callback_does_not_stop_later_returns(tmp_path, caplog):
    async def main():
        traffic = TrafficLP(ScriptedRuntime([]), str(tmp_path / "lp"))
        passed = []

        async def on_return(worker_id, ret):
            passed.append(ret.credit.id)
            if ret.credit.id == 0:
                raise RuntimeError("boom")

        held = traffic.hold(on_return)
        for num in (0, 1):
            await held("w", SimpleNamespace(credit=_credit(num), error="refused"))
        while len(passed) < 2:
            await asyncio.sleep(0.001)
        traffic.done.cancel()
        return passed

    assert _run(main()) == [0, 1]
    assert "a return callback raised" in caplog.text


def test_a_second_send_credit_call_site_refuses_the_package(tmp_path):
    # Another gate shard may be writing bytecode into __pycache__ as this copies.
    shutil.copytree(
        Path(aiperf.__file__).parent,
        tmp_path / "aiperf",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    fingerprint.check(tmp_path)
    path = tmp_path / "aiperf" / "timing" / "intervals.py"
    path.write_text(
        path.read_text() + "\n\nasync def _resend(router, credit):\n"
        "    await router.send_credit(credit=credit)\n"
    )
    with pytest.raises(
        RuntimeError, match=re.escape("2 .send_credit( call sites, not 1")
    ):
        fingerprint.check(tmp_path)


def test_the_timing_manager_builds_the_compass_router():
    from aiperf.timing import manager

    assert manager.StickyCreditRouter is CompassCreditRouter


def test_a_1p1d_run_file_sends_to_frontend_p_and_reads_from_frontend_d(
    tmp_path, monkeypatch
):
    """Behind a prefill-decode router, requests enter at frontend-P and the
    stream comes back from frontend-D, over the channels the run file's table
    declares."""
    pd = {"router_s": 2.0**-12, "kv_write_req_s": 2.0**-10}
    endpoint = f"inproc:test-traffic-lp-{uuid.uuid4().hex}"
    table = prefill_decode_table(**RUN, **pd)
    server = clock_transport.serve(ClockAuthority(table), endpoint)
    run_file = tmp_path / "run.json"
    run_file.write_text(
        json.dumps({"clock_endpoint": endpoint, "bound_s": 1e6, **RUN, **pd})
    )
    monkeypatch.setenv(RUN_ENV, str(run_file))
    monkeypatch.setenv(ADDRESS_ENV, str(tmp_path / "lp"))

    async def main():
        traffic = TrafficLP.from_env()
        try:
            traffic.send(_key(0))
            traffic._stamp_sends()  # as the hold does once aiperf has caught up
            traffic.rt.check_arrival("frontend-D->traffic:stream", 1.0, 0)
            return traffic.rt.send_log, traffic.rt.unreleased
        finally:
            for task in (*traffic._tasks, traffic.done):
                task.cancel()
            traffic._pub.close(linger=0)
            traffic._pull.close(linger=0)

    try:
        sent, unreleased = _run(main())
    finally:
        server.close()
    http = "traffic->frontend-P:http"
    assert sent == [(http, 0, table.lookahead(http))]
    assert unreleased == {("frontend-D->traffic:stream", 0): 1.0}


def test_behind_the_router_lp_requests_cancels_and_the_stream_cross_it(
    tmp_path, monkeypatch
):
    """With a router LP the traffic LP sends to it and reads the stream from it,
    and aiperf's cancel of a request open at the grace deadline, 2.0, is
    stamped to it on the request's channel, for the request's worker."""
    pd = {"router_s": 2.0**-12, "kv_write_req_s": 2.0**-10, "router_in_s": 2.0**-14}
    endpoint = f"inproc:test-traffic-lp-{uuid.uuid4().hex}"
    server = clock_transport.serve(
        ClockAuthority(prefill_decode_table(**RUN, **pd)), endpoint
    )
    run_file = tmp_path / "run.json"
    run_file.write_text(
        json.dumps({"clock_endpoint": endpoint, "bound_s": 1e6, **RUN, **pd})
    )
    monkeypatch.setenv(RUN_ENV, str(run_file))
    monkeypatch.setenv(ADDRESS_ENV, str(tmp_path / "lp"))
    sent = []

    class Runtime(ScriptedRuntime):
        def stamp_send(self, ch):
            sent.append(ch)
            return super().stamp_send(ch)

    monkeypatch.setattr(
        traffic_lp,
        "LPRuntime",
        lambda *_: Runtime([(2.0, set())], "router->traffic:stream"),
    )

    async def main():
        traffic = TrafficLP.from_env()
        stamps, reports = addresses(str(tmp_path / "lp"))
        sub = zmq.asyncio.Context.instance().socket(zmq.SUB)
        sub.connect(stamps)
        sub.subscribe(b"")
        await traffic.subscribed(1)
        push = zmq.Context.instance().socket(zmq.PUSH)
        push.connect(reports)

        async def on_return(worker_id, ret):
            pass

        held = traffic.hold(on_return)

        async def worker():
            published = [await sub.recv_pyobj() for _ in range(2)]
            push.send_pyobj((_key(0),))
            await held("w", SimpleNamespace(credit=_credit(0), error=None))
            return published

        worker = asyncio.ensure_future(worker())
        traffic.send(_key(0))
        while traffic.now() < 2.0:
            await traffic.wait(2.0)
        await traffic.cancel()
        traffic.finish()
        try:
            await asyncio.wait_for(traffic.done, 5)
        finally:
            push.close(linger=0)
            sub.close(linger=0)
        return traffic.http, traffic.stream, await worker

    try:
        http, stream, published = _run(main())
    finally:
        server.close()
    assert (http, stream) == ("traffic->router:http", "router->traffic:stream")
    assert sent == [http, http]
    assert published[1] == (_key(0), 2.0 + ADMIT, 1, "cancel")


#: frontend-P's URL, as a launcher names it; frontend-D's, as deploy.sh's
#: decode instance states it in its server.json.
PREFILL_URL, DECODE_URL = "http://127.0.0.1:8010", "http://127.0.0.1:8020"


@pytest.mark.parametrize(
    "router_in_s, chunked, channels, to",
    [
        (
            None,
            False,
            ("traffic->frontend-P:http", "frontend-D->traffic:stream"),
            (PREFILL_URL,),
        ),
        (
            None,
            True,
            ("traffic->frontend-P:http", "frontend-D->traffic:stream"),
            (PREFILL_URL, DECODE_URL),
        ),
        (2.0**-14, False, ("traffic->router:http", "router->traffic:stream"), ()),
    ],
    ids=["plain-router", "plain-router-chunked", "router-lp"],
)
def test_only_behind_a_plain_router_a_cancel_carries_frontend_p_s_url(
    router_in_s, chunked, channels, to, tmp_path, monkeypatch
):
    """With frontend-P's URL named, aiperf's cancel of a request open at the
    grace deadline, 2.0, is stamped on the request's channel and published with
    that URL behind a PD router that is no LP, which answers a cancel 400, and
    without it behind the router LP, which hands it on. Under a chunked
    transfer the request was stamped on frontend-D's channel too, and its
    cancel names frontend-D's URL beside frontend-P's."""
    pd = {"router_s": 2.0**-12, "kv_write_req_s": 2.0**-10}
    if router_in_s is not None:
        pd["router_in_s"] = router_in_s
    more = {}
    if chunked:
        more = {compass_run.KV_CHUNKED: True, "out_dir": str(tmp_path)}
        (tmp_path / "decode").mkdir()
        server = {"url": DECODE_URL}
        (tmp_path / "decode" / "server.json").write_text(json.dumps(server))
    endpoint = f"inproc:test-traffic-lp-{uuid.uuid4().hex}"
    server = clock_transport.serve(
        ClockAuthority(prefill_decode_table(**RUN, **pd, chunked=chunked)), endpoint
    )
    run_file = tmp_path / "run.json"
    run_file.write_text(
        json.dumps({"clock_endpoint": endpoint, "bound_s": 1e6, **RUN, **pd, **more})
    )
    monkeypatch.setenv(RUN_ENV, str(run_file))
    monkeypatch.setenv(ADDRESS_ENV, str(tmp_path / "lp"))
    monkeypatch.setenv(PREFILL_URL_ENV, PREFILL_URL)
    stream = channels[1]
    monkeypatch.setattr(
        traffic_lp, "LPRuntime", lambda *_: ScriptedRuntime([(2.0, set())], stream)
    )

    async def main():
        traffic = TrafficLP.from_env()
        stamps, reports = addresses(str(tmp_path / "lp"))
        sub = zmq.asyncio.Context.instance().socket(zmq.SUB)
        sub.connect(stamps)
        sub.subscribe(b"")
        await traffic.subscribed(1)
        push = zmq.Context.instance().socket(zmq.PUSH)
        push.connect(reports)

        async def on_return(worker_id, ret):
            pass

        held = traffic.hold(on_return)

        async def worker():
            published = [await sub.recv_pyobj() for _ in range(2)]
            push.send_pyobj((_key(0),))
            await held("w", SimpleNamespace(credit=_credit(0), error=None))
            return published

        worker = asyncio.ensure_future(worker())
        traffic.send(_key(0))
        while traffic.now() < 2.0:
            await traffic.wait(2.0)
        await traffic.cancel()
        traffic.finish()
        try:
            await asyncio.wait_for(traffic.done, 5)
        finally:
            push.close(linger=0)
            sub.close(linger=0)
        return traffic.http, await worker

    try:
        http, published = _run(main())
    finally:
        server.close()
    assert http == channels[0]
    assert published[1] == (_key(0), 2.0 + ADMIT, 1, "cancel", *to)


def test_a_1p1d_hold_settles_the_frontend_d_stream(tmp_path):
    """Behind a prefill-decode router the hold reads frontend-D's stream: the
    released final is reported, its return passed, and the run finishes."""
    stream = "frontend-D->traffic:stream"

    async def main():
        rt = ScriptedRuntime([(1.0, {0})], stream)
        traffic = TrafficLP(rt, str(tmp_path / "lp"), stream=stream)
        passed = []

        async def on_return(worker_id, ret):
            passed.append(ret.credit.id)

        held = traffic.hold(on_return)
        push = zmq.Context.instance().socket(zmq.PUSH)
        push.connect(addresses(str(tmp_path / "lp"))[1])
        traffic.send(_key(0))
        push.send_pyobj((_key(0), 0, 1.0, True))
        await held("worker_0", SimpleNamespace(credit=_credit(0), error=None))
        traffic.finish()
        try:
            await asyncio.wait_for(traffic.done, 5)
        finally:
            push.close(linger=0)
        return passed, rt.released

    passed, released = _run(main())
    assert passed == [0] and released == {stream: set()}


def test_a_cancel_closes_the_open_requests_and_drops_their_later_events(tmp_path):
    """Credits 0 and 1 are open at the grace deadline, 2.0. The cancel stamps
    one http send per credit for its worker; their cancelled returns, which
    the worker sends 50 ms after writing the cancels, are passed before the
    next clock call. Credit 0's event read past the cancel is recorded with no
    arrival check, and the event released after it that no worker reports is
    dropped, so the run finishes with nothing open."""

    class Runtime(ScriptedRuntime):
        def next_event(self, t, t_daemon=INF):
            passed_at.append(list(passed))
            return super().next_event(t, t_daemon)

    passed, passed_at = [], []

    async def main():
        rt = Runtime([(1.0, {0}), (3.0, {1, 2})])
        traffic = TrafficLP(rt, str(tmp_path / "lp"))
        stamps, reports = addresses(str(tmp_path / "lp"))
        sub = zmq.asyncio.Context.instance().socket(zmq.SUB)
        sub.connect(stamps)
        sub.subscribe(b"")
        await traffic.subscribed(1)
        push = zmq.Context.instance().socket(zmq.PUSH)
        push.connect(reports)

        async def on_return(worker_id, ret):
            passed.append(ret.credit.id)

        held = traffic.hold(on_return)

        async def worker():
            published = [await sub.recv_pyobj() for _ in range(4)]
            for key, *_ in published[2:]:
                push.send_pyobj((key,))
            push.send_pyobj((_key(0), 1, 2.5, False))
            await asyncio.sleep(0.05)
            for num in (1, 0):
                await held("w", SimpleNamespace(credit=_credit(num), error=None))
            return published

        worker = asyncio.ensure_future(worker())
        traffic.send(_key(0))
        traffic.send(_key(1))
        push.send_pyobj((_key(0), 0, 1.0, False))
        while traffic.now() < 2.0:
            await traffic.wait(2.0)
        await traffic.cancel()
        traffic.finish()
        try:
            await asyncio.wait_for(traffic.done, 5)
        finally:
            push.close(linger=0)
            sub.close(linger=0)
        return await worker, rt.calls

    published, calls = _run(main())
    assert published[2:] == [
        (_key(0), 2.0 + ADMIT, 2, "cancel"),
        (_key(1), 2.0 + ADMIT, 3, "cancel"),
    ]
    assert calls == [2.0, 2.0, INF, INF]
    assert passed_at == [[], [], [0, 1], [0, 1]]


def test_returns_whose_finals_share_a_grant_pass_in_stream_order(tmp_path):
    """Credits 0 and 1 end in one grant, stream seqs 0 and 1. Credit 1's return
    reaches the hold first and credit 0's 50 ms later; both are passed in
    stream order, so the next turns aiperf schedules from them keep it."""

    async def main():
        rt = ScriptedRuntime([(1.0, {0, 1})])
        traffic = TrafficLP(rt, str(tmp_path / "lp"))
        passed = []

        async def on_return(worker_id, ret):
            passed.append(ret.credit.id)

        held = traffic.hold(on_return)
        push = zmq.Context.instance().socket(zmq.PUSH)
        push.connect(addresses(str(tmp_path / "lp"))[1])
        traffic.send(_key(0))
        traffic.send(_key(1))
        push.send_pyobj((_key(0), 0, 1.0, True))
        push.send_pyobj((_key(1), 1, 1.0, True))
        await held("w", SimpleNamespace(credit=_credit(1), error=None))
        while rt.now < 1.0 or 1 not in traffic._reports:
            await asyncio.sleep(0.001)
        await asyncio.sleep(0.05)
        await held("w", SimpleNamespace(credit=_credit(0), error=None))
        traffic.finish()
        try:
            await asyncio.wait_for(traffic.done, 5)
        finally:
            push.close(linger=0)
        return passed

    assert _run(main()) == [0, 1]


def _stamped(tmp_path, instants, seed=None, **lp) -> list[tuple]:
    """aiperf sends the turns ``instants[k]`` lists, each (conversation, turn),
    at ``k`` s in the order listed, numbering each one's credit as it sends it.
    Each stamp as a worker reads it, its turn in place of its key: (arrival,
    seq, conversation, turn), then ``"jitter", <bytes>`` if it has one, by seq.
    `lp`: more arguments for the traffic LP."""

    async def main():
        prefix = str(tmp_path / f"lp-{uuid.uuid4().hex[:8]}")
        # A stream event released at 100 s that no worker reports holds the
        # traffic LP there, its requests open, while the stamps are read.
        traffic = TrafficLP(ScriptedRuntime([(100.0, {0})]), prefix, seed=seed, **lp)
        sub = zmq.asyncio.Context.instance().socket(zmq.SUB)
        sub.connect(addresses(prefix)[0])
        sub.subscribe(b"")
        await traffic.subscribed(1)
        turns = {}
        for at, sends in enumerate(instants):
            while traffic.now() < at:
                await traffic.wait(float(at))
            for conversation, turn in sends:
                credit = _credit(len(turns), conversation, turn)
                turns[_key(credit.id)] = conversation, turn
                traffic.send(_key(credit.id), send_rank(credit))
        stamps = [await sub.recv_pyobj() for _ in turns]
        traffic.done.cancel()
        sub.close(linger=0)
        return sorted(
            (arrival, seq, *turns[key], *more) for key, arrival, seq, *more in stamps
        )

    return _run(main())


#: Turns aiperf sends at one instant, as (conversation, turn): two sessions'
#: next turns, a subagent's first and a new session's first.
BURST = (("s2", 9), ("s1", 4), ("s1::sa:0", 0), ("s3", 0))


def test_one_instant_s_turns_take_the_same_seqs_whatever_order_aiperf_sends_them_in(
    tmp_path,
):
    """aiperf sends one instant's turns in the order its tasks happen to run,
    which follows the wall clock, and numbers their credits in that order:
    each turn's seq follows its conversation and turn instead."""
    runs = [
        _stamped(tmp_path, [sends])
        for sends in (BURST, BURST[::-1], BURST[2:] + BURST[:2])
    ]
    assert runs[1] == runs[0] and runs[2] == runs[0]
    assert runs[0] == [
        (ADMIT, 0, "s1", 4),
        (ADMIT, 1, "s1::sa:0", 0),
        (ADMIT, 2, "s2", 9),
        (ADMIT, 3, "s3", 0),
    ]


def test_a_seed_draws_one_instant_s_order_the_same_whatever_order_aiperf_sends_in(
    tmp_path,
):
    """Seeds 1 and 2 each give the burst's turns an order of their own,
    neither the one their ranks give, whatever order aiperf sends them in; the
    seqs and their arrival stay as they were."""
    plain = _stamped(tmp_path, [BURST])
    one = [_stamped(tmp_path, [sends], "1") for sends in (BURST, BURST[::-1])]
    two = _stamped(tmp_path, [BURST], "2")
    assert one[0] == one[1]
    assert len({tuple(s[2:] for s in run) for run in (plain, one[0], two)}) == 3
    assert [s[:2] for s in plain] == [s[:2] for s in one[0]] == [s[:2] for s in two]


def _jitters(stamps) -> dict[tuple, float]:
    """Each turn's intake jitter in `_stamped`'s stamps, by (conversation, turn)."""
    assert all(s[4] == "jitter" and len(s) == 6 for s in stamps)
    return {(s[2], s[3]): s[5] for s in stamps}


def test_a_seed_gives_each_turn_one_intake_jitter_whatever_order_aiperf_sends_in(
    tmp_path,
):
    """Seed 1 gives each of the burst's turns an intake jitter its stamp
    carries, drawn by its conversation and turn alone: the stamps are the same,
    jitters and all, whichever order aiperf sends the turns in, and at
    whichever instant."""
    runs = [
        _stamped(tmp_path, [sends], "1")
        for sends in (BURST, BURST[::-1], BURST[2:] + BURST[:2])
    ]
    assert runs[1] == runs[0] and runs[2] == runs[0]
    later = _jitters(_stamped(tmp_path, [[("x", 0)], BURST], "1"))
    assert {k: later[k] for k in BURST} == _jitters(runs[0])
    print(f"\n  seed 1 jitters (bytes): {_jitters(runs[0])}")
    assert all(0 < j < math.inf for j in _jitters(runs[0]).values())


def test_each_seed_draws_its_own_jitters_and_no_seed_draws_none(tmp_path):
    """Seeds 1 and 2 give each turn a jitter of their own, so the frontend
    orders the burst two ways; with no seed, or a median of 0, no stamp carries
    one, and the frontend takes the exact intake order."""
    one, two = (_jitters(_stamped(tmp_path, [BURST], seed)) for seed in ("1", "2"))
    assert all(one[k] != two[k] for k in BURST)
    assert sorted(BURST, key=one.get) != sorted(BURST, key=two.get)
    for stamps in (
        _stamped(tmp_path, [BURST]),
        _stamped(tmp_path, [BURST], "1", jitter_median=0.0),
    ):
        assert [len(s) for s in stamps] == [4] * len(BURST)
    with pytest.raises(ValueError, match="no count of bytes"):
        TrafficLP(None, str(tmp_path / "lp-x"), seed="1", jitter_median=-1.0)


def test_the_intake_jitter_is_log_normal_about_the_median_it_names():
    """Over many turns a seed's jitters have the median asked for, 48 kB by
    default, and the real burst's spread of send offsets in their logs; each
    draw is the same every time, and apart from the seq order's draw."""
    lp = SimpleNamespace(seed="7", jitter_median=INTAKE_JITTER_BYTES)
    ranks = [(0, "profiling", f"t{i}", i % 5, "") for i in range(4000)]
    draws = [TrafficLP.jitter(lp, rank) for rank in ranks]
    assert draws == [TrafficLP.jitter(lp, rank) for rank in ranks]
    logs = [math.log(d) for d in draws]
    median = sorted(draws)[len(draws) // 2]
    sigma = (sum((x - sum(logs) / len(logs)) ** 2 for x in logs) / len(logs)) ** 0.5
    print(f"\n  median {median:.0f} B, log sigma {sigma:.3f}")
    assert median == pytest.approx(INTAKE_JITTER_BYTES, rel=0.06)
    assert sigma == pytest.approx(INTAKE_JITTER_SIGMA, rel=0.04)
    halved = SimpleNamespace(seed="7", jitter_median=INTAKE_JITTER_BYTES / 2)
    assert [TrafficLP.jitter(halved, r) for r in ranks[:5]] == pytest.approx(
        [d / 2 for d in draws[:5]]
    )
    assert (
        TrafficLP.jitter(SimpleNamespace(seed=None, jitter_median=1.0), ranks[0])
        is None
    )


def test_turns_sent_at_a_later_instant_take_later_seqs_seeded_or_not(tmp_path):
    """Traces z and y send a turn at 0 s, b and a at 1 s: each instant's turns
    are stamped at the end of its own hold, so their ranks order them only
    within it."""
    instants = [[("z", 3), ("y", 1)], [("b", 0), ("a", 5)]]
    assert _stamped(tmp_path, instants) == [
        (ADMIT, 0, "y", 1),
        (ADMIT, 1, "z", 3),
        (1.0 + ADMIT, 2, "a", 5),
        (1.0 + ADMIT, 3, "b", 0),
    ]
    seeded = _stamped(tmp_path, instants, "1")
    assert [s[:2] for s in seeded] == [
        (ADMIT, 0),
        (ADMIT, 1),
        (1.0 + ADMIT, 2),
        (1.0 + ADMIT, 3),
    ]
    assert {s[2] for s in seeded[:2]} == {"y", "z"}


def test_a_cancel_closes_one_instant_s_requests_in_their_seqs_order(tmp_path):
    """aiperf sends trace b's turn, then trace a's, at 0 s, and cancels both
    at the grace deadline, 2.0: a's request takes seq 0 and its cancel seq 2,
    b's seq 1 and 3, whichever aiperf sent first."""

    async def main():
        prefix = str(tmp_path / "lp")
        traffic = TrafficLP(ScriptedRuntime([(2.0, set())]), prefix)
        stamps, reports = addresses(prefix)
        sub = zmq.asyncio.Context.instance().socket(zmq.SUB)
        sub.connect(stamps)
        sub.subscribe(b"")
        await traffic.subscribed(1)
        push = zmq.Context.instance().socket(zmq.PUSH)
        push.connect(reports)

        async def worker():
            published = [await sub.recv_pyobj() for _ in range(4)]
            for key, *_ in published[2:]:
                push.send_pyobj((key,))
            return published

        worker = asyncio.ensure_future(worker())
        for num, conversation in enumerate("ba"):
            traffic.send(_key(num), send_rank(_credit(num, conversation)))
        while traffic.now() < 2.0:
            await traffic.wait(2.0)
        await traffic.cancel()
        published = await worker
        traffic.done.cancel()  # the hold waits for their returns
        push.close(linger=0)
        sub.close(linger=0)
        return published

    a, b = _key(1), _key(0)
    assert _run(main()) == [
        (a, ADMIT, 0),
        (b, ADMIT, 1),
        (a, 2.0 + ADMIT, 2, "cancel"),
        (b, 2.0 + ADMIT, 3, "cancel"),
    ]


def test_every_cancel_of_a_burst_reaches_a_worker_that_lags_behind_it(tmp_path):
    """10,000 requests are open at the grace deadline, their send stamps read
    as each instant's are published. The worker reads nothing while the cancel
    burst is published, then reports each cancel it gets written: every one
    reaches it and `cancel` returns. Two c=512 runs on a loaded node hung
    there (#3819): PUB's high-water mark dropped the last 114 of 769 cancels
    to every worker, whose cancels were never written, so `cancel`, the hold
    and the clock waited forever."""
    n, per_instant = 10_000, 400

    async def main():
        prefix = str(tmp_path / "lp")
        traffic = TrafficLP(ScriptedRuntime([]), prefix)
        stamps, reports = addresses(prefix)
        # A blocking socket: while the worker reads, the loop, and with it the
        # owner's clock calls, waits.
        sub = zmq.Context.instance().socket(zmq.SUB)
        sub.setsockopt(zmq.RCVTIMEO, 2000)
        sub.connect(stamps)
        sub.subscribe(b"")
        await traffic.subscribed(1)
        push = zmq.Context.instance().socket(zmq.PUSH)
        push.setsockopt(zmq.SNDHWM, 0)  # the reports wait for the loop, unread
        push.connect(reports)
        try:
            for first in range(0, n, per_instant):
                for num in range(first, first + per_instant):
                    traffic.send(_key(num))
                t = 1.0 + first // per_instant
                while traffic.now() < t:
                    await traffic.wait(t)
                for _ in range(per_instant):
                    sub.recv_pyobj()
            cancelling = asyncio.ensure_future(traffic.cancel())
            while not traffic.cancelled:  # the burst is out once they are closed
                await asyncio.wait({cancelling}, timeout=0.05)
            got = []
            with contextlib.suppress(zmq.Again):
                while len(got) < n:
                    key, *_, kind = sub.recv_pyobj()
                    got.append((key, kind))
                    push.send_pyobj((key,))
            # Not one lost, or `cancel` waits for its report forever.
            assert got == [(_key(num), "cancel") for num in range(n)]
            await asyncio.wait_for(cancelling, 10)
        finally:
            traffic.done.cancel()  # the hold waits for their returns
            push.close(linger=0)
            sub.close(linger=0)

    _run(main())


def test_a_cancel_not_written_is_named_while_it_waits(tmp_path, monkeypatch, caplog):
    """A cancel its worker never reports written keeps `cancel` waiting, and
    the log names it every ``DIAG_S`` wall seconds."""
    monkeypatch.setattr(traffic_lp, "DIAG_S", 0.05)

    async def main():
        traffic = TrafficLP(ScriptedRuntime([]), str(tmp_path / "lp"))
        push = zmq.Context.instance().socket(zmq.PUSH)
        push.connect(addresses(str(tmp_path / "lp"))[1])
        traffic.send(_key(0))
        traffic.send(_key(1))
        while traffic.now() < 2.0:
            await traffic.wait(2.0)
        cancelling = asyncio.ensure_future(traffic.cancel())
        push.send_pyobj((_key(0),))
        try:
            await asyncio.wait({cancelling}, timeout=0.5)
            return cancelling.done()
        finally:
            cancelling.cancel()
            traffic.done.cancel()
            push.close(linger=0)

    assert not _run(main())
    named = [r.getMessage() for r in caplog.records if "cancels of" in r.getMessage()]
    assert named and f"cancels of [{_key(1)}] not reported written" in named[-1]


def test_a_cancel_waits_for_a_released_final_s_return_before_it_stamps(tmp_path):
    """Credits 0 and 1 are open; the grant at the grace deadline, 2.0, releases
    credit 0's final, whose return reaches the hold 50 ms after aiperf cancels.
    The cancel waits for that return, so credit 0 completes and only credit 1
    gets a cancel."""

    async def main():
        rt = ScriptedRuntime([(2.0, {0})])
        traffic = TrafficLP(rt, str(tmp_path / "lp"))
        stamps, reports = addresses(str(tmp_path / "lp"))
        sub = zmq.asyncio.Context.instance().socket(zmq.SUB)
        sub.connect(stamps)
        sub.subscribe(b"")
        await traffic.subscribed(1)
        push = zmq.Context.instance().socket(zmq.PUSH)
        push.connect(reports)
        passed, cancels = [], []

        async def on_return(worker_id, ret):
            passed.append(ret.credit.id)

        held = traffic.hold(on_return)

        async def worker():
            while True:
                key, *_, kind = await sub.recv_pyobj()
                if kind == "cancel":
                    cancels.append(key)
                    push.send_pyobj((key,))
                    ret = SimpleNamespace(credit=_credit(key[2]), error=None)
                    await held("w", ret)

        async def late_return():
            await asyncio.sleep(0.05)
            await held("w", SimpleNamespace(credit=_credit(0), error=None))

        worker = asyncio.ensure_future(worker())
        traffic.send(_key(0))
        traffic.send(_key(1))
        push.send_pyobj((_key(0), 0, 2.0, True))
        while traffic.now() < 2.0:
            await traffic.wait(2.0)
        late = asyncio.ensure_future(late_return())
        await traffic.cancel()
        traffic.finish()
        try:
            await asyncio.wait_for(traffic.done, 5)
        finally:
            worker.cancel()
            late.cancel()
            push.close(linger=0)
            sub.close(linger=0)
        return cancels, passed

    cancels, passed = _run(main())
    assert cancels == [_key(1)]
    assert passed == [0, 1]


#: An encoding ingress of two threads, its terms dyadic so its sums are exact.
INGRESS = {
    "fixed_s": 0.5,
    "per_byte_s": 2.0**-10,
    "workers": 2,
    "router_per_byte_s": 2.0**-13,
}


async def _behind_an_ingress(tmp_path, sizes, script, then=None):
    """aiperf sends credits 0, 1, ... at 0 s, one a trace, behind `INGRESS`;
    no stamp is published before every size is reported, which a worker does
    for credit k once 0.1 s has passed, with `sizes[k]` bytes, None for an
    errored return in its place. `then(traffic, sub, push)` runs after the
    stamps are read. The stamps, by seq, and what `then` returns."""
    prefix = str(tmp_path / f"lp-{uuid.uuid4().hex[:8]}")
    rt = ScriptedRuntime(script)
    traffic = TrafficLP(rt, prefix, ingress=Ingress(INGRESS, ADMIT))
    stamps, reports = addresses(prefix)
    sub = zmq.asyncio.Context.instance().socket(zmq.SUB)
    sub.connect(stamps)
    sub.subscribe(b"")
    await traffic.subscribed(1)
    push = zmq.Context.instance().socket(zmq.PUSH)
    push.connect(reports)

    async def on_return(worker_id, ret):
        pass

    held = traffic.hold(on_return)
    for num in range(len(sizes)):
        traffic.send(_key(num), send_rank(_credit(num)))
    try:
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(sub.recv_pyobj(), 0.1)
        for num, size in enumerate(sizes):
            if size is None:
                await held("w", SimpleNamespace(credit=_credit(num), error="refused"))
            else:
                push.send_pyobj((_key(num), size))
        published = sorted([await sub.recv_pyobj() for _ in sizes], key=lambda s: s[2])
        more = then and await then(traffic, sub, push)
    finally:
        traffic.done.cancel()
        push.close(linger=0)
        sub.close(linger=0)
    return published, more


def test_behind_an_ingress_requests_reach_the_router_past_its_threads(tmp_path):
    """Credits 0, 1 and 2 at 0 s with 1, 2 and 0.5 KiB bodies: none is
    stamped before its worker reports its size; each reaches the ingress at the
    channel's lookahead, where its record starts, and the receiver once a
    thread has held it 0.5 s plus 1 s a KiB and the router has parsed it, 1/8 s
    a KiB. The third waits for the first thread to come free."""
    published, _ = _run(_behind_an_ingress(tmp_path, [1024, 2048, 512], [(100.0, {0})]))
    first = ADMIT + 0.5 + 1.0  # the first thread comes free
    expected = [first + 0.125, ADMIT + 0.5 + 2.0 + 0.25, first + 0.5 + 0.5 + 0.0625]
    assert published == [
        (_key(num), at, num, "start", ADMIT) for num, at in enumerate(expected)
    ]


def test_behind_an_ingress_a_request_that_failed_unsent_takes_no_thread(tmp_path):
    """Credit 0's request failed before its worker sent it, so no size comes:
    its return lets the hold stamp it at the lookahead, and credit 1 takes the
    first thread."""
    published, _ = _run(_behind_an_ingress(tmp_path, [None, 1024], [(100.0, {0})]))
    assert published == [
        (_key(0), ADMIT, 0, "start", ADMIT),
        (_key(1), ADMIT + 0.5 + 1.0 + 0.125, 1, "start", ADMIT),
    ]


def test_behind_an_ingress_a_cancel_arrives_no_earlier_than_its_request(tmp_path):
    """aiperf cancels credits 0 and 1 at its grace deadline, 2.0: credit 1,
    with 4 KiB, reaches the router about 5.0 s, so its cancel does too, one seq
    later; credit 0 was there by then, and its cancel takes the lookahead. The
    grant at 2.0 releases an event of credit 0's, which holds the traffic LP at
    2.0 until it is reported, once aiperf is cancelling."""

    async def then(traffic, sub, push):
        while traffic.now() < 2.0:
            await asyncio.sleep(0.001)
        cancelling = asyncio.ensure_future(traffic.cancel())
        await asyncio.sleep(0)  # the cancel waits for the hold
        push.send_pyobj((_key(0), 0, 2.0, False))
        cancels = [await sub.recv_pyobj() for _ in range(2)]
        for key, *_ in cancels:
            push.send_pyobj((key,))
        await cancelling
        return cancels

    published, cancels = _run(
        _behind_an_ingress(tmp_path, [64, 4096], [(2.0, {0})], then)
    )
    last = ADMIT + 0.5 + 4.0 + 0.5
    assert [at for _, at, *_ in published] == [ADMIT + 0.5 + 0.0625 + 2.0**-7, last]
    assert cancels == [
        (_key(0), 2.0 + ADMIT, 2, "cancel"),
        (_key(1), last, 3, "cancel"),
    ]


def test_a_run_file_with_an_ingress_gives_the_traffic_lp_its_queue(
    tmp_path, monkeypatch
):
    """Behind the router LP the ingress sits in front of it, its lookahead the
    channel's into the router; the arrival the runtime logs is past the hop.
    A run file with none builds none, and its send is the lookahead's."""
    pd = {"router_s": 2.0**-12, "kv_write_req_s": 2.0**-10, "router_in_s": 2.0**-14}
    table = prefill_decode_table(**RUN, **pd)
    http = "traffic->router:http"
    sent = {}
    for name, more in (("plain", {}), ("ingress", {"ingress": INGRESS})):
        endpoint = f"inproc:test-traffic-lp-{uuid.uuid4().hex}"
        server = clock_transport.serve(ClockAuthority(table), endpoint)
        run_file = tmp_path / f"{name}.json"
        run = {"clock_endpoint": endpoint, "bound_s": 1e6, **RUN, **pd, **more}
        run_file.write_text(json.dumps(run))
        monkeypatch.setenv(RUN_ENV, str(run_file))
        monkeypatch.setenv(ADDRESS_ENV, str(tmp_path / f"lp-{name}"))

        async def main():
            traffic = TrafficLP.from_env()
            try:
                traffic.send(_key(0))
                traffic._take((_key(0), 1024))  # its worker's report
                traffic._stamp_sends()  # as the hold does once aiperf has caught up
                return traffic.ingress, traffic.rt.send_log
            finally:
                for task in (*traffic._tasks, traffic.done):
                    task.cancel()
                traffic._pub.close(linger=0)
                traffic._pull.close(linger=0)

        try:
            sent[name] = _run(main())
        finally:
            server.close()
    assert sent["plain"] == (None, [(http, 0, table.lookahead(http))])
    ingress, log = sent["ingress"]
    assert ingress.lookahead_s == table.lookahead(http)
    assert log == [(http, 0, table.lookahead(http) + 0.5 + 1.0 + 0.125)]


class StragglerRuntime(ScriptedRuntime):
    """``check_arrival`` with ``LPRuntime``'s rule: a report of an unreleased seq
    whose arrival the LP has already passed is a straggler."""

    def check_arrival(self, ch, arrival, seq) -> None:
        if seq not in self.released[ch] and arrival < self.now:
            raise Straggler(f"{ch} seq {seq} arrives at {arrival}, now {self.now}")


def test_a_cancelled_request_s_report_read_after_its_seq_is_dropped(tmp_path):
    """Credits 0 and 1 are cancelled at 2.0 and their returns passed. The grant
    at 3.0 releases credit 0's stream seq 1 with no request open, so the hold
    drops it; its report (arrival 2.5) is read after that and the run
    finishes."""

    async def main():
        rt = StragglerRuntime([(1.0, {0}), (3.0, {1})])
        traffic = TrafficLP(rt, str(tmp_path / "lp"))
        stamps, reports = addresses(str(tmp_path / "lp"))
        sub = zmq.asyncio.Context.instance().socket(zmq.SUB)
        sub.connect(stamps)
        sub.subscribe(b"")
        await traffic.subscribed(1)
        push = zmq.Context.instance().socket(zmq.PUSH)
        push.connect(reports)
        passed = []

        async def on_return(worker_id, ret):
            passed.append(ret.credit.id)

        held = traffic.hold(on_return)

        async def worker():
            published = [await sub.recv_pyobj() for _ in range(4)]
            for key, *_ in published[2:]:
                push.send_pyobj((key,))
            await asyncio.sleep(0.05)
            for num in (0, 1):
                await held("w", SimpleNamespace(credit=_credit(num), error=None))

        worker = asyncio.ensure_future(worker())
        traffic.send(_key(0))
        traffic.send(_key(1))
        push.send_pyobj((_key(0), 0, 1.0, False))
        while traffic.now() < 2.0:
            await traffic.wait(2.0)
        await traffic.cancel()
        await worker
        while traffic.now() < 3.0:
            await traffic.wait(3.0)
        push.send_pyobj((_key(0), 1, 2.5, False))
        await asyncio.sleep(0.1)
        traffic.finish()
        try:
            await asyncio.wait_for(traffic.done, 5)
        finally:
            push.close(linger=0)
            sub.close(linger=0)
        return passed

    assert _run(main()) == [0, 1]


def test_a_send_after_a_cancel_fails_the_run_by_name(tmp_path, monkeypatch):
    """Credit 0 is cancelled at 2.0, the grace deadline of a phase before the
    last, and aiperf then sends credit 5. Credit 0's worker stopped reading, so
    the grant at 3.0 releases its stream seq 0 and no one reports it: the send
    is refused and the run fails naming both, instead of holding for seq 0."""
    monkeypatch.setattr("compass_harness.traffic_lp.DIAG_S", 0.05)

    async def main():
        rt = ScriptedRuntime([(3.0, {0})])
        traffic = TrafficLP(rt, str(tmp_path / "lp"))
        stamps, reports = addresses(str(tmp_path / "lp"))
        sub = zmq.asyncio.Context.instance().socket(zmq.SUB)
        sub.connect(stamps)
        sub.subscribe(b"")
        await traffic.subscribed(1)
        push = zmq.Context.instance().socket(zmq.PUSH)
        push.connect(reports)

        async def on_return(worker_id, ret):
            pass

        held = traffic.hold(on_return)

        async def worker():
            _, (key, *_) = [await sub.recv_pyobj() for _ in range(2)]
            push.send_pyobj((key,))
            await held("w", SimpleNamespace(credit=_credit(0), error=None))

        worker = asyncio.ensure_future(worker())
        traffic.send(_key(0))
        while traffic.now() < 2.0:
            await traffic.wait(2.0)
        await traffic.cancel()
        await worker
        refused = None
        try:
            traffic.send(_key(5))
        except SendAfterCancel as e:
            refused = e
        try:
            with pytest.raises(SendAfterCancel) as failed:
                await asyncio.wait_for(traffic.done, 2)
        finally:
            push.close(linger=0)
            sub.close(linger=0)
        return refused, failed.value

    refused, failed = _run(main())
    assert failed is refused
    assert str(refused) == (
        f"request {_key(5)} sent after a cancel closed requests [{_key(0)}]: their "
        f"stream seqs released later on {STREAM} are never reported, so the hold "
        "would wait forever"
    )


def test_a_report_the_reader_rejects_fails_the_run(tmp_path):
    """Credit 0 is open and its stream seq 0 released at 1.0, unreported. A
    report of seq 1 with arrival 0.5, read after that grant, is a straggler:
    the run fails with it instead of holding for seq 0 forever."""

    async def main():
        rt = StragglerRuntime([(1.0, {0})])
        traffic = TrafficLP(rt, str(tmp_path / "lp"))
        push = zmq.Context.instance().socket(zmq.PUSH)
        push.connect(addresses(str(tmp_path / "lp"))[1])
        traffic.send(_key(0))
        while traffic.now() < 1.0:
            await traffic.wait(1.0)
        push.send_pyobj((_key(0), 1, 0.5, False))
        try:
            await asyncio.wait_for(traffic.done, 5)
        finally:
            push.close(linger=0)

    with pytest.raises(Straggler, match="seq 1 arrives at 0.5, now 1.0"):
        _run(main())


def test_a_failed_traffic_lp_tells_aiperf_the_run_failed_before_it_stops(
    tmp_path, monkeypatch
):
    """The reader rejects a report, as above, under the router. Before the loop
    stops, the router publishes an error the controller keeps as an exit error,
    then the cancel the records manager finishes on, whose log line names the
    error."""
    from aiperf.common.messages import ProfileCancelCommand
    from aiperf.controller.result_join_coordinator import ResultJoinCoordinator
    from aiperf.controller.system_controller import SystemController
    from aiperf.records.records_manager import RecordsManager

    sent = []

    class Bus:
        async def publish(self, message):
            message.to_json_bytes()  # as the pub client sends it
            sent.append(message)

    def init(self, **kw):
        self.comms = SimpleNamespace(create_pub_client=lambda address: Bus())

    monkeypatch.setattr(StickyCreditRouter, "__init__", init)
    monkeypatch.setattr(CompassCreditRouter, "exception", lambda self, msg: None)

    async def main():
        rt = StragglerRuntime([(1.0, {0})])
        lp = str(tmp_path / "lp")
        monkeypatch.setattr(TrafficLP, "from_env", classmethod(lambda cls: cls(rt, lp)))
        traffic = CompassCreditRouter(run=None, service_id="timing_manager").traffic
        push = zmq.Context.instance().socket(zmq.PUSH)
        push.connect(addresses(lp)[1])
        traffic.send(_key(0))
        while traffic.now() < 1.0:
            await traffic.wait(1.0)
        push.send_pyobj((_key(0), 1, 0.5, False))
        try:
            await asyncio.Event().wait()  # only the router stops the loop
        finally:
            push.close(linger=0)

    with pytest.raises(RuntimeError, match="Event loop stopped before Future"):
        _run(main())
    error, cancel = sent
    assert isinstance(cancel, ProfileCancelCommand)
    assert cancel.service_id == error.service_id == "timing_manager"

    async def shutdown_check():
        pass

    controller = SimpleNamespace(
        _exit_errors=[],
        _result_join_coordinator=ResultJoinCoordinator(),
        error=lambda msg: None,
        _check_and_trigger_shutdown=shutdown_check,
    )
    asyncio.run(SystemController._process_service_error_message(controller, error))
    (exit_error,) = controller._exit_errors
    assert exit_error.error_details.type == "Straggler"
    assert "seq 1 arrives at 0.5, now 1.0" in exit_error.error_details.message

    async def process_results(phase, cancelled):
        pass

    warned = []
    records = SimpleNamespace(
        warning=warned.append,
        _records_tracker=SimpleNamespace(mark_phase_cancelled=lambda phase: None),
        _process_results=process_results,
    )
    asyncio.run(RecordsManager._on_profile_cancel_command(records, cancel))
    (line,) = warned
    assert re.search(
        r'"reason_detail":"traffic LP: Straggler\(.* seq 1 arrives at 0.5, now 1.0',
        line,
    )


#: The records manager's results, as the controller's handler reads them.
RESULTS = SimpleNamespace(
    service_id="records_manager",
    results=SimpleNamespace(
        errors=[],
        fatal_errors=[],
        results=SimpleNamespace(records=["a record"], error_summary=None),
    ),
)


async def _nothing(*args) -> None:
    pass


async def _controller(monkeypatch, stop):
    """aiperf's system controller as plugin discovery resolves it, past each
    service's registration and the configure step; the records manager
    advertises ``profile``. Its own handlers and shutdown barrier run, and it
    stops by `stop`."""
    cls = plugins.get_class(PluginType.SERVICE, ServiceType.SYSTEM_CONTROLLER)
    ctl = cls.__new__(cls)
    HooksMixin.__init__(ctl)  # its hooks and logger, as BaseService builds them
    ctl.service_id, ctl.stop, ctl._exit_errors = "system_controller", stop, []
    ctl._processed_command_ids, ctl.publish = set(), _nothing
    ctl.service_manager = SimpleNamespace(service_id_map={}, service_map={})
    ctl._result_join_coordinator = ResultJoinCoordinator()
    ctl._shutdown_lock, ctl._shutdown_triggered = asyncio.Lock(), False
    for name, produces in (("records_manager", ("profile",)), ("timing_manager", ())):
        await ctl._handle_register_service_command(
            RegisterServiceCommand(
                service_id=name,
                service_type=ServiceType(name),
                state=LifecycleState.RUNNING,
                capabilities=tuple(map(make_result_producer_capability, produces)),
            )
        )
    monkeypatch.setattr(SystemController, "_profile_configure_all_services", _nothing)
    await ctl._profile_configure_all_services()
    return ctl


def _router_bus(monkeypatch, publish) -> None:
    """Each router built from here publishes by `publish` and logs nothing."""
    comms = SimpleNamespace(
        create_pub_client=lambda a: SimpleNamespace(publish=publish)
    )
    monkeypatch.setattr(
        StickyCreditRouter, "__init__", lambda s, **kw: setattr(s, "comms", comms)
    )
    monkeypatch.setattr(CompassCreditRouter, "exception", lambda self, msg: None)


async def _deliver(ctl, message) -> None:
    """`message` through aiperf's bus to each of `ctl`'s hooks on its type."""
    message = Message.from_json(message.to_json_bytes())
    for hook in ctl.get_hooks(AIPerfHook.ON_MESSAGE):
        if message.message_type in hook.resolve_params(ctl):
            await hook.func(message)


def test_aiperf_loads_the_compass_system_controller():
    found = plugins.get_class(PluginType.SERVICE, ServiceType.SYSTEM_CONTROLLER)
    assert found is CompassSystemController


#: The engine's work, which it holds past the moment every result is in.
HELD = 50.0


class HoldingEngine(threading.Thread):
    """The engine LP with work to `HELD`, which it holds in wall time until
    `release` is set. It keeps its next grant, or the refusal that ended the
    run, as `got`."""

    def __init__(self, endpoint) -> None:
        super().__init__(name="engine", daemon=True)
        self.endpoint, self.got = endpoint, None
        self.holding, self.release = threading.Event(), threading.Event()

    def run(self) -> None:
        rt = LPRuntime(ENGINE, TABLE, clock_transport.connect(ENGINE, self.endpoint))
        rt.next_event(HELD)
        self.holding.set()
        self.release.wait(10)
        try:
            self.got = rt.next_event(INF)
        except ClockAbort as abort:
            self.got = abort


class TimingManager(threading.Thread):
    """aiperf's timing manager: the router on its own event loop, the traffic
    LP's clock owner. aiperf has sent everything, so the traffic LP asks for
    time until its ``+inf`` grant."""

    def __init__(self) -> None:
        super().__init__(name="timing_manager", daemon=True)
        self.loop, self.router = asyncio.new_event_loop(), None

    def run(self) -> None:
        async def serve():
            self.router = CompassCreditRouter(run=None, service_id="timing_manager")
            self.router.traffic.finish()
            await asyncio.Event().wait()

        with contextlib.suppress(RuntimeError):  # the loop stopped
            self.loop.run_until_complete(serve())

    def kill(self) -> None:
        """aiperf's SIGKILL: the traffic LP's clock connection ends wherever
        the loop is, and the loop stops."""
        fd = os.dup(self.router.traffic.rt.conn.fileno())
        with socket.socket(fileno=fd) as sock, contextlib.suppress(OSError):
            sock.shutdown(socket.SHUT_RDWR)
        self.loop.call_soon_threadsafe(self.loop.stop)


def test_aiperf_s_shutdown_waits_for_the_traffic_lp_s_inf_grant(tmp_path, monkeypatch):
    """Every result reaches aiperf's controller while the engine holds work,
    so the traffic LP still waits for its ``+inf`` grant, its event loop
    blocked in that wait. The controller stops only once the router reports
    the traffic LP done, and its stop kills the timing manager as aiperf's
    does. Nothing ends the run: the engine is granted ``+inf``, and the
    traffic LP's final-response check runs and passes."""
    run = {**RUN, "bound_s": 100.0, "out_dir": str(tmp_path)}
    server = clock_transport.serve(
        compass_run._RecordingAuthority(run), "tcp://127.0.0.1:0"
    )
    frontend = clock_transport.connect(FRONTEND, server.endpoint)
    frontend.send((NER, INF, [], INF))  # idle throughout
    lp = str(tmp_path / "lp")

    def traffic(cls):
        conn = clock_transport.connect(compass_run.TRAFFIC, server.endpoint)
        return cls(LPRuntime(compass_run.TRAFFIC, TABLE, conn), lp)

    monkeypatch.setattr(TrafficLP, "from_env", classmethod(traffic))
    monkeypatch.setattr(ClockPacedLoopScheduler, "clock", None)
    tm, engine, kills = TimingManager(), HoldingEngine(server.endpoint), []

    async def main():
        loop, killed = asyncio.get_running_loop(), asyncio.Event()

        async def publish(message):  # from the timing manager's process
            loop.call_soon_threadsafe(asyncio.ensure_future, _deliver(ctl, message))

        _router_bus(monkeypatch, publish)

        async def stop():
            kills.append(tm.router.traffic.done.done())
            tm.kill()
            killed.set()

        ctl = await _controller(monkeypatch, stop)
        tm.start()
        engine.start()
        assert await asyncio.to_thread(engine.holding.wait, 10)
        await ctl._on_process_records_result_message(RESULTS)  # every result is in
        engine.release.set()
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(killed.wait(), 10)
        return ctl

    try:
        ctl = _run(main())
    finally:
        if not kills:
            tm.loop.call_soon_threadsafe(tm.loop.stop)
        tm.join(10)
        engine.join(10)
        frontend.close()
        server.close()
    assert engine.got == INF
    assert kills == [True]  # once, after the traffic LP was done
    done = tm.router.traffic.done
    assert done.result() is None and tm.router.traffic.now() == INF
    assert ctl._exit_errors == []


def test_a_failing_traffic_lp_still_fails_aiperf_s_run(tmp_path, monkeypatch):
    """Request 0 is open at the ``+inf`` grant, so the traffic LP fails its
    final-response check. The router's service error unregisters the timing
    manager from the traffic domain, so aiperf stops on the records manager's
    results and exits 1 naming the open request."""
    sent, stops, rt, lp = [], [], ScriptedRuntime([]), str(tmp_path / "lp")

    async def publish(message):
        sent.append(message)

    async def stop():
        stops.append("stop")

    _router_bus(monkeypatch, publish)
    monkeypatch.setattr(TrafficLP, "from_env", classmethod(lambda cls: cls(rt, lp)))

    async def timing_manager():
        traffic = CompassCreditRouter(run=None, service_id="timing_manager").traffic
        traffic.send(_key(0))
        traffic.finish()
        await asyncio.Event().wait()  # only the router stops the loop

    with pytest.raises(RuntimeError, match="Event loop stopped before Future"):
        _run(timing_manager())

    async def controller():
        ctl = await _controller(monkeypatch, stop)
        for message in sent:  # the service error, then the cancel
            await _deliver(ctl, message)
        assert stops == []  # the records manager's results are not in yet
        await ctl._on_process_records_result_message(RESULTS)
        return ctl

    ctl = asyncio.run(controller())
    assert stops == ["stop"]
    (error,) = ctl._exit_errors
    assert error.error_details.type == "UnansweredRequests"
    assert str(_key(0)) in error.error_details.message
