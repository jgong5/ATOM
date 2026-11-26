# SPDX-License-Identifier: MIT
"""A stream bypass, `compass run --stream-bypass` (#4228).

What each group defends:

* **The table.** Under the run file's ``stream_bypass`` each decode frontend's
  stream goes straight to traffic at the lookahead the router's two hops sum
  to, and only a stream's last event takes those hops, as ``final``. A run
  with no router LP, and a key that is no bool, are refused.
* **The flag.** `compass run --stream-bypass` writes the key, which is off by
  default and then absent; on, it is part of the configuration, as the
  channels its step table names are.
* **The decode frontend.** Its loop stamps a stream's events but its last on
  the straight channel, each naming it, and its ``[DONE]`` on ``final``.
* **The router.** atomesh as LP ``router`` passes the named events on as they
  are and holds the last, whose turn drops decode's load: every request takes
  the ranks it takes without the bypass, and no stream event but a last one
  is released to the router.
* **The client.** aiperf's transport reports each event on the channel its
  stamp names, with the TTFT and latency of the stamps; the traffic LP takes
  each report on that channel, passes a return at its final's release, and
  a settle whose released events are all reported reads no socket.
"""

import asyncio
import dataclasses
import json
import math
import uuid
from types import SimpleNamespace

import pytest
from test_drive_runfile import build
from test_router_lp_acceptance import (
    RANKS,
    atomesh,  # noqa: F401  the fixture
    run_router_lp,
)
from test_vertical_slice import _run_file

from atom.compass import clock_transport
from atom.compass import run as compass_run
from atom.compass.backends.step_stub import Coefficients
from atom.compass.carriers import sse_channel, sse_stamp
from atom.compass.clock import ClockAuthority, LpId, prefill_decode_table
from atom.compass.detect.scenario import check_run
from atom.utils import compass_loop
from atom.utils.clock import LPRuntime

HOPS = {
    "admission_s": 1e-5,
    "ipc_s": 1.3e-5,
    "stream_s": 1e-5,
    "router_s": 1.5e-3,
    "kv_write_req_s": 1.06e-4,
    "router_in_s": 1.3e-5,
}
TRAFFIC, ROUTER = LpId("traffic"), LpId("router")
STRAIGHT, FINAL = "frontend-D->traffic:stream", "router->traffic:final"


def _table(bypass: bool, decode: int = 2):
    return prefill_decode_table(**HOPS, decode=decode, stream_bypass=bypass)


def test_a_bypassed_stream_arrives_at_the_lookahead_of_the_router_s_hops():
    plain, bypass = _table(False), _table(True)
    via = plain.lookahead("frontend-D0->router:stream") + plain.lookahead(
        "router->traffic:stream"
    )
    assert bypass.lookahead("frontend-D0->traffic:stream") == pytest.approx(via)
    final = bypass.lookahead("frontend-D0->router:final")
    assert final + bypass.lookahead(FINAL) == via
    assert sorted(c.name for c in bypass.channels_into(TRAFFIC)) == [
        "frontend-D0->traffic:stream",
        "frontend-D1->traffic:stream",
        FINAL,
    ]

    def others(table):
        return {
            (c.name, c.lookahead_s)
            for c in table.channels_from(ROUTER)
            if not c.name.startswith("router->traffic")
        }

    assert others(bypass) == others(plain)


def test_a_bypass_with_no_router_lp_or_as_no_bool_is_refused(tmp_path):
    with pytest.raises(ValueError, match="no router_in_s"):
        prefill_decode_table(**{**HOPS, "router_in_s": None}, stream_bypass=True)
    one = json.loads(_run_file(tmp_path, stream_bypass=True).read_text())
    with pytest.raises(ValueError, match="stream_bypass"):
        compass_run.channel_table(one)
    with pytest.raises(TypeError, match="stream_bypass"):
        compass_run.channel_table(dict(one, stream_bypass="yes"))


@pytest.mark.parametrize("flags, on", [((), False), (("--stream-bypass",), True)])
def test_the_bypass_is_off_unless_the_command_turns_it_on(tmp_path, flags, on):
    stub = tmp_path / "stub.json"
    stub.write_text(json.dumps(dataclasses.asdict(Coefficients())))

    run = build(tmp_path, "--step-stub", str(stub), *flags)

    assert run.get(compass_run.STREAM_BYPASS) is (True if on else None)
    assert (compass_run.STREAM_BYPASS in compass_run.configuration(run)) is on


class _DecodeFrontend:
    """frontend-D0's runtime in its run, each send it stamps recorded."""

    def __init__(self, table) -> None:
        self.table, self.me, self.in_run, self.sent = (
            table,
            LpId("frontend-D0"),
            True,
            [],
        )

    def stamp_send(self, ch, at=None):
        self.sent.append(ch)
        return 0.5, self.sent.count(ch) - 1


def _stamped(rt, frames: list[str]) -> list[str]:
    """The comment lines of `frames`, each a body message of an SSE response,
    as the decode frontend's loop stamps them."""
    out = []

    async def send(message):
        out.append(message)

    async def respond():
        stamping = compass_loop._stamping_events(send, rt)
        start = [(b"content-type", b"text/event-stream")]
        await stamping({"type": "http.response.start", "headers": start})
        for frame in frames:
            body = {"type": "http.response.body", "body": frame.encode()}
            await stamping(body | {"more_body": True})

    asyncio.run(respond())
    text = "".join(m["body"].decode() for m in out[1:])
    return [line for line in text.split("\n") if line.startswith(": compass")]


@pytest.mark.parametrize("bypass", [False, True])
def test_a_decode_frontend_sends_only_a_stream_s_last_event_through_the_router(
    bypass,
):
    rt = _DecodeFrontend(_table(bypass))
    comments = _stamped(rt, ["data: 0\n\n", "data: 1\n\ndata: [DONE]\n\n"])
    if bypass:
        stream, last = "frontend-D0->traffic:stream", "frontend-D0->router:final"
        assert rt.sent == [stream, stream, last]
        assert [sse_stamp(c) for c in comments] == [(0.5, 0), (0.5, 1), (0.5, 0)]
        assert [sse_channel(c) for c in comments] == [stream, stream, None]
    else:
        assert rt.sent == ["frontend-D0->router:stream"] * 3
        assert [sse_stamp(c) for c in comments] == [(0.5, 0), (0.5, 1), (0.5, 2)]
        assert [sse_channel(c) for c in comments] == [None] * 3


def test_the_router_lp_holds_only_a_bypassed_stream_s_last_event(
    atomesh, tmp_path  # noqa: F811
):
    out = tmp_path / "bypass"
    assert run_router_lp(atomesh, out, "1", lambda: 0.0, stream_bypass=True) == RANKS
    assert not check_run([out])
    steps = (out / compass_run.STEP_TABLE_FILE).read_text()
    assert "->router:stream" not in steps
    assert "frontend-D->router:final" in steps and STRAIGHT in steps


# --- the client, where aiperf and compass_harness are installed --------------


@pytest.fixture
def harness():
    """test_harness_traffic_lp, whose helpers these tests use; skips as it does."""
    import test_harness_traffic_lp

    return test_harness_traffic_lp


@pytest.fixture
def transport():
    """test_harness_transport, whose stub server these tests use; skips as it does."""
    import test_harness_transport

    return test_harness_transport


class _Bypassed:
    """The transport tests' stand-in frontend as frontend-D behind a stream
    bypass, with a seq of its own on each channel."""

    me, table = LpId("frontend-D"), _table(True, decode=1)

    def __init__(self) -> None:
        self.arrival, self.seqs = 0.0, {}

    def stamp_send(self, ch: str) -> tuple[float, int]:
        self.seqs[ch] = self.seqs.get(ch, -1) + 1
        return self.arrival, self.seqs[ch]


def test_aiperf_s_transport_reports_each_event_on_the_channel_its_stamp_names(
    transport, tmp_path, monkeypatch
):
    t = transport

    async def run():
        async with t.Harness(tmp_path, monkeypatch) as h:
            h.frontend = _Bypassed()
            for num in t.SENDS:
                await h.stamp(num)
            records = [await h.send(num) for num in t.SENDS]
            n = sum(map(len, t.TOKENS.values())) + len(t.SENDS)
            return h, records, [await h.pull.recv_pyobj() for _ in range(n)]

    h, records, reports = t._run(run())
    expected, seq = [], 0
    for (num, (sent, _)), record in zip(t.SENDS.items(), records):
        tokens = t.TOKENS[num]
        assert record.error is None
        assert h.metric(record, t.TTFTMetric) == t._ns(tokens[0]) - t._ns(sent)
        latency = h.metric(record, t.RequestLatencyMetric)
        assert latency == t._ns(tokens[-1]) - t._ns(sent)
        for arrival in tokens:
            expected.append((t._key(num), seq, arrival, False, STRAIGHT))
            seq += 1
        expected.append((t._key(num), num, tokens[-1], True))
    assert reports == expected


def test_behind_a_bypass_the_traffic_lp_takes_each_report_on_its_channel(
    harness, tmp_path, monkeypatch
):
    h = harness
    endpoint = f"inproc:test-stream-bypass-{uuid.uuid4().hex}"
    run = {"clock_endpoint": endpoint, "bound_s": 1e6, **HOPS}
    run |= {"decode_instances": 2, "stream_bypass": True}
    server = clock_transport.serve(
        ClockAuthority(compass_run.channel_table(run)), endpoint
    )
    (tmp_path / "run.json").write_text(json.dumps(run))
    monkeypatch.setenv(compass_run.ENV, str(tmp_path / "run.json"))
    monkeypatch.setenv(h.ADDRESS_ENV, str(tmp_path / "lp"))
    d1 = "frontend-D1->traffic:stream"

    async def main():
        traffic = h.TrafficLP.from_env()
        try:
            traffic._take((h._key(0), 0, 1.0, False, d1))
            traffic._take((h._key(0), 0, 2.0, True))
            return traffic.streams, traffic.rt.unreleased
        finally:
            for task in (*traffic._tasks, traffic.done):
                task.cancel()
            traffic._pub.close(linger=0)
            traffic._pull.close(linger=0)

    try:
        streams, unreleased = h._run(main())
    finally:
        server.close()
    assert streams[0] == FINAL
    assert sorted(streams[1:]) == ["frontend-D0->traffic:stream", d1]
    assert unreleased == {(d1, 0): 1.0, (FINAL, 0): 2.0}


class _Grants:
    """A Clock Authority connection whose grant ``k``, at ``k + 1``, releases
    ``script[k]``; then ``+inf`` ends the run."""

    def __init__(self, script: list) -> None:
        self.script, self.k = script, 0

    def send(self, msg) -> None:
        pass

    def recv(self):
        if self.k == len(self.script):
            return math.inf, {}
        self.k += 1
        return float(self.k), self.script[self.k - 1]


def test_behind_a_bypass_a_settle_whose_released_events_are_reported_reads_no_socket(
    harness, tmp_path
):
    """Request 0's two events on decode's own channel and its last through the
    router are all reported before the first grant, and each grant releases
    one: the return passes at the last's release, and no settle reads the
    report socket."""
    h = harness
    reported = [(STRAIGHT, 0, 1.0, False), (STRAIGHT, 1, 2.0, False)]
    reported += [(FINAL, 0, 3.0, True)]

    async def main():
        script = [{ch: [(seq, at)]} for ch, seq, at, _ in reported]
        rt = LPRuntime(TRAFFIC, _table(True, decode=1), _Grants(script))
        prefix = str(tmp_path / "lp")
        http = "traffic->router:http"
        traffic = h.TrafficLP(rt, prefix, http, FINAL, bypass=(STRAIGHT,))
        reads, take_queued = [], traffic._take_queued

        def counted():
            reads.append(rt.now)
            take_queued()

        traffic._take_queued = counted
        passed = []

        async def on_return(worker_id, ret):
            passed.append(ret.credit.id)

        push = h.zmq.Context.instance().socket(h.zmq.PUSH)
        push.connect(h.addresses(prefix)[1])
        for ch, seq, at, final in reported:
            named = () if ch == FINAL else (ch,)
            push.send_pyobj((h._key(0), seq, at, final, *named))
        while sum(map(len, traffic._reports.values())) < len(reported):
            await asyncio.sleep(0.001)
        traffic.send(h._key(0))
        await traffic.hold(on_return)(
            "worker_0", SimpleNamespace(credit=h._credit(0), error=None)
        )
        traffic.finish()
        try:
            await asyncio.wait_for(traffic.done, 10)
        finally:
            push.close(linger=0)
        return passed, reads, rt

    passed, reads, rt = h._run(main())
    assert passed == [0] and rt.now == math.inf
    assert reads == []
