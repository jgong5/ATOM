# SPDX-License-Identifier: MIT
"""A simulated run under the DeepSeek-V4-Pro EP8 decode law captured on
mi355-gpu-40, whose HTTP intake is fitted on its own recorded server's lone
requests, charges their mean body about what that loop took to take one in."""

import asyncio
import json
import queue
import threading

import pytest
from test_compass_event_loop import (
    HTTP,
    INF,
    TABLE,
    TRAFFIC,
    _stamp,
    run,  # noqa: F401  the run fixture
)
from test_spec_verbs import SPEC_ROOT

from atom.compass import clock_transport
from atom.compass import run as compass_run
from atom.utils.clock import LPRuntime
from atom.utils.compass_loop import CompassEventLoop, HttpChannel

LAW = SPEC_ROOT / "scripts/compass/laws"
GPU40 = LAW / "deepseek-v4-pro-dpa8-ep8-mi355x-ptl-disabled-661ef7661-decode.json"
#: The law's decode-side record at 2d701d657 (cc-traces 256 to 392, c=16), over
#: its lone requests: the mean body, and the loop's mean CPU time to take one in.
CELL_BYTES, CELL_INTAKE_S = 509_461, 4.212e-3


def test_a_simulated_run_on_gpu40_charges_a_lone_cell_body_what_the_cell_took(
    run, monkeypatch, tmp_path  # noqa: F811
):
    path = tmp_path / "run.json"
    host = json.loads(GPU40.read_text())["host"]
    path.write_text(json.dumps({"bound_s": 600.0, "host": host}))
    monkeypatch.setenv(compass_run.ENV, str(path))
    stamps = queue.Queue()

    def traffic():
        rt = LPRuntime(TRAFFIC, TABLE, clock_transport.connect(TRAFFIC, run.endpoint))
        rt.start_run()
        stamps.put(rt.stamp_send(HTTP))
        rt.next_event(INF)
        rt.close()

    thread = threading.Thread(target=traffic, name="traffic", daemon=True)
    thread.start()
    arrival, entered = TABLE.lookahead(HTTP), []

    async def app(scope, receive, send):
        entered.append(asyncio.get_running_loop().time())

    async def receive():
        return {"type": "http.request", "body": b"x" * CELL_BYTES}

    async def go():
        scope = {"type": "http", "headers": [(b"x-test-stamp", b"%r 0" % arrival)]}
        await HttpChannel(app, _stamp)(scope, receive, None)

    loop = CompassEventLoop()
    try:
        loop.run_until_complete(go())
        loop.run_forever()
    finally:
        thread.join(10)
    loop.close()
    assert stamps.get(timeout=10) == (arrival, 0)
    print(f"\n  charged {(entered[0] - arrival) * 1e3:.3f} ms")
    assert entered[0] - arrival == pytest.approx(CELL_INTAKE_S, abs=0.5e-3)
