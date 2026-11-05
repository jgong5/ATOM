# SPDX-License-Identifier: MIT
"""mi355-gpu-40's spec states the per-byte intake rate that lone requests of a
served DeepSeek-V4.1-Flash cell's bodies read, beside the burst's fixed term at
that rate, and a simulated run on it charges the cell's mean body about what
the cell's loop took to take it in."""

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
from test_spec_verbs import SPEC_ROOT, _mi355x_spec

from atom.compass import clock_transport
from atom.compass import run as compass_run
from atom.utils.clock import LPRuntime
from atom.utils.compass_loop import CompassEventLoop, HttpChannel

GPU40 = SPEC_ROOT / "scripts" / "compass" / "machines" / "mi355x-gpu40.json"
#: The DeepSeek-V4.1-Flash TP4 DSpark5 cc-traces c=16 cell as a real recording
#: at 0cd22bcc3 read it, over its long prompts: the mean body, and the loop's
#: mean intake of one.
CELL_BYTES, CELL_INTAKE_S = 388_200, 4.59e-3


def test_gpu40_states_the_lone_requests_rate_and_the_burst_fixed_term_at_it(
    monkeypatch,
):
    script, spec = _mi355x_spec(monkeypatch, "mi355x-gpu40")
    model, _, root, rate, band, record = script.LONE
    origin = spec.origin("host.http_intake.per_byte_s")
    assert origin["origin"] == "probed"
    assert (origin["model"], origin["source_root"]) == (model, f"commit:{root}")
    assert origin["record"] == f"sha256:{record}"
    assert (origin["p10"], origin["p90"]) == band
    assert spec.value("host.http_intake.per_byte_s") == rate
    _, root, terms, bands, record = script.INTAKE
    origin = spec.origin("host.http_intake.fixed_s")
    assert (origin["source_root"], origin["record"]) == (
        f"commit:{root}",
        f"sha256:{record}",
    )
    assert (origin["p10"], origin["p90"]) == bands["fixed_s"]
    assert spec.value("host.http_intake.fixed_s") == terms["fixed_s"]


def test_a_simulated_run_on_gpu40_charges_a_lone_cell_body_what_the_cell_took(
    run, monkeypatch, tmp_path  # noqa: F811
):
    path = tmp_path / "run.json"
    machine = json.loads(GPU40.read_text())
    path.write_text(json.dumps({"bound_s": 600.0, "machine": machine}))
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
