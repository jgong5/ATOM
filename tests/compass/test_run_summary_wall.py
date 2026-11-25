# SPDX-License-Identifier: MIT
"""A run summary's wall seconds, which its speed divides by, are its client runs'.

Each client run counts from its client's first request to its end, so the
client's launch before its first clock call is not counted, nor, after its
client run ends, its exports and the stop: virtual time stands still through
both. A client run a stop finds open ends at the finish. Client run ``k``'s
summary counts client runs 0 to ``k``, as its simulated seconds do, and the
server's summary counts them all.

The authority is driven directly, on a wall clock the test sets, for a server
that serves client runs until it is stopped, as `compass run` starts one.
"""

import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

from atom.compass import run as compass_run
from atom.compass.clock import NER, TAR
from atom.compass.run import ENGINE, FRONTEND, TRAFFIC
from atom.utils import clock

INF = math.inf
HTTP, STREAM = "traffic->frontend:http", "frontend->traffic:stream"


def _serve(monkeypatch, out: Path, client_runs, stop: float, late=None) -> Path:
    """Serve `client_runs`, each the wall times of its client's first request
    and of its last, which ends it, or None for a client still bound at the
    stop; the server's LPs ask at wall 100. Each client run sends one request,
    which arrives a simulated second on and is answered a second after that.
    Then a stop at wall `stop`, and with `late`, a client that binds and asks
    at that wall time, after the finish. Returns the out_dir."""
    run = {"admission_s": 0.54e-3, "ipc_s": 2.0**-10, "stream_s": 2.0**-9}
    run.update(bound_s=100.0, client_runs=None, out_dir=str(out))
    wall = [100.0]
    fake = SimpleNamespace(monotonic=lambda: wall[0], sleep=time.sleep)
    monkeypatch.setattr(compass_run, "time", fake)
    monkeypatch.setattr(clock, "_installed", None)
    authority = compass_run._RecordingAuthority(run)
    now = 0.0  # the client's clock
    for k, (first, last) in enumerate(client_runs):
        authority.on_request(FRONTEND, NER, INF, [])
        authority.on_request(ENGINE, NER, INF, [])
        authority.on_bind(TRAFFIC)
        wall[0] = first
        authority.on_request(TRAFFIC, NER, INF, [(HTTP, k, now + 1)])
        authority.on_request(FRONTEND, NER, INF, [(STREAM, k, now + 2)])
        now += 2
        if last is None:
            break
        wall[0] = last
        authority.on_request(TRAFFIC, NER, INF, [])  # nothing left: the end
        compass_run._hand_in(run, "engine", [], k, priced={"steps": 1})
        compass_run._hand_in(run, "frontend", [], k)
        summary = out / compass_run.CLIENT_DIR.format(k) / compass_run.SUMMARY_FILE
        deadline = time.monotonic() + 10
        while not summary.exists():  # written on a thread of its own
            assert time.monotonic() < deadline, f"no client-{k} summary in 10 s"
            time.sleep(0.05)
    wall[0] = stop
    authority.on_request(FRONTEND, TAR, INF, [])
    wall[0] = stop + 1  # the hand-ins and the summary come after the finish
    if late is not None:
        wall[0] = late
        authority.on_bind(TRAFFIC)
        assert authority.on_request(TRAFFIC, NER, INF, []) == []
    compass_run._hand_in(run, "engine", [], priced={"steps": len(client_runs)})
    compass_run._hand_in(run, "frontend", [], stopped="SIGTERM")
    compass_run._write_summary(run, authority, ended=not authority.stopped_whole)
    return out


def _cost(where: Path) -> dict:
    return json.loads((where / compass_run.SUMMARY_FILE).read_text())["cost"]


def test_the_wall_leaves_out_the_client_s_launch_and_what_follows_its_last_request(
    monkeypatch, tmp_path
):
    out = _serve(monkeypatch, tmp_path, [(110.0, 1110.0)], stop=1117.0)
    for cost in (_cost(out), _cost(out / "client-0")):
        assert (cost["wall_seconds"], cost["simulated_seconds"]) == (1000.0, 2.0)
        assert cost["speed_ratio"] == 2.0 / 1000.0


def test_a_client_run_a_stop_finds_open_ends_at_the_finish(monkeypatch, tmp_path):
    out = _serve(monkeypatch, tmp_path, [(110.0, None)], stop=1117.0)
    assert _cost(out)["wall_seconds"] == 1007.0


def test_the_wall_counts_each_client_run_and_not_the_time_between_them(
    monkeypatch, tmp_path
):
    served = [(110.0, 1110.0), (1130.0, 2130.0)]
    out = _serve(monkeypatch, tmp_path, served, stop=2140.0)
    walls = [_cost(out / f"client-{k}")["wall_seconds"] for k in (0, 1)]
    assert walls == [1000.0, 2000.0]
    assert _cost(out)["wall_seconds"] == 2000.0


def test_a_client_that_asks_after_the_finish_opens_no_client_run(monkeypatch, tmp_path):
    out = _serve(monkeypatch, tmp_path, [(110.0, 1110.0)], stop=1117.0, late=1130.0)
    assert _cost(out)["wall_seconds"] == 1000.0
