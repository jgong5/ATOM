# SPDX-License-Identifier: MIT
"""Each client run a simulated server serves has its own summary.

A co-hosting frontend, an engine and one traffic LP per client run, each the
run bootstrap's own runtime, run over the authority the frontend serves. At
the end of a client run before the last, the engine and the frontend each
hand in at their marked grant, from inside their clock call, and the
authority's host writes that client run's summary; the finish writes the last
one's beside the run summary. A client run's counts are its own: the engine's
workers and the frontend's executor count from the server's start, and a
summary takes each hand-in less the one at the previous client run's end, so
it matches the summary a fresh server writes for the same client run. A stop
after a client run writes the run summary and leaves that client run's alone.

An engine in a run samples ATOM's prefix-cache and tier-load counters on its
clock after each step that moved one, and a client run's summary carries them
from the sample in force at its start; those tests step an engine core by hand
on a runtime with no authority, its scheduler's `EngineStats` ATOM's own.
"""

import json
import logging
import math
import re
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_dp_slice import _deployment

from atom.compass import run as compass_run
from atom.compass.clock import ClientRun
from atom.compass.clock_transport import connect
from atom.model_engine.engine_stats import EngineStats
from atom.utils import clock
from atom.utils.clock import LPRuntime
from atom.utils.distributed import utils

INF = math.inf
HTTP, STREAM = "traffic->frontend:http", "frontend->traffic:stream"
#: What each client run adds to the server's counts: the workers' refused
#: commands and priced steps, and the frontend executor's refused jobs.
CLIENT_RUNS = (
    (
        ["command:profile on rank 0"],
        {
            "steps": 3,
            "seconds": 0.75,
            "estimated_steps": 1,
            "treatments": Counter({"uniform routing": 3}),
        },
        [],
    ),
    (
        ["command:profile on rank 0", "cost:no source answered"],
        {
            "steps": 5,
            "seconds": 1.25,
            "analytical_steps": 2,
            "analytical_seconds": 0.5,
            "treatments": Counter(),
        },
        ["executor:render"],
    ),
)


def _serve(monkeypatch, out: Path, client_runs, served, stop=None) -> Path:
    """Serve one client run per entry of `served` on one server whose run file
    declares `client_runs`, None for ``null``; each client run sends one
    request, answered on the stream. With `stop`, a SIGTERM then stops the
    server: ``resigned``, once the last client run has ended, or ``bound``,
    with its client still bound after its answer. Returns the out_dir."""
    out.mkdir()
    run = {
        "admission_s": 0.54e-3,
        "ipc_s": 2.0**-10,
        "stream_s": 2.0**-9,
        "bound_s": 100.0,
        "clock_endpoint": f"inproc:{out}",
        "out_dir": str(out),
        "client_runs": client_runs,
    }
    (out / "run.json").write_text(json.dumps(run))
    monkeypatch.setenv(compass_run.ENV, str(out / "run.json"))
    for module, name in [
        (compass_run, "_authority"),
        (compass_run, "_member"),
        (clock, "_installed"),
        (utils, "LP_OF_RANK"),
    ]:
        monkeypatch.setattr(module, name, None)
    table = compass_run.channel_table(run)
    with compass_run.frontend(_deployment()):
        pass
    frontend = clock.installed()
    frontend.loop = SimpleNamespace(executor=SimpleNamespace(refusals=[]))
    with compass_run.engine(_deployment()):
        pass
    engine = clock.installed()
    workers = {
        "refused_commands": [],
        "priced_steps": {"treatments": Counter()},
        "predicted_memory": {},
    }
    core = SimpleNamespace(
        runner_mgr=SimpleNamespace(call_func=lambda name, wait_out: workers[name])
    )
    ended = len(served) - (stop == "bound")
    with compass_run.engine_loop(core):
        frontend.ask_next_event(INF)
        engine.ask_next_event(INF, 50.0)
        for k, (commands, priced, jobs) in enumerate(served):
            workers["refused_commands"] = workers["refused_commands"] + commands
            counts = workers["priced_steps"]
            workers["priced_steps"] = {
                c: counts.get(c, 0) + priced.get(c, 0) for c in {*counts, *priced}
            }
            frontend.loop.executor.refusals += jobs
            traffic = LPRuntime(
                compass_run.TRAFFIC,
                table,
                connect(compass_run.TRAFFIC, run["clock_endpoint"]),
            )
            traffic.start_run()
            traffic.stamp_send(HTTP)
            traffic.ask_next_event(INF)
            frontend.take_grant()
            frontend.stamp_send(STREAM)
            frontend.ask_next_event(INF)
            traffic.take_grant()
            if k == ended:  # the client stays bound, its client run open
                break
            traffic.ask_next_event(INF)
            # The end of the client run: the client resigns, and the engine
            # and the frontend hand in at their marked grants, or at the finish.
            assert traffic.take_grant() == INF
            last = k + 1 == len(served) and stop is None
            assert (
                (engine.take_grant() == INF) is (frontend.take_grant() == INF) is last
            )
            if not last:
                engine.ask_next_event(INF, 50.0)
                frontend.ask_next_event(INF)
        if stop is not None:  # the signal, as the frontend's event loop takes it
            frontend.loop.stopped = "SIGTERM"
            frontend.end_run()
            assert engine.take_grant() == INF
    monkeypatch.setattr(clock, "_installed", frontend)
    assert compass_run.frontend_done(SimpleNamespace(close=lambda: None))
    # An earlier client run's summary is written on a thread of its own.
    deadline = time.monotonic() + 10
    for k in range(ended):
        while not (out / f"client-{k}" / compass_run.SUMMARY_FILE).exists():
            assert time.monotonic() < deadline, f"no client-{k} summary in 10 s"
            time.sleep(0.05)
    return out


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


@pytest.fixture
def served(monkeypatch, tmp_path):
    return _serve(monkeypatch, tmp_path / "server", 2, CLIENT_RUNS)


def test_each_client_run_has_its_own_summary_over_its_own_steps(served):
    out = served
    whole = _read(out / compass_run.SUMMARY_FILE)
    runs = [_read(out / f"client-{k}" / compass_run.SUMMARY_FILE) for k in (0, 1)]
    for k, ((commands, priced, jobs), summary) in enumerate(zip(CLIENT_RUNS, runs)):
        schedule = summary["schedule"]
        print(f"\nclient run {k}:", json.dumps(summary["client_run"]), schedule)
        assert summary["client_run"]["index"] == k
        assert [r for r, n in schedule["refusals"]["reasons"] for _ in range(n)] == (
            sorted(commands + jobs)
        )
        assert schedule["refusals"]["steps"] == priced["steps"]
        assert schedule["predicted_seconds"] == priced["seconds"]
        assert schedule["treatments"] == dict(priced["treatments"])
        by_method = schedule["seconds_by_method"]
        assert by_method["analytical"]["steps"] == priced.get("analytical_steps", 0)
        assert by_method["estimated"]["steps"] == priced.get("estimated_steps", 0)
        assert summary["coverage_report"] is True
    # The server's summary counts both client runs.
    assert whole["schedule"]["refusals"]["count"] == 4
    assert whole["schedule"]["predicted_seconds"] == 2.0
    assert runs[1]["client_run"]["start"] == runs[0]["client_run"]["end"] > 0
    assert (out / "client-0" / compass_run.LP_FILE.format("engine")).exists()


def test_a_later_client_run_s_summary_is_a_fresh_server_s(
    monkeypatch, tmp_path, served
):
    later = _read(served / "client-1" / compass_run.SUMMARY_FILE)
    fresh_out = _serve(monkeypatch, tmp_path / "fresh", 1, CLIENT_RUNS[1:])
    fresh = _read(fresh_out / compass_run.SUMMARY_FILE)
    assert _read(fresh_out / "client-0" / compass_run.SUMMARY_FILE)["client_run"] == {
        "index": 0,
        "start": 0.0,
        "end": fresh["cost"]["simulated_seconds"],
    }
    for key in ("predicted_seconds", "refusals", "seconds_by_method", "treatments"):
        assert later["schedule"][key] == fresh["schedule"][key], key
    for key in ("coverage_report", "configuration", "schema_version"):
        assert later[key] == fresh[key], key
    span = later["client_run"]["end"] - later["client_run"]["start"]
    assert span == pytest.approx(fresh["cost"]["simulated_seconds"], rel=0, abs=1e-12)


def test_client_runs_is_a_positive_count_or_null(monkeypatch, tmp_path):
    for count in (0, 1.5, "2"):
        (tmp_path / "run.json").write_text(
            json.dumps({"bound_s": 1.0, "client_runs": count})
        )
        monkeypatch.setenv(compass_run.ENV, str(tmp_path / "run.json"))
        with pytest.raises(ValueError, match=f"client_runs is {count!r}"):
            compass_run.spec()


def test_a_stop_after_a_client_run_leaves_that_client_runs_summary(
    monkeypatch, tmp_path
):
    """A standalone authority stopped after client run 0 grants every server
    LP ``+inf`` and the resigned client nothing, then writes the run summary
    from every hand-in, with no client run open no stop reason, and client
    run 0's summary not again."""
    run = {"admission_s": 0.54e-3, "ipc_s": 2.0**-10, "stream_s": 2.0**-9}
    run.update(bound_s=100.0, client_runs=None, out_dir=str(tmp_path))
    monkeypatch.setenv(compass_run.ENV, str(tmp_path / "run.json"))
    authority = compass_run._RecordingAuthority(run)
    for lp in (compass_run.FRONTEND, compass_run.ENGINE, compass_run.TRAFFIC):
        authority.on_request(lp, "NER", INF, [])

    def hand_in(k, steps, **frontend):
        priced = {"steps": steps, "seconds": steps / 4}
        compass_run._hand_in(run, "engine", [], k, priced=priced)
        compass_run._hand_in(run, "frontend", [], k, **frontend)

    hand_in(0, 1)
    client_run = tmp_path / "client-0" / compass_run.SUMMARY_FILE
    deadline = time.monotonic() + 10
    while not client_run.exists():
        assert time.monotonic() < deadline, "no client-0 summary in 10 s"
        time.sleep(0.05)
    written = client_run.read_text()
    stop = authority.on_request(compass_run.FRONTEND, "TAR", INF, [])
    assert sorted(lp.name for lp, g, _ in stop if g == INF) == ["engine", "frontend"]
    hand_in(None, 3, stopped="SIGTERM")
    compass_run._write_summary(run, authority)
    summary = _read(tmp_path / compass_run.SUMMARY_FILE)
    assert summary["schedule"]["predicted_seconds"] == 0.75
    assert summary["schedule"]["refusals"]["reasons"] == []
    assert client_run.read_text() == written


#: The steps an engine takes over two client runs: each the
#: prefill it admits, as `EngineStats.update_cache`'s cached, full, compressed,
#: wanted and reusable tokens, or None; the tokens a tier load it completes
#: brings; and its forward's seconds.
STEPS = (
    ((64, 256, 64, 64, 192), 0, 0.25),
    (None, 0, 0.25),  # a decode: no counter moves
    (None, 128, 0.25),  # a tier load lands
    ((128, 512, 128, 128, 448), 0, 0.25),  # client run 0's last sample
    (None, 0, 0.25),
    ((192, 320, 192, 192, 256), 64, 0.25),
)
#: Where client run 0 ends: on its last sample, at 1.0, and after it, at the
#: decode ending at 1.25. Client run 1 opens with the 1.0 sample either way.
ENDS_0 = (1.0, 1.25)
IN_FORCE = 1.0
#: The samples' times without a tier and with one, whose load adds 0.75.
SAMPLED = {False: [0.0, 0.25, 1.0, 1.5], True: [0.0, 0.25, 0.75, 1.0, 1.5]}


class _Engine:
    """An engine core in a run whose `method` takes the next of `STEPS` as
    ATOM's does, admitting, then forwarding on its clock, then taking a load's
    completion, on an engine runtime with no authority; with `tier` its KV
    connector counts ``loaded_tokens``, without one it has none. `atom` holds
    ATOM's counters after each step, at the engine's clock."""

    def __init__(self, monkeypatch, tmp_path, method, tier, stats) -> None:
        run = {"admission_s": 0.54e-3, "ipc_s": 2.0**-10, "stream_s": 2.0**-9}
        run.update(bound_s=100.0, client_runs=2, out_dir=str(tmp_path))
        (tmp_path / "run.json").write_text(json.dumps(run))
        monkeypatch.setenv(compass_run.ENV, str(tmp_path / "run.json"))
        for module, name in [(compass_run, "_authority"), (compass_run, "_member")]:
            monkeypatch.setattr(module, name, None)
        self.run, self.method, self.stats = run, method, stats
        self.authority = compass_run._RecordingAuthority(run)
        self.rt = LPRuntime(compass_run.ENGINE, compass_run.channel_table(run), None)
        monkeypatch.setattr(clock, "_installed", self.rt)
        self.loaded = 0
        kv = SimpleNamespace(get_statistics=lambda: {"loaded_tokens": self.loaded})
        workers = {"refused_commands": [], "priced_steps": {}, "predicted_memory": {}}
        self.core = SimpleNamespace(
            runner_mgr=SimpleNamespace(call_func=lambda name, wait_out: workers[name]),
            scheduler=SimpleNamespace(
                engine_stats=stats, kv_connector=kv if tier else None
            ),
            **{method: self._step},
        )
        self.steps, self.tier, self.atom = iter(STEPS), tier, []

    def _step(self) -> None:
        prefill, loaded, seconds = next(self.steps)
        if prefill:
            self.stats.update_cache(*prefill)
        self.rt.now += seconds
        self.loaded += loaded

    def step(self) -> None:
        getattr(self.core, self.method)()
        c = self.stats.cache_statistics()
        loaded = self.loaded if self.tier else 0
        self.atom.append([self.rt.now, c["cached_tokens"], c["full_tokens"], loaded])

    def serve(self, end_0: float = 1.0) -> list:
        """Both client runs, split at `end_0`, each handed in at its end as its
        marked grant does, client run 0's summary written at its end and
        client run 1's at the finish; the summaries' ``hit_counters``."""
        with compass_run.engine_loop(self.core):
            while self.rt.now < end_0:
                self.step()
            self.rt.client_run_ended(0)
            compass_run._hand_in(self.run, "frontend", [], 0)
            a = self.authority
            a.ended.append(ClientRun(0, 0.0, end_0, ((compass_run.ENGINE, end_0),), {}))
            compass_run._write_client_run(self.run, a, 0)
            for _ in STEPS[len(self.atom) :]:
                self.step()
            end = self.rt.now
            a.ended.append(ClientRun(1, end_0, end, ((compass_run.ENGINE, end),), {}))
            self.rt.now = math.inf  # the finish
        compass_run._hand_in(self.run, "frontend", [], None)
        compass_run._write_summary(self.run, a)
        out = Path(self.run["out_dir"])
        return [
            _read(out / f"client-{k}" / compass_run.SUMMARY_FILE)["hit_counters"]
            for k in (0, 1)
        ]


@pytest.mark.parametrize("end_0", ENDS_0)
@pytest.mark.parametrize("method", ("_pp_head_step", "_process_engine_step"))
@pytest.mark.parametrize("tier", [True, False], ids=["tier", "no-tier"])
def test_an_engine_samples_its_hit_counters_after_each_step_that_moved_one(
    monkeypatch, tmp_path, method, tier, end_0
):
    """Every engine samples, its connector's ``loaded_tokens`` 0 with none, at
    its run's start and after each step that moved a counter, each sample
    ATOM's counters after that step at the engine's clock; client run 1's
    open with the sample in force at its start, taken at that very time or
    before it."""
    engine = _Engine(
        monkeypatch, tmp_path, method, tier, EngineStats(enable_prefix_caching=True)
    )
    runs = engine.serve(end_0)
    print(f"\n{method} tier={tier} end_0={end_0}:", json.dumps(runs))
    atom = {s[0]: s for s in [[0.0, 0, 0, 0], *engine.atom]}
    series = [atom[t] for t in SAMPLED[tier]]
    assert [list(r) for r in runs] == [["engine"], ["engine"]]
    assert runs[0]["engine"] == [s for s in series if s[0] <= IN_FORCE]
    assert runs[1]["engine"] == [s for s in series if s[0] >= IN_FORCE]
    assert runs[1]["engine"][-1] == engine.atom[-1]


def test_each_cache_stats_line_its_engine_logs_is_a_sample_in_order(
    monkeypatch, tmp_path, caplog
):
    """ATOM's cumulative ``[Cache Stats]`` line, logged here at each prefill,
    states the cached and full tokens of a sample, in order."""
    stats = EngineStats(enable_prefix_caching=True, cache_log_interval=1)
    engine = _Engine(monkeypatch, tmp_path, "_process_engine_step", True, stats)
    with caplog.at_level(logging.INFO, "atom"):
        runs = engine.serve()
    line = re.compile(r"\[Cache Stats +\] .*Cached/Reusable: (\d+)/\d+, .* of (\d+)")
    logged = [
        [int(g) for g in m.groups()] for m in map(line.search, caplog.messages) if m
    ]
    print("\nlogged:", logged)
    assert len(logged) == sum(1 for prefill, _, _ in STEPS if prefill)
    samples = iter(s[1:3] for s in runs[0]["engine"] + runs[1]["engine"][1:])
    assert all(any(s == said for s in samples) for said in logged)


def test_a_real_run_s_engine_samples_nothing(monkeypatch):
    """With no run file the engine's step is ATOM's own, unwrapped."""
    monkeypatch.delenv(compass_run.ENV, raising=False)
    monkeypatch.setattr(clock, "_installed", None)
    step = lambda: None
    core = SimpleNamespace(
        scheduler=SimpleNamespace(engine_stats=EngineStats(enable_prefix_caching=True)),
        _process_engine_step=step,
    )
    with compass_run.engine_loop(core):
        assert core._process_engine_step is step
