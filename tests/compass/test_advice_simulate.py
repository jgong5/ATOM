# SPDX-License-Identifier: MIT
"""`compass advice SPACE OUT`'s simulation: each searchable point runs on a
fresh server driven by one client run, in parallel, a failed point tried again
and then listed, a finished one skipped on a rerun.

The check is `test_advice_check`'s, its law the re-keyed DPA8 law as a law
file. `compass run`, `deploy.sh` and `client.sh` are stubs that log each call,
a server's with the ``STUB_X`` it runs under. A server answers at once, and on
SIGTERM writes the run summary and the run file, but no summary for the point
``STUB_NO_SUMMARY`` names, and exits 0, or 3 for the point
``STUB_SERVER_EXIT`` names; under a run directory it leaves before answering
for the points ``STUB_FAILS`` names, as many times as it says. The client
spends the CPU seconds ``STUB_CLIENT_CPU_S`` gives, sleeps the seconds
``STUB_CLIENT_S`` gives its point, waits, up to 60 s, until the calls log holds
the stop of the point ``STUB_AFTER`` maps its point to, writes the client
export, then stops the server through its ``server.json`` and waits for it to
leave; it exits 1 for the points ``STUB_CLIENT_FAIL`` names, and for the point
``STUB_HANG`` names writes its pid and sleeps. The tree each point runs on,
`serve.TREE`, reads as `TREE`, with no uncommitted file, unless a test says
otherwise.
"""

import itertools
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path

import platforms
import pytest
from test_advice_check import NETWORK, starts, stubbed  # noqa: F401  the fixture
from test_advice_reuse import law, law_file
from test_memory_readings import publish_readings
from test_run_step_grid import DOCUMENT

from atom.compass import cli
from atom.compass.advice import check, simulate
from atom.compass.artifacts.provenance import SourceRoot
from atom.compass.drive import reuse, serve

SERVER = """#!{python}
import json, os, signal, sys, time
from pathlib import Path

argv = sys.argv[1:]
at = Path(argv[0] if argv[0] != "run" else argv[argv.index("--out") + 1])
def log(**entry):
    with open(os.environ["STUB_CALLS"], "a") as calls:
        calls.write(json.dumps(entry) + "\\n")
log(argv=argv, x=os.environ.get("STUB_X"))
at.mkdir(parents=True)
def stop(*_):
    for name in ("summary.json", "run_file.json"):
        if name != "summary.json" or at.name != os.environ.get("STUB_NO_SUMMARY"):
            (at / name).write_text(json.dumps({{"point": at.name}}))
    log(down=str(at))
    sys.exit(3 if at.name == os.environ.get("STUB_SERVER_EXIT") else 0)
signal.signal(signal.SIGTERM, stop)
fails = json.loads(os.environ.get("STUB_FAILS", "{{}}")).get(at.name, 0)
if at.parent.name == "runs" and fails:
    tried = Path(os.environ["STUB_CALLS"] + "." + at.name)
    n = int(tried.read_text()) if tried.exists() else 0
    tried.write_text(str(n + 1))
    if n < fails:
        sys.exit("compass run: the server exited 1 before /health answered")
(at / "server.json").write_text(json.dumps({{"pid": os.getpid()}}))
log(up=str(at))
while True:
    time.sleep(0.05)
"""
CLIENT = """#!{python}
import json, os, signal, sys, time
from pathlib import Path

argv = sys.argv[1:]
at = Path(argv[0])
held = {{k: os.environ.get(k) for k in ("MODEL", "TRACES", "HARNESS_PYTHON")}}
with open(os.environ["STUB_CALLS"], "a") as calls:
    calls.write(json.dumps({{"client": argv, "env": held}}) + "\\n")
if at.name in os.environ.get("STUB_CLIENT_FAIL", "").split():
    sys.exit("client: aiperf: exited 1; see " + str(at / "client" / "aiperf.log"))
if at.name == os.environ.get("STUB_HANG"):
    Path(os.environ["STUB_CALLS"] + ".t").write_text(str(os.getpid()))
    os.rename(os.environ["STUB_CALLS"] + ".t", os.environ["STUB_CALLS"] + ".hung")
    time.sleep(600)
spent = time.process_time() + float(os.environ.get("STUB_CLIENT_CPU_S", 0))
while time.process_time() < spent:
    pass
time.sleep(json.loads(os.environ.get("STUB_CLIENT_S", "{{}}")).get(at.name, 0))
after = json.loads(os.environ.get("STUB_AFTER", "{{}}")).get(at.name)
down = json.dumps({{"down": str(at.parent / after)}}) if after else ""
for _ in range(1200):
    if down in Path(os.environ["STUB_CALLS"]).read_text():
        break
    time.sleep(0.05)
(at / "artifacts").mkdir()
(at / "artifacts" / "client_run.json").write_text(json.dumps({{"point": at.name}}))
pid = json.loads((at / "server.json").read_text())["pid"]
os.kill(pid, signal.SIGTERM)
stat = Path(f"/proc/{{pid}}/stat")
while stat.exists() and stat.read_text().rsplit(")", 1)[1].split()[0] != "Z":
    time.sleep(0.05)
"""
#: Three single-node candidates the check finds searchable at both points.
THREE = {"a": {}, "b": {"env": {"STUB_X": "1"}}, "c": {"env": {"STUB_X": "2"}}}
SIX = [f"{c}-c{n}" for c in "abc" for n in (1, 4)]
#: The tree each point runs on, and another.
TREE, ANOTHER = "1e47fbd62" + "0" * 31, "54a68b588" + "0" * 31


def tree(revision: str, dirty: int | None = 0):
    """`git_tree_root`, reading the tree each point runs on, `serve.TREE`, as
    `revision`, with `dirty` uncommitted files."""

    def read(name: str, root: str) -> SourceRoot:
        assert root == str(serve.TREE)
        return SourceRoot(name, root, revision, "tree", "a stub", dirty)

    return read


@pytest.fixture
def simulating(tmp_path, stubbed, monkeypatch):  # noqa: F811
    """`stubbed`, its law in a law file, each start and run the stub server's,
    each client run the stub client's, under a HARNESS_PYTHON."""
    law_file(tmp_path, law(stubbed))
    for name, text in (("server", SERVER), ("client", CLIENT)):
        (tmp_path / name).write_text(text.format(python=sys.executable))
        (tmp_path / name).chmod(0o755)
    monkeypatch.setattr(check, "COMPASS", tmp_path / "server")
    monkeypatch.setattr(check, "DEPLOY", tmp_path / "server")
    monkeypatch.setattr(simulate, "CLIENT", tmp_path / "client")
    monkeypatch.setattr(simulate, "git_tree_root", tree(TREE))
    monkeypatch.setenv("HARNESS_PYTHON", "/sim/python")
    stubs = "FAILS CLIENT_FAIL NO_SUMMARY HANG CLIENT_S CLIENT_CPU_S SERVER_EXIT AFTER"
    for name in stubs.split():
        monkeypatch.delenv(f"STUB_{name}", raising=False)
    return stubbed


def searched(
    tmp_path, dimensions: dict, out="out", network=None, whole=True, **workload
) -> dict:
    """`compass advice` on `dimensions`' candidates at concurrencies 1 and 4,
    on the suite's platform and `network`, with `workload`'s fields in place
    of its own, into `tmp_path`'s `out`, named whole or, unless `whole`, from
    `tmp_path` as the working directory; the outcomes the simulation writes."""
    flags = platforms.flags(tmp_path / "store", DOCUMENT, tmp_path)
    machine = {"device": flags[1], "ptl": flags[3]}
    space = {
        "model": str(tmp_path),
        "machine": machine | ({"network": network} if network else {}),
        "store": str(tmp_path / "store"),
        "workload": {
            "traces": str(tmp_path / "traces.jsonl"),
            "select": "agentx",
            "concurrency": [1, 4],
        }
        | workload,
        "slo": {},
        "common": {"basis": "a recipe"},
        "dimensions": {"config": dimensions},
    }
    (tmp_path / "space.json").write_text(json.dumps(space))
    given = str(tmp_path / out) if whole else out
    cli.main(["advice", str(tmp_path / "space.json"), given])
    return json.loads((tmp_path / out / simulate.SIMULATED).read_text())


def calls(tmp_path, what: str) -> list:
    """Each call the stubs logged holding `what`: ``argv`` a server's,
    ``client`` a client's, ``up`` and ``down`` a server's answer and stop."""
    return [c for c in starts(tmp_path) if what in c]


def ran(tmp_path, at: Path, what="argv") -> list:
    """Each server started into `at`, `compass run`'s ``--out`` or `deploy.sh`'s
    OUT, in order: its command, or, as `what` ``x``, the ``STUB_X`` it ran
    under."""

    def into(argv: list[str]) -> str:
        return argv[argv.index("--out") + 1] if argv[0] == "run" else argv[0]

    return [c[what] for c in calls(tmp_path, "argv") if str(at) == into(c["argv"])]


@pytest.mark.parametrize("whole", [True, False])
def test_each_searchable_point_runs_on_a_fresh_server_with_one_client_run(
    tmp_path, simulating, monkeypatch, capsys, whole
):
    # b reads readings of its own at both points: each is missing readings.
    dimensions = {"a": {"env": {"STUB_X": "1"}}}
    dimensions["b"] = {"per_concurrency": {"--max-model-len": 256}}
    monkeypatch.setenv("STUB_CLIENT_CPU_S", "0.2")
    if not whole:
        monkeypatch.chdir(tmp_path)

    found = searched(tmp_path, dimensions, whole=whole, select="0-7", duration_s=600)

    out = tmp_path / "out"
    assert list(found["points"]) == ["a-c1", "a-c4"]
    (started,) = ran(tmp_path, out / check.STARTS / "a-c1")
    for point, n in (("a-c1", 1), ("a-c4", 4)):
        at = out / simulate.RUNS / point
        held = found["points"][point]
        assert held["cpu_s"] >= 0.2
        assert held == {
            "state": "finished",
            "run": str(at),
            "cpu_s": held["cpu_s"],
            "tries": 1,
            "skipped": False,
        }
        for name in simulate.HELD:
            assert json.loads((at / name).read_text()) == {"point": point}
        # The check's start, into the point's own run directory.
        assert ran(tmp_path, at) == [
            [str(at) if w == str(out / check.STARTS / "a-c1") else w for w in started]
        ]
        assert ran(tmp_path, at, "x") == ["1"]
        (client,) = [c for c in calls(tmp_path, "client") if c["client"][0] == str(at)]
        assert client == {
            "client": [str(at), "--traces", "0-7", "--concurrency", str(n)]
            + ["--duration", "600", "--stop"],
            "env": {
                "MODEL": str(tmp_path),
                "TRACES": str(tmp_path / "traces.jsonl"),
                "HARNESS_PYTHON": "/sim/python",
            },
        }
    printed = capsys.readouterr().out
    assert f"  a-c4: finished in {out / simulate.RUNS / 'a-c4'}, " in printed
    assert "2 of 2 points finished, " in printed


@pytest.mark.parametrize(
    "stub, said",
    [
        ({"STUB_FAILS": '{"a-c1": 2}'}, "the server exited 1 before /health answered"),
        ({"STUB_CLIENT_FAIL": "a-c1"}, "client: aiperf: exited 1; see "),
        ({"STUB_NO_SUMMARY": "a-c1"}, "no summary.json in "),
        ({"STUB_SERVER_EXIT": "a-c1"}, "server exited 3"),
    ],
)
def test_a_failed_point_is_tried_again_then_listed_with_its_reason_and_left_out(
    tmp_path, simulating, monkeypatch, capsys, stub, said
):
    for name, value in stub.items():
        monkeypatch.setenv(name, value)

    found = searched(tmp_path, {"a": {}})

    failed = found["points"]["a-c1"]
    assert (failed["state"], failed["tries"]) == ("failed", 2)
    (why,) = failed["why"]
    assert said in why
    assert len(ran(tmp_path, tmp_path / "out" / simulate.RUNS / "a-c1")) == 2
    assert not (tmp_path / "out" / simulate.RUNS / "a-c1" / simulate.RAN).exists()
    assert found["points"]["a-c4"]["state"] == "finished"
    printed = capsys.readouterr().out
    assert f"  a-c1: failed after 2 tries\n    {why}\n" in printed
    assert "1 of 2 points finished, " in printed


def test_a_point_that_fails_once_finishes_on_its_second_try(
    tmp_path, simulating, monkeypatch
):
    monkeypatch.setenv("STUB_FAILS", '{"a-c1": 1}')

    found = searched(tmp_path, {"a": {}})

    assert found["points"]["a-c1"]["state"] == "finished"
    assert found["points"]["a-c1"]["tries"] == 2


def test_a_rerun_skips_each_finished_point_and_runs_what_did_not_finish_or_changed(
    tmp_path, simulating, monkeypatch
):
    monkeypatch.setenv("STUB_CLIENT_FAIL", "a-c1")
    first = searched(tmp_path, {"a": {}})
    monkeypatch.delenv("STUB_CLIENT_FAIL")
    before = len(calls(tmp_path, "client"))

    again = searched(tmp_path, {"a": {}})

    assert first["points"]["a-c1"]["state"] == "failed"
    assert again["points"]["a-c4"] == first["points"]["a-c4"] | {"skipped": True}
    assert again["points"]["a-c1"]["state"] == "finished"
    assert [c["client"][0] for c in calls(tmp_path, "client")[before:]] == [
        str(tmp_path / "out" / simulate.RUNS / "a-c1")
    ]
    before = len(calls(tmp_path, "client"))

    changed = searched(tmp_path, {"a": {}}, duration_s=60)

    assert [p["skipped"] for p in changed["points"].values()] == [False, False]
    assert len(calls(tmp_path, "client")) == before + 2
    before = len(calls(tmp_path, "client"))
    # The same space at another head, on the same tree.
    taken = reuse.head() | {"commit": "deadbeef" * 5}
    monkeypatch.setattr(reuse, "head", lambda commit=None: taken)

    moved = searched(tmp_path, {"a": {}}, duration_s=60)

    assert [p["skipped"] for p in moved["points"].values()] == [True, True]
    # On another tree; then twice with an uncommitted file; then clean again.
    for revision, dirty in ((ANOTHER, 0), (ANOTHER, 1), (ANOTHER, 1), (ANOTHER, 0)):
        monkeypatch.setattr(simulate, "git_tree_root", tree(revision, dirty))

        rerun = searched(tmp_path, {"a": {}}, duration_s=60)

        assert [p["skipped"] for p in rerun["points"].values()] == [False, False]
    assert len(calls(tmp_path, "client")) == before + 8


@pytest.mark.parametrize("cpus, tp, at_once", [(1, 1, 1), (4, 2, 2)])
def test_points_run_as_many_at_once_as_the_cpus_hold_the_widest_point_s_ranks(
    tmp_path, simulating, monkeypatch, cpus, tp, at_once
):
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(cpus)))
    monkeypatch.setenv("STUB_CLIENT_S", json.dumps(dict.fromkeys(SIX, 0.3)))
    dimensions = THREE | {"c": {"args": {"--tensor-parallel-size": tp}}}

    found = searched(tmp_path, dimensions)

    assert (found["cpus"], found["at_once"]) == (cpus, at_once)
    assert [p["state"] for p in found["points"].values()] == ["finished"] * 6
    up, most = 0, 0
    for c in starts(tmp_path):
        if f"/{simulate.RUNS}/" in c.get("up", c.get("down", "")):
            up += 1 if "up" in c else -1
            most = max(most, up)
    assert most == at_once


def test_the_same_inputs_give_the_same_results_whatever_order_points_end_in(
    tmp_path, simulating, monkeypatch
):
    # One at a time, then all at once with each point ending after the next.
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0})
    one = searched(tmp_path, THREE)
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(6)))
    monkeypatch.setenv("STUB_AFTER", json.dumps(dict(itertools.pairwise(SIX))))

    every = searched(tmp_path, THREE, out="again")

    def held(found: dict, out: str) -> dict:
        points = {}
        for name, outcome in found["points"].items():
            at = tmp_path / out / simulate.RUNS / name
            record = (at / simulate.RAN).read_text().replace(str(tmp_path / out), "OUT")
            files = [(at / f).read_text() for f in simulate.HELD]
            points[name] = (outcome["state"], json.loads(record)["ran"], files)
        return points

    assert (one["at_once"], every["at_once"]) == (1, 6)
    ended = [c["down"] for c in calls(tmp_path, "down") if "/again/runs/" in c["down"]]
    assert [Path(d).name for d in ended] == SIX[::-1]
    assert held(one, "out") == held(every, "again")
    assert list(one["points"]) == list(every["points"]) == SIX


def test_a_1p1d_point_runs_its_deployment_and_one_client_through_it(
    tmp_path, simulating, monkeypatch
):
    # A side's keys hold the AITER_LOG_LEVEL deploy.sh sets.
    monkeypatch.setenv("AITER_LOG_LEVEL", "WARNING")
    publish_readings(tmp_path / "store", simulating.config)
    law_file(tmp_path, law(simulating))
    side = {"deployment": "1p1d", "env": {"STUB_BOTH": "1"}}

    found = searched(tmp_path, {"p": side}, network=NETWORK)

    at = tmp_path / "out" / simulate.RUNS / "p-c1"
    assert found["points"]["p-c1"]["state"] == "finished"
    (deployed,) = ran(tmp_path, at)
    assert deployed[1:3] == ["run", "--device"]
    # ATOM's adapter, and none of the router's wall-clock detectors.
    assert deployed[deployed.index("--router") :] == [
        *("--router", "--", "--backend", "atom", "--disable-health-check"),
        *("--disable-circuit-breaker", "--disable-retries"),
    ]
    assert [c["client"][0] for c in calls(tmp_path, "client")].count(str(at)) == 1
    assert found["at_once"] == max(1, len(os.sched_getaffinity(0)) // 2)


def test_with_harness_python_unset_no_point_runs_and_each_says_why(
    tmp_path, simulating, monkeypatch, capsys
):
    monkeypatch.delenv("HARNESS_PYTHON")

    found = searched(tmp_path, {"a": {}})

    assert found["points"]["a-c1"] == {
        "state": "failed",
        "why": [simulate.UNSET],
        "tries": 0,
    }
    assert calls(tmp_path, "client") == []
    assert not (tmp_path / "out" / simulate.RUNS).exists()
    assert f"  a-c4: not run\n    {simulate.UNSET}\n" in capsys.readouterr().out


def _alive(pid: int) -> bool:
    stat = Path(f"/proc/{pid}/stat")
    try:
        return stat.read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return False


def test_sigterm_stops_every_point_s_commands_and_names_the_signal(
    tmp_path, simulating, monkeypatch
):
    monkeypatch.setenv("STUB_HANG", "a-c1")
    hung = Path(os.environ["STUB_CALLS"] + ".hung")

    def stop():
        for _ in range(1200):
            if hung.exists():
                os.kill(os.getpid(), signal.SIGTERM)
                return
            time.sleep(0.05)

    threading.Thread(target=stop, daemon=True).start()
    with pytest.raises(SystemExit, match="stopped by signal 15"):
        searched(tmp_path, {"a": {}})

    at = tmp_path / "out" / simulate.RUNS / "a-c1"
    assert not _alive(int(hung.read_text()))
    assert len(ran(tmp_path, at)) == 1
    assert str(at) in [c["down"] for c in calls(tmp_path, "down")]
    assert not (at / simulate.RAN).exists()
