# SPDX-License-Identifier: MIT
"""A DP request's session header, from the client to ATOM's router, the session
id a real and a simulated client's aiperf mint, and each request's rank
compared between a real and a simulated record.

`client.sh` runs with the stand-ins of ``test_client_sh.py``; the headers are
the ones aiperf's own transport builds under the environment it handed aiperf,
and ATOM's API server reads its session key from them. Requests are then routed
by ATOM's `CoreManager`, built bare as ``tests/test_dp_load_balance.py`` builds
it, behind the frontend `run.frontend` sets up on a recording server, and
`parity.compare` reads the routes it recorded.
"""

import json
import os
import pickle
import subprocess
import sys
from importlib import metadata
from types import SimpleNamespace

import pytest
from starlette.datastructures import Headers
from test_client_sh import AGENTX, DYNAMO, REAL, SIM, TREE, _client, _setup
from test_dp_load_balance import _make_mgr
from test_parity_record import BLOCK, _drive, _key

from atom.compass import run
from atom.compass.parity import ENV, ROUTES, compare
from atom.entrypoints.openai.api_server import _get_dp_session_affinity_ids
from atom.model_engine.engine_core_mgr import CoreManager, DisaggCoreManager
from atom.model_engine.sequence import Sequence

#: aiperf's transport building one request's headers under this environment:
#: a root's and a subagent child's, whose parent is "p".
HEADERS = """
import json
from types import SimpleNamespace as N
from aiperf.transports.base_transports import BaseTransport
def built(parent):
    endpoint = N(endpoint=N(session_header=None))
    info = N(x_request_id="r", x_correlation_id="c", parent_correlation_id=parent,
             model_endpoint=endpoint, endpoint_headers={}, turns=[])
    transport = N(base_headers={}, get_transport_headers=lambda _: {})
    return BaseTransport.build_headers(transport, info)
print(json.dumps([built(None), built("p")]))
"""


def _headers(env: dict) -> list[dict]:
    pytest.importorskip("aiperf")
    aiperf = {k: v for k, v in env.items() if k.startswith("AIPERF_HTTP_")}
    others = {k: v for k, v in os.environ.items() if not k.startswith("AIPERF_")}
    done = subprocess.run(
        [sys.executable, "-c", HEADERS],
        env=others | aiperf,
        capture_output=True,
        text=True,
        check=True,
        cwd="/",
    )
    return json.loads(done.stdout)


@pytest.mark.parametrize("server", [SIM, REAL], ids=["simulated", "real"])
def test_the_client_sends_the_submission_s_headers_and_atom_keys_them_alike(
    tmp_path, monkeypatch, server
):
    out, env = _setup(tmp_path, server)
    env = {k: v for k, v in env.items() if not k.startswith("AIPERF_")}
    done = _client(out, *AGENTX, env=env | {DYNAMO: "1"})
    assert done.returncode == 0, done.stdout + done.stderr
    sent = _headers(json.loads((tmp_path / "seen.json").read_text())[1])

    # The submission's client sets the Dynamo header alone.
    assert sent == _headers({DYNAMO: "1"})
    root, child = sent
    assert root == {
        "X-Request-ID": "r",
        "X-Correlation-ID": "c",
        "X-Dynamo-Session-ID": "c",
    }
    assert child == root | {"X-Dynamo-Parent-Session-ID": "p"}
    monkeypatch.setenv("ATOM_DP_SESSION_AFFINITY", "1")
    keys = [
        _get_dp_session_affinity_ids(SimpleNamespace(headers=Headers(h))) for h in sent
    ]
    assert keys == [("c", None), ("c", "p")]


#: aiperf minting one run's sessions: three lanes drawing traces t0, t1 and t0,
#: each from one instant of it, where its root and first subagent run, then
#: each root's next subagent spawned in the order $ORDER names, then t1 drawn
#: again as a lane is recycled. Prints each session's headers by lane and
#: conversation.
MINT = """
import json, os
from types import SimpleNamespace as N
from aiperf.common.models import ConversationMetadata, DatasetMetadata, TurnMetadata
from aiperf.common.models.branch import ConversationBranchInfo
from aiperf.timing.strategies.agentic_replay import AgenticReplayStrategy
from aiperf.timing.trajectory_source import TrajectorySource
from aiperf.transports.base_transports import BaseTransport
def conv(cid, *ts, **kw):
    turns = [TurnMetadata(timestamp_ms=t) for t in ts]
    return ConversationMetadata(conversation_id=cid, turns=turns, **kw)
convs = []
for t in ("t0", "t1"):
    subagents = [f"{t}::sa:{k}" for k in (0, 2)]
    branches = [ConversationBranchInfo(branch_id=f"{t}:{c[-1]}", mode="spawn",
                                       child_conversation_ids=[c]) for c in subagents]
    convs.append(conv(t, 0, 1000, 2000, 3000, branches=branches))
    convs += [conv(c, 500 * (2 * k + 1), 500 * (2 * k + 3), is_root=False, agent_depth=1)
              for k, c in zip((0, 2), subagents)]
draws = iter(["t0", "t1", "t0"])
dataset = DatasetMetadata(conversations=convs, sampling_strategy="sequential")
lanes = TrajectorySource(dataset_metadata=dataset, concurrency=3, random_seed=1,
    dataset_sampler=N(next_conversation_id=lambda: next(draws)),
    start_min_ratio=0.4, start_max_ratio=0.4, allow_dataset_wrap=True)
sessions = {f"{k}/{s.conversation_id}": s
            for k, lane in enumerate(lanes.trajectories) for s in lane.snapshot.states}
for k in map(int, os.environ["ORDER"]):
    root = lanes.trajectories[k].conversation_id
    parent = sessions[f"{k}/{root}"].x_correlation_id
    sessions[f"{k}/{root}::sa:2"] = lanes.start_branch_child(parent, f"{root}::sa:2", 1)
replay = N(conversation_source=lanes)
sessions["3/t1"] = AgenticReplayStrategy._build_session_for_trace(replay, "t1")
endpoint = N(endpoint=N(session_header=None))
transport = N(base_headers={}, get_transport_headers=lambda _: {})
print(json.dumps({name: BaseTransport.build_headers(transport, N(
    x_request_id="r", x_correlation_id=s.x_correlation_id, endpoint_headers={},
    parent_correlation_id=s.parent_correlation_id, model_endpoint=endpoint, turns=[]))
    for name, s in sessions.items()}))
"""
#: Runs a real client's aiperf as client.sh does, its arguments after ``-c``
#: given, under the default start method $START and no compass-harness. Its CLI
#: sets the start method $SET where given, as aiperf may, then starts a
#: process as aiperf starts its timing manager, which runs `MINT`.
REAL_AIPERF = """
import multiprocessing as mp, os, sys, types
mp.set_start_method(os.environ["START"])
sys.modules["compass_harness"] = None
def app():
    if "SET" in os.environ:
        mp.set_start_method(os.environ["SET"], force=True)
    mint = "import sys\\nsys.modules['compass_harness'] = None\\n" + os.environ["MINT"]
    service = mp.Process(target=exec, args=(mint, {}))
    service.start()
    service.join()
    sys.exit(service.exitcode)
cli = sys.modules["aiperf.cli"] = types.ModuleType("aiperf.cli")
cli.app = app
exec(sys.argv.pop(1), {})
"""
#: A simulated client's aiperf service, started by spawn: compass-harness as
#: aiperf's plugin discovery imports it in each process, then `MINT`.
SIMULATED_AIPERF = """
import multiprocessing as mp, os
mint = "from aiperf.plugin import plugins\\n" + os.environ["MINT"]
service = mp.get_context("spawn").Process(target=exec, args=(mint, {}))
service.start()
service.join()
"""


def test_a_real_and_a_simulated_run_send_one_session_id_per_trace_and_branch(
    tmp_path,
):
    pytest.importorskip("aiperf")
    try:
        metadata.version("compass-harness")
    except metadata.PackageNotFoundError:
        pytest.skip("needs compass-harness installed as aiperf's plugin")
    out, env = _setup(tmp_path, REAL)
    done = _client(out, *AGENTX, env=env)
    assert done.returncode == 0, done.stdout + done.stderr
    real = [REAL_AIPERF, *json.loads((tmp_path / "seen.json").read_text())[0][1:3]]
    run = {k: v for k, v in os.environ.items() if not k.startswith("AIPERF_")}

    def aiperf(code, **more):
        return subprocess.run(
            [sys.executable, "-c", *code],
            env=run | {DYNAMO: "1", "MINT": MINT} | more,
            capture_output=True,
            text=True,
            check=False,
            cwd="/",
        )

    # Each run spawns the subagents in its own order, as its clock has it.
    done = aiperf([SIMULATED_AIPERF], ORDER="210")
    assert done.returncode == 0, done.stderr
    sent = json.loads(done.stdout.splitlines()[-1])
    for start in ("fork", "spawn", "forkserver"):  # forkserver from Python 3.14
        done = aiperf(real, ORDER="012", START=start)
        assert done.returncode == 0, done.stderr
        assert json.loads(done.stdout.splitlines()[-1]) == sent, start
    done = aiperf(real, ORDER="012", START="fork", SET="spawn")
    assert done.returncode == 1 and not done.stdout
    assert "aiperf asked for start method spawn" in done.stderr
    ids = {name: headers["X-Dynamo-Session-ID"] for name, headers in sent.items()}
    assert len(set(ids.values())) == len(ids) == 3 * 3 + 1
    for k, root in enumerate(["t0", "t1", "t0"]):
        for child in (0, 2):
            headers = sent[f"{k}/{root}::sa:{child}"]
            assert headers["X-Dynamo-Parent-Session-ID"] == ids[f"{k}/{root}"]


def test_without_the_arm_s_header_the_client_sends_x_session_id_as_before(
    tmp_path, monkeypatch
):
    out, env = _setup(tmp_path, SIM)
    env = {k: v for k, v in env.items() if not k.startswith("AIPERF_")}
    done = _client(out, *AGENTX, env=env)
    assert done.returncode == 0, done.stdout + done.stderr
    root, child = _headers(json.loads((tmp_path / "seen.json").read_text())[1])
    assert root == {"X-Request-ID": "r", "X-Correlation-ID": "c", "X-Session-ID": "c"}
    assert child == root
    monkeypatch.setenv("ATOM_DP_SESSION_AFFINITY", "1")
    request = SimpleNamespace(headers=Headers(root))
    assert _get_dp_session_affinity_ids(request) == ("c", None)


#: Each request's session and prompt length. Session a's second turn extends its
#: first; d and e send one prompt, twins.
REQUESTS = [("a", 300), ("b", 100), ("c", 50), ("a", 350), ("d", 20), ("e", 20)]
FIRST = {"a": 1000, "b": 2000, "c": 3000, "d": 4000, "e": 4000}


def _tokens(session, n):
    return list(range(FIRST[session], FIRST[session] + n))


def _route(record, a_prefilled_before_c) -> list[int]:
    """Route `REQUESTS`, as the submission's DP points do, through ATOM's core
    manager behind the frontend of a server recording into `record`; the rank
    each was sent to. Session a's first turn finishes its prefill before c
    arrives where `a_prefilled_before_c`, after otherwise, as two runs whose
    clocks part there."""
    sent = []
    with pytest.MonkeyPatch.context() as patch:
        for cls in (CoreManager, DisaggCoreManager):  # ATOM's back after
            patch.setattr(cls, "add_request", cls.add_request)
        patch.delenv(run.ENV, raising=False)
        patch.setenv(ENV, str(record))
        with run.frontend(SimpleNamespace(runner_qualname=run.ATOM_RUNNER)):
            pass
        manager = _make_mgr(2, "least_tokens", 512, session_affinity=True)
        manager.pp_size, manager._seq_id_to_callback = 1, {}
        manager._send_request = lambda rank, payload: sent.extend(
            rank for _ in pickle.loads(payload)[1]
        )
        first = None
        for session, n in REQUESTS:
            if session == "c" and a_prefilled_before_c:
                manager._mark_seq_prefill_complete(first.id)
            seq = Sequence(_tokens(session, n), BLOCK, dp_session_id=session)
            first = first or seq
            manager.add_request([seq])
        run._move_line_files()  # as the server stops
    return sent


def _lines(record) -> list[dict]:
    return [json.loads(line) for line in (record / ROUTES).read_text().splitlines()]


def test_a_recording_frontend_writes_each_route_and_the_load_it_weighed(tmp_path):
    ranks = _route(tmp_path, a_prefilled_before_c=True)
    lines = _lines(tmp_path)
    assert [line["key"] for line in lines] == [_key(_tokens(*r)) for r in REQUESTS]
    assert [line["rank"] for line in lines] == ranks
    a, b = ranks[:2]
    assert {a, b} == {0, 1} and ranks[3] == a  # a's second turn stays with a
    assert lines[0] | {"key": None} == {
        "key": None,
        "rank": a,
        "reqs": [0, 0],
        "tokens": [0, 0],
    }
    # c sees a's prefill done and b's queued: one request a rank.
    assert lines[2]["reqs"] == [1, 1]
    assert lines[2]["tokens"][a] == 0 and lines[2]["tokens"][b] == 100


def test_a_recording_frontend_of_one_rank_writes_no_routes(tmp_path):
    with pytest.MonkeyPatch.context() as patch:
        for cls in (CoreManager, DisaggCoreManager):
            patch.setattr(cls, "add_request", cls.add_request)
        patch.delenv(run.ENV, raising=False)
        patch.setenv(ENV, str(tmp_path))
        with run.frontend(SimpleNamespace(runner_qualname=run.ATOM_RUNNER)):
            pass
        manager = _make_mgr(1, "least_tokens", 512, session_affinity=True)
        manager.pp_size, manager._seq_id_to_callback = 1, {}
        manager._send_request = lambda rank, payload: None
        manager.add_request([Sequence(_tokens("a", 10), BLOCK, dp_session_id="a")])
        run._move_line_files()
    assert not (tmp_path / ROUTES).exists()


@pytest.mark.usefixtures("unpriced")
def test_compare_names_the_first_request_two_runs_routed_apart(tmp_path, monkeypatch):
    real, simulated = tmp_path / "real", tmp_path / "simulated"
    for record in (real, simulated):
        _drive(record, monkeypatch, {0: [0], 1: [1]})
    ranks = {"real": _route(real, True), "simulated": _route(simulated, False)}
    lines = {"real": _lines(real), "simulated": _lines(simulated)}
    a = ranks["real"][0]
    assert ranks["simulated"][0] == a  # one session id, one tie-break
    # c went where a's prefill had left on the real run, and the other way
    # round on the simulated one; so did the second twin.
    assert [x == y for x, y in zip(*ranks.values())] == [1, 1, 0, 1, 1, 0]
    report = compare(real, simulated)["routes"]
    assert report == {
        "requests_per_rank": {
            name: [r.count(0), r.count(1)] for name, r in ranks.items()
        },
        "routed_by_both": 6,
        "routed_alike": 4,
        "first_unlike": {name: lines[name][2] for name in lines},
    }
    first = report["first_unlike"]
    assert first["real"]["tokens"][a] == 0 and first["simulated"]["tokens"][a] == 300

    done = subprocess.run(
        [sys.executable, "-m", "atom.compass.parity", str(real), str(simulated)],
        env=os.environ | {"PYTHONPATH": str(TREE)},
        capture_output=True,
        text=True,
        check=True,
        cwd="/",
    )
    assert json.loads(done.stdout)["routes"] == report
