# SPDX-License-Identifier: MIT
"""The router LP at 1P1D DP2, with no driver: atomesh, built from this tree, run
as `pd_sim.sh` runs the agentic cell's router (``--dp-aware``, dp_sticky), under
the standalone clock authority, between scripted frontends.

Each frontend is its LP and answers the router at a constant cost: prefill
sends its relay `PREFILL_S` after the request reaches it, decode one stream
event per token, `DECODE_S` apart. The engines are idle LPs. The run is made
twice, the second under another `PYTHONHASHSEED` and with a random wall delay
on every hop into and out of the router: before each send by the traffic LP and
the frontends, and before a frontend reads what the router posted.
`check_run` holds the two to the scenario checks, and each request's (prefill,
decode) rank must be the least-load choice worked out by hand in `RANKS`. The
same pair of runs under ``--policy random`` (as `pd_sim.sh` runs a cell that
maps ranks outside aiperf_agentic) must pick the same ranks, and
``--policy power_of_two`` must be refused by name.

cache_aware, as `pd_sim.sh` runs it, and prefix_hash make the same pair of
runs over requests that share prefixes, the frontends on the same ports in
both, as one deployment's are; cache_aware evicts every 300 s, `pd_sim.sh`'s
interval, past the run's end (atomesh refuses 0), and every 1 s, inside it.
Each pair meets the scenario checks and picks the same ranks under either
delay; the eviction inside the run moves a pick; prefix_hash sends each prefix
to one rank of each side.

Skips where cargo cannot resolve atomesh's dependencies, as
``test_mesh_clock_client.py`` does.
"""

import heapq
import json
import math
import os
import random
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from capabilities import capability
from test_pd_slice import PD
from test_vertical_slice import CONSTANT, TREE, _retry, _run_file

from atom.compass import run as compass_run
from atom.compass.carriers import (
    relay_stamp,
    stamp_events,
    tracestate_stamp,
    with_relay_stamp,
)
from atom.compass.clock import LpId
from atom.compass.clock_transport import connect
from atom.compass.detect.scenario import check_run
from atom.compass.parity.traffic import Traffic
from atom.utils.clock import LPRuntime, channel_of
from tests.compass.held_ports import held_ports

MESH = TREE / "atom" / "mesh"
WALL_S = 120.0
U = 2.0**-6
PREFILL_S, DECODE_S = 4 * U, U
# Each request's send time on the traffic clock, its max_tokens, and what its
# prompt starts with before `request <i>`.
WORKLOAD = [(0.0, 6, ""), (5.5 * U, 8, ""), (6 * U, 1, ""), (11.5 * U, 2, "")]
# Least load, ties to rank 0. Prefill load is held from placement until the
# relay is handled, 4.12U after the send; decode load until the stream's last
# event is, 0.02U plus U per token later.
#   0: nothing held                                         -> (0, 0)
#   1: p0 freed at 4.12U; d0 held by 0, streaming since 5.14U -> (0, 1)
#   2: p0 held by 1; d0 by 0, d1 by 1                       -> (1, 0)
#   3: p0 freed at 9.62U, p1 at 10.12U; d0 freed at 10.14U
#      and 11.14U, d1 held by 1                             -> (0, 0)
# A missed prefill drop sends 1 to p1, a missed decode drop 3 to d1, and a
# decode drop at the stream's first event 1 to d0.
RANKS = [(0, 0), (0, 1), (1, 0), (0, 0)]

# Two prefixes of 16 words each, which a prompt starts with before
# `request <i>`. cache_aware's eviction, at 1 s when on, falls between
# requests 2 and 3.
A, B = (" ".join(f"{f}{k}" for k in range(16)) + " " for f in "ab")
SHARED = [(0.0, 4, A), (U, 4, B), (1 - 2 * U, 4, A), (1 + U, 4, A), (1 + 16 * U, 4, B)]
# cache_aware, on each side: 0 matches nothing with both ranks idle, so it
# draws rank a; 1 matches nothing while a holds 0, so the other rank; 2 matches
# 0, so a. 3 matches 0 and 2, so a, unless the eviction emptied the tree: then
# the rank 2 does not hold. The arguments are `pd_sim.sh`'s, but the tree's
# size, in characters, below one prompt's.
CACHE_AWARE = ["--prefill-policy", "cache_aware", "--decode-policy", "cache_aware"]
CACHE_AWARE += ["--cache-threshold", "0.8", "--balance-abs-threshold", "20"]
CACHE_AWARE += ["--balance-rel-threshold", "2.0", "--max-tree-size", "16"]


@pytest.fixture(scope="module")
def atomesh(tmp_path_factory):
    resolved = shutil.which("cargo") and subprocess.run(
        ["cargo", "metadata", "--format-version", "1"],
        cwd=MESH,
        capture_output=True,
        check=False,
        timeout=600,
    )
    if not resolved or resolved.returncode:
        pytest.skip("cargo cannot resolve atomesh's dependencies here")
    subprocess.run(
        ["cargo", "build", "--bin", "atomesh"], cwd=MESH, check=True, timeout=1800
    )
    # Any later cargo call in atom/mesh, such as another gate shard's `cargo
    # test`, re-links target/debug/atomesh to the binary it built, with other
    # dependency features, and the path is missing while it does. The tests
    # start a copy in their own directory instead, which no cargo call touches.
    built = Path(os.environ.get("CARGO_TARGET_DIR", MESH / "target")) / "debug/atomesh"
    return Path(shutil.copy(built, tmp_path_factory.mktemp("atomesh")))


def test_a_later_cargo_call_leaves_the_binary_the_tests_start_alone(
    tmp_path, tmp_path_factory, monkeypatch
):
    # A stand-in cargo whose build and test each re-link target/debug/atomesh,
    # as cargo does, to a new binary that prints the command that linked it.
    (tmp_path / "cargo").write_text(
        '#!/bin/sh\n[ "$1" = metadata ] && exit 0\nd=$CARGO_TARGET_DIR/debug\n'
        'mkdir -p "$d" && rm -f "$d/atomesh" &&\n'
        'printf "#!/bin/sh\\necho %s\\n" "$1" >"$d/atomesh" && chmod +x "$d/atomesh"\n'
    )
    (tmp_path / "cargo").chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("CARGO_TARGET_DIR", str(tmp_path / "target"))
    binary = atomesh.__wrapped__(tmp_path_factory)
    subprocess.run(["cargo", "test"], check=True)
    started = subprocess.run([binary], capture_output=True, text=True, check=True)
    assert started.stdout == "build\n"


class Frontend:
    """Frontend LP ``frontend-<side>``: an HTTP server the router posts to, and
    the LP's clock owner, which replies to each request once it is released."""

    def __init__(self, run: dict, side: str, delay, model: str, port: int) -> None:
        self.run, self.side, self.delay, self.model = run, side, delay, model
        self.lp = LpId(f"frontend-{side}")
        self.cv = threading.Condition()
        self.bodies: dict[int, dict] = {}  # inbound seq -> its request body
        self.replies: dict[int, list] = {}  # inbound seq -> its reply's parts
        self.ranks: dict[int, int] = {}  # request -> the DP rank it was sent to
        self.errors: list[BaseException] = []
        self.http = ThreadingHTTPServer(("127.0.0.1", port), self._handler())
        self.url = f"http://127.0.0.1:{self.http.server_port}"
        threading.Thread(target=self.http.serve_forever, daemon=True).start()
        self.owner = threading.Thread(target=self._own, daemon=True)
        self.owner.start()

    def _handler(self):
        fe = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:
                pass

            def do_GET(self) -> None:  # the router's worker discovery
                role = "kv_producer" if fe.side == "P" else "kv_consumer"
                info = {"dp_size": 2, "tp_size": 1, "kv_role": role}
                info |= {"model_id": fe.model, "data": [{"id": fe.model}]}
                self._send(json.dumps(info).encode(), "application/json")

            def do_POST(self) -> None:
                time.sleep(fe.delay())
                raw = self.rfile.read(int(self.headers["content-length"]))
                body = json.loads(raw)
                stamp = (
                    tracestate_stamp(self.headers["tracestate"])
                    if fe.side == "P"
                    else relay_stamp(raw)
                )
                seq = stamp[1]
                with fe.cv:
                    request = int(re.search(r"request (\d+)$", body["prompt"])[1])
                    fe.ranks[request] = body["data_parallel_rank"]
                    fe.bodies[seq] = body
                    fe.cv.notify_all()
                if fe.side == "P":
                    self._send(fe.part(seq, 0).encode(), "application/json")
                    return
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.end_headers()
                for k in range(body["max_tokens"]):
                    self.wfile.write(fe.part(seq, k).encode())
                    self.wfile.flush()

            def _send(self, data: bytes, kind: str) -> None:
                self.send_response(200)
                self.send_header("content-type", kind)
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        return Handler

    def part(self, seq: int, k: int) -> str:
        """Part `k` of the reply to `seq`, once the owner has made it, after
        the injected delay."""
        with self.cv:
            self.cv.wait_for(lambda: len(self.replies.get(seq, ())) > k, WALL_S)
            text = self.replies[seq][k]
        time.sleep(self.delay())
        return text

    def _own(self) -> None:
        try:
            run = self.run
            conn = connect(self.lp, run["clock_endpoint"])
            rt = LPRuntime(self.lp, compass_run.channel_table(run), conn)
            inbound = channel_of(rt, "http" if self.side == "P" else "relay")
            out = channel_of(rt, "relay" if self.side == "P" else "stream")
            rt.start_run()
            due = []  # (time, inbound seq, part) of each send not made yet
            while (G := rt.next_event(due[0][0] if due else math.inf)) != math.inf:
                for seq in sorted(rt.released[inbound] - rt.handled[inbound]):
                    with self.cv:
                        assert self.cv.wait_for(lambda s=seq: s in self.bodies, WALL_S)
                        self.replies[seq] = []
                    with rt.lock:
                        rt.count_done_locked(inbound, seq)
                    n = 1 if self.side == "P" else self.bodies[seq]["max_tokens"]
                    for k in range(n):
                        cost = PREFILL_S if self.side == "P" else (k + 1) * DECODE_S
                        heapq.heappush(due, (G + cost, seq, k))
                while due and due[0][0] <= rt.now:
                    _, seq, k = heapq.heappop(due)
                    text = self._reply(rt, out, self.bodies[seq], k)
                    with self.cv:
                        self.replies[seq].append(text)
                        self.cv.notify_all()
            rt.close()
            compass_run._hand_in(run, self.lp.name, [])
        except BaseException as e:  # noqa: BLE001 - the test raises it
            self.errors.append(e)

    def _reply(self, rt, out: str, body: dict, k: int) -> str:
        if self.side == "P":
            params = {"kv_transfer_params": {"dp_rank": body["data_parallel_rank"]}}
            raw = json.dumps(params).encode()
            return with_relay_stamp(raw, lambda: rt.stamp_send(out)).decode()
        n = body["max_tokens"]
        events = [{"choices": [{"index": 0, "text": "x"}]}]
        if k == n - 1:
            events.append({"choices": [], "usage": {"completion_tokens": n}})
        text = "".join(f"data: {json.dumps(e)}\n\n" for e in events)
        return stamp_events(text + "data: [DONE]\n\n" * (k == n - 1), rt)


def _idle(run: dict, lp: str, member: str, errors: list) -> None:
    """A member of an engine LP that is never sent anything."""
    try:
        conn = connect(LpId(lp), run["clock_endpoint"], member)
        rt = LPRuntime(LpId(lp), compass_run.channel_table(run), conn)
        rt.start_run()
        while rt.next_event(math.inf) != math.inf:
            pass
        compass_run._hand_in(run, f"{lp}.{member}", [])
    except BaseException as e:  # noqa: BLE001 - the test raises it
        errors.append(e)


class DelayedTraffic(Traffic):
    """`Traffic`, sleeping `delay()` wall seconds before each send."""

    def __init__(self, *args, delay) -> None:
        super().__init__(*args)
        self.delay = delay

    def _read(self, body: dict, tracestate: str) -> None:
        time.sleep(self.delay())
        super()._read(body, tracestate)


def _launch(out: Path, name: str, args: list, **env) -> subprocess.Popen:
    with open(out / f"{name}.log", "w") as log:
        return subprocess.Popen(
            args, env=dict(os.environ, **env), stdout=log, stderr=subprocess.STDOUT
        )


def run_router_lp(
    atomesh: Path,
    out: Path,
    hash_seed: str,
    delay,
    policy: str = "dp_sticky",
    args: list = (),
    workload: list = WORKLOAD,
    model: str = "m",
    ports: tuple = (0, 0),
) -> list:
    """One run in `out`; each request's (prefill, decode) rank, in order.
    `args` are more router arguments; the frontends serve `model` on `ports`
    (prefill, decode), 0 for any."""
    out.mkdir()
    with held_ports(3) as (clock_port, router_port, metrics_port):
        run_file = _run_file(
            out,
            clock_endpoint=f"tcp://127.0.0.1:{clock_port}",
            data_parallel_size=2,
            step_stub=CONSTANT,
            router_in_s=2.0**-14,
            **PD,
        )
        run = json.loads(run_file.read_text())
        table = compass_run.channel_table(run)
        lookaheads = {
            c.name: table.lookahead(c.name)
            for c in table.channels_from(compass_run.ROUTER)
        }
        endpoint = run["clock_endpoint"]
        errors, procs, frontends = [], [], {}
        try:
            authority = _launch(
                out,
                "authority",
                [sys.executable, "-m", "atom.compass.run", "--compass-run"]
                + [str(run_file), "--compass-clock-endpoint", endpoint],
                PYTHONHASHSEED=hash_seed,
                PYTHONPATH=str(TREE),
            )
            procs.append(authority)
            host, port = endpoint.removeprefix("tcp://").rsplit(":", 1)
            _retry(
                "clock",
                authority,
                lambda: socket.create_connection((host, port)).close(),
            )
            frontends = {
                s: Frontend(run, s, delay, model, port) for s, port in zip("PD", ports)
            }
            idle = [
                threading.Thread(
                    target=_idle,
                    args=(run, f"engine-{s}", f"dp{r}", errors),
                    daemon=True,
                )
                for s in "PD"
                for r in range(2)
            ]
            for t in idle:
                t.start()
            router = _launch(
                out,
                "router",
                [str(atomesh), "launch", "--host", "127.0.0.1", "--port"]
                + [str(router_port), "--pd-disaggregation"]
                + ["--prefill", frontends["P"].url]
                + ["--decode", frontends["D"].url, "--policy", policy, "--dp-aware"]
                + ["--backend", "atom", "--disable-health-check"]
                + ["--disable-circuit-breaker", "--disable-retries"]
                + ["--prometheus-port", str(metrics_port), "--log-level", "info"]
                + list(args),
                ATOM_COMPASS_CLOCK_ENDPOINT=endpoint,
                ATOM_COMPASS_SEED="20260707",
                ATOM_COMPASS_ROUTER_LOOKAHEADS=json.dumps(lookaheads),
                # Each eviction pass's tenant sizes, which `evictions` reads.
                RUST_LOG="mesh=info,mesh::policies::tree=debug",
            )
            procs.append(router)
            deadline = time.monotonic() + WALL_S
            while True:
                try:
                    urllib.request.urlopen(f"http://127.0.0.1:{router_port}/v1/models")
                    break
                except OSError:
                    assert router.poll() is None, (out / "router.log").read_text()
                    assert time.monotonic() < deadline, f"no router in {WALL_S} s"
                    time.sleep(0.5)
            bodies = [
                {
                    "model": model,
                    "prompt": f"{prefix}request {i}",
                    "max_tokens": n,
                    "stream": True,
                }
                for i, (_, n, prefix) in enumerate(workload)
            ]
            offsets = [t for t, _, _ in workload]
            DelayedTraffic(run, router_port, bodies, offsets, delay=delay).run()
            for t in [*idle, *(f.owner for f in frontends.values())]:
                t.join(WALL_S)
            assert authority.wait(WALL_S) == 0, (out / "authority.log").read_text()
        finally:
            for p in procs:
                if p.poll() is None:
                    p.kill()
            for f in frontends.values():
                f.http.shutdown()
                f.http.server_close()
    errors += [e for f in frontends.values() for e in f.errors]
    if errors:
        raise errors[0]
    p, d = (frontends[s].ranks for s in "PD")
    return [(p[i], d[i]) for i in range(len(workload))]


@pytest.fixture(scope="module")
def runs(atomesh, tmp_path_factory):
    root = tmp_path_factory.mktemp("router_lp")
    wall = random.Random(1)
    delays = {"1": lambda: 0.0, "2": lambda: wall.uniform(0.0, 0.02)}
    return {
        root / f"seed{s}": run_router_lp(atomesh, root / f"seed{s}", s, delay)
        for s, delay in delays.items()
    }


@capability("router_lp")
def test_both_runs_meet_every_scenario_check(runs):
    failed = check_run(list(runs))
    assert not failed, "\n".join(failed)


def test_each_request_takes_the_least_load_ranks_under_any_host_delay(runs):
    print("\n(prefill, decode) rank per request, per run:", list(runs.values()))
    assert list(runs.values()) == [RANKS, RANKS]


def test_random_picks_the_same_ranks_from_one_seed_under_any_host_delay(
    atomesh, tmp_path
):
    wall = random.Random(2)
    first = run_router_lp(atomesh, tmp_path / "a", "1", lambda: 0.0, "random")
    second = run_router_lp(
        atomesh, tmp_path / "b", "2", lambda: wall.uniform(0.0, 0.02), "random"
    )
    print("\nrandom: (prefill, decode) rank per request, per run:", [first, second])
    assert first == second


def evictions(log: str) -> int:
    """The router's eviction passes that shrank a tree: each logs every
    tenant's size before it and after it (`atom/mesh/src/policies/tree.rs`)."""
    totals = []
    for line in log.splitlines():
        if "eviction - Used size per tenant" in line:
            totals.append(0)
        elif size := re.search(r"Tenant: \S+, Size: (\d+)", line):
            totals[-1] += int(size[1])
    return sum(after < before for before, after in zip(totals[::2], totals[1::2]))


def _model(directory: Path) -> str:
    """A model directory whose tokenizer, word-level over the prefixes' words,
    the router loads to give prefix_hash the tokens it hashes; its path is the
    model's name, as ATOM names a model by its path."""
    words = ["<unk>", *A.split(), *B.split()]
    vocab = {w: i for i, w in enumerate(words)}
    spec = {"version": "1.0", "truncation": None, "padding": None}
    spec |= {"added_tokens": [], "normalizer": None, "post_processor": None}
    spec |= {"pre_tokenizer": {"type": "WhitespaceSplit"}, "decoder": None}
    spec["model"] = {"type": "WordLevel", "vocab": vocab, "unk_token": "<unk>"}
    directory.mkdir()
    (directory / "tokenizer.json").write_text(json.dumps(spec))
    return str(directory)


def _pair(atomesh: Path, root: Path, model: str, policy: str, args: list) -> dict:
    """`policy`'s runs over `SHARED` from both seeds, the second under a random
    host delay, the frontends on the same ports in both: {out dir: ranks}."""
    root.mkdir()
    wall = random.Random(3)
    delays = {"1": lambda: 0.0, "2": lambda: wall.uniform(0.0, 0.02)}
    runs = {}
    with held_ports(2) as ports:
        for s, delay in delays.items():
            out = root / f"seed{s}"
            runs[out] = run_router_lp(
                atomesh, out, s, delay, policy, args, SHARED, model, ports
            )
    return runs


@pytest.fixture(scope="module")
def shared(atomesh, tmp_path_factory):
    root = tmp_path_factory.mktemp("router_lp_shared")
    model = _model(root / "model")
    runs = {
        "cache_aware": ("cache_aware", [*CACHE_AWARE, "--eviction-interval", "300"]),
        "cache_aware_evicting": (
            "cache_aware",
            [*CACHE_AWARE, "--eviction-interval", "1"],
        ),
        "prefix_hash": (
            "prefix_hash",
            ["--tokenizer-path", model, "--prefix-token-count", "16"],
        ),
    }
    return {
        name: _pair(atomesh, root / name, model, policy, args)
        for name, (policy, args) in runs.items()
    }


@capability("router_cache_aware")
@capability("router_prefix_hash")
def test_each_policy_meets_every_scenario_check_and_one_seed_picks_its_ranks(shared):
    failed = {name: check_run(list(pair)) for name, pair in shared.items()}
    print("\nrun: level, evictions, (prefill, decode) rank per request")
    for name, pair in shared.items():
        for out, ranks in pair.items():
            level = "failed" if failed[name] else "level 0"
            count = evictions((out / "router.log").read_text())
            print(f"{name}/{out.name}: {level}, {count} evictions, {ranks}")
    assert not any(failed.values()), "\n".join(f for fs in failed.values() for f in fs)
    for name, pair in shared.items():
        first, second = pair.values()
        assert first == second, f"{name} picks other ranks under a host delay"


def test_the_eviction_fires_inside_the_run_and_moves_a_pick(shared):
    kept, evicting = shared["cache_aware"], shared["cache_aware_evicting"]
    counts = [evictions((out / "router.log").read_text()) for out in [*kept, *evicting]]
    # One pass a side's tree at 1 s; none at 300 s.
    assert counts[:2] == [0, 0] and counts[2] == counts[3] > 0, counts
    (kept, _), (evicted, _) = kept.values(), evicting.values()
    a = kept[0]
    other = tuple(1 - r for r in a)
    assert kept == [a, other, a, a, other]
    assert evicted[:4] == [a, other, a, other], "the eviction must move request 3"


def test_prefix_hash_sends_each_prefix_to_one_rank_of_each_side(shared):
    # Its 16 words hash a prefix to one rank of each side, and no request finds
    # that rank above 1.25 times the mean load, which would send it elsewhere.
    ranks, _ = shared["prefix_hash"].values()
    taken = {p: {r for r, (_, _, q) in zip(ranks, SHARED) if q == p} for p in (A, B)}
    assert all(len(r) == 1 for r in taken.values()), taken


def test_power_of_two_is_refused_by_name(atomesh, tmp_path):
    # A clock that accepts the bind and never answers: the refusal comes first.
    clock = socket.create_server(("127.0.0.1", 0))
    with held_ports(2) as (router_port, metrics_port):
        router = _launch(
            tmp_path,
            "router",
            [str(atomesh), "launch", "--host", "127.0.0.1", "--port", str(router_port)]
            + ["--pd-disaggregation", "--prefill", "http://127.0.0.1:9"]
            + ["--decode", "http://127.0.0.1:9", "--policy", "power_of_two"]
            + ["--dp-aware", "--backend", "atom"]
            + ["--prometheus-port", str(metrics_port)],
            ATOM_COMPASS_CLOCK_ENDPOINT=f"tcp://127.0.0.1:{clock.getsockname()[1]}",
            ATOM_COMPASS_SEED="1",
            ATOM_COMPASS_ROUTER_LOOKAHEADS="{}",
        )
        try:
            code = router.wait(30)
        except subprocess.TimeoutExpired:
            router.kill()
            code = "still running after 30 s"
        finally:
            clock.close()
    log = (tmp_path / "router.log").read_text()
    assert code == 1, log
    assert "policy power_of_two is refused" in log
