# SPDX-License-Identifier: MIT
"""`deploy.sh` builds a simulated prefill-decode deployment's run file, starts
its clock authority, joins one `compass run` per instance to it, starts the
router between them, writes the deployment's `server.json`, and at SIGTERM
stops them all and writes its `run.json`.

`python3` is stubbed on `PATH`: the authority listens on its endpoint and
leaves once both instances have; a `compass run` serves `/health` on a port of
its own, writes its `server.json` and, at SIGTERM, its `run.json`. The router is
a planted `ATOMESH`, built at the commit and status `PLANTED_BUILD` names.
Every other `python3` is the real one, so the run file is
`runfile.prefill_decode`'s. Nothing needs a driver.
"""

import contextlib
import dataclasses
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import platforms
import psutil
import pytest
from test_launcher_held_ports import HELD, SQUAT
from test_memory_readings import DOCUMENT

from atom.compass import run as compass_run
from atom.compass.backends.step_stub import Coefficients
from atom.kv_transfer.offload.mooncake_store.config import parse_mooncake_store_config
from atom.model_engine.arg_utils import (
    compose_kv_offload_config,
    kv_offload_connector_config,
)

TREE = Path(__file__).resolve().parents[2]
SCRIPT = TREE / "scripts/compass/deploy.sh"
AUTHORITY = """\
import json, os, socket, sys, time
argv = sys.argv[1:]
run_file = argv[argv.index("--compass-run") + 1]
out = os.path.dirname(run_file)
json.dump(argv, open(out + "/authority.argv.json", "w"))
endpoint = argv[argv.index("--compass-clock-endpoint") + 1]
s = socket.create_server(("127.0.0.1", int(endpoint.rsplit(":", 1)[1])))
run = json.load(open(run_file))
# Each instance's directory: its side's name, with its index above one.
names = [
    f"{side}{k}" if n > 1 else side
    for side, n in (("prefill", run.get("prefill_instances", 1)), ("decode", run.get("decode_instances", 1)))
    for k in range(n)
]
deadline = time.monotonic() + 60
while not all(os.path.exists(f"{out}/{name}/run.json") for name in names):
    if time.monotonic() > deadline:
        sys.exit(9)
    time.sleep(0.1)
# A real authority summarises a while after both instances stop.
time.sleep(2)
open(out + "/summary.json", "w").write("{}")
"""
INSTANCE = """\
import http.server, json, os, signal, sys, threading, time
argv = sys.argv[1:]
out = argv[argv.index("--out") + 1]
json.dump(argv, open(out + ".argv.json", "w"))
json.dump(dict(os.environ), open(out + ".env.json", "w"))
if os.environ.get("PLANTED_FAIL") == os.path.basename(out):
    sys.exit(3)
os.makedirs(out)

class Health(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()

def leave(name, path):
    # Once `path` exists, `name` leaves with an error, as one that fails on its own.
    if name == os.path.basename(out):
        while not os.path.exists(path):
            time.sleep(0.1)
        os._exit(3)

if "PLANTED_DIE" in os.environ:
    threading.Thread(target=leave, args=os.environ["PLANTED_DIE"].split(), daemon=True).start()
host = argv[argv.index("--host") + 1] if "--host" in argv else "127.0.0.1"
# One serving at another node's address serves here at PLANTED_BIND.
server = http.server.HTTPServer((os.environ.get("PLANTED_BIND", host), 0), Health)
threading.Thread(target=server.serve_forever, daemon=True).start()

def stop(*_):
    open(out + "/run.json", "w").write("{}")
    os._exit(0)

signal.signal(signal.SIGTERM, stop)
ready = {"url": f"http://{host}:{server.server_port}", "pid": os.getpid()}
open(out + "/server.json.part", "w").write(json.dumps(ready))
os.rename(out + "/server.json.part", out + "/server.json")
while True:
    signal.pause()
"""
ROUTER = """\
import http.server, json, os, signal, sys, threading, time, urllib.request
argv = sys.argv[1:]
if argv == ["--version-verbose"]:
    commit, status = os.environ["PLANTED_BUILD"].split()
    print(f"atomesh 0.1.0\\n  Git Commit: {commit}\\n  Git Status: {status}")
    sys.exit()
lp = {k: v for k, v in os.environ.items() if k.startswith("ATOM_COMPASS_")}
import resource
nofile = resource.getrlimit(resource.RLIMIT_NOFILE)
json.dump({"argv": argv, "pid": os.getpid(), "env": lp, "nofile": nofile}, open(os.environ["PLANTED_ROUTER"], "w"))
def stop(*_):
    summary = os.path.exists(os.environ["PLANTED_SUMMARY"])
    open(os.environ["PLANTED_ROUTER"] + ".stop", "w").write(json.dumps(summary))
    os._exit(0)

signal.signal(signal.SIGTERM, stop)

def leave(path):
    # Once PLANTED_LEAVE exists: its words to the log, then exit 1, as fail() does.
    while not os.path.exists(path):
        time.sleep(0.1)
    print(open(path).read(), file=sys.stderr, flush=True)
    os._exit(1)

if "PLANTED_LEAVE" in os.environ:
    threading.Thread(target=leave, args=(os.environ["PLANTED_LEAVE"],), daemon=True).start()
# As atomesh does, it asks each instance first, through any proxy the environment names.
for flag in ("--prefill", "--decode"):
    urllib.request.urlopen(argv[argv.index(flag) + 1] + "/health", timeout=10)

class Models(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()

port = int(argv[argv.index("--port") + 1])
http.server.HTTPServer(("127.0.0.1", port), Models).serve_forever()
"""
PREFILL = ["-tp", "2", "--enable-dp-attention"]
ROUTER_ARGS = ["--policy", "random", "--backend", "atom", "--disable-health-check"]
ROUTER_LP_ARGS = [*ROUTER_ARGS, "--disable-retries", "--dp-aware"]


@pytest.fixture
def planted(tmp_path):
    """The stubbed `python3` and router, the suite's platform and a step stub;
    returns the environment and the run flags."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for name, body in (("authority", AUTHORITY), ("instance", INSTANCE)):
        (tmp_path / f"{name}.py").write_text(body)
    (bindir / "python3").write_text(
        "#!/bin/sh\n"
        f'case "$1 $2" in "-m atom.compass.run") exec {sys.executable} '
        f'{tmp_path}/authority.py "$@" ;; esac\n'
        f'case "$2" in *atom.compass.cli*) shift 2; exec {sys.executable} '
        f'{tmp_path}/instance.py "$@" ;; esac\n'
        f'exec {sys.executable} "$@"\n'
    )
    (bindir / "atomesh").write_text(f"#!{sys.executable}\n" + ROUTER)
    for tool in ("python3", "atomesh"):
        (bindir / tool).chmod(0o755)
    (tmp_path / "stub.json").write_text(json.dumps(dataclasses.asdict(Coefficients())))
    # The router stub records every ATOM_COMPASS_* it inherits, so none come
    # from the caller (the GPU gate sets the slice model's).
    env = {
        **{k: v for k, v in os.environ.items() if not k.startswith("ATOM_COMPASS_")},
        "PATH": f"{bindir}:{os.environ['PATH']}",
        "ATOMESH": str(bindir / "atomesh"),
        "PLANTED_ROUTER": str(tmp_path / "router.json"),
        "PLANTED_SUMMARY": str(tmp_path / "d" / "summary.json"),
        "WAIT_S": "60",
    }
    flags = ["--step-stub", str(tmp_path / "stub.json")]
    flags += platforms.flags(tmp_path / "store", DOCUMENT, tmp_path)
    flags += ["--store", str(tmp_path / "store")]
    return env, flags


def router_hop(tmp_path) -> list[str]:
    """`--host-laws` naming a law file whose host entry states the router's hop
    alone: a router LP splits it around its hop in, which the floor cannot hold."""
    law = tmp_path / "router_hop.json"
    host = {"key": {}, "laws": {}, "hops": {"router_s": 1.5e-3}}
    law.write_text(json.dumps({"host": host}))
    return ["--host-laws", str(law)]


def argv(tmp_path, flags, prefill=PREFILL, router=ROUTER_ARGS):
    model = ["--model", str(tmp_path)]
    return [
        "bash", str(SCRIPT), str(tmp_path / "d"), "run", *flags,
        "--prefill", "--", *model, *prefill,
        "--decode", "--", *model,
        "--router", "--", *router,
    ]  # fmt: skip


#: How long `_until` waits for deploy.sh's file: a few times the longest a
#: deployment has taken to write it on a loaded host, its run-file build
#: resolving each side's config in a process of its own.
UNTIL_S = 300


def _until(path: Path, script) -> None:
    """Return once deploy.sh `script` has written `path`. Fail if it exits
    first, or after UNTIL_S, once it and every process under it are killed."""
    router = path.with_name("router.log")
    deadline = time.monotonic() + UNTIL_S
    while not path.exists():
        assert script.poll() is None, (
            script.communicate(),
            router.read_text() if router.exists() else "no router.log",
        )
        if time.monotonic() > deadline:
            top = psutil.Process(script.pid)
            for proc in [top, *top.children(recursive=True)]:
                with contextlib.suppress(psutil.NoSuchProcess):
                    proc.kill()
            script.wait()
            pytest.fail(f"no {path} after {UNTIL_S} s")
        time.sleep(0.1)


def _parent(pid: int) -> int:
    return int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[1])


def test_a_deployment_joins_both_instances_to_its_authority_behind_the_router(
    tmp_path, planted
):
    env, flags = planted
    d = tmp_path / "d"
    script = subprocess.Popen(
        argv(tmp_path, flags), env=env, cwd=tmp_path, stdout=subprocess.PIPE, text=True
    )
    _until(d / "server.json", script)

    run_file = d / "run_file.json"
    run = json.loads(run_file.read_text())
    lps = {lp.name for lp in compass_run.channel_table(run).registry.ids()}
    assert lps == {"traffic", "frontend-P", "engine-P", "frontend-D", "engine-D"}
    assert run["out_dir"] == str(d) and run["client_runs"] is None
    assert compass_run._widths(run) == (2, 1)
    authority = json.loads((d / "authority.argv.json").read_text())
    clock = run["clock_endpoint"]
    assert authority[2:] == [
        "--compass-run",
        str(run_file),
        "--compass-clock-endpoint",
        clock,
    ]
    urls = {}
    for side, role, args in (("prefill", "P", PREFILL), ("decode", "D", [])):
        seen = json.loads((d / f"{side}.argv.json").read_text())
        joined = ["run", "--join", str(run_file), "--out", str(d / side), "--"]
        assert seen[:-2] == [*joined, "--model", str(tmp_path), *args]
        assert seen[-2] == "--kv-transfer-config"
        kv = json.loads(seen[-1])
        assert compass_run._pd_side(SimpleNamespace(kv_transfer_config=kv)) == role
        assert kv["compass_kv_write_req"].startswith("tcp://127.0.0.1:")
        urls[side] = json.loads((d / side / "server.json").read_text())["url"]
    router = json.loads((tmp_path / "router.json").read_text())
    port = router["argv"][router["argv"].index("--port") + 1]
    # It starts at the hard open-file limit: a c=1228 router at 1024 ran out (#3720).
    soft, hard = router["nofile"]
    assert soft == hard
    assert router["argv"][-len(ROUTER_ARGS) - 5 :] == [
        "--pd-disaggregation", "--prefill", urls["prefill"], "--decode", urls["decode"],
        *ROUTER_ARGS,
    ]  # fmt: skip
    ready = json.loads((d / "server.json").read_text())
    assert _parent(ready.pop("pid")) == script.pid
    assert ready == {
        "url": f"http://127.0.0.1:{port}",
        "clock_endpoint": clock,
        "run_file": str(run_file),
    }

    script.send_signal(signal.SIGTERM)
    out, _ = script.communicate(timeout=60)

    assert script.returncode == 0, out
    assert all((d / side / "run.json").exists() for side in ("prefill", "decode"))
    assert (d / "summary.json").exists()
    written = json.loads((d / "run.json").read_text())
    assert written["command"][1:] == argv(tmp_path, flags)[2:]
    assert written["run_file"] == run
    assert written["router"][1:] == router["argv"]


def test_two_prefill_instances_join_by_index_behind_a_router_lp_choosing_among_them(
    tmp_path, planted
):
    env, flags = planted
    env["PLANTED_BUILD"] = f"{_tree_commit()} clean"
    d = tmp_path / "d"
    model = ["--model", str(tmp_path)]
    words = argv(tmp_path, flags + router_hop(tmp_path), router=ROUTER_LP_ARGS)
    at = words.index("--decode")
    words[at:at] = ["--prefill", "--", *model, *PREFILL]
    script = subprocess.Popen(
        words, env=env, cwd=tmp_path, stdout=subprocess.PIPE, text=True
    )
    _until(d / "server.json", script)

    run = json.loads((d / "run_file.json").read_text())
    assert run["prefill_instances"] == 2 and "decode_instances" not in run
    lps = {lp.name for lp in compass_run.channel_table(run).registry.ids()}
    assert {"frontend-P0", "engine-P0", "frontend-P1", "engine-P1"} <= lps
    assert {"frontend-D", "engine-D", compass_run.ROUTER.name} <= lps
    kvs, urls = {}, {}
    for name in ("prefill0", "prefill1", "decode"):
        seen = json.loads((d / f"{name}.argv.json").read_text())
        assert seen[seen.index("--out") + 1] == str(d / name)
        kvs[name] = json.loads(seen[-1])
        urls[name] = json.loads((d / name / "server.json").read_text())["url"]
    assert [kvs[f"prefill{k}"]["compass_instance"] for k in (0, 1)] == [0, 1]
    assert "compass_instance" not in kvs["decode"]
    # One base every instance is given, one port a rank pair and instance pair.
    assert len({kv["compass_kv_write_req"] for kv in kvs.values()}) == 1
    router = json.loads((tmp_path / "router.json").read_text())
    workers = router["argv"][router["argv"].index("--pd-disaggregation") :][:7]
    assert workers == [
        "--pd-disaggregation", "--prefill", urls["prefill0"], "--prefill", urls["prefill1"],
        "--decode", urls["decode"],
    ]  # fmt: skip
    assert json.loads(router["env"]["ATOM_COMPASS_ROUTER_WORKERS"]) == {
        urls["prefill0"]: "frontend-P0",
        urls["prefill1"]: "frontend-P1",
    }
    # The run file names no worker_order: the router LP lists each side's
    # workers by instance, then rank, whichever ports the instances drew.
    dp = run["prefill_data_parallel_size"], run["decode_data_parallel_size"]
    instances = ("prefill0", dp[0]), ("prefill1", dp[0]), ("decode", dp[1])
    assert json.loads(router["env"]["ATOM_COMPASS_ROUTER_ORDER"]) == [
        [urls[name], r] for name, width in instances for r in range(width)
    ]

    script.send_signal(signal.SIGTERM)
    out, _ = script.communicate(timeout=60)

    assert script.returncode == 0, out
    written = json.loads((d / "run.json").read_text())
    assert written["instances"] == [
        {"name": "prefill0", "side": "P", "index": 0},
        {"name": "prefill1", "side": "P", "index": 1},
        {"name": "decode", "side": "D", "index": 0},
    ]


#: A prefill side's Mooncake Store as `pd_server_atom.sh`'s launcher names it to
#: each worker: rdma, one pool a NIC, each its master and metadata server.
STORE = (
    '{"mooncake_store.local_hostname":"10.0.0.1","mooncake_store.protocol":"rdma",'
    '"mooncake_store.pools":{'
    '"ionic_0":{"master":"10.0.0.1:50051","metadata":"http://10.0.0.1:8080/metadata"},'
    '"ionic_1":{"master":"10.0.0.1:50151","metadata":"http://10.0.0.1:8180/metadata"}}}'
)
PREFILL_ENV = {
    "ATOM_KV_OFFLOAD": "mooncake_store",
    "ATOM_KV_OFFLOAD_EXTRA_CONFIG": STORE,
}
DECODE_ENV = {"ATOM_HOST_IP": "10.0.0.2"}


@pytest.mark.parametrize("cmd", ["run", "record"])
def test_each_side_s_variables_reach_its_instances_alone_here_and_through_on(
    tmp_path, planted, cmd
):
    """2P1D, the second prefill instance started through a prefix that, as
    `ssh <node>` does, runs its words in a shell of its own: each prefill
    instance runs under PREFILL_ENV's Store, byte for byte, the decode under
    DECODE_ENV's variables, and neither side's command names the other's."""
    env, flags = planted
    prefix = tmp_path / "bin" / "on"
    prefix.write_text('#!/bin/sh\nshift\necho "$*" >>"$PLANTED_ON"\nexec sh -c "$*"\n')
    prefix.chmod(0o755)
    for name in (*PREFILL_ENV, *DECODE_ENV):  # none from the caller
        env.pop(name, None)
    env |= {
        "PLANTED_ON": str(tmp_path / "on.log"),
        "PLANTED_BUILD": f"{_tree_commit()} clean",
        "PREFILL_ENV": " ".join(f"{k}={v}" for k, v in PREFILL_ENV.items()),
        "DECODE_ENV": " ".join(f"{k}={v}" for k, v in DECODE_ENV.items()),
    }
    if cmd == "run":  # an instance on another node prices the KV on the network
        flags += [*router_hop(tmp_path), *platforms.network(DOCUMENT, tmp_path)]
    else:
        flags = ["--gpu", "0,1"]
    words = argv(tmp_path, flags, router=ROUTER_LP_ARGS)
    words[3] = cmd
    at = words.index("--decode")
    model = ["--model", str(tmp_path)]
    words[at:at] = ["--prefill", "--on", f"{prefix} node-b", "--", *model, *PREFILL]
    d = tmp_path / "d"
    script = subprocess.Popen(
        words, env=env, cwd=tmp_path, stdout=subprocess.PIPE, text=True
    )
    try:
        _until(d / "server.json", script)
    finally:
        script.send_signal(signal.SIGTERM)
        script.communicate(timeout=60)

    assert (tmp_path / "on.log").read_text().splitlines()[0] == f"bash {d}/prefill1.sh"
    for name in ("prefill0", "prefill1", "decode"):
        own, other = PREFILL_ENV, DECODE_ENV
        if name == "decode":
            own, other = other, own
        seen = json.loads((d / f"{name}.env.json").read_text())
        assert {k: seen.get(k) for k in own} == own, name
        assert not other.keys() & seen.keys(), name
        assert not any(k in (d / f"{name}.sh").read_text() for k in other), name
        if own is PREFILL_ENV:
            # ATOM folds it in beside the P/D connector, a Store it can address.
            kv = json.loads(json.loads((d / f"{name}.argv.json").read_text())[-1])
            offload = kv_offload_connector_config(
                seen["ATOM_KV_OFFLOAD"], seen["ATOM_KV_OFFLOAD_EXTRA_CONFIG"]
            )
            folded = json.loads(compose_kv_offload_config(json.dumps(kv), offload))
            assert folded["connectors"] == [kv, offload]
            pools = parse_mooncake_store_config(offload).pools
            assert sorted(pools) == ["ionic_0", "ionic_1"]


@pytest.mark.parametrize("node", ["10.0.0.2", None])
def test_a_record_pairs_the_clock_of_each_other_node_it_starts_an_instance_on(
    tmp_path, planted, node
):
    """2P1D recorded, the second prefill instance started through a prefix
    that, as `ssh <node>` does, runs its words in a shell of its own, and
    serving at `node`'s address: the deployment pairs that node's clock with
    this one's over the prefix as it starts, every CLOCK_PAIR_S and once it
    has stopped, each pair a line of OUT/clock_pairs.jsonl, which the router,
    asking prefill0 alone, never reaches. Serving at its node's loopback,
    None, it has no node the split tells apart from this one, so no pair."""
    env, _ = planted
    prefix = tmp_path / "bin" / "on"
    prefix.write_text('#!/bin/sh\nshift\necho "$*" >>"$PLANTED_ON"\nexec sh -c "$*"\n')
    prefix.chmod(0o755)
    env |= {"PLANTED_ON": str(tmp_path / "on.log"), "PLANTED_BIND": "127.0.0.1"}
    env |= {"CLOCK_PAIR_S": "1"}
    words = argv(tmp_path, ["--gpu", "0,1"])
    words[3] = "record"
    at = words.index("--decode")
    own = ["--on", f"{prefix} node-b", *(["--host", node] if node else [])]
    words[at:at] = ["--prefill", *own, "--", "--model", str(tmp_path), *PREFILL]
    d = tmp_path / "d"
    script = subprocess.Popen(
        words,
        env=env,
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _until(d / "server.json", script)
        ready = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
        time.sleep(3)
    finally:
        stopping = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
        script.send_signal(signal.SIGTERM)
        _, err = script.communicate(timeout=60)

    # No pair failed; and the run was written once all had stopped.
    assert script.returncode == 0, err
    assert "clock_pair" not in err, err
    assert (d / "run.json").exists()
    ran = (tmp_path / "on.log").read_text().splitlines()
    assert ran[0] == f"bash {d}/prefill1.sh"
    serve = f"python3 -I -u {TREE}/atom/compass/drive/clock_pair.py serve"
    paired = d / "clock_pairs.jsonl"
    if node is None:
        assert not paired.exists() and serve not in ran
        return
    rows = [json.loads(line) for line in paired.read_text().splitlines()]
    # As it started, once a second after while it served, and once stopped.
    # A pair the stop cut short ran through the prefix and wrote no line.
    assert len(rows) >= 3 and ran.count(serve) - len(rows) in (0, 1)
    assert {r["node"] for r in rows} == {node}
    times = [r["mono_ns"] for r in rows]
    assert times == sorted(times) and times[0] < ready + 2e9
    assert times[-1] > stopping
    # That node is this one, so each offset is none within the error it states.
    assert all(abs(r["offset_ns"]) <= r["error_ns"] for r in rows)


#: A prefix as a relay that carries no stdin, c2_r2's relay_client.py: it runs
#: its words in a shell of its own, as `ssh <node>` does, its stdin closed,
#: and lives as long as they do.
RELAY = (
    '#!/bin/sh\nshift\necho "$*" >>"$PLANTED_ON"\nsh -c "$*" </dev/null &\nwait $!\n'
)


@pytest.mark.parametrize("override", [True, False])
def test_a_node_whose_prefix_carries_no_stdin_is_paired_through_its_override(
    tmp_path, planted, override
):
    """prefill1 on 10.0.0.2, started through a relay that carries no stdin:
    with CLOCK_PAIR_PREFIX_10_0_0_2 naming a prefix that carries it, as `ssh`
    does, that node is paired through that prefix alone; without, each pair
    fails, so none is written, and nothing else does."""
    env, _ = planted
    bindir = tmp_path / "bin"
    (bindir / "relay").write_text(RELAY)
    (bindir / "ssh").write_text(
        '#!/bin/sh\nwhile [ "$1" = -o ]; do shift 2; done\nshift\n'
        'echo "$*" >>"$PLANTED_SSH"\nexec sh -c "$*"\n'
    )
    for tool in ("relay", "ssh"):
        (bindir / tool).chmod(0o755)
    env |= {"PLANTED_ON": str(tmp_path / "on.log"), "PLANTED_BIND": "127.0.0.1"}
    env |= {"PLANTED_SSH": str(tmp_path / "ssh.log"), "CLOCK_PAIR_S": "1"}
    if override:
        env["CLOCK_PAIR_PREFIX_10_0_0_2"] = f"{bindir / 'ssh'} -o BatchMode=yes node-b"
    words = argv(tmp_path, ["--gpu", "0,1"])
    words[3] = "record"
    at = words.index("--decode")
    own = ["--on", f"{bindir / 'relay'} node-b", "--host", "10.0.0.2"]
    words[at:at] = ["--prefill", *own, "--", "--model", str(tmp_path), *PREFILL]
    d = tmp_path / "d"
    script = subprocess.Popen(
        words,
        env=env,
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _until(d / "server.json", script)
        time.sleep(2)
    finally:
        script.send_signal(signal.SIGTERM)
        _, err = script.communicate(timeout=60)

    assert script.returncode == 0, err
    assert (d / "run.json").exists()
    serve = f"python3 -I -u {TREE}/atom/compass/drive/clock_pair.py serve"
    relayed = (tmp_path / "on.log").read_text().splitlines()
    assert relayed[0] == f"bash {d}/prefill1.sh"
    paired = d / "clock_pairs.jsonl"
    if not override:
        assert not paired.exists() and not (tmp_path / "ssh.log").exists()
        closed = (
            f"clock_pair: 10.0.0.2: `{bindir / 'relay'} node-b` closed with no answer"
        )
        # As it started and once it stopped at the least; one the stop cut
        # short ran through the relay and said nothing.
        assert err.count(closed) >= 2, err
        assert relayed.count(serve) - err.count(closed) in (0, 1)
        return
    assert "clock_pair" not in err, err
    rows = [json.loads(line) for line in paired.read_text().splitlines()]
    assert len(rows) >= 2 and {r["node"] for r in rows} == {"10.0.0.2"}
    ssh = (tmp_path / "ssh.log").read_text().splitlines()
    assert set(ssh) == {serve} and len(ssh) - len(rows) in (0, 1)
    assert serve not in relayed
    assert all(abs(r["offset_ns"]) <= r["error_ns"] for r in rows)


#: A prefix as `ssh <node>`: it runs its words in a shell of its own and ends
#: with their status, or, signalled itself, with 255, as ssh does at SIGTERM.
SSH = (
    '#!/bin/sh\ntrap "exit 255" TERM\nshift\necho "$*" >>"$PLANTED_ON"\n'
    'sh -c "$*" &\nwait $!\n'
)


@pytest.mark.parametrize("leaves", [False, True])
def test_a_record_s_stop_reaches_an_instance_elsewhere_through_its_prefix(
    tmp_path, planted, leaves
):
    """prefill1 started through a prefix that, as `ssh <node>` does, ends
    with its command's status, or with 255 when signalled itself: the stop
    reaches prefill1 through the prefix, never the prefix, so its status is
    prefill1's, and the record exits 0. prefill1 leaving with an error before
    any stop fails it."""
    env, _ = planted
    (tmp_path / "bin" / "ssh").write_text(SSH)
    (tmp_path / "bin" / "ssh").chmod(0o755)
    env |= {"PLANTED_ON": str(tmp_path / "on.log")}
    env |= {"PLANTED_DIE": f"prefill1 {tmp_path / 'leave'}"}
    words = argv(tmp_path, ["--gpu", "0,1"])
    words[3] = "record"
    at = words.index("--decode")
    own = ["--on", f"{tmp_path / 'bin' / 'ssh'} node-b"]
    words[at:at] = ["--prefill", *own, "--", "--model", str(tmp_path), *PREFILL]
    d = tmp_path / "d"
    script = subprocess.Popen(
        words,
        env=env,
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _until(d / "server.json", script)
        time.sleep(1)
        if leaves:
            (tmp_path / "leave").touch()
            script.wait(timeout=60)
    finally:
        script.send_signal(signal.SIGTERM)
        _, err = script.communicate(timeout=60)

    assert (d / "run.json").exists()
    pid = int((d / "prefill1.pid").read_text())
    ran = (tmp_path / "on.log").read_text().splitlines()
    assert ran[0] == f"bash {d}/prefill1.sh"
    if leaves:
        assert script.returncode == 1
        assert "deploy: a process of the deployment left before a stop" in err
        return
    assert script.returncode == 0, err
    assert ran[1:] == [f"kill -TERM {pid}"]
    assert (d / "prefill1" / "run.json").exists()


def test_a_side_s_variable_that_is_no_name_value_word_exits_2(tmp_path, planted):
    env, flags = planted
    env["PREFILL_ENV"] = (
        'ATOM_KV_OFFLOAD_EXTRA_CONFIG={"mooncake_store.protocol": "tcp"}'
    )

    done = subprocess.run(
        argv(tmp_path, flags),
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )

    assert done.returncode == 2, done.stderr
    assert 'take NAME=VALUE words, not "tcp"}' in done.stderr
    assert not (tmp_path / "d" / "run_file.json").exists()


@pytest.mark.parametrize(
    "where, refusal",
    [
        ("after decode", "in that order"),
        ("other args", "prefill1's ATOM arguments differ from prefill0's"),
        (
            "no retries",
            "choose among a side's instances as LP router but not --disable-retries",
        ),
    ],
)
def test_a_second_prefill_instance_out_of_place_exits_2_before_anything_starts(
    tmp_path, planted, where, refusal
):
    env, flags = planted
    model = ["--model", str(tmp_path)]
    retries = (
        ROUTER_ARGS if where == "no retries" else [*ROUTER_ARGS, "--disable-retries"]
    )
    words = argv(tmp_path, flags, router=retries)
    args = [] if where == "other args" else PREFILL
    at = words.index("--router" if where == "after decode" else "--decode")
    words[at:at] = ["--prefill", "--", *model, *args]

    done = subprocess.run(
        words, env=env, cwd=tmp_path, capture_output=True, text=True, check=False
    )

    assert done.returncode == 2, done.stderr
    assert refusal in done.stderr
    assert not (tmp_path / "d" / "run_file.json").exists()


def test_an_instance_that_exits_before_it_answers_fails_the_deployment_naming_it(
    tmp_path, planted
):
    env, flags = planted
    env["PLANTED_FAIL"] = "decode"

    done = subprocess.run(
        argv(tmp_path, flags),
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert done.returncode == 1
    assert "deploy: the decode instance exited before it answered" in done.stderr
    assert not (tmp_path / "d" / "server.json").exists()


def _tree_commit() -> str:
    stamp = TREE / ".compass-commit"
    if stamp.exists():
        return stamp.read_text().strip()
    git = ["git", "-C", str(TREE), "rev-parse", "HEAD"]
    return subprocess.run(
        git, capture_output=True, text=True, check=True
    ).stdout.strip()


def test_a_dp_aware_router_runs_as_the_router_lp_on_the_run_s_clock(tmp_path, planted):
    env, flags = planted
    env["PLANTED_BUILD"] = f"{_tree_commit()} clean"
    d = tmp_path / "d"
    words = argv(tmp_path, flags + router_hop(tmp_path), router=ROUTER_LP_ARGS)
    script = subprocess.Popen(
        words, env=env, cwd=tmp_path, stdout=subprocess.PIPE, text=True
    )
    _until(d / "server.json", script)

    run = json.loads((d / "run_file.json").read_text())
    assert run["router"] == {
        "path": env["ATOMESH"],
        "commit": _tree_commit(),
        "status": "clean",
    }
    assert run["router_in_s"] == compass_run.hop(run, "ipc_s")
    table = compass_run.channel_table(run)
    assert compass_run.ROUTER in table.registry.ids()
    router = json.loads((tmp_path / "router.json").read_text())
    assert router["argv"][-1] == "--dp-aware"
    lookaheads = json.loads(router["env"].pop("ATOM_COMPASS_ROUTER_LOOKAHEADS"))
    assert lookaheads == {
        c.name: table.lookahead(c.name) for c in table.channels_from(compass_run.ROUTER)
    }
    assert router["env"] == {
        "ATOM_COMPASS_CLOCK_ENDPOINT": run["clock_endpoint"],
        "ATOM_COMPASS_SEED": "20260707",
    }

    script.send_signal(signal.SIGTERM)
    out, _ = script.communicate(timeout=60)
    assert script.returncode == 0, out
    # The router LP outlives the run: it is stopped once the summary is written.
    assert json.loads((tmp_path / "router.json.stop").read_text()) is True


@pytest.mark.parametrize(
    "last_words, ended",
    [
        ("ClockAbort: traffic closed its connection, so the run ends here", True),
        ("no outbound channel router->frontend-P:http", False),
    ],
)
def test_a_router_lp_that_leaves_as_the_traffic_closes_waits_for_the_client_s_stop(
    tmp_path, planted, last_words, ended
):
    env, flags = planted
    env["PLANTED_BUILD"] = f"{_tree_commit()} clean"
    env["PLANTED_LEAVE"] = str(tmp_path / "leave")
    d = tmp_path / "d"
    words = argv(tmp_path, flags + router_hop(tmp_path), router=ROUTER_LP_ARGS)
    script = subprocess.Popen(
        words, env=env, cwd=tmp_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True,
    )  # fmt: skip
    try:
        _until(d / "server.json", script)
        router = json.loads((tmp_path / "router.json").read_text())["pid"]

        (tmp_path / "leave").write_text(f"router LP: {last_words}; ending the run")
        while Path(f"/proc/{router}").exists():
            time.sleep(0.1)
        if ended:
            time.sleep(1)  # the script has seen the router leave
            # client.sh --stop: SIGTERM to the parent of the process server.json names.
            ready = json.loads((d / "server.json").read_text())
            os.kill(_parent(ready["pid"]), signal.SIGTERM)
        _, err = script.communicate(timeout=60)
    finally:
        script.terminate()  # a deployment serving on stops with its test

    if not ended:
        assert script.returncode == 1, err
        assert "a process of the deployment left before a stop" in err
        return
    assert script.returncode == 0, err
    assert "left before a stop" not in err
    assert (d / "summary.json").exists() and (d / "run.json").exists()


def test_a_dp_aware_run_refuses_a_router_built_dirty_before_anything_starts(
    tmp_path, planted
):
    env, flags = planted
    env["PLANTED_BUILD"] = f"{_tree_commit()} dirty"
    words = argv(tmp_path, flags, router=ROUTER_LP_ARGS)

    done = subprocess.run(
        words, env=env, cwd=tmp_path, capture_output=True, text=True, check=False
    )

    assert done.returncode == 1, done.stderr
    assert f"deploy: router binary {env['ATOMESH']}" in done.stderr
    assert "built from an uncommitted tree" in done.stderr
    assert not (tmp_path / "d" / "run_file.json").exists()
    assert not (tmp_path / "d" / "authority.argv.json").exists()


@pytest.mark.parametrize(
    "case, refusal",
    [
        ("command", "the command after OUT is run or record"),
        ("kv", "--kv-transfer-config"),
        ("no router", "--router -- come once each"),
        ("occupied", "holds an earlier deployment's files"),
    ],
)
def test_bad_arguments_exit_2_before_anything_starts(tmp_path, planted, case, refusal):
    env, flags = planted
    words = argv(tmp_path, flags)
    if case == "command":
        words[3] = "trace"
    elif case == "kv":
        words.insert(words.index("--decode"), "--kv-transfer-config={}")
    elif case == "no router":
        del words[words.index("--router") :]
    else:
        (tmp_path / "d").mkdir()
        (tmp_path / "d" / "earlier").touch()

    done = subprocess.run(
        words, env=env, cwd=tmp_path, capture_output=True, text=True, check=False
    )

    assert done.returncode == 2, done.stderr
    assert refusal in done.stderr
    assert not (tmp_path / "d" / "authority.argv.json").exists()
    assert not (tmp_path / "d" / "run_file.json").exists()


@pytest.mark.parametrize(
    "words",
    [
        ["--prefill", "http://127.0.0.1:9"],
        ["--decode=http://127.0.0.1:9"],
        ["--max-concurrent-requests", "8"],
        ["--rate-limit-tokens-per-second=100"],
    ],
    ids=lambda words: words[0].split("=")[0],
)
def test_a_simulated_router_refuses_a_worker_more_or_a_limiter_before_anything_starts(
    tmp_path, planted, words
):
    env, flags = planted
    flag = words[0].split("=")[0]

    done = subprocess.run(
        argv(tmp_path, flags, router=[*ROUTER_ARGS, *words]),
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )

    assert done.returncode == 2, done.stderr
    assert f"the router's arguments carry {flag}," in done.stderr
    assert not (tmp_path / "d").exists()


def test_a_router_lp_without_disable_retries_exits_2_before_anything_starts(
    tmp_path, planted
):
    env, flags = planted

    done = subprocess.run(
        argv(tmp_path, flags, router=[*ROUTER_ARGS, "--dp-aware"]),
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )

    assert done.returncode == 2, done.stderr
    assert "carry --dp-aware but not --disable-retries:" in done.stderr
    assert not (tmp_path / "d").exists()


def test_a_simulated_router_bounds_a_request_by_a_year_of_wall_time(tmp_path, planted):
    # A stream lasts the wall time the run takes to reach its end: at c=256
    # past the router's default 1800 s, which cut it there.
    env, flags = planted
    script = subprocess.Popen(
        argv(tmp_path, flags), env=env, cwd=tmp_path, stdout=subprocess.PIPE, text=True
    )
    _until(tmp_path / "d" / "server.json", script)
    router = json.loads((tmp_path / "router.json").read_text())["argv"]
    script.send_signal(signal.SIGTERM)
    out, _ = script.communicate(timeout=60)

    assert script.returncode == 0, out
    # So are the prepare pool's wait and preparation, which a loaded host
    # stretches on the wall clock.
    for flag, scale in (
        ("--request-timeout-secs", 1),
        ("--prepare-queue-timeout-ms", 1000),
        ("--prepare-timeout-ms", 1000),
    ):
        assert router.count(flag) == 1, flag
        assert int(router[router.index(flag) + 1]) >= 365 * 24 * 3600 * scale, flag


@pytest.mark.parametrize(
    "words",
    [
        ["--request-timeout-secs", "60"],
        ["--request-timeout-secs=60"],
        ["--prepare-queue-timeout-ms", "5000"],
        ["--prepare-timeout-ms=15000"],
    ],
)
def test_a_simulated_router_refuses_the_caller_s_wall_clock_bounds(
    tmp_path, planted, words
):
    env, flags = planted

    done = subprocess.run(
        argv(tmp_path, flags, router=[*ROUTER_ARGS, *words]),
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )

    assert done.returncode == 2, done.stderr
    flag = words[0].split("=")[0]
    assert f"the router's arguments carry {flag}," in done.stderr
    assert not (tmp_path / "d").exists()


#: Another process, then a server, on the router's ports and the prefill's
#: block of write requests, two for DP 2 by DP 1.
SQUAT_ROUTER = """
import json, os, sys
a, d = sys.argv[1:], os.path.dirname(os.environ["PLANTED_SUMMARY"])
kv = json.loads(json.load(open(d + "/prefill.argv.json"))[-1])
base = int(kv["compass_kv_write_req"].rsplit(":", 1)[1])
ports = [a[a.index(f) + 1] for f in ("--port", "--prometheus-port")]
sys.argv[1:] = [*ports, str(base), str(base + 1)]
""" + SQUAT


def test_each_port_a_run_hands_a_server_is_held_from_another_process_first(
    tmp_path, planted
):
    env, flags = planted
    seen, bindir = tmp_path / "seen", Path(env["ATOMESH"]).parent
    env |= {"SQUAT": SQUAT, "SQUAT_ROUTER": SQUAT_ROUTER}
    python3 = (bindir / "python3").read_text().split("\n", 1)
    (bindir / "python3").write_text(
        f"{python3[0]}\n"
        'case "$1 $2" in "-m atom.compass.run")\n'
        f'    {sys.executable} -c "$SQUAT" "${{6##*:}}" >>{seen};;\n'
        f"esac\n{python3[1]}"
    )
    (bindir / "atomesh").rename(bindir / "router")
    (bindir / "atomesh").write_text(
        f'#!/bin/sh\n{sys.executable} -c "$SQUAT_ROUTER" "$@" >>{seen}\n'
        f'exec {bindir}/router "$@"\n'
    )
    (bindir / "atomesh").chmod(0o755)
    script = subprocess.Popen(
        argv(tmp_path, flags), env=env, cwd=tmp_path, stdout=subprocess.PIPE, text=True
    )
    try:
        _until(tmp_path / "d" / "server.json", script)
    finally:
        script.send_signal(signal.SIGTERM)
        script.communicate(timeout=60)

    squatted = [line.split()[1:] for line in seen.read_text().splitlines()]
    assert squatted == [HELD] * 5  # the clock's first
