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

import dataclasses
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_memory_readings import DOCUMENT

from atom.compass import run as compass_run
from atom.compass.backends.step_stub import Coefficients

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
deadline = time.monotonic() + 60
while not all(os.path.exists(f"{out}/{side}/run.json") for side in ("prefill", "decode")):
    if time.monotonic() > deadline:
        sys.exit(9)
    time.sleep(0.1)
open(out + "/summary.json", "w").write("{}")
"""
INSTANCE = """\
import http.server, json, os, signal, sys, threading
argv = sys.argv[1:]
out = argv[argv.index("--out") + 1]
json.dump(argv, open(out + ".argv.json", "w"))
if os.environ.get("PLANTED_FAIL") == os.path.basename(out):
    sys.exit(3)
os.makedirs(out)

class Health(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()

host = argv[argv.index("--host") + 1] if "--host" in argv else "127.0.0.1"
server = http.server.HTTPServer((host, 0), Health)
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
import http.server, json, os, signal, sys, urllib.request
argv = sys.argv[1:]
if argv == ["--version-verbose"]:
    commit, status = os.environ["PLANTED_BUILD"].split()
    print(f"atomesh 0.1.0\\n  Git Commit: {commit}\\n  Git Status: {status}")
    sys.exit()
lp = {k: v for k, v in os.environ.items() if k.startswith("ATOM_COMPASS_")}
json.dump({"argv": argv, "pid": os.getpid(), "env": lp}, open(os.environ["PLANTED_ROUTER"], "w"))
signal.signal(signal.SIGTERM, lambda *_: os._exit(0))
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


@pytest.fixture
def planted(tmp_path):
    """The stubbed `python3` and router, the suite's machine and a step stub;
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
    (tmp_path / "machine.json").write_text(json.dumps(DOCUMENT))
    (tmp_path / "stub.json").write_text(json.dumps(dataclasses.asdict(Coefficients())))
    env = {
        **os.environ,
        "PATH": f"{bindir}:{os.environ['PATH']}",
        "ATOMESH": str(bindir / "atomesh"),
        "PLANTED_ROUTER": str(tmp_path / "router.json"),
        "WAIT_S": "60",
    }
    flags = ["--step-stub", str(tmp_path / "stub.json")]
    flags += ["--machine", str(tmp_path / "machine.json")]
    flags += ["--store", str(tmp_path / "store")]
    return env, flags


def argv(tmp_path, flags, prefill=PREFILL, router=ROUTER_ARGS):
    model = ["--model", str(tmp_path)]
    return [
        "bash", str(SCRIPT), str(tmp_path / "d"), "run", *flags,
        "--prefill", "--", *model, *prefill,
        "--decode", "--", *model,
        "--router", "--", *router,
    ]  # fmt: skip


def _until(path: Path, script) -> None:
    for _ in range(600):
        if path.exists():
            return
        assert script.poll() is None, script.communicate()
        time.sleep(0.1)
    pytest.fail(f"no {path}")


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
    assert router["argv"][-len(ROUTER_ARGS) - 5 :] == [
        "--pd-disaggregation", "--prefill", urls["prefill"], "--decode", urls["decode"],
        *ROUTER_ARGS,
    ]  # fmt: skip
    ready = json.loads((d / "server.json").read_text())
    assert ready == {
        "url": f"http://127.0.0.1:{port}",
        "pid": router["pid"],
        "clock_endpoint": clock,
        "run_file": str(run_file),
    }
    assert _parent(router["pid"]) == script.pid

    script.send_signal(signal.SIGTERM)
    out, _ = script.communicate(timeout=60)

    assert script.returncode == 0, out
    assert all((d / side / "run.json").exists() for side in ("prefill", "decode"))
    assert (d / "summary.json").exists()
    written = json.loads((d / "run.json").read_text())
    assert written["command"][1:] == argv(tmp_path, flags)[2:]
    assert written["run_file"] == run
    assert written["router"][1:] == router["argv"]


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
    words = argv(tmp_path, flags, router=[*ROUTER_ARGS, "--dp-aware"])
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
    assert run["router_in_s"] == run["ipc_s"]
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


def test_a_dp_aware_run_refuses_a_router_built_dirty_before_anything_starts(
    tmp_path, planted
):
    env, flags = planted
    env["PLANTED_BUILD"] = f"{_tree_commit()} dirty"
    words = argv(tmp_path, flags, router=[*ROUTER_ARGS, "--dp-aware"])

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
