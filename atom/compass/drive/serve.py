# SPDX-License-Identifier: MIT
"""Start one ATOM instance's API server, real or simulated, and serve until the
command is stopped or the server exits.

A server handed a run file is simulated, and co-hosts its clock authority
unless it joins one its run file names. One without a run file is real, on the
GPUs the command names, which `gpu.quiet` checks before the server starts and
after it stops. ATOM's arguments pass verbatim; the driver assigns the host,
127.0.0.1 unless the command names another, its ports and the run file, and
refuses them, and the clock endpoint flag, among ATOM's arguments. Each start
takes a free `ATOM_DP_MASTER_PORT`: the port keys ATOM's compiled-graph cache,
and the first start that loads that cache autotunes every graph's kernels
again, whose configs stay loaded and read in `non_torch`.

The server runs this driver's tree in its own process group, so its engine
processes stop with it, and writes ``<out>/server.log`` and its step record in
``<out>/steps``. Once its ``/health`` first answers, ``<out>/server.json``
names its URL and process id and, for a simulated server, its clock endpoint
and run file. SIGINT or SIGTERM to the command, or the server's exit, stops the
whole group: SIGTERM first, SIGKILL for what is left after `STOP_GRACE_S`.
A real server's stop, for the GPU check after it, lasts until KFD, which
frees a process's GPU memory after it exits, lists no process it did not list
before the server started. Then ``<out>/run.json`` gets the run's identity,
read off the engine file its first step wrote. No command starts the client.
"""

import argparse
import http.client
import json
import os
import random
import signal
import socket
import subprocess
import sys
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

import atom
from atom.compass import parity
from atom.compass.drive.gpu import KFD, hip_visible, quiet
from atom.compass.drive.identity import identity
from atom.compass.run import CLOCK_ENV, ENV, _write_whole

SERVER = (sys.executable, "-m", "atom.entrypoints.openai.api_server")
HOST = "127.0.0.1"
#: Flags ATOM's arguments may not carry, in either spelling ATOM's parser
#: accepts or any abbreviation argparse expands: the driver assigns the host,
#: its ports and the run file, and a simulated server co-hosts its clock.
OWNED = (
    "--host",
    "--server-port",
    "--port",
    "--compass-run",
    "--compass-clock-endpoint",
    "--data-parallel-master-port",
)
#: ATOM's profiler: only a profiled pass may time ops, and no command runs one yet.
PROFILER = "--torch-profiler-dir"
#: As `cctraces_sim.sh` sets it, so a pooling client's idle connection is not
#: closed under it; before ATOM's arguments, so one there wins.
KEEP_ALIVE = ("--timeout-keep-alive", "1000000")
READY_FILE, LOG_FILE, RUN_JSON, STEPS = "server.json", "server.log", "run.json", "steps"
RUN_FILE = "run_file.json"
POLL_S = 0.5
#: Seconds the group has to leave after SIGTERM. ATOM's server waits
#: ATOM_SHUTDOWN_TIMEOUT_S for its engines, and they as long for their workers.
STOP_GRACE_S = 60
TREE = Path(atom.__file__).resolve().parents[1]
PORT_RANGE = Path("/proc/sys/net/ipv4/ip_local_port_range")


class ServeError(ValueError):
    """ATOM's arguments carry a flag the driver owns, or the server failed; a
    `ValueError`, so the console script exits with its message."""


def serve(
    atom_args, out, gpus=(), run=None, extra_env=None, host=HOST, joined=False
) -> None:
    """Serve one instance from ATOM's arguments `atom_args`, writing to the
    directory `out`, until SIGINT or SIGTERM or the server's exit.

    A real server names its GPUs in `gpus`; a simulated one is handed `run`,
    the run file's mapping, whose ``out_dir`` and ``clock_endpoint`` are set
    here, or, `joined`, kept: the server joins the authority its
    ``clock_endpoint`` names, which reads the hand-ins in its ``out_dir``. The
    server binds `host`, which its URL names. `extra_env` maps variables the
    server's environment holds over the command's. Raises `ServeError` when
    the server exits before ``/health`` answers, or on its own with a non-zero
    status, and `GpuRefusal` from the GPU checks. Once the server has stopped,
    writes run.json from the engine file, and raises `ServeError` when no step
    wrote one.
    """
    _refuse(atom_args)
    model = _model(atom_args)
    if (run is None) == (not gpus):
        raise ServeError(
            "a server is real, on the GPUs it names, or simulated, from a run "
            f"file, exactly one; got gpus={gpus} and a run file: {run is not None}"
        )
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ServeError(f"{out} holds an earlier run's files; one run per directory")
    out.mkdir(parents=True, exist_ok=True)
    server_port, engine_port, clock_port, dp_port = _free_ports(4)
    argv = [*SERVER, *KEEP_ALIVE, *atom_args, "--host", host]
    argv += ["--server-port", str(server_port), "--port", str(engine_port)]
    ready = {"url": f"http://{host}:{server_port}"}
    # Nor can the environment hand the server a run file or a clock.
    env = {**os.environ, **(extra_env or {})}
    env = {k: v for k, v in env.items() if k not in (ENV, CLOCK_ENV)}
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(TREE), env.get("PYTHONPATH")])
    )
    env[parity.ENV] = str(out / STEPS)
    env["ATOM_DP_MASTER_PORT"] = str(dp_port)
    if run is None:
        env["HIP_VISIBLE_DEVICES"] = hip_visible(gpus)
    else:
        run_file = out / RUN_FILE
        if joined:
            argv += ["--compass-clock-endpoint", run["clock_endpoint"]]
        else:
            endpoint = f"tcp://{host}:{clock_port}"
            run = {**run, "out_dir": str(out), "clock_endpoint": endpoint}
        run_file.write_text(json.dumps(run))
        argv += ["--compass-run", str(run_file)]
        ready |= {"clock_endpoint": run["clock_endpoint"], "run_file": str(run_file)}
    with _stop_signals() as stopped, quiet(gpus, out) if gpus else nullcontext():
        kfd = _kfd() if gpus else None
        with (out / LOG_FILE).open("w") as log:
            server = subprocess.Popen(
                argv,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
        try:
            while not (stopped or _answers(host, server_port)):
                if server.poll() is not None:
                    raise ServeError(
                        f"the server exited {server.returncode} before /health "
                        f"answered; see {out / LOG_FILE}"
                    )
                time.sleep(POLL_S)
            if not stopped:
                ready["pid"] = server.pid
                _write_whole(out / READY_FILE, json.dumps(ready))
            while not stopped and server.poll() is None:
                time.sleep(POLL_S)
        finally:
            _stop(server.pid, kfd)
            server.wait()
    if not stopped and server.returncode:
        raise ServeError(f"the server exited {server.returncode}; see {out / LOG_FILE}")
    # Any rank's: every DP rank resolves one engine configuration.
    engines = sorted((out / STEPS).glob("dp*.engine.json"))
    if not engines:
        raise ServeError(
            "the server served no step, so no engine file states the engine it "
            f"resolved, and {out / RUN_JSON} is not written"
        )
    _write_whole(
        out / RUN_JSON, json.dumps(identity(engines[0], model, sys.argv, gpus))
    )


def _model(atom_args) -> str:
    """The model ATOM's arguments name, whose config.json the host key digests."""
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--model")
    model = parser.parse_known_args(atom_args)[0].model
    if not model:
        raise ServeError(
            "ATOM's arguments name no --model; run.json digests its config"
        )
    return model


def _refuse(atom_args) -> None:
    for arg in atom_args:
        flag = arg.split("=", 1)[0].replace("_", "-")
        if len(flag) <= 2 or not flag.startswith("--"):
            continue
        if any(f.startswith(flag) for f in OWNED):
            raise ServeError(f"{arg} is among ATOM's arguments; the driver owns it")
        if PROFILER.startswith(flag):
            raise ServeError(f"{arg} profiles the server, and only a profiled pass may")


def _free_ports(n: int) -> list[int]:
    """`n` free ports, bound together on every address so no two are the same,
    from 10000 up to the kernel's ephemeral range: a bind to port 0, as ATOM's
    own ranks make while they load, or an outgoing connection never takes one."""
    low = int(PORT_RANGE.read_text().split()[0])
    held, ports = [], range(10000, low)
    try:
        for port in random.sample(ports, len(ports)):
            held.append(socket.socket())
            try:
                held[-1].bind(("", port))
            except OSError:
                held.pop().close()
            if len(held) == n:
                return [s.getsockname()[1] for s in held]
        raise ServeError(
            f"fewer than {n} free ports from 10000 up to the ephemeral range's "
            f"low bound {low} ({PORT_RANGE})"
        )
    finally:
        for s in held:
            s.close()


def _answers(host: str, port: int) -> bool:
    """Whether the server's ``/health`` answers 200."""
    conn = http.client.HTTPConnection(host, port, timeout=5)
    try:
        conn.request("GET", "/health")
        return conn.getresponse().status == 200
    except (OSError, http.client.HTTPException):
        return False
    finally:
        conn.close()


@contextmanager
def _stop_signals():
    """A list SIGINT and SIGTERM append to instead of ending the command, so
    the server is stopped before the command goes on."""
    got = []
    old = {
        s: signal.signal(s, lambda signum, frame: got.append(signum))
        for s in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        yield got
    finally:
        for s, handler in old.items():
            signal.signal(s, handler)


def _stop(group: int, kfd=None) -> None:
    """SIGTERM the process group `group`; SIGKILL what is left after
    `STOP_GRACE_S`. Given `kfd`, the processes KFD listed before the server
    started, the group has left only once KFD lists none it did not, as it
    lists an exited process until it has freed its GPU memory. KFD names a
    process by a PID a container's /proc may not show, so no PID is mapped:
    another tenant's process that opened a GPU since holds the stop as long."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(group, sig)
        except ProcessLookupError:
            pass
        for _ in range(int(STOP_GRACE_S / POLL_S)):
            if not _live(group, kfd):
                return
            time.sleep(POLL_S)


def _kfd() -> set[str]:
    """The processes KFD lists, by name; none where there is no KFD."""
    try:
        return {p.name for p in (KFD / "proc").iterdir()}
    except FileNotFoundError:
        return set()


def _live(group: int, kfd=None) -> bool:
    """Whether a process of `group` has not exited, a zombie has, or, given
    `kfd`, KFD lists a process `kfd` does not."""
    for stat in Path("/proc").glob("[0-9]*/stat"):
        try:
            state, _, pgrp = stat.read_text().rsplit(")", 1)[1].split()[:3]
        except OSError:
            continue
        if int(pgrp) == group and state != "Z":
            return True
    return kfd is not None and bool(_kfd() - kfd)
