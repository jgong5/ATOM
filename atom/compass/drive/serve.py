# SPDX-License-Identifier: MIT
"""Start one ATOM instance's API server, real or simulated, and serve until the
command is stopped or the server exits.

A server handed a run file is simulated, and co-hosts its clock authority
unless it joins one its run file names. One without a run file is real, on the
GPUs the command names, which `gpu.quiet` checks before the server starts and
after it stops; given an artifact store, its start merges the store's JIT cache
library for the stack into the container's caches and its stop writes back
what it finished compiling (`jit_cache`), and at DP and TP both above 1 it
runs under `LOG_MORE`. ATOM's arguments pass verbatim; the
driver assigns the host, 127.0.0.1 unless the command names another, its ports
and the run file, and
refuses them, and the clock endpoint flag, among ATOM's arguments; an op graph
run's file also holds ATOM's arguments, from which its engines trace a shape
the store lacks (`run.ATOM_ARGS`). A simulated server whose run file names
a machine starts under `on_arch`, which imports aiter as on that machine's
device first, so the server resolves its config, an FP8 quantization's dtype
among it, as there, whatever device this host has. Each start
takes a free `ATOM_DP_MASTER_PORT`: the port keys ATOM's compiled-graph cache,
and the first start that loads that cache autotunes every graph's kernels
again, whose configs stay loaded and read in `non_torch`. No later start loads
the directory, so the server's stop removes it (`drop_compiled`).

The server runs this driver's tree in its own process group, so its engine
processes stop with it, and writes ``<out>/server.log`` and its step record in
``<out>/steps``. Once its ``/health`` first answers, ``<out>/server.json``
names its URL and process id, for a simulated server its clock endpoint and
run file, and, given an artifact store, the JIT cache library's absolute path.
SIGINT or SIGTERM to the command, or the server's exit, stops the whole group:
SIGTERM first, SIGKILL for what is left after `STOP_GRACE_S`.
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
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from contextlib import ExitStack, contextmanager, nullcontext
from pathlib import Path

import atom
from atom.compass import parity
from atom.compass.backends.step_labels import TRACE
from atom.compass.drive import jit_cache
from atom.compass.drive.gpu import KFD, hip_visible, quiet
from atom.compass.drive.identity import _device, _stack, identity
from atom.compass.run import ATOM_ARGS, CLOCK_ENV, ENV, OP_GRAPH, _machine, _write_whole
from atom.utils.arg_parser import FlexibleArgumentParser

SERVER = (sys.executable, "-m", "atom.entrypoints.openai.api_server")
#: What a simulated server runs `SERVER`'s program under, before the arch.
ON_ARCH = ("-m", "atom.compass.drive.on_arch")
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
#: ATOM's profiler, which the driver points at `TRACE` for a trace run alone.
PROFILER = "--torch-profiler-dir"
#: A trace run's profiler: kernels and ranges, no Python stacks, shapes or
#: memory; and the seconds its stop, which writes every rank's trace, may take.
TRACE_STOP_S = 3600
TRACE_ENV = {
    "ATOM_PROFILER_WITH_STACK": "0",
    "ATOM_PROFILER_RECORD_SHAPES": "0",
    "ATOM_PROFILER_PROFILE_MEMORY": "0",
    "ATOM_PROFILER_TIMEOUT": str(TRACE_STOP_S),
}
#: The kineto config a trace run's server reads through ``KINETO_CONFIG``,
#: written in `out`. kineto's ROCm tracer keeps at most ROCTRACER_MAX_EVENTS
#: rows a process, launch calls and device rows alike, and drops every later
#: row unmarked; its default of 5,000,000 filled 149 s into a 300 s V4.1-Flash
#: TP4 trace run (#3106). The largest value the key takes, an int32: a trace
#: run's rows are held in host memory until its stop. A 120 s V4.1-Flash TP4
#: trace run then holds about 14.5M rows a rank, and its command leaves about
#: 11 min after SIGTERM, once it has written and reduced them; `client.sh`
#: --stop allows a trace run `TRACE_STOP_S` more for that.
KINETO_FILE, KINETO_CONF = "kineto.conf", "ROCTRACER_MAX_EVENTS=2147483647\n"
#: As `cctraces_sim.sh` sets it, so a pooling client's idle connection is not
#: closed under it; before ATOM's arguments, so one there wins.
KEEP_ALIVE = ("--timeout-keep-alive", "1000000")
READY_FILE, LOG_FILE, RUN_JSON, STEPS = "server.json", "server.log", "run.json", "steps"
RUN_FILE = "run_file.json"
POLL_S = 0.5
#: Seconds the group has to leave after SIGTERM. ATOM's server waits
#: ATOM_SHUTDOWN_TIMEOUT_S for its engines, and they as long for their workers.
STOP_GRACE_S = 60
#: ATOM_SHUTDOWN_TIMEOUT_S for a server this driver starts: a worker writes its
#: step record at its exit, and one blocked in a Mooncake sync write at SIGTERM
#: exits only once the write returns, within Mooncake's default 30 s timeout
#: (`mooncake_connector._mc_transfer_timeout_s`). Less than `STOP_GRACE_S`, so
#: ATOM's own kill of a worker still comes before the group's SIGKILL.
SHUTDOWN_TIMEOUT_S = 45
TREE = Path(atom.__file__).resolve().parents[1]
PORT_RANGE = Path("/proc/sys/net/ipv4/ip_local_port_range")
#: Where a server's ports are drawn, and where `deploy.sh` draws a deployment's:
#: apart, and each clear of the 15000 to 17000 Mooncake's transfer engine draws
#: each rank's RPC port from (rpc_min_port, rpc_max_port).
SERVE_PORTS, DEPLOY_PORTS = range(10000, 15000), range(17001, 65536)
#: The line each rank logs naming its graph's directory in the compiled-graph cache.
CACHE_LINE = re.compile(r"Using cache directory: (\S+) for vLLM's torch\.compile")
#: ATOM's `atom.utils.backends.VLLM_CACHE_ROOT`, stated here because importing
#: that module opens a HIP context on every GPU the calling process can see.
CACHE_ROOT = os.path.expanduser("~/.cache/atom")
#: ATOM's switch that heads each log line with its process, `<name>:<pid>`, so
#: `memory.check` can pair a DP x TP log's Memory budget lines with DP ranks.
LOG_MORE = {"ATOM_LOG_MORE": "1"}


def below_ephemeral(ports: range) -> range:
    """`ports` below the kernel's ephemeral range: a bind to port 0, as ATOM's
    own ranks make while they load, or an outgoing connection never takes one."""
    return range(ports.start, min(ports.stop, int(PORT_RANGE.read_text().split()[0])))


class ServeError(ValueError):
    """ATOM's arguments carry a flag the driver owns, or the server failed; a
    `ValueError`, so the console script exits with its message."""


def serve(
    atom_args,
    out,
    gpus=(),
    run=None,
    extra_env=None,
    host=HOST,
    joined=False,
    store=None,
    trace=False,
) -> None:
    """Serve one instance from ATOM's arguments `atom_args`, writing to the
    directory `out`, until SIGINT or SIGTERM or the server's exit.

    A real server names its GPUs in `gpus`; a simulated one is handed `run`,
    the run file's mapping, whose ``out_dir`` and ``clock_endpoint`` are set
    here, or, `joined`, kept: the server joins the authority its
    ``clock_endpoint`` names, which reads the hand-ins in its ``out_dir``. The
    server binds `host`, which its URL names. `extra_env` maps variables the
    server's environment holds over the command's. A real server under `trace`
    runs ATOM's profiler on every rank into `out`'s `TRACE` directory, started
    before `READY_FILE` is written and stopped before the server is. Raises `ServeError` when
    the server exits before ``/health`` answers, or on its own with a non-zero
    status, and `GpuRefusal` from the GPU checks. Given `store`, an artifact
    store's root, a real server's start merges the store's JIT cache library
    for the stack on `gpus` into the container's caches, and its stop writes
    back what it finished compiling since the merge. Once the server has
    stopped, writes run.json from the engine file, and raises `ServeError` when
    no step wrote one.
    """
    _refuse(atom_args)
    model = _model(atom_args)
    if (run is None) == (not gpus):
        raise ServeError(
            "a server is real, on the GPUs it names, or simulated, from a run "
            f"file, exactly one; got gpus={gpus} and a run file: {run is not None}"
        )
    if trace and run is not None:
        raise ServeError(
            "a trace run profiles a real server, and this one is simulated"
        )
    library = jit_cache.library(store, _stack(), _device(gpus)) if store else None
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ServeError(f"{out} holds an earlier run's files; one run per directory")
    out.mkdir(parents=True, exist_ok=True)
    held = ExitStack()
    server_port, engine_port, clock_port, dp_port = held.enter_context(_held_ports(4))
    argv = [*SERVER, *KEEP_ALIVE, *atom_args, "--host", host]
    argv += ["--server-port", str(server_port), "--port", str(engine_port)]
    ready = {"url": f"http://{host}:{server_port}"}
    if library:
        ready["library"] = str(library.absolute())
    # Nor can the environment hand the server a run file or a clock.
    env = {**os.environ, **(extra_env or {})}
    env = {k: v for k, v in env.items() if k not in (ENV, CLOCK_ENV)}
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(TREE), env.get("PYTHONPATH")])
    )
    env[parity.ENV] = str(out / STEPS)
    env["ATOM_DP_MASTER_PORT"] = str(dp_port)
    env["ATOM_SHUTDOWN_TIMEOUT_S"] = str(SHUTDOWN_TIMEOUT_S)
    if trace:
        argv += [PROFILER, str(out / TRACE)]
        (out / KINETO_FILE).write_text(KINETO_CONF)
        env.update(TRACE_ENV, KINETO_CONFIG=str((out / KINETO_FILE).absolute()))
    if run is None:
        env["HIP_VISIBLE_DEVICES"] = hip_visible(gpus)
        env |= log_more(atom_args)
    else:
        run_file = out / RUN_FILE
        if run.get(OP_GRAPH):
            run = {**run, ATOM_ARGS: list(atom_args)}
        if joined:
            argv += ["--compass-clock-endpoint", run["clock_endpoint"]]
        else:
            endpoint = f"tcp://{host}:{clock_port}"
            run = {**run, "out_dir": str(out), "clock_endpoint": endpoint}
        run_file.write_text(json.dumps(run))
        argv += ["--compass-run", str(run_file)]
        if "machine" in run:
            argv[1:1] = [*ON_ARCH, _machine(run).value("device.arch")]
        ready |= {"clock_endpoint": run["clock_endpoint"], "run_file": str(run_file)}
    with held, _stop_signals() as stopped, quiet(gpus, out) if gpus else nullcontext():
        kfd = _kfd() if gpus else None
        if library:
            jit_cache.restore(library)
        with (out / LOG_FILE).open("w") as log:
            # The log's creation, on the clock a cache file's mtime reads.
            since = os.fstat(log.fileno()).st_mtime_ns
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
                if trace:
                    _post(host, server_port, "/start_profile", STOP_GRACE_S)
                ready["pid"] = server.pid
                _write_whole(out / READY_FILE, json.dumps(ready))
            while not stopped and server.poll() is None:
                time.sleep(POLL_S)
            if trace and "pid" in ready and server.poll() is None:
                _post(host, server_port, "/stop_profile", TRACE_STOP_S)
        finally:
            _stop(server.pid, kfd)
            server.wait()
            drop_compiled(out / LOG_FILE)
            if library:
                jit_cache.save(library, since)
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


def drop_compiled(log) -> None:
    """Remove each directory ATOM derived in its compiled-graph cache that the
    stopped server's log `log` names. A directory configured outside the cache
    root stays."""
    root = Path(CACHE_ROOT, "torch_compile_cache")
    text = Path(log).read_text("utf-8", "replace")
    for used in map(Path, CACHE_LINE.findall(text)):
        if used.is_relative_to(root) and used != root:
            shutil.rmtree(root / used.relative_to(root).parts[0], ignore_errors=True)


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


def log_more(atom_args) -> dict:
    """`LOG_MORE` when ATOM's arguments run DP and TP both above 1, else
    nothing, so any other start logs in the format its environment gives.
    Under DP attention ATOM makes each TP rank a DP rank and runs TP 1."""
    parser = FlexibleArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=1)
    parser.add_argument("--data-parallel-size", "-dp", type=int, default=1)
    parser.add_argument("--enable-dp-attention", action="store_true")
    widths = parser.parse_known_args(atom_args)[0]
    if widths.enable_dp_attention:
        return {}
    if widths.tensor_parallel_size > 1 and widths.data_parallel_size > 1:
        return LOG_MORE
    return {}


def _refuse(atom_args) -> None:
    for arg in atom_args:
        flag = arg.split("=", 1)[0].replace("_", "-")
        if len(flag) <= 2 or not flag.startswith("--"):
            continue
        if any(f.startswith(flag) for f in OWNED):
            raise ServeError(f"{arg} is among ATOM's arguments; the driver owns it")
        if PROFILER.startswith(flag):
            raise ServeError(f"{arg} profiles the server, which only a trace run does")


@contextmanager
def _held_ports(n: int):
    """`n` free ports from `SERVE_PORTS`, bound together on every address so no
    two are the same, and held until leaving with SO_REUSEADDR set: no other
    process binds one but a listener that sets it too, as a server's do."""
    held, ports = [], below_ephemeral(SERVE_PORTS)
    try:
        for port in random.sample(ports, len(ports)):
            held.append(socket.socket())
            try:
                held[-1].bind(("", port))
                held[-1].setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            except OSError:
                held.pop().close()
            if len(held) == n:
                yield [s.getsockname()[1] for s in held]
                return
        raise ServeError(
            f"fewer than {n} free ports from {ports.start} up to {ports.stop} "
            f"({SERVE_PORTS.stop}, or the ephemeral range's low bound in {PORT_RANGE})"
        )
    finally:
        for s in held:
            s.close()


def _post(host: str, port: int, path: str, timeout: float) -> None:
    """POST `path` to the server; a reply other than 200 raises `ServeError`."""
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        conn.request("POST", path)
        reply = conn.getresponse()
        if reply.status != 200:
            raise ServeError(f"{path} answered {reply.status}: {reply.read()[:300]!r}")
    finally:
        conn.close()


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
    lists an exited process until the driver releases it; its memory can
    read as used a while longer, while the driver clears it. KFD names a
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
