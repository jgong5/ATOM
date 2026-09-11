"""Run one side of one cc-traces cell, and keep the evidence that it ran.

`cc_traces_plan.py` says which commands a cell needs; this executes them, in
that order, on one side at a time -- the real side on the leased node, the
modelled side in a container with no device, which is why they cannot be one
process. Both read the same `cell_steps()`, so what was reviewed is what runs.

What this enforces beyond "the command exited zero", because each of these has
already been a way a run looked finished and meant nothing:

* **A repeat is a process.** Every repeat starts its own server and stops it
  again. On the modelled side a predicting server's virtual epoch is fixed when
  it starts, so a second replay against the same process is stamped from an
  origin the first replay already moved -- the error `replay.py` refuses a
  warmed predictor for. The harness will not reuse a server across repeats, and
  refuses to continue if one is still alive when the next repeat begins.
* **The drain is the boundary.** The real side's artifact has to carry drained
  preparation records; the modelled side's has to carry none. A replay that
  exits 3 refused the run, and a refusal is kept as a refusal rather than
  retried into a pass.
* **What answered is what we think answered.** Each repeat's
  `/compass/provenance` is read from the server that served it, and a side whose
  server reports the other side's mode fails the repeat rather than the reading.
* **Only processes this harness started are ever signalled**, by the pid it
  recorded when it started them. Nothing here matches on a name.
* **Every repeat is an execution with a name.** A digest of an artifact says
  what is in a file, not which run produced it -- two repeats of the same cell
  can differ by noise alone, and a source residual and an independent repeat
  are then indistinguishable to anything reading the files later. So each
  fresh-server repeat gets one identifier, minted where the process is
  launched, carrying the process, the source, the configuration and the
  artifacts it produced; and it is written *into* the artifact, so a copy of
  the file is still that execution. The schema is `EXECUTION_SCHEMA` below.

Costs are collected from what actually happened -- startup is launch to healthy,
execution is the replay's own window through `compare.metrics` -- and the terms
nothing here can measure (`capture`, `calibration`, `derivation`, `load`) are
inputs to the merge, which refuses to write a `costs.json` without them.

    python scripts/compass/cc_traces_run.py side --cell RESULTS/tp2_long \
        --side real --tp 2 --class long
    python scripts/compass/cc_traces_run.py side --cell RESULTS/tp2_long \
        --side modelled --tp 2 --class long --replay-target target.json \
        --oracle transfer --oracle-option source=/path/to/tp1
    python scripts/compass/cc_traces_run.py costs RESULTS/tp2_long \
        --capture 412.0 --calibration 1980.0 --derivation 31.5 --load 96.0

No result is claimed by this file. It runs commands and writes down what they
did, including when what they did was fail.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str):
    """A sibling script as a module, the way the compass tests load them."""
    path = Path(__file__).resolve().parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"compass_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _core(name: str):
    """A stdlib-only module from the runtime package, loaded by path.

    Not imported as `atom.compass.core.<name>`: `atom/__init__.py` imports the
    sglang plugin, so a package import pulls in the engine, and checking who
    answered a request would then need a device. The same reasoning as
    `scripts/compass/execution_id.py`, for the same reason.
    """
    path = ROOT / "atom" / "compass" / "core" / f"{name}.py"
    if not path.exists():
        raise ImportError(
            f"the shared reading of process identity is missing: {path}. "
            f"Without it the harness cannot tell the server it launched from "
            f"one left over on the same port."
        )
    key = f"atom_compass_core_{name}"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module


compare = _load("compare")
plan_module = _load("cc_traces_plan")
execution_id = _load("execution_id")
process_identity = _core("process_identity")


def server_default_port():
    """The port ATOM's entry point listens on when `--server-port` is absent.

    Read from the entry point itself, not assumed here and not taken from
    `--port`: on that parser `--port` is the engine's internal port, so using
    it as a fallback would record a port the server never bound. None when the
    default cannot be read, so "unknown" stays unknown.
    """
    source = ROOT / "atom" / "entrypoints" / "openai" / "api_server.py"
    try:
        text = source.read_text()
    except OSError:
        return None
    for line in text.splitlines():
        head, sep, tail = line.partition("=")
        if sep and head.strip() == "DEFAULT_PORT" and tail.strip().isdigit():
            return tail.strip()
    return None


#: Seconds to wait for a server to answer /health before giving up on it. A
#: 262k-context model at TP=4 loads weights and captures graphs inside this.
STARTUP_TIMEOUT = 1800.0

#: Seconds between health probes. Short enough that `startup_real` is not
#: mostly quantisation error.
HEALTH_INTERVAL = 1.0

#: Seconds to let a server shut down after SIGTERM before SIGKILL. A server
#: killed outright can leave the device's memory allocated, which the next
#: repeat then fails to allocate -- so it is asked first, and only then killed.
STOP_GRACE = 60.0

#: `replay.py` exits 3 when it refuses: a warmed predictor, or a preparation it
#: could not drain. A refusal is evidence, not an error to retry away.
REFUSAL_EXIT = 3

#: The terms a cell's costs.json must carry; the four the harness cannot see
#: are named separately so the message can say which is missing and why.
MEASURED_TERMS = (
    "startup_real",
    "startup_modelled",
    "execution_real",
    "execution_modelled",
)
SUPPLIED_TERMS = ("capture", "calibration", "derivation", "load")

#: The cost record's schema. Version 2 names the clock every duration was
#: taken on, because version 1 wrote the modelled side's *virtual* served
#: window into `execution_modelled` -- a number that says how long the
#: prediction thinks the workload takes, not what producing it cost. A reader
#: that cannot tell the two apart cannot compute a speedup, so a record
#: without this schema is refused rather than reinterpreted.
COSTS_SCHEMA = "compass.costs/2"

#: The clock a duration a human would time with a stopwatch is taken on. The
#: only one a runtime cost may be measured on.
WALL_CLOCK = "wall"


# --------------------------------------------------------------------------
# what an execution is

#: The identity record written for every fresh-server repeat. Versioned
#: because other readers (the calibration registry, the MEMORY classifier)
#: key off it, and a field that changes meaning silently is worse than one
#: that changes name.
#:
#: {
#:   "schema":        "compass.execution/1",
#:   "execution_id":  "cx-<16 hex>",          # unique per launched repeat
#:   "id_rule":       how the id is derived, so it can be re-derived
#:   "id_inputs":     {host, cell, side, repeat, server_pid, launched_at_ns}
#:   "cell":          {"path", "tp", "class"},
#:   "side":          "real" | "modelled",
#:   "repeat":        1-based, within this side of this cell,
#:   "process":       {role, pid, command, log, launched_at, healthy_at,
#:                     startup_s, ended_at, exit},
#:   "replay":        {pid, command, started_at, ended_at, seconds, exit},
#:   "source":        {workload, workload_sha256, replay_target,
#:                     replay_target_sha256, oracle, oracle_options},
#:   "config":        {model, tp, port, mode, engine_args, provenance,
#:                     provenance_sha256},
#:   "server_process": {"said":     the server's own account of itself,
#:                      "observed": what we read from /proc ourselves,
#:                      "verified": whether the two are the same process},
#:   "artifacts":     {name: {"sha256", "bytes"}}
#: }
#:
#: Two repeats of one cell can produce byte-identical artifacts and still be
#: independent executions, so nothing here is derived from payload content:
#: the id is minted from where and when the process was launched.
# The rule itself lives in `execution_id.py`, which imports nothing but the
# standard library, so every reader of these artifacts -- this harness, the
# MEMORY classifier, a notebook opening one file -- verifies an id with the
# same code that minted it. A second implementation would be a second scheme,
# however carefully it was copied.
EXECUTION_SCHEMA = execution_id.EXECUTION_SCHEMA
ID_RULE = execution_id.ID_RULE
ID_FIELDS = execution_id.ID_FIELDS
derive_execution_id = execution_id.derive_execution_id
verify_execution_id = execution_id.verify_execution_id
stamp_of = execution_id.stamp_of
file_digest = execution_id.file_digest

#: What a run is *for*, carried in every execution record and in the stamp
#: inside every artifact. Acceptance is the default, so nothing written before
#: this field existed becomes diagnostic by omission; a diagnostic has to say
#: so. It travels inside the artifacts rather than in a file beside them
#: because a marker beside them is lost the moment somebody copies the
#: interesting files into a cell directory.
ACCEPTANCE = "acceptance"
DIAGNOSTIC = "diagnostic"
PURPOSES = (ACCEPTANCE, DIAGNOSTIC)


# --------------------------------------------------------------------------
# the seams: processes, health, provenance, time


class Processes:
    """Starting and stopping real processes, and nothing else.

    Every method takes an explicit handle or command. There is no lookup by
    name anywhere in this class, so there is no way for it to signal a process
    it did not start.
    """

    def start(self, command, *, log: Path, cwd=None, env=None):
        log.parent.mkdir(parents=True, exist_ok=True)
        handle = log.open("ab")
        proc = subprocess.Popen(
            list(command),
            stdout=handle,
            stderr=subprocess.STDOUT,
            cwd=str(cwd or ROOT),
            env=env,
        )
        proc._compass_log = handle  # closed in stop()
        return proc

    def run(self, command, *, log: Path, cwd=None, env=None) -> int:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("ab") as handle:
            return subprocess.call(
                list(command),
                stdout=handle,
                stderr=subprocess.STDOUT,
                cwd=str(cwd or ROOT),
                env=env,
            )

    def alive(self, proc) -> bool:
        return proc.poll() is None

    def stop(self, proc, *, grace: float = STOP_GRACE) -> int:
        """Ask this exact process to end, then insist. Returns its status."""
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=grace)
        handle = getattr(proc, "_compass_log", None)
        if handle is not None:
            handle.close()
        return proc.returncode


def http_get(url: str, timeout: float = 5.0):
    """A GET that answers with parsed JSON, or None if it could not."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            body = response.read().decode("utf-8", "replace")
    except (urllib.error.URLError, OSError, ValueError):
        return None
    if not body.strip():
        return {}
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return {}


# --------------------------------------------------------------------------
# one side of one cell


class SideRun:
    """The steps of one side, in order, with what each of them did."""

    def __init__(
        self,
        cell_plan: dict,
        side: str,
        *,
        processes=None,
        health=http_get,
        provenance=http_get,
        now=time.monotonic,
        wall=time.time,
        sleep=time.sleep,
        startup_timeout: float = STARTUP_TIMEOUT,
        health_interval: float = HEALTH_INTERVAL,
        host=None,
        probe=process_identity,
        purpose: str = ACCEPTANCE,
    ):
        self.plan = cell_plan
        self.side = side
        self.cell = Path(cell_plan["cell"])
        self.processes = processes or Processes()
        self.health = health
        self.provenance = provenance
        self.now = now
        self.wall = wall
        self.sleep = sleep
        self.startup_timeout = startup_timeout
        self.health_interval = health_interval
        self.host = host or socket.gethostname()
        #: reads /proc for us, so the server's account of itself can be
        #: checked rather than believed; injectable because the tests run
        #: against processes that were never started
        self.probe = probe
        #: acceptance or diagnostic, stamped into every record this run writes
        self.purpose = purpose
        #: step id -> the handle we started, so nothing is signalled by name
        self.running: dict[str, dict] = {}
        #: repeat -> its execution record, minted when its process launched
        self.executions: dict = {}
        self.journal: list[dict] = []
        self.failures: list[str] = []
        self.refused = False

    # -- the pieces ------------------------------------------------------

    def _record(self, step, **fields) -> dict:
        entry = {
            "id": step["id"],
            "role": step["role"],
            "side": step.get("side"),
            "repeat": step.get("repeat"),
            "command": list(step["command"]) if step.get("command") else None,
            "at": self.wall(),
            **fields,
        }
        self.journal.append(entry)
        return entry

    def _log(self, step) -> Path:
        return self.cell / "logs" / f"{step['id']}.log"

    def _mint(self, step, proc, launched_at) -> dict:
        """One execution record, made at the moment the process is launched.

        Not afterwards from the artifacts: by then the only things left to
        identify a run by are its bytes, and two repeats of one cell can
        produce the same bytes and still be independent executions.
        """
        command = list(step["command"])
        inputs = {
            "host": self.host,
            "cell": str(self.cell),
            "side": self.side,
            "repeat": step["repeat"],
            "server_pid": proc.pid,
            "launched_at_ns": int(launched_at * 1e9),
        }
        record = {
            "schema": EXECUTION_SCHEMA,
            "execution_id": derive_execution_id(**inputs),
            "id_rule": ID_RULE,
            "id_inputs": inputs,
            "cell": {
                "path": str(self.cell),
                "tp": self.plan["tp"],
                "class": self.plan["class"],
            },
            "side": self.side,
            "repeat": step["repeat"],
            "process": {
                "role": "server",
                "pid": proc.pid,
                "command": command,
                "log": str(self._log(step)),
                "launched_at": launched_at,
                "healthy_at": None,
                "startup_s": None,
                "ended_at": None,
                "exit": None,
            },
            # What this run is for. In the record and in every artifact stamp,
            # so a diagnostic stays a diagnostic after it is copied.
            "purpose": self.purpose,
            "replay": None,
            "source": self._source(step),
            "config": self._config(step, command),
            # Filled in once the server answers: the server's own account of
            # which process it is, and ours, kept apart. None until then, so
            # "never checked" never reads as "checked and fine".
            "server_process": None,
            "artifacts": {},
        }
        self.executions[step["repeat"]] = record
        return record

    def _source(self, step) -> dict:
        """What this repeat was run *from*, with digests where there is a file."""
        workload = ROOT / f"atom/compass/cc_traces_{self.plan['class']}.jsonl"
        command = step["command"]
        target = None
        if "--compass-replay-target" in command:
            target = command[command.index("--compass-replay-target") + 1]
        options = [
            command[i + 1]
            for i, part in enumerate(command)
            if part == "--compass-oracle-option"
        ]
        oracle = (
            command[command.index("--compass-oracle") + 1]
            if "--compass-oracle" in command
            else None
        )
        return {
            "workload": str(workload),
            "workload_sha256": (file_digest(workload) or {}).get("sha256"),
            "replay_target": target,
            "replay_target_sha256": (
                (file_digest(Path(target)) or {}).get("sha256") if target else None
            ),
            "oracle": oracle,
            "oracle_options": options,
        }

    def _config(self, step, command) -> dict:
        """The configuration as launched. What served it is filled in later."""

        def after(flag):
            return command[command.index(flag) + 1] if flag in command else None

        return {
            "model": after("--model"),
            "tp": after("-tp"),
            # The HTTP port this step is about: the listener a server binds,
            # or the one a client dials. A server's is `--server-port`, or the
            # entry point's own default when the flag is absent -- never its
            # `--port`, which is the engine's internal port and a different
            # thing. The internal port is left to the engine's own default and
            # is not what a client reaches.
            "port": (
                (after("--server-port") or server_default_port())
                if step["role"] == "serve"
                else after("--port")
            ),
            "mode": after("--compass-mode"),
            "engine_args": list(plan_module.ENGINE_ARGS),
            "provenance": None,
            "provenance_sha256": None,
        }

    def _serve(self, step) -> bool:
        """Start this repeat's server, wait for health, read what it is."""
        # Only another server: the sampler is meant to outlive every repeat,
        # and counting it here would refuse the run it exists to watch.
        servers = [
            name
            for name, held in self.running.items()
            if held["step"]["role"] == "serve"
        ]
        if servers:
            still = ", ".join(sorted(servers))
            self.failures.append(
                f"{step['id']}: {still} is still running; a repeat is a fresh "
                f"process and this one would have inherited a warmed server"
            )
            self._record(step, ok=False, reason="previous process still running")
            return False
        started = self.now()
        proc = self.processes.start(step["command"], log=self._log(step))
        execution = self._mint(step, proc, self.wall())
        self.running[step["id"]] = {
            "proc": proc,
            "pid": proc.pid,
            "step": step,
            "execution": execution,
        }
        healthy, reason = self._await_health(step, proc, started)
        seconds = self.now() - started
        entry = self._record(
            step,
            pid=proc.pid,
            execution_id=execution["execution_id"],
            ok=healthy,
            startup_s=seconds if healthy else None,
            reason=reason,
        )
        if healthy:
            execution["process"]["healthy_at"] = self.wall()
            execution["process"]["startup_s"] = seconds
        else:
            self.failures.append(f"{step['id']}: {reason}")
            return False
        said = self.provenance(step["health"].replace("/health", "/compass/provenance"))
        entry["provenance"] = self._check_provenance(step, said, execution)
        if entry["provenance"] is None:
            return False
        # Health and the configuration fields are all answerable by a server
        # left over on this port. Who actually replied is not.
        return self._check_server_process(step, said, execution, proc)

    def _await_health(self, step, proc, started):
        """Poll until it answers, it dies, or we run out of patience."""
        while True:
            if not self.processes.alive(proc):
                return False, (
                    f"the server exited {proc.returncode} before it was "
                    f"healthy; see {self._log(step)}"
                )
            if self.health(step["health"]) is not None:
                return True, None
            if self.now() - started > self.startup_timeout:
                return False, (
                    f"no /health in {self.startup_timeout:.0f}s; see "
                    f"{self._log(step)}"
                )
            self.sleep(self.health_interval)

    def _check_provenance(self, step, said, execution):
        """The server's own account of itself, and whether it is this side's."""
        if not isinstance(said, dict) or not said:
            self.failures.append(
                f"{step['id']}: the server did not answer /compass/provenance, "
                f"so what configuration served this repeat is undeclared"
            )
            return None
        want = "predict" if self.side == "modelled" else "measure"
        got = ((said.get("compass") or {}).get("mode")) or said.get("mode")
        if got != want:
            self.failures.append(
                f"{step['id']}: the server reports mode={got!r} on the "
                f"{self.side} side, which needs {want!r}"
            )
            return None
        # Field names below are the server's, from
        # `atom/entrypoints/openai/api_server.py::compass_provenance`.
        compass = said.get("compass") or {}
        if not compass.get("enabled"):
            self.failures.append(
                f"{step['id']}: the server reports compass disabled, so "
                f"--compass did not take and nothing it serves is a "
                f"Compass result"
            )
            return None
        # `replay.py::_clock_of` decides whether to refuse a warmed predictor
        # by reading exactly this field. A predicting server that reports no
        # virtual clock would never trigger that refusal, so the protection
        # the modelled side depends on would be silently absent.
        virtual = bool(compass.get("virtual_clock"))
        if self.side == "modelled" and not virtual:
            self.failures.append(
                f"{step['id']}: the predicting server reports no virtual "
                f"clock, so replay.py would not refuse a warmed predictor "
                f"and the protection this side depends on is absent"
            )
            return None
        if self.side == "real" and virtual:
            self.failures.append(
                f"{step['id']}: the measuring server is on a virtual clock, "
                f"so its seconds are modelled ones and not measurements"
            )
            return None
        declared = said.get("tensor_parallel_size")
        if declared is not None and int(declared) != int(self.plan["tp"]):
            self.failures.append(
                f"{step['id']}: the server reports tensor_parallel_size="
                f"{declared}, and this cell is tp{self.plan['tp']}"
            )
            return None
        path = self.cell / f"provenance.{self.side}.r{step['repeat']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(said, indent=1) + "\n")
        execution["config"]["provenance"] = said
        execution["config"]["provenance_sha256"] = (file_digest(path) or {}).get(
            "sha256"
        )
        execution["artifacts"][path.name] = file_digest(path)
        return said

    def _ancestry(self, pid, stop, limit=12) -> list:
        """The chain from `pid` upward, stopping at `stop` if we reach it.

        A server need not be the process we spawned. A launcher may fork, and
        then the thing holding the socket is a descendant of it. Being inside
        the tree this repeat started is the claim worth checking, and it is
        precisely the claim a leftover server cannot make.
        """
        chain, seen = [pid], {pid}
        while len(chain) < limit and chain[-1] not in (stop, 0, 1, None):
            parent = self.probe.parent_of(chain[-1])
            if parent is None or parent in seen:
                break
            seen.add(parent)
            chain.append(parent)
        return chain

    def _check_server_process(self, step, said, execution, proc) -> bool:
        """Whether the process that answered is the one this repeat launched.

        `server_code_sha256` is a fact about bytes on disk, not about who
        replied. A server left over from an earlier repeat, holding this port
        and started from the same tree with the same flags, matches on every
        build-and-configuration field this endpoint has. Health answers too --
        it answers *sooner*, because it is already warm, which is the failure
        worth worrying about: our own process is still loading weights, the
        stale one replies first, and the repeat is measured against a server
        nobody meant to start.

        So this compares what the server says about itself against what we can
        read out of `/proc` ourselves. The two accounts are kept apart in the
        record: `said` is the server's, `observed` is ours, and neither is
        written from the other.
        """
        theirs = said.get("server_process")
        port = (execution.get("config") or {}).get("port")
        if not isinstance(theirs, dict) or not theirs.get("pid"):
            self.failures.append(
                f"{step['id']}: the server does not report which process it "
                f"is (no server_process in /compass/provenance), so a reply "
                f"from a server left over on port {port} cannot "
                f"be told from a reply from the one this repeat launched"
            )
            return False
        observed = {
            "launched_pid": proc.pid,
            "host": self.host,
            "boot_id": self.probe.boot_id(),
            "start_ticks": self.probe.start_ticks(theirs["pid"]),
            "ancestry": self._ancestry(theirs["pid"], proc.pid),
            "alive_at_provenance": self.processes.alive(proc),
        }
        execution["server_process"] = {
            "said": theirs,
            "observed": observed,
            "verified": False,
        }

        def refuse(why):
            self.failures.append(f"{step['id']}: {why}")
            return False

        if not observed["alive_at_provenance"]:
            return refuse(
                f"the server this repeat launched (pid {proc.pid}) had "
                f"already exited when /compass/provenance was answered, so "
                f"pid {theirs['pid']} answering on port {port} "
                f"is a different server; see {self._log(step)}"
            )
        short = str(theirs.get("host") or "").split(".")[0]
        if short and short != str(self.host).split(".")[0]:
            return refuse(
                f"the server reports host {theirs['host']!r} but this repeat "
                f"was launched on {self.host!r}, so the reply came from "
                f"another machine and /proc here cannot vouch for it"
            )
        ours_boot = observed["boot_id"]
        if theirs.get("boot_id") and ours_boot and theirs["boot_id"] != ours_boot:
            return refuse(
                "the server reports a different boot than this one, so it "
                "is not a process this machine is currently running"
            )
        if proc.pid not in observed["ancestry"]:
            return refuse(
                f"pid {theirs['pid']} answered on port {port} "
                f"but it is not the process this repeat launched (pid "
                f"{proc.pid}) nor a descendant of it -- a server from an "
                f"earlier run is still holding the port, and its code digest "
                f"matching proves only that it was built from the same tree"
            )
        if observed["start_ticks"] is None:
            return refuse(
                f"pid {theirs['pid']} claims to have served this repeat but "
                f"no such process exists here, so nothing served it that we "
                f"can identify"
            )
        if theirs.get("start_ticks") != observed["start_ticks"]:
            return refuse(
                f"pid {theirs['pid']} started at tick "
                f"{theirs.get('start_ticks')!r} by its own account but at "
                f"{observed['start_ticks']!r} by ours, so the pid has been "
                f"reused and names a different process than the one that "
                f"answered"
            )
        execution["server_process"]["verified"] = True
        return True

    def _replay(self, step) -> bool:
        """This repeat's replay, against the server this repeat started."""
        serve_id = f"serve-{self.side}-{step['repeat']}"
        held = self.running.get(serve_id)
        if held is None:
            self.failures.append(f"{step['id']}: no server for this repeat")
            self._record(step, ok=False, reason="no server")
            return False
        execution = held["execution"]
        started_at, started = self.wall(), self.now()
        code = self.processes.run(step["command"], log=self._log(step))
        seconds = self.now() - started
        alive = self.processes.alive(held["proc"])
        execution["replay"] = {
            "pid": held["pid"],
            "command": list(step["command"]),
            "log": str(self._log(step)),
            "started_at": started_at,
            "ended_at": self.wall(),
            "seconds": seconds,
            "exit": code,
        }
        entry = self._record(
            step,
            pid=held["pid"],
            execution_id=execution["execution_id"],
            exit=code,
            seconds=seconds,
            ok=code == 0 and alive,
        )
        if code == REFUSAL_EXIT:
            self.refused = True
            entry["refusal"] = True
            self.failures.append(
                f"{step['id']}: the replay refused this run (exit 3); see "
                f"{self._log(step)}. A refusal is the result, not a retry."
            )
            return False
        if code != 0:
            self.failures.append(
                f"{step['id']}: the replay exited {code}; see {self._log(step)}"
            )
            return False
        if not alive:
            self.failures.append(
                f"{step['id']}: the server died during the replay, so the "
                f"artifact covers a run that did not finish being served"
            )
            return False
        return self._check_artifact(step, entry, execution)

    def _check_artifact(self, step, entry, execution) -> bool:
        """What the replay wrote, read back before the process is stopped."""
        path = self.cell / f"{self.side}.r{step['repeat']}.json"
        if not path.exists():
            self.failures.append(f"{step['id']}: {path.name} was not written")
            return False
        try:
            blob = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            self.failures.append(f"{step['id']}: {path.name} is not JSON: {exc}")
            return False
        manifest = blob.get("run") or {}
        bad = []
        # Manifest field names below are `replay.py`'s, from the dict it
        # writes as `run`.
        prepare = manifest.get("prepare") or {}
        if self.side == "real":
            if not manifest.get("paced"):
                bad.append(
                    "it was not paced, so its arrivals were declared to a "
                    "server that stamps on receipt"
                )
            if not prepare.get("drained"):
                bad.append(
                    "it carries no drained preparation, and the drain is "
                    "where the measurement starts"
                )
            elif not prepare.get("store_empty_after_drain"):
                bad.append(
                    "the engine's record store was not empty after the "
                    "drain, so a preparation row can enter the measurement"
                )
            elif not prepare.get("drained_records"):
                bad.append(
                    "the preparation drained no engine records at all, so "
                    "nothing shows it reached the engine it was warming"
                )
        else:
            if manifest.get("paced"):
                bad.append("it was paced against a virtual clock")
            if manifest.get("prepare"):
                bad.append(
                    "it was prepared, which lands inside every declared "
                    "arrival's TTFT"
                )
        # The frozen corpus, checked by its bytes rather than by its path:
        # the replay records the digest of the trace it actually read.
        want_trace = (execution.get("source") or {}).get("workload_sha256")
        got_trace = manifest.get("trace_sha256")
        if want_trace and got_trace and want_trace != got_trace:
            bad.append(
                f"it replayed a trace whose digest is {got_trace[:12]}, and "
                f"this cell's frozen workload is {want_trace[:12]}"
            )
        # And the server it reached. `replay.py` embeds the server's own
        # `/compass/provenance` under `run.server`; if that is not the process
        # this harness started, something else was listening on the port and
        # the repeat measured a server nobody recorded.
        served = manifest.get("server") or {}
        mine = (execution["config"].get("provenance") or {}).get("server_code_sha256")
        theirs = served.get("server_code_sha256")
        if not served:
            bad.append(
                "it carries no server provenance, so which process answered "
                "it cannot be recovered from the artifact"
            )
        elif mine and theirs and mine != theirs:
            bad.append(
                f"it was answered by a server whose code digest is "
                f"{theirs[:12]}, and the process this repeat started reports "
                f"{mine[:12]}: something else was listening on that port"
            )
        stamped = (blob.get("execution") or {}).get("execution_id")
        if stamped and stamped != execution["execution_id"]:
            # An artifact that already belongs to another execution: a stale
            # file left in the cell, or one copied in from elsewhere. Either
            # way it is not what this repeat produced.
            bad.append(
                f"{path.name} already carries execution {stamped}, which is "
                f"not this repeat's ({execution['execution_id']})"
            )
        entry["manifest_ok"] = not bad
        for reason in bad:
            self.failures.append(f"{step['id']}: {reason}")
        if bad:
            return False
        self._stamp(path, blob, execution)
        for name in (
            f"{self.side}.r{step['repeat']}.prepare.json",
            f"real.r{step['repeat']}_steps.jsonl",
        ):
            found = file_digest(self.cell / name)
            if found:
                execution["artifacts"][name] = found
        return True

    def _stamp(self, path, blob, execution):
        """Write the identity into the artifact, then digest what is on disk.

        Into the file rather than only beside it: a copy of an artifact is
        still that execution, and a reader who has the file has the id. The
        digest is taken after stamping, so it describes the bytes that exist.
        """
        stamp = stamp_of(execution)
        # Beyond the canonical identity fields, which the runtime package owns:
        # what the run was for, so a reader holding only this file can tell.
        stamp["purpose"] = execution.get("purpose", ACCEPTANCE)
        blob["execution"] = stamp
        path.write_text(json.dumps(blob, indent=1) + "\n")
        execution["artifacts"][path.name] = file_digest(path)

    def _stop(self, step) -> bool:
        """End exactly the process the named step started, by its own handle."""
        held = self.running.pop(step["stops"], None)
        if held is None:
            self._record(step, ok=True, reason="nothing of ours was running")
            return True
        code = self.processes.stop(held["proc"])
        execution = held.get("execution")
        if execution is not None:
            execution["process"]["ended_at"] = self.wall()
            execution["process"]["exit"] = code
            self._write_execution(execution)
        watched = True
        if held["step"]["role"] == "sample":
            # Now that it has stopped, ask what it saw -- before the run ends
            # and the answer is somebody else's problem.
            watched = self._check_sampler(step, self.cell / "gpu.jsonl")
        self._record(step, pid=held["pid"], exit=code, ok=watched)
        return watched

    def _check_sampler(self, step, path) -> bool:
        """Did the watch actually cover the window it was meant to cover?

        `isolation.py` judges what the samples say; this asks the question
        before that one -- whether there are samples spanning the repeats at
        all. A sampler that died after its first tick leaves a file that audits
        clean, and a clean audit over four seconds of a forty-minute window is
        not evidence that the node was quiet.
        """
        if not path.exists():
            self.failures.append(
                f"{step['id']}: {path.name} was not written, so the real "
                f"window went unwatched and no repeat in it can be attributed"
            )
            return False
        times, phases = [], set()
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(sample.get("t"), (int, float)):
                times.append(float(sample["t"]))
            if sample.get("phase"):
                phases.add(sample["phase"])
        if not times:
            self.failures.append(
                f"{step['id']}: {path.name} carries no timed sample, so "
                f"nothing in it can be placed against a repeat"
            )
            return False
        if "baseline" not in phases:
            # The only sample that can show a card was already somebody
            # else's: everything after the first server starts includes us.
            self.failures.append(
                f"{step['id']}: {path.name} has no baseline sample, so "
                f"whether the cards were already busy cannot be told"
            )
            return False
        launches = [
            e["process"]["launched_at"]
            for e in self.executions.values()
            if e["process"].get("launched_at") is not None
        ]
        ends = [
            e["process"]["ended_at"]
            for e in self.executions.values()
            if e["process"].get("ended_at") is not None
        ]
        first, last = min(times), max(times)
        if launches and first > min(launches):
            self.failures.append(
                f"{step['id']}: the watch starts {first - min(launches):.1f}s "
                f"after the first server did, so that much of the window is "
                f"unobserved"
            )
            return False
        if ends and last < max(ends):
            self.failures.append(
                f"{step['id']}: the watch stops {max(ends) - last:.1f}s "
                f"before the last server did, so that much of the window is "
                f"unobserved"
            )
            return False
        return True

    def _write_execution(self, execution):
        """The record beside the artifacts, as well as inside them."""
        path = self.cell / f"execution.{self.side}.r{execution['repeat']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(execution, indent=1) + "\n")

    def _command(self, step) -> bool:
        started = self.now()
        code = self.processes.run(step["command"], log=self._log(step))
        self._record(step, exit=code, seconds=self.now() - started, ok=code == 0)
        if code != 0:
            self.failures.append(f"{step['id']}: exited {code}; see {self._log(step)}")
            return False
        return True

    def _sample(self, step) -> bool:
        proc = self.processes.start(step["command"], log=self._log(step))
        self.running[step["id"]] = {"proc": proc, "pid": proc.pid, "step": step}
        self._record(step, pid=proc.pid, ok=True)
        return True

    # -- the run ---------------------------------------------------------

    def steps(self):
        return [s for s in self.plan["steps"] if s.get("side") == self.side]

    def run(self) -> int:
        """Every step of this side. Stops at the first failure, cleans up."""
        self.cell.mkdir(parents=True, exist_ok=True)
        try:
            for step in self.steps():
                role = step["role"]
                if role == "serve":
                    ok = self._serve(step)
                elif role == "replay":
                    ok = self._replay(step)
                elif role == "stop":
                    ok = self._stop(step)
                elif role == "sample":
                    ok = self._sample(step)
                else:
                    ok = self._command(step)
                if not ok:
                    break
        finally:
            self._cleanup()
        if not self.failures:
            self._write_costs()
        self._write_journal()
        if self.refused:
            return REFUSAL_EXIT
        return 1 if self.failures else 0

    def _cleanup(self):
        """Whatever of ours is still up, by the pid we recorded for it."""
        for step_id in list(self.running):
            held = self.running.pop(step_id)
            code = self.processes.stop(held["proc"])
            execution = held.get("execution")
            if execution is not None:
                execution["process"]["ended_at"] = self.wall()
                execution["process"]["exit"] = code
                self._write_execution(execution)
            self.journal.append(
                {
                    "id": f"cleanup-{step_id}",
                    "role": "stop",
                    "side": self.side,
                    "pid": held["pid"],
                    "execution_id": (execution["execution_id"] if execution else None),
                    "exit": code,
                    "at": self.wall(),
                    "ok": True,
                }
            )

    def _write_journal(self):
        path = self.cell / f"run.{self.side}.json"
        path.write_text(
            json.dumps(
                {
                    "schema": EXECUTION_SCHEMA,
                    "cell": str(self.cell),
                    "side": self.side,
                    "purpose": self.purpose,
                    "host": self.host,
                    "tp": self.plan["tp"],
                    "class": self.plan["class"],
                    "ok": not self.failures,
                    "executions": [self.executions[n] for n in sorted(self.executions)],
                    "refused": self.refused,
                    "failures": list(self.failures),
                    "steps": self.journal,
                },
                indent=1,
            )
            + "\n"
        )

    def _write_costs(self):
        """This side's own seconds, on both clocks, each one named.

        Two different durations were both called "the execution" before, and
        on the modelled side they differ by more than an order of magnitude:

        * the **served window**, from `compare.metrics` -- the same
          implementation the comparison reads, so a cost term and a metric
          cannot disagree about what was served. On a predicting server that
          window is on the engine's *virtual* clock: it is how long the
          prediction says the workload would have taken, which is a statement
          about accuracy, not about what running the predictor cost.
        * the **wall window**, the replay's own stopwatch around the client.
          That is the machine time the modelled path actually spent, and it is
          what a speedup over serving for real is a ratio of.

        So both are written, both are labelled with the clock they were taken
        on, and the cost term `execution_<side>` is the wall one. Every repeat
        is kept; the term is the median, under the one quantile convention.
        """
        startups = [
            e["startup_s"]
            for e in self.journal
            if e["role"] == "serve" and e.get("startup_s") is not None
        ]
        executions, served_windows, per_execution = [], [], []
        clocks = set()
        for entry in self.journal:
            if entry["role"] != "replay" or not entry.get("ok"):
                continue
            path = self.cell / f"{self.side}.r{entry['repeat']}.json"
            run = compare.load_run(str(path), f"{self.side}[{entry['repeat']}]")
            served = compare.metrics(run, sorted(run.joined))["window_s"]
            # The client's own elapsed, already recorded when the replay ran.
            wall = entry.get("seconds")
            clocks.add(run.clock)
            executions.append(wall)
            served_windows.append(served)
            # Each second attributed to the execution that spent it, so a
            # reader can tell a source residual from an independent repeat.
            per_execution.append(
                {
                    "execution_id": entry.get("execution_id"),
                    "repeat": entry["repeat"],
                    "execution_s": wall,
                    "served_window_s": served,
                    "startup_s": (self.executions.get(entry["repeat"]) or {})
                    .get("process", {})
                    .get("startup_s"),
                }
            )
        payload = {
            "cost_schema": COSTS_SCHEMA,
            "side": self.side,
            "repeats": len(executions),
            "startup_s": startups,
            "execution_s": executions,
            "served_window_s": served_windows,
            # Declared, not inferred from the side: the artifact says which
            # clock its records were stamped on, and that is what is recorded.
            "clocks": {
                "startup": WALL_CLOCK,
                "execution": WALL_CLOCK,
                "served_window": (
                    sorted(clocks) if len(clocks) > 1 else (next(iter(clocks), "?"))
                ),
            },
            "per_execution": per_execution,
            f"startup_{self.side}": _median(startups),
            f"execution_{self.side}": _median(executions),
            f"served_window_{self.side}": _median(served_windows),
            "convention": compare.QUANTILE_CONVENTION,
            "means": (
                "seconds. startup is launch to the first /health answer, on "
                "the wall clock. execution is the replay's own wall-clock "
                "window: the machine time this side spent. served_window is "
                "the window the engine reports having served, from "
                "compare.metrics, on the engine's own clock -- virtual on a "
                "predicting server, and never a runtime cost"
            ),
        }
        (self.cell / f"costs.{self.side}.json").write_text(
            json.dumps(payload, indent=1) + "\n"
        )


def _median(values):
    if not values:
        return None
    s = sorted(values)
    return s[min(len(s) - 1, int(0.5 * len(s)))]


# --------------------------------------------------------------------------
# the commands


def _cell_plan(args) -> dict:
    """This cell's steps, from the plan both sides read."""
    cell = Path(args.cell).resolve()
    built = plan_module.cell_steps(
        args.tp,
        args.klass,
        root=str(cell.parent),
        oracle=getattr(args, "oracle", None),
        options=getattr(args, "oracle_option", ()) or (),
        port=args.port,
        repeats=args.repeats,
        target=getattr(args, "replay_target", None),
        corpus=getattr(args, "corpus", None) or "$CC_TRACES_CORPUS",
    )
    if Path(built["cell"]).name != cell.name:
        raise SystemExit(
            f"--cell {cell.name} is not tp{args.tp}_{args.klass}: the "
            f"directory name is what the plan and the validator agree on"
        )
    built["cell"] = str(cell)
    for step in built["steps"]:
        if step.get("command"):
            step["command"] = [
                part.replace(built["cell"], str(cell)) for part in step["command"]
            ]
    return built


def side(args) -> int:
    if args.side == "modelled" and not args.replay_target:
        print(
            "the modelled side needs --replay-target: without a captured "
            "target there is no Config to serve from on a machine with no "
            "device, and inventing one would be inventing the prediction",
            file=sys.stderr,
        )
        return 2
    if args.side == "real" and args.replay_target:
        print(
            "--replay-target is the modelled side's input; the real side "
            "serves the model itself",
            file=sys.stderr,
        )
        return 2
    runner = SideRun(
        _cell_plan(args), args.side, purpose=getattr(args, "purpose", ACCEPTANCE)
    )
    code = runner.run()
    for reason in runner.failures:
        print(reason, file=sys.stderr)
    print(
        f"{args.side}: {len(runner.journal)} steps, "
        f"{'ok' if not runner.failures else 'FAILED'} -> "
        f"{runner.cell / f'run.{args.side}.json'}"
    )
    return code


def costs(args) -> int:
    """Merge both sides' measured seconds with the four supplied terms."""
    cell = Path(args.cell)
    merged, missing = {}, []
    for name in ("real", "modelled"):
        path = cell / f"costs.{name}.json"
        if not path.exists():
            missing.append(f"{path.name} (the {name} side has not run)")
            continue
        partial = json.loads(path.read_text())
        # Which clock this side's execution term was taken on travels with it.
        # Without it a later reader cannot tell a machine-time cost from a
        # predicted duration, and the two differ by an order of magnitude.
        said = partial.get("cost_schema")
        clock = (partial.get("clocks") or {}).get("execution")
        if said != COSTS_SCHEMA:
            missing.append(
                f"cost_schema in {path.name} (it says {said!r}, not "
                f"{COSTS_SCHEMA!r}, so which clock its seconds were taken on "
                f"is unrecorded)"
            )
        elif clock != WALL_CLOCK:
            missing.append(
                f"a wall-clock execution term in {path.name} (it was taken on "
                f"the {clock!r} clock, which is a predicted duration and not "
                f"a runtime cost)"
            )
        else:
            merged.setdefault("execution_clocks", {})[name] = clock
            served = partial.get(f"served_window_{name}")
            if isinstance(served, (int, float)) and math.isfinite(served):
                merged[f"served_window_{name}"] = float(served)
        for term in (f"startup_{name}", f"execution_{name}"):
            value = partial.get(term)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                missing.append(f"{term} in {path.name}")
            else:
                merged[term] = float(value)
    for term in SUPPLIED_TERMS:
        value = getattr(args, term)
        if value is None or not math.isfinite(value):
            missing.append(
                f"--{term} (nothing in this cell measures it; it is an input)"
            )
        else:
            merged[term] = float(value)
    if missing:
        print(
            "costs.json not written, because it would be missing: "
            + "; ".join(missing),
            file=sys.stderr,
        )
        return 2
    merged["cost_schema"] = COSTS_SCHEMA
    merged["means"] = (
        "seconds; startup and execution measured by cc_traces_run.py from "
        "this cell's own repeats, on the wall clock, the rest supplied at "
        "merge time. served_window_* is what the engine reports having "
        "served -- virtual on a predicting server -- and is carried for "
        "comparison, never as a cost"
    )
    merged["supplied"] = list(SUPPLIED_TERMS)
    merged["measured"] = list(MEASURED_TERMS)
    (cell / "costs.json").write_text(json.dumps(merged, indent=1) + "\n")
    print("costs.json: " + ", ".join(f"{t}={merged[t]:.3f}" for t in MEASURED_TERMS))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("side", help="run one side of one cell")
    s.add_argument("--cell", required=True)
    s.add_argument("--side", required=True, choices=("real", "modelled"))
    s.add_argument("--tp", type=int, required=True)
    s.add_argument("--class", dest="klass", required=True, choices=("short", "long"))
    s.add_argument("--repeats", type=int, default=plan_module.REPEATS)
    s.add_argument("--port", type=int, default=plan_module.PORT)
    s.add_argument("--oracle", default=None)
    s.add_argument("--oracle-option", action="append", default=[])
    s.add_argument("--replay-target", default=None)
    s.add_argument("--corpus", default=None)
    s.add_argument(
        "--purpose",
        default=ACCEPTANCE,
        choices=PURPOSES,
        help=(
            "what this run is for. Diagnostic runs are stamped into every "
            "execution record and artifact, and the validator refuses a cell "
            "built from them however the directory is named"
        ),
    )
    s.set_defaults(func=side)

    c = sub.add_parser("costs", help="merge the cell's cost terms")
    c.add_argument("cell")
    for term in SUPPLIED_TERMS:
        c.add_argument(f"--{term}", type=float, default=None)
    c.set_defaults(func=costs)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
