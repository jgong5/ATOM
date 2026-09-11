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
import hashlib
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


compare = _load("compare")
plan_module = _load("cc_traces_plan")

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
#:   "artifacts":     {name: {"sha256", "bytes"}}
#: }
#:
#: Two repeats of one cell can produce byte-identical artifacts and still be
#: independent executions, so nothing here is derived from payload content:
#: the id is minted from where and when the process was launched.
EXECUTION_SCHEMA = "compass.execution/1"

ID_RULE = (
    "sha256 of the id_inputs values joined by NUL, in the order "
    "host, cell, side, repeat, server_pid, launched_at_ns; first 16 hex "
    "characters, prefixed 'cx-'"
)


def derive_execution_id(host, cell, side, repeat, server_pid, launched_at_ns) -> str:
    """The one identifier, from the facts of the launch.

    Derived rather than random so that it can be checked: everything it is made
    of is recorded beside it, and `verify_execution_id` re-computes it. A pid is
    reused by the kernel eventually and a clock can be set backwards, which is
    why neither is the id on its own.
    """
    parts = [
        str(host),
        str(cell),
        str(side),
        str(repeat),
        str(server_pid),
        str(launched_at_ns),
    ]
    digest = hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()
    return f"cx-{digest[:16]}"


def verify_execution_id(record: dict) -> bool:
    """Does this record's id follow from its own recorded inputs?"""
    inputs = record.get("id_inputs") or {}
    try:
        expected = derive_execution_id(
            inputs["host"],
            inputs["cell"],
            inputs["side"],
            inputs["repeat"],
            inputs["server_pid"],
            inputs["launched_at_ns"],
        )
    except KeyError:
        return False
    return expected == record.get("execution_id")


def file_digest(path: Path):
    """A file's digest and size, or None if it is not there."""
    if not Path(path).exists():
        return None
    data = Path(path).read_bytes()
    return {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


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
            "replay": None,
            "source": self._source(step),
            "config": self._config(step, command),
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
            "port": after("--port"),
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
        return entry["provenance"] is not None

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
        path = self.cell / f"provenance.{self.side}.r{step['repeat']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(said, indent=1) + "\n")
        execution["config"]["provenance"] = said
        execution["config"]["provenance_sha256"] = (file_digest(path) or {}).get(
            "sha256"
        )
        execution["artifacts"][path.name] = file_digest(path)
        return said

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
        if self.side == "real":
            if not manifest.get("paced"):
                bad.append(
                    "it was not paced, so its arrivals were declared to a "
                    "server that stamps on receipt"
                )
            if not (manifest.get("prepare") or {}).get("drained"):
                bad.append(
                    "it carries no drained preparation, and the drain is "
                    "where the measurement starts"
                )
        else:
            if manifest.get("paced"):
                bad.append("it was paced against a virtual clock")
            if manifest.get("prepare"):
                bad.append(
                    "it was prepared, which lands inside every declared "
                    "arrival's TTFT"
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
        blob["execution"] = {
            "schema": EXECUTION_SCHEMA,
            "execution_id": execution["execution_id"],
            "id_rule": ID_RULE,
            "id_inputs": dict(execution["id_inputs"]),
            "cell": dict(execution["cell"]),
            "side": execution["side"],
            "repeat": execution["repeat"],
        }
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
        self._record(step, pid=held["pid"], exit=code, ok=True)
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
        """This side's own seconds: launch to healthy, and the window itself.

        The window comes from `compare.metrics`, the same implementation the
        comparison reads, so a cost term and a metric cannot disagree about
        what the measured window was. Every repeat is kept; the term is the
        median, under the repository's one quantile convention.
        """
        startups = [
            e["startup_s"]
            for e in self.journal
            if e["role"] == "serve" and e.get("startup_s") is not None
        ]
        executions, per_execution = [], []
        for entry in self.journal:
            if entry["role"] != "replay" or not entry.get("ok"):
                continue
            path = self.cell / f"{self.side}.r{entry['repeat']}.json"
            run = compare.load_run(str(path), f"{self.side}[{entry['repeat']}]")
            window = compare.metrics(run, sorted(run.joined))["window_s"]
            executions.append(window)
            # Each second attributed to the execution that spent it, so a
            # reader can tell a source residual from an independent repeat.
            per_execution.append(
                {
                    "execution_id": entry.get("execution_id"),
                    "repeat": entry["repeat"],
                    "execution_s": window,
                    "startup_s": (self.executions.get(entry["repeat"]) or {})
                    .get("process", {})
                    .get("startup_s"),
                }
            )
        payload = {
            "side": self.side,
            "repeats": len(executions),
            "startup_s": startups,
            "execution_s": executions,
            "per_execution": per_execution,
            f"startup_{self.side}": _median(startups),
            f"execution_{self.side}": _median(executions),
            "convention": compare.QUANTILE_CONVENTION,
            "means": (
                "startup is launch to the first /health answer; execution is "
                "the replay's own window, from compare.metrics"
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
    runner = SideRun(_cell_plan(args), args.side)
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
    merged["means"] = (
        "seconds; startup and execution measured by cc_traces_run.py from "
        "this cell's own repeats, the rest supplied at merge time"
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
