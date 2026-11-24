# SPDX-License-Identifier: MIT
"""The check `compass advice` makes before a search simulates: the reuse
check's verdicts (`drive.reuse`) on each point's laws and memory readings,
written into OUT as `REUSE`, and each candidate's one state, written as
`STATES`.

The head is taken once, at the commit SPACE freezes or the integration branch's
tip; a tree that is not the head gives every verdict ``refused``, requirement
1, with the reason. A candidate is looked up in each of `reuse.stores` in turn
and takes the first that holds every input of its points, whatever the verdicts
on them there: its platform readings compose with SPACE's machine, and no
verdict names requirement 1, so each point's readings, and each law that is not
a checked-in file, are at their key at the head's ATOM digest. With none, it
takes the first whose platform readings compose and that one of its laws is
read from, the store's own step linear law at the head's ATOM digest, whatever
the verdict on it, as a readings gap is measured into the store a law is read
from; else the shared store's. There, each instance of
each point is resolved as ATOM's server resolves its arguments, under its
variables, on the device arch of the machine spec that store's platform
readings compose with SPACE's machine; a 1P1D side as `deploy.sh run` starts
it, as `compass run --check` resolves one. The laws are judged at each point,
since an argument that varies with the concurrency moves their key. A refusal
of the composition, of an argument by ATOM's parser, of a configuration or of
the runner under a law is ``refused``, verbatim, naming no requirement; so is
any exception resolving a configuration, by its exception line, but one
Compass's own code raises other than as a refusal, a defect, which fails
`compass advice` with its traceback. A law
verdict ``calibrate``
names the command that makes the law as `compass run --check` prints it: on
one node the `calibrate.sh` command of its records into the candidate's
calibration directory under OUT's ``cal/`` (`calibrating`), its clients on
SPACE's traces, publishing into the store
the `INPUTS` of a calibration started there name, or its own when they name
none, as a resume must take the store it started with; before one starts, into
the shared store when the verdicts were judged there and it holds neither a
step linear law of the configuration at the head's ATOM digest nor readings at
their key, since it refuses a second publish at a key, else into the
calibration's own; on a 1P1D side `deploy.sh record` and a fit of that side.
Where an earlier candidate none of whose verdicts is refused calibrates a law
at the same key, by model, platform and engine config, a candidate names that
calibration in place of its own command, which the store would refuse as a
second law at the key, with the command measuring its readings into the store
that calibration publishes into, unless they are at their key there; into the
calibration's own store, only once it has started, as a publish there before
would leave its directory holding files, which `calibrate.sh` refuses.
A calibration that has ended records nothing more, so the next goes into a new
directory; where it would take what the ended one took and the law it fitted is
at the configuration's key at the head's ATOM digest (`_refits`), it would refit
that law, and the verdict says so in place of a command, as does a later
candidate's at that key, with the command measuring its readings into the store
that law was published into, unless they are at their key there.
A readings verdict ``measure`` names the `compass measure memory` command,
under the instance's variables, a 1P1D side's on the connector a real start of
it takes; an offloading instance's with the candidate's lmcache server beside
the engine, reached as a start of its point reaches its own, at the port of
the first of the candidate's points that runs alike. A point whose offloading
instance's variables cannot name that server is ``refused``, as its start
would be.
Where a verdict sends a point's readings to measure though an entry holds their
key in the store a candidate is looked up in, every point's command measures
its readings into a new store under OUT's ``cal/`` (`READINGS`), those at their
key in that store too, as a run reads one store; `reuse.stores` looks every
candidate up there.

`states` gives each candidate one state: ``refused``, with each refusal
verbatim, when any verdict is; else, when a law's verdict is ``calibrate``,
``calibrating`` while the `STATUS` file in the candidate's calibration
directory ends in a step's start or end, with each command as the
one that resumes it once no `calibrate.sh` runs, as after a kill that records
no failure; else ``missing a law``, with each command, which resumes a
calibration that stopped; either with the status file's last line and how many
steps it has seen end; else ``searchable``. A searchable candidate's point
whose readings a verdict sends to measure is ``missing readings``, with its
command. Each other point starts with no traffic, as the search will run it,
for what only a start shows, in the directory the check resolved its arguments
in: `compass run`, or `deploy.sh run` on 1P1D with `ROUTER`'s router arguments,
its decode side through a local ``--on`` prefix carrying its own variables where
SPACE names a network; a point offloading to an lmcache server with the run
file's ``lmcache`` entry, its server at the point's own port; a point of a
candidate naming a Mooncake Store's entry with that file, every point the same.
A point answers once the start's ``server.json`` names it, and is
``searchable``; a start that leaves first, or does not answer in `START_S`,
refuses the candidate with the start's last line and each log's first raised
error. Points whose instances run alike start once. Nothing is calibrated or
measured.
"""

import contextlib
import functools
import itertools
import json
import os
import shlex
import shutil
import signal
import subprocess
import time
import traceback
from pathlib import Path

from atom.compass import run as compass_run
from atom.compass.artifacts import (
    ArtifactRefusal,
    Kind,
    keyed_platform,
    machine_platform,
)
from atom.compass.drive import lacking, reuse, runfile, serve
from atom.compass.drive import run as drive_run
from atom.compass.fit.device import compose
from atom.compass.kv.lmcache_server import DECLARED_LAWS
from atom.compass.run import memory
from atom.compass.run.step_grid import keyed
from atom.compass.spec import SpecRefusal

#: The verdicts, in OUT.
REUSE = "reuse.json"
#: What a refusal of an input raises, recorded as its verdict.
REFUSALS = (ArtifactRefusal, SpecRefusal, ValueError, OSError)
#: Compass's own code: anything else raised in it resolving a config is a defect.
OWN = Path(__file__).resolve().parents[1]
#: Each candidate's state, in OUT.
STATES = "states.json"
#: Where each candidate's calibration writes, and each point starts, in OUT.
CAL, STARTS = "cal", "starts"
#: Beside what, in `CAL`, readings meeting a stale entry at their key are
#: measured into a new store (`lacking.readings_store`); no candidate's name
#: holds the ``@`` its store's directory name does.
READINGS = "readings"
#: Why readings at their key in a candidate's store are measured into the new
#: store, with those of its points whose key holds a stale entry there.
MOVED = (
    "at their key in {}, which holds a stale entry at another of the "
    "candidate's readings keys, so they go into {} with the others: a run "
    "reads one store"
)
#: The file a calibration appends each step's start, end or failure to, in its
#: directory, and ``end calibration`` once it has fitted both laws.
STATUS = "status"
#: What makes a law an earlier candidate's calibration makes at its key: that
#: calibration, named by its candidate and its directory, and the store it
#: publishes into.
BORROWED = (
    "{}'s calibration in {}, whose law is at this configuration's key, "
    "publishing into {}"
)
#: What stands for a borrower's readings command into the calibration's own
#: store until that calibration starts.
PENDING = (
    "its readings, into {}, once {}'s calibration in {} has started: compass "
    "advice then prints their command"
)
#: The file a calibration writes its inputs to, in its directory: a line
#: ``store <path>`` names the store it publishes into, when one was named.
INPUTS = "inputs"
#: A candidate's calibration directory after its first, in OUT's `CAL`, by
#: its count from 2: no candidate's name holds a ``+``.
AGAIN = "{}+{}"
#: The step linear law a calibration fits, in its directory, as `calibrate.sh`
#: writes it.
FITTED = Path("fit_steps", "step_linear_law.json")
#: What stands for the command of a candidate whose calibration has ended,
#: when one in a new directory would refit its law (`_refits`).
REFITS = (
    "no calibrate.sh command: the calibration in {} has ended on the inputs a "
    "new one would take, its law at this configuration's key at the head's ATOM "
    "digest, so a new one refits the law it published into {}"
)
#: What a start runs: a server on one node, a deployment on 1P1D.
COMPASS, DEPLOY = (
    serve.TREE / "scripts" / "compass" / n for n in ("compass", "deploy.sh")
)
#: Seconds a start has to answer, and its stop to leave: deploy.sh's default
#: wait for each of its instances and its router.
START_S = 3 * 600
#: The router arguments of a 1P1D start: ATOM's adapter, which asks the
#: prefill side for the KV transfer the decode side takes, where the router's
#: default is SGLang's; and none of the wall-clock failure detectors, which
#: `deploy.sh run` leaves its caller to switch off.
ROUTER = (
    "--backend",
    "atom",
    "--disable-health-check",
    "--disable-circuit-breaker",
    "--disable-retries",
)
#: The variable ATOM reads its KV offload connector's extra config from, which
#: names the lmcache server an offloading point reaches.
EXTRA = "ATOM_KV_OFFLOAD_EXTRA_CONFIG"


def check(expansion: dict, out: Path) -> dict:
    """`expansion`'s verdicts, written into `out`: the ``head``, the reason
    the tree is ``refused``, if it is, and per candidate the ``store`` it is
    looked up in and each point's ``law`` and ``readings`` verdicts per
    instance."""
    try:
        head = reuse.head(expansion["commit"])
        reuse.refuse_another_tree(head)
        refused = None
    except ValueError as error:
        head, refused = None, _refused(error, 1)
    judged, seen, machine, calibrations = {}, {}, expansion["machine"], {}
    for candidate in expansion["candidates"]:
        points = [p for p in expansion["points"] if p["candidate"] == candidate["name"]]
        if refused:
            verdicts = {
                p["name"]: _point(p, lambda *_: (refused, refused)) for p in points
            }
            judged[candidate["name"]] = {"store": None, "points": verdicts}
            continue
        holding_law, reached = None, _reached(candidate, points)
        directory, ended = calibrating(out, candidate["name"])
        for store in reuse.stores(expansion["store"], out):
            given = (machine["device"], store, machine["ptl"])
            given += (machine.get("stack"), machine.get("network"))
            spec = _or_refused(compose, *given)
            made = directory, expansion["workload"]["traces"]
            made += (expansion["store"], ended)
            judge = functools.partial(_judged, store, spec, head["atom"], seen, made)
            verdicts = {p["name"]: _point(p, judge) for p in reached}
            if isinstance(spec, dict):
                continue
            if all(v["requirement"] != 1 for v in _each(verdicts)):
                break
            laws = (v for point in verdicts.values() for v in point["law"].values())
            read = any(v.get("store") == str(store.root) for v in laws)
            if holding_law is None and read:
                holding_law = store, verdicts
        else:
            store, verdicts = holding_law or (store, verdicts)
        judged[candidate["name"]] = {"store": str(store.root), "points": verdicts}
        _borrowing(reached, judged[candidate["name"]], calibrations)
        new = lacking.readings_store(out / CAL / READINGS, head["atom"])
        _measuring(judged[candidate["name"]], new)
    written = {"head": head, "refused": refused, "candidates": judged}
    (out / REUSE).write_text(json.dumps(written, indent=1) + "\n")
    return written


def lines(judged: dict) -> list[str]:
    """A candidate's `check` verdicts as lines: where it is looked up, and each
    distinct verdict per input and instance, naming its points where they
    differ."""
    out = [f"  looked up in {judged['store']}"]
    for what in ("law", "readings"):
        held = {}
        for point, verdicts in judged["points"].items():
            for side, v in verdicts[what].items():
                held.setdefault((side, json.dumps(v)), []).append(point)
        for (side, v), points in held.items():
            v, every = json.loads(v), len(points) == len(judged["points"])
            at = "" if every else " at " + ", ".join(points)
            n = f", requirement {v['requirement']}" if v["requirement"] else ""
            out.append(f"  {what} {side}{at}: {v['verdict']}{n}: {v['reason']}")
    return out


def states(expansion: dict, judged: dict, out: Path) -> dict:
    """Each candidate's one state from its `check` verdicts `judged`, written
    into `out` as `STATES`: its ``state``, each distinct reason ``why`` and
    command ``made_by`` of the verdicts that decide it, and its ``points``,
    each with its own state, for a searchable candidate."""
    found = {}
    for candidate in expansion["candidates"]:
        held = judged["candidates"][candidate["name"]]
        laws = [v for point in held["points"].values() for v in point["law"].values()]
        if refused := [v for v in _each(held["points"]) if v["verdict"] == "refused"]:
            state = _state("refused", refused, points={})
        elif calibrate := [v for v in laws if v["verdict"] == "calibrate"]:
            directory, _ = calibrating(out, candidate["name"])
            state = _calibration(directory, calibrate)
        else:
            state = _searchable(expansion, candidate, held, out)
        found[candidate["name"]] = state
    (out / STATES).write_text(json.dumps(found, indent=1) + "\n")
    return found


def state_lines(state: dict) -> list[str]:
    """A candidate's `states` entry as lines: its state, then each reason and
    command, a calibrating one's as resuming it, then each point missing
    readings with its command."""
    searchable = [p for p, v in state["points"].items() if v["state"] == "searchable"]
    out = [f"  state: {state['state']}"]
    if state["points"]:
        out[0] += f", {len(searchable)} of {len(state['points'])} points searchable"
    out += [
        f"    refused: {why}" for why in state["why"] if state["state"] == "refused"
    ]
    if "calibration" in state:
        out.append(f"    calibration: {state['calibration']}")
    out += [f"    made by: {command}" for command in state["made_by"]]
    out += [
        f"    resumed, once no calibrate.sh runs, by: {command}"
        for command in state.get("resumed_by", [])
    ]
    for point, held in state["points"].items():
        out += [f"    {point}: {held['state']}, made by: {c}" for c in held["made_by"]]
    return out


def calibrating(out: Path, name: str) -> tuple[Path, Path | None]:
    """Where candidate `name`'s calibration runs, in `out`'s `CAL`: its latest
    directory, ``<name>`` or the last of ``<name>+2``, ``<name>+3``, ... there,
    and None; or, once the calibration there has ended, the next, and that
    one, since `calibrate.sh` records nothing more where one has ended."""
    cal, k = Path(out) / CAL, 2
    while (cal / AGAIN.format(name, k)).is_dir():
        k += 1
    latest = cal / (AGAIN.format(name, k - 1) if k > 2 else name)
    return (cal / AGAIN.format(name, k), latest) if _ended(latest) else (latest, None)


def _ended(directory: Path) -> bool:
    """Whether the calibration in `directory` has ended: its status file's
    last line is the ``end calibration`` a run that fits both laws writes."""
    status = directory / STATUS
    said = status.read_text().splitlines() if status.is_file() else []
    return bool(said) and said[-1].split()[1:3] == ["end", "calibration"]


def _candidate(directory: Path) -> str:
    """The candidate whose calibration runs in `directory`, ``<name>`` or one
    `AGAIN` names after it."""
    return directory.name.partition("+")[0]


def _calibration(at: Path, calibrate: list) -> dict:
    """The state of a candidate whose law verdicts `calibrate` send it to
    calibrate into `at`: ``calibrating`` while the `STATUS` file there ends in
    a step's start or end, with each verdict's command as ``resumed_by``, since
    a kill that records no failure, as a node's job end, leaves the file ending
    there too; else ``missing a law``, with each verdict's reason and command.
    With a status file, either states its last line and how many steps it has
    seen end."""
    status = at / STATUS
    said = status.read_text().splitlines() if status.is_file() else []
    lines = [line.split() for line in said if line]
    if not lines:
        return _state("missing a law", calibrate, points={})
    ended = {step for _, event, step, *_ in lines if event == "end"}
    ended.discard("calibration")
    said = f"{status}, {len(ended)} steps ended, last: {' '.join(lines[-1])}"
    held = [*calibrate, {"reason": said}]
    state = _state("missing a law", held, points={}, calibration=said)
    _, event, step, *_ = lines[-1]
    if event in ("start", "end") and step != "calibration":
        resumed = {"why": [said], "made_by": [], "resumed_by": state["made_by"]}
        state |= {"state": "calibrating"} | resumed
    return state


def _searchable(expansion: dict, candidate: dict, held: dict, out: Path) -> dict:
    """The state of `candidate`, whose every law its verdicts `held` reuse:
    each point whose readings a verdict sends to measure ``missing readings``,
    each other ``searchable`` once its start answers; or the candidate
    ``refused`` with what a start that does not answer says."""
    points, started = {}, {}
    for point in expansion["points"]:
        if point["candidate"] != candidate["name"]:
            continue
        verdicts = held["points"][point["name"]]
        if measure := [
            v for v in verdicts["readings"].values() if v["verdict"] == "measure"
        ]:
            points[point["name"]] = _state("missing readings", measure)
            continue
        alike = json.dumps(point["instances"], sort_keys=True)
        if alike not in started:
            given = expansion["machine"], candidate, point, verdicts, held["store"]
            started[alike] = _start(*given, out / STARTS / point["name"])
        if said := started[alike]:
            return _state("refused", [{"reason": said}], points={})
        points[point["name"]] = _state("searchable")
    return _state("searchable", points=points)


def _start(machine, candidate, point, verdicts, store, at: Path) -> str | None:
    """Start `point` of `candidate` with no traffic, on `machine` from `store`
    under the laws its `verdicts` name, into `at`: None once it answers, else
    why not."""
    try:
        command, env = serving(machine, candidate, point, verdicts, store, at)
    except ValueError as refused:
        return str(refused)
    return _started(command, env, at)


def serving(machine, candidate, point, verdicts, store, at: Path) -> tuple[list, dict]:
    """The command that serves `point` of `candidate` on `machine` from
    `store`, under the laws its `verdicts` name, into `at`, and the variables
    it runs under: `compass run`, or `deploy.sh run` on 1P1D. A point with an
    ``lmcache_port`` takes ``--lmcache``, its entry written beside `at`
    (`_lmcache`); a point of a candidate naming a ``mooncake_store`` entry
    takes ``--mooncake-store`` and that file, as each of its points does, since
    each deployment's Store binds a port of its own. A ValueError says why
    `deploy.sh` cannot start a 1P1D point, or why a point cannot reach its
    lmcache server."""
    flags = ["--device", machine["device"], "--ptl", machine["ptl"], "--store", store]
    flags += ["--stack", machine["stack"]] if machine.get("stack") else []
    for flag, i in (("--step-linear", 0), ("--host-laws", 1)):
        for value in dict.fromkeys(v["run"][i] for v in verdicts["law"].values()):
            flags += [flag, value]
    if "mooncake_store" in candidate:
        flags += ["--mooncake-store", candidate["mooncake_store"]]
    sides = point["instances"]
    if "lmcache_port" in point:
        entry, sides = _lmcache(candidate, point, at)
        flags += ["--lmcache", entry]
    if candidate["deployment"] == "single":
        (one,) = sides.values()
        return [COMPASS, "run", *flags, "--out", at, "--", *one["args"]], one["env"]
    prefill, decode = sides["prefill"]["env"], sides["decode"]["env"]
    words = [w for k in prefill if k not in decode for w in ("-u", k)]
    words += [f"{k}={v}" for k, v in decode.items() if prefill.get(k) != v]
    if spaced := [w for w in words if len(w.split()) != 1]:
        raise ValueError(
            "deploy.sh takes the decode side's own variables as one --on prefix "
            f"split on whitespace, and {', '.join(spaced)} holds whitespace"
        )
    on = []
    if network := machine.get("network"):
        # A local prefix starts the decode apart, so the KV link is inter_node.
        flags, on = flags + ["--network", network], ["--on", " ".join(["env", *words])]
    elif words:
        raise ValueError(
            "deploy.sh starts both sides of a deployment on one node under one "
            f"environment, and the decode side's differs from the prefill side's "
            f"by {' '.join(words)}"
        )
    command = [DEPLOY, at, "run", *flags, "--prefill", "--", *sides["prefill"]["args"]]
    command += ["--decode", *on, "--", *sides["decode"]["args"]]
    return command + ["--router", "--", *ROUTER], prefill


def _lmcache(candidate: dict, point: dict, at: Path) -> tuple[Path, dict]:
    """The run file's ``lmcache`` entry of `point` of `candidate`, written
    beside `at` in ``<at>.lmcache.json``: its lmcache server's flags and the
    request loop's declared laws; and the point's instances, each reaching
    that server (`_reaching`)."""
    server, sides = _reaching(candidate, point)
    entry = at.with_name(at.name + ".lmcache.json")
    entry.parent.mkdir(parents=True, exist_ok=True)
    entry.write_text(json.dumps({"server": server, **DECLARED_LAWS}))
    return entry, sides


def _reaching(candidate: dict, point: dict) -> tuple[list, dict]:
    """The flags of the lmcache server of `point` of `candidate`, at
    `serve.HOST` and the point's port; and the point's instances, each
    offloading through ``lmcache_mp`` reaching that server through `EXTRA`,
    which a ValueError refuses when not a JSON object or naming either, with
    the server's command as ``beside``, which a start of it alone runs beside
    the engine. ATOM refuses `EXTRA` with no offload, so an instance
    offloading through none keeps its variables."""
    port = point["lmcache_port"]
    server = [*candidate["lmcache_server"], "--host", serve.HOST, "--port", str(port)]
    reached = {"lmcache.mp.host": f"tcp://{serve.HOST}", "lmcache.mp.port": port}
    sides = dict(point["instances"])
    for side, one in sides.items():
        if one["env"].get("ATOM_KV_OFFLOAD") != "lmcache_mp":
            continue
        try:
            extra = json.loads(one["env"].get(EXTRA, "{}"))
        except ValueError:
            extra = None
        if not isinstance(extra, dict) or reached.keys() & extra.keys():
            raise ValueError(
                f"{EXTRA} is {one['env'][EXTRA]}: compass advice takes a JSON "
                f"object naming neither {' nor '.join(reached)}, which it assigns"
            )
        env = one["env"] | {EXTRA: json.dumps(extra | reached)}
        sides[side] = one | {"env": env, "beside": ["lmcache", "server", *server]}
    return server, sides


def _reached(candidate: dict, points: list) -> list:
    """`points`, `candidate`'s, each instance offloading to an lmcache server
    reaching one as `_reaching` gives it, at the port of the first of `points`
    that runs alike, as alike points start once (`_searchable`), so one
    readings command measures them all; or, where `_reaching` refuses a point,
    each of its instances holding the verdict refusing it as ``refused``."""
    first, reached = {}, []
    for point in points:
        alike = json.dumps(point["instances"], sort_keys=True)
        if "lmcache_port" in point and alike not in first:
            try:
                first[alike] = _reaching(candidate, point)[1]
            except ValueError as error:
                why = {"refused": _refused(error)}
                first[alike] = {s: one | why for s, one in point["instances"].items()}
        reached.append(point | {"instances": first.get(alike, point["instances"])})
    return reached


def launched(command: list, env: dict, at: Path, until) -> "Command":
    """`command` under `env`, run into `at` afresh, its output beside it in
    ``<at>.log``, and waited for until it leaves, the ``server.json`` it
    writes in `at` names its server, or `until()` holds, at most `START_S`."""
    shutil.rmtree(at, ignore_errors=True)
    at.parent.mkdir(parents=True, exist_ok=True)
    server = Command(command, env, at.with_name(at.name + ".log"))
    ready = at / serve.READY_FILE
    server.wait(int(START_S / serve.POLL_S), lambda: ready.is_file() or until())
    return server


def _started(command: list, env: dict, at: Path) -> str | None:
    """Run `command` under `env` until the ``server.json`` it writes in `at`
    names its server, then stop it with SIGTERM, as each is stopped: None.
    When it leaves first or does not answer in `START_S`, its last line and
    the first error raised in each log under `at`."""
    server = launched(command, env, at, lambda: False)
    ready = at / serve.READY_FILE
    answered, left = ready.is_file(), server.left()
    server.end()
    if answered:
        return None
    said = [] if left else [f"no {ready} in {START_S} s"]
    said += [_said(server.log, last=True)]
    said += [_said(p) for p in sorted(at.rglob("*.log"))]
    return "; ".join(dict.fromkeys(filter(None, said)))


class Command:
    """A command in a process group of its own, run in the caller's directory
    under this process's environment and `env`, its output in `log`; once
    reaped, its exit ``status`` and the CPU seconds it and the processes it
    waited for took."""

    def __init__(self, command: list, env: dict, log: Path):
        self.log = log
        with log.open("w") as written:
            self.process = subprocess.Popen(
                list(map(str, command)),
                stdout=written,
                stderr=subprocess.STDOUT,
                env=os.environ | env,
                start_new_session=True,
            )
        self.status, self.cpu_s = None, 0.0

    def left(self) -> bool:
        """Whether it has exited, reaped here for its status and CPU seconds."""
        if self.status is None:
            pid, status, usage = os.wait4(self.process.pid, os.WNOHANG)
            if pid:
                self.status = os.waitstatus_to_exitcode(status)
                self.process.returncode = self.status
                self.cpu_s = usage.ru_utime + usage.ru_stime
        return self.status is not None

    def wait(self, polls: int | None, until) -> None:
        """Until it exits or `until()` holds, for at most `polls` polls, or
        with no bound when None."""
        for _ in itertools.count() if polls is None else range(polls):
            if self.left() or until():
                return
            time.sleep(serve.POLL_S)

    def end(self) -> None:
        """Stop it with SIGTERM; SIGKILL its group after `START_S`; and reap
        it."""
        # Not Popen's own signalling: its poll would reap the process first.
        if not self.left():
            os.kill(self.process.pid, signal.SIGTERM)
            self.wait(int(START_S / serve.POLL_S), lambda: False)
        if not self.left():
            os.killpg(self.process.pid, signal.SIGKILL)
            self.wait(None, lambda: False)


def _said(log: Path, last: bool = False) -> str | None:
    """The exception line of the first traceback in `log`, else an argument
    parser's error line, else, with `last`, its last line."""
    lines = [line for line in log.read_text("utf-8", "replace").splitlines() if line]
    for i, line in enumerate(lines):
        if line.startswith("Traceback (most recent call last):"):
            return next((a for a in lines[i + 1 :] if not a[0].isspace()), None)
    if error := next((line for line in lines if ": error: " in line), None):
        return error
    return lines[-1] if last and lines else None


def _state(state: str, verdicts=(), **more) -> dict:
    """`state`, with each distinct reason and command of `verdicts`."""
    return {
        "state": state,
        "why": list(dict.fromkeys(v["reason"] for v in verdicts)),
        "made_by": list(
            dict.fromkeys(c for v in verdicts for c in v.get("made_by", []))
        ),
        **more,
    }


def _point(point: dict, judge) -> dict:
    """`point`'s ``law`` and ``readings`` verdicts per instance, as `judge`
    gives them for each, given its side."""
    verdicts = {"law": {}, "readings": {}}
    for side, one in point["instances"].items():
        verdicts["law"][side], verdicts["readings"][side] = judge(side, one)
    return verdicts


def _judged(store, machine, atom, seen: dict, made, side: str, one: dict) -> tuple:
    """The instance `one`'s law and readings verdicts in `store` on `machine`,
    a spec or the verdict its refusal gives, at the head's ATOM digest `atom`:
    each instance's config resolved once, under its variables, and judged once
    a store, both kept in `seen`. A 1P1D `side` takes the KV transfer config
    `deploy.sh` adds, on the simulated connector, and runs under its variables
    over this process's ``AITER_LOG_LEVEL``, with those the script sets over
    them (`drive_run.side_env`). `made` is the calibration's directory, traces
    and the shared store a ``calibrate`` verdict's command takes, and the
    directory of one that ended before it, or None, whose law a command
    refitting it (`_refits`) gives way to `REFITS`; a
    `calibrate.sh` command's verdict states, for `_borrowing`, its
    ``calibration``: the ``key`` its law is published at, by model, platform
    and engine config, its directory, ``in``, and the store it publishes
    ``into``; one giving way to `REFITS`, the ended one's directory and the
    store its law was published into, and that it ``refits``. An instance
    holding a verdict as ``refused`` takes it for both; one holding a command
    as ``beside`` gives it its readings' ``measuring``, to run beside the
    engine."""
    if isinstance(machine, dict):
        return machine, machine
    if "refused" in one:
        return one["refused"], one["refused"]
    args, env = one["args"], one["env"]
    if side in drive_run.ROLES:
        args = _or_refused(drive_run.side_args, side, args, drive_run.SIMULATED)
        if isinstance(args, dict):
            return args, args
        env = {"AITER_LOG_LEVEL": os.environ.get("AITER_LOG_LEVEL", "")} | env
        env |= drive_run.side_env(env)
    instance = (tuple(args), tuple(env.items()))
    if (store.root, *instance) not in seen:
        with _env(env):
            if instance not in seen:
                arch = machine.value("device.arch")
                seen[instance] = _or_refused(_resolved, args, arch)
            if isinstance(config := seen[instance], dict):
                verdicts = config, config
            else:
                law = _or_refused(reuse.law, store, config, machine, args, atom)
                readings = _or_refused(reuse.readings, store, config, machine, law)
                if law["verdict"] == "calibrate" and side in drive_run.ROLES:
                    fits = lacking.FITS.values()
                    law["made_by"] = [
                        lacking.SIDE_LAW.format(fit=f, side=side) for f in fits
                    ]
                elif law["verdict"] == "calibrate":
                    into = _publishing(store, made, law, readings)
                    with _env({"TRACES": made[1]}):
                        command = lacking.calibration(made[0], args, config, into)
                        law["made_by"] = [command]
                    key = [config.model, machine_platform(machine), keyed(config)]
                    into = str(into or made[0] / "store")
                    law["calibration"] = {"key": key, "in": str(made[0]), "into": into}
                    if (ended := made[3]) and _refits(ended, command, key, atom):
                        held = started_store(ended) or str(ended / "store")
                        law["made_by"] = [REFITS.format(ended, held)]
                        refitting = {"in": str(ended), "into": held, "refits": True}
                        law["calibration"] |= refitting
                if readings["verdict"] != "refused":
                    real = args
                    if side in drive_run.ROLES:
                        real = drive_run.side_args(side, one["args"], drive_run.REAL)
                    readings["measuring"] = (
                        memory.key_of(config, machine),
                        real,
                        env,
                        one.get("beside"),
                    )
                verdicts = law, readings
        seen[(store.root, *instance)] = verdicts
    return seen[(store.root, *instance)]


def _borrowing(points: list, held: dict, calibrations: dict) -> None:
    """Take each ``calibration`` `_judged` states out of the law verdicts in
    `held`, a candidate's at its `points`, and name, in place of a verdict's own
    `calibrate.sh` command, the calibration an earlier candidate's in
    `calibrations` makes at its key, since the store refuses a second law at a
    key: such a verdict is made by that calibration, or, where a new one would
    refit the law that one fitted, by why none serves (`REFITS`), and by what
    measures this candidate's readings into the store it publishes into
    (`_readings_by`), unless they are at their key there. Each other
    calibration joins `calibrations`, unless a verdict of the candidate is
    refused, whose state prints no command."""
    refused = any(v["verdict"] == "refused" for v in _each(held["points"]))
    for point in points:
        verdicts = held["points"][point["name"]]
        for side, law in verdicts["law"].items():
            law = dict(law)
            if (mine := law.pop("calibration", None)) is None:
                continue
            key = json.dumps(mine["key"], sort_keys=True)
            made = calibrations.get(key, mine)
            if made["in"] == mine["in"] and not refused:
                calibrations[key] = mine
            elif made["in"] != mine["in"]:
                lender, into = Path(made["in"]), made["into"]
                if made.get("refits"):
                    law["made_by"] = [REFITS.format(lender, into)]
                else:
                    law["made_by"] = [BORROWED.format(_candidate(lender), lender, into)]
                there = Path(into) == Path(held["store"])
                if verdicts["readings"][side]["verdict"] != "reuse" or not there:
                    one = point["instances"][side]
                    law["made_by"].append(_readings_by(lender, into, one))
            verdicts["law"][side] = law


def _measuring(held: dict, new: Path) -> None:
    """Name the command of each readings verdict in `held`, a candidate's in
    its store, that sends them to measure: into that store, or, once one sends
    readings whose key an entry there holds, which it refuses a second publish
    at, into `new` for every point, those at their key there too (`MOVED`), as
    a run reads one store. Each verdict sheds ``measuring``, its key, real ATOM
    arguments and variables, and the command run beside the engine, or None,
    which `_judged` gives it."""
    taken = []
    for point in held["points"].values():
        for side, readings in point["readings"].items():
            readings = point["readings"][side] = dict(readings)
            if (measuring := readings.pop("measuring", None)) is not None:
                taken.append((readings, *measuring))
    moved = any(v["verdict"] == "measure" and v["held"] for v, *_ in taken)
    into = new if moved else held["store"]
    for readings, key, real, env, beside in taken:
        reused = readings["verdict"] == "reuse"
        if moved and reused and lacking.holds(held["store"], key):
            why = MOVED.format(held["store"], into)
            readings |= {"verdict": "measure", "reason": why}
        if readings["verdict"] == "measure":
            readings["made_by"] = [lacking.measuring(into, key, real, env, beside)]


def _readings_by(lender: Path, into: str, one: dict) -> str:
    """The command measuring the instance `one`'s readings into `into`, the
    store the calibration in `lender` publishes into, with the command `one`
    holds as ``beside``, if any, beside the engine; while that is its own
    store and it has not started, `PENDING`: a publish there first leaves the
    calibration's directory holding files, which `calibrate.sh` refuses."""
    if Path(into) == lender / "store" and not (lender / INPUTS).is_file():
        return PENDING.format(into, _candidate(lender), lender)
    return lacking.measure_memory(into, one["args"], one["env"], one.get("beside"))


def _publishing(store, made, law: dict, readings: dict) -> str | None:
    """The store the calibration into `made`'s directory, whose verdicts `law`
    and `readings` were judged in `store`, publishes into, None its own. Once
    one has started there, the store its `INPUTS` name, if any: a resume must
    take the store it started with, though the readings or a law it published
    there now hold their keys. Before then, `made`'s shared store when it is
    `store` and holds neither a step linear law at the head's ATOM digest nor
    readings at their key, since a store refuses a second publish at a key."""
    directory, _, shared, _ = made
    if (directory / INPUTS).is_file():
        return started_store(directory)
    held = "store" in law or readings["verdict"] == "reuse" or readings.get("held")
    return shared if store.root == Path(shared) and not held else None


def started_store(directory: Path) -> str | None:
    """The store the `INPUTS` of the calibration started in `directory` name,
    None when they name none or no calibration started there."""
    inputs = directory / INPUTS
    given = inputs.read_text().splitlines() if inputs.is_file() else []
    return next((s[len("store ") :] for s in given if s.startswith("store ")), None)


def _refits(ended: Path, command: str, key: list, atom: str) -> bool:
    """Whether the `calibrate.sh` `command` refits the law the calibration
    that ended in `ended` fitted: it takes what that one took, as its `INPUTS`
    state it (ATOM's arguments, the traces and the records), and that law is
    at `key`, where the command's law publishes, at the head's ATOM digest
    `atom`, as `reuse.law` matches a store law: `key`'s model, the platform of
    the machine, and the engine config as `keyed` states it under the
    instance's environment, whose performance variables the command sets. No
    command, as `lacking.calibration` words why there is none, takes nothing;
    a directory holding no law refits none."""
    if str(lacking.CALIBRATE) not in (line := command.split("\n")[0]):
        return False
    words = shlex.split(line)
    at = words.index(str(lacking.CALIBRATE))
    env = dict(word.split("=", 1) for word in words[:at])
    taken = [f"atom --model {env['MODEL']} {env['SERVER_ARGS']}"]
    taken += [f"traces {env.get('TRACES')}"]
    taken += [f"record {record}" for record in words[at + 2 :]]
    inputs = ended / INPUTS
    given = inputs.read_text().splitlines() if inputs.is_file() else []
    # calibrate.sh writes `traces TRACES CAL_TRACES`.
    took = [s.rsplit(" ", 1)[0] if s.startswith("traces ") else s for s in given]
    took = [s for s in took if s.startswith(("atom ", "traces ", "record "))]
    if took != taken or not (ended / FITTED).is_file():
        return False
    model, platform, engine = key
    model = compass_run._file_digest(model, "config.json")
    held = {"model": model, "source_root": compass_run.source_root(atom)}
    held |= keyed_platform(Kind.STEP_LINEAR_LAW, platform)
    fitted = json.loads((ended / FITTED).read_text())["key"]
    return reuse.at_key(fitted, engine, **held)


def _resolved(args, arch: str):
    """ATOM's config of its arguments `args` on a device of `arch`, resolved
    as `drive_run._config` resolves it, in a child that hands back what it
    raises as text (`_resolving`), once `runfile._engine_args` reads them
    here: a value ATOM's parser rejects raises its error, where the child
    would exit. A refusal raises as a value error; Compass's own defect as a
    runtime error holding the child's traceback, which no verdict records."""
    from transformers.dynamic_module_utils import init_hf_modules

    from atom.compass.bench import ops
    from atom.utils.arg_parser import FlexibleArgumentParser

    class Refusing(FlexibleArgumentParser):
        def error(self, message):
            raise ValueError(f"ATOM's parser: {message}")

    runfile._engine_args(args, Refusing)
    # As `drive_run._config`: the child writes a model's remote code to
    # transformers' module cache, whose config classes then load here.
    init_hf_modules()
    config, raised, said = ops._child(_resolving, drive_run._declared, args, arch)
    if raised == "defect":
        raise RuntimeError(f"Compass's own code raised resolving {args}:\n{said}")
    if raised:
        raise ValueError(said)
    return config


def _resolving(declared, args, arch: str, of: str = "config") -> tuple:
    """In the child: `declared(args, arch)`, else what it raised as text, which
    always unpickles: ``refused`` with its exception line, ATOM's, or with its
    message, a refusal Compass raises; ``defect`` with its traceback, anything
    else raised in Compass's own code, its innermost frame under `OWN`."""
    try:
        return declared(args, arch), None, None
    except BaseException as error:  # noqa: BLE001 - each handed back as text
        *_, inner = traceback.extract_tb(error.__traceback__)
        if not Path(inner.filename).resolve().is_relative_to(OWN):
            said = traceback.format_exception_only(error)[-1].strip()
            return None, "refused", f"ATOM's {of}: {said}"
        if isinstance(error, REFUSALS):
            return None, "refused", str(error)
        return None, "defect", "".join(traceback.format_exception(error))


def _or_refused(call, *args):
    """`call(*args)`, or the verdict its refusal gives."""
    try:
        return call(*args)
    except REFUSALS as error:
        return _refused(error)


def _each(verdicts: dict):
    """Each verdict in `verdicts`, of each point, input and instance."""
    for point in verdicts.values():
        for of in point.values():
            yield from of.values()


def _refused(error: Exception, requirement: int | None = None) -> dict:
    return {"verdict": "refused", "requirement": requirement, "reason": str(error)}


@contextlib.contextmanager
def _env(env: dict):
    """This process's environment with `env` set, as the instance's server
    runs under; restored after."""
    saved = dict(os.environ)
    os.environ.update(env)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)
