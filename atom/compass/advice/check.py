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
takes the first whose platform readings compose and where a law is refused, as
one fitted on another host CPU is, so the refusal is named rather than the
shared store's calibrate; else the shared store's. There, each instance of
each point is resolved as ATOM's server resolves its arguments, under its
variables, on the device arch of the machine spec that store's platform
readings compose with SPACE's machine; a 1P1D side as `deploy.sh run` starts
it, as `compass run --check` resolves one. The laws are judged at each point,
since an argument that varies with the concurrency moves their key. A refusal
of the composition, of a configuration or of the runner under a law is
``refused``, verbatim, naming no requirement. A law verdict ``calibrate``
names the command that makes the law as `compass run --check` prints it: on
one node the `calibrate.sh` command of its records into OUT's
``cal/<candidate>``, its clients on SPACE's traces; on a 1P1D side `deploy.sh
record` and a fit of that side. A readings verdict ``measure`` names the
`compass measure memory` command, under the instance's variables, a 1P1D
side's on the connector a real start of it takes.

`states` gives each candidate one state: ``refused``, with each refusal
verbatim, when any verdict is; else ``missing a law``, with each command,
when a law's verdict is ``calibrate``; else ``searchable``. A searchable
candidate's point whose readings a verdict sends to measure is ``missing
readings``, with its command. Each other point starts with no traffic, as the
search will run it, for what only a start shows: `compass run`, or `deploy.sh
run` on 1P1D with no router arguments, its decode side through a local
``--on`` prefix carrying its own variables where SPACE names a network. A
point answers once the start's ``server.json`` names it, and is
``searchable``; a start that leaves first, or does not answer in `START_S`,
refuses the candidate with the start's last line and each log's first raised
error. Points whose instances run alike start once. Nothing is calibrated or
measured.
"""

import contextlib
import functools
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

from atom.compass.artifacts import ArtifactRefusal
from atom.compass.drive import lacking, reuse, serve
from atom.compass.drive import run as drive_run
from atom.compass.fit.device import compose
from atom.compass.spec import SpecRefusal

#: The verdicts, in OUT.
REUSE = "reuse.json"
#: What a refusal of an input raises, recorded as its verdict.
REFUSALS = (ArtifactRefusal, SpecRefusal, ValueError, OSError)
#: Each candidate's state, in OUT.
STATES = "states.json"
#: Where each candidate's calibration writes, and each point starts, in OUT.
CAL, STARTS = "cal", "starts"
#: What a start runs: a server on one node, a deployment on 1P1D.
COMPASS, DEPLOY = (
    serve.TREE / "scripts" / "compass" / n for n in ("compass", "deploy.sh")
)
#: Seconds a start has to answer, and its stop to leave: deploy.sh's default
#: wait for each of its instances and its router.
START_S = 3 * 600


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
    judged, seen, machine = {}, {}, expansion["machine"]
    for candidate in expansion["candidates"]:
        points = [p for p in expansion["points"] if p["candidate"] == candidate["name"]]
        if refused:
            verdicts = {
                p["name"]: _point(p, lambda *_: (refused, refused)) for p in points
            }
            judged[candidate["name"]] = {"store": None, "points": verdicts}
            continue
        refusing = None
        for store in reuse.stores(expansion["store"], out):
            given = (machine["device"], store, machine["ptl"])
            given += (machine.get("stack"), machine.get("network"))
            spec = _or_refused(compose, *given)
            made = out / CAL / candidate["name"], expansion["workload"]["traces"]
            judge = functools.partial(_judged, store, spec, head["atom"], seen, made)
            verdicts = {p["name"]: _point(p, judge) for p in points}
            if isinstance(spec, dict):
                continue
            if all(v["requirement"] != 1 for v in _each(verdicts)):
                break
            laws = (v for point in verdicts.values() for v in point["law"].values())
            if refusing is None and any(v["verdict"] == "refused" for v in laws):
                refusing = store, verdicts
        else:
            store, verdicts = refusing or (store, verdicts)
        judged[candidate["name"]] = {"store": str(store.root), "points": verdicts}
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
            state = _state("missing a law", calibrate, points={})
        else:
            state = _searchable(expansion, candidate, held, out)
        found[candidate["name"]] = state
    (out / STATES).write_text(json.dumps(found, indent=1) + "\n")
    return found


def state_lines(state: dict) -> list[str]:
    """A candidate's `states` entry as lines: its state, then each reason and
    command, then each point missing readings with its command."""
    searchable = [p for p, v in state["points"].items() if v["state"] == "searchable"]
    out = [f"  state: {state['state']}"]
    if state["points"]:
        out[0] += f", {len(searchable)} of {len(state['points'])} points searchable"
    out += [
        f"    refused: {why}" for why in state["why"] if state["state"] == "refused"
    ]
    out += [f"    made by: {command}" for command in state["made_by"]]
    for point, held in state["points"].items():
        out += [f"    {point}: {held['state']}, made by: {c}" for c in held["made_by"]]
    return out


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
    it runs under: `compass run`, or `deploy.sh run` on 1P1D. A ValueError
    says why `deploy.sh` cannot start a 1P1D point."""
    flags = ["--device", machine["device"], "--ptl", machine["ptl"], "--store", store]
    flags += ["--stack", machine["stack"]] if machine.get("stack") else []
    for flag, i in (("--step-linear", 0), ("--host-laws", 1)):
        for value in dict.fromkeys(v["run"][i] for v in verdicts["law"].values()):
            flags += [flag, value]
    sides = point["instances"]
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
    command += ["--decode", *on, "--", *sides["decode"]["args"], "--router", "--"]
    return command, prefill


def _started(command: list, env: dict, at: Path) -> str | None:
    """Run `command` under `env` until the ``server.json`` it writes in `at`
    names its server, then stop it with SIGTERM, as each is stopped: None.
    When it leaves first or does not answer in `START_S`, its last line and
    the first error raised in each log under `at`."""
    shutil.rmtree(at, ignore_errors=True)
    at.parent.mkdir(parents=True, exist_ok=True)
    log = at.with_name(at.name + ".log")
    with log.open("w") as written:
        server = subprocess.Popen(
            list(map(str, command)),
            stdout=written,
            stderr=subprocess.STDOUT,
            env=os.environ | env,
            cwd=serve.TREE,
            start_new_session=True,
        )
    ready = at / serve.READY_FILE
    for _ in range(int(START_S / serve.POLL_S)):
        if server.poll() is not None or ready.is_file():
            break
        time.sleep(serve.POLL_S)
    answered, left = ready.is_file(), server.poll() is not None
    server.terminate()
    try:
        server.wait(START_S)
    except subprocess.TimeoutExpired:
        os.killpg(server.pid, signal.SIGKILL)
        server.wait()
    if answered:
        return None
    said = [] if left else [f"no {ready} in {START_S} s"]
    said += [_said(log, last=True), *(_said(p) for p in sorted(at.rglob("*.log")))]
    return "; ".join(dict.fromkeys(filter(None, said)))


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
    `deploy.sh` adds, on the simulated connector, and runs under the
    ``AITER_LOG_LEVEL`` the script's `compass_env` sets when unset. `made` is
    the calibration's directory and traces a ``calibrate`` verdict's command
    takes."""
    if isinstance(machine, dict):
        return machine, machine
    args, env = one["args"], one["env"]
    if side in drive_run.ROLES:
        args = _or_refused(drive_run.side_args, side, args, drive_run.SIMULATED)
        if isinstance(args, dict):
            return args, args
        env = {"AITER_LOG_LEVEL": os.environ.get("AITER_LOG_LEVEL", "WARNING")} | env
    instance = (tuple(args), tuple(env.items()))
    if (store.root, *instance) not in seen:
        with _env(env):
            if instance not in seen:
                arch = machine.value("device.arch")
                seen[instance] = _or_refused(drive_run._config, args, arch)
            if isinstance(config := seen[instance], dict):
                verdicts = config, config
            else:
                law = _or_refused(reuse.law, store, config, machine, args, atom)
                readings = _or_refused(reuse.readings, store, config, machine)
                if law["verdict"] == "calibrate" and side in drive_run.ROLES:
                    fits = lacking.FITS.values()
                    law["made_by"] = [
                        lacking.SIDE_LAW.format(fit=f, side=side) for f in fits
                    ]
                elif law["verdict"] == "calibrate":
                    with _env({"TRACES": made[1]}):
                        law["made_by"] = [lacking.calibration(made[0], args, config)]
                if readings["verdict"] == "measure":
                    real = args
                    if side in drive_run.ROLES:
                        real = drive_run.side_args(side, one["args"], drive_run.REAL)
                    readings["made_by"] = [
                        lacking.measure_memory(store.root, real, env)
                    ]
                verdicts = law, readings
        seen[(store.root, *instance)] = verdicts
    return seen[(store.root, *instance)]


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
