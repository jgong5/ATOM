# SPDX-License-Identifier: MIT
"""The check `compass advice` makes before a search simulates: the reuse
check's verdicts (`drive.reuse`) on each point's laws and memory readings,
written into OUT as `REUSE`.

The head is taken once, at the commit SPACE freezes or the integration branch's
tip; a tree that is not the head gives every verdict ``refused``, requirement
1, with the reason. A candidate is looked up in each of `reuse.stores` in turn
and takes the first that holds every input of its points, whatever the verdicts
on them there, else the shared store's: its platform readings compose with
SPACE's machine, and no verdict names requirement 1, so each point's readings,
and each law that is not a checked-in file, are at their key at the head's ATOM
digest. There, each instance of each point is resolved as ATOM's server
resolves its arguments, under its variables, on the device arch of the machine
spec that store's platform readings compose with SPACE's machine; a 1P1D side
as `deploy.sh run` starts it, as `compass run --check` resolves one. The laws
are judged at each point, since an argument that varies with the concurrency
moves their key. A refusal of the composition, of a configuration or of the
runner under a law is ``refused``, verbatim, naming no requirement.
"""

import contextlib
import functools
import json
import os
from pathlib import Path

from atom.compass.artifacts import ArtifactRefusal
from atom.compass.drive import reuse
from atom.compass.drive import run as drive_run
from atom.compass.fit.device import compose
from atom.compass.spec import SpecRefusal

#: The verdicts, in OUT.
REUSE = "reuse.json"
#: What a refusal of an input raises, recorded as its verdict.
REFUSALS = (ArtifactRefusal, SpecRefusal, ValueError, OSError)


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
        for store in reuse.stores(expansion["store"], out):
            given = (machine["device"], store, machine["ptl"])
            given += (machine.get("stack"), machine.get("network"))
            spec = _or_refused(compose, *given)
            judge = functools.partial(_judged, store, spec, head["atom"], seen)
            verdicts = {p["name"]: _point(p, judge) for p in points}
            if not isinstance(spec, dict) and all(
                v["requirement"] != 1 for v in _each(verdicts)
            ):
                break
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


def _point(point: dict, judge) -> dict:
    """`point`'s ``law`` and ``readings`` verdicts per instance, as `judge`
    gives them for each, given its side."""
    verdicts = {"law": {}, "readings": {}}
    for side, one in point["instances"].items():
        verdicts["law"][side], verdicts["readings"][side] = judge(side, one)
    return verdicts


def _judged(store, machine, atom: str, seen: dict, side: str, one: dict) -> tuple:
    """The instance `one`'s law and readings verdicts in `store` on `machine`,
    a spec or the verdict its refusal gives, at the head's ATOM digest `atom`:
    each instance's config resolved once, under its variables, and judged once
    a store, both kept in `seen`. A 1P1D `side` takes the KV transfer config
    `deploy.sh` adds, on the simulated connector, and runs under the
    ``AITER_LOG_LEVEL`` the script's `compass_env` sets when unset."""
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
                verdicts = (
                    _or_refused(reuse.law, store, config, machine, args, atom),
                    _or_refused(reuse.readings, store, config, machine),
                )
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
