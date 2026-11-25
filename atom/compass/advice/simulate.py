# SPDX-License-Identifier: MIT
"""The simulation `compass advice` makes of each point its check finds
searchable: a fresh server per point, driven by one `client.sh` run.

Each point runs into OUT's ``runs/<point>``: its server is the command the
check's start runs (`check.serving`), `compass run`, or `deploy.sh run` on
1P1D, from the store and under the laws its verdicts name; once the server
names itself in ``server.json``, `client.sh` runs once against it on SPACE's
traces, selection and duration at the point's concurrency, under ``MODEL``
and ``TRACES``, with ``--stop``, and the server is waited for. The point is
finished once both exit 0 and its directory holds the run summary, the run
file and the client export (`HELD`); then ``point.json`` there records what
ran, the CPU seconds the two commands and the processes they waited for took,
and the tries. A rerun skips a point whose ``point.json`` records what it
would run, on the tree that runs it, by the ``HEAD^{tree}`` `git_tree_root`
reads, with no uncommitted file then or now; any other directory there is
removed first. A point that fails is tried again from a fresh directory, then
listed with each try's reason: the client's last line, the server command's,
and the first error raised in each log under the point's directory. Each
command's output is beside the directory, in ``<point>.log`` and
``<point>.client.log``, and an offloading point's lmcache entry in
``<point>.lmcache.json``. With ``HARNESS_PYTHON``, the Python `client.sh` runs
aiperf under, unset, no point runs, and each is listed with why.

Points start in SPACE's order, as many at once as this process's CPUs hold
the widest point's ranks, each rank one CPU; each server draws its own ports,
and an offloading point's lmcache server takes the point's own.
SIGINT or SIGTERM stops every point's commands, each as the check stops a
start, and the search exits naming the signal; a finished point keeps its
record. Each point's outcome is written into OUT as `SIMULATED`, with the
host's CPU and the points run at once.
"""

import concurrent.futures
import json
import os
import threading
from pathlib import Path

from atom.compass import run as compass_run
from atom.compass.advice import check
from atom.compass.artifacts.provenance import git_tree_root
from atom.compass.drive import serve

#: Each point's run directory, under OUT; and the record of what ran in it.
RUNS, RAN = "runs", "point.json"
#: Each point's outcome, in OUT.
SIMULATED = "simulated.json"
#: What drives each point's server.
CLIENT = serve.TREE / "scripts" / "compass" / "client.sh"
#: The tries a point gets.
TRIES = 2
#: Why no point runs with ``HARNESS_PYTHON`` unset.
UNSET = (
    "HARNESS_PYTHON is unset, and client.sh drives each simulated server from it: "
    "the Python with agentx-harness and compass-harness"
)
#: What a finished point's directory holds: the run summary, the run file and
#: the client export.
HELD = (compass_run.SUMMARY_FILE, serve.RUN_FILE, "artifacts/client_run.json")


def simulate(expansion: dict, judged: dict, found: dict, out: Path) -> dict:
    """Each point the `check.states` entries `found` call searchable, run as
    the `check.check` verdicts `judged` serve it, into `out`; the outcomes,
    written into `out` as `SIMULATED`."""
    candidates = {c["name"]: c for c in expansion["candidates"]}
    plans, tree = [], None
    for point in expansion["points"]:
        state = found[point["candidate"]]["points"].get(point["name"], {})
        if state.get("state") == "searchable":
            # Read once a point runs: a tree the check refused may be unreadable.
            tree = tree or git_tree_root("atom", str(serve.TREE))
            candidate = candidates[point["candidate"]]
            plans.append(_plan(expansion, judged, candidate, point, tree, out))
    cpus = len(os.sched_getaffinity(0))
    at_once = max(1, cpus // max((plan["ranks"] for plan in plans), default=1))
    stopping = threading.Event()
    unset = None if os.environ.get("HARNESS_PYTHON") else UNSET
    with (
        serve._stop_signals() as got,
        concurrent.futures.ThreadPoolExecutor(at_once) as pool,
    ):
        running = [pool.submit(_simulated, plan, stopping, unset) for plan in plans]
        while not all(f.done() for f in running):
            if got:
                stopping.set()
            concurrent.futures.wait(running, timeout=serve.POLL_S)
    if got:
        raise SystemExit(
            f"compass advice: stopped by signal {got[0]}; each finished point keeps "
            "its record, and a rerun skips it"
        )
    outcome = {
        "host_cpu": compass_run.host_cpu(),
        "cpus": cpus,
        "at_once": at_once,
        "points": {p["name"]: f.result() for p, f in zip(plans, running)},
    }
    (out / SIMULATED).write_text(json.dumps(outcome, indent=1) + "\n")
    return outcome


def lines(outcome: dict) -> list[str]:
    """`simulate`'s outcome as lines: each point's, then the CPU seconds the
    finished points took and the machine they took them on."""
    out = []
    for name, held in outcome["points"].items():
        if held["state"] == "failed":
            tries = (
                f"failed after {held['tries']} tries" if held["tries"] else "not run"
            )
            out.append(f"  {name}: {tries}")
            out += [f"    {why}" for why in held["why"]]
            continue
        earlier = ", finished earlier" if held["skipped"] else ""
        cpu_s = f"{held['cpu_s']:.0f} CPU s"
        out.append(f"  {name}: finished in {held['run']}, {cpu_s}{earlier}")
    finished = [p for p in outcome["points"].values() if p["state"] == "finished"]
    out.append(
        f"{len(finished)} of {len(outcome['points'])} points finished, "
        f"{sum(p['cpu_s'] for p in finished):.0f} CPU s, {outcome['at_once']} at "
        f"once on {outcome['cpus']} CPUs of {outcome['host_cpu']}"
    )
    return out


def _plan(expansion: dict, judged: dict, candidate: dict, point: dict, tree, out: Path):
    """What `point` of `candidate` runs into `out`'s `RUNS`: its ``at``; as
    ``ran``, the revision of `tree`, the tree that runs it, and its count of
    uncommitted files, and its server's and its client's commands and
    variables; and its ``ranks``, TP times DP over its instances."""
    held = judged["candidates"][candidate["name"]]
    at, verdicts = out / RUNS / point["name"], held["points"][point["name"]]
    machine, store = expansion["machine"], held["store"]
    server, env = check.serving(machine, candidate, point, verdicts, store, at)
    workload, picked = expansion["workload"], expansion["workload"]["select"]
    select = ["--agentx"] if picked == "agentx" else ["--traces", picked]
    client = [CLIENT, at, *select, "--concurrency", point["concurrency"]]
    client += ["--duration", workload["duration_s"]] if "duration_s" in workload else []
    ran = {
        "tree": {"revision": tree.revision, "dirty": tree.dirty},
        "server": list(map(str, server)),
        "env": env,
        "client": list(map(str, [*client, "--stop"])),
        "client_env": {"MODEL": expansion["model"], "TRACES": workload["traces"]},
    }
    ranks = 0
    for one in point["instances"].values():
        stated = serve.widths(one["args"])
        ranks += stated.tensor_parallel_size * stated.data_parallel_size
    return {"name": point["name"], "at": at, "ran": ran, "ranks": ranks}


def _record(plan: dict) -> dict | None:
    """The record of `plan`'s point, when it finished running what `plan` runs,
    on a tree with no uncommitted file."""
    if plan["ran"]["tree"]["dirty"] != 0:
        return None
    try:
        held = json.loads((plan["at"] / RAN).read_text())
    except (OSError, ValueError):
        return None
    return held if held.get("ran") == plan["ran"] else None


def _simulated(plan: dict, stopping: threading.Event, unset: str | None) -> dict:
    """`plan`'s point, skipped when it finished earlier, else, unless `unset`
    says why no client can run, tried until it finishes, up to `TRIES` times:
    its outcome."""
    if held := _record(plan):
        kept = {k: held[k] for k in ("state", "run", "cpu_s", "tries")}
        return kept | {"skipped": True}
    if unset:
        return {"state": "failed", "why": [unset], "tries": 0}
    why = []
    for tries in range(1, TRIES + 1):
        if stopping.is_set():
            break
        said, cpu_s = _try(plan, stopping)
        if said is None:
            done = {"state": "finished", "run": str(plan["at"]), "cpu_s": cpu_s}
            done["tries"] = tries
            (plan["at"] / RAN).write_text(json.dumps({"ran": plan["ran"], **done}))
            return done | {"skipped": False}
        why.append(said)
    return {"state": "failed", "why": list(dict.fromkeys(why)), "tries": len(why)}


def _try(plan: dict, stopping: threading.Event) -> tuple[str | None, float]:
    """One try of `plan`'s point from a fresh directory: None once it
    finished, else why not; and the CPU seconds its commands took."""
    at, ran = plan["at"], plan["ran"]
    server = check.launched(ran["server"], ran["env"], at, stopping.is_set)
    ready, client_log = at / serve.READY_FILE, at.with_name(at.name + ".client.log")
    said, cpu_s = [], 0.0
    served = ready.is_file() and not stopping.is_set()
    if served:
        client = check.Command(ran["client"], ran["client_env"], client_log)
        client.wait(None, stopping.is_set)
        client.end()
        cpu_s += client.cpu_s
        if client.status:
            exited = f"client.sh exited {client.status}"
            said.append(check._said(client_log, last=True) or exited)
    elif not ready.is_file() and not server.left():
        said.append(f"no {ready} in {check.START_S} s")
    server.end()
    cpu_s += server.cpu_s
    if server.status or not served:
        exited = f"{ran['server'][0]} exited {server.status}"
        said.append(check._said(server.log, last=True) or exited)
    if not said:
        lacking = [name for name in HELD if not (at / name).is_file()]
        return (f"no {', '.join(lacking)} in {at}" if lacking else None), cpu_s
    said += [check._said(p) for p in sorted(at.rglob("*.log"))]
    return "; ".join(dict.fromkeys(filter(None, said))), cpu_s
