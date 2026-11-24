# SPDX-License-Identifier: MIT
"""The reuse check: whether calibrated data serves a configuration of a search,
judged requirement by requirement, the first that fails named.

Only the head is supported. `head` takes it from the git repository of the tree
`atom` resolves under (`TREE`): the commit a space file freezes, which must be
one of the integration branch's (`INTEGRATION`), else that branch's tip at its
remote-tracking ref, as last fetched; and the ATOM digest of the ``atom/`` it
holds (`run.committed_atom_digest`). A tree that is not its repository's root,
a repository with no such ref, or with two remotes' refs at two commits,
refuses by name. `refuse_another_tree` refuses a tree whose ATOM digest
(`run.atom_digest`) is not the head's, naming both and the commit taken as the
head: every key reads that digest, so a tree that differs from the head only
outside ATOM's Python passes.

`law` gives the step linear law and host laws a resolved config runs under one
verdict, `reuse`, `calibrate` or `refused`, from each law file under `LAWS` and
each step linear law in a store whose key states the config's model, engine
config and platform, at any ATOM digest and form, the engine config at any
ladder the config's is a prefix of; of several, reuse comes
first, then refused, then calibrate. The requirement a verdict names:

1. the same key: none there, a store law, or host laws beside it, at another
   ATOM digest than the head's, or a law file whose host key states another,
   or a law the runner refuses for the prefill token-graph buckets it records
   whose records ran at another ATOM digest (`records_atom`) or state none,
   since a calibration of the config records them (calibrate); host laws
   fitted on another host CPU serve, predicting the host they timed;
2. not invalidated: a store law the store refuses to load under the head's
   readings (calibrate);
3. coverage: a decode rung the config schedules with no vector
   (`lacking.uncovered_rungs`), each listed (calibrate);
4. accuracy: no cost-accuracy figures, on the law or on one of its sources
   whose vectors price a step, which a source of host laws alone does not
   (calibrate); a figure on totals, holding out the costliest step or per step
   at the median past `GATE`, on some held-out run and kind, refused when the
   law's records ran at the head's ATOM digest (`records_atom`), since a
   calibration refits the same miss, else calibrate.

A verdict on a law found names, as ``run``, the ``--step-linear`` and
``--host-laws`` values a run of it takes, and one on a store's own step linear
law at the head's ATOM digest names that ``store``, whatever the verdict. A
refusal the runner makes of the config under the law, as of context
parallelism, propagates; so does one of the prefill token-graph buckets it
records, on a law whose records ran at the head's ATOM digest.
`readings` gives the config's memory readings in a
store, or where it holds none and the config's law verdict is ``reuse`` the
assumed ones carried by the law files it names as ``run``, as a run of it
reads them, `reuse`, or `measure` where none is at their key or the one there
was measured at another ATOM digest (requirement 1), or where it lacks the PAGE
bound a start of the config reads, which `compass run --check` finds stale
(`lacking.stale_page_bound`, requirement 2), ``held`` saying whether readings
are at their key, since the store refuses a second publish there. `stores`
orders the stores a candidate is looked up in.
"""

import json
import subprocess
from pathlib import Path

import atom
from atom.compass import run as compass_run
from atom.compass.artifacts import (
    ArtifactRefusal,
    ArtifactStore,
    Axis,
    Key,
    Kind,
    Rule,
    keyed_platform,
    machine_platform,
)
from atom.compass.artifacts.store import ENTRY_FILE
from atom.compass.backends.json_file import XZ, read_json
from atom.compass.backends.launch_features import GRAPHS
from atom.compass.backends.step_grid_plan import at_ladder
from atom.compass.drive import lacking, runfile
from atom.compass.drive.lacking import AXIS, stale_page_bound, uncovered_rungs
from atom.compass.run import STEP_LINEAR, memory
from atom.compass.run.step_grid import differ, keyed

#: The tree `atom` resolves under, whose repository the head is read from.
TREE = Path(atom.__file__).resolve().parents[1]
#: The checked-in law files, stored xz-compressed; a plain JSON one reads too.
LAWS = TREE / "scripts" / "compass" / "laws"
#: The integration branch, whose tip at its remote-tracking ref is the head.
INTEGRATION = "feature/atomcompass_new"
#: The cost-accuracy gate, and the figures of a held-out run's kind it holds.
GATE, GATED = 0.05, ("on_totals", "held_out", "median")
#: A configuration's verdicts on its laws, the first preferred.
PREFERRED = ("reuse", "refused", "calibrate")


def head(commit: str | None = None) -> dict:
    """The head: ``commit``, the commit taken as it, `commit` when a space file
    freezes one, else the tip of ``ref``, `INTEGRATION`'s remote-tracking ref;
    and ``atom``, the ATOM digest of the ``atom/`` it holds."""

    def git(*args) -> tuple[int, str]:
        done = subprocess.run(
            ["git", "-C", str(TREE), *args],
            capture_output=True,
            text=True,
            check=False,
        )
        return done.returncode, (done.stderr if done.returncode else done.stdout)

    code, top = git("rev-parse", "--show-toplevel")
    if code or Path(top.strip()).resolve() != TREE:
        raise ValueError(
            f"the tree atom resolves under, {TREE}, is not a git repository's "
            f"root, so the head's ATOM digest cannot be read: {top.strip()}"
        )
    tips = {}
    for remote in git("remote")[1].split():
        ref = f"{remote}/{INTEGRATION}"
        code, tip = git("rev-parse", "--verify", f"refs/remotes/{ref}^{{commit}}")
        if not code:
            tips[ref] = tip.strip()
    if len(set(tips.values())) != 1:
        raise ValueError(
            f"{TREE}'s repository has {len(set(tips.values()))} commits at a "
            f"remote-tracking ref of {INTEGRATION}, and the head is one: "
            + ("; ".join(f"{r} at {t}" for r, t in tips.items()) or "fetch it")
        )
    ref, tip = next(iter(tips.items()))
    taken = tip
    if commit is not None:
        code, taken = git("rev-parse", "--verify", f"{commit}^{{commit}}")
        taken = taken.strip()
        if code or git("merge-base", "--is-ancestor", taken, tip)[0]:
            raise ValueError(
                f"the frozen commit {commit} is not one of {ref}'s, at {tip}: a "
                "search runs at the head or at a commit of its branch"
            )
    atom_digest = compass_run.committed_atom_digest(TREE, taken)
    return {"commit": taken, "ref": ref, "atom": atom_digest}


def refuse_another_tree(taken: dict) -> None:
    """Refuse when the tree `atom` resolves under is not the head `taken`
    (`head`) by ATOM digest, naming both digests and the commit."""
    if (here := compass_run.atom_digest()) != taken["atom"]:
        raise ValueError(
            f"the tree atom resolves under, {TREE}, has ATOM digest {here}, and "
            f"the head, {taken['commit']} at {taken['ref']}, has {taken['atom']}: "
            "only the head is supported"
        )


def stores(shared: str, out: Path) -> list[ArtifactStore]:
    """The stores a candidate is looked up in, in order: each calibration's own
    under `out`'s ``cal/``, the latest published into first, then `shared`."""

    def latest(root: Path) -> float:
        entries = root.glob(f"*/*/{ENTRY_FILE}")
        return max((p.stat().st_mtime for p in entries), default=0.0)

    own = sorted(Path(out).glob("cal/*/store"), key=latest, reverse=True)
    return [ArtifactStore(root) for root in [*own, Path(shared)]]


def law(store: ArtifactStore, config, machine, atom_args, atom: str) -> dict:
    """The verdict on the laws `config`, ATOM's arguments `atom_args` resolved,
    runs under on `machine`, from `LAWS` and `store`, at the head's ATOM digest
    `atom`."""
    model = compass_run._file_digest(config.model, "config.json")
    engine, platform = keyed(config), machine_platform(machine)
    stated = keyed_platform(Kind.STEP_LINEAR_LAW, platform)
    found = []
    for path in sorted([*LAWS.glob("*.json"), *LAWS.glob(f"*.json{XZ}")]):
        held = read_json(path)
        host = held.get("host")
        at = host and (held.get("platform"), host["key"].get("model"))
        if at == (stated, model) and _same(held.get("engine_config"), engine):
            found.append(_law_file(str(path), held, config, machine, atom))
            found[-1]["run"] = [str(path)] * 2
    for key in _keyed(store, Kind.STEP_LINEAR_LAW, engine, model=model, **stated):
        where = f"{key.kind} {key.token()} in {store.root}"
        if (ran := key.values["source_root"]) != compass_run.source_root(atom):
            why = f"ran at {ran}, and a store entry is never re-keyed"
            found.append(_verdict("calibrate", 1, f"{where} {why}", law=where))
            continue
        read = {"law": where, "store": str(store.root)}
        try:
            step = runfile.resolve(store, STEP_LINEAR, key.token(), machine, atom_args)
        except ArtifactRefusal as refused:
            if refused.rule is not Rule.INVALIDATED:
                raise
            found.append(_verdict("calibrate", 2, f"{where}: {refused}", **read))
            continue
        here = {"model": model, "atom_src": atom}
        here |= keyed_platform(Kind.HOST_LAWS, platform)
        hosts = _keyed(store, Kind.HOST_LAWS, engine, **here)
        if not hosts:
            why = f"{where} has no host laws beside it at the head's ATOM digest"
            found.append(_verdict("calibrate", 1, why, **read))
            continue
        hosts.sort(key=lambda k: k.values["host_cpu"] != compass_run.host_cpu())
        host = runfile._member(store.read(hosts[0]), "params")
        found.append(_judged(where, step[0], host, config, machine, atom) | read)
        found[-1]["run"] = [key.token(), hosts[0].token()]
    if not found:
        why = (
            f"no step linear law at its key, model {model}, this engine config and "
            f"platform {stated}, in {store.root} or {LAWS}"
        )
        return _verdict("calibrate", 1, why)
    return min(found, key=lambda v: PREFERRED.index(v["verdict"]))


def readings(store: ArtifactStore, config, machine, law: dict) -> dict:
    """The verdict on `config`'s memory readings in `store` on `machine`, else,
    when `config`'s verdict `law` is ``reuse``, in the law files it names to run
    with, as `compass run` reads them: a calibration runs with the store law it
    publishes, which carries none."""
    carried = runfile.carried(law["run"]) if law["verdict"] == "reuse" else ()
    try:
        read = memory.load(str(store.root), config, machine, 0, carried)
    except ArtifactRefusal as refused:
        stale = refused.rule is Rule.INVALIDATED
        if refused.rule is not Rule.RESOLUTION and not (
            stale and AXIS.findall(str(refused)) == [str(Axis.ATOM_SRC)]
        ):
            raise
        return _verdict("measure", 1, str(refused), held=stale)
    held = (store.directory_for(memory.key_of(config, machine)) / ENTRY_FILE).is_file()
    if why := stale_page_bound(read, config):
        return _verdict("measure", 2, why, held=held)
    where = f"at their key in {store.root}" if held else f"in {read.source}"
    return _verdict("reuse", None, where)


def _law_file(where: str, held: dict, config, machine, atom: str) -> dict:
    if (ran := held["host"]["key"].get("atom")) != atom:
        why = f"{where} states ATOM digest {ran}, and the head's is {atom}"
        return _verdict("calibrate", 1, why, law=where)
    step = {k: v for k, v in held.items() if k not in runfile.NOT_STEP_LAW}
    return _judged(where, step, held["host"], config, machine, atom)


def _judged(where: str, law: dict, host: dict, config, machine, atom: str) -> dict:
    """The verdict on the step linear law `law` and host laws `host` found at
    `where`, both at the head's key, on any host CPU, the one they timed being
    the one predicted: requirement 1's prefill token-graph buckets, then
    requirements 3 and 4."""
    compass_run.refuse_unfitted_law(
        {STEP_LINEAR: law, "machine": machine.echo()}, config
    )
    compass_run.refuse_unfitted_host(host, config, machine_platform(machine))
    buckets = lacking.recorded_buckets(config, machine.value("device.arch"))
    try:
        compass_run.refuse_other_buckets(
            config, buckets, {STEP_LINEAR: law, "host": host}
        )
    except ValueError as refused:
        if (ran := law.get("records_atom")) == atom:
            raise
        ran = f"from ATOM digest {ran}" if ran else "stating no ATOM digest"
        why = f"{where}: {refused}, on records re-keyed {ran}"
        return _verdict("calibrate", 1, why, law=where)
    if rungs := [str(r) for r in uncovered_rungs(law, config)]:
        why = (
            f"{where} has no decode vector for rung{'s' * (len(rungs) > 1)} "
            f"{', '.join(rungs)}, which this configuration schedules"
        )
        return _verdict("calibrate", 3, why, law=where, rungs=rungs)
    past = []
    for name, held in _folded(law).items():
        of = f" of source {name}" if name else ""
        if not held.get("figures"):
            why = f"{where} states no cost-accuracy figures{of}"
            return _verdict("calibrate", 4, why, law=where)
        past += [
            (abs(v), f"run {run}, {kind}, {figure} {v:+.2%}{of}")
            for run, kinds in held["figures"].items()
            for kind, figures in kinds.items()
            for figure in GATED
            if (v := figures.get(figure)) is not None and abs(v) > GATE
        ]
    if not past:
        return _verdict("reuse", None, where, law=where)
    worst = f"{where}: {max(past)[1]}, past {GATE:.0%}"
    if (ran := law.get("records_atom")) == atom:
        why = f"{worst}, on records run at the head, which a calibration refits alike"
        return _verdict("refused", 4, why, law=where)
    ran = f"from ATOM digest {ran}" if ran else "stating no ATOM digest"
    return _verdict("calibrate", 4, f"{worst}, on records re-keyed {ran}", law=where)


def _folded(law: dict) -> dict:
    """What of `law` states cost-accuracy figures, by source name: each of its
    `sources` with a vector that prices a step, the prefill one or a decode one
    (`step_linear_fit._vector_of`), as ``step_linear_fit figures`` folds no
    other, such as one of host laws alone; else the law itself, named ``""``."""
    sources = (law.get("sources") or {}).items()
    priced = {
        name: source
        for name, source in sources
        if any(v == "prefill" or v.startswith("decode ") for v in source["vectors"])
    }
    return priced or {"": law}


def _keyed(store: ArtifactStore, kind: Kind, engine: dict, **fields) -> list[Key]:
    """The keys of `store`'s `kind` entries that state each of `fields` as
    given, and `engine` as their engine config, less its graph sets."""
    keys = []
    for path in sorted((store.root / kind.value).glob(f"*/{ENTRY_FILE}")):
        key = json.loads(path.read_text())["key"]
        theirs = json.loads(key["engine_config"])
        theirs.pop(GRAPHS, None)
        if all(key.get(f) == v for f, v in fields.items()) and _same(theirs, engine):
            keys.append(Key.of(kind, **key))
    return keys


def _same(there, here: dict) -> bool:
    """Whether `there` is the engine config `here`, as the runner compares one:
    `here` at `there`'s ladder where its own is a prefix of it (`at_ladder`)."""
    return isinstance(there, dict) and not differ(there, at_ladder(here, there), "")


def _verdict(verdict: str, requirement: int | None, reason: str, **more) -> dict:
    return {"verdict": verdict, "requirement": requirement, "reason": reason, **more}
