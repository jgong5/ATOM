# SPDX-License-Identifier: MIT
"""`compass run --check`: every input a run lacks, listed before any server.

`lacking` resolves the run's inputs as `runfile.build` and the runner do,
calling per input the function whose refusal the run would meet, so one
lacking input hides no other: the machine spec; its entry for the tokenizer
ATOM's server loads (`run.tokenizer_entry`); each store key a flag names
(`runfile.resolve`); a step linear law and host laws held against this run's
engine config, widths, machine spec and host key (`run.pick_fitted`,
`run.refuse_unfitted_law`, `run.refuse_unfitted_host`); a step grid bundle's
load; the configuration's memory readings; an op graph's price list; and an
`--acceptance-record`. An input is missing when nothing resolves for it, and
stale when the run refuses the entry it holds, naming each artifact axis that
moved. A memory readings entry holds every rank's member, or the store refuses
it, so rank 0's load stands for each rank's.

Printed apart, outside the exit status: the gaps `--coverage`'s check counts;
an op graph's absent price list, which prices every leaf by its analytical
law, unless `--coverage` asks for its check, which then refuses; and the
modules the step roofline leaves unclassified on the model built on the meta
device, when it prices the steps the backend refuses. No `compass run`
backend is the step roofline itself.
"""

import json
import re
import shlex
from pathlib import Path

from atom.compass import run as compass_run
from atom.compass.artifacts import (
    ArtifactRefusal,
    ArtifactStore,
    Rule,
    machine_spec_digest,
)
from atom.compass.drive import runfile, serve
from atom.compass.run import OP_GRAPH, PRICE_LIST, STEP_GRID, STEP_LINEAR
from atom.compass.spec import SpecRefusal

#: Each store input by its run-file key.
NAMES = {
    STEP_LINEAR: "step linear law",
    STEP_GRID: "step grid bundle",
    "host": "host laws",
    PRICE_LIST: "price list",
}
#: The commands that make each input.
MADE_BY = {
    "machine spec": "a developer's spec script, scripts/compass/<machine>_spec.py",
    "tokenizer entry": "the spec script that built {spec}, run again with "
    "--model {model} added",
    STEP_LINEAR: "compass record, then compass fit steps --backend step-linear",
    STEP_GRID: "compass trace steps, compass measure steps --grid, then compass "
    "fit steps --backend step-grid",
    "host": "compass record, then compass fit host",
    "memory readings": "compass measure memory --gpu G --out D --machine {machine} "
    "--store {store} -- {atom}",
    PRICE_LIST: "compass trace ops --population, compass measure ops --plan, then "
    "compass fit ops",
    "acceptance record": "compass record of a speculative run",
    "coverage input": "compass record or compass trace steps under these ATOM "
    "arguments",
    "coverage": "compass measure steps --grid, or compass measure ops --plan",
}
#: The artifact axis each cell of an invalidation refusal names.
AXIS = re.compile(r"` x `([^`]+)`")


def report(args, atom_args) -> int:
    """Print each input `lacking` finds, with the run's refusal and the command
    that makes it, then what it prints apart; 1 when one is lacking, else 0."""
    lacks, apart = lacking(args, atom_args)
    for name, state, refusal, made_by in lacks:
        print(f"lacking {name}: {state}\n  refused: {refusal}\n  made by: {made_by}")
    if not lacks:
        print("nothing lacking: the run meets no refusal before it serves")
    for line in apart:
        print(f"apart: {line}")
    return 1 if lacks else 0


def lacking(args, atom_args) -> tuple[list, list]:
    """Each input the run `args`, parsed by `run`, lacks under ATOM's arguments
    `atom_args`, as ``(name, state, refusal, made by)``; and the lines printed
    apart."""
    from atom.compass.drive import run as drive_run
    from atom.compass.run import memory, step_grid
    from atom.model_engine.llm_engine import _load_tokenizer

    lacks, apart = [], []

    def met(name, made_by, check, *given, state=None):
        """`check(*given)`'s value, True for None, or False when it refuses,
        its refusal recorded as `name` lacking."""
        try:
            return check(*given) or True
        except (ArtifactRefusal, SpecRefusal, ValueError) as refused:
            lacks.append((name, state or _state(refused), str(refused), made_by))
            return False

    try:
        document = json.loads(Path(args.machine).read_text())
        machine = compass_run._machine({"machine": document})
    except (OSError, SpecRefusal) as refused:
        return [("machine spec", "missing", str(refused), MADE_BY["machine spec"])], []
    store = ArtifactStore(Path(args.store).resolve())
    config, engine = drive_run._config(atom_args), runfile._engine_args(atom_args)
    run = {"machine": document, "store": str(store.root)}
    tok = _load_tokenizer(engine.model, engine.trust_remote_code)
    made = MADE_BY["tokenizer entry"].format(spec=document["name"], model=engine.model)
    met("tokenizer entry", made, compass_run.tokenizer_entry, machine, tok, config)

    flags = {k: getattr(args, k) for k in runfile.KINDS if getattr(args, k)}
    for name, tokens in (flags | {"host": args.host_laws or []}).items():
        tokens, held = [tokens] if isinstance(tokens, str) else tokens, []
        for token in tokens:
            made = MADE_BY[name]
            given = store, name, token, machine, atom_args
            if found := met(f"{NAMES[name]} {token}", made, runfile.resolve, *given):
                held.append(found[0])
        if held and len(held) == len(tokens):
            run[name] = held if len(held) > 1 else held[0]
    law, host = NAMES[STEP_LINEAR], NAMES["host"]
    if STEP_LINEAR in run and met(
        law, MADE_BY[STEP_LINEAR], compass_run.pick_fitted, run, STEP_LINEAR, config
    ):
        met(law, MADE_BY[STEP_LINEAR], compass_run.refuse_unfitted_law, run, config)
    if "host" in run and met(
        host, MADE_BY["host"], compass_run.pick_fitted, run, "host", config
    ):
        given = run["host"], config, machine_spec_digest(machine)
        met(host, MADE_BY["host"], compass_run.refuse_unfitted_host, *given)
    grid, made = NAMES[STEP_GRID], MADE_BY[STEP_GRID]
    checked = STEP_GRID in run and (
        met(grid, made, compass_run.refuse_context_parallel, STEP_GRID, config)
        and met(grid, made, step_grid.load, run[STEP_GRID], config, machine)
    )
    made = MADE_BY["memory readings"].format(
        machine=args.machine, store=args.store, atom=shlex.join(atom_args)
    )
    met("memory readings", made, memory.load, run["store"], config, machine, 0)

    if getattr(args, OP_GRAPH):
        run[OP_GRAPH], key = True, runfile._price_list_key(atom_args, machine)
        listed, made = NAMES[PRICE_LIST], MADE_BY[PRICE_LIST]
        if store.directory_for(key).exists():
            run[PRICE_LIST] = key.values
            checked = met(listed, made, compass_run._price_list, run, config, machine)
        elif args.coverage:
            given = args.coverage, run, atom_args
            met(listed, made, drive_run.check, *given, state="missing")
        else:
            apart.append(
                f"price list: none at {key}, so the op graph prices every leaf by "
                f"its analytical law; made by {made}"
            )
    if record := args.acceptance_record:
        steps = Path(record) / serve.STEPS
        drawn = any(steps.glob(compass_run.ACCEPTANCE_FILE.format("*")))
        given = record, atom_args, runfile._dp_width(atom_args)
        made, state = MADE_BY["acceptance record"], None if drawn else "missing"
        met("acceptance record", made, drive_run.measured, *given, state=state)
    if args.coverage and checked:
        made, given = MADE_BY["coverage input"], (args.coverage, run, atom_args)
        if found := met("coverage input", made, drive_run.check, *given):
            apart.append(_coverage(found))
    if args.step_roofline or STEP_GRID in flags:
        apart += _unclassified(atom_args, machine.value("device.arch"))
    return lacks, apart


def _state(refused) -> str:
    """``missing`` when nothing resolves for an input, else ``stale``, with each
    artifact axis that moved."""
    if isinstance(refused, SpecRefusal) or getattr(refused, "rule", None) is (
        Rule.RESOLUTION
    ):
        return "missing"
    axes = AXIS.findall(str(refused))
    return f"stale ({', '.join(axes)})" if axes else "stale"


def _coverage(report: dict) -> str:
    """The coverage check's `report`: its counts, and its gaps as JSON."""
    gaps = {k: v for k, v in report.items() if k not in ("inputs", "steps", "share")}
    return (
        f"coverage of {', '.join(report['inputs'])}: {gaps.pop('covered')} of "
        f"{report['steps']} steps covered; made by {MADE_BY['coverage']}\n"
        + json.dumps(gaps, indent=1)
    )


def _unclassified(atom_args, arch: str) -> list[str]:
    """The line naming each module the step roofline leaves unclassified on the
    model ATOM's arguments build on the meta device of `arch`, which refuses
    every step it is asked for; none when it classifies all. The build declares
    a device process-wide, so it runs in a process of its own."""
    from atom.compass.backends.step_roofline import UNCLASSIFIED
    from atom.compass.bench.ops import _child
    from atom.compass.validate.bound import walked

    modules = _child(walked, atom_args, arch)
    if names := [m.name for m in modules if m.kind == UNCLASSIFIED]:
        return [
            "the step roofline below the backend refuses each step it is asked "
            "for, leaving these modules unclassified: " + ", ".join(names)
        ]
    return []
