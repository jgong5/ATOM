# SPDX-License-Identifier: MIT
"""`compass run`: one simulated instance's server, serving whatever client runs
reach it until it is stopped, with run.json recording what grading reads.

    compass run <backend> --machine M --store S [--host-laws KEY]
        [--coverage INPUT...] [--on-refusal mark|abort] [--step-roofline]
        [--acceptance-record RECORD] --out D -- <ATOM args>

    compass run --join RUN_FILE --out D -- <ATOM args>

`runfile` builds the run file from the flags, the backend exactly one with no
default, and `serve` starts the server simulated from it with the step record
on. `--out` holds what `serve` writes and what the run writes there: each DP
rank's step record and cost lines, the step table, the run summary and each
LP's hand-in.

`--join` names a deployment's run file in place of every flag but `--out`: the
server joins the clock authority its ``clock_endpoint`` names, which writes the
step table, the run summary and the hand-ins in its ``out_dir``, and grading
reads them there. `scripts/compass/deploy.sh` builds that file, starts the
authority and joins each instance to it. A prefill or a decode instance shares a
clock with its peer, so it refuses without `--join`.

`--coverage` names record directories and `shape_population` tokens whose steps
the backend's coverage check reads before the server starts: the step grid's
is `step_grid_plan.coverage`, over the bundle `step_grid.load` loads under the
config ATOM's arguments resolve. An input whose DP rank's engine file (a
record's `steps/dp<r>.engine.json`, a population's `engine` member) states
another `SHAPING` field than that config refuses first, naming each field and
both values. The check refuses no step: it counts those the grid extrapolates
and those that fall to the step roofline below it, naming each with the grid's
reasons. The op graph's is `op_coverage`, over the backend `run.op_graph`
builds from the run file: a step is covered when its price list prices every
leaf, and a run file that names no price list refuses. Another backend has no
check, and refuses `--coverage`.

`--acceptance-record` names a speculative `compass record` directory: its
histograms' rates (`measured`) go to ATOM as ``--spec-decode-acceptance-rates``,
and the run file names the record, so the summary states the input ``measured``.
A deployment's run file that names one, a decode record, gives its rates to
each instance that joins it as `joined_atom_args` says.

Once the server stops, run.json gains, beside `serve`'s identity: `run_file`,
the run file the server was handed; `step_roofline`, the run file's key,
whether `--step-roofline` was on; `skip`, the host segments the run file's
host entry skips; `refusals`, client run 0's summary's, absent when it has
none, as when a stop came before it ended; `coverage`, the check's report with
the inputs it read, or None when no check ran; and `provenance`, the backend's
digest and fit provenance as the run file states them, and as
`memory_readings_assumed` the assumption each assumed `memory_readings`
member an engine read states, empty when every one was measured.
"""

import argparse
import json
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from atom.compass import run as compass_run
from atom.compass.artifacts import ArtifactStore, Kind
from atom.compass.backends import BatchView, CostRefused, RequestShape
from atom.compass.backends.step_grid_plan import BOUNDS, coverage
from atom.compass.drive import runfile, serve
from atom.compass.run import (
    OP_GRAPH,
    PRICE_LIST,
    STEP_GRID,
    STEP_LINEAR,
    STEP_STUB,
    step_grid,
)

PROG = "compass run"
#: The engine fields that shape a step's batches: the scheduler's bounds, the
#: capture ladder and the TP width. An input states them as this run resolves
#: them, or refuses.
SHAPING = (*BOUNDS, "ladder", "tensor_parallel_size")
#: A number standing alone in a reason, which `op_coverage` masks so one cause
#: at many steps' counts is one reason; a kernel's name keeps its digits.
NUMBER = r"(?<![\w.])\d+(?:\.\d+)?"


def main(argv: list[str]) -> None:
    cut = argv.index("--") if "--" in argv else len(argv)
    atom_args = argv[cut + 1 :]
    joining = argparse.ArgumentParser(prog=PROG, add_help=False)
    joining.add_argument("--join")
    if joining.parse_known_args(argv[:cut])[0].join:
        side = runfile.pd_side(atom_args)
        join(_joining_parser().parse_args(argv[:cut]), atom_args, side)
        return
    args = _parser().parse_args(argv[:cut])
    if args.coverage and not (getattr(args, STEP_GRID) or getattr(args, OP_GRAPH)):
        key = next(k for k in (STEP_STUB, STEP_LINEAR) if getattr(args, k))
        raise ValueError(
            f"--coverage: --{key.replace('_', '-')} has no coverage check; only "
            "--step-grid's and --op-graph's steps are checked"
        )
    runfile.pd_side(atom_args, refuse=True)
    run = runfile.build(args, atom_args)
    if record := args.acceptance_record:
        atom_args = with_measured_rates(record, atom_args, run["data_parallel_size"])
        run[compass_run.ACCEPTANCE_RECORD] = str(Path(record).resolve())
    report = check(args.coverage, run, atom_args) if args.coverage else None
    serve.serve(atom_args, args.out, run=run)
    write_grading(args.out, report)


def join(args, atom_args, side) -> None:
    """Serve one instance of a deployment whose authority already runs, from
    its run file `args.join`, and record what grading reads for its `side`."""
    run = json.loads(Path(args.join).read_text())
    if not {"clock_endpoint", "out_dir"} <= run.keys():
        raise ValueError(
            f"--join {args.join} names no clock_endpoint and out_dir; a "
            "deployment's run file names its authority and its directory"
        )
    atom_args = joined_atom_args(run, atom_args, side)
    serve.serve(atom_args, args.out, run=run, joined=True)
    write_grading(args.out, None, side)


def _joining_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=PROG, description=__doc__.split("\n\n")[0])
    p.add_argument("--join", required=True, metavar="RUN_FILE")
    p.add_argument("--out", required=True, help="an empty directory for its files")
    return p


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=PROG, description=__doc__.split("\n\n")[0])
    runfile.add_arguments(p)
    p.add_argument(
        "--coverage",
        nargs="+",
        metavar="INPUT",
        help="record directories and shape_population tokens the check reads",
    )
    p.add_argument(
        "--acceptance-record",
        metavar="RECORD",
        help="a speculative record directory whose acceptance the run draws from",
    )
    p.add_argument(
        "--out", required=True, help="an empty directory for the run's files and logs"
    )
    return p


def measured(record, atom_args, ranks: int) -> list[float]:
    """The rates the accepted-draft histograms of `record` give over DP ranks
    ``0..ranks-1``: entry ``i`` the share of rows that accepted more than ``i``
    drafts. Refuses no histogram, a missing rank's, another draft width, and a
    stated acceptance."""
    engine = runfile._engine_args(atom_args)
    for knob in ("rate", "length", "rates"):
        if getattr(engine, f"spec_decode_acceptance_{knob}") is not None:
            raise ValueError(
                f"--acceptance-record states the acceptance, and ATOM's arguments "
                f"state --spec-decode-acceptance-{knob} too"
            )
    width = engine.num_speculative_tokens if engine.method else 0
    files = sorted(
        (Path(record) / serve.STEPS).glob(compass_run.ACCEPTANCE_FILE.format("*"))
    )
    counts = Counter()
    for path in files:
        held = json.loads(path.read_text())
        if held["draft_tokens"] != width:
            raise ValueError(
                f"--acceptance-record: {path} drafted {held['draft_tokens']} "
                f"tokens a step, and ATOM's arguments draft {width}"
            )
        counts.update({int(a): n for a, n in held["distribution"].items()})
    rows = counts.total()
    if not rows:
        raise ValueError(
            f"--acceptance-record: {record} holds no accepted-draft histogram; "
            "a speculative record writes one per DP rank in its steps"
        )
    names = {compass_run.ACCEPTANCE_FILE.format(r) for r in range(ranks)}
    if missing := sorted(names - {p.name for p in files}):
        raise ValueError(
            f"--acceptance-record: {record} holds no {', '.join(missing)}, and "
            f"ATOM's arguments run {ranks} DP ranks"
        )
    return [sum(n for a, n in counts.items() if a > i) / rows for i in range(width)]


def with_measured_rates(record, atom_args, ranks: int) -> list[str]:
    """ATOM's arguments `atom_args` stating the rates `measured` reads from
    `record` over DP ranks ``0..ranks-1``."""
    rates = measured(record, atom_args, ranks)
    return [*atom_args, "--spec-decode-acceptance-rates", ",".join(map(str, rates))]


def joined_atom_args(run: dict, atom_args, side) -> list[str]:
    """ATOM's arguments `atom_args` of a deployment's `side` instance, stating
    the rates of the decode record its run file `run` names: the decode
    instance always, the prefill instance when it drafts, since its hand-in
    reads the same run file."""
    record = run.get(compass_run.ACCEPTANCE_RECORD)
    if not record or (side == "P" and not runfile._engine_args(atom_args).method):
        return atom_args
    try:
        return with_measured_rates(record, atom_args, run[compass_run.DP_WIDTHS["D"]])
    except ValueError as refused:
        name = "prefill" if side == "P" else "decode"
        raise ValueError(f"the {name} instance: {refused}") from None


def check(inputs, run: dict, atom_args) -> dict:
    """The coverage of the steps `inputs` hold by the step grid or the op
    graph's price list the run file `run` names, loaded under the config
    `atom_args` resolve; an input that ran under other `SHAPING` fields
    refuses."""
    store, config = ArtifactStore(run["store"]), _config(atom_args)
    here = step_grid.keyed(config) | {name: getattr(config, name) for name in BOUNDS}
    steps = [s for name in inputs for s in _steps(name, store, here)]
    machine = compass_run._machine(run)
    if run.get(OP_GRAPH):
        if PRICE_LIST not in run:
            raise ValueError(
                "--coverage: the store holds no price_list at this run's key, so "
                "the op graph prices every leaf by its roofline and no step is "
                "covered; compass fit ops publishes one"
            )
        # ATOM's MoE shards no experts without expert parallelism.
        ep = _ep_width(atom_args, machine) if config.enable_expert_parallel else 1
        backend = compass_run.op_graph(run, config, machine, ep)
        return {"inputs": list(inputs), **op_coverage(backend, steps)}
    backend, _ = step_grid.load(run[STEP_GRID], config, machine)
    return {"inputs": list(inputs), **coverage(backend, steps)}


def _ep_width(atom_args, machine) -> int:
    """`run.ep_width` of the model ATOM's arguments `atom_args` build on the meta
    device of `machine`'s arch, in a process of its own: the build declares a
    device process-wide, which the server this process starts must not inherit."""
    from atom.compass.bench.ops import _child

    return _child(_built_ep_width, atom_args, machine.value("device.arch"))


def _built_ep_width(atom_args, arch: str) -> int:
    from atom.compass.validate.bound import _build

    engine, kept = runfile._engine_args(atom_args), runfile.engine_only(atom_args)
    with tempfile.TemporaryDirectory(prefix="compass-run-ep-") as tmpdir:
        _, tree = _build(engine.model, engine.tensor_parallel_size, kept, arch, tmpdir)
    return compass_run.ep_width(tree)


def op_coverage(backend, steps) -> dict:
    """How the price list of `backend`, an `OpGraphBackend`, covers the named
    step-record lines `steps`: the steps and those covered by phase, and per
    leaf kind the steps it left out by why, numbers masked, one why (its last)
    per kind per step so a kind's counts sum to steps. A step is covered
    when the list prices each leaf the backend asks it for; a leaf it declines,
    on a line the fit declined among them, goes to the roofline and leaves its
    step out. A step the backend refuses is left out under ``step refused``."""
    by_phase, out = defaultdict(Counter), defaultdict(Counter)
    for _, line in steps:
        view = BatchView(
            tuple(RequestShape(*row) for row in line["rows"]),
            capture_rung=line["rung"],
            moe_rows=line.get("moe_rows"),
            produces_output=line["produces_output"],
        )
        try:
            cost = backend.price(view)
            terms = [
                (t.name.removeprefix("op."), t.refusals)
                for t in (*cost.terms, *cost.draft_terms)
            ]
        except CostRefused as refused:
            terms = [("step refused", refused.declined)]
        whys = {
            kind: re.sub(NUMBER, "#", r.reason.removeprefix(f"{kind}: "))
            for kind, refusals in terms
            for r in refusals
        }
        phase = "prefill" if view.prefill else "decode"
        by_phase[phase].update(steps=1, covered=int(not whys))
        for kind, why in whys.items():
            out[kind][why] += 1
    total = sum(by_phase.values(), Counter())
    return {
        "steps": total["steps"],
        "covered": total["covered"],
        "share": total["covered"] / max(total["steps"], 1),
        "by phase": {phase: dict(c) for phase, c in sorted(by_phase.items())},
        "left out, by leaf kind": {k: dict(v) for k, v in sorted(out.items())},
    }


def _steps(name: str, store, here: dict) -> list:
    """The steps of the record directory or `shape_population` token `name`,
    less DP's dummy runs, each named `<name> dp<r> step <n>`; a rank whose
    engine states other `SHAPING` fields than `here` refuses, naming each."""
    path = Path(name)
    if (path / serve.RUN_JSON).is_file():
        records = {
            int(p.stem[2:]): (p.read_text(), p.with_suffix(".engine.json").read_text())
            for p in (path / serve.STEPS).glob("dp*[0-9].jsonl")
        }
    else:
        entry = store.read(store.key_from_token(Kind.SHAPE_POPULATION, name))
        records = {
            rank.dp: (
                entry.read_member("steps", rank, "jsonl").decode(),
                entry.read_member("engine", rank, "json").decode(),
            )
            for rank in entry.topology.ranks()
        }
    for rank, (_, engine) in sorted(records.items()):
        ran = json.loads(engine)
        if differ := [k for k in SHAPING if ran.get(k) != here[k]]:
            raise ValueError(
                f"--coverage: {name} dp{rank} ran under another engine than "
                "ATOM's arguments resolve: "
                + "; ".join(
                    f"{k} {ran.get(k)!r} there, {here[k]!r} here" for k in differ
                )
            )
    return [
        (f"{name} dp{rank} step {line['step']}", line)
        for rank, (text, _) in sorted(records.items())
        for line in map(json.loads, text.splitlines())
        if not line["is_dummy_run"]
    ]


def _config(atom_args):
    """ATOM's `Config` for a server's arguments, as the server runs it."""
    from atom.compass.bench.measure import atom_config

    return atom_config(atom_args, server=True)


def write_grading(out, report, side=None) -> None:
    """Add what grading reads to the run.json of `out`, a stopped server's
    directory, the coverage check's `report` among it: from the run summary
    and hand-ins in its run file's ``out_dir``, its own engine's of a
    deployment's prefill (`side` ``P``) or decode (``D``) instance."""
    out = Path(out)
    run = json.loads((out / serve.RUN_FILE).read_text())
    where = Path(run.get("out_dir", out))
    first = where / compass_run.CLIENT_DIR.format(0) / compass_run.SUMMARY_FILE
    written = json.loads((out / serve.RUN_JSON).read_text())
    engine = compass_run.ENGINE.name + (f"-{side}" if side else "")
    engines = where.glob(compass_run.LP_FILE.format(f"{engine}*"))
    memory = [json.loads(p.read_text()).get("memory") or {} for p in engines]
    assumed = sorted({m["assumed"] for m in memory if m.get("assumed")})
    written |= {
        "run_file": run,
        "step_roofline": run["step_roofline"],
        "skip": [
            s for h in compass_run.entries(run, "host") for s in h.get("skip", [])
        ],
        "coverage": report,
        "provenance": {**run["backend"], "memory_readings_assumed": assumed},
    }
    if first.is_file():
        summary = json.loads(first.read_text())
        written["refusals"] = summary["schedule"]["refusals"]
    compass_run._write_whole(out / serve.RUN_JSON, json.dumps(written))
