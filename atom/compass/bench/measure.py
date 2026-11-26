# SPDX-License-Identifier: MIT
"""`compass measure steps`: batches replayed or planned, each distinct batch
measured on the bench once and published as one `step_measurement`.

    compass measure steps [INPUT...] [--grid INPUT... --grid-definition FILE]
        [--sample N --seed S | --sample all] [--ragged] [--sessions A:B]
        [--client-run INPUT EXPORT]... --sampling JSON --gpu G --store ROOT
        --out DIR -- <ATOM args>

An input is a record directory, `record`'s output, or a `shape_population`
entry named by the token `Key.token` prints, whose members are, per DP rank,
`steps` (`.jsonl`), the rank's step-record lines, and `engine` (`.json`), its
engine file. Its lines are read as `parity.served` reads a record, each with
the traces the client runs `--client-run` names for it sent. Lines a dummy run
wrote are dropped; under `--sessions A:B` so is every line serving no trace, or
a trace whose session, as its client run's export states it, lies outside A to
B, inclusive; a trace no export gives a session, or its exports give two, refuses.
An input refuses unless a `--client-run` names it, its kept lines served
requests under `--sampling` alone, and it states the engine ATOM's arguments
resolve, `keyed`'s fields, at data-parallel width 1, the bench's.

A prefill or decode context-parallel width above 1 refuses: a bundle states
no such width.

A batch is a line's view: its `rows`, `rung` and `produces_output`, and the
KV blocks its rows share where a planned line states them (`shared`). The
positional inputs' distinct views are replayed, origin `replay`, drawn by
`--sample N --seed S`, N from each stratum (a decode rung, or a prefill kind
in a token band of `step_linear_fit.TOKEN_BANDS`), or all of them by `--sample
all`. Under `--ragged` each drawn decode view is measured not itself but as
the ragged views `ragged.compose` makes of it, by `--seed` and the view's
digest, from the contexts of the positional inputs' decode rows, origin
`ragged`, naming those inputs and every trace their lines served; a drawn
prefill view is not measured. `--grid` runs the step grid's planner over its
inputs' lines for the engine they state and the nodes and margin
`--grid-definition` states, under `--ragged` over the ragged views too, whose
longest rows' cells alpha's fit reads, and its points are measured with
origin `plan`, the digest of the plan, which `<out>/plan.json` holds, and the
`--grid` inputs as their sources. A planned
point that is also a replayed view is measured once, as that replay alone: its
origin, sources and traces are the replay's, and it names no plan.

A measurement's key is the configuration (model, platform,
`keyed` engine, source root, `--sampling`), the step kind and the view's
digest, so a view the store holds is not measured again and a stopped
invocation resumes; a held entry taken under other readings than this run's
refuses, naming it; one held under an earlier protocol is kept, not measured
again. Its member `MEMBER` holds the view, `seconds` (the median
of the bench's kept repeats, under a drafter of their target's parts, and
`draft_seconds` that of their drafter's parts), the repeats and how many each
burst kept (`bursts`), the protocol, `bench_seconds`, the wall time the bench
took on it, the replay rule, the origin, the inputs it came from (`sources`) and the
`traces` their lines served, and `group_needs_eager`, true, where the bench
ran the batch eagerly because an attention backend refused it a graph, as a
step-record line states it. A batch the rule or the bench refuses is not
published; the command measures the rest, then fails naming each. `<out>`
also holds run.json, with the warm-up's and the measuring's boundaries, the
engine file and the GPU checks.

A point's repeats come in bursts (`window.py`): one pass gives every point its
first, the next its second, and a point whose two disagree runs on to
`window.REPEATS` at once; each is published after its last burst. Before
measuring, the command states each step kind's points, how many it measures,
and the hours they take at the median `bench_seconds` of the points of that
kind the store holds under this protocol at this model, platform, sampling and
engine on any ladder; after, the hours each kind took.

The bench holds the largest KV pool its inputs' engines state, whatever this
device's own budget gives, so every batch they ran, and every point planned
for that pool, fits it.

The API server's flags `compass record` serves with (`measure_memory.FRONTEND`)
are taken and dropped; any other flag ATOM's engine lacks refuses.
"""

import argparse
import datetime
import hashlib
import json
import os
import random
import statistics
import tempfile
import time
from collections import Counter
from pathlib import Path

from atom.compass.artifacts import (
    ArtifactRefusal,
    ArtifactStore,
    GateState,
    Key,
    Kind,
    Provenance,
    Topology,
    keyed_platform,
    member_name,
    roots_for,
)
from atom.compass.backends.step_grid_plan import plan
from atom.compass.backends.step_labels import digest as file_digest
from atom.compass.backends.step_linear_fit import recorded_level, token_band
from atom.compass.bench import ragged
from atom.compass.drive.gpu import add_gpu_flag, hip_visible, kfd_pid, quiet
from atom.compass.drive.identity import identity, phase
from atom.compass.fit.publish import conditions, platform
from atom.compass.fit.samples import MEMBER, run_fields
from atom.compass.parity.served import served
from atom.compass.run import _write_whole
from atom.compass.run.step_grid import BOUNDS, keyed, keyed_of
from atom.utils.arg_parser import FlexibleArgumentParser

PROG = "compass measure steps"
#: What of a step-record line is the batch it measures; a planned line may also
#: state the KV blocks its rows share, `shared`.
VIEW = ("rows", "rung", "produces_output")
#: What an input's engine and sampling are compared with, as a refusal names
#: each: here ATOM's arguments and `--sampling`.
AGAINST = ("ATOM's arguments resolve", "--sampling states")


def main(argv: list[str]) -> None:
    cut = argv.index("--") if "--" in argv else len(argv)
    args = _parser().parse_args(argv[:cut])
    _check(args)
    sampling = _sampling(args.sampling)
    os.environ["HIP_VISIBLE_DEVICES"] = hip_visible(args.gpu)
    kfd_pid()  # before atom_config opens a GPU, so quiet leaves this process out
    config = atom_config(argv[cut + 1 :])
    for name in ("prefill_context_parallel_size", "decode_context_parallel_size"):
        if (width := getattr(config, name)) > 1:
            raise ValueError(
                f"{name} {width}: a step grid bundle states no context-parallel "
                "width, so measure steps measures at width 1"
            )
    engine = keyed(config)
    store, out = ArtifactStore(args.store), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    points, num_blocks = _points(args, store, engine, sampling, out)
    if not points:
        raise ValueError("the inputs hold no batch to measure once filtered")
    engine_file = out / "engine.json"
    resolved = engine | {name: getattr(config, name) for name in BOUNDS}
    engine_file.write_text(json.dumps(resolved))
    command = [*PROG.split(), *argv]
    run, phases = identity(engine_file, config.model, command, args.gpu), {}
    configuration = run_fields(run)
    stated = configuration | {k: run[k] for k in ("stack", "host_key", "ptl")}
    readings = conditions({**stated, "engine_config": engine})
    configuration |= keyed_platform(Kind.STEP_MEASUREMENT, platform(stated))
    todo = _todo(points, configuration, engine, sampling, store, readings)
    past = _past(store, configuration, engine, sampling) if todo else {}
    print(_statement(points, todo, past), flush=True)
    refused = []
    try:
        if todo:
            with quiet(args.gpu, out):
                with phase(phases, "warm-up"):
                    bench = start_bench(config, num_blocks)
                try:
                    blocks = {"num_kvcache_blocks": config.num_kvcache_blocks}
                    engine_file.write_text(json.dumps(resolved | blocks))
                    with phase(phases, "measuring"):
                        refused = _measure(
                            bench,
                            config,
                            sampling,
                            store,
                            " ".join(command),
                            readings,
                            todo,
                        )
                finally:
                    bench.exit()
    finally:
        _write_whole(out / "run.json", json.dumps({**run, "phases": phases}))
    if refused:
        raise ValueError(
            f"{len(refused)} of {len(todo)} batches refused:\n" + "\n".join(refused)
        )


def _todo(points, configuration, engine, sampling, store, readings) -> list:
    """Each point's key and the point, for the points the store does not hold;
    a held entry measured under other readings than this run's refuses."""
    todo, stale = [], []
    for point in points.values():
        key = measurement_key(configuration, engine, sampling, point["view"])
        if not store.directory_for(key).exists():
            todo.append((key, point))
            continue
        try:
            store.load(key, conditions=readings, gates=GateState.of())
        except ArtifactRefusal as refusal:
            stale.append(f"{key.token()}: {refusal}")
    if stale:
        raise ValueError(
            f"{len(stale)} measurements in the store were taken under other "
            "readings than this run's:\n" + "\n".join(stale) + "\nA held entry "
            "is never replaced: measure into a fresh --store, or remove the "
            "entries named"
        )
    return todo


def _past(store, configuration: dict, engine: dict, sampling: dict) -> dict:
    """The `bench_seconds` of each point `store` holds measured under this
    protocol at `configuration` on any source root, `sampling` and `engine` on
    any ladder, by step kind: what an estimate reads."""
    from atom.compass.artifacts.store import ENTRY_FILE
    from atom.compass.bench.window import PROTOCOL

    stated = {k: v for k, v in configuration.items() if k != "source_root"}
    stated["sampling"] = json.dumps(sampling, sort_keys=True)
    unladdered = {k: v for k, v in engine.items() if k != "ladder"}
    past = {}
    folder = store.root / Kind.STEP_MEASUREMENT.value
    for path in sorted(folder.glob(f"*/{ENTRY_FILE}")):
        try:
            there = json.loads(path.read_bytes())["key"]
            ran = json.loads(there["engine_config"])
            if any(there.get(k) != v for k, v in stated.items()) or unladdered != {
                k: v for k, v in ran.items() if k != "ladder"
            }:
                continue
            key = Key.of(Kind.STEP_MEASUREMENT, **there)
            member = json.loads(store.read(key).read_member(*MEMBER))
        except (OSError, ValueError, KeyError, TypeError, ArtifactRefusal):
            continue  # an entry an estimate cannot read is no estimate's
        if member.get("protocol") == PROTOCOL and "bench_seconds" in member:
            past.setdefault(there["step_kind"], []).append(member["bench_seconds"])
    return past


def _statement(points: dict, todo: list, past: dict) -> str:
    """Each step kind's points, how many are measured and the hours they are
    estimated at, the median of the kind's `past` seconds a point."""
    kinds = Counter(_kind(point["view"]) for point in points.values())
    measuring = Counter(key.values["step_kind"] for key, _ in todo)
    held, hours = len(points) - len(todo), 0.0
    lines = [f"{len(points)} distinct batches, {held} in the store"]
    for kind in sorted(kinds, key=_in_order):
        n, line = measuring[kind], f"  {kind}: {kinds[kind]} points, "
        if not n:
            lines.append(line + "all in the store")
        elif seconds := past.get(kind):
            each = statistics.median(seconds)
            hours += n * each / 3600
            lines.append(
                line + f"{n} to measure, about {n * each / 3600:.2f} h at "
                f"{each:.1f} s a point, the median of {len(seconds)} the store holds"
            )
        else:
            lines.append(
                line + f"{n} to measure, no estimate: the store holds none of "
                "this kind measured under this protocol, model, platform and engine"
            )
    if hours:
        lines.append(f"about {hours:.2f} h estimated for the kinds above that have one")
    return "\n".join(lines)


def _in_order(kind: str) -> tuple:
    """A step kind's place in a statement, decode rungs by their width."""
    *words, last = kind.split()
    return (words, int(last)) if last.isdigit() else (kind.split(), 0)


def measurement_key(configuration: dict, engine: dict, sampling, view: dict) -> Key:
    """The key of `view`'s measurement under `configuration` (model, source
    root, platform), `engine`, `keyed`'s fields, and `sampling`."""
    return Key.of(
        Kind.STEP_MEASUREMENT,
        **configuration,
        engine_config=json.dumps(engine, sort_keys=True),
        sampling=json.dumps(sampling, sort_keys=True),
        step_kind=_kind(view),
        view=_digest(view),
    )


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=PROG, description=__doc__.split("\n\n")[0])
    p.add_argument(
        "inputs",
        nargs="*",
        metavar="INPUT",
        help="record directories and shape_population tokens to replay",
    )
    p.add_argument(
        "--grid",
        nargs="+",
        default=[],
        metavar="INPUT",
        help="record directories and shape_population tokens the planner reads",
    )
    p.add_argument("--grid-definition", metavar="FILE", help="the grid's nodes")
    p.add_argument("--sample", type=_count, help="N per stratum, or all")
    p.add_argument("--seed", type=int)
    p.add_argument(
        "--ragged",
        action="store_true",
        help="measure ragged views composed from each drawn decode view instead",
    )
    p.add_argument("--sessions", type=_span, metavar="A:B")
    p.add_argument(
        "--client-run",
        nargs=2,
        action="append",
        default=[],
        metavar=("INPUT", "EXPORT"),
        help="a client run an input's server served, by its export",
    )
    p.add_argument("--sampling", required=True, help="the request parameters, JSON")
    add_gpu_flag(p)
    p.add_argument("--store", required=True, help="the artifact store's root")
    p.add_argument("--out", required=True, help="where run.json and the plan go")
    return p


def _count(text: str):
    if text == "all" or text.isdigit() and int(text) > 0:
        return text if text == "all" else int(text)
    raise argparse.ArgumentTypeError(f"{text!r} is neither a positive count nor all")


def _span(text: str) -> tuple[int, int]:
    a, colon, b = text.partition(":")
    if colon and a.isdigit() and b.isdigit() and int(a) <= int(b):
        return int(a), int(b)
    raise argparse.ArgumentTypeError(f"{text!r} is not A:B, with A at most B")


def _check(args) -> None:
    named = {*args.inputs, *args.grid}
    for refused, why in (
        (not named, "it names no input to replay and no --grid input to plan over"),
        (
            args.inputs and args.sample is None,
            "replayed batches are drawn by --sample N --seed S or --sample all",
        ),
        (args.sample and not args.inputs, "--sample draws from no replayed input"),
        (args.ragged and not args.inputs, "--ragged composes from no replayed input"),
        (isinstance(args.sample, int) and args.seed is None, "--sample N needs --seed"),
        (
            args.grid and not args.grid_definition,
            "--grid plans with the nodes and margin --grid-definition FILE states",
        ),
        (args.grid_definition and not args.grid, "--grid-definition plans no --grid"),
    ):
        if refused:
            raise ValueError(why)
    exported = {i for i, _ in args.client_run}
    if stray := sorted(exported - named):
        raise ValueError(f"--client-run names {', '.join(stray)}, which is no input")
    if unnamed := sorted(named - exported):
        raise ValueError(
            f"{', '.join(unnamed)} is named by no --client-run; name the client "
            "runs its server served, so each measurement records the traces they sent"
        )


def _sampling(text: str) -> dict:
    from atom.sampling_params import SamplingParams

    try:
        sampling = json.loads(text)
        SamplingParams(**sampling)
    except (ValueError, TypeError) as error:
        raise ValueError(
            f"--sampling {text!r} is not a JSON object of SamplingParams fields: "
            f"{error}"
        ) from error
    return sampling


def atom_config(atom_args, server: bool = False):
    """ATOM's `Config` for its arguments, as `LLMEngine` resolves one and
    `CoreManager` rewrites it under DP attention, each TP rank a DP rank of TP 1:
    a server's node runs the DP ranks `CoreManager` starts there, and the
    benches run the whole group on one host.
    With `server`, `atom_args` are a server's, whose own flags pass unread, and
    no tokenizer loads; otherwise the server's flags `compass record` passes it
    (`measure_memory.FRONTEND`) are taken and dropped, any other flag the engine
    does not define refuses, and the config gets the tokenizer's ids and the
    end-of-sequence id apart from the stop ids."""
    import dataclasses

    from atom.compass.bench.ranks import layout
    from atom.compass.drive.measure_memory import FRONTEND
    from atom.compass.drive.runfile import _engine_args
    from atom.config import Config
    from atom.model_engine.arg_utils import EngineArgs
    from atom.model_engine.engine_core_mgr import iter_dp_rank_assignments
    from atom.model_engine.llm_engine import load_tokenizer

    parser = EngineArgs.add_cli_args(FlexibleArgumentParser(prog=f"{PROG} --"))
    for flag, kind in FRONTEND.items():
        parser.add_argument(flag, **kind)
    parsed = _engine_args(atom_args) if server else parser.parse_args(atom_args)
    engine_args = EngineArgs.from_cli_args(parsed)
    names = {f.name for f in dataclasses.fields(Config)}
    kwargs = engine_args._get_engine_kwargs().items()
    config = Config(engine_args.model, **{k: v for k, v in kwargs if k in names})
    if config.enable_dp_attention:
        # The engines `CoreManager` starts on this node, each its (global,
        # local) DP rank, the first the node's lowest.
        node = iter_dp_rank_assignments(config)
        layout(config)
        if server:
            config.parallel_config.data_parallel_size_local = len(node)
            config.parallel_config.data_parallel_rank = node[0][0]
    if server:
        return config
    tokenizer = load_tokenizer(config.model, config.trust_remote_code)
    config.bos_token_id = tokenizer.bos_token_id
    config.eos_token_id = tokenizer.eos_token_id
    config.stop_token_ids = sorted(set(config.stop_token_ids) - {config.eos_token_id})
    return config


def start_bench(config, num_blocks: int):
    """The bench, started on `config` with a pool of `num_blocks` KV blocks;
    imported here, since it needs a driver."""
    from atom.compass.bench.engine import Bench

    return Bench(config, num_blocks)


def _points(args, store, engine: dict, sampling: dict, out: Path) -> tuple[dict, int]:
    """Each distinct view to measure, by its digest: replayed views drawn as
    `--sample` says, or the ragged views composed from them, then planned
    points; and the largest KV pool an input's engine states."""
    read = read_inputs(args, [*args.inputs, *args.grid], store, engine, sampling)
    points = {}
    for name in args.inputs:
        source, lines, _ = read[name]
        for line in lines:
            _add(points, line, "replay", [source], line["traces"])
    strata = {}
    for view, point in points.items():
        strata.setdefault(_stratum(point["view"]), []).append(view)
    if isinstance(args.sample, int):
        drawn, rng = set(), random.Random(args.seed)
        for stratum in sorted(strata):
            views = sorted(strata[stratum])
            drawn.update(rng.sample(views, min(args.sample, len(views))))
        points = {v: p for v, p in points.items() if v in drawn}
    if args.ragged:
        points = _ragged(points, [read[n] for n in args.inputs], args.seed)
    if args.grid:
        composed = [(v, p["view"]) for v, p in points.items()] if args.ragged else []
        grid = grid_plan(args, read, composed)
        (out / "plan.json").write_text(json.dumps(grid["plan"], indent=1))
        planned = grid["plan"]
        print(
            f"the plan: {len(planned['lines'])} points, "
            f"{len(planned['infeasible'])} past a limit and not measured, "
            f"{len(planned['unplaced'])} steps outside the grid "
            f"({out / 'plan.json'})",
            flush=True,
        )
        made = grid["provenance"]
        for line in grid["plan"]["lines"]:
            _add(points, line, "plan", made["inputs"], made["traces"], made["run"])
    return points, max(r[2]["num_kvcache_blocks"] for r in read.values())


def _ragged(points: dict, read: list, seed) -> dict:
    """The ragged views of each decode view of `points`, from the decode rows
    of `read`'s lines, each input's source, lines and engine."""
    lines = [line for _, kept, _ in read for line in kept]
    contexts = sorted({c for line in lines for _, c, decode in line["rows"] if decode})
    engine = read[0][2]
    blocks = max(stated["num_kvcache_blocks"] for *_, stated in read)
    size, top = engine["kv_cache_block_size"], engine["max_model_len"]
    block_drafter = (engine.get("speculative") or (None,))[0] == "dspark"
    sources, traces = [s for s, *_ in read], {t for x in lines for t in x["traces"]}
    out = {}
    for digest, point in sorted(points.items()):
        if _kind(point["view"]).startswith("decode"):
            rng = random.Random(f"{seed} {digest}")
            for view in ragged.compose(
                point["view"],
                contexts,
                rng,
                blocks,
                size,
                top,
                engine["ladder"],
                block_drafter=block_drafter,
            ):
                _add(out, view, "ragged", sources, traces)
    return out


def read_inputs(args, names, store, engine: dict, sampling: dict, against=AGAINST):
    """Each input of `names`, read with the client runs `args.client_run` names
    for it under `args.sessions`, refusing as `_read` does."""
    read = {}
    for n in dict.fromkeys(names):
        runs = [e for r, e in args.client_run if Path(r) == Path(n)]
        read[n] = _read(n, store, runs, args.sessions, engine, sampling, against)
    return read


def grid_plan(args, read: dict, views=()) -> dict:
    """The step grid: the `definition` `args.grid_definition` holds and its
    digest, `grid`; the `plan` over the lines of the `--grid` inputs, as
    `read_inputs` read them into `read`, for the engine they state, and over
    the ragged `views`, named lines; and the plan's `provenance`: its digest,
    `run`, those inputs as sources and the traces their lines served."""
    definition = json.loads(Path(args.grid_definition).read_text())
    steps = [(f"{n} step {x['step']}", x) for n in args.grid for x in read[n][1]]
    engines = {f"{n} dp0": read[n][2] for n in args.grid}
    planned = plan(definition, steps, engines, views)
    provenance = {
        "source": "plan",
        "run": _digest(planned),
        "inputs": sorted({read[n][0] for n in args.grid}),
        "traces": sorted({t for _, x in steps for t in x["traces"]}),
    }
    grid = {"definition": definition, "grid": _digest(definition), "plan": planned}
    return grid | {"provenance": provenance}


def _read(name: str, store, client_runs, sessions, engine, sampling, against=AGAINST):
    """The input `name`: what names it as a source, its kept lines, and the
    engine its rank states. One whose engine, a record's over its level
    (`recorded_level`), or sampling is not `engine` or `sampling` refuses,
    naming what states those as `against` does."""
    path = Path(name)
    with tempfile.TemporaryDirectory() as unpacked:
        if (path / "run.json").is_file():
            steps = path / "steps"
            files = sorted(steps.glob("dp*[0-9].jsonl"), key=lambda p: int(p.stem[2:]))
            source = ", ".join(map(file_digest, files))
        else:
            key = store.key_from_token(Kind.SHAPE_POPULATION, name)
            entry, steps, source = store.read(key), Path(unpacked), str(key)
            for rank in entry.topology.ranks():
                for stem, extension, file in (
                    ("steps", "jsonl", f"dp{rank.dp}.jsonl"),
                    ("engine", "json", f"dp{rank.dp}.engine.json"),
                ):
                    (steps / file).write_bytes(entry.read_member(stem, rank, extension))
        record = served(steps, client_runs)
        if len(record) != 1:
            raise ValueError(
                f"{name} holds {len(record)} DP ranks, and the bench measures one"
            )
        ((rank, lines),) = record.items()
        stated = json.loads((steps / f"dp{rank}.engine.json").read_text())
    ran = recorded_level(path / "steps" / "dp0.jsonl") | keyed_of(stated)
    differ = [
        k for k in sorted(ran.keys() | engine.keys()) if ran.get(k) != engine.get(k)
    ]
    if differ:
        raise ValueError(
            f"{name} ran under another engine than {against[0]}: "
            + "; ".join(
                f"{k} {ran.get(k)!r} there, {engine.get(k)!r} here" for k in differ
            )
        )
    if sessions:
        exports = [json.loads(Path(c).read_text()) for c in client_runs]
        pairs = sorted(
            {(t, s) for e in exports for t, s in e.get("sessions", {}).items()}
        )
        given = dict(pairs)
        if twice := sorted({t for t, s in pairs if given[t] != s}):
            raise ValueError(
                f"the client runs naming {name} give "
                + "; ".join(
                    f"{t} sessions {' and '.join(str(s) for u, s in pairs if u == t)}"
                    for t in twice
                )
                + ", and --sessions selects a trace by its one session"
            )
        traces = sorted({r["trace"] for e in exports for r in e["requests"].values()})
        if unnumbered := [t for t in traces if t not in given]:
            raise ValueError(
                f"--sessions selects traces by the session their client run's "
                f"export states, and {name} served {', '.join(unnumbered)}, which "
                "no export gives one"
            )
        chosen = {t for t in traces if sessions[0] <= given[t] <= sessions[1]}
    lines = [
        line
        for line in lines
        if not line["is_dummy_run"]
        and (not sessions or line["traces"] and set(line["traces"]) <= chosen)
    ]
    wanted = json.dumps(sampling, sort_keys=True)
    if sent := sorted({p for line in lines for p in line["sampling"]} - {wanted}):
        raise ValueError(
            f"{name} served requests under {' and '.join(sent)}, and {against[1]} "
            f"{wanted}"
        )
    return source, lines, stated


def view_of(line: dict) -> dict:
    """The batch `line` states, as its measurement's key digests it."""
    view = {k: line[k] for k in VIEW}
    return view | {"shared": line["shared"]} if "shared" in line else view


def _add(points: dict, line: dict, origin: str, sources, traces, plan=None):
    view = view_of(line)
    point = points.setdefault(
        _digest(view),
        {"view": view, "origin": origin, "sources": set(), "traces": set()},
    )
    if point["origin"] != origin:
        return  # a replayed view the plan also places stays the replay's alone
    point["sources"].update(sources)
    point["traces"].update(traces)
    if plan:
        point["plan"] = plan


def _digest(document) -> str:
    text = json.dumps(document, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


def _kind(view: dict) -> str:
    """`decode rung R`, `decode eager`, or `prefill with output` or `without`."""
    if all(decode for *_, decode in view["rows"]):
        rung = view["rung"]
        return "decode eager" if rung is None else f"decode rung {rung}"
    return f"prefill {'with' if view['produces_output'] else 'without'} output"


def _stratum(view: dict) -> str:
    kind = _kind(view)
    if kind.startswith("decode"):
        return kind
    return f"{kind} tokens {token_band(sum(query for query, *_ in view['rows']))}"


def _measure(bench, config, sampling, store, command, readings, todo) -> list[str]:
    """Measure and publish each (key, point) of `todo`, its repeats in bursts
    (`window.py`): every point's first burst in one pass, then every point's
    second, a point whose two disagree running on to `window.REPEATS` at once;
    what was refused, and why. It prints the hours each step kind took."""
    from atom.compass.bench import window
    from atom.compass.bench.replay import ReplayRule

    rule, roots, refused = ReplayRule(config, sampling), None, []
    bursts, clock, hours = [[] for _ in todo], [0.0] * len(todo), Counter()
    for second in (False, True):
        for i, (key, point) in enumerate(todo):
            if second and not bursts[i]:
                continue  # refused in the first pass
            try:
                # Built again each pass: a row holds a token at each position.
                batch = rule.batch(point["view"])
            except ValueError as error:
                refused.append(f"{key}: {error}")
                continue
            repeats = window.BURST
            while repeats:
                started = time.monotonic()
                got = bench.measure(batch, repeats)
                clock[i] += time.monotonic() - started
                if "refused" in got:
                    break
                bursts[i].append(got)
                repeats = window.more(bursts[i]) if second else 0
            if "refused" in got:
                refused.append(f"{key}: {got['refused']}")
                bursts[i].clear()
                continue
            if not second:
                continue
            got = window.joined(bursts[i])
            drafter = got.get("draft_median_ns")
            member = {
                **point["view"],
                "seconds": got["median_ns"] / 1e9,
                **({} if drafter is None else {"draft_seconds": drafter / 1e9}),
                "statistic": "median",
                "repeats": got["repeats"],
                "bursts": got["bursts"],
                "discarded": got["discarded"],
                "preload_ns": got["preload_ns"],
                "protocol": got["protocol"],
                "bench_seconds": clock[i],
                "replay_rule": batch.replay_rule,
                "origin": point["origin"],
                "sources": sorted(point["sources"]),
                "traces": sorted(point["traces"]),
            } | ({"plan": point["plan"]} if "plan" in point else {})
            if got["group_needs_eager"]:
                member["group_needs_eager"] = True
            roots = roots or roots_for({"atom": "atom", "aiter": "aiter"})
            now = datetime.datetime.now(datetime.timezone.utc).isoformat()
            store.publish(
                key,
                provenance=Provenance(command, now, roots),
                topology=Topology(),
                conditions=readings,
                gates=GateState.of(),
                members={member_name(*MEMBER): json.dumps(member).encode()},
            )
            kind = key.values["step_kind"]
            hours[kind] += clock[i] / 3600
            ms, n = got["median_ns"] / 1e6, len(got["repeats"])
            named = f"{point['origin']} {kind} {key.token()}"
            print(f"measured {named}: {ms:.3f} ms, {n} repeats", flush=True)
    for kind in sorted(hours, key=_in_order):
        print(f"bench hours, {kind}: {hours[kind]:.3f} h", flush=True)
    return refused
