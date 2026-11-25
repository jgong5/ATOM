# SPDX-License-Identifier: MIT
"""`compass trace ops`: steps of a model, traced with no device, each shape
published as an `op_graph`.

    compass trace ops --model M [--tp N] [--width W ...] [--prefill T B ...]
                      [--whole] [--population P ...] [--symbolic] [--arch A]
                      [--gpu I] --store S [-- ATOM args]

A shape is a decode width (`--width`), a prefill of T tokens over B sequences
(`--prefill`), or every shape the steps of a record directory hold
(`--population`): a decode step at its CUDA-graph rung when it ran at one, else
at its width, and any other step as a prefill of its query tokens over its
rows, and under DP the rows the step's MoE gathered over the group, its line's
`moe_rows`; under a drafter a decode's drafter also at its rows padded to the
batch its later passes ran at, `draft_bs`, else to that rung, and the rows its
first pass gathered, `draft_moe_rows`, and a sampling prefill's or DP middle
chunk's drafter at its `draft_bs` where that differs from its rows; a step that
replays no graph yet pads past its rows, `target_bs`, at that width. Under
`--enable-tbo` a step its DP group split is its two
micro-batches, each a prefill at its own tokens and the group's rows of it,
by the split `validate.bound` rebuilds from every rank's line; a population's
prefill is traced as one forward, as it ran, where ATOM would split a step of
its tokens, a micro-batch's MoE on one stream as under overlap, and with
`--whole` each `--prefill` too. Each is traced at that
concrete shape (`forward`), decoder layers folded
into `Repeat` and forked streams into `Par` (`graph`). Under DP with no
drafter, `--symbolic` then traces each prefill's (T, B) once more, at its
width, with G a symbol, published at `forward.symbolic_shape` as a run's lookup
traces it (`forward.Tracer.symbolic`), so a run reads it for each G it admits;
without DP, or under a drafter, it refuses before any trace.

An `op_graph` is keyed by structure (`forward.structure`: the model's
architecture and config.json digest, `--arch`, the engine config a step law is
keyed by less its ladder and level (`forward.graph_engine`), which holds ATOM's
engine arguments and this process's performance environment, the TP width, the DP
and EP widths when set, the phase, the mode, whether the step produces
output and whether it is a two-batch-overlap micro-batch), by shape, which
names the rows a DP group's MoE gathers only off its default, every rank at
this one's tokens (`forward.Shape`), and by the torch it was traced under and
the digest of its files (`graph_key`): each file under ATOM's and aiter's roots
its process had run or read when its step was traced, with its own digest
(`files_of`). A lookup reads a graph while each of its files reads here as it
did then (`lookup`), so an edit to a file no trace ran misses no graph, and a
trace whose files read otherwise publishes its graph beside the old one.
A prefill shape is published twice, as a middle chunk, which samples nothing,
and as a final one. ATOM's engine arguments after `--`
configure the engine as they would a server's; `--arch` names the declared
device, which the entry also states as its `device` condition: a store holds a
graph per arch of one structure and shape. `--gpu` declares card I of this host
in its place, each property and the stepping as the card reports them
(`driverless.card`), its arch that card's, which an `--arch` beside it must
name: the step still runs no kernel, but picks the kernels it would on that
card. Its members are
``graph`` (`graph.to_json`), ``declared``, what the capture stated in place
of a device and the operators it left out of the graph, each with why, and
``files``. A TP
group's ranks run one graph, so rank 0's is the entry's. The declarations are
process-global, so a process runs one capture, of every shape it was given.
Trace in the environment and torch of the run or bench that reads the graphs:
a reader under another, or one that reads a graph's file otherwise, misses
it, and its refusal names each field and file that differs.
"""

import argparse
import dataclasses
import datetime
import functools
import hashlib
import json
import os
import pathlib
import sys
import tempfile
import time
from types import ModuleType

#: The roots whose files an `op_graph` states (`files_of`), by the name it
#: states each under, to the module that locates it.
ROOTS = {"atom": "atom", "aiter": "aiter"}
#: The kind of the ATOM-source reading of an `op_graph` keyed by its files.
FILES = "files"


def main(argv: list[str]) -> None:
    split = argv.index("--") if "--" in argv else len(argv)
    atom_args = argv[split + 1 :]
    p = argparse.ArgumentParser(prog="compass trace ops", description=__doc__)
    p.add_argument("--model", required=True, help="a model name or directory")
    p.add_argument("--tp", type=int, default=1, help="tensor-parallel width")
    p.add_argument(
        "--width", type=int, action="append", default=[], help="a decode width"
    )
    p.add_argument(
        "--prefill",
        type=int,
        nargs=2,
        action="append",
        default=[],
        metavar=("T", "B"),
        help="a prefill of T tokens over B sequences",
    )
    p.add_argument(
        "--whole",
        action="store_true",
        help="trace each --prefill as one forward where ATOM would split it",
    )
    p.add_argument(
        "--population",
        action="append",
        default=[],
        help="a record directory; every shape its steps hold",
    )
    p.add_argument(
        "--symbolic",
        action="store_true",
        help="under DP, also trace each prefill's (T, B) with G a symbol",
    )
    p.add_argument("--arch", help="the declared device's arch, gfx942 by default")
    p.add_argument("--gpu", type=int, help="declare this card as it reads")
    p.add_argument("--store", required=True, help="the artifact store")
    args = p.parse_args(argv[:split])

    from atom.compass import artifacts as a

    from . import driverless
    from .forward import capture, decode, prefill

    read = None if args.gpu is None else driverless.card(args.gpu)
    if read and args.arch not in (None, read["arch"]):
        raise ValueError(f"card {args.gpu} reads as {read['arch']}, not {args.arch}")
    arch = read["arch"] if read else args.arch or "gfx942"

    store = a.ArtifactStore(args.store)
    shapes = [decode(w) for w in args.width] + [prefill(*s) for s in args.prefill]
    if not shapes and not args.population:
        p.error("name a shape: --width, --prefill or --population")
    command = " ".join(["compass", "trace", "ops", *argv])
    with tempfile.TemporaryDirectory(prefix="compass-capture-") as tmpdir:
        began = time.perf_counter()  # the first shape's seconds include the build
        # A card's reading, `--whole` and `--symbolic` only when given: any
        # other call is as before.
        on = {"read": read} if read else {}
        if args.whole:
            on["whole"] = True
        if args.symbolic:
            on["symbolic"] = True
        captured = capture(
            args.model, args.tp, shapes, tmpdir, atom_args, arch, args.population, **on
        )
        for graph, model, declared in captured:
            seconds = time.perf_counter() - began
            key = publish(store, graph, model, declared, args.tp, atom_args, command)
            print(f"captured {graph.applicability.shape} in {seconds:.1f} s")
            print(f"published {key} as {key.token()}")
            began = time.perf_counter()


def population_shapes(names, tbo=False):
    """The shape of each step the record directories `names` hold, DP's dummy
    runs aside; under two-batch overlap (`tbo`) a split step's two
    micro-batches' (`population_steps`); each drafter's too."""
    return [shape for shape, _ in population_steps(names, tbo, drafters=True)]


def population_steps(names, tbo=False, drafters=False, stage=0):
    """Each step the record directories `names` hold, DP's dummy runs aside,
    of pipeline stage `stage` (above 0, its records ``pp<k>.dp<r>.jsonl``),
    as its shape, the one the op-level backend prices it at
    (`op_graph.step_shape`), with the rows its DP group's MoE
    gathered (`moe_rows`), and its step-record line, marked with how many of
    its group's ranks ran the step as a dummy forward (`group_dummies`) and,
    in a DP group, the most tokens a rank of it ran (`group_max_tokens`). A
    decode step verifies
    as many drafts in each row as its shortest row does. Under two-batch
    overlap (`tbo`, ATOM's `enable_tbo`) a step its ranks split, one rank
    included, by the rule `validate.bound` agrees over every rank's line at
    that step (`bound._split`), is its two micro-batches
    (`step_roofline.micro_batches`),
    each a prefill at its own tokens and the group's rows of it, marked a
    micro-batch (`forward.Shape`), its line holding its own rows and the
    most tokens a rank runs of it. With
    `drafters`, a drafter's shape follows it where they differ: a decode's
    at its rows padded to `draft_bs`, else to its rung, a sampling prefill's
    or a DP middle chunk's with `draft_bs` as its `width`. A target that
    replays no graph yet pads past its rows takes `target_bs` as its `width`."""
    from atom.compass.backends import BatchView, RequestShape
    from atom.compass.backends.op_graph import step_shape
    from atom.compass.backends.step_labels import group_dummies, group_peaks
    from atom.compass.backends.step_roofline import micro_batches
    from atom.compass.validate.bound import _split

    from .forward import decode, prefill

    for name in names:
        path = pathlib.Path(name)
        if not (path / "run.json").is_file():
            raise ValueError(f"{name} is not a record directory: it has no run.json")
        prefix = f"pp{stage}." if stage else ""
        paths = sorted((path / "steps").glob(f"{prefix}dp*[0-9].jsonl"))
        ranks = [[json.loads(t) for t in p.read_text().splitlines()] for p in paths]
        dp, splits, dummies = len(ranks), {}, group_dummies(ranks)
        # a line that states no rows, as a bare dummy run's, runs none
        peaks = group_peaks([s for s in lines if "rows" in s] for lines in ranks)
        if tbo and ranks:
            at = [{s["step"]: s for s in lines} for lines in ranks]
            for step in set.intersection(*map(set, at)):
                splits[step] = _split([r[step] for r in at])
        for rank, line in ((r, s) for r, lines in enumerate(ranks) for s in lines):
            if line["is_dummy_run"]:
                continue
            line["group_dummies"] = dummies[line["step"]]
            if dp > 1:
                line["group_max_tokens"] = peaks[line["step"]]
            rows = [(row[0], row[2]) for row in line["rows"]]
            split = splits.get(line["step"])
            try:
                drafts = min(q for q, _ in rows) - 1
                phase, tokens, seqs = step_shape(rows, line["rung"], drafts)
                if split:
                    held = [RequestShape(q, c, bool(d)) for q, c, d in line["rows"]]
                    view = BatchView(tuple(held), tbo_tokens=split, dp_rank=rank)
                    halves = micro_batches(view)
            except ValueError as exc:
                raise ValueError(f"{name} step {line['step']} {exc}") from None
            width, padded = line.get("draft_bs"), line.get("target_bs")
            if phase == "decode":
                target = decode(tokens, line.get("moe_rows"))._replace(width=padded)
                target = target.at(dp, drafts)
                yield target, line
                draft = decode(width or tokens, line.get("draft_moe_rows"), len(rows))
                if drafters and drafts and (draft := draft.at(dp, drafts)) != target:
                    yield draft, line
            elif not split:
                yield prefill(tokens, seqs, line.get("moe_rows"), padded).at(dp), line
                draft = prefill(tokens, seqs, line.get("draft_moe_rows"), width)
                if drafters and (line.get("produces_output") or dp > 1) and draft.width:
                    yield draft.at(dp), line
            else:
                for half, each in zip(halves, split):
                    rows = [(r.query_tokens, r.decode) for r in half.requests]
                    _, tokens, seqs = step_shape(rows, None)
                    own = {"rows": [dataclasses.astuple(r) for r in half.requests]}
                    own |= {"group_max_tokens": max(each)} if dp > 1 else {}
                    shape = prefill(tokens, seqs, half.moe_rows).at(dp)
                    yield shape._replace(micro_batch=True), line | own


@functools.cache
def traced_under() -> dict:
    """This process's ATOM source (`run.atom_source`) and torch, by axis: what
    every `op_graph` it traces is keyed and fingerprinted under, the ATOM
    source only for one that states no files, and every one it reads is
    looked up under. Read once, as a process traces and reads under the code
    it imported."""
    import torch

    from atom.compass.artifacts import Reading
    from atom.compass.run import atom_source

    return {"atom_src": atom_source(), "torch": Reading.stated(torch.__version__)}


def graph_key(structure: str, shape: str, files: dict | None = None):
    """The key of the `op_graph` at `structure` and `shape` traced under this
    process's torch and, with the `files` its trace ran and read (`files_of`),
    their digest (`files_reading`), else this process's ATOM source
    (`traced_under`): a graph traced under another torch or ATOM source, or
    whose files read otherwise, is another entry."""
    from atom.compass import artifacts as a

    under = traced_under()
    source = under["atom_src"] if files is None else files_reading(files)
    return a.Key.of(
        a.Kind.OP_GRAPH,
        structure=structure,
        shape=shape,
        source_root=f"{source.kind}-{source.value}",
        torch=under["torch"].value,
    )


def files_reading(files: dict):
    """The ATOM-source reading of an `op_graph` whose trace ran and read
    `files` (`files_of`): their digest, of kind `FILES`."""
    from atom.compass.artifacts import Reading

    text = json.dumps(files, sort_keys=True, separators=(",", ":"))
    return Reading(FILES, hashlib.sha256(text.encode()).hexdigest())


def files_of(reads) -> dict:
    """The files under `ROOTS` this process has run or read, as
    ``{root: {path under it: file_reading}}``: each module it imported, whose
    body ran, and each file of `reads` (`forward.watch_reads`). A module's
    path is read off its own namespace, past any `__getattr__` it defines."""
    modules = [m for m in list(sys.modules.values()) if isinstance(m, ModuleType)]
    paths = [vars(m).get("__file__") for m in modules]
    held = {}
    for path in sorted({p for p in paths if isinstance(p, str)} | set(reads)):
        if (place := _place(path)) and (reading := file_reading(*place)):
            held.setdefault(place[0], {})[place[1]] = reading
    return {root: dict(sorted(held[root].items())) for root in sorted(held)}


@functools.cache
def _roots() -> dict:
    """Each of `ROOTS`, its directory resolved, located without importing it."""
    from atom.compass.artifacts import module_root

    return {name: pathlib.Path(module_root(m)).resolve() for name, m in ROOTS.items()}


@functools.cache
def _place(path: str) -> tuple[str, str] | None:
    """The root of `ROOTS` that holds the file at `path`, and its path under
    that root; None for a file of no root, or one Python compiled."""
    real = pathlib.Path(path).resolve()
    for name, root in _roots().items():
        if real.is_relative_to(root) and real.suffix not in (".pyc", ".so"):
            return name, str(real.relative_to(root))
    return None


@functools.cache
def file_reading(root: str, path: str) -> str | None:
    """The sha256 of the file at `path` under the root `root` of `ROOTS` here,
    an ATOM file's as ATOM's source reading takes it (`run.atom_text`); None
    for a file that reading leaves out, or that is not there. Read once a
    process, which runs the files it read."""
    from atom.compass.run import atom_text

    if (held := _roots().get(root)) is None:
        return None
    try:
        if root == "atom":
            text = atom_text(held / path, pathlib.Path(path))
        else:
            text = (held / path).read_bytes()
    except OSError:
        return None
    return None if text is None else hashlib.sha256(text).hexdigest()


def lookup(store, structure: str, shape: str):
    """The key of the `op_graph` `store` holds at `structure` and `shape` that
    answers in this process, None for none: one traced under its ATOM source
    and torch (`graph_key`), else one traced under its torch whose files each
    read here as they did then (`changed`)."""
    key = graph_key(structure, shape)
    if store.directory_for(key).exists():
        return key
    return lookup_by_files(store, structure, shape)


def lookup_by_files(store, structure: str, shape: str):
    """The key of an `op_graph` `store` holds at `structure` and `shape`,
    traced under this process's torch, whose files each read here as they did
    then (`changed`); None for none."""
    torch = traced_under()["torch"].value
    keyed = _keyed_by_files(store).get((structure, shape, torch), ())
    return next((key for key in keyed if not changed(store, key)), None)


def changed(store, key) -> tuple[str, ...]:
    """Each file the `op_graph` at `key` in `store` states whose reading here
    (`file_reading`) is not its trace's, as ``<root>/<path>``."""
    return _here(str(store.root), key)[1]


def source_of(store, key):
    """The ATOM-source reading the `op_graph` at `key` in `store` loads under
    here: its files' as they read here, for one keyed by them; else this
    process's ATOM source."""
    if key.values["source_root"].startswith(f"{FILES}-"):
        return _here(str(store.root), key)[0]
    return traced_under()["atom_src"]


@functools.cache
def _here(root: str, key):
    """The reading of the files the `op_graph` at `key` in the store at `root`
    states, as they read here (`files_reading`), and each that reads
    otherwise than in its trace; read once a process."""
    from atom.compass.artifacts import ArtifactStore, Topology

    entry = ArtifactStore(root).read(key)
    rank = next(iter(Topology().ranks()))
    files = json.loads(entry.read_member("files", rank, "json"))
    here = {r: {p: file_reading(r, p) for p in held} for r, held in files.items()}
    moved = [
        f"{r}/{p}" for r, held in files.items() for p in held if here[r][p] != held[p]
    ]
    return files_reading(here), tuple(moved)


#: Per `op_graph` folder, its modification time when last listed, the entries
#: listed, and the keys of those keyed by their files (`_keyed_by_files`).
_LISTED: dict = {}


def _keyed_by_files(store) -> dict:
    """The keys of `store`'s `op_graph` entries keyed by their files, by
    structure, shape and torch: each entry's document read once a process,
    from 16 threads, and the folder listed again when its modification time
    moves, as a publish into it moves it."""
    from concurrent.futures import ThreadPoolExecutor

    from atom.compass.artifacts import Kind

    folder = store.root / Kind.OP_GRAPH.value
    try:
        moved = folder.stat().st_mtime_ns
    except FileNotFoundError:
        return {}
    at, seen, keyed = _LISTED.get(folder, (None, set(), {}))
    if moved != at:
        new = [n for n in os.listdir(folder) if not n.startswith(".") and n not in seen]
        with ThreadPoolExecutor(16) as pool:
            for key in filter(None, pool.map(_files_key, (folder / n for n in new))):
                held = key.values
                at_key = held["structure"], held["shape"], held["torch"]
                keyed.setdefault(at_key, []).append(key)
        _LISTED[folder] = moved, seen | set(new), keyed
    return keyed


def _files_key(place: pathlib.Path):
    """The key the entry at `place` restates when it is an `op_graph` keyed by
    its files; None for another, or for no entry."""
    from atom.compass.artifacts import ArtifactRefusal, Key, Kind

    try:
        key = json.loads((place / "entry.json").read_bytes())["key"]
        if str(key.get("source_root")).startswith(f"{FILES}-"):
            return Key.of(Kind.OP_GRAPH, **key)
    except (OSError, ValueError, KeyError, TypeError, ArtifactRefusal):
        pass
    return None


def publish(store, graph, model, declared, tp, atom_args, command):
    """Publish `graph` of `model` at TP `tp` as one `op_graph` in `store`, at
    its `graph_key`: keyed by the files `declared` states its trace ran and
    read (`files_of`) when it states them, which it holds as its ``files``
    member."""
    from atom.compass import artifacts as a
    from atom.compass.artifacts.keys import NO_PTL
    from atom.compass.run import host_cpu

    from .graph import to_json

    applicability = graph.applicability
    files = declared.get("files")
    key = graph_key(applicability.structure, applicability.shape, files)
    under = traced_under()
    if files is not None:
        under = under | {"atom_src": files_reading(files)}
    roots = a.roots_for({"atom": "atom", "aiter": "aiter"})
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    rank = next(iter(a.Topology().ranks()))
    stated = {name: value for name, value in declared.items() if name != "files"}
    members = {
        a.member_name("graph", rank, "json"): to_json(graph).encode(),
        a.member_name("declared", rank, "json"): json.dumps(stated).encode(),
    }
    if files is not None:
        members[a.member_name("files", rank, "json")] = json.dumps(files).encode()
    store.publish(
        key,
        provenance=a.Provenance(command, now, roots),
        topology=a.Topology(),
        conditions=a.Conditions.of(
            software_stack=a.Reading.of_source_root(roots[1]),
            **under,
            model=model,
            device=declared["arch"],
            ptl=NO_PTL,
            host_cpu=host_cpu(),
            engine_config=json.dumps(
                {"enforce_eager": True, "tp": tp, "atom_args": atom_args}
            ),
        ),
        gates=a.GateState.of(),
        members=members,
    )
    return key
