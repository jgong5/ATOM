# SPDX-License-Identifier: MIT
"""What a fit command shares: its inputs, and publishing what it fitted.

A fit reads record directories, each with the client runs its server served,
by their exports (`--client-run RECORD EXPORT`, in the order they ran), and a
fit taking replayed batches reads step measurements too, each named by the
token `Key.token` prints; it loads their samples under one configuration
(`samples.load`). A record no `--client-run` names refuses: its steps would
name no trace they served. A deployment's export names decode's response ids,
so a RECORD that is its `prefill/`, `decode/` beside it, has its export's
requests renamed as prefill's record names them, as grading renames them.
The platform is the one the inputs ran on: the device, software stack, PTL
state and torch the records state, or the step measurements' keys, which must
agree; no flag names it. `--ptl` states the PTL state of records that carry
none, and refuses for records that carry one; the report records it as
`stated_ptl`. The configuration gains `platform`, whose fields the fit's row
marks its key and params state, and `readings`, those a run on that platform
takes with the inputs' source root, model and engine, under which each
measurement is loaded. `--ops` takes the profiler's floor from the
`platform_readings` entry the store holds at the inputs' platform.

`publish` hands off one store entry whose members are `params`, what a run
file carries, and `report`, the fit's report, under the readings it is
given, and writes both, with the entry's key and token, as `<kind>.json` in
`--out`; its provenance names the command and the ATOM and aiter trees it ran
from.
"""

import argparse
import datetime
import functools
import json
from pathlib import Path

from atom.compass.artifacts import (
    AXES,
    PLATFORM,
    ArtifactStore,
    Conditions,
    GateState,
    Kind,
    Provenance,
    Reading,
    Topology,
    keyed_platform,
    member_name,
    platform_of,
    roots_for,
)
from atom.compass.backends.json_file import read_json
from atom.compass.backends.step_labels import Floor
from atom.compass.fit.device import readings
from atom.compass.fit.samples import UNRECORDED, load, run_fields
from atom.compass.parity import busy
from atom.compass.run import host_cpu
from atom.compass.spec.layers import stack_of
from atom.compass.validate.runs import SIDES, named_by_prefill

#: What a trace run's profiler adds to a kernel shorter than it.
FLOOR = "device.profiler_floor_s"
#: The rank a fit whose key states no layout writes at.
RANK = Topology().ranks()[0]
#: The engine fields the `measure ops` points a trace run's drafter kernels take
#: their own lengths from share with its records: what changes a kernel.
SAME_OPS = (
    "tensor_parallel_size",
    "data_parallel_size",
    "enable_dp_attention",
    "enable_expert_parallel",
    "kv_cache_dtype",
    "kv_cache_block_size",
    "index_cache_dtype",
    "speculative",
)


def ranks(key) -> tuple:
    """Every rank of the layout `key` states, one when it states none; a fit
    has one writer, so each rank holds the same file."""
    return Topology(**{a: key.values.get(a, 1) for a in AXES}).ranks()


def parser(prog: str, description: str) -> argparse.ArgumentParser:
    """A fit command's parser: its inputs, their client runs, the store and `--out`."""
    p = argparse.ArgumentParser(prog=prog, description=description)
    p.add_argument(
        "inputs",
        nargs="+",
        metavar="INPUT",
        help="record directories, and step_measurement tokens where the fit takes them",
    )
    p.add_argument(
        "--client-run",
        nargs=2,
        action="append",
        default=[],
        metavar=("RECORD", "EXPORT"),
        help="a client run RECORD's server served, by its export; every RECORD has one",
    )
    p.add_argument(
        "--ptl",
        choices=("enabled", "disabled"),
        help="the PTL state of records that carry none",
    )
    p.add_argument("--store", required=True, help="the artifact store's root")
    p.add_argument("--out", required=True, help="where the readable copy goes")
    return p


def inputs(args, sources, ops=()) -> tuple[list, dict]:
    """The samples of `args`' inputs, records each with its client runs and,
    where `sources` holds `replay`, step measurements, and their configuration;
    a fit taking only `sources` refuses any other input. With `ops`, `measure
    ops --out` directories read as `fit ops` reads them, a trace run's drafter
    labels lose the profiler's floor (`parity.busy.floored`), `_floor`, each
    kernel's own length from their points, which refuse unless they ran the
    records' model, platform, source root and `SAME_OPS`."""
    from atom.compass.fit import ops as fit_ops

    runs = {Path(i): [] for i in args.inputs if (Path(i) / "run.json").is_file()}
    tokens = [i for i in args.inputs if Path(i) not in runs]
    if tokens and "replay" not in sources:
        raise ValueError(
            f"{', '.join(tokens)} is not a record directory (no run.json), and "
            f"this fit takes {' and '.join(sources)} samples only"
        )
    for record, export in args.client_run:
        if Path(record) not in runs:
            raise ValueError(f"--client-run names {record}, which is not a RECORD")
        decode = Path(record).parent / SIDES[1]
        if Path(record).name == SIDES[0] and decode.is_dir():
            export = named_by_prefill(read_json(export), record, decode)["requests"]
        runs[Path(record)].append(export)
    unnamed = [str(d) for d, exports in runs.items() if not exports]
    if unnamed:
        raise ValueError(
            f"{', '.join(unnamed)} is named by no --client-run; name the client "
            "runs its server served, so the fit records the traces they sent"
        )
    store = ArtifactStore(args.store)
    measurements = [store.key_from_token(Kind.STEP_MEASUREMENT, t) for t in tokens]
    floored = None
    if ops:
        points, measured, keyed = fit_ops.read(store, map(Path, ops))
        floor = Floor(_floor(store, runs, measurements, args.ptl), points)
        floored = functools.partial(busy.floored, floor=floor)
    samples, config = load(runs, measurements, args.store, floored)
    if args.ptl:
        if config.get("ptl") != UNRECORDED:
            raise ValueError(
                f"--ptl {args.ptl} states the PTL state of records that carry "
                f"none, and these carry {config.get('ptl')!r}"
            )
        config["ptl"] = args.ptl
    elif config.get("ptl") == UNRECORDED:
        raise ValueError(
            "the records carry no PTL state; --ptl states their cards' state"
        )
    ran = platform(config) if runs else _keyed(measurements)
    if ops:
        want = {"model": config["model"], **keyed_platform(Kind.OP_MEASUREMENT, ran)}
        want["source_root"] = config["source_root"]
        got = {k: keyed.get(k) for k in want}
        want |= {k: config["engine_config"].get(k) for k in SAME_OPS}
        got |= {k: measured["engine_config"].get(k) for k in SAME_OPS}
        differ = [f"{k} {got[k]!r}, not {v!r}" for k, v in want.items() if got[k] != v]
        if differ:
            raise ValueError(
                "the --ops points ran under another configuration than the "
                "records: " + "; ".join(differ)
            )
    here = Conditions.of(
        **ran,
        host_cpu=host_cpu(),
        atom_src=Reading(*config["source_root"].split("-", 1)),
        model=config["model"],
        engine_config=json.dumps(config["engine_config"], sort_keys=True),
    )
    for key in measurements:
        store.load(key, conditions=here, gates=GateState.of())
    return samples, {
        **config,
        "platform": ran,
        "readings": here,
    }


def _floor(store, runs, measurements, ptl) -> float:
    """`FLOOR` in the readings `store` holds at the platform the inputs ran
    on: the first record's, its PTL state `ptl` where it records none, else the
    step measurements'; `load` holds every other input to it."""
    if runs:
        run = json.loads((next(iter(runs)) / "run.json").read_text())
        device = run_fields(run)["device"]
        on = platform_of(device, run["stack"], run.get("ptl", ptl))
    else:
        on = _keyed(measurements)
    key, entry = readings(store, on["device"], on["ptl"], stack_of(on))
    if FLOOR not in entry:
        raise ValueError(f"{key} as {key.token()} states no {FLOOR}, which --ops takes")
    return entry[FLOOR]


def _keyed(measurements) -> dict:
    """The platform the step measurements' keys state; two refuse, naming each."""
    each = {
        str(k): {a.field: k.values[a.field] for a in PLATFORM} for k in measurements
    }
    if len({json.dumps(p, sort_keys=True) for p in each.values()}) > 1:
        raise ValueError(
            "the step measurements ran on several platforms: "
            + "; ".join(f"{k} on {p}" for k, p in each.items())
        )
    return next(iter(each.values()))


def runs(samples) -> list[dict]:
    """Each run's provenance, in sample order: its first sample's, less the
    requests that step served, with the traces every sample of the run served."""
    held: dict[str, dict] = {}
    for s in samples:
        first = {k: v for k, v in s.provenance.items() if k != "requests"}
        run = held.setdefault(s.provenance["run"], {**first, "traces": set()})
        run["traces"] |= set(s.provenance["traces"])
    return [{**run, "traces": sorted(run["traces"])} for run in held.values()]


def platform(config) -> dict:
    """The platform `config`, the inputs' configuration, states."""
    return platform_of(config["device"], config["stack"], config["ptl"])


def conditions(config) -> Conditions:
    """The readings `config`, the inputs' configuration, states for every axis."""
    return Conditions.of(
        **platform(config),
        atom_src=Reading(*config["source_root"].split("-", 1)),
        model=config["model"],
        host_cpu=config["host_key"]["host_cpu"],
        engine_config=json.dumps(config["engine_config"], sort_keys=True),
    )


def publish(args, command: str, key, readings: Conditions, params, report):
    """Hand `params` and `report` off at `key` in `args.store` under `readings`,
    and write their readable copy in `args.out`."""
    if args.ptl:
        report = {**report, "stated_ptl": args.ptl}
    members, at = {"params": params, "report": report}, ranks(key)
    ArtifactStore(args.store).publish(
        key,
        provenance=Provenance(
            command,
            datetime.datetime.now(datetime.timezone.utc).isoformat(),
            roots_for({"atom": "atom", "aiter": "aiter"}),
        ),
        topology=at[0].topology,
        conditions=readings,
        gates=GateState.of(),
        members={
            member_name(stem, rank, "json"): json.dumps(body).encode()
            for stem, body in members.items()
            for rank in at
        },
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    copy = {"key": key.values, "token": key.token(), **members}
    (out / f"{key.kind}.json").write_text(json.dumps(copy, indent=1) + "\n")
    print(f"published {key} as {key.token()}")
