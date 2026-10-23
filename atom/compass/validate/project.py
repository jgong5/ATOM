# SPDX-License-Identifier: MIT
"""`compass validate project`: real runs' steps priced on a device none of them
ran on, at step and op level, published as a `projection`.

    compass validate project --model M [--tp N] --machine SPEC --store S
        -- RECORD [ATOM args] [-- RECORD [ATOM args] ...]

A cell is a real run's directory, its step record in `steps/`, and the ATOM
arguments it ran under. Its steps are priced as `validate bound` prices them,
on SPEC, the op_graphs S holds for SPEC's arch; nothing is compared with the
measured time, which is another device's. Each cell is priced in a process of
its own, since a build declares one engine per process.

The `projection` published in S states, per cell and granularity, the steps
priced and refused by reason, the projected seconds per phase, and the tokens
a second over the steps priced; per granularity, the cells ranked by those
tokens a second, its first (top-1) and first three (top-3); and each field of
SPEC whose origin is `assumed` that a price read, with its value, and for an
assumed arch the compute units and stepping its op_graphs were traced at.
Its `band` re-prices every cell at 0.8 and 1.2 times each carried ratio a price
read, one at a time (`band`), and ranks the cells again.

`projected` is the method that makes SPEC for a device that never ran, applied
to one that did, for `validate bound --ratios-from`.
"""

import argparse
import datetime
import hashlib
import json
import multiprocessing
import os
from pathlib import Path

from atom.compass.artifacts import (
    ArtifactStore,
    Conditions,
    GateState,
    Key,
    Kind,
    Provenance,
    Reading,
    Topology,
    machine_spec_key,
    member_name,
    roots_for,
)
from atom.compass.spec import MachineSpec

from . import bound

PROG = "compass validate project"
#: What a projected device takes from the probed one as it stands, no
#: datasheet stating it.
CARRIED = (
    "host.",
    "device.kernel_floor_s",
    "device.software_pinned_to.",
    "interconnect.intra_node.link_latency_s",
    "interconnect.inter_node.link_latency_s",
    "interconnect.router_relay_s",
)
CURVE = "device.memory.bandwidth_curve"
BANDWIDTH = "device.memory.bandwidth_bytes_per_s"


def projected(target: MachineSpec, source: MachineSpec) -> MachineSpec:
    """`target` at `source`'s probed ratios: target's peaks and what names it,
    source's derates and its bandwidth curve over its datasheet bandwidth
    times target's, and source's values where no datasheet states one
    (`CARRIED`), each assumed. A compute format source states no derate for is
    left out, so a price that reads its rate is refused."""

    def taken(path: str) -> bool:
        return "derate" in path.split(".") or path == CURVE or path.startswith(CARRIED)

    scale = target.value(BANDWIDTH) / source.value(BANDWIDTH)
    values = {p: v for p, v in target.values.items() if not taken(p)}
    values |= {p: v for p, v in source.values.items() if taken(p)}
    values[CURVE] = [[size, rate * scale] for size, rate in source.value(CURVE)]
    for fmt in ("bf16", "fp8", "fp4"):
        if f"device.compute.derate.{fmt}" not in values:
            values.pop(f"device.compute.{fmt}_flops", None)
    kept = target.values["provenance.origins"].items()
    values["provenance.origins"] = {
        **{p: o for p, o in kept if p in values and not taken(p)},
        **{
            p: {"origin": "assumed"}
            for p in source.values["provenance.origins"]
            if taken(p)
        },
    }
    values["name"] = f"{target.values['name']} at {source.values['name']} ratios"
    echo = MachineSpec(values, source.tokenizers).echo()
    return MachineSpec.from_mapping(echo)


class Reads:
    """A machine spec that keeps the path of each field a price reads."""

    def __init__(self, spec: MachineSpec):
        self.spec, self.read = spec, set()

    def value(self, path: str):
        self.read.add(path)
        return self.spec.value(path)

    def __getattr__(self, name):
        return getattr(self.spec, name)


def cell(model: str, tp: int, machine: dict, store: str, record: str, atom_args):
    """One cell priced at both granularities on the spec document `machine`:
    its report per granularity, the model's and engine's identity, its step
    records' digest and the spec fields its prices read."""
    from atom.compass.run import _file_digest
    from atom.compass.run.step_grid import keyed

    spec = Reads(MachineSpec.from_mapping(machine))
    config, _, priced = bound.backends(model, tp, atom_args, spec, store, [record])
    report = bound.gate(
        [record], priced, config.enable_tbo, config.speculative_config is not None
    )
    steps = hashlib.sha256()
    for path in sorted(Path(record, "steps").glob("dp*[0-9].jsonl")):
        steps.update(path.read_bytes())
    levels = {}
    for level, r in report.items():
        rate = r["ranking"] and r["ranking"]["tokens_per_s"][record]["analytical"]
        held = ("steps", "steps_refused", "refused", "analytical_s")
        levels[level] = {k: r[k] for k in held} | {"tokens_per_s": rate}
    return {
        "model": _file_digest(config.model, "config.json"),
        "engine": keyed(config, ladder=False),
        "steps": f"sha256:{steps.hexdigest()}",
        "levels": levels,
        "read": sorted(spec.read),
    }


def _each(jobs: list) -> list:
    """`cell` of each job, each in a process of its own, as many at once as
    this process may use CPUs."""
    spawn = multiprocessing.get_context("spawn")
    width = max(1, min(len(jobs), len(os.sched_getaffinity(0))))
    with spawn.Pool(width, maxtasksperchild=1) as pool:
        return pool.starmap(cell, jobs)


def _ranked(cells: dict) -> dict:
    """Per granularity, the cells by tokens a second, fastest first, and
    their first (top-1) and first three (top-3)."""
    ranking = {}
    for level in ("step", "op"):
        rates = {r: c["levels"][level]["tokens_per_s"] for r, (_, c) in cells.items()}
        order = sorted((r for r, t in rates.items() if t), key=lambda r: -rates[r])
        ranking[level] = {"order": order, "top_1": order[:1], "top_3": order[:3]}
    return ranking


def scaled(machine: MachineSpec, path: str, factor: float):
    """`machine` with the ratio at `path` times `factor`, capped at 1, the
    datasheet; for the bandwidth curve, each point's rate over the datasheet
    bandwidth. Also whether the cap applied."""
    values = dict(machine.values)
    if path == CURVE:
        top = machine.value(BANDWIDTH)
        curve = [[size, rate * factor] for size, rate in machine.value(CURVE)]
        capped = any(rate > top for _, rate in curve)
        values[CURVE] = [[size, min(rate, top)] for size, rate in curve]
    else:
        capped = machine.value(path) * factor > 1
        values[path] = min(machine.value(path) * factor, 1.0)
    return MachineSpec(values, machine.tokenizers), capped


def band(machine: MachineSpec, assumed: dict, jobs: list, base: dict) -> dict:
    """Each carried ratio a price read (an assumed derate, or the bandwidth
    curve over the datasheet bandwidth) at 0.8 and 1.2 times its value, the
    others held: per factor, whether the cap applied, the cells ranked, and
    per granularity whether the order of `base`, the ranking at the carried
    values, holds. A carried value no datasheet states a peak for, as the
    kernel floor or a link latency, is no ratio."""
    ratios = [p for p in assumed if p == CURVE or "derate" in p.split(".")]
    specs = [(p, f, *scaled(machine, p, f)) for p in ratios for f in (0.8, 1.2)]
    results = iter(
        _each([(*j[:2], s.echo(), *j[3:]) for _, _, s, _ in specs for j in jobs])
    )
    out = {}
    for path, factor, _, capped in specs:
        cells = {j[4]: (None, next(results)) for j in jobs}
        ranking = _ranked(cells)
        holds = {k: ranking[k]["order"] == base[k]["order"] for k in ranking}
        out.setdefault(path, {})[str(factor)] = {
            "capped": capped,
            "ranking": ranking,
            "order_holds": holds,
            "tokens_per_s": {
                k: {r: c["levels"][k]["tokens_per_s"] for r, (_, c) in cells.items()}
                for k in ranking
            },
        }
    return out


def projection(machine: MachineSpec, cells: dict) -> dict:
    """The document of `cells`, each a record to its ATOM arguments and its
    `cell` result, priced on `machine`. An assumed arch also names the compute
    units and stepping its op_graphs were traced at, which chose their kernels."""
    from atom.compass.capture.driverless import DEVICES, REVISIONS

    read = sorted(set().union(*(c["read"] for _, c in cells.values())))
    assumed = {
        p: machine.value(p) for p in read if machine.origin(p)["origin"] == "assumed"
    }
    if arch := assumed.get("device.arch"):
        assumed["capture.compute_units"] = DEVICES[arch][0]
        assumed["capture.revision"] = REVISIONS.get(arch, 0)
    return {
        "machine": {"name": machine.values["name"], "digest": machine.digest()},
        "assumed": assumed,
        "cells": {
            r: {"atom_args": a, "engine": c["engine"], **c["levels"]}
            for r, (a, c) in cells.items()
        },
        "ranking": _ranked(cells),
    }


def publish(store: str, document: dict, cells: dict, machine, command: str):
    """Publish `document` in `store` as a `projection`, keyed by its model,
    spec, ATOM tree and cells."""
    import torch

    from atom.compass.run import host_cpu

    model = next(iter(cells.values()))[1]["model"]
    roots = roots_for({"atom": "atom", "aiter": "aiter"})
    atom = next(r for r in roots if r.name == "atom")
    engines = json.dumps([c["engine"] for _, c in cells.values()], sort_keys=True)
    identity = json.dumps([[c["steps"], c["engine"]] for _, c in cells.values()])
    key = Key.of(
        Kind.PROJECTION,
        model=model,
        spec=machine.digest(),
        source_root=f"{atom.revision_kind}-{atom.revision}",
        cells=f"sha256:{hashlib.sha256(identity.encode()).hexdigest()}",
    )
    rank = Topology().ranks()[0]
    ArtifactStore(store).publish(
        key,
        provenance=Provenance(
            command, datetime.datetime.now(datetime.timezone.utc).isoformat(), roots
        ),
        topology=Topology(),
        conditions=Conditions.of(
            **machine_spec_key(machine).values,
            torch=torch.__version__,
            atom_src=Reading.of_source_root(atom),
            model=model,
            host_cpu=host_cpu(),
            engine_config=engines,
        ),
        gates=GateState.of(),
        members={
            member_name("projection", rank, "json"): json.dumps(document).encode()
        },
    )
    return key


def text(document: dict) -> str:
    assumed = ", ".join(document["assumed"]) or "none"
    out = [
        f"projection on {document['machine']['name']}; assumed fields read: {assumed}"
    ]
    for level, k in document["ranking"].items():
        out.append(f"{level} level: top-1 {k['top_1']}, top-3 {k['top_3']}")
        for record, c in document["cells"].items():
            r = c[level]
            rate = f"{r['tokens_per_s']:.0f}" if r["tokens_per_s"] else "-"
            phases = "; ".join(
                f"{p} {s['total']:.2f} s, median {s['median'] * 1e3:.3f} ms"
                for p, s in r["analytical_s"].items()
            )
            out.append(
                f"  {record}: {r['steps']} steps priced, {r['steps_refused']} "
                f"refused, {rate} tokens/s; {phases}"
            )
            out += [f"    refused {n} steps: {why}" for why, n in r["refused"].items()]
    for path, factors in document["band"].items():
        for factor, b in factors.items():
            cap = ", capped at 1" if b["capped"] else ""
            out.append(f"{path} x{factor}{cap}:")
            for level, k in b["ranking"].items():
                holds = "holds" if b["order_holds"][level] else "changes"
                out.append(
                    f"  {level} level: top-1 {k['top_1']}, top-3 {k['top_3']}; "
                    f"the order {holds}"
                )
    for level, k in document["ranking"].items():
        moved = [
            p
            for p, factors in document["band"].items()
            if any(b["ranking"][level]["top_1"] != k["top_1"] for b in factors.values())
        ]
        out.append(f"{level} level: top-1 changed at 0.8 or 1.2 of {moved or 'none'}")
    return "\n".join(out)


def main(argv: list[str]) -> None:
    cuts = [i for i, word in enumerate(argv) if word == "--"]
    p = argparse.ArgumentParser(prog=PROG, description=__doc__)
    p.add_argument("--model", required=True, help="a model name or directory")
    p.add_argument("--tp", type=int, default=1, help="tensor-parallel width")
    p.add_argument("--machine", required=True, help="the machine spec priced on")
    p.add_argument(
        "--store",
        required=True,
        help="the artifact store: op_graphs in, projection out",
    )
    args = p.parse_args(argv[: cuts[0] if cuts else len(argv)])
    groups = [argv[i + 1 : j] for i, j in zip(cuts, [*cuts[1:], len(argv)])]
    if not groups or not all(groups):
        p.error("name each cell after its own --: -- RECORD [ATOM args]")
    echo = json.loads(Path(args.machine).read_text())
    jobs = [(args.model, args.tp, echo, args.store, g[0], g[1:]) for g in groups]
    results = _each(jobs)
    cells = {g[0]: (g[1:], c) for g, c in zip(groups, results)}
    if len({c["model"] for _, c in cells.values()}) > 1:
        raise ValueError("the cells ran different models; a projection ranks one")
    machine = MachineSpec.from_mapping(echo)
    document = projection(machine, cells)
    document["band"] = band(machine, document["assumed"], jobs, document["ranking"])
    print(text(document))
    key = publish(args.store, document, cells, machine, " ".join([PROG, *argv]))
    print(f"published {key} as {key.token()}")
