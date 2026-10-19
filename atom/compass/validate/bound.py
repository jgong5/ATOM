# SPDX-License-Identifier: MIT
"""`compass validate bound`: the analytical price, at most the measured device
time on every measured step, at step and at op level.

    compass validate bound --model M [--tp N] --machine SPEC --store S
        RECORD... [-- ATOM args]

A RECORD is a real run's directory, its step record in `steps/`. Each forward
there that is not a DP dummy run is priced at step level by the step roofline,
over the modules `run.modules.walk` reads off the meta build ATOM's arguments
configure, and at op level by `OpGraphBackend`, over the `op_graph`s `compass
trace ops` published in S for the spec's arch; both price the rows a replayed
graph pads to. Each price is compared with the step's device busy time,
`d_busy_ns`.

Gate: a step priced above its measured time is a violation, named by its
costliest term, an op kind (`op.<kind>`) or a module kind (`roofline.<kind>`).
Any violation fails the gate and exits with status 1. With none, any refusal
makes the result a coverage report, its reasons counted by step.

Reported, not gated: per granularity and phase (decode, prefill, mixed), the
distribution of measured over analytical; and the records ranked by tokens a
second over the steps priced at that granularity, analytical against
measured: whether measured's fastest is analytical's fastest (top-1), or
among its first three (top-3).

Refused: a record whose engine file states another TP width or KV cache
format than ATOM's arguments resolve; a step with `moe_rows`, since the record
states neither the DP group's token count nor its two-batch-overlap split,
both of which its price reads.
"""

import argparse
import json
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

from atom.compass.backends import BatchView, CostRefused, RequestShape
from atom.compass.backends.provenance import Refusal
from atom.compass.backends.shape import sum_tokens

SOURCE = "validate-bound"
PHASES = ("decode", "prefill", "mixed")
#: The engine fields a record must state as ATOM's arguments resolve them.
PRICED = ("tensor_parallel_size", "kv_cache_dtype")
DP_MOE = (
    "a data-parallel MoE step: the record states neither the group's token "
    "count nor its two-batch-overlap split, which its price reads"
)


def steps(record):
    """Each forward of `record`'s ranks but DP dummy runs, as its name, its
    batch view and its measured device busy seconds."""
    for path in sorted(Path(record, "steps").glob("dp*[0-9].jsonl")):
        for s in map(json.loads, path.read_text().splitlines()):
            if s["is_dummy_run"]:
                continue
            if "d_busy_ns" not in s:
                raise ValueError(f"{path} has no device windows (`d_busy_ns`)")
            view = BatchView(
                tuple(RequestShape(q, c, bool(d)) for q, c, d in s["rows"]),
                capture_rung=s["rung"],
                moe_rows=s.get("moe_rows"),
                produces_output=s["produces_output"],
            )
            yield f"{record} {path.stem} step {s['step']}", view, s["d_busy_ns"] * 1e-9


def _phase(view: BatchView) -> str:
    if view.prefill and view.decode:
        return "mixed"
    return "prefill" if view.prefill else "decode"


def gate(records, backends: dict) -> dict:
    """The gate and its reports for `records`, every step priced by each of
    `backends`, a granularity's name to its `CostBackend`."""
    report = {}
    for level, backend in backends.items():
        violations, refused, rates = [], Counter(), {}
        ratios = {p: [] for p in PHASES}
        for record in records:
            tokens = measured_s = priced_s = 0.0
            for name, view, measured in steps(record):
                try:
                    if view.moe_rows is not None:
                        raise CostRefused(view, [Refusal(SOURCE, DP_MOE)])
                    cost = backend.price(view)
                except CostRefused as e:
                    refused.update(str(r) for r in dict.fromkeys(e.declined))
                    continue
                priced = cost.seconds
                ratios[_phase(view)].append(measured / priced)
                tokens += sum_tokens(view.requests)
                measured_s, priced_s = measured_s + measured, priced_s + priced
                if priced > measured:
                    top = max(cost.terms, key=lambda t: t.seconds)
                    violations.append(
                        {
                            "step": name,
                            "phase": _phase(view),
                            "measured_s": measured,
                            "priced_s": priced,
                            "named": top.name,
                            "named_s": top.seconds,
                        }
                    )
            if priced_s:
                rates[str(record)] = (tokens / measured_s, tokens / priced_s)
        steps_priced = sum(map(len, ratios.values()))
        verdict = "fail" if violations else "coverage report" if refused else "pass"
        report[level] = {
            "verdict": verdict,
            "steps": steps_priced,
            "violations": violations,
            "refused": dict(refused.most_common()),
            "measured_over_analytical": {
                p: dict(
                    zip(
                        ("min", "p10", "median", "p90", "max"),
                        np.percentile(r, (0, 10, 50, 90, 100)),
                    )
                )
                for p, r in ratios.items()
                if r
            },
            "ranking": _ranking(rates),
        }
    return report


def _ranking(rates: dict) -> dict | None:
    """Records by analytical tokens a second, and where measured's fastest sits."""
    if not rates:
        return None
    order = sorted(rates, key=lambda r: -rates[r][1])
    fastest = max(rates, key=lambda r: rates[r][0])
    rank = order.index(fastest) + 1
    return {
        "analytical_order": order,
        "measured_fastest": fastest,
        "top_1": rank == 1,
        "top_3": rank <= 3,
        "tokens_per_s": {
            r: {"measured": m, "analytical": a} for r, (m, a) in rates.items()
        },
    }


def text(report: dict) -> str:
    out = []
    for level, r in report.items():
        out.append(
            f"{level} level: {r['verdict']}, {r['steps']} steps priced, {len(r['violations'])} above measured"
        )
        if r["violations"]:
            named = Counter(v["named"] for v in r["violations"])
            out.append(
                f"  above measured, by costliest term: {json.dumps(dict(named.most_common()))}"
            )
        for v in r["violations"]:
            out.append(
                f"  above: {v['step']} ({v['phase']}) measured {v['measured_s'] * 1e3:.3f} ms,"
                f" priced {v['priced_s'] * 1e3:.3f} ms; costliest {v['named']} {v['named_s'] * 1e3:.3f} ms"
            )
        for reason, n in r["refused"].items():
            out.append(f"  refused {n} steps: {reason}")
        out.append("  measured over analytical: phase | min | p10 | median | p90 | max")
        for phase, q in r["measured_over_analytical"].items():
            out.append(f"  {phase} | " + " | ".join(f"{x:.3f}" for x in q.values()))
        if (k := r["ranking"]) is not None:
            out.append(
                f"  ranking: top-1 {k['top_1']}, top-3 {k['top_3']}; measured fastest {k['measured_fastest']}"
            )
            for record in k["analytical_order"]:
                t = k["tokens_per_s"][record]
                out.append(
                    f"  {record}: analytical {t['analytical']:.0f}, measured {t['measured']:.0f} tokens/s"
                )
    return "\n".join(out)


def _build(model: str, tp: int, atom_args, arch: str, tmpdir: str):
    """ATOM's config for its arguments, and the model it builds on the meta
    device with no driver, `arch` declared."""
    import torch

    from atom.compass.capture import driverless
    from atom.compass.capture.forward import _config
    from atom.compass.runner.overrides import FactoriesOnMeta
    from atom.config import set_current_atom_config
    from atom.utils import resolve_obj_by_qualname

    driverless.install(tmpdir, arch)
    config = _config(model, tp, atom_args)
    driverless.group(
        config.tensor_parallel_size, config.parallel_config.data_parallel_size
    )
    set_current_atom_config(config)
    from atom.model_engine.model_runner import (
        support_model_arch_dict,
    )  # needs the declared arch

    model_class = support_model_arch_dict[config.hf_config.architectures[0]]
    torch.set_default_dtype(config.torch_dtype)
    with FactoriesOnMeta():
        return config, resolve_obj_by_qualname(model_class)(config)


def main(argv: list[str]) -> None:
    cut = argv.index("--") if "--" in argv else len(argv)
    p = argparse.ArgumentParser(prog="compass validate bound", description=__doc__)
    p.add_argument(
        "records", nargs="+", metavar="RECORD", help="a real run's directory"
    )
    p.add_argument("--model", required=True, help="a model name or directory")
    p.add_argument("--tp", type=int, default=1, help="tensor-parallel width")
    p.add_argument("--machine", required=True, help="the machine spec")
    p.add_argument(
        "--store", required=True, help="the artifact store holding the op_graphs"
    )
    args = p.parse_args(argv[:cut])

    from atom.compass.backends.op_graph import OpGraphBackend
    from atom.compass.backends.step_roofline import StepRooflineBackend
    from atom.compass.run import _op_graphs
    from atom.compass.run.modules import walk
    from atom.compass.spec import MachineSpec

    machine = MachineSpec.from_mapping(json.loads(Path(args.machine).read_text()))
    arch = machine.value("device.arch")
    with tempfile.TemporaryDirectory(prefix="compass-bound-") as tmpdir:
        config, tree = _build(args.model, args.tp, argv[cut + 1 :], arch, tmpdir)
    for record in args.records:
        for path in sorted(Path(record, "steps").glob("dp*[0-9].engine.json")):
            ran = json.loads(path.read_text())
            if differ := [k for k in PRICED if ran.get(k) != getattr(config, k)]:
                raise ValueError(
                    f"{path} ran under another engine than ATOM's arguments resolve: "
                    + "; ".join(
                        f"{k} {ran.get(k)!r} there, {getattr(config, k)!r} here"
                        for k in differ
                    )
                )
    backends = {
        "step": StepRooflineBackend(machine, walk(tree)),
        "op": OpGraphBackend(
            machine,
            _op_graphs(args.store, config, arch),
            tp=config.tensor_parallel_size,
            kv=config.kv_cache_dtype,
        ),
    }
    report = gate(args.records, backends)
    print(text(report))
    if any(r["violations"] for r in report.values()):
        raise SystemExit(1)
