# SPDX-License-Identifier: MIT
"""`compass validate bound`: the analytical price, at most the measured device
time on every measured step, at step and at op level.

    compass validate bound --model M [--tp N] --machine SPEC --store S
        [--ops DIR] RECORD... [-- ATOM args]

A RECORD is a real run's directory, its step record in `steps/`. Each forward
there that is not a DP dummy run is priced at step level by the step roofline,
over the modules `run.modules.walk` reads off the meta build ATOM's arguments
configure, and at op level by `OpGraphBackend`, over the `op_graph`s `compass
trace ops` published in S for the spec's arch; both price the rows a replayed
graph pads to. Each price is compared with the step's device busy time,
`d_busy_ns`, or on an MTP record the target's, `d_target_ns`.

A data-parallel step (one with `moe_rows`) is read off every rank's line at
that step: its `moe_tokens` is the group's scheduled tokens, a dummy rank's
one, and under `--enable-tbo` its two-batch-overlap split is the one ATOM's
prefill rule agrees over those lines (`_split`).

Gate: a step priced above its measured time is a violation, named by its
costliest term, an op kind (`op.<kind>`) or a module kind (`roofline.<kind>`).
Any violation fails the gate and exits with status 1. With none, any refusal
makes the result a coverage report, its reasons counted by step.

Reported, not gated: per granularity and phase (decode, prefill, mixed), the
distribution of measured over analytical; and the records ranked by tokens a
second over the steps priced at that granularity, analytical against
measured: whether measured's fastest is analytical's fastest (top-1), or
among its first three (top-3).

Per op kind (`--ops`, a `measure ops --plan` run's `--out`): each point its
plan.jsonl holds measured, its cold time over the analytical price of the same
leaf at the point's shape and rows (`op_work.work`, `work.price`), as a
distribution per op kind and phase, with the points priced above their
measured time counted. Not gated: the bound holds per step, not per leaf. A
point whose leaf the analytical price refuses is counted by its reason; one
the plan refused to measure is counted apart.

Refused: a record whose engine block (`dp<r>.engine.json`, else the `engine`
of `dp<r>.key.json`) states another TP width or KV cache format than ATOM's
arguments resolve; two-batch overlap of decode steps or on one rank, whose
split no step line states; a data-parallel step some rank's record lacks; a
step of an MTP record without `d_target_ns`, whose `d_busy_ns` holds the
draft no backend prices.
"""

import argparse
import json
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from atom.compass.backends import BatchView, CostRefused, RequestShape
from atom.compass.backends.provenance import Refusal
from atom.compass.backends.shape import sum_tokens

SOURCE = "validate-bound"
PHASES = ("decode", "prefill", "mixed")
#: The engine fields a record must state as ATOM's arguments resolve them.
PRICED = ("tensor_parallel_size", "kv_cache_dtype")
PARTIAL = (
    "a data-parallel step some rank's record lacks: the group's token count "
    "and two-batch-overlap split, which its price reads, are unknown"
)
UNSPLIT = (
    "a step of an MTP record without `d_target_ns`: its `d_busy_ns` holds the "
    "draft, which no backend prices"
)


def steps(record, tbo=False, mtp=False):
    """Each forward of `record`'s ranks but DP dummy runs, as its name, its
    batch view or the `Refusal` it is counted by, and its measured device
    seconds. `tbo` is ATOM's `enable_tbo`; `mtp` says the run drafts, which
    any line with `d_target_ns` also says."""
    paths = sorted(Path(record, "steps").glob("dp*[0-9].jsonl"))
    ranks = [
        {s["step"]: s for s in map(json.loads, p.read_text().splitlines())}
        for p in paths
    ]
    mtp = mtp or any("d_target_ns" in s for r in ranks for s in r.values())
    for step in sorted(set().union(*ranks)):
        lines = [r.get(step) for r in ranks]
        whole = None not in lines
        # num_tokens_across_dp: every rank's scheduled tokens, a dummy's one.
        group = sum(q for s in lines if s for q, _, _ in s["rows"])
        split = _split(lines) if tbo and whole else None
        for path, s in zip(paths, lines):
            if s is None or s["is_dummy_run"]:
                continue
            if "d_busy_ns" not in s:
                raise ValueError(f"{path} has no device windows (`d_busy_ns`)")
            name = f"{record} {path.stem} step {step}"
            dp = s.get("moe_rows") is not None
            if dp and not whole:
                yield name, Refusal(SOURCE, PARTIAL), None
                continue
            if mtp and "d_target_ns" not in s:
                yield name, Refusal(SOURCE, UNSPLIT), None
                continue
            view = BatchView(
                tuple(RequestShape(q, c, bool(d)) for q, c, d in s["rows"]),
                capture_rung=s["rung"],
                moe_rows=s.get("moe_rows"),
                produces_output=s["produces_output"],
                tbo_tokens=split if dp else None,
                moe_tokens=group if dp else None,
            )
            yield name, view, s.get("d_target_ns", s["d_busy_ns"]) * 1e-9


def _split(lines):
    """The two-batch-overlap split ATOM agrees over a DP group's step `lines`,
    per micro-batch each rank's tokens, or None: each rank's prefill answer
    (`local_tbo_precompute`), then `sync_dp_metadata`'s OR of the min-token
    bar, AND of can-split and uniform prefill. A decoding rank, a dummy run
    among them, cannot split."""
    from atom.utils import envs
    from atom.utils.tbo import ubatching

    rule = (
        ubatching._precompute_prefill_token_split
        if envs.ATOM_TBO_PREFILL_TOKEN_SPLIT
        else ubatching._precompute_prefill_req_split
    )
    local, bar = [], envs.ATOM_TBO_PREFILL_MIN_TOKENS
    for s in lines:
        prefill = [q for q, _, d in s["rows"] if not d]
        if not prefill:
            return None
        local.append(rule(np.array(prefill), len(prefill), bar))
    meets, can, first, second = zip(*local)
    return (first, second) if any(meets) and all(can) else None


def _phase(view: BatchView) -> str:
    if view.prefill and view.decode:
        return "mixed"
    return "prefill" if view.prefill else "decode"


def gate(records, backends: dict, tbo=False, mtp=False) -> dict:
    """The gate and its reports for `records`, every step priced by each of
    `backends`, a granularity's name to its `CostBackend`; `tbo` and `mtp` as
    `steps` reads them."""
    report = {}
    for level, backend in backends.items():
        violations, refused, rates, unpriced = [], Counter(), {}, 0
        ratios = {p: [] for p in PHASES}
        for record in records:
            tokens = measured_s = priced_s = 0.0
            for name, view, measured in steps(record, tbo, mtp):
                try:
                    if isinstance(view, Refusal):
                        raise CostRefused(None, [view])
                    cost = backend.price(view)
                except CostRefused as e:
                    refused.update(str(r) for r in dict.fromkeys(e.declined))
                    unpriced += 1
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
            "steps_refused": unpriced,
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


def ops(out, machine, tp: int, kv: str, ep: int = 1) -> dict:
    """The points a `measure ops --plan` run at `out` measured, each its cold
    time over the analytical price of its leaf at its shape and rows on
    `machine`, in a TP group of `tp` and an EP group of `ep` with a `kv` KV cache,
    by op kind and phase.
    A leaf the analytical price says launches no kernel is counted apart, by
    whether its point timed any kernel."""
    from atom.compass.backends.op_work import work
    from atom.compass.backends.work import price
    from atom.compass.spec.rules import SpecRefusal

    lines = Path(out, "plan.jsonl").read_text().splitlines()
    ratios, refused, count = defaultdict(list), Counter(), Counter()
    for p in map(json.loads, lines):
        if "refused" in p:
            count["unmeasured"] += 1
            continue
        rows = tuple(RequestShape(q, c, bool(d)) for q, c, d in p["rows"])
        view = BatchView(rows, produces_output=True)
        try:
            w = work(json.loads(p["leaf"]), view, tp, kv, ep)
            priced = None if w is None else price(w, machine)
        except CostRefused as e:
            refused.update(str(r) for r in e.declined)
            continue
        except (SpecRefusal, ValueError) as e:
            refused[f"{p['op']}: {e}"] += 1
            continue
        if priced is None:
            count["no kernel, timed" if p["cold_ns"] else "no kernel"] += 1
            continue
        ratios[p["op"], p["shape"].split()[0]].append(p["cold_ns"] * 1e-9 / priced)
    return {
        "points": len(lines),
        "counts": dict(count),
        "refused": dict(refused.most_common()),
        "measured_over_analytical": {
            f"{kind} {phase}": {
                "points": len(r),
                "above": sum(x < 1 for x in r),
                **dict(
                    zip(
                        ("min", "p10", "median", "p90", "max"),
                        np.percentile(r, (0, 10, 50, 90, 100)),
                    )
                ),
            }
            for (kind, phase), r in sorted(ratios.items())
        },
    }


def ops_text(r: dict) -> str:
    rows, n = r["measured_over_analytical"], Counter(r["counts"])
    priced = sum(q["points"] for q in rows.values())
    above = sum(q["above"] for q in rows.values())
    out = [
        (
            f"op points: {r['points']} planned, {n['unmeasured']} unmeasured, "
            f"{n['no kernel'] + n['no kernel, timed']} of a leaf priced as launching "
            f"no kernel ({n['no kernel, timed']} of them timed a kernel), {priced} "
            f"priced, {sum(r['refused'].values())} refused; {above} priced above "
            "their measured time, reported, not gated: the bound holds per step"
        )
    ]
    out += [f"  refused {k} points: {reason}" for reason, k in r["refused"].items()]
    out.append(
        "  measured over analytical: kind phase | points | above | min | p10 | median | p90 | max"
    )
    for name, q in rows.items():
        out.append(
            f"  {name} | {q['points']} | {q['above']} | "
            + " | ".join(f"{q[k]:.3f}" for k in ("min", "p10", "median", "p90", "max"))
        )
    return "\n".join(out)


def text(report: dict) -> str:
    out = []
    for level, r in report.items():
        out.append(
            f"{level} level: {r['verdict']}, {r['steps']} steps priced, {r['steps_refused']} refused, {len(r['violations'])} above measured"
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
        # Reasons that differ only in their numbers, a shape or a row count, share a line.
        census, values = Counter(), Counter()
        for reason, n in r["refused"].items():
            census[key := re.sub(r"\b\d+\b", "N", reason)] += n
            values[key] += 1
        for reason, n in census.most_common():
            many = f" ({values[reason]} values of N)" if values[reason] > 1 else ""
            out.append(f"  refused {n} steps: {reason}{many}")
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
    device with no driver, `arch` declared, its post-load hooks run as a
    simulated run's are."""
    import torch

    from atom.compass.capture import driverless
    from atom.compass.capture.forward import _config
    from atom.compass.runner.overrides import (
        FactoriesOnMeta,
        process_weights_after_loading,
    )
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
        tree = resolve_obj_by_qualname(model_class)(config)
        process_weights_after_loading(tree)
    return config, tree


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
    p.add_argument(
        "--ops",
        metavar="DIR",
        help="a `measure ops --plan` run's --out, whose points are each priced "
        "at their leaf, shape and rows",
    )
    args = p.parse_args(argv[:cut])

    from atom.compass.backends.op_graph import OpGraphBackend
    from atom.compass.backends.step_roofline import StepRooflineBackend
    from atom.compass.run import _op_graphs, ep_width
    from atom.compass.run.modules import walk
    from atom.compass.spec import MachineSpec

    machine = MachineSpec.from_mapping(json.loads(Path(args.machine).read_text()))
    arch = machine.value("device.arch")
    with tempfile.TemporaryDirectory(prefix="compass-bound-") as tmpdir:
        config, tree = _build(args.model, args.tp, argv[cut + 1 :], arch, tmpdir)
    ep = ep_width(tree)
    dp = config.parallel_config.data_parallel_size
    if config.enable_tbo_decode or config.enable_tbo and dp == 1:
        raise ValueError(
            "two-batch overlap of decode steps or on one rank: no step line "
            "states its split, and only a data-parallel group's prefill split "
            "is rebuilt from the group's lines"
        )
    engines = []
    for record in args.records:
        for path in sorted(Path(record, "steps").glob("dp*[0-9].jsonl")):
            engine = path.with_suffix(".engine.json")
            if engine.exists():
                ran = json.loads(engine.read_text())
            else:
                ran = json.loads(path.with_suffix(".key.json").read_text())["engine"]
            engines.append((path.parent, ran))
    if args.ops:
        engines.append(
            (args.ops, json.loads(Path(args.ops, "engine.json").read_text()))
        )
    for where, ran in engines:
        if differ := [k for k in PRICED if ran.get(k) != getattr(config, k)]:
            raise ValueError(
                f"{where} ran under another engine than ATOM's arguments resolve: "
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
            ep=ep,
            kv=config.kv_cache_dtype,
        ),
    }
    mtp = config.speculative_config is not None
    report = gate(args.records, backends, config.enable_tbo, mtp)
    print(text(report))
    if args.ops:
        print(
            ops_text(
                ops(
                    args.ops,
                    machine,
                    config.tensor_parallel_size,
                    config.kv_cache_dtype,
                    ep,
                )
            )
        )
    if any(r["violations"] for r in report.values()):
        raise SystemExit(1)
