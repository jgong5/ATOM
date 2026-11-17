"""Replay a simulated run's recorded steps through OpGraphBackend with its price list.

usage: python3 replay.py SIDE_DIR RANK --model MCOPY [--first N] [--out F] [--profile F]

SIDE_DIR is a deployment side's directory (run_file.json, steps/dp<r>.jsonl).
Prices each line of steps/dp<RANK>.jsonl in order, as the run's runner did, and
writes each step's StepCost (every term's name, seconds, provenance, by_bound)
to --out, one JSON line a step, with the wall seconds pricing it took apart.
"""

import argparse
import cProfile
import dataclasses
import gc
import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path


def term(t):
    pv = t.provenance
    return [
        t.name,
        t.seconds,
        str(pv.method),
        pv.source,
        pv.validated,
        pv.note,
        pv.detail,
        [r.reason for r in pv.refusals],
        dict(t.by_bound),
    ]


def _declared(atom_args, arch):
    """A tree before `drive.run._config` took an arch: its config under a
    declared device of `arch`, as the tip builds it."""
    import tempfile

    from atom.compass.capture import driverless
    from atom.compass.drive.run import _config

    with tempfile.TemporaryDirectory(prefix="replay-config-") as tmpdir:
        driverless.install(tmpdir, arch)
        return _config(atom_args)


def config_of(atom_args, arch):
    import inspect

    from atom.compass.bench.ops import _child
    from atom.compass.drive.run import _config

    if len(inspect.signature(_config).parameters) > 1:
        return _config(atom_args, arch)
    from transformers.dynamic_module_utils import init_hf_modules

    init_hf_modules()
    return _child(_declared, atom_args, arch)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("side")
    p.add_argument("rank", type=int)
    p.add_argument("--model", required=True)
    p.add_argument("--first", type=int, default=0, help="price only the first N lines")
    p.add_argument("--skip", type=int, default=0, help="skip the first N lines")
    p.add_argument("--out")
    p.add_argument("--profile")
    p.add_argument("--digest", action="store_true", help="--out each step's sha256")
    p.add_argument("--env", default="", help="K=V,... the side's performance variables")
    p.add_argument("--dump-graphs", help="write each graph lookup's region here")
    p.add_argument("--graphs", help="serve graph lookups from a --dump-graphs file")
    p.add_argument("--params", help="the price list's params file, read directly")
    p.add_argument(
        "--graphs-only", action="store_true", help="only look each step's graph up"
    )
    args = p.parse_args()

    for kv in filter(None, args.env.split(",")):
        k, v = kv.split("=", 1)
        os.environ[k] = v

    import atom

    print("atom from", atom.__file__, flush=True)
    from atom.compass import run as compass_run
    from atom.compass.backends.resolver import CostRefused
    from atom.compass.backends.shape import BatchView, RequestShape
    from atom.compass.backends.step_labels import group_dummies, group_peaks

    side = Path(args.side)
    run = json.loads((side / "run_file.json").read_text())
    atom_args = list(run["atom_args"])
    atom_args[atom_args.index("--model") + 1] = args.model
    run = {k: v for k, v in run.items() if k != "atom_args"}  # no lazy traces
    try:
        machine = compass_run._machine(run)
    # A tip whose schema moved the router hop into host laws refuses it.
    except Exception as e:  # noqa: BLE001
        print("machine:", str(e)[:100], "; dropping interconnect.router_relay_s")
        run["machine"]["interconnect"].pop("router_relay_s")
        origins = run["machine"]["provenance"].get("origins", {})
        origins.pop("interconnect.router_relay_s", None)
        machine = compass_run._machine(run)
    if args.params:
        from atom.compass.fit.ops import PriceList

        t0 = time.perf_counter()
        params = json.loads(Path(args.params).read_text())
        print(f"params {time.perf_counter() - t0:.1f} s", flush=True)
        compass_run._price_list = lambda run, config, machine: PriceList(params)
    if args.graphs:
        with open(args.graphs, "rb") as f:
            held = pickle.load(f)
        compass_run._op_graphs = lambda *a, **k: lambda *a, **k: held.get(
            repr((a, sorted(k.items())))
        )
    t0 = time.perf_counter()
    config = config_of(atom_args, machine.value("device.arch"))
    print(f"config {time.perf_counter() - t0:.1f} s", flush=True)
    backend = compass_run.op_graph(run, config, machine, 1)
    graph, lookups = backend.graph, [0.0, 0]

    def timed_graph(*a, **k):
        t = time.perf_counter()
        try:
            return graph(*a, **k)
        finally:
            lookups[0] += time.perf_counter() - t
            lookups[1] += 1

    backend.graph = timed_graph
    dumped = {}
    if args.dump_graphs:

        def dumping(*a, **k):
            got = timed_graph(*a, **k)
            dumped[repr((a, sorted(k.items())))] = got
            return got

        backend.graph = dumping
    ranks = [
        [json.loads(x) for x in f.read_text().splitlines()]
        for f in sorted((side / "steps").glob("dp*[0-9].jsonl"))
    ]
    dummies, peaks = group_dummies(ranks), group_peaks(ranks)
    lines = ranks[args.rank][args.skip :]
    lines = lines[: args.first] if args.first else lines

    known = {f.name for f in dataclasses.fields(BatchView)}

    def view_of(s):
        dp = "moe_rows" in s
        split = s.get("tbo_tokens")
        kw = {
            "capture_rung": s["rung"],
            "moe_rows": s.get("moe_rows"),
            "draft_moe_rows": s.get("draft_moe_rows"),
            "draft_bs": s.get("draft_bs"),
            "draft_rows": s.get("draft_rows"),
            "target_bs": s.get("target_bs"),
            "produces_output": s["produces_output"],
            "tbo_tokens": None if split is None else tuple(map(tuple, split)),
            "moe_tokens": s.get("moe_tokens"),
            "dp_rank": s["dp_rank"] if dp else None,
            "dp_dummy": dp and bool(s["is_dummy_run"]),
            "group_dummies": dummies.get(s["step"], 0) if dp else 0,
            "group_max_tokens": peaks.get(s["step"]) if dp else None,
            "group_needs_eager": s.get("group_needs_eager", False),
        }
        kw = {k: v for k, v in kw.items() if k in known}
        rows = tuple(RequestShape(q, c, bool(d)) for q, c, d in s["rows"])
        return BatchView(rows, **kw)

    views = [view_of(s) for s in lines]
    if args.graphs_only:  # the regions `price` reads, with no drafter and no overlap
        missed = 0
        for v in views:
            assert v.tbo_tokens is None and not backend.drafts
            try:
                backend._region(v)
            except CostRefused:
                missed += 1
        with open(args.dump_graphs, "wb") as f:
            pickle.dump(dumped, f)
        print(f"steps {len(views)} graphs {len(dumped)} refused {missed}", flush=True)
        return
    # A run parses each graph once and freezes the live heap with it (`run._op_graphs`)
    gc.collect()
    gc.freeze()
    out = open(args.out, "w") if args.out else None  # noqa: SIM115
    prof = cProfile.Profile() if args.profile else None
    walls, total = [], 0.0
    t_all = time.perf_counter()
    for s, v in zip(lines, views):
        t = time.perf_counter()
        if prof:
            prof.enable()
        try:
            cost = backend.price(v)
            got = {"terms": [term(x) for x in cost.terms]}
            got["draft"] = [term(x) for x in cost.draft_terms]
            sec = sum(x.seconds for x in cost.terms)
        except CostRefused as e:
            got = {"refused": [r.reason for r in e.declined]}
            sec = None
        if prof:
            prof.disable()
        w = time.perf_counter() - t
        walls.append(w)
        total += sec or 0
        if out and args.digest:
            sha = hashlib.sha256(json.dumps(got, sort_keys=True).encode()).hexdigest()
            out.write(json.dumps({"step": s["step"], "s": sec, "wall": w, "sha": sha}))
            out.write("\n")
        elif out:
            out.write(
                json.dumps({"step": s["step"], "s": sec, "wall": w, **got}) + "\n"
            )
    elapsed = time.perf_counter() - t_all
    if prof:
        prof.dump_stats(args.profile)
    if args.dump_graphs:
        with open(args.dump_graphs, "wb") as f:
            pickle.dump(dumped, f)
    walls.sort()
    n = len(walls)
    print(
        f"steps {n} wall {elapsed:.2f} s, per step mean {elapsed / n * 1e3:.3f} ms "
        f"median {walls[n // 2] * 1e3:.3f} ms p90 {walls[n * 9 // 10] * 1e3:.3f} ms; "
        f"graph lookups {lookups[1]} {lookups[0]:.2f} s; simulated {total:.3f} s",
        flush=True,
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
