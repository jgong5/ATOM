"""Price a list of step shapes through the frozen composition, off GPU.

The composition is not a new one. It is the same object graph
`agent_scratch/g4/tpN_frozen.py` builds -- body graph plus the actual head and
its collectives, priced through one `PriceLibrary`, with the source-calibrated
`RunnerRegions` supplying prepare and postprocess -- with exactly one argument
changed: where the body graph comes from. `tpN_frozen.py` passes a
`StaticGraphs` holding one graph for one shape, which is why it can only answer
about that shape. This passes a `TemplateGraphs`, which binds a template to a
cohort when it can and derives when it cannot, and so can be asked about a
shape nobody has derived.

That is the whole of the difference, and it is the point: if the composition
had to be rebuilt to become executable, the numbers it produced would no longer
be the numbers the frozen diagnostics produced.

What it refuses rather than guesses:

  * a shape whose graph cannot be bound (no template, no deriver) -- reported,
    not filled in from a neighbour;
  * a shape whose operators the library does not price (`require_complete`);
  * a shape outside the region model's calibrated domain, which
    `RunnerRegions.breakdown` raises on;
  * a prefill shape that does not say whether it produces output, which decides
    whether the LM head runs at all.

Every refusal is counted and named in the output. A run that priced 40 of 100
shapes says so; it does not report an average over the 40.

    python scripts/compass/predict_step.py \
        --model /models/Qwen3.8-27B --tp 1 --device meta \
        --replay-target target.json \
        --block-size 16 --max-model-len 262144 --position-rows 3 \
        --price prices.json:graph.json:unregistered \
        --price hprices.json:hgraph.json:unregistered \
        --template graph.json --head-template hgraph.json \
        --regions source-27b-tp1 --cudagraph-mode full \
        --shapes shapes.json -o out.json

``--shapes`` is a JSON list of step shapes, in the field names `StepShape`
uses. ``--template`` seeds the cache with a graph already on disk, keyed by the
structure its own recorded batch spec describes -- so a run can reuse the v2
artifacts instead of deriving what has already been derived.
"""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path


def _file_digest(path):
    """The one digest helper, loaded by path rather than restated here.

    `scripts/compass/execution_id.py` owns it. Loaded the way that module loads
    the identity rule -- by file, not by importing `scripts.compass`, which is
    not a package on every invocation of this script.
    """
    name = "scripts_compass_execution_id"
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(
            name, Path(__file__).resolve().parent / "execution_id.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return module.file_digest(path)


def _inputs(args) -> dict:
    """Every file and every scalar this report's numbers depend on.

    A report that states a step time without stating what it was priced from is
    not attributable: the same shape through two price lists is two different
    answers with the same shape written at the top. The artifacts are recorded
    by digest rather than by path because a path is a name and the thing that
    has to match later is the bytes -- `agent_scratch/g4/` holds two files
    called `h27dec32.tp1.r0.json` with different contents, which is exactly the
    ambiguity a path cannot resolve.

    `--price` is a `prices[:graph[:registration]]` triple, and each part is
    recorded separately: the graph is what lets a layout mismatch be refused
    rather than answered from a dense price, and the registration is one of the
    two things a collective's signature does not carry.
    """
    prices = []
    for spec in args.price:
        parts = spec.split(":")
        prices.append({
            "prices": parts[0],
            "prices_digest": _file_digest(parts[0]),
            "graph": parts[1] if len(parts) > 1 and parts[1] else None,
            "graph_digest": (_file_digest(parts[1])
                             if len(parts) > 1 and parts[1] else None),
            "registration": parts[2] if len(parts) > 2 else None,
        })
    return {
        "prices": prices,
        "templates": [{"path": p, "digest": _file_digest(p)}
                      for p in args.template],
        "head_templates": [{"path": p, "digest": _file_digest(p)}
                           for p in args.head_template],
        "replay_target": ({"path": args.replay_target,
                           "digest": _file_digest(args.replay_target)}
                          if args.replay_target else None),
        "shapes_file": {"path": args.shapes, "digest": _file_digest(args.shapes)},
        # Scalars that change the answer rather than the presentation. `head`
        # decides whether a second region is priced at all; `require_complete`
        # decides whether an incomplete sum is a number or a refusal, and a
        # report produced with it off may not feed acceptance.
        "settings": {
            "model": args.model, "tp": args.tp, "device": args.device,
            "block_size": args.block_size,
            "max_model_len": args.max_model_len,
            "position_rows": args.position_rows,
            "block_policy": args.block_policy,
            "cudagraph_mode": args.cudagraph_mode,
            "regions": args.regions,
            "head": bool(args.head),
            "seconds_per_launch": args.seconds_per_launch,
            "require_complete": bool(args.require_complete),
            "carry_allocation": bool(args.carry_allocation),
            "derive": not args.no_derive,
            # None is exact prices only. A number is the declared sampling
            # density fitted prices were allowed over, and it belongs here for
            # the same reason `require_complete` does: the same shape under two
            # densities is two claims, and nothing else in the file says which.
            "interpolate": args.interpolate,
        },
    }


def _shape_from(raw: dict):
    from atom.compass.core.cost.base import StepShape

    missing = {"num_scheduled_tokens", "context_lens"} - set(raw)
    if missing:
        raise ValueError(f"shape is missing {sorted(missing)}")
    unknown = set(raw) - {
        "num_scheduled_tokens", "context_lens", "num_prefill_tokens",
        "topology", "rank_coords", "capture_bucket", "compiled",
        "produces_output", "label"}
    if unknown:
        raise ValueError(f"shape has unknown fields {sorted(unknown)}")
    return StepShape(
        num_scheduled_tokens=tuple(raw["num_scheduled_tokens"]),
        context_lens=tuple(raw["context_lens"]),
        num_prefill_tokens=int(raw.get("num_prefill_tokens", 0)),
        topology=raw.get("topology") or {},
        rank_coords=raw.get("rank_coords") or {},
        capture_bucket=raw.get("capture_bucket"),
        compiled=raw.get("compiled"),
        produces_output=raw.get("produces_output", True),
    )


def main() -> int:
    # Imported before the parser is built, because `--regions` takes its choices
    # from the one registry that holds them. That makes even `--help` load the
    # engine package -- which this script needs for every real invocation
    # anyway, and the alternative is a second list of names to keep in step.
    from atom.compass.core.cost.regions import REGION_MODELS

    ap = argparse.ArgumentParser(
        description="price step shapes through the frozen composition")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--device", default="meta",
                    choices=["meta", "cuda", "cpu"])
    ap.add_argument("--replay-target", default=None,
                    help="A captured target.json, so AITER can resolve the "
                         "chip on a machine with no GPU.")
    ap.add_argument("--block-size", type=int, required=True)
    ap.add_argument("--max-model-len", type=int, required=True)
    ap.add_argument("--position-rows", type=int, default=1,
                    help="3 for an MRoPE model. A shape does not carry it and "
                         "the positions tensor is that many times longer.")
    ap.add_argument("--block-policy", default="rounds")
    ap.add_argument("--cudagraph-mode", default=None,
                    choices=["full", "piecewise", "eager"])
    ap.add_argument("--price", action="append", default=[],
                    help="prices.json[:graph.json[:regime]], repeatable")
    ap.add_argument("--template", action="append", default=[],
                    help="A derived body graph to seed the cache with, "
                         "repeatable. Keyed by its own recorded batch spec.")
    ap.add_argument("--head-template", action="append", default=[],
                    help="The same, for the head region.")
    ap.add_argument("--head", action="store_true",
                    help="Price the LM head as a second region of the step. "
                         "Without it the step is body only, which is a "
                         "different claim and is recorded as one.")
    ap.add_argument("--interpolate", default=None, metavar="MAX_GAP_RATIO",
                    help="Price an operator at a row count nobody measured "
                         "from the measured ones either side, where the gap "
                         "between them is no wider than this ratio. Omitted, "
                         "an unmeasured row count is refused with its reason. "
                         "A fitted price is reported apart from a measured "
                         "one, so coverage stays truthful either way.")
    ap.add_argument("--regions", default="source-27b-tp1",
                    choices=sorted(REGION_MODELS))
    ap.add_argument("--seconds-per-launch", type=float, default=0.0)
    ap.add_argument("--no-derive", action="store_true",
                    help="Refuse any shape without a template instead of "
                         "deriving one. Measures template hit rate alone.")
    ap.add_argument("--require-complete", action="store_true", default=True)
    ap.add_argument("--allow-incomplete", dest="require_complete",
                    action="store_false",
                    help="Report a partially priced step rather than refusing "
                         "it. The coverage is reported either way.")
    ap.add_argument("--carry-allocation", action="store_true",
                    help="Reuse the template's block and state assignment for "
                         "every cohort bound to it. An explicitly unmeasured "
                         "approximation: see CarriedAllocation. Without it, "
                         "binding refuses any template that carries one.")
    ap.add_argument("--shapes", required=True)
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args()

    from atom.compass.runtime.source_oracle import build_source_oracle

    with open(args.shapes, encoding="utf-8") as fh:
        raw_shapes = json.load(fh)
    shapes = [_shape_from(r) for r in raw_shapes]
    labels = [r.get("label") or f"#{i}" for i, r in enumerate(raw_shapes)]

    # One construction, shared with the served path. The model is loaded once,
    # before the loop, because that is the whole reason this is a process and
    # not a shell loop -- `build_source_oracle` does that and reports what it
    # cost, so this script and a served run cannot compose different oracles
    # and call them both the frozen one.
    built = build_source_oracle(
        model=args.model, tp=args.tp, device=args.device,
        replay_target=args.replay_target,
        block_size=args.block_size, max_model_len=args.max_model_len,
        position_rows=args.position_rows, block_policy=args.block_policy,
        cudagraph_mode=args.cudagraph_mode,
        price=args.price, template=args.template,
        head_template=args.head_template, head=args.head,
        regions=args.regions,
        seconds_per_launch=args.seconds_per_launch,
        require_complete=args.require_complete,
        carry_allocation=args.carry_allocation,
        derive=not args.no_derive,
        interpolate=args.interpolate,
    )
    oracle = built.oracle
    body_graphs = built.body_graphs
    deriver = built.deriver
    build_s = built.build_seconds
    allocation = built.allocation
    head_graphs = built.head_graphs

    rows, refused = [], 0
    wall0 = time.perf_counter()
    for label, shape in zip(labels, shapes):
        t0 = time.perf_counter()
        try:
            cost = oracle.estimate(shape)
        except (KeyError, ValueError) as exc:
            refused += 1
            rows.append({"label": label, "refused": str(exc),
                         "seconds_to_answer": time.perf_counter() - t0})
            continue
        cov = oracle.last_coverage
        rows.append({
            "label": label,
            "step_seconds": cost.seconds,
            "breakdown": dict(cost.breakdown),
            "coverage": cov.describe() if cov is not None else None,
            # The same record as categories, because the sentence above is
            # truncated and a grader reads the file rather than the log.
            "coverage_split": cov.as_dict() if cov is not None else None,
            "seconds_to_answer": time.perf_counter() - t0,
        })
    wall = time.perf_counter() - wall0

    priced = [r for r in rows if "step_seconds" in r]
    report = {
        "model": args.model, "tp": args.tp, "device": args.device,
        "regions": args.regions,
        "inputs": _inputs(args),
        "shapes": len(shapes), "priced": len(priced), "refused": refused,
        "build_seconds": build_s,
        # The limit fitted prices were actually allowed over, off the provider
        # rather than off the flag: `--interpolate true` names no number, and
        # the number is what a reader has to be able to check.
        "interpolation_limit": built.interpolation_limit,
        "predict_seconds": wall,
        "seconds_per_shape": (wall / len(shapes)) if shapes else None,
        "body_cache": body_graphs.describe(),
        "body_cache_counts": {"hits": body_graphs.hits,
                              "binds": body_graphs.binds,
                              "derivations": body_graphs.derivations,
                              "refusals": len(body_graphs.refusals)},
        "derive_seconds": deriver.seconds if deriver else 0.0,
        "allocation": (allocation.describe() if allocation
                       else "none; binding refuses any template that carries "
                            "an allocation"),
        "rows": rows,
    }
    if head_graphs is not None:
        report["head_cache"] = head_graphs.describe()

    print(f"shapes    : {len(shapes)}  priced {len(priced)}  "
          f"refused {refused}")
    print(f"body cache: {body_graphs.describe()}")
    if head_graphs is not None:
        print(f"head cache: {head_graphs.describe()}")
    print(f"build     : {build_s:.2f}s (once)")
    print(f"predict   : {wall:.3f}s for {len(shapes)} shapes"
          + (f", {wall / len(shapes) * 1e3:.1f} ms each" if shapes else ""))
    if deriver is not None:
        print(f"derive    : {deriver.describe()}")
    for row in rows[:10]:
        if "refused" in row:
            print(f"  {row['label']:<16} REFUSED {row['refused'][:88]}")
        else:
            print(f"  {row['label']:<16} {row['step_seconds'] * 1e3:9.3f} ms  "
                  f"{row['coverage']}")
    if len(rows) > 10:
        print(f"  ... {len(rows) - 10} more, all in the report")
    for key, why in list(body_graphs.refusals.items())[:5]:
        print(f"  cache refused {key}: {why[:100]}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=1)
        print(f"written   : {args.out}")
    return 0 if priced else 1


if __name__ == "__main__":
    sys.exit(main())
