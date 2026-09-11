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
        --head-price hprices.json:hgraph.json:unregistered \
        --template graph.json --head-template hgraph.json \
        --regions source-27b-tp1 --cudagraph-mode full \
        --shapes shapes.json -o out.json

``--shapes`` is a JSON list of step shapes, in the field names `StepShape`
uses. ``--template`` seeds the cache with a graph already on disk, keyed by the
structure its own recorded batch spec describes -- so a run can reuse the v2
artifacts instead of deriving what has already been derived.
"""

import argparse
import json
import sys
import time

#: Region models, by the name a caller may ask for. A name that is not here is
#: an error rather than a default: running with no region model at all and
#: running with the source's is a difference of tens of percent in the step, and
#: a typo must not silently pick one.
REGION_MODELS = {
    "source-27b-tp1": ("atom.compass.core.cost.regions", "SOURCE_27B_TP1"),
    "none": None,
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


def _coords(saved):
    """A coordinate map as saved: a dict, or the pair list JSON turned it into."""
    if not saved:
        return {}
    if isinstance(saved, dict):
        return dict(saved)
    return {k: v for k, v in saved}


def _template_shape(graph: dict):
    """The structure a graph on disk is a graph of, from its own provenance.

    A derived artifact records the batch it was traced over
    (`provenance.batch_spec`). Reading it back is the only way to key a
    pre-derived graph that does not involve the caller restating the batch and
    getting it wrong -- and a graph whose provenance does not carry one cannot
    be used as a template at all, because nothing says what it is a template
    *for*.
    """
    from atom.compass.core.cost.base import StepShape

    prov = graph.get("provenance") or {}
    spec = prov.get("batch_spec")
    if not spec:
        raise ValueError(
            "this graph records no batch_spec, so nothing says which structure "
            "it is a template for. Derive it with --batch-spec.")
    key = graph.get("key") or {}
    # A saved key writes its coordinate dicts as pair lists, because JSON has
    # no dict-of-tuples. Reading one back as a dict is not a conversion, it is
    # the inverse of how it was written.
    topology = _coords(key.get("topology"))
    rank_coords = _coords(key.get("rank_coords"))
    queries = tuple(spec["query_lens"])
    contexts = tuple(spec["context_lens"])
    prefill = 0 if spec.get("kind") == "decode" else sum(queries)
    # Which bucket a graph was captured at is an execution fact, and that is
    # where the tracer records it -- not in the batch spec, which describes the
    # requests. A graph derived with no bucket declared keys as None: it is a
    # template for the uncaptured structure, and claiming a bucket it was not
    # traced at would make it answer for a padded graph nobody derived.
    execution = prov.get("execution") or {}
    return StepShape(
        num_scheduled_tokens=queries,
        context_lens=contexts,
        num_prefill_tokens=prefill,
        topology=topology, rank_coords=rank_coords,
        capture_bucket=execution.get("capture_bucket"),
        compiled=None,
        produces_output=True,
    )


def _load_prices(entries):
    """``prices.json[:graph.json[:regime]]`` triples, as PriceLibrary takes."""
    loaded = []
    for entry in entries or ():
        parts = entry.split(":")
        if len(parts) == 1:
            loaded.append((parts[0], None))
        elif len(parts) == 2:
            loaded.append((parts[0], parts[1] or None))
        elif len(parts) == 3:
            loaded.append((parts[0], parts[1] or None, parts[2]))
        else:
            raise ValueError(f"--price {entry!r}: expected at most "
                             "prices.json:graph.json:regime")
    return loaded


def main() -> int:
    ap = argparse.ArgumentParser(
        description="price step shapes through the frozen composition")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--rank", type=int, default=0)
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

    from atom.compass.core.cost.library import (LibraryCostOracle,
                                                PriceLibrary)
    from atom.compass.runtime.templates import (CarriedAllocation,
                                                TemplateGraphs, template_key)

    with open(args.shapes, encoding="utf-8") as fh:
        raw_shapes = json.load(fh)
    shapes = [_shape_from(r) for r in raw_shapes]
    labels = [r.get("label") or f"#{i}" for i, r in enumerate(raw_shapes)]

    regions = None
    if REGION_MODELS[args.regions] is not None:
        module, name = REGION_MODELS[args.regions]
        regions = getattr(__import__(module, fromlist=[name]), name)

    library = PriceLibrary.load(_load_prices(args.price))

    allocation = None
    if args.carry_allocation:
        allocation = CarriedAllocation(
            "predict_step --carry-allocation: the template's block assignment "
            "is reused for every cohort bound to it")

    # The deriver, and the model it needs. Built once, before the loop, because
    # that is the whole reason this is a process and not a shell loop.
    tracer = deriver = head_deriver = None
    build_s = 0.0
    if not args.no_derive:
        from atom.compass.runtime.tracer import ModelTracer, ShapeDeriver

        t0 = time.perf_counter()
        tracer = ModelTracer.build(args.model, args.tp, args.device,
                                   replay_target=args.replay_target)
        build_s = time.perf_counter() - t0
        common = dict(block_size=args.block_size,
                      max_model_len=args.max_model_len,
                      position_rows=args.position_rows,
                      block_policy=args.block_policy,
                      cudagraph_mode=args.cudagraph_mode)
        deriver = ShapeDeriver(tracer, region="body", **common)
        if args.head:
            head_deriver = ShapeDeriver(tracer, region="head", **common)

    def seed(paths, source):
        graphs = {}
        for path in paths or ():
            with open(path, encoding="utf-8") as fh:
                graph = json.load(fh)
            graphs[template_key(_template_shape(graph))] = graph
        return TemplateGraphs(graphs, derive=source, allocation=allocation)

    body_graphs = seed(args.template, deriver)
    head_graphs = seed(args.head_template, head_deriver) if args.head else None

    oracle = LibraryCostOracle(library, body_graphs,
                               seconds_per_launch=args.seconds_per_launch,
                               head_graphs=head_graphs,
                               regions=regions,
                               require_complete=args.require_complete)

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
            "seconds_to_answer": time.perf_counter() - t0,
        })
    wall = time.perf_counter() - wall0

    priced = [r for r in rows if "step_seconds" in r]
    report = {
        "model": args.model, "tp": args.tp, "device": args.device,
        "regions": args.regions,
        "shapes": len(shapes), "priced": len(priced), "refused": refused,
        "build_seconds": build_s,
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
