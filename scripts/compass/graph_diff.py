"""Step-3 check: does a meta-derived graph match one captured on hardware?

Traces one forward and writes the graph out, or compares two such graphs
operator by operator.

Tracing happens one graph per process because ATOM registers attention layers
in a global table, so a process can only build a model once. That suits the
intended use anyway: derive anywhere, capture on a GPU box, compare either.

This is the check that makes meta-derivation trustworthy. If the graphs agree,
a graph can be derived for a configuration nobody has run, and Compass can
sweep without a GPU per point. If they disagree, the disagreements are the bugs.

Weights are random: the comparison is structural, and values never enter it.

    python scripts/compass/graph_diff.py trace --device meta --model M -o meta.json
    python scripts/compass/graph_diff.py trace --device cuda --model M -o real.json
    python scripts/compass/graph_diff.py diff meta.json real.json

To validate derivation against what the engine really ran, capture through the
runner and compare with ``compare`` rather than ``diff`` — a capture holds the
runner's work as well as the model's, so containment is the question, not
equality::

    python scripts/compass/smoke.py --model M --level 0 --compass \
        --compass-mode trace --compass-graph-out capture.json
    python scripts/compass/graph_diff.py trace --device meta --model M \
        --tokens 1 -o derived.json
    python scripts/compass/graph_diff.py compare derived.json capture.json
"""

import argparse
import sys

# How a graph is derived, and what its provenance says, now live in the runtime
# package rather than in this script. Two reasons, and only one of them is
# tidiness: a graph per process pays the model build and ~9.5s of imports for
# every shape, which is the wrong cost for a cache that misses; and product code
# -- a cost oracle deriving a shape it has no template for -- cannot import a
# script. The CLI below is unchanged, and so is every field it writes.
# See `atom/compass/runtime/tracer.py`.
from atom.compass.runtime.tracer import (TRACE_TOKENS_DEFAULT, ModelTracer,
                                         TraceRequest)


def reconcile_declared(spec, args):
    """Put the declared deployment inputs on the batch, or say why not.

    ``--cudagraph-mode`` and ``--capture-bucket`` reach the provenance through
    the `TraceRequest`, and the *installed* batch through the spec. Nothing
    used to carry them across, so a spec silent on both was traced under the
    eager rule -- `max_seqlen_k` = the batch's longest context -- while the
    provenance beside it recorded `cudagraph_mode: full` from the command
    line. That is the seed a FULL bind now refuses: the label and the extent
    disagree, and under FULL the extent is kept verbatim rather than
    recomputed.

    So reconcile here, where both are in hand. A spec that declares them wins
    nothing and loses nothing -- it must simply agree with the flags -- and a
    spec that is silent takes the declaration, which is the only way a flag
    reaches `BatchSpec.launch_max_seqlen_k` at all. A flag left off takes the
    spec's value onto the request, so the provenance does not record `null`
    for something the batch declared.

    Returns ``(spec, None)`` or ``(spec, message)``; the message is a
    contradiction the caller should refuse on rather than resolve.
    """
    from atom.compass.runtime.batch_spec import BatchSpec

    overrides = {}
    for flag, cast in (("cudagraph_mode", str), ("capture_bucket", int)):
        declared = getattr(args, flag, None)
        on_spec = getattr(spec, flag)
        if declared is None:
            setattr(args, flag, on_spec)
            continue
        if on_spec is not None and cast(on_spec) != cast(declared):
            return spec, (
                f"--{flag.replace('_', '-')} {declared!r} contradicts the "
                f"batch spec, which declares {on_spec!r}; the trace installs "
                "the spec and the provenance records the flag, so the two "
                "would disagree in the artifact")
        if on_spec is None:
            overrides[flag] = declared
    if overrides:
        spec = BatchSpec.from_dict({**spec.to_dict(), **overrides})
    return spec, None


def _trace_cmd(args) -> int:
    # Read before anything else. A misspelled field or an impossible batch
    # should say so now, not after a 27B model has been built on meta.
    spec = None
    if args.batch_spec:
        from atom.compass.runtime.batch_spec import BatchSpec

        spec = BatchSpec.load(args.batch_spec)
        if args.tokens != spec.num_tokens:
            if args.tokens != TRACE_TOKENS_DEFAULT:
                print(f"--tokens {args.tokens} contradicts the batch spec, "
                      f"which computes {spec.num_tokens}", file=sys.stderr)
                return 2
            args.tokens = spec.num_tokens
        spec, why = reconcile_declared(spec, args)
        if why is not None:
            print(why, file=sys.stderr)
            return 2
    tracer = ModelTracer.build(args.model, args.tp, args.device,
                               replay_target=getattr(args, "replay_target",
                                                     None))
    if tracer.bootstrap:
        state = tracer.bootstrap
        print(f"### derive bootstrap: arch={state['arch']} "
              f"installed={state['installed']} ({state['reason']})", flush=True)

    traced = tracer.trace(spec, request=TraceRequest(
        tp=args.tp, rank=args.rank, region=getattr(args, "region", "body"),
        cudagraph_mode=getattr(args, "cudagraph_mode", None),
        capture_bucket=getattr(args, "capture_bucket", None),
        tokens=args.tokens, model=args.model))
    graph, trace_s = traced.graph, traced.trace_s
    build_s, redirected, device = tracer.build_s, traced.redirected, tracer.device
    graph.save(args.out)

    resident = ""
    if device.type == "cuda":
        import torch

        resident = f", {torch.cuda.memory_allocated() / 2**30:.1f} GiB resident"
    print(f"device    : {device.type}")
    if spec is not None:
        print(f"batch     : {spec.kind} bs={spec.batch_size} "
              f"tokens={spec.num_tokens} "
              f"context={min(spec.context_lens)}..{max(spec.context_lens)} "
              f"block={spec.block_size} bucket={spec.capture_bucket}")
    else:
        print("batch     : bare body pass, no forward context installed "
              "(attention will be unpriceable)")
    print(f"operators : {len(graph)} ({len(graph.op_names())} distinct)")
    print(f"built in  : {build_s:.2f}s | traced in {trace_s:.3f}s{resident}")
    if redirected:
        print(f"redirected: {redirected} cuda factory calls to meta "
              f"(dispatch keys only; no device was used)")
    print(f"written   : {args.out}")
    return 0


def _diff_cmd(args) -> int:
    from atom.compass.core.diff import diff_graphs
    from atom.compass.core.graph import OpGraph

    left, right = OpGraph.load(args.left), OpGraph.load(args.right)
    result = diff_graphs(left, right, compare_dtypes=not args.ignore_dtypes)
    print("ATOMCompass meta-vs-capture diff")
    print("=" * 66)
    if left.key and right.key and left.key != right.key:
        print("  WARNING: graphs describe different configurations")
        print(f"    left : {left.key}")
        print(f"    right: {right.key}")
    print(result.report())
    print("=" * 66)
    if result.identical:
        print("  Meta derivation is validated for this configuration:")
        print("  a graph for an un-captured configuration can be trusted.")
        return 0
    print("  Meta derivation does NOT reproduce hardware here.")
    return 1


def _compare_cmd(args) -> int:
    """Check that a derived graph is contained in a capture, in order.

    This is the validation that matters, and it is not equality. A capture also
    holds the runner's own work — batch metadata, the LM head, sampling — which
    the model body has no reason to contain. What must hold is that every
    operator the model performs appears in the capture, in order, at the same
    shapes.
    """
    from atom.compass.core.diff import align_graphs
    from atom.compass.core.graph import OpGraph

    derived = OpGraph.load(args.derived)
    captured = OpGraph.load(args.captured)

    print("ATOMCompass derivation-vs-capture check")
    print("=" * 66)
    for label, graph in (("derived", derived), ("captured", captured)):
        prov = graph.provenance or {}
        if prov:
            print(f"  {label:<9}: " + ", ".join(
                f"{k}={v}" for k, v in sorted(prov.items())))

    level = (captured.provenance or {}).get("compilation_level")
    if level:
        print(f"\n  WARNING: the capture was recorded at compilation level {level}.")
        print("  Inductor-fused operators reach neither tracer, so the capture is")
        print("  missing an unknown number of them. Recapture with --level 0.")

    dl = derived.key.batch_signature if derived.key else None
    cl = captured.key.batch_signature if captured.key else None
    if dl and cl and sum(dl) != sum(cl):
        print(f"\n  WARNING: different batches ({sum(dl)} tokens vs {sum(cl)}).")
        print("  Shapes will differ for reasons that carry no information;")
        print("  derive at the token count the capture used.")

    result = align_graphs(derived, captured, compare_dtypes=not args.ignore_dtypes)
    print()
    print(result.report())
    print("=" * 66)
    if result.contained:
        print("  Derivation reproduces hardware for this configuration.")
        print("  A graph derived for a configuration nobody has run can be trusted")
        print("  to the same extent.")
        return 0
    print("  Derivation does NOT reproduce hardware here.")
    print("  The unmatched operators above are the gap.")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description="ATOMCompass graph trace and diff")
    sub = ap.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("trace", help="trace one forward and write the graph")
    tr.add_argument("--model", required=True)
    tr.add_argument("--device", default="meta", choices=["meta", "cuda", "cpu"])
    tr.add_argument("--tokens", type=int, default=TRACE_TOKENS_DEFAULT,
                    help="Body tokens for a bare trace. Ignored when "
                         "--batch-spec is given, which says how many.")
    tr.add_argument("--tp", type=int, default=1)
    tr.add_argument("--rank", type=int, default=0,
                    help="Which rank of the group to derive. Any rank can be "
                         "derived from any process; nothing is communicated.")
    tr.add_argument("--replay-target", default=None,
                    help="A captured target.json to take the architecture "
                         "from, so a graph can be derived on a machine with "
                         "no GPU for AITER to interrogate.")
    tr.add_argument("--batch-spec", default=None,
                    help="A JSON BatchSpec saying which forward to derive: "
                         "prefill or decode, per-request query and context "
                         "lengths, block size, capture bucket. Without it the "
                         "trace is a bare --tokens body pass with no forward "
                         "context, and every attention operator comes out "
                         "unpriceable.")
    tr.add_argument("--region", default="body",
                    choices=["body", "head", "both"],
                    help="What to record: the model forward (body, the "
                         "default, so every existing invocation and artifact "
                         "is unchanged), the runner's compute_logits (head -- "
                         "the LM-head GEMM and its TP all-gather, which no "
                         "body graph contains and which is therefore priced "
                         "at zero today), or the two together.")
    tr.add_argument("--cudagraph-mode", default=None,
                    choices=["full", "piecewise", "eager"],
                    help="The deployment's cudagraph mode, a declared config "
                         "input. Without it a graph cannot say whether "
                         "production runs its LM head inside the replayed body "
                         "-- that depends on the branch run_model takes, not "
                         "on TP alone -- and the artifact records the question "
                         "as unanswered rather than guessing an answer.")
    tr.add_argument("--capture-bucket", type=int, default=None,
                    help="Rows the replayed body actually executes, when the "
                         "step replays a graph. A decode of 20 sequences whose "
                         "graph was captured at running_bs 32 forwards 32 "
                         "rows and slices to 20 afterwards; a graph traced at "
                         "20 is then the eager body, not the replayed one.")
    tr.add_argument("-o", "--out", required=True)
    tr.set_defaults(func=_trace_cmd)

    cp = sub.add_parser(
        "compare",
        help="check a derived graph is contained in a runner capture",
    )
    cp.add_argument("derived")
    cp.add_argument("captured")
    cp.add_argument("--ignore-dtypes", action="store_true")
    cp.set_defaults(func=_compare_cmd)

    df = sub.add_parser("diff", help="compare two written graphs")
    df.add_argument("left")
    df.add_argument("right")
    df.add_argument("--ignore-dtypes", action="store_true")
    df.set_defaults(func=_diff_cmd)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
