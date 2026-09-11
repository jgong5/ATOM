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

import contextlib
import argparse
import os
import sys
import time

import torch

#: What `--tokens` means when nobody said otherwise. Named so a batch spec can
#: tell "the default was left alone" from "the caller asked for this many".
TRACE_TOKENS_DEFAULT = 8


def _free_port() -> str:
    """Ask the OS for a port nobody is using.

    A fixed port is wrong here. The container runs with host networking on a
    shared machine, so a hardcoded number collides with whatever else happens to
    hold it — including a previous run of this same script — and the failure
    (``EADDRINUSE`` from the rendezvous store) says nothing about tracing.
    """
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return str(sock.getsockname()[1])


# Named per region so a graph says what it holds, and a sum over two graphs
# can be checked for double counting or for a hole. `sampler` and
# `input preparation` are excluded everywhere because no region traces them;
# they are the remaining named gap in the step, not a covered term.
_REGION_INCLUDES = {
    "body": ["model forward"],
    "head": ["compute_logits"],
    "both": ["model forward", "compute_logits"],
}
_REGION_EXCLUDES = {
    "body": ["compute_logits", "sampler", "input preparation"],
    "head": ["model forward", "sampler", "input preparation"],
    "both": ["sampler", "input preparation"],
}


def _init_env(tp: int) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", os.environ.get("MASTER_PORT") or _free_port())
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")


def _trace(model, input_ids, positions, topology=None, on_meta=False,
           spec=None, region="body"):
    """Run one forward under the tracers, returning the combined graph.

    ``region`` says what is recorded.

    ``body`` is the model forward, and is what every graph before this one
    held. ``head`` is the runner's ``compute_logits``: the LM-head projection
    and, at TP>1, the all-gather over its vocab shard. No body graph contains
    either, so both were priced at zero -- the failure mode this project keeps
    naming, an unmeasured family summed as nothing and the step called
    complete. ``both`` records the two together.

    For ``head`` the forward still runs, because the head needs its hidden
    states, but outside the tracers so its operators are not counted twice.

    The head is not a projection of the whole chunk. `ParallelLMHead.forward`
    reads the forward context and, on prefill, keeps one row per sequence
    (`x[cu_seqlens_q[1:] - 1]`); on decode it keeps every row, because decode
    already has one row per sequence. So this must be traced with the batch
    spec installed, exactly as attention is, or a 16384-token prefill would
    record a head 16384 rows wide against the 1 row production computes.

    Still outside every region: sampling, input preparation, and the runner's
    own host work. Those are named in the graph's provenance rather than left
    for a reader to assume covered.
    """
    from atom.compass.runtime.derive import (record_collectives,
                                             redirect_device_factories)
    from atom.compass.runtime.meta import MetaOpTracer
    from atom.compass.runtime.triton_trace import TritonLaunchTracer

    ops = MetaOpTracer(topology=topology)
    triton = TritonLaunchTracer(graph=ops.graph)
    collectives = record_collectives(ops.graph)
    # Only on meta, and reported: see redirect_device_factories.
    factories = (redirect_device_factories() if on_meta
                 else contextlib.nullcontext())
    # The batch's metadata, on the forward context, where attention reads it.
    # Without this the model still traces -- attention runs, its shapes are
    # right -- and every attention operator is recorded with no context, which
    # makes it unpriceable. See atom/compass/runtime/batch_spec.py.
    if spec is not None:
        from atom.compass.runtime import batch_spec as bs

        installed = bs.install(spec)
    else:
        installed = contextlib.nullcontext()
    if region not in ("body", "head", "both"):
        raise ValueError(f"unknown region {region!r}")

    with factories, installed, torch.inference_mode():
        if region == "head":
            # The forward still has to run, because the head needs its hidden
            # states -- but its operators are not this graph's, so they go
            # into a graph that is thrown away.
            #
            # It cannot simply run untraced. On meta a Triton launch reaches
            # the real Triton runtime and dies there ("0 active drivers"),
            # so the launch tracer is not only a recorder: it is what makes a
            # device-free forward possible at all. Hence a second, discarded
            # tracer rather than no tracer.
            scratch = MetaOpTracer(topology=topology)
            with record_collectives(scratch.graph), \
                    TritonLaunchTracer(graph=scratch.graph), scratch:
                hidden = model(input_ids, positions)
            t0 = time.perf_counter()
            with collectives, triton, ops:
                model.compute_logits(hidden)
        elif region == "both":
            t0 = time.perf_counter()
            with collectives, triton, ops:
                model.compute_logits(model(input_ids, positions))
        else:
            t0 = time.perf_counter()
            with collectives, triton, ops:
                model(input_ids, positions)
        trace_s = time.perf_counter() - t0
    return ops.graph, trace_s, getattr(factories, "redirected", 0)


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
    _init_env(args.tp)
    if getattr(args, "replay_target", None):
        # Deriving on a device-free machine. AITER resolves the chip at import
        # time by shelling out to `rocminfo`, which fails where there is no GPU,
        # and derivation's whole point is to run there. The replay path already
        # answers that question from a captured record; derivation asks it for
        # the same reason, so it uses the same seam rather than a second one.
        # Before the `aiter` import below, which is what triggers the query.
        from atom.compass.replay.bootstrap import install_from_target

        state = install_from_target(args.replay_target)
        print(f"### derive bootstrap: arch={state['arch']} "
              f"installed={state['installed']} ({state['reason']})", flush=True)
    from aiter import init_dist_env

    # Derivation builds the group at world size ONE, whatever TP width is being
    # derived, and then tells the group to report the wider width.
    #
    # A real group of size N needs N processes to arrive. That defeats the whole
    # purpose here: the point of derivation is to produce a sharded rank's graph
    # from one process, on no GPUs, for a configuration nobody has run. Asking
    # gloo for a world of 2 from a single process simply waits forever.
    #
    # ATOM already solves this for benchmarking -- `apply_simulated_tp` reports
    # a logical width while the real group stays physical, all the way down to
    # one rank -- and the reason it works there is the reason it works here:
    # every shard-size computation bottoms out at `get_tp_group().world_size`.
    # Its caveat, that collectives over absent ranks make the *output*
    # meaningless, costs derivation nothing, because derivation never looks at
    # the output. Only shapes are recorded, and shapes stay right.
    init_dist_env(1, rankID=0, backend="gloo",
                  distributed_init_method="env://", local_rank=0)

    from atom.compass.core.graph import GraphKey
    from atom.config import Config, set_current_atom_config
    from atom.model_engine.model_runner import support_model_arch_dict
    from atom.utils import resolve_obj_by_qualname

    config = Config(model=args.model, tensor_parallel_size=args.tp, load_dummy=True)
    set_current_atom_config(config)
    if args.tp > 1:
        from atom.compass.runtime.derive import simulate_group_width

        simulate_group_width(args.tp)
    arch = config.hf_config.architectures[0]
    model_class = resolve_obj_by_qualname(support_model_arch_dict[arch])

    device = torch.device(args.device)
    # Build in the model's own dtype. Left at the fp32 default, meta happily
    # traces kernels that real hardware rejects — AITER's fused qk-rmsnorm takes
    # only fp16/bf16 — so the two graphs would not be comparable in the one way
    # that matters.
    build_t0 = time.perf_counter()
    with torch.device(device):
        prev_dtype = torch.get_default_dtype()
        torch.set_default_dtype(config.torch_dtype)
        try:
            model = model_class(config)
        finally:
            torch.set_default_dtype(prev_dtype)
    if device.type != "meta":
        model = model.to(device)
    build_s = time.perf_counter() - build_t0

    if spec is None:
        from atom.compass.runtime.meta import derived_inputs

        inputs = derived_inputs(args.tokens, device)
    else:
        from atom.compass.runtime import batch_spec as bs

        inputs = bs.model_inputs(spec, device)

    # The trace runs in the model's dtype too, not only the build. A
    # library that creates a tensor without naming one gets the ambient
    # default, and AITER's dispatch dummy is exactly that: restored to
    # fp32 it is recorded as `...|1;4,5120;5120;4,5120|float32,bfloat16,
    # bfloat16,bfloat16`, against the capture's all-bfloat16, and every
    # fused qk-rmsnorm in the graph then misses its price by dtype alone.
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(config.torch_dtype)
    try:
        graph, trace_s, redirected = _trace(
            model, *inputs, topology={"tp": args.tp},
            on_meta=device.type == "meta", spec=spec,
            region=getattr(args, "region", "body"),
        )
    finally:
        torch.set_default_dtype(prev_dtype)
    graph.key = GraphKey.of(
        model_id=f"{arch}@{os.path.basename(args.model.rstrip('/'))}",
        topology={"tp": args.tp},
        rank_coords={"tp": args.rank},
        # Per-request scheduled tokens, which is what the runner records
        # (`runner.py`: `shape.num_scheduled_tokens`). A bare body pass is one
        # sequence of that many. The key still does not carry context length,
        # so a decode and a chunked prefill of the same query lengths key
        # alike -- the spec below is what tells them apart.
        batch_signature=spec.query_lens if spec else (args.tokens,),
    )
    graph.provenance = {
        "source": "derivation" if device.type == "meta" else "capture",
        "device": device.type,
        "compilation_level": 0,  # a bare model call is never compiled
        "tokens": args.tokens,
        # How many device-typed factory calls were sent to meta so the
        # operator after them could dispatch. Recorded, not hidden.
        "device_factories_redirected": redirected,
        # What this graph is a graph of, and -- as important -- what it is
        # still not. A reader summing it must be able to see the gap without
        # reading this script.
        "region": getattr(args, "region", "body"),
        "includes": _REGION_INCLUDES[getattr(args, "region", "body")],
        "excludes": _REGION_EXCLUDES[getattr(args, "region", "body")],
    }
    if getattr(args, "region", "body") in ("head", "both"):
        # Whether production runs this head at all, which is not a property of
        # the shape. A batch of pure middle chunks samples nothing and the
        # runner skips `compute_logits` entirely, so the graph below is then a
        # template for the step that *would* run it -- correct to derive, wrong
        # to charge. Recorded so a reader of the artifact cannot sum it without
        # meeting the question.
        graph.provenance["head_runs"] = (spec.produces_output()
                                         if spec is not None else None)
    if spec is not None:
        # The batch this graph is a graph of, written down in full, block
        # table included. Without it "4 tokens" is all a reader gets, and four
        # decodes at context 66 and a four-token prefill produce the same
        # number with two orders of magnitude between their KV traffic.
        graph.provenance["batch_spec"] = spec.to_dict()
    else:
        graph.provenance["forward_context"] = "none installed"
    graph.save(args.out)

    resident = ""
    if device.type == "cuda":
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
