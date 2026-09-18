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
import os
import sys
import time

import torch


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


def _init_env(tp: int) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", os.environ.get("MASTER_PORT") or _free_port())
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")


def _trace(model, input_ids, positions, topology=None):
    """Run one forward under the tracers, returning the combined graph."""
    from atom.compass.runtime.derive import record_collectives
    from atom.compass.runtime.meta import MetaOpTracer
    from atom.compass.runtime.triton_trace import TritonLaunchTracer

    ops = MetaOpTracer(topology=topology)
    triton = TritonLaunchTracer(graph=ops.graph)
    collectives = record_collectives(ops.graph)
    t0 = time.perf_counter()
    with collectives, triton, ops, torch.inference_mode():
        model(input_ids, positions)
    return ops.graph, time.perf_counter() - t0


def _trace_cmd(args) -> int:
    _init_env(args.tp)
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

    from atom.compass.runtime.meta import derived_inputs

    graph, trace_s = _trace(
        model, *derived_inputs(args.tokens, device), topology={"tp": args.tp}
    )
    graph.key = GraphKey.of(
        model_id=f"{arch}@{os.path.basename(args.model.rstrip('/'))}",
        topology={"tp": args.tp},
        rank_coords={"tp": args.rank},
        batch_signature=(args.tokens,),
    )
    graph.provenance = {
        "source": "derivation" if device.type == "meta" else "capture",
        "device": device.type,
        "compilation_level": 0,  # a bare model call is never compiled
        "tokens": args.tokens,
    }
    graph.save(args.out)

    resident = ""
    if device.type == "cuda":
        resident = f", {torch.cuda.memory_allocated() / 2**30:.1f} GiB resident"
    print(f"device    : {device.type}")
    print(f"operators : {len(graph)} ({len(graph.op_names())} distinct)")
    print(f"built in  : {build_s:.2f}s | traced in {trace_s:.3f}s{resident}")
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
    tr.add_argument("--tokens", type=int, default=8)
    tr.add_argument("--tp", type=int, default=1)
    tr.add_argument("--rank", type=int, default=0,
                    help="Which rank of the group to derive. Any rank can be "
                         "derived from any process; nothing is communicated.")
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
