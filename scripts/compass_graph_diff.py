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

    python scripts/compass_graph_diff.py trace --device meta --model M -o meta.json
    python scripts/compass_graph_diff.py trace --device cuda --model M -o real.json
    python scripts/compass_graph_diff.py diff meta.json real.json
"""

import argparse
import os
import sys
import time

import torch


def _init_env(tp: int) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29593")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")


def _trace(model, input_ids, positions):
    """Run one forward under both tracers, returning the combined graph."""
    from atom.compass.runtime.meta import MetaOpTracer
    from atom.compass.runtime.triton_trace import TritonLaunchTracer

    ops = MetaOpTracer()
    triton = TritonLaunchTracer(graph=ops.graph)
    t0 = time.perf_counter()
    with triton, ops, torch.inference_mode():
        model(input_ids, positions)
    return ops.graph, time.perf_counter() - t0


def _trace_cmd(args) -> int:
    _init_env(args.tp)
    from aiter import init_dist_env

    init_dist_env(args.tp, rankID=0, backend="gloo",
                  distributed_init_method="env://", local_rank=0)

    from atom.compass.core.graph import GraphKey
    from atom.config import Config, set_current_atom_config
    from atom.model_engine.model_runner import support_model_arch_dict
    from atom.utils import resolve_obj_by_qualname

    config = Config(model=args.model, tensor_parallel_size=args.tp, load_dummy=True)
    set_current_atom_config(config)
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

    graph, trace_s = _trace(
        model,
        torch.zeros(args.tokens, dtype=torch.long, device=device),
        torch.arange(args.tokens, dtype=torch.long, device=device),
    )
    graph.key = GraphKey.of(
        model_id=f"{arch}@{os.path.basename(args.model.rstrip('/'))}",
        topology={"tp": args.tp},
        rank_coords={"tp": 0},
        batch_signature=(args.tokens,),
    )
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


def main() -> int:
    ap = argparse.ArgumentParser(description="ATOMCompass graph trace and diff")
    sub = ap.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("trace", help="trace one forward and write the graph")
    tr.add_argument("--model", required=True)
    tr.add_argument("--device", default="meta", choices=["meta", "cuda", "cpu"])
    tr.add_argument("--tokens", type=int, default=8)
    tr.add_argument("--tp", type=int, default=1)
    tr.add_argument("-o", "--out", required=True)
    tr.set_defaults(func=_trace_cmd)

    df = sub.add_parser("diff", help="compare two written graphs")
    df.add_argument("left")
    df.add_argument("right")
    df.add_argument("--ignore-dtypes", action="store_true")
    df.set_defaults(func=_diff_cmd)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
