"""Step-2 probe: what stops ATOM's model from running on meta tensors?

Builds the model on the meta device — structure only, no weights, no GPU — and
runs a forward under a dispatch tracer. Reports the operators that executed and
the ones that could not, which is the meta-kernel worklist.

The list has to be discovered rather than read off the source: AITER registers
its operators lazily through JIT, so only a run reveals which ones a given model
actually reaches.

    python scripts/compass_meta_probe.py --model <path> [--tokens 8] [--tp 1]
"""

import argparse
import sys
import time

import torch

from atom.compass.runtime.meta import MetaOpTracer, MetaTrace
from atom.compass.runtime.triton_trace import TritonLaunchTracer


def main() -> int:
    ap = argparse.ArgumentParser(description="ATOMCompass meta probe")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokens", type=int, default=8, help="tokens in the probe batch")
    ap.add_argument("--tp", type=int, default=1, help="tensor parallel size to model")
    ap.add_argument("--show-ops", action="store_true", help="list executed operators")
    args = ap.parse_args()

    from atom.config import Config
    from atom.model_engine.model_runner import support_model_arch_dict
    from atom.utils import resolve_obj_by_qualname

    # ATOM's parallel layers query their communication group while being
    # constructed, so the groups have to exist before the model does — even at
    # world size one, and even though nothing will be communicated on meta.
    import os

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29591")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    from aiter import init_dist_env

    init_dist_env(
        args.tp,
        rankID=0,
        backend="gloo",
        distributed_init_method="env://",
        local_rank=0,
    )

    config = Config(model=args.model, tensor_parallel_size=args.tp)

    # Layers read the active config from module scope while being built.
    from atom.config import set_current_atom_config

    set_current_atom_config(config)

    arch = config.hf_config.architectures[0]
    qualname = support_model_arch_dict.get(arch)
    if qualname is None:
        print(f"no ATOM model class registered for architecture {arch!r}")
        return 2
    model_class = resolve_obj_by_qualname(qualname)
    print(f"architecture : {arch}\nmodel class  : {qualname}\ntokens       : {args.tokens}\n")

    # Structure only: meta tensors have shape and dtype but no storage, so this
    # allocates nothing and needs no GPU.
    build_t0 = time.perf_counter()
    try:
        with torch.device("meta"):
            model = model_class(config)
    except Exception as exc:  # noqa: BLE001
        print(f"model construction on meta failed: {type(exc).__name__}: {exc}")
        return 1
    build_s = time.perf_counter() - build_t0
    print(f"built on meta in {build_s:.2f}s\n")

    input_ids = torch.zeros(args.tokens, dtype=torch.long, device="meta")
    positions = torch.arange(args.tokens, dtype=torch.long, device="meta")

    tracer = MetaOpTracer()
    # Triton kernels never reach the dispatcher, so they need their own
    # interception. Both write into one graph, preserving execution order.
    triton_tracer = TritonLaunchTracer(graph=tracer.graph)
    failure = None
    completed = False
    try:
        with triton_tracer, tracer, torch.inference_mode():
            model(input_ids, positions)
        completed = True
    except Exception as exc:  # noqa: BLE001
        failure = f"{type(exc).__name__}: {' '.join(str(exc).split())[:160]}"

    trace = MetaTrace(
        graph=tracer.graph, missing=tracer.missing,
        seconds=tracer.seconds, completed=completed, failure=failure,
    )
    print(trace.report())
    print(triton_tracer.summary())

    if args.show_ops:
        print("\noperators executed (count):")
        for name, n in sorted(trace.graph.counts().items(), key=lambda kv: -kv[1]):
            print(f"  {n:6d}  {name}")

    return 0 if completed else 1


if __name__ == "__main__":
    sys.exit(main())
