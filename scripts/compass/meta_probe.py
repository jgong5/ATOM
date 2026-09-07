"""Step-2 probe: what stops ATOM's model from running on meta tensors?

Builds the model on the meta device — structure only, no weights, no GPU — and
runs a forward under a dispatch tracer. Reports the operators that executed and
the ones that could not, which is the meta-kernel worklist.

The list has to be discovered rather than read off the source: AITER registers
its operators lazily through JIT, so only a run reveals which ones a given model
actually reaches.

    python scripts/compass/meta_probe.py --model <path> [--tokens 8] [--tp 1]
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
    ap.add_argument("--profile-out",
                    help="write a memory profile for --compass-memory-model")
    ap.add_argument("--graph", help="traced op graph to name in the profile")
    ap.add_argument("--calibration",
                    help="collective constants to name in the profile")
    ap.add_argument("--weights-only", action="store_true",
                    help="report the weight and buffer terms and stop, without "
                         "tracing a forward")
    args = ap.parse_args()

    from atom.config import Config
    from atom.model_engine.model_runner import support_model_arch_dict
    from atom.utils import resolve_obj_by_qualname

    # ATOM's parallel layers query their communication group while being
    # constructed, so the groups have to exist before the model does — even at
    # world size one, and even though nothing will be communicated on meta.
    import os

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    # A fixed port collides on a host-networked container; let the OS choose.
    import socket

    with socket.socket() as _sock:
        _sock.bind(("127.0.0.1", 0))
        os.environ.setdefault("MASTER_PORT", str(_sock.getsockname()[1]))
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    from aiter import init_dist_env

    # One real rank, however wide the configuration being modelled. Asking
    # gloo for a world of N from one process waits forever for peers that will
    # never arrive -- which is what `--tp 2` used to do here, silently, until
    # the timeout. `simulate_group_width` is the mechanism that makes the
    # group *report* the wider size while staying one rank, so every shard
    # computation in the tree sizes itself for the width being modelled.
    init_dist_env(
        1,
        rankID=0,
        backend="gloo",
        distributed_init_method="env://",
        local_rank=0,
    )
    from atom.compass.runtime.derive import simulate_group_width

    # Before the model is built: layers read the width while being constructed,
    # so patching afterwards changes nothing already sized.
    simulate_group_width(args.tp, physical=1)

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
    # The model's dtype is the config's, not torch's default. Without this the
    # meta build comes out in float32 and every byte it reports is twice what
    # the loaded model holds -- which is exactly how this was caught.
    was_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(config.torch_dtype)
        with torch.device("meta"):
            model = model_class(config)
    except Exception as exc:  # noqa: BLE001
        print(f"model construction on meta failed: {type(exc).__name__}: {exc}")
        return 1
    finally:
        torch.set_default_dtype(was_dtype)
    build_s = time.perf_counter() - build_t0
    print(f"built on meta in {build_s:.2f}s\n")

    # What this rank's weights weigh, by ATOM's own sharding rather than by a
    # rule about it -- and without a device, so it answers for a width there
    # are no GPUs for. Meta tensors have shape and dtype and no storage.
    from atom.compass.core.memory_model import resident_bytes

    tied = bool(getattr(config.hf_config, "tie_word_embeddings", False))
    parameters, buffers = resident_bytes(model, tied_head=tied)
    print("parameters   : %.4f GiB per rank at tp=%d" % (parameters / 2**30, args.tp))
    print("buffers      : %.4f GiB (built at init, absent from the checkpoint)\n"
          % (buffers / 2**30))
    if args.profile_out:
        import json

        with open(args.profile_out, "w", encoding="utf-8") as fh:
            json.dump({
                "version": 1,
                "model": args.model,
                "world_size": args.tp,
                "parameters": parameters,
                "buffers": buffers,
                # Filled in by whoever has them; the runner resolves both.
                "graph": args.graph or None,
                "calibration": args.calibration or None,
            }, fh, indent=1)
        print("profile written to %s" % args.profile_out)
    if args.weights_only:
        return 0

    from atom.compass.runtime.meta import derived_inputs

    input_ids, positions = derived_inputs(args.tokens, "meta")

    tracer = MetaOpTracer(topology={"tp": args.tp})
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
