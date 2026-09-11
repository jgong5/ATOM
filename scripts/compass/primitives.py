"""Price a graph's operators without standing up the deployment they came from.

The price lists collected so far were produced by `scripts/compass/run.py`,
which starts a real ATOM server at the target width -- weights loaded, KV pool
allocated, scheduler running, a throwaway workload served -- and only then
prices the graph. That is a working collector and a fair diagnostic, but it is
not the workflow the PoC promises. If pricing a candidate configuration
requires standing that configuration up, nothing has been avoided; and at
TP=1/0.40 it is not merely expensive but impossible, because the 27B DeltaNet
state pool does not fit, so the primitives cannot be measured at all on a card
that runs every one of them comfortably.

What the operators actually need is narrower than a deployment:

* Most of them need nothing. `aiter::gemm_a16w16`, `silu_and_mul`, the rmsnorm
  kernels, every `aten::` view and elementwise op: hand them tensors of the
  recorded shapes and dtypes and they run. No config, no model, no cache.
* Collectives need a real process group of the target width, and nothing else.
  An N-way all-reduce is irreducibly an N-device measurement; it is not an
  N-device *model*.
* Attention needs a layer object registered under the name the graph records
  (`base_attention.unified_attention_with_output_base` looks it up in
  `static_forward_context`), the parameters that layer's implementation reads,
  and a KV region to address. It does not need the other 27 billion parameters,
  a scheduler, or a pool sized for 512 concurrent requests.

So this script offers the three in order, and says which one it used. Run it at
``--layers none`` and everything but attention prices; the numbers can then be
compared against a list collected the old way, which is what shows the server
was contributing nothing to them.

The batch is not inferred. Attention reads its metadata from the forward
context, and the graph carries the context it was derived under
(`atom/compass/runtime/batch_spec.py`), so pricing installs that -- the same
recorded context, whatever stood the layer up.

    python scripts/compass/primitives.py --graph 'g.tp1.r*.json' \
        --model Qwen/Qwen3.8-27B --tp 1 --layers none -o primitives.json
"""

import argparse
import json
import os
import sys
import time


def _sha256(path: str) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hashes(graphs) -> dict:
    """Content hashes for the code that priced and the graphs it priced.

    A price list is reproducible only against the exact collector and the exact
    graph, and both move. Recording the digests costs nothing and makes a stale
    comparison visible instead of silent.
    """
    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    sources = ["scripts/compass/primitives.py",
               "atom/compass/runtime/standalone.py",
               "atom/compass/runtime/microbench.py",
               "atom/compass/runtime/forward_ctx.py"]
    out: dict = {"sources": {}, "graphs": {}}
    for rel in sources:
        full = os.path.join(root, rel)
        if os.path.exists(full):
            out["sources"][rel] = _sha256(full)
    for path in graphs or []:
        if os.path.exists(path):
            out["graphs"][os.path.basename(path)] = _sha256(path)
    return out


def _device_identity(index: int) -> dict:
    """Which physical card this rank is on, by an id the process did not choose.

    A logical index is not an answer: every rank reports device 0 both when the
    ranks are correctly masked onto distinct cards and when they are all sharing
    one. The UUID or PCI address is the same string only if the card is the same
    card.
    """
    import torch

    props = torch.cuda.get_device_properties(index)
    ident = {}
    for name in ("uuid", "pci_bus_id", "pci_device_id", "name"):
        value = getattr(props, name, None)
        if value is not None:
            ident[name] = str(value)
    ident["logical_index"] = index
    ident["visible_devices"] = os.environ.get("HIP_VISIBLE_DEVICES")
    return ident


def _require_distinct_devices(rank: int, world: int) -> list:
    """Gather every rank's physical identity, and refuse to benchmark a clash.

    This is the check the first TP2/TP4 standalone run did not have. Four ranks
    on one card do not fail; they contend, and the prices come out 2-3x high
    with nothing in the artifact to say why.
    """
    import torch
    import torch.distributed as dist

    mine = _device_identity(torch.cuda.current_device())
    if world == 1 or not dist.is_initialized():
        return [mine]
    gathered: list = [None] * world
    dist.all_gather_object(gathered, mine)
    keys = [g.get("uuid") or g.get("pci_bus_id") for g in gathered]
    if any(k is None for k in keys):
        raise SystemExit(
            "cannot identify the physical device for every rank; refusing to "
            f"benchmark without it: {gathered}")
    if len(set(keys)) != world:
        raise SystemExit(
            f"ranks are not on distinct physical GPUs: {keys}. Prices from a "
            "shared card are contention, not cost.")
    return gathered


def _free_port() -> str:
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return str(sock.getsockname()[1])


def _init_env(rank: int, world: int) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", os.environ.get("MASTER_PORT")
                          or _free_port())
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    os.environ["LOCAL_RANK"] = str(rank)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Price a graph's operators without a serving deployment")
    ap.add_argument("--graph", required=True,
                    help="Graph path, comma list or glob, as run.py takes it")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--model", required=True,
                    help="Only read for its config: dtype, head counts, block "
                         "size. No weights are loaded from it.")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--layers", default="none",
                    choices=["none", "meta", "attention"],
                    help="How much of the model to stand up. 'none': no model "
                         "at all, so attention cannot resolve its layer and is "
                         "reported unpriced. 'meta': layers registered but "
                         "their parameters are meta. 'attention': attention "
                         "modules materialised with random parameters and a "
                         "KV region sized to the graph's own block tables.")
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--cache", default="graph",
                    choices=["hot", "cold", "graph"])
    ap.add_argument("--replay-target", default=None,
                    help="A captured target.json, for AITER's import-time "
                         "architecture query where rocminfo cannot answer.")
    args = ap.parse_args()

    _init_env(args.rank, args.tp)
    if args.replay_target:
        from atom.compass.replay.bootstrap import install_from_target

        state = install_from_target(args.replay_target)
        print(f"### bootstrap: arch={state['arch']} ({state['reason']})",
              flush=True)

    import torch  # noqa: F401  (imported for its side effects on aiter)
    from aiter import init_dist_env

    # Each rank on its own card, before anything can allocate on the default
    # one. Passing local_rank to init_dist_env does not move the current device,
    # and every rank landing on device 0 does not fail -- it contends, and the
    # prices come out 2-3x high with nothing in the artifact to say why.
    if torch.cuda.device_count() <= args.rank:
        raise SystemExit(f"rank {args.rank} of {args.tp} has no device: "
                         f"{torch.cuda.device_count()} visible")
    torch.cuda.set_device(args.rank)

    # A real group of the target width. Collectives are the one family that
    # genuinely needs the other devices, and nothing else here does.
    t0 = time.perf_counter()
    init_dist_env(args.tp, rankID=args.rank, backend="nccl",
                  distributed_init_method="env://", local_rank=args.rank)
    dist_s = time.perf_counter() - t0
    placement = _require_distinct_devices(args.rank, args.tp)

    from atom.config import Config, set_current_atom_config

    config = Config(model=args.model, tensor_parallel_size=args.tp,
                    load_dummy=True)
    set_current_atom_config(config)

    model_s = 0.0
    stood_up: dict = {"summary": "no model built"}
    if args.layers != "none":
        from atom.compass.runtime.standalone import stand_up_layers

        t0 = time.perf_counter()
        stood_up = stand_up_layers(config, args.graph, args.layers)
        model_s = time.perf_counter() - t0

    from atom.compass.runtime.microbench import load_ops, price_graph

    t0 = time.perf_counter()
    result = price_graph(args.graph, iters=args.iters, warmup=args.warmup,
                         cache=args.cache)
    price_s = time.perf_counter() - t0

    import torch

    device = torch.device("cuda", torch.cuda.current_device())
    _, paths, _ = load_ops(args.graph)
    result.setdefault("provenance", {})
    result["provenance"]["collector"] = {
        # What stood behind the numbers, recorded so a reader never has to
        # infer it. A price list collected this way and one collected through
        # a live server are the same kind of thing only if they agree.
        "script": "scripts/compass/primitives.py",
        "layers": args.layers,
        "stood_up": stood_up,
        "served_workload": False,
        "weights_loaded": False,
        "model_runner_initialized": False,
        "world_size": args.tp,
        "device_index": torch.cuda.current_device(),
        "visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
        "placement": placement,
        "rank": args.rank,
        "seconds": {"dist": round(dist_s, 2), "model": round(model_s, 2),
                    "price": round(price_s, 2)},
        "device_bytes": {
            "peak_process": torch.cuda.max_memory_allocated(device),
            "reserved_process": torch.cuda.max_memory_reserved(device),
        },
        "hashes": _hashes(paths),
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)

    p, u = result.get("prices") or {}, result.get("unpriced") or {}
    cov = result.get("coverage") or {}
    peak = torch.cuda.max_memory_allocated(device) / (1 << 20)
    census = (stood_up.get("census") or {}).get("rest") or {}
    print(f"layers    : {args.layers} ({stood_up['summary']})")
    print(f"priced    : {len(p)} signatures, {len(u)} not; "
          f"{cov.get('operators_priced', '?')}/{cov.get('operators', '?')} operators")
    print(f"seconds   : dist {dist_s:.1f}, model {model_s:.1f}, "
          f"price {price_s:.1f}")
    print(f"device    : peak {peak:.0f} MiB allocated; "
          f"{census.get('on_meta', 0)} target tensors left on meta "
          f"({census.get('bytes_avoided', 0) / (1 << 30):.1f} GiB not "
          f"materialised)")
    print(f"written   : {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
