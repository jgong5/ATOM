# SPDX-License-Identifier: MIT
"""What a capture states for the other ranks of its DP and EP groups.

A capture runs one rank, rank 0. Where ATOM reads its peers, two rules answer:

* Every peer scheduled what this rank did, but for its token count: the peers
  split evenly the rows the shape's MoE gathers (`Shape.gathered`) less this
  rank's (`split`), the first of them a prefill, as a decode gathering other
  than every rank's count has a prefilling peer, or each schedules this rank's
  count when the shape states none. The step's one DP all-gather
  (`sync_dp_metadata`) runs on the host with these rows. A drafter's pass
  all-reduces its own count (`DPMetadata.gather_running_tokens`): its first
  pass, the target's stream, takes the step's counts; a later one, a row a
  sequence, this rank's at every peer.
* What the DP group gathers for the MoE, or MORI dispatches to this rank, is
  every rank's rows: the shape's gathered rows, by default the group's width
  times this rank's; for MORI, the height ATOM trims its buffer to, `T_moe`. A
  drafter's later pass, a row a sequence, takes this rank's rows at every peer.
  The DP reduce-scatter gives back this rank's rows; MORI's combine gives back
  its buffer of `max_num_tokens_per_dp_rank` rows, which ATOM's finalize slices.
  In a symbolic trace the gathered rows are G, the recorder's symbol, where
  they are its hint (`forward._Recorder.rows`).

Each exchange is one `declared` op named `module:qualname` for the function
ATOM calls (`atom.model_ops.moe:all_gatherv`,
`mori.ops.dispatch_combine:EpDispatchCombineOp.dispatch`), its arguments the
ones ATOM passed by that function's parameter names, the group aside; MORI's
also state the EP width ATOM built their handle for, `num_ep_ranks`, which
their rows no longer give.

A graph that holds G is put at a G by `at`: G's slots take it, and each DP
exchange that holds G takes the `sizes` the first rule gives each rank there,
which ATOM keeps in numpy, where no symbol reaches.
"""

import inspect
import types

import numpy as np
import torch
from torch.utils import _pytree

from atom.compass import ir

#: MORI's dispatch-and-combine handle, whose two calls the EP exchanges are.
MORI = "mori.ops.dispatch_combine:EpDispatchCombineOp"
#: The DP exchanges whose `sizes` hold each rank's rows, this rank's first.
SIZED = ("atom.model_ops.moe:all_gatherv", "atom.model_ops.moe:reduce_scatterv")
#: G in a graph's JSON (`graph.to_json`).
SYMBOL = {"symbol": ir.G.name}
#: What `slots` marks a path to: G itself, or an exchange's `sizes`.
_G, _SIZES = "G", "sizes"


def split(rows: int, local: int, dp: int) -> list[int]:
    """Each rank's rows of the `rows` a DP group of `dp` gathers, this rank's
    `local` first: the peers split the rest evenly, the first ones a row more."""
    each, extra = divmod(rows - local, dp - 1)
    return [local] + [each + (k < extra) for k in range(dp - 1)]


def slots(region) -> dict:
    """Where `region`, a graph's JSON, holds G: a trie, keyed as the JSON is,
    of the paths to each `SYMBOL` and to the `sizes` of each exchange of
    `SIZED` that holds G."""
    trie = {}

    def mark(path, what):
        node = trie
        for key in path[:-1]:
            node = node.setdefault(key, {})
        node[path[-1]] = what

    def walk(node, path) -> bool:
        if node == SYMBOL:
            mark(path, _G)
            return True
        items = node.items() if isinstance(node, dict) else ()
        items = enumerate(node) if isinstance(node, list) else items
        # Every item walked: each may hold a slot of its own.
        held = sum(walk(value, (*path, key)) for key, value in items) > 0
        op = isinstance(node, dict) and node.get("op") in SIZED
        if held and op and "sizes" in node["args"]:
            mark((*path, "args", "sizes"), _SIZES)
        return held

    walk(region, ())
    return trie


def at(region, held: dict, g: int, dp: int, restated: bool = True):
    """`region`, a graph's JSON holding G at the `slots` `held`, at G = `g`:
    each G `g`, and unless `restated` is false each exchange's `sizes` this
    rank's then `split`'s over the `dp` ranks. What changes is copied, the
    rest shared."""
    node = region.copy()
    for key, what in held.items():
        if what == _G:
            node[key] = g
        elif what == _SIZES:
            node[key] = split(g, node[key][0], dp) if restated else node[key]
        else:
            node[key] = at(node[key], what, g, dp, restated)
    return node


def _declared(recorder, name, fn, kind=ir.NodeKind.DECLARED, stated=None):
    """`fn` as one op `name` of `kind`, its arguments as called and `stated`;
    what it dispatches inside is not recorded."""
    signature = inspect.signature(fn)

    def call(*args, **kwargs):
        with recorder.quiet():
            out = fn(*args, **kwargs)
        if recorder.on:
            slots = signature.bind(*args, **kwargs).arguments
            slots.update(slots.pop("kwargs", {}))
            slots.pop("group", None)
            slots.update(stated or {})
            outs = [t for t in _pytree.tree_leaves(out) if isinstance(t, torch.Tensor)]
            recorder.add(name, kind, slots, outs, None)
        return out

    return call


def patches(recorder):
    """`(owner, name, value)` for each patch that answers for the DP peers."""
    import torch.distributed as dist
    from aiter.dist.parallel_state import get_dp_group
    from torch._subclasses.fake_tensor import unset_fake_temporarily

    from atom.model_ops import moe
    from atom.utils.forward_context import DPMetadata
    from atom.utils.tbo import ubatching

    sync = ubatching.sync_dp_metadata
    dp = get_dp_group().world_size

    def on_host(fn, *args, **kwargs):
        with unset_fake_temporarily():
            return fn(*args, **kwargs)

    counts = recorder.counts = {}

    def peers(gathered, local, group=None, async_op=False):
        for slot in gathered:
            slot.copy_(local)
        if (rows := recorder.shape.gathered) is not None:
            for slot, n in zip(gathered[1:], split(rows, int(local[0]), dp)[1:]):
                slot[0] = n
            gathered[1][2] = 1  # is_prefill
        counts.clear()
        counts[int(local[0])] = [int(slot[0]) for slot in gathered]

    def gather_running_tokens(parallel_config, running_tokens):
        each = [running_tokens] * parallel_config.data_parallel_size
        return np.array(counts.get(running_tokens, each), dtype=np.int32)

    def sync_dp_metadata(**kwargs):
        gather, dist.all_gather = dist.all_gather, peers
        try:
            return on_host(sync, **kwargs)
        finally:
            dist.all_gather = gather

    def rows(x, height):
        if isinstance(x, (list, tuple)):
            return [rows(t, height) for t in x]
        return x.new_empty((recorder.rows(height), *x.shape[1:]))

    # `sizes` holds this rank's count in slot 0.
    def all_gatherv(x, sizes, group):
        return rows(x, sum(sizes))

    def reduce_scatterv(x, sizes, group):
        return rows(x, sizes[0])

    def reduce_scatter_with_unpadding(x, local_tokens):
        return rows(x, local_tokens)

    gathers = (all_gatherv, reduce_scatterv, reduce_scatter_with_unpadding)
    return [
        (ubatching, "sync_dp_metadata", sync_dp_metadata),
        (DPMetadata, "gather_running_tokens", staticmethod(gather_running_tokens)),
        *(
            (moe, f.__name__, _declared(recorder, f"{moe.__name__}:{f.__name__}", f))
            for f in gathers
        ),
    ]


def install_mori(recorder):
    """Give the EP group an all-to-all manager whose MORI handles are declared."""
    from aiter.dist.parallel_state import get_dp_group, get_ep_group

    group, dp = get_ep_group(), get_dp_group().world_size

    class Handle:
        """MORI's `EpDispatchCombineOp`, its two calls by its parameter names."""

        config = types.SimpleNamespace(
            kernel_type=types.SimpleNamespace(name="IntraNode")
        )

        def __init__(self, args):
            self.capacity = args["max_num_tokens_per_dp_rank"]
            width = {"num_ep_ranks": args["num_ep_ranks"]}
            self.dispatch = _declared(
                recorder, f"{MORI}.dispatch", self._dispatch, stated=width
            )
            self.combine = _declared(
                recorder, f"{MORI}.combine", self._combine, stated=width
            )

        def _dispatch(self, input, weights, scales, indices, **kwargs):
            rows, k = input.shape[0], indices.shape[1]
            n = sum(recorder.counts.get(rows, [rows] * dp)) * group.world_size // dp
            n = recorder.rows(n)
            return (
                input.new_empty((n, input.shape[1])),
                weights.new_empty((n, k)),
                None if scales is None else scales.new_empty((n, scales.shape[1])),
                indices.new_empty((n, k)),
                indices.new_empty((1,)),
            )

        def _combine(self, input, weights, indices, **kwargs):
            return input.new_empty((self.capacity, input.shape[-1])), None

    manager = types.SimpleNamespace(
        rank=group.rank_in_group,
        world_size=group.world_size,
        internode=False,
        get_handle=lambda args, index=0: Handle(args),
    )
    group.device_communicator = types.SimpleNamespace(
        all2all_manager=manager, destroy=lambda: None
    )
