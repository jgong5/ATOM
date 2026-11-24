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

aiter's comm-fused MoE, which ATOM takes where aiter holds winners for the
card and the layer (`driverless.NAMES`), reduces across its group inside its
own stage 2: one `declared` op, `COMM_FUSED` (`comm_fused`). Under DP its
input rows are gathered first by the DP group's `all_gatherv`, the call
`atom.model_ops.moe:all_gatherv` makes, declared as that one.

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
#: aiter's comm-fused MoE call: both expert stages and the group's reduction.
COMM_FUSED = "aiter.ops.comm_fused_moe_runtime:CommFusedMoeRuntime.run"


def split(rows: int, local: int, dp: int) -> list[int]:
    """Each rank's rows of the `rows` a DP group of `dp` gathers, this rank's
    `local` first: the peers split the rest evenly, the first ones a row more."""
    each, extra = divmod(rows - local, dp - 1)
    return [local] + [each + (k < extra) for k in range(dp - 1)]


def slots(region) -> list[list]:
    """Where `region`, a graph's JSON, holds G, as JSON: the path, keyed as
    the JSON is, to each `SYMBOL`, then ``G``, and to the `sizes` of each
    exchange of `SIZED` that holds G, then ``sizes``."""
    found = []

    def walk(node, path) -> bool:
        if node == SYMBOL:
            found.append([*path, _G])
            return True
        items = node.items() if isinstance(node, dict) else ()
        items = enumerate(node) if isinstance(node, list) else items
        # Every item walked: each may hold a slot of its own.
        held = sum(walk(value, (*path, key)) for key, value in items) > 0
        op = isinstance(node, dict) and node.get("op") in SIZED
        if held and op and "sizes" in node["args"]:
            found.append([*path, "args", "sizes", _SIZES])
        return held

    walk(region, ())
    return found


def at(region, held: list, g: int, dp: int, restated: bool = True):
    """`region`, a graph's JSON holding G at the `slots` `held`, at G = `g`:
    each G `g`, and unless `restated` is false each exchange's `sizes` this
    rank's then `split`'s over the `dp` ranks. What changes is copied, the
    rest shared."""
    trie = {}
    for *path, what in held:
        node = trie
        for key in path[:-1]:
            node = node.setdefault(key, {})
        node[path[-1]] = what
    return _put(region, trie, g, dp, restated)


def _put(region, trie: dict, g: int, dp: int, restated: bool):
    node = region.copy()
    for key, what in trie.items():
        if what == _G:
            node[key] = g
        elif what == _SIZES:
            node[key] = split(g, node[key][0], dp) if restated else node[key]
        else:
            node[key] = _put(node[key], what, g, dp, restated)
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
    from atom.model_ops.fused_moe.comm_fused_moe import CommFusedMoeBackend
    from atom.utils.forward_context import DPMetadata
    from atom.utils.tbo import ubatching

    sync = ubatching.sync_dp_metadata
    dp = get_dp_group().world_size
    gather_inputs = CommFusedMoeBackend._gather_dpa_inputs

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

    def gather_dpa_inputs(backend, rows):
        # The comm-fused MoE calls the DP group's communicator itself, with
        # the arguments `moe.all_gatherv` passes it; the capture's group has none.
        group = get_dp_group()
        held = group.device_communicator
        group.device_communicator = types.SimpleNamespace(
            all_gatherv=lambda x, dim, sizes: moe.all_gatherv(x, sizes, group)
        )
        try:
            return gather_inputs(backend, rows)
        finally:
            group.device_communicator = held

    gathers = (all_gatherv, reduce_scatterv, reduce_scatter_with_unpadding)
    return [
        (ubatching, "sync_dp_metadata", sync_dp_metadata),
        (DPMetadata, "gather_running_tokens", staticmethod(gather_running_tokens)),
        (CommFusedMoeBackend, "_gather_dpa_inputs", gather_dpa_inputs),
        *(
            (moe, f.__name__, _declared(recorder, f"{moe.__name__}:{f.__name__}", f))
            for f in gathers
        ),
    ]


def comm_fused(recorder):
    """`(owner, name, value)` for each patch that stands in for aiter's
    comm-fused MoE, which ATOM builds where aiter holds winners for its
    layer on the declared card.

    Its runners are the winners' configs alone, with no symmetric workspace
    or communicator built, each taking ragged rows where aiter's runner of
    its family does. A run is one `declared` op, `COMM_FUSED`: ATOM's
    arguments, aiter's enums among them by value as `fused_moe_` takes them,
    less the shared expert's producer, its join and their stream, plus the
    shape aiter built the runners for, `tp` ranks reducing by `comm`. It
    gives back the rows aiter's run does, of a ragged reduce-scatter this
    rank's, the first. The producer and the join run after it, as its stage 2
    calls them, so the shared expert is traced on its own stream. ATOM's
    custom op runs its body, as the other MoE wrappers do. No patch where
    aiter has no comm-fused MoE, as ATOM then builds none.
    """
    from aiter import ActivationType, QuantType

    from atom.model_ops.fused_moe import comm_fused_moe

    if (modules := comm_fused_moe._load_backend()) is None:
        return []
    host, runtime = modules

    class Runners(host._LazyRunners):
        def __init__(self, tp_group, shape, configs):
            self.tp_group, self.shape, self.configs = tp_group, shape, configs

        def __getitem__(self, tokens):
            runner = types.SimpleNamespace(config=self.configs[tokens])
            family = host._RUNNER_TYPES[type(runner.config)]
            runner.supports_ragged_m = family.supports_ragged_m.fget(runner)
            return runner

    def run(
        self,
        *,
        shared_partial,
        before_stage2_for_rows=None,
        before_shared_add=None,
        stage2_stream=None,
        reduce_scatter_sizes=None,
        **moe_args,
    ):
        hidden, sizes = moe_args["hidden_states"], reduce_scatter_sizes
        bucket = self.bucket_for(max(sizes) * len(sizes) if sizes else len(hidden))
        if sizes:
            rows = shared_rows = sizes[0]
        else:
            rows = self.runners.output_rows_for(bucket, len(hidden))
            shared_rows = self.runners.configs[bucket].output_rows
        with recorder.quiet():
            out = hidden.new_empty((rows, *hidden.shape[1:]))
        if recorder.on:
            shape = self.runners.shape
            # aiter's enums, which pickle cannot reduce, as `fused_moe_` takes them
            enums = (ActivationType, QuantType)
            slots = {
                k: v.value if isinstance(v, enums) else v for k, v in moe_args.items()
            }
            slots |= {"shared_partial": shared_partial, "tp": shape.tp}
            slots |= {"comm": shape.comm, "reduce_scatter_sizes": sizes}
            recorder.add(COMM_FUSED, ir.NodeKind.DECLARED, slots, [out], None)
        if before_stage2_for_rows is not None:
            before_stage2_for_rows(shared_rows)
        if before_shared_add is not None:
            before_shared_add()
        return out

    return [
        (host, "_LazyRunners", Runners),
        (runtime.CommFusedMoeRuntime, "run", run),
        (
            torch.ops.aiter,
            "comm_fused_moe_forward",
            comm_fused_moe.comm_fused_moe_forward,
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
