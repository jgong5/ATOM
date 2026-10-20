# SPDX-License-Identifier: MIT
"""`measure ops` on every rank of a parallel layout.

A layout runs D data-parallel ranks of T tensor-parallel ranks each, DP
attention's T ranks rewritten as T data-parallel ranks of one, as `CoreManager`
does (`layout`). Each rank is a process of its own on its own GPU (`launch`),
whose ATOM runner builds the groups through ATOM's own initialisation. The
ranks measure the union of their points in one order (`union`), agree on the
calls a graph holds (`agreed`) and take a barrier before each replay, so a
collective leaf runs on every rank at once. A point's record is rank 0's, with
each rank's cold time beside it (`measure`).
"""

import copy
import multiprocessing
from concurrent.futures import ProcessPoolExecutor


def layout(config) -> int:
    """`config` rewritten as `CoreManager` does under DP attention, every TP
    rank a DP rank of TP 1, and the ranks it runs."""
    if config.pipeline_parallel_size > 1 or config.prefill_context_parallel_size > 1:
        raise ValueError(
            "measure ops starts TP x DP ranks; pipeline and context parallelism "
            "are not laid out"
        )
    pc = config.parallel_config
    if config.enable_dp_attention:
        pc.data_parallel_size *= config.tensor_parallel_size
        pc.data_parallel_size_local = pc.data_parallel_size
        config.tensor_parallel_size = 1
    return pc.data_parallel_size * config.tensor_parallel_size


def launch(fn, config, *args) -> list:
    """`fn(rank_config, *args, rank=r)` on every rank of `config`'s layout, as
    `CoreManager` configures each DP rank, in spawned processes that meet at a
    store this process holds; what each returned, by rank."""
    from torch.distributed import TCPStore

    config = copy.deepcopy(config)
    pc = config.parallel_config
    store = TCPStore(
        pc.data_parallel_master_ip,
        pc.data_parallel_base_port,
        is_master=True,
        wait_for_workers=False,
    )
    pc.data_parallel_base_port, pc._managed_distributed_store = store.port, True
    calls = []
    for dp in range(pc.data_parallel_size):
        rank_config = copy.deepcopy(config)
        rank_config.parallel_config.data_parallel_rank = dp
        rank_config.parallel_config.data_parallel_rank_local = dp
        calls += [(rank_config, tp) for tp in range(config.tensor_parallel_size)]
    spawn = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(len(calls), mp_context=spawn) as pool:
        running = [pool.submit(fn, c, *args, rank=tp) for c, tp in calls]
        return [r.result() for r in running]


def measure(config, todo, llc: int, gpus: int) -> list[dict]:
    """`ops.measure` of `todo` on `gpus` GPUs, a rank of `config`'s layout on
    each: one in a child of its own; else rank 0's records, each with every
    rank's cold time, refused where any rank refused."""
    from atom.compass.bench import ops

    if gpus == 1:
        return ops._child(ops.measure, config, todo, llc)
    by_rank = launch(ops.measure, config, todo, llc)
    got = []
    for records in zip(*by_rank):
        record = records[0] | {"rank_cold_ns": [r.get("cold_ns") for r in records]}
        refused = [r["refused"] for r in records if "refused" in r]
        got.append(record | ({"refused": refused[0]} if refused else {}))
    return got


def cpu_group():
    """The host group over every rank, or None for one."""
    import torch.distributed as dist

    if not dist.is_initialized() or dist.get_world_size() == 1:
        return None
    from aiter.dist.parallel_state import get_world_group

    return get_world_group().cpu_group


def union(todo: list, group) -> list:
    """Every rank's `todo`, `(shape, context, {leaf: (op, instances)})` each, as
    one: each point and each leaf in the order the lowest rank naming it gives,
    so every rank measures the same leaves in the same order."""
    import torch.distributed as dist

    if group is None:
        return todo
    every = [None] * dist.get_world_size(group)
    dist.all_gather_object(every, todo, group=group)
    points: dict = {}
    for rank_todo in every:
        for shape, context, found in rank_todo:
            held = points.setdefault((shape, context), {})
            for sig, leaf in found.items():
                held.setdefault(sig, leaf)
    return [(shape, context, found) for (shape, context), found in points.items()]


def agreed(n: int, group, most: bool = False) -> int:
    """The least of every rank's `n`, or the most."""
    import torch
    import torch.distributed as dist

    if group is None:
        return n
    value = torch.tensor([n])
    op = dist.ReduceOp.MAX if most else dist.ReduceOp.MIN
    dist.all_reduce(value, op=op, group=group)
    return int(value)
