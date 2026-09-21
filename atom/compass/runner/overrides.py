# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""The model-runner methods that keep construction off the device.

These are the memory-owning methods of ATOM's `ModelRunner`: the ones that read
a checkpoint, warm the model, size the KV pool, allocate it, capture graphs, and
run a step. They live here rather than beside the subclass because this module
imports nothing from the engine, and so can be executed on a machine with no
driver.

Two of them run during construction (`_build_and_load_model`, `_maybe_warmup`)
and decline to do their work. The rest run afterwards, over the worker's RPC
channel; `allocate_kv_cache` does the arithmetic and none of the allocation,
`capture_cudagraph` captures nothing and says so in the shape its caller
unpacks, and the other two refuse by name because their answers are not this
module's to invent -- a wrong duration or a wrong block count would be
indistinguishable from a measured one.

Order matters when mixing this in: `NonAllocatingRunner` must precede
`ModelRunner` in the bases so these definitions win. The class deliberately
defines no `__init__`. The base class runs the whole of its own `__init__`
before a subclass body would get control, so anything these methods read has to
come from `self.config`, which the base sets first.

## The reply contract, and why it is not a matter of taste

A worker runs `AsyncIOProc.busy_loop`, which takes a name off a shared-memory
ring, resolves it with `getattr(runner, name, None)`, calls it, and forwards the
result **only when it is not None** (`async_proc.py:231-250`). The caller's side
is `AsyncIOProcManager.call_func`, whose `wait_out=True` form blocks on
`self.outputs_queue.get()` with no timeout. Three consequences, none of which a
single-process test can show:

* **A name the runner does not have is skipped, not raised.** `getattr` returns
  None, the loop moves on, the worker stays healthy, and a caller that asked for
  the reply waits for the life of the process. Absence is the quietest failure
  on this surface, which is why `RPC_SURFACE` below names every dispatched
  method rather than leaving it to whatever the class happens to inherit.
* **Returning None where the caller waits is the same failure.** The reply is
  never queued, so the shapes here are checked against what the call site
  unpacks, not against what reads well.
* **Raising is the loud option.** An exception leaves `busy_loop`, kills the
  worker, and the manager's process monitor turns that into a `SystemExit` on
  the output queue, which `call_func` re-raises in the caller. So a refusal
  reaches the caller as a traceback while a silence never reaches it at all.

`RPC_SURFACE` records, for each dispatched name, whether its caller waits for
the reply. The names themselves are not a list anyone typed: they are the ones
`engine_core`, `pp_engine_core` and `engine_utility` broadcast that a
`ModelRunner` answers, and the tests derive that intersection from ATOM's own
source and compare it with this table.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)

# Every name broadcast to a worker whose runner is a `ModelRunner`, and whether
# the caller blocks on the reply. True means a None return, or a missing method,
# parks that caller; False means a reply is not read and lands on the output
# queue for whoever asks next. The call site that fixes each shape is named
# beside it.
RPC_SURFACE: dict[str, bool] = {
    # Replaced here, in this module.
    "get_num_blocks": True,  # engine_core.py:132 reads four keys off a dict
    "allocate_kv_cache": True,  # engine_core.py:142-145 asserts the reply
    "capture_cudagraph": True,  # engine_core.py:149 unpacks three values
    "forward": True,  # engine_core.py:1264 hands it to Scheduler.postprocess
    # Answered by ATOM's own, which needs neither weights nor a device for them.
    "dummy_execution": True,  # engine_core.py:749, and it drives `forward`
    "exit": False,  # engine_core.py:260, the last call of the process's life
    "freeze_gc_heap": True,  # engine_core.py:204 waits and drops the count
    "process_kvconnector_output": False,  # engine_core.py:500 does not wait
    "async_proc_aggregation": True,  # engine_core.py:488 aggregates per worker
    "start_profiler": True,  # engine_utility.py:252 forwards the reply on
    "stop_profiler": True,  # engine_utility.py:264 logs and forwards a dict
    "flush_pp_send": True,  # pp_engine_core.py:81 waits before the next send
}


def unanswered_rpc_names(runner: Any) -> tuple[str, ...]:
    """The dispatched names `getattr` would answer with None for this runner.

    Each one is a caller that parks forever rather than an error anyone sees,
    so this is checked where the class is composed instead of being discovered
    by a deployment.
    """
    return tuple(name for name in RPC_SURFACE if getattr(runner, name, None) is None)


class RunnerRefusal(RuntimeError):
    """Raised where this runner has no answer and will not invent one."""


class UnbuiltModel(torch.nn.Module):
    """Stands in for the module tree a non-allocating runner never builds.

    It registers no parameter and no buffer, so it costs nothing on any device,
    and calling it raises instead of returning something a caller could mistake
    for a forward pass.
    """

    def __init__(self, model_class: Any) -> None:
        super().__init__()
        self.model_class_name = getattr(model_class, "__name__", repr(model_class))

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise RunnerRefusal(
            f"{self.model_class_name} was never built: this runner predicts what a "
            "forward pass costs instead of running one, so there are no weights "
            "to call."
        )


class NonAllocatingRunner:
    """Overrides that construct a model runner without touching device memory."""

    def _build_and_load_model(self, model_class: Any) -> None:
        """Build nothing and read no checkpoint.

        The base class constructs the model with the default device set to this
        rank's GPU and then fills it from disk. Both halves are skipped here, so
        no weight byte reaches the device and no checkpoint is read.
        """
        self.model = UnbuiltModel(model_class)
        # Cleared on the way out, as both of ATOM's own implementations do: the
        # caller set the default device to this rank's GPU before calling, and
        # the code that runs next is written against a cleared default.
        torch.set_default_device(None)
        logger.info(
            "%s not built and no checkpoint read; no weight bytes on the device.",
            self.model.model_class_name,
        )

    def _maybe_warmup(self) -> None:
        """Skip warmup.

        Warmup runs a forward over a dummy batch against real weights. There are
        none here, so there is nothing to warm. Skipping is also what lets
        construction finish at all: the base class warms the model from inside
        its own `__init__`, and the forward that warmup drives is this class's,
        which refuses.
        """
        return

    def get_num_blocks(self) -> dict[str, object]:
        """Refuse to size the KV pool.

        The base sizes it from what a real device reports free after the weights
        are resident. Neither figure exists here, and the substitute belongs to
        the memory model rather than to the runner, so answering would mean
        inventing a block count that the scheduler would then treat as measured.

        Whoever supplies that count answers a dict, not an integer:
        `engine_core.py:133-140` reads `num_kvcache_blocks` and `state_runtime`
        from it and takes `pool_entries` and `pool_entries_per_req` with a
        default, then hands the block count to the BlockManager.
        """
        raise RunnerRefusal(
            "a non-allocating runner cannot size the KV pool from a device it "
            "never allocated on; the block count has to come from a memory model "
            "this runner has not been given."
        )

    def allocate_kv_cache(self, num_kvcache_blocks: int) -> bool:
        """Record the block count and allocate nothing.

        The block accounting above this -- the pool, the prefix index, eviction,
        preemption -- is integer arithmetic and runs unmodified against the count
        recorded here. Only the tensors behind the blocks are absent.
        """
        self.config.num_kvcache_blocks = num_kvcache_blocks
        logger.info(
            "kv cache: %d blocks accounted, 0 bytes allocated.",
            num_kvcache_blocks,
        )
        return True

    def capture_cudagraph(self) -> tuple[float, list[int], int]:
        """Capture nothing, and say so in the three values the caller unpacks.

        The base reaches the model only after it has zeroed device buffers,
        opened a graph memory pool and entered a capture context, so inheriting
        it would put bytes on a device before it discovered there are no
        weights to trace. Nothing is captured here, and the reply says exactly
        that: no seconds spent, no batch sizes captured, no pool bytes held.

        The shape is fixed by `engine_core.py:149-151` and `1109-1111`, which
        both write `cap_cost, bs, pool_bytes = ...` and then format the first
        and third as numbers. Returning two values, or None, is not a test
        failure over there: the worker dies on the unpack, or never replies, and
        the caller waits for the rest of its life either way.

        `capture_sizes` and `capture_sizes_np` are deliberately left at the
        eager-fallback values the base set during construction, because the
        attention metadata builder reads them on every step.
        """
        logger.info("no cudagraph captured: there are no weights to trace.")
        return 0.0, [], 0

    def forward(self, batch: Any) -> Any:
        """Refuse to run a step.

        This class is the attachment point and not the replacement for a step. A
        step's output has to reproduce what the scheduler reads back from a real
        one, and a plausible-looking stub that does not is worse than no answer.

        What a replacement owes its callers: `engine_core.py:1264` passes the
        reply straight to `Scheduler.postprocess`, and the pipeline-parallel
        head at `pp_engine_core.py:379-385` reads `.req_ids` off it and puts it
        on the transport, so it is a `ScheduledBatchOutput` and not a tuple.
        """
        raise RunnerRefusal(
            "this runner has no cost model and no step semantics yet, so it "
            "cannot say what a step produced or how long it took."
        )
