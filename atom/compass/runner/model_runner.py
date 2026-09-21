# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""The runner class that `runner_qualname` names."""

from __future__ import annotations

from atom.compass.runner.overrides import (
    RPC_SURFACE,
    NonAllocatingRunner,
    RunnerRefusal,
    unanswered_rpc_names,
)
from atom.model_engine.model_runner import ModelRunner


class CompassModelRunner(NonAllocatingRunner, ModelRunner):
    """A `ModelRunner` that constructs without weights, KV tensors or a step.

    Everything the engine does around a step -- the scheduler, admission, the
    block manager, the prefix index -- is ATOM's own and runs unchanged. What
    this class removes is the weights, the KV tensors, the graph capture and the
    step itself, by overriding the six methods that own them; see `overrides`,
    which holds the bodies and says why each one does what it does.

    Construction is not free of device memory. What remains is the base's
    forward-vars ring from `allocate_forward_vars`, whose dominant term is a
    `max_num_batched_tokens` by `hidden_size` output buffer. It is sized by the
    batch budget and the model's hidden size, not by the model's weights, and
    no named tensor on the runner holds any of it.

    `NonAllocatingRunner` comes first so its methods win over the base's. There
    is deliberately no `__init__`: the base runs all of its own before a
    subclass body would get control, so there is no point at which state set
    here would be visible to it.
    """


_UNANSWERED = unanswered_rpc_names(CompassModelRunner)
if _UNANSWERED:
    # Checked here rather than left to a deployment. The worker resolves each
    # RPC with `getattr(runner, name, None)` and skips what comes back None, so
    # a name this class stops answering -- because ATOM renamed or dropped it --
    # produces no error anywhere: the worker stays healthy and nothing logs.
    # Failing the import turns that into a worker that dies at construction,
    # with a traceback.
    #
    # The two halves below fail differently, which is why the message
    # partitions them instead of asserting one story for all twelve. A waited
    # name parks its caller on an unbounded queue read for the life of the
    # process. An unwaited one parks nobody: `busy_loop` skips it and carries
    # on -- which for `exit` means the loop never breaks, and for
    # `process_kvconnector_output` means a KV load is silently never started.
    # Both are real failures; neither is a park.
    #
    # Two things about this raise itself. No CPU test tier can execute it:
    # importing this module imports `ModelRunner`, which runs aiter's
    # architecture probe and needs a driver, so a green CPU gate is not
    # evidence that the composed class answers the surface -- only an import on
    # a machine with a GPU is. And when it does fire inside a worker,
    # `AsyncIOProc.__init__` resolves the runner class (`async_proc.py:166`)
    # before assigning `self.runners = []` (`:167`), so the atexit finalizer
    # then fails on the half-built object and the worker log *ends* with
    # `AttributeError: 'AsyncIOProc' object has no attribute 'runners'`. The
    # refusal is the traceback above that one.
    _WAITED = [name for name in _UNANSWERED if RPC_SURFACE[name]]
    _UNREAD = [name for name in _UNANSWERED if not RPC_SURFACE[name]]
    raise RunnerRefusal(
        "the worker dispatches "
        + ", ".join(_UNANSWERED)
        + " by name and this runner answers none of them. Waited on, so a hole "
        "parks its caller for the life of the process: "
        + (", ".join(_WAITED) or "none")
        + ". Dispatched with no reader, so a hole is skipped and the worker "
        "carries on without it: " + (", ".join(_UNREAD) or "none") + "."
    )
