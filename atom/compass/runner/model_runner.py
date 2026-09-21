# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""The runner class that `runner_qualname` names."""

from __future__ import annotations

from atom.compass.runner.overrides import (
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
    # produces no error anywhere: the worker stays healthy and the caller that
    # asked for the reply blocks until the process is killed. Failing the import
    # turns that into a worker that dies at construction, with a traceback.
    raise RunnerRefusal(
        "the worker dispatches "
        + ", ".join(_UNANSWERED)
        + " by name and this runner answers none of them; each one would park "
        "its caller rather than raise."
    )
