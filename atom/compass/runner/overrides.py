# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""The model-runner methods that keep construction off the device.

These are the memory-owning methods of ATOM's `ModelRunner`: the ones that read
a checkpoint, warm the model, size the KV pool, allocate it, and run a step.
They live here rather than beside the subclass because this module imports
nothing from the engine, and so can be executed on a machine with no driver.

Two of them run during construction (`_build_and_load_model`, `_maybe_warmup`)
and decline to do their work. The rest run afterwards, over the worker's RPC
channel; `allocate_kv_cache` does the arithmetic and none of the allocation,
and the other two refuse by name because their answers are not this module's to
invent -- a wrong duration or a wrong block count would be indistinguishable
from a measured one.

Order matters when mixing this in: `NonAllocatingRunner` must precede
`ModelRunner` in the bases so these definitions win. The class deliberately
defines no `__init__`. The base class runs the whole of its own `__init__`
before a subclass body would get control, so anything these methods read has to
come from `self.config`, which the base sets first.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


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

    def forward(self, batch: Any) -> Any:
        """Refuse to run a step.

        This class is the attachment point and not the replacement for a step. A
        step's output has to reproduce what the scheduler reads back from a real
        one, and a plausible-looking stub that does not is worse than no answer.
        """
        raise RunnerRefusal(
            "this runner has no cost model and no step semantics yet, so it "
            "cannot say what a step produced or how long it took."
        )
