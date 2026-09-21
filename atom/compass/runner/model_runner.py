# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""The runner class that `runner_qualname` names."""

from __future__ import annotations

from atom.compass.runner.overrides import NonAllocatingRunner
from atom.model_engine.model_runner import ModelRunner


class CompassModelRunner(NonAllocatingRunner, ModelRunner):
    """A `ModelRunner` that constructs without owning any device memory.

    Everything the engine does around a step -- the scheduler, admission, the
    block manager, the prefix index -- is ATOM's own and runs unchanged. What
    this class removes is the weights, the KV tensors and the step itself, by
    overriding the five methods that own them; see `overrides`, which holds the
    bodies and says why each one does what it does.

    `NonAllocatingRunner` comes first so its methods win over the base's. There
    is deliberately no `__init__`: the base runs all of its own before a
    subclass body would get control, so there is no point at which state set
    here would be visible to it.
    """
