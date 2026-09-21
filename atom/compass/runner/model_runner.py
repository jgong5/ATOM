# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""The runner class that `runner_qualname` names."""

from __future__ import annotations

from atom.compass.runner.overrides import NonAllocatingRunner
from atom.model_engine.model_runner import ModelRunner


class CompassModelRunner(NonAllocatingRunner, ModelRunner):
    """A `ModelRunner` that constructs without weights, KV tensors or a step.

    Everything the engine does around a step -- the scheduler, admission, the
    block manager, the prefix index -- is ATOM's own and runs unchanged. What
    this class removes is the weights, the KV tensors and the step itself, by
    overriding the five methods that own them; see `overrides`, which holds the
    bodies and says why each one does what it does.

    Construction is not free of device memory, and what remains is the base's
    forward-vars ring. Measured at TP1 on Qwen3-0.6B with `enforce_eager`, the
    whole of `__init__` leaves 2,168,320 bytes allocated at a 1024-token,
    4-sequence budget and 17,668,096 at 8192 / 256; `allocate_forward_vars`
    accounts for 99.1% and 99.4% of those, and named tensors on the runner for
    0 bytes both times. That remainder is sized by the batch budget and not by
    the model: 16.9 MiB against the 1.40 GiB of weights the base makes resident
    for Qwen3-0.6B, and against 51.7 GiB for Qwen3.8-27B.

    `NonAllocatingRunner` comes first so its methods win over the base's. There
    is deliberately no `__init__`: the base runs all of its own before a
    subclass body would get control, so there is no point at which state set
    here would be visible to it.
    """
