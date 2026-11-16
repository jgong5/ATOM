# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""A ceiling on the torch caching allocator once a runner's startup ends.

Nothing else bounds the cache after KV sizing, and at the card's ceiling ROCr
meets a refused allocation by taking back every queue's scratch; a spilling
grid-barrier kernel (MoRI's intranode combine) then waits for blocks that
never start, or aborts with HSA_STATUS_ERROR_OUT_OF_RESOURCES. Under the cap
torch frees its cached blocks and retries before it asks the device, so a
margin of the card stays free.
"""

import logging
import time

import torch

from atom.utils import envs

logger = logging.getLogger("atom")

GIB = 1 << 30
#: Share of the card the cap keeps outside torch: room for every hardware
#: queue's scratch and for memory allocated outside torch.
CARD_MARGIN = 0.02
#: Seconds between two reads of torch's retry count by
#: `TorchCacheCap.log_first_free`.
CHECK_EVERY_S = 10.0


def torch_cache_limit(reserved: int, free: int, total: int, margin: float) -> int:
    """Bytes torch may reserve: the `reserved` it holds plus the `free` bytes,
    less `margin` of the card's `total`, and never less than it holds."""
    return max(reserved, reserved + free - int(total * margin))


class TorchCacheCap:
    """One device's cap. `log_first_free` logs once, the first time torch
    frees its cached blocks to retry an allocation (`num_alloc_retries`),
    reading the count at most every `every_s` seconds."""

    def __init__(
        self,
        cuda,
        device,
        limit: int,
        label: str,
        every_s: float = CHECK_EVERY_S,
        clock=time.monotonic,
    ):
        self._cuda, self._device, self._label = cuda, device, label
        self.limit = limit
        self._every_s, self._clock = every_s, clock
        self._retries = self._read_retries()
        self._next = clock() + every_s
        self.freed = False

    def _read_retries(self) -> int:
        return self._cuda.memory_stats(self._device).get("num_alloc_retries", 0)

    def log_first_free(self) -> None:
        if self.freed:
            return
        now = self._clock()
        if now < self._next:
            return
        self._next = now + self._every_s
        retries = self._read_retries()
        if retries > self._retries:
            self.freed = True
            logger.info(
                "%s: torch freed its cached blocks to retry an allocation under "
                "the %.2f GiB torch cache cap (reserved now %.2f GiB, %d retries "
                "since the cap)",
                self._label,
                self.limit / GIB,
                self._cuda.memory_reserved(self._device) / GIB,
                retries - self._retries,
            )


def cap_torch_cache(device, label: str, margin: float = CARD_MARGIN, cuda=torch.cuda):
    """Cap torch's cache on `device` at what it holds plus what is free now,
    less `margin` of the card (`torch_cache_limit`), and return the
    `TorchCacheCap`; None when ATOM_TORCH_CACHE_CAP is 0."""
    if not envs.ATOM_TORCH_CACHE_CAP:
        logger.info("%s: torch cache uncapped (ATOM_TORCH_CACHE_CAP=0)", label)
        return None
    free, total = cuda.mem_get_info(device)
    reserved = cuda.memory_reserved(device)
    limit = torch_cache_limit(reserved, free, total, margin)
    # torch takes the cap as a fraction of the device's own total.
    card = cuda.get_device_properties(device).total_memory
    cuda.set_per_process_memory_fraction(min(limit / card, 1.0), device)
    logger.info(
        "%s on %s: torch cache capped at %.2f GiB (held %.2f GiB, %.2f GiB "
        "free, %.2f GiB of the card kept outside torch)",
        label,
        device,
        limit / GIB,
        reserved / GIB,
        free / GIB,
        (total - limit) / GIB,
    )
    return TorchCacheCap(cuda, device, limit, label)
