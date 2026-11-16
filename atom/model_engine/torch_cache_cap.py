# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""A ceiling on the torch caching allocator once a runner's startup ends.

KV sizing budgets torch's allocation peak and leaves the rest of the card as
headroom, but nothing holds the cache to that afterwards: freed blocks stay
reserved, and steps of many different shapes can grow them until the card is
full. ROCr meets a device allocation it cannot back by taking every hardware
queue's scratch back and retrying, so the allocation succeeds and the next
kernel that spills has to find its scratch on a full card. A kernel that waits
at a grid-wide barrier for every block it launched, as MoRI's intranode
combine does, then either gets scratch for fewer waves than its grid and waits
for blocks that never start, or gets none and its queue aborts with
HSA_STATUS_ERROR_OUT_OF_RESOURCES.

The cap lets torch reserve what it holds when startup ends plus what is then
free, less a margin of the card. Past it torch frees its cached blocks and
retries, as it does at the card's ceiling, without asking the device for
memory first, so the margin stays free for scratch and for memory allocated
outside torch. A step whose live tensors do not fit under the cap fails with
torch's out-of-memory error. ATOM_TORCH_CACHE_CAP=0 leaves the allocator
uncapped.
"""

import logging
import time

import torch

from atom.utils import envs

logger = logging.getLogger("atom")

GIB = 1 << 30
#: Seconds between two reads of torch's retry count by `TorchCacheCap.poll`.
POLL_EVERY_S = 10.0


def torch_cache_limit(reserved: int, free: int, total: int, margin: float) -> int:
    """Bytes torch may reserve: the `reserved` it holds plus the `free` bytes,
    less `margin` of the card's `total`, and never less than it holds."""
    return max(reserved, reserved + free - int(total * margin))


class TorchCacheCap:
    """One device's cap, and the one line logged the first time torch frees
    its cached blocks under it.

    torch counts an allocation it retries after freeing every cached block
    (`num_alloc_retries`); under the cap that is how it stays below it, and it
    is also what it does when the card itself is full. `poll` reads the count
    at most every `every_s` seconds, so a forward between two reads costs one
    clock read."""

    def __init__(
        self,
        cuda,
        device,
        limit: int,
        label: str,
        every_s: float = POLL_EVERY_S,
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

    def poll(self) -> None:
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


def cap_torch_cache(device, margin: float, label: str, cuda=torch.cuda):
    """Cap torch's cache on `device` at what it holds plus what is free now,
    less `margin` of the card (`torch_cache_limit`), and return the
    `TorchCacheCap` to poll; None when ATOM_TORCH_CACHE_CAP is 0."""
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
