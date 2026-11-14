# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""The records a step linear law is fitted from, planned from a configuration
alone, as `scripts/compass/calibrate.sh` takes them: each its RECORD argument.

A law has a decode vector per capture rung (`step_linear.py`), and a rung N
comes from a record at N sessions that starts every trajectory at its first
turn: those sessions decode together at the start, where a record of sampled
starts batches far fewer of them
([#1840](https://github.com/jgong5/ATOM/issues/1840)). A data-parallel group
runs every rank at the rung of its widest rank's batch (`ForwardMode.decide`),
so the record takes N sessions a rank, N times the group's width. The rungs
are the capture ladder's, then the eager rung where the scheduler's bound
passes the widest captured size, taken at the bound.

The prefill vector fits from trace runs alone, and its held-out band needs two
(`step_linear_fit.fit`): one at concurrency 16 and one at 4, as the quickstart
takes them, where `compass record --trace` takes one; where it refuses, none
is planned and the law's prefill stays unfitted. The plan reads no law: a law
short of a rung is fitted again on the whole list, since a published law takes
no record.
"""

from atom.compass.backends.step_linear import EAGER

#: aiperf's arguments that start every trajectory at its first turn.
FIRST_TURN = "--trajectory-start-min-ratio 0 --trajectory-start-max-ratio 0"
#: The trace runs, each a `compass record --trace` of its concurrency's record.
TRACE_RUNS = ("trace 16", "trace 4")


def scheduled_rungs(ladder: list[int], bound: int) -> list:
    """Each decode rung a configuration schedules whose scheduler's widest
    decode batch is `bound`, and that captures `ladder`, ascending, each at
    most `bound`: each of `ladder`, then `EAGER` where `bound` passes the
    widest, as on an eager runner, which captures none."""
    return [*ladder, *([EAGER] if bound > max(ladder, default=0) else [])]


def records(ladder: list[int], bound: int, width: int, traced: bool) -> list[str]:
    """`calibrate.sh`'s RECORD arguments for a law with a vector at every rung
    `scheduled_rungs` gives: for each, ascending, a first-turn record of its
    batch, `bound` at the eager rung, at each of `width` data-parallel ranks;
    then, when `traced`, the trace runs."""
    rows = [bound if r == EAGER else r for r in scheduled_rungs(ladder, bound)]
    first_turn = [f"{n * width} {FIRST_TURN}" for n in rows]
    return first_turn + (list(TRACE_RUNS) if traced else [])
