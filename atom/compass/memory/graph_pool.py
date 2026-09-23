# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Two graph-pool numbers that are not the same number, kept apart on purpose.

This module keeps two graph-pool functions apart on purpose:

- **`reserves()` mirrors ATOM's own `_estimate_cudagraph_overhead`**
  (`model_runner.py:1546-1641`). It is not the better number and it is not meant
  to be. It is the number that *actually reserves the memory*, because it is
  what `get_num_blocks` subtracts from the budget, so substituting anything else
  there would predict a block count ATOM would never produce.
- **`predicts()` is what the pool really costs**, from the measured form in the
  spec: `91.1 MiB + 0.3033 MiB per captured token` at width 1, and flat above it.

**They disagree by 4-19x.** ATOM's is `0.2 x peak activations` of the *warmup*
shape and is blind to the capture ladder, while the measured pool moves over 4x
across ladders. Under-reserving is not an OOM: the capture loop re-checks free
memory per bucket and silently skips what will not fit, so the price is dropped
buckets and a decode cliff at those batch sizes -- which on a 192 GB card never
happened, which is why it went unnoticed. So the gap is a real defect with a
quiet failure mode, and collapsing the two functions into one would hide it
either way round: take the measured number and the predicted block count stops
being ATOM's; take ATOM's and the pool cost is wrong by up to 19x.

They are told apart by name rather than by a convention. `reserves()` names its
reading `cudagraph_overhead` and `predicts()` names its own `cudagraph_pool`,
and `readings.device_readings` refuses any reading but the first -- so "which
one reserves" is answered at the call site by the code, not by a comment that
can go stale.

One thing the width scaling rests on, recorded where the number is used: flat
above width 1 is well supported as a *shape* -- the allocated delta was
byte-identical, 79,692,800, across three widths and three ladders -- but that
the step happens at width 2 rather than the function merely looking flat over
the widths measured rests on one point above width 1.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from atom.compass.memory.terms import Basis, Reading, Term

#: The name `device_readings` accepts. See the module docstring.
RESERVES = "cudagraph_overhead"
#: The name it refuses, because this one does not reserve anything.
PREDICTS = "cudagraph_pool"

#: ATOM's declared live-tensors-per-layer coefficient (`model_runner.py:3601`).
#: Mirrored rather than re-derived: it is what the engine spends.
LIVE_TENSORS_PER_LAYER = 2.8
#: ATOM's whole-graph estimate, as a fraction of peak activations (`:1622`).
ACTIVATION_FRACTION = 0.2
#: The fraction of the utilisation budget the piecewise branch will reserve
#: before it stops taking buckets (`:1591`).
TARGET_RESERVE_FRACTION = 0.15


def piecewise_per_token_bytes(
    *, hidden_size: int, layers: int, dtype_bytes: int, dp_size: int = 1
) -> float:
    """ATOM's per-token retained estimate, from model geometry (`:3589-3612`).

    Mirrored including the sub-linear `dp ** 0.6` amplification, which is there
    because the MoE all-gathers hidden to about `dp_size` times the local tokens
    while attention does not amplify at all.
    """
    per_token = hidden_size * dtype_bytes * layers * LIVE_TENSORS_PER_LAYER
    return per_token * (float(dp_size) ** 0.6 if dp_size > 1 else 1.0)


def capture_token_shapes(
    capture_sizes: Sequence[int],
    *,
    q_buckets: Sequence[int] = (1,),
    max_num_batched_tokens: int,
) -> tuple[int, ...]:
    """The captured `num_tokens` shapes, which are `bs * q` and not `bs`.

    The capture loop uses a full q length for any speculative drafter, so a
    ladder read as `bs` alone under-counts the captured tokens by a factor of q
    -- a factor of 4 on plain MTP, which ATOM's own comment records as an 8.5 GB
    estimate against a 33 GB pool. A bucket over the token budget is not
    schedulable and is not captured, so it is dropped here too.
    """
    shapes = {bs * q for bs in capture_sizes for q in q_buckets}
    return tuple(sorted(s for s in shapes if s <= max_num_batched_tokens))


@dataclass(frozen=True, slots=True)
class PiecewiseCapture:
    """What the piecewise branch reserves for, and the buckets it gives up on."""

    per_token_bytes: float
    token_shapes: tuple[int, ...]
    budget_bytes: int

    def taken(self) -> tuple[tuple[int, ...], int]:
        """The buckets the capture loop keeps, and their token sum (`:1605-1611`).

        Greedy in ascending order and the first bucket is always taken, exactly
        as ATOM does it -- the cap is on how much of the budget the reservation
        may claim, not on how large one bucket may be.
        """
        target = TARGET_RESERVE_FRACTION * self.budget_bytes
        taken: list[int] = []
        total = 0
        for num_tokens in sorted(self.token_shapes):
            if taken and self.per_token_bytes * (total + num_tokens) > target:
                break
            taken.append(num_tokens)
            total += num_tokens
        return tuple(taken), total


def reserves(
    *,
    enforce_eager: bool = False,
    activation_bytes: int | None = None,
    piecewise: PiecewiseCapture | None = None,
) -> Reading:
    """The number that reserves: ATOM's own estimator, with its readings given.

    This is `_estimate_cudagraph_overhead` with the two `torch.cuda` calls
    replaced. The caller states which branch applies rather than this module
    inferring it from an engine it is not allowed to import.

    **Two of ATOM's own adjustments inside those branches are not mirrored
    here, and they pull in opposite directions.** Neither fires at M1 -- no
    drafter, one data-parallel rank -- and both are named because the cut that
    wires this into `get_num_blocks` under a block-count gate would otherwise
    read "nothing else" as fidelity:

    - A DSpark confidence-schedule drafter rescales the whole-graph branch by
      the captured bucket count, `activation_bytes * 0.2 * n_buckets`
      (`model_runner.py:1626-1632`). Without it this **under-reserves** by that
      factor and so predicts more KV blocks than ATOM would.
    - The piecewise branch drops buckets over `ATOM_PIECEWISE_DP_MAX_TOKENS`
      when `dp_size > 1` and a drafter is attached (`:1602-1604`).
      `capture_token_shapes` does not, so that configuration **over-reserves**
      and predicts fewer.
    """
    if enforce_eager:
        return Reading(
            RESERVES,
            (
                Term(
                    "no capture",
                    0,
                    Basis.DEPLOYMENT,
                    "config.enforce_eager",
                    "ATOM captures no graph under enforce_eager and reserves "
                    "nothing for one (model_runner.py:1556)",
                ),
            ),
        )
    if (activation_bytes is None) == (piecewise is None):
        raise ValueError(
            "state one branch: `activation_bytes` for the whole-graph estimate "
            "or `piecewise` for the per-token one. ATOM picks between them with "
            "`_piecewise_cg_active()`, which reads an engine this module does "
            "not import, so the caller that has the engine says which"
        )
    if piecewise is None:
        assert activation_bytes is not None
        return Reading(
            RESERVES,
            (
                Term(
                    "0.2 x peak activations",
                    int(activation_bytes * ACTIVATION_FRACTION),
                    Basis.DECLARED,
                    f"{ACTIVATION_FRACTION} x {activation_bytes} activation bytes "
                    "of the warmup shape (model_runner.py:1622)",
                    "ATOM's coefficient, mirrored because it is what reserves; "
                    "`predicts()` is the measured pool and disagrees by 4-19x",
                ),
            ),
        )
    taken, tokens = piecewise.taken()
    return Reading(
        RESERVES,
        (
            Term(
                "per-token x captured tokens",
                int(piecewise.per_token_bytes * tokens),
                Basis.DECLARED,
                f"{piecewise.per_token_bytes / (1 << 20):.3f} MiB/token x "
                f"{tokens} tokens over {len(taken)}/{len(piecewise.token_shapes)} "
                f"buckets, capped at {TARGET_RESERVE_FRACTION} x budget "
                "(model_runner.py:1565-1612)",
                "ATOM's geometry-derived coefficient, mirrored because it is "
                "what reserves; `predicts()` is the measured pool",
            ),
        ),
    )


def predicts(spec, *, tp_width: int, captured_tokens: int) -> Reading:
    """What the pool really costs. This reserves nothing; `reserves()` does.

    Below width 2 the measured form is a line in the captured tokens and both
    of its coefficients are spec fields. At and above width 2 it is flat, and
    the captured-token count does not enter -- which is the measured result and
    not an omission, so the argument is still taken and the source says that it
    did not enter rather than leaving a reader to infer it from an absence.
    """
    if captured_tokens < 0:
        raise ValueError(f"a ladder captures no negative tokens: {captured_tokens}")
    if tp_width < 1:
        raise ValueError(f"tensor-parallel width is at least 1: {tp_width}")
    field = "cudagraph_pool"
    if tp_width > 1:
        return Reading(
            PREDICTS,
            (
                Term(
                    "flat above width 1",
                    int(spec.runtime_constant(f"{field}.w_gt1_flat_bytes")),
                    Basis.SPEC,
                    f"device.runtime_constants.{field}.w_gt1_flat_bytes "
                    f"(flat; {captured_tokens} captured tokens do not enter)",
                    "flat is well supported as a shape -- the allocated delta "
                    "was byte-identical across three widths and three ladders "
                    "-- but that the step is at width 2 rests on one point "
                    "above width 1",
                ),
            ),
        )
    per_token = spec.runtime_constant(f"{field}.w1_bytes_per_captured_token")
    return Reading(
        PREDICTS,
        (
            Term(
                "base",
                int(spec.runtime_constant(f"{field}.w1_base_bytes")),
                Basis.SPEC,
                f"device.runtime_constants.{field}.w1_base_bytes",
            ),
            Term(
                "per captured token",
                int(per_token * captured_tokens),
                Basis.SPEC,
                f"device.runtime_constants.{field}.w1_bytes_per_captured_token"
                f" x {captured_tokens} captured tokens",
            ),
        ),
    )
