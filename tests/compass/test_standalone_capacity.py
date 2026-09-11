"""What the standalone harness says when the KV pool does not fit.

The pool is ``blocks x KV_VARIANTS``, and the variant count is a measurement
policy rather than anything the graph asks for. A bare allocator OOM reports
only the byte count, which sends a reader to the context length when the answer
is the rotation. These are about the message, and about the arithmetic behind
it being checkable without a device.

Device-free by construction: every function under test is pure.
"""

from __future__ import annotations

from atom.compass.runtime.standalone import (
    _demand_from_oom,
    capacity_refusal,
    paged_kv_bytes,
)

GIB = 1 << 30


def test_a_pool_that_fits_is_not_refused():
    assert capacity_refusal(demand=10 * GIB, free=190 * GIB, blocks=1000,
                            variants=64) is None


def test_a_pool_with_nothing_known_about_it_is_not_refused():
    """An unknown demand is not evidence of a problem, so it is not treated as one."""
    assert capacity_refusal(demand=0, free=190 * GIB, blocks=1000,
                            variants=64) is None
    assert capacity_refusal(demand=10 * GIB, free=0, blocks=1000,
                            variants=64) is None


def test_the_refusal_names_the_rotation_not_just_the_bytes():
    """The G1b case: 8,192 blocks x 64 variants = 512 GiB against 191 free."""
    why = capacity_refusal(demand=512 * GIB, free=191 * GIB,
                           blocks=8192 * 64, variants=64)
    assert why is not None
    assert "524,288 blocks" in why
    assert "64 KV variants" in why
    assert "512.00 GiB" in why
    assert "191.00 GiB free" in why
    # The size of one rotation copy is what says whether the context or the
    # policy is the problem: 8 GiB here, so the graph is not the difficulty.
    assert "8.00 GiB" in why


def test_the_refusal_does_not_offer_lowering_the_rotation_as_a_remedy():
    """Lowering it would answer with a number that is not a cold-call price.

    The message may explain what the knob is -- a reader needs that to
    understand the arithmetic -- and must not present turning it down as the
    fix, because how far the price moves when calls start re-reading a warmed
    copy is unmeasured.
    """
    why = capacity_refusal(demand=512 * GIB, free=191 * GIB,
                           blocks=8192 * 64, variants=64)
    assert "not measured" in why
    assert "not a cold-call price" in why
    for suggestion in ("try lowering", "set COMPASS_KV_VARIANTS=",
                       "reduce it to", "instead use"):
        assert suggestion not in why


def test_demand_is_read_back_from_the_allocator_rather_than_recomputed():
    """The allocator states what it asked for; a second estimate could disagree."""
    message = ("HIP out of memory. Tried to allocate 512.00 GiB. GPU 0 has a "
               "total capacity of 191.98 GiB of which 191.33 GiB is free.")
    assert _demand_from_oom(RuntimeError(message)) == 512 * GIB


def test_demand_handles_the_other_units_and_says_nothing_when_it_cannot_tell():
    assert _demand_from_oom(RuntimeError("Tried to allocate 1.50 MiB.")) == \
        int(1.5 * (1 << 20))
    assert _demand_from_oom(RuntimeError("something else went wrong")) == 0


def test_pool_bytes_is_blocks_times_a_block():
    assert paged_kv_bytes(1 << 20, 524_288) == 512 * GIB
    assert paged_kv_bytes(0, 100) == 0
    assert paged_kv_bytes(1 << 20, 0) == 0
