"""Per-operator timing, and the arithmetic that decides whether it is usable.

Measured on hardware, in-line CUDA events around each dispatched operator
inflate small kernels by an order of magnitude: 113 gemms read 15.368ms against
a whole replayed step of 3.946ms. The tracer is kept because the *question* it
answers is the right one and the answer has to be reproducible; what it rules
out is summing its numbers into a step cost.
"""

import pytest

from atom.compass.runtime.op_timing import OpTiming, OpTimingTracer


def _tracer(region, op_seconds):
    """A tracer with results already in place, as if a forward had run."""
    t = OpTimingTracer()
    t.region_seconds = region
    t.timings = [OpTiming(i, f"op{i}", s) for i, s in enumerate(op_seconds)]
    return t


class TestCoverage:
    def test_operators_accounting_for_the_region(self):
        s = _tracer(1.0, [0.5, 0.5]).summary()
        assert s["covered"] == pytest.approx(1.0)
        assert s["operators"] == 2
        assert s["sum_of_operators"] == pytest.approx(1.0)

    def test_time_no_operator_claimed(self):
        """The measured case: a third of the region belongs to nobody."""
        s = _tracer(1.0, [0.33, 0.33]).summary()
        assert s["covered"] == pytest.approx(0.66)

    def test_operators_overlapping_exceed_the_region(self):
        """Concurrent kernels are each credited with time the others used."""
        s = _tracer(1.0, [0.8, 0.8]).summary()
        assert s["covered"] == pytest.approx(1.6)

    def test_an_empty_region_does_not_divide_by_zero(self):
        s = _tracer(0.0, []).summary()
        assert s["covered"] != s["covered"], "expected nan, not a crash"


class TestTheRecord:
    def test_a_timing_keeps_its_position(self):
        """Position is how a timing is joined back to the graph's operator."""
        t = OpTiming(7, "aiter::gemm_a16w16", 0.000121)
        assert t.as_dict() == {"index": 7, "name": "aiter::gemm_a16w16",
                               "seconds": 0.000121}

    def test_without_a_device_nothing_is_timed(self):
        """No CUDA, no region: the tracer must not pretend to have measured."""
        t = OpTimingTracer()
        assert t.region_seconds == 0.0
        assert t.timings == []
