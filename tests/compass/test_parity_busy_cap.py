# SPDX-License-Identifier: MIT
"""A trace run's busy cap, the median queued gap, leaves out every gap of one
timestamp rounding step: two kernels that abut read a gap of about 1 ns, and on
a real trace most queued gaps are one."""

import json

import pytest
from test_parity_busy import _trace

from atom.compass.parity import busy

#: One forward's kernels, (name, start, end, launch), in us, each launch call 1
#: us long. K2 to K4 abut the kernel before, queued: gaps of 1 ns. K5 and K6 are
#: queued 6 and 8 after the kernel before ends. K7's launch returns at 170, 100
#: after K6 ends, so its gap is capped.
KERNELS = [
    ("k1", 10, 20, 1),
    ("k2", 20.001, 30, 2),
    ("k3", 30.001, 40, 3),
    ("k4", 40.001, 50, 4),
    ("k5", 56, 60, 5),
    ("k6", 68, 70, 6),
    ("k7", 170, 180, 169),
]


def _reduce(tmp_path, kernels):
    """A one-rank trace run of one forward running `kernels`, reduced."""
    (tmp_path / "steps").mkdir()
    line = {"step": 0, "rows": [[100, 100, False]]}
    (tmp_path / "steps" / "dp0.jsonl").write_text(json.dumps(line) + "\n")
    _trace(tmp_path / "trace/rank_0/m_ts_1.pt.trace.json.gz", kernels, marks=())
    return json.loads(busy.reduce(tmp_path).read_text())


def test_the_cap_is_the_median_queued_gap_past_one_rounding_step(tmp_path):
    """The union, 55.997, and the queued gaps, 0.003 + 6 + 8, make 70; K7's gap
    adds 7, the median of 6 and 8, not 0.001, the median of all five."""
    assert _reduce(tmp_path, KERNELS) == {"step": 0, "b_target_ns": 77000}


def test_queued_gaps_of_one_rounding_step_alone_refuse(tmp_path):
    with pytest.raises(ValueError, match="queued at a gap past one rounding step"):
        _reduce(tmp_path, KERNELS[:4] + KERNELS[-1:])
