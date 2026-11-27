# SPDX-License-Identifier: MIT
"""COMPASS_FIT_RING_CUT (#3720): a pipeline head's W_launch fitted on the forwards the IPC ring did not hold."""

import pytest

from atom.compass.backends.step_labels import ring_gated
from atom.compass.backends.step_linear_fit import RING_CUT_ENV, ring_cut_s


def _line(step, last_ms, dummy=False):
    return {"step": step, "is_dummy_run": dummy, "t_launch_ns": [0, int(last_ms * 1e6)]}


def test_a_head_forward_launching_last_right_after_stage_1_freed_its_slot_is_ring_held():
    # Stage 1's last launch of forwards 0..5 at 100 ms apart; the head's last launches.
    after = [_line(s, 100 * s) for s in range(6)]
    head = [_line(0, 0), _line(1, 10), _line(2, 20), _line(3, 5), _line(4, 300), _line(5, 210), _line(6, 999, dummy=True)]
    # Forward 3 (slot back: 0, at 0 ms) at 5 ms: held; 4 (back: 1, 100 ms) at 300: free; 5 (back: 2, 200) at 210: held.
    assert ring_gated(head, after, 3, 0.020) == {3, 5}
    # A forward whose slot-back forward stage 1 did not record counts as held.
    assert ring_gated(head, after[:1], 3, 0.020) == {3, 4, 5}


def test_the_switch_is_milliseconds(monkeypatch):
    monkeypatch.delenv(RING_CUT_ENV, raising=False)
    assert ring_cut_s() is None
    monkeypatch.setenv(RING_CUT_ENV, "20")
    assert ring_cut_s() == pytest.approx(0.020)
    monkeypatch.setenv(RING_CUT_ENV, "twenty")
    with pytest.raises(ValueError, match="milliseconds"):
        ring_cut_s()


def test_the_jit_cut_takes_a_threshold_in_seconds(monkeypatch):
    from atom.compass.backends.step_linear_fit import JIT_OVER_ENV, jit_over_s

    monkeypatch.delenv(JIT_OVER_ENV, raising=False)
    assert jit_over_s() is None
    monkeypatch.setenv(JIT_OVER_ENV, "1.0")
    assert jit_over_s() == 1.0
    monkeypatch.setenv(JIT_OVER_ENV, "one")
    with pytest.raises(ValueError, match="seconds"):
        jit_over_s()
