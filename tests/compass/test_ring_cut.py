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


def test_the_later_stages_launch_spans_are_read_ring_cut_and_stall_cut(tmp_path):
    import json as _json

    from atom.compass.backends.step_labels import digest
    from atom.compass.fit.host import record_stage_launches

    steps = tmp_path / "rec" / "steps"
    steps.mkdir(parents=True)

    def write(name, lasts):
        steps.joinpath(name).write_text(
            "".join(_json.dumps(_line(s, ms)) + "\n" for s, ms in enumerate(lasts))
        )

    write("dp0.jsonl", [0, 10, 20, 30, 40])
    write("pp1.dp0.jsonl", [0, 100, 200, 300, 2500])  # step 4's span 2.5 s: a compile
    write("pp2.dp0.jsonl", [0, 10, 20, 105, 120])  # step 3 at 105 ms: 5 ms past stage 3's step 0 (100): held
    write("pp3.dp0.jsonl", [100, 110, 120, 130, 140])
    got = record_stage_launches([tmp_path / "rec"], 0.020, 1.0)
    spans = got[digest(steps / "dp0.jsonl")]
    # Stage 1 (vs stage 2): step 3 at 300 is 300 ms past stage 2's step 0 (0): free; stage 2 step 3: held;
    # stage 3 uncut; stage 1's step 4 over 1 s: left out; stage 2's step 4 at 120, 10 ms past stage 3's step 1: held.
    assert sorted(spans[3]) == pytest.approx([0.130, 0.300])
    assert spans[4] == pytest.approx([0.140])
    assert len(spans[0]) == 3
