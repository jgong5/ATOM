# SPDX-License-Identifier: MIT
"""A forward's host segments (`step_labels._segments`) when its MTP status wait
comes after its launch, as MiniMax-M3.1's DSpark prefill forward stamps it:
`W_pre` runs to the launch less the staging gate's wait, and `W_inputs` is 0;
the launch is never counted twice. A wait before the launch keeps its split."""

from atom.compass.backends.step_labels import _segments

MS = 1_000_000


def line(mtp):
    return {
        "t_enter_ns": 0,
        # the staging gate waits on the previous forward
        "t_gate_ns": (1 * MS, 300 * MS),
        "t_launch_ns": (301 * MS, 341 * MS),
        "t_mtp_ns": mtp,
        "t_recv_ns": (343 * MS, 343 * MS),
        "t_exit_ns": 345 * MS,
    }


def test_a_status_wait_after_the_launch_leaves_w_pre_to_the_launch():
    host = _segments(line((341 * MS, 342 * MS)), True)

    # before the gate, and from the gate's end to the launch
    assert host["W_pre"] == 1 * MS + 1 * MS
    assert host["W_inputs"] == 0
    assert host["W_launch"] == 42 * MS


def test_a_status_wait_before_the_launch_keeps_w_pre_and_w_inputs():
    host = _segments(line((300 * MS + 200_000, 300 * MS + 700_000)), True)

    assert host["W_pre"] == 1 * MS + 200_000
    assert host["W_inputs"] == 301 * MS - (300 * MS + 700_000)
    assert host["W_launch"] == 42 * MS
