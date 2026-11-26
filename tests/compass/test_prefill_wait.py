# SPDX-License-Identifier: MIT
"""A prefill forward waits for the previous forward's device work (opt-in
COMPASS_PREFILL_WAITS_PREVIOUS=1).

ATOM's host blocks until the previous forward's device work is done before a
prefill forward launches: a middle chunk past its staging gate, where the wait
lands in `W_pre`, and an MTP output prefill past its MTP status wait, where it
lands in `W_inputs` (behind a middle chunk, which leaves no MTP status to wait
for, the status wait holds none of it). On Qwen3.8-Flash-Next's and
GLM-5.3-Flash's step B records the host part after the wait was 0.3-3.7 ms
(p50) where the wait ran to 0.5 s. Under the switch the labels take the wait
out of the one segment it sat in, and the simulated host runs that segment's
rest after the previous forward's device end.
"""

import collections
import math

import pytest

from atom.compass.backends import step_labels
from atom.compass.backends.step_labels import (
    _segments,
    prefill_wait_ns,
    take_prefill_wait,
)
from atom.compass.runner.overrides import HostDeviceRecurrence

MS = 1_000_000


def _middle(enter, gate_end, launch, busy):
    return {
        "t_enter_ns": enter,
        "t_gate_ns": (enter, gate_end),
        "t_launch_ns": (launch, launch + 70 * MS),
        "d_busy_ns": busy,
    }


def _output(enter, mtp_end, launch, busy):
    return {
        "t_enter_ns": enter,
        "t_gate_ns": (enter, enter),
        "t_mtp_ns": (enter, mtp_end),
        "t_launch_ns": (launch, launch + 70 * MS),
        "t_recv_ns": (launch + 75 * MS, launch + 75 * MS),
        "t_exit_ns": launch + 80 * MS,
        "d_busy_ns": busy,
    }


def test_a_middle_chunk_waits_from_its_gate_end_to_the_previous_device_end():
    prev = _middle(0, 0, 10 * MS, 600 * MS)  # device until 610 ms
    s = _middle(80 * MS, 80 * MS, 611 * MS, 600 * MS)  # launched once it was done
    assert prefill_wait_ns(prev, s, True) == 530 * MS
    early = _middle(700 * MS, 700 * MS, 701 * MS, 50 * MS)  # previous work long done
    assert prefill_wait_ns(prev, early, True) == 0
    assert prefill_wait_ns(None, s, True) == 0


def test_an_output_prefill_waits_from_its_mtp_status_wait_to_the_previous_device_end():
    prev = _output(0, 0, 10 * MS, 300 * MS)  # device until 310 ms
    s = _output(80 * MS, 81 * MS, 313 * MS, 200 * MS)  # launched 3 ms after it
    assert prefill_wait_ns(prev, s, False) == 229 * MS
    held = _output(80 * MS, 311 * MS, 314 * MS, 200 * MS)  # its status wait held it
    assert prefill_wait_ns(prev, held, False) == 0
    assert prefill_wait_ns(None, s, False) == 0
    unstamped = {k: v for k, v in s.items() if k != "t_mtp_ns"}
    assert prefill_wait_ns(prev, unstamped, False) == 0
    # a status wait stamped after the launch has no W_inputs to hold a wait
    late = {**s, "t_mtp_ns": (314 * MS, 315 * MS)}
    assert prefill_wait_ns(prev, late, False) == 0


def test_an_output_prefill_s_wait_leaves_its_labels_once():
    prev = _output(0, 0, 10 * MS, 300 * MS)
    s = _output(80 * MS, 81 * MS, 313 * MS, 200 * MS)
    raw = _segments(s, True)
    host = take_prefill_wait(raw, prev, s, "prefill")
    assert host["W_inputs"] == raw["W_inputs"] - 229 * MS == 3 * MS
    assert {k: v for k, v in host.items() if k != "W_inputs"} == {
        k: v for k, v in raw.items() if k != "W_inputs"
    }
    assert sum(raw.values()) - sum(host.values()) == 229 * MS
    assert raw["W_inputs"] == 232 * MS  # the segments themselves are untouched


def test_a_middle_chunk_s_wait_leaves_its_w_pre_and_a_decode_s_labels_stay():
    prev = _middle(0, 0, 10 * MS, 600 * MS)
    s = _middle(80 * MS, 80 * MS, 611 * MS, 600 * MS)
    raw = _segments(s, False)
    host = take_prefill_wait(raw, prev, s, "middle")
    assert host["W_pre"] == raw["W_pre"] - 530 * MS == 1 * MS
    assert sum(raw.values()) - sum(host.values()) == 530 * MS
    decode = _output(80 * MS, 81 * MS, 313 * MS, 200 * MS)
    assert take_prefill_wait(
        _segments(decode, True), prev, decode, "decode"
    ) == _segments(decode, True)


def test_the_switch_reads_its_variable(monkeypatch):
    monkeypatch.delenv(step_labels.PREFILL_WAIT_ENV, raising=False)
    assert not step_labels.prefill_waits()
    monkeypatch.setenv(step_labels.PREFILL_WAIT_ENV, "0")
    assert not step_labels.prefill_waits()
    monkeypatch.setenv(step_labels.PREFILL_WAIT_ENV, "1")
    assert step_labels.prefill_waits()
    monkeypatch.setenv(step_labels.PREFILL_WAIT_ENV, "yes")
    with pytest.raises(ValueError):
        step_labels.prefill_waits()


def _recurrence(mtp):
    host = ("E", "W_pre", "W_launch", "W_post")
    laws = {
        kind: {s: [0.001, 0.0, 0.0] for s in (*host, "W_inputs")}
        for kind in ("prefill", "decode")
    }
    laws["middle"] = {
        s: [0.001, 0.0, 0.0] for s in host
    }  # a middle chunk runs no MTP inputs
    r = object.__new__(HostDeviceRecurrence)
    r.laws, r.columns, r.paths, r.skip, r.mtp = laws, {}, {}, frozenset(), mtp
    r.ends = collections.deque(
        [-math.inf, 0.5], maxlen=2
    )  # the previous device end, 0.5 s
    r.output_end = -math.inf
    return r


def test_the_simulated_host_runs_a_prefill_s_rest_after_the_previous_device_end(
    monkeypatch,
):
    r = _recurrence(True)
    monkeypatch.delenv(step_labels.PREFILL_WAIT_ENV, raising=False)
    off = {
        kind: r.prepare_end(0.0, kind, (1, 0, 0))[0]
        for kind in ("middle", "prefill", "decode")
    }
    monkeypatch.setenv(step_labels.PREFILL_WAIT_ENV, "1")
    on = {
        kind: r.prepare_end(0.0, kind, (1, 0, 0))[0]
        for kind in ("middle", "prefill", "decode")
    }
    plain, _ = _recurrence(False).prepare_end(0.0, "prefill", (1, 0, 0))
    assert math.isclose(off["middle"], 0.002)  # E, W_pre; the device two back long done
    assert math.isclose(off["prefill"], 0.003)  # E, W_pre, W_inputs
    assert math.isclose(on["middle"], 0.501)  # the previous device end, then W_pre
    assert math.isclose(
        on["prefill"], 0.501
    )  # E, W_pre beside it; its end, then W_inputs
    assert math.isclose(on["decode"], off["decode"])  # a decode keeps its own waits
    assert math.isclose(plain, 0.003)  # no MTP: no W_inputs wait to move
