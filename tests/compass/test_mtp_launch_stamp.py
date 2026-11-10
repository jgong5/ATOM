# SPDX-License-Identifier: MIT
"""A launch's bound on the host clock's lead is read just before its event is
recorded, so a host that stalls after recording it does not lift the bound,
and the MTP status copy's labels stay the copy's latency."""

import pytest
from test_mtp_clock_drift import COPY, DRIFT, LATE, _decodes
from test_parity_device_window import _recorded
from test_parity_record import _config, _drive
from test_parity_recurrence import MS

from atom.compass.backends.step_labels import ANCHORS, mtp_copy
from atom.compass.parity import ENV, read

#: Milliseconds forward `AT`'s host stalls between recording its launch's event
#: and stamping the launch's entry: under the 0.5 ms margin from its launch
#: bound to the status waits', so a bound read at the entry stays in, and over
#: the 0.25 ms margin of a `LATE` forward's, so it would set the lead.
STALL, AT = 0.4, 120


def test_a_stall_under_the_status_waits_margin_keeps_the_status_copy_label():
    """Read at the entry stamp, the stalled bound would set the lead and
    label each copy 0.1 ms or less."""
    steps = _decodes()
    for s in steps:  # each launch's event is recorded as the launch is stamped
        s["t_launch_event_ns"] = s["t_launch_ns"][0]
    steps[AT]["t_launch_ns"][0] += round(STALL * MS)
    copy = mtp_copy(steps)
    waited = {n: c for n, (c, began) in copy.items() if began}
    assert waited.keys() == set(range(1, 200)) - LATE
    assert waited == {n: pytest.approx(COPY * MS, abs=ANCHORS * DRIFT) for n in waited}


def test_the_launch_is_read_before_its_event_is_recorded(tmp_path, monkeypatch):
    batches = _drive(tmp_path / "simulated", monkeypatch, {0: [0, 1]})[0][0]
    recorded, device = _recorded(batches[-1])
    events = []  # the host clock once each launch's event is recorded

    class _Timed(recorded):
        def device_event(self):
            launch = self.queued == ["launch"]
            event = super().device_event()
            if launch:
                events.append(self.step_clock_ns())
            return event

    device.tick = 10**12
    monkeypatch.setenv(ENV, str(tmp_path / "real"))
    runner = _Timed(0, _config())
    for b in batches:
        runner.forward(b)
    runner.exit()
    steps = read(tmp_path / "real")[0]
    assert len(steps) == len(events) == len(batches)
    for s, at in zip(steps, events):
        assert s["t_gate_ns"][1] < s["t_launch_event_ns"] < at < s["t_launch_ns"][0]
