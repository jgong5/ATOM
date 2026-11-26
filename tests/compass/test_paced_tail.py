# SPDX-License-Identifier: MIT
"""A host-paced forward's device tail under ``COMPASS_FIT_PACED_TAIL=1``.

Where a prefill's host launch paces its device, the device waits on each
kernel the host launches and runs only its last kernels past the launch's end;
a recurrence that ends such a forward's device with its launch starts the next
forward's launch, which waits for it (`step_labels.WAIT_ENV`), that much early.
A host fit under the switch states, per host kind, the median of how far a
paced forward's device ran past its launch's end (`step_labels.paced_tails`,
the entry's `step_labels.TAILS`), and a run's recurrence ends such a forward's
target no earlier than its launch end plus that. Unset, no entry states it and
every paced forward's device ends with its launch.

No device.
"""

import json

import pytest
from test_launch_wait import DECODE, LAWS, MS, PREFILL, _batch, _line

from atom.compass import run as compass_run
from atom.compass.backends import step_labels
from atom.compass.fit import host as fit_host
from atom.compass.runner.overrides import HostDeviceRecurrence

TAILS = step_labels.TAILS


def _lines():
    """A decode; a 500-token prefill and a middle chunk the host paced, their
    device 7 and 6 ms past their launch's end; a device-bound middle chunk,
    1.12 s past it; and a prefill whose device ended before its launch did."""
    return [
        _line(0, DECODE, True, (0, 0.2), (0, 10), None),
        _line(1, PREFILL, True, (11, 191), (11, 198), 10),
        _line(2, PREFILL, False, (200, 380), (200, 386), 198),
        _line(3, PREFILL, False, (390, 570), (390, 1690), 386),
        _line(4, PREFILL, True, (1700, 1880), (1700, 1870), 1690),
    ]


def test_a_paced_forward_s_tail_is_its_device_past_its_launch():
    assert step_labels.device_tails(_lines()) == {
        "prefill": [7 * MS],
        "middle": [6 * MS],
    }
    entry, report = step_labels.paced_tails([[_lines(), _lines()]])
    assert entry == pytest.approx({"prefill": 0.007, "middle": 0.006})
    assert report["middle"]["forwards"] == 2
    assert report["prefill"]["p90_s"] == pytest.approx(0.007)


def _record(tmp_path):
    steps = tmp_path / "rec" / "steps"
    steps.mkdir(parents=True)
    (steps / "dp0.jsonl").write_text("".join(json.dumps(x) + "\n" for x in _lines()))
    return tmp_path / "rec"


@pytest.mark.parametrize("value", [None, "0", "1"])
def test_a_host_fit_states_the_tails_under_the_switch_only(
    tmp_path, monkeypatch, value
):
    record = _record(tmp_path)
    fitted = {"laws": {"prefill": {"W_launch": [0.17, 0, 0]}}}
    monkeypatch.setattr(
        fit_host, "fit_host", lambda samples: (dict(fitted), {"unfitted": {}})
    )
    monkeypatch.setattr(fit_host.publish, "runs", lambda samples: {})
    for env in ("COMPASS_FIT_LAUNCH_WAIT", "COMPASS_FIT_LAUNCH_STALL"):
        monkeypatch.delenv(env, raising=False)
    if value is None:
        monkeypatch.delenv(step_labels.TAIL_ENV, raising=False)
    else:
        monkeypatch.setenv(step_labels.TAIL_ENV, value)
    host, report = fit_host.host_laws([], records=[record])
    if value == "1":
        assert host[TAILS] == pytest.approx({"prefill": 0.007, "middle": 0.006})
        assert report[TAILS]["prefill"]["forwards"] == 1
    else:
        assert host == fitted and TAILS not in report


def test_the_switch_takes_0_or_1(monkeypatch):
    monkeypatch.setenv(step_labels.TAIL_ENV, "on")
    with pytest.raises(ValueError, match=step_labels.TAIL_ENV):
        step_labels.paced_tail_on()


def _end(tails, device):
    """One output prefill of `device` s on the device, its launch 0.18 s:
    its device end."""
    rec = HostDeviceRecurrence(LAWS, (), lambda: 0.0, tails=tails)
    rec.charge(_batch(True, True, 500), device, 0.0)
    return rec.ends[-1]


def test_a_run_ends_a_paced_forward_s_device_its_tail_past_its_launch():
    # Unset: the paced forward's device ends with its 0.18 s launch.
    assert _end(None, 0.05) == pytest.approx(0.18)
    # Its tail: 7 ms past the launch's end.
    assert _end({"prefill": 0.007}, 0.05) == pytest.approx(0.187)
    # A device-bound forward ends with its own work either way.
    assert _end({"prefill": 0.007}, 1.0) == pytest.approx(1.0)
    # A kind the tails do not name keeps its launch end.
    assert _end({"middle": 0.007}, 0.05) == pytest.approx(0.18)


def test_a_run_file_s_host_entry_may_state_the_tails(tmp_path):
    host = {"laws": LAWS, TAILS: {"prefill": 0.007, "middle": 0.006}}
    compass_run._check_host(tmp_path / "run.json", host)
    for bad in ({"verify": 0.007}, {"prefill": -0.001}, [0.007]):
        with pytest.raises(ValueError, match=TAILS):
            compass_run._check_host(tmp_path / "run.json", {**host, TAILS: bad})
