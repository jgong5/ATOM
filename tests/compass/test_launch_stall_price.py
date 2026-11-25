# SPDX-License-Identifier: MIT
"""Launch stalls priced under ``COMPASS_FIT_LAUNCH_STALL=1``: a host fit states
the stalls `step_labels.launch_stalls` finds in its records, per launch path
(`step_labels.stall_prices`, the host entry's `step_labels.STALLS`), and a run
draws each prefill forward's stall from them on a per-rank seeded stream
(`runner.overrides.launch_stall`), added to its device time where a replayed
stall is, so the DP group's MAX holds every rank to the stalled one. Unset, no
host entry states stalls and a run draws none.

No device.
"""

import json
from types import SimpleNamespace

import pytest
from test_dp_step_max import dp2, prefill, runner  # noqa: F401  the dp2 fixture
from test_step_labels_launch_stall import _line

from atom.compass import run as compass_run
from atom.compass.backends import step_labels
from atom.compass.backends.shape import BatchView, RequestShape
from atom.compass.backends.step_stub import StepStubBackend
from atom.compass.fit import host as fit_host
from atom.compass.runner import overrides
from atom.compass.runner.overrides import HostDeviceRecurrence

STALLS = step_labels.STALLS


def _ranks(n=200):
    """Two ranks of `n` eager-body prefill lines of 50 ms: rank 0's every 20th
    stalled at 250 ms, rank 1's every 25th at 150 ms, 200 and 100 ms past their
    median; a decode and a dummy run, however long, weigh on nothing."""
    r0 = [_line(i, 0, 250 if i % 20 == 19 else 50) for i in range(n)]
    r0 += [_line(n, 0, 900, rows=((1, 900, True),)), _line(n + 1, 0, 900, dummy=True)]
    r1 = [_line(i, 1, 150 if i % 25 == 24 else 50) for i in range(n)]
    return [r0, r1]


def test_a_two_rank_run_s_stall_rate_and_excess_are_recovered_per_path():
    ranks = _ranks()
    got = step_labels.stall_prices([ranks])
    assert set(got) == {"eager_body"}
    body = got["eager_body"]
    assert (body["forwards"], body["stalled"]) == (400, 10 + 8)
    assert body["rate"] == pytest.approx(18 / 400)
    assert body["excess_s"] == pytest.approx((10 * 0.2 + 8 * 0.1) / 18)
    # The lines `launch_stalls` names stalled: steps 99 and 199 on both ranks.
    assert len(step_labels.launch_stalls(ranks)) == 18 - 2
    assert step_labels.stall_entry(got) == {
        "eager_body": {"rate": body["rate"], "excess_s": body["excess_s"]}
    }
    # Runs pool per path; a run with no stall prices none.
    assert step_labels.stall_prices([ranks, ranks])["eager_body"]["forwards"] == 800
    assert step_labels.stall_prices([[[_line(i, 0, 50) for i in range(5)]]]) == {}


def _record(tmp_path):
    steps = tmp_path / "rec" / "steps"
    steps.mkdir(parents=True)
    for r, lines in enumerate(_ranks()):
        text = "".join(json.dumps(x) + "\n" for x in lines)
        (steps / f"dp{r}.jsonl").write_text(text)
    # Beside the rank records and never read as one.
    (steps / "dp0.cost.jsonl").write_text('{"version": 1}\n')
    return tmp_path / "rec"


@pytest.mark.parametrize("value", [None, "0", "1"])
def test_a_host_fit_states_the_records_stalls_under_the_switch_only(
    tmp_path, monkeypatch, value
):
    record = _record(tmp_path)
    assert len(step_labels.record_ranks(record)) == 2
    fitted = {"laws": {"prefill": {"W_launch": [0.05, 0, 0]}}}
    monkeypatch.setattr(
        fit_host, "fit_host", lambda samples: (dict(fitted), {"unfitted": {}})
    )
    monkeypatch.setattr(fit_host.publish, "runs", lambda samples: {})
    if value is None:
        monkeypatch.delenv(step_labels.STALL_ENV, raising=False)
    else:
        monkeypatch.setenv(step_labels.STALL_ENV, value)
    host, report = fit_host.host_laws([], records=[record])
    if value == "1":
        prices = step_labels.stall_prices([_ranks()])
        assert report[STALLS] == prices
        assert host == {**fitted, STALLS: step_labels.stall_entry(prices)}
    else:
        assert host == fitted and STALLS not in report


def test_a_run_file_s_host_entry_takes_stalls_of_a_rate_and_an_excess():
    laws = {"prefill": {"W_launch": [1e-3, 0, 0]}}
    compass_run._check_host(
        "run.json", {"laws": laws, STALLS: {"replay": {"rate": 0.02, "excess_s": 0.2}}}
    )
    for bad in (
        {"replay": {"rate": 1.5, "excess_s": 0.2}},
        {"replay": {"rate": 0.1, "excess_s": -0.2}},
        {"replay": {"rate": 0.1}},
        {"replay": 0.2},
    ):
        with pytest.raises(ValueError, match=STALLS):
            compass_run._check_host("run.json", {"laws": laws, STALLS: bad})


def _view(prefill=True, dummy=False, rank=0):
    shape = RequestShape(600, 900, not prefill)
    return BatchView((shape,), dp_rank=rank, dp_dummy=dummy)


def _draws(seed, monkeypatch, n=2000, rate=0.05):
    monkeypatch.setenv(overrides.STALL_SEED, seed)
    host = HostDeviceRecurrence(
        {}, (), lambda: 0.0, stalls={"replay": {"rate": rate, "excess_s": 0.3}}
    )
    r = SimpleNamespace()
    got = []
    for i in range(n):
        got.append(overrides.launch_stall(r, _view(), host))
        # A decode, a DP dummy and a path the stalls do not name draw nothing.
        assert overrides.launch_stall(r, _view(prefill=False), host) == 0.0
        assert overrides.launch_stall(r, _view(dummy=True), host) == 0.0
    return [i for i, s in enumerate(got) if s], set(got)


def test_a_seeded_run_reproduces_its_stalls_at_the_fitted_rate(monkeypatch):
    first, sizes = _draws("1", monkeypatch)
    assert sizes == {0.0, 0.3}
    assert 70 <= len(first) <= 130  # 2000 draws at 5%
    assert _draws("1", monkeypatch)[0] == first
    assert _draws("2", monkeypatch)[0] != first
    # With no stalls a host prices none and names no path.
    bare = HostDeviceRecurrence({}, (), lambda: 0.0)
    assert overrides.launch_stall(SimpleNamespace(), _view(), bare) == 0.0
    assert bare.path(_view()) is None


def _group(run, stalls):
    """Rank 0 prefills 40 tokens, rank 1 8, each on a host of no laws, rank 0's
    stating `stalls`; each rank's device end, the group's MAX, and its own."""
    backend = StepStubBackend()
    ranks = [runner(backend, dp_size=2, dp_rank=r) for r in range(2)]
    for r, table in zip(ranks, (stalls, None)):
        r.compass_host = HostDeviceRecurrence({}, (), lambda: 0.0, stalls=table)
    run(lambda: ranks[0].forward(prefill(40)), lambda: ranks[1].forward(prefill(8)))
    own = [r.compass_host.target_end for r in ranks]
    return [vars(r)["_device_end"] for r in ranks], own


def test_a_drawn_stall_holds_the_whole_dp_group(dp2):  # noqa: F811
    ends, own = _group(dp2, None)
    assert ends == [own[0], own[0]] and own[0] > own[1]
    stalls = {"replay": {"rate": 1.0, "excess_s": 0.5}}
    held, held_own = _group(dp2, stalls)
    # Rank 0's device runs 0.5 s longer, and rank 1, its own target unmoved,
    # leaves the group's step at rank 0's end.
    assert held_own == [pytest.approx(own[0] + 0.5), pytest.approx(own[1])]
    assert held == [pytest.approx(own[0] + 0.5)] * 2
