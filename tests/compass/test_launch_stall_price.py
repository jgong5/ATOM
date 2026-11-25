# SPDX-License-Identifier: MIT
"""Launch stalls priced under ``COMPASS_FIT_LAUNCH_STALL=1`` as the first launch
of a shape on a rank: a host fit states, per launch path, the excess of the
stalls `step_labels.launch_stalls` finds in its records over the ranks' first
launches of a `step_labels.first_launch_key` (`step_labels.stall_prices`, the
host entry's `step_labels.STALLS`), and a run charges it on each first launch
of a key on a rank (`runner.overrides.launch_stall`), with no randomness, in its
device time where a replayed stall is, so the DP group's MAX holds every rank to
the stalled one. Unset, no host entry states stalls and a run charges none.

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

STALLS, KEY = step_labels.STALLS, step_labels.FIRST_LAUNCH_KEY


def _tokens(i):
    """100 to 1000 tokens in turn: power-of-2 keys 128, 256, 512 (first at 3rd)
    and 1024 (first at 6th)."""
    return 100 * (i % 10 + 1)


def _rank(rank, stalls, n=100):
    """`n` eager-body prefill lines of 50 ms, those at `stalls` (index:
    launch ms) longer."""
    return [
        _line(i, rank, stalls.get(i, 50), rows=((_tokens(i), 9000, False),))
        for i in range(n)
    ]


def _ranks():
    """Rank 0 stalls at its first 128 and 1024 launches by 200 ms, rank 1 at its
    first 256 by 100 ms: 0.5 s over 8 first launches. A decode and a dummy run,
    however long, weigh on nothing."""
    r0 = _rank(0, {0: 250, 5: 250})
    r0 += [_line(100, 0, 900, rows=((1, 900, True),)), _line(101, 0, 900, dummy=True)]
    return [r0, _rank(1, {1: 150})]


def test_a_two_rank_run_s_first_launch_cost_is_recovered_per_path():
    ranks = _ranks()
    assert step_labels.launch_stalls(ranks) == {0, 1, 5}
    got = step_labels.stall_prices([ranks])
    assert set(got) == {"eager_body"}
    body = got["eager_body"]
    assert (body["forwards"], body["stalled"], body["first"]) == (200, 3, 8)
    assert body["excess_s"] == pytest.approx(0.5)
    assert body["first_s"] == pytest.approx(0.5 / 8)
    assert step_labels.stall_entry(got) == {
        "eager_body": {"first_s": body["first_s"], "key": KEY}
    }
    # Two runs are two processes: each counts its own first launches.
    assert step_labels.stall_prices([ranks, ranks])["eager_body"]["first"] == 16
    assert step_labels.stall_prices([[_rank(0, {})]]) == {}


def test_the_key_is_the_prefill_tokens_up_to_a_power_of_2():
    def key(*rows):
        return step_labels.first_launch_key([RequestShape(*r) for r in rows])

    assert [key((q, 9000, False)) for q in (1, 2, 3, 100, 128, 129)] == [
        1,
        2,
        4,
        128,
        128,
        256,
    ]
    assert key((100, 900, False), (30, 40, False), (1, 800, True)) == 256


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


def test_a_run_file_s_host_entry_takes_stalls_on_its_own_key_only():
    laws = {"prefill": {"W_launch": [1e-3, 0, 0]}}
    good = {"replay": {"first_s": 0.1, "key": KEY}}
    compass_run._check_host("run.json", {"laws": laws, STALLS: good})
    for bad in (
        {"replay": {"first_s": 0.1, "key": "exact_prefill_tokens"}},
        {"replay": {"first_s": -0.1, "key": KEY}},
        {"replay": {"first_s": 0.1}},
        {"replay": 0.1},
    ):
        with pytest.raises(ValueError, match=STALLS):
            compass_run._check_host("run.json", {"laws": laws, STALLS: bad})


def _view(tokens=600, prefill=True, dummy=False):
    return BatchView((RequestShape(tokens, 9000, not prefill),), dp_dummy=dummy)


def _charges(runner_):
    host = HostDeviceRecurrence(
        {}, (), lambda: 0.0, stalls={"replay": {"first_s": 0.3, "key": KEY}}
    )
    views = [
        _view(600),  # the first 1024: charged
        _view(1000),  # 1024 again: none
        _view(1025),  # the first 2048
        _view(600, prefill=False),  # a decode
        _view(3000, dummy=True),  # a DP dummy, a new key: none, and not seen
        _view(3000),  # so the first 4096 here
        _view(600),
    ]
    return [overrides.launch_stall(runner_, v, host) for v in views]


def test_a_rank_s_first_launch_of_a_key_costs_its_path_s_first_s_once():
    r = SimpleNamespace()
    assert _charges(r) == [0.3, 0.0, 0.3, 0.0, 0.0, 0.3, 0.0]
    # The same runner, its keys seen: nothing more.
    assert _charges(r) == [0.0] * 7
    # Another process starts with none seen, and charges the same again.
    assert _charges(SimpleNamespace()) == [0.3, 0.0, 0.3, 0.0, 0.0, 0.3, 0.0]
    # A path the stalls do not name, or no stalls at all, charges nothing.
    named = HostDeviceRecurrence(
        {}, (), lambda: 0.0, stalls={"eager_body": {"first_s": 0.3, "key": KEY}}
    )
    assert overrides.launch_stall(SimpleNamespace(), _view(), named) == 0.0
    bare = HostDeviceRecurrence({}, (), lambda: 0.0)
    assert overrides.launch_stall(SimpleNamespace(), _view(), bare) == 0.0
    assert bare.path(_view()) is None


def _steps(run, stalls, n=4):
    """`n` prefill steps of a two-rank group, rank 0 of 40 tokens and rank 1 of
    8, rank 0's host stating `stalls`; each rank's device end after each."""
    backend = StepStubBackend()
    ranks = [runner(backend, dp_size=2, dp_rank=r) for r in range(2)]
    for r, table in zip(ranks, (stalls, None)):
        r.compass_host = HostDeviceRecurrence({}, (), lambda: 0.0, stalls=table)

    def steps(r, tokens):
        def go():
            ends = []
            for _ in range(n):
                r.forward(prefill(tokens))
                ends.append(vars(r)["_device_end"])
            return ends

        return go

    return run(steps(ranks[0], 40), steps(ranks[1], 8))


def test_a_first_launch_holds_the_dp_group_once_and_runs_repeat(dp2):  # noqa: F811
    plain = _steps(dp2, None)
    assert plain[0] == plain[1]
    stalls = {"replay": {"first_s": 0.5, "key": KEY}}
    held = _steps(dp2, stalls)
    # Rank 0's first 64-token launch holds both ranks 0.5 s; its repeats cost
    # nothing, so each later step ends 0.5 s after its plain twin.
    assert held[0] == held[1] == pytest.approx([e + 0.5 for e in plain[0]])
    assert _steps(dp2, stalls) == held
