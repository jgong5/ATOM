# SPDX-License-Identifier: MIT
"""A runner that does not defer its output reads its own sampled ids.

ATOM's non-deferred `prepare_sampled_ids` blocks in `tolist` until the
forward's own sampled ids are on the host, so such a runner's output forward
leaves no earlier than its own target end, as the last pipeline stage's does;
one that defers leaves at the previous output forward's.
"""

from types import SimpleNamespace

import pytest
from test_dp_step_max import dp2, prefill, runner  # noqa: F401
from test_draft_after_exit import Drafted
from test_memory_readings import META_8B
from test_parity_recurrence import MS, line
from test_run_step_grid import B37, decode, deployment, publish, run_file

from atom.compass import run as compass_run
from atom.compass.parity import recurrence
from atom.compass.runner.overrides import HostDeviceRecurrence

OUTPUT = SimpleNamespace(
    total_seqs_num_prefill=0,
    total_tokens_num=1,
    total_seqs_num=1,
    produces_output=lambda: True,
)


@pytest.mark.parametrize("defer, exits", [(True, [0, 2, 4]), (False, [2, 4, 6])])
def test_a_runner_that_reads_its_own_ids_leaves_at_its_own_device_end(defer, exits):
    """Three 2 s output forwards on one stage with no host laws, each called at
    the exit before it: the device ends at 2, 4 and 6 s either way."""
    clock, got = [0.0], []
    host = HostDeviceRecurrence({}, (), lambda: clock[0], defer=defer)
    for _ in exits:
        clock[0] += host.charge(OUTPUT, 2.0, 0.0)
        got.append((host.ends[-1], clock[0]))
    assert got == list(zip([2, 4, 6], exits))


@pytest.mark.parametrize("deferred, prices", [(True, [0, 1]), (False, [1, 2])])
def test_a_run_s_runner_waits_as_its_token_processor_defers(
    tmp_path, monkeypatch, deferred, prices
):
    """Two forwards called at the same time: the second's device work follows
    the first's, and a runner that does not defer leaves at its own end."""
    r = deployment(tmp_path)
    r.tokenID_processor = SimpleNamespace(is_deferred_out=deferred)
    publish(tmp_path, r)
    run = run_file(tmp_path, monkeypatch)
    monkeypatch.setenv(compass_run.NOW_ENV, str(tmp_path / "now.bin"))
    compass_run._PublishedClock(
        compass_run.ENGINE, compass_run.channel_table(run), None
    )
    compass_run.runner(r, META_8B)
    price = 8897.5e-6  # `test_run_step_grid`'s price of this batch
    charged = [r.forward(decode(B37)).predicted_s for _ in prices]
    assert charged == pytest.approx([n * price for n in prices])


def test_under_dp_a_runner_that_reads_its_own_ids_leaves_before_its_draft(
    dp2,  # noqa: F811
):
    """Two DP ranks that do not defer, each a 5 ms target and a 2 ms draft
    with no host laws: each queues its draft after it reads its ids, so both
    leave at the group's target end, and the device ends after the draft."""
    backend = Drafted(5, 2)
    ranks = [runner(backend, dp_size=2, dp_rank=r) for r in (0, 1)]
    for r in ranks:
        r.compass_host = HostDeviceRecurrence({}, (), lambda: 0.0, defer=False)
    replies = dp2(*(lambda r=r: r.forward(prefill(8)) for r in ranks))
    assert [x.predicted_s for x in replies] == pytest.approx([5e-3, 5e-3])
    assert [r.compass_host.ends[-1] for r in ranks] == pytest.approx([7e-3, 7e-3])


def own_reads(late=0):
    """Decodes of a runner that reads its own tokens, in ms: each enters 1
    after the last exit and launches from 1 to 2 after its entry, its 10 of
    device work starting at the launch; it reads its tokens from the launch's
    end to its device end, the fourth `late` after it, and leaves 1 later."""
    steps, enter = [], 0
    for n in range(4):
        end = enter + 11 + late * (n == 3)
        launch = (enter + 1, enter + 2)
        s = line(
            n, [[1, 151 + n, True]], True, enter, (enter,) * 2, launch, None, end + 1
        )
        s["t_read_ns"] = [launch[1] * MS, end * MS]
        steps.append(s)
        enter = end + 2
    return steps


def test_the_gate_releases_a_record_s_own_token_read_at_its_device_end():
    report = recurrence.check(own_reads())
    assert report["pass"] and report["total"] == pytest.approx(0, abs=1e-9)
    late = recurrence.check(own_reads(late=3))
    assert not late["pass"] and not late["kinds"]["decode"][-1]
    [(step, kind, err, residual, *_)] = late["failures"]
    assert (step, kind, residual) == (3, "decode", pytest.approx(-3))
    assert err == pytest.approx(100 * (13 / 16 - 1))
