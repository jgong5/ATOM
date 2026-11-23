# SPDX-License-Identifier: MIT
"""A runner that does not defer its output reads its own sampled ids.

ATOM's non-deferred `prepare_sampled_ids` blocks in `tolist` until the
forward's own sampled ids are on the host, so such a runner's output forward
leaves no earlier than its own target end, as the last pipeline stage's does;
one that defers leaves at the previous output forward's. Its record stamps
that read, which its labels and the launch's blocked time end at, as at a
token copy wait.
"""

import statistics
from types import SimpleNamespace

import pytest
from test_blocked_launch import _planted_tool
from test_dp_step_max import dp2, prefill, runner  # noqa: F401
from test_draft_after_exit import Drafted
from test_memory_readings import META_8B
from test_parity_device_window import BUSY, _Processor, _recorded
from test_parity_record import _config, _drive
from test_parity_recurrence import MS, _labelled, _law, line
from test_run_step_grid import B37, decode, deployment, publish, run_file

from atom.compass import run as compass_run
from atom.compass.backends.step_labels import HOST_SEGMENTS, label
from atom.compass.parity import ENV, read, recurrence
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


def test_under_dp_a_runner_that_reads_its_own_ids_leaves_after_its_draft(
    dp2,  # noqa: F811
):
    """Two DP ranks that do not defer, each a 5 ms target and a 2 ms draft
    with no host laws: each queues its draft after it reads its ids and reads
    the draft ids once it ends, so both leave at the group's device end."""
    backend = Drafted(5, 2)
    ranks = [runner(backend, dp_size=2, dp_rank=r) for r in (0, 1)]
    for r in ranks:
        r.compass_host = HostDeviceRecurrence({}, (), lambda: 0.0, defer=False)
    replies = dp2(*(lambda r=r: r.forward(prefill(8)) for r in ranks))
    assert [x.predicted_s for x in replies] == pytest.approx([7e-3, 7e-3])
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


def test_a_record_s_own_token_read_is_in_neither_its_launch_nor_its_post(tmp_path):
    """Each decode reads its own tokens for 9 ms, from its launch's end, and
    leaves 1 ms later: `W_launch` ends and `W_post` starts at that read."""
    samples = _labelled(tmp_path, own_reads())
    assert [s.host["W_launch"] for s in samples] == pytest.approx([1e-3] * 4)
    assert [s.host["W_post"] for s in samples] == pytest.approx([1e-3] * 4)


def _fitted(samples) -> dict:
    """`_law` with its decode vector and host laws at `samples`' median labels."""
    law = _law()
    device = statistics.median(s.device_s for s in samples)
    law["decode"]["eager"]["coefficients"]["step"] = device
    law["host"]["laws"] = {
        "decode": {
            g: [statistics.median(s.host[g] for s in samples if g in s.host), 0, 0]
            for g in HOST_SEGMENTS
        }
    }
    return law


@pytest.mark.parametrize("late", [0, 3])
def test_a_law_fitted_on_a_record_s_own_reads_grades_it_as_the_gate_does(
    tmp_path, late
):
    """Fitted on the on-time reads, the law charges each read once, after the
    forward's own device end; on the record whose fourth read ends 3 ms past
    it, that decode fails with the gate's residual."""
    law = _fitted(_labelled(tmp_path, own_reads()))
    steps = own_reads(late)
    gate = recurrence.check(steps)
    got = recurrence.priced(steps, _labelled(tmp_path, steps), law)
    assert got["pass"] == gate["pass"] == (not late)
    assert got["total"] == pytest.approx(gate["total"], abs=1e-9)
    assert got["kinds"] == pytest.approx(gate["kinds"])
    assert [r[:4] for r in got["failures"]] == pytest.approx(
        [r[:4] for r in gate["failures"]]
    )


def test_a_line_s_blocked_time_runs_to_its_own_token_read(tmp_path, monkeypatch):
    """A runner that reads its own tokens: each launch blocks 300 ns, an output
    forward's postprocess 50 ns more before its read, and the read itself
    7000 ns. `blocked_ns` holds the launch's and the postprocess's, as at a
    token copy wait, and `W_launch` runs to the read less them."""
    batches = _drive(tmp_path / "simulated", monkeypatch, {0: [0, 1]})[0][0]
    recorded, device = _recorded(batches[-1])
    tool, spend = _planted_tool(("hipEventSynchronize", "hipMemcpyWithStream"))
    late = [0]

    def blocking(call, ns, then=lambda *a: []):
        def blocked(*args):
            spend(call, ns)
            late[0] += ns
            return then(*args)

        return blocked

    class Blocking(recorded):
        def step_clock_ns(self):
            return recorded.step_clock_ns() + late[0]

    launch = recorded.__mro__[2].run_model
    monkeypatch.setattr(
        recorded.__mro__[2], "run_model", blocking("hipMemcpyWithStream", 300, launch)
    )
    monkeypatch.setattr(_Processor, "is_deferred_out", False, raising=False)
    monkeypatch.setattr(
        _Processor,
        "prepare_sampled_ids",
        lambda self, batch: blocking("hipMemcpyWithStream", 7000)(),
    )
    monkeypatch.setenv(ENV, str(tmp_path / "real"))
    runner = Blocking(0, _config())
    runner._blocked = tool
    read_ids = runner.tokenID_processor.prepare_sampled_ids
    runner.tokenID_processor.prepare_sampled_ids = blocking(
        "hipMemcpyWithStream", 50, read_ids
    )
    for batch in batches:
        runner.forward(batch)
    device.tick = BUSY // 3
    runner.exit()

    lines = read(tmp_path / "real")[0]
    outputs = [b.produces_output() for b in batches]
    assert any(outputs) and not all(outputs)
    assert [("t_read_ns" in x) for x in lines] == outputs
    want = [{"hipMemcpyWithStream": 350 if o else 300} for o in outputs]
    assert [x["blocked_ns"] for x in lines] == want
    for s, x, w in zip(label(tmp_path / "real" / "dp0.jsonl")[0], lines, want):
        l0, l1 = x["t_launch_ns"]
        wall = x.get("t_read_ns", [l1])[0] - l0
        assert s.host["W_launch"] == pytest.approx((wall - sum(w.values())) * 1e-9)
