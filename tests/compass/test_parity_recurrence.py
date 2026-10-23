# SPDX-License-Identifier: MIT
"""The recurrence's own check, on a hand-built record of one request.

Two middle chunks, the output prefill, then two decodes, each 10 ms of device
work and 1 ms per host segment. The stamps are what the recurrence's wait
rules give: the third forward's gate waits for the first one's device end,
and each decode waits to receive the previous output forward's.
"""

import json

import pytest

from atom.compass.backends.step_labels import label
from atom.compass.backends.step_linear import DECODE, PREFILL
from atom.compass.parity import recurrence

MS = 1_000_000


def line(step, rows, output, enter, gate, launch, recv, exit_):
    s = {
        "step": step,
        "dp_rank": 0,
        "is_dummy_run": False,
        "produces_output": output,
        "batch": [[7, rows[0][0], rows[0][1]]],
        "rows": rows,
        "t_enter_ns": enter * MS,
        "t_gate_ns": [g * MS for g in gate],
        "t_launch_ns": [t * MS for t in launch],
        "t_exit_ns": exit_ * MS,
        "d_busy_ns": 10 * MS,
    }
    if recv:
        s["t_recv_ns"] = [t * MS for t in recv]
    return s


def record(r1=41):
    """The last decode's recv wait ends at `r1` ms; 41 is step 3's device end."""
    return [
        line(0, [[50, 50, False]], False, 0, (0, 0), (1, 2), None, 3),
        line(1, [[50, 100, False]], False, 4, (4, 4), (5, 6), None, 7),
        line(2, [[50, 150, False]], True, 8, (8, 11), (12, 13), (13, 13), 14),
        line(3, [[1, 151, True]], True, 15, (15, 21), (22, 23), (23, 31), 32),
        line(4, [[1, 152, True]], True, 33, (33, 33), (34, 35), (35, r1), r1 + 1),
    ]


def test_measured_inputs_reproduce_a_record_the_wait_rules_wrote():
    report = recurrence.check(record())
    assert set(report["kinds"]) == {"idle", "middle", "prefill", "decode"}
    for n, p10, p50, p90, r1, ok in report["kinds"].values():
        assert (p10, p90) == pytest.approx((0, 0), abs=1e-9) and ok
    assert report["kinds"]["decode"][4] == pytest.approx(0, abs=1e-6)
    assert report["total"] == pytest.approx(0, abs=1e-9)
    assert report["pass"] and not report["failures"]


def test_an_output_forward_s_sampler_launch_overlaps_the_device_before_its_read():
    """Step 4 launches its sampler from 35 to 37 ms, then waits for step 3's
    device end at 41: the launch runs under the wait, so the exit stays 42."""
    steps = record()
    steps[4]["t_recv_ns"][0] = 37 * MS
    report = recurrence.check(steps)
    assert report["total"] == pytest.approx(0, abs=1e-9)
    assert report["pass"] and not report["failures"]


def test_a_host_bound_sampler_launch_delays_the_exit_past_the_device_end():
    """Step 4 launches its sampler from 34 to 42 ms, after step 3's device end
    at 41: the read starts at 42, so the exit is 43, not 42."""
    steps = record(r1=42)
    steps[4]["t_recv_ns"][0] = 42 * MS
    report = recurrence.check(steps)
    assert report["total"] == pytest.approx(0, abs=1e-9)
    assert report["pass"] and not report["failures"]


def test_a_late_r1_release_fails_its_kind_and_names_the_step():
    report = recurrence.check(record(r1=45))
    assert not report["kinds"]["decode"][-1]
    assert report["kinds"]["prefill"][-1]
    assert report["total"] == pytest.approx(100 * (42 / 46 - 1))
    assert not report["pass"]
    # step 4 launches at 34 behind step 3's device work, which ends at 41
    [(step, _, err, residual, queued, r1)] = report["failures"]
    assert (step, err) == (4, pytest.approx(100 * (10 / 14 - 1)))
    assert (residual, queued, r1) == pytest.approx((-4, 7, -4))


def test_the_script_reads_a_record_directory(tmp_path, capsys):
    (tmp_path / "dp0.jsonl").write_text(
        "".join(json.dumps(s | {"named": []}) + "\n" for s in record(r1=45))
    )
    assert recurrence.main(str(tmp_path)) == 1
    out = capsys.readouterr().out
    assert "decode: 1 steps outside the band" in out
    assert "step 4: -28.57%" in out


def test_every_kind_can_pass_while_the_run_total_still_fails():
    """Five decodes, each receiving 0.1 ms later than the record's first two:
    within every kind's step band, but their sum breaches the tighter total one."""
    steps, exit_ = record()[:4], 32
    for i in range(5):
        enter, launch = exit_ + 1, (exit_ + 2, exit_ + 3)
        recv_end = enter + 8.1
        steps.append(
            line(
                4 + i,
                [[1, 152 + i, True]],
                True,
                enter,
                (enter, enter),
                launch,
                (launch[1], recv_end),
                recv_end + 1,
            )
        )
        exit_ = recv_end + 1
    report = recurrence.check(steps)
    assert all(k[-1] for k in report["kinds"].values()) and not report["failures"]
    assert abs(report["total"]) > recurrence.TOTAL_BAND
    assert not report["pass"]


def test_one_middle_chunk_outside_the_band_fails_its_kind_of_nine():
    """Ten chunks of one request, 1 ms of device work each, 4 ms apart; the
    seventh waits 0.15 ms at a staging gate no rule predicts. Interpolated,
    the nine middle chunks' p10 is -0.72% and the kind would pass."""
    steps = []
    for n in range(10):
        enter = 4 * n + 0.15 * (n > 6)
        wait = 0.15 * (n == 6)
        launch = (enter + wait + 1, enter + wait + 2)
        rows = [[50, 50 * (n + 1), False]]
        s = line(n, rows, False, enter, (enter, enter + wait), launch, None, 0)
        s["t_exit_ns"], s["d_busy_ns"] = (launch[1] + 1) * MS, MS
        steps.append(s)
    report = recurrence.check(steps)
    n, p10, *_, ok = report["kinds"]["middle"]
    assert (n, p10, ok) == (9, pytest.approx(100 * (4 / 4.15 - 1)), False)
    [(step, kind, *_)] = report["failures"]
    assert (step, kind) == (6, "middle") and not report["pass"]


def test_one_decode_released_early_fails_its_kind_of_ten():
    """The record's decode, then nine whose receives end with the device but
    the sixth's 0.3 ms early: charged +3.09%; interpolated, p90 is +0.31%."""
    steps, exit_ = record()[:4], 32
    for i in range(9):
        enter, launch = exit_ + 1, (exit_ + 2, exit_ + 3)
        recv = (launch[1], enter + 8 - 0.3 * (i == 5))
        steps.append(
            line(
                4 + i,
                [[1, 152 + i, True]],
                True,
                enter,
                (enter,) * 2,
                launch,
                recv,
                recv[1] + 1,
            )
        )
        exit_ = recv[1] + 1
    report = recurrence.check(steps)
    n, *_, p90, _, ok = report["kinds"]["decode"]
    assert (n, p90, ok) == (10, pytest.approx(100 * (10 / 9.7 - 1)), False)
    [(step, kind, *_)] = report["failures"]
    assert (step, kind) == (9, "decode") and not report["pass"]


def test_a_record_without_device_windows_is_refused():
    steps = record()
    del steps[2]["d_busy_ns"]
    with pytest.raises(ValueError, match="step 2 has no device window"):
        recurrence.check(steps)


def test_a_decode_charges_the_device_gap_before_it_and_a_prefill_does_not():
    """A 4 ms gap before step 3 ends its device work at 45 ms, where step 4
    receives it; the same gap before the output prefill moves nothing."""
    steps = record(r1=45)
    steps[2]["d_gap_ns"] = steps[3]["d_gap_ns"] = 4 * MS
    report = recurrence.check(steps)
    assert report["total"] == pytest.approx(0, abs=1e-9)
    assert report["pass"] and not report["failures"]


def mtp_record():
    """Four MTP decodes of one request, written by the wait rules: 1 ms per
    host segment, 10 ms of target and 0.5 of draft. Each launch waits for the
    previous target's end; the draft copy is in by the time it is read."""
    steps = []
    for n, (enter, mtp_end) in enumerate([(0, 1), (4, 11), (14, 21.5), (24.5, 32)]):
        launch = (mtp_end, mtp_end + 1)
        s = line(n, [[2, 150 + n, True]], True, enter, (enter, enter), launch, None, 0)
        s["t_mtp_ns"] = [(enter + 1) * MS, mtp_end * MS]
        s["t_recv_ns"] = [launch[1] * MS] * 2
        s["t_draft_recv_ns"] = [(launch[1] + 1) * MS] * 2
        s["t_exit_ns"] = (launch[1] + 1) * MS
        s["d_busy_ns"], s["d_target_ns"] = 10.5 * MS, 10 * MS
        if n:
            s["d_gap_ns"] = 0  # each target starts as the draft before it ends
        steps.append(s)
    return steps


def test_an_mtp_record_the_wait_rules_wrote_passes_and_charged_without_mtp_fails():
    report = recurrence.check(mtp_record())
    assert report["kinds"]["decode"][1:4] == pytest.approx((0, 0, 0), abs=1e-9)
    assert report["kinds"]["decode"][4] == pytest.approx(0, abs=1e-6)
    assert report["pass"] and report["total"] == pytest.approx(0, abs=1e-9)
    # A wait that holds nothing is released where the host reaches it, inside
    # W_pre, not at the launch.
    steps = mtp_record()
    steps[0]["t_mtp_ns"] = [MS / 2] * 2
    assert recurrence.check(steps)["kinds"]["idle"][4] == pytest.approx(0, abs=1e-6)
    # A staging gate wait no rule predicts releases it earlier by that width.
    steps[0]["t_gate_ns"] = [0, MS / 4]
    assert recurrence.check(steps)["kinds"]["idle"][4] == pytest.approx(-0.25)
    # The draft copy wait follows no rule, so the charge misses its width.
    steps = mtp_record()
    steps[3]["t_draft_recv_ns"][1] += MS / 2
    steps[3]["t_exit_ns"] += MS / 2
    assert recurrence.check(steps)["total"] == pytest.approx(100 * (34 / 34.5 - 1))
    # Without the split, the token-copy wait sits after the launch and waits
    # for the whole device end:
    # step 1 exits at 12.5, not 13.
    steps = mtp_record()
    for s in steps:
        del s["d_target_ns"]
    report = recurrence.check(steps)
    assert not report["pass"]
    [(step, _, err, *_)] = report["failures"]
    assert (step, err) == (1, pytest.approx(-5))


def _law(prefill_s=0.01, decode=True, out_ms=0.0):
    """A law pricing `record()`'s forwards as it ran: 10 ms of device work
    each and 1 ms per host segment, `out_ms` of `E_out` after each exit
    when set; every vector's hull holds every step."""

    def vector(names, seconds):
        coefficients = {n: seconds * (n == "step") for n in names}
        return {"coefficients": coefficients, "hull": {"facets": [[0] * len(names)]}}

    host = {g: [1e-3, 0, 0] for g in ("E", "W_pre", "W_launch", "W_post")}
    if out_ms:
        host["E_out"] = [out_ms * 1e-3, 0, 0]
    return {
        "provenance": "the wait rules",
        "prefill": vector(PREFILL, prefill_s),
        "decode": {"eager": vector(DECODE, 0.01)} if decode else {},
        "host": {"laws": dict.fromkeys(("decode", "prefill", "middle"), host)},
    }


def _labelled(tmp_path, steps):
    path = tmp_path / "dp0.jsonl"
    lines = (s | {"rung": None, "named": []} for s in steps)
    path.write_text("".join(json.dumps(s) + "\n" for s in lines))
    return label(path)[0]


def test_a_law_that_prices_each_forward_as_it_ran_reproduces_its_period(tmp_path):
    steps = record()
    report = recurrence.priced(steps, _labelled(tmp_path, steps), _law())
    assert set(report["kinds"]) == {"idle", "middle", "prefill", "decode"}
    for n, p10, p50, p90, r1, ok in report["kinds"].values():
        assert (p10, p90) == pytest.approx((0, 0), abs=1e-9) and ok
    assert report["pass"] and report["refused"] == {}
    # A prefill vector 20% dear moves the output prefill's period and the
    # decode that receives its tokens.
    report = recurrence.priced(steps, _labelled(tmp_path, steps), _law(0.012))
    assert not report["kinds"]["prefill"][-1] and not report["pass"]
    assert {r[0] for r in report["failures"]} >= {2, 3}


def test_an_idle_forward_keeps_its_measured_wait_and_the_engine_runs_e_out(tmp_path):
    """Step 3 serves a new request 100 ms after step 2's input pull; every
    forward's exit is followed by 0.5 ms to its pull and 0.5 ms to the entry."""
    steps = record()[:3] + [
        line(
            3,
            [[1, 1, True]],
            True,
            114.5,
            (114.5,) * 2,
            (115.5, 116.5),
            (116.5,) * 2,
            117.5,
        ),
        line(
            4,
            [[1, 2, True]],
            True,
            118.5,
            (118.5,) * 2,
            (119.5, 120.5),
            (120.5, 125.5),
            126.5,
        ),
    ]
    for s in steps:
        s["t_pull_ns"] = s["t_exit_ns"] + MS // 2
    for s in steps[3:]:
        s["batch"][0][0] = 8
    law = _law(out_ms=0.5)
    law["host"]["laws"]["decode"]["E"] = [5e-4, 0, 0]  # one mapping serves every kind
    report = recurrence.priced(steps, _labelled(tmp_path, steps), law)
    assert report["kinds"]["idle"][0] == 2
    for n, p10, p50, p90, r1, ok in report["kinds"].values():
        assert (p10, p90) == pytest.approx((0, 0), abs=1e-9) and ok


def test_warm_and_refused_forwards_are_charged_and_not_graded(tmp_path):
    steps = record()
    report = recurrence.priced(
        steps, _labelled(tmp_path, steps), _law(decode=False), warm={2}
    )
    assert report["refused"] == {"decode": 2}
    assert set(report["kinds"]) == {"idle", "middle"}


def test_the_script_grades_a_record_on_a_law_and_names_its_refusals(tmp_path, capsys):
    steps = record()
    _labelled(tmp_path, steps)
    law = tmp_path / "law.json"
    law.write_text(json.dumps(_law(decode=False)))
    assert recurrence.main(str(tmp_path), law=str(law)) == 1
    assert "decode: 2 steps the law refuses" in capsys.readouterr().out
