# SPDX-License-Identifier: MIT
"""`validate.stalls` finds a rank whose launch its DP group waited on, and
whole-group pauses (#3778), and `runner.overrides.replayed_stall` replays both
on a simulated rank."""

import json
from types import SimpleNamespace

import pytest

from atom.compass.backends.shape import BatchView, RequestShape
from atom.compass.runner import overrides
from atom.compass.validate import stalls

MS = 1_000_000


def _line(step, rank, launch_ms, rows, dummy=False, out=True):
    enter = (1000 + step * 2000) * MS
    return {
        "step": step,
        "dp_rank": rank,
        "is_dummy_run": dummy,
        "produces_output": out,
        "rows": rows,
        "t_enter_ns": enter,
        "t_launch_ns": [enter + MS, enter + MS + launch_ms * MS],
    }


def test_stalls_names_the_rank_its_group_waited_on(tmp_path):
    prefill, decode = [[700, 40000, False]], [[3, 9000, True]]
    record = {
        0: [_line(0, 0, 120, prefill), _line(1, 0, 2, decode)],
        1: [_line(0, 1, 3120, prefill), _line(1, 1, 2, decode)],
        # A dummy forward's slow launch waits on the group; it stalls nothing.
        2: [_line(0, 2, 900, [[1, 1, False]], dummy=True), _line(1, 2, 2, decode)],
    }
    for rank, lines in record.items():
        text = "".join(json.dumps(x) + "\n" for x in lines)
        (tmp_path / f"dp{rank}.jsonl").write_text(text)
    found, first = stalls.stalls(tmp_path)
    assert first == 1.0
    assert [(s["step"], s["rank"]) for s in found] == [(0, 1)]
    assert abs(found[0]["excess_s"] - 3.0) < 1e-9


def _view(rank, decode):
    shape = RequestShape(4 if decode else 700, 9000, decode)
    return BatchView(requests=(shape,), dp_rank=rank)


def test_replayed_stall_charges_each_stall_once_on_its_rank(tmp_path, monkeypatch):
    path = tmp_path / "stalls.json"
    path.write_text(
        json.dumps(
            {
                "stalls": [
                    {"rank": 1, "t_s": 5.0, "seconds": 2.0},
                    {"rank": 1, "t_s": 6.0, "seconds": 0.5},
                    {"rank": 0, "t_s": 1.0, "seconds": 9.0},
                ]
            }
        )
    )
    monkeypatch.setenv(overrides.REPLAY_STALLS, str(path))
    clock = [100.0]
    host = SimpleNamespace(now=lambda: clock[0])
    runner = SimpleNamespace()
    assert overrides.replayed_stall(runner, _view(1, False), host) == 0.0
    clock[0] = 107.0
    # A decode step replays nothing; the next prefill step takes both due ones.
    assert overrides.replayed_stall(runner, _view(1, True), host) == 0.0
    assert overrides.replayed_stall(runner, _view(1, False), host) == 2.5
    assert overrides.replayed_stall(runner, _view(1, False), host) == 0.0
    monkeypatch.delenv(overrides.REPLAY_STALLS)
    assert overrides.replayed_stall(SimpleNamespace(), _view(1, False), host) == 0.0


#: Four DP ranks; a steady step's exit-to-exit cycle, in ms: a 2 ms gap, an
#: 18 ms MTP status wait, then a 2 ms launch.
RANKS, CYCLE_MS = 4, 22


def _group_record(path, n=60, pause=None, windows=True):
    """`n` decode steps on `RANKS` ranks, each waiting for its previous output
    in its MTP status wait (`t_mtp_ns`), as written to `path`/dp<r>.jsonl.

    `pause(step, rank)` gives a step's (gap_ms, wait_ms, busy_ms, output) on a
    rank where it differs from the steady (2, 18, 20, True): the time since its
    previous exit, its MTP wait, its device window and whether it produced
    output. Each rank's line follows its own previous exit."""
    t = {r: 1000 * MS for r in range(RANKS)}
    lines = {r: [] for r in range(RANKS)}
    for step in range(n):
        for r in range(RANKS):
            gap, wait, busy, out = (pause and pause(step, r)) or (2, 18, 20, True)
            enter = t[r] + gap * MS
            launch = enter + wait * MS
            exit_ = launch + 2 * MS
            line = {
                "step": step,
                "dp_rank": r,
                "is_dummy_run": False,
                "produces_output": out,
                "rows": [[1, 9000, True]],
                "t_enter_ns": enter,
                "t_exit_ns": exit_,
                "t_launch_ns": [launch, launch + MS],
            }
            if wait:
                line["t_mtp_ns"] = [enter + MS // 10, launch]
            if windows:
                line["d_busy_ns"] = busy * MS
            lines[r].append(line)
            t[r] = exit_
    for r, mine in lines.items():
        text = "".join(json.dumps(x) + "\n" for x in mine)
        (path / f"dp{r}.jsonl").write_text(text)
    return path


def test_a_pause_every_rank_waits_out_in_its_forward_is_device_side(tmp_path):
    # Step 39's device stopped for 4 s on every rank: each waits in step 40's
    # MTP status wait, released together.
    def pause(step, rank):
        if step == 39:
            return 2, 18, 4020, True
        if step == 40:
            return 2, 4018, 20, True

    found, first, ranks = stalls.group_pauses(_group_record(tmp_path, pause=pause))
    assert ranks == [0, 1, 2, 3] and first == pytest.approx(1.002)
    [p] = found
    assert (p["step"], p["rank"]) == (40, None)
    assert p["seconds"] == pytest.approx(4.0, abs=0.01)
    assert p["waits"] == {r: "mtp" for r in range(RANKS)}
    assert p["device_s"] == pytest.approx(4.0, abs=0.01)


def _no_output_rank(step, rank):
    # Rank 2's step 39 produced no output: it has nothing to wait for in step
    # 40 and waits for its group after it, between its forwards.
    if step == 39:
        return 2, 18, 4020, rank != 2
    if step == 40:
        return (2, 0, 20, True) if rank == 2 else (2, 4018, 20, True)
    if step == 41 and rank == 2:
        return 4020, 18, 20, True


def test_a_rank_with_no_output_to_wait_for_waits_for_its_group_between_steps(
    tmp_path,
):
    for windows in (True, False):
        path = tmp_path / str(windows)
        path.mkdir()
        record = _group_record(path, pause=_no_output_rank, windows=windows)
        [p] = stalls.group_pauses(record)[0]
        # Late to no forward, it held nobody, with device windows or without.
        assert (p["step"], p["rank"]) == (40, None)
        assert p["waits"] == {0: "mtp", 1: "mtp", 2: "between steps", 3: "mtp"}
        assert p["seconds"] == pytest.approx(4.0, abs=0.01)
        assert (p["device_s"] is None) == (not windows)


def _late_rank(step, rank):
    # Rank 1's host stopped for 4 s before step 40: the others launched it on
    # time, and their devices wait for rank 1's at its first collective.
    if step == 40:
        return (4002, 18, 20, True) if rank == 1 else (2, 18, 4020, True)
    if step == 41 and rank != 1:
        return 2, 4018, 20, True


def test_a_rank_late_to_its_forward_is_named(tmp_path):
    [p] = stalls.group_pauses(_group_record(tmp_path, pause=_late_rank))[0]
    assert (p["step"], p["rank"]) == (40, 1)
    assert p["waits"][1] == "late"
    assert p["seconds"] == pytest.approx(4.0, abs=0.01)


def test_an_idle_engine_and_a_short_wait_are_no_pause(tmp_path):
    def pause(step, rank):
        if step == 30:  # every rank idle for 10 s: no request in flight
            return 10000, 18, 20, True
        if step == 31:  # every rank 2.5 s past its step's median
            return 2, 2518, 20, True

    assert stalls.group_pauses(_group_record(tmp_path, pause=pause))[0] == []
    # A simulated record stamps no entry: there is nothing to read a pause from.
    for r in range(RANKS):
        (tmp_path / f"dp{r}.jsonl").write_text(json.dumps({"step": 0}) + "\n")
    assert stalls.group_pauses(tmp_path) == ([], 0.0, [0, 1, 2, 3])


def test_the_replay_file_holds_a_device_side_pause_on_every_rank(tmp_path):
    def pause(step, rank):
        if step == 39:
            return 2, 18, 4020, True
        if step == 40:
            return 2, 4018, 20, True

    record = _group_record(tmp_path, pause=pause)
    late = tmp_path / "late"
    late.mkdir()
    _group_record(late, pause=_late_rank)
    found = []
    for steps in (record, late):
        events, first, ranks = stalls.group_pauses(steps)
        found.append(
            {
                "record": str(steps),
                "first_s": first,
                "ranks": ranks,
                "events": [{"step": 3, "rank": 0, "t_s": first + 0.5, "excess_s": 0.3}],
                "pause_events": events,
            }
        )
    both = stalls.replay_file(found[0])["stalls"]
    assert both[0] == {"rank": 0, "t_s": pytest.approx(0.5), "seconds": 0.3}
    pauses = both[1:]
    assert [s["rank"] for s in pauses] == [0, 1, 2, 3]
    assert {s["step"] for s in pauses} == {40} and all(s["any_step"] for s in pauses)
    # From the record's first entry to step 40's, each 2 ms into its cycle.
    assert pauses[0]["t_s"] == pytest.approx(40 * CYCLE_MS / 1000)
    assert stalls.replay_file(found[0], "pauses")["stalls"] == pauses
    assert stalls.replay_file(found[0], "stalls")["stalls"] == both[:1]
    assert [s["rank"] for s in stalls.replay_file(found[1], "pauses")["stalls"]] == [1]


def test_replayed_stall_charges_a_pause_on_a_step_of_any_kind(tmp_path, monkeypatch):
    path = tmp_path / "stalls.json"
    stalled = [
        {"rank": 1, "t_s": 5.0, "seconds": 2.0},
        {"rank": 1, "t_s": 6.0, "seconds": 4.0, "any_step": True},
        {"rank": 0, "t_s": 6.0, "seconds": 4.0, "any_step": True},
    ]
    path.write_text(json.dumps({"stalls": stalled}))
    monkeypatch.setenv(overrides.REPLAY_STALLS, str(path))
    clock = [100.0]
    host = SimpleNamespace(now=lambda: clock[0])
    runner = SimpleNamespace()
    assert overrides.replayed_stall(runner, _view(1, True), host) == 0.0
    clock[0] = 107.0
    # A decode step takes the due pause and leaves the launch stall for a prefill.
    assert overrides.replayed_stall(runner, _view(1, True), host) == 4.0
    assert overrides.replayed_stall(runner, _view(1, True), host) == 0.0
    assert overrides.replayed_stall(runner, _view(1, False), host) == 2.0
    # An idle rank's dummy forward, a fabricated decode, takes a pause too.
    dummy = BatchView((RequestShape(1, 1, True),), dp_rank=0, dp_dummy=True)
    other = SimpleNamespace()
    clock[0] = 100.0
    assert overrides.replayed_stall(other, dummy, host) == 0.0
    clock[0] = 106.5
    assert overrides.replayed_stall(other, dummy, host) == 4.0


def test_each_pause_names_the_processes_kfd_evicted_over_it(tmp_path):
    pause = {"t_s": 100.0, "span_s": 4.0}
    lines = []
    for t in (1090.0, 1095.0, 1100.0, 1105.0, 1110.0):
        ours = 0 if t <= 1095.0 else 3900 if t >= 1105.0 else 1000
        lines += [
            f"{t}\t11\tstats_7\t{ours}\n",
            f"{t}\t11\tstats_8\t{ours // 2}\n",
            f"{t}\t22\tstats_7\t{250 if t >= 1105.0 else 0}\n",
            f"{t}\t33\tstats_7\t5\n",
        ]
    evicted = tmp_path / "evicted_ms.tsv"
    evicted.write_text("".join(lines))
    stalls.kfd(evicted, [pause], offset=1000.0)
    # From the sample at or before its start (1100) to the one at or after its end.
    assert pause["kfd"] == {
        "evicted_ms": [[11, 2900.0], [22, 250.0]],
        "samples": [1100.0, 1105.0],
    }
    # One the sampler did not run over is said to be so.
    early = {"t_s": 0.0, "span_s": 4.0, "step": 3, "seconds": 4.0, "rank": None}
    early["waits"] = {}
    stalls.kfd(evicted, [early], offset=1000.0)
    assert early["kfd"] == {"evicted_ms": [], "samples": None}
    assert "no sampler sample" in stalls._pause(early, 0.0)
    assert "2 processes (11 2.9 s, 22 0.2 s)" in stalls._pause(
        pause | {"step": 9, "seconds": 4.0, "rank": None, "waits": {}}, 0.0
    )
