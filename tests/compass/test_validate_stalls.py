# SPDX-License-Identifier: MIT
"""`validate.stalls` finds a rank whose launch its DP group waited on, and
`runner.overrides.replayed_stall` replays such stalls on a simulated rank."""

import json
from types import SimpleNamespace

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
