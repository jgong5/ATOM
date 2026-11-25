# SPDX-License-Identifier: MIT
"""`parity.parks`: a recorded engine's scheduler writes a line each time it
first counts a request parked for remote KV, with its sequence id, DP rank and
the engine clock's time; a request counted again before its count is dropped
writes none, one parked again after writes another. The file matches no
glob a record's other readers take.

No device.
"""

import json
from types import SimpleNamespace

import pytest

from atom.compass.parity import parks
from atom.model_engine import scheduler


@pytest.fixture
def restored(monkeypatch):
    """`Scheduler._count_inflight_load` restored after the test."""
    monkeypatch.setattr(
        scheduler.Scheduler,
        "_count_inflight_load",
        scheduler.Scheduler._count_inflight_load,
    )


def test_a_park_writes_a_line_and_a_park_after_its_count_dropped_another(
    tmp_path, restored
):
    ticks = iter(range(100, 200))
    path = tmp_path / parks.FILE.format(rank=3)
    parks.install(tmp_path / "earlier.jsonl", 3, lambda: 0)  # replaced below
    parks.install(path, 3, lambda: next(ticks))
    engine = SimpleNamespace(metrics=None, _num_parked_remote_kv=0)
    seq = SimpleNamespace(id=7)
    count = scheduler.Scheduler._count_inflight_load

    count(engine, seq)  # parked: 100
    count(engine, seq)  # counted already: no line
    scheduler.Scheduler._uncount_inflight_load(engine, seq)
    count(engine, seq)  # parked again: 102

    lines = [json.loads(x) for x in path.read_text().splitlines()]
    assert lines == [
        {"request_id": 7, "dp_rank": 3, "t_park_ns": 100},
        {"request_id": 7, "dp_rank": 3, "t_park_ns": 102},
    ]
    assert list(lines[0]) == list(parks.FIELDS)
    assert engine._num_parked_remote_kv == 1
    assert parks.read(tmp_path) == {3: lines}
    assert (tmp_path / "earlier.jsonl").read_text() == ""
    assert scheduler.Scheduler._count_inflight_load.parks_wrapped is not None


def test_no_record_reader_glob_takes_the_parks_file(tmp_path):
    (tmp_path / parks.FILE.format(rank=0)).write_text("")

    for glob in (
        "dp*[0-9].jsonl",
        "dp*.handoff.jsonl",
        "dp*.engine.json",
        "dp*.pulls.jsonl",
    ):
        assert not list(tmp_path.glob(glob)), glob
