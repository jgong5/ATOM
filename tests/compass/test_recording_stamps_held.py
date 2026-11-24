# SPDX-License-Identifier: MIT
"""A recorded real engine writes its pull, put and handoff stamps once it leaves
its loop, however it leaves it, or on its SIGTERM, never from inside the loop,
and each held line once.

A file write on the loop is time the loop spends: on a record kept on NFS, the
pull stamp's write stalled the loop about 1 ms inside the `E` it opens. ATOM's
busy loop runs on the step-loop tests' fake engine, under `engine_loop` as
`run_engine` runs it.
"""

import builtins
import json
import signal
import threading
from types import SimpleNamespace

import pytest
from test_step_loop_hooks import STEPS, Engine

from atom.compass import parity
from atom.compass import run as compass_run
from atom.model_engine.engine_core import EngineCore
from atom.model_engine.sequence import SequenceStatus, get_exit_sequence
from atom.utils import clock

CONFIG = SimpleNamespace(
    pipeline_parallel_size=1, parallel_config=SimpleNamespace(data_parallel_rank=0)
)


def _recorded_engine(tmp_path, monkeypatch):
    monkeypatch.delenv(compass_run.ENV, raising=False)
    monkeypatch.setenv(parity.ENV, str(tmp_path))
    monkeypatch.setattr(clock, "_stamps", None)
    with compass_run.engine(CONFIG):
        engine = Engine(SimpleNamespace())
        engine.output_queue = clock.relay_queue(CONFIG.parallel_config)
    engine.utility_handler.push_metrics = lambda: None
    return engine


def _sizes(tmp_path):
    return [
        (tmp_path / f).stat().st_size for f in ("dp0.pulls.jsonl", "dp0.handoff.jsonl")
    ]


def test_the_stamps_reach_the_record_when_the_loop_ends_not_while_it_runs(
    tmp_path, monkeypatch
):
    engine = _recorded_engine(tmp_path, monkeypatch)
    enqueue, seen = engine.runner_mgr.enqueue, []

    def step(msg):  # each forward sees the files as the loop left them
        seen.append(_sizes(tmp_path))
        enqueue(msg)
        if engine.runner_mgr.steps == STEPS:
            engine.input_queue.put([get_exit_sequence()])

    engine.runner_mgr.rpc_broadcast_mq.enqueue = step
    engine.input_queue.put(
        [SimpleNamespace(id=0, status=SequenceStatus.WAITING, left=STEPS)]
    )
    with compass_run.engine_loop(engine):
        EngineCore.busy_loop(engine)

    assert seen == [[0, 0]] * STEPS
    lines = (tmp_path / "dp0.pulls.jsonl").read_text().splitlines()
    stamps = [next(iter(json.loads(line).items())) for line in lines]
    assert [k for k, _ in stamps] == ["t_pull_ns"] * 2 + ["t_put_ns", "t_pull_ns"]
    assert [t for _, t in stamps] == sorted(t for _, t in stamps)
    # Once the loop has left, a stamp is written as it comes.
    clock._stamps.write({"rows": 1})
    assert (tmp_path / "dp0.handoff.jsonl").read_text().splitlines()[-1] == (
        '{"rows": 1}'
    )


def test_a_recorded_engines_sigterm_writes_its_stamps_before_atoms_handler_runs(
    tmp_path, monkeypatch
):
    engine = _recorded_engine(tmp_path, monkeypatch)
    clock._stamps.arm()
    clock._stamps.stamp("t_pull_ns")
    clock._stamps.write({"rows": 2})
    assert _sizes(tmp_path) == [0, 0]
    handled = []
    compass_run._stop_recording(
        engine, lambda *a: handled.append(_sizes(tmp_path)), signal.SIGTERM, None
    )
    assert len(handled) == 1 and all(handled[0])
    assert json.loads((tmp_path / "dp0.handoff.jsonl").read_text()) == {"rows": 2}


def test_a_loop_left_by_an_exception_still_writes_its_stamps(tmp_path, monkeypatch):
    engine = _recorded_engine(tmp_path, monkeypatch)
    enqueue = engine.runner_mgr.enqueue

    def step(msg):  # the second forward fails
        if engine.runner_mgr.steps == 1:
            raise RuntimeError("the second forward fails")
        enqueue(msg)

    engine.runner_mgr.rpc_broadcast_mq.enqueue = step
    engine.input_queue.put(
        [SimpleNamespace(id=0, status=SequenceStatus.WAITING, left=STEPS)]
    )
    with pytest.raises(RuntimeError), compass_run.engine_loop(engine):
        EngineCore.busy_loop(engine)

    lines = (tmp_path / "dp0.pulls.jsonl").read_text().splitlines()
    assert [next(iter(json.loads(line))) for line in lines] == ["t_pull_ns"]


def test_a_sigterm_caught_inside_the_write_writes_each_held_line_once(
    tmp_path, monkeypatch
):
    caught = []

    class File:  # the pulls file's flush is where the SIGTERM lands
        def __init__(self, path, mode):
            self.file = builtins.open(path, mode)  # noqa: SIM115
            self.pulls = "pulls" in str(path)

        def write(self, text):
            self.file.write(text)

        def flush(self):
            self.file.flush()
            if self.pulls and not caught:
                caught.append(1)
                compass_run._stop_recording(
                    engine, lambda *a: None, signal.SIGTERM, None
                )

    monkeypatch.setattr(clock, "open", File, raising=False)
    engine = _recorded_engine(tmp_path, monkeypatch)
    clock._stamps.arm()
    clock._stamps.stamp("t_pull_ns")
    clock._stamps.write({"rows": 2})
    # Its own thread: a lock the handler cannot take again fails the join, not the run.
    writer = threading.Thread(target=clock.write_stamps, daemon=True)
    writer.start()
    writer.join(10)

    assert not writer.is_alive() and caught == [1]
    assert len((tmp_path / "dp0.pulls.jsonl").read_text().splitlines()) == 1
    assert (tmp_path / "dp0.handoff.jsonl").read_text().splitlines() == ['{"rows": 2}']
