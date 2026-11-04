# SPDX-License-Identifier: MIT
"""A recorded real engine writes its pull, put and handoff stamps once it leaves
its loop, or on its SIGTERM, never from inside the loop.

A file write on the loop is time the loop spends: on a record kept on NFS, the
pull stamp's write stalled the loop about 1 ms inside the `E` it opens. ATOM's
busy loop runs on the step-loop tests' fake engine, under `engine_loop` as
`run_engine` runs it.
"""

import json
import signal
from types import SimpleNamespace

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
