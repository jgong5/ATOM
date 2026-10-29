# SPDX-License-Identifier: MIT
"""A recorded real run stamps the engine IO threads' queue handoffs; no other run does.

ATOM's busy loop runs on the step-loop tests' fake engine. Each request frame
put after the loop is entered is written with its receipt, put and the busy
loop's get; each output with its put and the output thread's next get.
`scripts/compass/handoff.py` reads them against the step record.
"""

import importlib.util
import json
import queue
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from test_dp_member_step_end import collectives  # noqa: F401
from test_dp_slice import _deployment
from test_step_loop_hooks import STEPS, Engine
from test_vertical_slice import _run_file

from atom.compass import parity
from atom.compass import run as compass_run
from atom.model_engine.engine_core import EngineCore
from atom.model_engine.sequence import SequenceStatus, get_exit_sequence
from atom.utils import clock
from atom.utils.distributed import utils

_spec = importlib.util.spec_from_file_location(
    "handoff", Path(__file__).resolve().parents[2] / "scripts/compass/handoff.py"
)
handoff = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(handoff)


def _frame(*prompts):
    timing = SimpleNamespace(received_at=time.perf_counter())
    return [
        SimpleNamespace(
            id=p,
            status=SequenceStatus.WAITING,
            left=STEPS,
            num_prompt_tokens=p,
            queue_timing=timing,
        )
        for p in prompts
    ]


def test_a_recorded_engine_writes_each_frame_and_output_handoff(tmp_path, monkeypatch):
    monkeypatch.delenv(compass_run.ENV, raising=False)
    monkeypatch.setenv(parity.ENV, str(tmp_path))
    monkeypatch.setattr(clock, "_stamps", None)
    config = _deployment()
    with compass_run.engine(config):
        engine = Engine(SimpleNamespace())
        engine.output_queue = clock.relay_queue(config.parallel_config)
    engine.input_queue.put(_frame(1))  # before the loop: run, not stamped
    enqueue, out = engine.runner_mgr.enqueue, engine.output_queue

    def step(msg):  # the input thread: one frame after step 1, the exit after 4
        enqueue(msg)
        steps = engine.runner_mgr.steps
        if steps in (1, 4):
            engine.input_queue.put(
                _frame(5, 6) if steps == 1 else [get_exit_sequence()]
            )

    engine.runner_mgr.rpc_broadcast_mq.enqueue = step
    engine.utility_handler.push_metrics = lambda: None
    sender = threading.Thread(target=lambda: list(iter(out.get, None)))
    sender.start()
    with compass_run.engine_loop(engine):
        EngineCore.busy_loop(engine)
    out.put(None)
    sender.join(5)

    lines = (tmp_path / "dp0.handoff.jsonl").read_text().splitlines()
    frames = [h for h in map(json.loads, lines) if "t_recv_ns" in h]
    outs = [h for h in map(json.loads, lines) if "rows" in h]
    assert [h["prompt_tokens"] for h in frames] == [11]
    f = frames[0]
    assert f["t_recv_ns"] <= f["t_in_put_ns"] <= f["t_in_get_ns"]
    assert 0 < f["t_pull_ns"] <= f["t_in_get_ns"]
    assert [h["rows"] for h in outs] == [1, 2]  # the frames' finished requests
    assert all(h["t_out_put_ns"] <= h["t_out_sent_ns"] for h in outs)


def test_a_real_run_without_a_record_stamps_nothing(tmp_path, monkeypatch):
    monkeypatch.delenv(compass_run.ENV, raising=False)
    monkeypatch.delenv(parity.ENV, raising=False)
    monkeypatch.setattr(clock, "_stamps", None)
    with compass_run.engine(_deployment()):
        q = clock.relay_queue(_deployment().parallel_config)
    inputs = queue.Queue()
    clock.stamp_input(SimpleNamespace(input_queue=inputs))
    assert (type(q), type(inputs), clock._stamps) == (queue.Queue, queue.Queue, None)


def test_a_simulated_run_with_a_record_stamps_nothing(
    tmp_path, monkeypatch, collectives  # noqa: F811
):
    monkeypatch.setenv(compass_run.ENV, str(_run_file(tmp_path)))
    monkeypatch.setenv(parity.ENV, str(tmp_path / "record"))
    monkeypatch.setattr(clock, "_stamps", None)
    monkeypatch.setattr(compass_run, "_member", None)
    monkeypatch.setattr(clock, "_installed", None)
    monkeypatch.setattr(utils, "LP_OF_RANK", None, raising=False)
    monkeypatch.setattr(compass_run, "connect", lambda *_: None)
    with compass_run.engine(_deployment()):
        q = clock.relay_queue(_deployment().parallel_config)
    inputs = queue.Queue()
    clock.stamp_input(SimpleNamespace(input_queue=inputs))
    assert (type(q), type(inputs), clock._stamps) == (
        clock.RelayQueue,
        queue.Queue,
        None,
    )
    assert not list(tmp_path.glob("**/*.handoff.jsonl"))


def test_the_reader_buckets_an_output_by_its_steps_batch(tmp_path):
    ms = 1_000_000
    # Forward entries 10 ms apart but for an idle 980 ms after the one-row step.
    enters, batches = (10, 20, 1000, 1010, 1020), (2, 1, 2, 2, 2)
    steps = [{"step": 0, "is_dummy_run": True, "batch": [], "t_enter_ns": 0}] + [
        {
            "step": k,
            "is_dummy_run": False,
            "batch": [[r] for r in range(n)],
            "t_enter_ns": t * ms,
            "t_exit_ns": (t + 5) * ms,
        }
        for k, (t, n) in enumerate(zip(enters, batches), 1)
    ]
    stamps = [
        {
            "t_recv_ns": 0,
            "t_in_put_ns": 5_000,
            "t_in_get_ns": 25_000,
            "t_pull_ns": 15_000,
            "prompt_tokens": 100,
        },
        # A finished list of two, put after the one-row step.
        {"t_out_put_ns": 26 * ms, "t_out_sent_ns": 26 * ms + 200_000, "rows": 2},
        # A finished list of one, put after a two-row step.
        {"t_out_put_ns": 16 * ms, "t_out_sent_ns": 16 * ms + 50_000, "rows": 1},
    ]
    for name, lines in (("dp0.jsonl", steps), ("dp0.handoff.jsonl", stamps)):
        (tmp_path / name).write_text("".join(json.dumps(x) + "\n" for x in lines))

    rows = handoff.report(tmp_path)

    assert "| out | 1 | 1 | 200.0 | 200.0 | 10000 | 2.00% |" in rows
    assert "| out | 2 | 1 | 50.0 | 50.0 | 10000 | 0.50% |" in rows
    assert "| queue | 64 | 1 | 20.0 | 20.0 | 10000 | 0.20% |" in rows
    assert "| past pull | 64 | 1 | 10.0 | 10.0 | 10000 | 0.10% |" in rows
    assert rows[-2:] == [
        "every p99 but queue's under 1% of its step: no",
        (
            "p10 at prompt bucket 64: recv+queue 25.0 us, recv+past pull 15.0 us; "
            "ipc_s 61.0 us"
        ),
    ]
