# SPDX-License-Identifier: MIT
"""A simulated runner opens each of its record files once for a run, and a
worker killed mid-run still leaves every line it recorded.

A spawned worker drives ATOM's `Scheduler` and the simulated runner's own
`forward`, recording to a directory, and counts each file open by path through
an audit hook, which sees an open whatever call makes it and stays in that
worker. It then dies by SIGKILL, closing nothing, and its files are read.
"""

import collections
import json
import multiprocessing
import os
import signal
import sys
from multiprocessing.connection import wait
from types import SimpleNamespace

from conftest import MockConfig
from test_dp_step_max import BLOCK, runner

from atom.compass import parity
from atom.compass.backends.step_stub import StepStubBackend
from atom.compass.runner.overrides import COST_TABLE_VERSION
from atom.model_engine.scheduler import Scheduler
from atom.model_engine.sequence import Sequence
from atom.sampling_params import SamplingParams

STEPS = 1000


def _record_steps(directory, conn):
    """Record `STEPS` simulated steps to `directory`, send how many times each
    path was opened, and die by SIGKILL."""
    opened = collections.Counter()

    def audit(event, args):
        if event == "open" and isinstance(args[0], (str, bytes)):
            opened[os.fsdecode(args[0])] += 1

    sys.addaudithook(audit)
    os.environ[parity.ENV] = str(directory)
    r = runner(StepStubBackend())
    # The engine fields a record's first step states.
    quant = SimpleNamespace(compute_hash=lambda: "q")
    fields = {"kv_cache_dtype": "bf16", "quant_config": quant}
    vars(r.config).update(fields, enforce_eager=True, tensor_parallel_size=1)
    scheduler = Scheduler(
        MockConfig(
            num_kvcache_blocks=128,
            kv_cache_block_size=BLOCK,
            max_model_len=2048,
            parallel_config=SimpleNamespace(data_parallel_rank=0),
        )
    )
    # It cannot finish before its `STEPS`-th forward: each one adds a token.
    params = SamplingParams(max_tokens=STEPS, ignore_eos=True)
    scheduler.add(Sequence(list(range(5, 13)), BLOCK, sampling_params=params))
    for _ in range(STEPS):
        batch, seqs = scheduler.schedule()
        reply = r.forward(batch)
        scheduler.postprocess(list(seqs.values()), reply, batch=batch)
    conn.send(dict(opened))
    os.kill(os.getpid(), signal.SIGKILL)


def test_a_thousand_simulated_steps_open_each_record_file_once_and_leave_every_line(
    tmp_path,
):
    spawn = multiprocessing.get_context("spawn")
    receive, send = spawn.Pipe(duplex=False)
    worker = spawn.Process(target=_record_steps, args=(tmp_path, send))
    worker.start()
    try:
        ready = wait([receive, worker.sentinel], timeout=600)
        assert receive in ready, f"the worker sent no open counts ({worker.exitcode})"
        opened = receive.recv()
        worker.join(60)
    finally:
        worker.kill()
    assert worker.exitcode == -signal.SIGKILL

    record, table = tmp_path / "dp0.jsonl", tmp_path / "dp0.cost.jsonl"
    assert (opened[str(record)], opened[str(table)]) == (1, 1)
    steps = [json.loads(line)["step"] for line in record.read_text().splitlines()]
    version, *costs = [json.loads(line) for line in table.read_text().splitlines()]
    assert version == {"version": COST_TABLE_VERSION}
    assert steps == [line["step"] for line in costs] == list(range(STEPS))
