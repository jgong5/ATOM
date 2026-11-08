# SPDX-License-Identifier: MIT
"""A simulated server whose host laws refuse its tokenizer exits on the refusal.

The API server checks its tokenizer against its host entry once its engines
have said READY, before it serves. Its engine is a process of its own that the
server's exit waits for, as ATOM's ``EngineCore-DP<n>`` is: a non-daemon
`multiprocessing.Process`. The refusal stops the run, as a signal would, so the
engine leaves its loop at its ``+inf`` grant and hands in, and the server exits
non-zero with its engine gone, the frontend's hand-in naming the refusal.
"""

import contextlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
from test_served_tokenizer import TERM

from atom.compass import run as compass_run
from tests.compass.held_ports import held_ports

TREE = Path(__file__).resolve().parents[2]
#: Wall seconds the server has to exit once it has checked its tokenizer.
EXIT_S = 30.0
#: The line the server prints with its engine's pid, before its tokenizer check.
ENGINE_LINE = re.compile(r"^engine (\d+)$", re.MULTILINE)
#: A co-hosting frontend over an engine LP in a spawned non-daemon process, which
#: asks for time until its ``+inf`` grant; once the engine has bound, the
#: frontend checks a fast tokenizer against a term fitted on the slow backend.
#: It prints the engine's pid before the check.
SERVER = """\
import math, multiprocessing
from types import SimpleNamespace

from atom.compass import run as compass_run
from atom.utils import clock


def engine(ready):
    rank = SimpleNamespace(data_parallel_size=1, data_parallel_rank=0)
    with compass_run.engine(SimpleNamespace(parallel_config=rank, kv_transfer_config={})):
        pass
    said = {"refused_commands": [], "priced_steps": {}, "predicted_memory": {}}
    workers = SimpleNamespace(call_func=lambda name, wait_out: said[name])
    ready.send(True)
    with compass_run.engine_loop(SimpleNamespace(runner_mgr=workers)):
        while not clock.idle(lambda: math.inf):
            pass


if __name__ == "__main__":
    deployment = SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_size=1, data_parallel_rank=0),
        tensor_parallel_size=1,
        enable_dp_attention=False,
        pipeline_parallel_size=1,
        enable_rapidserve=False,
        runner_qualname=compass_run.ATOM_RUNNER,
        kv_transfer_config={},
    )
    spawn = multiprocessing.get_context("spawn")
    with compass_run.frontend(deployment):
        bound, ready = spawn.Pipe(duplex=False)
        process = spawn.Process(target=engine, args=(ready,), name="EngineCore-DP0")
        process.start()
        bound.recv()
    print("engine", process.pid, flush=True)
    model = SimpleNamespace(hf_config=SimpleNamespace(architectures=["LlamaForCausalLM"]))
    compass_run.tokenizer(SimpleNamespace(is_fast=True), model)
"""


def test_a_server_whose_tokenizer_is_refused_exits_with_its_engine_gone(tmp_path):
    script = tmp_path / "server.py"
    script.write_text(SERVER)
    log = tmp_path / "server.log"
    with held_ports(1) as (clock_port,):
        run_file = tmp_path / "run.json"
        run_file.write_text(
            json.dumps(
                {
                    "clock_endpoint": f"tcp://127.0.0.1:{clock_port}",
                    "bound_s": 100.0,
                    "admission_path": "serving",
                    "ipc_s": 2.0**-10,
                    "stream_s": 2.0**-9,
                    "host": {"tokenizer": TERM | {"backend": "slow"}},
                    "out_dir": str(tmp_path),
                }
            )
        )
        env = dict(os.environ, PYTHONPATH=str(TREE), **{compass_run.ENV: str(run_file)})
        with log.open("w") as sink:
            server = subprocess.Popen(
                [sys.executable, str(script)],
                cwd=TREE,
                env=env,
                stdout=sink,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        try:
            deadline = time.monotonic() + 120
            while not (engine := ENGINE_LINE.search(log.read_text())):
                assert server.poll() is None, log.read_text()
                assert time.monotonic() < deadline, "no engine in 120 wall seconds"
                time.sleep(0.2)
            checked = time.monotonic()
            try:
                code = server.wait(timeout=EXIT_S)
            except subprocess.TimeoutExpired:
                pytest.fail(
                    f"the server was still up {EXIT_S} wall seconds after its "
                    f"tokenizer check:\n{log.read_text()}"
                )
            took = time.monotonic() - checked
        finally:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(server.pid, signal.SIGKILL)
            server.wait()
    out = log.read_text()
    print(f"\nexit {code} {took:.2f} wall seconds after the tokenizer check")
    assert code != 0 and "fitted on the slow backend" in out, out
    pid = int(engine[1])
    assert not Path(f"/proc/{pid}").exists(), f"engine {pid} outlived the server"
    # The engine left its loop at its +inf grant, rather than being killed.
    assert (tmp_path / compass_run.LP_FILE.format("engine")).exists(), out
    frontend = json.loads(
        (tmp_path / compass_run.LP_FILE.format("frontend")).read_text()
    )
    [ended] = frontend["refusals"]
    assert ended.startswith("ended:frontend raised ValueError("), ended
