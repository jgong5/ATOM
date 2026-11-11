# SPDX-License-Identifier: MIT
"""A trace run whose profiler dropped device rows refuses at its reduction,
and a trace run's server reads a kineto config that keeps every row (#3106).

kineto's ROCm tracer keeps at most ROCTRACER_MAX_EVENTS rows a process and
drops every later one, launch calls and device rows alike, with no mark in the
trace: a rank past it holds its forwards' CPU ranges and none of their kernels.
"""

import gzip
import json
import os
import signal
import subprocess
import sys

import pytest
from test_drive_serve import ARGS, TREE, _ready, planted  # noqa: F401, a fixture
from test_parity_busy import DRIVE, SERVER

from atom.compass.parity import busy

#: One forward's kernels, (name, start, end, launch), in us after its start,
#: each queued before the one before it ends: both gaps count whole.
KERNELS = [
    ("k1", 10, 20, 1),
    ("cross_device_reduce_1stage", 22, 30, 3),
    ("k3", 31, 40, 5),
]
#: The run's forwards, 100 tokens each, one every `SPAN` us.
FORWARDS, SPAN = 4, 1000


def _run(tmp_path, cut=(None, None), lost=()):
    """A two-rank trace run of `FORWARDS` forwards. Rank r's profiler keeps no
    row launched at or after `cut[r]` us, its launch call or its kernel, as
    kineto past its row cap. Each (forward, kernel) in `lost` keeps its launch
    call, which names its kernel, on both ranks, and loses its kernel's row."""
    steps = tmp_path / "steps"
    steps.mkdir()
    line = {"rows": [[100, 100, False]]}
    record = [json.dumps({"step": i, **line}) + "\n" for i in range(FORWARDS)]
    (steps / "dp0.jsonl").write_text("".join(record))
    for r, stop in enumerate(cut):
        events, corr = [], 0
        for i in range(FORWARDS):
            t = i * SPAN
            events.append(
                {
                    "cat": "user_annotation",
                    "name": "ATOM::forward tokens=100",
                    "ts": t,
                    "dur": 100,
                }
            )
            events.append(
                {
                    "cat": "user_annotation",
                    "name": "ATOM::run_model",
                    "ts": t,
                    "dur": 95,
                }
            )
            for name, start, end, launch in KERNELS:
                corr += 1
                if stop is not None and t + launch >= stop:
                    continue
                args = {"correlation": corr}
                events.append(
                    {
                        "cat": "cuda_runtime",
                        "name": "hipLaunchKernel",
                        "ts": t + launch,
                        "dur": 1,
                        "args": dict(args, kernel=name),
                    }
                )
                if (i, name) not in lost:
                    events.append(
                        {
                            "cat": "kernel",
                            "name": name,
                            "ts": t + start,
                            "dur": end - start,
                            "args": dict(args, stream=3),
                        }
                    )
        path = tmp_path / "trace" / f"rank_{r}" / "m_ts_1.pt.trace.json.gz"
        path.parent.mkdir(parents=True)
        with gzip.open(path, "wt") as f:
            json.dump({"schemaVersion": 1, "traceEvents": events}, f, indent=2)
    return tmp_path


def test_a_whole_trace_reduces(tmp_path):
    """Each forward: its kernels' union, 27 us, and its two gaps, 2 and 1, both
    queued."""
    got = busy.reduce(_run(tmp_path)).read_text().splitlines()
    assert [json.loads(x) for x in got] == [
        {"step": i, "b_target_ns": 30000} for i in range(FORWARDS)
    ]


def test_ranks_that_both_stop_recording_refuse_rather_than_price_no_busy_time(tmp_path):
    """Both ranks' profilers fill before forward 2. The ranks agree on forwards
    2 and 3, no collective on either, so their kernels alone would price both
    at no busy time."""
    with pytest.raises(ValueError, match="lacks device rows") as got:
        busy.reduce(_run(tmp_path, cut=(1500, 1500)))
    assert "rank_0 from forward 2, 0.0 s in (2 of 4)" in str(got.value)
    assert "rank_1 from forward 2, 0.0 s in (2 of 4)" in str(got.value)


def test_ranks_that_stop_apart_refuse_naming_each(tmp_path):
    """Rank 0 fills before forward 2, rank 1 before forward 3: the refusal
    names each rank's first forward without device rows, not the collectives
    their forward 2 disagrees on."""
    with pytest.raises(ValueError, match="lacks device rows") as got:
        busy.reduce(_run(tmp_path, cut=(1500, 2500)))
    assert "rank_0 from forward 2, 0.0 s in (2 of 4)" in str(got.value)
    assert "rank_1 from forward 3, 0.0 s in (1 of 4)" in str(got.value)


def test_a_launch_call_whose_kernel_row_is_lost_refuses(tmp_path):
    """Forward 1's K3 lost its row on both ranks while its launch call stayed:
    its kernels alone would price it 9 us short."""
    with pytest.raises(
        ValueError, match="rank_0 from forward 1, 0.0 s in \\(1 of 4\\)"
    ):
        busy.reduce(_run(tmp_path, lost={(1, "k3")}))


#: The planted trace-run server, logging the kineto config it was handed.
KINETO_SERVER = SERVER.replace(
    '"stack": os.environ.get("ATOM_PROFILER_WITH_STACK")',
    '"kineto": os.environ.get("KINETO_CONFIG"), "conf": os.environ.get("KINETO_CONFIG")'
    ' and open(os.environ["KINETO_CONFIG"]).read()',
)


def test_a_trace_run_s_server_keeps_every_profiler_row(planted):  # noqa: F811
    """The server reads a kineto config that sets the row cap to the largest
    the key takes, from the run's directory."""
    assert KINETO_SERVER != SERVER
    (planted / "server.py").write_text(KINETO_SERVER)
    out = planted / "out"
    env = dict(os.environ, PYTHONPATH=str(TREE))
    driver = subprocess.Popen(
        [sys.executable, "-c", DRIVE, str(planted / "server.py"), str(out), *ARGS],
        env=env,
    )
    _ready(out, driver, 600, 0.05)
    driver.send_signal(signal.SIGTERM)
    assert driver.wait(timeout=60) == 0

    seen = json.loads((planted / "argv.json").read_text())
    assert seen["kineto"] == str(out / "kineto.conf")
    assert seen["conf"] == "ROCTRACER_MAX_EVENTS=2147483647\n"
