# SPDX-License-Identifier: MIT
"""A trace run's busy time per pass, reduced from planted kineto traces of two
TP ranks; the server driver's profiler around a trace run; and the ranges the
recorder opens for the reducer while a profiler runs."""

import gzip
import json
import os
import signal
import subprocess
import sys

import pytest
import torch
from test_drive_serve import ARGS, TREE, _ready, planted  # noqa: F401, a fixture
from test_parity_device_window import _recorded
from test_parity_record import _config, _drive

from atom.compass.drive import serve
from atom.compass.parity import ENV, MARK, TARGET_DONE, busy

#: One MTP output prefill, in us. The target: K1, K2 queued 2 after K1 and
#: counted whole, the collective K3, queued on rank 0 before K2 ends but
#: started at 50 by rank 1, whose launch returns at 46, and K4 queued 1 after
#: it, and the sampler launched after the draft-KV pass's DK. The draft: DK,
#: then the propose pass's D1, its launch 45 late, and D2 queued 1 after D1.
#: The queued gaps are 2, 1 and 1, so every gap whose kernel was not queued
#: counts 1.
KERNELS = [
    ("k1", 10, 20, 1),
    ("k2", 22, 30, 3),
    ("cross_device_reduce_1stage", 40, 60, 25),
    ("k4", 61, 70, 37),
    ("dk", 72, 75, 81),
    ("sample", 76, 78, 92),
    ("d1", 120, 125, 111),
    ("d2", 126, 130, 113),
]
PROPOSE = MARK + "d_propose_ns"
#: Where the recorder's ranges open, each planted on a rank only where its real
#: recorder opens it (`_opened`).
MARKS = {MARK + "d_draft_kv_ns": (80, 90), TARGET_DONE: (100, 101), PROPOSE: (110, 200)}


def _trace(path, kernels, tokens=100, marks=MARKS):
    """A kineto trace of one forward running `kernels`, (name, start, end,
    launch), each launch call 1 us long, and opening `marks`; written over many
    lines."""
    events = [
        {
            "cat": "user_annotation",
            "name": f"ATOM::forward tokens={tokens}",
            "ts": 0,
            "dur": 1000,
        },
        {"cat": "user_annotation", "name": "ATOM::run_model", "ts": 0, "dur": 95},
        {"cat": "cpu_op", "name": "aten::mm", "ts": 5, "dur": 1},
    ]
    for name in marks:
        a, b = MARKS[name]
        events.append({"cat": "user_annotation", "name": name, "ts": a, "dur": b - a})
    for corr, (name, start, end, launch) in enumerate(kernels):
        args = {"correlation": corr}
        events.append(
            {
                "cat": "cuda_runtime",
                "name": "hipLaunchKernel",
                "ts": launch,
                "dur": 1,
                "args": args,
            }
        )
        events.append(
            {
                "cat": "kernel",
                "name": name,
                "ts": start,
                "dur": end - start,
                "args": args,
            }
        )
    path.parent.mkdir(parents=True)
    with gzip.open(path, "wt") as f:
        json.dump(
            {"schemaVersion": 1, "traceEvents": events, "deviceProperties": []},
            f,
            indent=2,
        )


def _run(
    tmp_path,
    ranks=("rank_0", "rank_1"),
    tokens=100,
    marks=(MARKS, MARKS),
    collective=KERNELS[2][0],
):
    """A trace run's directory: its record's one line and each rank's trace,
    opening its `marks`, rank 1's K1 starting 5 us earlier and its collective,
    named `collective`, at 50."""
    steps = tmp_path / "steps"
    steps.mkdir()
    line = {"step": 7, "rows": [[100, 100, False]], "d_target_ns": 1}
    line.update(d_draft_kv_ns=1, d_propose_ns=1)
    (steps / "dp0.jsonl").write_text(json.dumps(line) + "\n")
    late = {"k1": (5, 20, 1), KERNELS[2][0]: (50, 60, 45)}
    for r, name in enumerate(ranks):
        kernels = [(k, *late[k]) if r and k in late else (k, *x) for k, *x in KERNELS]
        kernels = [(collective if k == KERNELS[2][0] else k, *x) for k, *x in kernels]
        trace = tmp_path / "trace" / name / "m_ts_1.pt.trace.json.gz"
        _trace(trace, kernels, tokens, marks[r])
    return tmp_path


@pytest.mark.parametrize(
    "collective", ["cross_device_reduce_1stage", "atom::mono_step_begin_tp2_debug0"]
)
def test_each_pass_is_its_union_with_queued_gaps_whole_and_the_rest_capped(
    tmp_path, monkeypatch, collective
):
    """The named result. Rank 0's target: union 39, gaps 2 + 1 + 1 + 1, the
    collective (an all-reduce, or the mono runner's peer fence) from rank 1's
    start and its gap capped; rank 1's 5 more. The draft: DK, D1 and D2, their
    union 12 and gaps 1 + 1 + 1; the draft-KV pass DK, the propose pass D1 and
    D2. Each rank opens the ranges its real recorder opens."""
    marks = [set(_opened(tmp_path / f"r{r}", monkeypatch, r)[2]) for r in (0, 1)]
    monkeypatch.setattr(busy, "CHUNK", 64)  # a decode straddles every read
    path = busy.reduce(_run(tmp_path, marks=marks, collective=collective))

    assert path == tmp_path / "steps" / "dp0.busy.jsonl"
    got = [json.loads(x) for x in path.read_text().splitlines()]
    assert got == [
        {
            "step": 7,
            "b_target_ns": 49000,
            "b_draft_ns": 15000,
            "b_propose_ns": 11000,
            "b_draft_kv_ns": 4000,
        }
    ]


@pytest.mark.parametrize(
    "ranks, tokens, why",
    [
        (("dp0_tp0", "dp0_tp1"), 100, "not rank_<r> alone"),
        (("rank_0", "rank_1"), 99, "not the record's 1 lines with their tokens"),
    ],
)
def test_a_dp_trace_or_one_that_misses_the_record_refuses(tmp_path, ranks, tokens, why):
    with pytest.raises(ValueError, match=why):
        busy.reduce(_run(tmp_path, ranks, tokens))


def _opened(tmp_path, monkeypatch, rank):
    """TP rank `rank`'s recorded DSpark runner, the forwards it ran under a
    profiler, and the recorder's ranges they opened."""
    batches = _drive(tmp_path / "simulated", monkeypatch, {0: [0, 1]})[0][0]
    recorded, device = _recorded(batches[-1], drafter="dspark")
    device.tick = 10**12
    monkeypatch.setenv(ENV, str(tmp_path / "real"))
    runner = recorded(rank, _config())
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as p:
        for b in batches:
            runner.forward(b)
    return runner, batches, [e.name for e in p.events() if e.name.startswith(MARK)]


#: The planted server: answers ``/health`` and logs each POST with whether the
#: ready file exists yet.
SERVER = """\
import http.server, json, os, sys
d, argv = os.environ["PLANTED_DIR"], sys.argv[1:]
with open(d + "/argv.json", "w") as f:
    json.dump({"argv": argv, "stack": os.environ.get("ATOM_PROFILER_WITH_STACK")}, f)
os.makedirs(os.environ["ATOM_COMPASS_PARITY_RECORD"])
open(os.environ["ATOM_COMPASS_PARITY_RECORD"] + "/dp0.engine.json", "w").close()

class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()

    def do_POST(self):
        with open(d + "/posts", "a") as f:
            f.write(f"{self.path} {os.path.exists(d + '/out/server.json')}\\n")
        self.send_response(200)
        self.end_headers()

port = int(argv[argv.index("--server-port") + 1])
http.server.HTTPServer(("127.0.0.1", port), Handler).serve_forever()
"""
DRIVE = """\
import sys
from atom.compass.drive import serve
serve.SERVER = (sys.executable, sys.argv[1])
serve.identity = lambda *args: {"identity": "planted"}
serve.KFD = serve.Path(sys.argv[2], "kfd")
serve.serve(sys.argv[3:], sys.argv[2], (7,), trace=True)
"""


def test_a_trace_run_profiles_from_before_its_ready_file_to_its_stop(
    planted,  # noqa: F811
):
    (planted / "server.py").write_text(SERVER)
    out = planted / "out"
    env = dict(os.environ, PYTHONPATH=str(TREE))
    driver = subprocess.Popen(
        [sys.executable, "-c", DRIVE, str(planted / "server.py"), str(out), *ARGS],
        env=env,
    )
    _ready(out, driver, 600, 0.05)
    assert (planted / "posts").read_text() == "/start_profile False\n"
    driver.send_signal(signal.SIGTERM)
    assert driver.wait(timeout=60) == 0

    posts = (planted / "posts").read_text().splitlines()
    assert posts == ["/start_profile False", "/stop_profile True"]
    seen = json.loads((planted / "argv.json").read_text())
    at = seen["argv"].index(serve.PROFILER)
    assert seen["argv"][at + 1] == str(out / "trace") and seen["stack"] == "0"


@pytest.mark.parametrize("rank", [0, 1])
def test_only_a_profiled_forward_marks_its_target_end_and_each_drafter_pass(
    tmp_path, monkeypatch, rank
):
    """On every TP rank, though only rank 0 records."""
    runner, batches, names = _opened(tmp_path, monkeypatch, rank)
    outputs = sum(b.produces_output() for b in batches)
    assert names.count(TARGET_DONE) == names.count(PROPOSE) == outputs > 0
    assert names.count(MARK + "d_draft_kv_ns") == len(batches)

    def opened(name):
        raise AssertionError(f"{name} opened with no profiler running")

    monkeypatch.setattr(torch.profiler, "record_function", opened)
    for b in batches:
        runner.forward(b)
    runner.exit()
