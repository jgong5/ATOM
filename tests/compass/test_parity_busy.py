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
#: then the propose pass's D1, its launch 45 late, and D2, a collective, queued
#: 1 after D1.
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
    ("allgather_naive", 126, 130, 113),
]
PROPOSE = MARK + "d_propose_ns"
#: Where the recorder's ranges open, each planted on a rank where its real
#: recorder opens it (`_opened`).
MARKS = {MARK + "d_draft_kv_ns": (80, 90), TARGET_DONE: (100, 101), PROPOSE: (110, 200)}
#: The stream every planted kernel runs on, and ATOM's copy stream, which runs
#: only its copies to the host.
STREAM, COPY_STREAM = 3, 10


def _trace(path, kernels, tokens=100, marks=MARKS, copies=()):
    """A kineto trace of one forward running `kernels`, (name, start, end,
    launch), on `STREAM`, and `copies`, (start, end, launch, stream), each
    launch call 1 us long, and opening `marks`. Written over many lines."""
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
    gpu = [("kernel", "hipLaunchKernel", *k, STREAM) for k in kernels]
    gpu += [("gpu_memcpy", "hipMemcpyAsync", "Memcpy DtoH", *c) for c in copies]
    for corr, (cat, call, name, start, end, launch, stream) in enumerate(gpu):
        args = {"correlation": corr}
        events.append(
            {
                "cat": "cuda_runtime",
                "name": call,
                "ts": launch,
                "dur": 1,
                "args": args,
            }
        )
        events.append(
            {
                "cat": cat,
                "name": name,
                "ts": start,
                "dur": end - start,
                "args": dict(args, stream=stream),
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
    slow=(),
    late_draft=False,
    copies=(),
):
    """A trace run's directory: its record's one line and each rank's trace,
    opening its `marks` and running `copies`, rank 1's K1 starting 5 us
    earlier, its collective, named `collective`, at 50, under `late_draft` its
    D2 launched at 127 and starting at 128, and each kernel named in `slow`
    running 7 us longer."""
    steps = tmp_path / "steps"
    steps.mkdir()
    line = {"step": 7, "rows": [[100, 100, False]], "d_target_ns": 1}
    line.update(d_draft_kv_ns=1, d_propose_ns=1)
    (steps / "dp0.jsonl").write_text(json.dumps(line) + "\n")
    late = {"k1": (5, 20, 1), KERNELS[2][0]: (50, 60, 45)}
    if late_draft:
        late["allgather_naive"] = (128, 130, 127)
    for r, name in enumerate(ranks):
        kernels = [(k, *late[k]) if r and k in late else (k, *x) for k, *x in KERNELS]
        kernels = [(k, a, b + 7 * (r and k in slow), q) for k, a, b, q in kernels]
        kernels = [(collective if k == KERNELS[2][0] else k, *x) for k, *x in kernels]
        trace = tmp_path / "trace" / name / "m_ts_1.pt.trace.json.gz"
        _trace(trace, kernels, tokens, marks[r], copies)
    return tmp_path


@pytest.mark.parametrize(
    "collective",
    [
        "cross_device_reduce_1stage",
        "void aiter::ar_ll_gfx9<std::bfloat16_t, 2, 256>(std::bfloat16_t* const*)",
        "void aiter::allReduceQuantFp8<std::bfloat16_t, 2>(aiter::RankData*)",
        "void aiter::reduce_scatter_cross_device_store<std::bfloat16_t, 2>()",
        "atom::mono_step_begin_tp2_debug0",
    ],
)
def test_each_pass_is_its_union_with_queued_gaps_whole_and_the_rest_capped(
    tmp_path, monkeypatch, collective
):
    """The named result. Rank 0's target: union 39, gaps 2 + 1 + 1 + 1, the
    collective (an all-reduce, a low-latency one, a reduce-scatter, or the mono
    runner's peer fence) from rank 1's start and its gap capped, 44; rank 1's
    49, its K1 5 longer. The draft, alike on both ranks: DK, D1 and D2, their
    union 12 and gaps 1 + 1 + 1; the draft-KV pass DK, the propose pass D1 and
    D2. Each rank opens the ranges its real recorder opens under ATOM's
    profiler: every rank, every range."""
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


def test_each_pass_takes_its_largest_rank(tmp_path):
    """Rank 1 ran its K2, DK and D1 7 us longer: each pass is rank 1's, the
    target 53.5 (union 49, gaps 2 + 1 and its cap of 1.5 once), the draft 25,
    its draft-KV pass 11.5 and its propose pass 13.5."""
    path = busy.reduce(_run(tmp_path, slow=("k2", "dk", "d1")))

    got = json.loads(path.read_text())
    assert got["b_target_ns"] == 53500
    assert (got["b_draft_ns"], got["b_propose_ns"], got["b_draft_kv_ns"]) == (
        25000,
        13500,
        11500,
    )


def test_a_drafter_collective_counts_from_its_latest_rank_s_start(tmp_path):
    """Rank 1 launched D2, the propose pass's collective, at 127 and started it
    at 128, after rank 0's D1 ended at 125: on rank 0 its gap is then capped, as
    on rank 1, and both ranks' caps are 1.5. Each rank's propose pass is D1, 5
    and its capped gap, and D2, 2 and its capped gap: 10."""
    got = json.loads(busy.reduce(_run(tmp_path, late_draft=True)).read_text())
    assert (got["b_draft_ns"], got["b_propose_ns"], got["b_draft_kv_ns"]) == (
        14500,
        10000,
        4500,
    )


@pytest.mark.parametrize("stream, propose", [(COPY_STREAM, 11000), (STREAM, 21000)])
def test_a_copy_counts_on_the_kernels_stream_and_not_on_the_copy_stream(
    tmp_path, stream, propose
):
    """The propose pass queues a copy at 127 that runs 131 to 140, past D2. On
    the copy stream, which runs no kernel, it is left out; on the kernels'
    stream it adds 9 and its queued gap of 1 to the propose pass and the
    draft."""
    got = json.loads(
        busy.reduce(_run(tmp_path, copies=[(131, 140, 127, stream)])).read_text()
    )
    assert (got["b_draft_ns"], got["b_propose_ns"]) == (propose + 4000, propose)


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
        runner.profiler = p  # as ATOM's start_profiler keeps it
        for b in batches:
            runner.forward(b)
    runner.profiler = None  # as its stop_profiler leaves it
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
    """The target's end and each drafter pass on every TP rank, though only
    rank 0 records."""
    runner, batches, names = _opened(tmp_path, monkeypatch, rank)
    outputs = sum(b.produces_output() for b in batches)
    assert names.count(TARGET_DONE) == outputs > 0
    assert names.count(PROPOSE) == outputs
    assert names.count(MARK + "d_draft_kv_ns") == len(batches)

    def opened(name):
        raise AssertionError(f"{name} opened with no profiler running")

    monkeypatch.setattr(torch.profiler, "record_function", opened)
    for b in batches:
        runner.forward(b)
    runner.exit()
