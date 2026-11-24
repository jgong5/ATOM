# SPDX-License-Identifier: MIT
"""A trace run under pipeline stages: the recorder's ranges around a stage's
receive and send and its send's device start, and each stage's busy time,
reduced from planted kineto traces against its own record with its handoff in
no pass."""

import gzip
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_parity_record import GRAPHABLE, LADDER, _config, _drive
from test_step_labels_floor import FLOOR, KERNELS, MARKS, OWN, POINTS
from test_step_labels_floor import _run as _floor_run
from test_step_labels_floor import _trace as _floor_trace

from atom.compass.backends.step_labels import Floor
from atom.compass.parity import (
    ENV,
    MARK,
    PP_RECV,
    PP_SEND,
    SENDER,
    StepRecording,
    busy,
    read,
)

#: The stream the planted passes run on, and RCCL's, which runs the handoff's.
STREAM, RCCL = 3, 4
#: Stage 0's forward, in us: K1, then K2 queued 2 after it; its send, a copy of
#: its output and the send's kernel, which runs until stage 1 takes it; and K3,
#: launched after the send range with no kernel of a pass running.
SENDING = (
    [
        ("k1", 10, 20, 1, STREAM),
        ("k2", 22, 30, 3, STREAM),
        ("copy", 42, 44, 41, STREAM),
        ("ncclDevKernel_SendRecv", 45, 90, 43, RCCL),
        ("k3", 52, 55, 51, STREAM),
    ],
    [(PP_SEND, 40, 50)],
)
#: Stage 1's forward, in us: K1, its receive, whose kernel waits on RCCL's
#: stream from 40, where K1 ends, to 60 for stage 0's send, then K2, queued at
#: 32, 1 after the receive's kernel ends, and K3 1 after K2.
RECEIVING = (
    [
        ("k1", 10, 40, 1, STREAM),
        ("ncclDevKernel_SendRecv", 40, 60, 22, RCCL),
        ("k2", 61, 70, 31, STREAM),
        ("k3", 71, 80, 33, STREAM),
    ],
    [(PP_RECV, 21, 30)],
)


def _trace(path, kernels, marks):
    """A kineto trace of one forward of 100 tokens running `kernels`, each
    (name, start, end, launch, stream) in us, each launch call 1 us long, and
    opening `marks`, each (name, start, end)."""
    forward = {"name": "ATOM::forward tokens=100", "ts": 0, "dur": 1000}
    events = [forward, {"name": "ATOM::run_model", "ts": 0, "dur": 95}]
    events += [{"name": name, "ts": a, "dur": b - a} for name, a, b in marks]
    events = [{"cat": "user_annotation"} | e for e in events]
    for corr, (name, start, end, launch, stream) in enumerate(kernels):
        args = {"correlation": corr}
        call = {"cat": "cuda_runtime", "name": "hipLaunchKernel", "ts": launch}
        events.append(call | {"dur": 1, "args": args})
        row = {"cat": "kernel", "name": name, "ts": start, "dur": end - start}
        events.append(row | {"args": args | {"stream": stream}})
    path.parent.mkdir(parents=True)
    with gzip.open(path, "wt") as f:
        json.dump({"traceEvents": events}, f)


def _run(tmp_path, stages, names="pp{k}_rank_{r}"):
    """A trace run's directory: per stage of `stages`, its record's one line,
    of 100 tokens, and each of its ranks' trace, (kernels, marks), in the
    directory `names` gives the stage k and rank r."""
    steps = tmp_path / "steps"
    steps.mkdir()
    line = json.dumps({"step": 7, "rows": [[100, 100, False]]}) + "\n"
    for k, ranks in enumerate(stages):
        (steps / (f"pp{k}.dp0.jsonl" if k else "dp0.jsonl")).write_text(line)
        for r, (kernels, marks) in enumerate(ranks):
            trace = tmp_path / "trace" / names.format(k=k, r=r) / "m.pt.trace.json.gz"
            _trace(trace, kernels, marks)
    return tmp_path


def _busy(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def test_each_stage_s_busy_lines_lie_beside_its_own_record(tmp_path):
    """The named result. Stage 0: K1 and K2, 18, K2's queued gap of 2, and K3's
    gap capped at that 2: 25, its send's kernels in no pass and its range
    ending none. Stage 1: K1, K2 and K3, 48, and their gaps of 1 each: K2's
    gap from K1's end is 21, of which the receive's kernel spans 20."""
    out = _run(tmp_path, [[SENDING], [RECEIVING]])

    assert busy.reduce(out) == out / "steps" / "dp0.busy.jsonl"
    got = {p.name: _busy(p) for p in (out / "steps").glob("*.busy.jsonl")}
    assert got == {
        "dp0.busy.jsonl": [{"step": 7, "b_target_ns": 25000}],
        "pp1.dp0.busy.jsonl": [{"step": 7, "b_target_ns": 50000}],
    }


def test_each_rank_s_receive_spans_its_own_wait(tmp_path):
    """Two ranks of stage 1: rank 1's receive kernel starts at 45, 5 after rank
    0's, and rank 0's K3 runs to 90. Rank 0's target is its 58 of kernels and
    gaps of 1 and 1, 60, not the 65 rank 1's later receive start would leave
    it; rank 1's is 48 and K2's gap less its own receive's 15, 6, and 1: 55."""
    (k1, recv, k2, k3), marks = RECEIVING
    late = [k1, (recv[0], 45, *recv[2:]), k2, k3]
    long = [k1, recv, k2, ("k3", 71, 90, 33, STREAM)]
    out = _run(tmp_path, [[SENDING], [(long, marks), (late, marks)]])
    busy.reduce(out)

    assert _busy(out / "steps" / "pp1.dp0.busy.jsonl") == [
        {"step": 7, "b_target_ns": 60000}
    ]


def test_a_trace_run_mixing_stage_names_refuses(tmp_path):
    out = _run(tmp_path, [[SENDING]], names="rank_{r}")
    _trace(out / "trace" / "pp1_rank_0" / "m.pt.trace.json.gz", *RECEIVING)

    with pytest.raises(ValueError, match="not rank_<r> alone or pp<k>_rank_<r>"):
        busy.reduce(out)


def test_a_later_stage_s_drafter_loses_the_floor_from_its_own_traces(tmp_path):
    """The last of two stages runs the drafter: its drafter parts less the
    profiler's floor are a one-stage run's of the same forward, read from its
    own ``pp1_rank_<r>`` traces, not from stage 0's, which run no drafter."""
    (tmp_path / "one").mkdir()
    one = _floor_run(tmp_path / "one")
    two = tmp_path / "two"
    (two / "steps").mkdir(parents=True)
    (two / "run.json").write_text("{}")
    for name in ("dp0.jsonl", "pp1.dp0.jsonl"):
        (two / "steps" / name).write_text(one.read_text())
    for k, marks in enumerate(({}, MARKS)):
        for r in (0, 1):
            trace = two / "trace" / f"pp{k}_rank_{r}" / "m_ts_1.pt.trace.json.gz"
            _floor_trace(trace, [(100, KERNELS if k else KERNELS[:1], marks)])
    floor = Floor(FLOOR, [p | {"warm_ns": OWN[p["kernels"][0]]} for p in POINTS])

    got = busy.floored(two / "steps" / "pp1.dp0.jsonl", floor)
    assert got == busy.floored(one, floor)


#: How long a stage's settling of its previous send holds its stream, in ns.
HOLD = 50_000


def _middle_stage(tmp_path, monkeypatch, rank):
    """TP rank `rank` of the middle of three stages, recorded, its device one
    us further at each event and `HOLD` further at each settled send, and the
    batches it runs."""
    batches = _drive(tmp_path / "simulated", monkeypatch, {0: [0, 1, 2]})[0][0]
    device = SimpleNamespace(now=0)
    group = SimpleNamespace(is_first_rank=False, recv_tensor_dict=lambda **kw: {})
    parallel_state = SimpleNamespace(get_pp_group=lambda: group)
    monkeypatch.setitem(sys.modules, "aiter.dist.parallel_state", parallel_state)

    def commit(works):
        device.now += HOLD
        works.clear()

    sender = SimpleNamespace(
        commit_pp_send_work=commit,
        async_send_intermediate_tensors=lambda output, owned=(): ["work"],
    )
    monkeypatch.setitem(sys.modules, SENDER, sender)
    config = _config(pipeline_parallel_size=3)
    config.parallel_config.pipeline_parallel_rank = 1

    class _Event:
        def __init__(self):
            device.now += 1000
            self.t = device.now

        def query(self):
            return True

        def elapsed_time(self, other):
            return (other.t - self.t) / 1e6

    class _Base:
        """ATOM's runner: a forward receives its input, settles its last send
        if pending and sends its output, by `SENDER`'s names; a flush settles
        a pending send."""

        def __init__(self, rank, config):
            self.rank, self.config = rank, config
            self.capture_sizes_np, self.enforce_eager = np.array(LADDER), False
            self.running_tokens_per_dp_rank = np.zeros(1, dtype=np.int32)
            self.tbo_decode_graphs = False
            self.attn_metadata_builder = GRAPHABLE
            self._pp_pending_send = []

        def _gate_staging_reuse(self):
            pass

        def run_model(self, *args):
            group.recv_tensor_dict(src=0)
            if self._pp_pending_send:
                sender.commit_pp_send_work(self._pp_pending_send)
            self._pp_pending_send = sender.async_send_intermediate_tensors("out")
            return "logits"

        def forward(self, batch):
            self._gate_staging_reuse()
            self.run_model([], batch)
            return "ran"

        def flush_pp_send(self):
            if self._pp_pending_send:
                sender.commit_pp_send_work(self._pp_pending_send)
            return True

        def exit(self):
            pass

    class _Recorded(StepRecording, _Base):
        clock = iter(range(1, 10**9))

        def step_clock_ns(self):
            return next(self.clock)

        @staticmethod
        def device_event(event=None):
            return event or _Event()

    monkeypatch.setenv(ENV, str(tmp_path / "real"))
    return _Recorded(rank, config), batches


def test_a_sending_stage_s_send_starts_before_it_settles_the_last(
    tmp_path, monkeypatch
):
    """Events 1 us apart: the launch, the receive, the send, the exit. The
    second forward settles the first's send, a hold of 50 us, after its send's
    start; the third follows a flush that settled the second's, outside any
    launch, so it holds no settle and its send's start is its own."""
    runner, batches = _middle_stage(tmp_path, monkeypatch, 0)
    runner.forward(batches[0])
    runner.forward(batches[1])
    runner.flush_pp_send()
    runner.forward(batches[2])
    runner.exit()

    steps = read(tmp_path / "real", 1)[0]
    got = [(s["d_recv_ns"], s["d_send_ns"], s["d_busy_ns"]) for s in steps]
    assert got == [(1000, 2000, 3000), (1000, 2000, 53000), (1000, 2000, 3000)]
    assert ["t_flush_ns" in s for s in steps] == [False, False, True]


@pytest.mark.parametrize("rank", [0, 1])
def test_a_middle_stage_opens_its_handoff_ranges_on_every_rank_under_a_profiler(
    tmp_path, monkeypatch, rank
):
    """Each forward's receive and send, the second's settle of the first's
    send and the flush's settle each open their range, though only rank 0
    records; with no profiler running none opens."""
    runner, batches = _middle_stage(tmp_path, monkeypatch, rank)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as p:
        runner.forward(batches[0])
        runner.forward(batches[1])
        runner.flush_pp_send()
    names = [e.name for e in p.events() if e.name.startswith(MARK)]
    assert (names.count(PP_RECV), names.count(PP_SEND)) == (2, 4)

    def opened(name):
        raise AssertionError(f"{name} opened with no profiler running")

    monkeypatch.setattr(torch.profiler, "record_function", opened)
    runner.forward(batches[2])
    runner.flush_pp_send()
    runner.exit()
