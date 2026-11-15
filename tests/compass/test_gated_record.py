# SPDX-License-Identifier: MIT
"""A gated record: `compass record --gate S`, whose workers run each forward
holding prefill tokens behind a spin and event F, and the labels
`step_labels.label_run` gives a DP run's gated prefill steps, which
`step_linear_fit._groups` takes at the group's smallest rank.

Records are written two ways: by `StepRecord` under `StepRecording` on a planted
device whose events complete only once a host sync has run it past them, and
as a DP2 DSpark record's JSON lines planted step by step, its device running a
fixed time behind its host.
"""

import itertools
import json
import re
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from test_drive_identity import GIT
from test_drive_record import planted_gpu  # noqa: F401  the planted_gpu fixture
from test_parity_record import _config

from atom.compass import cli
from atom.compass.backends.shape import BatchView, RequestShape
from atom.compass.backends.step_labels import label, label_run
from atom.compass.backends.step_linear_fit import _groups
from atom.compass.drive import serve
from atom.compass.parity import ENV, GATE, StepRecording, compare
from atom.compass.runner import projection
from atom.utils import forward_context

GATE_NS, WORK = 400, 10  # ns


class _Device:
    """Work queued through `t`; events at or before `now` have completed."""

    def __init__(self):
        self.t, self.now, self.calls = 0, -1, []

    def sync(self):
        self.now = self.t


class _Event:
    def __init__(self, device):
        self.device, self.t = device, device.t

    def query(self):
        return self.t <= self.device.now

    def elapsed_time(self, other):
        return (other.t - self.t) / 1e6

    def synchronize(self):
        self.device.sync()


def _runner(device):
    """A recording runner whose forward makes `ModelRunner.forward`'s calls in
    its order, each queueing `WORK`; a batch's `sync` runs a host sync inside
    its launch or at the forward's end."""
    clock = itertools.count(1)

    class _Processor:
        def recv_async_output(self, handle):
            return handle

        def recv_spec_status(self):
            return None, None

        def prepare_sampled_ids(self, batch):
            return self.recv_async_output([])

        def recv_draft_token_ids(self):
            return []

    class _Base:
        def __init__(self):
            self.rank, self.config = 0, _config()
            self.tokenID_processor = _Processor()
            self.drafter = SimpleNamespace(compute_draft_kv=lambda: None)

        def _gate_staging_reuse(self):
            pass

        def run_model(self, batch):
            device.t += WORK
            if batch.sync == "launch":
                device.sync()

        def forward(self, batch):
            self._gate_staging_reuse()
            self.run_model(batch)
            self.drafter.compute_draft_kv()
            if batch.produces_output():
                self.tokenID_processor.prepare_sampled_ids(batch)
                self.propose_draft_token_ids(batch)
                self.tokenID_processor.recv_draft_token_ids()
            if batch.sync == "end":
                device.sync()
            return "ran"

        def propose_draft_token_ids(self, batch):
            device.t += WORK

        def exit(self):
            pass

    class _Recorded(StepRecording, _Base):
        @staticmethod
        def step_clock_ns():
            return next(clock)

        @staticmethod
        def device_event():
            return _Event(device)

        def calibrate_preload(self):
            device.calls.append("calibrate")

        def preload(self, ns):
            device.calls.append(("preload", ns))
            device.t += ns

    return _Recorded()


MIDDLE = BatchView((RequestShape(64, 64, False),), produces_output=False)
FINAL = BatchView((RequestShape(8, 200, False),), produces_output=True)
DECODE = BatchView((RequestShape(1, 201, True),), capture_rung=1, produces_output=True)


def _batch(view, dummy=False, sync=None):
    rows = view.requests
    return SimpleNamespace(
        view=view,
        sync=sync,
        is_dummy_run=dummy,
        total_tokens_num_prefill=sum(r.query_tokens for r in view.prefill),
        req_ids=list(range(len(rows))),
        num_scheduled_tokens=np.array([r.query_tokens for r in rows]),
        context_lens=np.array([r.context_tokens for r in rows]),
        is_final_chunk=None,
        external_request_ids=[None] * len(rows),
        produces_output=lambda: view.produces_output,
    )


#: Each forward with its line's (gated, qualified): F is read at the draft
#: copy wait where the forward has one, else once it returns.
FORWARDS = [
    (_batch(MIDDLE), (True, True)),
    (_batch(MIDDLE, sync="end"), (True, False)),
    (_batch(FINAL, sync="end"), (True, True)),
    (_batch(FINAL, sync="launch"), (True, False)),
    (_batch(DECODE), (False, None)),
    (_batch(MIDDLE, dummy=True), (False, None)),
]


def _take(directory, monkeypatch, gate):
    """`FORWARDS` recorded into `directory` under `gate`, or none: the lines,
    the device and how many spins each forward queued."""
    monkeypatch.setenv(ENV, str(directory))
    if gate is None:
        monkeypatch.delenv(GATE, raising=False)
    else:
        monkeypatch.setenv(GATE, str(gate))
    device = _Device()
    runner, spins = _runner(device), []
    for batch, _ in FORWARDS:
        before = len(device.calls)
        assert runner.forward(batch) == "ran"
        spins.append(device.calls[before:].count(("preload", GATE_NS)))
    device.sync()
    runner.exit()
    lines = (directory / "dp0.jsonl").read_text().splitlines()
    return [json.loads(x) for x in lines], device, spins


def test_a_gated_record_runs_each_prefill_forward_behind_its_spin_and_states_if_it_qualified(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(projection, "batch_view", lambda batch, mode, run: batch.view)
    # The mode each forward settles, which `run_model` keeps.
    context = SimpleNamespace(context=SimpleNamespace(forward_mode="settled"))
    monkeypatch.setattr(forward_context._forward_context_local, "ctx", context, False)

    plain, device, spins = _take(tmp_path / "plain", monkeypatch, None)
    assert not {"gated", "qualified"} & {k for s in plain for k in s}
    assert device.calls == [] and spins == [0] * len(FORWARDS)

    lines, device, spins = _take(tmp_path / "gated", monkeypatch, GATE_NS)
    assert [(s["gated"], s.get("qualified")) for s in lines] == [
        expected for _, expected in FORWARDS
    ]
    assert device.calls.count("calibrate") == 1
    assert spins == [int(s["gated"]) for s in lines]
    # The spin stays in the device gap before the launch, so the device
    # timeline the record lays out holds it.
    for s in lines[1:]:
        assert (s["d_gap_ns"] >= GATE_NS) == s["gated"]
    # The gate's keys say how a window was timed, not what a step decided.
    report = compare(tmp_path / "gated", tmp_path / "plain")
    assert report["first_divergence"] == {0: None}


#: Host and device time of the planted DP2 record: a step's host forward runs
#: `T` from `step * T`, and its device gap and busy time add up to `T`.
T, BUSY = 1_000_000, 600_000  # ns
TARGET, KV, PROPOSE = 500_000, 30_000, 40_000
OUTPUT, CHUNK, ONE = [[4000, 4000, False]], [[8192, 8192, False]], [[1, 300, True]]


def _line(step, rank, rows, *, output=True, dummy=False, gate=None, launch=20):
    """Rank `rank`'s line at `step`; `gate` is (gated, qualified), or None on an
    ordinary record; `launch` is where its model launch ends after its entry."""
    h, prefill = step * T, not rows[0][2]
    target = TARGET + 1000 * rank
    line = {
        "step": step,
        "dp_rank": rank,
        "is_dummy_run": dummy,
        "produces_output": output,
        "batch": [] if dummy else [[100 * step + rank, q, c] for q, c, _ in rows],
        "named": [],
        "rows": rows,
        "rung": None if prefill else 1,
        "moe_rows": sum(r[0] for r in rows),
        "moe_tokens": sum(r[0] for r in rows),
        "t_enter_ns": h,
        "t_launch_ns": [h + 10, h + launch],
        "t_exit_ns": h + 900_000,
        "d_gap_ns": T - BUSY,
        "d_busy_ns": BUSY,
        "d_target_ns": target,
    }
    if output:
        line["t_recv_ns"] = [h + 20, h + 30]
        wait = h + (700_000 if prefill else 25_000)
        line |= {"t_draft_recv_ns": [wait, wait + 10_000], "d_propose_ns": PROPOSE}
    if output or not prefill:
        line["d_draft_kv_ns"] = KV
    if gate is not None:
        line["gated"] = gate[0] and prefill and not dummy
        if line["gated"]:
            line["qualified"] = gate[1]
    return line


def _dp2(gated=True):
    """Each step's two lines: decodes around four prefill steps on rank 0,
    whose rank 1 runs a dummy, and one on both ranks."""
    g = (True, True) if gated else None
    one, out = {"rows": ONE, "gate": g}, {"rows": OUTPUT, "gate": g}
    dummy = {"rows": [[1, 1, True]], "dummy": True, "gate": g}
    steps = [
        (one, one),
        # Labelled: its dummy launched before the draft copy wait.
        (out, dummy),
        (one, one),
        # Labelled: a middle chunk has no draft copy wait, and its dummy
        # launched before it left the forward.
        ({"rows": CHUNK, "gate": g, "output": False}, dummy | {"launch": 800_000}),
        # Unqualified: F had completed by the draft copy wait.
        (out | {"gate": g and (True, False)}, dummy),
        # Its dummy launched after the draft copy wait began.
        (out, dummy | {"launch": 750_000}),
        # Both ranks prefill: the group's label is the smaller.
        (out, out),
        (one, one),
    ]
    return [
        [_line(i, r, **kw) for r, kw in enumerate(pair)] for i, pair in enumerate(steps)
    ]


def _records(tmp_path, gated=True):
    steps = tmp_path / "steps"
    steps.mkdir(parents=True, exist_ok=True)
    paths = []
    for rank in (0, 1):
        path = steps / f"dp{rank}.jsonl"
        lines = [pair[rank] for pair in _dp2(gated)]
        path.write_text("".join(json.dumps(s) + "\n" for s in lines))
        paths.append(path)
    return paths


def _labelled(run):
    return {(s.rank, s.step): s for samples in run for s in samples}


def test_a_qualified_gated_prefill_step_is_labelled_from_its_window_and_the_rest_are_cut_by_name(
    tmp_path,
):
    records = _records(tmp_path)
    run = [samples for samples, _ in label_run(records)]
    got = _labelled(run)

    prefill = {k: s for k, s in got.items() if s.view.prefill}
    assert {k: s.cut for k, s in prefill.items()} == {
        (0, 1): None,
        (0, 3): None,
        (0, 4): "unqualified",
        (0, 5): "late_dummy",
        (0, 6): None,
        (1, 6): None,
    }
    # The target pass alone: an output prefill's with its draft-KV pass split
    # off, as on any output prefill; no drafter pass is labelled.
    for (rank, step), s in prefill.items():
        if s.cut:
            continue
        output = s.view.produces_output
        target = (TARGET + 1000 * rank + KV * output) * 1e-9
        assert s.device_s == pytest.approx(target, rel=1e-12)
        assert s.draft_kv_s == (pytest.approx(KV * 1e-9) if output else None)
        assert (s.draft_s, s.propose_s) == (None, None)
    # A gated record gives no host label; its decodes keep their draft labels.
    assert all(s.host == {} for s in got.values())
    assert all(s.draft_s is not None for s in got.values() if not s.view.prefill)
    # Labelled alone, each record reads its peer's dummies beside it.
    assert {(0, s.step): s.cut for s in label(records[0])[0]} == {
        (0, s.step): s.cut for s in run[0]
    }

    # The same run recorded without a gate labels no prefill from its windows.
    ordinary = _labelled(s for s, _ in label_run(_records(tmp_path / "plain", False)))
    assert {s.cut for s in ordinary.values() if s.view.prefill} == {"untraced"}
    assert all(s.host for s in ordinary.values())


def test_groups_take_a_gated_step_at_its_smallest_rank_and_count_its_cuts(tmp_path):
    run = [samples for samples, _ in label_run(_records(tmp_path))]
    groups, cut, _ = _groups(run, "base")

    prefill = {
        g[next(iter(g))][3]: g for g in groups if next(iter(g.values()))[0] == "prefill"
    }
    assert sorted(prefill) == [1, 3, 6]
    assert {step: sorted(g) for step, g in prefill.items()} == {
        1: [0],
        3: [0],
        6: [0, 1],
    }
    smallest = (TARGET + KV) * 1e-9
    assert [r[2] for r in prefill[6].values()] == [pytest.approx(smallest)] * 2
    assert (cut["unqualified"], cut["late_dummy"]) == (1, 1)
    # Its rows count `seqs` and `out`, as every prefill row of a drafting run
    # does, though it carries no draft label.
    for step, g in prefill.items():
        output = step != 3
        for counts in (r[1] for r in g.values()):
            assert counts[-2:] == (1 if output else 0, int(output))


@pytest.mark.parametrize(
    "ours, atoms, refusal",
    [
        (["--gate", "1"], [], "--gate gates each DP rank's one worker"),
        (["--gate", "1"], ["-dp", "2", "-tp", "2"], "--gate gates each DP rank"),
        (
            ["--gate", "1"],
            ["--enable-dp-attention", "-tp", "2", "--pipeline-parallel-size=2"],
            "--gate gates each DP rank",
        ),
        (["--gate", "0"], ["--enable-dp-attention", "-tp", "2"], "--gate 0.0: a gate"),
        (
            ["--gate", "1", "--trace"],
            ["--enable-dp-attention", "-tp", "2"],
            "--trace reduces each TP",
        ),
    ],
)
def test_a_gate_off_dp_or_over_more_than_one_worker_refuses_before_a_server_starts(
    tmp_path, monkeypatch, ours, atoms, refusal
):
    monkeypatch.setattr(serve, "SERVER", (sys.executable, "-c", "pass"))
    out = tmp_path / "out"
    argv = ["record", "--gpu", "7", "--out", str(out), *ours, "--", "--model", "m"]

    with pytest.raises(SystemExit, match=f"^compass record: {re.escape(refusal)}"):
        cli.main([*argv, *atoms])
    assert not out.exists()


#: Put before the planted server: the gate its environment holds.
SEES = """\
import os
open(os.environ["PLANTED_SEEN"], "w").write(repr(os.environ.get("ATOM_COMPASS_PARITY_GATE")))
"""


@pytest.mark.skipif(GIT is None, reason="needs git")
def test_a_gated_record_s_server_and_run_json_hold_the_gate_in_nanoseconds(
    tmp_path, planted_gpu, monkeypatch  # noqa: F811
):
    server = tmp_path / "server.py"
    server.write_text(SEES + server.read_text())
    monkeypatch.setenv("PLANTED_SEEN", str(tmp_path / "seen"))
    out = tmp_path / "out"
    cli.main(
        f"record --gpu 7 --gate 0.25 --out {out} -- --model {planted_gpu} "
        "--enable-dp-attention -tp 2".split()
    )

    assert (tmp_path / "seen").read_text() == repr("250000000")
    assert json.loads((out / "run.json").read_text())["env"][GATE] == "250000000"
