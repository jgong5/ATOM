# SPDX-License-Identifier: MIT
"""Under MTP, R1 releases once the previous output forward's
`forward_done_event` has run, after its target and DSpark's draft-KV pass,
and then the status copy's latency, `L_mtp`: on the gate's measured inputs,
in a law's labels and fit, and in the runner's recurrence."""

import dataclasses
import json
from types import SimpleNamespace

import numpy as np
import pytest
from test_backend_step_linear_fit import _fit, _record, _steps, _view
from test_parity_recurrence import MS, line
from test_step_linear_draft import SHARE, _mtp_record, _output

from atom.compass import run as compass_run
from atom.compass.backends import StepLinearBackend, StepLinearLaw
from atom.compass.backends.step_labels import label, mtp_copy
from atom.compass.backends.step_linear_fit import fit_host, label_records
from atom.compass.parity import recurrence
from atom.compass.runner.overrides import (
    HOST_SEGMENTS,
    MTP_COPY,
    MTP_SEGMENT,
    HostDeviceRecurrence,
)

#: Milliseconds of a decode's draft-KV pass, and of the status copy after it.
KV, COPY = 0.25, 0.25
#: A draft-KV pass's seconds on (1, seqs).
KV_LAW = (2e-5, 1e-6)


def _waited(late=(), n=8):
    """`n` MTP decodes of one request written by the wait rules: 1 ms per host
    segment, 10 ms of target, `KV` of draft-KV pass, then 0.5 of draft. Each
    status wait returns `COPY` after the previous target and draft-KV pass,
    unless its forward entered 20 ms late (`late`) and found it done."""
    steps, exit_, end, done = [], -1.0, None, None
    for i in range(n):
        enter = exit_ + 1 + 20 * (i in late)
        m0 = enter + 1
        m1 = m0 if done is None else max(m0, done + COPY)
        start = m1 if end is None else max(end, m1)
        s = line(
            i, [[2, 150 + i, True]], True, enter, (enter,) * 2, (m1, m1 + 1), None, 0
        )
        s["t_mtp_ns"] = [m0 * MS, m1 * MS]
        s["t_recv_ns"] = [(m1 + 1) * MS] * 2
        s["t_draft_recv_ns"] = [(m1 + 2) * MS] * 2
        exit_ = m1 + 2
        s["t_exit_ns"] = exit_ * MS
        s["d_busy_ns"] = (10.5 + KV) * MS
        s["d_target_ns"], s["d_draft_kv_ns"] = 10 * MS, KV * MS
        if end is not None:
            s["d_gap_ns"] = (start - end) * MS
        done, end = start + 10 + KV, start + 10.5 + KV
        steps.append(s)
    return steps


def test_the_gate_releases_r1_after_the_draft_kv_pass_and_the_status_copy():
    steps = _waited(late={5})
    copy = mtp_copy(steps)
    assert sorted(copy) == list(range(1, 8))
    assert {n: c for n, (c, waited) in copy.items() if waited} == {
        n: pytest.approx(COPY * MS) for n in (1, 2, 3, 4, 6, 7)
    }
    report = recurrence.check(steps)
    assert report["kinds"]["decode"][1:4] == pytest.approx((0, 0, 0), abs=1e-9)
    assert report["kinds"]["decode"][4] == pytest.approx(0, abs=1e-6)
    assert report["pass"] and report["total"] == pytest.approx(0, abs=1e-9)


def test_a_law_labels_and_fits_the_status_copy_where_the_wait_began_first(
    tmp_path, monkeypatch
):
    """The record has decodes only, so the fit meets kinds no forward ran; it
    asks no rank of their empty designs, which numpy 2.3 refuses and 2.4
    answers; `strict` refuses on either."""
    rank = np.linalg.matrix_rank

    def strict(x, *args):
        if not x.size:
            raise ValueError("an empty design has no rank")
        return rank(x, *args)

    monkeypatch.setattr(np.linalg, "matrix_rank", strict)
    path = tmp_path / "dp0.jsonl"
    lines = (s | {"rung": None, "named": []} for s in _waited(late={5}))
    path.write_text("".join(json.dumps(s) + "\n" for s in lines))
    samples = label(path)[0]
    labelled = {s.step: s.host[MTP_COPY] for s in samples if MTP_COPY in s.host}
    assert labelled == {n: pytest.approx(COPY * 1e-3) for n in (1, 2, 3, 4, 6, 7)}
    kept = [s.device_s for s in samples if not s.cut]
    assert kept and kept == [pytest.approx((10 + KV) * 1e-3)] * len(kept)
    law = fit_host(label_records([path])[0])[0]["laws"]["decode"][MTP_COPY]
    assert law[0] == pytest.approx(COPY * 1e-3) and law[1:] == pytest.approx([0, 0])


@pytest.mark.parametrize("copy, exits", [(0.0, [4, 14, 28]), (0.5, [4, 14.5, 28.5])])
def test_the_runner_releases_r1_its_law_s_status_copy_after_the_target(copy, exits):
    """Three decodes, 1 s per host segment, 10 s of target and 4 of draft:
    the second forward's launch waits for the first target's end (12) and
    then the copy."""
    seg = {s: [1, 0, 0] for s in HOST_SEGMENTS}
    seg |= {MTP_SEGMENT: [0, 0, 0], MTP_COPY: [copy, 0, 0]}
    clock = [0.0]
    host = HostDeviceRecurrence({"decode": seg}, (), lambda: clock[0], mtp=True)
    b = SimpleNamespace(
        total_seqs_num_prefill=0,
        total_tokens_num=2,
        total_seqs_num=1,
        produces_output=lambda: True,
    )
    out = []
    for _ in exits:
        clock[0] += host.charge(b, 10, 4)
        out.append(clock[0])
    assert out == pytest.approx(exits)


def test_a_decode_s_draft_kv_pass_fits_the_draft_law_and_prices_as_target(tmp_path):
    """Each decode's device window holds its target, then a draft-KV pass of
    `KV_LAW` on its rows, then its draft: the law prices the pass apart,
    untagged, so the target's end is the end of the pass."""
    steps = _steps(3)
    record = _mtp_record(tmp_path / "dp0.jsonl", steps)
    lines = [json.loads(x) for x in record.read_text().splitlines()]
    for s in lines:
        if s["rows"][0][2]:
            kv = round((KV_LAW[0] + KV_LAW[1] * len(s["rows"])) * 1e9)
            s["d_draft_kv_ns"] = kv
            s["d_busy_ns"] += kv
    record.write_text("".join(json.dumps(x) + "\n" for x in lines))
    law = _fit([record])
    got = law["draft"]["draft_kv"]["coefficients"]
    assert [got["step"], got["seqs"]] == pytest.approx(KV_LAW, rel=1e-4)

    backend = StepLinearBackend(StepLinearLaw(law))
    for rows, rung, seconds in steps[:-1]:
        cost = backend.price(_output(rows, rung))
        kv = [t for t in cost.terms if t.name.startswith("draft.draft_kv.")]
        assert bool(kv) == rows[0][2]
        pass_s = KV_LAW[0] + KV_LAW[1] * len(rows) if kv else 0
        assert cost.target_seconds == pytest.approx(seconds + pass_s, rel=1e-4)
        assert cost.draft_seconds == pytest.approx(0.1 * seconds, rel=1e-4)


def test_a_run_file_whose_host_laws_carry_the_status_copy_loads(tmp_path, monkeypatch):
    seg = {s: [1, 0, 0] for s in (*HOST_SEGMENTS, MTP_SEGMENT, MTP_COPY)}
    path = tmp_path / "run.json"
    path.write_text(json.dumps({"bound_s": 1.0, "host": {"laws": {"decode": seg}}}))
    monkeypatch.setenv(compass_run.ENV, str(path))
    assert compass_run.spec()["host"]["laws"]["decode"][MTP_COPY] == [1, 0, 0]


#: An output prefill's draft-KV tail, and the draft it runs after the event:
#: seconds per step and per sequence, each; plus `REST` after its propose pass.
TAIL, AFTER, REST = (6e-5, 2e-5), (5e-4, 4e-5), 1e-5


def test_an_output_prefill_s_draft_kv_pass_prices_as_target_and_a_middle_chunk_s_as_draft(
    tmp_path,
):
    """A prefill's draft-KV pass is `SHARE` of its target, plus `TAIL` on an
    output prefill, which then runs its propose pass of `AFTER` and `REST`;
    every fourth prefill is a middle chunk; a decode runs `KV_LAW`'s pass, then
    a draft of `SHARE`. The law prices an output prefill's pass as target and
    the draft after it apart; a middle chunk's pass stays draft."""
    steps, plan = _steps(3), []
    for i, (rows, _, s) in enumerate(steps):
        n, out = len(rows), i % 8 != 0
        if rows[0][2]:
            plan.append((True, KV_LAW[0] + KV_LAW[1] * n, SHARE * s, 0))
        else:
            tail = (TAIL[0] + TAIL[1] * n) * out
            propose = (AFTER[0] + AFTER[1] * n) * out
            plan.append((out, SHARE * s + tail, (propose + REST) * out, propose))
    path = tmp_path / "dp0.jsonl"
    _record(path, [(r, g, s + kv + d) for (r, g, s), (_, kv, d, _) in zip(steps, plan)])
    lines = [json.loads(x) for x in path.read_text().splitlines()]
    for x, (_, _, s), (out, kv, _, propose) in zip(lines, steps, plan):
        x.update(produces_output=out, d_target_ns=round(s * 1e9))
        x["d_draft_kv_ns"] = round(kv * 1e9)
        if propose:
            x["d_propose_ns"] = round(propose * 1e9)
    path.write_text("".join(json.dumps(x) + "\n" for x in lines))

    samples = {s.step: s for s in label(path)[0]}
    for i in (2, 8):  # an output prefill, a middle chunk
        (_, _, s), (out, kv, draft, _) = steps[i], plan[i]
        got = samples[i]
        assert got.device_s == pytest.approx(s + kv * out)
        assert got.draft_s == pytest.approx(draft if out else kv)
        assert got.draft_kv_s == (pytest.approx(kv) if out else None)

    law = _fit([path])
    assert law["draft"]["draft_kv"]["coefficients"] == pytest.approx(
        dict(zip(("step", "seqs"), KV_LAW)), rel=1e-4
    )
    fitted = law["draft"]["draft_kv"]["fit"]
    decodes = sum(not s.view.prefill for s in samples.values())
    assert (fitted["samples"], fitted["dropped"]) == (decodes, 0)
    backend = StepLinearBackend(StepLinearLaw(law))
    for (rows, rung, s), (out, kv, draft, _) in zip(steps[:-1], plan):
        if rows[0][2]:
            continue
        view = dataclasses.replace(_view(rows, rung), produces_output=out)
        cost = backend.price(view)
        assert cost.target_seconds == pytest.approx(s + kv * out, rel=1e-4)
        assert cost.draft_seconds == pytest.approx(draft if out else kv, rel=1e-4)
