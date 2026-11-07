# SPDX-License-Identifier: MIT
"""A forward's draft starts no earlier than its exit: its host queues the
drafter's passes before it exits, and a pass the device reaches first waits
for it. So on a real MTP record a decode whose device reached its `propose`
pass before its host's draft copy wait gives no draft label: its window holds
the device's wait for the host."""

import json
from types import SimpleNamespace

import pytest
from test_dp_step_max import runner
from test_parity_recurrence import MS, line
from test_runner_step_semantics import _drive

from atom.compass.backends import (
    CostBackend,
    CostTerm,
    Granularity,
    Method,
    Provenance,
    StepCost,
)
from atom.compass.backends.step_labels import label_run
from atom.compass.runner import overrides
from atom.compass.runner.overrides import (
    MTP_COPY,
    MTP_SEGMENT,
    HostDeviceRecurrence,
)


class Drafted(CostBackend):
    """Every step's target and draft at stated milliseconds."""

    def __init__(self, target: float, draft: float) -> None:
        said = Provenance(Method.DECLARED, "stated", note="stated in this test; none")
        self.cost = StepCost(
            [CostTerm("target", target * 1e-3, said)],
            draft_terms=[CostTerm("draft", draft * 1e-3, said)] if draft else (),
        )

    @property
    def granularity(self):
        return Granularity.STEP

    def price(self, batch_view):
        return self.cost

    def describe(self):
        return "a stated target and draft"


def _host(post: float, **kw) -> dict:
    """Host laws, milliseconds a forward: 0.5 for `E` and for `W_pre`, 1 to
    launch, `post` after the token copy wait, none for the MTP segments."""
    seg = {"E": 0.5, "W_pre": 0.5, "W_launch": 1, "W_post": post}
    seg |= {MTP_SEGMENT: 0, MTP_COPY: 0}
    return {g: [ms * 1e-3, 0, 0] for g, ms in seg.items()}


@pytest.fixture(scope="module")
def decodes():
    """Three decodes of the driven streak's batches."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(overrides, "_group_step_seconds", lambda *_: 0.0)
        batches = _drive().batches
    return [b for b in batches if not b.total_seqs_num_prefill][:3]


@pytest.mark.parametrize(
    "target, draft, post, exits, ends, targets",
    [
        # Each target ends 1 ms after the launch, the host leaves 4 ms after
        # it: each draft runs from the exit, so the second target, ready at
        # 7 ms, waits until the first draft's end at 8 ms.
        (1, 2, 4, (6, 12, 18), (8, 14, 20), (2, 9, 15)),
        # The host leaves at its launch's end, before each target ends: each
        # draft runs right after its target.
        (3, 2, 0, (2, 5, 10), (6, 11, 16), (4, 9, 14)),
        # No draft: the device ends with the target or the launch.
        (1, 0, 4, (6, 12, 18), (2, 8, 14), (2, 8, 14)),
    ],
)
def test_a_simulated_runner_starts_each_draft_no_earlier_than_its_exit(
    decodes, target, draft, post, exits, ends, targets
):
    backend, clock = Drafted(target, draft), [0.0]
    r = runner(backend)
    r.compass_host = host = HostDeviceRecurrence(
        {"decode": _host(post)}, (), lambda: clock[0], mtp=True
    )
    got = []
    for batch in decodes:
        clock[0] += overrides._group_step_seconds(r, batch, backend)
        got.append((clock[0] * 1e3, host.ends[-1] * 1e3, host.target_end * 1e3))
    assert got == pytest.approx(list(zip(exits, ends, targets)))


def test_under_sync_a_draft_starts_no_earlier_than_its_launch_s_end():
    """The exit waits for the device's end, so the draft runs from the
    launch's end, 2 ms, to 4 ms, and the host leaves 4 ms after that."""
    batch = SimpleNamespace(
        total_seqs_num_prefill=1,
        total_tokens_num=8,
        total_seqs_num=1,
        produces_output=lambda: True,
    )
    host = HostDeviceRecurrence({"prefill": _host(4)}, (), lambda: 0.0, sync=True)
    assert host.charge(batch, 1e-3, 2e-3) == pytest.approx(8e-3)
    assert host.ends[-1] == pytest.approx(4e-3)


def _record(path, posts):
    """MTP decodes of one request, in milliseconds: each enters 1 after the
    last exit, waits for the last target's status 0.25 after it, launches for
    0.125, and leaves its token copy wait 1 later. Its host then queues the
    `propose` pass and enters its draft copy wait `posts[i]` after that, and
    leaves 0.5 after the wait. The device runs 10 of target once the last
    forward is done and the launch is queued, then the pass for 0.5 once the
    host has queued it."""
    lines, exit_, target, end = [], -1.0, None, None
    for i, post in enumerate(posts):
        enter = exit_ + 1
        m1 = enter + 1 if target is None else max(enter + 1, target + 0.25)
        start = m1 if end is None else max(end, m1)
        recv = m1 + 0.125
        d0 = recv + 1 + post
        d1 = d0 if end is None else max(d0, end + 0.25)
        exit_ = d1 + 0.5
        s = line(
            i, [[2, 150 + i, True]], True, enter, (enter,) * 2, (m1, recv), None, exit_
        )
        s["t_mtp_ns"] = [(enter + 1) * MS, m1 * MS]
        s["t_recv_ns"] = [recv * MS, (recv + 1) * MS]
        s["t_draft_recv_ns"] = [d0 * MS, d1 * MS]
        if end is not None:
            s["d_gap_ns"] = (start - end) * MS
        target = start + 10
        end = max(max(target, d0) + 0.5, exit_)
        s["d_busy_ns"], s["d_target_ns"] = (end - start) * MS, 10 * MS
        lines.append(s | {"rung": None, "named": []})
    path.write_text("".join(json.dumps(s) + "\n" for s in lines))
    return path


def test_a_decode_whose_device_reached_its_propose_pass_first_gives_no_draft_label(
    tmp_path,
):
    """The host enters step 3's draft copy wait 0.5 ms after the device
    reached the pass, and step 6's 0.125 ms before it did: step 3 keeps its
    target label and gives no draft label; every other kept step's is the
    pass's 0.5 ms. Under MTP the token copy wait waits for nothing, so a lead
    read off its exits would place step 3's device 0.875 ms late."""
    path = _record(tmp_path / "dp0.jsonl", [0, 0, 0, 9.625, 0, 0, 9, 0])
    [(samples, _)] = label_run([path])
    kept = {s.step: (s.device_s, s.draft_s) for s in samples if s.cut is None}
    assert {3, 6} <= kept.keys()
    assert kept == {
        n: (pytest.approx(10e-3), None if n == 3 else pytest.approx(0.5e-3))
        for n in kept
    }
