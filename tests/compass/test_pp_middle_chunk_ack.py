# SPDX-License-Identifier: MIT
"""A pipeline's head retires a middle chunk at the last stage's ack, as it
retires a forward that produces output at its tokens: ATOM's head keeps at
most `pp` forwards in flight and acks each, so its forward `n` waits for the
last stage's ack of forward `n - pp` whether or not that one produced output.
A record's labels and the simulated pipeline's computed timeline both
hold the head to it."""

import dataclasses
import json

import numpy as np
import pytest
from test_backend_step_labels import HOST_ADMIT, LAWS, pp2
from test_pp_stages import ACK, DATA, IPC6, META1, ROUND, TOKENS, WAKE, D, L, T, _woken

from atom.compass.backends.step_labels import host_kind, label_run
from atom.compass.backends.step_stub import Coefficients
from atom.compass.detect.determinism import StepTable
from atom.compass.detect.scenario import stage_timelines


def test_the_heads_e_after_a_middle_chunk_runs_from_the_last_stages_exit_of_it(
    tmp_path,
):
    """On `pp2`'s record with stage 1 at the head's own laws, stage 1 leaves
    some middle chunks after the head has left the forward after them, so the
    head's next forward enters after that exit, its ack, and labels the
    planted `E` from it, its admissions' time included."""
    head, last = pp2(tmp_path, 19, later=1)
    [(samples, _)] = label_run([tmp_path / "dp0.jsonl"], stage=0)
    sample = {s.step: s for s in samples}
    held = [
        n
        for n in range(2, len(head))
        if not head[n - 2]["produces_output"]
        and last[n - 2]["t_exit_ns"] > head[n - 1]["t_exit_ns"]
    ]
    assert held
    for n in held:
        rows = head[n]["rows"]
        x = (1, sum(r[0] for r in rows), len(rows))
        new = [(q, c - q) for q, c, _ in rows[1:]]
        admit = (min(len(new), 1), sum(q for q, _ in new), sum(c for _, c in new))
        law = LAWS[host_kind(sample[n].view)]["E"]
        planted = np.dot(law, x) + np.dot(HOST_ADMIT, admit)
        assert sample[n].host["E"] == pytest.approx(planted, abs=2e-9), n


#: One prompt's three chunks, the first two middle chunks, each `X` of device
#: work per stage: too little for the head's own device to hold its third
#: launch, which waits for the last stage's ack of the first.
X = 2.0**-9
CHUNKS = [
    {"batch": [["r", 4, c]], "rows": [[4, c, False]], "produces_output": c == 12}
    for c in (4, 8, 12)
]
#: Send n's gloo parts complete at `A<n>`: the last stage posts its receive of
#: send 0 at its ``meta``, and send 1 leaves at the head's wake on that. The
#: last stage returns a middle chunk at its metadata's arrival and acks it on
#: ``tokens``, `K<n>`, `IPC6` ahead; forward 2 launches at `K0`, its batch
#: reaching the last stage at `M2`, and ends its device work at `END2`.
A0 = 1 + IPC6 + T
A1 = A0 + WAKE + T
K0, K1 = A0 + IPC6, A1 + IPC6
M2 = K0 + IPC6
A2 = M2 + T
END2 = A2 + T + X
K2 = END2 + IPC6
CHUNK_TARS = [
    ("engine-pp0", 1.0, 1.0, "TAR"),
    ("engine-pp0", 1.0, 1.0, "TAR"),
    ("engine-pp0", A0, A0 + WAKE, "TAR"),
    ("engine-pp0", A1, A1 + _woken(A1 - A0), "TAR"),
    ("engine-pp0", K0, K0, "TAR"),
    ("engine-pp0", A2, A2 + _woken(A2 - A1), "TAR"),
    ("engine-pp1", 1 + IPC6, A0, "TAR"),
    ("engine-pp1", A0 + WAKE + L, A1, "TAR"),
    ("engine-pp1", M2, END2, "TAR"),
]
#: Each channel's arrivals by seq, which is all `stage_timelines` reads of a
#: release row.
CHUNK_RELEASES = {
    ("engine-pp1", META1): (1 + IPC6, 1 + IPC6, M2),
    ("engine-pp1", DATA): (1 + L, A0 + WAKE + L, K0 + L),
    ("engine-pp0", ACK): (A0, A1, A2),
    ("engine-pp0", TOKENS): (K0, K1, K2),
}


def _chunk_timelines(tars=CHUNK_TARS, releases=CHUNK_RELEASES) -> list[str]:
    """`stage_timelines` of `CHUNKS` on two stages of one layer each, against
    a step table of `tars` and `releases`."""
    run = {
        "admission_s": 0.54e-3,
        "ipc_s": IPC6,
        "stream_s": IPC6,
        "pipeline_parallel_size": 2,
        "machine": ROUND,
        "step_stub": dataclasses.asdict(Coefficients.constant(2 * X, 2 * D)),
    }
    table = StepTable(json.dumps(run))
    for row in tars:
        table.record(*row)
    for (lp, channel), ends in releases.items():
        for seq, end in enumerate(ends):
            table.record(lp, end, end, "release", channel, seq)
    return stage_timelines(
        table.text(),
        "engine",
        [CHUNKS, CHUNKS],
        2,
        lambda _: [("gloo", 8), ("rccl", 8)],
    )


def test_the_head_launches_pp_forwards_on_from_a_middle_chunk_at_its_ack():
    assert _chunk_timelines() == []
    # Forward 2 launched on the head's wake, before forward 0's ack.
    woke = CHUNK_TARS[3][2]
    early = ("engine-pp0", woke, woke, "TAR")
    tars = CHUNK_TARS[:4] + [early] + CHUNK_TARS[5:]
    assert _chunk_timelines(tars) == [
        f"engine-pp0: TAR row 4 is {early[1:3]}; the pipeline implies {(K0, K0)}"
    ]
    # The last stage acking only the forward that produces output.
    bare = CHUNK_RELEASES | {("engine-pp0", TOKENS): (K2,)}
    assert _chunk_timelines(releases=bare) == [
        f"{TOKENS} seq 0 is {(K2,)}; the pipeline implies {(K0,)}"
    ]
