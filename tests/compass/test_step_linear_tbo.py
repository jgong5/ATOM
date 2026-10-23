# SPDX-License-Identifier: MIT
"""A data-parallel prefill step the group runs without two-batch overlap pays
its MoE rows serially: the step linear law's `serial_moe_rows` column.

Two ranks' records are written as a real run before per-step split stamps
wrote them: no `moe_tokens`, no `tbo_tokens`, a host key whose engine states
`enable_tbo`. Labelling rebuilds each step's split by ATOM's own rule; the fit
finds the column from it and the backend prices it on an unsplit step only.
"""

import json

import numpy as np
import pytest
from test_backend_step_linear_fit import _record

from atom.compass.backends import RequestShape, StepLinearBackend, StepLinearLaw
from atom.compass.backends.shape import BatchView
from atom.compass.backends.step_labels import label
from atom.compass.backends.step_linear_fit import fit, label_records

LAW = {
    "step": 0.03,
    "tokens": 5e-5,
    "query_square": 7e-11,
    "query_cached": 1.4e-10,
    "moe_rows": 6e-6,
    "serial_moe_rows": 4e-6,
}


def _features(rows, moe_rows, split):
    q = np.array([r[0] for r in rows])
    n = np.array([r[1] for r in rows])
    serial = 0 if split else moe_rows
    return (1, q.sum(), (q * q).sum(), (q * (n - q)).sum(), moe_rows, serial)


def _old_dp2(tmp_path, steps, enable_tbo=True):
    """Rank 0 and 1's records of `steps`, each (rank 0's rows, rank 1's rows or
    None for a dummy forward, seconds), as a record older than split stamps."""
    paths = []
    for rank in (0, 1):
        own = [(s[rank] or [(1, 1, True)], None, s[2]) for s in steps]
        path = _record(tmp_path / f"dp{rank}.jsonl", own)
        lines = [json.loads(x) for x in path.read_text().splitlines()]
        for line, (r0, r1, _) in zip(lines, steps):
            line["dp_rank"] = rank
            line["is_dummy_run"] = rank == 1 and r1 is None
            line["moe_rows"] = sum(q for q, _, _ in r0 + (r1 or [(1, 1, True)]))
        path.write_text("".join(json.dumps(x) + "\n" for x in lines))
        key = {"engine": {"enable_tbo": enable_tbo}}
        path.with_suffix(".key.json").write_text(json.dumps(key))
        paths.append(path)
    return paths


@pytest.mark.parametrize("enable_tbo", [True, False])
def test_an_older_record_rebuilds_the_split_atom_s_gate_chose(tmp_path, enable_tbo):
    steps = [
        ([(8192, 8192, False)], [(100, 100, False)], 1.0),  # one rank at the bar
        ([(8192, 8192, False)], None, 1.0),  # a dummy peer cannot split
        ([(8000, 8000, False)], [(100, 100, False)], 1.0),  # nobody at the bar
    ]
    _old_dp2(tmp_path, steps, enable_tbo)

    split = ((4096, 50), (4096, 50)) if enable_tbo else None
    got = [s.view.tbo_tokens for s in label(tmp_path / "dp0.jsonl")[0]]
    assert got == [split, None, None]
    got = [s.view.tbo_tokens for s in label(tmp_path / "dp1.jsonl")[0]]
    assert got == [split, None]


def test_the_fit_prices_serial_moe_rows_on_an_unsplit_step_only(tmp_path):
    rng = np.random.default_rng(0)
    steps, views = [], []
    for i in range(240):
        r0 = [
            (q, q + int(rng.integers(0, 30000)), False)
            for q in map(int, rng.integers(2000, 9000, size=int(rng.integers(1, 3))))
        ]
        r1 = None if i % 3 == 0 else [(int(rng.integers(16, 512)), 600, False)]
        moe = sum(q for q, _, _ in r0 + (r1 or [(1, 1, True)]))
        split = r1 is not None and sum(q for q, _, _ in r0) >= 8192
        seconds = float(np.dot(list(LAW.values()), _features(r0, moe, split)))
        steps.append((r0, r1, seconds))
        views.append((r0, moe, split, seconds))
    assert {v[2] for v in views} == {True, False}

    records = _old_dp2(tmp_path, steps)
    law = fit(label_records(records, 2)[0], {"form": "base", "dp": 2})[0]
    assert law["prefill"]["coefficients"] == pytest.approx(LAW, rel=1e-6)

    backend = StepLinearBackend(StepLinearLaw(law), 2)
    for rows, moe, split, seconds in views:
        tbo = ((1, 1), (1, 1)) if split else None
        view = BatchView(
            tuple(RequestShape(*r) for r in rows), moe_rows=moe, tbo_tokens=tbo
        )
        assert backend.price(view).seconds == pytest.approx(seconds, rel=1e-6)
