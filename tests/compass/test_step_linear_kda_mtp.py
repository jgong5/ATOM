# SPDX-License-Identifier: MIT
"""A KDA model served with an MTP drafter (GLM-5.3-Flash: 34 KDA layers, its
checkpoint's MLA layer 45 drafting three tokens): the law's prefill vector
prices the target's KDA state work beside the draft's `seqs` and `out`, and its
draft law, over a draft layer that holds no KDA state, carries no KDA column.

The record is an MTP record (`d_target_ns`, a trace run's busy labels on
prefill) written from `test_step_linear_kda`'s known law for the target and a
draft that is a share of the target's work less its state work.
"""

import json

import pytest
from test_backend_step_linear_fit import traced
from test_step_linear_kda import CHUNK_S, PREFILL, STATE, _chunks, _kda_record, _law
from test_step_linear_kda import _steps as _kda_steps

from atom.compass.backends import (
    BatchView,
    RequestShape,
    StepLinearBackend,
    StepLinearLaw,
)
from atom.compass.backends.step_linear import KDA, KDA_CHUNK

#: Each forward's draft, as a share of its target's work less the state work.
SHARE = 0.1


def _steps(seed):
    """`test_step_linear_kda`'s steps, each with its draft's seconds."""
    out = []
    for rows, rung, seconds, counts in _kda_steps(seed):
        state = 0.0
        if not rows[0][2]:
            state = sum(c * n for c, n in zip(STATE.values(), counts))
            state += CHUNK_S * _chunks(rows)
        out.append((rows, rung, seconds, counts, SHARE * (seconds - state)))
    return out


def _mtp_kda_record(path, steps):
    """`steps` as an MTP record of a KDA model: each forward's target its price,
    then its draft, every line stating its `kda` counts."""
    _kda_record(path, [(r, g, s + d, c) for r, g, s, c, d in steps])
    lines = [json.loads(x) for x in path.read_text().splitlines()]
    for line, (_, _, seconds, _, _) in zip(lines, steps):
        line["d_target_ns"] = round(seconds * 1e9)
    path.write_text("".join(json.dumps(traced(x)) + "\n" for x in lines))
    return path


def test_a_kda_models_mtp_record_fits_state_columns_on_the_target_alone(tmp_path):
    steps = _steps(0)
    law = _law([_mtp_kda_record(tmp_path / "dp0.jsonl", steps)], kda=True)

    prefill, draft = law["prefill"], law["draft"]
    assert list(prefill["coefficients"]) == [*PREFILL, "seqs", "out", *KDA]
    assert prefill["kda_chunk"] == KDA_CHUNK
    expected = STATE | {"kda_pad_blocks": CHUNK_S}
    assert {k: prefill["coefficients"][k] for k in expected} == pytest.approx(
        expected, rel=1e-4
    )
    assert list(draft["prefill"]["coefficients"]) == [*PREFILL, "seqs", "out"]
    assert "kda_chunk" not in draft["prefill"]

    backend = StepLinearBackend(StepLinearLaw(law))
    for rows, rung, seconds, counts, drafted in steps[:-1]:
        view = BatchView(
            tuple(RequestShape(*r) for r in rows),
            capture_rung=rung,
            kda=counts,
            produces_output=True,
        )
        cost = backend.price(view)
        assert cost.target_seconds == pytest.approx(seconds, rel=1e-5)
        assert cost.draft_seconds == pytest.approx(drafted, rel=1e-4)
        assert not any("kda" in t.name for t in cost.draft_terms)
    priced = backend.price(
        BatchView(
            (RequestShape(100, 100, False),), kda=(1, 0, 0, 0), produces_output=True
        )
    )
    assert "prefill.kda_pad_blocks" in [t.name for t in priced.terms]
