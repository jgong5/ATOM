# SPDX-License-Identifier: MIT
"""`rowmod_vif` (`COMPASS_FIT_ROWMOD_VIF`): a decode rung adopts `rowmod(k)`
only where the column's variance inflation factor over the steps its fit keeps
is `ROWMOD_VIF` or less. A `rowmod` column the rung's other columns already
hold cuts the MAD by what they leave unfitted, not by the order of the rows;
the vector states the factor at each `k` read under `fit.rowmod_vif`.

Records are written as in `test_backend_step_linear_fit`'s `rowmod` tests: one
rank, every step at rung 8, its seconds its price under `ROWMOD`.
"""

import numpy as np
import pytest
from test_backend_step_linear_fit import ROWMOD, _record, _rowmod_steps, _view

from atom.compass.backends import StepLinearLaw
from atom.compass.backends.step_linear_fit import (
    ROWMOD_VIF,
    ROWMOD_VIF_ENV,
    env_options,
    fit,
    label_records,
)


def _held_steps(seed, n=150):
    """Rung-8 decode steps of 4, 6 or 8 rows, each row within 30% of its
    step's mean context, whose cost reads `rowmod(2)`: half the step's context
    and the little the order of its rows moves, which `context` holds."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        width = int(rng.choice((4, 6, 8)))
        mean = rng.uniform(1000, 60000)
        context = (mean * (1 + 0.3 * rng.uniform(-1, 1, size=width))).astype(int)
        rows = [(1, int(c), True) for c in context]
        seconds = (
            ROWMOD["step"]
            + ROWMOD["context"] * int(context.sum())
            + ROWMOD["graph_padding"] * _view(rows, 8).graph_padding
            + ROWMOD["rowmod"] * max(int(context[r::2].sum()) for r in range(2))
        )
        out.append((rows, 8, seconds))
    return out


def _rung(tmp_path, steps, gated):
    """Rung 8's vector fitted on two records of `steps(seed)`, gated or not."""
    tmp_path.mkdir()
    records = [_record(tmp_path / f"dp0-{s}.jsonl", steps(s)) for s in (20, 21)]
    options = {"form": "base", "dp": 1, **({"rowmod_vif": True} if gated else {})}
    law = fit(label_records(records)[0], options)[0]
    StepLinearLaw(law)
    return law["decode"]["8"]


def test_the_switch_reads_one_and_refuses_other_values():
    assert env_options({ROWMOD_VIF_ENV: "1"}) == {"rowmod_vif": True}
    assert env_options({ROWMOD_VIF_ENV: "0"}) == {}
    with pytest.raises(ValueError, match=ROWMOD_VIF_ENV):
        env_options({ROWMOD_VIF_ENV: "2"})


def test_a_rowmod_column_its_rung_cannot_hold_is_adopted_under_the_gate(tmp_path):
    def steps(seed):
        return _rowmod_steps(seed, 150)

    gated = _rung(tmp_path / "gated", steps, True)
    plain = _rung(tmp_path / "plain", steps, False)

    assert gated["rowmod"] == 4
    assert gated["fit"]["rowmod_vif"]["4"] <= ROWMOD_VIF
    assert "rowmod_vif" not in plain["fit"]
    del gated["fit"]["rowmod_vif"]
    assert gated == plain


def test_a_rowmod_column_its_rung_holds_is_refused_under_the_gate(tmp_path):
    plain = _rung(tmp_path / "plain", _held_steps, False)
    gated = _rung(tmp_path / "gated", _held_steps, True)

    # Without the gate every rule `_rowmod` states passes: the order moves
    # the cost, and shuffling the rows loses what the column cuts.
    assert plain["rowmod"] == 2
    evidence = plain["fit"]["rowmod"]
    assert evidence["shuffled_mad"] > 0.9 * evidence["mad_without"]
    # Yet `context` holds the column: half each step's context.
    assert "rowmod" not in gated
    assert set(gated["coefficients"]) == {"step", "context", "graph_padding"}
    assert gated["fit"]["rowmod_vif"]["2"] > 10 * ROWMOD_VIF
    assert gated["fit"]["mad"] == pytest.approx(evidence["mad_without"])
