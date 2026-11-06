# SPDX-License-Identifier: MIT
"""DeepSeek-V4-Pro's DPA8 EP8 laws for crsuse2-m2m-v3-005/013 price, inside the
hull each rung was fitted on, a step at the median context of each rung either
side of a 1P1D EP8 deployment reaches on a run neither law was fitted on: the
decode law its decode side's rungs, the prefill law the decodes its side runs for
the requests it holds. The prefill law prices its side's rung-2 decode, which none
of its records holds, by the decode law's rung-2 vector, declared.
"""

import json
from pathlib import Path

import pytest
from test_crs005_dpa8_laws import decode_step

from atom.compass.backends import Method, StepLinearBackend, StepLinearLaw
from atom.compass.backends.step_linear import EXTRAPOLATED

LAWS = Path(__file__).resolve().parents[2] / "scripts" / "compass" / "laws"


def law(side: str) -> dict:
    name = f"deepseek-v4-pro-dpa8-ep8-mi355x-crs005-{side}.json"
    return json.loads((LAWS / name).read_text())


#: (rung, the group's MoE tokens, each row's context): per rung, the recorded
#: step nearest the rung's median context, as its priced rank ran it, on each
#: side of a 1P1D deployment that served loads at c=1 to 256 on traces 256 to
#: 392 and that neither law was fitted on.
DECODE_SIDE = [
    (1, 8, (169105,)),
    (2, 9, (158366, 83962)),
    (4, 19, (223813, 151472, 56525)),
    (8, 16, (103161, 150439, 96501, 79890, 209928)),
]
PREFILL_SIDE = [(1, 8, (52298,))]
#: The prefill side's rung-2 decode nearest the median context of the three a 1P1D
#: deployment's c=64 run on traces 0 to 255 ran, as its costliest rank ran it.
PREFILL_RUNG_2 = (2, 10, (168434, 656))


@pytest.mark.parametrize(
    "side,rung,moe_tokens,contexts",
    [("decode", *s) for s in DECODE_SIDE] + [("prefill", *s) for s in PREFILL_SIDE],
)
def test_each_crs005_ep8_law_prices_an_unfitted_runs_median_steps_inside_its_hull(
    side, rung, moe_tokens, contexts
):
    backend = StepLinearBackend(StepLinearLaw(law(side)), dp=8)
    cost = backend.price(decode_step(rung, moe_tokens, contexts))
    assert cost.seconds > 0
    assert all(t.provenance.source != EXTRAPOLATED for t in cost.terms)


def test_the_crs005_ep8_prefill_law_prices_rung_2_by_the_decode_law_s_vector():
    prefill, decode = (
        StepLinearBackend(StepLinearLaw(law(side)), dp=8)
        for side in ("prefill", "decode")
    )
    step = decode_step(*PREFILL_RUNG_2)

    cost, fitted = prefill.price(step), decode.price(step)

    assert {t.provenance.method for t in cost.terms} == {Method.DECLARED}
    assert {t.provenance.method for t in fitted.terms} == {Method.ESTIMATED}
    assert cost.seconds == fitted.seconds > 0
