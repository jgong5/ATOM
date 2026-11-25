# SPDX-License-Identifier: MIT
"""DeepSeek-V4-Pro's DPA8 EP8 laws for crsuse2-m2m-v3-005/013 price, inside the
hull each rung was fitted on, a step at the median context of each rung a side
of a 1P1D EP8 deployment reaches on a run that law was not fitted on: the decode
law its decode side's rungs, the prefill law the decodes its side runs for the
requests it holds. The decode law prices inside its hull too the steps of that
run that a decode law fitted without a deployment serving c=64 then c=256 priced
outside it. The prefill law prices its side's rung-2 decode, which none of its
records holds, by the decode law's rung-2 vector, declared.
"""

from pathlib import Path

import pytest
from test_crs005_dpa8_laws import decode_step

from atom.compass.backends import Method, StepLinearBackend, StepLinearLaw
from atom.compass.backends.json_file import read_json
from atom.compass.backends.step_linear import EXTRAPOLATED

LAWS = Path(__file__).resolve().parents[2] / "scripts" / "compass" / "laws"


def law(side: str) -> dict:
    name = f"deepseek-v4-pro-dpa8-ep8-mi355x-crs005-{side}.json.xz"
    return read_json(LAWS / name)


#: (rung, the group's MoE tokens, each row's context): per rung, the recorded
#: step nearest the rung's median context, as its priced rank ran it: on the
#: decode side of a 1P1D deployment that served c=64 for 600 s and then c=256
#: for 900 s on traces 256 to 392, which the decode law was not fitted on; on the
#: prefill side of one that served loads at c=1 to 256 on the same traces, which
#: the prefill law was not fitted on.
DECODE_SIDE = [
    (1, 8, (193399,)),
    (2, 11, (94521, 218842)),
    (4, 14, (188820, 54945, 156952)),
    (8, 19, (97183, 207415, 224422, 162798)),
]
PREFILL_SIDE = [(1, 8, (52298,))]
#: (rung, the group's MoE tokens, each row's context, the group's dummies): the
#: decode side's steps at rungs 2 and 4, as its priced rank ran them, nearest the
#: median context of those a decode law fitted without a deployment serving c=64
#: then c=256 priced outside its hull.
OUTSIDE_BEFORE_C64_THEN_C256 = [
    (2, 9, (123085, 153637), 6),
    (4, 12, (101753, 211985, 167368, 193153), 3),
]
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


@pytest.mark.parametrize(
    "rung,moe_tokens,contexts,dummies", OUTSIDE_BEFORE_C64_THEN_C256
)
def test_the_crs005_ep8_decode_law_prices_inside_its_hull_steps_a_fit_without_c64_then_c256_left_outside(
    rung, moe_tokens, contexts, dummies
):
    backend = StepLinearBackend(StepLinearLaw(law("decode")), dp=8)
    cost = backend.price(decode_step(rung, moe_tokens, contexts, dummies))
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
