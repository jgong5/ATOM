# SPDX-License-Identifier: MIT
"""DeepSeek-V4-Pro's DPA8 decode law for crsuse2-m2m-v3-005/013 prices the
decode rungs the 1P1D cell's decode side reaches, rung 12 among them, inside
the hull each rung was fitted on; the gpu-45 decode law has no rung-12 vector.
"""

import json
from pathlib import Path

import pytest

from atom.compass.backends import (
    BatchView,
    CostRefused,
    RequestShape,
    StepLinearBackend,
    StepLinearLaw,
)
from atom.compass.backends.step_linear import EXTRAPOLATED

LAWS = Path(__file__).resolve().parents[2] / "scripts/compass/laws"


def backend(host: str, side: str) -> StepLinearBackend:
    law = json.loads(
        (LAWS / f"deepseek-v4-pro-dpa8-mi355x-{host}-{side}.json").read_text()
    )
    return StepLinearBackend(StepLinearLaw(law), dp=8)


#: (rung, the group's MoE tokens, each row's context): per rung, one recorded
#: decode step as its priced rank ran it, the one nearest the rung's median
#: context: rungs 1 to 8 from a 1P1D c=256 decode side on traces 256 to 392,
#: rung 12 from a lone decode server at c=112, since that 1P1D cell decoded at
#: no rung above 8.
CELL_STEPS = [
    (1, 8, (88093,)),
    (2, 10, (215755, 54830)),
    (4, 12, (90563, 180422, 109793)),
    (8, 16, (36186, 200051, 96886, 136088, 51536)),
    (12, 67, (61027, 246640, 83399, 252544, 131920, 220295, 240783, 58382)),
]


def decode_step(rung: int, moe_tokens: int, contexts: tuple[int, ...]) -> BatchView:
    shapes = tuple(RequestShape(1, context, True) for context in contexts)
    return BatchView(
        shapes, capture_rung=rung, moe_rows=8 * rung, moe_tokens=moe_tokens
    )


@pytest.mark.parametrize("rung,moe_tokens,contexts", CELL_STEPS)
def test_the_crs005_decode_law_prices_each_rung_the_cell_reaches_inside_its_hull(
    rung, moe_tokens, contexts
):
    cost = backend("crs005", "decode").price(decode_step(rung, moe_tokens, contexts))
    assert cost.seconds > 0
    assert all(t.provenance.source != EXTRAPOLATED for t in cost.terms)


def test_the_gpu45_decode_law_refuses_the_rung_12_step_the_crs005_law_prices():
    step = decode_step(*next(s for s in CELL_STEPS if s[0] == 12))
    with pytest.raises(CostRefused, match="no decode vector for rung 12"):
        backend("gpu45", "decode").price(step)
    assert backend("crs005", "decode").price(step).seconds > 0
