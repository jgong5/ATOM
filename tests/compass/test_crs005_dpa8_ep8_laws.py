# SPDX-License-Identifier: MIT
"""DeepSeek-V4-Pro's DPA8 EP8 laws for crsuse2-m2m-v3-005/013 price the decode steps
each side of a 1P1D EP8 cell runs, inside the hull each rung was fitted on: the
decode law each rung the decode side reaches, the prefill law the decodes its side
runs for the requests it holds. Each installs at its own side's EP8 engine config
and refuses EP off.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_backend_step_linear import _without_step_stub
from test_memory_readings import META_8B, publish_readings
from test_run_step_grid import deployment

from atom.benchmarks.results.metadata import PERFORMANCE_KEYS
from atom.compass import run as compass_run
from atom.compass.backends import (
    BatchView,
    RequestShape,
    StepLinearBackend,
    StepLinearLaw,
)
from atom.compass.backends.step_linear import EXTRAPOLATED, UNIFORM_ROUTING

COMPASS = Path(__file__).resolve().parents[2] / "scripts" / "compass"
CRS005 = json.loads((COMPASS / "machines" / "mi355x-crs005.json").read_text())


def law(side: str) -> dict:
    name = f"deepseek-v4-pro-dpa8-ep8-mi355x-crs005-{side}.json"
    return json.loads((COMPASS / "laws" / name).read_text())


#: (rung, each row's context, the group's MoE tokens): per rung, one recorded
#: decode step as its priced rank ran it, the one nearest the rung's median
#: context, from the sides of a 1P1D EP8 c=256 cell on traces 256 to 392.
DECODE_SIDE = [
    (1, (128099,), 8),
    (2, (193704, 125880), 12),
    (4, (236165, 122272, 114699), 15),
    (8, (91591, 183944, 225200, 115399, 40338), 19),
]
PREFILL_SIDE = [(1, (210477,), 8)]


def decode_step(rung: int, contexts: tuple[int, ...], moe_tokens: int) -> BatchView:
    shapes = tuple(RequestShape(1, context, True) for context in contexts)
    return BatchView(
        shapes, capture_rung=rung, moe_rows=8 * rung, moe_tokens=moe_tokens
    )


@pytest.mark.parametrize(
    "side,rung,contexts,moe_tokens",
    [("decode", *s) for s in DECODE_SIDE] + [("prefill", *s) for s in PREFILL_SIDE],
)
def test_each_crs005_ep8_law_prices_the_rungs_its_1p1d_side_reaches_inside_its_hull(
    side, rung, contexts, moe_tokens
):
    backend = StepLinearBackend(StepLinearLaw(law(side)), dp=8)
    cost = backend.price(decode_step(rung, contexts, moe_tokens))
    assert cost.seconds > 0
    assert all(t.provenance.source != EXTRAPOLATED for t in cost.terms)


@pytest.mark.parametrize("side", ["prefill", "decode"])
def test_each_crs005_ep8_law_installs_at_its_side_and_refuses_ep_off(
    monkeypatch, tmp_path, side
):
    fitted = law(side)
    engine = fitted["engine_config"]
    for key in PERFORMANCE_KEYS:
        monkeypatch.delenv(key, raising=False)
    for key, value in engine["env"].items():
        monkeypatch.setenv(key, value)
    run = _without_step_stub(tmp_path, step_linear=fitted, machine=CRS005)
    monkeypatch.setenv(compass_run.ENV, str(run))

    def installed(ep):
        r = deployment(
            tmp_path,
            **{k: engine[k] for k in ("kv_cache_block_size", "kv_cache_dtype")},
            quant_config=SimpleNamespace(compute_hash=lambda: engine["quantization"]),
            parallel_config=SimpleNamespace(
                data_parallel_size=8, data_parallel_rank=0, pipeline_parallel_rank=0
            ),
            capture_sizes=engine["ladder"],
            max_num_seqs=512,
            max_num_batched_tokens=16384,
            enable_dp_attention=True,
            enable_expert_parallel=ep,
            enable_tbo=engine["enable_tbo"],
        )
        publish_readings(tmp_path / "store", r.config, CRS005)
        compass_run.runner(r, META_8B)
        return r.compass_backend

    assert installed(True).treatments == (UNIFORM_ROUTING,)
    with pytest.raises(
        ValueError, match="enable_expert_parallel True in the law and False here"
    ):
        installed(False)
