# SPDX-License-Identifier: MIT
"""DeepSeek-V4's sparse attention and DeepSeek-R1's MLA, each served twice
under two `PYTHONHASHSEED` by `test_dp_slice`'s four requests and held to every
scenario check: at level 0 under the constant step stub, at level 1 under the
step roofline alone on `ROOFLINE_MACHINE`, an MI355X's spec document (`08`
D120).

Both serve at ``-tp 4 --kv-cache-dtype fp8``: DeepSeek-V4-Flash's config
(`deepseek_v4_flash_config.json`), ATOM's `DeepseekV4ForCausalLM`, its
attention sparse over a sliding window and the compressed KV, at the TP 4
`.github/benchmark/oot_benchmark_models.json` benchmarks it at and the fp8 KV
cache of `recipes/DeepSeek-V4.md`, with the fp8 indexer that recipe names as
the legacy path (``--index-cache-dtype fp8``, #4058); and DeepSeek-R1-0528's in MXFP4
(`deepseek_r1_0528_mxfp4_config.json`, amd/DeepSeek-R1-0528-MXFP4-MTP-MoEFP4's),
ATOM's `DeepseekV2ForCausalLM`, its attention MLA, as
`.github/benchmark/models.json` serves it as DeepSeek-R1-0528-MXFP4.

The runs need a driver, as in `test_vertical_slice`, and a model directory
holding a config and a tokenizer: `ATOM_COMPASS_V4_FLASH_MODEL` and
`ATOM_COMPASS_R1_MODEL`, each its config beside its checkpoint's tokenizer; no
weight is read. The rest run anywhere: each model's meta build at TP 4, walked
as `run.runner` walks it, every module classified and its steps priced by the
step roofline on `ROOFLINE_MACHINE`.
"""

import json
import os
import pathlib

import pytest
import test_dp_slice
import torch
from capabilities import capability
from test_dp_slice import (  # noqa: F401
    MAX_TOKENS,
    device_classes_put_back,
    run_dp_slice,
)
from test_moe_scenario import LEVELS, ONE_RANK, _terms, walked

from atom.compass.detect.scenario import check_run

TP4 = ["-tp", "4", "--kv-cache-dtype", "fp8"]
#: Each form's server flags.
SERVE = {
    # fp4, V4's default indexer, dies in aiter's OPUS config on the host (#4058).
    "deepseek_v4": TP4 + ["--index-cache-dtype", "fp8"],
    "mla": TP4,
}
HERE = pathlib.Path(__file__).parent
#: Each form: its model's variable, its config fixture, and the roofline terms
#: its attention prices as.
FORMS = {
    "deepseek_v4": (
        "ATOM_COMPASS_V4_FLASH_MODEL",
        HERE / "deepseek_v4_flash_config.json",
        {"roofline.sparse_attention", "roofline.mhc"},
    ),
    "mla": (
        "ATOM_COMPASS_R1_MODEL",
        HERE / "deepseek_r1_0528_mxfp4_config.json",
        {"roofline.mla_attention", "roofline.mla_up"},
    ),
}


@pytest.fixture(
    scope="module",
    params=[
        pytest.param((form, level), marks=capability(form, level), id=f"{form}-{level}")
        for form in FORMS
        for level in LEVELS
    ],
)
def runs(request, tmp_path_factory):
    """The form, the level and the form's two runs at that level."""
    form, level = request.param
    name = FORMS[form][0]
    model = os.environ.get(name)
    if not (torch.cuda.is_available() and model):
        pytest.skip(f"a run needs a driver and {name} naming a model directory")
    backend, machine = LEVELS[level]
    root = tmp_path_factory.mktemp(f"{form}-{level}")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(test_dp_slice, "MODEL", model)
        done = [
            run_dp_slice(
                root / f"seed{s}", s, SERVE[form], 1, machine=machine, backend=backend
            )
            for s in ("1", "2")
        ]
    return form, level, done


def test_each_model_meets_every_scenario_check(runs):
    """At level 1, the level-0 case's checks but one: the engine's
    hand-computed timeline, `check_run`'s records, which it computes only for a
    constant step stub, from that stub's prefill and decode costs. There the
    model's attention and its routed experts are among the terms the roofline
    priced."""
    form, level, done = runs
    run = done[0]
    print(
        f"\n{form} at level {level}: {run['summary']['schedule']['seconds_by_method']}"
    )
    print(run["table"])
    records = {"engine": run["out"] / "parity"} if level == 0 else None
    failed = check_run(
        [r["out"] for r in done],
        MAX_TOKENS,
        [line for _, line in run["events"]],
        records,
    )
    assert not failed, "\n".join(failed)
    if level == 1:
        terms = _terms(run)
        assert FORMS[form][2] | {"roofline.moe"} <= terms, sorted(terms)


# --- without a driver ---------------------------------------------------------


def _counts(modules) -> dict:
    """How many modules of each kind and dims, as ``(kind, dims, fmt)``."""
    found = {}
    for _, kind, dims, fmt, _ in modules:
        found[kind, tuple(dims), fmt] = found.get((kind, tuple(dims), fmt), 0) + 1
    return found


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """Each form's meta build at TP 4 with an fp8 KV cache, walked and priced."""
    fields = {"tensor_parallel_size": 4, "kv_cache_dtype": "fp8"}
    return {
        form: walked(
            json.loads(config.read_text()),
            fields,
            ONE_RANK,
            tmp_path_factory.mktemp(form),
        )
        for form, (_, config, _) in FORMS.items()
    }


@pytest.mark.parametrize("form", sorted(FORMS))
def test_each_model_classifies_every_module_and_prices_its_steps(built, form):
    out = built[form]
    print(f"\n{form}: {out['priced']}")
    assert [m[0] for m in out["modules"] if m[1] == "unclassified"] == []
    for name, terms in out["priced"].items():
        assert isinstance(terms, dict), f"{name}: {terms}"
        assert FORMS[form][2] | {"roofline.moe"} <= terms.keys(), (name, terms)
        assert all(s > 0 for s in terms.values()), (name, terms)
    priced = out["priced"]
    assert sum(priced["prefill"].values()) > sum(priced["decode"].values())


def test_v4_flash_at_tp4_is_its_43_sparse_layers_over_fp4_experts(built):
    """Each of the 43 layers attends sparsely at a quarter of the 64 heads of
    512, over a window of 128, at fp8, and mixes its 4 hyper-connections
    around each sub-layer; its MoE holds all 256 routed experts, 6 a token, a
    quarter of each expert's 2048 intermediate rows on a rank, in fp4."""
    found = _counts(built["deepseek_v4"]["modules"])
    attention = {k: n for k, n in found.items() if k[0] == "sparse_attention"}
    assert sum(attention.values()) == 43
    assert {k[1][:3] for k in attention} == {(16, 512, 128)}
    assert {k[2] for k in attention} == {"fp8"}
    assert found["mhc", (4, 4096), "bf16"] == 86
    moe = {k: n for k, n in found.items() if k[0] == "moe"}
    assert sum(moe.values()) == 43
    assert {k[1][:6] for k in moe} == {(256, 256, 6, 512, 4096, 4)}
    assert {k[2] for k in moe} == {"fp4"}


def test_r1_mxfp4_at_tp4_is_its_61_mla_layers_over_fp4_experts(built):
    """Each of the 61 layers runs MLA at a quarter of the 128 heads over the
    512-wide latent and its 64 rope dims, at the fp8 cache, and up-projects
    through `kv_b_proj`; 58 of them route to 256 experts, 8 a token, a quarter
    of each expert's 2048 intermediate rows on a rank, in fp4, with the shared
    expert fused beside them; the first 3 are dense."""
    found = _counts(built["mla"]["modules"])
    mla = {k: n for k, n in found.items() if k[0] == "mla_attention"}
    assert sum(mla.values()) == 61
    assert {(k[1][0], *k[1][2:4], k[2]) for k in mla} == {(32, 512, 64, "fp8")}
    assert sum(n for k, n in found.items() if k[0] == "mla_up") == 61
    moe = {k: n for k, n in found.items() if k[0] == "moe"}
    assert sum(moe.values()) == 58
    assert {(k[1][:6], k[1][-1]) for k in moe} == {((256, 256, 8, 512, 7168, 4), 1)}
    assert {k[2] for k in moe} == {"fp4"}
