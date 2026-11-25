# SPDX-License-Identifier: MIT
"""A pipeline with a drafter: ATOM builds the drafter on the last stage alone
(`ModelRunner.__init__`), so a simulated stage builds as one stage does, the
last charges the draft and an earlier one charges none. A config that declares
no acceptance is still refused.

Every stage builds ATOM's aux relay over its target (`PPAuxRelay`), which finds
the decoder layers through the model's `.model` wrappers; a simulated runner's
model is an `UnbuiltModel`. That check imports ATOM's drafter module, which
needs aiter declared on a device, so it owns a process.

The served run is `test_pd_prefill_pp`'s 1P1D at a prefill of PP2 x TP1, under
a constant step stub with a draft intercept, both sides drafting with
MiniMax-M3.1's DSpark draft at a declared acceptance. ATOM admits it only
because the `compass` connector counts as a P/D backend
(`factory.PD_TRANSFER_BACKENDS`): it speculates under PP on a P/D producer
(`Config._check_pp_support`). The target is Qwen3-0.6B's config, since an
M3.1 decode stops on meta in its mono drafter's bind (#4015); the tokenizer is
`test_filler_tokens`'s, and no weight is read, so the run needs no driver, only
atomesh on PATH.
"""

import dataclasses
import functools
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import test_pd_prefill_pp
import test_vertical_slice
from conftest import MockConfig
from test_dp_step_max import prefill, runner
from test_filler_tokens import model_dir  # noqa: F401  a fixture
from test_pd_slice import publish_pd_entries
from test_run_modules_minimax_m3 import M3
from test_speculative_step import _speculative
from test_vertical_slice import CONSTANT, MAX_TOKENS, publish_entry_on_arch

from atom.compass.backends.step_stub import Coefficients, StepStubBackend
from atom.compass.runner import overrides
from atom.compass.runner.overrides import (
    NonAllocatingRunner,
    RunnerRefusal,
    _group_step_seconds,
    _installed_backend,
)

TREE = Path(__file__).resolve().parents[2]
#: 10 s of prefill for the whole stack and a 4 s draft.
STUB = dataclasses.replace(Coefficients.constant(10.0, 1.0), draft_step=4.0)
#: ATOM's decoder-layer lookup on an unbuilt model whose tree wraps its stack.
RESOLVE = """
from atom.compass.capture.driverless import import_aiter_on
import_aiter_on("gfx950")
from torch import nn
from atom.compass.runner.overrides import UnbuiltModel
from atom.spec_decode.drafter import _resolve_decoder_layers
held = UnbuiltModel(nn.Module)
held.tree = nn.Module()
held.tree.model = nn.Module()
held.tree.model.layers = nn.ModuleList([nn.Identity()])
assert _resolve_decoder_layers(held) is held.tree.model.layers
"""

#: The served run's draft price, on every forward.
DRAFT_S = 2.0**-10
#: The served target: 28 layers, 14 a stage.
QWEN = json.loads(Path(__file__).with_name("qwen3_0_6b_config.json").read_text())
#: A one-layer M3.1 DSpark draft over the target's widths, with M3's
#: activation, tapping a layer on each stage, so stage 0 relays its tap to the
#: last; one KV head, which the TP1 decode's draft needs (`M3DSparkAttention`).
DRAFT = (
    QWEN
    | {k: M3[k] for k in ("hidden_act", "swiglu_alpha", "swiglu_limit")}
    | {
        "architectures": ["DSparkMiniMaxDraftModel"],
        "num_hidden_layers": 1,
        "num_key_value_heads": 1,
        "dspark_target_layer_ids": [7, 21],
        "dspark_ffn_hidden_size": 3072,
        "mask_token_id": QWEN["vocab_size"] - 1,
        "dspark_markov_rank": 256,
        "partial_rotary_factor": M3["partial_rotary_factor"],
    }
)
#: Each prefill stage's parity record, by stage.
STAGES = ("dp0", "pp1.dp0")


def _stage(rank: int):
    """Stage `rank` of two, priced for half the stack, its drafts declared."""
    stage = _speculative(runner(StepStubBackend(STUB, stage_share=(0.5, 0.5))))
    stage.config.pipeline_parallel_size = 2
    stage.config.parallel_config.pipeline_parallel_rank = rank
    return stage


@pytest.mark.parametrize("rates", [(0.5,), None], ids=["declared", "undeclared"])
def test_a_stage_builds_its_drafter_on_meta_only_on_a_declared_acceptance(
    monkeypatch, rates
):
    class Built(Exception):
        pass

    def built():
        raise Built

    monkeypatch.setattr(overrides, "drafts_on_meta", built)
    stage = NonAllocatingRunner()
    spec = SimpleNamespace(synthetic_acceptance_rates=rates)
    stage.config = MockConfig(pipeline_parallel_size=2, speculative_config=spec)
    with pytest.raises(RunnerRefusal if rates is None else Built) as raised:
        stage._build_and_load_model(object)
    assert rates is not None or "declares none" in str(raised.value)


def test_the_last_stage_alone_charges_the_draft():
    first, last = _stage(0), _stage(1)
    seconds = [
        _group_step_seconds(s, prefill(8), _installed_backend(s)) for s in (first, last)
    ]
    assert seconds == [5.0, 9.0]
    assert first.refused_commands() == last.refused_commands() == ()


def test_atom_finds_the_decoder_layers_of_an_unbuilt_model():
    env = dict(os.environ, PYTHONPATH=str(TREE))
    done = subprocess.run(
        [sys.executable, "-c", RESOLVE],
        capture_output=True,
        text=True,
        check=False,
        cwd=TREE,
        env=env,
        timeout=600,
    )
    assert done.returncode == 0, done.stderr[-4000:]


@pytest.fixture(scope="module")
def served(model_dir, tmp_path_factory):  # noqa: F811
    """One request through the run: its usage, and each prefill stage's cost
    lines, by stage."""
    if not shutil.which("atomesh"):
        pytest.skip("a run needs atomesh on PATH")
    (model_dir / "config.json").write_text(json.dumps(QWEN))
    draft = tmp_path_factory.mktemp("draft")
    (draft / "config.json").write_text(json.dumps(DRAFT))
    root = tmp_path_factory.mktemp("pp_drafter")
    spec = f"--method dspark --draft-model {draft} --num-speculative-tokens 3"
    with pytest.MonkeyPatch.context() as mp:
        for module in (test_pd_prefill_pp, test_vertical_slice):
            mp.setattr(module, "MODEL", str(model_dir))
        mp.setattr(test_pd_prefill_pp, "CONSTANT", CONSTANT | {"draft_step": DRAFT_S})
        # Each side's readings keyed as on_arch resolves its config on gfx950.
        on_arch = functools.partial(publish_pd_entries, publish=publish_entry_on_arch)
        mp.setattr(test_pd_prefill_pp, "publish_pd_entries", on_arch)
        mp.setenv("ATOM_COMPASS_PARITY_RECORD", str(root / "parity"))
        mp.setenv(
            "SERVER_ARGS",
            f"--enforce-eager --max-model-len 2048 {spec} "
            "--spec-decode-acceptance-rate 0.6",
        )
        ran = test_pd_prefill_pp.run_pd_pp(root / "seed1", "1", 2)
    (usage,) = [json.loads(e[6:])["usage"] for _, e in ran["events"] if '"usage"' in e]
    lines = []
    for name in STAGES:
        path = root / "parity" / "prefill" / f"{name}.cost.jsonl"
        lines.append([json.loads(s) for s in path.read_text().splitlines()[1:]])
    return usage, lines


def test_a_pipelined_prefill_with_a_drafter_serves_and_its_last_stage_charges_the_draft(
    served,
):
    usage, lines = served
    shown = ("step", "device_s", "draft_s", "refused")
    for name, rows in zip(STAGES, lines):
        print(f"\n{name}:")
        for row in rows:
            terms = {t["name"]: t["seconds"] for t in row["terms"]}
            print(f"  {json.dumps({k: row[k] for k in shown})} {terms}")
    assert usage["completion_tokens"] == MAX_TOKENS
    first, last = lines
    assert first and last
    assert all(row["refused"] is None for row in first + last)
    assert {row["draft_s"] for row in first} == {0.0}
    assert {row["draft_s"] for row in last} == {DRAFT_S}
