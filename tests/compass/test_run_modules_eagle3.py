# SPDX-License-Identifier: MIT
"""The step roofline lays out an Eagle3 drafter's passes (#3962).

`run.modules.drafter` builds the Eagle3 drafter ATOM maps an EAGLE3 Llama
checkpoint to (`Eagle3LlamaModel`) from its own config, as
`EagleProposer._build_draft_model` does, and `walk` reads it as a serial
drafter: each of its modules runs once a draft token, its `fc` and its aux
chunks' norms once before them, in the first pass
(`EagleProposer._propose`'s `combine_hidden_states`). Its layer norms the
embedding and the carried hidden state in one launch; its embedding is whole
on every rank; its final norm runs on the rows its LM head samples alone
unless the drafter norms its output in the forward (`norm_output`), where it
takes the down projection's all-reduce.

MiniMax-M3's EAGLE3 drafter (`test_draft_key.EAGLE3`, its config.json) is
built over MiniMax-M3 cut to two dense-attention layers, which the roofline
prices (`test_run_modules_minimax_m3`), at TP 4 on
`tests/compass/machines/mi355x-gpu40.json`; an EAGLE 3.1 head with
`fc_norm` and `norm_output` over Qwen3-0.6B at TP 2. Each is built in a
process of its own, as `test_run_modules_mtp` builds its models.
"""

import concurrent.futures
import json
import os
import pickle
import subprocess
import sys

import pytest
from test_dp_step_max import prefill
from test_draft_key import EAGLE3
from test_key_index_cache_dtype import QWEN
from test_run_modules_minimax_m3 import M3
from test_run_modules_mtp import (
    BUILD,
    HERE,
    MI308X,
    MI355X,
    TREE,
    later,
    serial,
    speculative,
    verify,
)

from atom.compass.backends.shape import BatchView, RequestShape, sum_tokens
from atom.compass.backends.step_roofline import UNCLASSIFIED, StepRooflineBackend
from atom.compass.backends.work import (
    allgather,
    allreduce,
    attention,
    elementwise,
    gemm,
    price,
)
from atom.compass.spec import MachineSpec

MACHINE = MachineSpec.from_mapping(json.loads(MI355X.read_text()))
#: ATOM builds its own Eagle3 drafter after the install, of the same names.
OWN = """
import copy
own = copy.copy(config)
own.hf_config = config.speculative_config.draft_model_hf_config
own.kv_cache_dtype = config.speculative_config.draft_kv_dtype
with FactoriesOnMeta():
    type(draft)(own, layer_offset=config.hf_config.num_hidden_layers)
"""
EAGLE3_BUILD = BUILD.replace(
    """with FactoriesOnMeta():  # ATOM builds its own drafter after, of the same names
    type(draft)(config)
""",
    OWN,
)
#: Both of MiniMax-M3's layers attend densely, with a dense MLP.
DENSE = {**M3["sparse_attention_config"], "sparse_attention_freq": [0, 0]}
#: An EAGLE 3.1 Llama head over Qwen3-0.6B's hidden states.
EAGLE31 = EAGLE3 | {
    "hidden_size": 1024,
    "intermediate_size": 3072,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "vocab_size": 151936,
    "draft_vocab_size": 151936,
    "fc_norm": True,
    "norm_output": True,
}
#: Each build: target config, draft config, TP width, machine, drafts, flags.
BUILDS = {
    "m3-eagle3": (
        M3 | {"sparse_attention_config": DENSE},
        EAGLE3,
        4,
        MI355X,
        3,
        ["--kv_cache_dtype", "fp8"],
    ),
    "eagle3.1-tp2": (json.loads(QWEN.read_text()), EAGLE31, 2, MI308X, 2, []),
}


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """By build name, its drafter's class name, whether ATOM's forward context
    kept its names, the walk's modules and the installed backend."""
    assert EAGLE3_BUILD != BUILD
    root = tmp_path_factory.mktemp("eagle3")

    def build(name):
        target, draft, tp, machine, k, args = BUILDS[name]
        for d, config in (("model", target), ("draft", draft)):
            (root / name / d).mkdir(parents=True)
            (root / name / d / "config.json").write_text(json.dumps(config))
        args = [
            *args,
            "--method",
            "eagle3",
            "--draft-model",
            str(root / name / "draft"),
        ]
        args += ["--num-speculative-tokens", str(k)]
        argv = [str(root / name), str(machine), str(HERE), str(tp), *args]
        done = subprocess.run(
            [sys.executable, "-c", EAGLE3_BUILD, *argv],
            capture_output=True,
            text=True,
            check=False,
            timeout=900,
            cwd=TREE,
            # M3's fused decode path reads a CUDA graph mode a bare config
            # leaves unset; it adds no module the walk sees.
            env={**os.environ, "PYTHONPATH": str(TREE), "ATOM_MONO_ENABLE": "0"},
        )
        assert done.returncode == 0, done.stderr[-4000:]
        return pickle.loads((root / name / "built.pkl").read_bytes())

    with concurrent.futures.ThreadPoolExecutor(len(BUILDS)) as pool:
        return dict(zip(BUILDS, pool.map(build, BUILDS)))


def eagle3_pass(rows, sampled: int, tokens: int = 0, first: bool = True) -> float:
    """One pass of MiniMax-M3's EAGLE3 drafter over `rows` at TP 4, module by
    module from its config, on `tokens` tokens where given, its final norm and
    LM head on `sampled` rows, and on the first pass its `fc` over the
    target's three aux hidden states: each module's kernel bound, then its
    collective, or the larger of the two where the norm runs the all-reduce."""
    tp, h, d = 4, EAGLE3["hidden_size"], EAGLE3["head_dim"]
    q, kv = EAGLE3["num_attention_heads"] // tp, EAGLE3["num_key_value_heads"] // tp
    inter, vocab = EAGLE3["intermediate_size"] // tp, EAGLE3["vocab_size"] // tp
    t = tokens or sum_tokens(rows)
    works = [
        elementwise(t * h),  # embed_tokens, whole on every rank
        elementwise(t * 2 * h),  # input_layernorm and hidden_norm, one launch
        gemm(t, (q + 2 * kv) * d, 2 * h),  # qkv_proj over both
        elementwise(t * d),  # rotary_emb
        attention(
            [r.query_tokens for r in rows],
            [r.context_tokens for r in rows],
            q,
            d,
            kv,
            kv_fmt="fp8",
        ),
        gemm(t, h, q * d),  # o_proj, its all-reduce left to the next norm
        gemm(t, 2 * inter, h),  # gate_up_proj
        elementwise(t * inter, reads=2),  # SiluAndMul
        gemm(t, h, inter) + allreduce(tp, t * h * 2),  # down_proj
    ]
    if first:
        works.append(gemm(t, h, 3 * h))  # fc, whole on every rank
    if sampled:
        works.append(elementwise(sampled * h))  # norm, in compute_draft_ids
        works.append(gemm(sampled, vocab, h) + allgather(tp, sampled * 8 * tp))
    # post_attention_layernorm with o_proj's all-reduce in its kernel.
    fused = price(elementwise(t * h) + allreduce(tp, t * h * 2), MACHINE)
    return sum(serial(w, MACHINE) for w in works) + fused


@pytest.mark.parametrize("name", BUILDS)
def test_each_eagle3_drafter_is_built_and_every_module_classified(built, name):
    found = built[name]
    k = BUILDS[name][4]
    drafted = [m for m in found["modules"] if m.passes]

    assert found["drafter"] == "Eagle3LlamaModel"
    assert found["kept"]
    assert [m.name for m in found["modules"] if m.kind == UNCLASSIFIED] == []
    assert all(m.name.startswith("drafter") for m in drafted)
    assert {m.name for m in drafted if m.passes == 1} == {
        "drafter.fc",
        *(["drafter.fc_norm"] if name == "eagle3.1-tp2" else []),
    }
    assert {m.passes for m in drafted} == {1, k}
    assert type(found["installed"]) is StepRooflineBackend


def test_m3_s_eagle3_drafter_runs_its_layer_at_the_config_s_shapes(built):
    h, d = EAGLE3["hidden_size"], EAGLE3["head_dim"]
    drafted = {
        m.name.removeprefix("drafter."): (m.kind, m.dims, m.ranks, m.head)
        for m in built["m3-eagle3"]["modules"]
        if m.passes
    }
    assert drafted == {
        "embed_tokens": ("embedding", (h,), 1, False),
        "fc": ("gemm", (h, 3 * h), 1, False),
        "midlayer.dual_norm": ("norm", (2 * h,), 1, False),
        "midlayer.self_attn.qkv_proj": ("gemm", ((16 + 2 * 16) * d, 2 * h), 1, False),
        "midlayer.self_attn.o_proj": ("gemm", (h, 16 * d), 1, False),
        "midlayer.self_attn.rotary_emb": ("rotary", (d,), 1, False),
        "midlayer.self_attn.attn": ("attention", (16, 16, d), 1, False),
        "midlayer.mlp.gate_up_proj": ("gemm", (2 * 4608, h), 1, False),
        "midlayer.mlp.down_proj": ("gemm", (h, 4608), 4, False),
        "midlayer.mlp.act_fn": ("act", (4608,), 1, False),
        "midlayer.post_attention_layernorm": ("norm", (h,), 4, False),
        "norm": ("norm", (h,), 1, True),
        "lm_head": ("lm_head", (200064 // 4, h), 4, False),
    }


def test_an_m3_eagle3_decode_is_priced_with_its_draft_by_hand(built):
    """The named result: MiniMax-M3's EAGLE3 drafter at TP 4 on MI355X, 4 rows
    verifying 3 drafts each. Before, the roofline over the target alone
    refuses the step (`prices no draft`); now the drafter's `fc` and first
    pass run the 16 tokens, its two later passes a row a request, each with
    its final norm and LM head on the 4 rows."""
    modules = built["m3-eagle3"]["modules"]
    installed = built["m3-eagle3"]["installed"]
    target = StepRooflineBackend(MACHINE, [m for m in modules if not m.passes])
    before, now = speculative(target), speculative(installed)
    assert before.forward(prefill(8)).predicted_s == 0
    assert before.refused_commands() == ("cost:StepRooflineBackend: prices no draft",)
    assert now.forward(prefill(8)).predicted_s > 0
    assert now.refused_commands() == ()

    step = verify(3)
    cost = installed.price(step)
    hand = eagle3_pass(step.requests, 4)
    hand += 2 * eagle3_pass(later(step), 4, first=False)
    terms = {t.name: t.seconds for t in cost.draft_terms}
    print(
        f"\nMiniMax-M3 EAGLE3 K3 TP4 on MI355X, 4 rows of 4 tokens: draft "
        f"{cost.draft_seconds:.9g} s, by hand {hand:.9g} s; "
        + ", ".join(f"{n} {s * 1e6:.2f} us" for n, s in sorted(terms.items()))
    )
    assert all(t.startswith("draft roofline.") for t in terms)
    assert cost.draft_seconds == pytest.approx(hand, rel=1e-12)


def test_a_middle_chunk_runs_the_fc_and_the_first_pass_alone_and_no_head(built):
    chunk = BatchView((RequestShape(512, 512, False),), produces_output=False)
    cost = built["m3-eagle3"]["installed"].price(chunk)
    by_name = {t.name: t.seconds for t in cost.draft_terms}

    assert by_name["draft roofline.lm_head"] == 0
    assert cost.draft_seconds == pytest.approx(
        eagle3_pass(chunk.requests, 0), rel=1e-12
    )


def test_an_eagle3_1_head_norms_its_aux_chunks_once_and_its_output_every_pass(
    built,
):
    """`fc_norm`: the three aux chunks' norms in one launch before the first
    pass; `norm_output`: the final norm in the forward, on every row, with
    the down projection's all-reduce inside it."""
    h, tp = EAGLE31["hidden_size"], 2
    found = {
        m.name.removeprefix("drafter."): m
        for m in built["eagle3.1-tp2"]["modules"]
        if m.passes
    }
    fc_norm, norm, down = (
        found[n] for n in ("fc_norm", "norm", "midlayer.mlp.down_proj")
    )
    assert (fc_norm.kind, fc_norm.dims, fc_norm.passes) == ("norm", (3 * h,), 1)
    assert (norm.dims, norm.ranks, norm.fused, norm.head) == ((h,), tp, True, False)
    assert down.ranks == 1
