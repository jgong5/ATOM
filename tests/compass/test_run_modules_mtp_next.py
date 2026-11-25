# SPDX-License-Identifier: MIT
"""The step roofline lays out Qwen3-Next's and MiMo-V2's MTP passes (#3984).

`run.modules.drafter` builds the drafters ATOM maps their MTP configs to
(`Qwen3NextMTP`, `MiMoV2MTP`) on the meta device, and `walk` marks each of
their modules with its passes, one a draft token, as it does Qwen3.5's
(`test_run_modules_mtp`). Qwen3-Next's MTP layer is a Qwen3-Next decoder layer
behind a Qwen3.5-style prologue whose `fc` output is all-gathered over the TP
group; each pass runs its first MTP layer. MiMo-V2's MTP block attends a
sliding window, priced over that window (#3965).

Qwen3-Next-80B-A3B's config is its Hugging Face checkpoint's
(`qwen3_next_80b_a3b_config.json`), MiMo-V2-Flash's too
(`mimo_v2_flash_config.json`), its target cut to its first layer, full
attention and a dense MLP, which the roofline prices. Each is
built at TP 4 on `tests/compass/machines/mi355x-gpu40.json`, in a process of
its own, as `test_run_modules_mtp` builds its models.
"""

import concurrent.futures
import json
import os
import pickle
import subprocess
import sys

import pytest
from test_dp_step_max import prefill
from test_run_modules_mtp import (
    BUILD,
    HERE,
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
    Work,
    activated_experts,
    allgather,
    allreduce,
    attention,
    elementwise,
    gemm,
    grouped_gemm,
    price,
)
from atom.compass.spec import MachineSpec

QWEN3_NEXT = HERE / "qwen3_next_80b_a3b_config.json"
MIMO = HERE / "mimo_v2_flash_config.json"
MACHINE = MachineSpec.from_mapping(json.loads(MI355X.read_text()))
TP = 4
#: Each build: its config, its edits, its drafts and its drafter's class.
BUILDS = {
    "qwen3-next-mtp2": (QWEN3_NEXT, {}, 2, "Qwen3NextMTP"),
    "mimo-v2-mtp1": (
        MIMO,
        {"num_hidden_layers": 1, "hybrid_layer_pattern": [0], "moe_layer_freq": [0]},
        1,
        "MiMoV2MTP",
    ),
}


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """By build name, its drafter's class name, whether ATOM's forward context
    kept its names, the walk's modules and the installed backend."""
    root = tmp_path_factory.mktemp("mtp-next")

    def build(name):
        config, edits, k, _ = BUILDS[name]
        hf = json.loads(config.read_text()) | edits
        hf.pop("auto_map", None)  # MiMo-V2-Flash names remote code it never runs
        (root / name / "model").mkdir(parents=True)
        (root / name / "model" / "config.json").write_text(json.dumps(hf))
        args = ["--method", "mtp", "--num-speculative-tokens", str(k)]
        argv = [str(root / name), str(MI355X), str(HERE), str(TP), *args]
        done = subprocess.run(
            [sys.executable, "-c", BUILD, *argv],
            capture_output=True,
            text=True,
            check=False,
            timeout=900,
            cwd=TREE,
            env={**os.environ, "PYTHONPATH": str(TREE)},
        )
        assert done.returncode == 0, done.stderr[-4000:]
        return pickle.loads((root / name / "built.pkl").read_bytes())

    with concurrent.futures.ThreadPoolExecutor(len(BUILDS)) as pool:
        return dict(zip(BUILDS, pool.map(build, BUILDS)))


def next_pass(rows, sampled: int, tokens: int = 0) -> float:
    """One pass of Qwen3-Next-80B-A3B's MTP layer over `rows` at TP 4, module
    by module from its config, on `tokens` tokens where given, its LM head on
    `sampled` rows: each module's kernel bound, then its collective."""
    hf = json.loads(QWEN3_NEXT.read_text())
    h, d = hf["hidden_size"], hf["head_dim"]
    q, kv = hf["num_attention_heads"] // TP, max(1, hf["num_key_value_heads"] // TP)
    experts, k = hf["num_experts"], hf["num_experts_per_tok"]
    inter = hf["moe_intermediate_size"] // TP
    rotary = int(d * hf["partial_rotary_factor"])
    t = tokens or sum_tokens(rows)
    # The routed experts and the one shared expert the kernel runs beside
    # them on every row (`Qwen3NextSparseMoeBlock`), each local expert's
    # gate-up and down weights read once; a row in and out, its ids and
    # weights; the output all-reduced over the group.
    groups = activated_experts(experts, k, t) + 1
    picks = t * k + t
    flops = grouped_gemm(picks, 2 * inter, h, groups)
    flops += grouped_gemm(picks, h, inter, groups)
    held = 3 * inter * h * 2
    moved = groups * held + t * (h * 4 + 2 * (k + 1) * 4)
    works = [
        elementwise(t * h) + allreduce(TP, t * h * 2),  # embed_tokens
        *[elementwise(t * h)] * 5,  # the two pre-fc norms, the layer's two, the last
        gemm(t, h // TP, 2 * h) + allgather(TP, t * h * 2),  # fc, gathered
        gemm(t, 2 * q * d + 2 * kv * d, h),  # qkv_proj, its query gated
        gemm(t, h, q * d) + allreduce(TP, t * h * 2),  # o_proj
        elementwise(t * rotary),  # rotary_emb, a quarter of the head
        *[elementwise(t * d)] * 2,  # q_norm, k_norm
        attention(
            [r.query_tokens for r in rows], [r.context_tokens for r in rows], q, d, kv
        ),
        # The router and the shared expert's gate in one product, 513 rows
        # padded to 528 (`MergedReplicatedLinear(output_row_alignment=16)`).
        gemm(t, -(-(experts + 1) // 16) * 16, h),
        Work(flops.flops_by_format, moved) + allreduce(TP, t * h * 2),  # experts
    ]
    if sampled:
        # Its own LM head on its vocabulary shard, each rank's argmax gathered,
        # a (value, id) pair a row in fp32 (`compute_argmax_token`).
        vocab = hf["vocab_size"] // TP
        works.append(gemm(sampled, vocab, h) + allgather(TP, sampled * 8 * TP))
    return sum(serial(w, MACHINE) for w in works)


def mimo_window(rows) -> Work:
    """MiMo-V2-Flash's MTP block attention at TP 4 over `rows` of decode, a
    query at position `p` attending its last `min(p, 128)` keys: Q read and
    O written, each request's last 128 keys and values read, in bf16."""
    hf = json.loads(MIMO.read_text())
    heads, kv = hf["swa_num_attention_heads"] // TP, hf["swa_num_key_value_heads"] // TP
    dim, window = hf["swa_head_dim"], hf["sliding_window_size"]
    keys = sum(
        min(x, window)
        for r in rows
        for x in range(r.context_tokens - r.query_tokens + 1, r.context_tokens + 1)
    )
    read = sum(min(r.context_tokens, window) for r in rows)
    moved = (sum_tokens(rows) * 2 * heads + read * 2 * kv) * 2 * dim
    return Work({"bf16": 4 * heads * dim * keys}, moved)


def mimo_pass(rows, sampled: int) -> float:
    """One pass of MiMo-V2-Flash's MTP layer over `rows` at TP 4, module by
    module from its config, its LM head on `sampled` rows: each module's
    kernel bound, then its collective. Its fp8 linears read fp8 activations;
    `eh_proj` and `o_proj`, which the checkpoint holds unscaled, run bf16."""
    hf = json.loads(MIMO.read_text())
    h, d = hf["hidden_size"], hf["swa_head_dim"]
    q, kv = hf["swa_num_attention_heads"] // TP, hf["swa_num_key_value_heads"] // TP
    inter, v = hf["intermediate_size"] // TP, hf["swa_v_head_dim"]
    rotary = int(d * hf["partial_rotary_factor"])
    t = sum_tokens(rows)
    works = [
        elementwise(t * h) + allreduce(TP, t * h * 2),  # embed_tokens
        *[elementwise(t * h)] * 5,  # enorm, hnorm, the block's two, final
        gemm(t, h, 2 * h),  # eh_proj over the embedding and the hidden state
        gemm(t, (q + kv) * d + kv * v, h, "fp8", "fp8"),  # qkv_proj
        elementwise(t * rotary),  # rotary_emb, a third of the head
        mimo_window(rows),
        gemm(t, h, q * v) + allreduce(TP, t * h * 2),  # o_proj
        gemm(t, 2 * inter, h, "fp8", "fp8"),  # gate_up_proj
        elementwise(t * inter, reads=2),  # SiluAndMul
        gemm(t, h, inter, "fp8", "fp8") + allreduce(TP, t * h * 2),  # down_proj
    ]
    if sampled:
        vocab = hf["vocab_size"] // TP
        works.append(gemm(sampled, vocab, h) + allgather(TP, sampled * 8 * TP))
    return sum(serial(w, MACHINE) for w in works)


@pytest.mark.parametrize("name", BUILDS)
def test_each_drafter_is_built_and_walked_its_passes_one_a_draft_token(built, name):
    found = built[name]
    _, _, k, drafter = BUILDS[name]
    drafted = [m for m in found["modules"] if m.passes]

    assert found["drafter"] == drafter
    assert found["kept"]
    assert drafted and {m.passes for m in drafted} == {k}
    assert all(m.name.startswith("drafter") for m in drafted)
    assert {"embedding", "lm_head"} <= {m.kind for m in drafted}
    assert type(found["installed"]) is StepRooflineBackend


def test_qwen3_next_classifies_every_module_and_gathers_its_fc(built):
    modules = built["qwen3-next-mtp2"]["modules"]
    fc = next(m for m in modules if m.name == "drafter.model.fc")

    assert [m.name for m in modules if m.kind == UNCLASSIFIED] == []
    assert (fc.dims, fc.ranks, fc.gathers) == ((2048 // TP, 2 * 2048), TP, True)


def test_a_qwen3_next_mtp2_decode_is_priced_with_its_draft_by_hand(built):
    """The named result: Qwen3-Next-80B-A3B MTP2 at TP 4 on MI355X, 4 rows
    verifying 2 drafts each. Before, the roofline over the target alone
    refuses the step (`prices no draft`); now its drafter's first pass runs
    the 12 tokens and its second a row a request, each with its LM head on
    the 4 rows."""
    modules = built["qwen3-next-mtp2"]["modules"]
    installed = built["qwen3-next-mtp2"]["installed"]
    target = StepRooflineBackend(
        installed.machine, [m for m in modules if not m.passes]
    )
    before, now = speculative(target), speculative(installed)
    assert before.forward(prefill(8)).predicted_s == 0
    assert before.refused_commands() == ("cost:StepRooflineBackend: prices no draft",)
    assert now.forward(prefill(8)).predicted_s > 0
    assert now.refused_commands() == ()

    step = verify(2)
    cost = installed.price(step)
    hand = next_pass(step.requests, 4) + next_pass(later(step), 4)
    terms = {t.name: t.seconds for t in cost.draft_terms}
    print(
        f"\nQwen3-Next-80B-A3B MTP2 TP4 on MI355X, 4 rows of 3 tokens: target "
        f"{cost.target_seconds:.9g} s, draft {cost.draft_seconds:.9g} s, by hand "
        f"{hand:.9g} s; "
        + ", ".join(f"{n} {s * 1e6:.2f} us" for n, s in sorted(terms.items()))
    )
    assert all(t.startswith("draft roofline.") for t in terms)
    assert cost.draft_seconds == pytest.approx(hand, rel=1e-12)


def test_a_middle_chunk_runs_the_first_qwen3_next_pass_alone_and_no_head(built):
    chunk = BatchView((RequestShape(512, 512, False),), produces_output=False)
    cost = built["qwen3-next-mtp2"]["installed"].price(chunk)
    assert cost.draft_seconds == pytest.approx(next_pass(chunk.requests, 0), rel=1e-12)


def test_mimo_v2_s_windowed_mtp_block_is_priced_over_its_window(built):
    """Every module of MiMo-V2-Flash's drafter is classified, its MTP block's
    sliding-window attention priced over its window (#3965), and its pass
    priced by hand: on 4 rows verifying 1 draft at contexts past the window,
    each query attends the last 128 keys and each request's last 128 keys
    and values are read. Before, its drafter was not built: `prices no
    draft`."""
    hf = json.loads(MIMO.read_text())
    heads, kv = hf["swa_num_attention_heads"] // TP, hf["swa_num_key_value_heads"] // TP
    dim, window = hf["swa_head_dim"], hf["sliding_window_size"]
    modules = built["mimo-v2-mtp1"]["modules"]
    (attn,) = [m for m in modules if m.passes and m.kind == "attention"]
    assert [m.name for m in modules if m.kind == UNCLASSIFIED] == []
    assert attn.name == "drafter.model.layers.1.mtp_block.self_attn.attn"
    assert attn.dims == (heads, kv, dim, window, 1)

    step = verify(1)
    rows = len(step.requests)
    assert mimo_window(step.requests).flops_by_format == {
        "bf16": 4 * heads * dim * rows * 2 * window
    }
    window_s = price(mimo_window(step.requests), MACHINE)
    alone = StepRooflineBackend(MACHINE, [attn]).price(step)
    assert alone.draft_seconds == pytest.approx(window_s, rel=1e-12)
    cost = built["mimo-v2-mtp1"]["installed"].price(step)
    hand = mimo_pass(step.requests, rows)
    terms = {t.name: t.seconds for t in cost.draft_terms}
    print(
        f"\nMiMo-V2-Flash MTP1 TP4 on MI355X, 4 rows of 2 tokens: draft "
        f"{cost.draft_seconds:.9g} s, by hand {hand:.9g} s, its windowed "
        f"attention {window_s:.9g} s; "
        + ", ".join(f"{n} {s * 1e6:.2f} us" for n, s in sorted(terms.items()))
    )
    assert terms["draft roofline.attention"] == pytest.approx(window_s, rel=1e-12)
    assert cost.draft_seconds == pytest.approx(hand, rel=1e-12)
