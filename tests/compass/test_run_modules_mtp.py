# SPDX-License-Identifier: MIT
"""The step roofline lays out a serial MTP drafter's passes.

`run.modules.drafter` builds the drafter ATOM maps Qwen3.5's, DeepSeek-V3's
and DeepSeek-V4's MTP configs to (`Qwen3_5MTP`, `DeepSeekMTP`,
`DeepseekV4MTP`) on the meta device, and `walk` marks each of its modules
with its passes, one a draft token; V4's, which holds no LM head, runs the
target's embedding and LM head. The step roofline prices them as the step's
draft terms: the first pass over the step's rows, each later one a row a
request, the LM head a row a request in each, and on a step that samples
nothing the first pass alone, unless a data-parallel group proposes for its
collectives. Each pass runs the first MTP layer; V4's block norms its inputs
and projects each residual copy, and reduces and norms its head's input on
the rows its head samples alone, where V4-Pro's target reduces and norms
every row before its logits, none on a middle chunk, and its LM head runs a
decode graph's pad rows too; at TP 2 Qwen3.5's gathers its `fc`'s output
and its argmax alone, and under a pure data-parallel group of two its later
passes run at the step's `running_bs`, draft graph or none, the group
sharding their argmax. Under V4-Pro's DP8 at TP 1 the target's LM head,
which its drafts share, is sharded over the group on a decode every rank
runs, a draft's never. A speculative step then answers
`_group_step_seconds`' "prices no draft" check, the roofline alone or below a
backend that refuses the step; a run with no drafter laid out, Eagle3's
(#3962), is still refused.

Each model is built in a process of its own, as
`test_backend_step_roofline_v41` builds V4.1's. Qwen3.5-0.8B's config is its
Hugging Face checkpoint's (`qwen3_5_0_8b_config.json`), and it is installed
through `run.runner` as the run's only backend on
`tests/compass/machines/mi308x-hjbog18.json`; the installed backend and the
walk's modules come back pickled.
"""

import concurrent.futures
import dataclasses
import json
import os
import pathlib
import pickle
import subprocess
import sys
from types import SimpleNamespace

import pytest
from test_dp_step_max import prefill, runner
from test_op_graph_drafter_width import settled

from atom.compass.backends.base import CostBackend, Granularity
from atom.compass.backends.provenance import Refusal
from atom.compass.backends.resolver import CostRefused
from atom.compass.backends.shape import BatchView, RequestShape, sum_tokens
from atom.compass.backends.step_roofline import (
    UNCLASSIFIED,
    Module,
    StepFallbackBackend,
    StepRooflineBackend,
)
from atom.compass.backends.work import (
    Work,
    allgather,
    attention,
    bounds,
    elementwise,
    gemm,
    price,
)
from atom.compass.run.modules import _dp_argmax, _dp_head
from atom.compass.spec import MachineSpec

HERE = pathlib.Path(__file__).parent
TREE = HERE.parents[1]
CONFIG = HERE / "qwen3_5_0_8b_config.json"
MI308X = HERE / "machines" / "mi308x-hjbog18.json"
#: V4-Pro's fp4 experts build on an MI355X's arch.
MI355X = HERE / "machines" / "mi355x-gpu40.json"
MACHINE = MachineSpec.from_mapping(json.loads(MI308X.read_text()))
V4_PRO = HERE / "deepseek_v4_pro_config.json"
V3 = ["--kv_cache_dtype", "fp8"]
#: Each build: its config, TP width, machine, draft tokens, ATOM arguments.
BUILDS = {
    "qwen3.5-mtp1": (CONFIG, 1, MI308X, 1, []),
    "qwen3.5-mtp3": (CONFIG, 1, MI308X, 3, []),
    "v3-mtp2": (HERE / "deepseek_v3_config.json", 8, MI308X, 2, V3),
    "v4-mtp2": (V4_PRO, 8, MI355X, 2, []),
    "qwen3.5-tp2": (CONFIG, 2, MI308X, 1, []),
    "qwen3.5-2layers": (CONFIG, 1, MI308X, 1, []),
    "qwen3.5-dp2-mtp3": (CONFIG, 1, MI308X, 3, ["--data-parallel-size", "2"]),
    "v4-dp8-mtp2": (V4_PRO, 8, MI355X, 2, ["--enable-dp-attention"]),
}
#: Builds whose text config is edited: two MTP layers.
TEXT = {"qwen3.5-2layers": {"mtp_num_hidden_layers": 2}}
DRAFTERS = {
    "qwen3.5-mtp1": "Qwen3_5MTP",
    "qwen3.5-mtp3": "Qwen3_5MTP",
    "qwen3.5-tp2": "Qwen3_5MTP",
    "qwen3.5-2layers": "Qwen3_5MTP",
    "qwen3.5-dp2-mtp3": "Qwen3_5MTP",
    "v3-mtp2": "DeepSeekMTP",
    "v4-mtp2": "DeepseekV4MTP",
    "v4-dp8-mtp2": "DeepseekV4MTP",
}

#: Builds the model under argv's root and its drafter on the meta device at
#: argv's TP width and machine's arch, as a simulated run builds them, walks them,
#: installs the step roofline alone through `run.runner` with the model's
#: memory readings published, then builds ATOM's own drafter, as ATOM does
#: after the install, and pickles the walk's modules and the installed backend.
BUILD = """
import json, os, pathlib, pickle, sys
from types import SimpleNamespace
root, machine = pathlib.Path(sys.argv[1]), json.loads(pathlib.Path(sys.argv[2]).read_text())
sys.path.insert(0, sys.argv[3])
tp, *atom_args = sys.argv[4:]
from atom.compass.validate.bound import _build
arch = machine["device"]["arch"]
config, tree = _build(str(root / "model"), int(tp), atom_args, arch, str(root))
from atom.compass.run.modules import drafter, walk
from atom.compass.runner.overrides import FactoriesOnMeta, _installed_backend
held = set(config.compilation_config.static_forward_context)
draft = drafter(config)
kept = set(config.compilation_config.static_forward_context) == held
modules = walk(tree, draft)
from test_memory_readings import publish_readings
from atom.compass import run as compass_run
publish_readings(root / "store", config, document=machine)
run = {"bound_s": 600.0, "step_roofline": True, "machine": machine, "store": str(root / "store")}
(root / "run.json").write_text(json.dumps(run))
os.environ[compass_run.ENV] = str(root / "run.json")
runner = SimpleNamespace(config=config, rank=0, _tr_hybrid_active=lambda: False)
compass_run.runner(runner, tree)
with FactoriesOnMeta():  # ATOM builds its own drafter after, of the same names
    type(draft)(config)
built = {"drafter": type(draft).__name__, "kept": kept, "modules": modules,
         "installed": _installed_backend(runner)}
(root / "built.pkl").write_bytes(pickle.dumps(built))
"""


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """By build name, its drafter's class name, whether ATOM's forward context
    kept its names, the walk's modules and the installed backend."""
    root = tmp_path_factory.mktemp("mtp")

    def build(name):
        config, tp, machine, k, args = BUILDS[name]
        hf = json.loads(config.read_text())
        hf.pop("auto_map", None)  # DeepSeek-V3's names remote code it never runs
        hf.get("text_config", {}).update(TEXT.get(name, {}))
        (root / name / "model").mkdir(parents=True)
        (root / name / "model" / "config.json").write_text(json.dumps(hf))
        args = [*args, "--method", "mtp", "--num-speculative-tokens", str(k)]
        argv = [str(root / name), str(machine), str(HERE), str(tp), *args]
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


def one_pass(rows, heads: int, tokens: int = 0, head: float | None = None) -> float:
    """One pass of Qwen3.5-0.8B's MTP layer over `rows` at TP 1, module by
    module from its config, each one roofline (`work.price`), on `tokens`
    tokens where given, its LM head on `heads` rows, none for 0, or at
    `head` seconds."""
    text = json.loads(CONFIG.read_text())["text_config"]
    h, d, inter = text["hidden_size"], text["head_dim"], text["intermediate_size"]
    q, kv = text["num_attention_heads"], text["num_key_value_heads"]
    rotary = int(d * text["rope_parameters"]["partial_rotary_factor"])
    t = tokens or sum(r.query_tokens for r in rows)
    works = [
        elementwise(t * h),  # embed_tokens
        *[elementwise(t * h)] * 5,  # the two pre-fc norms, the layer's two, the last
        gemm(t, h, 2 * h),  # fc, over the embedding and the hidden state
        gemm(t, 2 * q * d + 2 * kv * d, h),  # qkv_proj, its query gated
        gemm(t, h, q * d),  # o_proj
        elementwise(t * rotary),  # rotary_emb, a quarter of the head
        *[elementwise(t * d)] * 2,  # q_norm, k_norm
        attention(
            [r.query_tokens for r in rows], [r.context_tokens for r in rows], q, d, kv
        ),
        gemm(t, 2 * inter, h),  # gate_up_proj
        gemm(t, h, inter),  # down_proj
        elementwise(t * inter, reads=2),  # SiluAndMul
    ]
    if heads and head is None:
        works.append(gemm(heads, text["vocab_size"], h))  # lm_head
    return sum(price(w, MACHINE) for w in works) + (head or 0.0)


def serial(work, machine=MACHINE) -> float:
    """`work`'s kernel bound and then its collectives, as the roofline prices
    a module whose kernel runs no collective of its own."""
    b = bounds(work, machine)
    return max(b["floor"], b["compute"], b["memory"]) + b["comm"]


def verify(k: int, rows: int = 4) -> BatchView:
    """A decode step verifying `k` drafts of each of `rows` requests."""
    return BatchView(
        tuple(RequestShape(k + 1, 600 + i, True) for i in range(rows)),
        produces_output=True,
    )


def later(view: BatchView):
    """A serial drafter's later pass over `view`: a row a request."""
    return tuple(RequestShape(1, r.context_tokens, True) for r in view.requests)


@pytest.mark.parametrize("name", BUILDS)
def test_each_mtp_drafter_is_built_and_every_module_classified(built, name):
    found = built[name]
    k = BUILDS[name][3]
    drafted = [m for m in found["modules"] if m.passes]

    assert found["drafter"] == DRAFTERS[name]
    assert found["kept"]
    assert [m.name for m in found["modules"] if m.kind == UNCLASSIFIED] == []
    assert drafted and {m.passes for m in drafted} == {k}
    assert all(m.name.startswith("drafter") for m in drafted)
    # Each pass samples through an LM head and embeds its input.
    assert {"embedding", "lm_head"} <= {m.kind for m in drafted}
    assert type(found["installed"]) is StepRooflineBackend


def test_a_drafter_with_no_head_of_its_own_runs_the_targets(built):
    shared = [m for m in built["v4-mtp2"]["modules"] if "the target's" in m.name]
    assert [(m.kind, m.passes) for m in shared] == [("embedding", 2), ("lm_head", 2)]


def test_qwen3_5_0_8b_drafter_runs_its_mtp_layer_at_the_configs_shapes(built):
    drafted = sorted(
        (m.kind, m.dims) for m in built["qwen3.5-mtp1"]["modules"] if m.passes
    )
    assert drafted == sorted(
        [("embedding", (1024,)), ("gemm", (1024, 2048)), ("gemm", (5120, 1024))]
        + [
            ("gemm", (1024, 2048)),
            ("rotary", (64,)),
            ("norm", (256,)),
            ("norm", (256,)),
        ]
        + [("attention", (8, 2, 256)), ("gemm", (7168, 1024)), ("gemm", (1024, 3584))]
        + [("act", (3584,)), ("lm_head", (248320, 1024))]
        + [("norm", (1024,))] * 5
    )


def test_the_draft_term_of_an_mtp1_decode_step_at_batch_4_by_hand(built):
    """The named result: Qwen3.5-0.8B MTP1 at TP 1 on MI308X."""
    step = verify(1)
    cost = built["qwen3.5-mtp1"]["installed"].price(step)
    hand = one_pass(step.requests, 4)
    print(
        f"\ntarget {cost.target_seconds:.9g} s, draft {cost.draft_seconds:.9g} s,"
        f" by hand {hand:.9g} s"
    )
    assert cost.terms and all(t.name.startswith("roofline.") for t in cost.terms)
    assert cost.draft_terms
    assert all(t.name.startswith("draft roofline.") for t in cost.draft_terms)
    assert cost.draft_seconds == pytest.approx(hand, rel=1e-12)


def test_a_verify_step_drafts_one_pass_a_draft_token(built):
    """MTP3: the first pass over the four rows of four tokens, two more a row
    a request, each with its LM head on the four rows."""
    step = verify(3)
    cost = built["qwen3.5-mtp3"]["installed"].price(step)
    first, after = one_pass(step.requests, 4), one_pass(later(step), 4)
    assert cost.draft_seconds == pytest.approx(first + 2 * after, rel=1e-12)
    one = built["qwen3.5-mtp1"]["installed"].price(step)
    assert one.draft_seconds == pytest.approx(first, rel=1e-12)
    assert one.target_seconds == cost.target_seconds


def test_a_middle_chunk_runs_the_first_pass_alone_and_no_head(built):
    chunk = BatchView((RequestShape(512, 512, False),), produces_output=False)
    cost = built["qwen3.5-mtp3"]["installed"].price(chunk)
    by_name = {t.name: t.seconds for t in cost.draft_terms}

    assert by_name["draft roofline.lm_head"] == 0
    assert cost.draft_seconds == pytest.approx(one_pass(chunk.requests, 0), rel=1e-12)


def test_a_dp_middle_chunk_runs_every_pass_with_its_head(built):
    """Under a data-parallel group a middle chunk proposes for the group's
    collectives (`propose(align_only=True)`): the first pass with its LM head,
    then each later one, a row a request."""
    chunk = BatchView(
        (RequestShape(512, 512, False),),
        moe_rows=1024,
        draft_moe_rows=1024,
        produces_output=False,
    )
    cost = built["qwen3.5-mtp3"]["installed"].price(chunk)
    first, after = one_pass(chunk.requests, 1), one_pass(later(chunk), 1)
    assert cost.draft_seconds == pytest.approx(first + 2 * after, rel=1e-12)


def test_a_drafter_of_two_mtp_layers_runs_its_first_in_each_pass(built):
    """`EagleProposer` passes no `spec_step_idx`, so each pass runs layer 0."""

    def drafted(name):
        return sorted((m.kind, m.dims) for m in built[name]["modules"] if m.passes)

    assert drafted("qwen3.5-2layers") == drafted("qwen3.5-mtp1")


def test_at_tp2_qwen3_5_gathers_its_fc_and_its_argmax_alone(built):
    """`fc`'s output shards are all-gathered; the LM head gathers each rank's
    (value, id) pair a row in fp32 (`compute_argmax_token`), not the logits."""
    text = json.loads(CONFIG.read_text())["text_config"]
    h, vocab = text["hidden_size"], text["vocab_size"]
    found = {m.name: m for m in built["qwen3.5-tp2"]["modules"] if m.passes}
    step = verify(1, rows=64)
    t = sum_tokens(step.requests)
    hands = {
        "drafter.model.fc": gemm(t, h // 2, 2 * h) + allgather(2, t * h * 2),
        "drafter.lm_head": gemm(64, vocab // 2, h) + allgather(2, 64 * 2 * 4 * 2),
    }
    for name, hand in hands.items():
        cost = StepRooflineBackend(MACHINE, [found[name]]).price(step)
        assert cost.draft_seconds == pytest.approx(serial(hand), rel=1e-12)


def test_v4s_mtp_block_norms_its_inputs_and_projects_each_residual_copy(built):
    """`MTPBlock.forward` norms the token's embedding (`enorm`) and each of
    the target's residual copies (`hnorm`), projecting each copy (`h_proj`);
    `norm` runs before the head (`_head_input`)."""
    hf = json.loads(V4_PRO.read_text())
    hc, d = hf["hc_mult"], hf["hidden_size"]
    block = "drafter.model.mtp.0."
    found = {
        m.name.removeprefix(block): m for m in built["v4-mtp2"]["modules"] if m.passes
    }
    norms = [(found[n].kind, found[n].dims) for n in ("enorm", "hnorm", "norm")]
    assert norms == [("norm", (d,)), ("norm", (hc * d,)), ("norm", (d,))]
    h_proj = found["h_proj"]
    assert (found["e_proj"].dims, h_proj.dims) == ((d, d), (d, d, hc))
    chunk = BatchView((RequestShape(4096, 4096, False),), produces_output=False)
    hand = gemm(hc * 4096, d, d, h_proj.act, h_proj.fmt)
    mi355x = MachineSpec.from_mapping(json.loads(MI355X.read_text()))
    cost = StepRooflineBackend(mi355x, [h_proj]).price(chunk)
    assert cost.draft_seconds == pytest.approx(price(hand, mi355x), rel=1e-12)


def test_v4s_mtp_head_input_runs_on_the_rows_its_head_samples(built):
    """`DeepseekV4MTPModel._head_input` reduces the target's residual copies
    by the block's own weights (`ParallelHead.hc_head`) and norms the result,
    on the rows `compute_draft_ids` samples: none on a pass with no head, a
    row a request on each pass with one. A named result."""
    hf = json.loads(V4_PRO.read_text())
    hc, d = hf["hc_mult"], hf["hidden_size"]
    block = "drafter.model.mtp.0."
    found = {
        m.name.removeprefix(block): m for m in built["v4-mtp2"]["modules"] if m.passes
    }
    head = [found["hc_head"], found["norm"]]
    assert [(m.kind, m.dims, m.head) for m in head] == [
        ("hc_head", (hc, d), True),
        ("norm", (d,), True),
    ]
    mi355x = MachineSpec.from_mapping(json.loads(MI355X.read_text()))
    roofline = StepRooflineBackend(mi355x, head)
    chunk = BatchView((RequestShape(4096, 4096, False),), produces_output=False)
    headless = roofline.price(chunk)
    step = verify(2)
    # A row's copies read and one row written, in bf16; one gate a copy, an
    # fp32 projection of the row's copies; then the norm over the row.
    n = len(step.requests)
    gates = Work({"bf16": 2 * n * hc * hc * d}, n * (hc + 1) * d * 2 + hc * hc * d * 4)
    hand = 2 * (price(gates, mi355x) + price(elementwise(n * d), mi355x))
    cost = roofline.price(step)
    print(
        f"\nV4-Pro MTP2 head input: headless 4096-token chunk "
        f"{headless.draft_seconds:.9g} s; 4-row verify {cost.draft_seconds:.9g} s,"
        f" by hand {hand:.9g} s"
    )
    assert {t.name: t.seconds for t in headless.draft_terms} == {
        "draft roofline.hc_head": 0,
        "draft roofline.norm": 0,
    }
    assert cost.draft_seconds == pytest.approx(hand, rel=1e-12)


def test_v4_pro_s_target_reduces_every_row_before_its_logits(built):
    """`DeepseekV4ForCausalLM.compute_logits` reduces the target's residual
    copies by the model's own weights (`ParallelHead.hc_head`) on every row
    the forward ran, before the final norm and the LM head. A named result:
    its share of a V4-Pro TP8 MTP2 decode and prefill step on MI355X."""
    hf = json.loads(V4_PRO.read_text())
    hc, d = hf["hc_mult"], hf["hidden_size"]
    modules = built["v4-mtp2"]["modules"]
    (head,) = [m for m in modules if m.kind == "hc_head" and not m.passes]
    assert (head.name, head.dims, head.head, head.logits, head.layer) == (
        "model.hc_head",
        (hc, d),
        False,
        True,
        -1,
    )
    installed = built["v4-mtp2"]["installed"]
    machine = installed.machine
    others = StepRooflineBackend(machine, [m for m in modules if m is not head])
    # 64 requests verifying 2 drafts each; a 4096-token prompt.
    steps = {
        "decode": BatchView((RequestShape(3, 4096, True),) * 64, produces_output=True),
        "prefill": BatchView((RequestShape(4096, 4096, False),), produces_output=True),
    }
    lines = []
    for name, step in steps.items():
        n = sum_tokens(step.requests)
        # A row's copies read and one row written, in bf16; one gate a copy,
        # an fp32 projection of the row's copies.
        moved = n * (hc + 1) * d * 2 + hc * hc * d * 4
        hand = price(Work({"bf16": 2 * n * hc * hc * d}, moved), machine)
        cost = installed.price(step)
        by_name = {t.name: t.seconds for t in cost.terms}
        whole = cost.target_seconds + cost.draft_seconds
        assert by_name["roofline.hc_head"] == pytest.approx(hand, rel=1e-12)
        assert cost.target_seconds - others.price(step).target_seconds == (
            pytest.approx(hand, rel=1e-9)
        )
        lines.append(
            f"{name} {n} rows: {hand:.9g} s by hand, "
            f"{by_name['roofline.hc_head'] / cost.target_seconds:.4%} of the "
            f"target's {cost.target_seconds:.9g} s, "
            f"{by_name['roofline.hc_head'] / whole:.4%} of the step's {whole:.9g} s"
        )
    print("\nV4-Pro TP8 MTP2 target hc_head: " + "; ".join(lines))


def test_v4_pro_s_final_norm_and_lm_head_run_on_the_rows_atom_hands_them(built):
    """`DeepseekV4ForCausalLM.compute_logits` runs the final norm, which a
    middle chunk never calls, and the LM head on every row the runner hands
    it, a decode graph's pad rows too. A named result: a 4096-token V4-Pro
    TP8 middle chunk's final norm and the LM head of 60 requests verifying 2
    drafts in a rung-64 graph, on MI355X, before and after."""
    d = json.loads(V4_PRO.read_text())["hidden_size"]
    modules = built["v4-mtp2"]["modules"]
    (norm,) = [m for m in modules if m.name == "model.norm"]
    (lm_head,) = [m for m in modules if m.kind == "lm_head" and not m.passes]
    assert (norm.kind, norm.dims, norm.ranks, norm.logits) == ("norm", (d,), 1, True)
    assert (lm_head.dims[1], lm_head.fmt, lm_head.ranks) == (d, "bf16", 8)
    installed = built["v4-mtp2"]["installed"]
    machine, shard = installed.machine, lm_head.dims[0]
    # Before: the final norm on every step, the LM head on the real rows.
    every = [dataclasses.replace(m, logits=False) if m is norm else m for m in modules]
    before = StepRooflineBackend(machine, every)
    real = StepRooflineBackend(machine, modules, padded=False)

    def terms(backend, step):
        return {t.name: t.seconds for t in backend.price(step).terms}

    def head(rows):
        # The vocabulary shard over each row in bf16: the rows and the shard
        # read, the logits written, and gathered from the 8 ranks.
        moved = (rows * d + shard * d + rows * shard) * 2
        work = Work({"bf16": 2 * rows * shard * d}, moved)
        return serial(work + allgather(8, rows * shard * 8 * 2), machine)

    chunk = BatchView((RequestShape(4096, 4096, False),), produces_output=False)
    # Each row read and written once, in bf16.
    normed = price(Work({}, 4096 * d * 2 * 2), machine)
    was, now = terms(before, chunk), terms(installed, chunk)
    assert was.pop("roofline.norm") - now.pop("roofline.norm", 0) == (
        pytest.approx(normed, rel=1e-12)
    )
    assert now == was
    decode = BatchView(
        (RequestShape(3, 4096, True),) * 60, capture_rung=64, produces_output=True
    )
    padded = terms(installed, decode)["roofline.lm_head"]
    unpadded = terms(real, decode)["roofline.lm_head"]
    assert padded == pytest.approx(head(64 * 3), rel=1e-12)
    assert unpadded == pytest.approx(head(60 * 3), rel=1e-12)
    print(
        f"\nV4-Pro TP8 MTP2 on MI355X: a 4096-token middle chunk's final norm "
        f"{normed * 1e6:.2f} us before, none after; the LM head of 60 requests "
        f"of 3 tokens at rung 64, {unpadded * 1e6:.2f} us on 180 rows before, "
        f"{padded * 1e6:.2f} us on 192 after"
    )


def test_a_dp2_decode_drafts_its_later_passes_at_running_bs_sharding_their_argmax(
    built, monkeypatch
):
    """A named result: Qwen3.5-0.8B MTP3 at DP2 TP1 on MI308X, a decode of 3
    rows verifying 4 tokens each beside a peer's 3, its view `batch_view`'s
    for a drafter with no draft graph (MRoPE's): `running_bs` 4. The first
    pass runs the 12 tokens, its head the whole vocabulary on 4 rows, a height
    no peer is told; each later pass 4 rows, its head the group's sharded
    argmax: 8 rows gathered, half the vocabulary, their (value, id) pairs
    gathered (`_dp_sharded_logits`)."""
    rows = [(4, 600 + i, True) for i in range(3)]
    view = settled(monkeypatch, rows, [(3, 12, False)], step=False, drafts=3)
    assert view.draft_bs == 4
    text = json.loads(CONFIG.read_text())["text_config"]
    h, vocab = text["hidden_size"], text["vocab_size"]
    found = built["qwen3.5-dp2-mtp3"]["modules"]
    heads = [(m.passes, m.dims) for m in found if m.kind == "lm_head"]
    assert heads == [(3, (vocab, h, 2, 256))]  # the drafter's: the target ties
    sharded = gemm(8, vocab // 2, h) + allgather(2, 8 * h * 2) + allgather(2, 2 * 8 * 8)
    first = one_pass(view.requests, 4)
    after = one_pass(later(view), 4, tokens=4, head=serial(sharded))
    cost = built["qwen3.5-dp2-mtp3"]["installed"].price(view)
    print(
        f"\nQwen3.5-0.8B MTP3 DP2: draft {cost.draft_seconds:.9g} s, by hand "
        f"{first + 2 * after:.9g} s (first {first:.9g} s, later {after:.9g} s)"
    )
    assert cost.draft_seconds == pytest.approx(first + 2 * after, rel=1e-12)


def test_a_dp_group_shards_no_argmax_past_its_cap(built, monkeypatch):
    """`_can_use_dp_sharded_argmax` takes at most
    `ATOM_DP_DRAFT_ARGMAX_MAX_ROWS` rows, 256 by default: a later pass of 300
    runs the whole vocabulary on every rank, as the first pass does. Off, or
    on one rank, no head carries the group."""
    text = json.loads(CONFIG.read_text())["text_config"]
    h, vocab = text["hidden_size"], text["vocab_size"]
    found = built["qwen3.5-dp2-mtp3"]["modules"]
    head = next(m for m in found if m.passes and m.kind == "lm_head")
    step = dataclasses.replace(verify(3, rows=300), draft_bs=300)
    cost = StepRooflineBackend(MACHINE, [head]).price(step)
    assert cost.draft_seconds == pytest.approx(
        3 * price(gemm(300, vocab, h), MACHINE), rel=1e-12
    )
    group = SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_size=2))
    alone = SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_size=1))
    assert (_dp_argmax(group), _dp_argmax(alone)) == ((2, 256), ())
    monkeypatch.setenv("ATOM_DP_DRAFT_ARGMAX", "0")
    assert _dp_argmax(group) == ()


def test_a_dp_group_shards_its_targets_lm_head_at_tp1_by_its_mode(monkeypatch):
    """`ParallelLMHead._can_use_dp_sharded_head`: a head at TP 1, a group of
    two or more ranks that divides its vocabulary, `ATOM_DP_LM_HEAD_MODE`
    all2all, the default, or allgather. The head carries the group and the
    logits a row its exchange holds: a rank's shard, or the whole vocabulary;
    under any other mode ("default", ATOM's opt-out) it runs unsharded."""
    import atom.config

    def at(dp):
        config = SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_size=dp))
        monkeypatch.setattr(atom.config, "get_current_atom_config", lambda: config)

    head = Module("lm_head", "lm_head", (1000, 64))
    at(4)
    monkeypatch.delenv("ATOM_DP_LM_HEAD_MODE", raising=False)
    assert _dp_head(head).dims == (1000, 64, 4, 250)
    assert _dp_head(dataclasses.replace(head, ranks=2)) == dataclasses.replace(
        head, ranks=2
    )
    assert _dp_head(dataclasses.replace(head, dims=(1002, 64))).dims == (1002, 64)
    monkeypatch.setenv("ATOM_DP_LM_HEAD_MODE", "allgather")
    assert _dp_head(head).dims == (1000, 64, 4, 1000)
    monkeypatch.setenv("ATOM_DP_LM_HEAD_MODE", "default")
    assert _dp_head(head) == head
    monkeypatch.delenv("ATOM_DP_LM_HEAD_MODE")
    at(1)
    assert _dp_head(head) == head


def test_a_v4_dp8_target_shards_its_lm_head_and_its_drafts_do_not(built, monkeypatch):
    """V4-Pro MTP2 under `--enable-dp-attention -tp 8`, DP8 at TP 1, on
    MI355X. The target's LM head carries the group and an eighth of the
    vocabulary; its drafts', the same head, the group and their argmax's cap
    alone. `ForwardMode.decide` settles a verify while every rank decodes
    into a rung-4 graph of no prefilling group, which the head shards on, and
    one beside a prefilling peer into no graph of a prefilling group."""
    hf = json.loads(V4_PRO.read_text())
    vocab, d = hf["vocab_size"], hf["hidden_size"]
    found = built["v4-dp8-mtp2"]["modules"]
    heads = [(m.passes, m.dims) for m in found if m.kind == "lm_head"]
    assert heads == [(0, (vocab, d, 8, vocab // 8)), (2, (vocab, d, 8, 256))]
    rows = [(3, 600 + i, True) for i in range(3)]
    alike = settled(monkeypatch, rows, [(3, 9, False)] * 7)
    beside = settled(monkeypatch, rows, [(3, 9, False)] * 6 + [(1, 16, True)])
    assert (alike.capture_rung, alike.group_prefills) == (4, False)
    assert (beside.capture_rung, beside.group_prefills) == (None, True)


class _Refuses(CostBackend):
    """A step-level backend that prices no step, as a step grid with no point
    near one."""

    @property
    def granularity(self) -> Granularity:
        return Granularity.STEP

    def price(self, batch_view):
        raise CostRefused(batch_view, [Refusal("grid", "no point")])

    def describe(self) -> str:
        return "refuses every step"


def speculative(backend):
    """A runner priced by `backend` under a speculative config."""
    r = runner(backend)
    r.config.speculative_config = SimpleNamespace(synthetic_acceptance_rates=(0.5,))
    return r


def test_a_speculative_step_under_the_roofline_states_its_draft(built):
    roofline = built["qwen3.5-mtp1"]["installed"]
    alone = speculative(roofline)
    assert alone.forward(prefill(8)).predicted_s > 0
    assert alone.refused_commands() == ()
    below = speculative(StepFallbackBackend(_Refuses(), roofline))
    assert below.forward(prefill(8)).predicted_s > 0
    assert below.refused_commands() == ()


def test_a_run_whose_drafter_is_not_laid_out_is_still_refused(built):
    """As an Eagle3 run is (#3962): its drafter is not built, and the roofline
    over the target alone prices no draft."""
    target = [m for m in built["qwen3.5-mtp1"]["modules"] if not m.passes]
    r = speculative(StepRooflineBackend(MACHINE, target))
    assert r.forward(prefill(8)).predicted_s == 0
    assert r.refused_commands() == ("cost:StepRooflineBackend: prices no draft",)
