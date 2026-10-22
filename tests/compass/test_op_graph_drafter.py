# SPDX-License-Identifier: MIT
"""A drafter's step, traced and priced apart from its target's.

Under ``--method mtp`` `compass trace ops` traces a decode step whose rows
verify the drafter's tokens, and cuts each step that samples where ATOM records
`forward_done_event`: the target's work before the cut, the drafter's after
it, each an `op_graph` of its own. `OpGraphBackend` prices the drafter's graph
as terms tagged draft. Traced on Qwen3.5-35B-A3B-FP8's published config.
"""

import json
import os
import pathlib
import re
import subprocess
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from test_artifact_store import CONDITIONS, GATES, stanza
from test_backend_step_linear import _installed, view
from test_capture_ir import TREE, _ops, _record
from test_dp_step_max import runner
from test_memory_readings import WORKER
from test_op_graph import FLAT, GEMM, MIDDLE, _engine, by_name, g
from test_op_graph_arch import ENGINE
from test_work import SPEC

from atom.compass.artifacts import ArtifactStore, Key, Kind, Topology, member_name
from atom.compass.backends import BatchView, CostRefused, RequestShape
from atom.compass.backends.cost import fold_seconds
from atom.compass.backends.op_graph import OpGraphBackend
from atom.compass.backends.op_work import work
from atom.compass.backends.work import gemm, price
from atom.compass.capture import population_shapes
from atom.compass.capture.forward import structure
from atom.compass.run import _op_graphs
from atom.compass.runner import projection
from atom.compass.spec import MachineSpec

CONFIG = pathlib.Path(__file__).with_name("qwen3_5_35b_a3b_fp8_config.json")
CONFIG_ARCHITECTURES = json.loads(CONFIG.read_text())["architectures"]
MACHINE = TREE / "scripts" / "compass" / "machines" / "mi308x-hjbog39.json"
ATTENTION = "aiter.unified_attention_with_output_base.default"
SAMPLER = "atom.model_ops.rejection_sampler:rejection_sample"
ARGMAX = "aiter.ops.flydsl.topk.topk_per_row_argmax:topk_per_row_argmax"
MOE = "aiter.rocm_aiter_fused_moe.default"
#: Two rows, each verifying one draft and its bonus token.
VERIFY = view((2, 90, True), (2, 9, True), rung=2)

_CAPTURES: dict = {}


def capture(drafts, tmp_path_factory, dp=1):
    """`compass trace ops --width 2` under an MTP drafter of `drafts` tokens,
    or at DP `dp` `--population` of a record whose every rank verifies 2 rows
    at rung 2: the capture's directory and its graphs by structure."""
    if (drafts, dp) in _CAPTURES:
        return _CAPTURES[drafts, dp]
    root = tmp_path_factory.mktemp(f"mtp{drafts}")
    (root / "model").mkdir()
    (root / "model" / "config.json").write_bytes(CONFIG.read_bytes())
    argv = ["trace", "ops", "--model", str(root / "model"), "--width", "2"]
    if dp > 1:
        (root / "record").mkdir()
        line = {"rows": [[drafts + 1, 90, True]] * 2, "rung": 2}
        record = _record(root / "record", [line | {"moe_rows": dp * 2 * (drafts + 1)}])
        steps = root / "record" / "steps"
        for r in range(1, dp):
            (steps / f"dp{r}.jsonl").write_bytes((steps / "dp0.jsonl").read_bytes())
        argv[-2:] = ["--tp", str(dp), "--population", record]
    argv += ["--store", str(root / "store"), "--", "--method", "mtp"]
    argv += ["--enable-dp-attention"] * (dp > 1)
    argv += ["--num-speculative-tokens", str(drafts)]
    done = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from atom.compass.cli import main; main(sys.argv[1:])",
            *argv,
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=1800,
        cwd=TREE,
        env={**os.environ, "PYTHONPATH": str(TREE)},
    )
    tokens = re.findall(r"^published .* as (\w+)$", done.stdout, re.MULTILINE)
    assert tokens, f"rc {done.returncode}\n{done.stderr[-4000:]}"
    store, rank = ArtifactStore(str(root / "store")), Topology().ranks()[0]
    graphs = {}
    for token in tokens:
        entry = store.read(store.key_from_token(Kind.OP_GRAPH, token))
        graph = json.loads(entry.read_member("graph", rank, "json"))
        graphs[graph["structure"]] = graph
    _CAPTURES[drafts, dp] = root, graphs
    return root, graphs


@pytest.mark.parametrize("drafts", [1, 2])
def test_a_verify_step_is_cut_at_forward_done_event_into_target_and_drafter(
    drafts, tmp_path_factory
):
    _, graphs = capture(drafts, tmp_path_factory)
    [target] = [key for key in graphs if not key.endswith(" draft")]
    assert set(graphs) == {target, f"{target} draft"}
    assert f'"speculative":["mtp",{drafts}]' in target
    ours, theirs = (_ops(graphs[key]["region"]) for key in (target, f"{target} draft"))
    vocab = json.loads(CONFIG.read_text())["text_config"]["vocab_size"]

    # The target verifies each row's drafts and its bonus token, and samples.
    assert {op["args"]["q"]["shape"][0] for op in ours if op["op"] == ATTENTION} == {
        2 * (drafts + 1)
    }
    assert SAMPLER in {op["op"] for op in ours} - {op["op"] for op in theirs}
    # The drafter runs its layer once per draft, each through the shared
    # embedding and LM head.
    names = [op["op"] for op in theirs]
    assert names.count(ATTENTION) == names.count(ARGMAX) == drafts
    tables = [op for op in theirs if op["op"] == "aten.embedding.default"]
    assert [op["args"]["weight"]["shape"][0] for op in tables] == [vocab] * drafts


@pytest.mark.parametrize("ep", [1, 8], ids=["ep_off", "ep8"])
def test_a_run_prices_a_verify_step_s_drafter_from_its_own_graph(ep, tmp_path_factory):
    root, graphs = capture(1, tmp_path_factory)
    [drafter] = [graph for key, graph in graphs.items() if key.endswith(" draft")]
    text = json.loads(CONFIG.read_text())
    config = SimpleNamespace(
        model=str(root / "model"),
        hf_config=SimpleNamespace(architectures=text["architectures"]),
        tensor_parallel_size=1,
        parallel_config=SimpleNamespace(data_parallel_size=1),
        enable_expert_parallel=False,
        enforce_eager=True,
        speculative_config=SimpleNamespace(method="mtp", num_speculative_tokens=1),
        **_engine(drafter),
    )
    machine = MachineSpec.from_mapping(json.loads(MACHINE.read_text()))
    lookup = _op_graphs(str(root / "store"), config, "gfx942")
    cost = OpGraphBackend(machine, lookup, kv="bf16", ep=ep, drafts=1).price(VERIFY)

    def drafted(width):
        flat: dict = {}
        for op in _ops(drafter["region"]):
            if (w := work(op, VERIFY, 1, "bf16", width)) is not None:
                flat.setdefault(f"draft op.{op['op']}", []).append(price(w, machine))
        return {name: fold_seconds(s) for name, s in flat.items()}

    assert {t.name: t.seconds for t in cost.terms if t.draft} == drafted(ep)
    # The drafter's experts run over the target's EP ranks, as ATOM builds them.
    moe = f"draft op.{MOE}"
    assert (drafted(ep)[moe] == drafted(1)[moe]) == (ep == 1)
    assert f"op.{SAMPLER}" in {t.name for t in cost.terms if not t.draft}


FINAL = BatchView((RequestShape(9, 9, False),), produces_output=True)


@pytest.mark.parametrize(
    "batch,shape",
    [(VERIFY, ("decode", 2, 2, True)), (FINAL, ("prefill", 9, 1, True))],
    ids=["verify", "final-chunk"],
)
def test_a_step_that_samples_reads_its_drafter_s_graph_as_draft_terms(batch, shape):
    asked = []

    def graph(*at, draft=False):
        asked.append((*at, draft))
        return g(16, 64, 64) if draft else g(4, 8, 8)

    cost = OpGraphBackend(SPEC, graph, kv="bf16", drafts=1).price(batch)

    assert asked == [(*shape, False), (*shape, True)]
    assert (cost.target_seconds, cost.draft_seconds) == (
        price(gemm(4, 8, 8), SPEC),
        price(gemm(16, 64, 64), SPEC),
    )


@pytest.mark.parametrize(
    "batch,drafted,named",
    [
        (view((1, 9, True)), g(1, 8, 8), "a step that decodes a row of other than 2"),
        (MIDDLE, g(1, 8, 8), "a step under a drafter that produces no output"),
        (VERIFY, None, "over 2 sequences for its drafter"),
        (
            replace(VERIFY, tbo_tokens=((2,), (2,))),
            g(1, 8, 8),
            "a two-batch-overlap step under a drafter",
        ),
    ],
    ids=["unverified-row", "middle-chunk", "uncaptured-drafter", "two-batch-overlap"],
)
def test_a_step_under_a_drafter_with_no_draft_to_price_is_refused_by_name(
    batch, drafted, named
):
    def graph(*at, draft=False):
        return drafted if draft else g(1, 8, 8)

    with pytest.raises(CostRefused, match=named):
        OpGraphBackend(SPEC, graph, kv="bf16", drafts=1).price(batch)


def test_a_run_file_op_graph_under_a_drafter_prices_its_drafts(monkeypatch, tmp_path):
    drafter = SimpleNamespace(method="mtp", num_speculative_tokens=2)
    monkeypatch.setitem(WORKER, "speculative_config", drafter)

    assert _installed(monkeypatch, tmp_path, op_graph=True).drafts == 2


def test_a_population_captures_a_verify_step_at_its_width(tmp_path):
    record = _record(tmp_path, [{"rows": [[2, 90, True]] * 2, "rung": 2}])

    assert list(map(str, population_shapes([record]))) == ["decode B=2"]


# --- a data-parallel verify step ---------------------------------------------


def dp_view(width, rung, dp):
    """Rank 0's view of a decode of `width` rows, each verifying one draft, at
    `rung`, every rank of `dp` alike, as `batch_view` builds it from the
    step's mode."""
    batch = SimpleNamespace(
        total_seqs_num_prefill=0,
        num_scheduled_tokens=[2] * width,
        context_lens=[90] * width,
        req_ids=list(range(width)),
        produces_output=lambda: True,
    )
    r = runner(None, dp_size=dp)
    r.drafter = SimpleNamespace(is_block_drafter=False)
    counts = torch.tensor([2 * width] * dp)
    mode = SimpleNamespace(
        sync=SimpleNamespace(num_tokens_across_dp=counts, ub_tokens_across_dp=None),
        running_tokens_are_unified=True,
        running_tokens=2 * rung,
        running_bs=rung,
        use_cudagraph=True,
    )
    return projection.batch_view(batch, mode, r)


def test_a_balanced_dp_verify_step_prices_from_its_population_s_graphs(
    tmp_path_factory,
):
    root, graphs = capture(1, tmp_path_factory, dp=2)
    [target] = [key for key in graphs if not key.endswith(" draft")]
    drafter = graphs[f"{target} draft"]
    config = SimpleNamespace(
        model=str(root / "model"),
        hf_config=SimpleNamespace(architectures=CONFIG_ARCHITECTURES),
        tensor_parallel_size=1,
        parallel_config=SimpleNamespace(data_parallel_size=2),
        enable_expert_parallel=False,
        enforce_eager=True,
        speculative_config=SimpleNamespace(method="mtp", num_speculative_tokens=1),
        **{**_engine(drafter), "enable_dp_attention": True},
    )
    machine = MachineSpec.from_mapping(json.loads(MACHINE.read_text()))
    lookup = _op_graphs(str(root / "store"), config, "gfx942")
    step = dp_view(2, 2, dp=2)

    cost = OpGraphBackend(machine, lookup, kv="bf16", dp=2, drafts=1).price(step)

    assert cost.terms[0].provenance.detail.endswith("decode T=2 B=2 G=8")
    assert cost.target_seconds > 0 and cost.draft_seconds > 0
    # The drafter's first pass gathers every rank's verified tokens.
    first = next(op for op in _ops(drafter["region"]) if op["op"].endswith("gatherv"))
    assert sum(first["args"]["sizes"]) == step.draft_moe_rows == 8


def test_a_padded_dp_verify_step_reads_its_drafter_at_the_drafter_s_rows(tmp_path):
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "config.json").write_text("{}")
    config = SimpleNamespace(
        model=str(tmp_path / "model"),
        hf_config=SimpleNamespace(architectures=CONFIG_ARCHITECTURES),
        tensor_parallel_size=1,
        parallel_config=SimpleNamespace(data_parallel_size=8),
        enable_expert_parallel=False,
        enforce_eager=True,
        speculative_config=SimpleNamespace(method="mtp", num_speculative_tokens=1),
        **ENGINE,
    )
    # 3 rows at rung 4: the target gathers every rank's 8 padded tokens, the
    # drafter's first pass every rank's 6.
    stored = {
        ("decode B=4", False): g(4, 64, 64),
        ("decode B=4", True): g(4096 * 64, 128, 128),
        ("decode B=4 G=48", True): g(4096 * 48, 128, 128),
    }
    store, rank = ArtifactStore(str(tmp_path / "store")), Topology().ranks()[0]
    for (shape, draft), region in stored.items():
        at = structure(config, "decode", True, "gfx950", True, draft)
        store.publish(
            Key.of(Kind.OP_GRAPH, structure=at, shape=shape),
            provenance=stanza("compass trace ops"),
            topology=Topology(),
            conditions=CONDITIONS,
            gates=GATES,
            members={
                member_name("graph", rank, "json"): json.dumps(
                    {"region": region}
                ).encode()
            },
        )
    lookup = _op_graphs(store.root, config, "gfx950")
    backend = OpGraphBackend(SPEC, lookup, kv="bf16", dp=8, drafts=1, price_list=FLAT)
    step = dp_view(3, 4, dp=8)

    assert (step.moe_rows, step.draft_moe_rows) == (64, 48)
    # The price list holds the target's GEMM at `decode B=4`, no rows named.
    assert by_name(backend.price(step)) == {
        f"op.{GEMM}.price_list": pytest.approx(7e-6),
        f"draft op.{GEMM}": price(gemm(4096 * 48, 128, 128), SPEC),
    }
