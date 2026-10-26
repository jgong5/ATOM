# SPDX-License-Identifier: MIT
"""A drafter's step, traced and priced apart from its target's.

Under ``--method mtp`` `compass trace ops` traces a decode step whose rows
verify the drafter's tokens, and cuts each step that samples where ATOM records
`forward_done_event`: the target's work before the cut, the drafter's after
it, each an `op_graph` of its own. A middle chunk, which samples nothing, is
cut where its draft-KV pass starts. `OpGraphBackend` prices the drafter's graph
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
from test_capture_ir import MAIN, TREE, _ops, _record
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
from atom.compass.capture import forward, population_shapes
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
#: A prefill of 64 tokens in one sequence, and a middle chunk of it.
PREFILL = ("--prefill", "64", "1")
CHUNK = BatchView((RequestShape(64, 64, False),), produces_output=False)

_CAPTURES: dict = {}


def capture(drafts, tmp_path_factory, shape=("--width", "2"), dp=1):
    """`compass trace ops` at `shape` under an MTP drafter of `drafts` tokens,
    or at DP `dp` (`shape` naming `--tp`) or its `--population` of a record whose
    every rank verifies 2 rows at rung 2, then beside a peer prefilling 16 tokens: the capture's directory
    and its graphs by structure, the first shape's."""
    if (drafts, shape, dp) in _CAPTURES:
        return _CAPTURES[drafts, shape, dp]
    root = tmp_path_factory.mktemp(f"mtp{drafts}")
    (root / "model").mkdir()
    (root / "model" / "config.json").write_bytes(CONFIG.read_bytes())
    argv = ["trace", "ops", "--model", str(root / "model"), *shape]
    if dp > 1 and "--tp" not in shape:
        (root / "record").mkdir()
        line = {"rows": [[drafts + 1, 90, True]] * 2, "rung": 2}
        local = 2 * (drafts + 1)
        beside = line | {"rung": None, "moe_rows": local + 16}
        record = _record(root / "record", [line | {"moe_rows": dp * local}, beside])
        steps = root / "record" / "steps"
        for r in range(1, dp):
            (steps / f"dp{r}.jsonl").write_bytes((steps / "dp0.jsonl").read_bytes())
        argv[-len(shape) :] = ["--tp", str(dp), "--population", record]
    argv += ["--store", str(root / "store"), "--", "--method", "mtp"]
    argv += ["--enable-dp-attention"] * (dp > 1)
    argv += ["--num-speculative-tokens", str(drafts)]
    done = subprocess.run(
        [
            sys.executable,
            "-c",
            MAIN,
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
        graphs.setdefault(graph["structure"], graph)
    _CAPTURES[drafts, shape, dp] = root, graphs
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


def _priced(root, drafter, batch, ep=1, dp=1):
    """`batch` priced as a run prices it from the capture at `root` at DP `dp`,
    and its drafter's graph `drafter` priced at it over each EP width, as draft
    terms."""
    text = json.loads(CONFIG.read_text())
    config = SimpleNamespace(
        model=str(root / "model"),
        hf_config=SimpleNamespace(architectures=text["architectures"]),
        tensor_parallel_size=1,
        parallel_config=SimpleNamespace(data_parallel_size=dp),
        enable_expert_parallel=False,
        enforce_eager=True,
        speculative_config=SimpleNamespace(method="mtp", num_speculative_tokens=1),
        **{**_engine(drafter), "enable_dp_attention": dp > 1},
    )
    machine = MachineSpec.from_mapping(json.loads(MACHINE.read_text()))
    lookup = _op_graphs(str(root / "store"), config, "gfx942")
    backend = OpGraphBackend(machine, lookup, kv="bf16", ep=ep, dp=dp, drafts=1)
    cost = backend.price(batch)

    def drafted(width):
        flat: dict = {}
        for op in _ops(drafter["region"]):
            if (w := work(op, batch, 1, "bf16", width)) is not None:
                flat.setdefault(f"draft op.{op['op']}", []).append(price(w, machine))
        return {name: fold_seconds(s) for name, s in flat.items()}

    return cost, drafted


@pytest.mark.parametrize("ep", [1, 8], ids=["ep_off", "ep8"])
def test_a_run_prices_a_verify_step_s_drafter_from_its_own_graph(ep, tmp_path_factory):
    root, graphs = capture(1, tmp_path_factory)
    [drafter] = [graph for key, graph in graphs.items() if key.endswith(" draft")]
    cost, drafted = _priced(root, drafter, VERIFY, ep)

    assert {t.name: t.seconds for t in cost.draft_terms} == drafted(ep)
    # The drafter's experts run over the target's EP ranks, as ATOM builds them.
    moe = f"draft op.{MOE}"
    assert (drafted(ep)[moe] == drafted(1)[moe]) == (ep == 1)
    assert f"op.{SAMPLER}" in {t.name for t in cost.terms}


def test_a_middle_chunk_is_cut_where_its_draft_kv_pass_starts(tmp_path_factory):
    _, graphs = capture(1, tmp_path_factory, PREFILL)
    [middle] = [key for key in graphs if key.endswith(" no output")]
    final = middle.removesuffix(" no output")
    assert set(graphs) == {middle, f"{middle} draft", final, f"{final} draft"}
    ours, theirs, done = (
        _ops(graphs[key]["region"]) for key in (middle, f"{middle} draft", final)
    )
    names = [op["op"] for op in ours + theirs]

    # The drafter's layer takes each of the chunk's tokens, and nothing samples.
    queries = [op["args"]["q"]["shape"][0] for op in theirs if op["op"] == ATTENTION]
    assert queries == [64]
    assert ARGMAX not in names
    # The target holds no drafter pass: its attention is the final chunk's.
    attended = [sum(op["op"] == ATTENTION for op in ops) for ops in (ours, done)]
    assert attended[0] == attended[1]


def test_a_run_prices_a_middle_chunk_s_draft_kv_pass_from_its_own_graph(
    tmp_path_factory,
):
    root, graphs = capture(1, tmp_path_factory, PREFILL)
    [drafter] = [
        graph for key, graph in graphs.items() if key.endswith(" no output draft")
    ]
    cost, drafted = _priced(root, drafter, CHUNK)

    assert {t.name: t.seconds for t in cost.draft_terms} == drafted(1)


def test_a_block_drafter_is_refused_by_name_before_any_step():
    class DSparkProposer:
        is_block_drafter = True

    runner = SimpleNamespace(drafter=DSparkProposer())
    with pytest.raises(ValueError, match="DSparkProposer writes its draft KV before"):
        forward._drafter(runner, None)


FINAL = BatchView((RequestShape(9, 9, False),), produces_output=True)


@pytest.mark.parametrize(
    "batch,shape",
    [
        (VERIFY, ("decode", 2, 2, True)),
        (FINAL, ("prefill", 9, 1, True)),
        (MIDDLE, ("prefill", 9, 1, False)),
    ],
    ids=["verify", "final-chunk", "middle-chunk"],
)
def test_a_step_reads_its_drafter_s_graph_as_draft_terms(batch, shape):
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
        (VERIFY, None, "over 2 sequences for its drafter"),
        (
            replace(VERIFY, tbo_tokens=((2,), (2,))),
            g(1, 8, 8),
            "a two-batch-overlap step under a drafter",
        ),
    ],
    ids=["unverified-row", "uncaptured-drafter", "two-batch-overlap"],
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


def dp_view(width, rung, dp, counts=None):
    """Rank 0's view of a decode of `width` rows, each verifying one draft, at
    `rung`, every rank of `dp` alike or at its tokens in `counts`, as
    `batch_view` builds it from the step's mode."""
    batch = SimpleNamespace(
        total_seqs_num_prefill=0,
        num_scheduled_tokens=[2] * width,
        context_lens=[90] * width,
        req_ids=list(range(width)),
        produces_output=lambda: True,
        is_dummy_run=False,
    )
    r = runner(None, dp_size=dp)
    r.drafter = SimpleNamespace(is_block_drafter=False, mtp_k=1)
    counts = torch.tensor(counts or [2 * width] * dp)
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


def test_a_dp_drafter_beside_a_prefilling_peer_gathers_the_step_s_rows(
    tmp_path_factory,
):
    root, graphs = capture(1, tmp_path_factory, dp=2)
    [target] = [key for key in graphs if not key.endswith(" draft")]
    store, rank = ArtifactStore(str(root / "store")), Topology().ranks()[0]
    key = Key.of(Kind.OP_GRAPH, structure=f"{target} draft", shape="decode B=2 G=20")
    drafter = json.loads(store.read(key).read_member("graph", rank, "json"))

    # Its first pass gathers this rank's 4 verified tokens and the peer's 16.
    first = next(op for op in _ops(drafter["region"]) if op["op"].endswith("gatherv"))
    assert sum(first["args"]["sizes"]) == 20


def _lookup(tmp_path, dp, stored, *lazily):
    """`_op_graphs` at DP `dp` under MTP1 over a store of `stored`'s regions,
    each keyed by its phase, shape and whether it is the drafter's; `lazily`,
    its ATOM arguments and lazy counts."""
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "config.json").write_text("{}")
    config = SimpleNamespace(
        model=str(tmp_path / "model"),
        hf_config=SimpleNamespace(architectures=CONFIG_ARCHITECTURES),
        tensor_parallel_size=1,
        parallel_config=SimpleNamespace(data_parallel_size=dp),
        enable_expert_parallel=False,
        enforce_eager=True,
        speculative_config=SimpleNamespace(method="mtp", num_speculative_tokens=1),
        **ENGINE,
    )
    store, rank = ArtifactStore(str(tmp_path / "store")), Topology().ranks()[0]
    for (phase, shape, draft), region in stored.items():
        at = structure(config, phase, True, "gfx950", True, draft)
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
    return _op_graphs(store.root, config, "gfx950", *lazily)


def test_a_dp_verify_step_reads_its_drafter_at_its_rows_padded_too(tmp_path):
    # Rung 4 on 8 ranks: the target gathers every rank's 8 padded tokens, the
    # drafter's first pass every rank's own.
    lookup = _lookup(
        tmp_path,
        8,
        {
            ("decode", "decode B=4", False): g(4, 64, 64),
            ("decode", "decode B=4", True): g(4096 * 64, 128, 128),
            ("decode", "decode B=4 G=50", True): g(4096 * 50, 128, 128),
            ("decode", "decode B=4 G=48", True): g(4096 * 48, 128, 128),
            ("decode", "decode T=3 B=4", True): g(4096 * 47, 128, 128),
        },
    )
    backend = OpGraphBackend(SPEC, lookup, kv="bf16", dp=8, drafts=1, price_list=FLAT)
    at_rung = dp_view(4, 4, dp=8, counts=[8] + [6] * 7)
    padded = dp_view(3, 4, dp=8)

    assert (at_rung.moe_rows, at_rung.draft_moe_rows) == (64, 50)
    # The price list holds the target's GEMM at `decode B=4`, no rows named.
    assert by_name(backend.price(at_rung)) == {
        f"op.{GEMM}.price_list": pytest.approx(7e-6),
        f"draft op.{GEMM}": price(gemm(4096 * 50, 128, 128), SPEC),
    }
    # Padded, its drafter is its 3 rows' at rung 4, every rank's 6 tokens its
    # default G; `decode B=4 G=48`'s first pass would trace 8.
    assert (padded.moe_rows, padded.draft_moe_rows) == (64, 48)
    cost = backend.price(padded)
    assert by_name(cost)[f"draft op.{GEMM}"] == price(gemm(4096 * 47, 128, 128), SPEC)
    assert cost.draft_terms[-1].provenance.detail.endswith("decode T=3 B=4 G=48")


def test_a_dp_prefill_under_a_drafter_reads_its_graphs_at_their_default(tmp_path):
    # A prefill's default G is every rank's tokens: its rows verify no drafts.
    lookup = _lookup(
        tmp_path,
        2,
        {
            ("prefill", "prefill T=9 B=1", False): g(9, 64, 64),
            ("prefill", "prefill T=9 B=1", True): g(9, 128, 128),
        },
    )
    step = replace(FINAL, moe_rows=18, draft_moe_rows=18)

    cost = OpGraphBackend(SPEC, lookup, kv="bf16", dp=2, drafts=1).price(step)

    assert (cost.target_seconds, cost.draft_seconds) == (
        price(gemm(9, 64, 64), SPEC),
        price(gemm(9, 128, 128), SPEC),
    )


def test_a_dp_middle_chunk_is_cut_where_its_drafter_s_propose_starts(
    tmp_path_factory,
):
    root, graphs = capture(1, tmp_path_factory, (*PREFILL, "--tp", "2"), dp=2)
    [middle] = [key for key in graphs if key.endswith(" no output")]
    final = middle.removesuffix(" no output")
    assert set(graphs) == {middle, f"{middle} draft", final, f"{final} draft"}
    ours, theirs, done = (
        _ops(graphs[key]["region"]) for key in (middle, f"{middle} draft", final)
    )
    # Under DP ATOM runs the drafter's propose in place of its draft-KV pass:
    # one pass over the chunk's tokens, and the draft it drops.
    queries = [op["args"]["q"]["shape"][0] for op in theirs if op["op"] == ATTENTION]
    assert queries == [64]
    assert ARGMAX in {op["op"] for op in theirs}
    # The target holds no drafter pass: its attention is the final chunk's.
    attended = [sum(op["op"] == ATTENTION for op in ops) for ops in (ours, done)]
    assert attended[0] == attended[1]

    batch = SimpleNamespace(
        total_seqs_num_prefill=1,
        num_scheduled_tokens=[64],
        context_lens=[64],
        req_ids=[0],
        produces_output=lambda: False,
        is_dummy_run=False,
    )
    r = runner(None, dp_size=2)
    r.drafter = SimpleNamespace(is_block_drafter=False, mtp_k=1)
    mode = SimpleNamespace(
        sync=SimpleNamespace(
            num_tokens_across_dp=torch.tensor([64, 64]),
            max_bs_across_dp=1,
            ub_tokens_across_dp=None,
        ),
        running_tokens_are_unified=False,
        running_tokens=64,
        running_bs=1,
        use_cudagraph=False,
    )
    step = projection.batch_view(batch, mode, r)
    cost, drafted = _priced(root, graphs[f"{middle} draft"], step, dp=2)

    assert cost.draft_terms[-1].provenance.detail.endswith(
        "prefill T=64 B=1 G=128 that produces no output"
    )
    assert cost.target_seconds > 0
    assert {t.name: t.seconds for t in cost.draft_terms} == drafted(1)


# --- a decode padded to its rung ----------------------------------------------


def test_a_population_names_a_padded_decode_s_drafter_at_its_rows(tmp_path):
    drafted = {"rows": [[2, 90, True]] * 3, "rung": 4}
    record = _record(tmp_path, [drafted, drafted | {"rows": [[1, 90, True]] * 3}])

    # Without a drafter, no step after the cut to read at its rows.
    shapes = ["decode B=4", "decode T=3 B=4", "decode B=4"]
    assert list(map(str, population_shapes([record]))) == shapes


def _population(tmp_path_factory, line, drafts):
    """`compass trace ops --population` of a record of one step, `line`, under
    an MTP drafter of `drafts` tokens: the capture's directory and its graphs
    by whether each is the drafter's and its shape."""
    root = tmp_path_factory.mktemp(f"padded{drafts}")
    (root / "model").mkdir()
    (root / "model" / "config.json").write_bytes(CONFIG.read_bytes())
    (root / "record").mkdir()
    argv = ["trace", "ops", "--model", str(root / "model"), "--population"]
    argv += [_record(root / "record", [line]), "--store", str(root / "store")]
    argv += ["--", "--method", "mtp", "--num-speculative-tokens", str(drafts)]
    done = subprocess.run(
        [
            sys.executable,
            "-c",
            MAIN,
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
        graphs[graph["structure"].endswith(" draft"), graph["shape"]] = graph
    return root, graphs


def verify_view(width, rung):
    """One rank's view of a decode of `width` rows, each verifying one draft,
    replaying `rung`, as `batch_view` builds it from the step's mode."""
    batch = SimpleNamespace(
        total_seqs_num_prefill=0,
        num_scheduled_tokens=[2] * width,
        context_lens=[90] * width,
        req_ids=list(range(width)),
        produces_output=lambda: True,
    )
    r = runner(None)
    r.drafter = SimpleNamespace(is_block_drafter=False)
    mode = SimpleNamespace(
        sync=None, running_bs=rung, running_tokens=2 * rung, use_cudagraph=True
    )
    return projection.batch_view(batch, mode, r)


def test_a_decode_padded_to_its_rung_prices_its_drafter_at_its_rows(
    tmp_path_factory,
):
    line = {"rows": [[2, 90, True]] * 3, "rung": 4}
    root, graphs = _population(tmp_path_factory, line, 1)
    # The padded step's target ran at its rung: only its drafter is read at its rows.
    assert set(graphs) == {(False, "decode B=4"), (True, "decode B=4")} | {
        (True, "decode T=3 B=4")
    }
    # Its first pass runs the rows' 6 tokens; the rung's traces 8.
    for shape, tokens in (("decode T=3 B=4", 6), ("decode B=4", 8)):
        ops = _ops(graphs[True, shape]["region"])
        assert {op["args"]["q"]["shape"][0] for op in ops if op["op"] == ATTENTION} == {
            tokens
        }

    config = SimpleNamespace(
        model=str(root / "model"),
        hf_config=SimpleNamespace(architectures=CONFIG_ARCHITECTURES),
        tensor_parallel_size=1,
        parallel_config=SimpleNamespace(data_parallel_size=1),
        enable_expert_parallel=False,
        enforce_eager=True,
        speculative_config=SimpleNamespace(method="mtp", num_speculative_tokens=1),
        **_engine(graphs[True, "decode B=4"]),
    )
    machine = MachineSpec.from_mapping(json.loads(MACHINE.read_text()))
    lookup = _op_graphs(str(root / "store"), config, "gfx942")
    backend = OpGraphBackend(machine, lookup, kv="bf16", drafts=1)
    padded, full = backend.price(verify_view(3, 4)), backend.price(verify_view(4, 4))

    assert padded.draft_terms[-1].provenance.detail.endswith("decode T=3 B=4")
    assert padded.target_seconds == full.target_seconds
    # The rung's drafter, which this step read before, prices more than runs.
    assert 0 < padded.draft_seconds < full.draft_seconds


def test_a_padded_drafter_s_later_pass_runs_and_dispatches_at_the_rung(tmp_path):
    from test_capture_v4_pro import DPA_TP8, EP8, MORI, trace

    # DeepSeek-V4-Pro under MTP2 on 8 ranks, EP8: rank 0 verifies 3 rows at
    # rung 4, every peer 4, so the drafter's first pass gathers 9 + 7 x 12 rows.
    (tmp_path / "record").mkdir()
    line = {"rows": [[3, 90, True]] * 3, "rung": 4, "moe_rows": 96}
    record = _record(tmp_path / "record", [line | {"draft_moe_rows": 93}])
    for r in range(1, 8):
        idle = {"step": 0, "is_dummy_run": True}
        (tmp_path / "record" / "steps" / f"dp{r}.jsonl").write_text(json.dumps(idle))
    argv = (
        f"--population {record}{DPA_TP8}{EP8} --method mtp --num-speculative-tokens 2"
    )
    (tmp_path / "trace").mkdir()
    done = trace(tmp_path / "trace", argv)

    tokens = re.findall(r"^published .* as (\w+)$", done.stdout, re.MULTILINE)
    store, rank = (
        ArtifactStore(str(tmp_path / "trace" / "store")),
        Topology().ranks()[0],
    )
    graphs = {}
    for token in tokens:
        entry = store.read(store.key_from_token(Kind.OP_GRAPH, token))
        graph = json.loads(entry.read_member("graph", rank, "json"))
        graphs[graph["shape"]] = graph
    drafter = graphs["decode T=3 B=4 G=93"]
    assert drafter["structure"].endswith(" draft"), done.stderr[-4000:]
    # Its first pass sends its 9 tokens; its second, padded to the rung, 4 rows
    # from every rank, though the step's G is the first pass's.
    dispatched = [
        (op["args"]["input"]["shape"][0], op["out"][0][0])
        for op in _ops(drafter["region"])
        if op["op"] == f"{MORI}.dispatch"
    ]
    assert dispatched == [(9, 93), (4, 32)]
