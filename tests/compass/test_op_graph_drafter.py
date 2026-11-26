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
from test_artifact_store import stanza
from test_backend_step_linear import _installed, view
from test_capture_ir import MAIN, TREE, _ops, _record
from test_dp_step_max import runner
from test_memory_readings import WORKER
from test_op_graph import FLAT, GEMM, MIDDLE, _engine, by_name, g
from test_op_graph_arch import ENGINE, run_readings
from test_work import SPEC

from atom.compass.artifacts import (
    ArtifactStore,
    GateState,
    Kind,
    Topology,
    member_name,
)
from atom.compass.backends import BatchView, CostRefused, RequestShape
from atom.compass.backends.cost import fold_seconds
from atom.compass.backends.op_graph import DRAFT, DRAFT_KV, KV, OpGraphBackend
from atom.compass.backends.op_work import work
from atom.compass.backends.work import Work, gemm, price
from atom.compass.bench.ops import parts
from atom.compass.capture import graph_key, lookup, population_shapes
from atom.compass.capture.forward import Shape, structure
from atom.compass.run import _op_graphs
from atom.compass.runner import projection
from atom.compass.spec import MachineSpec

CONFIG = pathlib.Path(__file__).with_name("qwen3_5_35b_a3b_fp8_config.json")
CONFIG_ARCHITECTURES = json.loads(CONFIG.read_text())["architectures"]
MACHINE = TREE / "tests" / "compass" / "machines" / "mi308x-hjbog39.json"
ATTENTION = "aiter.unified_attention_with_output_base.default"
#: The target's verify of its rows' drafts, from its verify head's ids.
VERIFIER = "atom.model_ops.verify_ids:IdVerifier.launch"
ARGMAX = "aiter.ops.flydsl.topk.topk_per_row_argmax:topk_per_row_argmax"
MOE = "aiter.rocm_aiter_fused_moe.default"
#: Two rows, each verifying one draft and its bonus token.
VERIFY = view((2, 90, True), (2, 9, True), rung=2)
#: A prefill of 64 tokens in one sequence, and a middle chunk of it.
PREFILL, PREFILL_AT = ("--prefill", "64", "1"), "prefill T=64 B=1"
CHUNK = BatchView((RequestShape(64, 64, False),), produces_output=False)
#: A record's step of 3 rows, each verifying one draft, padded to rung 4.
PADDED = {"rows": [[2, 90, True]] * 3, "rung": 4}

_TRACES: dict = {}


def _trace(drafts, dp, tmp_path_factory):
    """`compass trace ops` under an MTP drafter of `drafts` tokens at DP `dp`,
    of every shape this file reads there, in one process, whose runner, built
    once, takes most of its seconds: without DP a decode of width 2; under
    MTP1 a prefill of 64 tokens in one sequence too, and the `--population` of
    a record whose step is `PADDED`, or under DP one whose every rank verifies
    2 rows at rung 2, then beside a peer prefilling 16 tokens. The capture's
    directory and its graphs."""
    if (drafts, dp) in _TRACES:
        return _TRACES[drafts, dp]
    root = tmp_path_factory.mktemp(f"mtp{drafts}")
    (root / "model").mkdir()
    (root / "model" / "config.json").write_bytes(CONFIG.read_bytes())
    argv = ["trace", "ops", "--model", str(root / "model")]
    argv += ["--width", "2"] if dp == 1 else ["--tp", str(dp)]
    if drafts == 1:
        (root / "record").mkdir()
        line = {"rows": [[drafts + 1, 90, True]] * 2, "rung": 2}
        local = 2 * (drafts + 1)
        beside = line | {"rung": None, "moe_rows": local + 16}
        lines = [line | {"moe_rows": dp * local}, beside] if dp > 1 else [PADDED]
        record = _record(root / "record", lines)
        steps = root / "record" / "steps"
        for r in range(1, dp):
            (steps / f"dp{r}.jsonl").write_bytes((steps / "dp0.jsonl").read_bytes())
        argv += [*PREFILL, "--population", record]
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
    graphs = []
    for token in tokens:
        entry = store.read(store.key_from_token(Kind.OP_GRAPH, token))
        graphs.append(json.loads(entry.read_member("graph", rank, "json")))
    _TRACES[drafts, dp] = root, graphs
    return root, graphs


def capture(drafts, tmp_path_factory, at="decode B=2", dp=1):
    """The directory of the capture under an MTP drafter of `drafts` tokens at
    DP `dp` (`_trace`) and its graphs of shape `at`, by structure."""
    root, graphs = _trace(drafts, dp, tmp_path_factory)
    return root, {g["structure"]: g for g in graphs if g["shape"] == at}


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

    # The target runs each row's drafts and its bonus token, and verifies them
    # from its head's ids.
    assert {op["args"]["q"]["shape"][0] for op in ours if op["op"] == ATTENTION} == {
        2 * (drafts + 1)
    }
    assert VERIFIER in {op["op"] for op in ours} - {op["op"] for op in theirs}
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
        """Each leaf priced under its kind, and a resource_bound `Par` as one
        roofline over its branches' summed work: the verify's ids, which the
        drafter's graph opens by copying to the host on a side stream."""
        flat: dict = {}

        def walk(region):
            if "par" in region:
                assert region["join"] == "resource_bound"
                works = (work(op, batch, 1, "bf16", width) for op in _ops(region))
                summed = sum((w for w in works if w is not None), Work())
                flat.setdefault("draft par.resource_bound", []).append(
                    price(summed, machine)
                )
            elif "op" in region:
                if (w := work(region, batch, 1, "bf16", width)) is not None:
                    name = f"draft op.{region['op']}"
                    flat.setdefault(name, []).append(price(w, machine))
            elif "repeat" in region:
                for _ in range(region["count"]):
                    walk(region["repeat"])
            else:
                for item in region["seq"]:
                    walk(item)

        walk(drafter["region"])
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
    assert f"op.{VERIFIER}" in {t.name for t in cost.terms}


def test_a_middle_chunk_is_cut_where_its_draft_kv_pass_starts(tmp_path_factory):
    _, graphs = capture(1, tmp_path_factory, PREFILL_AT)
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
    root, graphs = capture(1, tmp_path_factory, PREFILL_AT)
    [drafter] = [
        graph for key, graph in graphs.items() if key.endswith(" no output draft")
    ]
    cost, drafted = _priced(root, drafter, CHUNK)

    assert {t.name: t.seconds for t in cost.draft_terms} == drafted(1)


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
    ],
    ids=["unverified-row", "uncaptured-drafter"],
)
def test_a_step_under_a_drafter_with_no_draft_to_price_is_refused_by_name(
    batch, drafted, named
):
    def graph(*at, draft=False):
        return drafted if draft else g(1, 8, 8)

    with pytest.raises(CostRefused, match=named):
        OpGraphBackend(SPEC, graph, kv="bf16", drafts=1).price(batch)


def test_a_run_file_op_graph_under_a_drafter_prices_its_drafts(monkeypatch, tmp_path):
    drafter = SimpleNamespace(
        method="mtp", num_speculative_tokens=2, use_dspark=lambda: False
    )
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
        total_seqs_num=width,
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
        sync=SimpleNamespace(
            scheduled_tokens_per_dp_rank=counts,
            ub_tokens_across_dp=None,
            any_rank_needs_eager=False,
            max_final_requests_across_dp=0,
        ),
        is_prefill=False,
        tbo_collective_active=False,
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
    key = lookup(store, f"{target} draft", "decode B=2 G=20")
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
            graph_key(at, shape),
            provenance=stanza("compass trace ops"),
            topology=Topology(),
            conditions=run_readings(config.model),
            gates=GateState.of(),
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
    root, graphs = capture(1, tmp_path_factory, PREFILL_AT, dp=2)
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
        total_seqs_num=1,
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
            scheduled_tokens_per_dp_rank=torch.tensor([64, 64]),
            max_bs_across_dp=1,
            ub_tokens_across_dp=None,
            any_rank_needs_eager=False,
            max_final_requests_across_dp=0,
        ),
        is_prefill=True,
        tbo_collective_active=False,
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


def verify_view(width, rung):
    """One rank's view of a decode of `width` rows, each verifying one draft,
    replaying `rung`, as `batch_view` builds it from the step's mode."""
    batch = SimpleNamespace(
        total_seqs_num=width,
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
    root, traced = _trace(1, 1, tmp_path_factory)
    graphs = {
        (graph["structure"].endswith(" draft"), graph["shape"]): graph
        for graph in traced
        if graph["shape"].endswith(" B=4")
    }
    # The padded step's target ran at its rung: only its drafter is read at its rows.
    assert set(graphs) == {(False, "decode B=4"), (True, "decode B=4")} | {
        (True, "decode T=3 B=4")
    }
    # Its first pass runs the target's stream at the rung's 8 tokens, as
    # ATOM's propose does since upstream b7b5ec3f2; so does the rung's.
    for shape, tokens in (("decode T=3 B=4", 8), ("decode B=4", 8)):
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
    # Its target ran at its rung, its experts on its 6 tokens of the 8 rows.
    target = [{t.name: t.seconds for t in cost.terms} for cost in (padded, full)]
    experts = [terms.pop(f"op.{MOE}") for terms in target]
    assert target[0] == target[1] and experts[0] < experts[1]
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


@pytest.mark.parametrize(
    "batch,halves,shape",
    [
        (VERIFY, ((2,), (2,)), ("decode", 2, 2, True)),
        (FINAL, ((5,), (4,)), ("prefill", 9, 1, True)),
    ],
    ids=["verify", "final-chunk"],
)
def test_a_split_step_prices_its_drafter_after_both_micro_batches_whole(
    batch, halves, shape
):
    # ATOM wraps the target alone in UBatchWrapper: its drafter proposes after
    # both micro-batches, on the step's whole rows.
    asked = []

    def graph(*at, draft=False, micro_batch=False):
        asked.append((*at, draft, micro_batch))
        return g(16, 64, 64) if draft else g(4, 8, 8)

    split = replace(batch, tbo_tokens=halves)
    cost = OpGraphBackend(SPEC, graph, kv="bf16", drafts=1).price(split)

    assert [t.name for t in cost.terms] == ["op.tbo"]
    assert [a[-2:] for a in asked] == [(False, True), (False, True), (True, False)]
    assert asked[-1] == (*shape, True, False)
    assert cost.draft_seconds == price(gemm(16, 64, 64), SPEC)


def test_a_split_step_prices_its_block_drafter_s_draft_kv_pass_whole():
    # The draft-KV pass runs after both micro-batches too, on the step's whole
    # rows, from its graph at the step's shape, among the target's terms.
    asked = []

    def graph(*at, draft=False, micro_batch=False):
        asked.append((*at, draft, micro_batch))
        return g(8, 32, 32) if draft == KV else g(16, 64, 64) if draft else g(4, 8, 8)

    split = replace(FINAL, tbo_tokens=((5,), (4,)))
    cost = OpGraphBackend(SPEC, graph, kv="bf16", drafts=1, block=True).price(split)

    terms = {t.name: t for t in cost.terms}
    assert list(terms) == ["op.tbo", f"{DRAFT_KV}op.{GEMM}"]
    assert ("prefill", 9, 1, True, KV, False) in asked
    kv = terms[f"{DRAFT_KV}op.{GEMM}"]
    assert kv.seconds == price(gemm(8, 32, 32), SPEC)
    assert kv.provenance.detail.endswith("of the graph at prefill T=9 B=1")


def test_a_split_step_s_listed_shapes_hold_every_graph_its_drafter_is_priced_from(
    tmp_path, monkeypatch
):
    # One rank splits 9000 tokens in two rows into micro-batches of 4500; its
    # block drafter and draft-KV pass run past both, on the step's whole rows.
    monkeypatch.setenv("ATOM_TBO_PREFILL_MIN_TOKENS", "8192")
    rows = [[6000, 6000, False], [3000, 3000, False]]
    line = {"step": 0, "is_dummy_run": False, "rows": rows, "rung": None}
    (tmp_path / "steps").mkdir()
    (tmp_path / "run.json").write_text("{}")
    line |= {"produces_output": True}
    (tmp_path / "steps" / "dp0.jsonl").write_text(json.dumps(line) + "\n")
    asked = set()

    def graph(phase, t, b, output, gathered=None, draft=False, width=None, **micro):
        part = {True: DRAFT, KV: DRAFT_KV}.get(draft, "")
        shape = Shape(phase, t, b, gathered, width, **micro).at(1)
        asked.add((part, shape, output or not part, draft))
        return g(4, 8, 8)

    split = BatchView(
        tuple(RequestShape(*r) for r in rows),
        produces_output=True,
        tbo_tokens=((4500,), (4500,)),
    )
    OpGraphBackend(SPEC, graph, kv="bf16", drafts=1, block=True).price(split)

    spec = SimpleNamespace(use_dspark=lambda: True)
    config = SimpleNamespace(enable_tbo=True, speculative_config=spec)
    read = {r for _, reads in parts([str(tmp_path)], config) for r in reads}
    assert asked == read
    whole = Shape("prefill", 9000, 2)
    assert (DRAFT, whole, True, True) in asked
    listed = population_shapes([str(tmp_path)], tbo=True, drafted=True)
    assert {shape for _, shape, *_ in asked} == set(listed)
