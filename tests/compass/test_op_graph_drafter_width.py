# SPDX-License-Identifier: MIT
"""A drafter's later passes, read at the batch ATOM runs them at.

`EagleProposer.propose` runs every pass after its first at the step's
`running_bs` when the drafter declares a draft graph, on every step:
`ForwardMode.decide` settles it from the ladder and the DP group whether or
not the target replays. Each step's mode here is `decide`'s own.
"""

import json
import re
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_capture_ir import _ops, _record
from test_dp_step_max import runner
from test_op_graph import g
from test_op_graph_drafter import _lookup
from test_work import SPEC

from atom.compass.artifacts import ArtifactStore, Kind, Topology
from atom.compass.backends.op_graph import OpGraphBackend
from atom.compass.backends.resolver import CostRefused
from atom.compass.backends.work import gemm, price
from atom.compass.capture import population_shapes
from atom.compass.runner import projection
from atom.utils.forward_context import ForwardMode
from atom.utils.tbo import ubatching

LADDER = [1, 2, 4, 8]
#: Three rows, each verifying two drafts and its bonus token.
DECODE = [(3, 90, True)] * 3


def settled(monkeypatch, rows, peers=(), eager=False, step=True, output=True):
    """Rank 0's view of a step of `rows`, each ``(query, context, decode)``,
    under an MTP2 drafter beside DP `peers`, each ``(seqs, tokens,
    prefilling)``: its mode `ForwardMode.decide`'s on `LADDER`, or on ATOM's
    eager one, `[0]`; `step` that the drafter declares a draft graph, `output`
    that the step samples."""
    queries = [q for q, _, _ in rows]
    prefill = sum(not d for _, _, d in rows)
    batch = SimpleNamespace(
        total_seqs_num=len(rows),
        total_seqs_num_prefill=prefill,
        total_tokens_num=sum(queries),
        total_tokens_num_prefill=sum(q for q, _, d in rows if not d),
        num_scheduled_tokens=queries,
        context_lens=[c for _, c, _ in rows],
        req_ids=list(range(len(rows))),
        produces_output=lambda: output,
        is_dummy_run=False,
    )

    def sync(**_):
        counts = [sum(queries), *(t for _, t, _ in peers)]
        return SimpleNamespace(
            max_bs_across_dp=max([len(rows), *(b for b, _, _ in peers)]),
            any_rank_has_prefill=bool(prefill) or any(p for *_, p in peers),
            any_rank_needs_eager=False,
            max_seqlen_q_across_dp=None,
            tbo_collective_active=False,
            num_tokens_across_dp=torch.tensor(counts),
            ub_tokens_across_dp=None,
        )

    monkeypatch.setattr(ubatching, "sync_dp_metadata", sync)
    mode = ForwardMode.decide(
        batch=batch,
        dp_size=1 + len(peers),
        dp_group=None,
        enforce_eager=eager,
        step_needs_eager=False,
        capture_sizes=np.array([0] if eager else LADDER, dtype=np.int32),
        captured_tokens=None,
        is_block_drafter=False,
        tbo_on=False,
        local_tbo=(False, False, 0, 0),
        max_seqlen_q=3,
    )
    r = runner(None, dp_size=1 + len(peers))
    r.drafter = SimpleNamespace(is_block_drafter=False, step=object() if step else None)
    return projection.batch_view(batch, mode, r)


@pytest.mark.parametrize(
    "rows,peers,eager,step,drafter,width",
    [
        ([(3, 3, False)] * 3, (), False, True, ("prefill", 9, 3, True, 4), 4),
        (DECODE, [(1, 16, True)], False, True, ("decode", 3, 4, True, 25, None), 4),
        (DECODE, [(4, 12, False)], True, True, ("decode", 3, 4, True, 21, None), 4),
        (DECODE, [(4, 12, False)], True, False, ("decode", 3, 3, True, 21, None), 3),
    ],
    ids=["prefill", "beside-a-prefilling-peer", "eager-dp", "no-draft-graph"],
)
def test_a_drafter_s_later_passes_read_its_graph_at_the_step_s_running_bs(
    monkeypatch, rows, peers, eager, step, drafter, width
):
    asked = []

    def graph(*at, draft=False, width=None):
        asked.append((*at, width) if draft else at)
        # A drafter's graph at a wider batch runs a wider later pass.
        return g(1000 * (width or at[2]), 64, 64) if draft else g(4, 8, 8)

    view = settled(monkeypatch, rows, peers, eager, step)
    backend = OpGraphBackend(SPEC, graph, kv="bf16", dp=1 + len(peers), drafts=2)
    cost = backend.price(view)

    # No target replays a graph, yet the later passes pad to the step's batch.
    assert view.capture_rung is None
    assert view.draft_bs == (4 if step else None)
    assert asked[-1] == drafter
    assert cost.draft_seconds == price(gemm(1000 * width, 64, 64), SPEC)


@pytest.mark.parametrize(
    "peers,drafter,width",
    [
        ([(4, 12, True)], ("prefill", 9, 3, False, 21, 4), 4),
        ((), ("prefill", 9, 3, False, None), 3),
    ],
    ids=["dp", "one-rank"],
)
def test_a_middle_chunk_s_drafter_runs_its_later_passes_at_running_bs_under_dp(
    monkeypatch, peers, drafter, width
):
    asked = []

    def graph(*at, draft=False, width=None):
        asked.append((*at, width) if draft else at)
        return g(1000 * (width or at[2]), 64, 64) if draft else g(4, 8, 8)

    # Under DP ATOM runs the drafter's propose for its collectives; on one rank
    # only its draft-KV pass, the first pass's work.
    view = settled(monkeypatch, [(3, 3, False)] * 3, peers, output=False)
    backend = OpGraphBackend(SPEC, graph, kv="bf16", dp=1 + len(peers), drafts=2)
    cost = backend.price(view)

    assert view.draft_bs == 4
    assert asked[-1] == drafter
    assert cost.draft_seconds == price(gemm(1000 * width, 64, 64), SPEC)


def test_a_population_names_a_drafter_s_shape_at_the_batch_its_later_passes_ran(
    tmp_path,
):
    prefill = {"rows": [[3, 3, False]] * 3, "produces_output": True}
    record = _record(
        tmp_path,
        [
            {"rows": [[3, 90, True]] * 3, "draft_bs": 4},
            prefill | {"draft_bs": 4},
            # One rank's middle chunk runs no later pass; a batch at its rows pads none.
            prefill | {"produces_output": False, "draft_bs": 4},
            prefill | {"draft_bs": 3},
        ],
    )

    assert list(map(str, population_shapes([record]))) == [
        "decode B=3",
        "decode T=3 B=4",
        "prefill T=9 B=3",
        "prefill T=9 B=3 W=4",
        "prefill T=9 B=3",
        "prefill T=9 B=3",
    ]


def test_the_run_s_lookup_reads_a_prefill_s_drafter_at_its_width(monkeypatch, tmp_path):
    lookup = _lookup(
        tmp_path,
        1,
        {
            ("prefill", "prefill T=9 B=3", False): g(9, 64, 64),
            ("prefill", "prefill T=9 B=3", True): g(3000, 64, 64),
            ("prefill", "prefill T=9 B=3 W=4", True): g(4000, 64, 64),
        },
    )
    view = settled(monkeypatch, [(3, 3, False)] * 3)

    cost = OpGraphBackend(SPEC, lookup, kv="bf16", drafts=1).price(view)

    assert cost.terms[-1].provenance.detail.endswith("prefill T=9 B=3 W=4")
    assert cost.draft_seconds == price(gemm(4000, 64, 64), SPEC)


def test_a_run_traces_a_prefill_s_drafter_no_graph_holds_at_its_width(
    monkeypatch, tmp_path
):
    from atom.compass.bench import ops

    traced, lazy = [], {"lazy_traces": 0, "lazy_trace_wall_seconds": 0}
    monkeypatch.setattr(ops, "_child", lambda *a: traced.append(a[-1]))
    stored = {
        ("prefill", "prefill T=9 B=3", False): g(9, 64, 64),
        ("prefill", "prefill T=9 B=3", True): g(3000, 64, 64),
    }
    lookup = _lookup(tmp_path, 1, stored, ["--model", "m"], lazy)
    view = settled(monkeypatch, [(3, 3, False)] * 3)

    with pytest.raises(CostRefused, match="its later passes at 4"):
        OpGraphBackend(SPEC, lookup, kv="bf16", drafts=1).price(view)

    # A shape, not its text: the capture's cut sets `running_bs` from its width.
    assert traced == [("prefill", 9, 3, None, 4)] and traced[0].width == 4
    assert lazy["lazy_traces"] == 1


def test_a_prefill_s_drafter_is_captured_with_its_later_pass_at_its_width(tmp_path):
    from test_capture_v4_pro import trace

    # DeepSeek-V4-Pro under MTP2 at TP8: three prompts of 64 tokens, which
    # ATOM pads to a batch of 4 for the drafter's later pass.
    (tmp_path / "record").mkdir()
    line = {"rows": [[64, 64, False]] * 3, "produces_output": True}
    record = _record(tmp_path / "record", [line | {"draft_bs": 4}, line])
    (tmp_path / "trace").mkdir()
    argv = "--population {} -- --kv_cache_dtype fp8 --method mtp "
    done = trace(tmp_path / "trace", argv.format(record) + "--num-speculative-tokens 2")

    tokens = re.findall(r"^published .* as (\w+)$", done.stdout, re.MULTILINE)
    assert tokens, done.stderr[-4000:]
    store = ArtifactStore(str(tmp_path / "trace" / "store"))
    rank, graphs = Topology().ranks()[0], {}
    for token in tokens:
        entry = store.read(store.key_from_token(Kind.OP_GRAPH, token))
        graph = json.loads(entry.read_member("graph", rank, "json"))
        output = not graph["structure"].endswith(("no output", "no output draft"))
        graphs[graph["structure"].endswith(" draft"), output, graph["shape"]] = graph
    # At its width only the drafter's graph of the step that samples.
    rows, wide = "prefill T=192 B=3", "prefill T=192 B=3 W=4"
    assert set(graphs) == {
        (draft, output, rows) for draft in (False, True) for output in (False, True)
    } | {(True, True, wide)}
    # Its first pass embeds the prompts' tokens, its second a row a sequence.
    for shape, width in ((rows, 3), (wide, 4)):
        embedded = [
            op["args"]["x"]["shape"][0]
            for op in _ops(graphs[True, True, shape]["region"])
            if op["op"] == "aiter.masked_embedding.default"
        ]
        assert embedded == [192, width]


def test_a_dp_middle_chunk_s_drafter_is_captured_with_its_later_pass_at_its_width(
    tmp_path,
):
    from test_capture_v4_pro import trace

    # DeepSeek-V4-Pro under MTP2 at DPA DP2: three middle chunks of 64 tokens
    # on each rank, which ATOM pads to a batch of 4 for the drafter's later pass.
    (tmp_path / "record").mkdir()
    line = {"rows": [[64, 64, False]] * 3, "produces_output": False}
    line |= {"moe_rows": 384, "draft_moe_rows": 384}
    record = _record(tmp_path / "record", [line | {"draft_bs": 4}, line])
    idle = json.dumps({"step": 0, "is_dummy_run": True})
    (tmp_path / "record" / "steps" / "dp1.jsonl").write_text(idle + "\n")
    (tmp_path / "trace").mkdir()
    argv = "--population {} --tp 2 -- --kv_cache_dtype fp8 --enable-dp-attention "
    argv += "--method mtp --num-speculative-tokens 2"
    done = trace(tmp_path / "trace", argv.format(record))

    tokens = re.findall(r"^published .* as (\w+)$", done.stdout, re.MULTILINE)
    assert tokens, done.stderr[-4000:]
    store = ArtifactStore(str(tmp_path / "trace" / "store"))
    rank, graphs = Topology().ranks()[0], {}
    for token in tokens:
        entry = store.read(store.key_from_token(Kind.OP_GRAPH, token))
        graph = json.loads(entry.read_member("graph", rank, "json"))
        output = not graph["structure"].endswith(("no output", "no output draft"))
        graphs[graph["structure"].endswith(" draft"), output, graph["shape"]] = graph
    rows, wide = "prefill T=192 B=3", "prefill T=192 B=3 W=4"
    assert {(True, False, wide), (False, False, rows)} <= set(graphs)
    assert (False, False, wide) not in graphs
    # Its first pass embeds the chunks' tokens, its second a row a sequence.
    for shape, width in ((rows, 3), (wide, 4)):
        embedded = [
            op["args"]["x"]["shape"][0]
            for op in _ops(graphs[True, False, shape]["region"])
            if op["op"] == "aiter.masked_embedding.default"
        ]
        assert embedded == [192, width]
