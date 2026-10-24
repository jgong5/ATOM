# SPDX-License-Identifier: MIT
"""A target that replays no graph yet pads past its rows, read at that width.

`ForwardMode.decide` settles `running_bs` on every step. An eager decode every
rank of whose DP group decodes runs its model at it
(`ModelRunner._padded_decode_inputs`), and a prefill that samples runs its LM
head at it (`cu_seqlens_q`). Each step's mode here is `decide`'s own.
"""

import json
import re

import pytest
from test_capture_ir import _ops, _record
from test_op_graph import g
from test_op_graph_drafter_width import DECODE, settled
from test_work import SPEC

from atom.compass.artifacts import ArtifactStore, Kind, Topology
from atom.compass.backends.op_graph import OpGraphBackend
from atom.compass.backends.work import gemm, price
from atom.compass.capture import population_shapes

EMBED = "aten.embedding.default"
MOE = "aiter.fused_moe_.default"
GEMM = "aiter.gemm_a16w16.default"
ARGMAX = "aiter.ops.flydsl.topk.topk_per_row_argmax:topk_per_row_argmax"
#: DeepSeek-V4-Pro's vocabulary, its LM head's rows at TP1.
VOCAB = 129280


@pytest.mark.parametrize(
    "rows,peers,eager,target",
    [
        (DECODE, [(4, 12, False)], True, ("decode", 3, 3, True, 24, 4)),
        ([(3, 3, False)] * 3, (), False, ("prefill", 9, 3, True, 4)),
        (DECODE, [(4, 12, False)], False, ("decode", 4, 4, True, 24, None)),
        (DECODE, [(1, 16, True)], False, ("decode", 3, 3, True, 25, None)),
    ],
    ids=["eager-dp-decode", "sampling-prefill", "replayed", "beside-a-prefill"],
)
def test_a_target_padded_without_a_replay_reads_its_graph_at_running_bs(
    monkeypatch, rows, peers, eager, target
):
    asked = []

    def graph(*at, draft=False, width=None):
        if not draft:
            asked.append((*at, width))
        # A target's graph at a wider batch runs wider.
        return g(4, 8, 8) if draft else g(1000 * (width or at[2]), 64, 64)

    view = settled(monkeypatch, rows, peers, eager)
    backend = OpGraphBackend(SPEC, graph, kv="bf16", dp=1 + len(peers), drafts=2)
    cost = backend.price(view)

    assert asked == [target]
    assert view.target_bs == target[-1]
    width = target[-1] or target[2]
    assert cost.target_seconds == price(gemm(1000 * width, 64, 64), SPEC)


def test_a_population_names_a_padded_target_at_its_width(tmp_path):
    prefill = {"rows": [[3, 3, False]] * 3, "produces_output": True}
    decode = {"rows": [[3, 90, True]] * 3, "moe_rows": 24}
    lines = [decode | {"target_bs": 4}, decode, prefill | {"target_bs": 4}]
    record = _record(tmp_path, lines)
    steps = tmp_path / "steps"
    (steps / "dp1.jsonl").write_bytes((steps / "dp0.jsonl").read_bytes())

    # Two ranks each padded to 4 rows of 3 gather 24, the default at W=4.
    shapes = [
        "decode B=3 W=4",
        "decode B=3",
        "decode B=3 G=24",
        "decode B=3",
        "prefill T=9 B=3 W=4",
    ]
    assert list(map(str, population_shapes([record]))) == shapes * 2


def test_a_capture_runs_a_padded_target_at_its_width_and_samples_its_rows(tmp_path):
    from test_capture_v4_pro import DPA_TP8, trace

    # DeepSeek-V4-Pro, DPA DP8: rank 0 decodes 3 rows eagerly beside peers at
    # 4, and prefills three prompts of 64 tokens on a ladder that holds 4.
    (tmp_path / "record").mkdir()
    decode = {"rows": [[1, 90, True]] * 3, "moe_rows": 32, "target_bs": 4}
    prefill = {"rows": [[64, 64, False]] * 3, "produces_output": True}
    record = _record(tmp_path / "record", [decode, prefill | {"target_bs": 4}])
    for r in range(1, 8):
        idle = {"step": 0, "is_dummy_run": True}
        (tmp_path / "record" / "steps" / f"dp{r}.jsonl").write_text(json.dumps(idle))
    (tmp_path / "trace").mkdir()
    done = trace(tmp_path / "trace", f"--population {record}{DPA_TP8}")

    tokens = re.findall(r"^published .* as (\w+)$", done.stdout, re.MULTILINE)
    store = ArtifactStore(str(tmp_path / "trace" / "store"))
    rank = Topology().ranks()[0]
    graphs = {}
    for token in tokens:
        graph = store.read(store.key_from_token(Kind.OP_GRAPH, token))
        graph = json.loads(graph.read_member("graph", rank, "json"))
        graphs[graph["shape"]] = _ops(graph["region"])
    # Each target alone, and a wide prefill only as the step that samples.
    assert set(graphs) == {"decode B=3 W=4", "prefill T=192 B=3 W=4"}, done.stderr

    def rows(shape, op, arg):
        return [o["args"][arg]["shape"][0] for o in graphs[shape] if o["op"] == op]

    # The decode's model takes 4 rows, its MoE 4 from each of 8 ranks.
    assert rows("decode B=3 W=4", EMBED, "indices") == [4]
    assert set(rows("decode B=3 W=4", MOE, "hidden_states")) == {32}
    # The prefill's model takes its 192 tokens, its LM head 4 rows.
    assert rows("prefill T=192 B=3 W=4", EMBED, "indices") == [192]
    head = [
        o["args"]["A"]["shape"][0]
        for o in graphs["prefill T=192 B=3 W=4"]
        if o["op"] == GEMM and o["args"]["B"]["shape"][0] == VOCAB
    ]
    assert head == [4]
    # Each samples its 3 rows.
    for shape in graphs:
        assert rows(shape, ARGMAX, "scores") == [3]


def test_a_capture_keeps_an_eager_decode_s_drafter_only_from_its_own_shape(tmp_path):
    from test_capture_v4_pro import trace

    # DeepSeek-V4-Pro under MTP2 at DPA DP2: rank 0 decodes 3 rows eagerly,
    # padded to 4, and its drafter's later passes run at 4.
    (tmp_path / "record").mkdir()
    line = {"rows": [[3, 90, True]] * 3, "moe_rows": 24, "draft_moe_rows": 8}
    record = _record(tmp_path / "record", [line | {"target_bs": 4, "draft_bs": 4}])
    idle = json.dumps({"step": 0, "is_dummy_run": True})
    (tmp_path / "record" / "steps" / "dp1.jsonl").write_text(idle + "\n")
    (tmp_path / "trace").mkdir()
    argv = "--population {} --tp 2 -- --kv_cache_dtype fp8 --enable-dp-attention "
    argv += "--method mtp --num-speculative-tokens 2"
    done = trace(tmp_path / "trace", argv.format(record))

    tokens = re.findall(r"^published .* as (\w+)$", done.stdout, re.MULTILINE)
    assert tokens, done.stderr[-4000:]
    store = ArtifactStore(str(tmp_path / "trace" / "store"))
    rank, graphs = Topology().ranks()[0], set()
    for token in tokens:
        entry = store.read(store.key_from_token(Kind.OP_GRAPH, token))
        graph = json.loads(entry.read_member("graph", rank, "json"))
        graphs.add((graph["structure"].endswith(" draft"), graph["shape"]))
    # The target at its width; its drafter only at its rows on the batch of 4,
    # where the run reads it.
    assert graphs == {(False, "decode B=3 W=4"), (True, "decode T=3 B=4 G=8")}
