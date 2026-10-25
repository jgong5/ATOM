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
from test_op_graph import by_name, g, listed, pt
from test_op_graph_drafter_width import DECODE, settled
from test_op_work import leaf, t
from test_work import SPEC

from atom.compass.artifacts import ArtifactStore, Kind, Topology
from atom.compass.backends import BatchView, RequestShape
from atom.compass.backends.op_graph import LISTED, OpGraphBackend
from atom.compass.backends.resolver import CostSource
from atom.compass.backends.work import gemm, price
from atom.compass.capture import population_shapes

EMBED = "aten.embedding.default"
MOE = "aiter.fused_moe_.default"
GEMM = "aiter.gemm_a16w16.default"
ARGMAX = "aiter.ops.flydsl.topk.topk_per_row_argmax:topk_per_row_argmax"
MUL = "aten.mul.Tensor"
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


def argmax(rows):
    """A sampler leaf over `rows` rows of a vocabulary of 64."""
    return leaf(ARGMAX, (), scores=t([rows, 64]), indices=t([rows, 1]))


class Asked(CostSource):
    """A price list that notes each leaf's op, leading rows and T and B, and
    declines."""

    name = LISTED

    def __init__(self):
        self.asked = []

    def price(self, request):
        op, (_, tokens, seqs, *_), _ = request
        rows = [*next(iter(op["args"].values()))["shape"], None][0]
        self.asked.append((op["op"], rows, tokens, seqs))
        return self.refuse("asked")


def mixed(rows, width=4):
    """A mul of `rows` rows broadcast across `width`."""
    return leaf(MUL, [[width, rows, 64]], self=t([rows, 64]), other=t([width, 1, 64]))


@pytest.mark.parametrize(
    "rows,options,graph,asked",
    [
        (
            [(1, 90, True)] * 3,
            {},
            [g(4, 3, 64), argmax(3)],
            [(GEMM, 4, 4, 4), (ARGMAX, 3, 3, 3)],
        ),
        # Its drafter's at its rows, whatever its leaves lead with.
        (
            DECODE,
            {"drafts": 2},
            [g(12, 64, 64), argmax(6), argmax(9)],
            [(GEMM, 12, 4, 4), (ARGMAX, 6, 3, 3), (ARGMAX, 9, 3, 3)]
            + [(GEMM, 12, 3, 3), (ARGMAX, 6, 3, 3), (ARGMAX, 9, 3, 3)],
        ),
        # At DP2: its model at the width's tokens, an MoE over the width's gathered.
        (
            DECODE,
            {"drafts": 2, "dp": 2},
            [g(12, 64, 64), g(24, 32, 64), mixed(9, 12), argmax(9)],
            [(GEMM, 12, 4, 4), (GEMM, 24, 4, 4), (MUL, 9, 3, 3), (ARGMAX, 9, 3, 3)]
            + [(GEMM, 12, 3, 3), (GEMM, 24, 3, 3), (MUL, 9, 3, 3), (ARGMAX, 9, 3, 3)],
        ),
        # Its model at its rows, an MoE GEMM over 24 gathered rows among it.
        (
            [(3, 3, False)] * 3,
            {},
            [g(9, 64, 64), g(24, 32, 64), g(4, 128, 64), mixed(3), mixed(9), argmax(3)],
            [(GEMM, 9, 9, 3), (GEMM, 24, 9, 3), (GEMM, 4, 9, 4)]
            + [(MUL, 3, 9, 3), (MUL, 9, 9, 3), (ARGMAX, 3, 9, 3)],
        ),
        # Leaves that lead with no count of its rows or its width, at its rows.
        (
            [(1, 90, True)] * 3,
            {},
            [leaf(MUL, [s], self=t(s), other=t(s)) for s in ([1, 3, 64], [64, 4], [])],
            [(MUL, 1, 3, 3), (MUL, 64, 3, 3), (MUL, None, 3, 3)],
        ),
        # Leaves over its rows and its width, at its rows.
        (
            [(1, 90, True)] * 3,
            {},
            [
                mixed(3),
                leaf(
                    MUL,
                    [[3, 4, 64]],
                    self=t([3, 1, 64], role="weight"),
                    other=t([4, 64]),
                ),
                leaf(MUL, [[4, 3]], self=t([4, 3]), other=t([3], role="metadata")),
            ],
            [(MUL, 3, 3, 3), (MUL, 3, 3, 3), (MUL, 4, 3, 3)],
        ),
        # Unpadded, at its rows.
        ([(1, 90, True)] * 3, {"padded": False}, [g(4, 64, 64)], [(GEMM, 4, 3, 3)]),
    ],
    ids=[
        "eager-decode",
        "eager-mtp-decode",
        "mtp-dp2",
        "sampling-prefill",
        "unrecognised",
        "mixed",
        "real-rows",
    ],
)
def test_a_padded_target_asks_its_price_list_at_the_rows_each_leaf_ran(
    rows, options, graph, asked
):
    step = BatchView(
        tuple(RequestShape(*row) for row in rows), produces_output=True, target_bs=4
    )
    listing = Asked()
    backend = OpGraphBackend(
        SPEC,
        lambda *_, **__: {"seq": graph},
        kv="bf16",
        price_list=listing,
        **options,
    )
    backend.price(step)

    assert listing.asked == asked


# Each leaf measured on the graphs at rungs 3 and 4, where it runs the rung's
# rows; flat in context.
RUNGS = listed(
    *[
        pt(op, f"decode B={b}", ns, c)
        for b, ns, sampled in ((3, 5e3, 1e3), (4, 7e3, 2e3))
        for op, ns in ((g(b, 64, 64), ns), (argmax(b), sampled))
        for c in (100, 1000)
    ]
)


@pytest.mark.parametrize("dp,moe_rows", [(1, None), (2, 8)], ids=["one-rank", "dp2"])
def test_a_padded_decode_prices_its_model_at_its_width_and_its_sampler_at_its_rows(
    dp, moe_rows
):
    # Rank 0 decodes 3 rows eagerly, padded to 4; at DP2 its MoE gathers 4 from
    # each rank, the default at its width.
    step = BatchView((RequestShape(1, 90, True),) * 3, moe_rows=moe_rows, target_bs=4)
    graph = {"seq": [g(4, 64, 64), argmax(3)]}
    backend = OpGraphBackend(
        SPEC, lambda *_, **__: graph, kv="bf16", dp=dp, price_list=RUNGS
    )
    cost = backend.price(step)

    assert not cost.refusals
    assert by_name(cost) == {
        f"op.{GEMM}.price_list": pytest.approx(7e-6),
        f"op.{ARGMAX}.price_list": pytest.approx(1e-6),
    }


FOUR = (RequestShape(3, 90, True),) * 4


class AskedG(Asked):
    """`Asked`, with each leaf's G too."""

    def price(self, request):
        got = super().price(request)
        self.asked[-1] += (request[1][3],)
        return got


@pytest.mark.parametrize(
    "options,view,graph,asked",
    [
        # V4-Pro MTP2 at DP2 beside a peer's 4 rows: its first pass's MoE gathers
        # 9 + 12, every rank's later passes 4 rows each.
        (
            {"dp": 2},
            {"moe_rows": 24, "target_bs": 4, "draft_moe_rows": 21},
            [g(9, 64, 64), g(21, 32, 64), argmax(3), g(4, 64, 64), g(8, 32, 64)],
            [(GEMM, 9, 3, 3, 21), (GEMM, 21, 3, 3, 21), (ARGMAX, 3, 3, 3, 21)]
            + [(GEMM, 4, 4, 4, 8), (GEMM, 8, 4, 4, 8)],
        ),
        # At DP2 over 4 rows beside a peer's 6: its later passes' MoE gathers 12,
        # a first-pass count too, asked at the rows and the smaller G.
        (
            {"dp": 2},
            {"requests": FOUR, "draft_moe_rows": 30, "draft_bs": 6},
            [g(12, 64, 64), g(30, 32, 64), g(6, 64, 64), g(12, 32, 64)],
            [(GEMM, 12, 4, 4, 12), (GEMM, 30, 4, 4, 30), (GEMM, 6, 6, 6, 12)]
            + [(GEMM, 12, 4, 4, 12)],
        ),
        # At DP2 over 4 rows, its `draft_bs`: a later pass's 4 at its G.
        (
            {"dp": 2},
            {"requests": FOUR, "draft_moe_rows": 24},
            [g(12, 64, 64), g(24, 32, 64), g(4, 64, 64), g(8, 32, 64)],
            [(GEMM, 12, 4, 4, None), (GEMM, 24, 4, 4, None), (GEMM, 4, 4, 4, 8)]
            + [(GEMM, 8, 4, 4, 8)],
        ),
        # At DP8 beside peers' 5 and 1 rows: a count of both sides at
        # `draft_moe_rows`, below every rank's `draft_bs`.
        (
            {"dp": 8},
            {"draft_moe_rows": 42, "draft_bs": 6},
            [g(6, 64, 64), g(48, 32, 64)],
            [(GEMM, 6, 3, 3, 42), (GEMM, 48, 6, 6, 48)],
        ),
        # At DP2 over 2 rows beside a peer's 6: a count of both sides at the
        # rows' default G.
        (
            {"dp": 2},
            {"requests": FOUR[:2], "draft_moe_rows": 24, "draft_bs": 6},
            [g(6, 64, 64)],
            [(GEMM, 6, 2, 2, None)],
        ),
        # At DP4: its first pass gathers its default, its later passes 16; a
        # leaf over twice the width is no later pass's.
        (
            {"dp": 4},
            {"moe_rows": 48, "draft_moe_rows": 36},
            [g(9, 64, 64), g(36, 32, 64), g(4, 64, 64), g(16, 32, 64), g(8, 64, 64)],
            [(GEMM, 9, 3, 3, None), (GEMM, 36, 3, 3, None), (GEMM, 4, 4, 4, 16)]
            + [(GEMM, 16, 4, 4, 16), (GEMM, 8, 3, 3, None)],
        ),
        # Replayed at rung 4: its first pass at its rows.
        (
            {},
            {"capture_rung": 4},
            [g(9, 64, 64), g(4, 64, 64)],
            [(GEMM, 9, 3, 3, None), (GEMM, 4, 4, 4, None)],
        ),
    ],
    ids=[
        "mtp-dp2",
        "both",
        "draft-bs-rows",
        "both-low-g",
        "both-default-g",
        "dp4",
        "replayed",
    ],
)
def test_a_drafter_asks_its_price_list_at_the_rows_and_g_each_pass_ran(
    options, view, graph, asked
):
    rows = tuple(RequestShape(*row) for row in DECODE)
    step = BatchView(**{"requests": rows, "draft_bs": 4, **view})
    listing = AskedG()
    graphs = {False: {"seq": []}, True: {"seq": graph}}
    backend = OpGraphBackend(
        SPEC,
        lambda *_, draft=False, **__: graphs[draft],
        kv="bf16",
        drafts=2,
        price_list=listing,
        **options,
    )
    backend.price(step)

    assert listing.asked == asked


# The drafter's later-pass GEMM measured at B=3 and 4, its MoE gathering 8.
GATHERED = listed(
    *[
        pt(g(b, 64, 64), f"decode B={b} G=8", ns, c)
        for b, ns in ((3, 5e3), (4, 7e3))
        for c in (100, 1000)
    ]
)


@pytest.mark.parametrize(
    "dp,gathered,params",
    [(1, {}, RUNGS), (2, {"moe_rows": 24, "draft_moe_rows": 21}, GATHERED)],
    ids=["one-rank", "dp2"],
)
def test_a_drafter_prices_its_later_passes_at_draft_bs(dp, gathered, params):
    # MTP2 over 3 rows; its later passes run 4.
    step = BatchView(
        tuple(RequestShape(*row) for row in DECODE), draft_bs=4, **gathered
    )
    graphs = {False: {"seq": []}, True: {"seq": [g(4, 64, 64)]}}
    backend = OpGraphBackend(
        SPEC,
        lambda *_, draft=False, **__: graphs[draft],
        kv="bf16",
        dp=dp,
        drafts=2,
        price_list=params,
    )
    cost = backend.price(step)

    assert not cost.refusals
    assert by_name(cost) == {f"draft op.{GEMM}.price_list": pytest.approx(7e-6)}


@pytest.mark.parametrize(
    "peers,step,draft_rows",
    [
        # Every rank decoding: 9 + 12 tokens, 3 a row.
        ([(4, 12, False)], False, 7),
        # Beside a prefilling peer, 3 rows and one each peer, of 6.
        ([(2, 40, True), (1, 3, False)], False, 5),
        # Beside a wider prefilling peer, its 5 rows and one each peer, of 8.
        ([(5, 40, True)], False, 6),
        ([(4, 12, False)], True, None),
        ((), False, None),
    ],
    ids=[
        "dp2",
        "beside-a-prefill",
        "beside-a-wider-prefill",
        "draft-graph",
        "one-rank",
    ],
)
def test_an_eager_drafter_s_later_passes_gather_every_rank_s_rows(
    monkeypatch, peers, step, draft_rows
):
    view = settled(monkeypatch, DECODE, peers, eager=True, step=step)

    assert view.draft_rows == draft_rows


@pytest.mark.parametrize(
    "rows,peer,drafts,draft_rows",
    [
        # Three rows verifying one draft beside a peer's three: no later pass.
        ([(2, 90, True)] * 3, (3, 6, False), 1, None),
        # MTP2 beside a peer's four: its later passes gather 3 + 4 rows.
        (DECODE, (4, 12, False), 2, 7),
    ],
    ids=["mtp1", "mtp2"],
)
def test_an_eager_dp_drafter_gathers_later_pass_rows_only_when_it_has_later_passes(
    monkeypatch, rows, peer, drafts, draft_rows
):
    view = settled(monkeypatch, rows, [peer], eager=True, step=False, drafts=drafts)

    assert view.draft_rows == draft_rows


def test_an_eager_dp_drafter_asks_its_later_passes_at_the_g_they_gathered(
    monkeypatch,
):
    # V4-Pro MTP2 at DP2 with no draft graph, 3 rows beside a peer's 4: its
    # first pass's MoE gathers 9 + 12, its later passes' 3 + 4, which a capture
    # states as this rank's 3 at each rank.
    view = settled(monkeypatch, DECODE, [(4, 12, False)], eager=True, step=False)
    drafted = [g(9, 64, 64), g(21, 32, 64), argmax(3), g(3, 64, 64)]
    graphs = {
        False: {"seq": [g(4, 64, 64)]},
        True: {"seq": drafted + [g(6, 32, 64), g(7, 32, 64)]},
    }
    listing = AskedG()
    backend = OpGraphBackend(
        SPEC,
        lambda *_, draft=False, **__: graphs[draft],
        kv="bf16",
        dp=2,
        drafts=2,
        price_list=listing,
    )
    backend.price(view)

    assert listing.asked == [
        (GEMM, 4, 4, 4, None),
        (GEMM, 9, 3, 3, 21),
        (GEMM, 21, 3, 3, 21),
        (ARGMAX, 3, 3, 3, 7),
        (GEMM, 3, 3, 3, 7),
        (GEMM, 6, 3, 3, 7),
        (GEMM, 7, 3, 3, 7),
    ]


def test_an_eager_dp_mtp1_drafter_asks_its_one_pass_at_its_own_g(monkeypatch):
    # MTP1 at DP2 with no draft graph, 3 rows beside a peer's 3: its one pass's
    # MoE gathers 6 + 6, its default; it has no later pass at `draft_rows`.
    rows = [(2, 90, True)] * 3
    view = settled(monkeypatch, rows, [(3, 6, False)], eager=True, step=False, drafts=1)
    drafted = [g(6, 64, 64), g(12, 32, 64), argmax(3), g(3, 64, 64)]
    graphs = {False: {"seq": []}, True: {"seq": drafted}}
    listing = AskedG()
    backend = OpGraphBackend(
        SPEC,
        lambda *_, draft=False, **__: graphs[draft],
        kv="bf16",
        dp=2,
        drafts=1,
        price_list=listing,
    )
    backend.price(view)

    assert listing.asked == [
        (GEMM, 6, 3, 3, None),
        (GEMM, 12, 3, 3, None),
        (ARGMAX, 3, 3, 3, None),
        (GEMM, 3, 3, 3, None),
    ]
