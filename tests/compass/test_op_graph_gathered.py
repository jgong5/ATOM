# SPDX-License-Identifier: MIT
"""A data-parallel step's `op_graph` is keyed by the rows its MoE gathers over
the group: the shape names them only off their default, every rank at this
one's tokens. `trace ops --population` reads them off each step's line, a
capture gives the peers the rest of them, and `OpGraphBackend` reads the graph
at a step's own, and at each two-batch-overlap micro-batch's, refusing one no
store holds by its shape.
"""

import json
from types import SimpleNamespace

import pytest
import torch
from test_artifact_store import CONDITIONS, GATES, stanza
from test_capture_ir import _decoder
from test_capture_v4_pro import DPA_TP8, EP8, MOE, MORI, _rows, capture, trace
from test_dp_step_max import prefill, runner
from test_moe_gather_rows import decode
from test_op_graph import by_name, g
from test_op_graph_arch import ENGINE
from test_op_work import ONE, leaf, t
from test_op_work_v4 import close, experts, moe
from test_work import SPEC

from atom.compass.artifacts import ArtifactStore, Key, Kind, Topology, member_name
from atom.compass.backends import BatchView, CostRefused, RequestShape
from atom.compass.backends.op_graph import OpGraphBackend
from atom.compass.backends.op_work import work
from atom.compass.backends.work import activated_experts, gemm, price
from atom.compass.capture import population_shapes
from atom.compass.capture.forward import Shape, structure
from atom.compass.capture.forward import prefill as prefill_shape
from atom.compass.run import _op_graphs
from atom.compass.runner import projection

DP = 8
RANK = Topology().ranks()[0]
#: Rank 0 prefills 7 tokens; its peers 20 between them, not 7 each.
COUNTS = [7, 5, 10, 1, 1, 1, 1, 1]
#: The same 27 rows, split another way among the peers.
RESPLIT = [7, 1, 1, 1, 1, 1, 1, 14]
#: Each rank's two micro-batches: rank 0 cuts its 7 tokens 3 then 4.
SPLIT = ((3, 2, 5, 1, 1, 1, 1, 1), (4, 3, 5, 0, 0, 0, 0, 0))
RESPLIT_TBO = ((3, 1, 1, 1, 1, 1, 1, 6), (4, 0, 0, 0, 0, 0, 0, 8))
#: Stored: each graph one GEMM, its rows 4096 times those its MoE gathers, so
#: each prices above the kernel floor.
STORED = [
    Shape("prefill", 7, 1, 27),
    Shape("prefill", 7, 1),
    Shape("prefill", 3, 1, 15),
    Shape("prefill", 4, 1, 12),
    Shape("decode", 4, 4),
    Shape("decode", 1, 1, 27),
]


@pytest.fixture(params=[False, True], ids=["ep_off", "ep8"])
def config(request, tmp_path):
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "config.json").write_text("{}")
    return SimpleNamespace(
        model=str(tmp_path / "model"),
        hf_config=SimpleNamespace(architectures=["DeepseekV4ForCausalLM"]),
        tensor_parallel_size=1,
        parallel_config=SimpleNamespace(data_parallel_size=DP),
        enable_expert_parallel=request.param,
        enforce_eager=True,
        **ENGINE,
    )


def gathered(shape: Shape) -> int:
    return shape.gathered or DP * shape.tokens


@pytest.fixture
def backend(config, tmp_path):
    """`OpGraphBackend` over a store of `STORED`, as `compass run` reads it."""
    store = ArtifactStore(str(tmp_path / "store"))
    for shape in STORED:
        for output in (True, False) if shape.phase == "prefill" else (True,):
            at = structure(config, shape.phase, True, "gfx950", output)
            store.publish(
                Key.of(Kind.OP_GRAPH, structure=at, shape=str(shape)),
                provenance=stanza("compass trace ops"),
                topology=Topology(),
                conditions=CONDITIONS,
                gates=GATES,
                members={
                    member_name("graph", RANK, "json"): json.dumps(
                        {"region": g(4096 * gathered(shape), 64, 64)}
                    ).encode()
                },
            )
    return OpGraphBackend(SPEC, _op_graphs(store.root, config, "gfx950"), kv="bf16")


def view(batch, counts, split=None, unified=False, rung=None):
    """Rank 0's view of `batch` in a group whose ranks schedule `counts`
    tokens, split `split`, as `batch_view` builds it from the step's mode."""
    r = runner(None, dp_size=DP)
    r.drafter = None
    sync = SimpleNamespace(
        num_tokens_across_dp=torch.tensor(counts), ub_tokens_across_dp=split
    )
    mode = SimpleNamespace(
        sync=sync,
        running_tokens_are_unified=unified,
        running_tokens=rung,
        running_bs=rung,
        use_cudagraph=rung is not None,
    )
    return projection.batch_view(batch, mode, r)


def gemms(*rows):
    return sum(price(gemm(4096 * m, 64, 64), SPEC) for m in rows)


def test_an_unequal_prefill_prices_from_the_graph_at_its_gathered_rows(backend):
    cost = backend.price(view(prefill(7), COUNTS))

    assert by_name(cost) == {"op.aiter.gemm_a16w16.default": gemms(27)}
    assert cost.terms[0].provenance.detail.endswith("prefill T=7 B=1 G=27")


def test_a_split_step_prices_each_micro_batch_at_its_gathered_rows(backend):
    cost = backend.price(view(prefill(7), COUNTS, SPLIT))

    assert by_name(cost) == {"op.tbo": pytest.approx(gemms(15, 12))}
    assert cost.terms[0].provenance.detail.endswith(
        "prefill T=3 B=1 G=15 that produces no output and prefill T=4 B=1 G=12"
    )


def test_a_balanced_decode_prices_from_its_rung_s_graph(backend):
    cost = backend.price(view(decode(3), [3] * DP, unified=True, rung=4))

    assert by_name(cost) == {"op.aiter.gemm_a16w16.default": gemms(DP * 4)}
    assert cost.terms[0].provenance.detail.endswith("decode T=4 B=4 G=32")


def test_a_decode_beside_a_prefilling_peer_prices_from_the_graph_at_its_gathered_rows(
    backend,
):
    # Rank 1 prefills 20 tokens; the other seven decode one each.
    cost = backend.price(view(decode(1), [1, 20, 1, 1, 1, 1, 1, 1]))

    assert by_name(cost) == {"op.aiter.gemm_a16w16.default": gemms(27)}
    assert cost.terms[0].provenance.detail.endswith("decode T=1 B=1 G=27")


@pytest.mark.parametrize(
    "counts,split,named",
    [
        ([7, 5, 11, 1, 1, 1, 1, 1], None, "7 tokens over 1 sequences gathering 28"),
        (COUNTS, ((3,) * DP, (4,) * DP), "3 tokens over 1 sequences gathering 24"),
    ],
    ids=["prefill", "micro-batch"],
)
def test_rows_no_graph_gathers_are_refused_by_shape(backend, counts, split, named):
    with pytest.raises(
        CostRefused, match=f"no op_graph holds a prefill step of {named}"
    ):
        backend.price(view(prefill(7), counts, split))


@pytest.mark.parametrize("split", [None, SPLIT], ids=["prefill", "tbo"])
def test_two_peer_splits_of_one_gather_price_alike(backend, split):
    other = None if split is None else RESPLIT_TBO

    assert backend.price(view(prefill(7), COUNTS, split)) == backend.price(
        view(prefill(7), RESPLIT, other)
    )


# --- trace ops --------------------------------------------------------------------


def _record(root, ranks):
    """A record directory, one line a rank: rank r's rows and gathered rows."""
    (root / "steps").mkdir(parents=True)
    (root / "run.json").write_text("{}")
    for r, (rows, moe_rows) in enumerate(ranks):
        line = {"step": 1, "is_dummy_run": not rows, "rung": None}
        line.update(rows=rows or [[1, 1, False]], moe_rows=moe_rows)
        (root / "steps" / f"dp{r}.jsonl").write_text(json.dumps(line) + "\n")
    return str(root)


def test_a_population_names_a_dp_prefill_s_rows_off_their_default(tmp_path):
    ranks = [([[7, 7, False]], 27), ([[5, 5, False]], 27), ([[8, 8, False]], 32)]
    ranks += [([[1, 9, True]] * 2, 6)]
    record = _record(tmp_path, ranks)

    # Four ranks: 4 x 8 rows is the default, 4 x 2 a decode's.
    assert sorted(map(str, population_shapes([record]))) == [
        "decode B=2 G=6",
        "prefill T=5 B=1 G=27",
        "prefill T=7 B=1 G=27",
        "prefill T=8 B=1",
    ]


def test_a_population_s_drafted_decode_at_its_default_names_no_rows(tmp_path):
    # Two ranks, each two rows verifying one draft: 2 x 2 x 2 rows.
    record = _record(tmp_path, [([[2, 9, True]] * 2, 8)] * 2)

    assert list(map(str, population_shapes([record]))) == ["decode B=2"] * 2


def test_a_prefill_gathers_at_least_its_own_rows():
    with pytest.raises(ValueError, match="the gather holds this rank's"):
        prefill_shape(7, 1, 6)


def test_a_capture_on_one_rank_refuses_gathered_rows(tmp_path):
    record = _record(tmp_path / "record", [([[7, 7, False]], 27)])
    done = trace(tmp_path, f"--population {record}")

    assert "prefill T=7 B=1 G=27 gathers a DP group's rows on one rank" in done.stderr


@pytest.mark.parametrize("ep", [False, True], ids=["ep_off", "ep8"])
def test_a_capture_gives_the_peers_the_rest_of_the_gathered_rows(ep, tmp_path_factory):
    # Rank 0 prefills 64 tokens; the eight ranks' MoE gathers 767 rows.
    ranks = [([[64, 64, False]], 767)] + [(None, 767)] * (DP - 1)
    record = _record(tmp_path_factory.mktemp("record"), ranks)
    graphs = capture(tmp_path_factory, f"--population {record}" + DPA_TP8 + EP8 * ep)
    graph, _ = graphs["prefill T=64 B=1 G=767"]

    for layer in _decoder(graph):
        exchanged = [op for op in layer if op["kind"] == "declared"]
        names = [op["op"] for op in exchanged]
        if ep:
            assert names == [f"{MORI}.dispatch", f"{MORI}.combine"]
            assert {op["args"]["num_ep_ranks"] for op in exchanged} == {DP}
        else:
            assert names == [f"{MOE}:all_gatherv", f"{MOE}:reduce_scatterv"]
            assert _rows(exchanged[1]) == (767, 64)
        assert _rows(exchanged[0]) == (64, 767)


@pytest.mark.parametrize("ep", [False, True], ids=["ep_off", "ep8"])
def test_a_decode_at_gathered_rows_has_a_prefilling_peer(ep, tmp_path_factory):
    # Rank 0 decodes one token while a peer prefills: the MoE gathers 70 rows.
    ranks = [([[1, 9, True]], 70)] + [(None, 70)] * (DP - 1)
    record = _record(tmp_path_factory.mktemp("record"), ranks)
    graphs = capture(tmp_path_factory, f"--population {record}" + DPA_TP8 + EP8 * ep)
    graph, _ = graphs["decode B=1 G=70"]

    for layer in _decoder(graph):
        exchanged = [op for op in layer if op["kind"] == "declared"]
        if ep:
            assert [op["op"] for op in exchanged] == [
                f"{MORI}.dispatch",
                f"{MORI}.combine",
            ]
        else:
            # The variable-length gather, not every rank padded to one height.
            assert [op["op"] for op in exchanged] == [
                f"{MOE}:all_gatherv",
                f"{MOE}:reduce_scatterv",
            ]
            assert _rows(exchanged[1]) == (70, 1)
        assert _rows(exchanged[0]) == (1, 70)


@pytest.mark.parametrize("call", ["dispatch", "combine"])
def test_a_mori_exchange_at_gathered_rows_spans_the_ep_width(call):
    # This rank's 2 tokens; the group's 27 rows, not 8 x 2.
    exchange = leaf(
        f"{MORI}.{call}",
        [[27, 64]],
        input=t([27 if call == "combine" else 2, 64]),
        weights=None,
        scales=None,
        indices=t([2, 2], "torch.int32"),
        num_ep_ranks=DP,
    )
    [c] = work(exchange, ONE, 1, "bf16", DP).comm

    assert c.ranks == DP


def test_a_gather_s_pad_rows_activate_experts_as_one_token():
    # Sixteen rows gathered over the group, five of them real tokens.
    padded = BatchView(
        (RequestShape(1, 9, True),), capture_rung=2, moe_rows=16, moe_tokens=5
    )
    hit = activated_experts(16, 2, 6)

    close(work(moe(16, 16), padded, 1, "bf16"), experts(16, 32, hit))
