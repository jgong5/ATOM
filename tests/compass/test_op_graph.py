# SPDX-License-Identifier: MIT
"""The op-level backend: a step priced from the graph captured at its shape.

Each leaf is one roofline over its work; a `Seq` adds, a `Repeat` re-emits its
body's prices, a `max` `Par` takes its longest branch, and a
`resource_bound` `Par` costs one roofline over the branches' summed work. A step runs at its rung's
graph, or its width's under ``padding: none``, a prefill at its tokens and
rows; attention reads its context in the run's KV format. What cannot be
priced is refused by name, each reason once. The Qwen3.8-27B TP1 capture prices leaf by leaf, and `run.runner` installs the
backend from a run file's ``op_graph`` over the store's graphs.
"""

import copy
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from test_backend_step_linear import _installed, law, view
from test_backend_step_roofline import COMPUTE, MEMORY, ROOFLINE, _spec
from test_capture_ir import CONFIG, WIDTH_2, _ops, capture
from test_memory_readings import WORKER
from test_op_work import leaf, t
from test_work import SPEC

from atom.compass.backends import BatchView, CostRefused, Method, RequestShape
from atom.compass.backends.cost import fold_seconds
from atom.compass.backends.op_graph import NONE, OpGraphBackend
from atom.compass.backends.op_work import work
from atom.compass.backends.provenance import Refusal
from atom.compass.backends.step_roofline import overlap
from atom.compass.backends.work import attention, gemm, price
from atom.compass.bench.ops import leaves
from atom.compass.capture.peers import MORI
from atom.compass.run import _op_graphs
from atom.compass.spec import MachineSpec

GEMM = "aiter.gemm_a16w16.default"
ATTENTION = "aiter.unified_attention_with_output_base.default"
GATHER = "atom.model_ops.moe:all_gatherv"
DECODE = view((1, 900, True), (1, 300, True), rung=2)
PREFILLS = "--prefill 512 2 --prefill 300 3"


def g(m, n, k):
    """A GEMM leaf of `[m, k] @ [n, k]^T`."""
    return leaf(GEMM, [[m, n]], A=t([m, k]), B=t([n, k], role="weight"), bias=None)


def backend(region, machine=SPEC, kv="bf16", **kwargs):
    return OpGraphBackend(machine, lambda *shape: region, kv=kv, **kwargs)


def by_name(cost):
    return {term.name: term.seconds for term in cost.terms}


def test_a_step_is_one_roofline_per_leaf_composed_by_the_graph_s_joins():
    view_leaf = leaf("aten.view.default", [[2, 64]], self=t([2, 64]), size=[2, 64])
    region = {
        "seq": [
            g(2, 64, 64),
            view_leaf,
            {"repeat": g(2, 128, 64), "count": 3, "index": "l0", "start": 0, "step": 1},
            {"par": [g(2, 2048, 4096), {"seq": [g(2, 4096, 4096)]}], "join": "max"},
            {
                "par": [
                    {
                        "par": [
                            {"seq": [g(4096, 4096, 4096), g(4096, 4096, 4096)]},
                            g(4096, 2048, 4096),
                        ],
                        "join": "max",
                    },
                    g(4096, 1024, 4096),
                ],
                "join": "resource_bound",
            },
        ]
    }
    cost = backend(region).price(DECODE)
    one, body = price(gemm(2, 64, 64), SPEC), price(gemm(2, 128, 64), SPEC)
    longest = price(gemm(2, 4096, 4096), SPEC)
    big = gemm(4096, 4096, 4096)
    summed = big + big + gemm(4096, 2048, 4096) + gemm(4096, 1024, 4096)

    assert by_name(cost) == {
        f"op.{GEMM}": fold_seconds([one, body, body, body, longest]),
        "par.resource_bound": price(summed, SPEC),
    }


MIDDLE = BatchView((RequestShape(9, 9, False),), produces_output=False)
FINAL = BatchView((RequestShape(9, 9, False),), produces_output=True)


@pytest.mark.parametrize(
    "batch,padded,shape",
    [
        (view(*[(1, 90, True)] * 3, rung=4), True, ("decode", 4, 4, True)),
        (view(*[(1, 90, True)] * 3, rung=4), False, ("decode", 3, 3, True)),
        (view((1, 90, True), (1, 9, True)), True, ("decode", 2, 2, True)),
        (view((100, 100, False), (1, 50, True)), True, ("prefill", 101, 2, True)),
        (MIDDLE, True, ("prefill", 9, 1, False)),
        (FINAL, True, ("prefill", 9, 1, True)),
    ],
    ids=["rung", "padding-none", "eager-decode", "mixed", "middle", "final"],
)
def test_a_step_reads_the_graph_captured_at_its_shape(batch, padded, shape):
    asked = []

    def graph(*at):
        asked.append(at)
        return g(1, 8, 8)

    OpGraphBackend(SPEC, graph, kv="bf16", padded=padded).price(batch)

    assert asked == [shape]


@pytest.mark.parametrize(
    "batch,graph,named",
    [
        (
            view((2, 9, True)),
            g(1, 8, 8),
            "a step that decodes more than one token in a row",
        ),
        (view((1, 9, True)), None, "no op_graph holds a decode step of 1 tokens"),
        (MIDDLE, None, "of 9 tokens over 1 sequences that produces no output;"),
        (
            BatchView((RequestShape(9, 9, False),)),
            g(1, 8, 8),
            "a prefill step whose view does not say whether it produces output",
        ),
        (
            BatchView((RequestShape(9, 9, False),), image_tokens=5),
            g(1, 8, 8),
            "a step with image tokens",
        ),
    ],
    ids=[
        "speculative",
        "uncaptured",
        "uncaptured-middle",
        "output-unsaid",
        "images",
    ],
)
def test_a_step_with_no_graph_to_price_it_is_refused_by_name(batch, graph, named):
    with pytest.raises(CostRefused, match=named):
        backend(graph).price(batch)


@pytest.mark.parametrize(
    "steps,graph,reason",
    [
        (
            [view((2, 9, True)), view((3, 9, True), (1, 9, True))],
            g(1, 8, 8),
            (
                "a step that decodes more than one token in a row: a speculative "
                "decode step is not captured"
            ),
        ),
    ],
    ids=["speculative"],
)
def test_steps_declined_for_one_cause_share_one_reason(steps, graph, reason):
    """Steps of one kind the graph declines, whatever their counts, are
    refused for one reason."""
    reasons = set()
    for step in steps:
        with pytest.raises(CostRefused) as refused:
            backend(graph).price(step)
        reasons.update(refused.value.declined)

    assert reasons == {Refusal("op-graph", reason)}


def test_a_data_parallel_step_with_no_moe_exchange_prices_as_its_own_rows():
    dp = backend(g(9, 8, 8)).price(replace(FINAL, moe_rows=18))

    assert by_name(dp) == by_name(backend(g(9, 8, 8)).price(FINAL))


def test_each_reason_a_graph_refuses_is_named_once():
    region = {
        "seq": [
            leaf("foo.bar.default"),
            {
                "repeat": {"seq": [leaf("foo.bar.default"), g(2, 8, 8)]},
                "count": 4,
                "index": "l0",
                "start": 0,
                "step": 1,
            },
            leaf("foo.baz.default"),
        ]
    }
    with pytest.raises(CostRefused) as refused:
        backend(region).price(DECODE)

    assert refused.value.declined == (
        Refusal("op-work", "no extractor for the op kind foo.bar.default"),
        Refusal("op-work", "no extractor for the op kind foo.baz.default"),
    )


def test_a_spec_field_a_leaf_reads_and_the_spec_lacks_refuses_naming_it():
    lacking = SimpleNamespace(values={}, value=SPEC.value, origin=SPEC.origin)

    with pytest.raises(CostRefused, match=f"{GEMM} reads device.kernel_floor_s,"):
        backend(g(2, 8, 8), lacking).price(DECODE)


def test_a_collective_wider_than_the_scale_up_domain_refuses_naming_it():
    reduce = leaf("aiter.all_reduce_.default", [[2, 64]], tensor=t([2, 64]))

    with pytest.raises(CostRefused, match="over 16 ranks leaves the scale-up"):
        backend(reduce, tp=16).price(DECODE)


@pytest.mark.parametrize("kv", ["bf16", "fp8"])
def test_attention_reads_its_context_in_the_run_s_kv_format(kv):
    attn = leaf(ATTENTION, [[2, 6144]], q=t([2, 24, 256]), k=t([2, 4, 256]))
    long = view((1, 200_000, True), (1, 100_000, True), rung=2)
    cost = backend(attn, kv=kv).price(long)
    expected = attention([1, 1], [200_000, 100_000], 24 * 256, 1, 4 * 256, "bf16", kv)

    assert by_name(cost) == {f"op.{ATTENTION}": price(expected, SPEC)}


@pytest.mark.parametrize(
    "batch,reads",
    [
        (view((1, 900, True)), MEMORY),
        (BatchView((RequestShape(8192, 8192, False),), produces_output=True), COMPUTE),
    ],
    ids=["memory", "compute"],
)
@pytest.mark.parametrize("origin", ["probed", "assumed"])
def test_a_term_is_declared_iff_a_field_its_winning_bound_reads_is_stated(
    batch, reads, origin
):
    big = g(sum(r.query_tokens for r in batch.requests), 4096, 4096)
    (term,) = backend(big, _spec(origin)).price(batch).terms
    note = term.provenance.note

    if origin == "probed":
        assert term.provenance.method is Method.ANALYTICAL and not note
    else:
        assert term.provenance.method is Method.DECLARED
        assert [p for p in MEMORY + COMPUTE if p in note] == list(reads)
        assert note.endswith(NONE)


# --- two-batch overlap -------------------------------------------------------------

#: A machine whose kernel floor is a second, so each small leaf costs one.
FLOOR = copy.deepcopy(ROOFLINE)
FLOOR["device"]["kernel_floor_s"] = 1.0
SECOND = MachineSpec.from_mapping(FLOOR)
SCATTER = "atom.model_ops.moe:reduce_scatterv"
#: This rank's 7 tokens split 3 then 4, cutting the first request; its peer
#: splits the same.
SPLIT = BatchView(
    (RequestShape(4, 4, False), RequestShape(3, 10, False)),
    moe_rows=14,
    produces_output=True,
    tbo_tokens=((3, 3), (4, 4)),
)


def layers(*lead, count=3):
    """`lead`, then `count` layers of compute 3, gather 2, experts 4 and
    scatter 2, each leaf a second."""
    gather, scatter = leaf(GATHER, [[2, 8]], x=t([1, 8])), leaf(
        SCATTER, [[1, 8]], x=t([2, 8])
    )
    layer = [g(1, 8, 8)] * 3 + [gather] * 2 + [g(1, 8, 8)] * 4 + [scatter] * 2
    body = {"seq": layer}
    return {
        "seq": [
            *lead,
            {"repeat": body, "count": count, "index": "l0", "start": 0, "step": 1},
        ]
    }


def test_a_split_step_lays_both_micro_batches_on_the_compute_and_comm_streams():
    asked = []

    def graph(*at):
        asked.append(at)
        return layers()

    cost = OpGraphBackend(SECOND, graph, kv="bf16").price(SPLIT)

    # Each micro-batch from its own shape's graph, the second with the output.
    assert asked == [("prefill", 3, 1, False, 6), ("prefill", 4, 2, True, 8)]
    # The timeline ATOM's TBO ring runs: 44 seconds, against 66 one after the
    # other.
    assert by_name(cost) == {"op.tbo": 44.0}
    assert cost.terms[0].provenance.detail.endswith(
        "prefill T=3 B=1 G=6 that produces no output and prefill T=4 B=2 G=8"
    )


def test_a_gather_of_token_ids_runs_inline_on_the_compute_stream():
    ids = leaf(GATHER, [[2, 1]], x=t([1, 1], role="metadata"))
    cost = OpGraphBackend(SECOND, lambda *at: layers(ids, count=1), kv="bf16").price(
        SPLIT
    )

    assert by_name(cost) == {"op.tbo": overlap([[(4, 4), (2, 2), (4, 4), (2, 2)]])}


@pytest.mark.parametrize(
    "batch,graph,split,named",
    [
        (SPLIT, layers(), True, "on MoRI's split send and receive"),
        (
            replace(SPLIT, tbo_tokens=((3, 4), (4, 3))),
            layers(),
            False,
            r"split \(\(3, 4\), \(4, 3\)\) names no single pair",
        ),
        (SPLIT, None, False, "no op_graph holds a prefill step of 3 tokens"),
        (
            replace(SPLIT, produces_output=None),
            layers(),
            False,
            "does not say whether it produces output",
        ),
        (replace(SPLIT, image_tokens=5), layers(), False, "a step with image tokens"),
    ],
    ids=[
        "split-send-recv",
        "no-pair",
        "uncaptured",
        "output-unsaid",
        "images",
    ],
)
def test_a_split_step_with_no_price_is_refused_by_name(batch, graph, split, named):
    with pytest.raises(CostRefused, match=named):
        backend(graph, split_send_recv=split).price(batch)


@pytest.mark.parametrize("join", ["max", "resource_bound"])
def test_a_micro_batch_whose_exchange_shares_a_par_is_refused(join):
    routed = {"seq": [leaf(GATHER, [[2, 8]], x=t([1, 8])), g(1, 8, 8)]}
    dual = {"par": [routed, g(1, 8, 8)], "join": join}

    with pytest.raises(CostRefused, match="runs an MoE exchange beside a second"):
        OpGraphBackend(SECOND, lambda *at: dual, kv="bf16").price(SPLIT)


def test_micro_batches_whose_graphs_exchange_unalike_are_refused():
    def graph(phase, tokens, seqs, output, gathered):
        return layers(count=tokens - 2)

    with pytest.raises(CostRefused, match="change streams 3 and 7 times"):
        OpGraphBackend(SECOND, graph, kv="bf16").price(SPLIT)


def test_a_leaf_one_micro_batch_refuses_is_named_before_the_stream_count():
    def graph(phase, tokens, seqs, output, gathered):
        exchange = "atom.model_ops.moe:foo" if tokens == 3 else GATHER
        return {"seq": [g(1, 8, 8), leaf(exchange, [[2, 8]], x=t([1, 8])), g(1, 8, 8)]}

    with pytest.raises(CostRefused) as refused:
        OpGraphBackend(SECOND, graph, kv="bf16").price(SPLIT)

    assert refused.value.declined == (
        Refusal("op-work", "no extractor for the op kind atom.model_ops.moe:foo"),
    )


def test_the_first_micro_batch_issues_first_on_each_stream():
    gather = leaf(GATHER, [[2, 8]], x=t([1, 8]))

    def graph(phase, tokens, seqs, output, gathered):
        compute, exchange = {3: (1, 10), 4: (10, 1)}[tokens]
        return {"seq": [g(1, 8, 8)] * compute + [gather] * exchange}

    cost = OpGraphBackend(SECOND, graph, kv="bf16").price(SPLIT)

    # The first micro-batch's long exchange hides under the second's compute:
    # 12 seconds, against 21 issued the other way round.
    assert by_name(cost) == {"op.tbo": 12.0}


def test_a_mori_dispatch_with_an_unset_argument_rides_the_comm_stream():
    dispatch = leaf(
        f"{MORI}.dispatch",
        [[2, 8]],
        input=t([1, 8]),
        weights=t([1, 2]),
        scales=None,
        indices=t([1, 2]),
    )
    region = {"seq": [g(1, 8, 8), dispatch, g(1, 8, 8)]}
    cost = OpGraphBackend(SECOND, lambda *at: region, kv="bf16").price(SPLIT)

    assert by_name(cost) == {"op.tbo": overlap([[(1, 1), (1, 1), (1, 1)]])}


def test_an_exchange_two_pars_deep_is_refused():
    routed = {"seq": [leaf(GATHER, [[2, 8]], x=t([1, 8])), g(1, 8, 8)]}
    inner = {"par": [routed, g(1, 8, 8)], "join": "max"}
    nested = {"par": [inner, g(1, 8, 8)], "join": "max"}

    with pytest.raises(CostRefused, match="runs an MoE exchange beside a second"):
        OpGraphBackend(SECOND, lambda *at: nested, kv="bf16").price(SPLIT)


def test_a_split_step_s_note_names_the_stated_fields_of_every_price():
    region = {"seq": [g(8192, 4096, 4096), g(1, 4096, 4096)]}
    (term,) = backend(region, _spec("assumed")).price(SPLIT).terms

    assert term.provenance.method is Method.DECLARED
    assert [p for p in MEMORY + COMPUTE if p in term.provenance.note] == list(
        MEMORY + COMPUTE
    )


# --- the published 27B -----------------------------------------------------------


def test_the_27b_s_captured_decode_step_prices_leaf_by_leaf(tmp_path_factory):
    graph, _ = capture(1, WIDTH_2, tmp_path_factory)["decode B=2"]
    cost = backend(graph["region"]).price(DECODE)
    flat: dict = {}
    for op in _ops(graph["region"]):
        w = work(op, DECODE, 1, "bf16")
        if w is not None:
            flat.setdefault(f"op.{op['op']}", []).append(price(w, SPEC))

    assert by_name(cost) == {name: fold_seconds(s) for name, s in flat.items()}
    assert {f"op.{GEMM}", f"op.{ATTENTION}"} <= by_name(cost).keys()


def test_the_run_s_lookup_reads_the_eager_graph_at_a_shape(tmp_path_factory):
    graph, _ = capture(1, WIDTH_2, tmp_path_factory)["decode B=2"]
    base = tmp_path_factory.getbasetemp()
    hf = SimpleNamespace(architectures=json.loads(CONFIG.read_text())["architectures"])
    found = []
    for store in base.glob("tp1*/store"):
        config = SimpleNamespace(
            model=str(store.parent / "model"),
            hf_config=hf,
            tensor_parallel_size=1,
            parallel_config=SimpleNamespace(data_parallel_size=1),
            enable_expert_parallel=False,
            enforce_eager=False,
        )
        lookup = _op_graphs(str(store), config, "gfx942")
        found.append((lookup("decode", 2, 2), lookup("decode", 3, 3)))

    assert (graph["region"], None) in found


def test_a_27b_middle_chunk_prices_from_its_graph_with_no_lm_head(tmp_path_factory):
    capture(1, PREFILLS, tmp_path_factory)
    hf = SimpleNamespace(architectures=json.loads(CONFIG.read_text())["architectures"])
    lookups = [
        _op_graphs(
            str(store),
            SimpleNamespace(
                model=str(store.parent / "model"),
                hf_config=hf,
                tensor_parallel_size=1,
                parallel_config=SimpleNamespace(data_parallel_size=1),
                enable_expert_parallel=False,
                enforce_eager=True,
            ),
            "gfx942",
        )
        for store in tmp_path_factory.getbasetemp().glob("tp1*/store")
    ]
    lookup = next(at for at in lookups if at("prefill", 512, 2))
    final, middle = lookup("prefill", 512, 2, True), lookup("prefill", 512, 2, False)

    def rows(region):
        return {op["args"]["A"]["shape"][0] for op in _ops(region) if op["op"] == GEMM}

    # The head takes each sequence's last token; a middle chunk runs no head.
    assert (rows(final), rows(middle)) == ({512, 2}, {512})
    assert leaves(middle).keys() < leaves(final).keys()
    chunk = (RequestShape(256, 256, False),) * 2
    priced = {
        output: by_name(
            OpGraphBackend(SPEC, lookup, kv="bf16").price(
                BatchView(chunk, produces_output=output)
            )
        )
        for output in (True, False)
    }
    assert priced[False].keys() <= priced[True].keys()
    assert sum(priced[False].values()) < sum(priced[True].values())
    (head,) = [op for op in _ops(final) if rows(op) == {2}]
    dropped = priced[True][f"op.{GEMM}"] - priced[False][f"op.{GEMM}"]
    assert dropped == pytest.approx(price(work(head, FINAL, 1, "bf16"), SPEC))


@pytest.mark.parametrize("padding,padded", [({}, True), ({"padding": "none"}, False)])
def test_a_run_file_op_graph_installs_the_op_level_backend(
    monkeypatch, tmp_path, padding, padded
):
    monkeypatch.setitem(WORKER, "kv_cache_dtype", "fp8")
    installed = _installed(monkeypatch, tmp_path, op_graph=True, **padding)

    assert isinstance(installed, OpGraphBackend)
    assert (installed.tp, installed.kv, installed.padded) == (1, "fp8", padded)


def test_a_run_file_op_graph_under_low_latency_ep_refuses_its_split_steps(
    monkeypatch, tmp_path
):
    monkeypatch.setitem(WORKER, "enable_low_latency", True)
    split = [
        _installed(monkeypatch, tmp_path, op_graph=True, ep=ep) for ep in (False, True)
    ]

    assert [b.split_send_recv for b in split] == [False, True]


def test_a_run_file_op_graph_that_is_not_true_is_refused(monkeypatch, tmp_path):
    with pytest.raises(TypeError, match="op_graph is 'yes'; it is true"):
        _installed(monkeypatch, tmp_path, op_graph="yes")


def test_a_run_file_with_op_graph_and_another_backend_is_refused(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="has step_linear and op_graph"):
        _installed(monkeypatch, tmp_path, step_linear=law(), op_graph=True)
