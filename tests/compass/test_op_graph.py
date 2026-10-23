# SPDX-License-Identifier: MIT
"""The op-level backend: a step priced from the graph captured at its shape.

Each leaf is one roofline over its work; a `Seq` adds, a `Repeat` re-emits its
body's prices, a `max` `Par` takes its longest branch, and a
`resource_bound` `Par` costs one roofline over the branches' summed work. A step runs at its rung's
graph, or its width's under ``padding: none``, a prefill at its tokens and
rows; attention reads its context in the run's KV format. What cannot be
priced is refused by name, each reason once. The Qwen3.8-27B TP1 capture prices leaf by leaf, and `run.runner` installs the
backend from a run file's ``op_graph`` over the store's graphs. With a price
list, a leaf takes its price first, measured on a point or estimated between
two of one kernel, at the step's counts and its rows' context, a decode's
longest row and a prefill's mean cached tokens; anything
the list declines, a line the fit declined among them, is its roofline,
carrying why.
"""

import copy
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from test_backend_step_linear import _installed, law, view
from test_backend_step_roofline import COMPUTE, MEMORY, ROOFLINE, _spec
from test_capture_ir import CONFIG, WIDTH_2, _ops, capture
from test_memory_readings import DOCUMENT, WORKER
from test_op_graph_arch import ENGINE
from test_op_work import leaf, t
from test_work import SPEC

from atom.compass import run as compass_run
from atom.compass.artifacts import ArtifactRefusal
from atom.compass.backends import BatchView, CostRefused, Method, RequestShape
from atom.compass.backends.cost import fold_seconds
from atom.compass.backends.op_graph import NONE, OpGraphBackend
from atom.compass.backends.op_work import work
from atom.compass.backends.provenance import Refusal
from atom.compass.backends.step_roofline import overlap
from atom.compass.backends.work import attention, gemm, price
from atom.compass.bench.ops import leaves, signature
from atom.compass.capture.peers import MORI
from atom.compass.fit import ops as fit_ops
from atom.compass.run import _op_graphs
from atom.compass.runner.overrides import PricedSteps
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


def _engine(graph) -> dict:
    """`ENGINE` at the quantization hash the capture's own config keyed `graph`
    by, which a stand-in config cannot compute."""
    quantization = json.loads(graph["structure"].split()[3])["quantization"]
    return {
        **ENGINE,
        "quant_config": SimpleNamespace(compute_hash=lambda: quantization),
    }


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
            **_engine(graph),
        )
        lookup = _op_graphs(str(store), config, "gfx942")
        found.append((lookup("decode", 2, 2), lookup("decode", 3, 3)))

    assert (graph["region"], None) in found


def test_a_27b_middle_chunk_prices_from_its_graph_with_no_lm_head(tmp_path_factory):
    (graph, _), *_ = capture(1, PREFILLS, tmp_path_factory).values()
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
                **_engine(graph),
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


# --- the price list first ------------------------------------------------------


def pt(leaf, shape, ns, context=None, kernel="k", rows=None):
    """A `measure ops` point of `leaf` at `shape`."""
    point = {"shape": shape, "context": context, "op": leaf["op"], "cold_ns": ns}
    point |= {"leaf": signature(leaf), "kernels": [kernel]}
    return point | ({"rows": rows} if rows else {})


def listed(*points):
    """`compass fit ops`'s params over `points`, declined lines included, as
    the price list a run reads."""
    params = fit_ops.fit(list(points))
    fit_ops.held_out(params)
    return fit_ops.PriceList(params)


# A GEMM whose time does not move with context: its family runs over B alone.
FLAT = listed(
    *[
        pt(g(b, 64, 64), f"decode B={b}", ns, c)
        for b, ns in ((2, 5e3), (4, 7e3))
        for c in (100, 1000)
    ]
)


@pytest.mark.parametrize(
    "batch,padded,priced,method",
    [
        (DECODE, True, 5e-6, Method.MEASURED),
        (view(*[(1, 90, True)] * 3), False, 6e-6, Method.ESTIMATED),
    ],
    ids=["on-a-point", "between-two"],
)
def test_a_leaf_the_price_list_holds_is_priced_from_it(batch, padded, priced, method):
    (term,) = backend(g(2, 64, 64), padded=padded, price_list=FLAT).price(batch).terms

    assert (term.name, term.seconds) == (f"op.{GEMM}.price_list", pytest.approx(priced))
    assert (term.provenance.method, term.provenance.source) == (method, "price-list")
    assert not term.provenance.refusals


def test_a_leaf_the_price_list_declines_is_its_roofline_and_carries_why():
    wide = view(*[(1, 90, True)] * 5, rung=8)
    alone = backend(g(2, 64, 64)).price(wide)
    cost = backend(g(2, 64, 64), price_list=FLAT).price(wide)

    assert (
        by_name(cost) == by_name(alone) == {f"op.{GEMM}": by_name(alone)[f"op.{GEMM}"]}
    )
    assert cost.fell_back and cost.terms[0].provenance.method is Method.ANALYTICAL
    assert cost.refusals == (
        Refusal("price-list", f"{GEMM}: B = 8 is outside the measured 2 to 4"),
    )


# Along context at B = 2 the middle point reads +25% off its neighbours' line.
BENT = listed(
    *[
        pt(g(2, 64, 64), "decode B=2", ns, c)
        for c, ns in ((1000, 1e4), (2000, 2e4), (3000, 4e4))
    ]
)


@pytest.mark.parametrize(
    "rows,answer",
    [
        ([(1, 2000, True)] * 2, 2e-5),
        (
            [(1, 1500, True)] * 2,
            (
                f"{GEMM}: the fit declined context at B = 2: context = 2000 reads "
                "+25.0% off its neighbours"
            ),
        ),
    ],
    ids=["its-measured-point", "along-it"],
)
def test_a_line_the_fit_declined_answers_its_points_only(rows, answer):
    cost = backend(g(2, 64, 64), price_list=BENT).price(view(*rows, rung=2))

    if isinstance(answer, float):
        assert by_name(cost) == {f"op.{GEMM}.price_list": pytest.approx(answer)}
    else:
        assert cost.refusals == (Refusal("price-list", answer),)


# Linear in context, one kernel to 2000 and another at 3000.
BANDED = listed(
    *[
        pt(g(2, 64, 64), "decode B=2", ns, c, kernel)
        for c, ns, kernel in ((1000, 1e4, "a"), (2000, 2e4, "a"), (3000, 3e4, "b"))
    ]
)


@pytest.mark.parametrize(
    "rows,priced,method",
    [
        ([(1, 1500, True)] * 2, 1.5e-5, Method.ESTIMATED),
        ([(1, 1000, True), (1, 2000, True)], 2e-5, Method.MEASURED),
        ([(1, 1000, True), (1, 3000, True)], 3e-5, Method.MEASURED),
    ],
    ids=["uniform", "uneven-in-one-kernel", "uneven-across-two"],
)
def test_a_decode_step_prices_at_its_longest_row_s_context(rows, priced, method):
    # decode attention's time follows its longest row, so does the query
    cost = backend(g(2, 64, 64), price_list=BANDED).price(view(*rows, rung=2))

    assert by_name(cost) == {f"op.{GEMM}.price_list": pytest.approx(priced)}
    assert cost.terms[0].provenance.method is method


ATTN = leaf(ATTENTION, [[64, 6144]], q=t([64, 24, 256]), k=t([64, 4, 256]))
# A prefill of 64 tokens over two rows, each past 0, 1024 or 2048 cached tokens.
CACHED = listed(
    *[
        pt(ATTN, "prefill T=64 B=2", ns, c, rows=[[32, 32 + c, False]] * 2)
        for c, ns in ((0, 1e4), (1024, 2e4), (2048, 3e4))
    ]
)
# The same, one kernel to 1024 cached tokens and another at 2048.
BANDED_CACHED = listed(
    *[
        pt(ATTN, "prefill T=64 B=2", ns, c, kernel, [[32, 32 + c, False]] * 2)
        for c, ns, kernel in ((0, 1e4, "a"), (1024, 2e4, "a"), (2048, 3e4, "b"))
    ]
)


@pytest.mark.parametrize(
    "rows,params,answer",
    [
        ([(32, 32 + 512, False)] * 2, CACHED, 1.5e-5),
        ([(32, 32, False), (32, 32 + 2048, False)], CACHED, 2e-5),
        (
            [(32, 32, False), (32, 32 + 2048, False)],
            BANDED_CACHED,
            f"{ATTENTION}: its rows' cached tokens from 1024 to 2048 cross from a to b",
        ),
        (
            [(32, 32 + 512, False)] * 2,
            listed(pt(ATTN, "prefill T=64 B=2", 1e4)),
            (
                f"{ATTENTION}: at T = 64, at B = 2, context = 512 is outside the "
                "measured 0 to 0"
            ),
        ),
    ],
    ids=["uniform", "uneven", "uneven-across-two", "fresh-rows-only"],
)
def test_a_prefill_prices_linearly_in_its_rows_cached_tokens(rows, params, answer):
    final = replace(view(*rows), produces_output=True)
    cost = backend(ATTN, price_list=params).price(final)

    if isinstance(answer, float):
        assert by_name(cost) == {f"op.{ATTENTION}.price_list": pytest.approx(answer)}
    else:
        assert cost.refusals == (Refusal("price-list", answer),)


def test_a_resource_bound_par_prices_its_leaves_work_not_their_listed_prices():
    big = g(2, 4096, 4096)
    params = listed(
        *[pt(g(b, 4096, 4096), f"decode B={b}", 1e3, c) for b in (2, 4) for c in (1, 9)]
    )
    pooled = {"par": [big, big], "join": "resource_bound"}
    cost = backend(pooled, price_list=params).price(DECODE)

    summed = gemm(2, 4096, 4096) + gemm(2, 4096, 4096)

    assert by_name(cost) == {"par.resource_bound": price(summed, SPEC)}


def test_each_micro_batch_of_a_split_step_prices_at_its_own_shape():
    # flat in cached tokens, so each prefill family runs over T alone
    params = listed(
        *[
            pt(g(t, 8, 8), f"prefill T={t} B={b}", ns, c)
            for t, b, ns in ((3, 1, 3e9), (4, 2, 4e9))
            for c in (0, 8)
        ]
    )

    def graph(phase, tokens, seqs, output, gathered):
        return {"seq": [g(tokens, 8, 8)]}

    cost = OpGraphBackend(SECOND, graph, kv="bf16", dp=2, price_list=params)
    cost = cost.price(SPLIT)

    # each a second as a roofline: 3 and 4 seconds are the list's
    assert by_name(cost) == {"op.tbo.price_list": overlap([[(3.0, 4.0)]])}
    assert cost.terms[0].provenance.method is Method.MEASURED


@pytest.mark.parametrize(
    "rows,shape,ns,want",
    [(3, "prefill T=3 B=1", 3e9, (3.0, 1.0)), (4, "prefill T=4 B=2", 4e9, (1.0, 4.0))],
    ids=["first-listed", "second-listed"],
)
def test_a_split_step_the_list_prices_in_part_is_a_roofline_carrying_why(
    rows, shape, ns, want
):
    # the list holds one micro-batch's GEMM only; the other's is its
    # roofline, a second
    params = listed(*[pt(g(rows, 8, 8), shape, ns, c) for c in (0, 8)])

    def graph(phase, tokens, seqs, output, gathered):
        return {"seq": [g(tokens, 8, 8)]}

    cost = OpGraphBackend(SECOND, graph, kv="bf16", dp=2, price_list=params)
    cost = cost.price(SPLIT)

    assert by_name(cost) == {"op.tbo": overlap([[want]])}
    assert cost.refusals == (
        Refusal("price-list", f"{GEMM}: no point of its signature was measured"),
    )


def flat(n, *bs):
    """Decode points of a GEMM into `n` outputs, B microseconds at B, flat in
    context, so its family runs over B alone."""
    return [
        pt(g(b, n, 64), f"decode B={b}", b * 1e3, c) for b in bs for c in (100, 1000)
    ]


THIRTY_TWO = view(*[(1, 90, True)] * 32)


@pytest.mark.parametrize(
    "params,validated",
    [
        (listed(*flat(64, 1, 64)), False),
        (listed(*flat(64, 1, 8, 64)), True),
        (fit_ops.PriceList(fit_ops.fit(flat(64, 1, 8, 64))), False),
    ],
    ids=["B=1,64", "B=1,8,64", "no-brackets"],
)
def test_b_32_is_validated_only_inside_a_held_out_bracket(params, validated):
    # on 1, 8, 64 the held-out B = 8 was priced from 1 and 64, around 32
    listing = backend(g(2, 64, 64), padded=False, price_list=params)
    (term,) = listing.price(THIRTY_TWO).terms

    assert term.seconds == pytest.approx(32e-6)
    assert term.provenance.method is Method.ESTIMATED
    assert term.provenance.validated is validated


@pytest.mark.parametrize(
    "bs,contexts,context,validated",
    [
        ((1, 8, 64), (1000, 2000, 3000), 1500, True),
        ((1, 8, 64), (1000, 3000), 1500, False),
        ((1, 64), (1000, 2000, 3000), 1500, False),
        ((1, 8, 64), (1000, 3000), 1000, True),
    ],
    ids=["both-bracketed", "context-not", "b-not", "along-b-only"],
)
def test_a_query_on_two_dimensions_is_validated_only_when_both_lines_are(
    bs, contexts, context, validated
):
    params = listed(
        *[
            pt(g(b, 64, 64), f"decode B={b}", b * 1e4 + c * 10, c)
            for b in bs
            for c in contexts
        ]
    )
    step = view(*[(1, context, True)] * 32)
    (term,) = backend(g(2, 64, 64), padded=False, price_list=params).price(step).terms

    assert term.seconds == pytest.approx((32e4 + context * 10) * 1e-9)
    assert term.provenance.validated is validated


@pytest.mark.parametrize("bs,validated", [((1, 8, 64), True), ((1, 64), False)])
def test_a_term_of_measured_and_validated_leaves_is_validated(bs, validated):
    # N = 64 is measured at B = 32; N = 128 is estimated between its points
    params = listed(*flat(64, 1, 32, 64), *flat(128, *bs))
    pair = {"seq": [g(2, 64, 64), g(2, 128, 64)]}
    (term,) = backend(pair, padded=False, price_list=params).price(THIRTY_TWO).terms

    assert term.seconds == pytest.approx(64e-6)
    assert term.provenance.method is Method.ESTIMATED
    assert term.provenance.validated is validated


@pytest.mark.parametrize("ts,validated", [((2, 4, 5, 8), True), ((2, 4), False)])
def test_a_split_step_of_measured_and_validated_leaves_is_validated(ts, validated):
    # T = 4 is measured; T = 3 lies inside 2 to 5, which priced T = 4 held out
    params = listed(
        *[pt(g(t, 8, 8), f"prefill T={t} B=1", t * 1e9, c) for t in ts for c in (0, 8)]
    )

    def graph(phase, tokens, seqs, output, gathered):
        return {"seq": [g(tokens, 8, 8)]}

    cost = OpGraphBackend(SECOND, graph, kv="bf16", dp=2, price_list=params)
    (term,) = cost.price(SPLIT).terms

    assert term.name == "op.tbo.price_list"
    assert term.provenance.method is Method.ESTIMATED
    assert term.provenance.validated is validated


# An MoE GEMM over the rows a DP-2 group gathers, at T = 4 and two stated G;
# flat in cached tokens, so its family runs over G alone.
GATHERED = listed(
    *[
        pt(g(rows, 8, 8), f"prefill T=4 B=1 G={rows}", ns, c)
        for rows, ns in ((12, 3e9), (16, 4e9))
        for c in (0, 8)
    ]
)


@pytest.mark.parametrize(
    "rows,answer,method",
    [
        (12, 3.0, Method.MEASURED),
        (14, 3.5, Method.ESTIMATED),
        (32, f"{GEMM}: G = 32 is outside the measured 12 to 16", None),
        (8, f"{GEMM}: no point of its signature was measured", None),
    ],
    ids=["measured", "between-two", "unmeasured", "every-rank-s-t"],
)
def test_a_gathered_count_nobody_measured_is_declined(rows, answer, method):
    def graph(phase, tokens, seqs, output, gathered):
        return {"seq": [g(gathered, 8, 8)]}

    step = replace(FINAL, requests=(RequestShape(4, 4, False),), moe_rows=rows)
    cost = OpGraphBackend(SPEC, graph, kv="bf16", dp=2, price_list=GATHERED).price(step)

    if method is None:
        assert cost.refusals == (Refusal("price-list", answer),)
    else:
        assert by_name(cost) == {f"op.{GEMM}.price_list": pytest.approx(answer)}
        assert cost.terms[0].provenance.method is method


def test_a_prefill_kind_measured_at_one_row_count_prices_another():
    # no two points share T and cached tokens, so no point says rows move it
    params = listed(
        *[
            pt(g(t, 8, 8), f"prefill T={t} B=1", ns, c)
            for t, ns in ((4, 4e9), (8, 8e9))
            for c in (0, 8)
        ]
    )
    step = replace(FINAL, requests=(RequestShape(2, 2, False),) * 2)
    cost = backend(g(4, 8, 8), price_list=params).price(step)

    assert by_name(cost) == {f"op.{GEMM}.price_list": pytest.approx(4.0)}
    assert cost.terms[0].provenance.method is Method.MEASURED


def _published(tmp_path, root=None):
    """The fields of a `price_list` key holding `FLAT`, published under the
    readings a run on the suite's machine document takes from this tree, or
    from the tree `root`."""
    from atom.compass.artifacts import Key, Kind, Reading, git_tree_root, module_root
    from atom.compass.fit.publish import publish
    from atom.compass.run import _machine
    from atom.compass.run.step_grid import readings

    tree = git_tree_root("atom", module_root("atom"))
    source = Reading("tree", root) if root else Reading.of_source_root(tree)
    machine = _machine({"machine": DOCUMENT})
    conditions = readings(machine, atom_src=source, model="m", engine_config="{}")
    key = Key.of(
        Kind.PRICE_LIST,
        model="sha256:" + "0" * 64,
        dp=1,
        pp=1,
        pcp=1,
        tp=1,
        decode_context_parallel_size=1,
        enable_expert_parallel=False,
        enable_dp_attention=False,
        machine_spec="sha256:" + "1" * 64,
        source_root="tree-" + "a" * 40,
    )
    args = SimpleNamespace(store=str(tmp_path / "store"), out=str(tmp_path), ptl=None)
    publish(args, "compass fit ops", key, conditions, FLAT.params, {})
    return key.values


def test_a_run_file_price_list_prices_leaves_from_the_store_s_entry(
    monkeypatch, tmp_path
):
    fields = _published(tmp_path)
    installed = _installed(monkeypatch, tmp_path, op_graph=True, price_list=fields)

    assert installed.resolver.sources[0].params == FLAT.params


def test_a_run_file_price_list_under_dp_2_prices_a_step_gathering_2t(
    monkeypatch, tmp_path
):
    fields = _published(tmp_path)
    installed = _installed(
        monkeypatch, tmp_path, dp=2, op_graph=True, price_list=fields
    )
    installed.graph = lambda *shape: {"seq": [g(2, 64, 64)]}

    (term,) = installed.price(replace(DECODE, moe_rows=4)).terms

    assert (term.seconds, term.provenance.method) == (
        pytest.approx(5e-6),
        Method.MEASURED,
    )


def test_a_run_file_price_list_from_another_tree_is_refused_at_load(
    monkeypatch, tmp_path
):
    fields = _published(tmp_path, root="b" * 40)

    with pytest.raises(ArtifactRefusal, match="`price_list` x `ATOM src`"):
        _installed(monkeypatch, tmp_path, op_graph=True, price_list=fields)


def test_a_run_file_price_list_without_op_graph_is_refused(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="names a price_list and no op_graph"):
        _installed(monkeypatch, tmp_path, step_linear=law(), price_list={})


# --- a shape no graph holds, traced in the run -----------------------------------


def _reader(graph, model) -> SimpleNamespace:
    """A run's config of the 27B at `model`, keyed as the capture keyed `graph`."""
    hf = SimpleNamespace(architectures=json.loads(CONFIG.read_text())["architectures"])
    return SimpleNamespace(
        model=str(model),
        hf_config=hf,
        tensor_parallel_size=1,
        parallel_config=SimpleNamespace(data_parallel_size=1),
        enable_expert_parallel=False,
        enforce_eager=True,
        **_engine(graph),
    )


def _counts() -> dict:
    return {"lazy_traces": 0, "lazy_trace_wall_seconds": 0}


def test_a_shape_no_graph_holds_is_traced_in_the_run_as_trace_ops_traces_it(
    tmp_path_factory, tmp_path
):
    (graph, _), *_ = capture(1, PREFILLS, tmp_path_factory).values()
    stores = tmp_path_factory.getbasetemp().glob("tp1*/store")
    model = next(tmp_path_factory.getbasetemp().glob("tp1*/model"))
    config, lazy = _reader(graph, model), _counts()
    offline = next(
        lookup
        for lookup in (_op_graphs(str(s), config, "gfx942") for s in stores)
        if lookup("prefill", 512, 2)
    )
    lookup = _op_graphs(str(tmp_path), config, "gfx942", ["--model", str(model)], lazy)
    chunk = (RequestShape(256, 256, False),) * 2
    for output in (True, False):
        view = BatchView(chunk, produces_output=output)
        OpGraphBackend(SPEC, lookup, kv="bf16").price(view)

    # Both chunks of the shape came from one trace, the graphs trace ops publishes.
    assert [lookup("prefill", 512, 2, o) for o in (True, False)] == [
        offline("prefill", 512, 2, o) for o in (True, False)
    ]
    assert lazy["lazy_traces"] == 1 and lazy["lazy_trace_wall_seconds"] > 0


def _lazily(tmp_path_factory, root, *atom_args, **engine):
    """The run's lookup of the 27B under `engine`'s config fields into the
    store at `root`, tracing a missing shape from the server's `atom_args`, and
    its counts."""
    (graph, _), *_ = capture(1, PREFILLS, tmp_path_factory).values()
    model = next(tmp_path_factory.getbasetemp().glob("tp1*/model"))
    config, lazy = _reader(graph, model), _counts()
    vars(config).update(engine)
    args = ["--model", str(model), *atom_args]
    return _op_graphs(str(root), config, "gfx942", args, lazy), lazy


def test_a_prefill_two_batch_overlap_would_split_is_traced_lazily_as_one_forward(
    tmp_path_factory, tmp_path, monkeypatch
):
    # ATOM splits a prefill under --enable-tbo from this many tokens on.
    monkeypatch.setenv("ATOM_TBO_PREFILL_MIN_TOKENS", "256")
    lookup, lazy = _lazily(tmp_path_factory, tmp_path, "--enable-tbo", enable_tbo=True)

    assert isinstance(lookup("prefill", 512, 2), dict)
    assert lazy["lazy_traces"] == 1


def test_a_lazy_trace_reads_atom_s_arguments_in_either_spelling_as_its_server_does(
    tmp_path_factory, tmp_path
):
    args = "--kv-cache-dtype", "fp8"
    lookup, lazy = _lazily(tmp_path_factory, tmp_path, *args, kv_cache_dtype="fp8")

    assert isinstance(lookup("decode", 2, 2), dict)
    assert lazy["lazy_traces"] == 1


def test_a_lazy_trace_that_fails_ends_the_run_naming_the_shape(tmp_path):
    (tmp_path / "model").mkdir()  # holds no config.json, so no capture builds it
    (tmp_path / "read").mkdir()
    (tmp_path / "read" / "config.json").write_text("{}")
    config = SimpleNamespace(
        model=str(tmp_path / "read"),
        hf_config=SimpleNamespace(architectures=["Qwen3ForCausalLM"]),
        tensor_parallel_size=1,
        parallel_config=SimpleNamespace(data_parallel_size=1),
        enable_expert_parallel=False,
        enforce_eager=True,
        **ENGINE,
    )
    lazy = _counts()
    args = ["--model", str(tmp_path / "model")]
    lookup = _op_graphs(str(tmp_path / "store"), config, "gfx942", args, lazy)
    rows = BatchView((RequestShape(100, 100, False),) * 3, produces_output=True)

    with pytest.raises(RuntimeError, match="a lazy trace of prefill T=300 B=3 failed"):
        OpGraphBackend(SPEC, lookup, kv="bf16").price(rows)
    assert lazy["lazy_traces"] == 0


def test_a_run_file_op_graph_traces_from_the_server_s_arguments(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(
        compass_run, "_op_graphs", lambda *a: seen.append(a[3:]) or (lambda *s: None)
    )
    _installed(monkeypatch, tmp_path, op_graph=True, atom_args=["--model", "m"])

    assert seen == [(["--model", "m"], _counts())]


def test_the_run_summary_totals_each_engine_s_lazy_traces(monkeypatch, tmp_path):
    run = {"admission_path": "serving", "ipc_s": 2.0**-10, "stream_s": 2.0**-9}
    run.update(bound_s=100.0, client_runs=None, out_dir=str(tmp_path))
    monkeypatch.setenv(compass_run.ENV, str(tmp_path / "run.json"))
    authority = compass_run._RecordingAuthority(run)
    worker = SimpleNamespace(compass_lazy={**_counts(), "lazy_traces": 2})
    worker.compass_lazy["lazy_trace_wall_seconds"] = 9.5
    compass_run._hand_in(run, "engine", [], priced=PricedSteps.priced_steps(worker))
    compass_run._hand_in(run, "frontend", [])
    compass_run._write_summary(run, authority, ended=True)

    cost = json.loads((tmp_path / compass_run.SUMMARY_FILE).read_text())["cost"]
    assert (cost["lazy_traces"], cost["lazy_trace_wall_seconds"]) == (2, 9.5)
