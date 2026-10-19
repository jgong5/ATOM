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
two of one kernel, at the step's counts and its rows' mean context; anything
the list declines, a line the fit declined among them, is its roofline,
carrying why.
"""

import json
from types import SimpleNamespace

import pytest
from test_backend_step_linear import _installed, law, view
from test_backend_step_roofline import COMPUTE, MEMORY, _spec
from test_capture_ir import CONFIG, WIDTH_2, _ops, capture
from test_memory_readings import DOCUMENT
from test_op_work import leaf, t
from test_work import SPEC

from atom.compass.artifacts import ArtifactRefusal
from atom.compass.backends import BatchView, CostRefused, Method, RequestShape
from atom.compass.backends.cost import fold_seconds
from atom.compass.backends.op_graph import NONE, OpGraphBackend
from atom.compass.backends.op_work import work
from atom.compass.backends.provenance import Refusal
from atom.compass.backends.work import attention, gemm, price
from atom.compass.bench.ops import signature
from atom.compass.fit import ops as fit_ops
from atom.compass.run import _op_graphs

GEMM = "aiter.gemm_a16w16.default"
ATTENTION = "aiter.unified_attention_with_output_base.default"
DECODE = view((1, 900, True), (1, 300, True), rung=2)


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
                    {"par": [g(4096, 4096, 4096), g(4096, 2048, 4096)], "join": "max"},
                    g(4096, 1024, 4096),
                ],
                "join": "resource_bound",
            },
        ]
    }
    cost = backend(region).price(DECODE)
    one, body = price(gemm(2, 64, 64), SPEC), price(gemm(2, 128, 64), SPEC)
    longest = price(gemm(2, 4096, 4096), SPEC)
    summed = gemm(4096, 4096, 4096) + gemm(4096, 2048, 4096) + gemm(4096, 1024, 4096)

    assert by_name(cost) == {
        f"op.{GEMM}": fold_seconds([one, body, body, body, longest]),
        "par.resource_bound": price(summed, SPEC),
    }


@pytest.mark.parametrize(
    "batch,padded,shape",
    [
        (view(*[(1, 90, True)] * 3, rung=4), True, ("decode", 4, 4)),
        (view(*[(1, 90, True)] * 3, rung=4), False, ("decode", 3, 3)),
        (view((1, 90, True), (1, 9, True)), True, ("decode", 2, 2)),
        (view((100, 100, False), (1, 50, True)), True, ("prefill", 101, 2)),
    ],
    ids=["rung", "padding-none", "eager-decode", "mixed"],
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
        (view((2, 9, True)), g(1, 8, 8), "a step that decodes 2 tokens in a row"),
        (view((1, 9, True)), None, "no op_graph holds a decode step of 1 tokens"),
        (
            BatchView((RequestShape(9, 9, False),), tbo_tokens=((5,), (4,))),
            g(1, 8, 8),
            "two-batch-overlap",
        ),
        (
            BatchView((RequestShape(9, 9, False),), image_tokens=5),
            g(1, 8, 8),
            "5 image tokens",
        ),
        (
            BatchView((RequestShape(9, 9, False),), moe_rows=18),
            g(1, 8, 8),
            "a data-parallel step whose MoE gathers 18 rows",
        ),
    ],
    ids=["speculative", "uncaptured", "tbo", "images", "data-parallel"],
)
def test_a_step_with_no_graph_to_price_it_is_refused_by_name(batch, graph, named):
    with pytest.raises(CostRefused, match=named):
        backend(graph).price(batch)


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
    [(view((1, 900, True)), MEMORY), (view((8192, 8192, False)), COMPUTE)],
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
        lookup = _op_graphs(str(store), config)
        found.append((lookup("decode", 2, 2), lookup("decode", 3, 3)))

    assert (graph["region"], None) in found


@pytest.mark.parametrize("padding,padded", [({}, True), ({"padding": "none"}, False)])
def test_a_run_file_op_graph_installs_the_op_level_backend(
    monkeypatch, tmp_path, padding, padded
):
    installed = _installed(monkeypatch, tmp_path, op_graph=True, **padding)

    assert isinstance(installed, OpGraphBackend)
    assert (installed.tp, installed.kv, installed.padded) == (1, "bf16", padded)


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
    "rows,answer",
    [
        ([(1, 1500, True)] * 2, 1.5e-5),
        ([(1, 1000, True), (1, 2000, True)], 1.5e-5),
        (
            [(1, 1000, True), (1, 3000, True)],
            f"{GEMM}: its rows' context from 2000 to 3000 crosses from a to b",
        ),
    ],
    ids=["uniform", "uneven-in-one-kernel", "uneven-across-two"],
)
def test_a_step_s_rows_price_at_their_mean_inside_one_kernel_to_their_max(rows, answer):
    cost = backend(g(2, 64, 64), price_list=BANDED).price(view(*rows, rung=2))

    if isinstance(answer, float):
        assert by_name(cost) == {f"op.{GEMM}.price_list": pytest.approx(answer)}
        assert cost.terms[0].provenance.method is Method.ESTIMATED
    else:
        assert cost.refusals == (Refusal("price-list", answer),)


ATTN = leaf(ATTENTION, [[64, 6144]], q=t([64, 24, 256]), k=t([64, 4, 256]))
# A prefill of 64 tokens over two rows, each past 0, 1024 or 2048 cached tokens.
CACHED = listed(
    *[
        pt(ATTN, "prefill T=64 B=2", ns, c, rows=[[32, 32 + c, False]] * 2)
        for c, ns in ((0, 1e4), (1024, 2e4), (2048, 3e4))
    ]
)


@pytest.mark.parametrize(
    "rows,params,answer",
    [
        ([(32, 32 + 512, False)] * 2, CACHED, 1.5e-5),
        ([(32, 32, False), (32, 32 + 2048, False)], CACHED, 2e-5),
        (
            [(32, 32 + 512, False)] * 2,
            listed(pt(ATTN, "prefill T=64 B=2", 1e4)),
            (
                f"{ATTENTION}: at T = 64, at B = 2, context = 512 is outside the "
                "measured 0 to 0"
            ),
        ),
    ],
    ids=["uniform", "uneven", "fresh-rows-only"],
)
def test_a_prefill_prices_linearly_in_its_rows_cached_tokens(rows, params, answer):
    cost = backend(ATTN, price_list=params).price(view(*rows))

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


def test_a_run_file_price_list_from_another_tree_is_refused_at_load(
    monkeypatch, tmp_path
):
    fields = _published(tmp_path, root="b" * 40)

    with pytest.raises(ArtifactRefusal, match="`price_list` x `ATOM src`"):
        _installed(monkeypatch, tmp_path, op_graph=True, price_list=fields)


def test_a_run_file_price_list_without_op_graph_is_refused(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="names a price_list and no op_graph"):
        _installed(monkeypatch, tmp_path, step_linear=law(), price_list={})
