# SPDX-License-Identifier: MIT
"""A run's decode steps priced through a published price list, as a run asks
them: each step prices as a backend new to it does, while each leaf's
signature, and the work of a leaf whose work reads nothing of the step, is
taken once over the run, the work of a leaf whose work reads the step only
where the list declines it, a leaf whose answer reads nothing the steps move
is asked at a graph's first two steps alone, and a step at a view priced
before asks the list nothing; and a backend keeps alive the graphs of no more steps than a run's
graph lookup keeps graphs.
"""

import collections
import gc
import weakref
from types import SimpleNamespace

from test_artifact_store import clean_checkout  # noqa: F401  autouse
from test_fit_ops import SPEC, _at, _fit, measured  # noqa: F401  fixture
from test_op_work import leaf, t

from atom.compass import run
from atom.compass.artifacts import ArtifactStore
from atom.compass.backends import BatchView, RequestShape, op_graph
from atom.compass.backends.cost import CostTerm
from atom.compass.backends.op_graph import LISTED, OpGraphBackend
from atom.compass.backends.provenance import Method, Provenance
from atom.compass.backends.resolver import CostSource
from atom.compass.bench import ops as bench
from atom.compass.fit import ops, publish

GEMM = "aiter.gemm_a16w16.default"
ATTENTION = "aiter.unified_attention_with_output_base.default"
VIEW = "aten.view.default"
#: Each step's one row's context: one past the measured range, and the first
#: again, as a DP rank's dummy forwards repeat one view.
CONTEXTS = (1000, 1001, 1002, 2000, 5000, 2001, 1000)


def _gemm():
    return leaf(GEMM, [[1, 64]], A=t([1, 64]), B=t([64, 64], role="weight"), bias=None)


def _attention():
    return leaf(ATTENTION, [[1, 8, 64]], q=t([1, 8, 64]), k=t([1, 1, 64]))


def _layers():
    """Two layers of a view, a GEMM and an attention, each leaf its own object."""
    view = leaf(VIEW, [[1, 64]], self=t([1, 64]), size=[1, 64])
    return {"seq": [view, _gemm(), _attention(), _gemm(), _attention()]}


def _terms(backend, step):
    cost = backend.price(step)
    return cost.terms, cost.draft_terms


def test_steps_price_as_new_while_each_leaf_is_signed_and_worked_out_once(
    measured, tmp_path, monkeypatch  # noqa: F811
):
    # the GEMM flat over context, attention moving with it, both at 1000 and 3000
    points = [_at(_gemm(), "decode B=1", 5e3, c) for c in (1000, 3000)]
    points += [_at(_attention(), "decode B=1", c * 10.0, c) for c in (1000, 3000)]
    d, _ = measured("m", points)
    got = _fit(tmp_path / "fit", tmp_path / "store", d)
    _, config, _ = ops.read(ArtifactStore(str(tmp_path / "store")), [d])
    monkeypatch.setattr(
        "atom.compass.run.step_grid.conditions",
        lambda c, m: publish.conditions(config),
    )

    def backend(region):
        listed = {"store": str(tmp_path / "store"), "price_list": got["key"]}
        return OpGraphBackend(
            SPEC,
            lambda *_, **__: region,
            kv="bf16",
            price_list=run._price_list(listed, None, None),
        )

    region = _layers()
    steps = [BatchView((RequestShape(1, c, True),)) for c in CONTEXTS]
    new = [_terms(backend(region), step) for step in steps]
    works, signed = collections.Counter(), []
    work, signature = op_graph.work, bench.signature

    def counted(leaf, *args):
        works[leaf["op"]] += 1
        return work(leaf, *args)

    monkeypatch.setattr(op_graph, "work", counted)
    monkeypatch.setattr(
        bench, "signature", lambda leaf: signed.append(1) or signature(leaf)
    )
    priced = backend(region)
    listed, asked = priced.resolver.sources[0], []
    price = listed.price
    monkeypatch.setattr(listed, "price", lambda r: asked.append(1) or price(r))

    assert [_terms(priced, step) for step in steps] == new
    assert [sorted(t.name for t in terms) for terms, _ in new] == [
        [f"op.{GEMM}.price_list", f"op.{ATTENTION}" + (".price_list" * (c <= 3000))]
        for c in CONTEXTS
    ]
    # attention's work at the first step and where the list declines it
    assert works == {VIEW: 1, GEMM: 1, ATTENTION: 2}
    assert len(signed) == 2
    # the GEMM at the graph's first two steps, its answer reading no row's
    # context; attention once a step, but at the view priced before
    assert len(asked) == 2 + len(set(CONTEXTS))


class _Region(dict):
    """A graph's region a weak reference can follow."""


class _Flat(CostSource):
    """A price list that prices every leaf at 1 us."""

    name = LISTED

    def price(self, request):
        return CostTerm(f"op.{request[0]['op']}", 1e-6, Provenance(Method.MEASURED))


def test_a_backend_keeps_alive_no_more_steps_graphs_than_a_run_s_lookup_keeps(
    tmp_path,
):
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_size=1), speculative_config=None
    )
    kept = run._op_graphs(str(tmp_path), config, "gfx950").cache_info().maxsize
    # a lookup that parses each step's graph anew, as `run._op_graphs` does a
    # graph it dropped, over one step more than it keeps
    parsed = []

    def graph(*_, **__):
        region = _Region(seq=[_gemm(), _attention()])
        parsed.append(weakref.ref(region))
        return region

    backend = OpGraphBackend(SPEC, graph, kv="bf16", price_list=_Flat())
    for c in range(kept + 1):
        backend.price(BatchView((RequestShape(1, 1000 + c, True),)))
    gc.collect()

    assert sum(r() is not None for r in parsed) <= kept
