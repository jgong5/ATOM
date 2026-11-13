# SPDX-License-Identifier: MIT
"""A padded decode's MoE priced at its real rows: `measure ops` routes a
point's MoE rows past its real ones (``R=``) to no expert, as ATOM routes a
pad row, `--plan` measures each rung's routed leaves at the real rows its
steps' MoE ran, `fit ops` interpolates along them, and `OpGraphBackend` asks a
step's own."""

import json
import pathlib
from types import SimpleNamespace

import pytest
import torch
from test_artifact_store import clean_checkout  # noqa: F401  autouse
from test_bench_ops import KEY, _graph, _leaf, _t
from test_bench_ops_moe import MOE, _spec, _stub, _timed
from test_bench_ops_plan import CONFIG, bench  # noqa: F401  a fixture
from test_op_graph import g
from test_work import SPEC

from atom.compass import ir
from atom.compass.backends import BatchView, RequestShape
from atom.compass.backends.op_graph import LISTED, OpGraphBackend
from atom.compass.backends.resolver import CostSource
from atom.compass.bench import ops
from atom.compass.capture.forward import decode
from atom.compass.fit import ops as fit_ops

FUSED = "aiter.fused_moe_.default"
#: The real rows of 13 spread over 8 ranks of 4 gathered rows each.
SPREAD = [2, 2, 2, 2, 2, 1, 1, 1]


def test_a_point_routes_each_rank_s_rows_past_its_real_ones_to_no_expert():
    stub = _stub((16, 2))
    # ATOM's ids buffer, its third column a fused shared expert
    stub.shared = torch.arange(96, dtype=torch.int32).view(32, 3)
    every = stub._route(_spec([32, 3]), 0)
    stub.real = (13, 8, 32)  # 13 real rows of the 32 a DP group of 8 gathers

    ids = stub._route(_spec([32, 3]), 0)

    real = [4 * r + i for r, n in enumerate(SPREAD) for i in range(n)]
    pad = sorted(set(range(32)) - set(real))
    assert (ids[pad, :2] == -1).all()
    assert torch.equal(ids[real, :2], every[real, :2])  # one seeded draw
    assert torch.equal(ids[:, 2], stub.shared[:, 2])  # its shared expert on every row
    assert sum(stub.loads[-1]) == 13 * 2
    # a rank's own rows before the gather route as they are
    assert (stub._route(_spec([4, 2]), 0) >= 0).all()


def test_a_point_records_the_rows_its_ids_route_and_draws_its_real_rows_alone():
    stub = _stub((16, 2))
    stub.real = (2, 1, 4)  # one rank's two real rows of its rung of 4

    record, copies = _timed(stub, MOE, 3)

    assert record["routed_rows"] == 4
    for c in copies:
        assert (c["topk_ids"][2:] == -1).all() and (c["topk_ids"][:2] >= 0).all()
    assert record["routing"]["activated"] == [
        len(c["topk_ids"][:2].unique()) for c in copies
    ]


def test_a_shape_states_its_real_rows_and_its_graph_is_its_rung_s(monkeypatch):
    shape = decode(4, 64)._replace(real=13)
    asked = []
    monkeypatch.setattr("atom.compass.capture.forward.structure", lambda *a, **k: "")
    monkeypatch.setattr(
        "atom.compass.capture.graph_key", lambda at, text: asked.append(text)
    )
    store = SimpleNamespace(
        directory_for=lambda key: pathlib.Path("/"),
        read=lambda key: SimpleNamespace(read_member=lambda *a: '{"region": {}}'),
    )

    ops.graph(store, None, shape)

    assert str(shape) == "decode B=4 G=64 R=13"
    assert ops._shape(str(shape)) == shape
    assert asked == ["decode B=4 G=64"]


def _routed_graph(dp):
    """`test_bench_ops_plan`'s graph with an MoE leaf whose ids hold the rows
    its group of `dp` gathers."""

    def at(store, config, shape, arch=None):
        rows = shape.tokens if shape.phase == "prefill" else shape.seqs
        attn = _leaf(
            "aiter.attn",
            ("q", _t([rows, 8], "activation")),
            ("layer", ir.ContextRef(KEY)),
            key=KEY,
        )
        ids = _t([dp * rows, 2], "activation", dtype="torch.int32")
        return _graph(attn, _leaf(FUSED, ("topk_ids", ids)), layers=2)

    return at


def _routed_bench(fn, config, todo, llc):
    """A device timing the MoE leaf a microsecond per real row, every other
    leaf at a microsecond, and recording the rows the MoE's ids hold."""
    got = []
    for shape, context, found in todo:
        for sig, (op, _) in found.items():
            point = {"shape": str(shape), "context": context, "op": op["op"]}
            point |= {"leaf": sig, "kernels": ["k"], "cold_ns": 1e3}
            if op["op"] == FUSED:
                rows = op["args"]["topk_ids"]["shape"][0]
                point |= {"routed_rows": rows, "cold_ns": 1e3 * (shape.real or rows)}
            got.append(point)
    return got


def _steps(path, dp, *steps):
    """A record directory of decode `steps` at rung 4, each its rows' contexts
    and, under DP, its group's real MoE rows, on each of `dp` ranks."""
    (path / "steps").mkdir(parents=True)
    (path / "run.json").write_text("{}")
    for rank in range(dp):
        lines = []
        for i, (contexts, real) in enumerate(steps):
            line = {"step": i, "is_dummy_run": False, "rung": 4}
            line["rows"] = [[1, c, True] for c in contexts]
            line |= {"moe_rows": 4 * dp, "moe_tokens": real} if dp > 1 else {}
            lines.append(json.dumps(line) + "\n")
        (path / "steps" / f"dp{rank}.jsonl").write_text("".join(lines))
    return str(path)


@pytest.mark.parametrize(
    "dp, steps, real",
    [
        # one rank: each step's own rows, from one row up
        (1, [([1000, 3000], None), ([2000] * 3, None)], [1, 2, 3]),
        # DP 2: each step's group's tokens, of the 8 rows its MoE gathers
        (2, [([1000, 3000], 5), ([2000] * 3, 6)], [1, 5, 6]),
    ],
)
def test_a_plan_measures_a_rung_s_routed_leaf_at_its_steps_real_rows(
    bench, tmp_path, monkeypatch, dp, steps, real  # noqa: F811
):
    monkeypatch.setattr(ops, "graph", _routed_graph(dp))
    monkeypatch.setattr(ops, "_child", _routed_bench)
    parallel = SimpleNamespace(data_parallel_size=dp)
    config = SimpleNamespace(**vars(CONFIG), parallel_config=parallel)
    store, root, key, run = bench
    population = _steps(tmp_path / "plan", dp, *steps)
    args = SimpleNamespace(plan=[population], store=root, llc_mib=256)
    out = tmp_path / "out"
    out.mkdir()

    ops.plan(store, key, config, run, args, "compass measure ops --plan", out)

    records = [json.loads(x) for x in (out / "plan.jsonl").read_text().splitlines()]

    stated = {(r["op"], r["shape"]) for r in records if " R=" in r["shape"]}
    assert stated == {(FUSED, f"decode B=4 R={n}") for n in real}
    params = fit_ops.fit(records)
    fit_ops.held_out(params)
    [(op, _)] = [
        leaf
        for leaf in ops.leaves(
            _routed_graph(dp)(None, None, decode(4))["region"]
        ).values()
        if leaf[0]["op"] == FUSED
    ]
    listed = fit_ops.PriceList(params)

    def price(*r):
        return listed.price((op, ("decode", 4, 4, None, 3000, 2000.0, *r), None))

    assert price(real[-1]).seconds == pytest.approx(real[-1] * 1e-6)
    assert price(None).seconds == pytest.approx(4 * dp * 1e-6)  # every row
    assert price().seconds == pytest.approx(4 * dp * 1e-6)


class _Asked(CostSource):
    """A price list that notes each bound it is asked at, and declines."""

    name = LISTED

    def __init__(self):
        self.bounds = []

    def price(self, request):
        self.bounds.append(request[1])
        return self.refuse("asked")


@pytest.mark.parametrize(
    "view, dp, real",
    [
        # one rank: its own tokens, two rows at rung 4
        (BatchView((RequestShape(1, 900, True),) * 2, capture_rung=4), 1, 2),
        # DP 8: the group's 14 tokens, its two dummy forwards' rows as one
        (
            BatchView(
                (RequestShape(1, 900, True),) * 2,
                capture_rung=4,
                moe_rows=32,
                moe_tokens=14,
                group_dummies=2,
            ),
            8,
            13,
        ),
        # a DP step stating no real tokens: every row
        (
            BatchView((RequestShape(1, 900, True),), capture_rung=4, moe_rows=32),
            8,
            None,
        ),
        # a prefill: no pad rows
        (BatchView((RequestShape(9, 9, False),), produces_output=True), 1, None),
    ],
)
def test_a_step_is_asked_at_the_real_rows_its_moe_runs(view, dp, real):
    asked = _Asked()
    backend = OpGraphBackend(
        SPEC, lambda *s, **_: g(4, 64, 64), kv="bf16", dp=dp, price_list=asked
    )

    backend.price(view)

    assert [b[-1] for b in asked.bounds] == [real]
