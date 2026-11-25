# SPDX-License-Identifier: MIT
"""`compass measure ops --plan` over prefill steps whose shapes state the rows
their DP group's MoE gathers, G, which `fit ops` keeps apart from a shape that
states none: a filled point lands in the family its steps are priced from, on
its own points' graph, the plan reads no graph of a shape stating none, and
`fit ops` reads a leaf that does not read G at any G.
The bench and graphs are `test_bench_ops_plan`'s stand-ins.
"""

import json

import pytest
from test_artifact_store import clean_checkout  # noqa: F401  autouse
from test_bench_ops_plan import _moving_bench, _plan, bench  # noqa: F401  a fixture

from atom.compass.backends.provenance import Refusal
from atom.compass.bench import ops
from atom.compass.capture.forward import prefill
from atom.compass.fit import ops as fit_ops

#: The rows a prefill step's DP peers add to its own in its MoE's gather.
PEERS = 7


def _attention(tokens):
    """A prefill attention leaf over `tokens` query rows."""
    q = {"shape": [tokens, 8], "stride": [8, 1], "dtype": "torch.bfloat16"}
    args = {"q": q | {"role": "activation"}}
    return {"op": "aiter.attn", "args": args, "out": [[tokens, 8]]}


def _point(shape, context):
    """The attention's point at `shape` and `context` cached tokens, its time
    a microsecond per token of its row."""
    leaf = ops.signature(_attention(shape.tokens))
    point = {"shape": str(shape), "context": context, "op": "aiter.attn"}
    ns = 1e3 * (shape.tokens + (context or 0))
    return point | {"leaf": leaf, "kernels": ["p"], "cold_ns": ns}


def test_a_fill_beside_points_that_state_g_prices_a_step_that_states_g_between_them():
    at = {t: prefill(t, 1, t + PEERS) for t in (40, 60, 100)}
    # 60 holds its fresh rows alone; 40 and 100 hold 2000 cached tokens too
    records = [_point(at[t], c) for t in (40, 100) for c in (None, 2000)]
    records.append(_point(at[60], None))
    families = ops._families(records, ops._dependent(records))

    want = ops._filled(families, set(), 1 << 20)
    records += [_point(shape, context) for shape, context in want]
    params = fit_ops.fit(records)
    fit_ops.held_out(params)
    # between 40 and 60, at 1000 cached tokens and a G no point states
    step = ("prefill", 50, 1, 50 + PEERS, 1000, 1000)

    got = fit_ops.PriceList(params).price((_attention(50), step, None))

    assert not isinstance(got, Refusal), got.reason
    assert got.seconds == pytest.approx(1050e-6)


def test_a_fill_stating_g_reads_its_own_points_graph_not_a_drafter_s_beside_it():
    at = {t: prefill(t, 1, t + PEERS) for t in (40, 60, 100)}
    # a DP drafter's later pass at T = 60, gathering its batch of 4 on two
    # ranks: its graph there holds the drafter's leaf alone
    x = {"shape": [4, 8], "stride": [8, 1], "dtype": "torch.bfloat16"}
    step = {"op": "aiter.step", "args": {"x": x}, "out": [[4, 8]]}
    later = {"shape": "prefill T=60 B=1 G=8", "context": None, "op": "aiter.step"}
    records = [later | {"leaf": ops.signature(step), "kernels": ["s"], "cold_ns": 4e3}]
    records += [_point(at[t], c) for t in (40, 100) for c in (None, 2000)]
    records.append(_point(at[60], None) | {"micro_batch": True})
    families = ops._families(records, ops._dependent(records))

    want = ops._filled(families, set(), 1 << 20)

    # the micro-batch's own graph, which its attention point was read off
    assert set(want) == {(at[60]._replace(micro_batch=True), 2000)}


def _gathering(path, *steps):
    """A record directory of `steps`, each `(rows, rung)`, a prefill's group
    gathering `PEERS` rows past its own."""
    (path / "steps").mkdir(parents=True)
    (path / "run.json").write_text("{}")
    lines = []
    for i, (rows, rung) in enumerate(steps):
        line = {"step": i, "is_dummy_run": False, "rows": rows, "rung": rung}
        if rung is None:
            line["moe_rows"] = sum(row[0] for row in rows) + PEERS
        lines.append(json.dumps(line) + "\n")
    (path / "steps" / "dp0.jsonl").write_text("".join(lines))
    return str(path)


def test_a_plan_over_steps_that_state_g_asks_no_shape_that_states_none(
    bench, tmp_path, monkeypatch  # noqa: F811
):
    monkeypatch.setattr(ops, "_child", _moving_bench)
    read, stand_in = set(), ops.graph

    def graph(store, config, shape, arch=None):
        read.add(str(shape))
        return stand_in(store, config, shape, arch)

    monkeypatch.setattr(ops, "graph", graph)
    plan = _gathering(
        tmp_path / "plan",
        ([[1, 1000, True], [1, 3000, True]], 2),
        ([[40, 40, False]], None),
        ([[100, 2100, False]], None),
        ([[150, 3150, False]], None),
        ([[30, 1030, False], [30, 3030, False]], None),  # B = 2 at T = 60
        ([[75, 75, False], [75, 75, False]], None),  # and at T = 150
    )

    _plan(bench, tmp_path, plan)

    # B = 1 holds no step at T = 60, nor B = 2 at T = 100, and GEMM switches
    # kernels at T = 100: no shape stating G but a step's has a graph
    assert {s for s in read if s.startswith("prefill")} == {
        "prefill T=40 B=1 G=47",
        "prefill T=100 B=1 G=107",
        "prefill T=150 B=1 G=157",
        "prefill T=60 B=2 G=67",
        "prefill T=150 B=2 G=157",
    }
