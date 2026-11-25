# SPDX-License-Identifier: MIT
"""A compressor's decode leaf, whose cost alternates with its rows' contexts
against its compress ratio: `fit ops` reads it at whether the step's rows
complete a compressed entry, a run and `measure ops --coverage` price it so,
and `measure ops --plan` halves each of its two lines on its own."""

import pytest
from test_bench_ops import _graph, _t
from test_bench_ops import _leaf as _ir_leaf
from test_bench_ops_plan import CONFIG, _records, bench  # noqa: F401  a fixture
from test_fit_ops import _priced
from test_op_work import leaf, t

from atom.compass.backends import BatchView, RequestShape
from atom.compass.bench import ops as bench_ops
from atom.compass.capture.forward import decode
from atom.compass.fit import ops

OP = "aiter.compress"
RATIO = 4


def _compressor(tokens):
    """A compressor's leaf over `tokens` query tokens at `RATIO`."""
    return leaf(OP, (), kv_in=t([tokens, 1024]), ratio=RATIO)


def _completing(rows) -> bool:
    """Whether an entry's attention runs: some row's context ends fewer past a
    multiple of `RATIO` than the row's query tokens."""
    return any(c % RATIO < q for q, c, *_ in rows)


def _ns(rows, context):
    """6.8 us where an entry completes, else 1.6 us, each rising with context."""
    return 6800 + 0.01 * context if _completing(rows) else 1600 + 0.002 * context


def _point(context, b=2, q=2):
    """`measure ops`' point at `context` on `b` rows alike, each verifying `q`
    tokens, as MTP1 does: a row holds its context and the draft past it."""
    rows = [[q, context + q - 1, True]] * b
    point = {"shape": f"decode B={b}", "context": context, "op": OP, "rows": rows}
    point |= {"leaf": bench_ops.signature(_compressor(q * b)), "kernels": ["k"]}
    return point | {"cold_ns": _ns(rows, context)}


MEASURED = [_point(c) for c in range(1000, 1100, 3)]


@pytest.mark.parametrize("context", [1031, 1032, 1050, 1053])
def test_a_compressor_s_alternating_points_fit_one_line_each_way(context):
    # 1031 and 1032 complete an entry, 1050 and 1053 do not; none was measured
    params = ops.fit(MEASURED)
    ops.held_out(params)
    (f,) = params["families"].values()
    rows = [(2, context + 1)]
    bound = ("decode", 2, 2, None, context, context, 4, None, frozenset(rows))

    got = ops.PriceList(params).price((_compressor(4), bound, 0))

    assert (f["dims"], f["declined"]) == (["B", "completes", "context"], [])
    assert got.seconds == pytest.approx(_ns(rows, context) * 1e-9, rel=0.05)


def test_a_step_completing_an_entry_on_a_row_short_of_its_longest_prices_so():
    # The longest row, at 1007 unmeasured, ends 3 past a multiple of 4 and
    # completes none; the other ends 1 past one, so the step runs the entry's
    # attention.
    rows = (2, 1007), (2, 801)
    step = BatchView(tuple(RequestShape(q, c, True) for q, c in rows))

    got = _priced(
        MEASURED, step, lambda draft, _: {"seq": []} if draft else _compressor(4), 1
    )

    assert got == {f"op.{OP}.price_list": pytest.approx(_ns(rows, 1007) * 1e-9)}


def _lines(bend, q):
    """A compressor's family on one row of `q` query tokens: its completing
    line bends at `bend`; the other is straight and reaches neither 1000 nor
    1400."""
    times = {1000: 6000, bend: 7000, 1400: 6200, 1101: 1600, 1201: 1620, 1301: 1640}
    records = [_point(c, b=1, q=q) | {"cold_ns": ns} for c, ns in times.items()]
    return bench_ops._families(records, bench_ops._dependent(records))


def test_a_plan_halves_each_line_of_a_compressor_on_it_and_reaches_its_ends():
    # No drafter: an entry completes at a multiple of 4.
    want = bench_ops._densify(_lines(1204, 1), set())

    assert {at: len(fams) for at, fams in want.items()} == {
        (decode(1), 1100): 1,  # 1102 halfway completes none; 1100 does
        (decode(1), 1300): 1,  # so 1302 and 1300
        (decode(1), 1001): 1,  # the other line's nearest to 1000
        (decode(1), 1399): 1,  # and to 1400
    }


def test_a_plan_reads_a_drafted_row_s_line_past_the_draft_it_verifies():
    # MTP1: the row holds its context and the draft, so an entry completes at
    # contexts 0 or 3 past a multiple of 4.
    want = bench_ops._densify(_lines(1203, 2), set(), drafts=1)

    assert {context for _, context in want} == {1100, 1300, 1001, 1398}


def test_coverage_prices_a_compressor_s_step_on_the_line_its_rows_hold(
    bench, tmp_path, monkeypatch  # noqa: F811
):
    # No drafter: a row completes an entry at a multiple of 4.
    store, root, _, _ = bench
    compress = _ir_leaf(OP, ("kv_in", _t([1, 8], "activation")), ("ratio", RATIO))
    graph = _graph(compress, layers=1)
    monkeypatch.setattr(bench_ops, "graph", lambda *_, **__: graph)
    [sig] = bench_ops.leaves(graph["region"])
    records = [
        _point(c, b=1, q=1) | {"leaf": sig, "cold_ns": _ns([(1, c)], c)}
        for c in range(1000, 1100, 3)
    ]
    rows = [[1, 1032, True]], [[1, 1033, True]], [[1, 1052, True]]
    graded = _records(tmp_path / "graded", *((r, 1) for r in rows))

    report = bench_ops.coverage(store, root, CONFIG, [graded], records, "gfx942")

    assert report["by phase"] == {"decode": {"steps": 3, "covered": 3}}
