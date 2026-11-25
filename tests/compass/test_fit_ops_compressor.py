# SPDX-License-Identifier: MIT
"""A compressor's decode leaf, whose cost alternates with its rows' contexts
against its compress ratio: `fit ops` reads it at whether the step's rows
complete a compressed entry and whether most that do run past it, a run and
`measure ops --coverage` price it so, and `measure ops --plan` halves each of
its lines on its own."""

import pytest
from test_bench_ops import _graph, _t
from test_bench_ops import _leaf as _ir_leaf
from test_bench_ops_plan import CONFIG, _records, bench  # noqa: F401  a fixture
from test_fit_ops import _priced
from test_op_work import leaf, t

from atom.compass.backends import BatchView, RequestShape
from atom.compass.backends.op_graph import counted_rows
from atom.compass.bench import ops as bench_ops
from atom.compass.capture.forward import decode
from atom.compass.fit import ops

OP = "aiter.compress"
RATIO = 4


def _compressor(tokens):
    """A compressor's leaf over `tokens` query tokens at `RATIO`."""
    return leaf(OP, (), kv_in=t([tokens, 1024]), ratio=RATIO)


def _completing(rows) -> tuple[bool, bool]:
    """Whether an entry's attention runs, some row's context ending fewer past
    a multiple of `RATIO` than the row's query tokens, and whether most such
    rows end past one."""
    past = [c % RATIO > 0 for q, c, *_ in rows if c % RATIO < q]
    return bool(past), sum(past) > len(past) / 2


def _ns(rows, context):
    """1.6 us where no entry completes, else 6.4 us, or 6.9 us where most rows
    completing one run past it, each rising with context."""
    runs, past = _completing(rows)
    if not runs:
        return 1600 + 0.002 * context
    return (6900 if past else 6400) + 0.01 * context


def _point(context, b=2, q=2):
    """`measure ops`' point at `context` on `b` rows alike, each verifying `q`
    tokens, as MTP1 does: a row holds its context and the draft past it."""
    rows = [[q, context + q - 1, True]] * b
    point = {"shape": f"decode B={b}", "context": context, "op": OP, "rows": rows}
    point |= {"leaf": bench_ops.signature(_compressor(q * b)), "kernels": ["k"]}
    return point | {"cold_ns": _ns(rows, context)}


MEASURED = [_point(c) for c in range(1000, 1100, 3)]


@pytest.mark.parametrize("context", [1031, 1032, 1050, 1053])
def test_a_compressor_s_alternating_points_fit_a_line_each_way(context):
    # 1031 completes an entry on its last token, 1032 on its first and runs
    # past it, 1050 and 1053 complete none; none was measured
    params = ops.fit(MEASURED)
    ops.held_out(params)
    (f,) = params["families"].values()
    rows = [(2, context + 1)]
    bound = ("decode", 2, 2, None, context, context, 4, None, counted_rows(rows))

    got = ops.PriceList(params).price((_compressor(4), bound, 0))

    assert (f["dims"], f["declined"]) == (["B", "completes", "past", "context"], [])
    assert got.seconds == pytest.approx(_ns(rows, context) * 1e-9, rel=0.05)


def test_a_step_completing_an_entry_on_a_row_short_of_its_longest_prices_so():
    # The longest row, at 1007 unmeasured, ends 3 past a multiple of 4 and
    # completes none; the other ends 1 past one, so the step runs the entry's
    # attention and the next entry's first token.
    rows = (2, 1007), (2, 801)
    step = BatchView(tuple(RequestShape(q, c, True) for q, c in rows))

    got = _priced(
        MEASURED, step, lambda draft, _: {"seq": []} if draft else _compressor(4), 1
    )

    assert got == {f"op.{OP}.price_list": pytest.approx(_ns(rows, 1007) * 1e-9)}


def test_a_step_whose_rows_repeat_counts_each_of_them():
    # Two rows at 1033 run past the entry they complete and one at 1032 ends
    # on it: most of the three run past.
    rows = (2, 1033), (2, 1033), (2, 1032)
    step = BatchView(tuple(RequestShape(q, c, True) for q, c in rows))
    measured = [_point(c, b=3) for c in range(1000, 1100, 3)]

    got = _priced(
        measured, step, lambda draft, _: {"seq": []} if draft else _compressor(6), 1
    )

    assert got == {f"op.{OP}.price_list": pytest.approx(_ns(rows, 1033) * 1e-9)}


def test_a_compressor_s_point_stating_no_rows_is_left_out_by_why():
    # a point measured before points stated their rows
    rowless = {k: v for k, v in _point(1006).items() if k != "rows"}

    params = ops.fit([*MEASURED, rowless])

    (f,) = params["families"].values()
    assert [p["why"] for p in params["left_out"]] == [
        "a compressor's point states no rows to read it at"
    ]
    assert len(f["points"]) == len(MEASURED)


def _family(times, q):
    """A compressor's family on one row of `q` query tokens, measured at
    `times`, {context: ns}."""
    records = [_point(c, b=1, q=q) | {"cold_ns": ns} for c, ns in times.items()]
    return bench_ops._families(records, bench_ops._dependent(records))


def test_a_plan_halves_each_line_of_a_compressor_on_it_and_reaches_its_ends():
    # No drafter: an entry completes at a multiple of 4. The completing line
    # bends at 1204; the other is straight and reaches neither 1000 nor 1400.
    times = {1000: 6000, 1204: 7000, 1400: 6200, 1101: 1600, 1201: 1620, 1301: 1640}

    want = bench_ops._densify(_family(times, 1), set())

    assert {at: len(fams) for at, fams in want.items()} == {
        (decode(1), 1100): 1,  # 1102 halfway completes none; 1100 does
        (decode(1), 1300): 1,  # so 1302 and 1300
        (decode(1), 1001): 1,  # the other line's nearest to 1000
        (decode(1), 1399): 1,  # and to 1400
    }


def test_a_plan_reads_a_drafted_row_s_line_past_the_draft_it_verifies():
    # MTP1: the row holds its context and the draft, so an entry completes on
    # the row's last token at a context 3 past a multiple of 4 and runs past
    # it at 0 past one. The line running past bends at 1204; the other
    # completing line, at 1003 and 1399, reaches no context nearer its ends.
    times = {1000: 6900, 1204: 7900, 1400: 6950, 1003: 6400, 1399: 6450}
    times |= {1101: 1600, 1201: 1620, 1301: 1640}

    want = bench_ops._densify(_family(times, 2), set(), drafts=1)

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
