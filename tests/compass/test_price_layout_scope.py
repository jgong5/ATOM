"""Operand layout must travel with the price that was selected, not the file.

Two defects, one cause: layout is recorded once per *signature* while prices are
selected per *scope*. `signature_of` deliberately excludes layout, so the same
key can hold a TP2 measurement of a dense rebuild and a TP4 measurement of a
strided one. Validating a request against a signature-wide layout then checks it
against whichever file happened to load first, which is not necessarily the
measurement being spent.

Everything here goes through the real `PriceLibrary` / `ParametricPriceLibrary`
`add` and `lookup` with price and graph files on disk. Nothing mocks the base
refusal: a test that stubs the thing under test would pass either way.
"""

from __future__ import annotations

import json

import pytest

from atom.compass.core.cost.families import ParametricPriceLibrary, coverage_split
from atom.compass.core.cost.library import (
    MEASURED_KEY,
    REGISTERED,
    PriceLibrary,
)
from atom.compass.runtime.microbench import signature_of

# -- operators -------------------------------------------------------------
#
# `layouts` as `meta._layouts_of` records them: (position, (stride, offset,
# elements, owner)). An op with no `layouts` is a dense rebuild, and that is the
# common case rather than a missing field.

def all_reduce(*, layouts=()) -> dict:
    """One collective. Same signature whether or not a layout is recorded."""
    op = {
        "name": "c10d::all_reduce_",
        "input_shapes": [[32, 5120]],
        "dtypes": ["bfloat16"],
        "group": "tp",
    }
    if layouts:
        op["layouts"] = layouts
    return op


STRIDED = [[0, [5120, 4096, 163840, "buf0"]]]


def gemm(rows: int, *, layouts=()) -> dict:
    op = {
        "name": "aiter::gemm_a16w16",
        "input_shapes": [[rows, 17408], [5120, 17408]],
        "dtypes": ["bfloat16", "bfloat16"],
        "scalars": [["#2", "None"]],
    }
    if layouts:
        op["layouts"] = layouts
    return op


def strided_gemm(rows: int, *, offset: int = 4096) -> dict:
    """A gemm reading a strided view whose extent scales with the rows."""
    return gemm(rows, layouts=[[0, [17408, offset, rows * 17408, "buf0"]]])


def _write(tmp_path, tag, op, seconds, *, topology=None, registration=None,
           rows=None):
    """One price file and the graph it was priced from, as `add` reads them."""
    provenance = {}
    if topology is not None:
        provenance["topology"] = topology
    if registration is not None:
        provenance[MEASURED_KEY] = registration
    graph = {"ops": [op]}
    if rows is not None:
        graph["provenance"] = {"execution": {"body_rows_traced": rows}}
    prices = {
        "provenance": provenance,
        "prices": {signature_of(op): {
            "seconds": seconds, "kernels": {"k": seconds},
            "occurrences": 1, "name": op["name"]}},
    }
    gpath = tmp_path / f"g_{tag}.json"
    ppath = tmp_path / f"p_{tag}.json"
    gpath.write_text(json.dumps(graph))
    ppath.write_text(json.dumps(prices))
    return str(ppath), str(gpath)


# == (1) scoped selection vs signature-wide layout ==========================

TP2 = {"tp": 2}
TP4 = {"tp": 4}


def _two_scope_library(tmp_path, order):
    """TP2 priced dense, TP4 priced strided, loaded in the given order."""
    files = {
        "tp2": _write(tmp_path, "tp2", all_reduce(), 1e-4,
                      topology=TP2, registration=REGISTERED),
        "tp4": _write(tmp_path, "tp4", all_reduce(layouts=STRIDED), 4e-4,
                      topology=TP4, registration=REGISTERED),
    }
    library = PriceLibrary()
    for key in order:
        library.add(*files[key])
    return library


@pytest.mark.parametrize("order", [("tp2", "tp4"), ("tp4", "tp2")])
def test_the_scoped_strided_measurement_is_spent_on_its_own_request(
        tmp_path, order):
    """TP4 was measured strided; a strided TP4 request is exactly that price.

    Refusing it is refusing a measurement that exists, under the layout it was
    taken at, because a different scope's file was read first.
    """
    library = _two_scope_library(tmp_path, order)
    record, detail = library.lookup(
        all_reduce(layouts=STRIDED), topology=TP4, registration=REGISTERED)
    assert record is not None, detail
    assert record["seconds"] == pytest.approx(4e-4)


@pytest.mark.parametrize("order", [("tp2", "tp4"), ("tp4", "tp2")])
def test_a_dense_request_is_not_paid_from_a_strided_measurement(
        tmp_path, order):
    """TP4 holds only a strided price, so a dense TP4 rebuild has no price.

    This is the direction that corrupts a number rather than losing one: the
    selected record is TP4's strided measurement, and answering with it prices
    a dense rebuild at a strided rebuild's cost.
    """
    library = _two_scope_library(tmp_path, order)
    record, detail = library.lookup(
        all_reduce(), topology=TP4, registration=REGISTERED)
    assert record is None, (
        f"answered a dense TP4 request with {record} -- the only TP4 "
        "measurement was taken on a strided operand")
    assert "layout" in detail


@pytest.mark.parametrize("order", [("tp2", "tp4"), ("tp4", "tp2")])
def test_the_dense_scope_still_answers_its_own_dense_request(tmp_path, order):
    """The fix must not refuse everything: TP2 was measured dense."""
    library = _two_scope_library(tmp_path, order)
    record, detail = library.lookup(
        all_reduce(), topology=TP2, registration=REGISTERED)
    assert record is not None, detail
    assert record["seconds"] == pytest.approx(1e-4)


@pytest.mark.parametrize("order", [("tp2", "tp4"), ("tp4", "tp2")])
def test_a_strided_request_at_the_dense_scope_is_refused(tmp_path, order):
    library = _two_scope_library(tmp_path, order)
    record, detail = library.lookup(
        all_reduce(layouts=STRIDED), topology=TP2, registration=REGISTERED)
    assert record is None, detail
    assert "layout" in detail


def test_an_unrecorded_layout_does_not_start_refusing(tmp_path):
    """A price file loaded without its graph says nothing about layout.

    Absent is not dense. Such a record keeps answering, as it did before, and
    the absence is what makes that safe to state.
    """
    ppath, _ = _write(tmp_path, "nolayout", all_reduce(), 1e-4,
                      topology=TP2, registration=REGISTERED)
    library = PriceLibrary()
    library.add(ppath, None)
    for op in (all_reduce(), all_reduce(layouts=STRIDED)):
        record, detail = library.lookup(op, topology=TP2,
                                        registration=REGISTERED)
        assert record is not None, detail


# == (2) interpolation across layouts ======================================

def _rows_library(tmp_path, ops: dict) -> ParametricPriceLibrary:
    library = ParametricPriceLibrary(max_gap_ratio=2.0)
    for rows, (op, seconds) in sorted(ops.items()):
        library.add(*_write(tmp_path, f"r{rows}", op, seconds, rows=rows))
    return library


def test_a_strided_request_does_not_interpolate_dense_measurements(tmp_path):
    """The defect that claims coverage it does not have.

    Only dense 32 and 64 were measured. An exact-width strided request is
    already refused by the layout check on the base path -- so answering the
    48-row strided one by interpolating those same dense measurements is the
    library contradicting itself, and it reports the result as covered.
    """
    library = _rows_library(tmp_path, {32: (gemm(32), 1e-4),
                                       64: (gemm(64), 2e-4)})

    exact, why_exact = library.lookup(strided_gemm(32))
    assert exact is None, "the exact-width strided request should refuse"

    record, _detail = library.lookup(strided_gemm(48))
    assert record is None, (
        f"interpolated a strided 48-row request from dense measurements "
        f"({record}) while refusing the exact 32-row one: {why_exact}")


def test_such_a_refusal_is_not_counted_as_coverage(tmp_path):
    """A refusal must reach the coverage split as a refusal.

    The failure mode is not only a wrong price: it is a wrong price that
    reports itself complete.
    """
    library = _rows_library(tmp_path, {32: (gemm(32), 1e-4),
                                       64: (gemm(64), 2e-4)})
    split = coverage_split(library, {"ops": [strided_gemm(48)]})
    assert split["refused"] == 1, split
    assert split["interpolated"] == 0, split
    assert not split["complete_measured"], split
    assert not split["complete_accounted"], split


def test_a_strided_ladder_still_interpolates_its_own_width(tmp_path):
    """Row-adjusted compatibility, not blanket layout equality.

    The 32- and 64-row measurements are strided, and the extent of the view
    scales with the rows exactly as the shapes do. A 48-row request on the same
    view is the same operator at a third width, and refusing it would throw
    away a real measurement.
    """
    library = _rows_library(tmp_path, {32: (strided_gemm(32), 1e-4),
                                       64: (strided_gemm(64), 2e-4)})
    record, detail = library.lookup(strided_gemm(48))
    assert record is not None, detail
    assert record["interpolated"] is True


def test_a_layout_that_does_not_scale_with_the_rows_is_a_different_operator(
        tmp_path):
    """Same family, same widths, a view that moves under it.

    The offset is a fixed byte position in both measurements and the request
    changes it. That is not this operator at a third width.
    """
    library = _rows_library(tmp_path, {32: (strided_gemm(32), 1e-4),
                                       64: (strided_gemm(64), 2e-4)})
    record, _detail = library.lookup(strided_gemm(48, offset=8192))
    assert record is None, f"answered a moved view with {record}"


def test_the_dense_ladder_is_unaffected(tmp_path):
    """Regression guard: the existing dense interpolation must keep working."""
    library = _rows_library(tmp_path, {32: (gemm(32), 1e-4),
                                       64: (gemm(64), 2e-4)})
    record, detail = library.lookup(gemm(48))
    assert record is not None, detail
    assert record["interpolated"] is True
