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
    UNREGISTERED,
    PriceLibrary,
)
from atom.compass.runtime.microbench import cost_key_of, signature_of

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

TP1 = {"tp": 1}
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


# == (3) one signature, two scopes, two layouts ============================
#
# `_ops` is keyed by signature alone and keeps the first operator seen under
# it. A signature does not carry layout or scope, so one file's dense rebuild
# and another's strided view share a key -- and pairing every scoped price with
# the first-seen operator put both sets of seconds on the first layout's curve.
# Two failures at once: the dense median is contaminated by strided
# measurements, and no strided curve exists for a strided request to sit in.

def _scoped(tmp_path, tag, op, seconds, rows, topology):
    return _write(tmp_path, tag, op, seconds, topology=topology,
                  registration=REGISTERED, rows=rows)


def _mixed_library(tmp_path, order=("dense", "strided")):
    """Dense at tp2 and strided at tp4, same signature, same widths."""
    files = {
        "dense": [
            _scoped(tmp_path, "d32", gemm(32), 1e-4, 32, TP2),
            _scoped(tmp_path, "d64", gemm(64), 2e-4, 64, TP2),
        ],
        "strided": [
            _scoped(tmp_path, "s32", strided_gemm(32), 5e-4, 32, TP4),
            _scoped(tmp_path, "s64", strided_gemm(64), 10e-4, 64, TP4),
        ],
    }
    library = ParametricPriceLibrary(max_gap_ratio=2.0)
    for which in order:
        for price, graph in files[which]:
            library.add(price, graph)
    return library


@pytest.mark.parametrize("order", [("dense", "strided"), ("strided", "dense")])
def test_a_dense_interpolation_is_not_contaminated_by_strided_prices(
        tmp_path, order):
    """48 dense rows sits between 1e-4 and 2e-4, and nowhere near 5e-4.

    The strided measurements are five times the dense ones here precisely so
    that a contaminated median is a number no honest dense interpolation could
    produce.
    """
    library = _mixed_library(tmp_path, order)
    record, detail = library.lookup(gemm(48), topology=TP2,
                                    registration=REGISTERED)
    assert record is not None, detail
    assert 1e-4 <= record["seconds"] <= 2e-4, (
        f"dense 48 came back as {record['seconds']:.3e}, outside the dense "
        "bracket [1e-4, 2e-4]")


@pytest.mark.parametrize("order", [("dense", "strided"), ("strided", "dense")])
def test_the_strided_measurements_keep_their_own_support(tmp_path, order):
    """Strided 32 and 64 were measured, so strided 48 is interpolable.

    Losing this is the quieter half: the measurements exist, were paid for,
    and a request that should have been answered from them is refused because
    they were filed under another layout's curve.
    """
    library = _mixed_library(tmp_path, order)
    record, detail = library.lookup(strided_gemm(48), topology=TP4,
                                    registration=REGISTERED)
    assert record is not None, detail
    assert record["interpolated"] is True
    assert 5e-4 <= record["seconds"] <= 10e-4, record["seconds"]


def test_a_local_operator_is_reusable_across_scopes(tmp_path):
    """TP2 measured dense, TP4 measured strided, and both are just matmuls.

    A GEMM's cost is a property of its shapes and its operand layout, not of
    the deployment around it, so the strided curve answers a strided request
    whichever group width asks. The separation that matters here is layout,
    and it still holds: this is answered from the strided measurements, not
    from the dense ones five times cheaper.
    """
    library = _mixed_library(tmp_path)
    record, detail = library.lookup(strided_gemm(48), topology=TP2,
                                    registration=REGISTERED)
    assert record is not None, detail
    assert 5e-4 <= record["seconds"] <= 10e-4, (
        "answered from the dense curve rather than the strided one")


def test_a_local_operator_needs_no_scope_named_at_all(tmp_path):
    """Most callers never name one, and a matmul does not need them to."""
    library = _mixed_library(tmp_path)
    record, detail = library.lookup(gemm(48))
    assert record is not None, detail
    assert 1e-4 <= record["seconds"] <= 2e-4


# == (4) a local operator's cost does not depend on the group ==============
#
# `PriceLibrary.lookup` scopes exact prices for COLLECTIVES only, and the body
# hands the graph's registration to every lookup. Scoping local families the
# same way made the two paths disagree about one operator: the exact-width GEMM
# was answered from an unregistered price list, while the in-support
# interpolation of that same GEMM was refused for the same list being
# unregistered.

def _local(tmp_path, tag, rows, seconds, topology, registration):
    return _write(tmp_path, tag, gemm(rows), seconds, topology=topology,
                  registration=registration, rows=rows)


def test_a_gemm_interpolates_across_registration_regimes(tmp_path):
    """A GEMM does not care which path the collectives elsewhere took."""
    library = ParametricPriceLibrary(max_gap_ratio=2.0)
    library.add(*_local(tmp_path, "u32", 32, 1e-4, TP1, UNREGISTERED))
    library.add(*_local(tmp_path, "u64", 64, 2e-4, TP1, UNREGISTERED))

    exact, why = library.lookup(gemm(32), topology=TP1,
                                registration=REGISTERED)
    assert exact is not None, why

    between, detail = library.lookup(gemm(48), topology=TP1,
                                     registration=REGISTERED)
    assert between is not None, (
        "the exact width was answered from this list and the interpolation "
        f"was not: {detail}")
    assert between["interpolated"] is True


def test_a_gemm_interpolates_across_group_widths(tmp_path):
    """TP1 and TP2 shards of the same shape are the same matrix multiply."""
    library = ParametricPriceLibrary(max_gap_ratio=2.0)
    library.add(*_local(tmp_path, "t32", 32, 1e-4, TP1, UNREGISTERED))
    library.add(*_local(tmp_path, "t64", 64, 2e-4, TP2, REGISTERED))

    record, detail = library.lookup(gemm(48), topology=TP4,
                                    registration=REGISTERED)
    assert record is not None, detail
    assert record["interpolated"] is True
    assert 1e-4 <= record["seconds"] <= 2e-4


def test_a_collective_is_not_answered_by_the_parametric_path_at_all(tmp_path):
    """Which is why scoping local families on the group was wrong.

    `c10d::all_reduce_` has no rows family contract, so a collective never
    reaches the curve machinery: its scope separation is `PriceLibrary`'s
    exact-price selection by width and path, and that is where it belongs.
    Scoping local families the same way bought nothing and cost real
    interpolations.
    """
    library = ParametricPriceLibrary(max_gap_ratio=2.0)
    for tag, rows, seconds in (("c32", 32, 1e-4), ("c64", 64, 2e-4)):
        op = dict(all_reduce(), input_shapes=[[rows, 5120]])
        library.add(*_write(tmp_path, tag, op, seconds, topology=TP2,
                            registration=REGISTERED, rows=rows))

    wide = dict(all_reduce(), input_shapes=[[48, 5120]])
    record, detail = library.lookup(wide, topology=TP2,
                                    registration=REGISTERED)
    assert record is None, f"interpolated a collective: {record}"
    assert "no declared family contract" in detail, detail


# == (5) one graph, one signature, two calls ==============================

def test_a_duplicate_signature_in_one_graph_takes_the_first_occurrence(
        tmp_path):
    """The collector prices the first; this must label it as the first.

    `microbench` keys its example operator with `example.setdefault` and
    `PriceLibrary._ingest` captures the measured layout with
    `layouts.setdefault`, so a graph holding a dense and a strided call under
    one signature is PRICED as the dense one. Labelling it strided here would
    disagree with the measurement that was actually taken.
    """
    op_dense, op_strided = gemm(32), strided_gemm(32)
    assert signature_of(op_dense) == signature_of(op_strided), (
        "the premise: layout is not in the signature")

    graph = {"ops": [op_dense, op_strided],
             "provenance": {"execution": {"body_rows_traced": 32}}}
    prices = {"prices": {signature_of(op_dense): {
        "seconds": 1e-4, "kernels": {"k": 1e-4}, "occurrences": 2,
        "name": op_dense["name"]}}}
    gpath = tmp_path / "g_dup.json"
    ppath = tmp_path / "p_dup.json"
    gpath.write_text(json.dumps(graph))
    ppath.write_text(json.dumps(prices))

    library = ParametricPriceLibrary(max_gap_ratio=2.0)
    library.add(str(ppath), str(gpath))
    library._build()

    recorded = next(iter(library._observations.values()))[0][0]
    assert not recorded.get("layouts"), (
        "the second, strided call was taken as the representative for a price "
        "the collector measured on the first, dense one")


# == (4) the cost key collapses observations; layout must not follow it =====
#
# `core/cost/identity` files two observations that differ only in allocator
# addresses under one cost key, so one measurement answers for both. That is a
# LOOKUP index. Layout is a per-record fact: the two collapsed observations may
# have been recorded on differently arranged memory, and whichever the graph
# happens to list first is not necessarily the one whose price is kept.
#
# Keying the ingest-time layout table by the cost key reintroduces the defect
# this module exists for, one level down -- so these go through the same real
# `add`/`lookup`, with the two failure modes the review named: a price file
# whose order is the reverse of the graph's, and a first raw observation that
# carries no price at all.

MEASURED_SLOTS = [9102, 9118, 9134, 9150, 9166, 9182, 9198, 9214]
SHIFTED_SLOTS = [1150, 2302, 3454, 4606, 5758, 6910, 8062, 9214]


def attn(slots, *, layouts=()) -> dict:
    """Decode attention. Same cost key at either allocation, and the two
    differ in the one component the normalisation removes."""
    op = {
        "name": "aiter::unified_attention_with_output_base",
        "input_shapes": [[8, 6144], [8, 1024]],
        "dtypes": ["bfloat16", "bfloat16"],
        "context": [["slot_mapping", list(slots)],
                    ["context_lens", [1151] * 8],
                    ["max_seqlen_k", 1151]],
    }
    if layouts:
        op["layouts"] = layouts
    return op


ATTN_STRIDED = [[0, [6144, 4096, 49152, "buf0"]]]


def _collapsed_file(tmp_path, tag, graph_ops, priced_ops, seconds):
    """One file whose graph records `graph_ops` in that order and whose price
    list records `priced_ops` in that order. Both raw signatures share a cost
    key; the price list may name a subset, and in a different order."""
    prices = {
        "provenance": {"topology": TP1, MEASURED_KEY: REGISTERED},
        "prices": {signature_of(op): {
            "seconds": seconds, "kernels": {"k": seconds},
            "occurrences": 1, "name": op["name"]} for op in priced_ops},
    }
    gpath = tmp_path / f"g_{tag}.json"
    ppath = tmp_path / f"p_{tag}.json"
    gpath.write_text(json.dumps({"ops": list(graph_ops)}))
    ppath.write_text(json.dumps(prices))
    library = PriceLibrary()
    library.add(str(ppath), str(gpath))
    return library


def test_the_kept_price_carries_its_own_layout_not_the_graphs_first(tmp_path):
    """Graph lists the dense call first; the price list names the strided one
    first, so the strided record is the one kept for this scope.

    Its layout has to be the strided one it was measured under. Taking the
    graph's first entry instead stamps it dense, and a dense request is then
    answered from a measurement of a strided read.
    """
    dense = attn(MEASURED_SLOTS)
    strided = attn(SHIFTED_SLOTS, layouts=ATTN_STRIDED)
    library = _collapsed_file(tmp_path, "rev", [dense, strided],
                              [strided, dense], 3e-4)

    record, detail = library.lookup(attn(SHIFTED_SLOTS, layouts=ATTN_STRIDED),
                                    topology=TP1, registration=REGISTERED)
    assert record is not None, detail
    assert record["seconds"] == pytest.approx(3e-4)

    refused, why = library.lookup(attn(MEASURED_SLOTS),
                                  topology=TP1, registration=REGISTERED)
    assert refused is None, (
        "a dense request was paid from a price measured on a strided read, "
        f"because the layout came from the graph's first entry: {why}")


def test_an_unpriced_first_observation_does_not_lend_its_layout(tmp_path):
    """The graph's first call has no price at all -- the collector refused it,
    or it simply is not in the list. The record that does exist must still
    carry the layout of the observation it was taken from."""
    dense = attn(MEASURED_SLOTS)
    strided = attn(SHIFTED_SLOTS, layouts=ATTN_STRIDED)
    library = _collapsed_file(tmp_path, "gap", [dense, strided],
                              [strided], 3e-4)

    record, detail = library.lookup(attn(SHIFTED_SLOTS, layouts=ATTN_STRIDED),
                                    topology=TP1, registration=REGISTERED)
    assert record is not None, detail

    refused, why = library.lookup(attn(MEASURED_SLOTS),
                                  topology=TP1, registration=REGISTERED)
    assert refused is None, (
        "the only price in the file was measured strided, and a dense request "
        f"was answered from it: {why}")


def mrope(positions, *, layouts=()) -> dict:
    """Rotary embedding over 8 decode rows. A rows family -- so the parametric
    curve is actually built for it -- that carries `positions`, which is an
    allocator-relative component the cost key normalises away. The two
    allocations below therefore collapse onto one cost key while remaining
    distinct observations."""
    op = {
        "name": "triton::_mrope_qk_kernel",
        "input_shapes": [[8, 6144], [8, 1024]],
        "dtypes": ["bfloat16", "bfloat16"],
        "context": [["positions", list(positions)]],
    }
    if layouts:
        op["layouts"] = layouts
    return op


MROPE_STRIDED = [[0, [6144, 4096, 49152, "buf0"]]]


def test_the_parametric_join_uses_the_records_own_observation(tmp_path):
    """`ParametricPriceLibrary` pairs each price with the operator it priced,
    to read a feature map off it. Joining on the cost key hands it whichever
    collapsed sibling the graph listed first -- the same defect as (3), and
    the curve is then built on the wrong layout.

    The graph lists the dense call first; the only price in the file was
    measured on the strided one. A cost-key join returns the dense operator,
    so the observation carries no layout and the strided curve never exists.
    """
    dense = mrope(MEASURED_SLOTS)
    strided = mrope(SHIFTED_SLOTS, layouts=MROPE_STRIDED)
    assert cost_key_of(dense) == cost_key_of(strided), "the premise"
    assert signature_of(dense) != signature_of(strided), "the premise"

    prices = {
        "provenance": {"topology": TP1, MEASURED_KEY: REGISTERED,
                       "execution": {"body_rows_traced": 8}},
        "prices": {signature_of(strided): {
            "seconds": 3e-4, "kernels": {"k": 3e-4},
            "occurrences": 1, "name": strided["name"]}},
    }
    gpath = tmp_path / "g_join.json"
    ppath = tmp_path / "p_join.json"
    gpath.write_text(json.dumps(
        {"ops": [dense, strided],
         "provenance": {"execution": {"body_rows_traced": 8}}}))
    ppath.write_text(json.dumps(prices))
    library = ParametricPriceLibrary()
    library.add(str(ppath), str(gpath))
    library._build()

    seen = 0
    for observations in library._observations.values():
        for op, *_ in observations:
            seen += 1
            assert op.get("layouts"), (
                "a price measured on the strided call was attributed to the "
                "dense one, because the join went through the cost key")
    assert seen == 1, (
        "the join produced no observation, so the layout assertion above "
        f"never ran: {seen} observations")
