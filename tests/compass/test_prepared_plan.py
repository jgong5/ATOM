"""A prepared plan preserves scope, coverage, dynamic boundaries and refusals."""
import json

import pytest

from atom.compass.core.cost.families.adapter import ParametricPriceLibrary
from atom.compass.core.cost.prepared import prepare_static_operator
from atom.compass.core.cost.prepared_plan import PreparedGraph, prepare_plan
from atom.compass.runtime.microbench import signature_of

from .test_attention_family import _unified
from .test_family_prices import _head_file, head_gemm
from .test_output_visibility import SCOPE
from .test_prepared_identities import library_for, operation


def prepared_graph(ops, **fields):
    prepared = [prepare_static_operator(op) or op for op in ops]
    return PreparedGraph(dict(fields, ops=prepared), prepare_plan(prepared))


def add_book(library, tmp_path, label, ops, seconds):
    graph, price = tmp_path / f"{label}.graph.json", tmp_path / f"{label}.price.json"
    graph.write_text(json.dumps({"ops": ops}))
    price.write_text(json.dumps({"prices": {
        signature_of(op): {"seconds": value, "kernels": {"k": value}, "name": op["name"]}
        for op, value in zip(ops, seconds)}}))
    library.add(str(price), str(graph))


def test_segment_plan_keeps_the_last_dynamic_sync_prefix(tmp_path):
    library = ParametricPriceLibrary()
    library.request_attention_scope = SCOPE
    attention = _unified([8], [16], is_prefill=True, has_cached=True)
    def static(name):
        return {"name": name, "input_shapes": [], "dtypes": []}
    ops = [static("prefix"), dict(attention, scalars=[["layer", 3]]),
           static("between"), dict(attention, scalars=[["layer", 7]]), static("suffix")]
    add_book(library, tmp_path, "ordered", ops, [2., 3., 5., 7., 11.])
    graph = prepared_graph(ops)
    expected_timing = {}
    expected = library.body(dict(graph), timing=expected_timing)
    for _ in range(2):
        timing = {}
        assert library.body(graph, timing=timing) == expected
        assert timing == expected_timing
        assert timing["seconds"] == 10.0 and timing["priced_operator_index"] == 3


def test_fitted_head_segment_tracks_interpolation_configuration(tmp_path):
    library = ParametricPriceLibrary(max_gap_ratio=2.0)
    _head_file(library, tmp_path, 2, 1e-4)
    _head_file(library, tmp_path, 4, 2e-4)
    graph = prepared_graph([head_gemm(3)])
    for _ in range(2):
        seconds, coverage, _ = library.body(graph)
        assert seconds == pytest.approx(1.5e-4)
        assert coverage.interpolated == 1 and coverage.measured == 0
    library.max_gap_ratio = 1.1
    assert not library.body(graph)[1].complete


def test_new_source_closes_a_previously_refused_static_plan(tmp_path):
    library = ParametricPriceLibrary()
    one, two = operation(), dict(operation(), name="other::kernel")
    add_book(library, tmp_path, "first", [one], [.001])
    graph = prepared_graph([one, two])
    assert not library.body(graph)[1].complete
    add_book(library, tmp_path, "second", [two], [.002])
    seconds, coverage, _ = library.body(graph)
    assert seconds == .003 and coverage.complete and coverage.measured == 2


def test_static_address_counts_survive_warm_reuse_and_failed_dynamic_pass(tmp_path):
    measured = operation()
    measured["scalars"] = [["slot_mapping", [1, 2]]]
    library = library_for(tmp_path, measured)
    current = dict(measured, scalars=[["slot_mapping", [11, 12]]])
    graph = prepared_graph([current, current, current])
    for expected in (3, 6):
        assert library.body(graph)[1].measured == 3
        assert sum(library.address_shifted.values()) == expected
    missing = {"name": "missing", "input_shapes": [], "dtypes": [],
               "context": [["current", [1]]]}
    incomplete = prepared_graph([current, missing])
    for expected in (7, 8):
        result = library.body(incomplete)
        assert result[1].measured == 1 and sum(result[1].refused.values()) == 1
        assert sum(library.address_shifted.values()) == expected


def test_replacing_prepared_operator_order_uses_the_live_graph(tmp_path):
    op = operation()
    library = library_for(tmp_path, op)
    graph = prepared_graph([op, op])
    assert library.body(graph)[1].measured == 2
    graph["ops"] = [op]
    assert library.body(graph)[1].measured == 1
