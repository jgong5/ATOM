"""Repeated fitted calls may share a body result; measured layers stay distinct."""
import copy

import pytest

from atom.compass.core.cost.library import PriceLibrary
from .test_attention_family import (
    SCOPE, _cold_designs, _gdn_designs, _gdn_full, _layer_copy, _library, _unified,
)


def without_model_memo(library, graph, **kwargs):
    original = library._body_lookup
    library._body_lookup = PriceLibrary._body_lookup.__get__(library)
    try:
        return library.body(graph, **kwargs)
    finally:
        library._body_lookup = original


def test_repeated_fitted_calls_keep_current_context_and_order(tmp_path):
    library = _library(tmp_path, _cold_designs())
    library.request_attention_scope = dict(SCOPE)
    short = _unified([641], [641], is_prefill=True, has_cached=False)
    long = _unified([777], [777], is_prefill=True, has_cached=False)
    first = dict(short, scalars=[["layer_name", "model.layers.3.self_attn"]])
    second = dict(short, scalars=[["layer_name", "model.layers.7.self_attn"]])
    graph = {"ops": [first, second, long, first]}
    assert library.body(graph) == without_model_memo(library, graph)
    changed = copy.deepcopy(graph)
    changed["ops"][1] = long
    assert library.body(changed) == without_model_memo(library, changed)
    assert library.body(changed)[0] != library.body(graph)[0]


def test_scope_and_geometry_changes_do_not_reuse_a_modelled_answer(tmp_path):
    library = _library(tmp_path, _cold_designs())
    library.request_attention_scope = dict(SCOPE)
    op = _unified([641], [641], is_prefill=True, has_cached=False)
    changed = dict(op, dtypes=["float32"])
    graph = {"ops": [op, changed, op]}
    assert library.body(graph) == without_model_memo(library, graph)
    assert library.body(graph)[1].interpolated == 2
    library.request_attention_scope = dict(SCOPE, kv_cache_dtype="float32")
    assert library.body(graph) == without_model_memo(library, graph)
    assert library.body(graph)[1].interpolated == 0


def test_exact_layer_prices_remain_individual(tmp_path):
    designs = _gdn_designs()
    library = _library(tmp_path, designs, layers=2)
    library.request_attention_scope = dict(SCOPE)
    first, second = (_layer_copy(designs[0][0], layer) for layer in (0, 1))
    one, _ = library.lookup(first)
    two, _ = library.lookup(second)
    assert one is not None and two is not None and one is not two
    one["seconds"], two["seconds"] = .001, .003
    graph = {"ops": [first, second, first, second]}
    seconds, coverage, _ = library.body(graph)
    assert seconds == pytest.approx(.008)
    assert coverage.measured == 4 and coverage.interpolated == 0


def test_modelled_refusals_keep_their_full_graph_evidence(tmp_path):
    library = _library(tmp_path, _cold_designs())
    library.request_attention_scope = dict(SCOPE)
    allowed = _unified([641], [641], is_prefill=True, has_cached=False)
    outside = _unified([999999], [999999], is_prefill=True, has_cached=False)
    graph = {"ops": [allowed, outside, allowed, outside]}
    answer = library.body(graph)
    assert answer == without_model_memo(library, graph)
    assert sum(answer[1].refused.values()) == 2


def test_exact_gdn_reuse_keeps_current_inputs_and_record_layout(tmp_path):
    source = _gdn_full(2, 2)
    library = _library(tmp_path, [(source, .001)], layers=1)
    library.request_attention_scope = dict(SCOPE)
    op = _layer_copy(source, 0)
    graph = {"ops": [op]}
    assert library.body(graph)[1].measured == 1
    changed = copy.deepcopy(graph)
    changed["ops"][0]["dtypes"] = ["float32"]
    assert library.body(changed) == without_model_memo(library, changed)
    assert library.body(changed)[1].measured == 0
    record, _ = library.lookup(op)
    record["layout"] = "different"
    assert library.body(graph) == without_model_memo(library, graph)
    assert library.body(graph)[1].measured == 0


def test_exact_gdn_reuse_repays_every_address_shift(tmp_path):
    source = _gdn_full(2, 2)
    library = _library(tmp_path, [(source, .001)], layers=1)
    library.request_attention_scope = dict(SCOPE)
    op = _layer_copy(source, 0)
    op = copy.deepcopy(op)
    for pair in op["context"]:
        if pair[0] == "non_spec_state_indices_tensor":
            pair[1] = [11, 12]
    graph = {"ops": [op, op]}
    for expected in (2, 4, 6):
        assert library.body(graph)[1].measured == 2
        assert sum(library.address_shifted.values()) == expected


def test_new_selected_treatment_cannot_spend_cached_exact_gdn_price(tmp_path):
    source = _gdn_full(2, 2)
    library = _library(tmp_path, [(source, .001)], layers=1)
    library.request_attention_scope = dict(SCOPE)
    graph = {"ops": [_layer_copy(source, 0)]}
    assert library.body(graph)[1].measured == 1
    library.request_attention_treatments = {"gdn.decode": {"cache": "cold"}}
    assert library.body(graph) == without_model_memo(library, graph)
    assert library.body(graph)[1].measured == 0
