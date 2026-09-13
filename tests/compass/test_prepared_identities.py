"""Prepared templates own immutable identities; ordinary inputs remain live."""
from dataclasses import FrozenInstanceError
import json

import pytest

from atom.compass.core.cost.library import PriceLibrary
from atom.compass.core.cost.prepared import (
    PreparedOperator, materialize_graph, prepare_static_operator,
)
from atom.compass.runtime.microbench import signature_of
from atom.compass.runtime.templates import TemplateGraphs

from .test_templates import shape


def operation():
    return {"name": "custom::kernel", "input_shapes": [[2, 4]],
            "dtypes": ["bf16"], "scalars": [["axes", [-1, 4]], ["scale", -0.0]]}


def library_for(tmp_path, op):
    graph = tmp_path / "graph.json"
    price = tmp_path / "price.json"
    graph.write_text(json.dumps({"ops": [op], "key": {"topology": [["tp", 2]]}}))
    price.write_text(json.dumps({"prices": {signature_of(op): {"seconds": .001,
                               "name": op["name"]}}}))
    library = PriceLibrary()
    library.add(str(price), str(graph), registration="unregistered")
    return library


def test_preparation_owns_a_snapshot_and_mutable_lookup_stays_live(tmp_path):
    op = operation()
    library = library_for(tmp_path, op)
    prepared = prepare_static_operator(op)
    assert isinstance(prepared, PreparedOperator)
    first = library.lookup(prepared)
    op["input_shapes"][0][0] = 3
    op["scalars"][0][1][0] = 9
    assert library.lookup(op)[0] is None
    assert library.lookup(prepared) == first
    exported = prepared.as_dict()
    exported["input_shapes"][0][0] = 99
    assert prepared.as_dict()["input_shapes"] == [[2, 4]]
    with pytest.raises(FrozenInstanceError):
        prepared.signature = "changed"


def test_preparation_keeps_python_scalar_representation():
    op = operation()
    prepared = prepare_static_operator(op)
    assert prepared.signature == signature_of(prepared.as_dict())
    op["scalars"][0][1] = (-1, 4)
    assert prepare_static_operator(op).signature != prepared.signature
    op["scalars"][1][1] = 0.0
    assert "scale=0.0" in prepare_static_operator(op).signature


def test_dynamic_and_custom_values_remain_on_the_live_path():
    class CustomInt(int):
        pass

    op = operation()
    op["scalars"] = [["value", CustomInt(2)]]
    assert prepare_static_operator(op) is None
    op["scalars"] = [["value", bytearray(b"x")]]
    assert prepare_static_operator(op) is None
    op["scalars"] = []
    op["context"] = [["slot_mapping", [1]]]
    assert prepare_static_operator(op) is None
    op["context"] = []
    op["int_values"] = [[0, [1]]]
    assert prepare_static_operator(op) is None


def test_prepared_identity_still_checks_layout_topology_and_registration(tmp_path):
    op = operation()
    op["group"] = "tp"
    library = library_for(tmp_path, op)
    prepared = prepare_static_operator(op)
    assert library.lookup(prepared, {"tp": 2}, "unregistered")[0] is not None
    assert library.lookup(prepared, {"tp": 4}, "unregistered")[0] is None
    assert library.lookup(prepared, {"tp": 2}, "registered")[0] is None
    op["layouts"] = [[0, [[8, 1], 8, 16, 0]]]
    other = prepare_static_operator(op)
    assert library.lookup(other, {"tp": 2}, "unregistered")[0] is None


def test_template_public_copies_and_replacement_cannot_poison_preparation():
    original = {"ops": [operation()]}
    source = TemplateGraphs()
    point = shape([1], [128])
    source.add(point, original)
    original["ops"][0]["input_shapes"][0][0] = 8
    public = source.graph_for(point)
    public["ops"][0]["input_shapes"][0][0] = 9
    prepared = source.prepared_graph_for(point)
    assert prepared["ops"][0].as_dict()["input_shapes"] == [[2, 4]]
    assert materialize_graph(prepared)["ops"][0]["input_shapes"] == [[2, 4]]
    source.add(point, original)
    assert source.prepared_graph_for(point)["ops"][0].as_dict()["input_shapes"] == [[8, 4]]


def test_prepared_lookup_keeps_address_shift_counters(tmp_path):
    measured = operation()
    measured["scalars"] = [["slot_mapping", [1, 2]]]
    library = library_for(tmp_path, measured)
    current = dict(measured, scalars=[["slot_mapping", [11, 12]]])
    prepared = prepare_static_operator(current)
    for _ in range(3):
        assert library.lookup(prepared)[0] is not None
    assert sum(library.address_shifted.values()) == 3
