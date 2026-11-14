"""Where each field of a machine spec came from, and that it survives a save.

A spec of several origins records each field's in `provenance.origins`. Each
test here reads a document as a file holds it, so what a writer knew and did
not write down is what these check for.
"""

import copy
import json

import pytest
from test_spec_schema import DOCUMENT

from atom.compass.spec import MachineSpec, SpecRefusal
from atom.compass.spec.fields import SCHEMA
from atom.compass.spec.rules import Rule

CARD = {"model": "Qwen/Qwen3.8-27B", "width": 1, "source_root": "commit:983e42973"}
LATENCY = "interconnect.inter_node.link_latency_s"


def saved(document):
    """A document as a file holds it: JSON."""
    return json.loads(json.dumps(document))


def labels(spec):
    """Every field the spec holds, as `origin` is asked for it."""
    for field in SCHEMA:
        path = field.path
        if path in ("name", "schema_version") or path.startswith("provenance."):
            continue
        if path in spec.values:
            yield path


def flat(body, prefix=""):
    """A document body as its fields' dotted paths, each to its value."""
    fields = {}
    for key, value in body.items():
        if isinstance(value, dict):
            fields |= flat(value, f"{prefix}{key}.")
        else:
            fields[prefix + key] = value
    return fields


def mixed(name, *parts):
    """The saved document of machine `name` holding each `(entry, body)` part's
    fields, written dotted, and recording `entry` as the origin of each; a later
    part's field replaces an earlier one's."""
    values, origins = {}, {}
    for entry, body in parts:
        values |= flat(body)
        origins |= dict.fromkeys(flat(body), entry)
    provenance = {
        "authored_by": "a person",
        "date": "2026-09-18",
        "origin": "mixed",
        "origins": origins,
    }
    return saved({"schema_version": 1, "name": name, "provenance": provenance} | values)


def three_origins():
    """A saved spec of one machine whose fields have each origin: the device
    probed on two subjects, a latency from a datasheet, the rest assumed."""
    return mixed(
        DOCUMENT["name"],
        ({"origin": "probed", **CARD}, {"device": DOCUMENT["device"]}),
        ({"origin": "probed", "model": "Qwen/Qwen3-8B"}, {"device.ptl": "disabled"}),
        ({"origin": "assumed"}, {"interconnect": DOCUMENT["interconnect"]}),
        ({"origin": "datasheet"}, {LATENCY: 5.0e-6}),
    )


def test_a_saved_spec_answers_origin_for_every_field():
    spec = MachineSpec.from_mapping(three_origins())
    assert spec.value("provenance.origin") == "mixed"
    answered = {label: spec.origin(label)["origin"] for label in labels(spec)}
    assert set(answered.values()) == {"probed", "datasheet", "assumed"}
    capacity = "device.memory.capacity_bytes"
    assert spec.origin(capacity) == {"origin": "probed", **CARD}
    assert spec.origin(LATENCY) == {"origin": "datasheet"}
    assert spec.origin("interconnect.intra_node.derate") == {"origin": "assumed"}
    assert spec.origin("device.ptl") == {
        "origin": "probed",
        "model": "Qwen/Qwen3-8B",
    }


def test_a_mixed_document_without_its_map_is_refused_by_name():
    document = three_origins()
    del document["provenance"]["origins"]
    spec = MachineSpec.from_mapping(document)
    with pytest.raises(SpecRefusal) as refused:
        spec.origin("device.memory.capacity_bytes")
    assert refused.value.rule is Rule.TOTALITY
    assert "records no origin for `device.memory.capacity_bytes`" in str(refused.value)


def test_a_document_of_one_origin_answers_without_the_map():
    stanza = dict(DOCUMENT["provenance"], subject={"model": "Qwen/Qwen3-8B"})
    spec = MachineSpec.from_mapping(dict(DOCUMENT, provenance=stanza))
    assert "provenance.origins" not in spec.values
    assert spec.origin("interconnect.router_relay_s") == {
        "origin": "probed",
        "model": "Qwen/Qwen3-8B",
    }


@pytest.mark.parametrize(
    "stated", ["guessed", "transferred-from:mi300x-8gpu", "transferred-from:", "", 7]
)
def test_an_origin_nobody_defined_is_refused(stated):
    stanza = dict(DOCUMENT["provenance"], origin=stated)
    with pytest.raises(SpecRefusal, match="which is not an origin"):
        MachineSpec.from_mapping(dict(DOCUMENT, provenance=stanza))


def test_one_field_is_never_mixed_and_only_a_probed_one_has_a_subject():
    document = three_origins()
    origins = document["provenance"]["origins"]
    for entry, wanted in (
        ({"origin": "mixed"}, "which is not an origin"),
        ({"origin": "assumed", "model": "m"}, "a probed field"),
        ({"origin": "probed", "rack": "r7"}, "a mapping of some of"),
    ):
        broken = copy.deepcopy(document)
        broken["provenance"]["origins"] = dict(
            origins, **{"interconnect.router_relay_s": entry}
        )
        with pytest.raises(SpecRefusal, match=wanted):
            MachineSpec.from_mapping(broken)
    stanza = dict(DOCUMENT["provenance"], origin="assumed", subject={"model": "m"})
    with pytest.raises(SpecRefusal, match="state it on a probed document only"):
        MachineSpec.from_mapping(dict(DOCUMENT, provenance=stanza))


def test_origin_of_a_host_field_is_refused_as_of_any_unknown_path():
    spec = MachineSpec.from_mapping(three_origins())
    for path in ("host.http_intake.fixed_s", "device.memory.colour"):
        with pytest.raises(SpecRefusal, match="not a field of this schema"):
            spec.origin(path)
