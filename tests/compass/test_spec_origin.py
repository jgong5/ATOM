"""Where each field of a machine spec came from, and that it survives a save.

A spec is merged from fragments of different origins, written to a file and
read back, sometimes merged again with next week's probe. Each test here goes
through that save: what a merge knew and did not write down is what these
check for.
"""

import copy
import json

import pytest
from test_spec_schema import DOCUMENT
from test_spec_verbs import CLASS1, LINKS, fragment

from atom.compass.spec import MachineSpec, SpecRefusal, explain, merge
from atom.compass.spec.fields import SCHEMA
from atom.compass.spec.rules import Rule

CARD = {"model": "Qwen/Qwen3.8-27B", "width": 1, "source_root": "commit:983e42973"}


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


def three_origins():
    """Probed and assumed fragments of one machine, the probes of two subjects."""
    return [
        fragment(
            "ptl", {"device": {"ptl": "disabled"}}, subject={"model": "Qwen/Qwen3-8B"}
        ),
        fragment("class1", CLASS1, subject=CARD),
        fragment("links", LINKS, origin="assumed"),
    ]


def test_a_saved_spec_answers_origin_for_every_field():
    spec = MachineSpec.from_mapping(saved(merge(three_origins()).document))
    assert spec.value("provenance.origin") == "mixed"
    answered = {label: spec.origin(label)["origin"] for label in labels(spec)}
    assert set(answered.values()) == {"probed", "assumed"}
    capacity = "device.memory.capacity_bytes"
    assert spec.origin(capacity) == {"origin": "probed", **CARD}
    assert spec.origin("interconnect.intra_node.derate") == {"origin": "assumed"}
    assert spec.origin("device.ptl") == {
        "origin": "probed",
        "model": "Qwen/Qwen3-8B",
    }


def test_a_field_two_fragments_agree_on_is_assumed_only_if_both_assumed_it():
    links = fragment("links", LINKS, origin="assumed")
    probed = fragment("probe", {"interconnect": {"inter_node": {"derate": 0.80}}})
    sheet = fragment(
        "datasheet",
        {"interconnect": {"inter_node": {"link_latency_s": 5.0e-6}}},
        origin="datasheet",
    )
    fragments = [*three_origins()[:2], probed, sheet, links]
    document = saved(merge(fragments).document)
    spec = MachineSpec.from_mapping(document)
    assert spec.origin("interconnect.inter_node.derate") == {"origin": "probed"}
    assert spec.origin("interconnect.inter_node.link_latency_s") == {
        "origin": "datasheet"
    }
    assert spec.origin("interconnect.inter_node.link_bandwidth_bytes_per_s") == {
        "origin": "assumed"
    }


def test_a_saved_spec_merged_again_keeps_each_fields_origin():
    first = saved(merge(three_origins()).document)
    week_later = {"interconnect": {"inter_node": {"link_latency_s": 5.0e-6}}}
    again = merge(
        [
            fragment("machine.json", {k: v for k, v in first.items() if k != "name"}),
            fragment("probe", week_later, date="2026-09-25"),
        ]
    )
    spec = MachineSpec.from_mapping(saved(again.document))
    assert spec.origin("interconnect.inter_node.link_latency_s") == {"origin": "probed"}
    assert spec.origin("interconnect.intra_node.link_latency_s") == {
        "origin": "assumed"
    }
    capacity = "device.memory.capacity_bytes"
    assert spec.origin(capacity) == {"origin": "probed", **CARD}


def test_a_mixed_document_without_its_map_is_refused_by_name():
    document = saved(merge(three_origins()).document)
    del document["provenance"]["origins"]
    spec = MachineSpec.from_mapping(document)
    with pytest.raises(SpecRefusal) as refused:
        spec.origin("device.memory.capacity_bytes")
    assert refused.value.rule is Rule.TOTALITY
    assert "records no origin for `device.memory.capacity_bytes`" in str(refused.value)
    with pytest.raises(SpecRefusal, match="records no origin"):
        explain(spec, "device.memory.capacity_bytes")
    with pytest.raises(SpecRefusal, match="records no origin"):
        merge(
            [fragment("machine.json", {"device": document["device"]}, origin="mixed")]
        )


def test_a_document_of_one_origin_answers_without_the_map():
    stanza = dict(DOCUMENT["provenance"], subject={"model": "Qwen/Qwen3-8B"})
    spec = MachineSpec.from_mapping(dict(DOCUMENT, provenance=stanza))
    assert "provenance.origins" not in spec.values
    assert spec.origin("interconnect.inter_node.derate") == {
        "origin": "probed",
        "model": "Qwen/Qwen3-8B",
    }


@pytest.mark.parametrize(
    "stated", ["guessed", "transferred-from:mi300x-8gpu", "transferred-from:", "", 7]
)
def test_an_origin_nobody_defined_is_refused(stated):
    with pytest.raises(SpecRefusal, match="which is not an origin"):
        fragment("class1", CLASS1, origin=stated)


def test_one_field_is_never_mixed_and_only_a_probed_one_has_a_subject():
    document = saved(merge(three_origins()).document)
    origins = document["provenance"]["origins"]
    for entry, wanted in (
        ({"origin": "mixed"}, "which is not an origin"),
        ({"origin": "assumed", "model": "m"}, "a probed field"),
        ({"origin": "probed", "rack": "r7"}, "a mapping of some of"),
    ):
        broken = copy.deepcopy(document)
        broken["provenance"]["origins"] = dict(
            origins, **{"interconnect.inter_node.derate": entry}
        )
        with pytest.raises(SpecRefusal, match=wanted):
            MachineSpec.from_mapping(broken)
    with pytest.raises(SpecRefusal, match="state it on a probed document only"):
        fragment("links", LINKS, origin="assumed", subject={"model": "m"})


def test_origin_of_a_host_field_is_refused_as_of_any_unknown_path():
    spec = MachineSpec.from_mapping(saved(merge(three_origins()).document))
    for path in ("host.http_intake.fixed_s", "device.memory.colour"):
        with pytest.raises(SpecRefusal, match="not a field of this schema"):
            spec.origin(path)


def test_explain_names_each_contributions_origin():
    combination = merge(three_origins())
    spec = MachineSpec.from_mapping(saved(combination.document))
    rows = {row.path: row for row in explain(spec, "device").contributions}
    assert rows["device.memory.capacity_bytes"].origin == {"origin": "probed", **CARD}
    assert str(rows["device.memory.capacity_bytes"]).endswith(
        "probed on model Qwen/Qwen3.8-27B, width 1, source_root commit:983e42973"
    )
    stated = {row.path: row.origin for row in explain(spec, "provenance").contributions}
    assert set(stated.values()) == {None}
