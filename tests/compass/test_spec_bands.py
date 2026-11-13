# SPDX-License-Identifier: MIT
"""A probed number is published with the band its probe measured, and every
reader still reads the one value."""

import json
import pathlib

import pytest
from test_spec_origin import labels, saved, three_origins
from test_spec_verbs import fragment

from atom.compass.artifacts import machine_platform
from atom.compass.spec import MachineSpec, SpecRefusal, merge
from atom.compass.spec.bands import refuse_unbanded
from atom.compass.spec.rules import Rule

ROOT = pathlib.Path(__file__).resolve().parents[2]
#: A probed number of `three_origins`, to state a band on.
BANDWIDTH = "device.memory.bandwidth_bytes_per_s"


def _banded(document, low=0.5, high=2.0):
    """`document` with every probed origin given a band around its value."""
    spec = MachineSpec.from_mapping(document)
    for label, found in document["provenance"]["origins"].items():
        if found["origin"] != "probed":
            continue
        if isinstance(spec.value(label), int | float):
            found["p10"], found["p90"] = (
                spec.value(label) * low,
                spec.value(label) * high,
            )
    return document


def test_a_probed_number_without_its_band_is_refused_by_name():
    spec = MachineSpec.from_mapping(saved(merge(three_origins()).document))
    with pytest.raises(SpecRefusal) as refused:
        refuse_unbanded(spec)
    assert refused.value.rule is Rule.BAND
    said = refused.value.what
    for owed in (BANDWIDTH, "device.compute.bf16_flops"):
        assert f"`{owed}`" in said
    # Read, not sampled: a count, a text, the capacity; a derate; an assumed link.
    for free in (
        "count_per_node",
        "device.name",
        "capacity_bytes",
        "derate",
        "interconnect",
    ):
        assert free not in said


def test_a_band_beside_each_probed_number_publishes_and_moves_no_value():
    plain = MachineSpec.from_mapping(saved(merge(three_origins()).document))
    banded = MachineSpec.from_mapping(_banded(saved(merge(three_origins()).document)))
    refuse_unbanded(banded)
    for label in labels(plain):
        assert banded.value(label) == plain.value(label), label
    assert machine_platform(banded) == machine_platform(plain)
    rate = banded.origin(BANDWIDTH)
    assert (rate["p10"], rate["p90"]) == (4.0e12, 1.6e13)


def test_a_saved_band_survives_a_second_merge():
    first = _banded(saved(merge(three_origins()).document))
    body = {k: v for k, v in first.items() if k != "name"}
    again = merge([fragment("machine.json", body)])
    spec = MachineSpec.from_mapping(saved(again.document))
    refuse_unbanded(spec)
    assert spec.origin("device.compute.fp8_flops")["p90"] == 1.0e16


@pytest.mark.parametrize(
    "band",
    [
        {"p10": 2.0, "p90": 1.0},
        {"p10": 1.0},
        {"p10": True, "p90": 2.0},
        {"p10": "1", "p90": 2.0},
        {"p10": {"a": 1.0}, "p90": {"b": 2.0}},
    ],
)
def test_a_band_is_two_ends_the_low_one_first(band):
    document = saved(merge(three_origins()).document)
    document["provenance"]["origins"][BANDWIDTH] |= band
    with pytest.raises(SpecRefusal, match="a band, a `p10` at most its `p90`"):
        MachineSpec.from_mapping(document)


def test_only_a_probed_field_states_a_band():
    document = saved(merge(three_origins()).document)
    assumed = document["provenance"]["origins"]["interconnect.router_relay_s"]
    assumed |= {"p10": 1e-3, "p90": 2e-3}
    with pytest.raises(SpecRefusal, match="only kind with a subject"):
        MachineSpec.from_mapping(document)


@pytest.mark.parametrize("name", ["mi308x-hjbog18", "mi308x-hjbog39", "mi355x-gpu45"])
def test_every_checked_in_spec_publishes_its_probed_numbers_with_a_band(name):
    path = ROOT / "tests" / "compass" / "machines" / f"{name}.json"
    refuse_unbanded(MachineSpec.from_mapping(json.loads(path.read_text())))
