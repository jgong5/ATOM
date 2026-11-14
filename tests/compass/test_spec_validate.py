# SPDX-License-Identifier: MIT
"""Checking a spec document, and comparing the ranks' readings of one machine.

The document is the reference spec of `test_spec_schema`, so a test that breaks
it breaks exactly what it says it does.

The rank numbers are the measured ones, because these checks are judgements
about size: the cross-rank spread accepted is the one the hardware really
shows, and the spread refused is the one a neighbour really produced. Round
numbers would make both tests vacuous.
"""

import pytest
from test_spec_schema import document, drop

from atom.compass import spec as spec_package
from atom.compass.spec import SpecRefusal, across_ranks, validate
from atom.compass.spec.rules import Rule

MIB = 1024**2


# --- validate: every reason at once, and the refusal list --------------------


def test_a_complete_spec_validates_and_resolves():
    checked = validate(document())
    assert checked.ok, [str(refusal) for refusal in checked.refusals]
    assert str(checked) == "ok: 0 refusal(s)"


def test_every_missing_field_is_reported_and_not_only_the_first():
    thin = drop(
        drop(document(), "device", "arch"), "device", "memory", "capacity_bytes"
    )
    checked = validate(thin)
    missing = {
        refusal.what.split("`")[1]
        for refusal in checked.refusals
        if "`" in refusal.what
    }
    assert {"device.arch", "device.memory.capacity_bytes"} <= missing
    assert not checked.ok


def test_a_spec_peak_with_no_derate_beside_it_is_refused():
    checked = validate(drop(document(), "device", "compute", "derate", "bf16"))
    assert [refusal.rule for refusal in checked.refusals] == [Rule.DERATE]


def test_an_unknown_key_is_refused_where_it_sits():
    checked = validate(dict(document(), gpu_memory_utilization=0.9))
    assert [refusal.rule for refusal in checked.refusals] == [Rule.SEPARATION]


def test_a_field_stated_twice_is_reported_beside_the_rest_rather_than_raised():
    # The reader refuses a document that states one field under two spellings.
    # The check walks the same document, so it reports that refusal with the
    # others and keeps walking: raised out of the walk instead, it would take
    # every other result of the run with it, the unknown key beside it included.
    written = dict(document(), gpu_memory_utilization=0.9)
    assert "link_latency_s" in written["interconnect"]["inter_node"]
    written["interconnect.inter_node.link_latency_s"] = 1
    checked = validate(written)
    rules = sorted(refusal.rule.name for refusal in checked.refusals)
    assert rules == ["SEPARATION", "SHAPE"]
    assert any("stated twice" in refusal.what for refusal in checked.refusals)


def test_one_mistyped_key_does_not_take_the_rest_of_the_document_with_it():
    # The refusal a mistyped key earns is about that key; the field it was
    # meant to state is reported missing beside it.
    written = document()
    memory = written["device"]["memory"]
    memory["capacity_byte"] = memory.pop("capacity_bytes")
    checked = validate(written)
    rules = [refusal.rule for refusal in checked.refusals]
    assert rules.count(Rule.SEPARATION) == 1
    assert rules.count(Rule.SHAPE) == 1


def test_a_block_holding_a_scalar_does_not_take_the_rest_with_it_either():
    # The other way the walk refuses: a block written as a number. The fields
    # under it are gone, and everything beside it is still checked.
    written = document()
    written["device"]["memory"] = 288.0e9
    checked = validate(written)
    rules = [refusal.rule for refusal in checked.refusals]
    assert rules.count(Rule.DERATE) == 1


def test_a_value_of_the_wrong_shape_is_reported():
    written = document()
    written["device"]["count_per_node"] = True
    checked = validate(written)
    assert [refusal.rule for refusal in checked.refusals] == [Rule.SHAPE]


# --- one reading per rank ----------------------------------------------------


@pytest.mark.parametrize(
    "tp_width,reading,spread",
    [
        (1, 926 * MIB, 0),
        (2, 6906 * MIB, 0),
        (4, 7266 * MIB, 192 * MIB),
        (8, 10704 * MIB, 640 * MIB),
    ],
)
def test_the_spread_this_hardware_really_shows_is_not_an_error(
    tp_width, reading, spread
):
    across_ranks("non_torch", [reading] * (tp_width - 1) + [reading + spread])


def test_a_neighbour_holding_fifty_times_the_rank_is_refused():
    # The case this exists for: a rank had reserved 2.94 GB and the card read
    # 152.01 GB, and six runs died on a negative memory budget before anything
    # compared the ranks.
    with pytest.raises(SpecRefusal) as refused:
        across_ranks("non_torch", [2.94e9, 2.94e9, 2.94e9, 152.01e9])
    assert refused.value.rule is Rule.RANK_AGREEMENT
    assert "2940000000.0" in refused.value.what
    assert "152010000000.0" in refused.value.what


def test_a_reading_no_spec_could_hold_is_refused_where_the_ranks_are():
    with pytest.raises(SpecRefusal) as refused:
        across_ranks("non_torch", [0, 970.0e6])
    assert refused.value.rule is Rule.SHAPE


# --- what the package reaches -------------------------------------------------


def test_the_verbs_are_reachable_by_name():
    for name in ("validate", "across_ranks"):
        assert name in spec_package.__all__
        assert getattr(spec_package, name, None) is not None
