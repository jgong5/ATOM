# SPDX-License-Identifier: MIT
"""Consulting cost sources in order, and what falling through leaves behind.

The failure these tests exist for is a run that silently answers from the
bottom of the ladder: every number present, every total plausible, and nothing
in the record saying the measured price was never found. So the assertions are
about what survives a fall-through, not about which source won -- the answer
carries the refusal that preceded it, the resolution carries all of them, and
an exhausted ladder raises with the complete list rather than the first entry.
"""

import pytest

from atom.compass.backends import (
    CostRefused,
    CostSource,
    CostTerm,
    Provenance,
    Refusal,
    Resolver,
    Species,
)


class Answers(CostSource):
    """A source that always answers, at a stated species."""

    def __init__(self, name, seconds, species):
        self._name = name
        self._seconds = seconds
        self._species = species

    @property
    def name(self):
        return self._name

    def price(self, request):
        return CostTerm(request, self._seconds, Provenance(self._species, self._name))


class Declines(CostSource):
    """A source that always declines, for a stated reason."""

    def __init__(self, name, reason):
        self._name = name
        self._reason = reason

    @property
    def name(self):
        return self._name

    def price(self, request):
        return self.refuse(self._reason)


PRICED = Answers("price_list", 0.004, Species.MEASURED)
LAW = Answers("analytic_law", 0.005, Species.ANALYTICAL)
UNPRICED = Declines("price_list", "no entry for this leaf")
OUTSIDE = Declines("nearest_key", "outside the measured range")


def test_the_top_source_answers_and_nothing_is_marked():
    got = Resolver([PRICED, LAW]).resolve("attention")
    assert got.source == "price_list"
    assert not got.fell_through
    assert got.term.provenance == Provenance(Species.MEASURED, "price_list")


def test_a_fall_through_is_stamped_on_the_answer():
    got = Resolver([UNPRICED, LAW]).resolve("attention")
    assert got.source == "analytic_law"
    assert got.fell_through
    assert got.term.provenance.refusal == Refusal(
        "price_list", "no entry for this leaf"
    )
    assert str(got.term.provenance) == (
        "refused(price_list: no entry for this leaf) -> analytical (analytic_law)"
    )


def test_every_source_passed_over_is_recorded_in_order():
    got = Resolver([UNPRICED, OUTSIDE, LAW]).resolve("attention")
    assert [r.source for r in got.declined] == ["price_list", "nearest_key"]
    assert got.term.provenance.refusal.source == "price_list"


def test_an_exhausted_ladder_refuses_with_the_whole_list():
    """One refusal per run turns one coverage gap into one iteration each."""
    with pytest.raises(CostRefused) as raised:
        Resolver([UNPRICED, OUTSIDE]).resolve("attention")
    assert [r.source for r in raised.value.declined] == ["price_list", "nearest_key"]
    assert raised.value.request == "attention"
    assert "outside the measured range" in str(raised.value)


def test_a_resolver_needs_sources():
    with pytest.raises(ValueError, match="no sources"):
        Resolver([])


def test_rungs_are_read_by_name_so_names_are_unique():
    with pytest.raises(ValueError, match="duplicate source"):
        Resolver([PRICED, Answers("price_list", 0.009, Species.FITTED)])


def test_the_ladder_reports_the_order_it_will_consult():
    assert Resolver([UNPRICED, OUTSIDE, LAW]).names == (
        "price_list",
        "nearest_key",
        "analytic_law",
    )


def test_a_source_cannot_pin_its_refusal_on_another_source():
    """Misattributed refusals make the reason counts name the wrong thing."""

    class Misattributes(Declines):
        def price(self, request):
            return Refusal("somebody_else", self._reason)

    resolver = Resolver([Misattributes("price_list", "no entry"), LAW])
    with pytest.raises(ValueError, match="has to name who declined"):
        resolver.resolve("attention")


def test_a_source_that_returns_a_bare_number_is_a_bug_not_a_cost():
    class ReturnsSeconds(CostSource):
        @property
        def name(self):
            return "loose"

        def price(self, request):
            return 0.004

    with pytest.raises(TypeError, match="not a cost term"):
        Resolver([ReturnsSeconds()]).resolve("attention")
