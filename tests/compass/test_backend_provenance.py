# SPDX-License-Identifier: MIT
"""The vocabulary every cost is labelled with.

The point of these tests is that the label cannot be omitted. A cost whose
origin is unknown is worse than a missing cost, because it is counted, summed
and reported like a real one; the constructors are written so that producing
one takes a deliberate lie rather than an oversight, and these assert that
property rather than the spellings.

The refusal rendering is asserted too. It is not cosmetic: the string is what
lands in a run record, and a stand-in answer that renders the same as a
first-choice one would hide exactly the fall-through the record exists to show.
"""

import pytest

from atom.compass.backends import Provenance, Refusal, Species


def test_the_species_vocabulary_is_closed():
    """Five ways an answer is obtained, and no sixth added in passing."""
    assert {s.value for s in Species} == {
        "analytical",
        "measured",
        "fitted",
        "interpolated",
        "extrapolated",
    }


def test_provenance_has_no_default_species():
    """There is no constructor call that leaves the origin unstated."""
    with pytest.raises(TypeError):
        Provenance()


def test_provenance_refuses_a_species_shaped_string():
    """A bare string would be silently accepted by the dataclass otherwise."""
    with pytest.raises(TypeError):
        Provenance("measured")


def test_a_refusal_names_who_declined_and_why():
    r = Refusal("price_list", "no entry for gemm(4096,8192)")
    assert r.source == "price_list"
    assert str(r) == "refused(price_list: no entry for gemm(4096,8192))"


@pytest.mark.parametrize(
    "source,reason",
    [("", "a reason"), ("  ", "a reason"), ("a_source", ""), ("a_source", " ")],
)
def test_a_refusal_with_a_blank_half_is_not_a_refusal(source, reason):
    """Half a refusal cannot be acted on and cannot be grouped when counted."""
    with pytest.raises(ValueError):
        Refusal(source, reason)


def test_plain_provenance_renders_species_and_unit():
    assert str(Provenance(Species.MEASURED)) == "measured"
    assert str(Provenance(Species.MEASURED, "op-level")) == "measured (op-level)"


def test_a_stand_in_answer_renders_the_refusal_it_replaced():
    """Reading the record, a fall-through is visible without comparing runs."""
    p = Provenance(Species.ANALYTICAL, "roofline").after(
        Refusal("price_list", "unpriced leaf")
    )
    assert p.is_refused
    assert str(p) == "refused(price_list: unpriced leaf) -> analytical (roofline)"


def test_the_first_refusal_is_the_one_kept():
    """Two sources declined; the top one is what an operator has to measure."""
    p = Provenance(Species.ANALYTICAL).after(Refusal("price_list", "unpriced leaf"))
    kept = p.after(Refusal("nearest_key", "outside the measured range"))
    assert kept.refusal == Refusal("price_list", "unpriced leaf")
