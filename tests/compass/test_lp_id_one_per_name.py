# SPDX-License-Identifier: MIT
"""`LpId` equality is identity, so every way of making one must reach the same object.

A copy, an unpickled identity or one rebuilt from its name that came back as a
second object would compare unequal to the registered one and miss every
mapping keyed by it, with no error.
"""

import copy
import dataclasses
import pickle
import sys

import pytest

from atom.compass.clock import LpId, LpRegistry
from atom.compass.clock_transport import wire

MADE = {
    "constructed": lambda lp: LpId(lp.name),
    "by keyword": lambda lp: LpId(name=lp.name),
    "copied": copy.copy,
    "deep-copied": copy.deepcopy,
    "replaced": lambda lp: dataclasses.replace(lp, name=lp.name),
    **{
        f"pickled, protocol {p}": (lambda lp, p=p: pickle.loads(pickle.dumps(lp, p)))
        for p in range(pickle.HIGHEST_PROTOCOL + 1)
    },
}


@pytest.mark.parametrize("make", MADE.values(), ids=MADE.keys())
def test_every_way_of_making_an_identity_keys_the_registered_one(make):
    registry = LpRegistry()
    decode = registry.register(LpId("decode-one-per-name"))
    other = make(decode)
    assert other == decode and hash(other) == hash(decode)
    assert registry.require(other) is decode
    assert {decode: 1}[other] == 1


def test_a_bare_name_still_keys_nothing():
    decode = LpId("decode")
    assert decode != "decode" and "decode" not in {decode: 1}
    assert LpRegistry().register(decode) is decode


def test_a_refused_name_leaves_nothing_behind():
    with pytest.raises(ValueError):
        LpId("pp stage")
    with pytest.raises(ValueError):
        LpId("pp stage")


def test_a_bound_identity_off_the_wire_keys_the_registered_one():
    registry = LpRegistry()
    engine = registry.register(LpId("engine-one-per-name"))
    _, bound = wire.decode(wire.encode((wire.BIND, engine)))
    assert registry.require(bound) is engine


def test_a_lookup_by_an_equal_identity_runs_no_python_code():
    # The clock authority keys its tables by identity and looks one up dozens of
    # times per request; a generated `__hash__` or `__eq__` is a Python call each.
    table = {LpId("decode"): 1}
    probe = LpId("decode")
    calls = []
    sys.setprofile(lambda frame, event, _: event == "call" and calls.append(frame))
    try:
        found = table[probe] == 1 and probe == next(iter(table))
    finally:
        sys.setprofile(None)
    assert found and not calls, [frame.f_code.co_qualname for frame in calls]
