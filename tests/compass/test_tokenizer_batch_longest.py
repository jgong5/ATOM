# SPDX-License-Identifier: MIT
"""COMPASS_TOKENIZER_BATCH_LONGEST: a several-item encode_batch charged over its
longest encoding (`atom.utils.compass_loop.wrap_encode_batch`)."""

from types import SimpleNamespace

import pytest

from atom.utils import compass_loop
from atom.utils.compass_loop import BATCH_LONGEST_ENV, wrap_encode_batch

ONE = {"fixed_s": 0.001, "tokens_per_s": 1e6}
SEVERAL = {"fixed_s": 0.002, "tokens_per_s": 5e5}


@pytest.fixture
def charges(monkeypatch):
    got = []
    monkeypatch.setattr(compass_loop, "charge", got.append)
    compass_loop._encoded.ids = 0
    return got


def batch(*lengths):
    return lambda *a, **k: [SimpleNamespace(ids=[0] * n) for n in lengths]


@pytest.mark.parametrize("value", [None, "", "0"])
def test_off_a_batch_is_charged_over_all_its_ids(charges, monkeypatch, value):
    if value is None:
        monkeypatch.delenv(BATCH_LONGEST_ENV, raising=False)
    else:
        monkeypatch.setenv(BATCH_LONGEST_ENV, value)
    wrap_encode_batch(batch(30000, 10000, 2000), ONE, SEVERAL)(["a", "b", "c"])
    wrap_encode_batch(batch(500), ONE, SEVERAL)(["a"])
    assert charges == [
        pytest.approx(0.002 + 42000 / 5e5),
        pytest.approx(0.001 + 500 / 1e6),
    ]


def test_on_a_batch_is_charged_over_its_longest_item(charges, monkeypatch):
    monkeypatch.setenv(BATCH_LONGEST_ENV, "1")
    wrap_encode_batch(batch(30000, 10000, 2000), ONE, SEVERAL)(["a", "b", "c"])
    wrap_encode_batch(batch(500), ONE, SEVERAL)(["a"])
    wrap_encode_batch(batch(), ONE, SEVERAL)([])
    assert charges == [
        pytest.approx(0.002 + 30000 / 5e5),
        pytest.approx(0.001 + 500 / 1e6),
        pytest.approx(0.002),
    ]
    # the segment tokenizer's own work is still over every id encoded
    assert compass_loop._encoded.ids == 42500


def test_a_batch_inside_an_encode_is_the_encode_s_to_charge(charges, monkeypatch):
    monkeypatch.setenv(BATCH_LONGEST_ENV, "1")
    compass_loop._in_encode.on = True
    try:
        wrap_encode_batch(batch(30000, 10000), ONE, SEVERAL)(["a", "b"])
    finally:
        compass_loop._in_encode.on = False
    assert charges == []
