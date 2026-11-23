# SPDX-License-Identifier: MIT
"""`lacking.uncovered_rungs` holds a prefill-decode producer to no decode rung.

A producer's requests leave after their prefill, and the one decode-phase step
it takes, a DP group's dummy with every rank idle, is priced without a vector
(`cost.idle_group_dummy`), so `compass run --check` on a prefill side's law
names no decode rung it lacks, whatever its capture ladder. Whether a config
is one reads its KV transports' roles, a `multi` expanded and its offload
tiers left out, as the C1228 recipe's prefill runs a Mooncake ``kv_producer``
beside its ``mooncake_store`` offload.
"""

from types import SimpleNamespace

import pytest

from atom.compass.drive import lacking

#: A step linear law with no decode vector at all.
NO_DECODE = {"prefill": {"1": [0.0]}}


def _config(kv):
    """A config with `kv` as its ``kv_transfer_config``; nothing else is read
    before a producer is told apart."""
    return SimpleNamespace(kv_transfer_config=kv)


PRODUCERS = {
    "mooncake": {"kv_role": "kv_producer", "kv_connector": "mooncake"},
    "compass": {"kv_role": "kv_producer", "kv_connector": "compass"},
    "recipe multi": {
        "kv_connector": "multi",
        "connectors": [
            {"kv_role": "kv_producer", "kv_connector": "mooncake"},
            {"kv_role": "kv_both", "kv_connector": "mooncake_store"},
        ],
    },
}
DECODERS = {
    "consumer": {"kv_role": "kv_consumer", "kv_connector": "mooncake"},
    "both": {"kv_role": "kv_both", "kv_connector": "mooncake"},
    "no role": {"kv_connector": "mooncake"},
    "no transfer": None,
    "offload alone": {"kv_role": "kv_both", "kv_connector": "mooncake_store"},
    "consumer beside a producer": {
        "kv_connector": "multi",
        "connectors": [
            {"kv_role": "kv_producer", "kv_connector": "mooncake"},
            {"kv_role": "kv_consumer", "kv_connector": "mooncake"},
        ],
    },
}


@pytest.mark.parametrize("kv", PRODUCERS.values(), ids=PRODUCERS)
def test_a_producer_schedules_no_decode_rung_its_law_must_cover(kv):
    config = _config(kv)

    assert lacking._producer(config)
    assert lacking.uncovered_rungs(NO_DECODE, config) == []


@pytest.mark.parametrize("kv", DECODERS.values(), ids=DECODERS)
def test_any_other_config_decodes(kv):
    assert not lacking._producer(_config(kv))


def test_a_consumer_s_rungs_are_still_held_to_its_law(monkeypatch):
    """The exemption is the producer's alone: a consumer's scheduled rungs are
    what the law is held to, as before."""
    monkeypatch.setattr(lacking, "_ladder", lambda config: [1, 2, 4])
    monkeypatch.setattr(lacking, "schedulable_bound", lambda config: 4)
    config = _config(DECODERS["consumer"])
    law = {**NO_DECODE, "decode": {"1": [0.0], "2": [0.0]}}

    assert lacking.uncovered_rungs(law, config) == [
        r for r in lacking.scheduled_rungs([1, 2, 4], 4) if str(r) not in ("1", "2")
    ]
