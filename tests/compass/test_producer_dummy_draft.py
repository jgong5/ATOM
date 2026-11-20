# SPDX-License-Identifier: MIT
"""A P/D producer's DP dummy runs no drafter: a producer never speculates, its
dummies included, and its drafter only writes its requests' context KV. A
dummy beside a prefilling peer, priced by the law's prefill vector, states no
draft term and is charged, not refused as pricing no draft; the same dummy on a
server that proposes is still refused."""

import copy

import pytest
from test_dp_step_max import dp2, prefill, runner  # noqa: F401  the dp2 fixture
from test_speculative_step import _speculative
from test_validate_cost import LAW

from atom.compass.backends import StepLinearBackend, StepLinearLaw

#: A producer's law: its draft prices an output prefill's draft-KV pass among
#: the target's terms and has no propose vector (#3720 cell B2's prefill side).
PRODUCER_LAW = {**LAW, "draft": {**copy.deepcopy(LAW), "output_prefill_kv": True}}
PRODUCER = {"kv_role": "kv_producer", "kv_connector": "compass"}


def _ranks(kv):
    ranks = []
    for r in range(2):
        backend = StepLinearBackend(StepLinearLaw(PRODUCER_LAW), 2)
        ranks.append(_speculative(runner(backend, dp_size=2, dp_rank=r)))
        ranks[-1].config.kv_transfer_config = kv
    return ranks


@pytest.mark.parametrize("multi", [False, True])
def test_a_producer_s_dummy_beside_a_prefilling_peer_is_charged(
    dp2, multi  # noqa: F811
):
    kv = {"kv_connector": "multi", "connectors": [PRODUCER]} if multi else PRODUCER
    ranks = _ranks(kv)
    busy, idle = dp2(lambda: ranks[0].forward(prefill(40)), ranks[1].dummy_execution)
    assert [r.refused_commands() for r in ranks] == [(), ()]
    assert busy.predicted_s > 0 and idle.group_targets == busy.group_targets


def test_the_same_dummy_on_a_server_that_proposes_is_still_refused(
    dp2,  # noqa: F811
):
    ranks = _ranks(None)
    dp2(lambda: ranks[0].forward(prefill(40)), ranks[1].dummy_execution)
    assert ranks[1].refused_commands() == ("cost:StepLinearBackend: prices no draft",)
