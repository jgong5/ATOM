# SPDX-License-Identifier: MIT
"""A DP dummy forward needs no host law for its kind, decode: no fit samples a
dummy's host time, and a P/D producer's host laws, fitted on its prefills and
middle chunks, hold no decode kind. Its idle group's dummies and a dummy beside
a prefilling peer are charged, not refused (#3689); a real decode on those
host laws is still refused."""

import numpy as np
import pytest
from test_dp_step_max import decode, dp2, prefill, runner  # noqa: F401  dp2
from test_eager_dp_dummy import no_eager

from atom.compass.backends.cost import EAGER_DP_DUMMY, IDLE_DP_GROUP
from atom.compass.runner.overrides import HostDeviceRecurrence

#: A producer's host laws: its prefills and middle chunks, no decode kind.
PREFILL_ONLY = {
    kind: {s: [0.001, 0.0, 0.0] for s in ("E", "W_pre", "W_launch", "W_post")}
    for kind in ("prefill", "middle")
}


def _ranks(captured: bool):
    ranks = [runner(no_eager(2), dp_size=2, dp_rank=r) for r in range(2)]
    for r in ranks:
        r.compass_host = HostDeviceRecurrence(PREFILL_ONLY, (), lambda: 0.0)
        if captured:  # the law holds rung 2 alone: a dummy replays rung 1
            r.capture_sizes_np, r.enforce_eager = np.array([1, 2], np.int32), False
    return ranks


@pytest.mark.parametrize(
    "captured, treatment", [(False, EAGER_DP_DUMMY), (True, IDLE_DP_GROUP)]
)
def test_an_idle_group_s_dummies_are_charged_with_no_decode_host_law(
    dp2, captured, treatment  # noqa: F811
):
    ranks = _ranks(captured)
    replies = dp2(*(r.dummy_execution for r in ranks))
    assert [r.refused_commands() for r in ranks] == [(), ()]
    assert [r.predicted_s for r in replies] == [0.0, 0.0]
    assert all(r._priced.treatments == {treatment: 1} for r in ranks)


def test_a_dummy_beside_a_prefilling_peer_is_priced_by_the_law_and_not_refused(
    dp2,  # noqa: F811
):
    """Its row priced by the law's prefill vector, under no treatment."""
    ranks = _ranks(False)
    busy, idle = dp2(lambda: ranks[0].forward(prefill(40)), ranks[1].dummy_execution)
    assert [r.refused_commands() for r in ranks] == [(), ()]
    assert not {EAGER_DP_DUMMY, IDLE_DP_GROUP} & set(ranks[1]._priced.treatments)
    assert idle.group_targets == busy.group_targets


def test_a_decode_on_host_laws_with_no_decode_kind_is_still_refused(
    dp2,  # noqa: F811
):
    ranks = _ranks(False)
    dp2(lambda: ranks[0].forward(decode(40)), ranks[1].dummy_execution)
    refused = ranks[0].refused_commands()
    assert len(refused) == 1 and "host: no law for decode.E" in refused[0]
    assert ranks[1].refused_commands() == ()
