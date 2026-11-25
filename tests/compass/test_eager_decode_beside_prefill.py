# SPDX-License-Identifier: MIT
"""A DP rank's decode that replays no graph while a peer prefills is priced 0 s
under the `eager decode beside prefill` treatment by a step linear law with no
`eager` vector and by a replay step grid, where each refused it before. The
projection says whether a rank of the step's group prefills. An eager decode
with no prefilling peer is still refused, a law with an `eager` vector and an
eager bundle price the decode themselves, and the run summary counts the
treatment by step.
"""

import pytest
from test_backend_step_grid import ON_NODE, eager, replay
from test_backend_step_linear import law
from test_dp_step_max import Recording, dp2, prefill, runner  # noqa: F401  dp2
from test_eager_dp_dummy import no_eager
from test_moe_gather_rows import decode
from test_run_treatments import dp_summary

from atom.compass.backends import (
    BatchView,
    CostRefused,
    Method,
    RequestShape,
    StepLinearBackend,
    StepLinearLaw,
)

BESIDE = "eager decode beside prefill"


def eager_decode(group_prefills=True):
    rows = (RequestShape(1, 900, True),)
    return BatchView(rows, dp_rank=1, group_prefills=group_prefills)


@pytest.mark.parametrize("backend", [no_eager, replay], ids=["step-linear", "grid"])
def test_an_eager_decode_beside_a_prefilling_peer_is_0_s_under_its_treatment(backend):
    cost = backend().price(eager_decode())

    assert cost.seconds == 0.0
    assert cost.treatments == (BESIDE,)
    assert [t.provenance.method for t in cost.terms] == [Method.DECLARED]


@pytest.mark.parametrize(
    "backend, reason",
    [
        (no_eager, "no decode vector for rung eager"),
        (replay, "a decode step replays no graph, and a replay bundle has no eager"),
    ],
    ids=["step-linear", "grid"],
)
def test_an_eager_decode_with_no_prefilling_peer_still_refuses(backend, reason):
    with pytest.raises(CostRefused, match=reason):
        backend().price(eager_decode(group_prefills=False))


def test_a_law_with_an_eager_vector_prices_the_decode_with_it():
    cost = StepLinearBackend(StepLinearLaw(law()), 2).price(eager_decode())

    assert cost.seconds > 0.0
    assert BESIDE not in cost.treatments


def test_an_eager_bundle_prices_the_decode_from_its_grid():
    rows = tuple(RequestShape(*r) for r in ON_NODE)
    cost = eager().price(BatchView(rows, dp_rank=1, group_prefills=True))

    assert cost.seconds > 0.0
    assert BESIDE not in cost.treatments


def test_each_rank_s_view_says_whether_a_rank_of_its_group_prefills(
    dp2,  # noqa: F811
):
    backends = [Recording(), Recording()]
    ranks = [runner(b, dp_size=2, dp_rank=r) for r, b in enumerate(backends)]
    # A step rank 0 prefills while rank 1 decodes, then one both decode.
    dp2(
        lambda: [ranks[0].forward(b) for b in (prefill(40), decode(1))],
        lambda: [ranks[1].forward(b) for b in (decode(1), decode(1))],
    )

    assert [[v.group_prefills for v in b.views] for b in backends] == [
        [True, False],
        [True, False],
    ]
    assert [v.capture_rung for v in backends[1].views] == [None, None]


def test_the_run_summary_counts_each_eager_decode_beside_prefill(
    monkeypatch, tmp_path, dp2  # noqa: F811
):
    ranks = [runner(no_eager(2), dp_size=2, dp_rank=r) for r in range(2)]
    # Two steps per rank: rank 1 decodes, eagerly, while rank 0 prefills.
    dp2(
        lambda: [ranks[0].forward(prefill(40)) for _ in range(2)],
        lambda: [ranks[1].forward(decode(1)) for _ in range(2)],
    )

    out = dp_summary(monkeypatch, tmp_path, ranks)

    assert out["schedule"]["treatments"] == {BESIDE: 2, "uniform routing": 2}
    assert [r.refused_commands() for r in ranks] == [(), ()]


def test_an_eager_decode_beside_decoding_peers_still_refuses(dp2):  # noqa: F811
    ranks = [runner(no_eager(2), dp_size=2, dp_rank=r) for r in range(2)]
    dp2(lambda: ranks[0].forward(decode(1)), lambda: ranks[1].forward(decode(2)))

    refused = ("cost:step-linear: no decode vector for rung eager",)
    assert [r.refused_commands() for r in ranks] == [refused, refused]
