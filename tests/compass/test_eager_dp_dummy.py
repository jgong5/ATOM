# SPDX-License-Identifier: MIT
"""A DP dummy forward that replays no graph, beside a peer with work, is priced
0 s under the `eager DP dummy` treatment by a step linear law with no `eager`
vector and by a replay step grid, where each refused it before. A real eager
decode step and a dummy forward on one rank are still refused, an eager bundle
prices the dummy from its grid, and the run summary counts the treatment by step.
Under a speculative config the dummy needs no draft term; any other step without
one is still refused.
"""

import pytest
from test_backend_step_grid import ON_NODE, eager, replay
from test_backend_step_linear import law
from test_dp_step_max import dp2, prefill, runner  # noqa: F401  the dp2 fixture
from test_run_treatments import dp_summary
from test_speculative_step import _speculative

from atom.compass.backends import (
    BatchView,
    CostRefused,
    Method,
    RequestShape,
    StepLinearBackend,
    StepLinearLaw,
)
from atom.compass.backends.cost import EAGER_DP_DUMMY


def no_eager(dp=1):
    spec = law()
    del spec["decode"]["eager"]
    return StepLinearBackend(StepLinearLaw(spec), dp)


def dummy(rung=None, dp_dummy=True):
    return BatchView((RequestShape(1, 1, True),), capture_rung=rung, dp_dummy=dp_dummy)


@pytest.mark.parametrize("backend", [no_eager, replay], ids=["step-linear", "grid"])
def test_an_eager_dp_dummy_is_0_s_under_its_treatment(backend):
    cost = backend().price(dummy())

    assert cost.seconds == 0.0
    assert cost.treatments == (EAGER_DP_DUMMY,)
    assert [t.provenance.method for t in cost.terms] == [Method.DECLARED]


@pytest.mark.parametrize(
    "backend, reason",
    [
        (no_eager, "no decode vector for rung eager"),
        (replay, "a decode step replays no graph, and a replay bundle has no eager"),
    ],
    ids=["step-linear", "grid"],
)
def test_an_eager_decode_step_that_is_no_dp_dummy_still_refuses(backend, reason):
    with pytest.raises(CostRefused, match=reason):
        backend().price(dummy(dp_dummy=False))


def test_a_law_with_an_eager_vector_prices_the_dummy_with_it():
    cost = StepLinearBackend(StepLinearLaw(law()), 2).price(dummy())

    assert cost.seconds > 0.0
    assert EAGER_DP_DUMMY not in cost.treatments


def test_the_run_summary_counts_each_eager_dummy_step(
    monkeypatch, tmp_path, dp2  # noqa: F811
):
    ranks = [runner(no_eager(2), dp_size=2, dp_rank=r) for r in range(2)]
    # Two steps per rank: rank 1 has no work, so it runs ATOM's DP dummy.
    dp2(
        lambda: [ranks[0].forward(prefill(40)) for _ in range(2)],
        lambda: [ranks[1].dummy_execution() for _ in range(2)],
    )

    out = dp_summary(monkeypatch, tmp_path, ranks)

    assert out["schedule"]["treatments"] == {
        EAGER_DP_DUMMY: 2,
        "uniform routing": 2,
    }
    assert [r.refused_commands() for r in ranks] == [(), ()]


def test_an_eager_bundle_prices_the_dummy_from_its_grid():
    rows = tuple(RequestShape(*r) for r in ON_NODE)
    cost = eager().price(BatchView(rows, dp_dummy=True))

    assert cost.seconds > 0.0
    assert EAGER_DP_DUMMY not in cost.treatments


def test_a_dummy_forward_on_one_rank_still_refuses():
    r = runner(no_eager())

    assert r.dummy_execution().predicted_s == 0.0
    assert r.refused_commands() == (
        "cost:step-linear: no decode vector for rung eager",
    )


def test_under_a_speculative_config_the_dummy_is_0_s_and_a_draftless_step_refuses(
    monkeypatch, tmp_path, dp2  # noqa: F811
):
    ranks = [_speculative(runner(no_eager(2), dp_size=2, dp_rank=r)) for r in range(2)]
    dp2(lambda: ranks[0].forward(prefill(40)), lambda: ranks[1].dummy_execution())

    out = dp_summary(monkeypatch, tmp_path, ranks)

    assert out["schedule"]["treatments"] == {EAGER_DP_DUMMY: 1}
    assert [r.refused_commands() for r in ranks] == [
        ("cost:StepLinearBackend: prices no draft",),
        (),
    ]
