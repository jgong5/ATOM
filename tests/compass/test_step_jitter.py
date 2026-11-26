# SPDX-License-Identifier: MIT
"""``COMPASS_STEP_JITTER``: per-step multiplicative noise on each rank's priced
device time (`overrides.step_jitter`), off unless set, the same factors for a
seed in every run, mean one. A DP group's step is its ranks' max as before.
Two ranks run as two threads on a gloo group (`test_dp_step_max`'s `dp2`).
No device.
"""

import math
import statistics

import pytest
from test_dp_step_max import decode, dp2, prefill, runner  # noqa: F401  the dp2 fixture

from atom.compass.backends.step_stub import Coefficients, StepStubBackend
from atom.compass.runner import overrides

SPEC = {"decode": 0.05, "prefill": 0.02, "output": 0.0}


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for name in (overrides.STEP_JITTER_ENV, overrides.STEP_JITTER_SEED_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("COMPASS_HARNESS_SEND_ORDER_SEED", raising=False)


@pytest.mark.parametrize("raw", [None, ""])
def test_off_unset_or_empty(monkeypatch, raw):
    if raw is not None:
        monkeypatch.setenv(overrides.STEP_JITTER_ENV, raw)
    assert overrides.step_jitter_spec() is None
    assert overrides.step_jitter("decode", 0, 7, None) == 1.0


def test_a_spec_names_kinds_and_sigmas(monkeypatch):
    monkeypatch.setenv(overrides.STEP_JITTER_ENV, "decode=0.05,prefill=0.02,output=0")
    assert overrides.step_jitter_spec() == SPEC


@pytest.mark.parametrize(
    "raw", ["decode", "decode=x", "decode=-0.1", "mixed=0.1", "decode=inf"]
)
def test_a_bad_spec_is_refused(monkeypatch, raw):
    monkeypatch.setenv(overrides.STEP_JITTER_ENV, raw)
    with pytest.raises(ValueError, match=overrides.STEP_JITTER_ENV):
        overrides.step_jitter_spec()


def test_factors_repeat_per_seed_differ_by_rank_step_and_seed(monkeypatch):
    monkeypatch.setenv(overrides.STEP_JITTER_SEED_ENV, "3")
    a = [overrides.step_jitter("decode", r, s, SPEC) for r in (0, 1) for s in range(50)]
    assert a == [
        overrides.step_jitter("decode", r, s, SPEC) for r in (0, 1) for s in range(50)
    ]
    assert a[:50] != a[50:]
    monkeypatch.setenv(overrides.STEP_JITTER_SEED_ENV, "4")
    assert a != [
        overrides.step_jitter("decode", r, s, SPEC) for r in (0, 1) for s in range(50)
    ]
    # without its own seed it takes the client's send-order seed
    monkeypatch.delenv(overrides.STEP_JITTER_SEED_ENV)
    monkeypatch.setenv("COMPASS_HARNESS_SEND_ORDER_SEED", "3")
    assert a[:5] == [overrides.step_jitter("decode", 0, s, SPEC) for s in range(5)]


def test_factors_have_mean_one_and_the_kind_s_sigma():
    f = [overrides.step_jitter("decode", 0, s, SPEC) for s in range(20000)]
    assert statistics.mean(f) == pytest.approx(1.0, abs=2e-3)
    assert statistics.stdev(math.log(x) for x in f) == pytest.approx(0.05, rel=0.05)
    # a kind with sigma 0, one the spec does not name, or a dummy (None): none
    assert overrides.step_jitter("output", 0, 1, SPEC) == 1.0
    assert overrides.step_jitter("output", 0, 1, {"decode": 0.1}) == 1.0
    assert overrides.step_jitter(None, 0, 1, SPEC) == 1.0


def _stub(seconds):
    return StepStubBackend(
        Coefficients.constant(prefill_seconds=seconds, decode_seconds=seconds)
    )


def test_a_rank_s_step_is_priced_times_its_factor_and_the_group_takes_the_max(
    dp2, monkeypatch  # noqa: F811
):
    # a whole prompt in one chunk is an output prefill
    monkeypatch.setenv(overrides.STEP_JITTER_ENV, "decode=0.05,output=0.05")
    monkeypatch.setenv(overrides.STEP_JITTER_SEED_ENV, "11")
    ranks = [runner(_stub(0.1), dp_size=2, dp_rank=r) for r in (0, 1)]
    got = dp2(
        lambda: ranks[0].forward(prefill(40)), lambda: ranks[1].forward(prefill(8))
    )
    # each rank's first prefill forward: factor of (seed, rank, step 0)
    want = [
        0.1 * overrides.step_jitter("output", r, 0, {"output": 0.05}) for r in (0, 1)
    ]
    assert [g.predicted_s for g in got] == pytest.approx([max(want)] * 2)
    assert want[0] != want[1]
    # a second forward on each rank draws its next factor
    got = dp2(
        lambda: ranks[0].forward(prefill(40)), lambda: ranks[1].forward(prefill(8))
    )
    want = [
        0.1 * overrides.step_jitter("output", r, 1, {"output": 0.05}) for r in (0, 1)
    ]
    assert [g.predicted_s for g in got] == pytest.approx([max(want)] * 2)
