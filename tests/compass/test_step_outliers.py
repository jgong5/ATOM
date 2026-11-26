# SPDX-License-Identifier: MIT
"""``COMPASS_STEP_OUTLIERS``: rare slow prefill steps, drawn per group step from
a seeded digest, the same on every rank, only at a step where some rank of
the data-parallel group finishes a prompt's prefill and at least two ranks run
real rows. Unset or empty: none. No device."""

from types import SimpleNamespace

import pytest

from atom.compass.runner.overrides import (
    STEP_OUTLIERS_ENV,
    STEP_OUTLIERS_SEED_ENV,
    step_outlier,
    step_outliers_spec,
)


def _view(tail=True, dummies=0):
    return SimpleNamespace(group_tail=tail, group_dummies=dummies)


def test_unset_or_empty_is_none(monkeypatch):
    monkeypatch.delenv(STEP_OUTLIERS_ENV, raising=False)
    assert step_outliers_spec() is None
    monkeypatch.setenv(STEP_OUTLIERS_ENV, "")
    assert step_outliers_spec() is None
    assert step_outlier(_view(), 4, 0, None) == 0.0


def test_the_spec_takes_a_rate_and_an_excess(monkeypatch):
    monkeypatch.setenv(STEP_OUTLIERS_ENV, "rate=0.05,excess_s=0.24")
    assert step_outliers_spec() == {"rate": 0.05, "excess_s": 0.24}


@pytest.mark.parametrize(
    "raw",
    [
        "rate=2,excess_s=0.1",
        "rate=0.1",
        "rate=0.1,excess_s=-1",
        "rate=x,excess_s=1",
        "r=0.1,excess_s=1",
    ],
)
def test_a_bad_spec_is_refused(monkeypatch, raw):
    monkeypatch.setenv(STEP_OUTLIERS_ENV, raw)
    with pytest.raises(ValueError, match=STEP_OUTLIERS_ENV):
        step_outliers_spec()


def test_only_a_multi_rank_tail_step_can_run_slow(monkeypatch):
    monkeypatch.setenv(STEP_OUTLIERS_SEED_ENV, "7")
    always = {"rate": 1.0, "excess_s": 0.25}
    assert step_outlier(_view(), 4, 3, always) == 0.25
    assert step_outlier(_view(tail=False), 4, 3, always) == 0.0
    assert step_outlier(_view(dummies=3), 4, 3, always) == 0.0  # one real rank
    assert step_outlier(_view(dummies=2), 4, 3, always) == 0.25
    assert step_outlier(None, 4, 3, always) == 0.0
    assert step_outlier(_view(), 4, 3, {"rate": 0.0, "excess_s": 0.25}) == 0.0


def test_the_draw_is_the_seed_s_and_the_step_s_alone(monkeypatch):
    spec = {"rate": 0.3, "excess_s": 0.2}
    monkeypatch.setenv(STEP_OUTLIERS_SEED_ENV, "1")
    a = [step_outlier(_view(), 4, k, spec) for k in range(400)]
    b = [step_outlier(_view(dummies=1), 2 + 2, k, spec) for k in range(400)]
    assert a == b  # every rank of a group draws the same
    assert 0.2 < sum(1 for x in a if x) / len(a) < 0.4
    monkeypatch.setenv(STEP_OUTLIERS_SEED_ENV, "2")
    assert [step_outlier(_view(), 4, k, spec) for k in range(400)] != a
