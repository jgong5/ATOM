# SPDX-License-Identifier: MIT
"""``COMPASS_PREFILL_PHASE_MAX``: a DP group's prefill step priced as its longest
attention over cached history plus its largest rest (`history_phase_wait`),
not as its slowest rank's own sum. Two ranks run as two threads on a gloo
group (`test_dp_step_max`'s `dp2`). No device.
"""

from types import SimpleNamespace

import pytest
import torch
from test_dp_step_max import dp2, prefill, runner  # noqa: F401  the dp2 fixture

from atom.compass.backends.cost import StepCost
from atom.compass.backends.step_stub import StepStubBackend
from atom.compass.runner import overrides


class Split(StepStubBackend):
    """Prices every step as `history` seconds of history attention and `rest`."""

    def __init__(self, history, rest):
        super().__init__()
        self.history, self.rest = history, rest

    def price(self, batch_view):
        return StepCost(
            [
                self._term("prefill.query_cached", 1, self.history),
                self._term("prefill.tokens", 1, self.rest),
            ]
        )


def _terms(history, rest):
    return [
        SimpleNamespace(name="prefill.query_cached", seconds=history * 0.75),
        SimpleNamespace(name="prefill.history", seconds=history * 0.25),
        SimpleNamespace(name="prefill.tokens", seconds=rest),
        SimpleNamespace(name="draft.prefill.history", seconds=1.0),
    ]


def test_off_unset_or_without_a_group(monkeypatch):
    monkeypatch.delenv("COMPASS_PREFILL_PHASE_MAX", raising=False)
    monkeypatch.setattr(torch.distributed, "all_reduce", None)
    assert overrides.history_phase_wait(_terms(0.1, 0.2), object()) == 0.0
    monkeypatch.setenv("COMPASS_PREFILL_PHASE_MAX", "1")
    assert overrides.history_phase_wait(_terms(0.1, 0.2), None) == 0.0


def test_each_rank_waits_for_the_group_s_longest_history(monkeypatch):
    monkeypatch.setenv("COMPASS_PREFILL_PHASE_MAX", "1")
    ranks = [(0.30, 0.10), (0.05, 0.40), (0.0, 0.20)]  # (history, rest) s
    top = max(h for h, _ in ranks)
    monkeypatch.setattr(
        torch.distributed, "all_reduce", lambda t, op=None, group=None: t.fill_(top)
    )
    priced = [h + r + overrides.history_phase_wait(_terms(h, r), object()) for h, r in ranks]
    assert max(priced) == pytest.approx(top + max(r for _, r in ranks))
    # a refused or dummy rank, no history of its own, waits the whole top; draft terms are not history
    monkeypatch.setenv("COMPASS_PREFILL_PHASE_MAX", "0.5")
    assert overrides.history_phase_wait((), object()) == pytest.approx(0.15)


def test_a_dp_group_prices_longest_history_plus_largest_rest(dp2, monkeypatch):
    ranks = [runner(Split(0.30, 0.10), dp_size=2, dp_rank=0), runner(Split(0.05, 0.40), dp_size=2, dp_rank=1)]
    off = dp2(lambda: ranks[0].forward(prefill(40)), lambda: ranks[1].forward(prefill(8)))
    assert [r.predicted_s for r in off] == pytest.approx([0.45, 0.45])
    monkeypatch.setenv("COMPASS_PREFILL_PHASE_MAX", "1")
    on = dp2(lambda: ranks[0].forward(prefill(40)), lambda: ranks[1].forward(prefill(8)))
    assert [r.predicted_s for r in on] == pytest.approx([0.70, 0.70])
