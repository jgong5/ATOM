# SPDX-License-Identifier: MIT
"""A new request's first forward after a busy engine's step keeps its `E`.

Request 7's prefill and decode, then request 8's first forward, sharing no
request with the decode before it, then a last decode. While request 7 is
still to decode there, the engine went from its pull straight to request 8's
forward: the gap is `E`, labelled and priced by the law like any other. Once
request 7 has finished, the engine waited there for request 8's arrival, and
the gap is cut as idle.
"""

import pytest
from test_parity_recurrence import MS, _labelled, _law, line

from atom.compass.parity import recurrence


def _record(held: bool) -> list[dict]:
    """Request 8 enters 3 ms after the pull that follows request 7's decode."""
    steps = [
        line(0, [[50, 50, False]], True, 0, (0, 0), (1, 2), (2, 2), 3),
        line(1, [[1, 51, True]], True, 4, (4, 4), (5, 6), (6, 14), 15),
        line(2, [[40, 40, False]], True, 19, (19, 19), (20, 21), (21, 25), 26),
        line(3, [[1, 41, True]], True, 27, (27, 27), (28, 29), (29, 37), 38),
    ]
    steps[2]["batch"] = [[8, 40, 40]]
    steps[3]["batch"] = [[8, 1, 41]]
    if held:  # request 7 decodes again beside request 8
        steps[3]["rows"] = [[1, 52, True], [1, 41, True]]
        steps[3]["batch"] = [[7, 1, 52], [8, 1, 41]]
    for s in steps:
        s["t_pull_ns"] = s["t_exit_ns"] + MS
    return steps


@pytest.mark.parametrize("held", [True, False])
def test_a_first_forward_keeps_its_e_only_while_the_engine_holds_a_request(
    tmp_path, held
):
    steps = _record(held)
    samples = {s.step: s for s in _labelled(tmp_path, steps)}

    assert samples[2].idle is not held
    if held:
        assert samples[2].host["E"] == pytest.approx(3e-3, rel=1e-12)
    else:
        assert "E" not in samples[2].host
    assert [s.step for s in samples.values() if "E" not in s.host] == (
        [0] if held else [0, 2]
    )


@pytest.mark.parametrize("held", [True, False])
def test_the_recurrence_prices_a_busy_first_forward_s_e_by_its_law(tmp_path, held):
    steps = _record(held)
    kinds = {"idle": 1, "decode": 2, "prefill": 1} if held else {"idle": 2, "decode": 2}

    assert {k: v[0] for k, v in recurrence.check(steps)["kinds"].items()} == kinds
    report = recurrence.priced(steps, _labelled(tmp_path, steps), _law())
    assert {k: v[0] for k, v in report["kinds"].items()} == kinds
