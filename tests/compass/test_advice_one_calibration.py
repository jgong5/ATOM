# SPDX-License-Identifier: MIT
"""`compass advice`'s check of two candidates whose laws share one key, as the
DeepSeek-V4-Pro template's S2 and S5, whose LMCache no law key holds: the
first prints its `calibrate.sh` command, and the other names that calibration
and the command measuring its readings into the store it publishes into,
unless they are at their key there. A candidate refused at some point prints
no command, so the next at its key prints the calibration.

The configuration, its stubbed resolution and starts are `test_advice_check`'s.
"""

import pytest
from test_advice_calibrating import _env, _readings_at_another_digest
from test_advice_check import (
    advise,
    resolved,
    stubbed,  # noqa: F401  the fixture
)
from test_advice_reuse import resolved_here
from test_memory_readings import publish_readings

from atom.compass.advice import check
from atom.compass.drive import lacking

#: A candidate with no law, and one at its law key: the --max-model-len the
#: stubbed resolution reads keys its readings, and no law; and their arguments.
S2, ARGS2 = {"args": {"--max-num-seqs": 12}}, ["--max-num-seqs", "12"]
S5 = {"args": S2["args"] | {"--max-model-len": 512}}
ARGS5 = [*ARGS2, "--max-model-len", "512"]


def _calibrations(found: dict) -> list[str]:
    """Each `calibrate.sh` command the candidates' states print."""
    return [c for s in found.values() for c in s["made_by"] if "calibrate.sh " in c]


@pytest.mark.parametrize("readings", ["none", "S2's held", "S5's at their key"])
def test_two_candidates_at_one_law_key_print_one_calibration(
    tmp_path, stubbed, capsys, readings  # noqa: F811
):
    store, cal = tmp_path / "store", tmp_path / "out" / "cal" / "S2"
    if readings == "S2's held":
        # The store refuses S2's readings again: it calibrates into its own.
        _readings_at_another_digest(store, resolved(stubbed.config, ARGS2))
    if readings == "S5's at their key":
        publish_readings(store, resolved(stubbed.config, ARGS5))

    found = advise(tmp_path, {"S2": S2, "S5": S5})

    assert found["S2"]["state"] == found["S5"]["state"] == "missing a law"
    (calibration,) = _calibrations(found)
    assert f"calibrate.sh {cal} " in calibration
    into = _env(found, "S2").get("STORE", str(cal / "store"))
    measure = f"compass measure memory --gpu G --out D --store {into} -- --model "
    measure += " ".join([str(tmp_path), *ARGS5])
    made = [check.BORROWED.format("S2", into)]
    made += [] if readings == "S5's at their key" else [measure]
    assert found["S5"]["made_by"] == made
    printed = capsys.readouterr().out
    # Once in S2's state, once in its rows of the results.
    assert printed.count(f"{lacking.CALIBRATE} ") == 2
    assert "".join(f"    made by: {c}\n" for c in made) in printed


def test_a_candidate_refused_at_a_point_leaves_the_calibration_to_the_next(
    tmp_path, stubbed, monkeypatch  # noqa: F811
):
    def refusing(atom_args):
        if atom_args[-2:] == ["--decode-context-parallel-size", "4"]:
            raise AssertionError("tp_size (1) must be divisible by dcp_size (4)")
        return resolved(stubbed.config, atom_args)

    resolved_here(monkeypatch, refusing)
    # Refused at concurrency 4; at 1, at S2's law key.
    refused = S2 | {"per_concurrency": {"--decode-context-parallel-size": 1}}

    found = advise(tmp_path, {"R": refused, "S2": S2})

    assert found["R"]["state"] == "refused"
    (calibration,) = _calibrations(found)
    assert f"calibrate.sh {tmp_path / 'out' / 'cal' / 'S2'} " in calibration
