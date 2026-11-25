# SPDX-License-Identifier: MIT
"""`compass advice`'s check of two candidates whose laws share one key, as the
DeepSeek-V4-Pro template's S2 and S5, whose LMCache no law key holds: the
first prints its `calibrate.sh` command, and the other names that calibration
and the command measuring its readings into the store it publishes into,
unless they are at their key there; into the calibration's own store, only
once it has started. A candidate refused at some point prints no command, so
the next at its key prints the calibration. Once the first's calibration has
ended where a new one would refit its law, the other says so too, with the
command measuring its readings into the store that law was published into.
Where the shared store holds a stale entry at the other's readings key, which
it refuses a second publish at, the first calibrates into its own store, unless
the other is refused, as it then prints no readings command; a 1P1D first does
too. A first that has started there, or ended, keeps that store, so the other
calibrates its own law into its own store.

The configuration, its stubbed resolution and starts are `test_advice_check`'s.
"""

import pytest
from test_advice_calibrating import (
    PAIR,
    _calibrate,
    _env,
    _fitted,
    _readings_at_another_digest,
    sides,  # noqa: F401  the fixture
)
from test_advice_check import (
    NETWORK,
    advise,
    resolved,
    stubbed,  # noqa: F401  the fixture
)
from test_advice_results import _calibrated
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


@pytest.mark.parametrize(
    "held, at_key, started",
    [
        (False, False, False),
        (False, True, False),
        (True, False, False),
        (True, True, False),
        (True, True, True),
    ],
    ids=[
        "into the shared store",
        "S5's readings there",
        "into S2's own, not started",
        "into S2's own, not started, S5's readings in the shared",
        "into S2's own, started, S5's readings in the shared",
    ],
)
def test_two_candidates_at_one_law_key_print_one_calibration(
    tmp_path, stubbed, capsys, held, at_key, started  # noqa: F811
):
    store, cal = tmp_path / "store", tmp_path / "out" / "cal" / "S2"
    if held:
        # The store refuses S2's readings again: it calibrates into its own.
        _readings_at_another_digest(store, resolved(stubbed.config, ARGS2))
    if at_key:
        publish_readings(store, resolved(stubbed.config, ARGS5))
    if started:
        cal.mkdir(parents=True)
        (cal / "inputs").write_text("tree c0ffee\nrecord 1\n")

    found = advise(tmp_path, {"S2": S2, "S5": S5})

    assert found["S2"]["state"] == found["S5"]["state"] == "missing a law"
    (calibration,) = _calibrations(found)
    assert f"calibrate.sh {cal} " in calibration
    into = str(cal / "store" if held else store)
    assert _env(found, "S2").get("STORE", str(cal / "store")) == into
    measure = f"compass measure memory --gpu G --out D --store {into} -- --model "
    measure += " ".join([str(tmp_path), *ARGS5])
    made = [check.BORROWED.format("S2", cal, into)]
    if held or not at_key:
        # A measure into S2's own store before it starts makes it refuse.
        made += [
            check.PENDING.format(into, "S2", cal) if held and not started else measure
        ]
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


def test_a_calibration_taken_again_is_named_by_its_candidate_and_its_directory(
    tmp_path, stubbed  # noqa: F811
):
    """S2's calibration in ``cal/S2`` has ended, and the store refuses its
    readings again, so its next goes into ``cal/S2+2``, publishing into its own
    store: S5 names S2's calibration there, and its readings once it starts."""
    store, cal = tmp_path / "store", tmp_path / "out" / "cal"
    _readings_at_another_digest(store, resolved(stubbed.config, ARGS2))
    (cal / "S2").mkdir(parents=True)
    ended = "2026-11-24T07:00:00Z end calibration 0.20 node-hours\n"
    (cal / "S2" / "status").write_text(ended)

    found = advise(tmp_path, {"S2": S2, "S5": S5})

    again = cal / "S2+2"
    (calibration,) = _calibrations(found)
    assert f"calibrate.sh {again} " in calibration
    into = str(again / "store")
    assert found["S5"]["made_by"] == [
        check.BORROWED.format("S2", again, into),
        check.PENDING.format(into, "S2", again),
    ]


def test_a_candidate_at_the_key_of_a_calibration_that_refits_names_it_not_its_own(
    tmp_path, stubbed, monkeypatch  # noqa: F811
):
    """S2's calibration into its own store, as the store refuses its readings
    again, run by `calibrate.sh` in `test_calibrate`'s stub tree, ends, its law
    keyed at S2's key as a fit keys it, so a new one would refit that law: S5,
    at that key, says so as S2 does, with the command measuring its readings
    into S2's store, and prints no calibration of its own."""
    monkeypatch.setenv("AIPERF_PYTHON", "/real/python")
    store, cal = tmp_path / "store", tmp_path / "out" / "cal" / "S2"
    _readings_at_another_digest(store, resolved(stubbed.config, ARGS2))
    (first,) = _calibrations(advise(tmp_path, {"S2": S2, "S5": S5}))
    assert _calibrated(tmp_path, first)[0] == 0
    _fitted(cal, resolved(stubbed.config, ARGS2))

    found = advise(tmp_path, {"S2": S2, "S5": S5})

    into = str(cal / "store")
    refits = check.REFITS.format(cal, into)
    assert found["S2"]["made_by"] == [refits]
    measure = f"compass measure memory --gpu G --out D --store {into} -- --model "
    measure += " ".join([str(tmp_path), *ARGS5])
    assert found["S5"]["made_by"] == [refits, measure]


def test_a_borrower_s_stale_readings_send_its_lender_into_its_own_store(
    tmp_path, stubbed, capsys  # noqa: F811
):
    """The shared store holds S5's readings measured at another ATOM source,
    and nothing of S2's: S2's calibration publishes into its own store, as the
    shared one refuses S5's readings again, and S5's readings command measures
    them there once S2's has started, as its inputs then name no store."""
    store, cal = tmp_path / "store", tmp_path / "out" / "cal" / "S2"
    _readings_at_another_digest(store, resolved(stubbed.config, ARGS5))
    into = str(cal / "store")
    borrowed = check.BORROWED.format("S2", cal, into)
    measure = f"compass measure memory --gpu G --out D --store {into} -- --model "
    measure += " ".join([str(tmp_path), *ARGS5])

    for readings in (check.PENDING.format(into, "S2", cal), measure):
        found = advise(tmp_path, {"S2": S2, "S5": S5})

        (calibration,) = _calibrations(found)
        assert f"calibrate.sh {cal} " in calibration
        assert "STORE" not in _env(found, "S2")
        assert found["S5"]["made_by"] == [borrowed, readings]
        printed = capsys.readouterr().out
        assert f"    made by: {calibration}\n" in printed
        assert f"    made by: {borrowed}\n    made by: {readings}\n" in printed
        # S2's calibration starts, publishing into its own store.
        cal.mkdir(parents=True, exist_ok=True)
        (cal / "inputs").write_text("tree c0ffee\nrecord 1\n")


def test_a_refused_borrower_s_stale_readings_leave_its_lender_in_the_shared_store(
    tmp_path, stubbed, monkeypatch  # noqa: F811
):
    """R, at S2's law key and S5's readings key at concurrency 1 and refused at
    4, prints no readings command, so its stale readings in the shared store
    keep S2's calibration there."""

    def refusing(atom_args):
        if atom_args[-2:] == ["--decode-context-parallel-size", "4"]:
            raise AssertionError("tp_size (1) must be divisible by dcp_size (4)")
        return resolved(stubbed.config, atom_args)

    resolved_here(monkeypatch, refusing)
    store = tmp_path / "store"
    _readings_at_another_digest(store, resolved(stubbed.config, ARGS5))
    refused = S5 | {"per_concurrency": {"--decode-context-parallel-size": 1}}

    found = advise(tmp_path, {"S2": S2, "R": refused})

    assert found["R"]["state"] == "refused"
    assert _env(found, "S2")["STORE"] == str(store)


def test_a_1p1d_lender_s_borrower_s_stale_readings_send_it_into_its_own_store(
    tmp_path, sides  # noqa: F811
):
    """S, a lone server at A's prefill side's law key, borrows that side's law,
    and the shared store holds S's readings stale, and nothing of A's: A fits
    both sides into its own store, and S's readings wait on A's start."""
    store, cal = tmp_path / "store", tmp_path / "out" / "cal" / "A"
    s = {"args": {"--max-num-seqs": 8, "--max-model-len": 512}}
    args = ["--max-num-seqs", "8", "--max-model-len", "512"]
    _readings_at_another_digest(store, resolved(sides, args))

    found = advise(tmp_path, {"A": PAIR["A"], "S": s}, network=NETWORK)

    assert "STORE" not in _calibrate(found, "A")[0]
    into = str(cal / "store")
    assert found["S"]["made_by"] == [
        check.BORROWED.format("A", cal, into),
        check.PENDING.format(into, "A", cal),
    ]


@pytest.mark.parametrize("ended", [False, True], ids=["started", "ended, refitting"])
def test_a_borrower_s_stale_readings_in_its_started_lender_s_store_calibrate_its_own(
    tmp_path, stubbed, monkeypatch, capsys, ended  # noqa: F811
):
    """S2's calibration has started into the shared store, named through a
    link, or, run there by `calibrate.sh` in `test_calibrate`'s stub tree, has
    ended, so a new one would refit its law; the shared store holds S5's
    readings measured at another ATOM source, which it refuses again, and S2's
    keeps that store, as its inputs name it, resolved: S5 prints its own
    calibrate.sh command, into its own store, and no readings command."""
    monkeypatch.setenv("AIPERF_PYTHON", "/real/python")
    store, cal = tmp_path / "store", tmp_path / "out" / "cal"
    (tmp_path / "shared").mkdir()
    store.symlink_to(tmp_path / "shared")
    if ended:
        (first,) = _calibrations(advise(tmp_path, {"S2": S2, "S5": S5}))
        assert _calibrated(tmp_path, first)[0] == 0
        _fitted(cal / "S2", resolved(stubbed.config, ARGS2))
    else:
        (cal / "S2").mkdir(parents=True)
        inputs = f"tree c0ffee\nstore {store.resolve()}\nrecord 1\n"
        (cal / "S2" / "inputs").write_text(inputs)
    _readings_at_another_digest(store, resolved(stubbed.config, ARGS5))
    capsys.readouterr()

    found = advise(tmp_path, {"S2": S2, "S5": S5})

    if ended:
        refits = check.REFITS.format(cal / "S2", store.resolve())
        assert found["S2"]["made_by"] == [refits]
    else:
        assert _env(found, "S2")["STORE"] == str(store.resolve())
    measure = "compass measure memory "
    assert not [c for c in found["S5"]["made_by"] if measure in c]
    env, given = _calibrate(found, "S5")
    assert "STORE" not in env and given[0] == str(cal / "S5")
    printed = capsys.readouterr().out
    assert measure not in printed
    assert "".join(f"    made by: {c}\n" for c in found["S5"]["made_by"]) in printed
