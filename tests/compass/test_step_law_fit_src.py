# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""A `step_linear_law` key holds ``fit_src``, the sha256 of the law
(`step_linear_fit.fit_src`): two `compass fit steps` of one configuration
whose laws differ, on other records or by changed fit code, publish two
entries into one store, of which the reuse check takes the newer. The reuse
check passes by a store law whose key states none, and the calibration check
does not read an ended calibration's law stating none as one a new
calibration would refit."""

import json

import pytest
from test_advice_calibrating import UNREAD, _fitted
from test_advice_check import advise as states
from test_advice_check import resolved, stubbed  # noqa: F401  the fixture
from test_advice_results import _calibrated
from test_advice_reuse import (  # noqa: F401  fixtures
    advise,
    at_head,
    searched,
    stored,
    verdicts,
)
from test_fit_steps import (  # noqa: F401  planted_roots: autouse
    LINEAR,
    clients,
    fit,
    member,
    planted_roots,
    record,
)
from test_memory_readings import publish_readings

from atom.compass import run as compass_run
from atom.compass.advice import check
from atom.compass.artifacts import ArtifactStore, Kind
from atom.compass.artifacts.store import ENTRY_FILE
from atom.compass.backends import step_linear_fit
from atom.compass.drive import lacking


def refits(tmp_path, monkeypatch, r, cause: str, swapped: bool) -> list[str]:
    """The tokens of two `compass fit steps` of `r`'s configuration into
    `tmp_path`'s store, in the order they ran: on other records, one on record
    a and one on a and b; by changed code, one by this tree's fit and one by a
    fit pricing each prefill coefficient a nanosecond higher; `swapped`, the
    second first."""
    a, b = (
        record(tmp_path / n, seed, key=compass_run.host_key(r.config), trace=n)
        for n, seed in (("a", 19), ("b", 20))
    )
    fitted = step_linear_fit.fit

    def changed(*args, **kwargs):
        law, report = fitted(*args, **kwargs)
        coefficients = law["prefill"]["coefficients"]
        law["prefill"]["coefficients"] = {n: c + 1e-9 for n, c in coefficients.items()}
        return law, report

    second = ((a, b), fitted) if cause == "other records" else ((a,), changed)
    runs = [((a,), fitted), second]
    tokens = []
    for records, code in reversed(runs) if swapped else runs:
        monkeypatch.setattr(step_linear_fit, "fit", code)
        fit(tmp_path, "fit steps", *LINEAR, *records, *clients(*records))
        out = json.loads((tmp_path / "out/step_linear_law.json").read_text())
        tokens.append(out["token"])
    return tokens


@pytest.mark.parametrize("swapped", [False, True], ids=["as named", "swapped"])
@pytest.mark.parametrize("cause", ["other records", "changed code"])
def test_two_fits_whose_laws_differ_are_two_entries_and_the_reuse_check_takes_the_newer(
    tmp_path, monkeypatch, searched, cause, swapped  # noqa: F811
):
    tokens = refits(tmp_path, monkeypatch, searched, cause, swapped)

    store = ArtifactStore(tmp_path / "store")
    keys = [store.key_from_token(Kind.STEP_LINEAR_LAW, t) for t in tokens]
    laws = [member(store.root, k, "params") for k in keys]
    assert laws[0] != laws[1]
    assert [k.values["fit_src"] for k in keys] == list(
        map(step_linear_fit.fit_src, laws)
    )
    found = verdicts(advise(tmp_path), "law")
    assert [v["law"] for v in found] == [
        f"{keys[1].kind} {tokens[1]} in {store.root}"
    ] * 2


def test_a_store_law_whose_key_states_no_fit_src_is_passed_by(
    tmp_path, searched  # noqa: F811
):
    stored(tmp_path, searched)
    for entry in (tmp_path / "store" / Kind.STEP_LINEAR_LAW.value).glob(
        f"*/{ENTRY_FILE}"
    ):
        document = json.loads(entry.read_text())
        del document["key"]["fit_src"]
        entry.write_text(json.dumps(document))

    found = verdicts(advise(tmp_path), "law")

    assert [(v["verdict"], v["requirement"]) for v in found] == [("calibrate", 1)] * 2
    assert "no step linear law at its key" in found[0]["reason"]


def test_an_ended_calibration_whose_law_states_no_fit_src_is_not_refitted(
    tmp_path, stubbed, monkeypatch  # noqa: F811
):
    """As the ended calibration in `test_advice_calibrating` that a new one
    would refit, but for its law's key, which states no ``fit_src``: the
    reuse check passes that law by, so the next calibration goes into
    ``cal/b+2``."""
    monkeypatch.setenv("AIPERF_PYTHON", "/real/python")
    monkeypatch.setenv("KEYED", "ATOM_USE_TRITON_MOE")
    cal = tmp_path / "out" / "cal"
    (first,) = states(tmp_path, {"b": UNREAD})["b"]["made_by"]
    assert _calibrated(tmp_path, first)[0] == 0
    config = resolved(stubbed.config, ["--max-num-seqs", "4"])
    publish_readings(tmp_path / "store", config)
    _fitted(cal / "b", config)
    law = cal / "b" / check.FITTED
    held = json.loads(law.read_text())
    del held["key"]["fit_src"]
    law.write_text(json.dumps(held))

    (command,) = states(tmp_path, {"b": UNREAD})["b"]["made_by"]

    assert f" {lacking.CALIBRATE} {cal / 'b+2'} " in command
