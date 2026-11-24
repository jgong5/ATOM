# SPDX-License-Identifier: MIT
"""The tokenizer and intake fits on host-clock windows of a record
(`step_linear_fit.within`): each lone call and request is still found among
all of the record's stamps, and only those within a window are fitted; with no
window, or None for a record, the fit is as it was. `test_fit_host.stamped`'s
record: eight lone requests a second apart from 1001 s on, then a burst.
"""

import pytest
from test_fit_host import stamped

from atom.compass import run as compass_run
from atom.compass.backends.step_linear_fit import fit_intake, fit_tokenizer


def test_the_intake_and_tokenizer_terms_read_only_the_windows_given(tmp_path):
    (tmp_path / "steps").mkdir()
    stamped(tmp_path, burst=(40_000, 600_000, 50_000))
    intake = tmp_path / "steps" / compass_run.INTAKE_RECORD
    calls = intake.with_name(compass_run.TOKENIZER_RECORD)
    whole, report = fit_intake([intake])
    assert fit_intake([intake], windows=[None]) == (whole, report)
    # The first six lone requests' app calls, a second apart from 1001 s on,
    # before the burst's.
    first = [(10**12, 10**12 + 65 * 10**8)]
    term, part = fit_intake([intake], windows=[first])
    assert part["samples"] == 6 and term["fixed_s"] == term["idle_s"]
    assert term["per_byte_s"] == pytest.approx(whole["per_byte_s"], rel=1e-6)
    _, fitted = fit_tokenizer([calls], [intake], windows=[first])
    # Their one-item encodes: the first, third and fifth request's.
    assert fitted["encode"]["samples"] + fitted["encode"]["dropped"] == 3
    assert fit_tokenizer([calls], [intake], windows=[None]) == fit_tokenizer(
        [calls], [intake]
    )
