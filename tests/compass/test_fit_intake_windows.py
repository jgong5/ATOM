# SPDX-License-Identifier: MIT
"""The tokenizer and intake fits on host-clock windows of a record
(`step_linear_fit.within`): each lone call and request is still found among
all of the record's stamps, and only those within a window are fitted; with no
window, or None for a record, the fit is as it was. `test_fit_host.stamped`'s
record: eight lone requests a second apart from 1001 s on, then a burst.
"""

import itertools
import json

import numpy as np
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
    # Their bodies alone, of the eight lone sizes the first six.
    sizes = np.linspace(30_000, 400_000, 8).astype(int).tolist()
    assert term["bytes"] == [sizes[0], sizes[5]] != whole["bytes"]
    _, fitted = fit_tokenizer([calls], [intake], windows=[first])
    # Their one-item encodes: the first, third and fifth request's.
    assert fitted["encode"]["samples"] + fitted["encode"]["dropped"] == 3
    assert fit_tokenizer([calls], [intake], windows=[None]) == fit_tokenizer(
        [calls], [intake]
    )


def _all_but(*ns) -> list:
    """Host-clock windows holding every time but the nanoseconds `ns`."""
    cuts = sorted(ns)
    return list(zip([0, *(n + 1 for n in cuts)], [*cuts, 10**18]))


def test_a_call_within_a_window_is_judged_lone_among_all_of_the_record_s(tmp_path):
    (tmp_path / "steps").mkdir()
    stamped(tmp_path, burst=(40_000, 600_000, 50_000))
    intake = tmp_path / "steps" / compass_run.INTAKE_RECORD
    calls = intake.with_name(compass_run.TOKENIZER_RECORD)
    _, whole = fit_tokenizer([calls], [intake])
    made = [json.loads(line) for line in calls.read_text().splitlines()[1:]]
    # The two decodes at once, 500 ns apart, and the encode_batch the second
    # lone request's segment call made.
    decodes = sorted(c["t_start_ns"] for c in made if c["call"] == "decode")
    (late,) = [b for a, b in itertools.pairwise(decodes) if b - a == 500]
    (inner,) = [
        c["t_start_ns"]
        for c in made
        if c["call"] == "encode_batch" and (c["items"], c["tokens"]) == (3, 5000)
    ]
    windows = [_all_but(late - 500, inner)]
    _, fitted = fit_tokenizer([calls], [intake], windows=windows)
    # The later decode still overlaps the earlier one, which starts outside;
    # the segment call still made the encode_batch, which starts outside and
    # alone leaves the fit.
    assert fitted["decode"] == whole["decode"]
    assert fitted["segment"] == whole["segment"]
    batch = [
        r["encode_batch"]["samples"] + r["encode_batch"]["dropped"]
        for r in (fitted, whole)
    ]
    assert batch[0] == batch[1] - 1
