# SPDX-License-Identifier: MIT
"""A decode right after an output prefill: labelled, its excess fitted and charged.

The output prefill's host work evicts the worker's input-prep data from the
core's cache, so the decode after it runs its host segments longer, more after
a larger prefill. `step_labels.label` gives such a decode the prefill's view
(`Sample.after`); `fit_host` fits each decode segment's law without those
decodes and `after_prefill` on the prefill's `(1, tokens, seqs)` over what they
spend beyond it; `HostDeviceRecurrence` adds it to the decode it charges right
after an output prefill, in a run and in the recurrence's priced grade.
"""

import dataclasses
import json
from types import SimpleNamespace

import numpy as np
import pytest
from test_fit_steps import DIGEST
from test_memory_readings import META_8B
from test_parity_recurrence import _labelled, _law, line, record
from test_run_step_grid import deployment, publish, run_file

from atom.compass import run as compass_run
from atom.compass.backends.shape import BatchView, RequestShape
from atom.compass.backends.step_labels import Sample
from atom.compass.backends.step_linear_fit import fit_host, host_held_out
from atom.compass.parity import recurrence
from atom.compass.runner.overrides import (
    FORWARD_KINDS,
    HOST_SEGMENTS,
    MTP_SEGMENT,
    HostDeviceRecurrence,
)

#: A decode's segments on its own `(1, tokens, seqs)`; under MTP 6 tokens a row.
LAW = {"E": [2.3e-4, 1e-6, 0], MTP_SEGMENT: [4e-4, 5e-7, 0]}
#: What a decode right after an output prefill spends beyond `LAW`, on the
#: prefill's `(1, tokens, seqs)`.
TERM = {"E": [5e-5, 0, 2e-5], MTP_SEGMENT: [1.6e-4, 2e-8, 0]}
OTHER = 1e-4  # every other segment, with no excess


def _prefill(tokens, seqs):
    rows = tuple(
        RequestShape(tokens // seqs, tokens // seqs, False) for _ in range(seqs)
    )
    return BatchView(rows, produces_output=True)


def _cell(rng, plain=300, after=120, run="a", term=True):
    """`plain` decodes after a decode and `after` decodes right after output
    prefills of 200 to 16000 tokens over 1 to 8 rows, which pay `TERM` with
    `term`; each segment within 1%."""
    out = []
    for i in range(plain + after):
        seqs = int(rng.integers(1, 9))
        x = (1, 6 * seqs, seqs)
        view = BatchView(
            tuple(RequestShape(6, 600, True) for _ in range(seqs)), produces_output=True
        )
        before = None
        if i >= plain:
            before = _prefill(int(rng.integers(200, 16000)), int(rng.integers(1, 9)))
        p = before and (
            1,
            sum(r.query_tokens for r in before.requests),
            len(before.requests),
        )
        host = {}
        for g in (*HOST_SEGMENTS, MTP_SEGMENT):
            s = np.dot(LAW[g], x) if g in LAW else OTHER
            s += np.dot(TERM[g], p) if term and before and g in TERM else 0.0
            host[g] = s * rng.uniform(0.99, 1.01)
        provenance = {"source": "record", "run": run, "requests": [i], "config": {}}
        sample = Sample(i, 0, view, 0.01, None, host, False, provenance)
        out.append(dataclasses.replace(sample, after=before))
    return out


def test_a_decode_right_after_an_output_prefill_carries_that_prefill_s_view(tmp_path):
    """`record()` runs two middle chunks, an output prefill, then two decodes;
    a DP dummy forward right after a prefill leaves the decode after it nothing."""
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    samples = _labelled(tmp_path / "a", record())
    assert [s.after for s in samples] == [None, None, None, samples[2].view, None]

    steps = [
        line(0, [[50, 50, False]], True, 0, (0, 0), (1, 2), (2, 2), 3),
        line(1, [[1, 1, True]], False, 4, (4, 4), (5, 6), None, 7),
        line(2, [[1, 51, True]], True, 15, (15, 15), (16, 17), (17, 24), 25),
    ]
    steps[1] |= {"is_dummy_run": True, "batch": []}
    samples = _labelled(tmp_path / "b", steps)
    assert [(s.step, s.after) for s in samples] == [(0, None), (2, None)]


def test_a_decode_law_leaves_out_the_decodes_after_an_output_prefill_and_prices_them():
    """Half the decodes follow an output prefill: too many for the MAD drops
    to keep them out of a decode law fitted on every decode."""
    host, report = fit_host(_cell(np.random.default_rng(2218), after=300))
    laws, terms = host["laws"]["decode"], host["after_prefill"]
    x, p = (1, 24, 4), (1, 8000, 4)
    for g, law in LAW.items():
        assert np.dot(laws[g], x) == pytest.approx(np.dot(law, x), rel=0.01)
        assert np.dot(terms[g], p) == pytest.approx(np.dot(TERM[g], p), rel=0.05)
        assert report[f"after_prefill.{g}"]["samples"] >= 285
    for g in set(HOST_SEGMENTS) - set(LAW):
        assert np.dot(laws[g], x) == pytest.approx(OTHER, rel=0.01)
        assert np.dot(terms[g], p) < 0.03 * OTHER
    assert not [k for k in report["unfitted"] if k.startswith("after_prefill")]
    compass_run._check_host("run.json", host)


def test_too_few_decodes_after_an_output_prefill_stay_in_the_decode_law():
    """Three of them, costing what the others do, fit no term, so each
    decode law keeps them as it did before `after_prefill` existed, and the
    report says why."""
    cell = _cell(np.random.default_rng(1), after=3, term=False)
    host, report = fit_host(cell)
    assert "after_prefill" not in host
    assert report["unfitted"][f"after_prefill.{MTP_SEGMENT}"] == (
        "3 forwards for 3 coefficients"
    )
    unmarked = [dataclasses.replace(s, after=None) for s in cell]
    assert host["laws"] == fit_host(unmarked)[0]["laws"]


def test_the_held_out_report_prices_a_decode_after_an_output_prefill_with_its_term():
    rng = np.random.default_rng(7)
    errors = host_held_out(_cell(rng) + _cell(rng, run="b"))
    after = errors[f"host after_prefill {MTP_SEGMENT}"]
    assert len(after) == 240
    assert np.median(np.abs(after)) < 0.02
    assert len(errors[f"host decode {MTP_SEGMENT}"]) == 600


def _batch(kind, tokens, seqs):
    return SimpleNamespace(
        total_seqs_num_prefill=seqs * (kind == "prefill"),
        total_tokens_num=tokens,
        total_seqs_num=seqs,
        produces_output=lambda: kind != "middle",
    )


def test_a_run_charges_the_term_on_the_decode_right_after_an_output_prefill():
    """Only the decode charged right after the 8000-token prefill of two rows
    pays the term, on that prefill's tokens and rows; an entry without
    `after_prefill` charges it none."""
    laws = {k: {g: [1e-4, 0, 0] for g in HOST_SEGMENTS} for k in FORWARD_KINDS}
    after = {"E": [5e-5, 0, 2e-5], "W_pre": [0, 1e-8, 0]}
    forwards = [
        _batch("prefill", 8000, 2),
        _batch("decode", 12, 2),
        _batch("decode", 12, 2),
        _batch("middle", 4000, 1),
        _batch("decode", 6, 1),
        _batch("prefill", 300, 1),
        _batch("prefill", 300, 1),
    ]

    def charged(after):
        host = HostDeviceRecurrence(laws, (), lambda: 0.0, after_prefill=after)
        out = []
        for batch in forwards:
            host.charge(batch, 0.01, 0.0)
            out += [host.segments["E"], host.segments["W_pre"]]
        return out

    plain = [1e-4] * 2 * len(forwards)
    paid = plain[:2] + [1e-4 + 5e-5 + 2 * 2e-5, 1e-4 + 8000e-8] + plain[4:]
    assert charged(after) == pytest.approx(paid)
    assert charged(None) == pytest.approx(plain)


def test_the_recurrence_grade_charges_the_term_on_the_decode_after_the_prefill(
    tmp_path,
):
    """`_law()` prices `record()` as it ran. A term of 20 us a prefill token
    adds 1 ms to step 3's `W_post`, after the 50-token prefill: step 3 exits
    1 ms late, and step 4, held by step 3's device, exits as measured."""
    steps = record()
    samples = _labelled(tmp_path, steps)
    law = _law()
    assert recurrence.priced(steps, samples, law)["failures"] == []
    law["host"]["after_prefill"] = {"W_post": [0, 2e-5, 0]}
    report = recurrence.priced(steps, samples, law)
    residuals = {r[0]: r[3] for r in report["failures"]}
    assert residuals == pytest.approx({3: 1.0, 4: -1.0})


def test_a_run_file_s_after_prefill_reaches_the_runner_and_a_malformed_one_refuses(
    tmp_path, monkeypatch
):
    r = deployment(tmp_path)
    publish(tmp_path, r)
    laws = {k: {g: [1e-4, 0, 0] for g in HOST_SEGMENTS} for k in FORWARD_KINDS}
    key = {**compass_run.host_key(r.config), "machine_spec": DIGEST}
    host = {"key": key, "laws": laws, "after_prefill": {"E": [2e-4, 0, 0]}}
    run_file(tmp_path, monkeypatch, host=host)
    monkeypatch.setenv(compass_run.NOW_ENV, str(tmp_path / "now.bin"))
    compass_run._PublishedClock(
        compass_run.ENGINE,
        compass_run.channel_table(json.loads((tmp_path / "run.json").read_text())),
        None,
    )
    compass_run.runner(r, META_8B)
    r.compass_host.charge(_batch("prefill", 8000, 1), 0.01, 0.0)
    r.compass_host.charge(_batch("decode", 1, 1), 0.01, 0.0)
    assert r.compass_host.segments["E"] == pytest.approx(3e-4)

    for bad in ({"W_none": [2e-4, 0, 0]}, {"E": [2e-4, 0]}):
        run_file(tmp_path, monkeypatch, host=host | {"after_prefill": bad})
        with pytest.raises(ValueError, match="after_prefill, segments to three"):
            compass_run.spec()
