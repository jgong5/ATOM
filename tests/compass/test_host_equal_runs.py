# SPDX-License-Identifier: MIT
"""COMPASS_FIT_HOST_EQUAL_RUNS (#734): listed host segments fitted with each run weighing the same in all, a weighted
Gamma fit (`_fit_form`'s `weight`), every other segment as before."""

import json

import numpy as np
import pytest
from test_backend_step_labels import _pair

from atom.compass.backends.step_linear_fit import (
    HOST_EQUAL_RUNS_ENV,
    _host_law,
    equal_run_weights,
    fit_host,
    host_equal_runs,
    label_records,
)


def test_each_run_weighs_the_same_and_the_weights_average_one():
    w = equal_run_weights(["a", "a", "a", "b"])
    assert w.tolist() == pytest.approx([2 / 3, 2 / 3, 2 / 3, 2.0])
    assert w.mean() == pytest.approx(1.0)
    assert equal_run_weights([("r", 0), ("r", 0)]).tolist() == [1.0, 1.0]


def test_the_switch_lists_kind_segments(monkeypatch):
    monkeypatch.delenv(HOST_EQUAL_RUNS_ENV, raising=False)
    assert host_equal_runs() == frozenset()
    monkeypatch.setenv(HOST_EQUAL_RUNS_ENV, "prefill.E,prefill.E_out")
    assert host_equal_runs() == {"prefill.E", "prefill.E_out"}
    monkeypatch.setenv(HOST_EQUAL_RUNS_ENV, "E")
    with pytest.raises(ValueError, match="kind.segment"):
        host_equal_runs()


def test_a_weight_counts_as_that_many_copies_and_unit_weights_change_nothing():
    rng = np.random.default_rng(7)
    x = np.column_stack([np.ones(40), rng.integers(100, 4000, 40), rng.integers(1, 9, 40)]).astype(float)
    y = x @ [1e-3, 2e-7, 4e-4] * rng.uniform(0.8, 1.25, 40)
    names = ("step", "tokens", "seqs")
    plain = _host_law(x, y, names)[0]
    assert _host_law(x, y, names, weight=np.ones(40))[0] == plain
    w = np.ones(40)
    w[:10] = 3.0
    copies = _host_law(np.vstack([x, x[:10], x[:10]]), np.concatenate([y, y[:10], y[:10]]), names)[0]
    assert _host_law(x, y, names, weight=w)[0] == pytest.approx(copies, rel=1e-6, abs=1e-12)
    assert _host_law(x, y, names, weight=w)[0] != pytest.approx(plain, rel=1e-6)


def test_fit_host_weighs_only_the_listed_segment(tmp_path, monkeypatch):
    records = _pair(tmp_path / "a", ((3,), (2,)))[0] + _pair(tmp_path / "b", ((2,), (3,)))[0]
    for r in records[2:]:  # another run: its records' bytes, so its name, differ
        lines = [json.loads(x) | {"run_b": 1} for x in r.read_text().splitlines()]
        r.write_text("".join(json.dumps(x) + "\n" for x in lines))
    samples = label_records(records, 2)[0]
    keep = {samples[0].provenance["run"]}
    # Two runs of unequal size: the first run's samples, then the second's last few.
    other = [s for s in samples if s.provenance["run"] not in keep][-12:]
    samples = [s for s in samples if s.provenance["run"] in keep] + other
    monkeypatch.delenv(HOST_EQUAL_RUNS_ENV, raising=False)
    off, _ = fit_host(samples)
    monkeypatch.setenv(HOST_EQUAL_RUNS_ENV, "prefill.W_pre")
    on, report = fit_host(samples)
    assert report["prefill.W_pre"]["equal_runs"] == 2
    for kind, laws in off["laws"].items():
        for seg, coef in laws.items():
            if (kind, seg) != ("prefill", "W_pre"):
                assert on["laws"][kind][seg] == coef
