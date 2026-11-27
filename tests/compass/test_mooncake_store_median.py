# SPDX-License-Identifier: MIT
"""COMPASS_FIT_STORE_MEDIAN=load_s,... (#3720): a Mooncake Store law fitted by
least absolute log error, so its median label over price is 1 at every count."""

import json

import numpy as np
import pytest
from test_mooncake_store_fit import CAPACITY, FLAGS, START, copy, logged, lookups

from atom.compass.backends import step_linear_fit as fit
from atom.compass.backends.step_linear_fit import _store_median, offload_main
from atom.compass.kv.mooncake_store_server import check_entry, of_rank


def tailed(fixed, per, counts, tail=(1.0, 1.0, 1.0, 4.0, 9.0)):
    """``(1, count, seconds)`` copies on the law, a share of them a factor over
    it (a copy that waited behind the stage's forwards), as `tail` cycles."""
    return np.array(
        [(1, n, (fixed + per * n) * tail[i % len(tail)]) for i, n in enumerate(counts)]
    )


def test_a_tailed_copy_is_priced_at_its_median_at_every_count():
    rows = tailed(0.02, 2e-4, [8, 40, 120, 300, 700, 1200] * 10)
    mean_law = [0.08, 8e-4]  # what the solver's relative least squares prices
    law, fitted = _store_median(rows, mean_law, {"centre": 0.4, "mad": 0.6})
    assert law == pytest.approx([0.02, 2e-4])
    assert fitted["centre"] == pytest.approx(0.0)
    centred = fitted["median_centred"]
    assert centred["solver"] == mean_law and centred["solver_centre"] == 0.4
    assert centred["copies"] == len(rows) and centred["form"] == "least absolute log error"
    count, label = rows[:, 1], rows[:, 2]
    for n in np.unique(count):  # 1 at each count, not only overall
        assert np.median(label[count == n] / (law[0] + law[1] * n)) == pytest.approx(1.0)


def test_a_copy_of_no_positive_label_is_left_out_and_two_are_needed():
    counts = [10, 20, 30, 40, 50] * 3  # each count once a factor over
    rows = np.vstack([tailed(0.01, 1e-4, counts, tail=(1.0, 1.0, 4.0)), [(1, 25, 0.0)]])
    law, fitted = _store_median(rows, [0.0, 0.0], {"centre": 0.0})
    assert law == pytest.approx([0.01, 1e-4]) and fitted["median_centred"]["copies"] == 15
    with pytest.raises(ValueError, match="1 copies"):
        _store_median([(1, 10, 0.5), (1, 20, 0.0)], [0.0, 0.0], {"centre": 0.0})


def test_pairs_come_from_a_thinned_set_but_every_copy_is_scored(monkeypatch):
    monkeypatch.setattr(fit, "STORE_MEDIAN_PAIRS", 12)
    rows = tailed(0.03, 5e-4, list(range(1, 301)))
    law, fitted = _store_median(rows, [0.1, 1e-3], {"centre": 0.5})
    assert law == pytest.approx([0.03, 5e-4])
    assert fitted["median_centred"]["copies"] == 300


def pp4_log(tmp_path, monkeypatch, caplog):
    """A PP4 prefill's Store log: per-rank chunk bytes, each rank's loads on
    ``(1 + r) ms + 1.5 ms`` a chunk with a one-sided tail, saves on a line."""
    chunk = [3 << 20, 3 << 20, 3 << 20, 5 << 20]
    lines = [
        logged(
            "Mooncake Store offload worker",
            START | {"rank": r, "world": 4, "chunk_bytes": chunk[r]},
        )
        for r in range(4)
    ]
    for r in range(4):
        for i, n in enumerate([1, 3, 6, 10, 20, 40] * 5):
            ms = {"total_ms": 1.0 + 2.0 * n, "pack_ms": 1, "put_ms": 1}
            lines.append(copy("[OFFLOAD-SAVE-PROF]", n, rank=r, **ms))
            tail = (1.0, 1.0, 1.0, 6.0, 20.0)[i % 5]
            ms = {"total_ms": ((1.0 + r) + 1.5 * n) * tail, "get_ms": 1, "unpack_ms": 1}
            lines.append(copy("[OFFLOAD-LOAD-PROF]", n, rank=r, **ms))
    lines += lookups(monkeypatch, caplog, range(1, 4))
    log = tmp_path / "p.log"
    log.write_text("\n".join(lines) + "\n")
    return log


def fitted_entry(log, tmp_path, name):
    out = tmp_path / name
    argv = [str(log), "--out", str(out), "--master-flags", FLAGS]
    offload_main([*argv, "--capacity-bytes", str(CAPACITY)])
    return json.loads(out.read_text())


def test_the_switch_centres_each_rank_s_load_law_and_leaves_the_rest(
    tmp_path, monkeypatch, caplog
):
    log = pp4_log(tmp_path, monkeypatch, caplog)
    monkeypatch.delenv(fit.STORE_MEDIAN_ENV, raising=False)
    off = fitted_entry(log, tmp_path, "off.json")
    monkeypatch.setenv(fit.STORE_MEDIAN_ENV, "load_s")
    on = fitted_entry(log, tmp_path, "on.json")
    check_entry(on)
    for r in range(4):
        assert of_rank(on, "load_s", r) == pytest.approx([(1.0 + r) * 1e-3, 1.5e-3])
        # The solver prices the tail into the law; the report keeps its law.
        solver = of_rank(off, "load_s", r)
        assert solver[0] + 10 * solver[1] > 1.2 * ((1.0 + r) + 15.0) * 1e-3
        kept = on["fit"]["load_s"]["ranks"][r]["median_centred"]
        assert kept["solver"] == pytest.approx(solver)
    # Laws it does not list, and every other field, are the unset run's.
    assert {k: v for k, v in on.items() if k not in ("load_s", "fit")} == {
        k: v for k, v in off.items() if k not in ("load_s", "fit")
    }
    assert on["fit"]["store_s"] == off["fit"]["store_s"]
    assert "median_centred" not in json.dumps(off)


def test_the_switch_refuses_a_name_that_is_not_a_store_law(
    tmp_path, monkeypatch, caplog
):
    log = pp4_log(tmp_path, monkeypatch, caplog)
    monkeypatch.setenv(fit.STORE_MEDIAN_ENV, "load_s,save_s")
    with pytest.raises(ValueError, match=r"names \['save_s'\], not a Store law"):
        fitted_entry(log, tmp_path, "bad.json")
