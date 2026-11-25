# SPDX-License-Identifier: MIT
"""The host and step laws' solver divides each equation by its price under the
fitted coefficients, iterated from the solve that divides it by its label, so
a law prices its samples' mean; its MAD test, and `validate.cost`'s rule for a
slow leading step, read the residual it solves, price less label over price."""

import numpy as np
import pytest
from test_backend_step_linear_fit import HOST, _steps, _timed_record
from test_fit_host import BATCH, DECODE, ENCODE, stamped

from atom.compass import run as compass_run
from atom.compass.backends.accuracy import _leading_slow
from atom.compass.backends.step_linear_fit import (
    _fit_form,
    _host_law,
    _solve,
    fit_host,
    fit_tokenizer,
    label_records,
)


@pytest.mark.parametrize("width", [1, 2], ids=["one rank", "two ranks"])
def test_one_shape_at_once_and_twice_its_price_is_priced_at_their_mean(width):
    # Twenty steps of one shape, half at 10 ms and half at 20 ms. Divided by
    # its label, the solve prices them at 12 ms, sum(1/y) / sum(1/y^2).
    y = np.repeat([0.01, 0.02], 10)
    x = np.zeros((20, width, 2))
    x[..., 0] = 1.0
    x[:, :, 1] = [3.0, 1.0][:width]
    solved = _fit_form(x, np.ones((20, width), dtype=bool), y, ["step", "context"])

    price = (x @ solved["coef"]).max(axis=1)
    assert price == pytest.approx(np.full(20, 0.015), rel=1e-9)
    assert solved["keep"].all()
    # The residual the MAD test reads: price less label, over price.
    assert solved["residual"] == pytest.approx(np.repeat([1 / 3, -1 / 3], 10))


def test_each_encode_shape_of_two_records_is_priced_at_its_calls_mean(tmp_path):
    # Record b's lone one-item encodes take twice record a's, call for call:
    # the encode term prices each call at the mean of the two, 1.5 times a's.
    # Divided by its label, the solve priced each at 1.2 times a's.
    a, b = (tmp_path / d / "steps" for d in "ab")
    for d, enc in ((a, ENCODE), (b, (2 * ENCODE[0], ENCODE[1] / 2))):
        d.mkdir(parents=True)
        stamped(d.parent, rates=(enc, BATCH, DECODE))
    term, report = fit_tokenizer(
        [d / compass_run.TOKENIZER_RECORD for d in (a, b)],
        [d / compass_run.INTAKE_RECORD for d in (a, b)],
    )

    assert term["encode"] == pytest.approx(
        {"fixed_s": 1.5 * ENCODE[0], "tokens_per_s": ENCODE[1] / 1.5}, rel=1e-4
    )
    assert (report["encode"]["samples"], report["encode"]["dropped"]) == (8, 0)


def test_a_step_slow_by_its_price_s_scatter_is_dropped_and_validate_cost_names_it():
    # Thirty steps at 0.95, 1 and 1.05 times a law, the first at 1.3 times:
    # its residual, 1 - 1.3, lies past MAD_K scaled MADs, where 1 / 1.3 - 1,
    # its error over its label, does not. Graded by the scatter the fit
    # recorded, the same step leads its record slow.
    tokens = np.linspace(1000, 30_000, 30)
    x = np.column_stack([np.ones(30), tokens])
    law = x @ [0.01, 1e-6]
    seconds = law * np.r_[1.3, np.resize([0.95, 1.0, 1.05], 29)]
    solved = _solve(x, seconds, ["step", "tokens"])

    assert np.flatnonzero(~solved["keep"]).tolist() == [0]
    price = x @ solved["coef"]
    priced = [("prefill", "", i, s, p) for i, (s, p) in enumerate(zip(seconds, price))]
    fitted = {"prefill": {"fit": {k: solved[k] for k in ("centre", "mad")}}}
    assert _leading_slow(priced, fitted) == {0}


def test_a_step_no_column_prices_is_dropped_and_divides_no_equation():
    # The first step's only feature is zero, so every vector prices it at no
    # time: its residual is minus infinity and the MAD test drops it.
    x = np.arange(5.0)[:, None]
    solved = _solve(x, np.array([1.0, 1.0, 2.0, 3.0, 4.0]), ["tokens"])

    assert solved["keep"].tolist() == [False, True, True, True, True]
    assert solved["coef"] == pytest.approx([1.0], rel=1e-12)


def test_a_host_law_keeps_a_sample_at_twice_its_price_which_the_step_solve_drops():
    # Thirty samples at 0.98, 1 and 1.02 times a law, the first at twice it,
    # as a call after an idle CPU runs. The step solve's MAD test drops it;
    # the host law keeps every sample and prices their mean ratio at 1.
    tokens = np.linspace(1000, 30_000, 30)
    x = np.column_stack([np.ones(30), tokens])
    seconds = x @ [0.01, 1e-6] * np.r_[2.0, np.resize([0.98, 1.0, 1.02], 29)]
    coef, report = _host_law(x, seconds, ["fixed", "tokens"])

    assert (report["samples"], report["dropped"]) == (30, 0)
    assert np.mean(seconds / (x @ coef)) == pytest.approx(1.0, rel=1e-9)
    assert not _solve(x, seconds, ["fixed", "tokens"], kept="fixed")["keep"][0]


def test_the_jit_cut_still_drops_a_compile_the_host_laws_then_leave_out(tmp_path):
    # Output prefill 110's launch stalls on a compile, 20 times its law, and
    # decode 251's runs 6 times its own. The JIT cut's MAD test names both;
    # with them and the forwards after them left out, every host law, which
    # drops no sample, is the planted one.
    scale = {(110, "W_launch"): 20, (251, "W_launch"): 6}
    record, _ = _timed_record(tmp_path / "dp0.jsonl", _steps(14, n=400), scale=scale)
    host, report = fit_host(label_records([record])[0])

    assert list(report["jit_steps"].values()) == [[110, 251]]
    for kind, segments in HOST.items():
        for segment, c in segments.items():
            got = host["laws"][kind][segment]
            assert got == pytest.approx(c, rel=1e-4, abs=1e-12), f"{kind}.{segment}"
            assert report[f"{kind}.{segment}"]["dropped"] == 0
