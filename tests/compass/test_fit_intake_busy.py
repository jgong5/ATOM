# SPDX-License-Identifier: MIT
"""The ``http_intake`` term's ``fixed_s`` from the API loop's busy periods
(`step_linear_fit.fit_intake`'s ``busy``, `step_linear_fit._busy`): a chat
prefill server's burst, its templates rendered and its pool jobs run off the
loop, after `test_fit_host.stamped`'s eight lone requests. Without ``busy``
the term is as it was.
"""

import json

import pytest
from test_fit_host import _stamp, host_laws_key, stamped
from test_fit_steps import clients, fit, member, planted_roots, record  # noqa: F401
from test_run_step_grid import deployment, publish

from atom.compass import run as compass_run
from atom.compass.backends.step_linear_fit import _busy, fit_intake
from atom.compass.parity.client_run import RAW

#: The burst's start, after the lone requests', and its loop thread's CPU clock
#: then; the threads its templates render on and its pool jobs run on.
T, C0, RENDER, POOLED = 2 * 10**12, 10**12, 7, 8
SIZES = (2000, 3000, 2500, 1500)
#: The loop's CPU from the first app call to the last send, and each template
#: render's and pool job's CPU, in ns.
LOOP_NS, RENDER_NS, POOL_NS = 3_000_000, 40_000, 100_000


def _burst(intake) -> None:
    """Append a burst of `SIZES` to the stamps at `intake`: each app call with
    its body's read begun at once, the bodies read, each template rendered on
    `RENDER` and each pool job run on `POOLED` (which makes no tokenizer call,
    so no tokenizer law prices it), then the sends, the last at `LOOP_NS` of
    the loop's CPU after the first app call."""
    lines = []
    for k, size in enumerate(SIZES):
        a = T + 1000 * k
        lines += [_stamp("app", a, a, cpu=(C0 + 30_000 * k,) * 2)]
        lines += [_stamp("body", a + 100, T + 10_000 + 1000 * k, size)]
        r = T + 20_000 + 1000 * k
        lines += [_stamp("tmpl", r, r + 200, size, (0, RENDER_NS), RENDER)]
        p = T + 30_000 + 1000 * k
        lines += [_stamp("prep", p, p + 500_000, 0, (0, POOL_NS), POOLED)]
        s = T + 600_000 + 200_000 * k
        end = C0 + LOOP_NS * (k + 1) // len(SIZES)
        lines += [_stamp("send", s, s + 50_000, 0, (end - 10, end))]
    with open(intake, "a") as f:
        f.write("\n".join(lines) + "\n")


@pytest.fixture
def stamps(tmp_path):
    (tmp_path / "steps").mkdir()
    stamped(tmp_path)
    intake = tmp_path / "steps" / compass_run.INTAKE_RECORD
    _burst(intake)
    return intake, intake.with_name(compass_run.TOKENIZER_RECORD)


def test_a_busy_period_holds_the_loop_s_cpu_and_the_off_loop_work(stamps):
    intake, calls = stamps
    seconds = (LOOP_NS + len(SIZES) * (RENDER_NS + POOL_NS)) * 1e-9
    assert _busy(intake, calls) == [(4, sum(SIZES), pytest.approx(seconds))]
    # The period begins at the first app call: a window from after it has none.
    assert _busy(intake, calls, [(T + 1, T + 10**9)]) == []


def test_busy_fits_fixed_s_on_the_periods_and_leaves_the_rest(stamps):
    intake, calls = stamps
    plain = fit_intake([intake], [calls])
    assert fit_intake([intake], [calls], None, None) == plain
    term, report = fit_intake([intake], [calls], None, [None])
    assert {k: term[k] for k in ("idle_s", "per_byte_s", "bytes")} == {
        k: plain[0][k] for k in ("idle_s", "per_byte_s", "bytes")
    }
    seconds = (LOOP_NS + len(SIZES) * (RENDER_NS + POOL_NS)) * 1e-9
    rest = seconds - term["idle_s"] - term["per_byte_s"] * sum(SIZES)
    assert term["fixed_s"] == pytest.approx(rest / (len(SIZES) - 1))
    assert report["busy"]["periods"] == 1 and report["busy"]["requests"] == 4
    assert "fixed_s_is_idle_s" not in report


def test_with_no_busy_period_in_the_spans_fixed_s_is_idle_s(stamps):
    intake, calls = stamps
    for spans in ([], [(T + 1, T + 10**9)]):
        term, report = fit_intake([intake], [calls], None, [spans])
        assert term["fixed_s"] == term["idle_s"]
        assert report["fixed_s_is_idle_s"] == "no busy period of two requests or more"


def test_fit_host_fits_fixed_s_on_the_intake_burst_runs_spans(tmp_path):
    r = deployment(tmp_path)
    publish(tmp_path, r)
    key = compass_run.host_key(r.config)
    a = record(tmp_path / "a", 19, key=key)
    stamped(a)
    _burst(a / "steps" / compass_run.INTAKE_RECORD)
    # The client run's raw records: sent from just before the burst, the last
    # answered 5 ms on.
    meta = {"request_start_ns": 7, "request_end_ns": 7 + 5_000_000}
    sent = [{"start_perf_ns": T - 1000 + k, "metadata": meta} for k in range(4)]
    (a / RAW).write_text("".join(json.dumps(x) + "\n" for x in sent))
    burst = ("--intake-burst-run", a, a / "client.json")
    fit(tmp_path, "fit host", a, *clients(a), *burst)

    host = member(tmp_path / "store", host_laws_key(key), "params")
    report = member(tmp_path / "store", host_laws_key(key), "report")
    span = [T - 1000, T + 3 - 1000 + 5_000_000]
    assert report["intake_burst_runs"] == [[str(a), str(a / "client.json"), span]]
    assert report["http_intake"]["busy"]["requests"] == 4
    intake = host["http_intake"]
    seconds = (LOOP_NS + len(SIZES) * (RENDER_NS + POOL_NS)) * 1e-9
    rest = seconds - intake["idle_s"] - intake["per_byte_s"] * sum(SIZES)
    assert intake["fixed_s"] == pytest.approx(rest / (len(SIZES) - 1))
