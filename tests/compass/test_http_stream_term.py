# SPDX-License-Identifier: MIT
"""A streaming server's ``http_stream`` term: its API loop's CPU per SSE event.

On r256 (c=256, pit2-p03-g08, #3720) decode's API loop thread spent 22% of its
wall time on CPU, ~50 us per output row its output threads took in (one SSE
event of a streamed response each) besides its intakes; the relays and
intakes it takes in wait behind that work. `step_linear_fit.fit_stream` fits
the term on the loop thread's CPU clock over windows of a record's intake
stamps; the run reads it (`run.http_stream`) and `HttpChannel` charges it
before stamping each event (`tests/compass/test_carriers.py`).
"""

import json

import numpy as np
import pytest

from atom.compass import run as compass_run
from atom.compass.backends.step_linear_fit import (
    STREAM_WINDOW_S,
    _loop_windows,
    fit_stream,
    streams,
)

LOOP, OUT = 7, 9  # the API loop's thread, an output thread
PER_EVENT, PER_REQUEST, PER_BYTE, IDLE = 50e-6, 0.6e-3, 12e-9, 0.005


def _record(path, windows, head=True) -> None:
    """Intake stamps of a server whose loop spends, in each 10 s window of
    `windows`' ``(events, requests, body bytes)``, the CPU the module's rates
    give, at an even rate across the window: its own stamps every 100 ms (the
    first an SSE head unless not `head`) and each request's app call, body
    read and send, its thread CPU clock read off that rate; each event's
    output row on another thread."""
    lines, cpu0, width = [], 0, int(STREAM_WINDOW_S * 1e9)
    for w, (events, requests, nbytes) in enumerate(windows):
        t0 = (w + 1) * width
        busy = IDLE * STREAM_WINDOW_S + PER_EVENT * events
        busy += PER_REQUEST * requests + PER_BYTE * nbytes
        at = [
            (t0 + k * width // 100 + 1, "head" if head and not k else "recv", 0)
            for k in range(100)
        ]
        for k in range(requests):
            t = t0 + 1000 + k * (width - 2000) // requests
            at += [
                (t, "app", 0),
                (t + 10, "body", nbytes // requests),
                (t + 20, "send", 0),
            ]
        for t, kind, size in sorted(at):
            cpu = cpu0 + round(busy * 1e9 * (t - t0) / width)
            lines.append(f"{t} {LOOP} {kind} {t - 5} {size} {cpu} {cpu}")
        cpu0 += round(busy * 1e9)
        for k in range(events):
            t = t0 + 500 + k * (width - 1000) // max(events, 1)
            lines.append(f"{t} {OUT} out {t - 5} 0 0 0")
    path.write_text("".join(line + "\n" for line in lines))


def _windows(n, seed=0):
    rng = np.random.default_rng(seed)
    return [
        (int(rng.integers(500, 30_000)), int(rng.integers(1, 120)), 0) for _ in range(n)
    ]


def test_the_loop_s_cpu_per_event_is_fitted_over_its_windows(tmp_path):
    sizes = np.random.default_rng(1).integers(10_000, 2_000_000, 24)
    windows = [(e, r, r * int(b)) for (e, r, _), b in zip(_windows(24), sizes)]
    _record(tmp_path / "a", windows[:12])
    _record(tmp_path / "b", windows[12:])
    term, report = fit_stream([tmp_path / "a", tmp_path / "b"])
    assert term == {"per_event_s": pytest.approx(PER_EVENT, rel=0.02)}
    coef = report["coefficients"]
    assert coef["requests"] == pytest.approx(PER_REQUEST, rel=0.05)
    assert coef["bytes"] == pytest.approx(PER_BYTE, rel=0.05)
    assert report["windows"] == 24


def test_a_window_the_loop_s_stamps_do_not_span_is_left_out(tmp_path):
    _record(tmp_path / "a", _windows(3))
    lines = (tmp_path / "a").read_text().splitlines()
    # The loop stamps only the first half of the second window.
    width = STREAM_WINDOW_S * 1e9
    kept = [
        line
        for line in lines
        if not (
            line.split()[1] == str(LOOP)
            and 2.5 * width < int(line.split()[0]) < 3 * width
        )
    ]
    (tmp_path / "a").write_text("".join(line + "\n" for line in kept))
    assert len(_loop_windows(tmp_path / "a")) == 2


def test_a_server_that_streamed_nothing_gives_no_term(tmp_path):
    _record(tmp_path / "a", _windows(12), head=False)
    assert not streams(tmp_path / "a") and _loop_windows(tmp_path / "a") == []
    with pytest.raises(ValueError, match="no record streamed an SSE response"):
        fit_stream([tmp_path / "a"])


def write(tmp_path, monkeypatch, **run):
    path = tmp_path / "run.json"
    path.write_text(json.dumps({"bound_s": 1.0, **run}))
    monkeypatch.setenv(compass_run.ENV, str(path))


def test_a_run_reads_its_server_s_http_stream_term(tmp_path, monkeypatch):
    write(tmp_path, monkeypatch, host={"http_stream": {"per_event_s": 5e-5}})
    assert compass_run.http_stream() == 5e-5
    write(tmp_path, monkeypatch, host={})
    assert compass_run.http_stream() is None


@pytest.mark.parametrize(
    "term",
    [
        {},
        {"per_event_s": -1e-5},
        {"per_event_s": "5e-5"},
        {"per_event_s": 1e-5, "x": 1},
    ],
)
def test_an_http_stream_term_the_frontend_would_misread_is_refused(
    tmp_path, monkeypatch, term
):
    write(tmp_path, monkeypatch, host={"http_stream": term})
    with pytest.raises(ValueError, match="host's http_stream is"):
        compass_run.spec()


@pytest.mark.parametrize("head", [True, False], ids=["streamed", "unstreamed"])
def test_compass_fit_host_fits_the_term_for_a_server_that_streamed_only(
    tmp_path, monkeypatch, head
):
    from atom.compass.fit import host as fit_host_cli

    monkeypatch.setattr(
        fit_host_cli, "fit_host", lambda samples: ({}, {"unfitted": {}})
    )
    sizes = np.random.default_rng(2).integers(10_000, 2_000_000, 12)
    windows = [(e, r, r * int(b)) for (e, r, _), b in zip(_windows(12), sizes)]
    (tmp_path / "decode" / "steps").mkdir(parents=True)
    _record(tmp_path / "decode" / "steps" / compass_run.INTAKE_RECORD, windows, head)
    # A tokenizer record with no call: its terms are left unfitted.
    (tmp_path / "decode" / "steps" / compass_run.TOKENIZER_RECORD).write_text(
        json.dumps({"backend": "fast", "fingerprint": "sha256:0"}) + "\n"
    )
    host, report = fit_host_cli.host_laws([], [tmp_path / "decode"])
    if head:
        assert host["http_stream"]["per_event_s"] == pytest.approx(PER_EVENT, rel=0.02)
        assert "http_stream" not in report["unfitted"]
    else:
        assert "http_stream" not in host and "http_stream" not in report["unfitted"]
