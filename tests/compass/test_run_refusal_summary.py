# SPDX-License-Identifier: MIT
"""What the run summary says about the steps a run priced.

Steps are priced through the runner's own `forward`, the worker's answers are
read back as `engine_done` reads them over the worker RPC, and the frontend's
`frontend_done` writes ``summary.json``.
"""

import json
import math
import time
from types import SimpleNamespace

import pytest
import torch
from test_backend_coarse import law
from test_dp_step_max import prefill, runner
from test_runner_control_refusals import Worker

from atom.compass import run as compass_run
from atom.compass.backends import (
    BatchView,
    CoarseBackend,
    CoarseLaw,
    Coefficients,
    CostTerm,
    Refusal,
    RequestShape,
    ShapeStubBackend,
    StepCost,
)
from atom.utils import clock
from atom.utils.clock import LPRuntime

STAGE1 = Coefficients(decode_request=0.0)
CHAIN = (
    Refusal("price_list", "no entry for this shape"),
    Refusal("nearest_key", "outside the measured range"),
)
# Prefill of 1 to 32 tokens with nothing cached: the chord of t^2 from 1 to 32.
SHORT_PREFILL_HULL = {
    "equal": {"query_cached": 0},
    "facets": [[-1, 0, 1], [0, -1, 1], [-33, 1, 32], [1, 0, -32]],
}


class StandsIn(ShapeStubBackend):
    """The stub, standing in after two rungs refused a prefill of `tokens` or more."""

    def __init__(self, coefficients, tokens):
        super().__init__(coefficients)
        self.tokens = tokens

    def estimate(self, batch_view):
        cost = super().estimate(batch_view)
        if sum(r.query_tokens for r in batch_view.prefill) < self.tokens:
            return cost
        return StepCost(
            [
                CostTerm(t.name, t.seconds, t.provenance.resolved("stub", CHAIN))
                for t in cost.terms
                if t.seconds > 0.0
            ]
        )


def seconds(backend, tokens):
    return backend.estimate(BatchView((RequestShape(tokens, tokens, False),))).seconds


def summary(monkeypatch, tmp_path, backend, *prompts):
    """The run summary after one engine's runner priced a prefill of each prompt."""
    monkeypatch.setattr(torch.distributed, "all_reduce", None)
    run = {
        "admission_path": "serving",
        "ipc_s": 0.001,
        "stream_s": 0.002,
        "bound_s": 10.0,
        "clock_endpoint": "inproc:unused",
        "out_dir": str(tmp_path),
    }
    (tmp_path / "run.json").write_text(json.dumps(run))
    monkeypatch.setenv(compass_run.ENV, str(tmp_path / "run.json"))
    table = compass_run.channel_table(run)
    authority = compass_run._RecordingAuthority(run)
    authority.started = authority.finished = time.monotonic()

    r = runner(backend)
    for tokens in prompts:
        r.forward(prefill(tokens))
    engine = LPRuntime(compass_run.ENGINE, table, None)
    engine.now = math.inf
    monkeypatch.setattr(clock, "_installed", engine)
    compass_run.engine_done(SimpleNamespace(runner_mgr=Worker(r)))

    monkeypatch.setattr(compass_run, "_authority", authority)
    frontend = LPRuntime(compass_run.FRONTEND, table, None)
    frontend.now = math.inf
    frontend.loop = SimpleNamespace(executor=SimpleNamespace(refusals=[]))
    monkeypatch.setattr(clock, "_installed", frontend)
    assert compass_run.frontend_done(SimpleNamespace(close=lambda: None))
    return json.loads((tmp_path / compass_run.SUMMARY_FILE).read_text())


def test_steps_priced_after_a_refusal_are_counted_by_step_and_by_second(
    monkeypatch, tmp_path
):
    backend = StandsIn(STAGE1, tokens=40)
    out = summary(monkeypatch, tmp_path, backend, 8, 8, 8, 40)

    short, long = seconds(backend, 8), seconds(backend, 40)
    refusals = out["schedule"]["refusals"]
    reason = "cost:price_list: no entry for this shape; nearest_key: outside the measured range"
    assert refusals["count"] == 1
    assert refusals["steps"] == 4
    assert refusals["fraction_of_steps"] == 0.25
    assert refusals["fraction_of_predicted_seconds"] == pytest.approx(
        long / (3 * short + long)
    )
    assert refusals["reasons"] == [[reason, 1]]
    assert out["schedule"]["predicted_seconds"] == pytest.approx(3 * short + long)
    assert out["coverage_report"]


@pytest.mark.parametrize("clean,report", [(18, True), (20, False)])
def test_more_than_five_percent_of_seconds_refused_is_a_coverage_report(
    monkeypatch, tmp_path, clean, report
):
    flat = Coefficients.constant(prefill_seconds=0.001, decode_seconds=0.001)
    out = summary(monkeypatch, tmp_path, StandsIn(flat, 40), *[8] * clean, 40)

    fraction = out["schedule"]["refusals"]["fraction_of_predicted_seconds"]
    assert fraction == pytest.approx(1 / (clean + 1))
    assert (fraction > compass_run.REFUSED_SECONDS_GATE) == report
    assert out["coverage_report"] is report


def test_steps_outside_the_hull_are_extrapolated_and_never_refused(
    monkeypatch, tmp_path
):
    backend = CoarseBackend(CoarseLaw(law(prefill_hull=SHORT_PREFILL_HULL)))
    out = summary(monkeypatch, tmp_path, backend, 8, 8, 8, 40)

    short, long = seconds(backend, 8), seconds(backend, 40)
    assert out["schedule"]["extrapolated"] == {
        "steps": 1,
        "fraction_of_predicted_seconds": pytest.approx(long / (3 * short + long)),
    }
    refusals = out["schedule"]["refusals"]
    assert (refusals["count"], refusals["steps"]) == (0, 4)
    assert refusals["fraction_of_predicted_seconds"] == 0.0
    assert not out["coverage_report"]


def test_a_run_with_neither_reports_zero_of_both_beside_its_seconds(
    monkeypatch, tmp_path
):
    backend = ShapeStubBackend(STAGE1)
    out = summary(monkeypatch, tmp_path, backend, 8, 40)

    schedule = out["schedule"]
    assert schedule["predicted_seconds"] == pytest.approx(
        seconds(backend, 8) + seconds(backend, 40)
    )
    assert schedule["extrapolated"] == {
        "steps": 0,
        "fraction_of_predicted_seconds": 0.0,
    }
    assert schedule["refusals"]["fraction_of_predicted_seconds"] == 0.0
    assert schedule["refusals"]["fraction_of_steps"] == 0.0
    assert not out["coverage_report"]
