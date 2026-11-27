# SPDX-License-Identifier: MIT
"""A host segment whose every label is 0 fits a zero law
(`step_linear_fit.fit_host`). DeepSeek-V4.1-Flash's DSpark output prefill
waits on its MTP status just after its launch, so it labels `W_pre` up to
the launch and `W_inputs` 0 (`step_labels._segments`); the zero law then
prices `W_inputs` at no time, and a run charges such a forward rather than
refusing it for want of a law."""

import dataclasses
import json
from types import SimpleNamespace

import numpy as np
import pytest
from test_backend_step_linear_fit import HOST
from test_dummy_host import SETS, _dummy_batch, _samples, planted  # noqa: F401
from test_mtp_prepare_split import INPUTS, _mtp_record

from atom.compass import run as compass_run
from atom.compass.backends.launch_features import GRAPHS, stated
from atom.compass.backends.step_linear_fit import (
    DUMMY,
    HOST_FEATURES,
    fit_host,
    label_records,
)
from atom.compass.runner.overrides import MTP_COPY, MTP_SEGMENT, HostDeviceRecurrence


def _late_record(path):
    """`_mtp_record`'s record, each output prefill's status wait moved after
    its launch as #4248's target record stamps all 2469 of its own: from 2 us
    past the launch's end, 9 us wide."""
    record = _mtp_record(path)
    lines = [json.loads(x) for x in record.read_text().splitlines()]
    for s in lines:
        if s["produces_output"] and not s["rows"][0][2]:
            end = s["t_launch_ns"][1]
            s["t_mtp_ns"] = [end + 2_000, end + 11_000]
    record.write_text("".join(json.dumps(s) + "\n" for s in lines))
    return record


def test_a_status_wait_after_each_output_prefill_s_launch_fits_a_zero_w_inputs(
    tmp_path,
):
    samples = label_records([_late_record(tmp_path / "dp0.jsonl")])[0]
    host, report = fit_host(samples)

    assert report["unfitted"] == {}
    prefill = host["laws"]["prefill"]
    assert prefill[MTP_SEGMENT] == [0.0] * len(HOST_FEATURES)
    assert report[f"prefill.{MTP_SEGMENT}"]["zero"] == list(HOST_FEATURES)
    # W_pre runs to the launch, never into it.
    assert prefill["W_pre"] == pytest.approx(HOST["prefill"]["W_pre"], rel=1e-4)
    # A decode's wait before its launch keeps its split.
    assert host["laws"]["decode"][MTP_SEGMENT] == pytest.approx(
        INPUTS["decode"], rel=1e-4, abs=1e-12
    )

    compass_run._check_host("run.json", host)
    rec = HostDeviceRecurrence(host["laws"], [MTP_COPY], lambda: 0.0, mtp=True)
    batch = SimpleNamespace(
        total_seqs_num_prefill=1,
        total_tokens_num=2048,
        total_seqs_num=1,
        produces_output=lambda: True,
    )
    rec.charge(batch, 0.05, 0.0)
    assert rec.segments[MTP_SEGMENT] == 0.0


def test_a_dummy_segment_whose_every_label_is_0_fits_a_zero_law():
    samples = _samples(np.random.default_rng(3746), 80)
    rows = tuple(
        (*r[:2], r[2] | {MTP_SEGMENT: 0.0}, r[3]) for r in samples[0].dummy_host
    )
    samples = [dataclasses.replace(s, dummy_host=rows) for s in samples]
    host, report = fit_host(samples)

    assert host["laws"][DUMMY][MTP_SEGMENT] == [0.0] * len(HOST_FEATURES)
    assert report[f"{DUMMY}.{MTP_SEGMENT}"] == {
        "samples": 80,
        "zero": list(HOST_FEATURES),
        "forwards": 0,
    }
    assert host["laws"][DUMMY]["W_pre"] == pytest.approx([1e-3, 0, 0], rel=0.02)

    # Beside a decode `W_inputs` law that adopts the eager column, an eager
    # dummy is charged no `W_inputs`, and an eager decode its column.
    host["laws"]["decode"][MTP_SEGMENT] = [1e-3, 0.0, 0.0]
    host["columns"].setdefault("decode", {})[MTP_SEGMENT] = {"eager": 0.02}
    launch = stated({"engine": {GRAPHS: SETS}})
    rec = HostDeviceRecurrence(
        host["laws"],
        [MTP_COPY],
        lambda: 0.0,
        mtp=True,
        columns=host["columns"],
        launch=launch,
    )
    rec.charge(_dummy_batch(True, 1.0), 0.01, 0.0)
    assert rec.segments[MTP_SEGMENT] == 0.0
    rec.charge(_dummy_batch(False, 1.0), 0.01, 0.0)
    assert rec.segments[MTP_SEGMENT] == pytest.approx(0.021)
