# SPDX-License-Identifier: MIT
"""A run prices the launch-path constants a kind-block host fit states
(`fit.host_blocks`, COMPASS_FIT_HOST_KIND_BLOCKS=1) under its host entry's
``paths``: the run file check takes them on segments its laws state
(`run._check_host`), the recurrence prices a forward's path constant in place
of its law's own (`runner.overrides.HostDeviceRecurrence`, from the batch's
`compass_path`), a simulated run sets that path from the view it prices
(`launch_features.path`), and the recurrence grade prices it on each step
(`parity.recurrence.priced`).

A planted provider's prefill past `BUCKET` tokens runs its body eagerly, its
launch `EAGER_S`.
"""

import json

import pytest
from test_backend_step_grid import scheduler_of
from test_fit_steps import PLATFORM
from test_memory_readings import META_8B
from test_parity_recurrence import _labelled, _law
from test_parity_recurrence import record as recurrence_record
from test_run_step_grid import deployment, publish, run_file

from atom.compass import run as compass_run
from atom.compass.artifacts import Kind, keyed_platform
from atom.compass.backends.launch_features import (
    GRAPHS,
    PATHS,
    REGISTRY,
    LaunchFeatures,
)
from atom.compass.backends.shape import BatchView, RequestShape, sum_tokens
from atom.compass.parity import recurrence
from atom.compass.runner.overrides import (
    FORWARD_KINDS,
    HOST_SEGMENTS,
    HostDeviceRecurrence,
)

ARCH = "PathsForCausalLM"
BUCKET = 4000
SETS = {"architecture": ARCH, "token": {"prefill": [BUCKET], "decode": []}}
EAGER_S = 0.04


class Paths(LaunchFeatures):
    features = ("eager",)

    def predict(self, view, graphs):
        bucket = max(graphs["token"]["prefill"])
        return {
            "eager": float(bool(view.prefill) and sum_tokens(view.requests) > bucket)
        }


@pytest.fixture(autouse=True)
def provider(monkeypatch):
    monkeypatch.setitem(REGISTRY, ARCH, Paths())


def test_a_recurrence_prices_a_path_constant_in_place_of_its_law_s():
    laws = {k: {g: (1e-4, 0.0, 0.0) for g in HOST_SEGMENTS} for k in FORWARD_KINDS}
    paths = {"decode": {"W_launch": {"rung16": 3e-3}}}
    rec = HostDeviceRecurrence(laws, (), lambda: 0.0, paths=paths)
    x = (1, 96, 16)
    assert rec.prepare_end(0.0, "decode", x, path="rung16")[1][
        "W_launch"
    ] == pytest.approx(3e-3)
    assert rec.prepare_end(0.0, "decode", x, path="rung8")[1]["W_launch"] == 1e-4
    assert rec.prepare_end(0.0, "decode", x)[1]["W_launch"] == 1e-4
    with pytest.raises(ValueError, match="pipeline stage"):
        HostDeviceRecurrence(laws, (), lambda: 0.0, stage=(0, 2), paths=paths)
    bare = HostDeviceRecurrence(laws, (), lambda: 0.0)
    assert bare.path(BatchView((RequestShape(6, 9, True),), capture_rung=1)) is None


def test_a_run_file_s_host_entry_takes_paths_on_segments_its_laws_state():
    laws = {"decode": {"W_launch": [1e-4, 0, 0]}}
    compass_run._check_host(
        "run.json", {"laws": laws, PATHS: {"decode": {"W_launch": {"rung1": 2e-3}}}}
    )
    for bad in (
        {"decode": {"W_pre": {"rung1": 2e-3}}},
        {"decode": {"W_launch": {"rung1": -1.0}}},
        {"prefill": {"W_launch": {"eager": 1.0}}},
    ):
        with pytest.raises(ValueError, match=PATHS):
            compass_run._check_host("run.json", {"laws": laws, PATHS: bad})


def test_a_run_prices_a_prefill_at_its_path_s_constant(tmp_path, monkeypatch):
    r = deployment(tmp_path)
    publish(tmp_path, r)
    key = {
        **compass_run.host_key(r.config),
        **keyed_platform(Kind.HOST_LAWS, PLATFORM),
    }
    key["engine"] = {**key["engine"], GRAPHS: SETS}
    laws = {k: {g: [1e-4, 0, 0] for g in HOST_SEGMENTS} for k in FORWARD_KINDS}
    paths = {"prefill": {"W_launch": {"eager": EAGER_S}}}
    run_file(tmp_path, monkeypatch, host={"key": key, "laws": laws, PATHS: paths})
    monkeypatch.setenv(compass_run.NOW_ENV, str(tmp_path / "now.bin"))
    compass_run._PublishedClock(
        compass_run.ENGINE,
        compass_run.channel_table(json.loads((tmp_path / "run.json").read_text())),
        None,
    )
    compass_run.runner(r, META_8B)
    batch, _ = scheduler_of([1000] * 5).schedule()
    assert batch.total_tokens_num > BUCKET
    r.forward(batch)
    assert batch.compass_path == "eager"
    assert r.compass_host.segments["W_launch"] == pytest.approx(EAGER_S)


def test_the_recurrence_grade_prices_a_path_constant_on_each_step(tmp_path):
    """`record()`'s output prefill of 50 tokens runs past a bucket of 32: its
    eager path's launch constant, 2 ms over its law's, moves its exit 2 ms."""
    steps = recurrence_record()
    samples, law = _labelled(tmp_path, steps), _law()
    bare = recurrence.priced(steps, samples, law)
    law["host"]["key"] = {"engine": {GRAPHS: {**SETS, "token": {"prefill": [32]}}}}
    base = law["host"]["laws"]["prefill"]["W_launch"][0]
    law["host"][PATHS] = {"prefill": {"W_launch": {"eager": base + 2e-3}}}
    priced = recurrence.priced(steps, samples, law)

    def residual(report):
        return {r[0]: r[3] for r in report["failures"]}

    assert residual(priced)[2] == pytest.approx(residual(bare).get(2, 0) + 2.0)
