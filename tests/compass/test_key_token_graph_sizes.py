# SPDX-License-Identifier: MIT
"""``ATOM_TOKEN_GRAPH_SIZES`` sets ATOM's prefill token buckets, and the largest
moves a longer prefill's host launch, so a run whose buckets ATOM reads off its
default refuses a step law fitted at the default, naming them, and a run at the
default keys as before, however it spells the default. The default once stopped
at 3072: a law recorded there states those buckets, and a run at today's
default refuses it.

No device.
"""

import pytest
from test_drive_runfile import build, install, step_linear
from test_run_step_grid import deployment

from atom.benchmarks.results.metadata import PERFORMANCE_KEYS
from atom.compass.backends import StepLinearBackend
from atom.compass.run.step_grid import keyed

BUCKETS = "ATOM_TOKEN_GRAPH_SIZES"
#: ATOM's default buckets before they reached 7168, which the DeepSeek-V4.1-Flash
#: laws in scripts/compass/laws/ were recorded at.
FORMER = "256,512,1024,2048,3072"


@pytest.fixture(autouse=True)
def environment(monkeypatch):
    """No performance variable and ATOM's default buckets."""
    for name in (*PERFORMANCE_KEYS, BUCKETS):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("buckets", ["256,512", "256,512,1024,2048,3072,4096", ""])
def test_a_run_at_other_buckets_refuses_a_law_fitted_at_the_default(
    monkeypatch, tmp_path, buckets
):
    law = step_linear(tmp_path, deployment(tmp_path))
    run = build(tmp_path, "--step-linear", law.key.token())
    monkeypatch.setenv(BUCKETS, buckets)

    with pytest.raises(ValueError, match="fitted under another engine config") as e:
        install(tmp_path, monkeypatch, run, deployment(tmp_path))
    assert f"env {BUCKETS} None in the law and {buckets!r} here" in str(e.value)


@pytest.mark.parametrize(
    "spelled",
    [
        "256,512,1024,2048,3072,4096,5120,6144,7168",
        " 256, 512,1024,2048,3072,4096,5120,6144, 7168",
    ],
)
def test_a_run_at_the_default_buckets_keys_as_before(monkeypatch, tmp_path, spelled):
    config = deployment(tmp_path).config
    unset = keyed(config)
    monkeypatch.setenv(BUCKETS, spelled)

    assert unset["env"] == {}
    assert keyed(config) == unset


def test_a_law_recorded_at_the_former_default_refuses_a_run_at_the_default(
    monkeypatch, tmp_path
):
    monkeypatch.setenv(BUCKETS, FORMER)
    law = step_linear(tmp_path, deployment(tmp_path))
    run = build(tmp_path, "--step-linear", law.key.token())
    monkeypatch.delenv(BUCKETS)

    with pytest.raises(ValueError, match="fitted under another engine config") as e:
        install(tmp_path, monkeypatch, run, deployment(tmp_path))
    assert f"env {BUCKETS} {FORMER!r} in the law and None here" in str(e.value)

    monkeypatch.setenv(BUCKETS, FORMER)
    r = deployment(tmp_path)
    install(tmp_path, monkeypatch, run, r)
    assert isinstance(r.compass_backend, StepLinearBackend)
