# SPDX-License-Identifier: MIT
"""The run file's ``rare_full_gc``: every process of a simulated run raises
CPython's generation-2 collection threshold, generations 0 and 1 as before;
without it nothing changes.

A process the run spawns takes it as it imports `atom.compass.run`, or aiperf's
as plugin discovery imports ``compass_harness``: each such case imports in a
fresh interpreter. The API server and the standalone authority import that
module before their command line names the run file, and take it as they start.
"""

import dataclasses
import gc
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from test_cohosted_frontend_clock import tcp_run  # noqa: F401  the fixture
from test_dp_slice import _deployment
from test_drive_runfile import build
from test_vertical_slice import _run_file

from atom.compass import run as compass_run
from atom.compass.backends.step_stub import Coefficients

ROOT = Path(__file__).resolve().parents[2]
PROBE = """
import gc
try:
    import {module}
except Exception:  # the harness checks aiperf after the setting, all this reads
    pass
print(*gc.get_threshold())
"""


@pytest.fixture(autouse=True)
def thresholds():
    """This process's thresholds, put back after the test."""
    saved = gc.get_threshold()
    yield saved
    gc.set_threshold(*saved)


def imported(module: str, run_file: Path | None) -> tuple[int, ...]:
    """A fresh interpreter's thresholds once it imports `module`, its
    environment naming `run_file` as the run's, or none."""
    env = {k: v for k, v in os.environ.items() if k != compass_run.ENV}
    if run_file is not None:
        env[compass_run.ENV] = str(run_file)
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), str(ROOT / "compass_harness")])
    done = subprocess.run(
        [sys.executable, "-c", PROBE.format(module=module)],
        check=False,
        env=env,
        cwd="/",
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert done.returncode == 0, done.stderr
    return tuple(int(t) for t in done.stdout.split()[-3:])


@pytest.fixture(scope="module")
def default() -> tuple[int, ...]:
    """A fresh interpreter's thresholds, which nothing here changes."""
    return imported("os", None)


def raised(default) -> tuple[int, ...]:
    return (*default[:2], compass_run.RARE_FULL_GC_GEN2)


@pytest.mark.parametrize("module", ["atom.compass.run", "compass_harness"])
def test_a_process_the_run_spawns_raises_generation_2_alone(tmp_path, module, default):
    run_file = _run_file(tmp_path, rare_full_gc=True)
    assert imported(module, run_file) == raised(default)


@pytest.mark.parametrize("module", ["atom.compass.run", "compass_harness"])
@pytest.mark.parametrize("keys", [None, {}, {"rare_full_gc": False}])
def test_without_the_option_a_process_keeps_its_thresholds(
    tmp_path, module, keys, default
):
    run_file = None if keys is None else _run_file(tmp_path, **keys)
    assert imported(module, run_file) == default


def test_the_api_server_takes_it_as_its_frontend_starts(tcp_run, default):  # noqa: F811
    gc.set_threshold(*default)
    path = Path(os.environ[compass_run.ENV])
    path.write_text(json.dumps({**tcp_run, compass_run.RARE_FULL_GC: True}))
    with compass_run.frontend(_deployment()):
        pass
    assert gc.get_threshold() == raised(default)


@pytest.mark.parametrize("on", [False, True])
def test_the_standalone_authority_takes_it_as_it_starts(
    tmp_path, monkeypatch, default, on
):
    gc.set_threshold(*default)
    monkeypatch.setenv(compass_run.ENV, str(_run_file(tmp_path, rare_full_gc=on)))
    seen = []
    monkeypatch.setattr(
        compass_run, "serve", lambda a, e, select: seen.append(gc.get_threshold())
    )
    monkeypatch.setattr(compass_run, "_until_end", lambda run, a: False)
    monkeypatch.setattr(compass_run, "_write_summary", lambda *_: None)
    compass_run.authority("tcp://127.0.0.1:1")
    assert seen == [raised(default) if on else default]


@pytest.mark.parametrize("flags, on", [((), False), (("--rare-full-gc",), True)])
def test_the_option_is_off_unless_the_run_command_turns_it_on(tmp_path, flags, on):
    stub = tmp_path / "stub.json"
    stub.write_text(json.dumps(dataclasses.asdict(Coefficients())))
    run = build(tmp_path, "--step-stub", str(stub), *flags)
    assert run[compass_run.RARE_FULL_GC] is on
    off = dict(run, **{compass_run.RARE_FULL_GC: not on})
    assert compass_run.configuration(run) == compass_run.configuration(off)


def test_a_run_refuses_an_option_that_is_not_true_or_false(tmp_path, monkeypatch):
    monkeypatch.setenv(compass_run.ENV, str(_run_file(tmp_path, rare_full_gc="yes")))
    with pytest.raises(ValueError, match="rare_full_gc is 'yes'"):
        compass_run.spec()
