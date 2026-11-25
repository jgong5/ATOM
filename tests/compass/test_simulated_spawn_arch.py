# SPDX-License-Identifier: MIT
"""A simulated server's engines and workers start as on its machine's device,
on a host with no driver.

ATOM's API server spawns its engine cores, and each engine core its workers;
each imports aiter again, whose Triton helpers probe the device for its arch.
`serve` starts the server under `on_arch`, and every process below it declares
the same arch before aiter loads, the server's module run as ``python -m`` runs
it or by a wrapper that makes it ``__main__`` in its place, as `http_intake`
runs ATOM's. The planted server spawns as ATOM's does, two levels deep, and
each level records the arch aiter's probe answers.

Then ATOM's own server, its engine core and TP1 worker, starts from `serve` on
a step stub and answers ``/health``. Its worker asks `torch.cuda` about the
device as it builds, its attention metadata builder the CU count; with no
driver the declared device answers.

No device: each declares its machine's.
"""

import dataclasses
import json
import os
import shutil
import signal
import subprocess
import sys
import textwrap

import pytest
from test_check_fp8_arch import FP8, QWEN, machine
from test_drive_runfile import build
from test_drive_serve import PLANTED, _ready, planted  # noqa: F401  a fixture
from test_drive_trace_steps import plant_module
from test_filler_tokens import model_dir  # noqa: F401  a fixture
from test_memory_readings import publish_readings

from atom.compass.backends.step_stub import Coefficients
from atom.compass.drive import run as drive_run
from atom.compass.drive import serve

#: Each level imports ATOM's model ops' fp8 helper, which probes the arch, and
#: records the arch; the first two spawn the next, as ATOM's server does.
CHILD = """\
import multiprocessing


def probe(d, depth):
    from aiter.ops.triton.utils._triton import arch_info
    from aiter.ops.triton.utils.types import get_fp8_dtypes

    get_fp8_dtypes()
    with open(f"{d}/arch{depth}", "w") as f:
        f.write(arch_info.get_arch())
    if depth < 2:
        level = multiprocessing.get_context("spawn").Process(
            target=probe, args=(d, depth + 1)
        )
        level.start()
        level.join()
"""
SPAWNS = """\
import os
from planted_child import probe

if __name__ == "__main__":
    probe(os.environ["PLANTED_DIR"], 0)
"""
#: Runs the module argv[1] names as `http_intake serve` runs ATOM's server.
WRAPPER = """\
import runpy, sys

runpy.run_module(sys.argv.pop(1), run_name="__main__", alter_sys=True)
"""
#: `serve` on the run file in argv[2], into argv[1], ATOM's arguments after.
DRIVE = """\
import json, sys
from pathlib import Path
from atom.compass.drive import serve
serve.serve(sys.argv[3:], sys.argv[1], run=json.loads(Path(sys.argv[2]).read_text()))
"""


@pytest.mark.parametrize("wrapped", [False, True], ids=["module", "wrapped"])
@pytest.mark.parametrize("arch", sorted(FP8))
def test_a_simulated_server_s_spawned_processes_probe_its_machine_s_arch(
    planted, monkeypatch, arch, wrapped  # noqa: F811
):
    monkeypatch.setenv("PLANTED_EXIT", "late:0")
    plant_module(planted, monkeypatch, SPAWNS + textwrap.indent(PLANTED, "    "))
    (planted / "planted_child.py").write_text(CHILD)
    if wrapped:
        (planted / "wrapper.py").write_text(WRAPPER)
        monkeypatch.setattr(serve, "SERVER", (*serve.SERVER[:2], "wrapper", "planted"))
    run = {"bound_s": 1.0, "machine": machine(arch)}

    serve.serve(["--model", "m"], planted / "out", run=run)

    levels = sorted(planted.glob("arch*"))
    assert [p.name for p in levels] == ["arch0", "arch1", "arch2"]
    assert {p.read_text() for p in levels} == {arch}


def test_atom_s_own_simulated_engine_starts(model_dir, tmp_path):  # noqa: F811
    for name in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy(model_dir / name, tmp_path / name)
    shutil.copy(QWEN, tmp_path / "config.json")
    (tmp_path / "stub.json").write_text(json.dumps(dataclasses.asdict(Coefficients())))
    atom = ["--enforce-eager", "--max-model-len", "2048"]
    run = build(tmp_path, "--step-stub", str(tmp_path / "stub.json"), atom=atom)
    atom = ["--model", str(tmp_path), *atom]
    publish_readings(tmp_path / "store", drive_run._config(atom, "gfx950"))
    (tmp_path / "run.json").write_text(json.dumps(run))
    (tmp_path / "drive.py").write_text(DRIVE)
    out = tmp_path / "out"
    driver = subprocess.Popen(
        [sys.executable, tmp_path / "drive.py", out, tmp_path / "run.json", *atom],
        env=dict(os.environ, PYTHONPATH=str(serve.TREE)),
    )
    try:
        assert _ready(out, driver)["pid"]
    finally:
        driver.send_signal(signal.SIGTERM)
        driver.wait(timeout=120)
    assert "built on meta and no checkpoint read" in (out / "server.log").read_text()
