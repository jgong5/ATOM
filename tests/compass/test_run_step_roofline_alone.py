# SPDX-License-Identifier: MIT
"""A run file naming ``step_roofline`` and none of ``step_stub``,
``step_linear``, ``step_grid`` and ``op_graph`` installs the step roofline as
the run's only backend, every step priced by it; beside one of the four it
stays the fallback below it; with none of the five a run refuses, naming all
five. `compass run --step-roofline` alone builds that run file.

The run file is installed by `run.runner`: on Qwen3-0.6B's meta build at TP1
(`qwen3_0_6b_config.json`) under `tests/compass/machines/mi308x-hjbog18.json`,
built in a process of its own because the driverless declarations are
process-global, as `test_backend_step_roofline_mla` builds its models; and on
the suite's 8B worker (`test_backend_step_linear._installed`) for the keys.
"""

import argparse
import dataclasses
import json
import os
import pathlib
import subprocess
import sys

import pytest
from test_backend_step_linear import _installed
from test_drive_runfile import PLATFORM_AND_STORE, build, install
from test_run_step_grid import deployment

from atom.compass.backends.step_roofline import (
    StepFallbackBackend,
    StepRooflineBackend,
)
from atom.compass.backends.step_stub import Coefficients, StepStubBackend
from atom.compass.drive import run as drive_run
from atom.compass.drive import runfile

TREE = pathlib.Path(__file__).resolve().parents[2]
CONFIG = pathlib.Path(__file__).with_name("qwen3_0_6b_config.json")
MI308X = TREE / "tests" / "compass" / "machines" / "mi308x-hjbog18.json"
#: The five run-file keys a run prices its steps from, as the refusal names them.
FIVE = ("step_stub", "step_linear", "step_grid", "op_graph", "step_roofline")

#: Builds the model under argv's root on the meta device of argv's machine's
#: arch, installs argv's run file through `run.runner` with the model's memory
#: readings published in the store it names, and prints the installed backend's
#: class and its price of one prefill and one decode batch.
RUN = """
import json, pathlib, sys
root, machine = pathlib.Path(sys.argv[1]), json.loads(pathlib.Path(sys.argv[2]).read_text())
sys.path.insert(0, sys.argv[3])
from atom.compass.capture import driverless
driverless.install(str(root), machine["device"]["arch"])
driverless.group(1, 1)
import os, torch
from types import SimpleNamespace
from atom.config import Config, set_current_atom_config
config = Config(model=str(root / "model"), tensor_parallel_size=1, load_dummy="empty",
                enforce_eager=True)
set_current_atom_config(config)
from atom.model_engine.model_runner import support_model_arch_dict
from atom.utils import resolve_obj_by_qualname
from atom.compass.runner.overrides import FactoriesOnMeta, process_weights_after_loading
torch.set_default_dtype(config.torch_dtype)
with FactoriesOnMeta():
    tree = resolve_obj_by_qualname(support_model_arch_dict[config.hf_config.architectures[0]])(config)
    process_weights_after_loading(tree)
from test_memory_readings import publish_readings
from atom.compass import run as compass_run
from atom.compass.backends import BatchView, RequestShape
from atom.compass.runner.overrides import _installed_backend
publish_readings(root / "store", config, document=machine)
run = {"bound_s": 600.0, "step_roofline": True, "machine": machine, "store": str(root / "store")}
(root / "run.json").write_text(json.dumps(run))
os.environ[compass_run.ENV] = str(root / "run.json")
runner = SimpleNamespace(config=config, rank=0, _tr_hybrid_active=lambda: False)
compass_run.runner(runner, tree)
backend = _installed_backend(runner)
prefill = BatchView((RequestShape(512, 512, False),))
decode = BatchView(tuple(RequestShape(1, 600 + i, True) for i in range(4)))
print(json.dumps({
    "installed": type(backend).__name__,
    "unclassified": list(backend.unclassified),
    "prefill_s": backend.price(prefill).seconds,
    "decode_s": backend.price(decode).seconds,
}))
"""


@pytest.fixture(scope="module")
def qwen3(tmp_path_factory):
    """What the Qwen3-0.6B build's run printed."""
    root = tmp_path_factory.mktemp("qwen3")
    (root / "model").mkdir()
    (root / "model" / "config.json").write_text(CONFIG.read_text())
    here = str(pathlib.Path(__file__).parent)
    done = subprocess.run(
        [sys.executable, "-c", RUN, str(root), str(MI308X), here],
        capture_output=True,
        text=True,
        check=False,
        timeout=900,
        cwd=TREE,
        env={**os.environ, "PYTHONPATH": str(TREE)},
    )
    assert done.returncode == 0, done.stderr[-4000:]
    return json.loads(done.stdout.splitlines()[-1])


def test_qwen3_0_6b_on_mi308x_is_priced_by_the_step_roofline_alone(qwen3):
    print(f"\n{qwen3}")
    assert qwen3["installed"] == StepRooflineBackend.__name__
    assert qwen3["unclassified"] == []
    # One prefill of 512 tokens costs more than one decode step of four rows.
    assert qwen3["prefill_s"] > qwen3["decode_s"] > 0


def test_a_run_file_naming_step_roofline_alone_installs_it_unwrapped(
    monkeypatch, tmp_path
):
    installed = _installed(monkeypatch, tmp_path, step_roofline=True)

    assert type(installed) is StepRooflineBackend
    assert installed.padded is True


def test_a_run_file_naming_a_backend_beside_it_keeps_the_fallback(
    monkeypatch, tmp_path
):
    stub = dataclasses.asdict(Coefficients())
    installed = _installed(monkeypatch, tmp_path, step_stub=stub, step_roofline=True)

    assert isinstance(installed, StepFallbackBackend)
    assert isinstance(installed.first, StepStubBackend)


@pytest.mark.parametrize("keys", [{}, {"step_roofline": False}], ids=["none", "off"])
def test_a_run_file_naming_no_backend_refuses_naming_all_five(
    monkeypatch, tmp_path, keys
):
    with pytest.raises(ValueError, match="this one has none") as refused:
        _installed(monkeypatch, tmp_path, **keys)
    assert all(key in str(refused.value) for key in FIVE)


def test_compass_run_step_roofline_alone_builds_and_installs_it(monkeypatch, tmp_path):
    run = build(tmp_path, "--step-roofline")

    assert run["step_roofline"] is True
    assert not {"step_stub", "step_linear", "step_grid", "op_graph"} & run.keys()
    assert run["backend"]["provenance"] is None
    install(tmp_path, monkeypatch, run, r := deployment(tmp_path))
    assert type(r.compass_backend) is StepRooflineBackend


def test_compass_run_with_no_backend_flag_refuses_naming_all_five(tmp_path):
    flags = [f"--{key.replace('_', '-')}" for key in FIVE]
    parser = argparse.ArgumentParser()
    runfile.add_arguments(parser)
    args = parser.parse_args(PLATFORM_AND_STORE)

    with pytest.raises(ValueError, match="name none") as refused:
        runfile.build(args, ["--model", str(tmp_path)])
    assert all(flag in str(refused.value) for flag in flags)
    with pytest.raises(ValueError, match="name none"):
        drive_run.main([*PLATFORM_AND_STORE, "--out", str(tmp_path), "--", "-tp", "1"])


def test_coverage_beside_the_step_roofline_alone_refuses_naming_it(tmp_path):
    argv = [*PLATFORM_AND_STORE, "--step-roofline", "--coverage", "r", "--out", "o"]

    with pytest.raises(ValueError, match="--step-roofline has no coverage check"):
        drive_run.main([*argv, "--", "-tp", "1"])
