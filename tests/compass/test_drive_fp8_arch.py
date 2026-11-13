# SPDX-License-Identifier: MIT
"""A simulated server, a run file's hops and the op graph `compass validate`
reads key an FP8 quantization as a worker on their machine's device does,
whatever device this host has. aiter resolves its fp8 dtype at import from the
device it finds (`test_check_fp8_arch`), and ATOM's config holds that dtype in
the quantization config, whose hash keys a run's memory readings and laws.

`serve` starts a simulated server in a process whose aiter is imported as on
its machine's device; the planted server parses ATOM's config from the
arguments it is handed, as ATOM's API server does. The hops and the op graph
build the config on the machine's arch (`drive.run._config`).

No device: each declares its machine's.
"""

import json
import os
import textwrap
from types import SimpleNamespace

import pytest
import torch
from test_check_fp8_arch import FP8, machine, worker
from test_drive_run import stub
from test_drive_runfile import build
from test_drive_serve import PLANTED, planted  # noqa: F401  a fixture
from test_key_quantization_remap import QWEN35
from test_run_hops import entry

from atom.compass import run as compass_run
from atom.compass.drive import serve
from atom.compass.run.step_grid import keyed
from atom.compass.validate import __main__ as validate

#: ATOM's arguments past the model: a TP4 server of Qwen3.5's FP8 checkpoint.
TP4 = ["-tp", "4"]
#: Ahead of the planted server: the quantization of ATOM's config as the API
#: server parses the arguments it is handed, and what the process holds after.
#: Under the main guard: the quantization's hash runs in a spawned child.
PARSED = """\
import json, os, sys, torch
from atom.compass.bench.measure import atom_config
from atom.compass.run.step_grid import keyed

if __name__ == "__main__":
    config = atom_config(sys.argv[1:], server=True)
    with open(os.environ["PLANTED_DIR"] + "/parsed.json", "w") as f:
        json.dump({
            "quantization": keyed(config)["quantization"],
            "available": torch.cuda.is_available(),
            "environ": {k: os.environ.get(k) for k in ("PATH", "GPU_ARCHS")},
        }, f)
"""


@pytest.mark.parametrize("arch", sorted(FP8))
def test_a_simulated_server_keys_fp8_as_its_machine_s_worker_does(
    planted, monkeypatch, arch  # noqa: F811
):
    monkeypatch.setenv("PLANTED_EXIT", "late:0")
    monkeypatch.delenv("GPU_ARCHS", raising=False)
    (planted / "planted.py").write_text(PARSED + textwrap.indent(PLANTED, "    "))
    config = worker(planted, monkeypatch, arch, QWEN35, *TP4)
    run = {"bound_s": 1.0, "machine": machine(arch)}

    serve.serve(["--model", config.model, *TP4], planted / "out", run=run)

    parsed = json.loads((planted / "parsed.json").read_text())
    assert parsed["quantization"] == keyed(config)["quantization"]
    # Only aiter's answers stay: the processes the server starts find this host's.
    assert parsed["available"] == torch.cuda.is_available()
    assert parsed["environ"] == {"PATH": os.environ["PATH"], "GPU_ARCHS": None}


@pytest.mark.parametrize("arch", sorted(FP8))
def test_a_run_file_takes_the_hops_of_the_host_entry_its_machine_s_worker_keys(
    tmp_path, monkeypatch, arch
):
    (tmp_path / "config.json").write_bytes(QWEN35.read_bytes())
    hops, laws = {}, []
    for k, fitted in enumerate(sorted(FP8)):
        (tmp_path / fitted).mkdir()
        config = worker(tmp_path / fitted, monkeypatch, fitted, QWEN35, *TP4)
        hops[fitted] = {"ipc_s": (k + 1) * 1e-4}
        law = entry(
            tmp_path, f"{fitted}.json", SimpleNamespace(config=config), hops[fitted]
        )
        laws += ["--host-laws", law]

    run = build(
        tmp_path, "--step-stub", stub(tmp_path), *laws, atom=TP4, machine=machine(arch)
    )

    assert run["ipc_s"] == hops[arch]["ipc_s"]


def test_an_op_graph_is_read_under_the_config_its_machine_s_worker_keys(
    tmp_path, monkeypatch
):
    config = worker(tmp_path, monkeypatch, "gfx942", QWEN35, *TP4)
    side = tmp_path / "side"
    (side / "steps").mkdir(parents=True)
    argv = ["run", "--", "--model", config.model, *TP4]
    run = {"argv": argv, "run_file": {"machine": machine("gfx942")}}
    (side / "run.json").write_text(json.dumps(run))
    (side / "steps/dp0.engine.json").write_text(json.dumps({"env": {}}))
    read = []
    monkeypatch.setattr(
        compass_run, "op_graph", lambda run_file, c, *args, **kw: read.append(c)
    )

    validate._op_graph(side)

    assert [keyed(c)["quantization"] for c in read] == [keyed(config)["quantization"]]
