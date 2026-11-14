# SPDX-License-Identifier: MIT
"""A simulated server a script starts keys an FP8 quantization as a worker on
its run file's machine's device does, whatever device this host has, as
`compass run`'s does (`test_drive_fp8_arch`): each script starts it under
`on_arch`, which imports aiter as on that machine's device first.

`python3` is stubbed on `PATH` to plant modules in every process it starts:
ATOM's API servers, the clock authority and the parity traffic. A planted
simulated server writes the quantization of ATOM's config as it parses the
arguments it is handed and exits, before the script reads it ready; a real one
only says it listens. The rest, `on_arch` among it, runs as written. No device.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from test_check_fp8_arch import machine, worker
from test_key_quantization_remap import QWEN35

from atom.compass.run.step_grid import keyed

TREE = Path(__file__).resolve().parents[2]
#: A module named by a file in PLANTED_DIR runs that file.
SITE = """\
import importlib.util, os, sys
d = os.environ["PLANTED_DIR"]
planted = {f[: -len(".py")]: os.path.join(d, f) for f in os.listdir(d)}
class Planted:
    def find_spec(name, path=None, target=None):
        if name in planted:
            return importlib.util.spec_from_file_location(name, planted[name])
sys.meta_path.insert(0, Planted)
"""
#: Under the main guard: the quantization's hash runs in a spawned child.
SERVER = """\
import os, sys, time
if __name__ == "__main__":
    if "--compass-run" not in sys.argv:
        print("Uvicorn running", flush=True)
        time.sleep(600)
    from atom.compass.bench.measure import atom_config
    from atom.compass.run.step_grid import keyed
    config = atom_config(sys.argv[1:], server=True)
    with open(os.environ["PARSED"], "a") as f:
        f.write(keyed(config)["quantization"] + "\\n")
"""
AUTHORITY = """\
import socket, sys, time
endpoint = sys.argv[sys.argv.index("--compass-clock-endpoint") + 1]
listening = socket.create_server(("127.0.0.1", int(endpoint.rpartition(":")[2])))
time.sleep(600)
"""
PLANTED = {
    "atom.entrypoints.openai.api_server": SERVER,
    "atom.entrypoints.openai_server": SERVER,
    "atom.compass.run.__main__": AUTHORITY,
    "atom.compass.parity.traffic": "",
}
EAGER = ["--enforce-eager"]
#: Each script, the run file it is given (None: it writes its own), its
#: variables, its simulated servers' ATOM arguments past the model and
#: `SERVER_ARGS`, and the arch of the machine.
CASES = {
    "tp1-gfx950": ("cctraces_sim.sh", {}, {}, [], "gfx950"),
    "tp1-gfx942": ("cctraces_sim.sh", {}, {}, [], "gfx942"),
    "http-intake": ("cctraces_sim.sh", {}, {"HTTP_INTAKE": "x"}, [], "gfx950"),
    "1p1d": ("cctraces_sim.sh", {"router_s": 1e-5}, {}, [], "gfx942"),
    "parity": (
        "parity_one_node.sh",
        None,
        {},
        ["-tp", "2", "--enable-dp-attention"],
        "gfx950",
    ),
}


@pytest.mark.parametrize("script, run, more, args, arch", CASES.values(), ids=CASES)
def test_a_script_s_simulated_server_keys_fp8_as_its_machine_s_worker_does(
    tmp_path, monkeypatch, script, run, more, args, arch
):
    config = worker(tmp_path, monkeypatch, arch, QWEN35, *args, *EAGER)
    for d in ("site", "planted", "bin"):
        (tmp_path / d).mkdir()
    (tmp_path / "site/sitecustomize.py").write_text(SITE)
    for name, source in PLANTED.items():
        (tmp_path / f"planted/{name}.py").write_text(source)
    (tmp_path / "bin/python3").write_text(
        f'#!/bin/sh\nPYTHONPATH={tmp_path}/site:$PYTHONPATH exec {sys.executable} "$@"\n'
    )
    # The import probe: this tree's compass_harness, as cctraces_sim.sh requires.
    (tmp_path / "bin/harness").write_text(
        '#!/bin/sh\necho "${PYTHONPATH%%:*}/compass_harness/x.py"\n'
    )
    for stub in (tmp_path / "bin").iterdir():
        stub.chmod(0o755)
    (tmp_path / "machine.json").write_text(json.dumps(machine(arch)))
    run_file = [] if run is None else [tmp_path / "run.json"]
    (tmp_path / "run.json").write_text(
        json.dumps({**(run or {}), "machine": machine(arch)})
    )
    (tmp_path / "traces.jsonl").write_text('{"id": 0}\n')
    inherited = {"CONFIG", "HIP_VISIBLE_DEVICES", "HTTP_INTAKE"}
    env = dict(
        {k: v for k, v in os.environ.items() if k not in inherited},
        PATH=f"{tmp_path}/bin:{os.environ['PATH']}",
        PLANTED_DIR=str(tmp_path / "planted"),
        PARSED=str(tmp_path / "parsed"),
        MODEL=config.model,
        SERVER_ARGS=" ".join(EAGER),
        WAIT_S="120",
        TRACES=str(tmp_path / "traces.jsonl"),
        HARNESS_PYTHON=str(tmp_path / "bin/harness"),
        MACHINE=str(tmp_path / "machine.json"),
        STORE=str(tmp_path / "store"),
    ) | {name: str(tmp_path / value) for name, value in more.items()}

    done = subprocess.run(
        ["bash", TREE / "scripts/compass" / script, *run_file, tmp_path / "out"],
        cwd="/",
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )

    parsed = tmp_path / "parsed"
    assert parsed.exists(), done.stdout + done.stderr
    assert set(parsed.read_text().split()) == {keyed(config)["quantization"]}
