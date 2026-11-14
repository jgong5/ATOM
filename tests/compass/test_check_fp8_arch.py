# SPDX-License-Identifier: MIT
"""`compass run --check` keys an FP8 quantization as a worker on its machine's
device does, wherever it runs. aiter resolves its fp8 dtype at import from the
device it finds, ``float8_e4m3fn`` on gfx950, ``float8_e4m3fnuz`` on gfx942
and ``uint8`` where it finds none; ATOM's parser holds that dtype in the
quantization config, whose hash keys a run's memory readings.

The worker's config is parsed here with aiter's arch probe and fp8 dtype
answering its arch, as on its device; `compass measure memory` publishes the
readings under its key.

The check builds its config where its machine's device is declared, a process
of its own, so a config class a model's remote code defines comes back from
there.

No device: the check declares its machine's.
"""

import json
import os
import pathlib
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from test_drive_run import recorded, stub
from test_key_index_cache_dtype import served
from test_key_quantization_remap import M3_VL, QWEN35
from test_memory_readings import DOCUMENT, publish_readings
from test_run_lacking import lacked
from test_run_step_grid import KEY, publish
from test_step_grid_plan import LADDER, MIXED, located, measured, plan

from atom.compass.artifacts import Key, Kind
from atom.compass.drive import run as drive_run
from atom.compass.drive import serve

#: MiniMax-M3's online FP8 quantization, as its MI355X submission serves it.
PTPC_FP8 = (
    '{"global_quant_config":"ptpc_fp8","exclude_layer":["lm_head",'
    '"model.embed_tokens","vision_tower","multi_modal_projector",'
    '"patch_merge_mlp","*block_sparse_moe"]}'
)
#: aiter's fp8 dtype on each arch's device.
FP8 = {"gfx950": torch.float8_e4m3fn, "gfx942": torch.float8_e4m3fnuz}
#: A device of each arch.
DEVICES = {"gfx950": "MI355X", "gfx942": "MI308X"}
TREE = pathlib.Path(__file__).resolve().parents[2]
QWEN = pathlib.Path(__file__).with_name("qwen3_0_6b_config.json")
#: The config class of a checkpoint that ships its own, which ATOM loads only
#: under --trust-remote-code.
REMOTE = (
    "from transformers import Qwen3Config\n\n\n"
    "class TinyConfig(Qwen3Config):\n"
    '    model_type = "tiny_remote"\n'
)


def machine(arch):
    """DOCUMENT, its device the one of `arch`."""
    device = {**DOCUMENT["device"], "name": DEVICES[arch], "arch": arch}
    return {**DOCUMENT, "device": device}


def worker(tmp_path, monkeypatch, arch, config_json, *args):
    """The config a server of `config_json`'s model parses under ATOM's
    arguments `args` on a device of `arch`."""
    with monkeypatch.context() as device:
        from aiter.utility import dtypes

        device.setitem(dtypes.d_dtypes, "fp8", FP8[arch])
        return served(tmp_path, device, arch, config_json, *args)


@pytest.mark.parametrize(
    "model, args, arch",
    [
        ("MiniMax-M3", ["-tp", "4", "--online_quant_config", PTPC_FP8], "gfx950"),
        ("Qwen3.5", ["-tp", "4"], "gfx950"),
        ("MiniMax-M3", ["-tp", "4", "--online_quant_config", PTPC_FP8], "gfx942"),
    ],
    ids=["m3-ptpc-fp8-gfx950", "qwen3.5-fp8-gfx950", "m3-ptpc-fp8-gfx942"],
)
def test_a_check_finds_the_readings_its_machine_s_worker_keys(
    tmp_path, monkeypatch, capsys, model, args, arch
):
    config_json = QWEN35
    if model == "MiniMax-M3":
        config_json = tmp_path / "m3.json"
        config_json.write_text(json.dumps(M3_VL))
    document = machine(arch)
    config = worker(tmp_path, monkeypatch, arch, config_json, *args)
    publish_readings(tmp_path / "store", config, document)
    (tmp_path / "machine.json").write_text(json.dumps(document))
    flags = ["--step-stub", stub(tmp_path), "--machine", str(tmp_path / "machine.json")]
    flags += ["--store", str(tmp_path / "store"), "--check"]

    with pytest.raises(SystemExit) as exited:
        drive_run.main([*flags, "--", "--model", config.model, *args])

    assert lacked(capsys.readouterr().out) == []
    assert exited.value.code == 0


def test_a_check_s_coverage_loads_the_bundle_its_machine_s_worker_keys(
    tmp_path, monkeypatch, capsys
):
    # The record's engine: `test_step_grid_plan`'s bounds and ladder at TP 1.
    directory = recorded(tmp_path, monkeypatch, MIXED)
    planned = plan(directory / serve.STEPS)
    member = measured(planned, located(tmp_path, planned))
    args = ["--online_quant_config", '{"global_quant_config":"ptpc_fp8"}']
    args += ["--max-num-seqs", "8", "--max-num-batched-tokens", "64"]
    args += ["--max-model-len", "2048", "--cudagraph-capture-sizes", str(LADDER)]
    config = worker(tmp_path, monkeypatch, "gfx950", QWEN, *args)
    publish(tmp_path, SimpleNamespace(config=config), member=member)
    publish_readings(tmp_path / "store", config)
    (tmp_path / "machine.json").write_text(json.dumps(DOCUMENT))
    token = Key.of(Kind.STEP_GRID_BUNDLE, **KEY).token()
    flags = ["--step-grid", token, "--machine", str(tmp_path / "machine.json")]
    flags += ["--store", str(tmp_path / "store"), "--coverage", str(directory)]

    with pytest.raises(SystemExit) as exited:
        drive_run.main([*flags, "--check", "--", "--model", config.model, *args])

    out = capsys.readouterr().out
    assert lacked(out) == [], out
    assert f"apart: coverage of {directory}: " in out
    assert exited.value.code == 0


def test_a_check_reads_the_config_of_a_model_with_remote_code(tmp_path):
    model = tmp_path / "tiny"
    model.mkdir()
    config = json.loads(QWEN.read_text()) | {"model_type": "tiny_remote"}
    config["auto_map"] = {"AutoConfig": "configuration_tiny.TinyConfig"}
    (model / "config.json").write_text(json.dumps(config))
    (model / "configuration_tiny.py").write_text(REMOTE)
    (tmp_path / "machine.json").write_text(json.dumps(DOCUMENT))
    flags = ["--step-stub", stub(tmp_path), "--machine", str(tmp_path / "machine.json")]
    flags += ["--store", str(tmp_path / "store"), "--check"]
    atom = ["--model", str(model), "--trust-remote-code"]
    # A command of its own, as the check runs: no remote code is loaded in it.
    cli = [sys.executable, "-c", "from atom.compass.cli import main; main()", "run"]
    env = dict(os.environ, PYTHONPATH=str(TREE))
    env["HF_MODULES_CACHE"] = str(tmp_path / "modules")

    done = subprocess.run(
        [*cli, *flags, "--", *atom],
        check=False,
        env=env,
        capture_output=True,
        text=True,
    )

    assert lacked(done.stdout) == ["memory readings: missing"], done.stderr[-3000:]
