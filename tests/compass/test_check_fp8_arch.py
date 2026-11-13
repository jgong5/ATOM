# SPDX-License-Identifier: MIT
"""`compass run --check` keys an FP8 quantization as a worker on its machine's
device does, wherever it runs. aiter resolves its fp8 dtype at import from the
device it finds, ``float8_e4m3fn`` on gfx950, ``float8_e4m3fnuz`` on gfx942
and ``uint8`` where it finds none; ATOM's parser holds that dtype in the
quantization config, whose hash keys a run's memory readings.

The worker's config is parsed here with aiter's arch probe and fp8 dtype
answering its arch, as on its device; `compass measure memory` publishes the
readings under its key.

No device: the check declares its machine's.
"""

import json

import pytest
import torch
from test_drive_run import stub
from test_key_index_cache_dtype import served
from test_key_quantization_remap import M3_VL, QWEN35
from test_memory_readings import DOCUMENT, publish_readings
from test_run_lacking import lacked

from atom.compass.drive import run as drive_run

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


def machine(arch):
    """DOCUMENT, its device the one of `arch`."""
    device = {**DOCUMENT["device"], "name": DEVICES[arch], "arch": arch}
    return {**DOCUMENT, "device": device}


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
    with monkeypatch.context() as device:
        from aiter.utility import dtypes

        device.setitem(dtypes.d_dtypes, "fp8", FP8[arch])
        worker = served(tmp_path, device, arch, config_json, *args)
    publish_readings(tmp_path / "store", worker, document)
    (tmp_path / "machine.json").write_text(json.dumps(document))
    flags = ["--step-stub", stub(tmp_path), "--machine", str(tmp_path / "machine.json")]
    flags += ["--store", str(tmp_path / "store"), "--check"]

    with pytest.raises(SystemExit) as exited:
        drive_run.main([*flags, "--", "--model", worker.model, *args])

    assert lacked(capsys.readouterr().out) == []
    assert exited.value.code == 0
