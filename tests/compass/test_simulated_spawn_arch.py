# SPDX-License-Identifier: MIT
"""A simulated server's engines and workers start as on its machine's device,
on a host with no driver.

ATOM's API server spawns its engine cores, and each engine core its workers;
each imports aiter again, whose Triton helpers probe the device for its arch.
`serve` starts the server under `on_arch`, and every process below it declares
the same arch before aiter loads. The planted server spawns as ATOM's does, two
levels deep, and each level records the arch aiter's probe answers.

A worker's meta build asks `torch.cuda` for the current device and whether its
stream captures (DeepSeek-V4's HCA preparation); under a run file it is
answered as the simulated device, and `torch.cuda` is the host's again after.

No device: each declares its machine's.
"""

import textwrap

import pytest
import torch
from test_check_fp8_arch import FP8, machine
from test_drive_serve import PLANTED, planted  # noqa: F401  a fixture
from test_drive_trace_steps import plant_module
from test_weights_meta_build import TiedPacked, _built

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


@pytest.mark.parametrize("arch", sorted(FP8))
def test_a_simulated_server_s_spawned_processes_probe_its_machine_s_arch(
    planted, monkeypatch, arch  # noqa: F811
):
    monkeypatch.setenv("PLANTED_EXIT", "late:0")
    plant_module(planted, monkeypatch, SPAWNS + textwrap.indent(PLANTED, "    "))
    (planted / "planted_child.py").write_text(CHILD)
    run = {"bound_s": 1.0, "machine": machine(arch)}

    serve.serve(["--model", "m"], planted / "out", run=run)

    levels = sorted(planted.glob("arch*"))
    assert [p.name for p in levels] == ["arch0", "arch1", "arch2"]
    assert {p.read_text() for p in levels} == {arch}


def test_a_meta_build_asks_torch_cuda_about_the_simulated_device(monkeypatch, tmp_path):
    host = torch.cuda.current_device
    seen = []

    class AsksTheDevice(TiedPacked):
        def __init__(self, config):
            super().__init__(config)
            seen.append(
                (
                    torch.cuda.current_device(),
                    torch.cuda.is_current_stream_capturing(),
                    torch.cuda.get_device_properties(0).gcnArchName,
                )
            )

    _built(monkeypatch, tmp_path, AsksTheDevice)
    assert seen == [(0, False, "gfx950")]
    assert torch.cuda.current_device is host
