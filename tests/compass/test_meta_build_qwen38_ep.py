# SPDX-License-Identifier: MIT
"""What a simulated Qwen3.8-Flash-Next start under expert parallelism needs to build on no device.

Found starting TP2 and TP4 with --enable-expert-parallel on the CPU image: the
sparse MoE block wraps its experts' weight loaders as it builds, and first
reads this rank's expert range off the MoE's `expert_map` with
`torch.nonzero`, which refuses a meta map. The loaders only read a checkpoint;
on a meta map nothing is installed, on any other ATOM's installer runs.
"""

import subprocess
import sys

import pytest
import torch

INSTALL = """
from types import SimpleNamespace
from atom.compass.drive.on_arch import declare
declare("gfx950")
import torch
from torch import nn
from atom.compass.runner.overrides import stacked_experts_unloaded_on_meta
from atom.models import qwen4_exp


def experts(device):
    e = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(tp_rank=0),
        expert_layout=SimpleNamespace(num_routed=4),
        expert_map=torch.tensor([-1, -1, 0, 1, -1], dtype=torch.int32, device=device),
        w13_weight=nn.Parameter(torch.zeros(2, 4, 3, device=device), requires_grad=False),
        w2_weight=nn.Parameter(torch.zeros(2, 3, 2, device=device), requires_grad=False),
    )
    e.w13_weight.weight_loader = e.w2_weight.weight_loader = stock
    return e


def stock(*args, **kwargs):
    pass


meta = experts("meta")
try:
    qwen4_exp.install_stacked_expert_loaders(meta)
except NotImplementedError:
    print("refused")
stacked_experts_unloaded_on_meta()
stacked_experts_unloaded_on_meta()  # a second call wraps nothing more
meta = experts("meta")
print(qwen4_exp.install_stacked_expert_loaders(meta), meta.w13_weight.weight_loader is stock)
host = experts("cpu")
qwen4_exp.install_stacked_expert_loaders(host)
print(host.w13_weight.weight_loader is stock)
print(qwen4_exp.install_stacked_expert_loaders.__wrapped__.__wrapped__ if hasattr(
    qwen4_exp.install_stacked_expert_loaders.__wrapped__, "__wrapped__") else "once")
"""


@pytest.mark.skipif(torch.cuda.is_available(), reason="a driver answers its own arch")
def test_a_meta_expert_map_installs_no_stacked_loader():
    done = subprocess.run(
        [sys.executable, "-c", INSTALL],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-3000:]
    assert done.stdout.splitlines()[-4:] == ["refused", "None True", "False", "once"]
