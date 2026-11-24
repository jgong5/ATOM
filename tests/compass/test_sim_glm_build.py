# SPDX-License-Identifier: MIT
"""What a simulated GLM-5.3-Flash start at TP4 with MTP3 needs to build on no device.

One place stopped it, found starting the server on the CPU gate's image: the
post-load hooks refused every KDA layer, whose fused input projection records
each of the q, k and v shards its loader took and whose hook refuses a load
missing one, where the meta build loads none. The hook then folds the
low-rank output gate into the projection's fourth shard, which runs on meta as
on the GPU.
"""

import subprocess
import sys

import pytest
import torch

#: Run where aiter imports as a simulated server's does (`drive.on_arch`), which
#: `atom.models.glm5_next` needs.
WHOLE = """
from types import SimpleNamespace
from atom.compass.drive.on_arch import declare
declare("gfx950")
import torch
from atom.compass.runner.overrides import process_weights_after_loading
from atom.config import use_custom_atom_config
from atom.models.glm5_next import Glm5NextKDAAttention
from atom.models.kimi_k3 import KimiKDAAttention

KimiKDAAttention.process_weights_after_loading = lambda self: print("parent")
attention = Glm5NextKDAAttention.__new__(Glm5NextKDAAttention)
torch.nn.Module.__init__(attention)
attention._loaded_input_shards = set()
attention.local_proj_size, attention.hidden_size = 4, 8
with torch.device("meta"):
    attention.in_proj = torch.nn.Linear(8, 16, bias=False)
    attention.g_a_proj = torch.nn.Linear(8, 2, bias=False)
    attention.g_b_proj = torch.nn.Linear(2, 4, bias=False)
tree = torch.nn.Module()
tree.layer = torch.nn.Module()
tree.layer.self_attn = attention
with use_custom_atom_config(SimpleNamespace(load_dummy=None)):
    try:
        attention.process_weights_after_loading()
    except RuntimeError as refused:
        print(refused)
    process_weights_after_loading(tree)
print(sorted(attention._loaded_input_shards))
print(attention.g_a_proj.weight.numel(), attention.g_b_proj.weight.numel())
"""


@pytest.mark.skipif(torch.cuda.is_available(), reason="a driver answers its own arch")
def test_a_meta_glm_kda_projection_stands_for_a_whole_load_and_folds_its_gate():
    done = subprocess.run(
        [sys.executable, "-c", WHOLE],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-3000:]
    refused = "Incomplete GLM-5.3 KDA input projection: missing checkpoint shards"
    assert done.stdout.splitlines()[-4:] == [
        f"{refused} [0, 1, 2] from q/k/v",
        "parent",
        "[0, 1, 2]",
        "0 0",
    ]
