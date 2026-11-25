# SPDX-License-Identifier: MIT
"""What a simulated Kimi-K3.1 start at TP8 DCP8 needs to build on no device.

Two places stopped it, each found starting the server on the CPU gate's image:
every rank aborted where a decode context-parallel MLA layer asks aiter whether
a persistent decode serves its gathered heads, since aiter's MLA planner reads
the device's arch in C++ and a failed `hipGetDeviceCount` aborts there; and the
post-load hooks refused the DSA layers' fused indexer projection, whose loader
records each checkpoint slice it took and whose hook refuses a load missing
one, where the meta build loads none.
"""

import subprocess
import sys

import pytest
import torch


@pytest.mark.skipif(torch.cuda.is_available(), reason="a driver answers its own arch")
def test_a_declared_process_asks_aiters_mla_planner_as_its_arch_and_lives():
    code = (
        "from atom.compass.drive.on_arch import declare\n"
        "declare('gfx950')\n"
        "from aiter.ops import attention\n"
        "from atom.distributed.dcp_utils import dcp_persistent_supported\n"
        "print(attention._mla_cpp_gpu_arch(0))\n"
        "print(dcp_persistent_supported(16, 'bf16', 8))\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-3000:]
    arch, supported = done.stdout.splitlines()[-2:]
    assert arch == "gfx950"
    assert supported in ("True", "False")


#: Run where aiter imports as a simulated server's does (`drive.on_arch`), which
#: `atom.models.deepseek_v2` needs.
WHOLE = """
from atom.compass.drive.on_arch import declare
declare("gfx950")
import torch
from atom.compass.runner.overrides import process_weights_after_loading
from atom.models.deepseek_v2 import FusedIndexerProjection, MergedReplicatedLinear

MergedReplicatedLinear.process_weights_after_loading = lambda self: print("parent")
projection = FusedIndexerProjection.__new__(FusedIndexerProjection)
torch.nn.Module.__init__(projection)
projection._checkpoint_shards, projection._require_checkpoint_shards = set(), True
try:
    projection.process_weights_after_loading()
except RuntimeError as refused:
    print(refused)
tree = torch.nn.Module()
tree.layer = torch.nn.Module()
tree.layer.fused_qkv_a_proj = projection
process_weights_after_loading(tree)
print(sorted(projection._checkpoint_shards))
"""


@pytest.mark.skipif(torch.cuda.is_available(), reason="a driver answers its own arch")
def test_a_meta_fused_indexer_projection_stands_for_a_whole_load():
    done = subprocess.run(
        [sys.executable, "-c", WHOLE],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-3000:]
    assert done.stdout.splitlines()[-3:] == [
        "Incomplete fused indexer projection: missing checkpoint slices [0, 1, 2, 3]",
        "parent",
        "[0, 1, 2, 3]",
    ]
