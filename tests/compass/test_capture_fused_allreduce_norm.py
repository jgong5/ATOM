# SPDX-License-Identifier: MIT
"""A capture's TP group traces ATOM's fused all-reduce norm as one leaf.

aiter asks the group for a device communicator before it dispatches the
per-group FP8 variant, and a capture declines the communicator; the norm still
dispatches aiter's registered op, whose fake answers.
"""

import os
import subprocess
import sys
import types

import torch
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from atom.compass.capture import driverless, forward

FP8_BLOCK = {
    "quant_method": "fp8",
    "activation_scheme": "dynamic",
    "fmt": "e4m3",
    "weight_block_size": [128, 128],
}


def test_the_per_group_fp8_fused_norm_is_one_leaf_at_tp2(request, tmp_path):
    if "COMPASS_ARCH_DECLARED" not in os.environ:
        # The declared device and group are process-global: a process of its own.
        env = {**os.environ, "COMPASS_ARCH_DECLARED": "1"}
        done = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"]
            + [f"{__file__}::{request.node.name}"],
            env=env,
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
        )
        assert done.returncode == 0, done.stdout[-4000:]
        return
    driverless.install(str(tmp_path), "gfx942")
    driverless.group(2, 1)
    from atom.config import QuantizationConfig
    from atom.model_ops.layernorm import RMSNorm

    hf = types.SimpleNamespace(
        torch_dtype=torch.bfloat16, quantization_config=FP8_BLOCK
    )
    norm = RMSNorm(
        256, fused_allreduce=True, fused_quant=True, quant_config=QuantizationConfig(hf)
    )
    recorder = forward._Recorder({}, driverless.driverless_mode(ShapeEnv()))
    with recorder.fake_mode, recorder:
        x = torch.empty(4, 256, dtype=torch.bfloat16, device="cuda")
        residual = torch.empty_like(x)
        recorder.on = True
        (out, scale), residual = norm(x, residual)
    assert [op.name for _, op in recorder.items] == [
        "aiter.fused_allreduce_rmsnorm_quant_per_group_.default"
    ]
    assert (out.shape, scale.shape, residual.shape) == ((4, 256), (4, 2), (4, 256))
