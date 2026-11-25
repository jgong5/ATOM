# SPDX-License-Identifier: MIT
"""What a simulated Qwen3.8-Flash-Next start needs to build its PLE n-gram table on no device.

Found starting the server on the CPU image: the post-load hook of the FP8
n-gram table checks the checkpoint's one global scale with `Tensor.item()`,
which refuses on meta. And with the factories on meta, ATOM's placement check
(`_host_table_available`) built the table in HBM, ~51 GB of FP8 at TP1, where
a gfx950 start page-locks it on the host: the simulated KV pool came out that
much smaller than the real one.
"""

import subprocess
import sys

import pytest
import torch

#: Run where aiter imports as a simulated server's does (`drive.on_arch`), which
#: `atom.model_ops.qwen4_exp.ple_layer` needs.
PLACED = """
import os
from types import SimpleNamespace
from atom.compass.drive.on_arch import declare
declare("gfx950")
from atom.compass.runner import overrides
from atom.config import use_custom_atom_config
from atom.model_ops.qwen4_exp import ple_layer

check = ple_layer._host_table_available
with use_custom_atom_config(SimpleNamespace(load_dummy=None)):
    with overrides.FactoriesOnMeta():
        print(check())
        with overrides.ple_table_placed_on_device():
            print(ple_layer._host_table_available())
            os.environ["ATOM_ENGRAM_UVA"] = "0"
            print(ple_layer._host_table_available())
            del os.environ["ATOM_ENGRAM_UVA"]
with use_custom_atom_config(SimpleNamespace(load_dummy=True)):
    with overrides.FactoriesOnMeta(), overrides.ple_table_placed_on_device():
        print(ple_layer._host_table_available())
print(ple_layer._host_table_available is check)
"""

#: The two tables' hooks: on meta skipped, on the host ATOM's.
HOOKS = """
from atom.compass.drive.on_arch import declare
declare("gfx950")
import torch
from torch import nn
from atom.compass.runner.overrides import process_weights_after_loading
from atom.model_ops.qwen4_exp import ple_layer


def table(cls, device):
    t = cls.__new__(cls)
    nn.Module.__init__(t)
    t.prefix, t.fp8 = "model.layers.2.ple.ngram_embedding", True
    t.weight = nn.Parameter(
        torch.empty(4, 8, dtype=torch.float8_e4m3fn, device=device), requires_grad=False
    )
    t.weight_scale = nn.Parameter(
        torch.full((1,), float("nan"), device=device), requires_grad=False
    )
    return t


for cls in (ple_layer._Qwen4ExpFP8Embedding, ple_layer._Qwen4ExpHostEmbedding):
    hook = cls.process_weights_after_loading
    meta = table(cls, "meta")
    try:
        meta.process_weights_after_loading()
    except RuntimeError as refused:
        print(type(refused).__name__)
    tree = nn.Module()
    tree.ple = meta
    process_weights_after_loading(tree)
    print(cls.process_weights_after_loading is hook)
    tree.ple = table(cls, "cpu")
    try:
        process_weights_after_loading(tree)
    except Exception as refused:
        print("nan" in str(refused) or "not finite" in str(refused))
"""


def _run(script):
    done = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-3000:]
    return done.stdout.splitlines()


@pytest.mark.skipif(torch.cuda.is_available(), reason="a driver answers its own arch")
def test_the_ple_table_is_placed_as_on_the_simulated_device():
    """On meta factories ATOM's check says HBM; placed as on gfx950 it says
    host, but where ATOM_ENGRAM_UVA=0 or a dummy load keeps it in HBM; the check
    is ATOM's own again after the build."""
    assert _run(PLACED)[-5:] == ["False", "True", "False", "False", "True"]


@pytest.mark.skipif(torch.cuda.is_available(), reason="a driver answers its own arch")
def test_a_meta_ple_table_stands_for_a_whole_load():
    """A meta table's scale check refuses on its own and passes in the meta
    build's post-load walk; a table on the host still has its NaN scale
    refused, and each class gets its own hook back."""
    lines = _run(HOOKS)
    assert lines[-6:] == ["RuntimeError", "True", "True"] * 2
