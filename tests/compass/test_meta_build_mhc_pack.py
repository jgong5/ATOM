# SPDX-License-Identifier: MIT
"""A meta build of DeepSeek-V4.1 keeps the mHC packs a real gfx950 start keeps.

After load each V4.1 block packs its two mHC projections into buffers of its
own (`pack_mhc_fn`) on a gfx950 GPU, and ATOM's pack keeps nothing for a meta
weight, which is on no GPU. A real DTP2 start therefore held more buffer bytes
than the simulated one counted, and the simulated KV pool came out larger.

The published V4.1-Flash config builds at DTP2 with the submission's KV and
index planes, on a declared gfx950 with no driver, its post-load hooks run as a
simulated run's are (`validate.bound._build`). A meta device's properties are
refused, as torch refuses them on a host with a GPU, so the arch the pack sees
is the declared device's. The build owns its process because the driverless
declarations are process-global; each backend `ATOM_DSV41_MHC_BACKEND` names
builds once.
"""

import json
import os
import pathlib
import subprocess
import sys

import pytest

TREE = pathlib.Path(__file__).resolve().parents[2]
CONFIG = TREE / "tests" / "models" / "deepseek_v41" / "fixtures" / "config.json"

BUILD = """
import json, sys
import torch
from atom.compass.capture import driverless

install = driverless.install


def on_a_host(*args, **kwargs):
    said = install(*args, **kwargs)
    declared = torch.cuda.get_device_properties

    def properties(device=None):
        if device is not None and torch.device(device).type != "cuda":
            raise ValueError(f"Expected a cuda device, but got: {device}")
        return declared(device)

    torch.cuda.get_device_properties = properties
    return said


driverless.install = on_a_host
from atom.compass.validate.bound import _build

args = ["--enable-dp-attention", "--kv_cache_dtype", "fp8", "--index-cache-dtype",
        "fp4", "--method", "dspark", "--num-speculative-tokens", "5"]
config, tree = _build(sys.argv[1] + "/model", 2, args, "gfx950", sys.argv[1])
from atom.compass.backends.provenance import Method
from atom.compass.memory.readings import ModelTerms
from atom.compass.memory.terms import Term
from atom.model_ops.deepseek_v41 import mhc_pre_delayed
from atom.models.deepseek_v41 import model

none = Term("activations", 0, Method.MEASURED, "none")
buffers = ModelTerms.from_tree(tree, activations=none).buffers
layers = []
for layer in tree.layers:
    held = {}
    for sublayer in ("attn", "ffn"):
        fn = getattr(layer, f"hc_{sublayer}_fn")
        pack = getattr(layer, f"hc_{sublayer}_fn_packed")
        held[sublayer] = [fn.untyped_storage().nbytes()] + (
            [] if pack is None
            else [str(pack.dtype), pack.is_meta, pack.untyped_storage().nbytes()]
        )
    layers.append(held)
print(json.dumps({
    "dp": config.parallel_config.data_parallel_size,
    "buffers": [buffers.nbytes, buffers.source],
    "layers": layers,
    "put_back": [
        not hasattr(mhc_pre_delayed.pack_mhc_fn, "__wrapped__"),
        model.pack_mhc_fn is mhc_pre_delayed.pack_mhc_fn,
        torch.cuda.get_device_properties.__name__ == "properties",
    ],
}))
"""


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """Each backend's build, by the backend's name."""
    root = tmp_path_factory.mktemp("v41")
    (root / "model").mkdir()
    (root / "model" / "config.json").write_bytes(CONFIG.read_bytes())
    out = {}
    for backend in ("auto", "triton"):
        done = subprocess.run(
            [sys.executable, "-c", BUILD, str(root)],
            capture_output=True,
            text=True,
            check=False,
            timeout=900,
            cwd=TREE,
            env={
                **os.environ,
                "PYTHONPATH": str(TREE),
                "ATOM_DSV41_MHC_BACKEND": backend,
            },
        )
        assert done.returncode == 0, done.stderr[-4000:]
        out[backend] = json.loads(done.stdout.splitlines()[-1])
    return out


def test_each_v41_block_keeps_both_mhc_packs_on_meta_as_a_gfx950_start(built):
    """Each pack is the HIP layout's int32 tensor, the projection's FP32 bytes
    as BF16 high and low halves, on meta; the build puts back the pack and the
    device properties after it."""
    done = built["auto"]
    text = json.loads(CONFIG.read_text())["text_config"]
    assert done["dp"] == 2
    assert len(done["layers"]) == text["num_hidden_layers"]
    projection = 24 * text["hc_mult"] * text["hidden_size"] * 4
    for held in done["layers"]:
        assert held == {s: [projection, "torch.int32", True, projection] for s in held}
    assert done["put_back"] == [True, True, True]


def test_the_buffers_count_each_pack_only_where_atom_packs(built):
    """Under the Triton backend ATOM packs nothing, nor does its meta build;
    under its default each pack is one more buffer storage, of its
    projection's bytes."""
    auto, triton = built["auto"], built["triton"]
    text = json.loads(CONFIG.read_text())["text_config"]
    packs = 2 * text["num_hidden_layers"]
    projection = 24 * text["hc_mult"] * text["hidden_size"] * 4
    assert all(len(v) == 1 for held in triton["layers"] for v in held.values())
    stored = [int(t["buffers"][1].split()[0]) for t in (auto, triton)]
    assert stored[0] - stored[1] == packs
    assert auto["buffers"][0] - triton["buffers"][0] == packs * projection
