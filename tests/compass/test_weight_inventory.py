"""The weight term counts what the engine builds, not what the file ships.

`weight_bytes` sums every safetensors header entry. A checkpoint may ship
tensors for a module the configured model class never constructs -- on the
27B, the ``mtp.*`` draft block -- and summing the headers counts them into the
memory budget, where they cost KV blocks that are never recovered.

These tests pin the mechanism rather than the 27B's number: a synthetic
checkpoint with one built tensor and one unbuilt one, and a model that builds
only the first. No device, no real checkpoint, no engine.
"""

import json
import os
import struct

import torch

from atom.compass.core.memory_model import resident_bytes, weight_bytes


def _write_shard(path, tensors):
    """One safetensors file with `tensors` as {name: (dtype, shape)}."""
    header, offset = {}, 0
    for name, (dtype, shape) in tensors.items():
        n = 1
        for s in shape:
            n *= s
        size = n * (2 if dtype in ("F16", "BF16") else 4)
        header[name] = {"dtype": dtype, "shape": list(shape),
                        "data_offsets": [offset, offset + size]}
        offset += size
    blob = json.dumps(header).encode()
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)
        fh.write(b"\0" * offset)
    return offset


def _checkpoint(tmp_path, tied=False):
    built = _write_shard(str(tmp_path / "model-00001-of-00002.safetensors"),
                         {"model.layers.0.weight": ("BF16", (64, 32))})
    unbuilt = _write_shard(str(tmp_path / "model-00002-of-00002.safetensors"),
                           {"mtp.fc.weight": ("BF16", (16, 32))})
    with open(os.path.join(str(tmp_path), "config.json"), "w") as fh:
        json.dump({"tie_word_embeddings": tied}, fh)
    # A sharded checkpoint is read through its index, as the real one is.
    with open(os.path.join(str(tmp_path),
                           "model.safetensors.index.json"), "w") as fh:
        json.dump({"metadata": {"total_size": built + unbuilt},
                   "weight_map": {
                       "model.layers.0.weight":
                           "model-00001-of-00002.safetensors",
                       "mtp.fc.weight":
                           "model-00002-of-00002.safetensors"}}, fh)
    return built, unbuilt


class _BuiltModel(torch.nn.Module):
    """Only the tensor the checkpoint's first shard carries."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.empty(64, 32, dtype=torch.bfloat16, device="meta"))


def test_headers_count_a_module_the_model_does_not_build(tmp_path):
    built, unbuilt = _checkpoint(tmp_path)
    headers = weight_bytes(str(tmp_path), 1)
    assert headers == built + unbuilt

    parameters, _ = resident_bytes(_BuiltModel())
    assert parameters == built

    # The whole of the difference is the unbuilt module, and nothing else:
    # no padding, no alias, no sharding rule is involved at TP=1.
    assert headers - parameters == unbuilt


def test_the_two_routes_agree_when_everything_is_built(tmp_path):
    built = _write_shard(str(tmp_path / "model.safetensors"),
                         {"model.layers.0.weight": ("BF16", (64, 32))})
    with open(os.path.join(str(tmp_path), "config.json"), "w") as fh:
        json.dump({"tie_word_embeddings": False}, fh)

    parameters, _ = resident_bytes(_BuiltModel())
    assert weight_bytes(str(tmp_path), 1) == built == parameters
