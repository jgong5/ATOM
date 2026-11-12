# SPDX-License-Identifier: MIT
"""A meta build re-quantizes a row-parallel weight online with no collective.

Under `--online_quant_config` ptpc_fp8, ATOM's `RowParallelLinear` gathers its
shard across TP before it quantizes per output row. On the meta device the
gather answers with its shape, and the weights left are this rank's FP8 shard
and its per-row scales.
"""

import torch
import torch.distributed as dist
from test_weights_meta_build import _built
from transformers import PretrainedConfig

TP, RANK, IN, OUT = 4, 1, 1024, 512
# The InferenceX MiniMax-M3 submission's `--online_quant_config`.
QUANT = {
    "global_quant_config": "ptpc_fp8",
    "exclude_layer": [
        "lm_head",
        "model.embed_tokens",
        "vision_tower",
        "multi_modal_projector",
        "patch_merge_mlp",
        "*block_sparse_moe",
    ],
}


def _group(monkeypatch):
    """aiter's TP coordinator at width TP, and the gathers it starts, which
    refuse a meta tensor as c10d's does."""
    from aiter.dist.parallel_state import GroupCoordinator

    group = GroupCoordinator.__new__(GroupCoordinator)
    group.world_size, group.rank_in_group, group.device_group = TP, RANK, None
    ran = []

    def all_gather_into_tensor(output, input, group=None, async_op=False):
        ran.append(tuple(input.shape))
        if input.is_meta:
            raise NotImplementedError("c10d::_allgather_base_ with Meta tensors")
        output.copy_(input.expand(output.shape))

    monkeypatch.setattr(dist, "all_gather_into_tensor", all_gather_into_tensor)
    return group, ran


def test_a_row_parallel_weight_quantizes_online_on_meta(monkeypatch, tmp_path):
    from aiter.ops.triton.utils._triton import arch_info

    # No device to probe at import; nothing here launches a kernel.
    monkeypatch.setattr(arch_info, "_CACHED_ARCH", arch_info._CACHED_ARCH or "gfx950")
    from atom.config import QuantizationConfig
    from atom.model_ops import linear

    group, ran = _group(monkeypatch)
    monkeypatch.setattr(linear, "get_tp_group", lambda: group)

    class OProj(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            quant = QuantizationConfig(PretrainedConfig(), online_quant_config=QUANT)
            self.o_proj = linear.RowParallelLinear(
                IN, OUT, quant_config=quant, prefix="model.layers.0.self_attn.o_proj"
            )

    runner = _built(monkeypatch, tmp_path, OProj)
    o_proj = runner.model.tree.o_proj
    assert ran == []
    assert o_proj.quant_type.name == "per_Token"
    assert tuple(o_proj.weight.shape) == (OUT, IN // TP)
    assert o_proj.weight.element_size() == 1
    weights = runner.compass_readings.peak_torch.terms[0]
    assert weights.nbytes == OUT * IN // TP + OUT * 4


def test_a_gather_of_real_bytes_still_runs_the_collective(monkeypatch):
    from aiter.dist.parallel_state import GroupCoordinator

    from atom.compass.runner.overrides import gathers_by_shape

    group, ran = _group(monkeypatch)
    real = GroupCoordinator.all_gather
    with gathers_by_shape():
        meta = group.all_gather(torch.empty(OUT, IN // TP, device="meta"), dim=1)
        host = group.all_gather(torch.ones(OUT, IN // TP), dim=1)
    assert (meta.is_meta, tuple(meta.shape)) == (True, (OUT, IN))
    assert tuple(host.shape) == (OUT, IN) and ran == [(OUT, IN // TP)]
    assert GroupCoordinator.all_gather is real
