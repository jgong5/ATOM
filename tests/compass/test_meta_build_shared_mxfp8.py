# SPDX-License-Identifier: MIT
"""MiniMax-M3.1's Quark MXFP4 export builds on meta as it starts on the GPU.

Under ATOM_FUSE_MXFP8_SHARED_INTO_MXFP4 its MoE stages the MXFP8 shared expert
while loading and fuses it into the expert's MXFP4 slot after the load. A meta
build loads nothing: the staging is freed as the fuse frees it, and the slots
are the weights left. With a DSpark drafter, MiniMax's mono installer binds the
drafter into the model the runner holds, which on a simulated run is the
stand-in for the meta tree.
"""

import importlib
from types import SimpleNamespace

import pytest
import torch
from test_weights_meta_build import _built
from torch import nn

from atom.compass.runner.overrides import UnbuiltModel

H, I_LOCAL, ROUTED, SHARED, PAD = 64, 32, 2, 1, 16
E, HALF = ROUTED + SHARED, I_LOCAL + PAD
#: The bytes of the four MXFP4 buffers, the shared slot among them.
SLOTS = E * 2 * HALF * (H // 2 + H // 32) + E * H * (I_LOCAL // 2 + I_LOCAL // 32)
STAGING = "atom.model_ops.fused_moe.mxfp8_shared_into_mxfp4"


@pytest.fixture
def arch(monkeypatch):
    """The arch aiter's Triton probe answers, declared: no device to probe at
    import, and nothing here launches a kernel."""
    from aiter.ops.triton.utils._triton import arch_info

    monkeypatch.setattr(arch_info, "_CACHED_ARCH", arch_info._CACHED_ARCH or "gfx950")


@pytest.fixture
def moe(monkeypatch, arch):
    """ATOM's MoE module. It refuses any MoE backend but mega's after the
    fuse, which is not the one checked here."""
    moe = importlib.import_module("atom.model_ops.moe")
    monkeypatch.setattr(
        moe, "get_current_atom_config", lambda: SimpleNamespace(moe_backend="aiter")
    )
    return moe


def _slot(*shape):
    return nn.Parameter(torch.zeros(*shape, dtype=torch.uint8), requires_grad=False)


def _experts(moe):
    """An MXFP4 FusedMoE with one fused shared slot, its buffers laid out as the
    target lays them (w13 halves padded), its shared expert staged in MXFP8 on
    the device its buffers are on."""
    from aiter import QuantType, dtypes

    layer = object.__new__(moe.FusedMoE)
    nn.Module.__init__(layer)
    layer.prefix = layer.layer_name = "model.layers.3.block_sparse_moe.experts"
    layer.online_quant = False
    layer.layer_quant_config = SimpleNamespace(
        quant_type=QuantType.per_1x32, quant_dtype=dtypes.fp4x2
    )
    layer.expert_layout = SimpleNamespace(
        routed_physical_per_rank=ROUTED,
        num_fused_shared_experts=SHARED,
        shared_is_routed=False,
    )
    layer.w13_weight = _slot(E, 2 * HALF, H // 2)
    layer.w13_weight_scale = _slot(E, 2 * HALF, H // 32)
    layer.w2_weight = _slot(E, H, I_LOCAL // 2)
    layer.w2_weight_scale = _slot(E, H, I_LOCAL // 32)
    staging = importlib.import_module(STAGING)
    layer.shared_mxfp8_into_mxfp4 = staging.Mxfp8SharedIntoMxfp4(
        layer, None, H, I_LOCAL
    )
    return layer


def _holding(experts):
    """A model class whose tree holds `experts`, built by the call."""

    class Holding(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.experts = experts()

    return Holding


def test_a_shared_expert_staged_on_meta_fuses_in_place(monkeypatch, tmp_path, moe):
    runner = _built(monkeypatch, tmp_path, _holding(lambda: _experts(moe)))
    experts = runner.model.tree.experts
    assert experts.shared_mxfp8_into_mxfp4.staged is None
    assert experts.w13_weight.is_meta
    assert tuple(experts.w13_weight.shape) == (E, 2 * HALF, H // 2)
    weights = runner.compass_readings.peak_torch.terms[0]
    assert weights.nbytes == SLOTS
    fuse = importlib.import_module(STAGING).Mxfp8SharedIntoMxfp4.fuse
    assert not hasattr(fuse, "__wrapped__")


def test_a_staging_of_real_bytes_fuses_as_before(moe):
    """On the host the staging holds bytes, and ATOM's fuse runs: with no
    shared tensor loaded it refuses as on a real start."""
    from atom.compass.runner.overrides import shared_mxfp8_fused_on_meta

    staging = importlib.import_module(STAGING).Mxfp8SharedIntoMxfp4
    experts, fuse = _experts(moe), staging.fuse
    with shared_mxfp8_fused_on_meta():
        assert staging.fuse is not fuse
        with pytest.raises(RuntimeError, match="were not loaded from the checkpoint"):
            experts.shared_mxfp8_into_mxfp4.fuse(experts)
    assert staging.fuse is fuse
    assert experts.shared_mxfp8_into_mxfp4.staged is not None


def test_minimax_binds_its_drafter_into_the_meta_tree(arch):
    """ATOM's runner hands its installer the model it holds after the build;
    MiniMax's binds the DSpark drafter into the model's mono router."""
    from atom.models.registry import mono_decode_installers
    from atom.utils import resolve_obj_by_qualname

    bound = []

    class Tree(nn.Module):
        def bind_mono_drafter(self, drafter):
            bound.append(drafter)

    held = UnbuiltModel(Tree)
    held.tree = Tree()
    drafter = object()
    for architecture in (
        "MiniMaxM3SparseForConditionalGeneration",
        "MiniMaxM3SparseForCausalLM",
    ):
        install = resolve_obj_by_qualname(mono_decode_installers[architecture])
        assert install(held, None, drafter) is held
    assert bound == [drafter, drafter]
    held.tree = nn.Module()
    with pytest.raises(AttributeError, match="bind_mono_drafter"):
        install(held, None, drafter)
