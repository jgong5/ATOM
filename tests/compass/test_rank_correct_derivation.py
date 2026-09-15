# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Derivation on a rank that is not zero, and back again.

Simulated TP made the group *report* a logical width while one process held
one shard, and that was enough for shapes: every shard-size computation reads
`world_size`. It was not enough for rank. `rank_in_group` stayed 0, so a
TP4 derivation produced rank 0's model four times and stamped four different
`rank_coords` on it.

Two seams, tested separately because they fail separately:

* `_patch_group` reports a logical rank as well as a logical width, and can be
  undone, so one process can hold rank 0, then rank 1, then rank 0 again with
  nothing of the middle one left over.
* `rebind_logical_rank` moves an already-built model onto another rank --
  necessary because a process may only build one model -- and refuses any
  module whose `forward` reads a rank-derived value it cannot recompute.
"""

import sys
import types

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

from atom.compass.runtime.derive import (
    RankRebindRefusal,
    rebind_logical_rank,
    simulate_group_width,
)
from atom.distributed.simulated_tp import _patch_group, restore_group

LOGICAL = 4


def _unpatched(*args, **kwargs):
    raise AssertionError("collective was not replaced by _patch_group")


def _stub_group():
    grp = types.SimpleNamespace(
        device_group=dist.group.WORLD,
        rank_in_group=0,
        ranks=[0],
        world_size=1,
        all_reduce=_unpatched,
    )
    return grp


@pytest.fixture
def group():
    if not dist.is_available():
        pytest.skip("torch.distributed is unavailable")
    created = not dist.is_initialized()
    if created:
        dist.init_process_group(
            backend="gloo", init_method="tcp://127.0.0.1:29593",
            world_size=1, rank=0)
    yield _stub_group()
    if created:
        dist.destroy_process_group()


# --------------------------------------------------------------------------
# The group seam.


def test_the_group_reports_the_rank_it_was_asked_for(group):
    _patch_group(group, LOGICAL, 1, logical_rank=2)
    assert group.world_size == LOGICAL
    assert group.rank_in_group == 2
    assert group.simulated_tp_logical_rank == 2


def test_the_shard_this_rank_keeps_is_its_own(group):
    # reduce_scatter hands each rank the slice at its own offset. Rank 2 of 4
    # keeps rows 4:6 of an 8-row reduction, not rows 0:2.
    _patch_group(group, LOGICAL, 1, logical_rank=2)
    x = torch.arange(8, dtype=torch.float32).unsqueeze(1)
    assert torch.equal(group.reduce_scatter_tensor(x, dim=0),
                       torch.tensor([[4.0], [5.0]]))


def test_the_gathered_result_holds_this_rank_at_its_own_position(group):
    _patch_group(group, LOGICAL, 1, logical_rank=1)
    x = torch.ones(2, 3)
    out = group.all_gather(x, dim=0)
    assert out.shape == (8, 3)
    # Rows 2:4 are ours; the rest are the absent ranks, which read as zero.
    assert out[2:4].eq(1).all() and out[:2].eq(0).all() and out[4:].eq(0).all()


def test_a_simulated_rank_needs_a_group_with_no_real_peers(group):
    # With peers present, `rank_in_group` is this process's address inside a
    # real collective; lying about it sends the message to the wrong rank.
    with pytest.raises(ValueError, match="physical ranks"):
        _patch_group(group, LOGICAL, 2, logical_rank=1)


def test_a_rank_outside_the_group_is_refused(group):
    with pytest.raises(ValueError, match="outside a TP4 group"):
        _patch_group(group, LOGICAL, 1, logical_rank=4)


def test_restore_puts_the_group_back_exactly(group):
    _patch_group(group, LOGICAL, 1, logical_rank=3)
    restore_group(group)
    assert group.world_size == 1
    assert group.rank_in_group == 0
    assert group.all_reduce is _unpatched
    assert not hasattr(group, "simulated_tp_logical_rank")
    # Idempotent: a second restore is not an error and changes nothing.
    restore_group(group)
    assert group.world_size == 1


def test_rank0_then_rank1_then_rank0_leaves_nothing_of_rank1(group):
    x = torch.arange(8, dtype=torch.float32).unsqueeze(1)

    _patch_group(group, LOGICAL, 1, logical_rank=0)
    first = group.reduce_scatter_tensor(x, dim=0).clone()
    first_gather = group.all_gather(torch.ones(2, 3), dim=0).clone()

    _patch_group(group, LOGICAL, 1, logical_rank=1)
    assert group.rank_in_group == 1
    assert not torch.equal(group.reduce_scatter_tensor(x, dim=0), first)

    _patch_group(group, LOGICAL, 1, logical_rank=0)
    assert group.rank_in_group == 0
    assert group.world_size == LOGICAL
    # Not "close to": the same process must give rank 0 the same answer the
    # second time, or the middle rank left something behind.
    assert torch.equal(group.reduce_scatter_tensor(x, dim=0), first)
    assert torch.equal(group.all_gather(torch.ones(2, 3), dim=0), first_gather)


# --------------------------------------------------------------------------
# The model seam.


class VocabLike(nn.Module):
    """Shaped like `VocabParallelEmbedding`: rank enters the forward as bounds."""

    def __init__(self, per_partition, rank):
        super().__init__()
        self.tp_rank = rank
        self.num_embeddings_per_partition = per_partition
        self.vocab_start_idx = per_partition * rank
        self.vocab_end_idx = self.vocab_start_idx + per_partition

    def forward(self, x):
        return x + self.vocab_start_idx


class LoaderOnly(nn.Module):
    """Shaped like the parallel linears: rank is read by the weight loader."""

    def __init__(self, rank):
        super().__init__()
        self.tp_rank = rank

    def weight_loader(self, param, loaded):
        return loaded.narrow(0, self.tp_rank, 1)

    def forward(self, x):
        return x * 2


class ReadsRankInForward(nn.Module):
    def __init__(self, rank):
        super().__init__()
        self.tp_rank = rank

    def forward(self, x):
        return x + self.tp_rank


class Model(nn.Module):
    def __init__(self, extra=None):
        super().__init__()
        self.embed = VocabLike(32, 0)
        self.proj = LoaderOnly(0)
        self.norm = nn.LayerNorm(4)  # no rank at all
        if extra is not None:
            self.odd = extra


@pytest.fixture
def fake_tp_group(group, monkeypatch):
    """`simulate_group_width` reaching a stub group instead of aiter's."""
    parallel_state = types.ModuleType("aiter.dist.parallel_state")
    parallel_state.get_tp_group = lambda: group
    dist_mod = types.ModuleType("aiter.dist")
    dist_mod.parallel_state = parallel_state
    aiter_mod = sys.modules.get("aiter") or types.ModuleType("aiter")
    monkeypatch.setitem(sys.modules, "aiter", aiter_mod)
    monkeypatch.setitem(sys.modules, "aiter.dist", dist_mod)
    monkeypatch.setitem(sys.modules, "aiter.dist.parallel_state", parallel_state)
    yield group
    restore_group(group)


def test_rebinding_moves_the_vocab_bounds_and_the_group_together(fake_tp_group):
    model = Model()
    moved = rebind_logical_rank(model, 3, logical=LOGICAL)
    assert moved == 2  # the vocab module and the loader-only one
    assert (model.embed.vocab_start_idx, model.embed.vocab_end_idx) == (96, 128)
    assert model.proj.tp_rank == 3
    # The group has to agree, or the next layer built or collective performed
    # would answer for the old rank.
    assert fake_tp_group.rank_in_group == 3


def test_rebinding_back_restores_rank_zeros_own_bounds(fake_tp_group):
    model = Model()
    before = (model.embed.vocab_start_idx, model.embed.vocab_end_idx)
    rebind_logical_rank(model, 1, logical=LOGICAL)
    rebind_logical_rank(model, 0, logical=LOGICAL)
    assert (model.embed.vocab_start_idx, model.embed.vocab_end_idx) == before
    assert model.proj.tp_rank == 0
    assert fake_tp_group.rank_in_group == 0


def test_a_module_that_reads_its_rank_in_forward_is_refused_by_name(
        fake_tp_group):
    model = Model(extra=ReadsRankInForward(0))
    with pytest.raises(RankRebindRefusal, match="odd \\(ReadsRankInForward\\)"):
        rebind_logical_rank(model, 2, logical=LOGICAL)


def test_a_rank_outside_the_width_is_refused(fake_tp_group):
    with pytest.raises(ValueError, match="outside a TP4 group"):
        simulate_group_width(LOGICAL, 1, rank=7)


def test_an_unsimulated_group_cannot_be_asked_for_another_rank(fake_tp_group):
    # A TP1 group has no second rank to be.
    with pytest.raises(ValueError, match="outside a TP1 group"):
        simulate_group_width(1, 1, rank=1)
    # A group whose ranks are all really there is whichever rank it really is:
    # nothing is being simulated, so there is no shard to move onto.
    with pytest.raises(RuntimeError, match="not simulated"):
        simulate_group_width(2, 2, rank=1)


# --------------------------------------------------------------------------
# The served path.


def test_the_tracer_moves_the_model_before_it_traces(fake_tp_group):
    from atom.compass.runtime.tracer import ModelTracer

    model = Model()
    tracer = ModelTracer(model=model, config=None, arch="Stub",
                         device=torch.device("meta"), tp=LOGICAL,
                         model_path="/stub", build_s=0.0)
    assert tracer.rank == 0
    assert tracer.set_rank(2) == 2
    assert model.embed.vocab_start_idx == 64
    assert tracer.rank_rebinds == 1
    # Same rank twice is not a second rebind.
    assert tracer.set_rank(2) == 0
    assert tracer.rank_rebinds == 1
    tracer.set_rank(0)
    assert model.embed.vocab_start_idx == 0
    assert tracer.rank_rebinds == 2


def test_a_tp1_tracer_has_no_other_rank_to_offer():
    from atom.compass.runtime.tracer import BuildRefusal, ModelTracer

    tracer = ModelTracer(model=Model(), config=None, arch="Stub",
                         device=torch.device("meta"), tp=1,
                         model_path="/stub", build_s=0.0)
    with pytest.raises(BuildRefusal, match="only rank 0"):
        tracer.set_rank(1)
