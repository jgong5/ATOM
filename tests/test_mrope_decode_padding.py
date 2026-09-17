# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""M-RoPE decode positions must be packed at the width the graph reads them.

The M-RoPE buffer is the one per-token tensor a decode step writes that is not
read as a contiguous prefix. `ModelRunner._mrope_positions_view(n)` strides a
flat buffer by `n` to make three sections of `n` positions, and a captured
decode graph holds that view at the bucket it was captured with --
`running_bs * max_seqlen_q`. The builder packed the *scheduled* width instead,
so at three requests replaying a bucket of four the graph read each section
rotated by one and every section but the first carried another's positions.

Every other decode tensor survived the same short write because it is 1-D: a
short write there leaves a stale tail but never moves a real value. Here it
moves all of them, which is why this is checked by reading the buffer back
through the runner's own view rather than by inspecting what was written.
"""

import importlib.util
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch


def _why_aiter_will_not_load() -> str | None:
    """Decide the skip without importing AITER, because trying costs the run.

    The helpers under test are pure numpy/torch, but they live in a module that
    pulls in AITER, and on a device-free box that import dies partway through
    its arch detection. A `try`/`except ImportError` around it is not enough and
    not harmless: the half-finished import leaves `aiter.jit` module objects
    whose native registry never got filled, and every later test that touches
    AITER inherits them -- collecting this file next to `tests/compass` took 25
    unrelated tests down with it, and clearing `sys.modules` afterwards only
    changed the error they died with. So the two known prerequisites are checked
    first, and nothing is imported until both hold.
    """
    if "jax" not in sys.modules and importlib.util.find_spec("jax") is None:
        # aiter/ops/triton/utils/_triton/arch_info.py reads the arch off jax.
        return "jax is not installed"
    rocminfo = shutil.which("rocminfo")
    if rocminfo is None:
        # aiter/jit/utils/chip_info.py: get_gfx_runtime always shells out to it.
        return "rocminfo is not on PATH"
    if subprocess.run(rocminfo, capture_output=True, check=False).returncode != 0:
        return "rocminfo does not name a device"
    return None


_unavailable = _why_aiter_will_not_load()
if _unavailable:
    pytest.skip(f"needs an importable AITER build: {_unavailable}",
                allow_module_level=True)

from atom.model_engine.model_runner import ModelRunner
from atom.model_ops.attentions.backends import CommonAttentionBuilder
from atom.utils import CpuGpuBuffer

MAX_TOKENS = 32
STALE = -999  # whatever the previous step left; must not survive as a position


class Runner:
    """The two things the builder asks the runner for, and nothing else."""

    use_mrope = True

    def __init__(self):
        buf = CpuGpuBuffer(
            3, MAX_TOKENS, dtype=torch.int64,
            device=torch.device("cpu"), pin_memory=False,
        )
        buf.np.fill(STALE)
        buf.gpu.fill_(STALE)
        self.forward_vars = {"mrope_positions": buf}

    _mrope_positions_view = ModelRunner._mrope_positions_view


class Builder:
    """The real packing helpers, on a runner with no model behind it."""

    def __init__(self, runner):
        self.model_runner = runner

    _mrope_cpu_view = CommonAttentionBuilder._mrope_cpu_view
    _copy_mrope_to_gpu = CommonAttentionBuilder._copy_mrope_to_gpu
    _build_mrope_decode_positions = (
        CommonAttentionBuilder._build_mrope_decode_positions
    )


class Batch:
    def __init__(self, context_lens, max_seqlen_q, deltas=None):
        self.req_ids = [f"r{i}" for i in range(len(context_lens))]
        self.total_tokens_num_decode = len(context_lens) * max_seqlen_q
        self.mrope_position_deltas = deltas or {}


def _pack(context_lens, bs, max_seqlen_q=1, deltas=None):
    """Pack a decode step, then read it back the way a graph would.

    Returns the runner's view at the padded width -- which is what a replay
    reads -- not the tensor the builder handed back.
    """
    runner = Runner()
    builder = Builder(runner)
    lens = np.asarray(context_lens, dtype=np.int32)
    returned = builder._build_mrope_decode_positions(
        Batch(context_lens, max_seqlen_q, deltas), lens, max_seqlen_q, bs
    )
    read = runner._mrope_positions_view(bs * max_seqlen_q)
    return returned, read.tolist()


def test_a_padded_decode_is_read_back_section_by_section():
    """Three requests replaying the bucket of four: the commonest padded step
    there is. Each section holds its three positions and then the pad, at the
    offsets the graph's stride puts them."""
    _, read = _pack([10, 20, 30], bs=4)
    assert read == [[9, 19, 29, 0]] * 3


def test_the_sections_are_not_rotated_by_the_padding():
    """The defect, stated as what it did. Packing the scheduled width put
    section one's first element where the graph reads section zero's pad, so
    each section came back shifted by the missing row."""
    _, read = _pack([10, 20, 30], bs=4)
    rotated = [[9, 19, 29, 9], [19, 29, 9, 19], [29, 0, 0, 0]]
    assert read != rotated
    assert len({tuple(section) for section in read}) == 1


def test_the_pad_carries_a_legal_position_not_the_last_step():
    """Attention never walks the padded rows -- `cu_seqlens_q` repeats its
    last offset, so they are empty sequences -- but the rotary embedding and
    the MoE path do read them. Zero is a legal position; what the previous
    step left is not."""
    _, read = _pack([10, 20, 30], bs=4)
    assert all(section[3] == 0 for section in read)
    assert STALE not in [value for section in read for value in section]


def test_an_unpadded_decode_is_packed_exactly_as_before():
    """Eager, or a batch that lands on its bucket: `bs` is the scheduled
    count and nothing is added. The control for the case above."""
    _, read = _pack([10, 20, 30], bs=3)
    assert read == [[9, 19, 29]] * 3


def test_the_builder_still_returns_the_view_it_packed():
    """The eager path uses the returned tensor directly, so it has to be the
    padded view and not a narrower slice of it."""
    returned, read = _pack([10, 20, 30], bs=4)
    assert list(returned.shape) == [3, 4]
    assert returned.tolist() == read


def test_a_speculative_step_pads_whole_rows():
    """`max_seqlen_q > 1`: the width is rows times tokens-per-row, so two
    verified requests in a bucket of three pad by one row of two tokens, not
    by one token."""
    _, read = _pack([10, 20], bs=3, max_seqlen_q=2)
    assert read == [[8, 9, 18, 19, 0, 0]] * 3


def test_the_position_deltas_are_unchanged_by_the_padding():
    """M-RoPE's three sections are the logical rows of one request's
    position, not three requests: the delta shifts all three together, and
    the padding must not make them disagree."""
    _, read = _pack([10, 20, 30], bs=4, deltas={"r1": 5})
    assert read == [[9, 24, 29, 0]] * 3


def test_no_mrope_no_positions():
    """A model without M-RoPE gets None and the buffer is never touched."""
    runner = Runner()
    runner.use_mrope = False
    builder = Builder(runner)
    batch = Batch([10, 20, 30], 1)
    got = builder._build_mrope_decode_positions(
        batch, np.asarray([10, 20, 30], dtype=np.int32), 1, 4
    )
    assert got is None
    assert torch.equal(
        runner.forward_vars["mrope_positions"].gpu,
        torch.full((3, MAX_TOKENS), STALE, dtype=torch.int64),
    )
