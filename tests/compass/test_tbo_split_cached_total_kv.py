# SPDX-License-Identifier: MIT
"""A TBO prefill split whose batch holds cached context reads no tensor onto the
host: each micro-batch's `total_kv` comes from the CPU lengths the attention
builder published, so on a GPU the split does not wait for the device to drain
before either micro-batch launches."""

import numpy as np
import pytest
import torch
from torch.overrides import TorchFunctionMode

from atom.utils.forward_context import AttentionMetaData
from atom.utils.tbo.ubatch_splitting import (
    UBatchSlice,
    attach_tbo_cpu_lens,
    split_attn_metadata,
)

# Each copies a tensor's values to the host; on a GPU tensor it waits for the device.
HOST_READS = {
    "item",
    "cpu",
    "tolist",
    "numpy",
    "__array__",
    "__int__",
    "__index__",
    "__float__",
    "__bool__",
}

CACHED = np.array([5, 0, 7], dtype=np.int32)
NEW = np.array([4, 6, 3], dtype=np.int32)
RUNNING_BS = 4

# A token-midpoint split at token 6 cuts request 1 in two.
SLICES = [
    UBatchSlice(request_slice=slice(0, 2), token_slice=slice(0, 6)),
    UBatchSlice(request_slice=slice(1, 3), token_slice=slice(6, 13)),
]


class HostReads(TorchFunctionMode):
    """Names every host read of a tensor taken inside it."""

    def __init__(self):
        super().__init__()
        self.seen = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        name = getattr(func, "__name__", "")
        if name in HOST_READS:
            self.seen.append(name)
        return func(*args, **(kwargs or {}))


def cu(lens):
    return np.concatenate([[0], np.cumsum(lens)]).astype(np.int32)


def prefill_with_cached_context(with_cu_k: bool) -> AttentionMetaData:
    """A three-request prefill, two with a cached prefix, padded to
    `RUNNING_BS`, and the CPU lengths its builder publishes under TBO."""
    ctx = np.zeros(RUNNING_BS, dtype=np.int32)
    ctx[: len(NEW)] = CACHED + NEW
    cu_q = np.concatenate([cu(NEW), [NEW.sum()]]).astype(np.int32)
    cu_k = np.concatenate([cu(CACHED + NEW), [(CACHED + NEW).sum()]]).astype(np.int32)
    md = AttentionMetaData(
        cu_seqlens_q=torch.from_numpy(cu_q.copy()),
        cu_seqlens_k=torch.from_numpy(cu_k.copy()) if with_cu_k else None,
        max_seqlen_q=int(NEW.max()),
        max_seqlen_k=int(ctx.max()),
        context_lens=torch.from_numpy(ctx.copy()),
        has_cached=True,
        total_kv=int(ctx.sum()),
    )
    bs = len(NEW)
    attach_tbo_cpu_lens(md, "context_lens", ctx[:bs].copy())
    attach_tbo_cpu_lens(md, "cu_seqlens_q", cu_q[: bs + 1].copy())
    attach_tbo_cpu_lens(md, "cu_seqlens_k", cu_k[: bs + 1].copy())
    return md


@pytest.mark.parametrize("with_cu_k", [True, False], ids=["cu_seqlens_k", "context"])
@pytest.mark.parametrize("ub", [0, 1])
def test_a_cached_prefill_split_takes_total_kv_without_a_host_read(with_cu_k, ub):
    md = prefill_with_cached_context(with_cu_k)
    rs = SLICES[ub].request_slice

    with HostReads() as reads:
        split = split_attn_metadata(md, SLICES[ub], RUNNING_BS)

    assert reads.seen == []
    # What the device read gave: this ubatch's requests' cached plus new tokens.
    assert split.total_kv == int(md.context_lens[rs].sum()) == [15, 16][ub]
    if with_cu_k:
        assert split.total_kv == int(split.cu_seqlens_k[rs.stop - rs.start])
