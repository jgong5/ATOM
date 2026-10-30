# SPDX-License-Identifier: MIT
# Prefill attention reads no lengths back from the device.
#
# A read of a device value makes the host wait for every kernel queued before
# it, so a prefill launch would return only when the device is nearly through
# the forward. The GDN chunk indices come from the builder's host copy of
# `cu_seqlens_q`; the prefix-cache KV gather sizes its buffers from the host
# `total_kv` and builds no token map of its own. Here the device copy is stale
# or on the meta device, where a read gets the wrong answer or raises.

import importlib.abc
import importlib.machinery
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from atom.model_ops.attentions.token_layout.batch_ids import build_batch_ids_device
from atom.utils import CpuGpuBuffer


def _stand_in_attr(attr):
    """A mock for each name an importer binds, and none for a dunder:
    `inspect` caches every module's `__file__`, and a mock there breaks the
    real aiter's custom-op registration later in the process."""
    if attr.startswith("__"):
        raise AttributeError(attr)
    return MagicMock()


class _AiterStandIn(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Every `aiter` module as a stand-in, where aiter is not installed:
    nothing here calls it."""

    def find_spec(self, name, path, target=None):
        if name == "aiter" or name.startswith("aiter."):
            return importlib.machinery.ModuleSpec(name, self, is_package=True)

    def create_module(self, spec):
        module = types.ModuleType(spec.name)
        module.__path__ = []
        module.__getattr__ = _stand_in_attr
        return module

    def exec_module(self, module):
        pass


def _import():
    from atom.model_ops import attention_mha
    from atom.model_ops.attentions import aiter_attention, gdn_attn
    from atom.model_ops.fla_ops import index

    return attention_mha, aiter_attention, gdn_attn, index


try:
    from aiter.ops.triton.utils._triton import arch_info as _arch_info
except ImportError:
    _arch_info = None

if _arch_info is not None:
    # A real import, so no engine module keeps a stand-in for a later importer
    # to take. With no device, aiter's Triton helpers cannot probe the arch at
    # import; nothing here launches a kernel, so they are lent one.
    _lent = _arch_info._CACHED_ARCH is None
    if _lent:
        _arch_info._CACHED_ARCH = "gfx950"
    try:
        attention_mha, aiter_attention, gdn_attn, index = _import()
    finally:
        if _lent:
            _arch_info._CACHED_ARCH = None
else:
    _finder, _before = _AiterStandIn(), set(sys.modules)
    sys.meta_path.insert(0, _finder)
    try:
        attention_mha, aiter_attention, gdn_attn, index = _import()
    finally:
        sys.meta_path.remove(_finder)
        # Only the stand-ins go: an aiter imported for real before stays
        # whole, and none of its packages keeps a stand-in bound.
        for _name in [
            n for n in set(sys.modules) - _before if n.split(".")[0] == "aiter"
        ]:
            _module = sys.modules.pop(_name)
            _parent, _, _child = _name.rpartition(".")
            if getattr(sys.modules.get(_parent), _child, None) is _module:
                delattr(sys.modules[_parent], _child)


def buf(n):
    return CpuGpuBuffer(n, dtype=torch.int32, device="cpu", pin_memory=False)


def test_gdn_chunk_indices_follow_the_host_lengths(monkeypatch):
    # Lengths 17, 8, 1 at chunk 16: two chunks, one, one. The device copy
    # still holds the previous step's 8, 8, 8.
    host_cu, stale_cu = [0, 17, 25, 26], [0, 8, 16, 24, 24]
    var = {
        "cu_seqlens_q": buf(8),
        "gdn_batch_ptr": buf(64),
        "gdn_token_chunk_offset_ptr": buf(64),
        "non_spec_state_indices": buf(8),
        "non_spec_state_indices_in": buf(8),
    }
    var["cu_seqlens_q"].np[: len(host_cu)] = host_cu
    attn_metadata = SimpleNamespace(
        cu_seqlens_q=torch.tensor(stale_cu, dtype=torch.int32),
        num_cached_tokens=None,
    )
    monkeypatch.setattr(
        aiter_attention.AiterAttentionMetadataBuilder,
        "prepare_prefill",
        lambda self, b, rbs, rts: (attn_metadata, None),
    )
    builder = object.__new__(gdn_attn.GDNAttentionMetadataBuilder)
    builder.__dict__.update(
        model_runner=SimpleNamespace(
            forward_vars=var,
            config=SimpleNamespace(hf_config=SimpleNamespace(linear_conv_kernel_dim=4)),
        ),
        use_spec_decode=False,
        replayssm=False,
        device="cpu",
        prepare_state_indices=lambda batch, with_spec: None,
        prepare_block_tables=lambda batch: None,
        _checkpoint_targets=lambda batch: None,
    )
    batch = SimpleNamespace(
        block_tables=[[0], [1], [2]],
        total_seqs_num_decode=0,
        total_seqs_num_prefill=3,
        total_tokens_num_decode=0,
        total_tokens_num_prefill=26,
        total_seqs_num=3,
        total_tokens_num=26,
    )

    md, _ = builder.prepare_prefill(batch, 3, 26)
    cu = md.gdn_metadata.non_spec_query_start_loc
    var["cu_seqlens_q"].np[:4] = 0  # the next step reuses the buffer
    indices = index.prepare_chunk_indices(cu, 16)

    assert indices.tolist() == [[0, 0], [0, 1], [1, 0], [2, 0]]
    assert indices.dtype == cu.dtype


def test_chunk_indices_without_a_host_copy_read_the_device_lengths():
    """The plugins pass only their device `query_start_loc`."""
    cu = torch.tensor([0, 17, 25, 26], dtype=torch.int32)
    assert index.prepare_chunk_indices(cu, 16).tolist() == [
        [0, 0],
        [0, 1],
        [1, 0],
        [2, 0],
    ]


def test_prefix_kv_gather_takes_its_token_map_from_the_metadata(monkeypatch):
    calls = []
    monkeypatch.setattr(
        attention_mha, "cp_mha_gather_cache", lambda **kw: calls.append(kw)
    )
    impl = object.__new__(attention_mha.PagedAttentionImpl)
    impl.__dict__.update(_cache_format="NHD", kv_cache_dtype="bf16")
    meta = {"device": "meta", "dtype": torch.bfloat16}
    k = torch.empty(5, 2, 64, **meta)
    context_lens = torch.empty(3, dtype=torch.int32, device="meta")
    # Handed its total, the map builds without reading the meta lengths.
    token_map = build_batch_ids_device(context_lens, total=40)
    attn_metadata = SimpleNamespace(
        cu_seqlens_k=torch.empty(4, dtype=torch.int32, device="meta"),
        context_lens=context_lens,
        total_kv=40,
        block_tables=None,
        seq_starts=None,
        batch_id_per_k_token=token_map,
    )
    k_cache = torch.empty(9, 16, 2, 64, **meta)

    out = impl._gather_prefix_and_concat_kv(
        k, k, k, k_cache, k_cache, None, None, attn_metadata
    )

    assert token_map.shape == (40,)
    assert calls[0]["batch_id_per_k_token"] is token_map
    assert out[1].shape == out[2].shape == (40, 2, 64)
