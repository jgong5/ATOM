# SPDX-License-Identifier: MIT
"""Engram's UVA shard held in HBM (`ATOM_ENGRAM_UVA_HBM`) instead of page-locked.

A host that cannot page-lock every rank's shard can still serve the UVA
lookup if each rank copies its shard into its own HBM; the kernel reads the
same rows either way. What these tests hold is what the serving start relies
on: the rows the kernel addresses arrive whole, the host never holds more
than one bounce chunk of them nor keeps the checkpoint pages it copied, a copy
that fails holds nothing and votes the TP group onto the host path, and
without the option the shard is page-locked as before.

CPU-only: the "device" is `torch.device("cpu")` and the page-locked bounce
chunk is ordinary memory, since a host with no driver has none to give
(`torch.empty(..., pin_memory=True)` raises there).
"""

import os
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

PAGE = 4096


@pytest.fixture
def engram(monkeypatch):
    """The engram modules, with every bounce chunk recorded and no real
    page-locking reachable: a test that wants the page-locked path installs
    its own `cudaHostRegister`."""
    from atom.model_ops.engram import host, host_memory, tables

    bounces = []

    def bounce(nbytes):
        bounces.append(nbytes)
        return torch.empty(nbytes, dtype=torch.uint8)

    monkeypatch.setattr(host_memory, "_bounce", bounce)
    monkeypatch.setattr(host_memory, "_host_allocator", lambda: None)
    monkeypatch.delenv("ATOM_ENGRAM_UVA_HBM", raising=False)
    return SimpleNamespace(
        host=host, host_memory=host_memory, tables=tables, bounces=bounces
    )


def scaled_table(engram, rows=4096, dim=64, block=16):
    """A quantized table as the checkpoint stores one: fp8 rows, E8M0 scales."""
    rng = np.random.default_rng(0)
    raw = rng.integers(0, 256, size=(rows, dim), dtype=np.uint8)
    raw[(raw & 0x7F) == 0x7F] = 0x3C  # no fp8 e4m3 NaN codes
    weight = torch.from_numpy(raw).view(torch.float8_e4m3fn)
    codes = rng.integers(117, 137, size=(rows, dim // block), dtype=np.uint8)
    scale = torch.from_numpy(codes).view(torch.float8_e8m0fnu)
    return engram.tables.HostEmbeddingTable(weight, rows, dim, scale=scale)


def test_an_hbm_shard_holds_exactly_the_rows_the_kernel_addresses(engram, monkeypatch):
    def page_locked(sources):
        raise AssertionError("the HBM option page-locked the shard")

    monkeypatch.setattr(engram.host_memory, "pin_rows", page_locked)
    # Many bounce chunks per source, so the shard crosses chunk edges.
    monkeypatch.setattr(engram.host_memory, "_HBM_BOUNCE_BYTES", 4 * PAGE)
    table = scaled_table(engram)
    device = torch.device("cpu")
    assert table.enable_uva(123, 4000, device=device) is True
    weight, scales, start, end = table.registered_shard()
    assert (start, end) == (123, 4000)
    assert weight.device == device and weight.dtype == torch.float8_e4m3fn
    assert torch.equal(
        weight.view(torch.uint8), table._tensor[123:4000].view(torch.uint8)
    )
    assert torch.equal(scales, table._scale[123:4000].view(torch.uint8))
    # Copies, not views of the checkpoint.
    assert weight.data_ptr() != table._tensor[123:4000].data_ptr()
    # The host held one bounce chunk, never the shard.
    assert engram.bounces == [4 * PAGE]
    table.disable_uva()
    assert table._uva is None and table._shard == ()


def resident_pages(tensor):
    """Pages of `tensor` mapped in this process (`/proc/self/pagemap`)."""
    lo = tensor.data_ptr() // PAGE
    hi = -(-(tensor.data_ptr() + tensor.nbytes) // PAGE)
    fd = os.open("/proc/self/pagemap", os.O_RDONLY)
    try:
        e = np.frombuffer(os.pread(fd, 8 * (hi - lo), 8 * lo), dtype="<u8")
    finally:
        os.close(fd)
    return int(np.count_nonzero(e >> np.uint64(63)))


def test_an_hbm_copy_drops_each_copied_chunk_of_the_checkpoint(
    engram, monkeypatch, tmp_path
):
    """The checkpoint mapping is read once; each chunk's pages leave this
    process once they are on the device, so the source is not held beside
    the shard."""
    data = np.random.default_rng(1).standard_normal((4096, 64)).astype(np.float32)
    path = tmp_path / "table.bin"
    data.tofile(path)
    mapped = torch.from_numpy(np.memmap(path, np.float32, "c", shape=data.shape))
    dropped = []
    real = engram.host_memory._source_dropper

    def watched(src):
        drop = real(src)
        assert drop is not None  # a file mapping

        def record(offset, nbytes):
            dropped.append((offset, nbytes))
            drop(offset, nbytes)

        record.close = drop.close
        return record

    monkeypatch.setattr(engram.host_memory, "_source_dropper", watched)
    monkeypatch.setattr(engram.host_memory, "_HBM_BOUNCE_BYTES", 1 << 14)
    table = engram.tables.HostEmbeddingTable(mapped, num_rows=4096, head_dim=64)
    assert table.enable_uva(0, 4096, device=torch.device("cpu")) is True
    assert len(dropped) > 2 and sum(n for _, n in dropped) == data.nbytes
    # Fault-around while copying a later chunk can map a few clean pages of an
    # earlier one back in (page cache, harmless).
    assert resident_pages(mapped) < data.nbytes // PAGE // 8
    assert torch.equal(table.registered_shard()[0], torch.from_numpy(data))
    assert torch.equal(mapped, torch.from_numpy(data))  # faulted back in


def test_an_hbm_copy_with_no_page_locked_memory_holds_nothing(engram, monkeypatch):
    def none_to_give(nbytes):
        raise RuntimeError("No HIP GPUs are available")

    monkeypatch.setattr(engram.host_memory, "_bounce", none_to_give)
    table = scaled_table(engram)
    assert table.enable_uva(0, 4096, device=torch.device("cpu")) is False
    assert table._uva is None and table._shard == ()


def test_an_hbm_copy_that_fails_on_the_scales_keeps_no_values(engram, monkeypatch):
    """All-or-nothing across a table's two sources."""
    copies = []
    real = engram.host_memory._copy_into

    def fail_second_source(dst, src, pool, drop=None):
        copies.append(src.data_ptr())
        if src.data_ptr() == table._scale.data_ptr():
            raise MemoryError("out of HBM")
        real(dst, src, pool, drop)

    monkeypatch.setattr(engram.host_memory, "_copy_into", fail_second_source)
    table = scaled_table(engram)
    assert table.enable_uva(0, 4096, device=torch.device("cpu")) is False
    assert table._scale.data_ptr() in copies
    assert table._uva is None and table._shard == ()


# --- the TP group's decision ---------------------------------------------------


def tiny_runtime(engram):
    """A host runtime of two layers, four hash heads each, over arange tables."""
    from dataclasses import replace

    from atom.model_ops.engram.mapping import (
        EngramConfig,
        NgramHashMapping,
        _next_prime,
    )

    class Tokenizer:
        lookup_table = np.arange(512, dtype=np.int64)

        def __len__(self):
            return len(self.lookup_table)

    cfg = EngramConfig(
        layer_ids=(0, 2),
        num_embeddings=(0, 0),
        max_ngram_size=3,
        vocab_size=1024,
        n_heads=2,
        head_dim=8,
        pad_token_id=0,
        compressed_vocab_size=512,
    )
    # The row counts the mapping derives for this shape, in its prime order.
    seen, totals = set(), []
    for _ in cfg.layer_ids:
        total = 0
        for _ in cfg.ngram_orders:
            start = cfg.vocab_size - 1
            for _ in range(cfg.n_heads):
                start = _next_prime(start, seen)
                seen.add(start)
                total += start
        totals.append(total)
    cfg = replace(cfg, num_embeddings=tuple(totals))
    mapping = NgramHashMapping(cfg, Tokenizer())
    tables = {
        layer: engram.tables.HostEmbeddingTable(
            torch.arange(rows * 8, dtype=torch.float32).reshape(rows, 8),
            num_rows=rows,
            head_dim=8,
        )
        for layer, rows in zip(cfg.layer_ids, cfg.num_embeddings)
    }
    return engram.host.EngramHost(
        engram.host.EngramPrefetcher(mapping, tables),
        max_num_tokens=8,
        num_hash_heads=cfg.num_hash_heads,
        head_dim=8,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def tp_group(monkeypatch, world_size, rank):
    """Serve `from aiter.dist.parallel_state import get_tp_group` a stub,
    undone at teardown so no later `importorskip("aiter")` finds it."""
    import sys

    group = SimpleNamespace(
        world_size=world_size, rank_in_group=rank, cpu_group=object()
    )
    parallel_state = ModuleType("aiter.dist.parallel_state")
    parallel_state.get_tp_group = lambda: group
    for name, module in (
        ("aiter", ModuleType("aiter")),
        ("aiter.dist", ModuleType("aiter.dist")),
        ("aiter.dist.parallel_state", parallel_state),
    ):
        monkeypatch.setitem(sys.modules, name, module)


def test_the_option_puts_this_ranks_heads_of_every_table_on_its_device(
    engram, monkeypatch, caplog
):
    monkeypatch.setenv("ATOM_ENGRAM_UVA_HBM", "1")
    monkeypatch.setattr(
        engram.host_memory,
        "pin_rows",
        lambda sources: pytest.fail("the HBM option page-locked the shard"),
    )
    tp_group(monkeypatch, world_size=2, rank=1)
    votes = []
    monkeypatch.setattr(
        engram.host.dist,
        "all_reduce",
        lambda t, op=None, group=None: votes.append(int(t)),
    )
    rt = tiny_runtime(engram)
    mapping = rt.prefetcher._hash_mapping
    heads = mapping.config.num_hash_heads
    with caplog.at_level("INFO", logger=engram.host.__name__):
        assert rt._enable_uva(rt.prefetcher, heads) is True
    assert votes == [1]
    assert "GiB in HBM on this rank" in caplog.text
    tables = rt.prefetcher._tables
    assert rt._pinned_tables == list(tables.values())
    per = -(-heads // 2)
    for layer, table in tables.items():
        offsets = mapping.head_offsets[layer]
        lo = int(offsets[per])
        hi = int(offsets[heads - 1]) + int(mapping.head_vocab_sizes[layer][heads - 1])
        weight, _, start, end = table.registered_shard()
        assert (start, end) == (lo, hi)
        assert torch.equal(weight, table._tensor[lo:hi])
        assert weight.data_ptr() != table._tensor[lo:hi].data_ptr()
    rt.shutdown()
    assert all(table._uva is None for table in tables.values())


def test_a_rank_whose_hbm_copy_fails_turns_the_group_around(engram, monkeypatch):
    """The device lookup ends in an all-gather: a rank that cannot hold its
    shard in HBM votes no, and the whole group takes the host path."""
    monkeypatch.setenv("ATOM_ENGRAM_UVA_HBM", "1")
    tp_group(monkeypatch, world_size=2, rank=0)
    votes = []
    monkeypatch.setattr(
        engram.host.dist,
        "all_reduce",
        lambda t, op=None, group=None: votes.append(int(t)),
    )
    taken = []
    real = engram.host_memory.to_device

    def second_table_refused(sources, device):
        taken.append(sources[0].data_ptr())
        if len(taken) == 2:
            return None
        return real(sources, device)

    monkeypatch.setattr(engram.host_memory, "to_device", second_table_refused)
    rt = tiny_runtime(engram)
    heads = rt.prefetcher._hash_mapping.config.num_hash_heads
    assert rt._enable_uva(rt.prefetcher, heads) is False
    assert votes == [0]
    assert len(taken) == 2
    assert rt._pinned_tables == []
    assert all(t._uva is None for t in rt.prefetcher._tables.values())
    rt.shutdown()


def test_without_the_option_the_shard_is_page_locked_as_before(engram, monkeypatch):
    registered = []

    class Cudart:
        def cudaHostRegister(self, ptr, size, flags):
            registered.append(ptr)
            return 0

        def cudaHostUnregister(self, ptr):
            registered.remove(ptr)
            return 0

    monkeypatch.setattr(torch.cuda, "cudart", lambda: Cudart())
    monkeypatch.setattr(
        engram.host_memory,
        "to_device",
        lambda sources, device: pytest.fail("copied to the device without the option"),
    )
    tp_group(monkeypatch, world_size=2, rank=0)
    monkeypatch.setattr(engram.host.dist, "all_reduce", lambda *a, **k: None)
    rt = tiny_runtime(engram)
    heads = rt.prefetcher._hash_mapping.config.num_hash_heads
    assert rt._enable_uva(rt.prefetcher, heads) is True
    assert len(registered) == len(rt.prefetcher._tables)
    assert engram.bounces == []
    rt.shutdown()
    assert registered == []
