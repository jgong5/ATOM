# SPDX-License-Identifier: MIT
"""`compass measure ops` binds DeepSeek-V4's window write at a prefill padded
past its rows to what ATOM passes it there, with no device.

ATOM runs a step at the `running_bs` `ForwardMode.decide` settles on its
ladder; a prefill's rows past the batch's take no query tokens and slot 0, so
the write writes nothing for them. The stand-in runner is `test_bench_ops.py`'s,
its step's width settled by ATOM's own `decide`.
"""

from types import SimpleNamespace

import numpy as np
import torch
from test_bench_ops import _graph, _leaf, _Runner, _t

from atom.compass.bench import ops
from atom.compass.capture.forward import prefill
from atom.utils import forward_context as fctx
from atom.utils.forward_context import ForwardMode

SWA = "atom.model_ops.v4_kernels.state_writes:swa_write"
I32 = "torch.int32"


class _Csa(torch.nn.Module):
    """A DeepSeek-V4 CSA layer, beside which a window write takes the state
    slots ATOM's step writes to."""

    indexer, compress_ratio = SimpleNamespace(index_topk=8), 4

    def _fill_csa_paged_compress(self, *args):
        pass


class _V4Runner(_Runner):
    """`_Runner` of a DeepSeek-V4 CSA layer, preparing a prefill as ATOM's
    builder does at the width `decide` settles on the runner's ladder, ATOM's
    eager one unless the bench pads it: past the batch's rows, flat query
    offsets and slot 0. Past that width, its buffers hold an earlier step's."""

    capture_sizes_np = np.array([0], dtype=np.int32)

    def __init__(self):
        super().__init__()
        self.model.append(_Csa())
        self.forward_vars["cu_seqlens_q"] = torch.tensor([0, 3625, 7250, 0]).int()
        self.forward_vars["v4_meta_state_slot_out"] = torch.tensor([5, 9, 0]).int()

    def prepare_model(self, batch):
        mode = ForwardMode.decide(
            batch=batch,
            dp_size=1,
            dp_group=None,
            enforce_eager=True,
            step_needs_eager=False,
            capture_sizes=self.capture_sizes_np,
            is_block_drafter=False,
            tbo_on=False,
            local_tbo=(False, False, 0, 0),
            max_seqlen_q=1,
            running_tokens_per_dp_rank=np.zeros(1, dtype=np.int32),
        )
        rows, width = batch.total_seqs_num, mode.running_bs
        tokens = batch.total_tokens_num
        cu = self.forward_vars["cu_seqlens_q"]
        slots = self.forward_vars["v4_meta_state_slot_out"]
        cu[1 : rows + 1] = torch.tensor(np.cumsum(batch.num_scheduled_tokens))
        cu[rows + 1 : width + 1] = tokens
        slots[:rows], slots[rows:width] = torch.tensor(batch.slots), 0
        md = fctx.AttentionMetaData(cu_seqlens_q=cu[: width + 1])
        md.state_slot_out = slots[:width]
        context = fctx.Context(positions=torch.arange(tokens))
        fctx._forward_context = fctx.ForwardContext(attn_metadata=md, context=context)
        return torch.zeros(tokens, dtype=torch.int32), None, None, None, True, False


def test_a_padded_prefill_s_window_write_takes_the_slots_and_offsets_atom_pads(
    monkeypatch,
):
    # A prompt of 462 tokens over one row whose LM head ran padded to two
    # rows: the write, captured there, takes two rows' state slots and three
    # query offsets. A step prepared at its one row publishes neither past it,
    # and the write took an earlier step's: an offset 7250 tokens into a step
    # of 462.
    monkeypatch.setattr(fctx._forward_kv_cache_context, "kv_cache_data", {})

    def took(stub, op, sig, instances, n):
        kwargs, _ = stub._copy(op, sig, instances, 0)
        return {"took": {slot: t.tolist() for slot, t in kwargs.items()}}

    def batch(view, block, slot):
        tokens = [query for query, _, _ in view["rows"]]
        return SimpleNamespace(
            total_seqs_num=len(tokens),
            total_seqs_num_prefill=len(tokens),
            total_tokens_num=sum(tokens),
            total_tokens_num_prefill=sum(tokens),
            num_scheduled_tokens=tokens,
            num_cached_tokens=[0] * len(tokens),
            final_prefill_requests=lambda: len(tokens),
            is_dummy_run=False,
            slots=[3 + slot],  # apart from the padded row's slot 0
        )

    monkeypatch.setattr(ops.Stub, "_time", took)
    write = _leaf(
        SWA,
        ("cu_seqlens_q", _t([3], "metadata", dtype=I32)),
        ("state_slot_per_seq", _t([2], "metadata", dtype=I32)),
    )
    found = ops.leaves(_graph(write, layers=1)["region"])
    rule = SimpleNamespace(block_size=16, batch=batch)
    shape = prefill(462, 1, gathered=8389, width=2)

    [record] = ops.Stub(_V4Runner(), 10_000).point(rule, shape, None, found)

    assert record["metadata"] == {
        "cu_seqlens_q": "cu_seqlens_q",
        "state_slot_per_seq": "state_slot_out",
    }
    assert record["took"] == {
        "cu_seqlens_q": [0, 462, 462],
        "state_slot_per_seq": [3, 0],
    }
