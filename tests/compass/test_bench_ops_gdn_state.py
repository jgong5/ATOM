# SPDX-License-Identifier: MIT
"""The bytes `measure ops` counts for a GDN layer's per-request state at a
point of a model under a drafter.

ATOM's GDN mixer is one leaf, `aiter.linear_attention_with_output_base`,
naming its layer, whose state ATOM's KV registry holds: its conv state
(`k_cache`) is handed each row's first state slot, and its recurrent state
(`v_cache`) at a decode every slot the row holds, two under MTP1, at a prefill
the first alone (`attention_gdn.py`). The layer is Qwen3.5-27B's at TP1,
sized by `GDNStateMixin._state_shape`; batches are the replay rule's for a GDN
model under MTP1 (`test_bench_replay_state`), their state indices prepared by
ATOM's own `GDNStateMixin.prepare_state_indices` beside `test_bench_ops`'s
stand-in runner.
"""

import json
from types import SimpleNamespace

import pytest
import torch
from test_bench_ops import KEY, TREE, _graph, _leaf, _t
from test_bench_ops import _Runner as OpsRunner
from test_bench_replay_state import MTP1, gdn_config
from test_filler_tokens import model_dir  # noqa: F401  the model_dir fixture
from test_gdn_prefill_metadata_host import GDNStateMixin  # aiter stood in

from atom.compass import ir
from atom.compass.bench import ops, replay
from atom.compass.capture.forward import decode, prefill
from atom.config import KVCacheTensor
from atom.utils import forward_context as fctx

pytestmark = pytest.mark.usefixtures("unpriced")

BF16 = "torch.bfloat16"
#: The layer's state slots: three placements of two rows of two each.
SLOTS = 12
#: Last-level cache bytes: 2 copies of a decode's call pass twice it.
LLC = 4 << 20


def geometry():
    """Qwen3.5-27B's GDN heads and dims, as its config states them."""
    path = TREE / "tests/compass/qwen3_5_27b_config.json"
    text = json.loads(path.read_text())["text_config"]
    return (
        text["linear_num_key_heads"],
        text["linear_num_value_heads"],
        text["linear_key_head_dim"],
        text["linear_value_head_dim"],
        text["linear_conv_kernel_dim"],
    )


class GDNRunner(OpsRunner):
    """`test_bench_ops`'s stand-in runner holding one GDN layer's state, two
    slots a request, whose input preparation writes a batch's state indices
    as ATOM's GDN builder does, the spec-decode table at a decode; `slots`
    holds each placement's conv and recurrent state slots, as the kernels are
    handed them."""

    def __init__(self):
        super().__init__({KEY.format(L=0): SimpleNamespace(layer_num=0)})
        self.pool_plan = SimpleNamespace(entries={"state": SLOTS})
        self.config.pool_entries_per_req = {"state": MTP1}
        self.forward_vars |= GDNStateMixin._state_index_buffers(8, MTP1 - 1, "cpu")
        self.builder = object.__new__(GDNStateMixin)
        self.builder.model_runner, self.builder.replayssm = self, False
        self.slots = []

    def prepare_model(self, batch):
        n, spec = batch.total_seqs_num, not batch.total_seqs_num_prefill
        self.builder.prepare_state_indices(batch, with_spec=spec)
        if spec:
            table = self.forward_vars["spec_state_indices"].np[:n]
            self.slots.append((table[:, 0].tolist(), table.flatten().tolist()))
        else:
            committed = self.forward_vars["non_spec_state_indices"].np[:n].tolist()
            self.slots.append((committed, committed))
        return super().prepare_model((0, 0))  # its forward context


@pytest.mark.parametrize("shape", [decode(2, rows=4), prefill(4, 2)], ids=str)
def test_a_gdn_point_counts_each_state_slot_its_layer_s_kernels_are_handed(
    monkeypatch,
    model_dir,  # noqa: F811
    shape,
):
    k_heads, v_heads, k_dim, v_dim, kernel = geometry()
    conv, state = GDNStateMixin._state_shape(
        1, k_heads, v_heads, k_dim, v_dim, kernel, MTP1 - 1
    )
    bf16 = torch.bfloat16
    layer = KVCacheTensor(
        0,
        torch.empty(SLOTS, *conv, dtype=bf16),
        torch.empty(SLOTS, *state, dtype=bf16),
        per_request_state=True,
    )
    registry = {"layer_0": layer}
    monkeypatch.setattr(fctx._forward_kv_cache_context, "kv_cache_data", registry)
    monkeypatch.setattr(ops.Stub, "_time", lambda *_: {"cold_ns": 1.0})
    t = shape.tokens
    args = [
        ("mixed_qkv", _t([t, conv[1]], "activation", dtype=BF16)),
        ("b", _t([t, v_heads], "activation", dtype=BF16)),
        ("a", _t([t, v_heads], "activation", dtype=BF16)),
        ("core_attn_out", _t([t, v_heads, v_dim], "activation", dtype=BF16)),
    ]
    leaf = _leaf(
        "aiter.linear_attention_with_output_base.default",
        *args,
        ("layer_name", ir.ContextRef(KEY)),
        key=KEY,
    )
    found = ops.leaves(_graph(leaf, layers=1)["region"])
    runner = GDNRunner()
    rule = replay.ReplayRule(gdn_config(model_dir))

    [record] = ops.Stub(runner, LLC).point(rule, shape, 20, found)

    # A decode's two rows of two tokens: each placement hands the conv kernel
    # a slot a row and the recurrent one two, 81920 and 1572864 bytes a slot.
    # A prefill hands both the first slot a row. Each copy is a placement of
    # its own, the one layer's state there read once.
    acts = 2 * (t * conv[1] + 2 * t * v_heads + t * v_heads * v_dim)
    each = layer.k_cache[0].nbytes, layer.v_cache[0].nbytes
    read = [
        acts + len(convs) * each[0] + len(recurrent) * each[1]
        for convs, recurrent in runner.slots
    ]
    width = MTP1 if shape.phase == "decode" else 1
    assert [len(r) for _, r in runner.slots] == [2 * width] * len(read)
    assert record["footprint_bytes"] == sum(read)
    placements = {"decode": 2, "prefill": 3}[shape.phase]
    assert record["placements"] == len(read) == placements
