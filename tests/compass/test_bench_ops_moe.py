# SPDX-License-Identifier: MIT
"""`compass measure ops` on an MoE leaf: the router's top-k expert ids it reads
are drawn independently and uniformly per copy, and their loads recorded."""

import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from test_bench_ops import _Runner

from atom.compass.bench import ops
from atom.utils import forward_context as fctx

I32, F32 = "torch.int32", "torch.float32"


class _Experts(torch.nn.Module):
    """An MoE layer as the stub reads one: its routed experts and top-k."""

    def __init__(self, experts, top_k):
        super().__init__()
        self.global_num_experts, self.top_k = experts, top_k


def _stub(*layers):
    runner = _Runner()
    for experts, top_k in layers:
        runner.model.append(_Experts(experts, top_k))
    stub = ops.Stub(runner, 10_000)
    stub.placed = [(fctx.ForwardContext(), {})]
    return stub


def _spec(shape, dtype=I32, role="activation"):
    stride = list(torch.empty(shape).stride())
    return {"shape": shape, "dtype": dtype, "stride": stride, "role": role}


MOE = {
    "op": "aiter.fused_moe_.default",
    "args": {
        "hidden_states": _spec([4, 3], F32),
        "topk_weight": _spec([4, 2], F32),
        "topk_ids": _spec([4, 2]),
        "token_ids": _spec([4]),
    },
}


def _timed(stub, op, n):
    """`op` through `_time` over `n` copies, and the copies its cold graph
    cycled over."""
    graphs = []

    def replay(fn, calls, flush=None):
        graphs.append(calls)
        return [1.0] * ops.REPEATS

    with (
        patch.object(ops, "_function", return_value=lambda **kwargs: None),
        patch.object(ops, "_kernels", return_value=(["k"], 1.0)),
        patch.object(ops, "_replay", side_effect=replay),
    ):
        record = stub._time(op, "sig", [{}], n)
    return record, [kwargs for kwargs, _ in graphs[0][: record["copies"]]]


def test_each_copy_draws_each_token_s_experts_distinct_and_records_their_loads():
    record, copies = _timed(_stub((16, 2)), MOE, 8)

    ids = [c["topk_ids"] for c in copies]
    for t in ids:
        assert t.dtype == torch.int32 and t.shape == (4, 2)
        assert all(len(set(row)) == 2 for row in t.tolist())
        assert 0 <= t.min() and t.max() < 16
    assert len({tuple(t.flatten().tolist()) for t in ids}) == 8
    assert record["routing"] == {
        "seed": ops.SEED,
        "experts": 16,
        "activated": [len(t.unique()) for t in ids],
        "max_load": [int(t.flatten().bincount().max()) for t in ids],
    }
    # A token id and the router's weights are no expert ids.
    assert all(not c["token_ids"].any() for c in copies)
    assert all(c["topk_weight"].dtype == torch.float32 for c in copies)


@pytest.mark.parametrize("past", ["the cold copies", "the copies"])
def test_a_copy_on_another_copy_s_inputs_draws_experts_of_its_own(past):
    # Copy 1 takes copy 0's hidden states: past the cold copies (one passes the
    # cache), or past the copies with the second of the leaf's two weights.
    stub = _stub((16, 2))
    w = {"w": _spec([3, 3], F32, "weight")}
    moe = {"op": MOE["op"], "args": MOE["args"] | w, "occurrences": 2}
    if past == "the cold copies":
        stub.cold_copies = {"sig": 1}

    record, copies = _timed(stub, moe, 2 if past == "the cold copies" else 1)

    assert copies[1]["hidden_states"] is copies[0]["hidden_states"]
    assert not torch.equal(copies[0]["topk_ids"], copies[1]["topk_ids"])
    assert len(record["routing"]["activated"]) == 2


def test_a_leaf_records_only_its_own_draws():
    stub = _stub((16, 2))
    _timed(stub, MOE, 3)

    record, copies = _timed(stub, MOE, 2)

    assert record["routing"]["activated"] == [
        len(c["topk_ids"].unique()) for c in copies
    ]


def test_the_draw_is_seeded_by_the_copy_and_the_seed():
    def ids(seed):
        with patch.object(ops, "SEED", seed):
            return [c["topk_ids"].tolist() for c in _timed(_stub((16, 2)), MOE, 3)[1]]

    assert ids(0) == ids(0)
    assert ids(0) != ids(1)


def test_a_leaf_after_another_under_the_model_s_default_dtype_draws_as_alone():
    # ATOM's runner leaves the model's dtype as torch's default; a draw at
    # bfloat16 ties within a row, so its top-k ids depend on how ties break.
    moe = {"op": MOE["op"], "args": {"topk_ids": _spec([256, 6])}}

    def ids(stub):
        return [c["topk_ids"].tolist() for c in _timed(stub, moe, 3)[1]]

    alone = ids(_stub((256, 6)))
    stub = _stub((256, 6))
    default = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        other = {"op": MOE["op"], "args": {"topk_ids": _spec([64, 6])}}
        assert _timed(stub, other, 2)[0]["routing"]  # another leaf draws first
        after = ids(stub)
    finally:
        torch.set_default_dtype(default)

    assert after == alone


def test_a_dense_model_draws_nothing():
    record, [copy] = _timed(_stub(), MOE, 1)

    assert not copy["topk_ids"].any() and record["routing"] is None


@pytest.mark.parametrize("tokens", [8, 16, 128])
def test_activated_experts_match_the_uniform_expectation(tokens):
    # DeepSeek-V4-Pro routes 6 of 384 experts per token.
    experts, k, copies = 384, 6, 256
    stub = _stub((experts, k))

    for c in range(copies):
        stub._route(_spec([tokens, k]), c)

    got = torch.tensor(ops._routing(stub.loads)["activated"], dtype=torch.float64)
    expected = experts * (1 - (1 - k / experts) ** tokens)
    assert abs(got.mean() - expected) < 4 * got.std() / copies**0.5
    assert sum(stub.loads[0]) == tokens * k


def test_a_model_of_two_routings_refuses_by_name():
    stub = _stub((16, 2), (8, 2))

    with pytest.raises(ValueError, match=r"route \(experts, top-k\) \[\(8, 2\)"):
        stub._copy(MOE, "sig", [{}], 0)


# ATOM's ids buffer as `init_aiter_topK_meta_data` builds it for 16 routed
# experts, top-2, over 6 rows: without EP two fused shared experts, ids 16 and
# 17 on every row; under EP rank 1 of 2 one shared expert plus one column, ids
# 16 and 17 on the rows rank 1 owns and the fake expert 17 on the others. The
# top-k columns are the router's, here -1.
FUSED = {
    "without EP": [[-1, -1, 16, 17]] * 6,
    "under EP": [[-1, -1, 17, 17], [-1, -1, 16, 17]] * 3,
}


def _fused(monkeypatch, buffer):
    topk = SimpleNamespace(aiter_topK_meta_data=(None, torch.tensor(buffer)))
    monkeypatch.setitem(sys.modules, "atom.model_ops.topK", topk)
    return _stub((16, 2))


def _moe(width):
    return {"op": MOE["op"], "args": MOE["args"] | {"topk_ids": _spec([4, width])}}


@pytest.mark.parametrize("ep", FUSED)
def test_fused_shared_expert_ids_draw_routed_columns_and_keep_atom_s_shared_ones(
    monkeypatch, ep
):
    record, copies = _timed(_fused(monkeypatch, FUSED[ep]), _moe(4), 8)

    ids = [c["topk_ids"] for c in copies]
    for t in ids:
        assert t.dtype == torch.int32 and t.shape == (4, 4)
        assert all(len(set(row)) == 2 for row in t[:, :2].tolist())
        assert 0 <= t[:, :2].min() and t[:, :2].max() < 16
        assert t[:, 2:].tolist() == [row[2:] for row in FUSED[ep][:4]]
    assert len({tuple(t[:, :2].flatten().tolist()) for t in ids}) == 8
    assert record["routing"]["activated"] == [len(t[:, :2].unique()) for t in ids]


@pytest.mark.parametrize("width", [3, 5])
def test_ids_wider_than_top_k_that_atom_s_buffer_does_not_explain_draw_nothing(
    monkeypatch, width
):
    # Width 3 with no fused shared experts; width 5 past a buffer 4 wide.
    stub = _fused(monkeypatch, FUSED["under EP"]) if width == 5 else _stub((16, 2))

    record, [copy] = _timed(stub, _moe(width), 1)

    assert not copy["topk_ids"].any() and record["routing"] is None


def test_ids_and_weights_viewed_in_atom_s_top_k_buffers_draw_ids_and_read_weights(
    monkeypatch,
):
    # DeepSeek-V4 on gfx942 fuses its shared expert: its fused-MoE leaf reads
    # views of ATOM's top-k buffers, which the capture holds as metadata.
    weights = torch.full((6, 4), 0.5)
    topk = SimpleNamespace(
        aiter_topK_meta_data=(
            weights,
            torch.tensor(FUSED["without EP"], dtype=torch.int32),
        )
    )
    monkeypatch.setitem(sys.modules, "atom.model_ops.topK", topk)
    runner = _Runner()
    runner.model.append(_Experts(16, 2))
    stub = ops.Stub(runner, 10_000)
    viewed = {
        "topk_weight": _spec([4, 4], F32, "metadata"),
        "topk_ids": _spec([4, 4], role="metadata"),
    }
    # a weight of 9 occurrences: copy 8 is copy 0's call on the ninth weight
    viewed["w"] = _spec([3, 3], F32, "weight")
    moe = {"op": MOE["op"], "args": MOE["args"] | viewed, "occurrences": 9}
    stub.leaves = {"sig": (moe, [{}])}
    stub._prepare((0, 0))

    record, copies = _timed(stub, moe, 8)

    ids = [c["topk_ids"] for c in copies]
    for t in ids:
        assert all(len(set(row)) == 2 for row in t[:, :2].tolist())
        assert 0 <= t[:, :2].min() and t[:, :2].max() < 16
        assert t[:, 2:].tolist() == [[16, 17]] * 4
    assert len({tuple(t[:, :2].flatten().tolist()) for t in ids}) == 9
    assert record["routing"]["activated"] == [len(t[:, :2].unique()) for t in ids]
    assert torch.equal(copies[0]["topk_weight"], weights[:4])
    assert stub.placed[0][1]["sig", "topk_weight"][0] == "topk_weights"
    assert stub.placed[0][1]["sig", "topk_ids"][0] == "topk_ids"


@pytest.mark.parametrize("width", [2, 4])
def test_a_block_table_as_wide_as_top_k_or_atom_s_ids_buffer_keeps_its_blocks(
    monkeypatch, width
):
    # Top-k 2, and ATOM's ids buffer 4 wide: integer metadata of either width
    # that is not a view of the ids buffer is read live, never drawn.
    topk = SimpleNamespace(
        aiter_topK_meta_data=(
            None,
            torch.tensor(FUSED["without EP"], dtype=torch.int32),
        )
    )
    monkeypatch.setitem(sys.modules, "atom.model_ops.topK", topk)
    runner = _Runner()
    runner.model.append(_Experts(16, 2))
    stub = ops.Stub(runner, 10_000)
    tables = {
        "q": _spec([4, 3], F32),
        "block_tables": _spec([4, width], role="metadata"),
    }
    attention = {"op": "aiter.some_attention.default", "args": tables}
    stub.leaves = {"sig": (attention, [{}])}
    stub._prepare((5, 0))

    record, copies = _timed(stub, attention, 4)

    assert stub.placed[0][1]["sig", "block_tables"][0] == "block_tables"
    assert all(c["block_tables"].eq(5).all() for c in copies)
    assert record["routing"] is None
