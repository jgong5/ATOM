# SPDX-License-Identifier: MIT
"""`compass measure ops` on an MoE leaf: the router's top-k expert ids it reads
are drawn independently and uniformly per copy, and their loads recorded."""

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

    def replay(fn, calls):
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
    # cache), or past the copies with the model's second Linear(3, 3) weight.
    stub = _stub((16, 2))
    moe = {"op": MOE["op"], "args": MOE["args"] | {"w": _spec([3, 3], F32, "weight")}}
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


@pytest.mark.parametrize("streams", [None, 4])
def test_the_stub_hands_the_logits_head_each_token_s_residual_streams(streams):
    # DeepSeek-V4's logits head reduces `hc_mult` residual streams a token; a
    # model without them takes one hidden state a token.
    runner = _Runner()
    if streams:
        runner.config.hf_config.get_text_config().hc_mult = streams
    stub, got = ops.Stub(runner, 10_000), []

    with patch.object(runner.model, "forward", lambda *a: got.append(stub(*a))):
        runner.forward((0, 0))

    assert got[0].shape == ((2, 4, 3) if streams else (2, 3))
