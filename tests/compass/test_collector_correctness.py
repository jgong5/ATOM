"""Three ways a benchmark can report work it did not do.

Each of these produces a plausible number. That is what makes them worth a
test: a price that is absent gets chased, and a price that is wrong by a
believable factor gets used.

None of these needs a device. The two that would -- rotating a real KV region,
reducing over a real group -- are exercised through the seams the production
code already has, so what is asserted is the decision the collector makes, not
the hardware's response to it.
"""

from __future__ import annotations

import json

import pytest

from atom.compass.runtime import forward_ctx, microbench

# == (1) the capture-unsupported fallback ==================================
#
# Chunked-prefill attention cannot be graph-captured, so it is timed
# back-to-back instead. That path installed variant 0 and then called in it
# `iters` times while the artifact recorded `kv_regions: len(variants)` -- 64
# cold regions claimed for a loop that revisited one warm one.

class _Uncapturable(RuntimeError):
    def __init__(self):
        super().__init__("operation not permitted when stream is capturing")


def _attention_graph(tmp_path):
    path = tmp_path / "graph.json"
    path.write_text(json.dumps({"ops": [{
        "name": "aiter::unified_attention_with_output_base",
        "input_shapes": [], "dtypes": [], "scalars": [],
        "context": [["context_lens", [8]]]}]}))
    return str(path)


@pytest.fixture
def fallback(tmp_path, monkeypatch):
    """A pricing run whose capture always fails, with the rotation observable.

    `install` hands back real thunks that record which region they installed,
    so "did the timed loop rotate" is answered by what the thunks saw rather
    than by inspecting the call.
    """
    from atom.utils import forward_context

    visited: list[int] = []

    def fake_install(name, recorded, variants=1):
        return [lambda v=v: visited.append(v) for v in range(variants)]

    def fake_time_in_graph(*args, **kwargs):
        raise _Uncapturable()

    def fake_time_over(fn, sets, iters, warmup, before=None):
        for i in range(warmup + iters):
            if before is not None:
                before(i)
        return 1e-5, 1e-6

    monkeypatch.setattr(forward_context, "reset_forward_context", lambda: None)
    monkeypatch.setattr(microbench, "_resolve", lambda n: (lambda *a, **k: None))
    monkeypatch.setattr(microbench, "_build_arg_sets", lambda op, cache, fn: [((), {})])
    monkeypatch.setattr(forward_ctx, "install", fake_install)
    monkeypatch.setattr(microbench, "_time_in_graph", fake_time_in_graph)
    monkeypatch.setattr(microbench, "_time_over", fake_time_over)

    class _Torch:
        class cuda:
            @staticmethod
            def synchronize():
                return None

    monkeypatch.setitem(__import__("sys").modules, "torch", _Torch)
    return visited


def test_the_fallback_rotates_the_regions_it_installed(fallback, tmp_path):
    """Otherwise every call after the first runs against a resident region."""
    result = microbench.price_graph(_attention_graph(tmp_path), iters=8,
                                    warmup=0, cache="graph")
    assert result["prices"], result["unpriced"]
    assert len(set(fallback)) > 1, (
        "the fallback installed one region and timed every call in it; "
        f"regions visited: {sorted(set(fallback))}")


def test_the_region_count_is_what_the_loop_visited(fallback, tmp_path):
    """`kv_regions` is the claim a reader uses to decide a price is cold.

    It must come from the loop that ran, not from how many regions were
    prepared for a path that then failed.
    """
    result = microbench.price_graph(_attention_graph(tmp_path), iters=8,
                                    warmup=0, cache="graph")
    record = next(iter(result["prices"].values()))
    assert record["cache"] == "over", "expected the fallback path"
    assert record["kv_regions"] == len(set(fallback)), (
        f"claimed {record['kv_regions']} regions, visited "
        f"{len(set(fallback))}")
    assert record["kv_regions"] > 1


# == (2) the width a collective was actually reduced over ==================

def _collective_graph(tmp_path, topology):
    path = tmp_path / "coll.json"
    path.write_text(json.dumps({
        "key": {"topology": [["tp", topology]]},
        "ops": [{"name": "aiter::all_reduce_", "input_shapes": [[32, 5120]],
                 "dtypes": ["bfloat16"], "scalars": [], "group": "tp"}],
    }))
    return str(path)


@pytest.fixture
def priceable(monkeypatch):
    from atom.utils import forward_context

    monkeypatch.setattr(forward_context, "reset_forward_context", lambda: None)
    monkeypatch.setattr(microbench, "_resolve", lambda n: (lambda *a, **k: None))
    monkeypatch.setattr(microbench, "_build_arg_sets",
                        lambda op, cache, fn: [((), {})])
    monkeypatch.setattr(microbench, "_time_over",
                        lambda *a, **k: (1e-5, 1e-6))
    monkeypatch.setattr(microbench, "_time_isolated",
                        lambda *a, **k: (1e-5, 1e-6))


def _group_of(monkeypatch, width):
    """Make this process look like a rank of a group `width` wide.

    Patches torch.distributed rather than anything in `microbench`, so the test
    states the situation and not the implementation that has to notice it -- and
    so it runs identically against a build that has no such implementation yet.
    """
    import torch.distributed as dist

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: width is not None)
    if width is not None:
        monkeypatch.setattr(dist, "get_world_size", lambda *a, **k: width)


def test_a_four_rank_graph_is_not_priced_over_two_ranks(tmp_path, monkeypatch,
                                                        priceable):
    """The label comes from the graph and the group comes from `--tp`.

    Nothing compared them, so `--tp 2` on a TP4 graph reduced over two ranks
    and filed a four-rank price -- which matches a real TP4 request exactly,
    since a collective's signature carries its message and not its group.
    """
    _group_of(monkeypatch, 2)
    result = microbench.price_graph(_collective_graph(tmp_path, 4),
                                    iters=4, warmup=0, cache="hot")
    assert not result["prices"], (
        "priced a 4-rank collective while reducing over 2 ranks, and labelled "
        f"it {result['provenance']['topology']}: {result['prices']}")
    why = next(iter(result["unpriced"].values()))
    assert "4" in why and "2" in why, why


def test_the_matching_width_is_priced(tmp_path, monkeypatch, priceable):
    """The check must not refuse a run that is what it says it is."""
    _group_of(monkeypatch, 4)
    result = microbench.price_graph(_collective_graph(tmp_path, 4),
                                    iters=4, warmup=0, cache="hot")
    assert result["prices"], result["unpriced"]


def test_a_collective_graph_with_no_communicator_is_refused(tmp_path,
                                                            monkeypatch,
                                                            priceable):
    """A local copy is not a four-rank reduction, however fast it is."""
    _group_of(monkeypatch, None)
    result = microbench.price_graph(_collective_graph(tmp_path, 4),
                                    iters=4, warmup=0, cache="hot")
    assert not result["prices"], (
        "priced a 4-rank collective with no group at all, and labelled it "
        f"{result['provenance']['topology']}")


def test_a_tp4_shaped_gemm_on_one_device_is_still_priced(tmp_path, monkeypatch,
                                                         priceable):
    """The check is about collectives, not about the width of a shape.

    A GEMM shaped for a TP4 shard is an ordinary single-device matrix multiply.
    Its cost does not depend on how many other ranks exist and refusing it
    would remove the largest measurable part of a sharded model from any
    single-GPU pricing run.
    """
    path = tmp_path / "gemm.json"
    path.write_text(json.dumps({
        "key": {"topology": [["tp", 4]]},
        "ops": [{"name": "aiter::gemm_a16w16",
                 "input_shapes": [[32, 4352], [5120, 4352]],
                 "dtypes": ["bfloat16", "bfloat16"], "scalars": []}],
    }))
    _group_of(monkeypatch, None)
    result = microbench.price_graph(str(path), iters=4, warmup=0, cache="hot")
    assert result["prices"], result["unpriced"]


# == (3) the padding sentinel in a shifted slot_mapping ====================

def test_padding_stays_padding_in_every_variant():
    """Three real rows in a bucket of four, at several regions.

    The fourth entry is -1: not a slot, a statement that there is no slot. The
    kernel tests the sign and skips the write. Adding the region offset to it
    makes it a valid address -- and specifically an address in the region the
    previous call just wrote, so the padding row overwrites live KV.
    """
    slots = [0, 1, 2, -1]
    for v in range(1, 5):
        offset = v * 64 * 16
        shifted = forward_ctx.shift_addresses(slots, offset)
        assert shifted[-1] == -1, (
            f"variant {v} turned the padding sentinel into {shifted[-1]}, "
            "which is a real address in an earlier region")
        assert shifted[:3] == [offset, offset + 1, offset + 2]


def test_variant_zero_is_the_recorded_call_unchanged():
    slots = [0, 1, 2, -1]
    assert forward_ctx.shift_addresses(slots, 0) == slots


def test_block_table_padding_is_not_shifted_into_a_real_block():
    """The same shift is applied to block ids, so the same rule applies."""
    assert forward_ctx.shift_addresses([7, 8, -1], 128) == [135, 136, -1]


def test_every_negative_is_carried_through_not_only_minus_one():
    """The contract is "negative means no address", not "-1 is special"."""
    assert forward_ctx.shift_addresses([-7, 3], 100) == [-7, 103]


def test_the_arithmetic_this_replaced_aliased_into_the_previous_region():
    """Kept as the reason the helper exists, not as a test of deleted code.

    Three rows in a bucket of four, one region along: the shift the installer
    used to apply mapped the -1 to 1023, which is not a sentinel and not an
    unused address -- it is the last slot of region 0, the region the previous
    call wrote. The padding row's write lands on live KV.
    """
    slots = [0, 1, 2, -1]
    stride, block_size, v = 64, 16, 1
    offset = v * stride * block_size
    aliased = [x + offset for x in slots]
    assert aliased[-1] == 1023
    assert 0 <= aliased[-1] < offset, "inside the previous region"
    assert forward_ctx.shift_addresses(slots, offset)[-1] == -1
