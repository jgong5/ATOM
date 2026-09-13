"""Deferred token identity and host-visible publication are separate contracts."""

from dataclasses import replace
import queue
import types

import pytest

from atom.compass.core.cost.base import StepCost, StepShape
from atom.compass.core.cost.families.adapter import ParametricPriceLibrary
from atom.compass.core.cost.library import LibraryCostOracle
from atom.model_engine.engine_core import EngineCore
from atom.utils.clock import VirtualClock, WallClock, get_clock, reset_clock, set_clock

from .test_attention_family import _unified
from .test_rank_aggregation import _runner


SCOPE = {"unified": {"attention_backend": (
    ("backend", "atom.model_ops.attentions.aiter_attention.AiterBackend"),
    ("impl", "atom.model_ops.attention_mha.PagedAttentionImpl"),
    ("ATOM_USE_UNIFIED_ATTN", "False"),
)}}


class TimedLibrary(ParametricPriceLibrary):
    def lookup(self, op, *_args):
        return {"seconds": op["test_seconds"], "kernels": {"test": 0.0}}, "test"


def composition(*, cached=True, prefill=True, produces=True):
    library = TimedLibrary()
    library.request_attention_scope = SCOPE
    attn = _unified([8], [16 if cached else 8],
                    is_prefill=prefill, has_cached=cached)
    ops = [{"name": "prefix", "test_seconds": 2.0},
           dict(attn, test_seconds=3.0),
           {"name": "between", "test_seconds": 5.0},
           dict(attn, test_seconds=7.0),
           {"name": "suffix", "test_seconds": 11.0}]
    graph = {"key": {"topology": [["tp", 1]]}, "ops": ops}
    source = types.SimpleNamespace(graph_for=lambda shape: graph)
    regions = types.SimpleNamespace(refusal=lambda shape: None,
        breakdown=lambda shape: {"<prepare>": 1.0, "<postprocess>": 2.0})
    oracle = LibraryCostOracle(library, source, seconds_per_launch=0.1,
                               regions=regions, require_complete=True)
    shape = StepShape((8,), (16 if cached else 8,), 8 if prefill else 0,
                      produces_output=produces)
    return oracle, shape, attn


def test_ordered_prefix_leaves_the_last_opaque_operator_and_suffix_after_output():
    oracle, shape, _ = composition()
    cost = oracle.estimate(shape)
    assert cost.seconds == pytest.approx(31.5)
    # 2 + 3 + 5 before the last sync, plus three launch terms and preparation.
    assert cost.output_ready_seconds == pytest.approx(11.3)
    assert cost.seconds - cost.output_ready_seconds == pytest.approx(20.2)
    assert cost.output_ready_basis["priced_operator_index"] == 3
    # Cached totals must preserve the boundary, without another pricing pass.
    assert oracle.estimate(shape) == cost
    assert oracle.price_cache_hits == 1


@pytest.mark.parametrize("over", [{"cached": False}, {"prefill": False},
                                  {"produces": False}])
def test_cold_decode_and_outputless_paths_keep_the_old_timing(over):
    oracle, shape, _ = composition(**over)
    assert oracle.estimate(shape).output_ready_seconds == 0.0


def test_unified_triton_or_unstated_backend_does_not_inherit_asm_synchronization():
    oracle, shape, _ = composition()
    oracle.library.request_attention_scope = {
        "unified": {"attention_backend": dict(SCOPE["unified"]["attention_backend"],
                                               ATOM_USE_UNIFIED_ATTN="True")}}
    assert oracle.estimate(shape).output_ready_seconds == 0.0
    other, shape, _ = composition()
    other.library.request_attention_scope = None
    assert other.estimate(shape).output_ready_seconds == 0.0


def test_offset_outside_the_step_cannot_drop_or_add_time():
    with pytest.raises(ValueError, match="within the step"):
        StepCost(2.0, output_ready_seconds=3.0)


def test_actual_engine_loop_waits_for_the_blocking_rank_without_changing_token_identity():
    class PerRank:
        def estimate(self, shape):
            rank = shape.rank_coords["tp"]
            return StepCost(4.0 if rank == 0 else 5.0,
                            output_ready_seconds=3.0 if rank == 0 else 1.0)

    clock = VirtualClock(epoch=100.0)
    set_clock(clock)
    try:
        runner = _runner("slowest", PerRank())
        runner._deferred_output = [0]  # a short request's first token is pending
        runner._offer_allocation = lambda batch: None
        runner._describe = lambda batch: StepShape((8,), (16,), 8,
            topology={"tp": 2}, rank_coords={"tp": 0})
        rows = []
        runner._record_measurement = lambda *args, **kw: rows.append(kw)
        batch = types.SimpleNamespace(req_ids=[1], total_seqs_num_prefill=1,
            total_seqs_num_decode=0, is_dummy_run=False,
            produces_output=lambda: True)
        published = []
        core = EngineCore.__new__(EngineCore)
        core.scheduler = types.SimpleNamespace(
            schedule=lambda: (batch, {1: object()}), take_rejected=lambda: [],
            compute_detailed_aggregates=lambda *_args: None, prefill_delayer=None,
            postprocess=lambda seqs, out, **kw: published.append(
                (get_clock().time(), out.req_ids, out.is_deferred_out)) or [])
        core.runner_mgr = types.SimpleNamespace(
            call_func=lambda name, b, **kw: runner.forward(b))
        core.kv_transfer_enabled = False
        core._poll_kv_transfer_progress = lambda: None
        core.output_queue = queue.Queue()
        core.stream_output_queue = queue.Queue()
        assert core._process_engine_step_inner() is True
        assert published == [(103.0, [0], True)]
        assert clock.time() == 105.0  # max total charged exactly once
        assert rows[0]["ranks"]["slowest_rank"] == 1
        assert rows[0]["ranks"]["output_ready_rank"] == 0
    finally:
        reset_clock()
