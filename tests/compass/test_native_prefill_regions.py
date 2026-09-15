"""A/P selection uses native path facts and replaces the base terms once."""
from dataclasses import replace

import pytest

from atom.compass.core.cost.native_prefill_regions import NativePrefillRegions
from atom.compass.core.cost.regions import Measured, RunnerRegions
from atom.compass.runtime.source_oracle import region_snapshot
from .test_native_region_context import native_context


SCOPE = dict(model="Qwen/Qwen3.8-27B", kv_cache_dtype="bf16", compilation_level=3,
    cudagraph_mode="FULL", pipeline_parallel_size=1, enable_prefix_caching=True,
    checkpoint_demand=True, checkpoint_interval_tokens=8192, block_size=16,
    max_model_len=262144, position_rows=3, speculative_config_absent=True,
    num_spec_step=0, state_maintenance_empty=True, midstep_saves_empty=True)


def source_selector(query, history, output):
    return dict(tp=1, pp=1, dp=1, level=3, compiled=True, cudagraph_mode="FULL",
                capture_bucket=None, n=1, q=query, history=history,
                produces_output=output, allocation_blocks=704,
                path="within_request_checkpoint" if history else "cold",
                fork=bool(history), temperature=1.0, top_p=1.0, top_k=-1)


def point(query, history, output, prepare):
    return dict(selector=source_selector(query, history, output),
                components=dict(prepare=dict(median=prepare),
                                postprocess=dict(median=.0001 if output else 0)))


def adapter(allocation):
    measured = Measured(.04, .03, .05, 6, "test base")
    base = RunnerRegions(measured, measured, measured, measured, measured,
                         (1,), (1,), (1, 16384), (1,))
    cells = (
        dict(chunk=0, method="pooled_source_median", **point(8192, 0, False, .002)),
        dict(chunk=1, method="pooled_source_median", **point(3056, 8192, False, .001)),
        dict(chunk=2, method="linear_q_1_to_16_no_extrapolation",
             endpoints=[point(1, 11248, True, .0005), point(16, 11248, True, .0008)]),
    )
    return NativePrefillRegions(base, cells, SCOPE, "a" * 64, (), allocation)


def test_final_region_uses_frozen_endpoints_and_replaces_the_base_terms():
    allocation, shape, _ = native_context()
    regions = adapter(allocation)
    assert regions.base.breakdown(shape) == {"<prepare>": .04, "<postprocess>": .04}
    assert regions.breakdown(shape) == pytest.approx({"<prepare>": .00064, "<postprocess>": .0001})
    assert regions.seconds(shape) == pytest.approx(.00074)
    with pytest.raises(ValueError, match="uncertainty"):
        regions.band(shape)
    # The runtime context source is not a cost coefficient in the snapshot.
    assert region_snapshot("native-prefill", regions)["sha256"]


def test_prefix_hit_cannot_fall_back_to_the_less_specific_base():
    allocation, shape, _ = native_context(prefix_hit=11248)
    regions = adapter(allocation)
    assert regions.base.refusal(shape) is None
    assert "prefix" in regions.refusal(shape)
    with pytest.raises(ValueError, match="prefix"):
        regions.breakdown(shape)


def test_later_continuation_does_not_price_the_historical_admission_hit_again():
    cold, shape, _ = native_context()
    resumed, _, _ = native_context(prefix_hit=8192, continuation=True)
    assert resumed.region_context_for(shape)["prefix_cache_hit_tokens"] == (8192,)
    assert adapter(resumed).breakdown(shape) == adapter(cold).breakdown(shape)


def test_missing_context_refuses_new_cells_and_preserves_unrelated_base_answers():
    allocation, shape, _ = native_context()
    regions = adapter(allocation)
    allocation.clear()
    assert "no native region context" in regions.refusal(shape)
    other = replace(shape, num_scheduled_tokens=(32,), context_lens=(32,), num_prefill_tokens=32,
                    produces_output=False)
    assert regions.breakdown(other) == regions.base.breakdown(other)


@pytest.mark.parametrize("changes", [
    {"blocks": 703}, {"blocks": 705}, {"midstep": True},
    {"batch_changes": {"return_logprobs": [True]}},
    {"batch_changes": {"needs_independent_noise": [True]}},
    {"batch_changes": {"temperatures": [0.0]}},
    {"batch_changes": {"top_ks": [10]}},
    {"batch_changes": {"top_ps": [.9]}},
    {"batch_changes": {"num_spec_step": 1}},
    {"batch_changes": {"state_save_all": [[(9, 11248)]]}},
    {"batch_changes": {"prefill_continuations": [None]}},
    {"config_changes": {"speculative_config": object()}},
    {"config_changes": {"pipeline_parallel_size": 2}},
    {"config_changes": {"model": "another-model"}},
])
def test_incompatible_native_paths_are_refused_despite_permissive_base(changes):
    allocation, shape, _ = native_context(**changes)
    regions = adapter(allocation)
    assert regions.base.refusal(shape) is None
    assert regions.refusal(shape) is not None


def test_library_composition_charges_selected_ap_exactly_once(tmp_path):
    from atom.compass.core.cost.library import LibraryCostOracle, PriceLibrary, StaticGraphs
    from .test_library_oracle import _graph, _op, _price_list

    allocation, shape, _ = native_context()
    regions = adapter(allocation)
    operator = _op("triton::norm", [[8, 16]])
    prices = _price_list(tmp_path, "body.json", [operator], .01)
    oracle = LibraryCostOracle(PriceLibrary.load([(prices, None)]),
        StaticGraphs({StaticGraphs.key(shape): _graph([operator])}), regions=regions, require_complete=True)
    result = oracle.estimate(shape)
    assert result.breakdown["<body>"] == .01
    assert result.breakdown["<prepare>"] == pytest.approx(.00064)
    assert result.breakdown["<postprocess>"] == .0001
    assert result.seconds == pytest.approx(.01074)
    assert result.preparation_seconds == pytest.approx(.00064)
    # Cached primitive arithmetic must not bypass the current native path guard.
    allocation.clear()
    with pytest.raises(ValueError, match="no native region context"):
        oracle.estimate(shape)
