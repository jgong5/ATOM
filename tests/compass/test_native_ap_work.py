"""Work selection and source isolation for the native chain A/P consumer."""
import copy
import json
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from atom.compass.core.cost.native_ap_work import (
    A_FEATURES, A_SCALES, P_SCALES, NativeAPWorkRegions,
    _source_scope, observation_vectors, validate_source_model,
)
from atom.compass.runtime.templates import NativeAllocation
from .test_native_ap_exact import offer
from .test_native_ap_regions import descriptor, point
from .test_native_prefill_regions import adapter


@pytest.fixture
def candidate():
    allocation = NativeAllocation(block_size=16, max_model_len=262144,
        position_rows=3, cudagraph_mode="full", capture_region_context=True)
    base = adapter(allocation)
    model = dict(observed_feature_bounds=dict(zip(A_FEATURES[1:],
        ([1, 16384], [3, 16632], [1, 3], [0, 3], [0, 1], [0, 1]))),
        models=dict(prepare=dict(scales=A_SCALES, coefficients=[.001, .002, .003, .004, .005, .006, .007]),
                    postprocess=dict(scales=P_SCALES, coefficients=[.01, .02, .03],
                                     source_groups=[dict(features=[1, 1, 1]), dict(features=[1, 3, 3])])) )
    return NativeAPWorkRegions(base, model, "source", (), allocation), allocation


def fresh_descriptor(q, history, blocks, output):
    d = descriptor(point("unused-label", q, history, blocks, output))
    # Independent request KV tables are the chain source's observed scope.
    d["block_tables"] = [list(range(100000 * (i + 1), 100000 * (i + 1) + b)) for i, b in enumerate(d["blocks"])]
    d.update(midstep_saves_empty=True, state_maintenance=dict(relocations=[], checkpoint_stores=[], checkpoint_restores=[]))
    d["forward_context"]["scope"] = dict(d["forward_context"]["scope"])
    return d


def test_unseen_n3_work_uses_all_sampled_rows_and_preserves_renaming(candidate):
    regions, allocation = candidate
    d = fresh_descriptor([7, 8192, 48], [195984, 8192, 688], [12250, 1024, 46], True)
    shape = offer(allocation, d, change={"prior_sampled_batch_rows": 2})
    first = regions.breakdown(shape)
    assert first["<postprocess>"] == pytest.approx(.01 + .02 + .03 * 2 / 3)
    offer(allocation, d, change={"prior_sampled_batch_rows": 2}, rename=500000)
    assert regions.breakdown(shape) == first
    assert regions.source_qualified is False


def test_p0_has_structural_zero_with_empty_previous_output(candidate):
    regions, allocation = candidate
    d = fresh_descriptor([8192, 48, 48], [180224, 0, 0], [12250, 4, 4], False)
    shape = offer(allocation, d, change={"prior_sampled_batch_rows": 0})
    assert regions.breakdown(shape)["<postprocess>"] == 0


@pytest.mark.parametrize("change", [
    {"prior_sampled_batch_rows": 4}, {"prior_sampled_has_logprobs": True},
    {"num_spec_step": 1}, {"state_maintenance_empty": False},
])
def test_unobserved_runtime_or_queue_refuses(candidate, change):
    regions, allocation = candidate
    d = fresh_descriptor([15], [222640], [13916], True)
    shape = offer(allocation, d, change=change)
    assert regions.refusal(shape)


def test_large_allocation_refuses_and_shared_kv_preserves_ap_work(candidate):
    regions, allocation = candidate
    d = fresh_descriptor([15], [222640], [17000], True)
    shape = offer(allocation, d)
    assert "bounds" in regions.refusal(shape)
    d = fresh_descriptor([15, 15], [32, 32], [3, 3], True)
    shape = offer(allocation, d)
    expected = regions.breakdown(shape)
    d["block_tables"][1][0] = d["block_tables"][0][0]
    shape = offer(allocation, d)
    assert regions.breakdown(shape) == expected
    d["state_fork_srcs"][0] = d["state_slots"][1]
    shape = offer(allocation, d)
    assert "state alias" in regions.refusal(shape)


def test_retained_tiny_source_wins_without_work_context(candidate, monkeypatch):
    regions, _ = candidate
    shape = SimpleNamespace()
    monkeypatch.setattr(type(regions.base), "_cell", lambda *_: {"retained": True})
    monkeypatch.setattr(type(regions.base), "refusal", lambda *_: None)
    monkeypatch.setattr(type(regions.base), "breakdown", lambda *_: {"<prepare>": .0002, "<postprocess>": .0001})
    assert regions.breakdown(shape) == {"<prepare>": .0002, "<postprocess>": .0001}


def test_sampling_feature_is_native_tensor_shape_not_output_mask():
    d = fresh_descriptor([15, 8192, 7504], [195984, 8192, 688], [12250, 2191, 2191], True)
    row = dict(descriptor=d, forward_context_after_return=dict(previous_sampled_ids=dict(shape=[3])))
    assert sum(d["output_rows"]) == 1
    assert observation_vectors(row)[1] == [1, 3, 1]
    del row["forward_context_after_return"]
    with pytest.raises(ValueError, match="tensor"):
        observation_vectors(row)


def test_empty_prior_uses_separate_P_source_without_refitting_A(candidate):
    regions, allocation = candidate
    initial = dict(conditions={str(n): dict(postprocess_seconds=.00012 + n * .000001) for n in (1, 2, 3)})
    added = replace(regions, allocation=allocation, initial_postprocess=initial)
    d = fresh_descriptor([15, 8192, 7504], [195984, 8192, 688], [12250, 2191, 2191], True)
    shape = offer(allocation, d, change={"prior_sampled_batch_rows": 0})
    assert "sampler/queue bounds" in regions.refusal(shape)
    assert added.breakdown(shape)["<postprocess>"] == pytest.approx(.000123)
    assert added.model is regions.model
    from atom.compass.core.cost.composition_qualification import offer_observation

    decode = fresh_descriptor([1], [33], [3], True)
    decode.update(prefill_rows=0, capture_bucket=1)
    shape = offer_observation(allocation, dict(descriptor=decode))
    assert added.breakdown(shape) == regions.breakdown(shape)


def test_signed_negative_prepare_is_retained_and_short_cohort_refuses():
    d = fresh_descriptor([32], [0], [3], False)
    row = dict(chain_id="source_chain", chain_step=0, repetition=0, role="source", normal_return=True,
               descriptor=d, seconds=dict(prepare=-.000001, run_model=.1, postprocess=0., forward=.099999))
    with pytest.raises(ValueError, match="six source"):
        _source_scope([row], d["forward_context"]["scope"], {"source_chain"})
    assert row["seconds"]["prepare"] == -.000001


def test_actual_frozen_source_model_and_tampering():
    root = os.environ.get("ATOMCOMPASS_AP_WORK_SOURCE")
    if not root:
        pytest.skip("set ATOMCOMPASS_AP_WORK_SOURCE to an immutable source snapshot")
    root = Path(root)
    model = json.loads((root / "SOURCE_MODEL_FREEZE.json").read_text())
    rows = json.loads((root / "SOURCE_ROWS.json").read_text())["rows"]
    assert validate_source_model(model, rows)
    wrong = copy.deepcopy(model)
    wrong["models"]["prepare"]["coefficients"][0] += .001
    with pytest.raises(ValueError, match="coefficients"):
        validate_source_model(wrong, rows)
    wrong_rows = copy.deepcopy(rows)
    wrong_rows[0]["role"] = "heldout"
    with pytest.raises(ValueError, match="source-only"):
        validate_source_model(model, wrong_rows)
