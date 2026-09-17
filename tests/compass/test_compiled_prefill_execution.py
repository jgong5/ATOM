"""The effective M calibration preserves raw prices, A/P and decode behavior."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from atom.compass.core.cost.base import StepCost, StepShape
from atom.compass.core.cost.compiled_prefill_execution import CompiledPrefillExecution


@pytest.fixture
def rule():
    return CompiledPrefillExecution(dict(parameters=dict(alpha=.95, floor_seconds=dict(cold=.10, cached=.12)),
        observed_domain=dict(query_tokens=[1, 16384], rows=[1, 3])), "source", ())


def shape(q=32, history=0, prefill=True, bucket=None):
    return StepShape((q,), (q + history,), num_prefill_tokens=q if prefill else 0,
                     produces_output=True, compiled=True, capture_bucket=bucket, topology={"tp": 1})


def cost(body, head=0., prefix=0.):
    return StepCost(seconds=body + head + .002 + .0001,
        breakdown={"<body>": body, "<head>": head, "<prepare>": .002, "<postprocess>": .0001},
        output_ready_seconds=prefix, preparation_seconds=.002)


def test_floor_is_inside_M_and_original_prices_are_preserved(rule):
    original = cost(.02, .01)
    result = rule.apply(original, shape(history=32))
    assert result.model_seconds == .12
    assert result.seconds == pytest.approx(.1221)
    assert result.output_ready_seconds == pytest.approx(.122)
    assert result.breakdown["<body>"] == .02 and result.breakdown["<head>"] == .01
    assert result.breakdown["<prepare>"] == .002 and result.breakdown["<postprocess>"] == .0001
    assert sum(result.breakdown.values()) == pytest.approx(result.seconds)


def test_large_cold_forward_keeps_asynchronous_return(rule):
    result = rule.apply(cost(8.), shape(q=8192))
    assert result.model_seconds == pytest.approx(7.6)
    assert result.output_ready_seconds == pytest.approx(.102)
    assert result.output_ready_seconds < result.seconds / 10
    assert result.breakdown["<compiled-prefill-execution-adjustment>"] == pytest.approx(-.4)


def test_cached_prefix_scales_without_double_charging_MHA_over(rule):
    result = rule.apply(cost(8., prefix=4.002), shape(q=8192, history=8192))
    assert result.output_ready_seconds == pytest.approx(3.802)
    assert result.seconds == pytest.approx(7.6021)


def test_captured_decode_is_unchanged(rule):
    original = cost(.025)
    assert rule.apply(original, shape(q=1, history=32, prefill=False, bucket=1)) is original


def test_width_four_candidate_is_cached_only_and_preserves_parameters(rule):
    added = CompiledPrefillExecution(dict(parameters=rule.parameters,
        observed_domain=rule.domain, native_width_extension=dict(maximum_rows=4,cached_only=True)), "later-source", ())
    cached = StepShape((1,)*4,(33,)*4,num_prefill_tokens=4,produces_output=True,
                       compiled=True,topology={"tp":1})
    with pytest.raises(ValueError, match="outside its source work scope"):
        rule.apply(cost(.2),cached)
    assert added.apply(cost(.2),cached).model_seconds == pytest.approx(.19)
    assert added.parameters == rule.parameters
    cold = StepShape((1,)*4,(1,)*4,num_prefill_tokens=4,produces_output=True,
                     compiled=True,topology={"tp":1})
    with pytest.raises(ValueError, match="cached prefill only"):
        added.apply(cost(.2),cold)
    too_wide = StepShape((1,)*5,(33,)*5,num_prefill_tokens=5,produces_output=True,
                         compiled=True,topology={"tp":1})
    with pytest.raises(ValueError, match="outside its source work scope"):
        added.apply(cost(.2),too_wide)


def test_separate_width_fit_preserves_all_original_regimes(rule):
    added = CompiledPrefillExecution(dict(parameters=rule.parameters,
        observed_domain=rule.domain, native_width_extension=dict(maximum_rows=4, cached_only=True,
            parameters=dict(alpha=1.07, floor_seconds=.107))), rule.sha256, ())
    for history in (0, 32):
        for body in (.02, 5.4):
            original_shape = shape(history=history)
            assert added.apply(cost(body), original_shape) == rule.apply(cost(body), original_shape)
    cached = StepShape((4096,)*4, (4128,)*4, num_prefill_tokens=16384,
                      produces_output=False, compiled=True, topology={'tp': 1})
    assert added.apply(cost(5.4), cached).model_seconds == pytest.approx(5.778)
    assert added.apply(cost(.02), cached).model_seconds == pytest.approx(.107)


def test_actual_frozen_rule_source_integrity():
    value = os.environ.get("ATOMCOMPASS_EXECUTION_RULE")
    if not value:
        pytest.skip("set ATOMCOMPASS_EXECUTION_RULE to the frozen source rule")
    import hashlib
    path = Path(value);data = json.loads(path.read_text())
    quotes = json.loads(Path(data["evidence"]["body_quotes"]["path"]).read_text())
    inputs = json.loads(Path(quotes["loaded_inputs"]["path"]).read_text())
    options = json.loads(Path(data["body_options"]["path"]).read_text())
    oracle = SimpleNamespace(compass_loaded_inputs=inputs)
    loaded = CompiledPrefillExecution.load(str(path), hashlib.sha256(path.read_bytes()).hexdigest(), oracle=oracle, options=options)
    assert loaded.parameters == data["parameters"]
    # A/P sources are not B inputs; a changed primitive remains a hard failure.
    oracle.compass_loaded_inputs = [dict(row, sha256="0" * 64) if row["role"].startswith("oracle.native_ap_regions") else row for row in inputs]
    CompiledPrefillExecution.load(str(path), hashlib.sha256(path.read_bytes()).hexdigest(), oracle=oracle, options=options)
    oracle.compass_loaded_inputs = [dict(row, sha256="0" * 64) if row["role"] == "oracle.attention_scope" else row for row in inputs]
    with pytest.raises(ValueError, match="B sources changed"):
        CompiledPrefillExecution.load(str(path), hashlib.sha256(path.read_bytes()).hexdigest(), oracle=oracle, options=options)
