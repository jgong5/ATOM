"""Cached source additions must not silently expand the accepted domain."""

from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace

import pytest

from atom.compass.core.cache_policy import cache_on_policy
from atom.compass.core.cost.base import StepShape
from atom.compass.core.cost.regions import REGION_MODELS
from atom.compass.runtime import cache_region_oracle as runtime
from atom.compass.runtime.source_oracle import region_snapshot


BASE = "source-27b-tp1-history-2m"


@pytest.fixture
def artifact():
    return {
        "schema": runtime.SCHEMA,
        "name": "test-cached-regions",
        "model": "Qwen/Qwen3.8-27B",
        "base": {"name": BASE, "snapshot_sha256": region_snapshot(
            BASE, REGION_MODELS[BASE])["sha256"]},
        "cache_policy": cache_on_policy(),
        "evidence": {"target_timings_used": False},
        "final": {
            "query_tokens": 16, "cached_history": [16400, 114688],
            "prepare_intercept": .0006, "prepare_slope": -1e-9,
            "postprocess": .0001,
            "validation": {"status": "PASSED", "checks": [{"pass": True}]},
        },
        "outputless": {
            "histories": [16384, 98304], "queries": [16, 1024, 4096, 16384],
            "prepare": [[.0008, .001, .0017, .0046],
                        [.0009, .0011, .0018, .0047]],
            "validation": {"status": "FAILED", "checks": [
                {"pass": True}, {"pass": False, "error_seconds": .00018}]},
        },
    }


def shape(query=16, history=65536, **changes):
    return replace(StepShape(num_scheduled_tokens=(query,),
                            context_lens=(history + query,),
                            num_prefill_tokens=query, compiled=True,
                            produces_output=True, topology={"tp": 1}), **changes)


def test_final_query_boundaries_and_frozen_formula(artifact):
    model = runtime.model_from_artifact(artifact)
    for history in (16400, 65536, 114688):
        s = shape(history=history)
        assert model.refusal(s) is None
        assert model.breakdown(s) == pytest.approx({
            "<prepare>": .0006 - history * 1e-9, "<postprocess>": .0001})
        with pytest.raises(ValueError, match="no calibrated uncertainty"):
            model.band(s)


@pytest.mark.parametrize("s", [
    shape(query=14), shape(query=17), shape(history=0),
    shape(history=16399), shape(history=114689),
    shape(compiled=False), shape(compiled=None), shape(capture_bucket=16),
    shape(topology={"tp": 2}), shape(topology={"dp": 2}),
    shape(rank_coords={"tp": 1}), shape(produces_output=False),
    shape(num_scheduled_tokens=(8, 8), context_lens=(32776, 32776)),
])
def test_new_support_does_not_escape_source_scope(artifact, s):
    model = runtime.model_from_artifact(artifact)
    assert model._selection(s) is None
    assert model.refusal(s) == model.base.refusal(s)


def test_failed_outputless_requires_two_explicit_flags(artifact):
    with pytest.raises(ValueError, match="diagnostic_only"):
        runtime.model_from_artifact(artifact, include_failed_outputless=True)
    final = runtime.model_from_artifact(artifact)
    diagnostic = runtime.model_from_artifact(
        artifact, include_failed_outputless=True, diagnostic_only=True)
    assert final.diagnostic_outputless is None
    assert "FAILED" in diagnostic.diagnostic_outputless.validation
    assert region_snapshot("selected", final)["sha256"] != region_snapshot(
        "selected", diagnostic)["sha256"]
    assert diagnostic.breakdown(shape(256, 65536, produces_output=False))[
        "<postprocess>"] == 0


def test_diagnostic_keeps_frozen_piecewise_surface_even_at_base_points(artifact):
    model = runtime.model_from_artifact(
        artifact, include_failed_outputless=True, diagnostic_only=True)
    for history, row in zip(artifact["outputless"]["histories"],
                            artifact["outputless"]["prepare"]):
        for query, seconds in zip(artifact["outputless"]["queries"], row):
            assert model.breakdown(shape(query, history, produces_output=False))[
                "<prepare>"] == seconds
    # Halfway along both axes between the 1024 and 4096 source queries.
    assert model.breakdown(shape(2560, 57344, produces_output=False))[
        "<prepare>"] == pytest.approx(.0014)
    for s in (shape(15, 65536, produces_output=False),
              shape(16385, 65536, produces_output=False),
              shape(1024, 98305, produces_output=False)):
        assert model._selection(s) is None


def test_published_presets_and_old_answers_stay_unchanged(artifact):
    before = {name: region_snapshot(name, value)
              for name, value in REGION_MODELS.items()}
    model = runtime.model_from_artifact(artifact)
    for s in (shape(16384, 16384, produces_output=False),
              shape(16384, 16384),
              shape(1, 65536, num_prefill_tokens=0, capture_bucket=1)):
        assert model.refusal(s) is None
        assert model.breakdown(s) == model.base.breakdown(s)
        assert model.band(s) == model.base.band(s)
    assert before == {name: region_snapshot(name, value)
                      for name, value in REGION_MODELS.items()}


@pytest.mark.parametrize("mutation", [
    lambda a: a["base"].update(snapshot_sha256="wrong"),
    lambda a: a["final"]["validation"].update(status="FAILED"),
    lambda a: a["final"]["validation"]["checks"][0].update({"pass": False}),
    lambda a: a["final"].update(query_tokens=14),
    lambda a: a["final"].update(cached_history=[1, 1]),
    lambda a: a["final"].update(prepare_intercept=float("nan")),
    lambda a: a["final"].update(postprocess=-1),
])
def test_invalid_source_artifact_is_refused(artifact, mutation):
    mutation(artifact)
    with pytest.raises(ValueError):
        runtime.model_from_artifact(artifact)


def test_failed_source_cannot_be_relabelled_as_passed(artifact):
    artifact["outputless"]["validation"]["status"] = "PASSED"
    with pytest.raises(ValueError, match="FAILED qualification"):
        runtime.model_from_artifact(
            artifact, include_failed_outputless=True, diagnostic_only=True)


def test_factory_attests_loaded_bytes_and_deployment_scope(artifact, tmp_path, monkeypatch):
    path = tmp_path / "overlay.json"
    path.write_text(json.dumps(artifact))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    calls = []

    def build(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(oracle=SimpleNamespace(compass_loaded_inputs=()))

    monkeypatch.setattr(runtime, "build_source_oracle", build)
    options = dict(region_overlay=str(path), region_overlay_sha256=digest,
                   regions=BASE, model="Qwen/Qwen3.8-27B", tp=1,
                   block_size=16, max_model_len=262144, position_rows=3,
                   cudagraph_mode="FULL", allocation="native")
    oracle = runtime.source_cost_oracle(**options)
    assert oracle.compass_loaded_inputs[-1].sha256 == digest
    assert oracle.regions.diagnostic_outputless is None
    assert calls[0]["regions"] == BASE
    assert oracle.compass_region_snapshot == region_snapshot(
        artifact["name"], oracle.regions)
    with pytest.raises(ValueError, match="SHA-256"):
        runtime.source_cost_oracle(**dict(options, region_overlay_sha256="wrong"))
    for key, value in (("tp", 2), ("block_size", 32), ("allocation", "synthetic"),
                       ("cudagraph_mode", "piecewise"), ("position_rows", 1)):
        with pytest.raises(ValueError, match="source scope"):
            runtime.source_cost_oracle(**dict(options, **{key: value}))
    with pytest.raises(ValueError, match="diagnostic_only"):
        runtime.source_cost_oracle(**dict(options, include_failed_outputless="1"))
    assert len(calls) == 1


def test_q16_addition_binds_the_artifact_deployment_scope(artifact, tmp_path, monkeypatch):
    from atom.compass.core.cost import cached_q16

    path = tmp_path / "overlay.json"
    scope = SimpleNamespace(role="oracle.attention_scope", sha256="native-scope")
    added = SimpleNamespace(role="oracle.q16_sources", sha256="q16-source")
    monkeypatch.setattr(runtime, "build_source_oracle", lambda **kw: SimpleNamespace(
        oracle=SimpleNamespace(compass_loaded_inputs=(scope,),
                               library=SimpleNamespace(loaded_inputs=(scope,)))))
    monkeypatch.setattr(cached_q16, "CachedQ16Prices", lambda base, *args: SimpleNamespace(
        loaded_inputs=base.loaded_inputs + (added,)))
    options = dict(region_overlay=str(path), regions=BASE, model="Qwen/Qwen3.8-27B",
                   tp=1, block_size=16, max_model_len=262144, position_rows=3,
                   cudagraph_mode="full", allocation="native",
                   q16_handoff="handoff.json", q16_handoff_sha256="q16-source")
    for declared in (None, "other-scope", "native-scope"):
        artifact["q16_request_scope"] = {"sha256": declared}
        path.write_text(json.dumps(artifact))
        options["region_overlay_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        if declared != "native-scope":
            with pytest.raises(ValueError, match="pinned deployment"):
                runtime.source_cost_oracle(**options)
        else:
            oracle = runtime.source_cost_oracle(**options)
            assert oracle.compass_loaded_inputs[-1] is added
