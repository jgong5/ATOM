"""Source evidence is pinned and cannot be refitted or relabelled at loading."""
import copy
import hashlib
import json
from statistics import median
from types import SimpleNamespace

import pytest

from atom.compass.core.cache_policy import cache_on_policy
from atom.compass.core.cost.native_prefill_regions import NativePrefillRegions
from .test_native_prefill_regions import SCOPE, adapter, source_selector
from .test_native_region_context import native_context
from .test_cache_region_overlay import artifact as overlay_fixture


DEPLOYMENT_SCOPE = {"attention_scope": {"unified": {"attention_backend": "source-fixture"},
                                       "gdn": {"gdn_decode_lossy_fast": False}}}
DEPLOYMENT_SCOPE_BYTES = json.dumps(DEPLOYMENT_SCOPE, sort_keys=True)
DEPLOYMENT_SCOPE_SHA = hashlib.sha256(DEPLOYMENT_SCOPE_BYTES.encode()).hexdigest()


def bundle(tmp_path, mutation=None):
    pins = {}

    def write(role, value):
        if mutation:
            mutation(role, value)
        path = tmp_path / (role + ".json")
        path.write_text(json.dumps(value, sort_keys=True))
        pins[role] = dict(path=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        return pins[role]

    runtime = dict(compilation_level=3, cudagraph_mode="FULL", block_size=16, kv_cache_dtype="bf16")
    plan = dict(engine_args=dict(model="Qwen/Qwen3.8-27B", pipeline_parallel_size=1,
                                max_model_len=262144), cache_policy=cache_on_policy(), required_runtime=runtime)
    write("plan", plan)
    identity = dict(plan_sha256=pins["plan"]["sha256"], ownership_sha256="a" * 64,
                    arm="unprofiled_control")
    sources, heldouts, structure = [], [], []
    for case, role, final_q in [("low", "source", 1), ("high", "source", 16),
                                ("left", "heldout", 8), ("right", "heldout", 9)]:
        for rep in range(6):
            for chunk, q, history, output in [(0, 8192, 0, False), (1, 3056, 8192, False),
                                               (2, final_q, 11248, True)]:
                prepare = .002 if chunk == 0 else .001 if chunk == 1 else .0005 + (q - 1) / 15 * (.0008 - .0005)
                post = .0001 if output else 0
                seq_id = case + str(rep)
                row = dict(case=case, role=role, repetition=rep, chunk=chunk,
                    selector=source_selector(q, history, output),
                    seconds=dict(prepare=prepare, postprocess=post, run_model=.01, forward=.01 + prepare + post),
                    timeline=dict(req_ids=[seq_id], q=[q], context=[q + history]))
                (sources if role == "source" else heldouts).append(row)
                structure.append(dict(req_ids=[seq_id], query_lens=[q], context_lens=[q + history],
                                      state_maintenance=dict(relocations=[], checkpoint_stores=0, checkpoint_restores=0)))
    write("source", dict(**identity, rows=sources))
    write("heldout", dict(**identity, rows=heldouts))
    cells = []
    for chunk in range(3):
        points = []
        cases = ("low",) if chunk < 2 else ("low", "high")
        for case in cases:
            selected = [row for row in sources if row["chunk"] == chunk and (chunk < 2 or row["case"] == case)]
            components = {}
            for component in ("prepare", "postprocess"):
                values = [row["seconds"][component] for row in selected]
                components[component] = dict(n=len(values), median=median(values), raw=values)
            points.append(dict(selector=selected[0]["selector"], components=components))
        cells.append(dict(chunk=chunk, method="pooled_source_median", **points[0]) if chunk < 2 else
                     dict(chunk=chunk, method="linear_q_1_to_16_no_extrapolation", endpoints=points))
    write("rule", dict(schema="compass.region_source_rule/1", source_only=True, source_rows=36,
                        gate_seconds=.000110, cells=cells, source_identity=identity,
                        source_sha256=pins["source"]["sha256"], source_plan_sha256=pins["plan"]["sha256"]))
    write("freeze", dict(rule=copy.deepcopy(pins["rule"]), source_input=copy.deepcopy(pins["source"]),
                          source_plan_sha256=pins["plan"]["sha256"], heldout_validation_started=False))
    checks = []
    for case in ("left", "right"):
        for chunk in range(3):
            selected = [row for row in heldouts if row["case"] == case and row["chunk"] == chunk]
            for component in ("prepare", "postprocess"):
                values = [row["seconds"][component] for row in selected]
                checks.append(dict(case=case, chunk=chunk, component=component, predicted=values[0],
                    observed=dict(median=median(values), raw=values), absolute_median_error=0, passed=True))
    write("verdict", dict(schema="compass.region_heldout_validation/1", gate_seconds=.000110,
        passed=True, heldout_rows=36, component_checks=checks, source_identity=identity,
        source_input_sha256=pins["source"]["sha256"], heldout_input_sha256=pins["heldout"]["sha256"],
        source_plan_sha256=pins["plan"]["sha256"], frozen_rule_sha256=pins["rule"]["sha256"]))
    write("structure", dict(steps=structure))
    write("final", dict(storage=dict(resolved_runtime=runtime),
        cache=dict(policy=cache_on_policy(), quiescence=dict(idle=True)), structure=copy.deepcopy(pins["structure"])))
    write("observations", dict(**identity, region_sources=copy.deepcopy(pins["source"]),
        region_heldouts=copy.deepcopy(pins["heldout"]), final=copy.deepcopy(pins["final"])))
    write("native_complete", dict(**identity, success=True, engine_closed=True,
                                 observations=copy.deepcopy(pins["observations"])))
    (tmp_path / "deployment_scope.json").write_text(DEPLOYMENT_SCOPE_BYTES)
    handoff = dict(schema="compass.native_prefill_region_sources/1", scope=SCOPE, evidence=copy.deepcopy(pins),
                   deployment_scope=dict(path="deployment_scope.json", sha256=DEPLOYMENT_SCOPE_SHA))
    reference = write("handoff", handoff)
    return tmp_path / reference["path"], reference["sha256"]


def test_loader_binds_source_freeze_native_completion_and_heldout(tmp_path):
    path, sha = bundle(tmp_path)
    allocation, shape, _ = native_context()
    region = NativePrefillRegions.load(adapter(allocation).base, str(path), sha, allocation,
                                      deployment_scope_sha256=DEPLOYMENT_SCOPE_SHA)
    assert region.breakdown(shape) == pytest.approx({"<prepare>": .00064, "<postprocess>": .0001})
    assert len(region.loaded_inputs) == 11
    assert region.source_handoff_sha256 == sha


@pytest.mark.parametrize("role,change", [
    ("rule", lambda value: value["cells"][0]["components"]["prepare"].update(median=.05)),
    ("heldout", lambda value: value.update(ownership_sha256="b" * 64)),
    ("verdict", lambda value: value.update(passed=False)),
    ("verdict", lambda value: value["component_checks"][0].update(predicted=.5)),
    ("native_complete", lambda value: value.update(engine_closed=False)),
    ("structure", lambda value: value["steps"][0]["state_maintenance"].update(checkpoint_stores=1)),
])
def test_changed_qualification_cannot_be_hidden_by_repinning_files(tmp_path, role, change):
    path, sha = bundle(tmp_path, lambda name, value: change(value) if name == role else None)
    allocation, _, _ = native_context()
    with pytest.raises(ValueError):
        NativePrefillRegions.load(adapter(allocation).base, str(path), sha, allocation,
                                 deployment_scope_sha256=DEPLOYMENT_SCOPE_SHA)


def test_source_bytes_changed_after_handoff_are_refused(tmp_path):
    path, sha = bundle(tmp_path)
    with (tmp_path / "source.json").open("a") as stream:
        stream.write(" ")
    allocation, _, _ = native_context()
    with pytest.raises(ValueError, match="evidence changed: source"):
        NativePrefillRegions.load(adapter(allocation).base, str(path), sha, allocation,
                                 deployment_scope_sha256=DEPLOYMENT_SCOPE_SHA)


def test_different_deployment_attention_scope_cannot_activate_source(tmp_path):
    path, sha = bundle(tmp_path)
    allocation, _, _ = native_context()
    with pytest.raises(ValueError, match="deployment attention scope"):
        NativePrefillRegions.load(adapter(allocation).base, str(path), sha, allocation,
                                 deployment_scope_sha256="f" * 64)


def test_factory_keeps_review_candidate_inactive_until_explicit_source_qualification(
        tmp_path, overlay_fixture, monkeypatch):
    from atom.compass.runtime import cache_region_oracle as runtime
    from atom.compass.core.loaded_input import load_json

    path, sha = bundle(tmp_path)
    overlay = tmp_path / "overlay.json"
    overlay.write_text(json.dumps(overlay_fixture))
    allocation, shape, _ = native_context()
    _, scope_input = load_json(str(tmp_path / "deployment_scope.json"), role="oracle.attention_scope")
    monkeypatch.setattr(runtime, "build_source_oracle", lambda **kwargs:
        SimpleNamespace(oracle=SimpleNamespace(native_allocation=allocation, compass_loaded_inputs=(),
                                               library=SimpleNamespace(loaded_inputs=(scope_input,)))))
    options = dict(region_overlay=str(overlay), region_overlay_sha256=hashlib.sha256(overlay.read_bytes()).hexdigest(),
        regions=overlay_fixture["base"]["name"], native_prefill_handoff=str(path),
        native_prefill_handoff_sha256=sha, model="Qwen/Qwen3.8-27B", tp=1,
        block_size=16, max_model_len=262144, position_rows=3, cudagraph_mode="FULL", allocation="native")
    with pytest.raises(ValueError, match="review candidate"):
        runtime.source_cost_oracle(**options)
    handoff = json.loads(path.read_text())
    handoff["source_qualified"] = True
    path.write_text(json.dumps(handoff))
    options["native_prefill_handoff_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    result = runtime.source_cost_oracle(**options)
    assert result.regions.breakdown(shape) == pytest.approx({"<prepare>": .00064, "<postprocess>": .0001})
    assert result.compass_region_snapshot["requested"] == "native-prefill-sources"
    assert result.compass_loaded_inputs[-1].role == "oracle.native_prefill_regions.structure"
