"""Exact supplement composition needs fresh source closure, never a candidate alone."""

from dataclasses import replace
import ast
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from atom.compass.core.cost.base import StepShape
from atom.compass.core.cost.region_supplement import (
    EVIDENCE_ROLES, PrefillRegionSupplement, SUPPLEMENT_KEYS,
)
from atom.compass.core.cost.root_prefill import ExactPrefillRegions
from atom.compass.core.loaded_input import LoadedInput
from atom.compass.runtime import cache_region_oracle as runtime
from .test_cache_region_overlay import artifact as overlay_fixture


ORIGINAL_KEYS = ((496, 496, False), (2, 498, True), (32, 32, False),
                 (16352, 16384, False), (2960, 2992, False), (1, 2993, True),
                 (8160, 8192, False), (3040, 11232, False), (10, 11242, True))
SCOPE = "a" * 64


def pin(name):
    return {"path": name, "sha256": hashlib.sha256(name.encode()).hexdigest()}


def summary(values):
    low, median, high = sorted(values)
    return dict(seconds=median, all_three=values, minimum=low, maximum=high,
                range_over_median=(high - low) / median if median else None,
                structural_zero=not median)


def bundle(tmp_path, mutate=lambda objects: None):
    cells, predictions, references, joins = [], {}, {}, {}
    for i, (q, history, output) in enumerate(sorted(SUPPLEMENT_KEYS)):
        name = f"region_{i}"
        cells.append(dict(cell_id=name, query=q, total_history=history,
                          produces_output=output, cached_prefix=history - q, request=i % 7))
        values = {"prepare": [.001, .003, .002],
                  "postprocess": [.0001, .00012, .00011] if output else [0., 0., 0.]}
        references[name] = {c: summary(v) for c, v in values.items()}
        predictions[name] = {c: v["seconds"] for c, v in references[name].items()}
        joins[name] = [dict(repeat=r, observed_phase=f"regions_reference_r{r}_request{i % 7}",
                           **{c + "_seconds": v[r - 1] for c, v in values.items()}) for r in (1, 2, 3)]
    fixtures = dict(warm=[pin("fresh-warm")], heldout=[pin(f"fresh-{r}") for r in (1, 2, 3)])
    pins = {role: {"path": str(tmp_path / f"{role}.json")} for role in EVIDENCE_ROLES}
    candidate = dict(schema="compass.root1493_region_supplement_candidate/1", cells=cells,
        predictions=predictions, reference_points=references, reference_join_rows=joins,
        fresh_confirmation_timings_read=False, target_timings_used=False, candidate_activated=False,
        source_consumption_forbidden=True, old_r3_heldouts_previously_inspected=True,
        relative_spread_qualification_gate=None, absolute_error_limits=dict(prepare=.00011, postprocess=.00011),
        original_nine_cell_freeze=pin("original-freeze"), original_nine_cell_verdict=pin("original-verdict"),
        fresh_fixtures=fixtures, fresh_fixture_specification=dict(warm_seed=1495000,
                                                               heldout_seeds=[1495001, 1495002, 1495003]))
    plan = dict(schema="compass.root1493_region_supplement_executable/1",
                candidate_freeze=pins["candidate"], cells=cells, fresh_fixtures=fixtures)
    phase = dict(schema="compass.root1493_region_supplement_phase/1", plan=pins["plan"],
        prediction_freeze=pins["candidate"], phase="heldout", fixtures=fixtures,
        source_requests=28, primitive_requests=0, target_requests=0,
        records=[dict(cell_id=c["cell_id"], repeat=r, fixture=fixtures["heldout"][r - 1],
                      raw=pin(f"fresh-{r}-{c['cell_id']}")) for r in (1, 2, 3) for c in cells])
    checks = []
    for cell in cells:
        for component, value in predictions[cell["cell_id"]].items():
            checks.append(dict(cell_id=cell["cell_id"], component=component, prediction=value,
                observed=summary([value] * 3), error_seconds=0., absolute_limit_seconds=.00011,
                **{"pass": True}))
    verdict = dict(schema="compass.root1493_region_supplement_verdict/1", plan=pins["plan"],
        prediction_freeze=pins["candidate"], heldout_evidence=dict(phase_result=pins["heldout_phase"]),
        source_qualified=True, all_error_gates_pass=True, candidate_activated=False,
        target_timings_used=False, fresh_confirmation_only=True,
        old_r3_heldouts_used_for_validation=False, checks=checks)
    complete = dict(schema="compass.root1493_region_supplement_native_complete/1", plan=pins["plan"],
        prediction_freeze=pins["candidate"], phase_result=pins["heldout_phase"], verdict=pins["verdict"],
        source_requests=28, primitive_requests=0, target_requests=0, candidate_activated=False)
    handoff = dict(schema="compass.root_prefill_region_supplement_export/1", evidence=pins,
        scope=dict(model="Qwen/Qwen3.8-27B", topology={"tp": 1}, dtype="bfloat16",
                   num_sequences=1, request_scope_sha256=SCOPE),
        source_qualified=True, all_error_gates_pass=True, candidate_activated=False,
        target_timings_used=False, fit_inputs_are_references_only=True,
        heldout_timings_used_as_fit_inputs=False)
    objects = dict(candidate=candidate, plan=plan, heldout_phase=phase, verdict=verdict,
                   native_complete=complete, handoff=handoff)
    mutate(objects)
    for role in ("candidate", "plan", "heldout_phase", "verdict", "native_complete", "handoff"):
        path = tmp_path / f"{role}.json"
        path.write_text(json.dumps(objects[role], sort_keys=True))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if role != "handoff":
            pins[role]["sha256"] = digest
    loaded = tuple(LoadedInput("oracle.root_prefill_region_" + role,
                              str(tmp_path / f"original-{role}.json"),
                              str(tmp_path / f"original-{role}.json"), False,
                              pin("original-" + role)["sha256"], 1) for role in ("freeze", "verdict"))
    root = SimpleNamespace(region_points=tuple((q, h, o, .004, .0002 if o else 0.)
        for q, h, o in ORIGINAL_KEYS), handoff_sha256="b" * 64, source_qualified=True, loaded_inputs=loaded)
    overlay = runtime.model_from_artifact(overlay_fixture.__wrapped__())
    base = ExactPrefillRegions(overlay, root.region_points, root.handoff_sha256)
    return str(path), digest, root, base


def load_bundle(args):
    path, sha, root, base = args
    return PrefillRegionSupplement.load(base, root, path, sha, deployment_scope_sha256=SCOPE)


def test_all_nineteen_prefill_shapes_use_exact_qualified_sources(tmp_path):
    args = bundle(tmp_path)
    model = load_bundle(args)
    assert len(model.points) == 10 and len(model.base.points) == 9
    assert model.base is args[3] and model.base.points is args[2].region_points
    assert model.base.base.diagnostic_outputless is None
    assert model.base.base.diagnostic_final is None
    for q, history, output, prepare, postprocess in (*model.points, *model.base.points):
        shape = StepShape((q,), (history,), num_prefill_tokens=q, compiled=True, produces_output=output)
        assert model.refusal(shape) is None
        assert model.breakdown(shape) == {"<prepare>": prepare, "<postprocess>": postprocess}
        assert (model._selection(shape) is not None) == ((q, history, output) in SUPPLEMENT_KEYS)
    for other in (replace(shape, topology={"tp": 2}), replace(shape, capture_bucket=1),
                  replace(shape, compiled=False), replace(shape, num_prefill_tokens=0)):
        assert model._selection(other) is None
        assert model.refusal(other) == model.base.refusal(other)
    assert {s.role for s in model.loaded_inputs} == {
        "oracle.region_supplement_" + role for role in (*EVIDENCE_ROLES, "sources")}


@pytest.mark.parametrize("mutate", [
    lambda o: o["handoff"].update(source_qualified=False),
    lambda o: o["verdict"].update(source_qualified=False),
    lambda o: o["verdict"].update(old_r3_heldouts_used_for_validation=True),
    lambda o: o["native_complete"].update(source_requests=27),
    lambda o: o["heldout_phase"]["records"].pop(),
    lambda o: o["heldout_phase"]["records"][0].update(fixture=pin("old-r3-heldout")),
    lambda o: o["verdict"]["checks"].__setitem__(0, o["verdict"]["checks"][1]),
    lambda o: o["verdict"]["checks"][0].update(absolute_limit_seconds=.1),
    lambda o: o["candidate"]["predictions"]["region_0"].update(prepare=.01),
    lambda o: o["candidate"]["reference_join_rows"]["region_0"][0].update(observed_phase="regions_heldout_r1_request0"),
    lambda o: o["candidate"].update(original_nine_cell_freeze=pin("replacement")),
    lambda o: o["candidate"]["cells"][0].update(total_history=44834),
])
def test_unqualified_or_mixed_source_closure_is_refused(tmp_path, mutate):
    with pytest.raises(ValueError):
        load_bundle(bundle(tmp_path, mutate))


def test_relabelled_error_gate_and_nonzero_outputless_postprocess_are_refused(tmp_path):
    def relabel(objects):
        check = objects["verdict"]["checks"][0]
        check.update(observed=summary([.1] * 3), error_seconds=abs(.1 - check["prediction"]))
    with pytest.raises(ValueError, match="absolute error gate"):
        load_bundle(bundle(tmp_path, relabel))
    def nonzero(objects):
        name = next(c["cell_id"] for c in objects["candidate"]["cells"] if not c["produces_output"])
        check = next(c for c in objects["verdict"]["checks"] if c["cell_id"] == name and c["component"] == "postprocess")
        check.update(observed=summary([.00001] * 3), error_seconds=.00001)
    with pytest.raises(ValueError, match="structural zeros"):
        load_bundle(bundle(tmp_path, nonzero))


def test_candidate_only_tampered_and_wrong_scope_handoffs_are_refused(tmp_path):
    path, sha, root, base = bundle(tmp_path)
    with pytest.raises(ValueError):
        PrefillRegionSupplement.load(base, root, str(tmp_path / "candidate.json"), sha,
                                     deployment_scope_sha256=SCOPE)
    with pytest.raises(ValueError):
        PrefillRegionSupplement.load(base, root, path, sha, deployment_scope_sha256="wrong")
    (tmp_path / "verdict.json").write_text("{}")
    with pytest.raises(ValueError, match="evidence pin"):
        PrefillRegionSupplement.load(base, root, path, sha, deployment_scope_sha256=SCOPE)


@pytest.mark.parametrize("damage", [None, "missing_read", "unregistered", "aggregate", "unconfigured"])
def test_runtime_api_and_pair_checker_preserve_supplement_inputs(tmp_path, monkeypatch, damage):
    from .test_opening_harness import wrapper_evidence, opening, validate
    from atom.compass.core.cost import root_prefill
    from atom.compass.core.loaded_input import load_json, manifest

    compass, _ = wrapper_evidence.__wrapped__(tmp_path)
    options = compass["oracle_options"]
    options.update(include_failed_outputless=False, include_failed_final=False, diagnostic_only=False)
    scope_sha = hashlib.sha256(Path(options["attention_scope"]).read_bytes()).hexdigest()
    source_dir = tmp_path / "supplement"
    source_dir.mkdir()
    path, sha, root, _ = bundle(source_dir, lambda o: o["handoff"]["scope"].update(request_scope_sha256=scope_sha))
    root_path = tmp_path / "root-handoff.json"
    root_path.write_text("{}")
    _, root_input = load_json(str(root_path), role="oracle.root_prefill_sources")
    # The original primitive/nine-region loader is covered separately. This
    # isolates the added source, through the real runtime, API grouping and checker.
    def original_loader(base, *args, **kwargs):
        return SimpleNamespace(region_points=root.region_points, handoff_sha256=root_input.sha256,
            source_qualified=True, loaded_inputs=base.loaded_inputs + (root_input,) + root.loaded_inputs)
    monkeypatch.setattr(root_prefill, "RootPrefillPrices", original_loader)
    options.update(root_prefill_handoff=str(root_path), root_prefill_handoff_sha256=root_input.sha256,
                   region_supplement_handoff=path, region_supplement_handoff_sha256=sha)
    oracle = runtime.source_cost_oracle(**options)
    rank = manifest(oracle.compass_loaded_inputs)
    rank["regions"] = oracle.compass_region_snapshot
    compass["loaded_inputs"]["ranks"] = [rank]
    api_path = Path(__file__).resolve().parents[2] / "atom/entrypoints/openai/api_server.py"
    tree = ast.parse(api_path.read_text())
    nodes = [n for n in tree.body if
             isinstance(n, ast.FunctionDef) and n.name == "_loaded_option_files" or
             isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "_ROLE_OPTIONS" for t in n.targets)]
    namespace = {"os": os}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(api_path), "exec"), namespace)
    grouped = namespace["_loaded_option_files"]([rank])
    assert len(grouped["region_supplement_handoff"]) == 6
    digests = {k: next(iter(v.values())) if len(v) == 1 else validate._rolled_digest(v) for k, v in grouped.items()}
    compass.update(oracle_option_files=grouped, oracle_option_sha256=digests)
    provenance = dict(kind="source_calibration", measured_at_tp=1, from_target_engine=False,
        sources=[pin("isolated-source")], code={"collector.py": "9" * 64})
    artifacts = [dict(provenance, sha256=row["sha256"], contents={Path(row["path"]).name: row["sha256"]})
                 for row in rank["inputs"]]
    artifacts += [dict(provenance, sha256=digests[key], contents=value) for key, value in grouped.items()]
    artifacts.append(dict(provenance, kind="region_model", sha256=rank["regions"]["sha256"]))
    if damage == "missing_read":
        rank["inputs"] = [r for r in rank["inputs"] if r["role"] != "oracle.region_supplement_verdict"]
    elif damage == "unregistered":
        verdict_sha = next(r["sha256"] for r in rank["inputs"] if r["role"] == "oracle.region_supplement_verdict")
        artifacts = [r for r in artifacts if r["sha256"] != verdict_sha]
    elif damage == "aggregate":
        grouped["region_supplement_handoff"].pop("candidate.json")
    elif damage == "unconfigured":
        options.pop("region_supplement_handoff")
        options.pop("region_supplement_handoff_sha256")
    modelled = SimpleNamespace(manifest={"server": {"compass": compass, "tensor_parallel_size": 1}})
    bad, notes = opening.check_source_contract(modelled, {"artifacts": artifacts}, "7" * 64, {}, "fixture")
    assert bool(bad) is (damage is not None), bad
    assert not any("FAILED" in note for note in notes)
