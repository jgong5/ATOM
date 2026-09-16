"""Exact selectors plus optional tests against the preserved N3 acquisition.

Set ATOMCOMPASS_N3_ARTIFACT_ROOT to the acquisition's agent_scratch directory
and ATOMCOMPASS_N3_CLOSEOUT to its owned EXIT.json to exercise real evidence.
Raw traces and measurement artifacts are intentionally not copied into Git.
"""
import copy
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from atom.compass.core.cost.base import StepShape
from atom.compass.core.cost.native_ap_exact import POLICY, SCHEMA, canonical_state_slots
from atom.compass.core.cost.native_ap_regions import NativeAPFamilyRegions
from atom.compass.runtime.source_oracle import region_snapshot
from atom.compass.runtime.templates import NativeAllocation, NativeStepAllocation
from .test_native_prefill_regions import SCOPE, adapter


def _pin(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


@pytest.fixture
def observed_bundle(tmp_path):
    root = os.environ.get("ATOMCOMPASS_N3_ARTIFACT_ROOT")
    closeout = os.environ.get("ATOMCOMPASS_N3_CLOSEOUT")
    if not root or not closeout:
        pytest.skip("set ATOMCOMPASS_N3_ARTIFACT_ROOT and ATOMCOMPASS_N3_CLOSEOUT for real acquisition evidence")
    root = Path(root)
    run = root / "n3src1/r1/native"
    paths = dict(design=root / "n3src1/DESIGN.json", acquisition_plan=root / "n3src1/PLAN.json",
        source=run / "private/SOURCE_ROWS.json", freeze=run / "SOURCE_FREEZE.json",
        heldout=run / "private/HELDOUT_ROWS.json", verdict=run / "HELDOUT_VALIDATION.json",
        body=root / "gemm8288v1/BODY_RECHECK.json", native_complete=run / "NATIVE_COMPLETE.json",
        initial_runtime=run / "private/INITIAL_RUNTIME.json", final_runtime=run / "private/FINAL_RUNTIME.json",
        copy_closeout=Path(closeout), ordinary_holdout=root / "whole_forward_generalization_v1/ORDINARY_N3_FORWARD_HOLDOUT.json")
    pins = {}
    for role, path in paths.items():
        target = tmp_path / (role + ".json")
        target.write_bytes(path.read_bytes())
        pins[role] = _pin(target)
    allocation = NativeAllocation(block_size=16, max_model_len=262144, position_rows=3,
                                  cudagraph_mode="full", capture_region_context=True)
    base = replace(adapter(allocation), source_qualified=True, allocation=allocation)
    deployment = json.loads(paths["acquisition_plan"].read_text())["request_scope"]["sha256"]
    qualification = dict(schema="compass.native_ap_exact_forward_qualification/1", policy=POLICY,
        source_qualified=True, evidence_sha256={k: v["sha256"] for k, v in pins.items()})
    target = tmp_path / "qualification.json"
    target.write_text(json.dumps(qualification))
    pins["qualification"] = _pin(target)
    handoff = dict(schema=SCHEMA, scope=base.scope, source_qualified=True,
        retained_native_handoff_sha256=base.source_handoff_sha256,
        deployment_scope={"sha256": deployment}, cells=[dict(evidence=pins)])
    return base, allocation, handoff, deployment, tmp_path


def load_bundle(bundle):
    base, allocation, handoff, deployment, path = bundle
    target = path / "handoff.json"
    target.write_text(json.dumps(handoff))
    regions = NativeAPFamilyRegions.load(base, str(target), _pin(target)["sha256"], allocation,
                                        deployment_scope_sha256=deployment)
    return regions, allocation


def offer(allocation, descriptor, *, change=None, rename=0):
    d = copy.deepcopy(descriptor)
    context = {**SCOPE, "prefill_continuations": tuple(d["prefill_continuation"]),
        "output_rows": tuple(d["output_rows"]), "prefix_cache_hit_tokens": tuple(d["prefix_cache_hit_tokens"]),
        "output_state_representation": "predictive_deferred_batch", "prior_sampled_batch_rows": 1,
        "prior_sampled_has_logprobs": False}
    context.update({key: tuple(d[key]) for key in
        ("temperatures", "top_ks", "top_ps", "return_logprobs", "independent_noise")})
    context.update(change or {})
    allocation.offer(NativeStepAllocation(rows=list(zip(d["q"], d["context"])),
        block_tables=[[value + rename for value in row] for row in d["block_tables"]],
        state_slots=[value + rename for value in d["state_slots"]],
        state_fork_srcs=[value + rename if value >= 0 else value for value in d["state_fork_srcs"]],
        state_rows=d["state_rows"], num_prefill_seqs=len(d["q"]), rank_coords={"tp": 0}, region_context=context))
    return StepShape(tuple(d["q"]), tuple(d["context"]), num_prefill_tokens=sum(d["q"]),
        produces_output=d["produces_output"], compiled=True, topology={"tp": 1}, rank_coords={"tp": 0})


def observed_descriptor(bundle):
    reference = bundle[2]["cells"][0]["evidence"]["source"]
    return json.loads(Path(reference["path"]).read_text())["rows"][0]["descriptor"]


def test_real_frozen_source_and_both_forward_checks_load(observed_bundle):
    regions, allocation = load_bundle(observed_bundle)
    shape = offer(allocation, observed_descriptor(observed_bundle))
    assert regions.breakdown(shape) == {"<prepare>": 0.0010659179687504405, "<postprocess>": 0.0}
    assert regions.exact_cells[0]["validation"] == pytest.approx(
        {"independent_relative_error": .012611342098719733, "ordinary_relative_error": .002943307655741007})
    assert len(regions.loaded_inputs) == 14
    verdict_path = observed_bundle[2]["cells"][0]["evidence"]["verdict"]["path"]
    verdict = json.loads(Path(verdict_path).read_text())
    assert verdict["component_residuals"]["prepare"]["absolute_median_error"] > .000110
    assert region_snapshot("native-ap-families", regions)["sha256"]
    offer(allocation, observed_descriptor(observed_bundle), rename=1000000,
          change={"incidental_request_name": "unseen request", "native_debug_counter": 900})
    assert regions.seconds(shape) == 0.0010659179687504405
    with pytest.raises(ValueError, match="uncertainty"):
        regions.band(shape)


@pytest.mark.parametrize("change", [
    {"prior_sampled_batch_rows": 0}, {"prior_sampled_batch_rows": 3},
    {"prior_sampled_has_logprobs": True}, {"output_state_representation": "unknown"},
])
def test_real_source_does_not_qualify_a_different_queue(observed_bundle, change):
    regions, allocation = load_bundle(observed_bundle)
    shape = offer(allocation, observed_descriptor(observed_bundle), change=change)
    assert "queue differs" in regions.refusal(shape)


@pytest.mark.parametrize("change", ["query", "blocks", "state_alias", "short_short_sharing", "nonprefix_sharing"])
def test_exact_source_does_not_widen_geometry_or_aliases(observed_bundle, change):
    regions, allocation = load_bundle(observed_bundle)
    d = copy.deepcopy(observed_descriptor(observed_bundle))
    if change == "query":
        d["q"][1] += 16
        d["context"][1] += 16
    elif change == "blocks":
        d["block_tables"][1].append(999999)
    elif change == "state_alias":
        d["state_slots"][1] = d["state_slots"][0]
    elif change == "short_short_sharing":
        d["block_tables"][2][0] = d["block_tables"][1][0]
    else:
        d["block_tables"][2][-1] = d["block_tables"][1][0]
    shape = offer(allocation, d)
    assert regions.refusal(shape) is not None
    with pytest.raises(ValueError):
        regions.breakdown(shape)


def inflate_heldouts(data):
    for row in data["rows"]:
        row["seconds"]["forward"] += 2
        row["seconds"]["run_model"] += 2


@pytest.mark.parametrize("role,damage", [
    ("qualification", lambda d: d["policy"].update(limit=1.0)),
    ("native_complete", lambda d: d.update(engine_closed=False)),
    ("copy_closeout", lambda d: d["cleanup"].update(writers_released=False)),
    ("heldout", lambda d: d["rows"].pop()),
    ("heldout", inflate_heldouts),
    ("heldout", lambda d: d["rows"][1]["descriptor"].update(req_ids=d["rows"][0]["descriptor"]["req_ids"])),
    ("verdict", lambda d: d.update(host_return_law_inferred=True)),
])
def test_repinning_does_not_qualify_incomplete_or_changed_evidence(observed_bundle, role, damage):
    pins = observed_bundle[2]["cells"][0]["evidence"]
    path = Path(pins[role]["path"])
    data = json.loads(path.read_text())
    damage(data)
    path.write_text(json.dumps(data))
    pins[role] = _pin(path)
    if role != "qualification":
        qpath = Path(pins["qualification"]["path"])
        qualification = json.loads(qpath.read_text())
        qualification["evidence_sha256"][role] = pins[role]["sha256"]
        qpath.write_text(json.dumps(qualification))
        pins["qualification"] = _pin(qpath)
    with pytest.raises(ValueError):
        load_bundle(observed_bundle)


def test_pairwise_facts_include_the_two_short_rows():
    from atom.compass.runtime.templates import block_sharing_pairs

    assert block_sharing_pairs([[1, 2], [3, 4], [3, 5]]) == [[0, 1, 0, 0], [0, 2, 0, 0], [1, 2, 1, 0]]
    assert block_sharing_pairs([[1, 2], [3, 4], [5, 3]])[-1] == [1, 2, 0, 1]
    assert canonical_state_slots([90, 91, 92], [89, -1, -1]) == dict(destinations=[0, 1, 2], sources=[3, -1, -1])


def test_refusal_diagnostic_preserves_offered_queue(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from atom.compass.core.cost import library

    monkeypatch.setenv("COMPASS_REFUSAL_DUMP", str(tmp_path))
    monkeypatch.setattr(library, "_DUMPS_WRITTEN", 0)
    shape = StepShape((8192, 48, 48), (188416, 48, 48), num_prefill_tokens=8288, produces_output=False)
    context = {"prior_sampled_batch_rows": 1, "prior_sampled_has_logprobs": False}
    allocation = SimpleNamespace(region_context_for=lambda value: context)
    library._dump_region_refusal(shape, "unqualified geometry", allocation)
    evidence = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert evidence["why"] == "unqualified geometry"
    assert evidence["native_region_context"] == context
