"""Fixture-only family coefficients exercise transfer guards and source binding."""
import copy
import hashlib
import json
from dataclasses import replace
from statistics import median

import numpy as np
import pytest

from atom.compass.core.cost.base import StepShape
from atom.compass.core.cost.native_ap_regions import (
    EVIDENCE,
    FAMILIES,
    PATH_STRATA,
    NativeAPFamilyRegions,
    source_halfspaces,
    work_features,
)
from atom.compass.core.cost.native_prefill_regions import (
    NativePrefillRegions,
    _breakdown,
    _cell_for,
)
from atom.compass.runtime.templates import NativeAllocation, NativeStepAllocation

from .test_native_prefill_regions import SCOPE, adapter
from .test_native_prefill_sources import DEPLOYMENT_SCOPE_SHA
from .test_native_prefill_sources import bundle as native_bundle


def point(name, q, history, blocks, output, *, role="source"):
    q, history, blocks = [q] if isinstance(q, int) else q, [history] if isinstance(history, int) else history, [blocks] if isinstance(blocks, int) else blocks
    return dict(id=name, role=role, q=q, history=history, blocks=blocks,
                produces_output=output, prefill_continuation=[bool(x) for x in history],
                output_rows=[output and t <= 16 for t in q])


def descriptor(p, *, rename=0):
    n = len(p["q"])
    shared = 2 if n == 2 else 0
    tables = [list(range(rename, rename + shared)) +
              list(range(rename + 10000 * (index + 1), rename + 10000 * (index + 1) + b - shared))
              for index, b in enumerate(p["blocks"])]
    cached = any(p["history"])
    return dict(q=p["q"], history=p["history"], blocks=p["blocks"],
        context=[t + h for t, h in zip(p["q"], p["history"])], block_tables=tables,
        state_slots=list(range(rename + 100, rename + 100 + n)),
        state_fork_srcs=list(range(rename + 200, rename + 200 + n)) if cached else [-1],
        state_rows=list(range(n)),
        prefill_continuation=p["prefill_continuation"], output_rows=p["output_rows"],
        prefix_cache_hit_tokens=[0] * n, temperatures=[1.0] * n, top_ks=[-1] * n,
        top_ps=[1.0] * n, return_logprobs=[False] * n, independent_noise=[False] * n,
        produces_output=p["produces_output"], prefill_rows=n, compiled=True,
        capture_bucket=None, topology={"tp": 1}, rank_coords={"tp": 0},
        forward_context=dict(context_source="native_runner_cpu_metadata", scope=SCOPE,
            deferred_output=True, prior_sampled_batch_rows=1, pending_token_copies=1, pending_logprob_entries=1,
            pending_nonnull_logprob_copies=0, pending_mtp_status_copies=0))


def offer(allocation, p, *, rename=0, changes=None):
    d = descriptor(p, rename=rename)
    n = len(p["q"])
    context = dict(SCOPE, prefill_continuations=tuple(p["prefill_continuation"]),
        output_rows=tuple(p["output_rows"]), temperatures=(1.0,) * n, top_ks=(-1,) * n,
        top_ps=(1.0,) * n, return_logprobs=(False,) * n, independent_noise=(False,) * n,
        output_state_representation="predictive_deferred_batch", prior_sampled_batch_rows=1,
        prior_sampled_has_logprobs=False)
    context["prefix_cache_hit_tokens"] = (0,) * n
    context.update(changes or {})
    allocation.offer(NativeStepAllocation(rows=list(zip(d["q"], d["context"])),
        block_tables=d["block_tables"], state_slots=d["state_slots"], state_fork_srcs=d["state_fork_srcs"],
        state_rows=list(range(n)), num_prefill_seqs=n, rank_coords={"tp": 0}, region_context=context))
    return StepShape(tuple(d["q"]), tuple(d["context"]), num_prefill_tokens=sum(d["q"]),
        produces_output=p["produces_output"], compiled=True, topology={"tp": 1}, rank_coords={"tp": 0})


def family_bundle(path, base, *, failed=None, mutation=None, deployment_scope_sha=DEPLOYMENT_SCOPE_SHA):
    path.mkdir(exist_ok=True)
    pins = {}

    def write(role, value):
        if mutation:
            mutation(role, value)
        target = path / (role + ".json")
        target.write_text(json.dumps(value, sort_keys=True))
        pins[role] = dict(path=target.name, sha256=hashlib.sha256(target.read_bytes()).hexdigest())
        return copy.deepcopy(pins[role])

    points = [point("cold", 32, 0, 33, False)]
    for index, (q, h, b) in enumerate(((16, 32, 33), (16, 32, 704), (8192, 48, 704), (16384, 16384, 2627))):
        points.append(point("n1p0_" + str(index), q, h, b, False))
    for index, (q, h, b) in enumerate(((1, 512, 33), (8, 512, 33), (9, 512, 33), (1, 46576, 2912))):
        points.append(point("n1p1_" + str(index), q, h, b, True))
    for q in (1, 16):
        points.append(point("retained" + str(q), q, 11248, 704, True, role="retained_source"))
    for index, (q, h, b) in enumerate((([1056, 2944], [40960, 48], [2627, 188]),
            ([8192, 16], [32768, 32], [2627, 188]), ([8192, 2944], [32768, 48], [2627, 188]),
            ([8192, 2960], [32768, 48], [2627, 189]))):
        points.append(point("n2p0_" + str(index), q, h, b, False))
    for index, (q, h, b) in enumerate((([7, 16], [42016, 2992], [2627, 188]),
            ([1056, 1], [40960, 3008], [2627, 189]), ([1056, 15], [40960, 2992], [2627, 188]),
            ([1056, 16], [40960, 2992], [2627, 188]))):
        points.append(point("n2p1_" + str(index), q, h, b, True))
    for name in ("cold", "n1p0_0", "n1p1_0", "n2p0_0", "n2p1_0"):
        heldout = copy.deepcopy(next(p for p in points if p["id"] == name))
        heldout.update(id="heldout_" + name, role="heldout")
        points.append(heldout)
    points += [point("transfer_cold", 8192, 0, 704, False, role="transfer"),
               point("transfer_continuing", 3056, 8192, 704, False, role="transfer"),
               point("transfer_final", 8, 11248, 704, True, role="transfer")]
    design = dict(points=points, target_timing_inputs=[], changes_existing_refusals=False)
    write("design", design)
    domains = {}
    for family in FAMILIES:
        strata = copy.deepcopy(PATH_STRATA[family])
        strata["prefill_continuations"] = [[False]] if family == "N1_cold_P0" else [[True] * int(family[1])]
        strata["output_rows"] = PATH_STRATA[family]["output_rows"]
        strata["prior_sampled_batch_rows"] = [1]
        strata["shared_prefix_blocks"] = [2] if family.startswith("N2") else [None]
        vectors = [work_features(p["q"], p["history"], p["blocks"], p["produces_output"])[1]
                   for p in points if p["role"] in ("source", "retained_source") and
                   work_features(p["q"], p["history"], p["blocks"], p["produces_output"])[0] == family]
        domains[family] = dict(strata, **({"exact": {"q": [32], "history": [0], "blocks": [33]}}
            if family == "N1_cold_P0" else {"halfspaces": source_halfspaces([v[1:] for v in vectors])}))
        domains[family]["source_anchor_ids"] = [p["id"] for p in points if p["role"] in ("source", "retained_source")
            and work_features(p["q"], p["history"], p["blocks"], p["produces_output"])[0] == family]
        if family != "N1_cold_P0":
            domains[family]["source_feature_vectors"] = [vector[1:] for vector in vectors]
    write("domain", dict(schema="compass.native_ap_family_domain/1", families=domains,
        design=pins["design"], feature_axes=["totalQ", "totalBlocks", "convWork"]))
    write("acquisition_plan", dict(design=pins["design"], domain=pins["domain"],
                                   request_scope={"sha256": deployment_scope_sha}))
    rows, anchors = {"source": [], "heldout": [], "transfer": []}, {}
    for p in points:
        family, vector = work_features(p["q"], p["history"], p["blocks"], p["produces_output"])
        prepare = .0002 if family == "N1_cold_P0" else sum(x * y for x, y in zip(vector, [.0002, 1e-8, 1e-8, 1e-8]))
        post = .0001 if p["produces_output"] else 0.
        if p["role"] == "retained_source":
            # The retained values are the same unchanged source endpoints as the base fixture.
            prepare = .0005 if p["q"] == [1] else .0008
        if p["role"] == "transfer":
            cell = _cell_for(base.cells, p["q"][0], p["history"][0], p["produces_output"])
            prepare = _breakdown(cell, p["q"][0])["<prepare>"]
        if p["role"] in ("source", "retained_source"):
            anchors[p["id"]] = dict(prepare=prepare, postprocess=post)
        if p["role"] in rows:
            for repetition in range(6):
                rows[p["role"]].append(dict(point_id=p["id"], role=p["role"], repetition=repetition,
                    normal_return=True, descriptor=descriptor(p, rename=100000 * (repetition + 1)),
                    seconds=dict(prepare=prepare, postprocess=post, run_model=.01, forward=.01 + prepare + post)))
    write("source", dict(rows=rows["source"]))
    prepare_models = {}
    for family in FAMILIES:
        names = [p["id"] for p in points if p["id"] in anchors and
                 work_features(p["q"], p["history"], p["blocks"], p["produces_output"])[0] == family]
        selected = [next(p for p in points if p["id"] == name) for name in names]
        vectors = [work_features(p["q"], p["history"], p["blocks"], p["produces_output"])[1] for p in selected]
        coefficients = np.linalg.lstsq(np.asarray(vectors, dtype=float), np.asarray([anchors[name]["prepare"] for name in names]), rcond=None)[0]
        prepare_models[family] = dict(prepare_coefficients=coefficients.tolist(), source_anchor_ids=names)
    rule = dict(schema="compass.native_ap_family_rule/1", source_only=True, source_input=pins["source"],
        prepare=prepare_models, postprocess={"1": .0001, "2": .0001}, median_gate_seconds=.000110,
        uncertainty_band_claim=False)
    write("rule", rule)
    write("freeze", dict(source_only=True, heldout_started=False, rule=pins["rule"], source_input=pins["source"], design=pins["design"]))
    # Independent fixture observations follow the frozen rule; optional failure is kept explicit.
    for row in rows["heldout"]:
        p = next(p for p in points if p["id"] == row["point_id"])
        family, vector = work_features(p["q"], p["history"], p["blocks"], p["produces_output"])
        value = sum(x * y for x, y in zip(vector, rule["prepare"][family]["prepare_coefficients"]))
        row["seconds"]["prepare"] = value + (.001 if family == failed else 0)
        row["seconds"]["forward"] = .01 + row["seconds"]["prepare"] + row["seconds"]["postprocess"]
    write("heldout", dict(rows=rows["heldout"]))
    checks = []
    for p in points:
        if p["role"] != "heldout":
            continue
        family, vector = work_features(p["q"], p["history"], p["blocks"], p["produces_output"])
        prediction = {"prepare": sum(x * y for x, y in zip(vector, rule["prepare"][family]["prepare_coefficients"])),
                      "postprocess": .0001 if p["produces_output"] else 0.}
        for component in prediction:
            values = [r["seconds"][component] for r in rows["heldout"] if r["point_id"] == p["id"]]
            error = abs(prediction[component] - median(values))
            checks.append(dict(point_id=p["id"], component=component, family=family,
                prediction=prediction[component], observed_median=median(values), absolute_error=error,
                passed=error <= .000110, raw=values))
    write("verdict", dict(checks=checks, source_refitted=False, passed=failed is None))
    transfer_checks = []
    for p in points:
        if p["role"] != "transfer":
            continue
        for component in ("prepare", "postprocess"):
            values = [r["seconds"][component] for r in rows["transfer"] if r["point_id"] == p["id"]]
            transfer_checks.append(dict(point_id=p["id"], component=component, passed=True,
                prediction=values[0], observed_median=median(values), raw=values))
    write("placement_transfer", dict(passed=True, checks=transfer_checks, rows=rows["transfer"], source_refitted=False))
    write("native_complete", dict(success=True, engine_closed=True, plan_sha256=pins["acquisition_plan"]["sha256"]))
    for role in ("initial_runtime", "final_runtime"):
        write(role, dict(inherited_run_model=True, attention_scope={"fixture": True}))
    write("copy_closeout", dict(passed=True))
    statuses = {family: family != failed for family in FAMILIES}
    write("qualification", dict(schema="compass.native_ap_family_qualification/1",
        evidence_sha256={role: pin["sha256"] for role, pin in pins.items()}, families=statuses))
    assert set(pins) == set(EVIDENCE)
    handoff = dict(schema="compass.native_ap_family_sources/1", evidence=pins, scope=base.scope,
        deployment_scope={"sha256": deployment_scope_sha}, retained_native_handoff_sha256=base.source_handoff_sha256,
        source_qualified=True, families={family: dict(source_qualified=value) for family, value in statuses.items()})
    target = path / "handoff.json"
    target.write_text(json.dumps(handoff, sort_keys=True))
    return target, hashlib.sha256(target.read_bytes()).hexdigest()


def load_fixture(tmp_path, **kwargs):
    allocation = NativeAllocation(block_size=16, max_model_len=262144, position_rows=3,
                                  cudagraph_mode="full", capture_region_context=True)
    directory = tmp_path / "retained"
    directory.mkdir()
    path, sha = native_bundle(directory)
    base = NativePrefillRegions.load(adapter(allocation).base, str(path), sha, allocation,
                                     deployment_scope_sha256=DEPLOYMENT_SCOPE_SHA)
    base = replace(base, source_qualified=True, allocation=allocation)
    path, sha = family_bundle(tmp_path / "family", base, **kwargs)
    loaded = NativeAPFamilyRegions.load(base, str(path), sha, allocation,
                                       deployment_scope_sha256=DEPLOYMENT_SCOPE_SHA)
    return loaded, allocation


def test_interpolated_native_work_uses_hull_without_request_or_absolute_allocation_keys(tmp_path):
    regions, allocation = load_fixture(tmp_path)
    p = point("unseen-request", 5, 512, 33, True)
    shape = offer(allocation, p)
    first = regions.breakdown(shape)
    assert first["<prepare>"] > 0 and first["<postprocess>"] == .0001
    offer(allocation, p, rename=1000000)
    assert regions.breakdown(shape) == first
    with pytest.raises(ValueError, match="uncertainty"):
        regions.band(shape)


def test_failed_family_does_not_fall_through_to_permissive_base(tmp_path):
    regions, allocation = load_fixture(tmp_path, failed="N1_cached_P0")
    shape = offer(allocation, point("failed", 16, 32, 33, False))
    assert "unqualified or failed" in regions.refusal(shape)
    with pytest.raises(ValueError, match="unqualified or failed"):
        regions.breakdown(shape)


def test_old_native_exact_values_and_refusals_retain_precedence(tmp_path):
    regions, allocation = load_fixture(tmp_path, failed="N1_cached_P1")
    shape = offer(allocation, point("retained", 8, 11248, 704, True))
    assert regions.breakdown(shape) == pytest.approx({"<prepare>": .00064, "<postprocess>": .0001})
    offer(allocation, point("incompatible-retained", 8, 11248, 704, True), changes={"prefill_continuations": (False,)})
    assert "prefix" in regions.refusal(shape)


@pytest.mark.parametrize("change,why", [
    ({"prior_sampled_batch_rows": 4}, "prior_sampled_batch_rows"),
    ({"prior_sampled_has_logprobs": True}, "logprob"),
    ({"output_state_representation": "invented_queue"}, "deferred"),
    ({"output_rows": (False,)}, "output_rows"),
    ({"midstep_saves_empty": False}, "deployment"),
])
def test_unqualified_transfer_paths_refuse(tmp_path, change, why):
    regions, allocation = load_fixture(tmp_path)
    shape = offer(allocation, point("request", 5, 512, 33, True), changes=change)
    assert why in regions.refusal(shape)


def test_no_extrapolation_beyond_source_feature_hull(tmp_path):
    regions, allocation = load_fixture(tmp_path)
    shape = offer(allocation, point("outside", 17, 496, 33, True))
    assert "hull" in regions.refusal(shape)


@pytest.mark.parametrize("role,damage", [
    ("rule", lambda v: v["prepare"]["N1_cached_P0"]["prepare_coefficients"].__setitem__(0, .1)),
    ("domain", lambda v: v["families"]["N1_cached_P0"]["halfspaces"].pop()),
    ("native_complete", lambda v: v.update(engine_closed=False)),
    ("qualification", lambda v: v["evidence_sha256"].update(rule="0" * 64)),
])
def test_repinning_cannot_refit_widen_or_qualify_incomplete_evidence(tmp_path, role, damage):
    with pytest.raises(ValueError):
        load_fixture(tmp_path, mutation=lambda name, value: damage(value) if name == role else None)


def test_source_hull_uses_exact_integer_facets():
    assert source_halfspaces([[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2]]) == [
        [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [1, 1, 1, 2]]


def test_passed_final_q16_survives_failed_new_family(tmp_path):
    from atom.compass.core.cost.cache_regions import (
        CachedPrefillRegions,
        FinalPrefillRegion,
    )

    regions, allocation = load_fixture(tmp_path, failed="N1_cached_P1")
    inherited = CachedPrefillRegions(regions.base.base,
        FinalPrefillRegion(16, (33792, 66560), .0002, 1e-9, .0003, "passed fixture"),
        None, "fixture", "qualified fixture only")
    base = replace(regions.base, base=inherited, allocation=allocation)
    regions = replace(regions, base=base, allocation=allocation)
    shape = offer(allocation, point("retained-final", 16, 44832, 2803, True))
    assert regions.breakdown(shape) == inherited.breakdown(shape)


def test_n2_joint_aliases_and_prefix_sharing_are_scope_not_absolute_keys(tmp_path):
    regions, allocation = load_fixture(tmp_path)
    p = point("n2", [7, 16], [42016, 2992], [2627, 188], True)
    shape = offer(allocation, p)
    expected = regions.breakdown(shape)
    offer(allocation, p, rename=1000000)
    assert regions.breakdown(shape) == expected
    record = allocation._record
    record.state_fork_srcs = (record.state_slots[0], record.state_fork_srcs[1])
    assert "alias relation" in regions.refusal(shape)
    offer(allocation, p)
    record = allocation._record
    second = list(record.block_tables[1])
    second[-1] = record.block_tables[0][-1]
    record.block_tables = (record.block_tables[0], tuple(second))
    assert "shared KV" in regions.refusal(shape)


def test_placement_pass_flag_cannot_hide_a_failed_raw_transfer(tmp_path):
    def corrupt(role, value):
        if role == "placement_transfer":
            for row in value["rows"]:
                row["seconds"]["prepare"] += .001
                row["seconds"]["forward"] += .001

    with pytest.raises(ValueError, match="placement transfer"):
        load_fixture(tmp_path, mutation=corrupt)
