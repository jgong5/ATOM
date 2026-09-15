"""Exact attention supplements are witnessed measurements, not unscoped prices."""
import copy
import hashlib
import json

import pytest

from atom.compass.core.cost.families import attention as A, attention_scope
from atom.compass.core.cost.families.adapter import ParametricPriceLibrary, _acquisition_policy, _measurement_identity
from atom.compass.core.cost.families.exact_attention import TAG_SCHEMA, WITNESS_SCHEMA, EQUIVALENCE_SCHEMA, _digest
from atom.compass.core.cost.identity import cost_key
from atom.compass.core.cost.library import PriceLibrary
from atom.compass.runtime.microbench import signature_of
from .test_attention_family import _resolved_record, _unified


def fixture(tmp_path):
    def write(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value))
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    op = _unified([8], [16], is_prefill=True, has_cached=True)
    op.update(input_shapes=[[8, 6144], [8, 1024], [8, 1024]],
              dtypes=["bfloat16"] * 3, layouts=[])
    signature, geometry = signature_of(op), A.geometry_of(op)
    key = cost_key(signature)
    resolved = _resolved_record()
    scope = attention_scope.read_resolved(resolved, where="fixture").for_op(op)
    scope_ref = write("scope.json", resolved)
    policy = {"observed_cache_required": "over", "kv_variants": 8, "repeats": 3}
    plan = write("plan.json", {"policy": policy})
    graph = write("graph.json", {"ops": [op], "key": {"topology": [["tp", 1]]}})
    counts = {"kernel_a": 2, "kernel_b": 7}
    witness = write("witness.json", {"schema": WITNESS_SCHEMA, "tp": 1, "rank": 0,
        "calls": [{"cost_key": key, "geometry": geometry, "layouts": [],
                   "input_shapes": op["input_shapes"], "kernel_counts": counts, "launch_count": 9}]})
    fields = {name + "_sha256": _digest(value) for name, value in (
        ("geometry", geometry), ("layouts", []), ("input_shapes", op["input_shapes"]),
        ("dtypes", op["dtypes"]))}
    equivalence = write("equivalence.json", {"schema": EQUIVALENCE_SCHEMA, "tp": 1,
        "representative_rank": 0, "all_144_calls_equal": True,
        "graphs": [{"rank": 0, "calls": [dict(fields, cost_key=key)]}],
        "scopes": [{"rank": 0, "resolved_scope_sha256": _digest(scope),
                    "timing_and_witness_scopes_equal": True,
                    "timing_and_witness_conditioning_equal": True}]})
    exports = []
    for repeat, seconds in enumerate((1.0, 3.0, 2.0), 1):
        record = {"name": op["name"], "seconds": seconds, "cache": "over",
                  "kv_regions": 8, "kernels": {}}
        provenance = {"topology": {"tp": 1}, "collector": {"rank": 0, "world_size": 1,
            "served_workload": False, "weights_loaded": False, "model_runner_initialized": False,
            "hashes": {"sources": {"atom/compass/runtime/microbench.py": "a" * 64}}}}
        raw = {"prices": {signature: record}, "provenance": provenance}
        raw_ref = write(f"raw{repeat}.json", raw)
        identity = _measurement_identity(record, _acquisition_policy(raw))
        treatment = {"kernels": identity[0], **dict(identity[1:])}
        composition = dict(witness, schema=WITNESS_SCHEMA, cost_key=key,
                           kernel_counts=counts, launch_count=9,
                           measurement_treatment=treatment, rank_equivalence=equivalence)
        tag = {"schema": TAG_SCHEMA, "regime": "unified.prefill.cached",
               "selection_plan": plan, "selection_policy": policy, "measurement_treatment": treatment}
        export = {"prices": {signature: dict(record, composition_witness=composition)},
                  "resolved_scope": resolved,
                  "provenance": dict(provenance, exact_only=True, raw_inputs=[raw_ref],
                                     resolved_scope_source=scope_ref, attention_exact_override=tag)}
        exports.append(write(f"export{repeat}.json", export)["path"])
    library = ParametricPriceLibrary()
    library.request_attention_scope = attention_scope.Declaration(scopes={"unified.prefill.cached": scope})
    return library, op, graph["path"], exports


def test_exact_override_uses_raw_repeat_median_and_witnessed_launch_count(tmp_path):
    library, op, graph, exports = fixture(tmp_path)
    for path in exports:
        library.add(path, graph)
    record, _ = library.lookup(op, {"tp": 1})
    assert record["attention_exact_override"] is True
    assert record["seconds"] == 2.0 and record["kernels"] == {}
    assert record["launch_count"] == 9
    assert library._attention_obs == []
    seconds, coverage, launches = library.body({"key": {"topology": [["tp", 1]]}, "ops": [op, op]})
    assert seconds == 4.0 and launches == 18 and coverage.measured == 2
    plain = PriceLibrary()
    plain.add(exports[0], graph)
    assert plain.lookup(op, {"tp": 1})[0] is None


def test_exact_override_refuses_wrong_scope_treatment_or_layout(tmp_path):
    library, op, graph, exports = fixture(tmp_path)
    for path in exports:
        library.add(path, graph)
    assert library.lookup(op, {"tp": 1})[0] is not None
    library.request_attention_treatments = {"unified.prefill.cached": {"cache": "graph"}}
    assert library.lookup(op, {"tp": 1})[0] is None
    library.request_attention_treatments = {}
    changed = copy.deepcopy(op)
    changed["layouts"] = [[0, [[9, 1], 2, 16, 0]]]
    assert library.lookup(changed, {"tp": 1})[0] is None
    scopes = copy.deepcopy(library.request_attention_scope.scopes)
    scopes["unified.prefill.cached"]["kv_cache_dtype"] = "fp8"
    library.request_attention_scope = attention_scope.Declaration(scopes=scopes)
    assert library.lookup(op, {"tp": 1})[0] is None


def test_target_capacity_is_not_relabelled_as_source_capacity(tmp_path):
    library, op, graph, exports = fixture(tmp_path)
    for path in exports:
        library.add(path, graph)
    scopes = json.loads(json.dumps(library.request_attention_scope.scopes))
    for _name, fields in scopes["unified.prefill.cached"]["kv_cache_layout"]:
        for name, value in fields:
            if name == "shape":
                value[0] = 65536
    library.request_attention_scope = attention_scope.Declaration(scopes=scopes)
    record, _ = library.lookup(op, {"tp": 1})
    assert record is not None
    source = dict(record["source_conditioning"]["native_scope"]["kv_cache_layout"])
    assert dict(source["k"])["shape"][0] == 131072


@pytest.mark.parametrize("field", ["seconds", "launch_count"])
def test_unwitnessed_duration_or_count_is_not_admitted(tmp_path, field):
    library, op, graph, exports = fixture(tmp_path)
    blob = json.loads(open(exports[0]).read())
    record = next(iter(blob["prices"].values()))
    if field == "seconds":
        record["seconds"] = 99.0
    else:
        record["composition_witness"]["launch_count"] = 8
    path = tmp_path / "altered.json"
    path.write_text(json.dumps(blob))
    with pytest.raises(ValueError):
        library.add(str(path), graph)


def test_missing_request_scope_refuses_instead_of_raising(tmp_path):
    from types import SimpleNamespace

    library, op, graph, exports = fixture(tmp_path)
    library.add(exports[0], graph)
    for declaration in (None, SimpleNamespace(for_op=lambda _op: None)):
        library.request_attention_scope = declaration
        record, reason = library.lookup(op, {"tp": 1})
        assert record is None and "scope" in reason


def test_scope_treatment_cannot_overrule_an_explicit_conflicting_selector(tmp_path):
    library, op, graph, exports = fixture(tmp_path)
    library.add(exports[0], graph)
    scopes = copy.deepcopy(library.request_attention_scope.scopes)
    scopes["unified.prefill.cached"]["measurement_treatment"] = {"cache": "over"}
    library.request_attention_scope = attention_scope.Declaration(scopes=scopes)
    library.request_attention_treatments = {"unified.prefill.cached": {"cache": "graph"}}
    assert library.lookup(op, {"tp": 1})[0] is None
