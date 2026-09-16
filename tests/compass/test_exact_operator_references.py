"""Exact event references retain real repetitions without claiming dispatch."""
import copy
import hashlib
import json
import os
from pathlib import Path

import pytest

from atom.compass.core.cost import exact_operator_references as E
from atom.compass.core.cost.families.adapter import ParametricPriceLibrary
from atom.compass.core.cost.families.attention_scope import Declaration
from atom.compass.core.cost.library import INTERPOLATED_FLAG, _record_launch_count
from atom.compass.core.cost.prepared import prepare_static_operator
from atom.compass.core.cost.records import OperatorEventRecord


def pin(path):
    return dict(path=str(path), sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())


@pytest.fixture
def actual(tmp_path, monkeypatch):
    source = os.environ.get("ATOMCOMPASS_EXACT_OPERATOR_SOURCE")
    scope = os.environ.get("ATOMCOMPASS_EXACT_OPERATOR_SCOPE")
    if not source or not scope:
        pytest.skip("set ATOMCOMPASS_EXACT_OPERATOR_SOURCE and ATOMCOMPASS_EXACT_OPERATOR_SCOPE for actual references")
    source = json.loads(Path(source).read_text())
    local_source = tmp_path / "source.json"
    local_source.write_text(json.dumps(source))
    scope_data = json.loads(Path(scope).read_text())
    handoff = dict(schema=E.SCHEMA, source_handoff=pin(local_source), deployment_scope=pin(scope),
        diagnostic_only=True, cases=[case["name"] for case in source["cases"]])
    base = ParametricPriceLibrary()
    base.launch_charge_seconds = 0.0
    base.request_attention_scope = Declaration(scopes=scope_data["attention_scope"])
    base.request_attention_treatments = {"gdn.prefill": {"kernels": ("old parametric treatment",)}}
    monkeypatch.setattr(E, "live_body_flags", lambda: {"FLA_GDN_FIX_BT": 0, "USE_DEFAULT_FLA_NORM": 0})
    return tmp_path, source, handoff, base


def load(actual):
    path, source, handoff, base = actual
    source_path = path / "source.json"
    source_path.write_text(json.dumps(source))
    handoff["source_handoff"] = pin(source_path)
    target = path / "handoff.json"
    target.write_text(json.dumps(handoff))
    return E.ExactOperatorReferences(base, str(target), pin(target)["sha256"],
        deployment_scope_sha256=handoff["deployment_scope"]["sha256"], diagnostic_only=True)


def test_actual_three_references_preempt_treatment_without_training_or_dispatch_claims(actual):
    library = load(actual)
    for case in actual[1]["cases"]:
        record, _ = library.lookup(case["operator"], {"tp": 1})
        assert isinstance(record, OperatorEventRecord)
        assert record["seconds"] == case["reference_median_seconds"]
        assert record["all_three"] == case["reference_values_seconds"]
        assert record["relative_spread"] == case["relative_spread"]
        assert record["precision_warning"] == (case["relative_spread"] > .05)
        assert record["source_qualified"] is False and record["whole_forward_validation_required"]
        assert record["kernel_count"] is None and record["launch_count"] is None
        assert record["kernel_dispatch_observed"] is False
        assert not record.get(INTERPOLATED_FLAG)
        assert _record_launch_count(record) == 0
        seconds, coverage, charges = library.body({"ops": [case["operator"]], "key": {"topology": [["tp", 1]]}})
        assert seconds == record["seconds"] and coverage.complete and charges == 0
    assert actual[3]._attention_obs == []


def test_layer_label_and_joint_address_renaming_preserve_exact_work(actual):
    library = load(actual)
    op = copy.deepcopy(actual[1]["cases"][0]["operator"])
    op["scalars"][0][1] = op["scalars"][0][1].replace("layers.0.", "layers.1.")
    for name, value in op["context"]:
        if name in ("non_spec_state_indices_tensor", "non_spec_state_indices_in_tensor"):
            value[0][:] = [index + 5 if index >= 0 else index for index in value[0]]
    record, _ = library.lookup(op, {"tp": 1})
    assert record["seconds"] == actual[1]["cases"][0]["reference_median_seconds"]
    assert record["layer_label_equivalence"]["source"] == 0
    assert record["layer_label_equivalence"]["target"] == 1
    assert not record.get(INTERPOLATED_FLAG) and _record_launch_count(record) == 0


def test_prepared_mrope_uses_the_same_finite_record(actual):
    library = load(actual)
    case = next(case for case in actual[1]["cases"] if case["operator"]["name"] == E.MROPE)
    prepared = prepare_static_operator(case["operator"])
    assert prepared is not None
    record, _ = library._body_lookup(prepared, {"tp": 1}, None, {})
    assert record["seconds"] == case["reference_median_seconds"] and _record_launch_count(record) == 0


@pytest.mark.parametrize("damage", ["stride", "offset", "backing", "alias", "missing_alias", "output_dtype", "topology"])
def test_inexact_work_does_not_reach_finite_reference(actual, damage):
    library = load(actual)
    op = copy.deepcopy(actual[1]["cases"][0]["operator"])
    topology = {"tp": 1}
    if damage == "stride":
        op["layouts"][0][1][0][0] += 1
    elif damage == "offset":
        op["layouts"][1][1][1] += 1
    elif damage == "backing":
        op["layouts"][0][1][2] += 1
    elif damage == "alias":
        context = dict(op["context"])
        context["non_spec_state_indices_tensor"][0][0] = context["non_spec_state_indices_in_tensor"][0][0]
    elif damage == "missing_alias":
        op["context"] = [item for item in op["context"] if item[0] != "non_spec_state_indices_in_tensor"]
    elif damage == "output_dtype":
        op["output_dtypes"] = ["float32"]
    else:
        topology = {"tp": 2}
    record, _ = library.lookup(op, topology)
    assert record is None


def test_extra_launch_charge_and_changed_backend_flags_refuse(actual, monkeypatch):
    library = load(actual)
    op = actual[1]["cases"][0]["operator"]
    actual[3].launch_charge_seconds = .000001
    assert library.lookup(op, {"tp": 1})[0] is None
    actual[3].launch_charge_seconds = 0
    monkeypatch.setattr(E, "live_body_flags", lambda: {"FLA_GDN_FIX_BT": 1, "USE_DEFAULT_FLA_NORM": 0})
    assert library.lookup(op, {"tp": 1})[0] is None


@pytest.mark.parametrize("damage", ["median", "spread", "drop_repeat", "invent_dispatch", "wrong_backend_code"])
def test_repinning_cannot_trim_refit_or_invent_evidence(actual, damage):
    case = actual[1]["cases"][0]
    if damage == "median":
        case["reference_median_seconds"] *= 1.1
    elif damage == "spread":
        case["relative_spread"] = 0
    elif damage == "drop_repeat":
        case["references"].pop()
    elif damage == "invent_dispatch":
        case["kernel_dispatch_observed"] = True
    else:
        plan = json.loads(Path(case["plan"]["path"]).read_text())
        entry = next(p for p in plan["code_pins"] if p["path"].endswith("atom/model_ops/attention_gdn.py"))
        entry["sha256"] = "0" * 64
        target = actual[0] / "changed_plan.json"
        target.write_text(json.dumps(plan))
        case["plan"] = pin(target)
    with pytest.raises(ValueError):
        load(actual)


def test_shared_projection_views_are_not_contiguous_defaults():
    op = {"input_shapes": [[3, 4], [3, 2]], "dtypes": ["bfloat16"] * 2,
          "layouts": [[0, [[8, 1], 0, 24, 0]], [1, [[8, 1], 6, 24, 0]]]}
    views = E.argument_views(op)
    assert views[0]["stride"] == [8, 1]
    assert views[1]["offset"] == 6 and views[1]["elements"] == 24 and views[1]["owner"] == 0


def test_synthetic_bulk_schema_selects_one_signature_and_retains_shared_pool(actual):
    """Schema fixture only; this does not claim a new batched acquisition."""
    from atom.compass.core.cost.library import _signature_of

    directory, source, handoff, _ = actual
    case = source["cases"][0]
    handoff["cases"] = [case["name"]]
    op = case["operator"]
    other = source["cases"][1]["operator"]
    batch = directory / "batch.json"
    batch.write_text(json.dumps({"ops": [op, other]}))
    plan = json.loads(Path(case["plan"]["path"]).read_text())
    plan["graph"] = pin(batch)
    plan["no_end_to_end_timing_inputs"] = True
    plan.pop("fitting_end_to_end_timings")
    for entry in plan["cases"]:
        entry["case_id"] = entry.pop("name")
    plan["input_pins"] = plan.pop("code_pins")
    plan_path = directory / "batch_plan.json"
    plan_path.write_text(json.dumps(plan))
    case["plan"] = pin(plan_path)
    case.pop("native_context")
    signature = _signature_of(op)
    for index, reference in enumerate(case["references"]):
        raw = json.loads(Path(reference["artifact"]["path"]).read_text())
        raw["acquisition"]["plan_sha256"] = case["plan"]["sha256"]
        raw["acquisition"]["case_id"] = raw["acquisition"].pop("case")
        raw["acquisition"].pop("source_only")
        raw["acquisition"].pop("dispatch_seconds_used_as_prices")
        raw["acquisition"]["by_signature"] = {signature: {"case_id": case["name"]}}
        setup = raw["provenance"]["native_layout_context_setup"]
        setup.pop("native_context")
        setup["graph"] = pin(batch)
        raw["provenance"]["native_layout_context_setup"] = {
            "standup": setup["standup"], "by_signature": {signature: setup}}
        raw["provenance"]["collector"]["hashes"]["graphs"] = {batch.name: pin(batch)["sha256"]}
        raw["prices"][_signature_of(other)] = dict(raw["prices"][signature])
        target = directory / f"batch_raw{index}.json"
        target.write_text(json.dumps(raw))
        reference["artifact"] = pin(target)
    library = load(actual)
    assert library.lookup(op, {"tp": 1})[0]["seconds"] == case["reference_median_seconds"]


def test_mha_scope_and_each_shifted_kv_context_are_required():
    from atom.compass.core.cost.families import attention as A, attention_scope
    from atom.compass.core.cost.library import _signature_of
    from atom.compass.runtime.forward_ctx import shift_addresses
    from .test_attention_family import _resolved_record, _unified

    op = _unified([8], [16], is_prefill=True, has_cached=True)
    op["context"] += [["block_tables", [0]], ["slot_mapping", list(range(8, 16))]]
    resolved = _resolved_record()
    expected = A.scoped(op, attention_scope.read_resolved(resolved, where="fixture").for_op(op))
    context = dict(op["context"])
    installs = []
    for region in range(2):
        shifted = dict(context, block_tables=shift_addresses(context["block_tables"], region),
                       slot_mapping=shift_addresses(context["slot_mapping"], region * expected["kv_cache_block_size"]))
        installs.append(dict(region=region, block_stride=1, observed_context_sha256=E._digest(shifted),
                             expected_context_sha256=E._digest(shifted)))
    raw = dict(raw={"resolved_scope": resolved}, prices={_signature_of(op): {"kv_regions": 2}},
        provenance={"native_layout_context_setup": {"context_sha256": E._digest(op["context"]), "context_installs": installs}})
    E._registered_scope(op, raw, expected, {})
    installs[0]["observed_context_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="shifted KV context"):
        E._registered_scope(op, raw, expected, {})
