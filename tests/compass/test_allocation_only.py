from types import SimpleNamespace

import pytest

from atom.compass.core.cost import allocation_only as A
from atom.compass.core.cost.exact_operator_references import ExactOperatorReferences
from atom.compass.core.cost.prepared import prepare_static_operator


def policy(deterministic=False, fill=True):
    return dict(torch_git_version=A.TORCH_REVISION, torch_version="pinned native version",
                deterministic_algorithms=deterministic, fill_uninitialized_memory=fill)


def operator():
    rows=8240
    return dict(name="aten::empty_like",input_shapes=[[rows,48,128]],dtypes=["bfloat16"],
        output_shapes=[[rows,48,128]],output_dtypes=["bfloat16"],output_aliases=[None],
        layouts=[[0,[[16480,128,1],10240,rows*16480,0]]],scalars=[["pin_memory",False]])


def model(monkeypatch):
    monkeypatch.setattr(A,"runtime_policy",policy)
    result=A.AllocationOnly.__new__(A.AllocationOnly)
    result.native_policy=policy();result.proof={"path":"native proof","sha256":"a"*64}
    return result


@pytest.mark.parametrize("prepared",[False,True])
def test_uninitialized_allocation_has_zero_device_cost_and_retains_output_footprint(monkeypatch,prepared):
    op=operator()
    if prepared:op=prepare_static_operator(op)
    record,_=model(monkeypatch).finish(op,{"tp":1},(None,"missing native view"))
    assert record["zero_work"]and record["seconds"]==0 and record["kernel_count"]==0
    assert record["allocation_only"]["output_logical_bytes"]==8240*48*128*2
    assert record["allocation_only"]["host_allocation_and_footprint_separate"]
    assert record["allocation_only"]["timings_used_to_infer_zero"]is False


@pytest.mark.parametrize("damage",["copy","dtype","device","alias","capacity","bad_owner","bad_stride"])
def test_other_work_or_malformed_descriptors_keep_the_original_refusal(monkeypatch,damage):
    op=operator()
    if damage=="copy":op["name"]="aten::clone"
    elif damage=="dtype":op["output_dtypes"]=["float32"]
    elif damage=="device":op["scalars"].append(["device","cpu"])
    elif damage=="alias":op["output_aliases"]=[0]
    elif damage=="capacity":op["layouts"][0][1][2]=1
    elif damage=="bad_owner":op["layouts"][0][1][3]=1
    else:op["layouts"][0][1][0][0]=-1
    original=(None,"original refusal")
    assert model(monkeypatch).finish(op,{"tp":1},original)==original


def test_enabling_fill_invalidates_both_quote_and_prepared_cache_key(monkeypatch):
    allocation=model(monkeypatch)
    library=ExactOperatorReferences.__new__(ExactOperatorReferences)
    library.allocation_only=allocation;library.handoff_sha256="frozen"
    library.base=SimpleNamespace(_prepared_config_key=lambda *args:())
    old_key=library._prepared_config_key({"tp":1},None)
    monkeypatch.setattr(A,"runtime_policy",lambda:policy(True,True))
    assert library._prepared_config_key({"tp":1},None)!=old_key
    # A known incompatible fill policy cannot fall through to an older price.
    result,reason=allocation.finish(operator(),{"tp":1},({"seconds":.1},"old measured"))
    assert result is None and "disabled deterministic filling"in reason


def test_existing_prices_remain_selected_under_a_compatible_policy(monkeypatch):
    original=({"seconds":.000001,"source":"existing"},"existing")
    assert model(monkeypatch).finish(operator(),{"tp":1},original)is original


@pytest.mark.parametrize("value",[{},policy(True,True),dict(policy(),torch_git_version="unknown"),
    dict(policy(),deterministic_algorithms=0)])
def test_unknown_or_enabled_fill_is_not_a_zero_work_proof(value):
    assert not A.uninitialized(value)


def test_runtime_snapshot_reads_worker_flags_instead_of_configuration_claims(monkeypatch):
    from atom.compass.core.resolved_runtime import worker_snapshot

    actual=policy(True,True)
    monkeypatch.setattr(A,"runtime_policy",lambda:actual)
    runner=SimpleNamespace(config=SimpleNamespace(allocation_policy=policy()),rank=0)
    snapshot=worker_snapshot(runner)
    assert snapshot['reader']['component']=='ModelRunner'
    assert snapshot['allocation_policy']is actual


@pytest.mark.parametrize('side',['real','modelled'])
@pytest.mark.parametrize('damage',['missing','enabled','unknown_revision'])
def test_opening_admission_checks_actual_worker_policy_even_with_safe_frontend(side,damage):
    from .test_opening_harness import opening,runtime_readings,run
    from atom.compass.core.cache_policy import cache_on_policy

    server=runtime_readings(side)
    server['allocation_policy']=policy()  # A frontend claim cannot replace the worker.
    if damage=='missing':server['worker_runtime'][0].pop('allocation_policy')
    elif damage=='enabled':server['worker_runtime'][0]['allocation_policy']=policy(True,True)
    else:server['worker_runtime'][0]['allocation_policy']=dict(policy(),torch_git_version='unknown')
    bad=opening.check_server_configuration(server,{'target_model':run.plan_module.MODEL,'cache_policy':cache_on_policy()},side)
    assert any('deterministic filling disabled'in reason for reason in bad)
