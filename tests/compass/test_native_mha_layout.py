"""Only the proved native V address class reaches the bounded model fallback."""
import copy
from types import SimpleNamespace

import pytest

from atom.compass.core.cost import native_mha_layout as N
from atom.compass.core.cost.families.attention_scope import Declaration
from atom.compass.core.cost.library import INTERPOLATED_FLAG


def operator(rows=1):
    return dict(name=N.attention.UNIFIED, group=None, abi="",
        input_shapes=[[rows,6144],[rows,1024],[rows,1024]], dtypes=["bfloat16"]*3,
        output_shapes=[[rows,6144]], output_dtypes=["bfloat16"], output_aliases=[None],
        scalars=[["#1",None],["#4",None],["#5","language_model.model.layers.3.self_attn"],["#6",False],["#7",None]],
        layouts=[[2,[[14336,1],13312,rows*14336,2]]], int_values=[], int_ranges=[],
        context=[["is_prefill",False],["has_cached",False],["state","prefill_native"],
            ["cu_seqlens_q",list(range(rows+1))],["cu_seqlens_k",None],
            ["context_lens",[64]*rows],["max_seqlen_q",1],["max_seqlen_k",64],
            ["block_tables_shape",[rows,4]],["block_tables",list(range(rows*4))],
            ["slot_mapping",[63+64*i for i in range(rows)]]])


def wrapper(*, exact=None, refuse=False):
    scope={"unified":{"kv_cache_dtype":"bf16"},"unified.decode":dict(N.DECODE_SCOPE)}
    calls=[]
    def modelled(op, reason, contract, topology, registration=None, *, _memo=None):
        calls.append(copy.deepcopy(op))
        if refuse:
            return None,"existing bounded model refused context domain"
        return {"seconds":.123, "kernels":{}, INTERPOLATED_FLAG:True,
                "interpolation":{"regime":N.REGIME}},"interpolated://existing-model"
    obj=N.NativeMhaDecodeLayout.__new__(N.NativeMhaDecodeLayout)
    obj.base=SimpleNamespace(lookup=lambda *args:(exact,"original source result"))
    obj.family=SimpleNamespace(request_attention_scope=Declaration(scopes=copy.deepcopy(scope)),_modelled=modelled)
    obj.declaration=Declaration(scopes=scope)
    obj.evidence={"source_refitted":False,"observed_source_residuals":[-.203,-.226]}
    return obj,calls


@pytest.mark.parametrize("rows",[1,2,4])
def test_scoped_model_transfer_preserves_runtime_layout_and_residuals(rows):
    op=operator(rows);before=copy.deepcopy(op);library,calls=wrapper()
    record,reason=library.lookup(op,{"tp":1})
    assert record["seconds"]==.123 and record[INTERPOLATED_FLAG]
    assert len(calls)==1 and calls[0]["layouts"]==[] and op==before
    transfer=record["native_mha_layout_transfer"]
    assert transfer["actual_layouts"]==op["layouts"]
    assert transfer["exact_measured_coverage"] is False
    assert transfer["observed_source_residuals"]==[-.203,-.226]
    assert "modelled" in reason


def test_exact_reference_precedes_model_and_model_domain_refusal_survives():
    exact={"seconds":.456,"source":"exact-native-reference"}
    library,calls=wrapper(exact=exact)
    assert library.lookup(operator(),{"tp":1})[0] is exact and not calls
    library,calls=wrapper(refuse=True)
    assert library.lookup(operator(),{"tp":1})==(None,"existing bounded model refused context domain")
    assert len(calls)==1


@pytest.mark.parametrize("damage",["prefill","row8","stride","offset","capacity","alias","q_layout","mla","q_scale","output_dtype","layer"])
def test_changed_work_does_not_enter_layout_transfer(damage):
    op=operator(8 if damage=="row8" else 1)
    if damage=="prefill":
        op["context"][0][1]=True
    elif damage in ("stride","offset","capacity","alias"):
        layout=op["layouts"][0][1]
        if damage=="stride":layout[0][0]+=1
        elif damage=="offset":layout[1]+=1
        elif damage=="capacity":layout[2]+=1
        else:layout[3]=0
    elif damage=="q_layout":op["layouts"].append([0,[[6144,1],1,6145,0]])
    elif damage=="mla":op["scalars"][3][1]=True
    elif damage=="q_scale":op["scalars"][0][1]=1.0
    elif damage=="output_dtype":op["output_dtypes"]=["float32"]
    elif damage=="layer":op["scalars"][2][1]="language_model.model.layers.0.self_attn"
    library,calls=wrapper()
    assert library.lookup(op,{"tp":1})[0] is None and not calls


@pytest.mark.parametrize("topology",[None,{"tp":2},{"tp":True},{"tp":1,"pp":2}])
def test_transfer_does_not_change_parallel_scope(topology):
    library,calls=wrapper()
    assert library.lookup(operator(),topology)[0] is None and not calls


@pytest.mark.parametrize("label,key,value",[("unified","kv_cache_dtype","fp8"),
    ("unified.decode","decode_dispatch_topology","different-chip")])
def test_current_physical_and_model_scopes_are_rechecked(label,key,value):
    library,calls=wrapper()
    library.family.request_attention_scope.scopes[label][key]=value
    record,why=library.lookup(operator(),{"tp":1})
    assert record is None and "scope differs" in why and not calls


def test_transfer_does_not_bypass_exact_reference_launch_charge_guard():
    library,calls=wrapper()
    library.family.launch_charge_seconds=.001
    record,why=library.lookup(operator(),{"tp":1})
    assert record is None and "zero added launch charge" in why and not calls
