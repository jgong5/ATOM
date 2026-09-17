"""The extension prices a bounded domain only after the complete old lookup refuses."""
import copy
from types import SimpleNamespace

import pytest

from atom.compass.core.cost import native_mha_prefill as N
from atom.compass.core.cost.families.attention_scope import Declaration
from atom.compass.core.cost.library import INTERPOLATED_FLAG, _record_launch_count


def operator(q=11, prefix=36352, *, native=True, layer=3):
    history = prefix + q
    return dict(name=N.MHA, group=None, abi="", input_shapes=[[q,6144],[q,1024],[q,1024]],
        dtypes=["bfloat16"]*3, output_shapes=[[q,6144]], output_dtypes=["bfloat16"],
        output_aliases=[None], int_values=[], int_ranges=[], launch=[],
        scalars=[["#1",None],["#4",None],["#5",f"language_model.model.layers.{layer}.self_attn"],["#6",False],["#7",None]],
        layouts=[[2,[[14336,1],13312,q*14336,2]]] if native else [],
        context=[["context_lens",[history]],["cu_seqlens_q",[0,q]], ["cu_seqlens_k",[0,history]],
            ["max_seqlen_q",q],["max_seqlen_k",history],["min_seqlen_q",0],
            ["num_cached_tokens",[prefix]],["total_kv",history],["has_cached",True],
            ["is_prefill",True],["state","prefill_prefix"],["seq_starts",[0]],
            ["positions",list(range(prefix,history))*3],
            ["slot_mapping",list(range(prefix,history))],["block_tables_shape",[1,16384]]])


def wrapper(result=None):
    obj=N.NativeMhaPrefillFallback.__new__(N.NativeMhaPrefillFallback)
    scope={"unified":{"kv_cache_dtype":"bf16"}}
    obj.declaration=Declaration(scopes=copy.deepcopy(scope))
    obj.family=SimpleNamespace(request_attention_scope=Declaration(scopes=copy.deepcopy(scope)),launch_charge_seconds=0)
    obj.handoff_sha256="test-handoff"
    obj.endpoints={}
    for q in N.QUERIES:
        for prefix,seconds in zip(N.PREFIXES,(q*.001,q*.002)):
            key=N._identity(operator(q,prefix,native=False))[3]
            obj.endpoints[key,prefix]=dict(name=f"q{q}_{prefix}",seconds=seconds)
    obj.review=dict(comparison_summary={"full_original_domain":{"min_relative_error":-.10076989092815347}},
        native_comparisons=[{"relative_error":-.10076989092815347},{"relative_error":-.0007504098457836728}],
        address_argument={"limitations":["Native controls use four KV regions; sources use eight"]})
    calls=[]
    def old(*args):
        calls.append(args)
        return result,"original result"
    obj.base=SimpleNamespace(lookup=old,_body_lookup=old)
    return obj,calls


@pytest.mark.parametrize("q",range(1,16))
@pytest.mark.parametrize("prefix",[32752,36352,49136,65520])
@pytest.mark.parametrize("native",[False,True])
def test_exact_query_brackets_cover_only_their_measured_prefix_interval(q,prefix,native):
    op=operator(q,prefix,native=native,layer=63)
    before=copy.deepcopy(op); library,calls=wrapper()
    record,why=library.lookup(op,{"tp":1})
    assert len(calls)==1 and calls[0][0] is op and op==before
    expected,weights=N.mha_prefix_interpolation(q*.001,q*.002,prefix)
    assert record["seconds"]==expected and record[INTERPOLATED_FLAG]
    assert record["kernel_count"] is None and _record_launch_count(record)==0
    assert [source["weight"] for source in record["interpolation"]["sources"]]==weights
    assert record["interpolation"]["target_layer"]==63
    transfer=record["native_mha_prefill_transfer"]
    assert transfer["exact_measured_coverage"] is False
    assert transfer["timing_equivalence_proven"] is False and transfer["source_refitted"] is False
    assert transfer["source_conditioning"]["kv_regions"]==8
    assert transfer["native_control_conditioning"]["kv_regions"]==4
    assert min(transfer["observed_source_residuals"])==-.10076989092815347


@pytest.mark.parametrize("body",[False,True])
@pytest.mark.parametrize("record",[{"seconds":.31,"source":"exact-native"},
    {"seconds":.47,INTERPOLATED_FLAG:True,"source":"old-model"}, {"seconds":0.,"zero_work":True}])
def test_every_previously_priced_result_keeps_priority_and_identity(body,record):
    library,calls=wrapper(result=record)
    library.endpoints=None  # A previously served path must not inspect the extension.
    if body:
        actual,why=library._body_lookup(operator(),{"tp":1},None,{})
    else:
        actual,why=library.lookup(operator(),{"tp":1})
    assert actual is record and why=="original result" and len(calls)==1


@pytest.mark.parametrize("q,prefix",[(0,36352),(16,36352),(32,36352),(11,32736),(11,65536),(11,36353)])
def test_no_query_or_prefix_extrapolation(q,prefix):
    library,calls=wrapper()
    assert library.lookup(operator(q,prefix),{"tp":1})==(None,"original result")
    assert len(calls)==1


@pytest.mark.parametrize("damage",["cold","decode","mixed","history","positions","layer","q_scale",
    "stride","offset","capacity","owner","q_layout","output_alias","output_dtype","input_dtype"])
def test_unproved_path_or_layout_keeps_old_refusal(damage):
    op=operator(); ctx=dict(op["context"])
    if damage=="cold":ctx["has_cached"]=False
    elif damage=="decode":ctx["is_prefill"]=False
    elif damage=="mixed":ctx["context_lens"].append(65)
    elif damage=="history":ctx["num_cached_tokens"]=[36336]
    elif damage=="positions":ctx["positions"][0]+=1
    elif damage=="layer":op["scalars"][2][1]="language_model.model.layers.0.self_attn"
    elif damage=="q_scale":op["scalars"][0][1]=1.
    elif damage=="stride":op["layouts"][0][1][0][0]+=1
    elif damage=="offset":op["layouts"][0][1][1]+=1
    elif damage=="capacity":op["layouts"][0][1][2]+=1
    elif damage=="owner":op["layouts"][0][1][3]=0
    elif damage=="q_layout":op["layouts"].append([0,[[6144,1],1,11*6144+1,0]])
    elif damage=="output_alias":op["output_aliases"]=[0]
    elif damage=="output_dtype":op["output_dtypes"]=["float32"]
    elif damage=="input_dtype":op["dtypes"][2]="float32"
    op["context"]=list(ctx.items())
    library,calls=wrapper()
    assert library.lookup(op,{"tp":1})==(None,"original result") and len(calls)==1


@pytest.mark.parametrize("topology",[None,{"tp":2},{"tp":True},{"tp":1,"pp":2}])
def test_parallel_scope_remains_strict(topology):
    library,_=wrapper()
    assert library.lookup(operator(),topology)==(None,"original result")


def test_scope_changes_and_extra_launch_charges_are_not_bypassed():
    library,_=wrapper(); library.family.request_attention_scope.scopes["unified"]["kv_cache_dtype"]="fp8"
    assert "scope differs" in library.lookup(operator(),{"tp":1})[1]
    library,_=wrapper(); library.family.launch_charge_seconds=.1
    assert "zero added launch" in library.lookup(operator(),{"tp":1})[1]


def test_missing_endpoint_keeps_refusal_and_body_lookup_uses_same_policy():
    library,_=wrapper(); key=N._identity(operator())[3]
    del library.endpoints[key,N.PREFIXES[1]]
    assert library.lookup(operator(),{"tp":1})==(None,"original result")
    library,calls=wrapper(); op=operator()
    record,_=library._body_lookup(op,{"tp":1},None,{})
    assert record[INTERPOLATED_FLAG] and len(calls)==1
