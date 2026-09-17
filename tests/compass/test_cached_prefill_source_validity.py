"""Invalid cached-gather offsets cannot influence a source-fitted law."""
import copy
import json

import pytest

from atom.compass.core.cost.families import attention as A
from atom.compass.core.cost.families.adapter import ParametricPriceLibrary
from atom.compass.runtime.microbench import signature_of
from .test_attention_family import SCOPE, _unified


def op(q, histories, starts=None, *, prefill=True, cached=True):
    value=_unified(q,[a+b for a,b in zip(q,histories)],is_prefill=prefill,has_cached=cached)
    if starts is not None:value['context'].append(['seq_starts',starts])
    return value


@pytest.mark.parametrize('starts',[[0,64],[64,0],[-1,0],[0],[0,False],1])
def test_non_native_cached_row_offsets_are_refused(starts):
    refused=A.regime_of(op([4,8],[64,128],starts),scope=SCOPE)
    assert isinstance(refused,A.Refusal)
    assert refused.reason==A.CACHED_ROW_STARTS_REFUSAL


def test_zero_starts_and_unrelated_cold_decode_paths_remain_unchanged():
    assert A.regime_of(op([4,8],[64,128],[0,0]),scope=SCOPE).name=='unified.prefill.cached'
    assert A.regime_of(op([4,8],[0,0],[0,64],cached=False),scope=SCOPE).name=='unified.prefill.cold'
    assert A.regime_of(op([1,1],[64,128],[0,64],prefill=False),scope=SCOPE).name.startswith('unified.decode.')


def test_invalid_observations_do_not_change_fit_treatment_or_launch_composition(tmp_path):
    library=ParametricPriceLibrary();library.request_attention_scope=copy.deepcopy(SCOPE)
    library.launch_charge_seconds=0
    def add(name,operator,seconds,kernel='native_kernel'):
        graph=tmp_path/(name+'.graph.json');price=tmp_path/(name+'.price.json')
        graph.write_text(json.dumps({'ops':[operator]}))
        price.write_text(json.dumps({'provenance':{'attention_scope':SCOPE},
            'prices':{signature_of(operator):{'seconds':seconds,'kernels':{kernel:None},'cache':'over','kv_regions':8}}}))
        library.add(str(price),str(graph))
        return price.read_bytes()
    for q in (2,4,8,16):
        for h in (64,128):
            operator=op([q],[h],[0])
            features=A.features_for(A.REGIMES['unified.prefill.cached'],A.structure_of(operator),SCOPE)
            add(f'q{q}_h{h}',operator,sum((i+1)*1e-8*x for i,x in enumerate(features)))
    query=op([6],[96],[0])
    fit_before=next(iter(library.attention_model().fits.values()))
    coefficients=fit_before.coefficients
    treatment=library._treatment_for(query,SCOPE)
    kernels=library._launch_composition('unified.prefill.cached',fit_before)
    bad=op([4,8],[80,100],[0,80])
    raw=add('bad_same_treatment',bad,50.)
    add('bad_other_treatment',bad,100.,kernel='unrelated_kernel')
    fit_after=next(iter(library.attention_model().fits.values()))
    assert fit_after.coefficients==coefficients and fit_after.points==fit_before.points
    assert library._treatment_for(query,SCOPE)==treatment
    assert library._launch_composition('unified.prefill.cached',fit_after)==kernels
    # Even an undeclared requested regime cannot give invalid sources a vote.
    unknown=op([6],[96],[0],prefill=None)
    assert library._treatment_for(unknown,SCOPE)==treatment
    exclusions=library.attention_coverage()['excluded_from_fits']
    assert sum(row['observations'] for row in exclusions if row['reason']==A.CACHED_ROW_STARTS_REFUSAL)==2
    assert (tmp_path/'bad_same_treatment.price.json').read_bytes()==raw
    assert len(library._attention_obs)==10  # Evidence retained, not erased.


def test_diagnostic_exact_reference_cannot_admit_invalid_cached_metadata():
    from atom.compass.core.cost.diagnostic_references import DiagnosticReferencePrices

    overlay=DiagnosticReferencePrices.__new__(DiagnosticReferencePrices)
    overlay._selected={}
    overlay.excluded_references=[]
    invalid=op([7,16],[42016,2992],[0,42016])
    raw={'seconds':.123,'reference_cell_id':'invalid_reference','source_graph':{'sha256':'f'*64}}
    assert overlay._insert(invalid,raw)==0 and not overlay._selected
    assert overlay.excluded_references==[dict(reference_cell_id='invalid_reference',
        source_graph={'sha256':'f'*64},reason=A.CACHED_ROW_STARTS_REFUSAL)]
    assert raw['seconds']==.123
