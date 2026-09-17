"""Fresh four-control qualification cannot omit errors or reuse the old freeze."""
import copy
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from atom.compass.core.cost import native_ap_width_qualification as Q
from atom.compass.core.cost.base import StepCost
from atom.compass.core.cost.library import Coverage
from atom.compass.core.cost.composition_qualification import code_identity, geometry, host_rule, source_selection
from .test_native_ap_work import fresh_descriptor


@pytest.fixture
def evidence(monkeypatch):
    documents={}
    def pin(name,value):
        documents['/proof/'+name]=value
        return dict(path='/proof/'+name,sha256=name)
    def read(path,*,role):
        return copy.deepcopy(documents[path]),SimpleNamespace(sha256=path.rsplit('/',1)[1],role=role)
    monkeypatch.setattr(Q,'load_json',read)
    monkeypatch.setattr(Q,'offer_observation',lambda allocation,row:row)
    cost=StepCost(1.11,{'<body>':1.,'<prepare>':.1,'<postprocess>':.01},
                  output_ready_seconds=.1,preparation_seconds=.1,model_seconds=1.)
    coverage=Coverage(operators=2,measured=2,seconds=1.,sources={'fixed':2})
    oracle=SimpleNamespace(require_complete=True,last_coverage=coverage,
        native_allocation=SimpleNamespace(clear=lambda:None),estimate=lambda row:cost,
        execution_model=SimpleNamespace(sha256='M'))
    sources=[];heldouts=[];predictions=[]
    for index,(name,step) in enumerate(Q.PRIMARY.items()):
        n=3 if name=='allocation3_extent' else 4
        d=fresh_descriptor([1]*n,[32]*n,[3]*n,True)
        d['seq_starts']=[0]*n
        d['req_ids']=list(range(n))
        for repetition in range(6):
            source=dict(chain_id=name,chain_step=step,repetition=repetition,role='source',
                        descriptor=copy.deepcopy(d),normal_return=True,seconds={'forward':1.11})
            sources.append(source)
            heldouts.append(dict(copy.deepcopy(source),role='heldout'))
        predictions.append(dict(chain_id=name,chain_step=step,geometry=geometry(d),
                                cost=asdict(cost),coverage=asdict(coverage)))
    width=dict(source=pin('source',dict(rows=sources)),candidate=dict(sha256='candidate'),plan=dict(sha256='p'*64))
    regions=SimpleNamespace(width_extension=width)
    options=dict(region_overlay='retained',region_overlay_sha256='retained',regions='retained')
    prediction_pin=pin('predictions',dict(complete=True,refused=0,rows=predictions))
    identity=dict(schema='compass.complete_predictor_identity/1',complete_identity=True,
        target_end_to_end_timings_used=False,body_book=dict(loaded_inputs=[],source_selection=source_selection(options)),
        code=code_identity(),host_rule=host_rule(oracle),validation_predictions=prediction_pin)
    identity_pin=pin('identity',identity)
    source_run=dict(execution_plan_sha256='e'*64,plan_sha256='p'*64,ownership_token_sha256='s'*64,pid=10,started_at=100.)
    heldout_run=dict(execution_plan_sha256='f'*64,plan_sha256='q'*64,ownership_token_sha256='h'*64,pid=11,started_at=200.)
    source_pin=pin('source_run',source_run);heldout_pin=pin('heldout_run',heldout_run)
    complete_pin=pin('complete',dict(success=True,engine_closed=True,plan_sha256='q'*64))
    source_close=pin('source_closeout',dict(cleanup=dict(writers_released=True),terminal=dict(unprofiled_control={
        'RUNNER_START.json':dict(sha256=source_pin['sha256'])})))
    closeout=pin('closeout',dict(exit_code=0,plan_sha256='f'*64,cleanup=dict(verified=True,writers_released=True),
        collection=dict(copy_complete=True),terminal=dict(unprofiled_control={
            'RUNNER_START.json':dict(sha256=heldout_pin['sha256']),
            'native/NATIVE_COMPLETE.json':dict(sha256=complete_pin['sha256'])})))
    data=dict(schema=Q.SCHEMA,passed=True,source_refitted=False,relative_limit=.10,
        primary_controls=Q.PRIMARY,old_heldouts_are_regression_only=True,final_e2e_proof_required=True,
        predictor_identity=identity_pin,predictor_freeze=pin('freeze',dict(frozen_before_heldout_warmups=True,
            source_refitted=False,predictor_identity=identity_pin,source_model=dict(sha256='candidate'))),
        heldout=pin('heldouts',dict(rows=heldouts)),source_run=source_pin,heldout_run=heldout_pin,
        native_complete=complete_pin,copy_closeout=closeout,
        source_copy=pin('source_copy',dict(original_closeout=source_close,collection=dict(
            copy_complete=True,owned_writers_released_before_collection=True))))
    return dict(data=data,documents=documents,oracle=oracle,regions=regions,options=options)


def qualify(evidence):
    return Q.validate(evidence['data'],SimpleNamespace(sha256='receipt'),path='/proof/receipt',sha256='receipt',
        inputs=[],options=evidence['options'],regions=evidence['regions'],oracle=evidence['oracle'])


def test_fresh_controls_keep_old_measurements_outside_independent_proof(evidence):
    result,_=qualify(evidence)
    assert result['independent_forward_steps']==4 and result['independent_forward_observations']==24
    assert result['old_heldouts_are_regression_only'] and result['final_e2e_proof_required']


def test_one_failed_control_cannot_be_omitted(evidence):
    for row in evidence['documents']['/proof/heldouts']['rows'][:6]:row['seconds']['forward']=1.5
    with pytest.raises(ValueError,match='not under 10%'):
        qualify(evidence)


def test_omitted_prediction_and_missing_repeat_refuse(evidence):
    evidence['documents']['/proof/predictions']['rows'].pop()
    with pytest.raises(ValueError,match='omit or duplicate'):
        qualify(evidence)


def test_changed_predictor_and_unclosed_worker_refuse(evidence):
    evidence['documents']['/proof/identity']['code']={}
    with pytest.raises(ValueError,match='complete predictor'):
        qualify(evidence)
    evidence['documents']['/proof/identity']['code']=code_identity()
    evidence['documents']['/proof/closeout']['cleanup']['writers_released']=False
    with pytest.raises(ValueError,match='closed collection'):
        qualify(evidence)
