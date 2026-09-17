"""Reference precision never substitutes for controls or complete body accuracy."""
from copy import deepcopy

import pytest

from atom.compass.core.cost.library import PriceLibrary
from atom.compass.core.cost.reference_precision import REFERENCE_PRECISION_POLICY
from .test_reached_primitive_prices import conditioned_campaign, make_campaign, load, mha
from .test_prepared_plan import add_book


def seal_all(store):
    # Fixture-only pins may point to records appended later in construction.
    for _ in range(12):
        before = {name: pin['sha256'] for name, pin in store.pins.items()}
        store.seal()
        for label in ('gpu3', 'validation'):
            store.data[f'{label}/EXIT.json']['plan_sha256'] = store.pins[f'{label}/EXECUTION.json']['sha256']
        if before == {name: pin['sha256'] for name, pin in store.pins.items()}:
            return
    raise AssertionError('fixture dependency pins did not settle')


def precision_campaign(tmp_path):
    store, domain, parent = conditioned_campaign(tmp_path)
    original = store.data['gpu3/FREEZE.json']
    reference = original['reference_points']['gemm_ref']
    values = [1.9, 2., 2.1]
    for repeat, seconds in enumerate(values, 1):
        store.data[f'gpu3/reference/gemm_ref.r{repeat}.json']['prices'][reference['signature']]['seconds'] = seconds
    reference.update(all_three=values, range_over_median=(2.1 - 1.9) / 2., source_qualified=False)
    original['source_qualified'] = False
    pin = make_campaign(store, domain, 'validation', groups=('gemm',))
    data = store.data
    handoff, plan, frozen, verdict = (data['validation/' + name] for name in
                                    ('HANDOFF.json', 'PLAN.json', 'FREEZE.json', 'VERDICT.json'))
    condition = data['gpu3/PLAN.json']['conditioning_policy']
    for name in ('PLAN.json', 'MANIFEST.json'):
        data['validation/' + name]['conditioning_policy'] = deepcopy(condition)
    for phase in ('reference', 'heldout'):
        for row in data[f'validation/{phase}/PHASE_RESULT.json']['records']:
            row['settings']['GRAPH_BATCH'] = 32
            row['treatment']['conditioning_policy'] = deepcopy(condition)
            data[f"validation/{phase}/{row['cell_id']}.r{row['repeat']}.json"]['provenance']['conditioning'] = dict(
                policy=deepcopy(condition), timer_invocations=1, completed=True)
        data[f'validation/{phase}/PREFLIGHT.json']['runtime_identity'] = data['gpu3/DISPATCH.json']['runtime_identity']
    data['validation/DISPATCH.json']['runtime_identity'] = data['gpu3/DISPATCH.json']['runtime_identity']
    parent_evidence = data['gpu3/HANDOFF.json']['evidence']
    for role in ('reference_plan', 'reference_phase', 'reference_preflight', 'original_failure'):
        handoff['evidence'][role] = parent_evidence[role]
    data['gpu3/ORIGINAL_FAILURE.json'].update(science=parent_evidence['plan'], reference_evidence=original['reference_evidence'])
    plan['reference_source'] = dict(plan=parent_evidence['plan'], phase=parent_evidence['reference_phase'],
                                  failure=parent_evidence['original_failure'])
    frozen.update(original_plan=parent_evidence['plan'], whole_campaign_source_qualified=False,
        reference_evidence=original['reference_evidence'], reference_points=original['reference_points'],
        source_qualified=False, reference_values_admitted=True, reused_reference_failure=parent_evidence['original_failure'])
    frozen['predictions']['gemm_control'].update(source_qualified=False, reference_values_admitted=True)
    handoff['entries'] = data['gpu3/HANDOFF.json']['entries']
    data['validation/EVENT1.json']['payload'] = dict(phase=parent_evidence['reference_phase'], reused=True, original_plan=parent_evidence['plan'])
    data['validation/EVENT2.json']['payload']['reference_phase'] = parent_evidence['reference_phase']
    verdict.update(source_qualified=False, controls_qualified=True, ready_for_body_validation=True,
                   ready_for_source_review=False, reference_precision_passed=False)
    for record in (plan, data['validation/MANIFEST.json'], frozen, verdict, handoff):
        record['reference_precision_policy'] = dict(REFERENCE_PRECISION_POLICY)

    descriptor = dict(q=[3056], history=[8192], context=[11248], blocks=[704], produces_output=False,
                      block_tables=[list(range(2, 706))])
    native = store.add('native_body.json', dict(source_refitted=False, rows=[dict(
        point_id='independent_body', repetition=i, role='transfer', normal_return=True,
        descriptor=deepcopy(descriptor), seconds={'run_model': seconds})
        for i, seconds in enumerate([3., 4.49, 4.5, 4.5, 4.51, 5.])]))
    rules = store.add('body_rules.json', dict(native_source=native, point_id='independent_body',
        component='run_model', observations=6, relative_error_limit=.10,
        descriptor={key: descriptor[key] for key in ('q', 'history', 'blocks', 'produces_output')}))
    plan['independent_native_body_transfer'] = rules
    data['gpu3/PLAN.json']['independent_native_body_transfer'] = rules
    attention = deepcopy(mha(8192))
    attention['context'] = list(dict(attention['context'], block_tables=descriptor['block_tables'][0][:703],
                                    context_lens=[11248], cu_seqlens_q=[0, 3056]).items())
    base = PriceLibrary()
    add_book(base, tmp_path, 'body_remainder', [attention], [.5])
    graph = store.add('body_graph.json', dict(key=dict(batch_signature=[3056], topology=[['tp', 1]]),
        ops=[domain.ops['gemm_ref'], domain.ops['gemm_ref'], attention], provenance=dict(
            binding=dict(rows=[[3056, 11248]], allocation_measured=True),
            execution=dict(body_rows_traced=3056, body_rows_executed=3056, capture_bucket=None, step_kind='prefill'),
            includes=['model forward'], excludes=['compute_logits', 'sampler', 'input preparation'],
            head_placement={'in_this_graph': False})))
    inputs = store.add('baseline_inputs.json', dict(loaded_inputs=[item.as_dict() for item in base.loaded_inputs]))
    composition = store.add('composition.json', dict(schema='compass.q3056_body_composition/1', graph=graph,
        baseline_library_inputs=inputs, body_registration=None, seconds_per_launch=0,
        native_descriptor=descriptor, shape=dict(num_scheduled_tokens=[3056], context_lens=[11248], produces_output=False, topology={'tp': 1}),
        reference_bindings=[dict(reference_cell_id='gemm_ref', signature=reference['signature'])]))
    handoff['native_body_validation'] = store.add('body_validation.json', dict(schema='compass.q3056_native_body_validation/1',
        composition=composition, prediction_freeze=store.pins['validation/FREEZE.json'], native_source=native,
        point_id='independent_body', repetitions=list(range(6))))
    seal_all(store)
    return store, domain, pin, base


def test_new_policy_keeps_failed_precision_visible_and_derives_body_from_actual_base(tmp_path):
    store, domain, pin, base = precision_campaign(tmp_path)
    library = load(pin, base)
    price = library.lookup(domain.ops['gemm_control'])[0]
    assert price['seconds'] == 2. and price['source_qualified'] is True
    assert price['reference_precision_passed'] is False
    assert store.data['gpu3/FREEZE.json']['reference_points']['gemm_ref']['source_qualified'] is False
    body = library.campaigns[0]['native_body_validation']
    assert body['reference_multiplicities'] == {'gemm_ref': 2}
    assert body['qualified_remainder_seconds'] == .5 and body['frozen_body_seconds'] == 4.5
    assert body['native_median_seconds'] == 4.5 and len(body['native_seconds']) == 6
    assert body['native_range_over_median'] > .4  # Median gate does not assert each row passed.


@pytest.mark.parametrize('damage', ['missing_body', 'policy_mismatch', 'precision_relabel', 'refit',
    'body_multiplicity', 'extra_missing_op', 'baseline_inputs', 'native_rows', 'native_error', 'allocation'])
def test_policy_cannot_bypass_precision_provenance_or_independent_body_accuracy(tmp_path, damage):
    store, _, pin, base = precision_campaign(tmp_path)
    data = store.data
    if damage == 'missing_body': del data['validation/HANDOFF.json']['native_body_validation']
    elif damage == 'policy_mismatch': del data['validation/VERDICT.json']['reference_precision_policy']
    elif damage == 'precision_relabel': data['validation/FREEZE.json']['reference_points']['gemm_ref']['source_qualified'] = True
    elif damage == 'refit': data['validation/FREEZE.json']['predictions']['gemm_control']['seconds'] = 2.04
    elif damage == 'body_multiplicity': data['body_graph.json']['ops'].insert(0, data['body_graph.json']['ops'][0])
    elif damage == 'extra_missing_op': data['body_graph.json']['ops'].append(dict(name='unpriced_other', input_shapes=[], dtypes=[], scalars=[]))
    elif damage == 'baseline_inputs': data['baseline_inputs.json']['loaded_inputs'] = []
    elif damage == 'native_rows': data['native_body.json']['rows'].pop()
    elif damage == 'native_error':
        for row in data['native_body.json']['rows']: row['seconds']['run_model'] = 3.
    elif damage == 'allocation': dict(data['body_graph.json']['ops'][-1]['context'])['block_tables'].pop()
    seal_all(store)
    with pytest.raises(ValueError): load(pin, base)


@pytest.mark.parametrize('damage', ['spread', 'error', 'dispatch'])
def test_reference_precision_policy_keeps_independent_control_gates_hard(tmp_path, damage):
    store, _, pin, base = precision_campaign(tmp_path)
    data = store.data
    if damage == 'dispatch':
        data['validation/DISPATCH.json']['cells']['gemm_control']['kernel_profile'] = [['wrong_kernel', 8]]
    else:
        values = [1.8, 2.04, 2.2] if damage == 'spread' else [2.5] * 3
        for repeat, value in enumerate(values, 1):
            raw = data[f'validation/heldout/gemm_control.r{repeat}.json']
            next(iter(raw['prices'].values()))['seconds'] = value
    seal_all(store)
    with pytest.raises(ValueError): load(pin, base)
