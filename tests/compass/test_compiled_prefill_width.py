"""Cached-N4 M may fit source medians, but cannot drop rows or consume heldouts."""
import copy

import pytest

from atom.compass.core.cost.compiled_prefill_width import (
    METHOD, SCHEMA, fit_scale_floor, source_groups, validate_width_fit,
)
from atom.compass.core.cost.composition_qualification import geometry
from .test_native_ap_work import fresh_descriptor


@pytest.fixture
def evidence():
    source, observations, quoted = [], [], {}
    for key, q, body, model in [('short', 1, .02, .1), ('medium', 16, .04, .1), ('large', 4096, 5., 5.5)]:
        descriptor = fresh_descriptor([q]*4, [32]*4, [3]*4, q == 1)
        quoted[key] = dict(complete=True, geometry=geometry(descriptor), B=body, body_seconds=body, head_seconds=0.)
        for repetition in range(6):
            observations.append(dict(source_row_index=len(source), geometry_key=key, B=body, M=model))
            source.append(dict(role='source', normal_return=True, descriptor=copy.deepcopy(descriptor),
                               seconds=dict(run_model=model), repetition=repetition))
    source_pin = dict(path='source', sha256='source-digest')
    quotes = dict(source_only=True, heldout_rows_read=False, e2e_timings_read=False, source=source_pin,
                  loaded_inputs=dict(path='inputs', sha256='inputs-digest'),
                  observations=observations, quotes=quoted, refused=[])
    groups = source_groups(source, quotes)
    parameters = fit_scale_floor([(g['B'], g['median_M']) for g in groups])
    fit = dict(schema=SCHEMA, method=METHOD, source_only=True, frozen=True, source_qualified=False,
               heldout_rows_read=False, e2e_timings_read=False, original_parameters_unchanged=True,
               frozen_before_fresh_heldouts=True, full_forward_qualification_required=True,
               source=source_pin, body_quotes=dict(path='quotes', sha256='quotes-digest'),
               source_groups=groups, parameters=parameters)
    documents = dict(source=dict(rows=source), quotes=quotes, fit=fit, inputs=[])
    return documents, dict(execution_fit=dict(path='fit', sha256='fit-digest'), parameters=parameters), source_pin


def validate(evidence):
    documents, width, source_pin = evidence
    return validate_width_fit(width, dict(source=source_pin), dict(source_input=source_pin),
                              lambda pin, role: documents[pin['path']], [])


def test_source_fit_recovers_floor_and_large_body_scale(evidence):
    assert validate(evidence) == pytest.approx(dict(alpha=1.1, floor_seconds=.1))


def test_any_eligible_source_observation_cannot_be_dropped(evidence):
    evidence[0]['quotes']['observations'].pop()
    with pytest.raises(ValueError, match='omits or duplicates'):
        validate(evidence)


def test_measured_M_and_frozen_fit_cannot_be_changed(evidence):
    evidence[0]['quotes']['observations'][0]['M'] *= 2
    with pytest.raises(ValueError, match='native source M'):
        validate(evidence)


def test_matching_artifact_parameters_still_must_reproduce_source_fit(evidence):
    documents, width, _ = evidence
    documents['fit']['parameters'] = width['parameters'] = dict(alpha=1.2, floor_seconds=.11)
    with pytest.raises(ValueError, match='complete frozen source fit'):
        validate(evidence)


def test_heldout_label_is_never_a_source_observation(evidence):
    evidence[0]['source']['rows'][-1]['role'] = 'heldout'
    with pytest.raises(ValueError, match='cached-N4 source observations'):
        validate(evidence)
