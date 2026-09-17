"""A validated width supplement extends only its declared work boundaries."""
import copy
from dataclasses import replace

import pytest

from .test_native_ap_work import candidate, fresh_descriptor
from .test_native_ap_exact import offer


def extended(candidate):
    original, allocation = candidate
    bounds = copy.deepcopy(original.model["observed_feature_bounds"])
    bounds.update(rows=[1,4],state_fork_rows=[0,4],allocated_block_entries=[3,30137])
    width = dict(bounds=bounds,max_rows=4,max_prior_rows=4,initial_P4=.00012844049930572509)
    initial = dict(conditions={str(n):dict(postprocess_seconds=.00012) for n in (1,2,3)})
    return replace(original,allocation=allocation,width_extension=width,initial_postprocess=initial)


def test_no_supplement_retains_n4_and_prior4_refusals(candidate):
    original, allocation = candidate
    for n,prior in ((4,1),(1,4)):
        d=fresh_descriptor([15]*n,[32]*n,[3]*n,True)
        shape=offer(allocation,d,change={"prior_sampled_batch_rows":prior})
        assert original.refusal(shape)


def test_supplement_uses_source_P4_and_keeps_original_coefficients(candidate):
    original, allocation = candidate
    before=copy.deepcopy(original.model)
    added=extended(candidate)
    d=fresh_descriptor([1]*4,[32]*4,[3]*4,True)
    shape=offer(allocation,d,change={"prior_sampled_batch_rows":0})
    assert added.breakdown(shape)["<postprocess>"] == added.width_extension["initial_P4"]
    d=fresh_descriptor([15]*2,[32]*2,[3]*2,True)
    shape=offer(allocation,d,change={"prior_sampled_batch_rows":4})
    assert added.breakdown(shape)["<postprocess>"] == pytest.approx(.01 + .02*2/3 + .03*4/3)
    assert added.model is original.model and original.model == before
    assert added.source_qualified is False


@pytest.mark.parametrize("n,prior,blocks",[(5,1,[3]*5),(4,5,[3]*4),(4,1,[12251,4327,7126,6434])])
def test_larger_width_prior_or_table_extent_still_refuses(candidate,n,prior,blocks):
    _,allocation=candidate
    added=extended(candidate)
    d=fresh_descriptor([1]*n,[32]*n,blocks,True)
    shape=offer(allocation,d,change={"prior_sampled_batch_rows":prior})
    assert added.refusal(shape)
