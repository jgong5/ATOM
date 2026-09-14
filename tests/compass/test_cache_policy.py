"""Deployment policy is explicit and does not invent legacy capture facts."""

from types import SimpleNamespace

from atom.compass.core import cache_policy


def test_policy_snapshot_uses_the_native_demand_override():
    config = SimpleNamespace(enable_prefix_caching=True,
                             state_checkpoint_interval_tokens=8192,
                             state_checkpoint_demand=True)
    runtime = cache_policy.cache_on_policy()["state_runtime"]
    actual = cache_policy.snapshot(config, runtime, environment={})
    assert not cache_policy.policy_errors(actual, cache_policy.cache_on_policy())
    disabled = cache_policy.snapshot(config, runtime,
                                     environment={"ATOM_STATE_CHECKPOINT_DEMAND": "0"})
    assert disabled["state_checkpoint_demand"] is False
    assert cache_policy.policy_errors(disabled, actual)


def test_missing_legacy_fields_remain_unknown():
    actual = cache_policy.snapshot(SimpleNamespace(enable_prefix_caching=False), None,
                                   environment={})
    assert actual["enable_prefix_caching"] is False
    assert actual["state_checkpoint_interval_tokens"] is None
    assert actual["state_checkpoint_demand"] is None
    assert actual["state_runtime"] is None
    assert cache_policy.policy_errors(actual, cache_policy.cache_on_policy())


def test_policy_checks_types_and_native_transfer_semantics():
    wanted = cache_policy.cache_on_policy()
    altered = cache_policy.cache_on_policy()
    altered["enable_prefix_caching"] = 1
    assert cache_policy.policy_errors(altered, wanted)
    altered = cache_policy.cache_on_policy()
    altered["state_runtime"]["transfer"]["readable_midstep"] = True
    assert cache_policy.policy_errors(altered, wanted)
    assert cache_policy.cache_on_policy() == wanted
