"""Cache policy carried by a deployment, independent of its pool capacity."""

from copy import deepcopy
import json
import os


SCHEMA = "compass.cache_policy/1"
CONFIG_FIELDS = ("enable_prefix_caching", "state_checkpoint_interval_tokens",
                 "state_checkpoint_demand")


def configuration(config, *, environment=None):
    """Record declared config plus the native demand override; never fill gaps.

    A BlockManager already resolved its policy, so its caller passes an empty
    environment. Worker config snapshots use the same override as the native
    BlockManager. The pinned 8192 interval is aligned to the PoC's block size.
    """
    env = os.environ if environment is None else environment
    fields = {key: getattr(config, key, None) for key in CONFIG_FIELDS}
    if "ATOM_STATE_CHECKPOINT_DEMAND" in env:
        fields["state_checkpoint_demand"] = env["ATOM_STATE_CHECKPOINT_DEMAND"] == "1"
    return fields


def snapshot(config, state_runtime, *, environment=None):
    runtime = (state_runtime.to_wire() if hasattr(state_runtime, "to_wire")
               else state_runtime)
    return {"schema": SCHEMA, **configuration(config, environment=environment),
            "state_runtime": deepcopy(runtime)}


def cache_on_policy():
    """The explicit TP1 diagnostic policy; no claim of measured memory reuse."""
    return {
        "schema": SCHEMA,
        "enable_prefix_caching": True,
        "state_checkpoint_interval_tokens": 8192,
        "state_checkpoint_demand": True,
        "state_runtime": {
            "transfer": {"kind": "fork", "fork_tokens": 1,
                         "paged_layout_id": None, "readable_midstep": False},
            "checkpoint_spec": None,
        },
    }


def policy_errors(observed, expected):
    """Require the whole policy, including native state transfer and types."""
    if not isinstance(observed, dict) or not isinstance(expected, dict):
        return ["cache policy is missing or is not an object"]
    if set(observed) != set(expected):
        return ["cache policy fields differ from the declared policy"]
    return [f"cache policy {key} differs: observed {observed[key]!r}, expected {value!r}"
            for key, value in expected.items()
            if json.dumps(observed[key], sort_keys=True) != json.dumps(value, sort_keys=True)]
