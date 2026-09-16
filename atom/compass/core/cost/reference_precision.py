"""Explicit reference precision policy; independent validation remains mandatory."""
import math
from statistics import median


REFERENCE_PRECISION_POLICY = {
    "schema": "compass.reference_precision_diagnostic/1",
    "reference_statistic": "median_of_all_three",
    "reference_spread_limit": 0.05,
    "reference_spread_is_gate": False,
    "reuse_all_reference_observations": True,
    "control_spread_limit": 0.05,
    "control_relative_error_limit": 0.10,
    "native_body_relative_error_limit": 0.10,
}


def reference_precision_policy(record):
    if "reference_precision_policy" not in record:
        return None
    policy = record["reference_precision_policy"]
    if (not isinstance(policy, dict) or policy != REFERENCE_PRECISION_POLICY
            or any(type(policy[key]) is not type(value) for key, value in REFERENCE_PRECISION_POLICY.items())):
        raise ValueError("undeclared reference precision policy")
    return policy


def reference_values_admitted(point, policy):
    """Admit unchanged medians without relabeling their original precision flag."""
    if policy is None:
        return point["source_qualified"] is True
    values = point.get("all_three", [])
    if (len(values) != 3 or any(type(value) not in (int, float) or not math.isfinite(value)
                              or value <= 0 for value in values)
            or point.get("kernel_qualified", True) is not True):
        return False
    center = median(values)
    spread = (max(values) - min(values)) / center
    return (point.get("seconds") == center and point.get("range_over_median") == spread
            and point.get("source_qualified") is (spread <= policy["reference_spread_limit"]))
