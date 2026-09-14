"""Opt-in cache-on region artifacts; registered/default source factories stay unchanged."""

import json
import math

from atom.compass.core.cache_policy import cache_on_policy, policy_errors
from atom.compass.core.cost.cache_regions import (
    CachedPrefillRegions, DiagnosticFinalTransfer, DiagnosticOutputlessRegion, FinalPrefillRegion,
)
from atom.compass.core.cost.regions import REGION_MODELS
from atom.compass.core.loaded_input import load_json
from atom.compass.runtime.source_oracle import (
    _flag, _rank_coords, build_source_oracle, region_snapshot,
)


SCHEMA = "compass.cached_prefill_region_overlay/1"


def _number(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("source coefficients must be finite numbers")
    return float(value)


def _axis(values, *, count=None):
    if (not isinstance(values, list) or len(values) < 2
            or (count is not None and len(values) != count)
            or any(type(x) is not int or x <= 0 for x in values)
            or values != sorted(set(values))):
        raise ValueError("source axes must be strictly increasing positive integers")
    return tuple(values)


def model_from_artifact(data, *, include_failed_outputless=False,
                        include_failed_final=False, diagnostic_only=False):
    if data.get("schema") != SCHEMA or data.get("model") != "Qwen/Qwen3.8-27B":
        raise ValueError("unknown cached-prefill source artifact")
    errors = policy_errors(data.get("cache_policy"), cache_on_policy())
    if errors:
        raise ValueError("; ".join(errors))
    base_spec = data["base"]
    base = REGION_MODELS[base_spec["name"]]
    if region_snapshot(base_spec["name"], base)["sha256"] != base_spec["snapshot_sha256"]:
        raise ValueError("cached-prefill artifact's base preset changed")
    final = None
    if data.get("final") is not None:
        spec = data["final"]
        validation = spec["validation"]
        checks = validation.get("checks") or []
        if (validation.get("status") != "PASSED" or not checks
                or not all(check.get("pass") is True for check in checks)):
            raise ValueError("final-prefill source addition needs passing heldout evidence")
        history = _axis(spec["cached_history"], count=2)
        query = spec["query_tokens"]
        if type(query) is not int or query != 16:
            raise ValueError("this source schema validates only the final 16-token query")
        intercept, slope, post = map(_number, (spec["prepare_intercept"], spec["prepare_slope"], spec["postprocess"]))
        if post < 0 or any(intercept + slope * h < 0 for h in history):
            raise ValueError("final-prefill cost is negative inside its source domain")
        final = FinalPrefillRegion(query, history, intercept, slope, post,
                                   json.dumps(validation, sort_keys=True))
    final_transfer = None
    if include_failed_final:
        if not diagnostic_only:
            raise ValueError("failed final-query transfer requires diagnostic_only=1")
        spec = data["failed_final_transfer"]
        validation = spec["validation"]
        checks = validation.get("checks") or []
        controls = [check for check in checks if check.get("role") == "baseline_q16_control"]
        transfer = [check for check in checks if check.get("role") != "baseline_q16_control"]
        if (validation.get("status") != "FAILED"
                or validation.get("baseline_q16_controls_pass") is not True
                or not controls or not all(check.get("pass") is True for check in controls)
                or not any(check.get("pass") is False for check in transfer)):
            raise ValueError("final-query transfer must retain FAILED transfer and passing q16 controls")
        queries = _axis(spec["queries"], count=2)
        history = _axis(spec["cached_history"], count=2)
        if queries != (1, 15) or history != (33792, 66560):
            raise ValueError("failed final-query transfer is bounded to q1..15 and history [33792,66560]")
        if (final is None or spec.get("formula_source") != "final"
                or final.cached_history[0] > history[0] or final.cached_history[1] < history[1]
                or any(key in spec for key in ("prepare_intercept", "prepare_slope", "postprocess"))):
            raise ValueError("failed final-query transfer must reuse the unchanged selected final16 formula")
        final_transfer = DiagnosticFinalTransfer(queries, history, final,
                                                 json.dumps(validation, sort_keys=True))
    diagnostic = None
    if include_failed_outputless:
        if not diagnostic_only:
            raise ValueError("failed outputless source requires diagnostic_only=1")
        spec = data["outputless"]
        validation = spec["validation"]
        if (validation.get("status") != "FAILED" or not validation.get("checks")
                or not any(check.get("pass") is False for check in validation["checks"])):
            raise ValueError("diagnostic source must retain its FAILED qualification")
        histories = _axis(spec["histories"], count=2)
        queries = _axis(spec["queries"])
        values = tuple(tuple(_number(v) for v in row) for row in spec["prepare"])
        if len(values) != 2 or any(len(row) != len(queries) or min(row) < 0 for row in values):
            raise ValueError("outputless source grid must cover the complete source rectangle")
        diagnostic = DiagnosticOutputlessRegion(histories, queries, values,
                                                 json.dumps(validation, sort_keys=True))
    if final is None and diagnostic is None:
        raise ValueError("no cached-prefill source addition was selected")
    return CachedPrefillRegions(base, final, diagnostic, data["name"],
        json.dumps({"cache_policy": data["cache_policy"], "evidence": data["evidence"],
                    "diagnostic_outputless_enabled": include_failed_outputless,
                    "diagnostic_final_transfer_enabled": include_failed_final}, sort_keys=True),
        diagnostic_final=final_transfer)


def source_cost_oracle(*, region_overlay, region_overlay_sha256, regions,
                       include_failed_outputless=False, include_failed_final=False, diagnostic_only=False,
                       q16_handoff=None, q16_handoff_sha256=None,
                       rank_coords=None, **options):
    """Build the existing source composition, then select a separate region object.

    This deliberately distinct factory is not registered as an acceptance
    factory. It mutates neither REGION_MODELS nor the base preset. The harness
    must independently verify the artifact's required native cache policy.
    """
    expected = {"model": "Qwen/Qwen3.8-27B", "tp": 1, "block_size": 16,
                "max_model_len": 262144, "position_rows": 3,
                "cudagraph_mode": "full", "allocation": "native"}
    for key, value in expected.items():
        actual = options.get(key, 1 if key == "tp" else None)
        if isinstance(value, int):
            actual = int(actual) if actual is not None else None
        elif key == "cudagraph_mode":
            actual = str(actual).lower()
        if actual != value:
            raise ValueError(f"cached-prefill source scope requires {key}={value!r}")
    if any(rank != 0 for rank in _rank_coords(rank_coords).values()):
        raise ValueError("cached-prefill source scope requires rank zero")
    if bool(q16_handoff) != bool(q16_handoff_sha256):
        raise ValueError("q16 source handoff and its explicit SHA-256 are required together")
    data, loaded = load_json(region_overlay, role="oracle.region_overlay")
    if loaded.sha256 != region_overlay_sha256:
        raise ValueError("region overlay differs from its explicit SHA-256")
    if regions != data["base"]["name"]:
        raise ValueError("region overlay and requested base preset differ")
    model = model_from_artifact(data,
        include_failed_outputless=_flag(include_failed_outputless, "include_failed_outputless"),
        include_failed_final=_flag(include_failed_final, "include_failed_final"),
        diagnostic_only=_flag(diagnostic_only, "diagnostic_only"))
    result = build_source_oracle(regions=regions, rank_coords=rank_coords, **options).oracle
    result.regions = model
    result.compass_region_snapshot = region_snapshot(data["name"], model)
    result.compass_loaded_inputs += (loaded,)
    if q16_handoff:
        from atom.compass.core.cost.cached_q16 import CachedQ16Prices

        # The operator ABI does not encode the installed backend or KV/state
        # layout. Bind this addition to the explicit deployment scope in its
        # reviewed artifact; this is not a claim that it independently proves
        # every live backend flag. The maintained harness checks live policy.
        expected_scope = (data.get("q16_request_scope") or {}).get("sha256")
        scopes = [entry for entry in result.library.loaded_inputs
                  if entry.role == "oracle.attention_scope"]
        if not expected_scope or len(scopes) != 1 or scopes[0].sha256 != expected_scope:
            raise ValueError("q16 addition requires its pinned deployment request scope")
        base_inputs = len(result.library.loaded_inputs)
        result.library = CachedQ16Prices(result.library, q16_handoff, q16_handoff_sha256)
        result.compass_loaded_inputs += result.library.loaded_inputs[base_inputs:]
    return result
