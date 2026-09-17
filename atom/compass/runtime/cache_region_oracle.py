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
                       native_prefill_handoff=None, native_prefill_handoff_sha256=None,
                       native_ap_handoff=None, native_ap_handoff_sha256=None,
                       reached_primitive_handoffs=None,
                       diagnostic_reference_handoff=None, diagnostic_reference_handoff_sha256=None,
                       exact_operator_handoff=None, exact_operator_handoff_sha256=None,
                       native_mha_decode_layout_handoff=None, native_mha_decode_layout_handoff_sha256=None,
                       native_mha_prefill_handoff=None, native_mha_prefill_handoff_sha256=None,
                       composition_qualification=None, composition_qualification_sha256=None,
                       composition_extension=None, composition_extension_sha256=None,
                       compiled_prefill_execution_handoff=None, compiled_prefill_execution_handoff_sha256=None,
                       include_failed_outputless=False, include_failed_final=False, diagnostic_only=False,
                       q16_handoff=None, q16_handoff_sha256=None,
                       low_q_handoff=None, low_q_handoff_sha256=None,
                       low_q_allow_failed_spread=False,
                       root_prefill_handoff=None, root_prefill_handoff_sha256=None,
                       root_prefill_allow_failed_spread=False,
                       region_supplement_handoff=None, region_supplement_handoff_sha256=None,
                       root_prefill_diagnostic_handoff=None, root_prefill_diagnostic_handoff_sha256=None,
                       root_prefill_diagnostic_workload_sha256=None,
                       rank_coords=None, **options):
    """Build the existing source composition, then select a separate region object.

    This deliberately distinct factory is not registered as an acceptance
    factory. It mutates neither REGION_MODELS nor the base preset. The harness
    must independently verify the artifact's required native cache policy and
    actual effective backend/environment equality. The pinned attention scope
    is a deployment declaration, not a fresh observation of the live worker.
    """
    composition_options = dict(locals())
    composition_options.update(composition_options.pop("options"))
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
    if bool(native_prefill_handoff) != bool(native_prefill_handoff_sha256):
        raise ValueError("native prefill source handoff and its explicit SHA-256 are required together")
    if bool(native_ap_handoff) != bool(native_ap_handoff_sha256):
        raise ValueError("native A/P handoff and its explicit SHA-256 are required together")
    if bool(diagnostic_reference_handoff) != bool(diagnostic_reference_handoff_sha256):
        raise ValueError("diagnostic reference handoff and its explicit SHA-256 are required together")
    if bool(exact_operator_handoff) != bool(exact_operator_handoff_sha256):
        raise ValueError("exact operator handoff and its SHA-256 are required together")
    if bool(native_mha_decode_layout_handoff) != bool(native_mha_decode_layout_handoff_sha256):
        raise ValueError("native MHA layout handoff and its SHA-256 are required together")
    if native_mha_decode_layout_handoff and not (_flag(diagnostic_only, "diagnostic_only") or composition_qualification):
        raise ValueError("native MHA layout transfer requires diagnostic mode or composition qualification")
    if bool(native_mha_prefill_handoff) != bool(native_mha_prefill_handoff_sha256):
        raise ValueError("native MHA prefill handoff and its SHA-256 are required together")
    if native_mha_prefill_handoff and not (_flag(diagnostic_only, "diagnostic_only") or composition_qualification):
        raise ValueError("native MHA prefill fallback requires diagnostic mode or composition qualification")
    if bool(composition_qualification) != bool(composition_qualification_sha256):
        raise ValueError("composition qualification and its SHA-256 are required together")
    if bool(composition_extension) != bool(composition_extension_sha256):
        raise ValueError("composition extension and its SHA-256 are required together")
    if composition_extension and not (composition_qualification and compiled_prefill_execution_handoff):
        raise ValueError("composition extension requires the original qualification and unchanged execution model")
    if bool(compiled_prefill_execution_handoff) != bool(compiled_prefill_execution_handoff_sha256):
        raise ValueError("compiled-prefill execution source and its SHA-256 are required together")
    if compiled_prefill_execution_handoff and not (_flag(diagnostic_only, "diagnostic_only") or composition_qualification):
        raise ValueError("compiled-prefill execution requires diagnostic mode or composition qualification")
    if exact_operator_handoff and (
            not (_flag(diagnostic_only, "diagnostic_only") or composition_qualification)
            or not _flag(options.get("require_complete", True), "require_complete")):
        raise ValueError("exact operator references require diagnostic mode or composition qualification, and complete coverage")
    if diagnostic_reference_handoff and (
            not (_flag(diagnostic_only, "diagnostic_only") or composition_qualification)
            or not _flag(options.get("require_complete", True), "require_complete")):
        raise ValueError("diagnostic references require diagnostic mode or composition qualification, and complete coverage")
    if native_ap_handoff and not native_prefill_handoff:
        raise ValueError("native A/P families require their retained native-prefill source")
    if bool(q16_handoff) != bool(q16_handoff_sha256):
        raise ValueError("q16 source handoff and its explicit SHA-256 are required together")
    if bool(low_q_handoff) != bool(low_q_handoff_sha256):
        raise ValueError("low-query source handoff and its explicit SHA-256 are required together")
    if bool(root_prefill_handoff) != bool(root_prefill_handoff_sha256):
        raise ValueError("root prefill handoff and its explicit SHA-256 are required together")
    if bool(region_supplement_handoff) != bool(region_supplement_handoff_sha256):
        raise ValueError("region supplement handoff and its explicit SHA-256 are required together")
    if region_supplement_handoff and not root_prefill_handoff:
        raise ValueError("region supplement requires the original root prefill handoff")
    if bool(root_prefill_diagnostic_handoff) != bool(root_prefill_diagnostic_handoff_sha256):
        raise ValueError("root diagnostic handoff and its explicit SHA-256 are required together")
    if bool(root_prefill_diagnostic_handoff) != bool(root_prefill_diagnostic_workload_sha256):
        raise ValueError("root diagnostic handoff and its fixed workload are required together")
    if root_prefill_diagnostic_handoff:
        if (not _flag(diagnostic_only, "diagnostic_only") or root_prefill_handoff or region_supplement_handoff
                or not root_prefill_diagnostic_workload_sha256):
            raise ValueError("root diagnostic source requires explicit diagnostic mode/workload and separate source selection")
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
    if low_q_handoff:
        from atom.compass.core.cost.low_query import LowQueryPrices

        scopes = [entry for entry in result.library.loaded_inputs
                  if entry.role == "oracle.attention_scope"]
        if len(scopes) != 1:
            raise ValueError("low-query addition requires one loaded deployment request scope")
        base_inputs = len(result.library.loaded_inputs)
        result.library = LowQueryPrices(result.library, low_q_handoff, low_q_handoff_sha256,
            deployment_scope_sha256=scopes[0].sha256,
            allow_failed_spread=_flag(low_q_allow_failed_spread, "low_q_allow_failed_spread"),
            diagnostic_only=_flag(diagnostic_only, "diagnostic_only"))
        result.compass_loaded_inputs += result.library.loaded_inputs[base_inputs:]
    if root_prefill_handoff:
        from atom.compass.core.cost.root_prefill import ExactPrefillRegions, RootPrefillPrices

        scopes = [entry for entry in result.library.loaded_inputs
                  if entry.role == "oracle.attention_scope"]
        if len(scopes) != 1:
            raise ValueError("root prefill addition requires one loaded deployment request scope")
        base_inputs = len(result.library.loaded_inputs)
        result.library = RootPrefillPrices(result.library, root_prefill_handoff, root_prefill_handoff_sha256,
            deployment_scope_sha256=scopes[0].sha256,
            allow_failed_spread=_flag(root_prefill_allow_failed_spread, "root_prefill_allow_failed_spread"),
            diagnostic_only=_flag(diagnostic_only, "diagnostic_only"))
        result.regions = ExactPrefillRegions(result.regions, result.library.region_points,
                                             result.library.handoff_sha256)
        result.compass_region_snapshot = region_snapshot("root-prefill-sources", result.regions)
        result.compass_loaded_inputs += result.library.loaded_inputs[base_inputs:]
    if region_supplement_handoff:
        from atom.compass.core.cost.region_supplement import PrefillRegionSupplement

        result.regions = PrefillRegionSupplement.load(result.regions, result.library,
            region_supplement_handoff, region_supplement_handoff_sha256,
            deployment_scope_sha256=scopes[0].sha256)
        result.compass_region_snapshot = region_snapshot("root-prefill-supplement", result.regions)
        result.compass_loaded_inputs += result.regions.loaded_inputs
    if root_prefill_diagnostic_handoff:
        from atom.compass.core.cost.root_diagnostic import DiagnosticRootPrefillPrices
        from atom.compass.core.cost.root_prefill import ExactPrefillRegions

        scopes = [entry for entry in result.library.loaded_inputs if entry.role == "oracle.attention_scope"]
        if len(scopes) != 1:
            raise ValueError("root diagnostic addition requires one loaded deployment request scope")
        base_inputs = len(result.library.loaded_inputs)
        result.library = DiagnosticRootPrefillPrices(result.library, root_prefill_diagnostic_handoff,
            root_prefill_diagnostic_handoff_sha256, deployment_scope_sha256=scopes[0].sha256,
            workload_sha256=root_prefill_diagnostic_workload_sha256, diagnostic_only=True)
        result.regions = ExactPrefillRegions(result.regions, result.library.region_points, result.library.handoff_sha256)
        result.regions = ExactPrefillRegions(result.regions, result.library.supplement_points, result.library.handoff_sha256)
        result.compass_region_snapshot = region_snapshot("root-reference-diagnostic", result.regions)
        result.compass_loaded_inputs += result.library.loaded_inputs[base_inputs:]
    if reached_primitive_handoffs:
        from atom.compass.core.cost.reached_primitives import ReachedPrimitivePrices

        scopes = [entry for entry in result.library.loaded_inputs if entry.role == "oracle.attention_scope"]
        if len(scopes) != 1:
            raise ValueError("reached primitive sources require one loaded deployment attention scope")
        previous = len(result.library.loaded_inputs)
        result.library = ReachedPrimitivePrices(result.library, reached_primitive_handoffs,
            deployment_scope_sha256=scopes[0].sha256)
        result.compass_loaded_inputs += result.library.loaded_inputs[previous:]
    if native_prefill_handoff:
        from atom.compass.core.cost.native_prefill_regions import NativePrefillRegions

        scopes = [entry for entry in result.library.loaded_inputs
                  if entry.role == "oracle.attention_scope"]
        if len(scopes) != 1:
            raise ValueError("native prefill sources require one loaded deployment attention scope")
        selected = NativePrefillRegions.load(result.regions, native_prefill_handoff,
            native_prefill_handoff_sha256, result.native_allocation,
            deployment_scope_sha256=scopes[0].sha256)
        if not selected.source_qualified:
            raise ValueError("native prefill source is a review candidate; qualification is required for activation")
        result.regions = selected
        result.compass_region_snapshot = region_snapshot("native-prefill-sources", result.regions)
        result.compass_loaded_inputs += result.regions.loaded_inputs
    if diagnostic_reference_handoff:
        from atom.compass.core.cost.diagnostic_references import DiagnosticReferencePrices

        scopes = [entry for entry in result.library.loaded_inputs if entry.role == "oracle.attention_scope"]
        if len(scopes) != 1:
            raise ValueError("diagnostic references require one loaded deployment attention scope")
        previous = len(result.library.loaded_inputs)
        result.library = DiagnosticReferencePrices(result.library, diagnostic_reference_handoff,
            diagnostic_reference_handoff_sha256, deployment_scope_sha256=scopes[0].sha256,
            diagnostic_only=True)
        result.compass_loaded_inputs += result.library.loaded_inputs[previous:]
    if exact_operator_handoff:
        from atom.compass.core.cost.exact_operator_references import ExactOperatorReferences

        scopes = [entry for entry in result.library.loaded_inputs if entry.role == "oracle.attention_scope"]
        if len(scopes) != 1:
            raise ValueError("exact operator sources require one loaded deployment attention scope")
        if result.seconds_per_launch != 0:
            raise ValueError("unobserved kernel dispatch requires zero added launch charge")
        previous = len(result.library.loaded_inputs)
        result.library = ExactOperatorReferences(result.library, exact_operator_handoff,
            exact_operator_handoff_sha256, deployment_scope_sha256=scopes[0].sha256, diagnostic_only=True)
        result.compass_loaded_inputs += result.library.loaded_inputs[previous:]
    if native_mha_decode_layout_handoff:
        from atom.compass.core.cost.native_mha_layout import NativeMhaDecodeLayout

        scopes = [entry for entry in result.library.loaded_inputs if entry.role == "oracle.attention_scope"]
        if len(scopes) != 1:
            raise ValueError("native MHA layout transfer requires one loaded deployment attention scope")
        previous = len(result.library.loaded_inputs)
        result.library = NativeMhaDecodeLayout(result.library, native_mha_decode_layout_handoff,
            native_mha_decode_layout_handoff_sha256, deployment_scope_sha256=scopes[0].sha256)
        result.compass_loaded_inputs += result.library.loaded_inputs[previous:]
    if native_mha_prefill_handoff:
        from atom.compass.core.cost.native_mha_prefill import NativeMhaPrefillFallback

        scopes = [entry for entry in result.library.loaded_inputs if entry.role == "oracle.attention_scope"]
        if len(scopes) != 1 or result.seconds_per_launch != 0:
            raise ValueError("native MHA prefill fallback requires one deployment scope and zero added launch charge")
        previous = len(result.library.loaded_inputs)
        result.library = NativeMhaPrefillFallback(result.library, native_mha_prefill_handoff,
            native_mha_prefill_handoff_sha256, deployment_scope_sha256=scopes[0].sha256)
        result.compass_loaded_inputs += result.library.loaded_inputs[previous:]
    if native_ap_handoff:
        from atom.compass.core.cost.native_ap_regions import NativeAPFamilyRegions

        selected = NativeAPFamilyRegions.load(result.regions, native_ap_handoff,
            native_ap_handoff_sha256, result.native_allocation,
            deployment_scope_sha256=scopes[0].sha256)
        if not selected.source_qualified and not (
                selected.version == "native-ap-work/1"
                and (_flag(diagnostic_only, "diagnostic_only") or composition_qualification)):
            raise ValueError("native A/P source is a review candidate; family qualification is required for activation")
        result.regions = selected
        result.compass_region_snapshot = region_snapshot("native-ap-families", selected)
        result.compass_loaded_inputs += selected.loaded_inputs
    extension = None
    if composition_extension:
        from atom.compass.core.cost.composition_extension import CompositionExtension

        extension = CompositionExtension(composition_extension, composition_extension_sha256,
                                         options=composition_options)
    if compiled_prefill_execution_handoff:
        from atom.compass.core.cost.compiled_prefill_execution import CompiledPrefillExecution

        if not result.require_complete or result.seconds_per_launch != 0:
            raise ValueError("compiled-prefill execution requires complete raw coverage and zero extra launch charge")
        result.execution_model = CompiledPrefillExecution.load(compiled_prefill_execution_handoff,
            compiled_prefill_execution_handoff_sha256, oracle=result, options=composition_options,
            extension=extension)
        result.compass_loaded_inputs += result.execution_model.loaded_inputs
    if composition_qualification:
        from atom.compass.core.cost.composition_qualification import validate

        if result.seconds_per_launch != 0 or result.regions.version != "native-ap-work/1":
            raise ValueError("composition qualification requires the source-work A/P model and zero extra launch charge")
        result.compass_composition_qualification, loaded = validate(
            composition_qualification, composition_qualification_sha256,
            inputs=result.compass_loaded_inputs, options=composition_options, regions=result.regions,
            oracle=result, extension=extension)
        result.compass_loaded_inputs += loaded
    return result
