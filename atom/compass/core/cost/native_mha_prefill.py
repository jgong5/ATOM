"""Refusal-only cached-prefill interpolation; endpoints never enter the base book."""
from math import isfinite
from pathlib import Path
from statistics import median

from atom.compass.core.cost.exact_operator_references import argument_views
from atom.compass.core.cost.families import attention, attention_scope
from atom.compass.core.cost.families.adapter import ParametricPriceLibrary
from atom.compass.core.cost.families.exact_attention import _layout_without_capacity
from atom.compass.core.cost.library import INTERPOLATED_FLAG, PriceLibrary
from atom.compass.core.cost.low_query import MHA, PREFIXES, QUERIES, _mha_identity, mha_prefix_interpolation
from atom.compass.core.cost.prepared import PreparedOperator
from atom.compass.core.cost.reached_primitive_evidence import Evidence, graph_for, same_pin
from atom.compass.core.cost.records import OperatorEventRecord

SCHEMA = "compass.native_mha_prefill_fallback/1"
ROLE_PREFIX = "oracle.native_mha_prefill."


def _identity(op):
    """The dense or proved native-V class of one small cached-prefill query."""
    if (op.get("name") != MHA or op.get("group") is not None
            or op.get("int_values") or op.get("int_ranges")
            or op.get("output_aliases") != [None]):
        return None
    identity = _mha_identity(op)
    if identity is None:
        return None
    q = identity[0]
    shapes = [[q, 6144], [q, 1024], [q, 1024]]
    scalars = dict(op.get("scalars") or ())
    if ({key: value for key, value in scalars.items() if key != "#5"}
            != {"#1": None, "#4": None, "#6": False, "#7": None}
            or op.get("input_shapes") != shapes or op.get("dtypes") != ["bfloat16"] * 3
            or op.get("output_shapes") != [[q, 6144]] or op.get("output_dtypes") != ["bfloat16"]):
        return None
    wanted = [dict(shape=shape, dtype="bfloat16", stride=[shape[1], 1],
                   offset=0, elements=q * shape[1], owner=index)
              for index, shape in enumerate(shapes)]
    views = argument_views(op)
    if views != wanted:
        wanted[2].update(stride=[14336, 1], offset=13312, elements=q * 14336)
        if views != wanted:
            return None
    return _mha_identity(dict(op, layouts=[]))


def _endpoints(reader, handoff):
    plan = reader.read(handoff["source_plan"], "source_plan")
    frozen = reader.read(handoff["source_freeze"], "source_freeze")
    if (frozen.get("schema") != "compass.low_q_reference_freeze/1"
            or not same_pin(frozen.get("plan"), handoff["source_plan"])
            or any(frozen.get(key) is not False for key in
                   ("heldout_timings_read", "target_timings_used", "candidate_activated"))):
        raise ValueError("cached-prefill fallback requires original reference-only frozen endpoints")
    args = plan["engine_args"]
    if (args.get("model") != "Qwen/Qwen3.8-27B" or args.get("tensor_parallel_size") != 1
            or args.get("kv_cache_dtype") != "bf16"):
        raise ValueError("cached-prefill endpoint deployment differs")
    cases = {case["cell_id"]: case for case in plan["cases"]}
    raw_pins = {Path(pin["path"]).name: pin for pin in frozen["reference_evidence"]["raw_prices"]}
    result = {}
    for q in QUERIES:
        for prefix in PREFIXES:
            name = f"reference_mha_q{q}_l3_C{prefix}"
            case, point = cases[name], frozen["reference_points"][name]
            op = graph_for(reader, case, name + ".graph")
            identity = _identity(op)
            if (identity is None or identity[:3] != (q, prefix, 3) or op.get("layouts")
                    or case["phase"] != "reference" or case["family"] != "mha"
                    or (case["observed_cache"], case["kv_regions"], case["arg_sets"]) != ("over", 8, 64)
                    or point.get("source_qualified") is not True):
                raise ValueError("cached-prefill endpoint changes its query, layout or treatment")
            values, pins = [], []
            for repeat in (1, 2, 3):
                pin = raw_pins[f"{name}.r{repeat}.json"]
                raw = reader.read(pin, name + f".raw{repeat}")
                price = raw.get("prices", {}).get(case["signature"], {})
                provenance = raw.get("provenance", {})
                seconds = price.get("seconds")
                if (raw.get("unpriced") or set(raw.get("prices", {})) != {case["signature"]}
                        or type(seconds) not in (int, float) or not isfinite(seconds) or seconds <= 0
                        or (price.get("cache"), price.get("kv_regions"), price.get("arg_sets")) != ("over", 8, 64)
                        or provenance.get("cache") != "graph" or provenance.get("iters") != case["iters"]
                        or provenance.get("graph") != case["graph"]["path"]
                        or provenance.get("only") != MHA or provenance.get("topology") != {"tp": 1}
                        or provenance.get("observed_group_width") != 1):
                    raise ValueError("cached-prefill endpoint changes its original event observations")
                values.append(seconds)
                pins.append(pin)
            seconds = median(values)
            if (point["all_three"] != values or point["seconds"] != seconds
                    or point["signature"] != case["signature"]
                    or point["range_over_median"] != (max(values) - min(values)) / seconds):
                raise ValueError("cached-prefill endpoint changes its frozen median or spread")
            result[(identity[3], prefix)] = dict(name=name, seconds=seconds, all_three=values,
                graph=case["graph"], raw_prices=pins, reference_precision_passed=True)
    return result


def _review(reader, handoff, endpoints):
    review = reader.read(handoff["review"], "review")
    if (review.get("schema") != "compass.cached_prefill_native_v_domain_review/1"
            or any(review.get(key) is not False for key in
                   ("accepted", "source_qualified", "timing_equivalence_proven", "predictions_refitted",
                    "target_end_to_end_timings_used", "oracle_selection_changed"))
            or not same_pin(review["evidence"]["dense_reference_plan"], handoff["source_plan"])
            or not same_pin(review["evidence"]["dense_reference_freeze"], handoff["source_freeze"])
            or not same_pin(review["evidence"]["request_scope"], handoff["deployment_scope"])):
        raise ValueError("cached-prefill transfer changes its source proof or residuals")
    argument = review["address_argument"]
    if (any(argument.get(key) is not True for key in (
            "arithmetic_and_launch_shape_equal", "intra_wave_coalescing_equal",
            "cache_write_destinations_equal", "downstream_prefill_reads_gathered_KV_cache_not_original_V"))
            or not argument.get("limitations")):
        raise ValueError("cached-prefill transfer lacks its cached-path address proof")
    for name in ("native_attention", "cache_kernel"):
        reader.read(review["evidence"][name], name, json_data=False)
    root = Path(__file__).resolve().parents[4]
    reader.read(dict(path=str(root / "atom/model_ops/attention_mha.py"),
                     sha256=review["evidence"]["native_attention"]["sha256"]), "current_attention", json_data=False)
    native = reader.read(review["evidence"]["native_reference_handoff"], "native_reference_handoff")
    expected = {}
    for case in native["cases"]:
        identity = _identity(case["operator"])
        if identity is None:
            continue
        q, prefix, _, key = identity
        seconds, _ = mha_prefix_interpolation(endpoints[(key, PREFIXES[0])]["seconds"],
            endpoints[(key, PREFIXES[1])]["seconds"], prefix)
        values = case["reference_values_seconds"]
        observed = median(values)
        if (case["reference_median_seconds"] != observed or case["timer_mode"] != "over"
                or any(ref["arg_sets"] != 64 or ref["kv_regions"] != 4 for ref in case["references"])):
            raise ValueError("native-prefill comparison changes its measured treatment")
        expected[case["name"]] = (q, prefix, seconds, observed, values,
                                  (max(values) - min(values)) / observed, seconds / observed - 1)
    comparisons = review["native_comparisons"]
    if len(comparisons) != len(expected) or {row["name"] for row in comparisons} != set(expected):
        raise ValueError("cached-prefill review omits native comparison residuals")
    for row in comparisons:
        actual = tuple(row[key] for key in ("q", "prefix", "predicted_seconds", "measured_seconds",
                                             "native_all_three", "native_relative_spread", "relative_error"))
        if actual != expected[row["name"]] or row["native_kv_regions"] != [4] or row["native_arg_sets"] != [64]:
            raise ValueError("cached-prefill review refits or trims native residuals")
    errors = [row["relative_error"] for row in comparisons]
    summary = dict(count=len(comparisons), q_values=sorted({row["q"] for row in comparisons}),
                   min_relative_error=min(errors), median_relative_error=median(errors),
                   max_relative_error=max(errors))
    if review["comparison_summary"]["full_original_domain"] != summary:
        raise ValueError("cached-prefill review changes its complete residual summary")
    return review


class NativeMhaPrefillFallback(PriceLibrary):
    """Answer only a prior refusal; never insert sources into the base book."""
    requires_zero_launch_charge = True

    def __init__(self, base, handoff_path, handoff_sha256, *, deployment_scope_sha256):
        super().__init__()
        reader = Evidence(Path(handoff_path).parent, "native_mha_prefill")
        reader.prefix = ROLE_PREFIX
        handoff = reader.read(dict(path=str(handoff_path), sha256=handoff_sha256), "handoff")
        if (handoff.get("schema") != SCHEMA
                or any(handoff.get(key) is not True for key in ("old_lookup_first", "refusal_only", "modelled_transfer"))
                or any(handoff.get(key) is not False for key in
                       ("base_source_book_changed", "source_refitted", "timing_equivalence_proven"))
                or handoff["deployment_scope"]["sha256"] != deployment_scope_sha256):
            raise ValueError("cached-prefill fallback changes its refusal-only source policy")
        self.endpoints = _endpoints(reader, handoff)
        self.review = _review(reader, handoff, self.endpoints)
        preflight = reader.read(handoff["source_preflight"], "source_preflight")
        if not same_pin(preflight.get("plan"), handoff["source_plan"]):
            raise ValueError("cached-prefill source preflight belongs to a different plan")
        live = preflight["family_abi"]
        expected_abi = self.review["address_argument"]["live_mha_abi"]
        names = {name for name in live["all_layers"] if name.endswith(".self_attn")}
        if names != {f"language_model.model.layers.{layer}.self_attn" for layer in range(3,64,4)}:
            raise ValueError("cached-prefill source lacks all sixteen native MHA layer ABIs")
        for name, abi in live["all_layers"].items():
            if not name.endswith(".self_attn"):
                continue
            for key in ("class", "parameters", "norms", "rotary_class", "alibi", "sinks"):
                if abi[key] != expected_abi[key]:
                    raise ValueError("cached-prefill source registered layer ABI differs from its proof")
            for field in ("k_cache", "v_cache"):
                actual, expected = abi["pools"][field], expected_abi["pools"][field]
                if any(actual[key] != expected[key] for key in ("dtype", "stride", "element_size")) or actual["shape"][1:] != expected["shape"][1:]:
                    raise ValueError("cached-prefill source KV views differ from its proof")
        declared = reader.read(handoff["deployment_scope"], "deployment_scope")
        self.declaration = attention_scope.Declaration(scopes=declared["attention_scope"])
        physical = self.declaration.for_family("unified")
        backend = dict(physical.get("attention_backend") or ())
        if (backend.get("backend") != "atom.model_ops.attentions.aiter_attention.AiterBackend"
                or backend.get("impl") != "atom.model_ops.attention_mha.PagedAttentionImpl"
                or any(backend.get(key) not in (False, "False", "false", 0, "0") for key in
                       ("ATOM_USE_UNIFIED_ATTN", "ATOM_FORCE_ATTN_TRITON"))
                or physical.get("kv_cache_dtype") != "bf16" or physical.get("kv_cache_block_size") != 16
                or physical.get("sliding_window") != -1):
            raise ValueError("cached-prefill transfer is outside its physical backend scope")
        self.base = base
        provider = base
        while provider is not None and not isinstance(provider, ParametricPriceLibrary):
            provider = getattr(provider, "base", None)
        if provider is None:
            raise ValueError("cached-prefill fallback requires the original scoped source library")
        self.family = provider
        self.handoff_sha256 = handoff_sha256
        self.loaded_inputs = base.loaded_inputs + tuple(reader.inputs)
        self.sources = base.sources + [str(handoff_path)]
        self._prices, self.address_shifted = base._prices, base.address_shifted

    def _fallback(self, op, topology, registration, original):
        if isinstance(op, PreparedOperator):
            op = op.as_dict()
        identity = _identity(op)
        if (identity is None or not topology or topology.get("tp") != 1
                or any(type(value) is not int or value != 1 for value in topology.values())):
            return original
        if getattr(self.family, "launch_charge_seconds", 0) != 0:
            return None, "cached-prefill fallback requires zero added launch charge"
        declared = self.family.request_attention_scope
        if not isinstance(declared, attention_scope.Declaration):
            try:
                declared = attention_scope.declaration_of(declared, where="cached-prefill fallback current scope")
            except ValueError:
                return None, "cached-prefill fallback has no current deployment scope"
        expected = _layout_without_capacity(attention.scoped(op, self.declaration.for_family("unified")))
        actual = _layout_without_capacity(attention.scoped(op, declared.for_family("unified")))
        if attention._scope_matches(expected, actual) is not None:
            return None, "cached-prefill fallback current backend/KV scope differs"
        q, prefix, layer, key = identity
        sources = [self.endpoints.get((key, bound)) for bound in PREFIXES]
        if any(source is None for source in sources):
            return original
        seconds, weights = mha_prefix_interpolation(sources[0]["seconds"], sources[1]["seconds"], prefix)
        source = "interpolated://native-mha-prefill/bounded-prefix"
        return OperatorEventRecord(seconds=seconds, kernels={}, source=source, **{INTERPOLATED_FLAG: True},
            source_qualified=False, whole_forward_validation_required=True,
            kernel_dispatch_observed=False, kernel_count=None, launch_count=None,
            timer_scope="modelled interpolation of whole-operator over-time event intervals",
            interpolation=dict(family=MHA, regime="unified.prefill.cached.low_query", q=q,
                prefix=prefix, prefix_bounds=list(PREFIXES), source_layer=3, target_layer=layer,
                sources=[dict(item, weight=weight) for item, weight in zip(sources, weights)]),
            native_mha_prefill_transfer=dict(handoff_sha256=self.handoff_sha256,
                old_lookup_refusal=original[1], exact_measured_coverage=False,
                actual_layouts=op.get("layouts", []), model_lookup_layouts=[],
                source_refitted=False, timing_equivalence_proven=False,
                source_conditioning=dict(cache="over", arg_sets=64, kv_regions=8),
                native_control_conditioning=dict(cache="over", arg_sets=64, kv_regions=4),
                comparison_summary=self.review["comparison_summary"],
                observed_source_residuals=[row["relative_error"] for row in self.review["native_comparisons"]],
                limitations=self.review["address_argument"]["limitations"])), source

    def lookup(self, op, topology=None, registration=None):
        original = self.base.lookup(op, topology, registration)
        return original if original[0] is not None else self._fallback(op, topology, registration, original)

    def _body_lookup(self, op, topology, registration, modelled_memo):
        original = self.base._body_lookup(op, topology, registration, modelled_memo)
        return original if original[0] is not None else self._fallback(op, topology, registration, original)

    def host_sync_reason(self, op):
        return self.base.host_sync_reason(op)

    def _can_reuse_prepared_lookups(self):
        return self.base._can_reuse_prepared_lookups()

    def _prepared_config_key(self, topology, registration):
        key = self.base._prepared_config_key(topology, registration)
        return None if key is None else (key, self.handoff_sha256)

    def add(self, *args, **kwargs):
        raise ValueError("build baseline prices before attaching cached-prefill fallback")

    def describe(self):
        return f"NativeMhaPrefillFallback(Q1-15 bounded prefix; old lookup first; base={self.base.describe()})"
