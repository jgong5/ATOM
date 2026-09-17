"""Finite native operator-event references, with no kernel-dispatch inference.

The witnessed ExactAttentionOverrides schema remains separate. This reader
shares the reached-source work identity and evidence reader, but consumes whole
operator event durations only, before any legacy family/treatment fallback.
"""
from math import isfinite
from pathlib import Path
from statistics import median

from atom.compass.core.cost.cached_q16 import GDN, MHA
from atom.compass.core.cost.families import attention, attention_scope
from atom.compass.core.cost.families.exact_attention import _digest, _layout_without_capacity
from atom.compass.core.cost.library import PriceLibrary, _signature_of
from atom.compass.core.cost.prepared import PreparedOperator
from atom.compass.core.cost.reached_primitive_evidence import Evidence
from atom.compass.core.cost.reached_primitives import INVALID_ALIAS, work_identity
from atom.compass.core.cost.records import OperatorEventRecord

SCHEMA = "compass.exact_operator_reference_sources/1"
ROLE_PREFIX = "oracle.exact_operator_references."
MROPE = "triton::_mrope_qk_tiled_kernel"
CODE = {
    GDN: ("atom/model_ops/base_attention.py", "atom/model_ops/attention_gdn.py",
          "atom/model_ops/attentions/gdn_attn.py", "atom/models/qwen3_5.py"),
    MHA: ("atom/model_ops/paged_attention.py", "atom/model_ops/attention_mha.py", "atom/model_ops/attentions/aiter_attention.py",
          "atom/models/qwen3_5.py"),
    MROPE: ("atom/model_ops/triton_mrope.py",),
}


def argument_views(op):
    """Decode the graph's operand layouts, including shared backing storage."""
    layouts = dict(op.get("layouts") or ())
    result = []
    for index, (shape, dtype) in enumerate(zip(op["input_shapes"], op["dtypes"])):
        stride, size = [], 1
        for value in reversed(shape):
            stride.insert(0, size)
            size *= value
        layout = layouts.get(index, [stride, 0, size, index])
        result.append(dict(shape=shape, dtype=dtype.removeprefix("torch."), stride=layout[0],
                           offset=layout[1], elements=layout[2], owner=layout[3]))
    return result


def live_body_flags():
    from atom.model_ops.fla_ops import chunk_o, l2norm

    return {"FLA_GDN_FIX_BT": int(bool(chunk_o.FLA_GDN_FIX_BT)),
            "USE_DEFAULT_FLA_NORM": int(l2norm.USE_DEFAULT_FLA_NORM)}


def _registered_scope(op, raw, expected, plan):
    setup = raw["provenance"]["native_layout_context_setup"]
    if op["name"] == MROPE:
        if dict(op.get("launch") or ()).get("origin") != "atom.model_ops.triton_mrope:_mrope_qk_tiled_kernel":
            raise ValueError("exact MRoPE source lacks its pinned Triton entrypoint")
        return
    if op["name"] == MHA:
        # MHA needs the full live KV/backend scope, not a backend guessed from
        # the layer's name. Existing resolved-scope machinery owns this ABI.
        resolved = attention_scope.read_resolved(raw.get("resolved_scope") or raw.get("raw", {}).get("resolved_scope") or {},
                                                where="exact MHA source")
        actual = _layout_without_capacity(attention.scoped(op, resolved.for_op(op)))
        if attention._scope_matches(actual, _layout_without_capacity(expected)) is not None:
            raise ValueError("exact MHA source resolved backend/KV scope differs")
        from atom.compass.runtime.forward_ctx import shift_addresses

        context = dict(op["context"])
        if setup.get("context_sha256") != _digest(op["context"]):
            raise ValueError("exact MHA source lacks its installed full context identity")
        installs = setup.get("context_installs") or []
        regions = raw["prices"][_signature_of(op)]["kv_regions"]
        if sorted(item["region"] for item in installs) != list(range(regions)):
            raise ValueError("exact MHA source lacks every measured KV region context")
        stride = max(context["block_tables"]) + 1
        for item in installs:
            shifted = dict(context)
            shifted["block_tables"] = shift_addresses(context["block_tables"], item["region"] * stride)
            shifted["slot_mapping"] = shift_addresses(context["slot_mapping"], item["region"] * stride * expected["kv_cache_block_size"])
            if (item["block_stride"] != stride or item["observed_context_sha256"] != _digest(shifted)
                    or item["expected_context_sha256"] != _digest(shifted)):
                raise ValueError("exact MHA source measured a different shifted KV context")
        return
    state = dict(expected["gdn_state_geometry"])
    standup = setup["standup"]
    if (standup["backend"] != state["backend"] or standup["implementation"] != state["impl"]
            or standup["kv_cache_dtype"] != "bf16" or standup["model_dtype"] != "torch.bfloat16"):
        raise ValueError("exact GDN source registered backend or dtype differs")
    installs = setup["context_installs"]
    if not installs or any(item["context"] != op["context"] for item in installs):
        raise ValueError("exact GDN source did not install its full operator context")
    expected_views = dict(state["state_view"])
    for item in installs:
        if len(item["state_views"]) != 2 or any(view["shape"][0] != standup["state_slots"] for view in item["state_views"]):
            raise ValueError("exact GDN source state views disagree with its actual shared pool")
        for name, actual in zip(("k", "v"), item["state_views"]):
            view = dict(expected_views[name])
            if (actual["shape"][1:] != list(view["shape"])[1:]
                    or actual["stride"] != list(view["stride"])
                    or actual["dtype"] != view["dtype"].removeprefix("torch.")):
                raise ValueError("exact GDN state view differs beyond the addressed capacity")
        capacity = item["state_views"][0]["shape"][0]
        indices = dict(op["context"])
        if any(value >= capacity for key in ("non_spec_state_indices_tensor", "non_spec_state_indices_in_tensor")
               for value in indices[key][0]):
            raise ValueError("exact GDN source addressed state outside its actual pool")
    if any(int(plan["environment"][name]) != value for name, value in live_body_flags().items()):
        raise ValueError("exact GDN source and current registered backend flags differ")


def _case(reader, case, declaration):
    name = case["name"]
    graph = reader.read(case["graph"], name + ".graph")
    if case["operator"] not in graph.get("ops", ()):
        raise ValueError("exact operator source graph differs from its handoff")
    op = case["operator"]
    if op["name"] not in CODE or op.get("group") is not None:
        raise ValueError("exact source operator is outside the supported noncollective ABI")
    key, layer = work_identity(op)
    if key is None or key[1] == INVALID_ALIAS or _digest(key) != case["work_identity_sha256"]:
        raise ValueError("exact source work identity or joint state aliases differ")
    plan = reader.read(case["plan"], name + ".plan")
    signature = _signature_of(op)
    declared = next((item for item in plan["cases"] if item.get("name", item.get("case_id")) == name), None)
    named_in_batch = plan.get("case_ids", {}).get(signature) == name
    if ((not named_in_batch and (declared is None or declared["graph"]["sha256"] != case["graph"]["sha256"]))
            or plan["model"] != "Qwen/Qwen3.8-27B"
            or not (plan.get("fitting_end_to_end_timings") is False or plan.get("no_end_to_end_timing_inputs") is True)
            or case.get("interpolation_authorized") is not False or case.get("kernel_dispatch_observed") is not False
            or case.get("source_qualified") is not False or case.get("whole_forward_validation_required") is not True):
        raise ValueError("exact source changes its plan, dispatch status or qualification")
    if case.get("native_context"):
        native = reader.read(case["native_context"], name + ".native_context")
        if native.get("operator_context") != op.get("context"):
            raise ValueError("exact source native context differs from its operator graph")
    code = {str(Path(pin["path"]).relative_to(plan["source_root"])): pin for pin in plan.get("code_pins", plan.get("input_pins", []))
            if Path(pin["path"]).is_relative_to(plan["source_root"])}
    root = Path(__file__).resolve().parents[4]
    for relative in CODE[op["name"]]:
        pin = code.get(relative)
        if pin is None:
            raise ValueError("exact operator source does not pin registered backend code: " + relative)
        reader.read(pin, name + ".source_code." + relative, json_data=False)
        reader.read(dict(path=str(root / relative), sha256=pin["sha256"]),
                    name + ".current_code." + relative, json_data=False)
    expected = attention.scoped(op, declaration.for_op(op) or {}) if op["name"] in (GDN, MHA) else {}
    # Exact operator events witness the registered module and its live KV
    # views. A per-call decode declaration names the parametric dispatch law;
    # it is retained for lookup, but is not the physical scope measured here.
    registered = (attention.scoped(op, declaration.for_family("unified"))
                  if op["name"] == MHA else expected)
    views = argument_views(op)
    # The pinned graph collector explicitly falls back to its over-time event
    # interval for uncapturable MHA. The handoff must disclose that observed
    # mode; it is not a kernel-only measurement or an extra launch charge.
    timer_mode = case.get("timer_mode", "graph")
    if timer_mode not in (("graph", "over") if op["name"] == MHA else ("graph",)):
        raise ValueError("exact source does not declare a supported operator-event timer mode")
    values, host_values, seeds, treatments = [], [], set(), set()
    for repeat, reference in enumerate(case["references"], 1):
        raw = reader.read(reference["artifact"], name + f".raw{repeat}")
        closed = reader.read(reference["exit"], name + f".exit{repeat}")
        acquisition = {**raw["acquisition"], **raw["acquisition"].get("by_signature", {}).get(signature, {})}
        collector = raw["provenance"]["collector"]
        batched = "by_signature" in raw["provenance"]["native_layout_context_setup"]
        if batched:
            batch_graph = reader.read(plan["graph"], name + ".batch_graph")
            if (op not in batch_graph.get("ops", [])
                    or collector["hashes"]["graphs"].get(Path(plan["graph"]["path"]).name) != plan["graph"]["sha256"]):
                raise ValueError("exact batch reference did not measure its declared batch graph")
        if (closed.get("exit_code") != 0 or (not batched and closed["run"].get("case", closed["run"].get("case_id")) != name)
                or closed["run"]["repetition"] != repeat or closed["run"]["seed"] != reference["seed"]
                or acquisition.get("case", acquisition.get("case_id")) != name or acquisition["repetition"] != repeat
                or acquisition["role"] != "reference" or acquisition["seed"] != reference["seed"]
                or acquisition["plan_sha256"] != case["plan"]["sha256"]
                or acquisition.get("source_only") is False
                or acquisition.get("dispatch_seconds_used_as_prices") is True
                or collector["rank"] != 0 or collector["world_size"] != 1
                or any(collector.get(field) is not False for field in
                       ("served_workload", "weights_loaded", "model_runner_initialized"))):
            raise ValueError("exact reference did not complete its independent source acquisition")
        hashes = collector["hashes"]
        if not hashes["graphs"] or (not batched and hashes["graphs"].get(Path(case["graph"]["path"]).name) != case["graph"]["sha256"]):
            raise ValueError("exact reference measured a different graph")
        for relative, digest in hashes["sources"].items():
            if relative not in code or code[relative]["sha256"] != digest:
                raise ValueError("exact reference collector code differs from its plan")
        setup = raw["provenance"]["native_layout_context_setup"]
        if batched:
            if signature not in setup["by_signature"]:
                raise ValueError("exact batch reference lacks this signature's live operand/context evidence")
            setup = {**setup, **setup["by_signature"][signature]}
            raw = {**raw, "provenance": {**raw["provenance"], "native_layout_context_setup": setup}}
        allowed_graphs = {case["graph"]["sha256"]}
        if batched:
            allowed_graphs.add(plan["graph"]["sha256"])
        if (setup.get("before_timer_only") is not True or setup["graph"]["sha256"] not in allowed_graphs
                or (setup.get("native_context") and case.get("native_context")
                    and setup["native_context"]["sha256"] != case["native_context"]["sha256"])
                or not setup.get("operand_sets") or any(actual != views for actual in setup["operand_sets"])):
            raise ValueError("exact reference did not measure the complete shared operand layouts")
        _registered_scope(op, raw, registered, plan)
        if signature not in raw["prices"] or (not batched and set(raw["prices"]) != {signature}) or raw.get("unpriced"):
            raise ValueError("exact reference changed its one complete operator inventory")
        record = raw["prices"][signature]
        seconds = record["seconds"]
        host_seconds = record.get("host_seconds")
        if (type(seconds) not in (int, float) or not isfinite(seconds) or seconds <= 0
                or record.get("kernels") != {} or record.get("composition_witness")
                or seconds != reference["seconds"] or record.get("cache") != timer_mode
                or (timer_mode == "over" and raw["provenance"].get("cache") != "graph")
                or type(host_seconds) not in (int, float) or not isfinite(host_seconds) or host_seconds < 0
                or record.get("arg_sets") != len(setup["operand_sets"])
                or record.get("arg_sets") != reference["arg_sets"]
                or record.get("kv_regions") != reference["kv_regions"]
                or raw["provenance"].get("iters") != plan["timed_iterations"]
                or case["timed_iterations"] != plan["timed_iterations"]):
            raise ValueError("exact operator event reference changed its duration, layout or unobserved dispatch")
        values.append(seconds)
        host_values.append(host_seconds)
        seeds.add(acquisition["seed"])
        treatments.add((record["cache"], record["arg_sets"], record["kv_regions"], raw["provenance"]["iters"]))
    if len(values) != 3 or len(seeds) != 3 or len(treatments) != 1:
        raise ValueError("exact references require all three distinct fixed-seed repetitions")
    center = median(values)
    spread = (max(values) - min(values)) / center
    if (case["reference_values_seconds"] != values or case["reference_median_seconds"] != center
            or case["relative_spread"] != spread):
        raise ValueError("exact reference median/spread trims or changes its raw observations")
    result = OperatorEventRecord(seconds=center, kernels={}, source=case["references"][values.index(center)]["artifact"]["path"],
        source_references=case["references"], all_three=values, relative_spread=spread,
        source_host_seconds=host_values,
        registered_attention_scope=registered,
        timer_scope=("device event interval including host starvation, context installation and dispatch"
                     if timer_mode == "over" else "captured graph event interval"),
        precision_warning=spread > .05, source_qualified=False, whole_forward_validation_required=True,
        kernel_dispatch_observed=False, kernel_count=None, launch_count=None,
        dispatch_status=case["dispatch_status"], source_layer=layer, work_identity_sha256=_digest(key),
        source_conditioning=dict(cache=timer_mode, requested_cache="graph", arg_sets=record["arg_sets"], kv_regions=record["kv_regions"],
            state_slots=setup.get("standup", {}).get("state_slots")),
        registered_body_flags={key: int(plan["environment"][key]) for key in
            ("FLA_GDN_FIX_BT", "USE_DEFAULT_FLA_NORM")} if op["name"] == GDN else {})
    return key, result, expected


class ExactOperatorReferences(PriceLibrary):
    """Thin finite overlay; absent work reaches the unchanged base library."""
    requires_zero_launch_charge = True

    def __init__(self, base, handoff_path, handoff_sha256, *, deployment_scope_sha256, diagnostic_only=False):
        super().__init__()
        if diagnostic_only is not True or getattr(base, "launch_charge_seconds", 0) != 0:
            raise ValueError("exact operator references require diagnostic mode and zero added launch charge")
        self.base = base
        reader = Evidence(Path(handoff_path).parent, "exact")
        reader.prefix = ROLE_PREFIX
        handoff = reader.read(dict(path=str(handoff_path), sha256=handoff_sha256), "handoff")
        if (handoff.get("schema") != SCHEMA or handoff.get("diagnostic_only") is not True
                or handoff["deployment_scope"]["sha256"] != deployment_scope_sha256):
            raise ValueError("exact operator handoff changes deployment scope or diagnostic status")
        scope = reader.read(handoff["deployment_scope"], "deployment_scope")
        declaration = attention_scope.Declaration(scopes=scope["attention_scope"])
        source = reader.read(handoff["source_handoff"], "source_handoff")
        if (source.get("schema") != "compass.exact_native_primitive_reference_handoff/1"
                or source.get("exact_only") is not True or source.get("interpolation_authorized") is not False
                or source.get("source_qualified") is not False or source.get("target_end_to_end_timings_used") is not False
                or source.get("kernel_dispatch_observed") is not False):
            raise ValueError("exact operator source does not preserve its unqualified, unprofiled scope")
        self.acquisition_evidence = {}
        for name in ("preserved_dispatch_failure", "foreign_work_timeline", "native_layout_proof"):
            if name in source:
                self.acquisition_evidence[name] = reader.read(source[name], name)
        for name in ("original_supervisor_exits", "release_and_collection"):
            self.acquisition_evidence[name] = [reader.read(pin, name + str(i)) for i, pin in enumerate(source.get(name, []))]
        for i, pin in enumerate(handoff.get("extra_evidence", [])):
            reader.read(pin, "extra_evidence" + str(i))
        self.allocation_only = None
        if handoff.get("allocation_only"):
            from atom.compass.core.cost.allocation_only import AllocationOnly

            self.allocation_only = AllocationOnly(reader, handoff["allocation_only"])
        cases = {case["name"]: case for case in source["cases"]}
        selected = handoff["cases"]
        if not selected or len(set(selected)) != len(selected) or any(name not in cases for name in selected):
            raise ValueError("exact operator selected cases are empty, duplicated or unknown")
        self._selected = {}
        for name in selected:
            key, record, expected = _case(reader, cases[name], declaration)
            if key in self._selected:
                raise ValueError("exact operator handoff repeats a work identity")
            self._selected[key] = (record, expected)
        self._work = {key[0][0] for key in self._selected}
        self.source_qualified = False
        self.handoff_sha256 = handoff_sha256
        self.loaded_inputs = base.loaded_inputs + tuple(reader.inputs)
        self.sources = base.sources + [str(handoff_path)]
        self._prices, self.address_shifted = base._prices, base.address_shifted

    def _source_lookup(self, op, topology):
        if op.get("name") not in CODE:
            return None
        if isinstance(op, PreparedOperator):
            op = op.as_dict()
        key, layer = work_identity(op)
        if op.get("name") in (GDN, MHA) and (key is None or key[1] == INVALID_ALIAS):
            return None, "exact operator source requires valid registered layer and joint state-alias evidence"
        if key not in self._selected:
            if key is not None and key[0][0] in self._work:
                return None, "exact operator reference differs in complete layout/context or joint state aliases"
            return None
        if op.get("group") is not None or any(int(value) != 1 for value in (topology or {}).values()):
            return None, "exact operator reference requires its TP1 noncollective work"
        record, expected = self._selected[key]
        if op["name"] == GDN and record["registered_body_flags"] != live_body_flags():
            return None, "exact GDN reference current registered backend flags differ"
        provider = self.base
        scope_checked = op["name"] not in (GDN, MHA)
        while provider is not None:
            if getattr(provider, "launch_charge_seconds", 0) != 0:
                return None, "unobserved kernel dispatch cannot receive an added launch charge"
            if hasattr(provider, "_declared_scope") and op["name"] in (GDN, MHA):
                actual = attention.scoped(op, provider._declared_scope(op) or {})
                if attention._scope_matches(_layout_without_capacity(expected), _layout_without_capacity(actual)) is not None:
                    return None, "exact operator reference current registered attention scope differs"
                if op["name"] == MHA:
                    declared = getattr(provider, "request_attention_scope", None)
                    if not isinstance(declared, attention_scope.Declaration):
                        try:
                            declared = attention_scope.declaration_of(declared, where="exact MHA current scope")
                        except ValueError:
                            return None, "exact MHA reference lacks the current physical backend/KV scope"
                    physical = attention.scoped(op, declared.for_family("unified"))
                    if attention._scope_matches(_layout_without_capacity(record["registered_attention_scope"]),
                                                _layout_without_capacity(physical)) is not None:
                        return None, "exact MHA reference current physical backend/KV scope differs"
                scope_checked = True
                break
            provider = getattr(provider, "base", None)
        if not scope_checked:
            return None, "exact operator reference lacks the current registered attention scope"
        if layer is not None and layer != record["source_layer"]:
            record = OperatorEventRecord(record, layer_label_equivalence={"source": record["source_layer"], "target": layer,
                "basis": "same registered model/backend code and complete operator work identity; no numeric interpolation"})
        return record, record["source"]

    def lookup(self, op, topology=None, registration=None):
        selected = self._source_lookup(op, topology)
        result = selected if selected is not None else self.base.lookup(op, topology, registration)
        allocation = getattr(self, "allocation_only", None)
        return allocation.finish(op, topology, result) if allocation is not None else result

    def _body_lookup(self, op, topology, registration, modelled_memo):
        selected = self._source_lookup(op, topology)
        result = selected if selected is not None else self.base._body_lookup(op, topology, registration, modelled_memo)
        allocation = getattr(self, "allocation_only", None)
        return allocation.finish(op, topology, result) if allocation is not None else result

    def add(self, *args, **kwargs):
        raise ValueError("build baseline prices before attaching exact operator references")

    def host_sync_reason(self, op):
        return self.base.host_sync_reason(op)

    def _can_reuse_prepared_lookups(self):
        return self.base._can_reuse_prepared_lookups()

    def _prepared_config_key(self, topology, registration):
        key = self.base._prepared_config_key(topology, registration)
        allocation = getattr(self, "allocation_only", None)
        policy = allocation.configuration_key() if allocation is not None else None
        return None if key is None else (key, self.handoff_sha256, policy)

    def describe(self):
        return f"ExactOperatorReferences({len(self._selected)} finite event references; dispatch unobserved; no interpolation; base={self.base.describe()})"
