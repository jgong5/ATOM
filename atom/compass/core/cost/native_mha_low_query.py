"""Bounded native cached-MHA laws with independent operator heldouts.

The owning private MHA library checks deployment scope and tries old and exact
prices first. This reader never inserts observations into a generic source book.
"""
from math import isclose, isfinite
from statistics import median

from atom.compass.core.cost.exact_operator_references import _case
from atom.compass.core.cost.families import attention_scope
from atom.compass.core.cost.library import _signature_of
from atom.compass.core.cost.prepared import PreparedOperator
from atom.compass.core.cost.reached_primitive_evidence import Evidence, same_pin

SCHEMA = "compass.native_mha_low_query/1"
VALIDATION_PREFIX = "validation.native_mha_low_query."
GROUPS = {"N1_causal", "N2_causal", "N3_causal", "N4_causal"}
MHA = "aiter::unified_attention_with_output_base"


def coordinates(operator):
    """The measured native projection layout and cached-prefill metadata."""
    op = operator.as_dict() if isinstance(operator, PreparedOperator) else operator
    if (op.get("name") != MHA or op.get("group") is not None
            or op.get("int_values") or op.get("int_ranges")
            or op.get("abi", "") or op.get("launch") or op.get("param_names")
            or op.get("output_aliases") != [None]):
        return None
    context = dict(op.get("context") or ())
    cu = context.get("cu_seqlens_q") or []
    lengths = context.get("context_lens") or []
    if (len(cu) != len(lengths) + 1 or not cu or cu[0] != 0
            or any(type(v) is not int for v in cu + lengths)):
        return None
    queries = [b - a for a, b in zip(cu, cu[1:])]
    history = [c - q for c, q in zip(lengths, queries)]
    n, total = len(queries), sum(queries)
    if (n not in (1, 2, 3, 4) or any(q < 1 or q > 16 for q in queries)
            or any(h < 512 or h > 249952 or h % 16 for h in history)
            or (n > 1 and queries.count(1) > 1)
            or context.get("is_prefill") is not True or context.get("has_cached") is not True
            or context.get("state") != "prefill_prefix"
            or context.get("seq_starts") != [0] * n
            or context.get("num_cached_tokens") != history
            or context.get("total_kv") != sum(lengths)
            or context.get("min_seqlen_q") != 0
            or len(context.get("slot_mapping") or []) != total
            or any(type(v) is not int or v < 0 for v in context["slot_mapping"])
            or context.get("max_seqlen_q") != max(queries)
            or context.get("max_seqlen_k") != max(lengths)):
        return None
    cu_k = [0]
    for length in lengths:
        cu_k.append(cu_k[-1] + length)
    scalars = dict(op.get("scalars") or ())
    layer = scalars.pop("#5", None)
    layers = {f"language_model.model.layers.{i}.self_attn" for i in range(3, 64, 4)}
    if (context.get("cu_seqlens_k") != cu_k or layer not in layers
            or scalars != {"#1": None, "#4": None, "#6": False, "#7": None}
            or op.get("input_shapes") != [[total, 6144], [total, 1024], [total, 1024]]
            or op.get("dtypes") != ["bfloat16"] * 3
            or op.get("output_shapes") != [[total, 6144]]
            or op.get("output_dtypes") != ["bfloat16"]
            or op.get("layouts") != [[2, [[14336, 1], 13312, total * 14336, 2]]]):
        return None
    return dict(q=queries, context=lengths, history=history, n=n,
                sum_context=sum(lengths), max_context=max(lengths), max_query=max(queries))


def sharing_pattern(operator, row):
    """Cached prefix aliases, excluding each row's writable query-tail block."""
    op = operator.as_dict() if isinstance(operator, PreparedOperator) else operator
    context = dict(op.get("context") or ())
    flat = context.get("block_tables") or []
    shape = context.get("block_tables_shape") or []
    n = row["n"]
    if len(shape) != 2 or shape[0] != n or len(flat) % n:
        return None
    width = len(flat) // n
    if any(h // 16 + 1 > min(width, shape[1]) for h in row["history"]):
        return None
    tables = [flat[i * width:i * width + h // 16] for i, h in enumerate(row["history"])]
    tails = [flat[i * width + h // 16] for i, h in enumerate(row["history"])]
    cached = {block for table in tables for block in table}
    if (len(set(tails)) != n or cached.intersection(tails)
            or context.get("slot_mapping") != [block * 16 + offset
                for block, q in zip(tails, row["q"]) for offset in range(q)]):
        return None
    pattern = []
    for i, first in enumerate(tables):
        for second in tables[i + 1:]:
            prefix = 0
            for left, right in zip(first, second):
                if left != right:
                    break
                prefix += 1
            if len(set(first).intersection(second)) != prefix:
                return None
            pattern.append(prefix)
    return tuple(sorted(pattern))


def features(row, group):
    values = [1.0, row["sum_context"] / 100000.0]
    if group in ("N2_causal", "N3_causal"):
        values.append(row["max_context"] / 100000.0)
    return values


class NativeMhaLowQueryModel:
    """A source-frozen operator model; complete-forward qualification is separate."""

    def __init__(self, reader, handoff_pin, *, deployment_scope_sha256):
        read = lambda pin, role: reader.read(pin, "low_query." + role)
        validation_reader = Evidence(reader.directory, "native_mha_low_query")
        validation_reader.prefix = VALIDATION_PREFIX
        # This contract pins independent validation as well as its source
        # model. Its admission dependency is not a fitting observation.
        handoff = validation_reader.read(handoff_pin, "handoff")
        if (handoff.get("schema") != SCHEMA or set(handoff.get("active_groups", [])) != GROUPS
                or handoff.get("old_lookup_first") is not True
                or handoff.get("q1_model_activated") is not False
                or handoff.get("whole_forward_validation_required") is not True
                or handoff["deployment_scope"]["sha256"] != deployment_scope_sha256):
            raise ValueError("native low-query handoff changes scope or validation requirements")
        scope = read(handoff["deployment_scope"], "deployment_scope")
        declaration = attention_scope.Declaration(scopes=scope["attention_scope"])
        source = read(handoff["source_model"], "source_model")
        if (source.get("schema") != "compass.selected_root_mha_source_model/1"
                or source.get("heldout_timings_read") is not False
                or source.get("e2e_timings_read") is not False
                or source.get("raw_observations_trimmed") is not False):
            raise ValueError("native low-query model changes its source-only fit")
        cases = {}
        for index, pin in enumerate(handoff["source_handoffs"]):
            book = read(pin, f"source_handoff{index}")
            if (book.get("source_qualified") is not False
                    or book.get("target_end_to_end_timings_used") is not False):
                raise ValueError("native low-query source changes its qualification or provenance")
            for case in book["cases"]:
                if case["name"] in cases:
                    raise ValueError("native low-query source names overlap")
                cases[case["name"]] = case
        self.models = {name: source["models"][name] for name in GROUPS}
        self.patterns = {name: set() for name in GROUPS}
        source_rows = {row["name"]: row for row in source["source_rows"]}
        if (len(source_rows) != len(source["source_rows"])
                or set(source["models"]) != GROUPS | {"N1_q1"}):
            raise ValueError("native low-query model repeats a source row")
        for group, fit in source["models"].items():
            selected = [row for row in source_rows.values() if row["group"] == group]
            coefficients = fit["coefficients_seconds"]
            if (len(coefficients) != (3 if group in ("N2_causal", "N3_causal") else 2)
                    or any(type(v) not in (int, float) or not isfinite(v) or v < 0 for v in coefficients)
                    or fit["source_count"] != len(selected)
                    or len(fit["residuals"]) != len(selected)
                    or {r["name"] for r in fit["residuals"]} != {r["name"] for r in selected}):
                raise ValueError("native low-query model changes its coefficients or source membership")
            residuals = {r["name"]: r for r in fit["residuals"]}
            observed_coordinates = []
            for row in selected:
                case = cases[row["name"]]
                coordinate = coordinates(case["operator"])
                if (coordinate is None or coordinate["q"] != row["q"]
                        or coordinate["context"] != row["context"]):
                    raise ValueError("native low-query fit uses an invalid or different source context")
                expected_group = ("N1_q1" if coordinate["n"] == 1 and coordinate["max_query"] == 1
                                  else f'N{coordinate["n"]}_causal')
                if group != expected_group:
                    raise ValueError("native low-query source belongs to another row-count or causal regime")
                _, record, _ = _case(reader, case, declaration)
                if record["all_three"] != row["values"] or record["seconds"] != row["median"]:
                    raise ValueError("native low-query fit changes its original observations")
                predicted = sum(a * b for a, b in zip(features(coordinate, group), coefficients))
                residual = residuals[row["name"]]
                if (not isclose(predicted, residual["predicted_seconds"], rel_tol=1e-10, abs_tol=1e-12)
                        or not isclose(predicted / row["median"] - 1, residual["relative_error"], abs_tol=1e-10)):
                    raise ValueError("native low-query source residual differs from its frozen law")
                observed_coordinates.append(coordinate)
                if group in GROUPS:
                    pattern = sharing_pattern(case["operator"], coordinate)
                    if pattern is None:
                        raise ValueError("native low-query source has unsupported non-prefix aliases")
                    self.patterns[group].add(pattern)
            limits = {name: [min(r[name] for r in observed_coordinates), max(r[name] for r in observed_coordinates)]
                      for name in ("sum_context", "max_context", "max_query")}
            if fit["measured_feature_intervals"] != limits:
                raise ValueError("native low-query model expands its measured feature intervals")
        self.handoff_sha256 = handoff_pin["sha256"]
        self.source_model = handoff["source_model"]
        inventory = read(handoff["prompt_tail_inventory"], "prompt_tail_inventory")
        if inventory.get("all_observed_payloads_exact") is not True:
            raise ValueError("native low-query model lacks its prepared-prompt identity checks")
        self.allowed_rows = {(row["tail_query"], row["tail_history"]) for row in inventory["rows"]}
        self.sharing_audit = read(handoff["sharing_audit"], "sharing_audit")
        self._validate_heldouts(validation_reader.read, handoff, source)
        reader.inputs.extend(validation_reader.inputs)

    def _validate_heldouts(self, read, handoff, source):
        report = read(handoff["heldout_validation"], "heldout_validation")
        if (not same_pin(report["source_model"], handoff["source_model"])
                or report.get("source_refitted_after_heldouts") is not False
                or report.get("e2e_timings_used") is not False
                or report.get("all_raw_observations_retained") is not True
                or report.get("threshold") != .10 or report.get("passed") is not True
                or set(report["independently_tested_model_groups"]) != GROUPS):
            raise ValueError("native low-query heldout receipt changes independence or acceptance")
        plan = read(report["plan"], "heldout_plan")
        if (not same_pin(plan["source_freeze"], handoff["source_model"])
                or read(plan["source_freeze"], "heldout_frozen_model") != source
                or not read(report["release"], "heldout_release")["verified"]
                or not read(report["collection"], "heldout_collection")["copy_complete"]):
            raise ValueError("native low-query heldouts lack their frozen source model or closed evidence")
        graph = read(plan["graph"], "heldout_graph")
        by_name = {plan["case_ids"][_signature_of(op)]: op for op in graph["ops"]}
        if (len(by_name) != 16 or len(report["rows"]) != 16
                or {r["name"] for r in report["rows"]} != set(by_name)):
            raise ValueError("native low-query validation omits or repeats heldout geometries")
        for row in report["rows"]:
            op = by_name[row["name"]]
            quote = self.quote(op)
            if quote is None or not isclose(quote["seconds"], row["predicted_seconds"], rel_tol=1e-10):
                raise ValueError("native low-query heldout is outside the source law")
            values, seeds, repeats = [], set(), set()
            signature = _signature_of(op)
            for observation in row["observations"]:
                raw = read(observation["raw"], row["name"] + ".raw")
                closed = read(observation["exit"], row["name"] + ".exit")
                acquired = {**raw["acquisition"], **raw["acquisition"]["by_signature"][signature]}
                price = raw["prices"][signature]
                if (closed["exit_code"] != 0 or closed["started_at"] <= source["created_at"]
                        or acquired["role"] != "heldout" or acquired["source_only"] is not False
                        or acquired["plan_sha256"] != report["plan"]["sha256"]
                        or acquired["case"] != row["name"] or raw["unpriced"]
                        or closed["run"]["role"] != "heldout"
                        or closed["run"]["seed"] != acquired["seed"]
                        or closed["run"]["repetition"] != acquired["repetition"]
                        or (price["cache"], price["arg_sets"], price["kv_regions"]) != ("over", 64, 4)
                        or price["seconds"] != observation["seconds"]):
                    raise ValueError("native low-query heldout changes its raw observation or treatment")
                values.append(price["seconds"])
                seeds.add(acquired["seed"])
                repeats.add(acquired["repetition"])
            if len(values) != 3 or len(seeds) != 3 or repeats != {1, 2, 3}:
                raise ValueError("native low-query heldout requires three independent observations")
            center = median(values)
            error = quote["seconds"] / center - 1
            if (center != row["observed_median_seconds"] or abs(error) > .10
                    or not isclose(error, row["relative_error"], abs_tol=1e-10)):
                raise ValueError("native low-query independent heldout exceeds its frozen error limit")
        self.validation = handoff["heldout_validation"]

    def quote(self, operator):
        row = coordinates(operator)
        if row is None or row["max_query"] == 1:
            return None
        if any(pair not in self.allowed_rows for pair in zip(row["q"], row["history"])):
            return None
        group = f'N{row["n"]}_causal'
        fit = self.models[group]
        pattern = sharing_pattern(operator, row)
        if (any(not lo <= row[name] <= hi for name, (lo, hi) in fit["measured_feature_intervals"].items())
                or pattern not in self.patterns[group]):
            return None
        seconds = sum(a * b for a, b in zip(features(row, group), fit["coefficients_seconds"]))
        return dict(seconds=seconds, model_provenance=dict(
            handoff_sha256=self.handoff_sha256, source_model=self.source_model,
            group=group, features={k: row[k] for k in ("sum_context", "max_context", "max_query")},
            measured_feature_intervals=fit["measured_feature_intervals"],
            cached_prefix_sharing_pattern=list(pattern),
            source_qualified=False, whole_forward_validation_required=True,
            kernel_dispatch_observed=False, q1_range_activated=False))
