"""A bounded native cached-MHA law for larger queries.

Only the token-derived native V capacity is abstracted. Source observations
remain whole-operator events, and independent validation remains an admission
dependency rather than a fitting input.
"""
from collections import Counter
from math import isclose, isfinite
from statistics import median

from atom.compass.core.cost.exact_operator_references import _case, _registered_scope, argument_views
from atom.compass.core.cost.families import attention, attention_scope
from atom.compass.core.cost.library import _signature_of
from atom.compass.core.cost.prepared import PreparedOperator
from atom.compass.core.cost.reached_primitive_evidence import Evidence, same_pin

SCHEMA = "compass.native_mha_large_query/1"
SOURCE_SCHEMA = "compass.native_mha_large_query_source_model/1"
VALIDATION_PREFIX = "validation.native_mha_large_query."
REGIME = attention.REGIMES["unified.prefill.cached"]
DOMAIN = dict(min_max_query=512, min_max_history=8192, max_query_rows=16384,
              sequence_counts=[1, 2, 3, 4])
OPERAND_CLASS = "native BF16 Q6144/KV1024; V stride14336 offset13312 owner2 capacity=query_total*14336"


def argument_set_count(row):
    """The pinned collector's 1-GiB input rotation, including native V storage."""
    per_set_bytes = sum(row["q"]) * (6144 + 1024 + 14336) * 2
    return max(2, min(64, (1 << 30) // per_set_bytes))


def coordinates(operator):
    """Validate the complete native operand class and cached allocation."""
    op = operator.as_dict() if isinstance(operator, PreparedOperator) else operator
    if (op.get("name") != attention.UNIFIED or op.get("group") is not None
            or op.get("int_values") or op.get("int_ranges")
            or op.get("abi", "") or op.get("launch") or op.get("param_names")
            or op.get("output_aliases") != [None]):
        return None
    context = dict(op.get("context") or ())
    cu = context.get("cu_seqlens_q") or []
    lengths = context.get("context_lens") or []
    if (not lengths or len(cu) != len(lengths) + 1 or cu[0] != 0
            or any(type(v) is not int for v in cu + lengths)):
        return None
    q = [b - a for a, b in zip(cu, cu[1:])]
    history = [c - a for c, a in zip(lengths, q)]
    n, total = len(q), sum(q)
    if (n not in DOMAIN["sequence_counts"] or any(a < 1 for a in q)
            or total > DOMAIN["max_query_rows"] or max(q) < DOMAIN["min_max_query"]
            or max(history) < DOMAIN["min_max_history"]
            or any(h < 0 or h % 16 for h in history)
            or any(c > 262144 for c in lengths)
            or context.get("is_prefill") is not True or context.get("has_cached") is not True
            or context.get("state") != "prefill_prefix"
            or context.get("seq_starts") != [0] * n
            or context.get("num_cached_tokens") != history
            or context.get("total_kv") != sum(lengths)
            or context.get("min_seqlen_q") != 0
            or context.get("max_seqlen_q") != max(q)
            or context.get("max_seqlen_k") != max(lengths)):
        return None
    cu_k = [0]
    for length in lengths:
        cu_k.append(cu_k[-1] + length)
    scalars = dict(op.get("scalars") or ())
    layer = scalars.pop("#5", None)
    if (context.get("cu_seqlens_k") != cu_k
            or layer not in {f"language_model.model.layers.{i}.self_attn" for i in range(3, 64, 4)}
            or scalars != {"#1": None, "#4": None, "#6": False, "#7": None}
            or op.get("input_shapes") != [[total, 6144], [total, 1024], [total, 1024]]
            or op.get("dtypes") != ["bfloat16"] * 3
            or op.get("output_shapes") != [[total, 6144]]
            or op.get("output_dtypes") != ["bfloat16"]
            or op.get("layouts") != [[2, [[14336, 1], 13312, total * 14336, 2]]]):
        return None
    flat = context.get("block_tables") or []
    shape = context.get("block_tables_shape") or []
    if (len(shape) != 2 or shape[0] != n or type(shape[1]) is not int
            or len(flat) % n or any(type(b) is not int or b < 0 for b in flat)):
        return None
    width = len(flat) // n
    if any((c + 15) // 16 > min(width, shape[1]) for c in lengths):
        return None
    tables = [flat[i * width:i * width + (c + 15) // 16] for i, c in enumerate(lengths)]
    prefixes = [table[:h // 16] for table, h in zip(tables, history)]
    writes = [table[h // 16:] for table, h in zip(tables, history)]
    cached = {b for table in prefixes for b in table}
    written = [b for table in writes for b in table]
    if (any(len(table) != len(set(table)) for table in tables)
            or len(written) != len(set(written)) or cached.intersection(written)
            or context.get("slot_mapping") != [table[token // 16] * 16 + token % 16
                for table, h, c in zip(tables, history, lengths) for token in range(h, c)]):
        return None
    sharing = []
    for i, first in enumerate(prefixes):
        for second in prefixes[i + 1:]:
            shared = 0
            for left, right in zip(first, second):
                if left != right:
                    break
                shared += 1
            if len(set(first).intersection(second)) != shared:
                return None
            sharing.append(shared)
    return dict(q=q, history=history, n=n, sharing=sorted(sharing))


def source_scope(declaration):
    return dict(declaration.for_family("unified"),
                measurement_treatment=("native_operator_event", "over", 4),
                operand_geometry=OPERAND_CLASS)


def fit_sources(rows, scope):
    """One source-only fit, preserving every selected observation and residual."""
    fit = attention.fit_regime(REGIME, [(attention.structure_of(row["operator"]),
        row["median"], row["name"], scope) for row in rows])
    if isinstance(fit, attention.Refusal):
        raise ValueError("native larger-query source law refused: " + fit.reason)
    residuals = []
    for row in rows:
        values = attention.features_for(REGIME, attention.structure_of(row["operator"]), scope)
        by_name = dict(zip(REGIME.features, values))
        seconds = fit.predict([by_name[name] for name in fit.features])
        residuals.append(dict(name=row["name"], predicted_seconds=seconds,
                              relative_error=seconds / row["median"] - 1))
    snapshot = dict(features=list(fit.features), coefficients=list(fit.coefficients),
        scales=list(fit.scales), pinned=list(fit.pinned), bounded=list(fit.bounded),
        measured_feature_intervals={name: [min(r[i] for r in fit.domain), max(r[i] for r in fit.domain)]
                                   for i, name in enumerate(fit.features)},
        sequence_counts=dict(sorted(Counter(str(len(row["q"])) for row in rows).items())),
        sharing_patterns={str(n): sorted({tuple(row["sharing"]) for row in rows if len(row["q"]) == n})
                          for n in sorted({len(row["q"]) for row in rows})},
        source_count=len(rows), residuals=residuals)
    # Keep the artifact representation deterministic across JSON round trips.
    snapshot["sharing_patterns"] = {n: [list(p) for p in patterns]
                                    for n, patterns in snapshot["sharing_patterns"].items()}
    return fit, snapshot


class NativeMhaLargeQueryModel:
    """Private larger-query law; exact prices and forward qualification stay outside."""

    def __init__(self, reader, handoff_pin, *, deployment_scope_sha256):
        read = lambda pin, role: reader.read(pin, "large_query." + role)
        validation = Evidence(reader.directory, "native_mha_large_query")
        validation.prefix = VALIDATION_PREFIX
        handoff = validation.read(handoff_pin, "handoff")
        if (handoff.get("schema") != SCHEMA or handoff.get("domain") != DOMAIN
                or handoff.get("old_lookup_first") is not True
                or handoff.get("whole_forward_validation_required") is not True
                or handoff["deployment_scope"]["sha256"] != deployment_scope_sha256):
            raise ValueError("native larger-query handoff changes its source domain or scope")
        declared = read(handoff["deployment_scope"], "deployment_scope")
        declaration = attention_scope.Declaration(scopes=declared["attention_scope"])
        self.declaration = declaration
        source = read(handoff["source_model"], "source_model")
        if (source.get("schema") != SOURCE_SCHEMA or source.get("domain") != DOMAIN
                or source.get("heldout_timings_read") is not False
                or source.get("e2e_timings_read") is not False
                or source.get("raw_observations_trimmed") is not False):
            raise ValueError("native larger-query model changes its source-only fit")
        cases = {}
        for index, pin in enumerate(source["source_handoffs"]):
            book = read(pin, f"source_handoff{index}")
            if (book.get("source_qualified") is not False
                    or book.get("target_end_to_end_timings_used") is not False):
                raise ValueError("native larger-query source changes its qualification")
            for case in book["cases"]:
                row = coordinates(case["operator"])
                if row is None:
                    continue
                if case["name"] in cases:
                    raise ValueError("native larger-query source names overlap")
                cases[case["name"]] = case
        rows = []
        for row in source["source_rows"]:
            case = cases.pop(row["name"])
            actual = coordinates(case["operator"])
            _, record, _ = _case(reader, case, declaration)
            if (record["all_three"] != row["values"] or record["seconds"] != row["median"]
                    or record["relative_spread"] != row["spread"]
                    or any(actual[key] != row[key] for key in ("q", "history", "sharing"))
                    or case["timer_mode"] != "over"
                    or any((ref["arg_sets"], ref["kv_regions"]) != (argument_set_count(actual), 4)
                           for ref in case["references"])):
                raise ValueError("native larger-query source changes its observations or geometry")
            rows.append(dict(row, operator=case["operator"]))
        if cases:
            raise ValueError("native larger-query fit omits eligible source observations")
        self.scope = source_scope(declaration)
        self.fit, snapshot = fit_sources(rows, self.scope)
        if snapshot != source["fit"]:
            raise ValueError("native larger-query source freeze differs from its deterministic fit")
        self.snapshot = snapshot
        self.handoff_sha256 = handoff_pin["sha256"]
        self.source_model = handoff["source_model"]
        self.source_geometries = {(tuple(row["q"]), tuple(row["history"]), tuple(row["sharing"])) for row in rows}
        self._validate_heldouts(validation.read, handoff, source)
        reader.inputs.extend(validation.inputs)

    def _validate_heldouts(self, read, handoff, source):
        report = read(handoff["heldout_validation"], "heldout_validation")
        if (not same_pin(report["source_model"], handoff["source_model"])
                or report.get("source_refitted_after_heldouts") is not False
                or report.get("e2e_timings_used") is not False
                or report.get("all_raw_observations_retained") is not True
                or report.get("threshold") != .10 or report.get("passed") is not True):
            raise ValueError("native larger-query heldouts change independence or acceptance")
        plan = read(report["plan"], "heldout_plan")
        if (not same_pin(plan["source_freeze"], handoff["source_model"])
                or read(plan["source_freeze"], "heldout_frozen_model") != source
                or not read(report["release"], "heldout_release")["verified"]
                or not read(report["collection"], "heldout_collection")["copy_complete"]):
            raise ValueError("native larger-query heldouts lack the frozen model or closed evidence")
        graph = read(plan["graph"], "heldout_graph")
        by_name = {plan["case_ids"][_signature_of(op)]: op for op in graph["ops"]}
        if (len(by_name) != len(report["rows"])
                or {r["name"] for r in report["rows"]} != set(by_name)):
            raise ValueError("native larger-query validation omits or repeats heldout geometries")
        tested_counts = set()
        for row in report["rows"]:
            op = by_name[row["name"]]
            coordinate = coordinates(op)
            quote = self.quote(op)
            if (quote is None or not isclose(quote["seconds"], row["predicted_seconds"], rel_tol=1e-10)
                    or (tuple(coordinate["q"]), tuple(coordinate["history"]), tuple(coordinate["sharing"]))
                        in self.source_geometries):
                raise ValueError("native larger-query heldout is outside the law or repeats source geometry")
            tested_counts.add(str(coordinate["n"]))
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
                        or (price["cache"], price["arg_sets"], price["kv_regions"])
                            != ("over", argument_set_count(coordinate), 4)
                        or price["seconds"] != observation["seconds"]):
                    raise ValueError("native larger-query heldout changes its raw observation or treatment")
                setup = raw["provenance"]["native_layout_context_setup"]
                setup = {**setup, **setup["by_signature"][signature]}
                if (setup.get("before_timer_only") is not True
                        or setup["graph"]["sha256"] != plan["graph"]["sha256"]
                        or len(setup["operand_sets"]) != price["arg_sets"]
                        or any(views != argument_views(op) for views in setup["operand_sets"])
                        or not setup["standup"]["extent_preflight"]["passed"]
                        or not setup["standup"]["gather_correctness_preflight"]["passed"]):
                    raise ValueError("native larger-query heldout changes its live layout or gather proof")
                observed = {**raw, "provenance": {**raw["provenance"], "native_layout_context_setup": setup}}
                _registered_scope(op, observed, attention.scoped(op, self.declaration.for_family("unified")), plan)
                values.append(price["seconds"])
                seeds.add(acquired["seed"])
                repeats.add(acquired["repetition"])
            if len(values) != 3 or len(seeds) != 3 or repeats != {1, 2, 3}:
                raise ValueError("native larger-query heldout requires three independent observations")
            center = median(values)
            error = quote["seconds"] / center - 1
            if (center != row["observed_median_seconds"] or abs(error) > .10
                    or not isclose(error, row["relative_error"], abs_tol=1e-10)):
                raise ValueError("native larger-query heldout exceeds its frozen error limit")
        if tested_counts != set(self.snapshot["sequence_counts"]):
            raise ValueError("native larger-query heldouts omit a measured sequence count")
        self.validation = handoff["heldout_validation"]

    def quote(self, operator):
        row = coordinates(operator)
        if row is None or row["sharing"] not in self.snapshot["sharing_patterns"].get(str(row["n"]), []):
            return None
        op = operator.as_dict() if isinstance(operator, PreparedOperator) else operator
        values = attention.features_for(REGIME, attention.structure_of(op), self.scope)
        by_name = dict(zip(REGIME.features, values))
        if any(by_name[name] for name in self.fit.pinned):
            return None
        selected = [by_name[name] for name in self.fit.features]
        if attention._outside_domain(self.fit, selected):
            return None
        seconds = self.fit.predict(selected)
        if not isfinite(seconds) or seconds <= 0:
            return None
        return dict(seconds=seconds, model_provenance=dict(
            handoff_sha256=self.handoff_sha256, source_model=self.source_model,
            features=by_name, measured_feature_intervals=self.snapshot["measured_feature_intervals"],
            sequence_count=row["n"], measured_sequence_counts=self.snapshot["sequence_counts"],
            cached_prefix_sharing_pattern=row["sharing"], operand_class=OPERAND_CLASS,
            source_conditioning=dict(cache="over", kv_regions=4, arg_sets=argument_set_count(row),
                                     argument_rotation_bytes=1 << 30),
            source_qualified=False, whole_forward_validation_required=True, kernel_dispatch_observed=False))
