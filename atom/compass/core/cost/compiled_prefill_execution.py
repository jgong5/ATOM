"""Source-calibrated elapsed M for compiled prefill without CUDA graph replay.

The raw primitive quote B remains visible. Alpha and the two floors are
effective elapsed-time parameters, not isolated CPU or GPU measurements.
"""
from dataclasses import replace
from math import isfinite
from pathlib import Path
from statistics import median

from atom.compass.core.cost.composition_qualification import geometry, input_identity, source_selection
from atom.compass.core.cost.native_ap_work import eligible
from atom.compass.core.loaded_input import load_json

SCHEMA = "compass.compiled_prefill_execution_model/1"
PREFIX = "oracle.compiled_prefill_execution."
REGION_PREFIXES = ("oracle.native_prefill_regions", "oracle.native_ap_regions",
                   "oracle.region_overlay", "oracle.native_mha_decode_layout.", PREFIX, "validation.")


def body_inputs(inputs):
    return [row for row in input_identity(inputs) if not row["role"].startswith(REGION_PREFIXES)]


def body_options(options):
    excluded = ("native_prefill_handoff", "native_ap_handoff", "region_overlay",
                "compiled_prefill_execution_handoff", "composition_qualification", "native_mha_decode_layout_handoff")
    return {key: value for key, value in source_selection(options).items()
            if not key.startswith(excluded) and key not in
            ("regions", "include_failed_outputless", "include_failed_final", "diagnostic_only")}


class CompiledPrefillExecution:
    def __init__(self, data, sha256, inputs):
        self.parameters = data["parameters"]
        self.domain = data["observed_domain"]
        self.maximum_rows = data.get("native_width_extension", {}).get("maximum_rows", self.domain["rows"][1])
        self.width_cached_only = bool(data.get("native_width_extension"))
        self.width_parameters = data.get("native_width_extension", {}).get("parameters")
        self.sha256 = sha256
        self.loaded_inputs = tuple(inputs)

    @classmethod
    def load(cls, path, sha256, *, oracle, options, extension=None):
        inputs = []
        def read(pin, role):
            value, loaded = load_json(str(Path(path).parent / pin["path"]), role=PREFIX + role)
            if loaded.sha256 != pin["sha256"]:
                raise ValueError("compiled-prefill execution input changed: " + role)
            inputs.append(loaded)
            return value
        data = read(dict(path=str(path), sha256=sha256), "rule")
        if (data.get("schema") != SCHEMA or data.get("source_only") is not True
                or data.get("frozen") is not True or data.get("source_qualified") is not False
                or data.get("heldout_rows_read") is not False or data.get("e2e_timings_read") is not False
                or data.get("AP_rule_changed") is not False or data.get("captured_decode_changed") is not False
                or data.get("primitive_price_records_changed") is not False
                or data.get("frozen_before_independent_main_heldouts") is not True
                or data.get("fitting_target") != "native run_model event interval M"
                or data.get("eligibility") != dict(capture_bucket=None, compiled=True, prefill=True, topology={"tp": 1})):
            raise ValueError("compiled-prefill execution model changes its source-only scope")
        parameters = data["parameters"]
        if (set(parameters) != {"alpha", "floor_seconds"}
                or set(parameters["floor_seconds"]) != {"cold", "cached"}
                or any(type(v) not in (int, float) or not isfinite(v) or v <= 0
                       for v in (parameters["alpha"], *parameters["floor_seconds"].values()))):
            raise ValueError("compiled-prefill execution parameters are invalid")
        evidence = {role: read(pin, role) for role, pin in data["evidence"].items()}
        width = data.get("native_width_extension")
        if width:
            active = getattr(oracle.regions, "width_extension", None)
            candidate = read(width["candidate"], "width_source_candidate")
            if (active is None or width.get("maximum_rows") != 4
                    or width.get("cached_only") is not True
                    or width.get("source_only") is not True
                    or width.get("full_forward_qualification_required") is not True
                    or width["candidate"] != active["candidate"]
                    or candidate.get("schema") != "compass.native_width_four_ap_candidate/1"
                    or candidate.get("heldout_started") is not False
                    or candidate.get("source_refitted") is not False):
                raise ValueError("compiled-prefill width extension lacks its independent native source candidate")
        quotes = evidence["body_quotes"]
        historical_inputs = read(quotes["loaded_inputs"], "body_loaded_inputs")
        historical_options = read(data["body_options"], "body_options")
        if width and ("execution_fit" in width or "parameters" in width):
            from atom.compass.core.cost.compiled_prefill_width import validate_width_fit
            validate_width_fit(width, active, candidate, read, historical_inputs)
        if extension is None:
            if body_inputs(historical_inputs) != body_inputs(oracle.compass_loaded_inputs):
                raise ValueError("compiled-prefill B sources changed; a new source fit is required")
            if body_options(historical_options) != body_options(options):
                raise ValueError("compiled-prefill B source selection changed")
        else:
            extension.check_body_identity(historical_inputs, historical_options, oracle, options)
        if (quotes.get("source_only") is not True or quotes.get("heldout_rows_read") is not False
                or quotes.get("e2e_timings_read") is not False
                or quotes["source"]["sha256"] != data["evidence"]["source"]["sha256"]):
            raise ValueError("compiled-prefill quotes mix source and target observations")
        source = evidence["source"]["rows"]
        expected = {i for i, row in enumerate(source) if eligible(row)}
        observations = quotes["observations"]
        if (any(row.get("role") != "source" for row in source)
                or len(observations) != len(expected)
                or {row["source_row_index"] for row in observations} != expected):
            raise ValueError("compiled-prefill fit omits or duplicates source observations")
        if extension is not None:
            extension.check_calibration(oracle, source, quotes)
        groups = {}
        for observation in observations:
            row = source[observation["source_row_index"]]
            quoted = quotes["quotes"][observation["geometry_key"]]
            if (quoted.get("complete") is not True or quoted["geometry"] != geometry(row["descriptor"])
                    or quoted["B"] != quoted["body_seconds"] + quoted["head_seconds"]
                    or observation["B"] != quoted["B"] or observation["M"] != row["seconds"]["run_model"]):
                raise ValueError("compiled-prefill fit changes its measured M or raw B quote")
            groups.setdefault(observation["geometry_key"], []).append(observation["M"])
        errors = []
        for key, values in groups.items():
            quote = quotes["quotes"][key]
            cached = any(quote["geometry"]["history"])
            predicted = max(parameters["alpha"] * quote["B"], parameters["floor_seconds"]["cached" if cached else "cold"])
            errors.append(abs(predicted - median(values)) / median(values))
        if (len(groups) != data["source_group_metrics"]["count"]
                or abs(max(errors) - data["source_group_metrics"]["max_absolute"]) > 1e-10):
            raise ValueError("compiled-prefill frozen parameters or source residuals differ")
        fit = evidence["fit"]["parameters"]
        if (parameters["alpha"] != fit["alpha"]
                or parameters["floor_seconds"] != dict(cold=fit["cold_floor_seconds"], cached=fit["cached_floor_seconds"])):
            raise ValueError("compiled-prefill rule differs from its frozen fit")
        return cls(data, sha256, inputs)

    def apply(self, cost, shape):
        if not shape.is_prefill or not shape.compiled or shape.capture_bucket is not None:
            return cost
        cached = any(c > q for c, q in zip(shape.context_lens, shape.num_scheduled_tokens))
        if self.width_cached_only and shape.batch_size > self.domain["rows"][1] and not cached:
            raise ValueError("compiled-prefill width-four source supports cached prefill only")
        if (dict(shape.topology) != {"tp": 1} or shape.num_prefill_tokens != shape.total_tokens
                or not self.domain["query_tokens"][0] <= shape.total_tokens <= self.domain["query_tokens"][1]
                or not self.domain["rows"][0] <= shape.batch_size <= self.maximum_rows):
            raise ValueError("compiled-prefill execution is outside its source work scope")
        raw = cost.breakdown["<body>"] + cost.breakdown.get("<head>", 0.)
        if raw <= 0:
            raise ValueError("compiled-prefill execution needs a positive raw B quote")
        floor = self.parameters["floor_seconds"]["cached" if cached else "cold"]
        alpha = self.parameters["alpha"]
        if self.width_parameters is not None and shape.batch_size == 4:
            alpha, floor = self.width_parameters["alpha"], self.width_parameters["floor_seconds"]
        model = max(alpha * raw, floor)
        adjustment = model - raw
        prepare = cost.preparation_seconds
        if prepare is None:
            raise ValueError("compiled-prefill execution needs its separately sourced preparation boundary")
        prefix = max(0., cost.output_ready_seconds - prepare)
        ready = min(prepare + model, max(prepare + prefix * model / raw, prepare + floor))
        return replace(cost, seconds=cost.seconds + adjustment, model_seconds=model,
            breakdown={**cost.breakdown, "<compiled-prefill-execution-adjustment>": adjustment},
            output_ready_seconds=ready,
            output_ready_basis={**cost.output_ready_basis, "compiled_prefill_execution": dict(
                source_sha256=self.sha256, raw_body_plus_head_seconds=raw, effective_model_seconds=model,
                empirical_caller_floor_seconds=floor,
                interpretation="Empirical M and readiness transform; no isolated CPU/GPU attribution")})
