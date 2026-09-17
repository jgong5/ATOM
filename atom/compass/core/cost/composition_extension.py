"""Explicit equivalence evidence for a later, refusal-only source addition.

The old predictor remains the one frozen before its heldouts. This receipt
does not backdate new code or qualify the newly added domain.
"""
from dataclasses import asdict
import hashlib
import json
from math import ulp
from pathlib import Path

from atom.compass.core.cost.composition_qualification import (
    code_identity, geometry, input_identity, offer_observation, source_selection,
)
from atom.compass.core.cost.native_ap_work import eligible
from atom.compass.core.loaded_input import LoadedInput, load_json

SCHEMA = "compass.forward_composition_extension/1"
PREFIX = "validation.forward_extension."
ADDED_PREFIX = "oracle.native_mha_prefill."
ADDED_OPTIONS = {"native_mha_prefill_handoff", "native_mha_prefill_handoff_sha256"}
CODE_PATHS = {
    "compass/core/cost/composition_extension.py",
    "compass/core/cost/composition_qualification.py",
    "compass/core/cost/compiled_prefill_execution.py",
    "compass/core/cost/native_mha_prefill.py",
    "compass/runtime/cache_region_oracle.py",
    "entrypoints/openai/api_server.py",
}


def observation_key(row):
    d = row["descriptor"]
    context = d["forward_context"]
    value = dict(geometry=geometry(d), context={
        "scope": context["scope"],
        "prior_sampled_batch_rows": context["prior_sampled_batch_rows"],
        **{key: d[key] for key in ("prefix_cache_hit_tokens", "temperatures", "top_ks",
                                   "top_ps", "return_logprobs", "independent_noise")},
    })
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def quote_snapshot(oracle, row):
    cost = oracle.estimate(offer_observation(oracle.native_allocation, row))
    coverage = oracle.last_coverage
    if not oracle.require_complete or not coverage.complete:
        raise ValueError("extension equivalence requires complete original coverage")
    # All raw components, M, F, readiness and complete coverage provenance.
    return json.loads(json.dumps(dict(cost=asdict(cost), coverage=asdict(coverage))))


def cohort_snapshot(oracle, sources, heldouts):
    quotes, observations = {}, []
    for role, rows in (("source_calibration", sources), ("heldout", heldouts)):
        for index, row in enumerate(rows):
            if role == "source_calibration" and not eligible(row):
                continue
            key = observation_key(row)
            if key not in quotes:
                quotes[key] = quote_snapshot(oracle, row)
            observations.append(dict(role=role, index=index, key=key))
    oracle.native_allocation.clear()
    return dict(observations=observations, quotes=quotes)


class CompositionExtension:
    def __init__(self, path, sha256, *, options):
        self.path = Path(path)
        self.loaded_inputs = []
        self.data = self.read(dict(path=str(path), sha256=sha256), "receipt")
        data = self.data
        if (data.get("schema") != SCHEMA
                or data.get("new_code_predates_old_heldouts") is not False
                or data.get("source_refitted") is not False
                or data.get("accepted") is not False
                or data.get("final_e2e_proof_required") is not True
                or data.get("validation_scope") != "unchanged_original_geometries"):
            raise ValueError("extension must preserve the old freeze and require new E2E proof")
        old = self.read(data["old_qualification"], "old_qualification")
        if data["old_qualification"] != dict(path=options["composition_qualification"],
                                            sha256=options["composition_qualification_sha256"]):
            raise ValueError("extension names a different original qualification")
        if old["predictor_identity"] != data["old_predictor_identity"]:
            raise ValueError("extension changes the original predictor identity")
        self.old_identity = self.read(data["old_predictor_identity"], "old_identity")
        self.old_quotes = self.read(data["old_quotes"], "old_quotes")
        if (self.old_quotes.get("schema") != "compass.original_predictor_quotes/1"
                or self.old_quotes.get("old_qualification") != data["old_qualification"]
                or self.old_quotes.get("old_predictor_identity") != data["old_predictor_identity"]
                or self.old_quotes.get("code") != self.old_identity["code"]
                or self.old_quotes.get("source_refitted") is not False
                or self.old_quotes.get("original_qualification_passed") is not True
                or self.old_quotes.get("heldout") != old["heldout"]):
            raise ValueError("extension baseline is not the original qualified predictor")
        for role in ("collector", "snapshot_helper"):
            self.read(self.old_quotes[role], "old_quotes." + role, json_data=False)
        current = code_identity()
        if data["new_identity"]["code"] != current:
            raise ValueError("extension new predictor code changed")
        before = self.old_identity["code"]
        changes = {name: dict(before=before.get(name), after=current.get(name))
                   for name in before.keys() | current.keys() if before.get(name) != current.get(name)}
        if (not changes or set(changes) - CODE_PATHS or changes != data["code_changes"]
                or "compass/core/cost/native_mha_prefill.py" not in changes):
            raise ValueError("extension changes code outside the bounded fallback and its validators")
        self.added_inputs = data["added_inputs"]
        if (not self.added_inputs or self.added_inputs != input_identity(self.added_inputs)
                or any(not v["role"].startswith(ADDED_PREFIX) for v in self.added_inputs)):
            raise ValueError("extension must enumerate exactly its additional fallback inputs")
        handoff = self.read(data["fallback_handoff"], "fallback_handoff")
        if (data["fallback_handoff"] != dict(path=options["native_mha_prefill_handoff"],
                                           sha256=options["native_mha_prefill_handoff_sha256"])
                or handoff.get("schema") != "compass.native_mha_prefill_fallback/1"
                or any(handoff.get(k) is not True for k in ("old_lookup_first", "refusal_only", "modelled_transfer"))
                or any(handoff.get(k) is not False for k in
                       ("base_source_book_changed", "source_refitted", "timing_equivalence_proven"))):
            raise ValueError("extension requires an explicitly bounded refusal-only fallback")
        self.handoff = handoff

    def read(self, pin, role, *, json_data=True):
        requested = str(self.path.parent / pin["path"])
        if json_data:
            value, loaded = load_json(requested, role=PREFIX + role)
        else:
            actual = Path(requested).resolve()
            value = actual.read_bytes()
            loaded = LoadedInput(PREFIX + role, requested, str(actual), False,
                                 hashlib.sha256(value).hexdigest(), len(value))
        if loaded.sha256 != pin["sha256"]:
            raise ValueError("extension evidence changed: " + role)
        self.loaded_inputs.append(loaded)
        return value

    def inherited_inputs(self, inputs):
        current = input_identity(inputs)
        additions = [v for v in current if v["role"].startswith(ADDED_PREFIX)]
        if additions != self.added_inputs:
            raise ValueError("extension added source identities differ")
        # Subtract only exact pinned additions after checking their complete set.
        for item in self.added_inputs:
            current.remove(item)
        return current

    def inherited_options(self, options):
        current = source_selection(options)
        if current != self.data["new_identity"]["body_book"]["source_selection"]:
            raise ValueError("extension new source options changed")
        return {key: value for key, value in current.items() if key not in ADDED_OPTIONS}

    def check_identity(self, identity, inputs, options, oracle):
        if (identity != self.old_identity
                or input_identity(inputs) != self.data["new_identity"]["body_book"]["loaded_inputs"]
                or self.inherited_inputs(inputs) != identity["body_book"]["loaded_inputs"]
                or self.inherited_options(options) != identity["body_book"]["source_selection"]):
            raise ValueError("extension changes the original source book or source selection")
        from atom.compass.core.cost.native_mha_prefill import NativeMhaPrefillFallback
        if (not isinstance(oracle.library, NativeMhaPrefillFallback)
                or oracle.library.handoff_sha256 != self.data["fallback_handoff"]["sha256"]):
            raise ValueError("extension requires the reviewed refusal-only adapter")
        self.primitive_control_max_error = max(abs(row["relative_error"])
            for row in oracle.library.review["native_comparisons"])

    def check_body_identity(self, historical_inputs, historical_options, oracle, options):
        from atom.compass.core.cost.compiled_prefill_execution import body_inputs, body_options
        if (body_inputs(historical_inputs) != body_inputs(self.inherited_inputs(oracle.compass_loaded_inputs))
                or body_options(historical_options) != body_options(self.inherited_options(options))):
            raise ValueError("extension changes the inherited compiled-prefill B identity")

    def check_calibration(self, oracle, source, original_quotes):
        cache, self.calibration_comparisons = {}, []
        original_observations = {row["index"]: row["key"] for row in self.old_quotes["observations"]
                                 if row["role"] == "source_calibration"}
        for observation in original_quotes["observations"]:
            index = observation["source_row_index"]
            row = source[index]
            key = observation_key(row)
            if original_observations.get(index) != key:
                raise ValueError("extension source-calibration geometry differs from the old predictor")
            if key not in cache:
                cost = oracle.estimate(offer_observation(oracle.native_allocation, row))
                if not oracle.last_coverage.complete:
                    raise ValueError("extension source-calibration coverage changed")
                cache[key] = cost.breakdown["<body>"] + cost.breakdown.get("<head>", 0.)
            old_parts = self.old_quotes["quotes"][key]["cost"]["breakdown"]
            old = old_parts["<body>"] + old_parts.get("<head>", 0.)
            new, historical = cache[key], observation["B"]
            comparison = dict(source_row_index=index, geometry=geometry(row["descriptor"]),
                old_actual_B=old, new_actual_B=new, historical_export_B=historical,
                absolute_old_new_difference=abs(new-old), old_new_difference_ulps=abs(new-old)/ulp(old),
                absolute_export_difference=abs(new-historical), export_difference_ulps=abs(new-historical)/ulp(old))
            self.calibration_comparisons.append(comparison)
            if new != old:
                raise ValueError("extension changes an original source-calibration raw B quote: " + json.dumps(comparison))
            # Existing complete-predictor export comparison policy. This applies
            # only to historical summation roundoff; old/new equality above is exact.
            if abs(new-historical) > 1e-10:
                raise ValueError("extension historical source-calibration B differs beyond the existing export policy: " + json.dumps(comparison))
        oracle.native_allocation.clear()

    def check_quotes(self, oracle, sources, heldouts, source_pin):
        if source_pin != self.old_quotes["source"]:
            raise ValueError("extension baseline changes the original source cohort")
        snapshot = cohort_snapshot(oracle, sources, heldouts)
        if (snapshot["observations"] != self.old_quotes["observations"]
                or snapshot["quotes"] != self.old_quotes["quotes"]):
            raise ValueError("extension changes an original component, timing, readiness or coverage")
        return dict(original_observations=len(snapshot["observations"]),
                    unique_original_geometries=len(snapshot["quotes"]),
                    exact_equality=True, new_code_predates_old_heldouts=False,
                    added_domain_qualified_by_old_heldouts=False,
                    primitive_control_max_relative_error=self.primitive_control_max_error,
                    primitive_controls_all_under_ten_percent=self.primitive_control_max_error < .10,
                    source_calibration=dict(old_new_exact=True, historical_export_absolute_limit_seconds=1e-10,
                                            comparisons=self.calibration_comparisons),
                    final_e2e_proof_required=True)
