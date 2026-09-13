"""Guarded finite attention measurements with independent composition evidence."""

from collections import defaultdict
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path

from atom.compass.core.loaded_input import load_json


TAG_SCHEMA = "compass.attention_exact_override/1"
WITNESS_SCHEMA = "compass.attention_composition_witness/1"
EQUIVALENCE_SCHEMA = "compass.attention_rank_equivalence/2"


class AttestedAttentionRecord(dict):
    """Only validated evidence can create the special launch-count result."""

    __slots__ = ("attested_launch_count",)

    def __init__(self, record, count, **metadata):
        super().__init__(record, **metadata)
        self.attested_launch_count = count
        self["attention_exact_override"] = True
        self["launch_count"] = count


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@lru_cache(maxsize=32)
def _reference(path, expected):
    blob, loaded = load_json(path, role="oracle.attention_exact.evidence")
    if loaded.sha256 != expected:
        raise ValueError(f"attention exact evidence digest mismatch: {path}")
    # Witnesses retain large argument dumps for audit. Pricing needs only the
    # independently recorded identity, views and composition, indexed once.
    if blob.get("schema") == WITNESS_SCHEMA:
        fields = ("cost_key", "geometry", "layouts", "input_shapes", "kernel_counts", "launch_count")
        blob = {"schema": blob["schema"], "tp": blob["tp"], "rank": blob["rank"],
                "calls": {call["cost_key"]: {k: call[k] for k in fields}
                          for call in blob["calls"]}}
    elif blob.get("schema") == EQUIVALENCE_SCHEMA:
        blob = dict(blob, graph_index={
            item["rank"]: {call["cost_key"]: call for call in item["calls"]}
            for item in blob["graphs"]})
    return blob, loaded


def _layout_without_capacity(scope):
    """KV allocation extent is source conditioning, not the view's layout."""
    from atom.compass.core.cost.families.attention import _hashable

    result = {key: _hashable(value) for key, value in scope.items()}
    layout = result.get("kv_cache_layout")
    if not isinstance(layout, (list, tuple)):
        return result
    try:
        views = dict(layout)
        if set(views) != {"k", "v"}:
            return result
        normalized = []
        for name, description in layout:
            fields = dict(description)
            shape, stride = fields.get("shape"), fields.get("stride")
            if (not isinstance(shape, (list, tuple)) or len(shape) != 5
                    or not isinstance(stride, (list, tuple)) or len(stride) != 5
                    or type(shape[0]) is not int or shape[0] <= 0):
                return result
            normalized.append((name, tuple((key, ("*blocks",) + tuple(value[1:])
                                           if key == "shape" else value)
                                          for key, value in description)))
        result["kv_cache_layout"] = _hashable(normalized)
    except (TypeError, ValueError):
        pass
    return result


class ExactAttentionOverrides:
    def __init__(self):
        self.entries = defaultdict(list)
        self._seen_refs = set()

    def _read(self, library, ref, where):
        if (not isinstance(ref, dict) or not isinstance(ref.get("path"), str)
                or not isinstance(ref.get("sha256"), str) or len(ref["sha256"]) != 64):
            raise ValueError(f"{where}: a hashed evidence reference is required")
        path = Path(ref["path"])
        if not path.is_absolute():
            path = Path(where).parent / path
        key = (str(path), ref["sha256"])
        blob, loaded = _reference(*key)
        if key not in self._seen_refs:
            library.loaded_inputs += (loaded,)
            self._seen_refs.add(key)
        return blob

    def add(self, library, price_path, blob, graph):
        from atom.compass.core.cost.families import attention as A, attention_scope
        from atom.compass.core.cost.families.adapter import (
            _acquisition_policy, _attention_scope, _hashable, _measurement_identity,
        )
        from atom.compass.core.cost.identity import cost_key
        from atom.compass.core.cost.library import _layout_fingerprint
        from atom.compass.runtime.microbench import signature_of

        provenance = blob.get("provenance") or {}
        tag = provenance.get("attention_exact_override")
        if (provenance.get("exact_only") is not True or not isinstance(tag, dict)
                or tag.get("schema") != TAG_SCHEMA
                or tag.get("regime") != "unified.prefill.cached" or graph is None):
            raise ValueError(f"{price_path}: invalid exact attention override declaration")
        plan = self._read(library, tag.get("selection_plan"), price_path)
        selection = tag.get("selection_policy")
        if selection != plan.get("policy"):
            raise ValueError(f"{price_path}: selection policy differs from its frozen plan")
        raw_inputs = provenance.get("raw_inputs") or ()
        if not raw_inputs:
            raise ValueError(f"{price_path}: exact override has no raw timing inputs")
        raw = [self._read(library, ref, price_path) for ref in raw_inputs]
        resolved_blob = blob.get("resolved_scope")
        if not isinstance(resolved_blob, dict):
            raise ValueError(f"{price_path}: live resolved scope is required")
        scope_source = self._read(library, provenance.get("resolved_scope_source"), price_path)
        if resolved_blob != scope_source:
            raise ValueError(f"{price_path}: embedded and witnessed resolved scopes differ")
        resolved = attention_scope.read_resolved(resolved_blob, where=price_path)
        declared = _attention_scope(blob, None)
        policy = _acquisition_policy(blob)
        collector = provenance.get("collector") or {}
        if (policy == ("unevidenced",)
                or collector.get("served_workload") is not False
                or collector.get("weights_loaded") is not False
                or collector.get("model_runner_initialized") is not False):
            raise ValueError(f"{price_path}: a recorded source-only acquisition is required")
        rank = int(provenance["collector"]["rank"])
        width = int(provenance["collector"]["world_size"])
        for item in raw:
            actual = (item.get("provenance") or {}).get("collector") or {}
            if (_acquisition_policy(item) != policy
                    or actual.get("rank") != rank or actual.get("world_size") != width
                    or any(actual.get(field) is not False for field in (
                        "served_workload", "weights_loaded", "model_runner_initialized"))):
                raise ValueError(f"{price_path}: raw acquisition identity differs from the export")
        operators = {signature_of(op): op for op in graph.get("ops") or ()}
        for signature, record in (blob.get("prices") or {}).items():
            op = operators.get(signature)
            if op is None or op.get("name") != A.UNIFIED or op.get("group") is not None:
                raise ValueError(f"{price_path}: exact timing has no matching native operator")
            seconds = record.get("seconds")
            if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
                raise ValueError(f"{price_path}: invalid exact duration")
            originals = [item.get("prices", {}).get(signature) for item in raw]
            if not any(item is not None and item.get("seconds") == seconds
                       and _measurement_identity(item, policy) == _measurement_identity(record, policy)
                       for item in originals):
                raise ValueError(f"{price_path}: exact duration/treatment is not copied from raw evidence")
            identity = _measurement_identity(record, policy)
            treatment = {"kernels": identity[0], **dict(identity[1:])}
            if _hashable(treatment) != _hashable(tag.get("measurement_treatment")):
                raise ValueError(f"{price_path}: actual treatment differs from its declaration")
            if (record.get("cache") != selection.get("observed_cache_required")
                    or record.get("kv_regions") != selection.get("kv_variants")):
                raise ValueError(f"{price_path}: acquisition did not realize the selected policy")
            scope = dict(declared)
            scope.update(resolved.for_op(op))
            regime = A.regime_of(op, None, scope)
            if isinstance(regime, A.Refusal) or regime.name != tag["regime"]:
                raise ValueError(f"{price_path}: exact operator is not in the declared regime")
            if any(field not in scope for field in regime.required_scope):
                raise ValueError(f"{price_path}: exact source omits required native scope")
            witness_ref = record.get("composition_witness")
            witness = self._read(library, witness_ref, price_path)
            key = cost_key(signature)
            if (witness.get("schema") != WITNESS_SCHEMA or witness.get("tp") != width
                    or witness_ref.get("schema") != WITNESS_SCHEMA or witness_ref.get("cost_key") != key
                    or _hashable(witness_ref.get("measurement_treatment")) != _hashable(treatment)):
                raise ValueError(f"{price_path}: composition witness identifies different work/treatment")
            call = witness["calls"].get(key)
            geometry = A.geometry_of(op)
            if (call is None or _hashable(call["geometry"]) != _hashable(geometry)
                    or call["layouts"] != op.get("layouts")
                    or call["input_shapes"] != op.get("input_shapes")):
                raise ValueError(f"{price_path}: composition geometry/layout differs")
            counts = call["kernel_counts"]
            count = call["launch_count"]
            if (type(count) is not int or count <= 0 or not counts
                    or any(not isinstance(k, str) or type(v) is not int or v <= 0 for k, v in counts.items())
                    or sum(counts.values()) != count or witness_ref.get("launch_count") != count
                    or witness_ref.get("kernel_counts") != counts):
                raise ValueError(f"{price_path}: invalid witnessed kernel multiplicities")
            equivalence = self._read(library, witness_ref.get("rank_equivalence"), price_path)
            if (equivalence.get("schema") != EQUIVALENCE_SCHEMA
                    or equivalence.get("tp") != width or not equivalence.get("all_144_calls_equal")
                    or equivalence.get("representative_rank") != witness.get("rank")):
                raise ValueError(f"{price_path}: cross-rank composition equivalence is not established")
            proof = equivalence["graph_index"].get(rank, {}).get(key)
            if proof is None or any(proof.get(field + "_sha256") != _digest(value) for field, value in (
                    ("geometry", geometry), ("layouts", op.get("layouts")),
                    ("input_shapes", op.get("input_shapes")), ("dtypes", op.get("dtypes")))):
                raise ValueError(f"{price_path}: this rank's call differs from its equivalence proof")
            scope_proof = next((item for item in equivalence["scopes"] if item["rank"] == rank), None)
            if (scope_proof is None or not scope_proof.get("timing_and_witness_scopes_equal")
                    or not scope_proof.get("timing_and_witness_conditioning_equal")
                    or scope_proof.get("resolved_scope_sha256") != _digest(resolved.for_op(op))):
                raise ValueError(f"{price_path}: this rank's live scope/conditioning is not witnessed")
            source_scope = A.scoped(op, scope)
            normalized = _layout_without_capacity(source_scope)
            result = AttestedAttentionRecord(record, count, source=price_path, signature=signature,
                composition_witness=dict(witness_ref),
                source_conditioning={"native_scope": source_scope},
                exact_selection_plan=dict(tag["selection_plan"]))
            self.entries[key].append({"record": result, "scope": normalized,
                                      "layout": _layout_fingerprint(op), "treatment": treatment,
                                      "composition": _hashable(counts), "regime": regime.name})

    def lookup(self, library, op, topology):
        from atom.compass.core.cost.families import attention as A
        from atom.compass.core.cost.families.adapter import _hashable, _topology_key
        from atom.compass.core.cost.identity import cost_key
        from atom.compass.core.cost.library import _layout_fingerprint
        from atom.compass.runtime.microbench import signature_of

        if not self.entries or op.get("name") != A.UNIFIED:
            return None
        context = dict(op.get("context") or ())
        if context.get("is_prefill") is not True or context.get("has_cached") is not True:
            return None
        signature = signature_of(op)
        key = cost_key(signature)
        candidates = self.entries.get(key)
        if not candidates:
            return None
        if op.get("group") is not None:
            return None, "exact attention override cannot answer a collective call"
        requested = library._declared_scope(op)
        explicit_treatment = requested.pop("measurement_treatment", None)
        if topology:
            requested.setdefault("topology", _topology_key(topology))
        requested = _layout_without_capacity(A.scoped(op, requested))
        layout = _layout_fingerprint(op)
        selected = []
        for entry in candidates:
            if entry["layout"] != layout or A._scope_matches(entry["scope"], requested) is not None:
                continue
            selector = dict(library.request_attention_treatments.get(entry["regime"], {}))
            if explicit_treatment is not None:
                selector.update({"kernels": explicit_treatment[0], **dict(explicit_treatment[1:])})
            if any(_hashable(entry["treatment"].get(k, A.ABSENT)) != _hashable(v)
                   for k, v in selector.items()):
                continue
            selected.append(entry)
        if not selected:
            return None, "exact attention override differs in current layout, native scope, or selected treatment"
        groups = {(A.scope_key(item["scope"]), A.scope_key(item["treatment"]), item["composition"])
                  for item in selected}
        if len(groups) != 1:
            return None, "exact attention override has multiple distinct matching acquisition/composition groups"
        selected.sort(key=lambda item: item["record"]["seconds"])
        result = selected[len(selected) // 2]["record"]
        if result["signature"] != signature:
            library.address_shifted[key] = library.address_shifted.get(key, 0) + 1
        return result, result["source"]
