"""Isolated q16 source prices, without extending any parametric query family."""

import math
from pathlib import Path

from atom.compass.core.cost.library import (
    INTERPOLATED_FLAG, PriceLibrary, _cost_key_of, _layout_fingerprint,
)
from atom.compass.core.cost.prepared import PreparedOperator
from atom.compass.core.loaded_input import load_json


GDN = "aiter::linear_attention_with_output_base"
MHA = "aiter::unified_attention_with_output_base"
GATHER = "aten::index.Tensor"
HISTORIES = (32768, 65536, 131072)


def _mha_identity(op):
    """Remove only the validated history coordinate, preserving layer and ABI."""
    context = dict(op.get("context") or ())
    histories = context.get("context_lens")
    if not isinstance(histories, (list, tuple)) or len(histories) != 1:
        return None
    history = histories[0]
    if (type(history) is not int or not HISTORIES[0] <= history <= HISTORIES[-1]
            or (history - 16) % 16
            or context.get("cu_seqlens_q") != [0, 16]
            or context.get("max_seqlen_q") != 16
            or context.get("cu_seqlens_k") != [0, history]
            or context.get("max_seqlen_k") != history
            or context.get("total_kv") != history
            or context.get("num_cached_tokens") != [history - 16]
            or context.get("has_cached") is not True
            or context.get("is_prefill") is not True
            or context.get("state") != "prefill_prefix"):
        return None
    normalized = dict(op)
    replacements = {"context_lens": [0], "cu_seqlens_k": [0, 0],
                    "max_seqlen_k": 0, "total_kv": 0, "num_cached_tokens": [0]}
    normalized["context"] = [[key, replacements.get(key, value)]
                             for key, value in op["context"]]
    return (history, (_cost_key_of(normalized), _layout_fingerprint(op),
                     tuple(tuple(shape) for shape in op.get("output_shapes") or ()),
                     tuple(op.get("output_dtypes") or ())))


class CachedQ16Prices(PriceLibrary):
    """Compose a frozen, bounded addition with the unchanged baseline provider.

    All source files live in a separate exact library. In particular, its
    low-query observations never enter the baseline's fitted families. Body
    pricing retains the baseline's per-body memo and host synchronization
    rules, as well as reuse of prepared static graph segments.
    """

    def __init__(self, base, handoff_path, handoff_sha256):
        super().__init__()
        self.base = base
        handoff, loaded = load_json(handoff_path, role="oracle.q16_sources")
        if loaded.sha256 != handoff_sha256:
            raise ValueError("q16 source handoff differs from its explicit SHA-256")
        scope = handoff.get("scope") or {}
        if (handoff.get("schema") != "compass.q16_reference_export/1"
                or handoff.get("source_qualified") is not True
                or handoff.get("all_error_gates_pass") is not True
                or handoff.get("fit_inputs_are_references_only") is not True
                or handoff.get("heldout_timings_used_as_fit_inputs") is not False
                or scope.get("model") != "Qwen/Qwen3.8-27B"
                or scope.get("topology") != {"tp": 1}
                or scope.get("query_tokens") != 16
                or scope.get("num_sequences") != 1
                or scope.get("dtype") != "bfloat16"):
            raise ValueError("unqualified or incompatible q16 source handoff")
        if getattr(base, "launch_charge_seconds", 0) != 0:
            raise ValueError("q16 sources do not qualify a kernel launch count")
        self.source = PriceLibrary()
        self._mha = {}
        self.handoff_sha256 = loaded.sha256
        expected = {"gdn_fork_ref": ("gdn", 48, "graph", 1),
                    "gather_h16_m1": ("gather", 1, "graph", 1),
                    **{f"mha_h{h}": ("mha", 16, "over", 8) for h in HISTORIES}}
        seen = set()
        for entry in handoff["entries"]:
            case = entry["case_id"]
            if case in seen or case not in expected:
                raise ValueError("unexpected or duplicate q16 source case")
            seen.add(case)
            family, count, timer, regions = expected[case]
            if (entry.get("family") != family or entry.get("source_qualified") is not True
                    or entry.get("observed_cache") != timer or entry.get("kv_regions") != regions):
                raise ValueError("q16 source acquisition treatment differs")
            # Export bundles are relocatable: each hashed price/graph is
            # adjacent to the handoff; original acquisition paths stay intact.
            paths = [Path(handoff_path).parent / Path(entry[k]["path"]).name
                     for k in ("price", "graph")]
            blob, graph = self.source._ingest(str(paths[0]), str(paths[1]), None, None)
            for key, actual in zip(("price", "graph"), self.source.loaded_inputs[-2:]):
                if actual.sha256 != entry[key]["sha256"]:
                    raise ValueError(f"q16 {key} differs from its handoff SHA-256")
            if (len(blob.get("prices") or {}) != count or blob.get("unpriced")
                    or len(graph.get("ops") or ()) != count):
                raise ValueError("q16 source graph/price coverage differs")
            for op in graph["ops"]:
                record, why = self.source.lookup(op)
                if (record is None or record.get("cache") != timer
                        or record.get("kv_regions") != regions
                        or not math.isfinite(float(record["seconds"]))
                        or float(record["seconds"]) < 0):
                    raise ValueError(f"invalid q16 source price: {why}")
                if family == "mha":
                    identity = _mha_identity(op)
                    if op.get("name") != MHA or identity is None:
                        raise ValueError("q16 MHA source is outside the native history domain")
                    history, key = identity
                    if case != f"mha_h{history}":
                        raise ValueError("q16 MHA source history differs from case")
                    self._mha.setdefault(key, {})[history] = record
                elif op.get("name") != (GDN if family == "gdn" else GATHER):
                    raise ValueError("q16 source operator differs from its declared family")
        if seen != set(expected) or len(self._mha) != 16 or any(
                set(points) != set(HISTORIES) for points in self._mha.values()):
            raise ValueError("q16 sources do not retain all per-layer reference anchors")
        self.loaded_inputs = base.loaded_inputs + (loaded,) + self.source.loaded_inputs
        self.sources = base.sources + self.source.sources
        # Existing exact records remain eligible for the body's prepared
        # lookup cache. Source additions have their own bounded lookup below.
        self._prices = base._prices
        self.address_shifted = base.address_shifted

    def add(self, *args, **kwargs):
        raise ValueError("build baseline prices before attaching the frozen q16 addition")

    def _source_lookup(self, op, topology):
        name = op.get("name")
        if name not in (GDN, MHA, GATHER):
            return None
        if any(int(width) != 1 for width in (topology or {}).values()):
            return None
        if isinstance(op, PreparedOperator):
            op = op.as_dict()
        if op.get("group") is not None:
            return None
        if name == MHA:
            identity = _mha_identity(op)
            if identity is None:
                return None
            history, key = identity
            points = self._mha.get(key)
            if points is None:
                return None
            if history in points:
                return points[history], points[history]["source"]
            lo, hi = (HISTORIES[:2] if history < HISTORIES[1] else HISTORIES[1:])
            weight = (history - lo) / (hi - lo)
            low, high = points[lo], points[hi]
            record = dict(low, seconds=(1 - weight) * low["seconds"] + weight * high["seconds"],
                          **{INTERPOLATED_FLAG: True})
            record["interpolation"] = {
                "family": MHA, "basis": "q16 per-layer piecewise total history",
                "histories": [lo, hi], "weights": [1 - weight, weight],
                "source_signatures": [low["signature"], high["signature"]],
                "source_handoff_sha256": self.handoff_sha256,
                "launch_count_qualified": False,
            }
            return record, "interpolated://cached-q16/per-layer-history"
        if name == GATHER:
            if (op.get("input_shapes") != [[16, 5120], [1]]
                    or dict(op.get("int_values") or ()).get(1) != [15]):
                return None
        # The source's exact normalized key keeps query, N, layer, dtypes,
        # initial-state presence and input/output state counts. Only allocator
        # addresses are erased by the existing identity rule; fork, relocation
        # and in-place q16 controls were independently qualified.
        result = self.source.lookup(op, topology)
        return result if result[0] is not None else None

    def lookup(self, op, topology=None, registration=None):
        selected = self._source_lookup(op, topology)
        return selected if selected is not None else self.base.lookup(op, topology, registration)

    def _body_lookup(self, op, topology, registration, modelled_memo):
        selected = self._source_lookup(op, topology)
        return (selected if selected is not None else
                self.base._body_lookup(op, topology, registration, modelled_memo))

    def host_sync_reason(self, op):
        return self.base.host_sync_reason(op)

    def _can_reuse_prepared_lookups(self):
        return self.base._can_reuse_prepared_lookups()

    def _prepared_config_key(self, topology, registration):
        key = self.base._prepared_config_key(topology, registration)
        return None if key is None else (key, self.handoff_sha256)

    def describe(self):
        return (f"CachedQ16Prices({self.base.describe()}; exact q16 GDN/gather; "
                "q16 MHA per-layer total history [32768,131072]; "
                f"source={self.handoff_sha256}; launch counts unqualified)")
