"""Explicit native-V address transfer into the existing bounded decode model.

Exact native references retain priority. Only a private lookup copy loses the
proved V address translation; the runtime graph and its scope remain intact.
The result is modelled coverage, with the observed source bias retained.
"""
from pathlib import Path
import re

from atom.compass.core.cost.exact_operator_references import argument_views
from atom.compass.core.cost.families import attention, attention_scope
from atom.compass.core.cost.families.adapter import ParametricPriceLibrary
from atom.compass.core.cost.families.exact_attention import _layout_without_capacity
from atom.compass.core.cost.families.features import contract_for
from atom.compass.core.cost.library import INTERPOLATED_FLAG, PriceLibrary
from atom.compass.core.cost.prepared import PreparedOperator
from atom.compass.core.cost.reached_primitive_evidence import Evidence

SCHEMA = "compass.native_mha_decode_layout_transfer/1"
ROLE_PREFIX = "oracle.native_mha_decode_layout."
REGIME = "unified.decode.paged_gluon_dispatch"
DECODE_SCOPE = {"attention_backend": "paged_gluon", "compute_units": 80,
                "decode_dispatch_topology": "gfx942-spx-4xcc-80cu",
                "kv_cache_block_size": 16, "kv_cache_dtype": "bfloat16",
                "kv_cache_layout": "SHUFFLE", "num_kv_heads": 4, "sliding_window": -1}


def native_decode_layout(op):
    """The exact operand/address class whose per-token cache writes were proved."""
    if op.get("name") != attention.UNIFIED or op.get("group") is not None:
        return False
    scalars = dict(op.get("scalars") or ())
    if ({key: value for key, value in scalars.items() if key != "#5"}
            != {"#1": None, "#4": None, "#6": False, "#7": None}
            or op.get("int_values") or op.get("int_ranges")):
        return False
    structure = attention.structure_of(op)
    if structure is None or structure.is_prefill is not False:
        return False
    # Validate this registered layer without copying/hashing the large native
    # block table a second time after the exact-reference lookup already did so.
    layer = scalars.get("#5")
    match = re.fullmatch(r"language_model\.model\.layers\.(\d+)\.self_attn", layer) if isinstance(layer, str) else None
    if match is None or int(match[1]) not in range(3, 64, 4):
        return False
    shapes = [list(shape) for shape in op.get("input_shapes", ())]
    if not shapes or not shapes[0] or shapes[0][0] not in (1, 2, 4):
        return False
    rows = shapes[0][0]
    expected = [[rows, 6144], [rows, 1024], [rows, 1024]]
    if (shapes != expected or op.get("output_aliases") != [None]
            or [list(shape) for shape in op.get("output_shapes", ())] != [[rows, 6144]]
            or list(op.get("output_dtypes", ())) != ["bfloat16"]):
        return False
    views = argument_views(op)
    wanted = [dict(shape=shape, dtype="bfloat16", stride=[shape[1], 1],
                   offset=0, elements=rows * shape[1], owner=index)
              for index, shape in enumerate(expected)]
    wanted[2].update(stride=[14336, 1], offset=13312, elements=rows * 14336)
    return views == wanted


class NativeMhaDecodeLayout(PriceLibrary):
    """A scoped model fallback around exact sources, never a new measurement."""
    def __init__(self, base, handoff_path, handoff_sha256, *, deployment_scope_sha256):
        super().__init__()
        reader = Evidence(Path(handoff_path).parent, "native_mha_layout")
        reader.prefix = ROLE_PREFIX
        handoff = reader.read(dict(path=str(handoff_path), sha256=handoff_sha256), "handoff")
        if (handoff.get("schema") != SCHEMA or handoff.get("modelled_transfer") is not True
                or handoff.get("source_refitted") is not False
                or handoff["deployment_scope"]["sha256"] != deployment_scope_sha256):
            raise ValueError("native MHA layout transfer changes its modelled policy or deployment scope")
        proof = reader.read(handoff["address_proof"], "address_proof")
        if (proof.get("schema") != "compass.mha_decode_v_address_translation_proof/1"
                or any(proof.get(key) is not True for key in (
                    "arithmetic_and_launch_shape_equal", "cache_write_destinations_equal",
                    "V_read_addresses_differ_only_by_per_CTA_translation",
                    "downstream_decode_reads_KV_cache_not_original_V"))
                or proof.get("timing_equivalence_proven") is not False):
            raise ValueError("native MHA layout transfer lacks its scoped address proof")
        reader.read(proof["native_attention"], "native_attention", json_data=False)
        root = Path(__file__).resolve().parents[4]
        reader.read(dict(path=str(root / "atom/model_ops/attention_mha.py"),
                         sha256=proof["native_attention"]["sha256"]), "current_attention", json_data=False)
        kernel = proof["current_remote_kernel"]
        reader.read(dict(path=kernel["local_snapshot"], sha256=kernel["sha256"]), "cache_kernel", json_data=False)
        controls = reader.read(proof["source_handoff"], "controls")
        if len(controls["cases"]) != 7 or not all(native_decode_layout(case["operator"]) for case in controls["cases"]):
            raise ValueError("native MHA layout proof controls changed their operand class")
        comparison = reader.read(handoff["bias_comparison"], "bias_comparison")
        if (comparison.get("predictions_refitted") is not False
                or comparison["source"] != proof["source_handoff"]):
            raise ValueError("native MHA transfer erased or refitted its source residuals")
        declared = reader.read(handoff["deployment_scope"], "deployment_scope")
        self.declaration = attention_scope.Declaration(scopes=declared["attention_scope"])
        if self.declaration.for_family("unified.decode") != DECODE_SCOPE:
            raise ValueError("native MHA layout transfer requires its pinned paged-gluon decode scope")
        physical = self.declaration.for_family("unified")
        backend = dict(physical.get("attention_backend") or ())
        if (backend.get("backend") != "atom.model_ops.attentions.aiter_attention.AiterBackend"
                or backend.get("impl") != "atom.model_ops.attention_mha.PagedAttentionImpl"
                or any(backend.get(key) not in (False, "False", "false", 0, "0")
                       for key in ("ATOM_USE_UNIFIED_ATTN", "ATOM_FORCE_ATTN_TRITON"))
                or physical.get("kv_cache_dtype") != "bf16"
                or physical.get("kv_cache_block_size") != 16 or physical.get("sliding_window") != -1):
            raise ValueError("native MHA layout transfer physical backend/KV scope is outside its proof")
        self.base = base
        provider = base
        while provider is not None and not isinstance(provider, ParametricPriceLibrary):
            provider = getattr(provider, "base", None)
        if provider is None:
            raise ValueError("native MHA layout transfer requires an existing bounded family model")
        self.family = provider
        self.handoff_sha256 = handoff_sha256
        self.evidence = dict(handoff_sha256=handoff_sha256, address_proof=handoff["address_proof"],
                             bias_comparison=handoff["bias_comparison"],
                             observed_source_residuals=comparison["comparisons"])
        self.loaded_inputs = base.loaded_inputs + tuple(reader.inputs)
        self.sources = base.sources + [str(handoff_path)]
        self._prices, self.address_shifted = base._prices, base.address_shifted

    def _fallback(self, op, topology, registration, original, memo=None):
        if isinstance(op, PreparedOperator):
            op = op.as_dict()
        if (not native_decode_layout(op) or not topology or topology.get("tp") != 1
                or any(type(v) is not int or v != 1 for v in topology.values())):
            return original
        if getattr(self.family, "launch_charge_seconds", 0) != 0:
            return None, "native MHA layout transfer requires its zero added launch charge"
        declared = self.family.request_attention_scope
        if not isinstance(declared, attention_scope.Declaration):
            try:
                declared = attention_scope.declaration_of(declared, where="native MHA transfer current scope")
            except ValueError:
                return None, "native MHA transfer has no current deployment scope"
        for label in ("unified", "unified.decode"):
            expected = _layout_without_capacity(attention.scoped(op, self.declaration.for_family(label)))
            actual = _layout_without_capacity(attention.scoped(op, declared.for_family(label)))
            if attention._scope_matches(expected, actual) is not None:
                return None, "native MHA layout transfer current backend/KV scope differs"
        # Only this private model query changes. The bounded model still checks
        # context features, treatments, dispatch topology and source support.
        query = dict(op, layouts=[])
        record, why = self.family._modelled(query, original[1], contract_for(op["name"]),
                                            topology, registration, _memo=memo)
        if record is None:
            return None, why
        if not record.get(INTERPOLATED_FLAG) or record.get("interpolation", {}).get("regime") != REGIME:
            return None, "native MHA layout transfer did not reach its bounded decode model"
        return dict(record, native_mha_layout_transfer=dict(self.evidence,
            actual_layouts=op["layouts"], model_lookup_layouts=[], exact_measured_coverage=False,
            timing_equivalence_proven=False, source_refitted=False)), why + "; native V address transfer (modelled)"

    def lookup(self, op, topology=None, registration=None):
        original = self.base.lookup(op, topology, registration)
        return original if original[0] is not None else self._fallback(op, topology, registration, original)

    def _body_lookup(self, op, topology, registration, modelled_memo):
        original = self.base._body_lookup(op, topology, registration, modelled_memo)
        return original if original[0] is not None else self._fallback(op, topology, registration, original, modelled_memo)

    def host_sync_reason(self, op):
        return self.base.host_sync_reason(op)

    def _can_reuse_prepared_lookups(self):
        return self.base._can_reuse_prepared_lookups()

    def _prepared_config_key(self, topology, registration):
        key = self.base._prepared_config_key(topology, registration)
        return None if key is None else (key, self.handoff_sha256)

    def add(self, *args, **kwargs):
        raise ValueError("build source prices before attaching the native MHA layout transfer")

    def describe(self):
        return f"NativeMhaDecodeLayout(modelled TP1 rows1/2/4; exact references first; base={self.base.describe()})"
