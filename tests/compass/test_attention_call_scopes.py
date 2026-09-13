"""A measured decode kernel scope must not replace a prefill backend scope."""
import hashlib
import json

from atom.compass.core.cost.families.attention import UNIFIED
from atom.compass.core.cost.families.attention_scope import declaration_of
from atom.compass.runtime.source_oracle import _price_library, _DEFAULT_GAP_RATIO

PREFILL = {"attention_backend": [["backend", "AiterBackend"],
                                  ["impl", "PagedAttentionImpl"]],
           "kv_cache_dtype": "bf16", "kv_cache_layout": "declared prefill view",
           "kv_cache_block_size": 16, "sliding_window": -1}
DECODE = dict(PREFILL, attention_backend="paged_gluon", num_kv_heads=4,
              compute_units=80, kv_cache_layout="declared decode view")


def op(prefill, cached=False):
    return {"name": UNIFIED,
            "context": [["is_prefill", prefill], ["has_cached", cached],
                        ["cu_seqlens_q", [0, 1]], ["context_lens", [32]]],
            "input_shapes": [[1, 24, 256]]}


def test_native_branch_selects_complete_scope_without_overwriting_prefill():
    declared = declaration_of({"attention_scope": {
        "unified": PREFILL, "unified.decode": DECODE}}, where="combined")
    assert declared.for_op(op(False)) == DECODE
    assert declared.for_op(op(True)) == declared.for_family("unified")
    assert declared.for_op(op(True, True)) == declared.for_family("unified")
    assert declared.for_op(op(None)) == declared.for_family("unified")
    assert "compute_units" not in declared.for_op(op(True))


def test_cold_and_cached_native_scopes_can_be_declared_independently():
    cold = dict(PREFILL, kv_cache_layout="cold view")
    cached = dict(PREFILL, kv_cache_layout="cached view")
    declared = declaration_of({"unified.decode": DECODE,
                               "unified.prefill.cold": cold,
                               "unified.prefill.cached": cached}, where="branches")
    assert declared.for_op(op(True))["kv_cache_layout"] == "cold view"
    assert declared.for_op(op(True, True))["kv_cache_layout"] == "cached view"
    assert declared.for_op(op(False))["attention_backend"] == "paged_gluon"
    assert declared.for_op(op(None)) == {}


def test_combined_file_reaches_factory_and_attests_original_bytes(tmp_path):
    path = tmp_path / "combined.json"
    path.write_text(json.dumps({"attention_scope": {
        "unified": PREFILL, "unified.decode": DECODE}}))
    raw = path.read_bytes()
    library = _price_library([], _DEFAULT_GAP_RATIO, attention_scope=str(path))
    assert library._declared_scope(op(False))["compute_units"] == 80
    assert isinstance(library._declared_scope(op(True))["attention_backend"], tuple)
    loaded = [r for r in library.loaded_inputs if r.role == "oracle.attention_scope"]
    assert len(loaded) == 1
    assert loaded[0].sha256 == hashlib.sha256(raw).hexdigest()
