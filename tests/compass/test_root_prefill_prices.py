"""Bounded prefill transfer checks the semantics erased by generic cost keys."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from atom.compass.core.cost.base import StepShape
from atom.compass.core.cost.library import PriceLibrary
from atom.compass.core.cost.low_query import GEMM, _key
from atom.compass.core.cost.root_prefill import ExactPrefillRegions, RootPrefillPrices, _cold_gdn_identity
from .test_cached_q16_prices import gdn, mha


def cold(layer=0, slot=1):
    op = gdn(layer=layer, query=32, read=slot, write=slot)
    values = dict(op["context"])
    values.update(has_initial_state=[[0], "bool"], num_spec_decodes=0, num_spec_decode_tokens=0)
    op["context"] = list(values.items())
    return op


def adapter():
    result = RootPrefillPrices.__new__(RootPrefillPrices)
    PriceLibrary.__init__(result)
    result.base = PriceLibrary()
    result.handoff_sha256 = "a" * 64
    result._exact, result._cold_gdn, result._warm_gdn, result._bounded_mha = {}, {}, {}, {}
    return result


def test_cold_gdn_transfers_only_valid_equal_slots_and_gdn_layers():
    library = adapter()
    layer, key = _cold_gdn_identity(cold())
    library._cold_gdn[key] = {"seconds": .001, "source": "cold-source"}
    for layer in (0, 32, 62):
        for slot in (0, 1, 31):
            assert library.lookup(cold(layer, slot))[0]["seconds"] == .001
    assert _cold_gdn_identity(cold(layer=3)) is None


@pytest.mark.parametrize("field,value", [
    ("has_initial_state", [[1], "bool"]), ("has_initial_state", None),
    ("has_initial_state", [[0], "int32"]),
    ("non_spec_state_indices_in_tensor", [[-1], "int32"]),
    ("non_spec_state_indices_in_tensor", [[2], "int32"]),
    ("non_spec_state_indices_tensor", [[32], "int32"]),
    ("num_spec_decodes", 1), ("num_spec_decode_tokens", 1),
    ("num_prefills", 2), ("non_spec_query_start_loc", [[0, 31], "int32"]),
    ("replayssm", True),
])
def test_cold_gdn_refuses_before_alias_address_normalization(field, value):
    op = cold()
    context = dict(op["context"])
    context[field] = value
    op["context"] = list(context.items())
    assert _cold_gdn_identity(op) is None


def test_exact_mha_does_not_extend_query_history_or_tp():
    library = adapter()
    op = mha(history=2993, query=1)
    library._exact[_key(op)] = {"seconds": .001, "source": "exact-mha"}
    assert library.lookup(mha(history=2993, query=1, layer=63))[0]["seconds"] == .001
    for other in (mha(history=2994, query=1), mha(history=2993, query=2)):
        assert library.lookup(other) == library.base.lookup(other)
    assert library.lookup(op, {"tp": 2}) == library.base.lookup(op, {"tp": 2})


def test_m9_replacement_intercepts_public_body_and_prepared_configuration():
    library = adapter()
    op = {"name": GEMM, "input_shapes": [[9, 17408], [5120, 17408]],
          "dtypes": ["bfloat16", "bfloat16"], "output_shapes": [[9, 5120]],
          "output_dtypes": ["bfloat16"]}
    library._exact[_key(op)] = {"seconds": .003, "source": "mandatory-new-m9"}
    for call in (lambda: library.lookup(op), lambda: library._body_lookup(op, None, None, {})):
        assert call()[0]["seconds"] == .003
    library.base._prepared_config_key = lambda *args: ("base",)
    assert library._prepared_config_key(None, None) == (("base",), "a" * 64)
    library.handoff_sha256 = "b" * 64
    assert library._prepared_config_key(None, None) == (("base",), "b" * 64)


def test_exact_regions_preserve_structural_zero_and_delegate_outside_domain():
    base = SimpleNamespace(topologies=(1,), refusal=lambda s: "base refusal",
                           breakdown=lambda s: {"<base>": .9}, describe=lambda: "base")
    regions = ExactPrefillRegions(base, ((32, 32, False, .002, 0.), (2, 498, True, .001, .0001)), "a" * 64)
    shape = StepShape((32,), (32,), num_prefill_tokens=32, compiled=True, produces_output=False)
    assert regions.refusal(shape) is None
    assert regions.breakdown(shape) == {"<prepare>": .002, "<postprocess>": 0.}
    for other in (replace(shape, produces_output=True), replace(shape, compiled=False),
                  replace(shape, context_lens=(48,)), replace(shape, topology={"tp": 2}),
                  replace(shape, capture_bucket=1), replace(shape, num_prefill_tokens=0)):
        assert regions.refusal(other) == "base refusal"
        assert regions.breakdown(other) == {"<base>": .9}
    with pytest.raises(ValueError, match="uncertainty"):
        regions.band(shape)
