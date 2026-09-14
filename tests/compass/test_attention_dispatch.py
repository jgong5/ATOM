"""The dispatch-group model must preserve the measured ordering mechanism."""
import pytest

from atom.compass.core.cost.families.attention import (
    REGIMES, Refusal, features_for, regime_of, structure_of,
)


CONTEXTS = (190001, 56265, 38802, 62299, 44836, 68333, 50870, 33407,
            56904, 39441, 62938, 45475, 68972, 51509, 34046, 57543,
            188417, 63577, 46114, 69611, 52148, 34685, 58182, 40719,
            64216, 46753, 70250, 52787, 35324, 58821, 41358, 64855)
SCOPE = {"kv_cache_dtype": "bfloat16", "kv_cache_layout": "SHUFFLE",
         "kv_cache_block_size": 16, "sliding_window": -1,
         "attention_backend": "paged_gluon", "num_kv_heads": 4,
         "compute_units": 80,
         "decode_dispatch_topology": "gfx942-spx-4xcc-80cu"}


def op(contexts):
    return {"name": "aiter::unified_attention_with_output_base",
            "input_shapes": [[len(contexts), 6144]], "dtypes": ["bfloat16"],
            "context": [["cu_seqlens_q", list(range(len(contexts) + 1))],
                        ["context_lens", list(contexts)],
                        ["is_prefill", False], ["has_cached", True],
                        ["capture_bucket", len(contexts)]]}


def test_measured_permutation_uses_local_cu_balance():
    # Source dispatch probe: all 4096 workgroups stay in linear-ID mod16
    # groups of five active CUs. Recorded maximum assigned tile loads are
    # 1479–1482 for A and exactly 869 for B; neither is a fitted duration.
    a = op(CONTEXTS)
    permuted = list(CONTEXTS)
    permuted[1], permuted[16] = permuted[16], permuted[1]
    b = op(permuted)
    regime = regime_of(a, scope=SCOPE)
    assert regime.name == "unified.decode.paged_gluon_dispatch"
    av = dict(zip(regime.features, features_for(regime, structure_of(a), SCOPE)))
    bv = dict(zip(regime.features, features_for(regime, structure_of(b), SCOPE)))
    assert av['dispatch_max'] == 1479
    assert bv['dispatch_max'] == 869
    assert av['cu_tiles'] == bv['cu_tiles']
    # Pinned old fits retain their original descriptor and feature meanings.
    old = REGIMES['unified.decode.paged_gluon_order']
    assert features_for(old, structure_of(b), SCOPE)[1] == 988


@pytest.mark.parametrize('change', [
    {'decode_dispatch_topology': None},
    {'decode_dispatch_topology': 'unobserved'},
    {'compute_units': 160},
])
def test_dispatch_feature_requires_the_supported_hardware_scope(change):
    scope = {**SCOPE, **change}
    regime = REGIMES['unified.decode.paged_gluon_dispatch']
    assert isinstance(features_for(regime, structure_of(op(CONTEXTS)), scope), Refusal)


def test_missing_dispatch_declaration_preserves_existing_regime():
    scope = {key: value for key, value in SCOPE.items()
             if key != 'decode_dispatch_topology'}
    assert regime_of(op(CONTEXTS), scope=scope).name == 'unified.decode.paged_gluon_order'
