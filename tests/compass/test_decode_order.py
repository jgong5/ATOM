"""Order-aware decode retains native padding, grid identity and domain refusal."""
import pytest

from atom.compass.core.cost.families import attention as A
from atom.compass.runtime.batch_spec import BatchSpec

NAME = "unified.decode.paged_gluon_order"
SCOPE = {"attention_backend": "paged_gluon", "sliding_window": -1,
         "num_kv_heads": 4, "compute_units": 80,
         "kv_cache_dtype": "bfloat16", "kv_cache_layout": "NHD",
         "kv_cache_block_size": 16}


def op_for(contexts, bucket=None):
    spec = BatchSpec(kind="decode", query_lens=tuple([1] * len(contexts)),
                     context_lens=tuple(contexts), prompt_lens=tuple([1] * len(contexts)),
                     block_size=16, max_model_len=262144,
                     capture_bucket=bucket or len(contexts), cudagraph_mode="full")
    rows = bucket or len(contexts)
    return {"name": A.UNIFIED,
            "input_shapes": [[rows, 24, 256], None, [rows, 4, 256], [rows, 4, 256]],
            "context": spec.attention_context()}


def features(op):
    return A.features_for(A.REGIMES[NAME], A.structure_of(op), SCOPE)


def test_residue_class_sum_distinguishes_equal_work_orders():
    # With two classes, alternating tile counts1/2/3/4 gives class sums4/6;
    # swapping the middle requests gives3/7. No scheduler claim is involved.
    first = A.Structure((1,) * 4, (255, 511, 767, 1023))
    second = A.Structure((1,) * 4, (255, 767, 511, 1023))
    assert first.cu_class_work(256, 1, 1, 2) == 6
    assert second.cu_class_work(256, 1, 1, 2) == 7
    assert first.split_tiles(256, 1) == second.split_tiles(256, 1) == 10


def test_native_padding_keeps_its_grid_positions_and_adds_no_tile_work():
    op = op_for([4096] * 3, bucket=4)
    structure = A.structure_of(op)
    assert structure.contexts() == (4096, 4096, 4096, 0)
    assert structure.executed_rows == 4
    padded = features(op)
    full = features(op_for([4096] * 4))
    assert not isinstance(padded, A.Refusal)
    assert padded[2] < full[2]


@pytest.mark.parametrize("missing", ["compute_units", "num_kv_heads"])
def test_an_unknown_grid_axis_refuses(missing):
    scope = dict(SCOPE)
    scope.pop(missing)
    out = A.features_for(A.REGIMES[NAME], A.structure_of(op_for([4096] * 4)), scope)
    assert isinstance(out, A.Refusal)
    assert missing in out.reason


def test_partial_context_metadata_does_not_shrink_the_launched_grid():
    op = op_for([4096] * 4)
    op["input_shapes"][0][0] = 8
    out = features(op)
    assert isinstance(out, A.Refusal)
    assert "different decode grids" in out.reason


def test_model_uses_recorded_order_and_keeps_domain_refusal():
    assert A.DECODE_KERNELS["paged_gluon"] == NAME
    balanced = op_for([4096] * 16 + [128] * 16)
    concentrated = op_for([4096, 128] * 16)
    a, b = features(balanced), features(concentrated)
    assert a[2] == b[2] and a[1] < b[1]
    scope = A.scoped(balanced, SCOPE)
    regime = A.REGIMES[NAME]
    fit = A.Fit(regime, regime.features, (), (1e-5, 1e-6, 0.0),
                (1.0,) * 3, 4, 1, 0.0, [a, b], scope)
    model = A.Model()
    model.fits[A._label(NAME, A.scope_key(scope))] = fit
    assert model.price(balanced, SCOPE) < model.price(concentrated, SCOPE)
    outside = model.price(op_for([262144] * 32), SCOPE)
    assert isinstance(outside, A.Refusal)
    assert "outside the measured range" in outside.reason
