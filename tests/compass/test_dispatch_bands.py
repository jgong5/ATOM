"""The dispatch probe as evidence: a repeated tile name is not one curve.

The case these tests are written from is real and is the one that would have
been priced wrong. `MT256x192x64_MI32x32x1` serves 8256..9216 and again
14400..15360 on this source, with four other tiles in between, so a library
holding measurements at 9216 and 14400 sees two endpoints that agree and, on
the endpoint test alone, would interpolate every row between them.
"""

import json

import pytest

from atom.compass.core.cost.families.dispatch_bands import (
    BandEvidence,
    geometry_key,
    load_band_map,
    static_shapes,
)
from atom.compass.core.cost.families.support import MeasuredCurve, RowSupport

A = "MT256x192x64_MI32x32x1"
B = "MT256x224x64_MI16x16x1"
C = "MT256x256x32_MI32x32x1"


def _probe(path, rows_kernels, weight=(5120, 6144)):
    """A probe file in the shape `dispatch_probe.py` writes."""
    path.write_text(json.dumps({
        "graph": ["b27_tp1_r0_pref_body_16384.json"],
        "rows": [r for r, _ in rows_kernels],
        "geometries": [{
            "weight": list(weight),
            "dtypes": ["bfloat16", "bfloat16"],
            "points": [{"rows": r, "kernels": [k],
                        # Present in the real files and deliberately unread:
                        # no timing in a probe is a price.
                        "seconds": {k: 1.0}}
                       for r, k in rows_kernels],
        }],
    }))
    return path


def _curve(points, bands=None):
    curve = MeasuredCurve(template_key="t", family="aiter::gemm_a16w16",
                          bands=bands)
    for rows, seconds, kernels in points:
        curve.add(rows, seconds, f"p{rows}.json", kernels)
    return curve


def test_band_starts_are_the_first_row_known_to_run_a_new_kernel(tmp_path):
    probe = _probe(tmp_path / "bands.json",
                   [(8192, A), (8256, A), (8320, B), (8384, B), (8448, A)])
    band_map = load_band_map([probe], "aiter::gemm_a16w16")
    evidence = band_map.get(geometry_key(
        "aiter::gemm_a16w16", ["bfloat16", "bfloat16"], [(5120, 6144)]))
    assert evidence.starts == (8320, 8448)
    assert (evidence.low, evidence.high, evidence.step) == (8192, 8448, 64)


def test_shards_merge_into_one_span(tmp_path):
    """The probe ran in shards, and a switch can straddle two of them."""
    lower = _probe(tmp_path / "lower.json", [(8192, A), (8256, A)])
    upper = _probe(tmp_path / "upper.json", [(8320, B), (8384, B)])
    band_map = load_band_map([lower], "aiter::gemm_a16w16")
    for key in (other := load_band_map([upper], "aiter::gemm_a16w16")).keys():
        band_map.add(key, other.get(key))
    evidence = band_map.get(geometry_key(
        "aiter::gemm_a16w16", ["bfloat16", "bfloat16"], [(5120, 6144)]))
    assert (evidence.low, evidence.high) == (8192, 8384)
    # Per shard neither file witnesses a switch; only the union does, and the
    # union is what the bracket is judged against.
    assert evidence.starts == (8320,)


def test_repeated_tile_name_across_disjoint_bands_is_refused():
    """The case that motivated this: same name at both ends, four tiles inside."""
    points = [(9216, 1.0e-3, (A,)), (14400, 1.6e-3, (A,))]
    plain = RowSupport(_curve(points), max_gap_ratio=2.0)
    # Without the probe the endpoint test sees one kernel and prices it.
    answer = plain.price(12288)
    assert getattr(answer, "basis", None) == "interpolated"

    guarded = RowSupport(_curve(points, bands=BandEvidence(
        starts=(9280, 10304, 11328, 12352, 14400), low=8192, high=16384,
        step=64)), max_gap_ratio=2.0)
    refusal = guarded.price(12288)
    assert refusal.component == "kernel_switch"
    assert "9280" in refusal.reason and "12352" in refusal.reason


def test_the_bracketing_pair_still_prices_the_row_it_was_measured_for():
    """14400..15360 is one band, so 14592 is answerable and must stay so."""
    support = RowSupport(
        _curve([(14400, 1.6e-3, (A,)), (15360, 1.7e-3, (A,))],
               bands=BandEvidence(starts=(9280, 10304, 11328, 12352, 14400,
                                          15424),
                                  low=8192, high=16384, step=64)),
        max_gap_ratio=2.0)
    price = support.price(14592)
    assert price.basis == "interpolated"
    assert price.kernels == (A,)


def test_a_start_at_the_upper_end_still_refuses():
    """The row below `hi` ran another tile, so reaching down from it crosses."""
    support = RowSupport(
        _curve([(11328, 1.2e-3, (A,)), (12352, 1.4e-3, (B,))],
               bands=BandEvidence(starts=(12352,), low=8192, high=16384,
                                  step=64)),
        max_gap_ratio=2.0)
    assert support.price(12288).component == "kernel_switch"


def test_no_probe_is_silence_not_a_clean_bracket():
    support = RowSupport(_curve([(9216, 1.0e-3, (A,)), (14400, 1.6e-3, (A,))]),
                         max_gap_ratio=2.0)
    price = support.price(12288)
    assert price.basis == "interpolated"
    assert "dispatch probe" not in support.describe()


def test_unrecorded_kernels_do_not_invent_a_band(tmp_path):
    probe = tmp_path / "bands.json"
    probe.write_text(json.dumps({"geometries": [{
        "weight": [5120, 6144], "dtypes": ["bfloat16", "bfloat16"],
        "points": [{"rows": 8192, "kernels": [A]},
                   {"rows": 8256, "kernels": []},
                   {"rows": 8320, "kernels": [A]}]}]}))
    evidence = load_band_map([probe], "aiter::gemm_a16w16").get(geometry_key(
        "aiter::gemm_a16w16", ["bfloat16", "bfloat16"], [(5120, 6144)]))
    assert evidence.starts == ()
    assert (evidence.low, evidence.high) == (8192, 8320)


def test_static_shapes_drops_every_operand_carrying_the_row_count():
    op = {"name": "aiter::gemm_a16w16",
          "input_shapes": [[14592, 5120], [5120, 6144]],
          "dtypes": ["bfloat16", "bfloat16"]}
    assert static_shapes(op, 14592) == ((5120, 6144),)


def test_describe_names_the_probe_and_the_brackets_it_rules_out():
    support = RowSupport(
        _curve([(9216, 1.0e-3, (A,)), (14400, 1.6e-3, (A,))],
               bands=BandEvidence(starts=(9280, 12352), low=8192, high=16384,
                                  step=64)),
        max_gap_ratio=2.0)
    text = support.describe()
    assert "dispatch probe covers 8192..16384" in text
    assert "9216->14400" in text


@pytest.mark.parametrize("rows", [9216, 14400])
def test_a_measured_row_is_never_refused_by_a_band(rows):
    """Bands constrain interpolation only; a reading stands on its own."""
    support = RowSupport(
        _curve([(9216, 1.0e-3, (A,)), (14400, 1.6e-3, (C,))],
               bands=BandEvidence(starts=(9280, 12352), low=8192, high=16384,
                                  step=64)),
        max_gap_ratio=2.0)
    assert support.price(rows).basis == "measured"
