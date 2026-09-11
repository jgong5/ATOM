"""Behaviour of the per-family parametric prices.

These are about what the module refuses and how it labels what it answers, not
about whether any particular number is right. A price that is wrong by 3% is a
measurement problem; a price that is an interpolation wearing a measurement's
label is a reporting problem, and it is the one that survives review.
"""

from __future__ import annotations

import json

import pytest

from atom.compass.core.cost.families import (
    INTERPOLATED_SCHEME,
    MeasuredCurve,
    ParametricPriceLibrary,
    Refusal,
    RowSupport,
    aligns,
    contract_for,
    coverage_split,
    infer_rows,
)


def gemm(rows: int, n: int = 5120, k: int = 17408) -> dict:
    """A gemm operator as the graphs record it: (rows, k) x (n, k)."""
    return {
        "name": "aiter::gemm_a16w16",
        "input_shapes": [[rows, k], [n, k]],
        "dtypes": ["bfloat16", "bfloat16"],
        "scalars": [["#2", "None"]],
    }


def triton_norm(rows: int) -> dict:
    """A Triton launch whose grid is rows * 28 -- proportional, not constant."""
    return {
        "name": "triton::_fused_qk_norm_single_kernel",
        "input_shapes": [[rows, 24, 256], [rows, 4, 256]],
        "dtypes": ["bfloat16", "bfloat16"],
        "scalars": [["#7", rows], ["#9", 14336]],
        "launch": [["grid", [rows * 28]]],
    }


def attention(rows: int, context: int, slots: list[int]) -> dict:
    return {
        "name": "aiter::unified_attention_with_output_base",
        "input_shapes": [[rows, 6144], [rows, 1024], [rows, 1024]],
        "dtypes": ["bfloat16", "bfloat16", "bfloat16"],
        "context": [["context_lens", [context] * rows],
                    ["slot_mapping", slots],
                    ["max_seqlen_k", context],
                    ["state", "prefill_native"]],
        "scalars": [["#5", "language_model.model.layers.11.self_attn"]],
    }


# -- matching two widths of one operator --------------------------------


def test_architectural_constant_divisible_by_the_row_count_is_not_scaled():
    """17408 is an exact multiple of 32, and it is not a function of the batch.

    A template abstracted from the 32-row measurement alone cannot tell the
    two apart. Comparing the two widths against each other can: 17408 equals
    17408, so equality settles it before divisibility is ever consulted.
    """
    assert aligns(gemm(32), 32, gemm(16384), 16384)
    assert infer_rows(gemm(16384), gemm(32), 32) == 16384
    assert infer_rows(gemm(32), gemm(16384), 16384) == 32


def test_different_weight_shapes_are_not_the_same_operator():
    """Two gemms in one step differ in K, and no row count makes them one."""
    assert not aligns(gemm(32, n=5120, k=17408), 32,
                      gemm(32, n=14336, k=5120), 32)
    assert infer_rows(gemm(32, n=14336, k=5120), gemm(32, n=5120, k=17408),
                      32) is None


def test_a_proportional_launch_grid_tracks_the_row_count():
    assert aligns(triton_norm(32), 32, triton_norm(16384), 16384)
    assert infer_rows(triton_norm(4096), triton_norm(32), 32) == 4096


def test_a_row_count_must_satisfy_every_position_not_just_one():
    """Positions that imply different widths cannot both be believed.

    Here the operand shapes say 64 rows and the launch grid says 128. Solving
    from either one alone gives an answer; putting the answer back through
    every position is what rejects it.
    """
    inconsistent = triton_norm(64)
    inconsistent["launch"] = [["grid", [128 * 28]]]
    assert infer_rows(inconsistent, triton_norm(32), 32) is None


def test_a_scalar_equal_to_the_measured_width_is_read_as_the_constant_it_is():
    """Equality is tested first, so a coincidence is never read as scaling.

    A scalar that holds 32 in both operators is a constant, whatever the row
    counts are -- including when one of those row counts is also 32.
    """
    wide = triton_norm(4096)
    wide["scalars"] = [["#7", 4096], ["#9", 14336]]
    assert infer_rows(wide, triton_norm(32), 32) == 4096


# -- what the support region will and will not answer --------------------


def curve_at(points: dict[int, float]) -> MeasuredCurve:
    curve = MeasuredCurve(template_key="t", family="aiter::gemm_a16w16")
    for rows, seconds in points.items():
        curve.add(rows, seconds, f"file_{rows}.json", ("k",))
    return curve


def test_a_measured_width_is_reported_as_measured():
    support = RowSupport(curve_at({32: 1e-4, 64: 2e-4}))
    price = support.price(32)
    assert not isinstance(price, Refusal)
    assert price.basis == "measured"
    assert price.seconds == pytest.approx(1e-4)


def test_a_width_beyond_the_measured_range_is_refused_not_extrapolated():
    support = RowSupport(curve_at({32: 1e-4, 64: 2e-4}))
    refusal = support.price(128)
    assert isinstance(refusal, Refusal)
    assert "outside the measured range" in refusal.reason
    assert "extrapolation" in refusal.reason


def test_a_width_inside_an_unsampled_gap_is_refused():
    """Measuring 32 and 16384 does not make 512 supported.

    A coordinate-wise bound would say it does. The gap rule is what stops the
    range being mistaken for coverage of the range.
    """
    support = RowSupport(curve_at({32: 1e-4, 16384: 5e-2}), max_gap_ratio=2.0)
    refusal = support.price(512)
    assert isinstance(refusal, Refusal)
    assert "gap" in refusal.reason
    assert "32..16384" in refusal.reason


def test_a_width_inside_a_densely_sampled_gap_is_interpolated():
    support = RowSupport(curve_at({32: 1e-4, 64: 2e-4}), max_gap_ratio=2.0)
    price = support.price(48)
    assert not isinstance(price, Refusal)
    assert price.basis == "interpolated"
    assert 1e-4 < price.seconds < 2e-4
    assert price.uncertainty > 0.0
    assert set(price.sources) == {"file_32.json", "file_64.json"}


def test_interpolation_uncertainty_grows_with_the_gap():
    narrow = RowSupport(curve_at({100: 1e-4, 150: 2e-4}), max_gap_ratio=2.0)
    wide = RowSupport(curve_at({100: 1e-4, 199: 2e-4}), max_gap_ratio=2.0)
    assert wide.price(120).uncertainty > narrow.price(120).uncertainty


# -- the fallback fires behind one refusal and no other -------------------


class _Stub(ParametricPriceLibrary):
    """A library whose exact lookup refuses for a reason we choose."""

    def __init__(self, reason):
        super().__init__()
        self._reason = reason

    def lookup(self, op, topology=None, registration=None):
        record, detail = (None, self._reason)
        if record is not None or detail != "no entry for this signature":
            return record, detail
        return self._parametric(op, detail)


def test_a_layout_refusal_is_returned_untouched():
    """A measurement of this operator exists and is of a different memory.

    That is a finding. Answering it from a curve would replace a known-bad
    case with a number that looks like a price.
    """
    reason = "priced under a different operand layout (dense vs strided)"
    record, detail = _Stub(reason).lookup(gemm(48))
    assert record is None
    assert detail == reason


def test_a_pricing_time_refusal_is_returned_untouched():
    reason = "refused when priced: stride past its tensors"
    record, detail = _Stub(reason).lookup(gemm(48))
    assert record is None
    assert detail == reason


# -- the ragged family refuses and says what is missing -------------------


def test_attention_refuses_and_names_the_unmeasured_component():
    library = ParametricPriceLibrary()
    record, detail = library.lookup(attention(32, 1151, list(range(32))))
    assert record is None
    assert "ragged" in detail
    assert "nobody has measured" in detail


def test_attention_contract_marks_allocator_state_unmeasured():
    contract = contract_for("aiter::unified_attention_with_output_base")
    assert "slot_mapping" in contract.unmeasured_nuisances
    assert "positions" in contract.unmeasured_nuisances
    # The layer index is the one nuisance a measurement does cover.
    layer = next(n for n in contract.nuisances if n.component == "layer")
    assert layer.measured and layer.spread > 0.0


# -- measured and interpolated must stay apart ---------------------------


def _library_with(tmp_path, widths: dict[int, float]) -> ParametricPriceLibrary:
    """A library holding one gemm priced at each of several widths."""
    from atom.compass.runtime.microbench import signature_of

    library = ParametricPriceLibrary(max_gap_ratio=2.0)
    for rows, seconds in widths.items():
        op = gemm(rows)
        graph = {"ops": [op],
                 "provenance": {"execution": {"body_rows_traced": rows}}}
        prices = {"prices": {signature_of(op): {
            "seconds": seconds, "kernels": {"k": seconds},
            "occurrences": 1, "name": op["name"]}}}
        gpath = tmp_path / f"g{rows}.json"
        ppath = tmp_path / f"p{rows}.json"
        gpath.write_text(json.dumps(graph))
        ppath.write_text(json.dumps(prices))
        library.add(str(ppath), str(gpath))
    return library


def test_an_interpolated_record_says_so_and_names_its_measurements(tmp_path):
    library = _library_with(tmp_path, {32: 1e-4, 64: 2e-4})
    record, source = library.lookup(gemm(48))
    assert record is not None, source
    assert record["interpolated"] is True
    assert source.startswith(INTERPOLATED_SCHEME)
    block = record["interpolation"]
    assert block["basis"] == "interpolated"
    assert block["measured_rows"] == [32, 64]
    assert len(block["measured_sources"]) == 2
    assert block["uncertainty"] > 0.0


def test_an_exact_width_is_answered_by_the_measurement_itself(tmp_path):
    """The exact key still wins, and its record is the measured one untouched.

    A measured record carries no ``interpolated`` flag at all -- the flag is
    something this module adds to what it derives, not a field it back-fills
    onto measurements -- so absence is the measured case and callers must read
    it that way.
    """
    library = _library_with(tmp_path, {32: 1e-4, 64: 2e-4})
    record, source = library.lookup(gemm(64))
    assert not record.get("interpolated")
    assert not source.startswith(INTERPOLATED_SCHEME)
    assert record["seconds"] == pytest.approx(2e-4)


def test_coverage_split_separates_measured_from_interpolated(tmp_path):
    library = _library_with(tmp_path, {32: 1e-4, 64: 2e-4})
    graph = {"ops": [gemm(32), gemm(48)], "provenance": {}}
    split = coverage_split(library, graph)
    assert split["operators"] == 2
    assert split["measured"] == 1
    assert split["interpolated"] == 1
    assert split["refused"] == 0
    assert split["complete_measured"] is False


def test_a_step_priced_entirely_by_interpolation_is_not_complete(tmp_path):
    """The property lead's PRICING_API.md asks for, stated as a test.

    ``PriceLibrary.body`` returns a ``Coverage`` whose ``complete`` is
    ``priced == operators``. An interpolated record increments ``priced`` like
    any other, so a step summed entirely from interpolations reports itself
    complete and the distinction is gone from every downstream report.

    This test is expected to fail until ``Coverage`` in
    ``atom/compass/core/cost/library.py`` carries an ``interpolated`` count of
    its own and excludes it from ``complete``. That file is the lead's; this is
    the failing test requested against it.
    """
    library = _library_with(tmp_path, {32: 1e-4, 64: 2e-4})
    graph = {"ops": [gemm(48)], "provenance": {}}

    seconds, coverage, _launches = library.body(graph)

    assert seconds > 0.0
    split = coverage_split(library, graph)
    assert split["interpolated"] == 1 and split["measured"] == 0

    assert hasattr(coverage, "interpolated"), (
        "Coverage does not distinguish an interpolated price from a measured "
        "one, so a step summed entirely from interpolations is indistinguishable "
        "from one summed from measurements")
    assert coverage.interpolated == 1
    assert coverage.measured == 0
    # Not an acceptance failure: a validated interpolation may well be enough
    # for complete predictive coverage. What must survive into the report is
    # that this step was covered by a model and not by a measurement, so the
    # two completeness questions are asked separately.
    assert coverage.complete_accounted
    assert not coverage.complete_measured


def test_the_four_states_of_an_operator_stay_distinguishable(tmp_path):
    """Measured, interpolated, known zero-work and refused are four answers.

    Collapsing any pair of them loses something a reader needs: a modelled
    price read as a measurement overstates the evidence, and a zero-work case
    read as a refusal understates the coverage.
    """
    library = _library_with(tmp_path, {32: 1e-4, 64: 2e-4})
    graph = {"ops": [gemm(32), gemm(48), gemm(4096)], "provenance": {}}
    split = coverage_split(library, graph)
    assert split["measured"] == 1          # exact key at a measured width
    assert split["interpolated"] == 1      # inside a densely sampled gap
    assert split["refused"] == 1           # beyond the measured range
    assert split["zero_work"] == 0
    assert split["accounted"] == 2
    assert not split["complete_accounted"]
    assert not split["complete_measured"]
