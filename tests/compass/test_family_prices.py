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


# -- a bracket can be close enough and still be two curves ----------------

#: The decode rungs the prefill campaign interpolated from, in seconds, and the
#: tile the library served each with. From CC's head delivery
#: (`results_prefill/prefill_gemm_validation.json`, campaign val_commit_3e04c166
#: on MI308X): 16 rows is the last point served by MT64x16x256 and 32 is served
#: by MT128x32x128, with the switch itself at 20.
PREFILL_RUNGS = {8: 0.0009640, 16: 0.0009707, 32: 0.0011603}
SMALL_TILE = ("aiter::gemm_a16w16_MT64x16x256",)
LARGE_TILE = ("aiter::gemm_a16w16_MT128x32x128",)


def switched_curve(kernels_by_rows: dict[int, tuple]) -> MeasuredCurve:
    curve = MeasuredCurve(template_key="t", family="aiter::gemm_a16w16")
    for rows, kernels in kernels_by_rows.items():
        curve.add(rows, PREFILL_RUNGS[rows], f"file_{rows}.json", kernels)
    return curve


def test_a_bracket_that_crosses_a_kernel_switch_is_refused():
    """16 to 32 is a ratio of exactly 2.0, so the gap rule lets it through.

    It should not go through. The library serves 16 rows with one tile and 32
    with another, and pricing 20 or 24 off a line between them missed the
    measured points by 12.21% and 8.23%.
    """
    support = RowSupport(switched_curve({16: SMALL_TILE, 32: LARGE_TILE}),
                         max_gap_ratio=2.0)
    for rows in (20, 24):
        refusal = support.price(rows)
        assert isinstance(refusal, Refusal), rows
        assert refusal.component == "kernel_switch"
        assert "MT64x16x256" in refusal.reason
        assert "MT128x32x128" in refusal.reason


def test_the_gap_rule_on_its_own_would_have_allowed_that_bracket():
    """The same two row counts, same times, one tile: priced.

    So the refusal above is about kernel identity and not about distance --
    which is the whole point, because the distance test passes here.
    """
    support = RowSupport(switched_curve({16: SMALL_TILE, 32: SMALL_TILE}),
                         max_gap_ratio=2.0)
    price = support.price(20)
    assert not isinstance(price, Refusal)
    assert price.basis == "interpolated"


def test_a_switch_outside_the_bracket_is_not_this_bracket_s_problem():
    # 8 and 16 are both on the small tile; the switch at 32 is further up the
    # ladder and says nothing about a 12-row price.
    support = RowSupport(
        switched_curve({8: SMALL_TILE, 16: SMALL_TILE, 32: LARGE_TILE}),
        max_gap_ratio=2.0)
    price = support.price(12)
    assert not isinstance(price, Refusal)
    assert price.kernels == SMALL_TILE


def test_an_unlabelled_bracket_is_priced_and_says_it_was_not_checked():
    """An absence is not a switch, and it is not a confirmation either.

    A producer that recorded no kernel names leaves the identity unknown. The
    price still stands -- refusing on an absence would discard every older
    artifact -- but it carries that it was never checked.
    """
    support = RowSupport(switched_curve({16: (), 32: LARGE_TILE}),
                         max_gap_ratio=2.0)
    price = support.price(20)
    assert not isinstance(price, Refusal)
    assert "unchecked" in price.detail


def test_a_checked_bracket_does_not_claim_to_be_unchecked():
    support = RowSupport(switched_curve({16: SMALL_TILE, 32: SMALL_TILE}),
                         max_gap_ratio=2.0)
    assert "unchecked" not in support.price(20).detail


def test_describe_names_the_switch_so_an_acquisition_can_see_it():
    support = RowSupport(switched_curve({16: SMALL_TILE, 32: LARGE_TILE}),
                         max_gap_ratio=2.0)
    said = support.describe()
    assert "kernel switches inside a bracket: 16->32" in said
    # And not as a gap: the gap rule has no objection to this ladder.
    assert "gaps too wide" not in said


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
    """An empty library refuses, and the refusal has to be actionable.

    What matters is not the wording but that a reader can tell this apart
    from an ordinary unmeasured width: the family is named, the statement is
    that nothing was measured rather than that this call is out of support,
    and the reason a row count cannot stand in is given. Asserting the
    sentence verbatim makes every rewording a failure and every loss of one
    of those three a pass.
    """
    library = ParametricPriceLibrary()
    op = attention(32, 1151, list(range(32)))
    record, detail = library.lookup(op)
    assert record is None

    lowered = detail.lower()
    # It names the family this is, ...
    assert op["name"].split("::")[-1].split("_with")[0] in lowered \
        or "attention" in lowered
    # ... says the absence is of evidence, not of support -- there is no law
    # here to be outside of, ...
    assert "measured" in lowered
    assert "outside" not in lowered and "out of support" not in lowered
    # ... and says why no row count stands in for one, which is the thing a
    # reader would otherwise try next.
    assert "batch" in lowered or "row" in lowered


def test_attention_contract_marks_allocator_state_unmeasured():
    contract = contract_for("aiter::unified_attention_with_output_base")
    assert "slot_mapping" in contract.unmeasured_nuisances
    assert "positions" in contract.unmeasured_nuisances
    # The layer index is the one nuisance a measurement does cover.
    layer = next(n for n in contract.nuisances if n.component == "layer")
    assert layer.measured and layer.spread > 0.0


# -- measured and interpolated must stay apart ---------------------------


def _library_with(tmp_path, widths: dict[int, float],
                  kernels: dict | None = None) -> ParametricPriceLibrary:
    """A library holding one gemm priced at each of several widths.

    ``kernels`` names the kernel each width was served by, where a test cares.
    Everything else gets one name, so every bracket is on a single curve.
    """
    from atom.compass.runtime.microbench import signature_of

    library = ParametricPriceLibrary(max_gap_ratio=2.0)
    for rows, seconds in widths.items():
        op = gemm(rows)
        graph = {"ops": [op],
                 "provenance": {"execution": {"body_rows_traced": rows}}}
        kernel = (kernels or {}).get(rows, "k")
        prices = {"prices": {signature_of(op): {
            "seconds": seconds, "kernels": {kernel: seconds},
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


def test_a_step_priced_entirely_by_interpolation_is_complete_but_not_measured(
        tmp_path):
    """The two completeness questions are different questions.

    ``complete`` asks whether anything was refused -- whether every operator
    has a predicted cost at all. A validated in-support interpolation counts,
    and it has to: predictive coverage is the claim being made, and a rule that
    made any fitted step incomplete forever would put that claim out of reach.

    ``complete_measured`` asks the stricter thing: whether none of it was
    fitted. A step summed entirely from interpolations answers yes to the first
    and no to the second, and a report that cannot tell them apart overstates
    its evidence.

    Named against the seam in ``agent_scratch/COVERAGE_SEAM.md``; the counts are
    asserted explicitly so neither is recoverable only by subtraction.
    """
    library = _library_with(tmp_path, {32: 1e-4, 64: 2e-4})
    graph = {"ops": [gemm(48)], "provenance": {}}

    seconds, coverage, _launches = library.body(graph)

    assert seconds > 0.0
    split = coverage_split(library, graph)
    assert split["interpolated"] == 1 and split["measured"] == 0

    assert coverage.interpolated == 1
    assert coverage.measured == 0
    assert coverage.zero_work == 0
    assert coverage.priced == 1
    assert coverage.complete
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


# -- the kernel switch has to survive the whole library path --------------

#: Both tiles, the two rung times CC measured either side of the switch, and
#: the row count whose interpolation the delivery reports as 12.21% high.
SWITCH_WIDTHS = {16: 9.707e-4, 32: 1.1603e-3}
SWITCH_KERNELS = {16: "gemm_MT64x16x256", 32: "gemm_MT128x32x128"}


def test_the_library_refuses_a_row_count_across_a_tile_switch(tmp_path):
    """A refusal in the support region has to arrive as a refusal here.

    The adapter is where a `Refusal` becomes `(None, reason)`, and a reason
    that lost the kernel names would leave a reader to rediscover why 20 rows
    is not answerable from 16 and 32.
    """
    library = _library_with(tmp_path, SWITCH_WIDTHS, kernels=SWITCH_KERNELS)
    record, detail = library.lookup(gemm(20))
    assert record is None
    assert "same kernel" in detail
    assert "MT64x16x256" in detail and "MT128x32x128" in detail
    assert "measured at [16, 32]" in detail


def test_the_same_ladder_on_one_tile_still_answers(tmp_path):
    # The refusal above is the switch and not the ladder: identical row counts
    # and identical times, one tile, and 20 rows is priced.
    library = _library_with(tmp_path, SWITCH_WIDTHS)
    record, source = library.lookup(gemm(20))
    assert record is not None, source
    assert record["interpolated"] is True


def test_a_switched_bracket_is_counted_refused_and_not_interpolated(tmp_path):
    """It lands in the refused column, with its reason, not the priced one.

    The standing rule is that the four states stay apart and every refusal
    keeps its provenance. A switch that was counted as coverage would be the
    12.21% error reported as a prediction.
    """
    library = _library_with(tmp_path, SWITCH_WIDTHS, kernels=SWITCH_KERNELS)
    graph = {"ops": [gemm(16), gemm(20)], "provenance": {}}

    split = coverage_split(library, graph)
    assert split["measured"] == 1
    assert split["interpolated"] == 0
    assert split["refused"] == 1
    assert not split["complete_measured"]

    _seconds, coverage, _launches = library.body(graph)
    assert coverage.refused == {"aiter::gemm_a16w16": 1}
    assert not coverage.complete
    # `refused` and `reasons` are keyed by operator name, so a switched
    # bracket has to show up under the family it was refused for, not as a bare
    # count that could have come from anywhere.
    reason = coverage.reasons["aiter::gemm_a16w16"]
    assert "same kernel" in reason
    assert "MT64x16x256" in reason and "MT128x32x128" in reason


# -- the width an observation is filed at -------------------------------------
#
# A price file states one width for the whole file. In the body that is also
# the width every operator in it ran at. In the head it is not: the graph is
# traced over the hidden state handed to `compute_logits`, and the LM-head
# GEMM runs after that state has been narrowed to one row per request.


def head_gemm(rows: int) -> dict:
    """The LM-head GEMM as run 5 executed it: (M, 5120) x (vocab, 5120)."""
    return {
        "name": "aiter::gemm_a16w16",
        "input_shapes": [[rows, 5120], [248320, 5120]],
        "dtypes": ["bfloat16", "bfloat16"],
        "scalars": [["#2", "None"]],
    }


def _head_file(library, tmp_path, rows: int, seconds: float, traced=None):
    """One head price file: a GEMM at `rows`, in a graph traced at `traced`."""
    from atom.compass.runtime.microbench import signature_of

    op = head_gemm(rows)
    graph = {"ops": [op]}
    if traced is not None:
        graph["provenance"] = {"execution": {"body_rows_traced": traced}}
    prices = {"prices": {signature_of(op): {
        "seconds": seconds, "kernels": {"k": seconds},
        "occurrences": 1, "name": op["name"]}}}
    gpath = tmp_path / f"hg{rows}.json"
    ppath = tmp_path / f"hp{rows}.json"
    gpath.write_text(json.dumps(graph))
    ppath.write_text(json.dumps(prices))
    library.add(str(ppath), str(gpath))
    return str(gpath)


def test_a_declared_family_reads_its_width_off_its_own_operator():
    from atom.compass.core.cost.families.features import executed_rows

    assert executed_rows(head_gemm(2)) == 2
    assert executed_rows(gemm(640)) == 640


def test_a_family_with_no_declared_reading_says_so_rather_than_guessing():
    from atom.compass.core.cost.families.features import executed_rows

    # `aten::view` carries no operand that says which dimension the batch is,
    # which is why the reading is declared per family instead of assumed to be
    # operand 0 dimension 0 everywhere.
    assert executed_rows({"name": "aten::view",
                          "input_shapes": [[32, 17408]]}) is None


def test_the_head_gemm_is_filed_at_the_rows_it_ran_not_the_rows_traced(
        tmp_path):
    """The whole point: 2 rows of work must not join the 16384-row curve.

    Both files below are traced over the full hidden height, because that is
    what a head graph is traced over. Their GEMMs ran at 1 and 2 rows.
    """
    library = ParametricPriceLibrary(max_gap_ratio=2.0)
    _head_file(library, tmp_path, 1, 1e-4, traced=16384)
    _head_file(library, tmp_path, 2, 2e-4, traced=16384)
    _head_file(library, tmp_path, 4, 4e-4, traced=16384)

    # Asked at a width no file holds, so the answer can only come from the
    # curve, and the curve names the widths it was built from. Those are the
    # widths the GEMMs ran at, not the height the graphs were traced over.
    record, source = library.lookup(head_gemm(3))
    assert record is not None, source
    assert record["interpolation"]["measured_rows"] == [1, 2, 4]
    assert record["interpolation"]["rows"] == 3

    # And the width the files state is outside that range entirely, so it is
    # refused rather than answered from these three.
    record, why = library.lookup(head_gemm(16384))
    assert record is None
    assert "outside the measured range" in why


def test_a_graph_stating_no_width_of_its_own_is_still_read_per_operator(
        tmp_path):
    """A head graph has no embedding and no `body_rows_traced`.

    Before the width was read off the operator this ended the file's
    usefulness, which is why every head price file in run 5 was
    exact-signature only and the M=2 step refused with "no entry for this
    signature".
    """
    library = ParametricPriceLibrary(max_gap_ratio=2.0)
    graph = _head_file(library, tmp_path, 1, 1e-4)
    _head_file(library, tmp_path, 2, 2e-4)

    assert graph in "".join(library.no_file_width.values())
    assert graph not in "".join(library.unbuildable.values())
    record, source = library.lookup(head_gemm(2))
    assert record is not None, source
    assert record["seconds"] == 2e-4


def test_the_head_metadata_operators_have_contracts_and_the_slice_is_a_view():
    """The three head metadata operators, and what each contract may claim.

    This test used to assert the slice had no contract at all. That was the
    right answer while the only thing known about it was its shapes: the
    slice is affine in requests, its operand is the cumulative offset vector
    whose length is requests + 1, and no row count makes two of its widths
    the same operator -- so a "rows" contract would have been a false
    declaration, and it stays false today.

    What changed is not the shape reasoning but the recording. The trace now
    carries ``output_aliases``, and on the real head graph the slice's entry
    is ``[-1]``: it allocated nothing and wrote into storage that existed
    before the step. That is a structural fact about the recording, not an
    inference from a small time, and it is the only basis on which the zero
    is allowed. The sibling tests below hold the other two directions -- an
    allocating slice is a copy and refuses, and a slice with no recorded
    alias refuses rather than assuming one.

    The sub and index contracts stay "rows". Both were refused in run 5,
    both were measured at 2.0e-06 s and 4.3e-06 s at M=2, and neither is
    zero.
    """
    assert contract_for("aten::sub.Tensor").kind == "rows"
    assert contract_for("aten::index.Tensor").kind == "rows"
    slice_contract = contract_for("aten::slice.Tensor")
    assert slice_contract is not None
    assert slice_contract.kind == "view"
    assert slice_contract.rows_from is None
    assert slice_contract.values == ()


def test_the_gather_is_not_answered_across_the_height_it_gathers_from():
    """`aten::index` has a second dimension and it is not collapsed.

    The selected count is the width; the height selected FROM stays part of
    the key. A measurement that gathered 2 rows out of 640 does not answer a
    request that gathers 2 out of 16384 -- that would be a claim about the
    source height nobody has measured at a fixed width.
    """
    def gather(selected: int, height: int) -> dict:
        return {"name": "aten::index.Tensor",
                "input_shapes": [[height, 5120], [selected]],
                "dtypes": ["bfloat16", "int32"]}

    assert infer_rows(gather(2, 16384), gather(1, 16384), 1) == 2
    assert infer_rows(gather(2, 16384), gather(2, 640), 2) is None


def _gather(selected: int, height: int) -> dict:
    """The head's last-token gather: `selected` rows out of a `height` state."""
    return {"name": "aten::index.Tensor",
            "input_shapes": [[height, 5120], [selected]],
            "dtypes": ["bfloat16", "int32"]}


def test_the_gather_refuses_a_pair_that_moved_the_height_with_the_width():
    """The height must not be solved along with the rows.

    Holding the height fixed and moving the selected count -- which is what
    the test above does -- never exercises this: at a fixed height the height
    is equal on both sides and passes on the equality branch. The failure
    needs the two to move TOGETHER. 2 out of 8192 and 4 out of 16384 share a
    coefficient of 4096 on the height and 1 on the count, so without a
    declaration they read as one operator at two widths, and a request for 3
    out of 12288 is solved from them at a source height nobody measured.
    """
    assert infer_rows(_gather(3, 12288), _gather(2, 8192), 2) is None
    assert infer_rows(_gather(4, 16384), _gather(2, 8192), 2) is None
    assert not aligns(_gather(4, 16384), 4, _gather(2, 8192), 2)


def test_the_gather_still_scales_at_one_height(tmp_path):
    """The case the head actually needs is not collateral damage.

    A fixed source height with a moving selected count is the ordinary head
    step, and it must still reach a price: the declaration fixes one extent,
    it does not turn the family into exact-match.
    """
    assert infer_rows(_gather(4, 16384), _gather(2, 16384), 2) == 4
    assert aligns(_gather(4, 16384), 4, _gather(2, 16384), 2)

    from atom.compass.runtime.microbench import signature_of

    library = ParametricPriceLibrary(max_gap_ratio=2.0)
    for selected, seconds in ((2, 2e-06), (4, 4e-06)):
        op = _gather(selected, 16384)
        graph = {"ops": [op],
                 "provenance": {"execution": {"body_rows_traced": 16384}}}
        prices = {"prices": {signature_of(op): {
            "seconds": seconds, "kernels": {"k": seconds},
            "occurrences": 1, "name": op["name"]}}}
        gpath = tmp_path / f"gg{selected}.json"
        ppath = tmp_path / f"gp{selected}.json"
        gpath.write_text(json.dumps(graph))
        ppath.write_text(json.dumps(prices))
        library.add(str(ppath), str(gpath))

    record, source = library.lookup(_gather(3, 16384))
    assert record is not None, source
    assert record["interpolation"]["measured_rows"] == [2, 4]
    assert record["interpolation"]["rows"] == 3

    # ... and the same request against an unmeasured height is refused.
    record, why = library.lookup(_gather(3, 12288))
    assert record is None


def test_a_graph_that_contradicts_itself_about_its_width_is_refused_whole(
        tmp_path):
    """Silence about the width and a contradiction about it differ.

    A head graph states no width and its operators are still trustworthy --
    that is what `no_file_width` is for. A body graph whose provenance says
    640 rows while its own embedding runs 1024 states one twice and disagrees
    with itself, and the operators whose family declares `rows_from` come off
    that same graph. Reading them anyway would let a file that used to be
    exact-key only start contributing points to a curve on the strength of
    the one number nobody is disputing, which is not a stronger source than
    it was before.
    """
    from atom.compass.runtime.microbench import signature_of

    library = ParametricPriceLibrary(max_gap_ratio=2.0)
    op = gemm(1024)
    embed = {"name": "aten::embedding",
             "input_shapes": [[151936, 5120], [1024]],
             "dtypes": ["bfloat16", "int64"]}
    graph = {"ops": [op, embed],
             "provenance": {"execution": {"body_rows_traced": 640}}}
    prices = {"prices": {signature_of(op): {
        "seconds": 1e-3, "kernels": {"k": 1e-3},
        "occurrences": 1, "name": op["name"]}}}
    gpath = tmp_path / "conflict.graph.json"
    ppath = tmp_path / "conflict.price.json"
    gpath.write_text(json.dumps(graph))
    ppath.write_text(json.dumps(prices))
    library.add(str(ppath), str(gpath))

    assert str(ppath) in library.unbuildable
    assert str(ppath) not in library.no_file_width
    assert "different widths" in library.unbuildable[str(ppath)]

    # And it contributed nothing: no curve exists to answer another width,
    # nor the width it was measured at by any route but the exact key.
    record, why = library.lookup(gemm(512))
    assert record is None


def _selector_gather(selected: int, height: int = 16384, rows=None) -> dict:
    """The head's gather as it is really recorded: with its row numbers.

    The default selectors are what a batch of equal-length requests produces --
    the last row of each slice of the state -- so they are in bounds, distinct
    and as many as the width.
    """
    if rows is None:
        step = height // selected
        rows = [(i + 1) * step - 1 for i in range(selected)]
    return {"name": "aten::index.Tensor",
            "input_shapes": [[height, 5120], [selected]],
            "dtypes": ["bfloat16", "int32"],
            "int_values": [[1, list(rows)]]}


def _offsets(width: int, height: int = 16384) -> dict:
    """The head's integer subtract, carrying the cumulative offsets."""
    step = height // width
    return {"name": "aten::sub.Tensor",
            "input_shapes": [[width]],
            "dtypes": ["int32"],
            "scalars": [["#1", 1]],
            "int_values": [[0, [(i + 1) * step for i in range(width)]]]}


def test_the_gather_matches_two_widths_once_its_selectors_are_validated():
    """Validated row numbers stand for their count, so two widths line up.

    Without this the head's gather could never be priced at any width but the
    one measured: the selectors differ at every width and at every mix of
    request lengths, so the literal vectors made two recordings of the same
    operator incomparable. The abstraction is the count, and the count is the
    width -- which is exactly what `aligns` is built to recognise.
    """
    assert aligns(_selector_gather(4), 4, _selector_gather(2), 2)
    assert infer_rows(_selector_gather(4), _selector_gather(2), 2) == 4

    # And two different batches at the SAME width, whose requests were split
    # differently, are the same operator rather than two.
    assert aligns(_selector_gather(4, rows=[0, 1, 2, 16383]), 4,
                  _selector_gather(4), 4)


def test_a_gather_that_reads_one_row_twice_keeps_its_literal_selectors():
    """Distinctness is a condition of the declaration, not a detail.

    The same count reading one row four times is a different amount of memory
    traffic from one reading four distinct rows, and only the second was
    measured. A vector that repeats a row fails validation, keeps its values,
    and is refused -- which is the fail-closed direction.
    """
    repeated = _selector_gather(4, rows=[4095, 4095, 4095, 4095])
    assert not aligns(repeated, 4, _selector_gather(2), 2)
    assert infer_rows(repeated, _selector_gather(2), 2) is None


def test_a_gather_whose_selector_leaves_the_source_keeps_its_values():
    """An index outside the source height is not a row of this tensor."""
    outside = _selector_gather(4, rows=[0, 1, 2, 16384])
    assert not aligns(outside, 4, _selector_gather(2), 2)
    negative = _selector_gather(4, rows=[-1, 1, 2, 3])
    assert not aligns(negative, 4, _selector_gather(2), 2)


def test_a_selector_that_does_not_fill_its_operand_keeps_its_values():
    """As many indices as the operand says, or the count means nothing."""
    short = _selector_gather(4)
    short["int_values"] = [[1, [4095, 8191, 12287]]]
    assert not aligns(short, 4, _selector_gather(2), 2)


def test_the_integer_subtract_matches_two_widths_of_unrelated_offsets():
    """Elementwise integer work at a fixed dtype and extent is the same work.

    The offsets themselves are unrelated between two batches -- they are
    cumulative token counts -- so without the declaration the subtract refused
    every width but its own, exactly as the gather did.
    """
    assert aligns(_offsets(4), 4, _offsets(2), 2)
    assert infer_rows(_offsets(4), _offsets(2), 2) == 4
    odd = {"name": "aten::sub.Tensor",
           "input_shapes": [[4]],
           "dtypes": ["int32"],
           "scalars": [["#1", 1]],
           "int_values": [[0, [7, 11, 13, 17]]]}
    assert aligns(odd, 4, _offsets(2), 2)


def test_a_slice_that_aliases_its_operand_is_priced_at_zero_structurally():
    """A view dispatches no kernel, and the recording is what says so.

    Not a measurement that came out small: `output_aliases` records, per
    output, whether the operator allocated it, decided as the trace ran by
    whether the output's storage is one of its own inputs. An index there
    means it wrote into a tensor that already existed.
    """
    library = ParametricPriceLibrary(max_gap_ratio=2.0)
    view = {"name": "aten::slice.Tensor",
            "input_shapes": [[5]],
            "output_shapes": [[4]],
            "dtypes": ["int32"],
            "scalars": [["#1", 0], ["#2", 1], ["#3", 9223372036854775807]],
            "int_values": [[0, [0, 4096, 8192, 12288, 16384]]],
            "output_aliases": [-1]}
    record, source = library.lookup(view)
    assert record is not None, source
    assert record["seconds"] == 0.0
    assert record["zero_work"] is True
    assert record["structural"]["basis"] == "alias"
    assert not record.get("interpolation")


def test_a_slice_that_allocated_its_output_is_refused():
    """Then it copied rather than viewed, and a copy is unmeasured work."""
    library = ParametricPriceLibrary(max_gap_ratio=2.0)
    copied = {"name": "aten::slice.Tensor",
              "input_shapes": [[5]],
              "output_shapes": [[4]],
              "dtypes": ["int32"],
              "output_aliases": [None]}
    record, why = library.lookup(copied)
    assert record is None
    assert "copied rather than viewed" in why


def test_a_slice_with_no_recorded_alias_is_refused_rather_than_assumed():
    """Empty means not known, which is not the same as not allocated.

    A graph written before `output_aliases` was recorded carries no statement
    about what the slice did. Reading that silence as a view would turn every
    such graph's slices into free work on no evidence.
    """
    library = ParametricPriceLibrary(max_gap_ratio=2.0)
    silent = {"name": "aten::slice.Tensor",
              "input_shapes": [[5]],
              "output_shapes": [[4]],
              "dtypes": ["int32"]}
    record, why = library.lookup(silent)
    assert record is None
    assert "not known" in why


def test_a_view_stays_structural_when_the_ragged_attention_law_is_live(
        tmp_path):
    """The view branch is tested before the ragged branch, and that order
    carries weight rather than reading well.

    `_parametric` asks three questions in sequence: is this a view, is it a
    row family, otherwise hand it to the ragged law. The third is a
    catch-all -- anything whose kind is not "rows" reaches it -- so a view
    only escapes the fitted attention law because it is intercepted first.
    Swap the two and the slice stops being structural. Verified by mutation
    rather than asserted: moving the ragged branch ahead of the view branch
    makes this test fail, and the observed failure is a refusal, not a wrong
    number -- the law declines an operator it has no design point for, the
    structural zero is lost, and a head graph that was fully accounted
    becomes incomplete. That is the mild version of the hazard. The severe
    version is a library whose law does match, which would answer with
    seconds for work that was never dispatched; this test cannot exhibit
    that, and does not claim to.
    A library with no attention observations cannot show this. `_modelled`
    refuses immediately when `_attention_obs` is empty, so a misordered
    dispatch would still produce a refusal and the test would pass for the
    wrong reason. This builds the library from the attention fixtures first,
    so the law is genuinely live, and only then asks for the slice.
    """
    from tests.compass.test_attention_family import _cold_designs, _library

    library = _library(tmp_path, _cold_designs())
    library._build()
    assert library._attention_obs  # the law is live, not an empty stub

    viewed = {
        "name": "aten::slice.Tensor",
        "input_shapes": [[3]],
        "output_shapes": [[2]],
        "dtypes": ["int32"],
        "output_dtypes": ["int32"],
        "output_aliases": [-1],
        "int_values": [],
    }
    record, source = library.lookup(viewed, {"tp": 1}, None)
    assert record is not None
    assert record["seconds"] == 0.0
    assert record["zero_work"] is True
    assert record["structural"]["basis"] == "alias"
    assert source.startswith("structural://")
    # and it did NOT come from the law, which would have marked it a
    # prediction rather than a structural absence
    assert not record.get("interpolated")
    assert "interpolated://" not in source

    # The same operator, with the alias evidence removed, refuses -- it does
    # not fall through to the law either. Not known is not not-allocated, and
    # it is not an invitation to model.
    unaliased = dict(viewed, output_aliases=[])
    record, why = library.lookup(unaliased, {"tp": 1}, None)
    assert record is None
    assert "output_aliases" in why
