"""The parts of a step that are neither the body graph nor the head graph.

A prediction composed of two graphs is a prediction of `run_model`. The runner
also prepares inputs and postprocesses logits, and at TP>1 broadcasts the
sampled ids, and none of that is in either graph. What these check is that the
region model supplies it from measurement and refuses everywhere else -- a
region model that answers for any shape has stopped being a measurement and
become a fitted constant.
"""

import json

import pytest

from atom.compass.core.cost.base import StepShape
from atom.compass.core.cost.library import (
    LibraryCostOracle, PriceLibrary, StaticGraphs)
from atom.compass.core.cost.regions import (
    SOURCE_27B_TP1, SOURCE_27B_TP1_CONC, SOURCE_27B_TP1_CONC_V2, Measured)


def _decode(seqs=32, context=1151, tp=1):
    return StepShape(num_scheduled_tokens=(1,) * seqs,
                     context_lens=(context,) * seqs,
                     topology={"tp": tp}, capture_bucket=32)


def _prefill(seqs=16, tokens=16384, tp=1):
    per = tokens // seqs
    return StepShape(num_scheduled_tokens=(per,) * seqs,
                     context_lens=(per,) * seqs,
                     num_prefill_tokens=tokens, topology={"tp": tp})


class TestTheDomainIsTheMeasurement:

    def test_the_measured_decode_is_answered(self):
        assert SOURCE_27B_TP1.refusal(_decode()) is None
        parts = SOURCE_27B_TP1.breakdown(_decode())
        assert set(parts) == {"<postprocess>", "<prepare>"}
        assert SOURCE_27B_TP1.seconds(_decode()) == pytest.approx(2.333e-4)

    def test_an_unmeasured_batch_size_is_refused(self):
        why = SOURCE_27B_TP1.refusal(_decode(seqs=20))
        assert why is not None and "20 sequences" in why
        with pytest.raises(ValueError, match="no measured region"):
            SOURCE_27B_TP1.seconds(_decode(seqs=20))

    def test_an_unmeasured_width_is_refused(self):
        why = SOURCE_27B_TP1.refusal(_decode(tp=8))
        assert why is not None and "tp=8" in why

    def test_a_prefill_outside_the_measured_extent_is_refused(self):
        why = SOURCE_27B_TP1.refusal(_prefill(seqs=16, tokens=1024))
        assert why is not None and "1024 tokens" in why

    def test_the_measured_prefill_is_answered(self):
        assert SOURCE_27B_TP1.refusal(_prefill()) is None
        # Prefill preparation is an order of magnitude above decode's: it
        # stages a whole chunk's ids and block tables, not one token each.
        assert (SOURCE_27B_TP1.breakdown(_prefill())["<prepare>"]
                > 8 * SOURCE_27B_TP1.breakdown(_decode())["<prepare>"])


class TestTheOneThingTPChanges:

    def test_the_broadcast_is_charged_only_above_one_rank(self):
        assert "<tp-broadcast>" not in SOURCE_27B_TP1.breakdown(_decode(tp=1))
        for tp in (2, 4):
            parts = SOURCE_27B_TP1.breakdown(_decode(tp=tp))
            assert parts["<tp-broadcast>"] == pytest.approx(2.86e-5)

    def test_the_rest_of_postprocess_does_not_move_with_tp(self):
        # `compute_logits` all-gathers the vocab shards before postprocess sees
        # them, so the sampler runs on the same shape at every width. This is
        # the claim that lets a TP1 calibration transfer with one added term.
        one = SOURCE_27B_TP1.breakdown(_decode(tp=1))
        two = SOURCE_27B_TP1.breakdown(_decode(tp=2))
        assert one["<postprocess>"] == two["<postprocess>"]
        assert one["<prepare>"] == two["<prepare>"]


class TestTheNumbersCarryTheirOwnSpread:

    def test_the_band_brackets_the_point_estimate(self):
        for shape in (_decode(), _decode(tp=2), _prefill()):
            low, high = SOURCE_27B_TP1.band(shape)
            assert low <= SOURCE_27B_TP1.seconds(shape) <= high

    def test_each_region_says_how_it_was_measured(self):
        text = SOURCE_27B_TP1.describe()
        for fragment in ("p50", "n=255", "bcast_probe", "cap_subspan"):
            assert fragment in text

    def test_the_cold_first_use_row_is_not_in_the_warm_model(self):
        # Five prefills were captured and four are used. The first is the first
        # use of that shape -- 7.071 s against 4.47-4.98 s -- and the acceptance
        # protocol is warmed, so it stays classified as cold rather than being
        # averaged into a warm constant.
        assert SOURCE_27B_TP1.postprocess_prefill.samples == 4
        assert SOURCE_27B_TP1.prepare_prefill.samples == 4
        assert "warm" in SOURCE_27B_TP1.prepare_prefill.how

    def test_a_measured_value_describes_itself(self):
        m = Measured(1e-4, 9e-5, 1.1e-4, 7, "p50 [p10,p90] of something")
        assert "0.1000 ms" in m.describe() and "n=7" in m.describe()


class TestTheOracleChargesTheRegionsItWasGiven:

    def _oracle(self, tmp_path, **kwargs):
        op = {"name": "aiter::gemm", "input_shapes": [[32, 4096], [4096, 4096]],
              "dtypes": ["bfloat16"]}
        from atom.compass.runtime.microbench import signature_of
        blob = {"prices": {signature_of(op): {
                    "name": op["name"], "seconds": 1e-3, "occurrences": 1,
                    "kernels": {"k0": 1e-3}}},
                "unpriced": {}, "coverage": {},
                "provenance": {"topology": {"tp": 1}}}
        path = tmp_path / "p.json"
        path.write_text(json.dumps(blob))
        shape = _decode()
        graphs = StaticGraphs({StaticGraphs.key(shape): {
            "ops": [op], "key": {"topology": [["tp", 1]]},
            "provenance": {"execution": {"body_rows_traced": 32}}}})
        return shape, LibraryCostOracle(
            PriceLibrary.load([(str(path), None)]), graphs, **kwargs)

    def test_the_regions_appear_by_name_in_the_breakdown(self, tmp_path):
        shape, oracle = self._oracle(tmp_path, regions=SOURCE_27B_TP1)
        cost = oracle.estimate(shape)
        assert cost.breakdown["<postprocess>"] == pytest.approx(1.019e-4)
        assert cost.breakdown["<prepare>"] == pytest.approx(1.314e-4)
        assert cost.seconds == pytest.approx(1e-3 + 2.333e-4)

    def test_a_shape_outside_the_domain_is_refused_not_answered(self, tmp_path):
        shape, oracle = self._oracle(tmp_path, regions=SOURCE_27B_TP1)
        wide = StepShape(num_scheduled_tokens=shape.num_scheduled_tokens,
                         context_lens=shape.context_lens,
                         topology={"tp": 8}, capture_bucket=32)
        # The graph lookup would answer; the region model is what refuses.
        oracle.graphs._graphs[StaticGraphs.key(wide)] = (
            oracle.graphs._graphs[StaticGraphs.key(shape)])
        with pytest.raises(ValueError, match="no measured region"):
            oracle.estimate(wide)

    def test_a_scalar_and_a_region_model_are_not_both_accepted(self, tmp_path):
        with pytest.raises(ValueError, match="twice"):
            self._oracle(tmp_path, regions=SOURCE_27B_TP1,
                         extra_seconds=2.3e-4)


class TestTheConcurrencyProfileIsKeyedOnTheRungThatRan:
    """`SOURCE_27B_TP1_CONC`: fourteen decode concurrencies, ten cells.

    What the single-concurrency profile could not say, and what the cells say
    instead -- including where they still refuse.
    """

    def _dec(self, seqs, bucket, context=1151, tp=1):
        return StepShape(num_scheduled_tokens=(1,) * seqs,
                         context_lens=(context,) * seqs,
                         topology={"tp": tp}, capture_bucket=bucket)

    def test_a_concurrency_the_old_profile_refused_is_answered(self):
        """Sixteen sequences was outside a domain measured only at 32."""
        assert SOURCE_27B_TP1.refusal(self._dec(16, 16)) is not None
        assert SOURCE_27B_TP1_CONC.refusal(self._dec(16, 16)) is None

    def test_a_padded_batch_is_priced_from_its_padded_cell(self):
        """Twenty sequences replays bucket 32, and does not cost what 32 does."""
        padded = SOURCE_27B_TP1_CONC.breakdown(self._dec(20, 32))["<prepare>"]
        exact = SOURCE_27B_TP1_CONC.breakdown(self._dec(32, 32))["<prepare>"]
        assert padded == pytest.approx(1.381e-4)
        assert exact == pytest.approx(1.256e-4)
        assert padded > exact

    def test_preparation_is_not_monotone_in_the_batch(self):
        """Why a cell table and not a curve: two sequences cost more than
        eight, so anything rising with batch size is wrong here."""
        two = SOURCE_27B_TP1_CONC.breakdown(self._dec(2, 2))["<prepare>"]
        eight = SOURCE_27B_TP1_CONC.breakdown(self._dec(8, 8))["<prepare>"]
        assert two > eight

    def test_postprocess_is_one_constant_across_every_measured_batch(self):
        one = SOURCE_27B_TP1_CONC.breakdown(self._dec(1, 1))["<postprocess>"]
        for seqs, bucket in ((2, 2), (16, 16), (20, 32), (32, 32)):
            assert (SOURCE_27B_TP1_CONC.breakdown(
                self._dec(seqs, bucket))["<postprocess>"] == one)

    def test_a_pairing_the_ladder_never_produces_is_refused(self):
        """Nine sequences replay bucket 16, so nine at bucket 32 never ran.

        The (32, padded) cell is real -- N=17, 20 and 31 filled it. It is not
        an answer for nine, which the engine sends to a different rung and pads
        a different distance.
        """
        why = SOURCE_27B_TP1_CONC.refusal(self._dec(9, 32))
        assert why is not None and "replays 9 at 16" in why
        assert SOURCE_27B_TP1_CONC.refusal(self._dec(9, 16)) is None

    def test_a_bucket_off_the_ladder_is_refused_against_the_cells(self):
        """Above the ladder there is no rung to check a declared bucket
        against, so the measured cells do the refusing instead."""
        why = SOURCE_27B_TP1_CONC.refusal(self._dec(33, 64))
        assert why is not None and "measured only at" in why

    def test_a_bucket_narrower_than_the_batch_is_refused(self):
        why = SOURCE_27B_TP1_CONC.refusal(self._dec(20, 16))
        assert why is not None and "narrower" in why

    def test_an_eager_step_is_refused_rather_than_read_as_the_first_rung(self):
        """`capture_bucket=None` means nothing replayed; every measured row
        here replayed."""
        why = SOURCE_27B_TP1_CONC.refusal(self._dec(1, None))
        assert why is not None and "replayed no captured graph" in why

    def test_a_long_history_is_refused_because_nothing_measured_one(self):
        """The bursts ran 1024-token prompts to 128 outputs. cc-traces reaches
        109 741 tokens, where the term that grows with history is no longer
        below its dispatch threshold."""
        why = SOURCE_27B_TP1_CONC.refusal(self._dec(32, 32, context=109_741))
        assert why is not None and "grows with history" in why
        assert SOURCE_27B_TP1_CONC.refusal(self._dec(32, 32, context=1152)) is None

    def test_a_mixed_batch_is_refused_on_its_longest_history(self):
        shape = StepShape(num_scheduled_tokens=(1, 1),
                          context_lens=(1100, 40_000),
                          topology={"tp": 1}, capture_bucket=2)
        assert "40000" in SOURCE_27B_TP1_CONC.refusal(shape)

    def test_the_broadcast_is_still_added_at_width(self):
        at_one = SOURCE_27B_TP1_CONC.breakdown(self._dec(16, 16))
        at_two = SOURCE_27B_TP1_CONC.breakdown(self._dec(16, 16, tp=2))
        assert "<tp-broadcast>" not in at_one
        assert at_two["<tp-broadcast>"] == pytest.approx(2.86e-5)

    def test_the_band_covers_the_pooled_spread(self):
        low, high = SOURCE_27B_TP1_CONC.band(self._dec(20, 32))
        assert low < SOURCE_27B_TP1_CONC.seconds(self._dec(20, 32)) < high

    def test_the_prefill_terms_are_the_older_profiles_own_objects(self):
        """Carried, not re-stated: neither capture ran a prefill in that
        domain, so a second copy of the number would be copying."""
        assert (SOURCE_27B_TP1_CONC.prepare_prefill
                is SOURCE_27B_TP1.prepare_prefill)
        assert (SOURCE_27B_TP1_CONC.postprocess_prefill
                is SOURCE_27B_TP1.postprocess_prefill)
        assert SOURCE_27B_TP1_CONC.tp_broadcast is SOURCE_27B_TP1.tp_broadcast

    def test_the_ladder_is_offered_but_not_used_to_fill_a_missing_bucket(self):
        """A caller may resolve a rung; the region model may not guess one."""
        assert SOURCE_27B_TP1_CONC.bucket_for(12) == 16
        assert SOURCE_27B_TP1_CONC.bucket_for(32) == 32
        assert SOURCE_27B_TP1_CONC.bucket_for(33) is None
        assert SOURCE_27B_TP1_CONC.refusal(self._dec(12, None)) is not None


class TestTheFrozenProfileDidNotMove:
    """The TP2/TP4 transfer reports were computed from `SOURCE_27B_TP1`.

    A newer measurement does not improve a prediction that has already been
    made and checked against a target, so these pin the old numbers against the
    new ones rather than migrating them.
    """

    def test_the_decode_constants_are_what_they_were_measured_at(self):
        assert SOURCE_27B_TP1.prepare_decode.seconds == pytest.approx(1.314e-4)
        assert SOURCE_27B_TP1.postprocess_decode.seconds == pytest.approx(1.019e-4)
        assert SOURCE_27B_TP1.decode_sequences == (32,)

    def test_the_two_profiles_disagree_and_that_is_the_point(self):
        """0.1314 ms from 64 paced requests settling at 32, 0.1256 ms from 32
        released in lockstep -- 4.4% apart, un-attributed. Reconciling them by
        overwriting one would hide the disagreement, not resolve it."""
        old = SOURCE_27B_TP1.breakdown(_decode())["<prepare>"]
        new = SOURCE_27B_TP1_CONC.breakdown(_decode())["<prepare>"]
        assert old != new
        assert abs(old - new) / old == pytest.approx(0.044, abs=0.005)


class TestTheSecondInstanceProfile:
    """`/2` exists because a second server disagreed with the first.

    Not because a row was lost. The lost row is one cell's sample count; the
    between-instance shift is every prepare cell's band.
    """

    def _dec(self, seqs, bucket, context=1151, tp=1):
        return StepShape(num_scheduled_tokens=(1,) * seqs,
                         context_lens=(context,) * seqs,
                         topology={"tp": tp}, capture_bucket=bucket)

    def test_the_first_profile_was_not_edited(self):
        """Including the cell whose capture was truncated. A published number
        that changes later is a number nobody can check."""
        cells = dict(SOURCE_27B_TP1_CONC.prepare_decode_cells)
        assert cells[(32, False)].samples == 383
        assert cells[(32, False)].seconds == pytest.approx(1.256e-4)
        assert SOURCE_27B_TP1_CONC.version == "source-27b-tp1-conc/1"

    def test_the_re_measured_cell_has_its_full_three_bursts(self):
        cells = dict(SOURCE_27B_TP1_CONC_V2.prepare_decode_cells)
        assert cells[(32, False)].samples == 384
        assert all(m.samples in (384, 1152)
                   for _, m in SOURCE_27B_TP1_CONC_V2.prepare_decode_cells)

    def test_every_band_contains_the_other_instances_answer(self):
        """The property a one-server band did not have.

        For the seven cells both servers measured, `/1`'s number falls inside
        `/2`'s band. That is what makes `/2` usable for a prediction about a
        server that has not started yet.
        """
        one = dict(SOURCE_27B_TP1_CONC.prepare_decode_cells)
        two = dict(SOURCE_27B_TP1_CONC_V2.prepare_decode_cells)
        shared = [c for c in two if "ONE INSTANCE" not in two[c].how]
        assert len(shared) == 7
        for cell in shared:
            assert two[cell].low <= one[cell].seconds <= two[cell].high, cell

    def test_the_single_instance_cells_say_so(self):
        """Three cells only one server ever measured. Their bands are narrower
        than the others' for a reason that is not precision."""
        two = dict(SOURCE_27B_TP1_CONC_V2.prepare_decode_cells)
        alone = {c for c in two if "ONE INSTANCE" in two[c].how}
        assert alone == {(2, False), (4, False), (16, True)}

    def test_the_shift_is_a_level_and_not_a_shape(self):
        """Every shared cell moved up by between 4.0% and 5.0%. A shift that
        uniform is the server, not the batch: if it varied by cell, the profile
        would be measuring different work, not the same work more slowly."""
        one = dict(SOURCE_27B_TP1_CONC.prepare_decode_cells)
        two = dict(SOURCE_27B_TP1_CONC_V2.prepare_decode_cells)
        shifts = [two[c].seconds / one[c].seconds - 1
                  for c in two if "ONE INSTANCE" not in two[c].how]
        assert min(shifts) > 0.039 and max(shifts) < 0.051

    def test_the_padding_drop_reproduced_on_a_second_server(self):
        """The structural finding, independently. A batch of exactly 32
        prepares faster than a batch of 17, 20 or 31 padded up to 32 -- about
        9% faster on both instances, which is why it is the shape and not the
        level that this profile claims to have measured."""
        for profile in (SOURCE_27B_TP1_CONC, SOURCE_27B_TP1_CONC_V2):
            cells = dict(profile.prepare_decode_cells)
            drop = cells[(32, False)].seconds / cells[(32, True)].seconds - 1
            assert drop == pytest.approx(-0.091, abs=0.004)

    def test_postprocess_is_the_one_region_that_survived_the_restart(self):
        """0.43% apart across two servers, against prepare's 4-5%. Whatever
        the restart changed, it did not change this."""
        one = SOURCE_27B_TP1_CONC.postprocess_decode.seconds
        two = SOURCE_27B_TP1_CONC_V2.postprocess_decode.seconds
        assert abs(two / one - 1) < 0.01

    def test_it_answers_the_same_shapes_and_refuses_the_same_ones(self):
        """A newer profile, not a wider one: same ladder, same domain."""
        assert SOURCE_27B_TP1_CONC_V2.refusal(self._dec(20, 32)) is None
        assert SOURCE_27B_TP1_CONC_V2.refusal(self._dec(9, 32)) is not None
        assert SOURCE_27B_TP1_CONC_V2.refusal(self._dec(1, None)) is not None
        assert SOURCE_27B_TP1_CONC_V2.refusal(
            self._dec(32, 32, context=109_741)) is not None
        assert SOURCE_27B_TP1_CONC_V2.version == "source-27b-tp1-conc/2"

    def test_the_prefill_terms_are_still_the_older_profiles_own_objects(self):
        assert (SOURCE_27B_TP1_CONC_V2.prepare_prefill
                is SOURCE_27B_TP1.prepare_prefill)
        assert SOURCE_27B_TP1_CONC_V2.tp_broadcast is SOURCE_27B_TP1.tp_broadcast
