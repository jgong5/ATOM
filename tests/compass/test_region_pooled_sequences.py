"""One pooled group for three-or-more sequences, and nothing else pooled.

`source-27b-tp1-prefill-interp` refuses every prefill above two sequences. The
corrected client workload reaches thirty-two, because clients are top-level
sessions rather than an in-flight cap, so the replay cannot run against that
model at all. `source-27b-tp1-prefill-seqs` answers 3..32 from one pooled token
axis.

What these check is the boundary of that relaxation, not that it is a good
approximation -- end-to-end accuracy decides the latter. Specifically: that the
one- and two-sequence groups still answer exactly what they answered, that the
pooled group refuses outside its measured token span, that a refusal names the
group rather than the sentinel, and that the predecessor is untouched.
"""

import pytest

from atom.compass.core.cost.base import StepShape
from atom.compass.core.cost.regions import (
    POOLED_SEQS, REGION_MODELS, SOURCE_27B_TP1_PREFILL_INTERP,
    SOURCE_27B_TP1_PREFILL_SEQS)

INTERP = SOURCE_27B_TP1_PREFILL_INTERP
SEQS = SOURCE_27B_TP1_PREFILL_SEQS


def _prefill(seqs, tokens, tp=1):
    """A final-chunk prefill: every sequence's prompt completes, so it samples.

    The remainder goes on the first sequence rather than being spread, because
    the key is (count, total, produces_output) and the split within the batch
    is not part of it.
    """
    per = tokens // seqs
    sched = (tokens - per * (seqs - 1),) + (per,) * (seqs - 1)
    return StepShape(num_scheduled_tokens=sched, context_lens=sched,
                     num_prefill_tokens=tokens, topology={"tp": tp})


class TestThePredecessorIsUntouched:

    def test_the_interp_model_still_refuses_above_two_sequences(self):
        why = INTERP.refusal(_prefill(32, 16384))
        assert why is not None and "32 sequence(s)" in why

    def test_it_still_carries_exactly_five_anchors_per_term(self):
        assert len(INTERP.prepare_prefill_anchors) == 5
        assert len(INTERP.postprocess_prefill_anchors) == 5

    def test_the_new_model_carries_the_old_anchors_as_the_same_objects(self):
        assert (SEQS.prepare_prefill_anchors[:5]
                == INTERP.prepare_prefill_anchors)
        assert (SEQS.postprocess_prefill_anchors[:5]
                == INTERP.postprocess_prefill_anchors)

    def test_it_declares_no_pooling_of_its_own(self):
        assert INTERP.prefill_pooled_sequences == ()


class TestOneAndTwoSequencesAreNotPooled:
    """The relaxation starts at three, and this is why it has to.

    Preparation over two sequences totalling 15360 tokens is roughly five
    times what one sequence of 15232 costs. Pooling those would be the
    indefensible version of this change, so the pooled range starts above them
    and the two groups must answer bit for bit what they did before.
    """

    @pytest.mark.parametrize("seqs,tokens", [(1, 8192), (1, 7680), (1, 1024),
                                             (1, 9216), (2, 2048)])
    def test_the_answer_is_identical_to_the_predecessors(self, seqs, tokens):
        shape = _prefill(seqs, tokens)
        assert INTERP.refusal(shape) is None
        assert SEQS.refusal(shape) is None
        assert SEQS.seconds(shape) == INTERP.seconds(shape)
        assert SEQS.band(shape) == INTERP.band(shape)

    def test_the_run_7_tail_is_still_interpolated_and_not_pooled(self):
        # (1, 9216) is the shape run 7 died on. It must still come from the
        # one-sequence anchors either side of it, not from the new group,
        # whose 8192 and 10240 values are twice as large.
        shape = _prefill(1, 9216)
        assert SEQS.seconds(shape) == pytest.approx(1.0e-3, rel=0.2)
        assert SEQS._pool(1) == 1
        assert SEQS._pool(2) == 2


class TestThePooledGroupAnswersTheSchedulersDomain:

    @pytest.mark.parametrize("seqs", [3, 4, 5, 12, 17, 20, 24, 28, 32])
    def test_every_sequence_count_through_32_is_answered(self, seqs):
        shape = _prefill(seqs, 12288)
        assert SEQS.refusal(shape) is None
        assert SEQS.seconds(shape) > 0

    def test_a_measured_shape_returns_its_own_anchor_exactly(self):
        # 32x512 is the largest step the scheduler can build and was measured.
        assert SEQS.breakdown(_prefill(32, 16384))["<prepare>"] == (
            pytest.approx(3.985405e-3))

    def test_the_sequence_count_does_not_move_the_answer(self):
        # The pooling claim, stated as a test: same tokens, eight-fold
        # difference in sequence count, one answer.
        assert (SEQS.seconds(_prefill(4, 16384))
                == SEQS.seconds(_prefill(32, 16384)))

    def test_an_unmeasured_token_count_interpolates_within_the_group(self):
        low = SEQS.breakdown(_prefill(20, 10240))["<prepare>"]
        high = SEQS.breakdown(_prefill(20, 12288))["<prepare>"]
        mid = SEQS.breakdown(_prefill(20, 11264))["<prepare>"]
        assert low < mid < high

    def test_an_interpolated_answer_says_it_is_one(self):
        how = SEQS._prefill_at(
            SEQS._prefill_table(SEQS.prepare_prefill_cells,
                                SEQS.prepare_prefill_anchors),
            (POOLED_SEQS, 11264, True)).how
        assert "INTERPOLATED" in how
        assert "pooled 3..32-sequence group" in how

    def test_the_band_is_the_union_and_never_tighter_than_the_evidence(self):
        lo, hi = SEQS.band(_prefill(20, 11264))
        # 10240's low edge is a flat-mode row at 7.92e-4; an interpolated band
        # that excluded it would be claiming precision nobody measured.
        assert lo <= 7.9247e-4 + 1.006e-4


class TestTheGroupRefusesOutsideWhatWasMeasured:

    def test_below_the_measured_span_is_refused(self):
        why = SEQS.refusal(_prefill(3, 900))
        assert why is not None and "outside the measured token span" in why

    def test_above_32_sequences_is_refused(self):
        # max_num_seqs is 32, so this is not a shape the scheduler builds; if
        # one ever appears it must refuse rather than borrow the group.
        why = SEQS.refusal(_prefill(33, 16384))
        assert why is not None
        assert "33 sequence(s)" in why

    def test_a_refusal_names_the_group_and_not_the_sentinel(self):
        why = SEQS.refusal(_prefill(3, 900))
        assert "pooled 3..32-sequence group" in why
        assert str(POOLED_SEQS) not in why


class TestTheModelIsReachableAndDescribesItself:

    def test_it_is_registered_under_its_name(self):
        assert REGION_MODELS["source-27b-tp1-prefill-seqs"] is SEQS

    def test_the_predecessors_are_still_registered(self):
        assert REGION_MODELS["source-27b-tp1-prefill-interp"] is INTERP

    def test_the_description_states_the_pooling(self):
        text = SEQS.describe()
        assert "sequence counts 3..32 share one token axis" in text
        assert "1 and 2 remain separate groups" in text

    def test_every_pooled_prepare_anchor_has_a_postprocess_one(self):
        # A sampling step whose postprocess is missing is a gap, not zero, and
        # `refusal` enforces that. Checking the tuples directly catches a
        # half-populated stanza before a shape has to.
        prep = {k for k, _ in SEQS.prepare_prefill_anchors
                if k[0] == POOLED_SEQS}
        post = {k for k, _ in SEQS.postprocess_prefill_anchors
                if k[0] == POOLED_SEQS}
        assert prep == post

    def test_the_provenance_names_the_campaigns_and_the_exclusion(self):
        assert "regionseqs" in SEQS.provenance
        assert "regiongaps" in SEQS.provenance
        assert "cap_conc" in SEQS.provenance
        assert "waiting=0" in SEQS.provenance
