"""The prefill profile, declared for the widths the matrix actually runs.

`source-27b-tp1-prefill-seqs` shipped saying `topologies=(1,)` while carrying
its decode cells, its scalar prefill terms and its broadcast from
`source-27b-tp1-conc-v2`, which says `(1, 2, 4)`. Nothing was measured at TP2
or TP4 to justify the narrowing and nothing was measured at TP1 to justify the
parent's width either -- the parent's transfer is a source-code argument plus
one separately measured collective, and the descendant inherited the numbers
without the declaration. What that cost was concrete: the TP2 and TP4 cells of
the client matrix had to run `regions=none`, which does not widen a prediction,
it deletes preparation and postprocess from it.

So these check the transfer as a transfer, not a preset as a preset:

* the coefficients are the same objects they were, so TP1 answers what it
  answered and no number moved under cover of a domain change;
* at every declared width the two regions are identical and the only
  difference is `<tp-broadcast>`, which is the one term measured on real two-
  and four-rank groups;
* a step that samples nothing pays no broadcast at TP4, because the collective
  is of the sampled token;
* `build_source_group` prices all three widths through the real composition
  with this profile named, and the region snapshot records the widened domain;
* the one preparation term that really is sized by the rank's own KV head
  count -- the block_size 256/1024 metadata build -- is refused at build time
  rather than answered from a TP1 measurement of different work.
"""

import json

import pytest

from atom.compass.core.cost.base import StepShape
from atom.compass.core.cost.regions import (
    KV_HEAD_SIZED_PREPARE_BLOCK_SIZES, POOLED_SEQS, REGION_MODELS,
    SOURCE_27B_TP1, SOURCE_27B_TP1_CONC_V2, SOURCE_27B_TP1_PREFILL_CELLS,
    SOURCE_27B_TP1_PREFILL_INTERP, SOURCE_27B_TP1_PREFILL_SEQS,
    wide_tp_precondition)

SEQS = SOURCE_27B_TP1_PREFILL_SEQS
WIDTHS = (1, 2, 4)

#: A decode inside the profile's own domain: a replayed bucket it measured, at
#: a history length its context bound covers.
BUCKET = 16
CONTEXT = 1100


def _decode(tp=1, seqs=BUCKET, bucket=BUCKET, rank=0):
    return StepShape(num_scheduled_tokens=(1,) * seqs,
                     context_lens=(CONTEXT,) * seqs,
                     num_prefill_tokens=0, topology={"tp": tp},
                     rank_coords={"tp": rank},
                     capture_bucket=bucket, compiled=None,
                     produces_output=True)


def _prefill(seqs, tokens, tp=1, produces_output=True):
    """A prefill in the pooled group's token axis, as the runner reports one."""
    per = tokens // seqs
    sched = (tokens - per * (seqs - 1),) + (per,) * (seqs - 1)
    return StepShape(num_scheduled_tokens=sched, context_lens=sched,
                     num_prefill_tokens=tokens, topology={"tp": tp},
                     produces_output=produces_output)


class TestNoCoefficientMoved:
    """A domain change that also moved a number would be unreviewable."""

    def test_the_decode_cells_are_the_parent_s_own_objects(self):
        assert (SEQS.prepare_decode_cells
                is SOURCE_27B_TP1_CONC_V2.prepare_decode_cells)
        assert (SEQS.postprocess_decode
                is SOURCE_27B_TP1_CONC_V2.postprocess_decode)

    def test_the_prefill_anchors_are_the_predecessor_s_own_objects(self):
        assert (SEQS.prepare_prefill_anchors[:5]
                == SOURCE_27B_TP1_PREFILL_INTERP.prepare_prefill_anchors)
        assert (SEQS.prepare_prefill_cells
                is SOURCE_27B_TP1_PREFILL_CELLS.prepare_prefill_cells)

    def test_the_broadcast_is_the_one_measured_on_real_groups(self):
        """The only term TP>1 adds, and it is not a residual: it comes from
        `SOURCE_27B_TP1`, where a standalone probe timed it on the real two-
        and four-rank groups."""
        assert SEQS.tp_broadcast is SOURCE_27B_TP1.tp_broadcast
        assert "bcast_probe" in SEQS.tp_broadcast.how
        assert SEQS.tp_broadcast.samples > 0

    def test_the_pooled_group_still_spans_three_to_thirty_two(self):
        assert SEQS.prefill_pooled_sequences == (3, 32)


class TestTheDeclaredWidths:

    def test_it_declares_the_parent_s_widths(self):
        assert SEQS.topologies == (1, 2, 4)
        assert SEQS.topologies is SOURCE_27B_TP1_CONC_V2.topologies

    def test_the_version_moves_with_the_domain(self):
        """`region_snapshot` digests the domain along with the numbers, so a
        run stamped against the narrow declaration must not be confusable with
        one stamped against this."""
        assert SEQS.version == "prefill-seqs-2026-09-13"

    def test_the_provenance_says_where_the_numbers_were_measured(self):
        assert "measured at TP1" in SEQS.provenance
        assert "No full-engine or serving measurement at TP2 or TP4" in (
            SEQS.provenance)

    def test_it_names_what_it_does_not_cover(self):
        for missing in ("inter-rank skew", "process control", "logprobs"):
            assert missing in SEQS.provenance, missing

    @pytest.mark.parametrize("tp", WIDTHS)
    def test_no_width_is_refused_for_being_wide(self, tp):
        assert SEQS.refusal(_decode(tp=tp)) is None
        assert SEQS.refusal(_prefill(8, 12288, tp=tp)) is None

    def test_an_undeclared_width_is_still_refused(self):
        why = SEQS.refusal(_decode(tp=8))
        assert why is not None and "outside the measured widths" in why

    def test_the_ordinary_wide_configuration_names_this_profile(self):
        """What the earlier `regions=none` was: a caller configuration, not a
        missing implementation. The acceptance matrix names this profile for
        every cell, so the profile has to answer at every width the matrix
        runs."""
        from scripts.compass.cc_traces_registry import SHARED_OPTIONS

        options = dict(SHARED_OPTIONS)
        assert REGION_MODELS[options["regions"]] is SEQS
        assert set(WIDTHS) <= set(SEQS.topologies)
        assert int(options["block_size"]) not in (
            KV_HEAD_SIZED_PREPARE_BLOCK_SIZES)


class TestOnlyTheBroadcastDiffersAcrossWidths:

    @pytest.mark.parametrize("tp", WIDTHS)
    def test_preparation_and_postprocess_are_the_tp1_numbers(self, tp):
        at_one = SEQS.breakdown(_decode(tp=1))
        wide = SEQS.breakdown(_decode(tp=tp))

        assert wide["<prepare>"] == at_one["<prepare>"]
        assert wide["<postprocess>"] == at_one["<postprocess>"]

    @pytest.mark.parametrize("tp", (2, 4))
    def test_the_whole_difference_is_one_broadcast(self, tp):
        wide = SEQS.seconds(_decode(tp=tp))

        assert wide - SEQS.seconds(_decode(tp=1)) == pytest.approx(
            SEQS.tp_broadcast.seconds, rel=1e-12)

    def test_tp1_is_charged_no_collective(self):
        assert "<tp-broadcast>" not in SEQS.breakdown(_decode(tp=1))

    @pytest.mark.parametrize("tp", (2, 4))
    def test_the_broadcast_is_not_scaled_by_the_width(self, tp):
        """Two ranks and four ranks are charged the same measured number.
        Anything else here would be a model of the collective, and nothing
        measured one as a function of the width."""
        assert SEQS.breakdown(_decode(tp=tp))["<tp-broadcast>"] == (
            SEQS.tp_broadcast.seconds)

    @pytest.mark.parametrize("tp", (2, 4))
    def test_a_prefill_that_samples_pays_it_too(self, tp):
        assert "<tp-broadcast>" in SEQS.breakdown(_prefill(8, 12288, tp=tp))

    def test_a_middle_chunk_pays_no_broadcast_at_tp4(self):
        """The collective is of the sampled token. A chunk that samples
        nothing never runs it, and billing one would be a collective invented
        by the width declaration."""
        middle = _prefill(1, 16384, tp=4, produces_output=False)

        assert "<tp-broadcast>" not in SEQS.breakdown(middle)

    @pytest.mark.parametrize("tp", (2, 4))
    def test_the_band_widens_by_the_broadcast_s_own_band(self, tp):
        low, high = SEQS.band(_decode(tp=tp))
        base_low, base_high = SEQS.band(_decode(tp=1))

        assert low == pytest.approx(base_low + SEQS.tp_broadcast.low, rel=1e-12)
        assert high == pytest.approx(base_high + SEQS.tp_broadcast.high,
                                     rel=1e-12)


class TestTheOneWidthSizedPreparationTerm:
    """`set_aiter_persistent_worker_buffers` sizes its tables by
    `num_key_value_heads // world_size`, and runs only at block_size 256 or
    1024. On those block sizes a TP1 preparation measurement is a measurement
    of different work, so the transfer argument does not hold and the pairing
    is refused where both facts are known: at build time. The shape a region
    model is handed carries no block size, so `refusal` cannot see this.
    """

    @pytest.mark.parametrize("block_size", KV_HEAD_SIZED_PREPARE_BLOCK_SIZES)
    @pytest.mark.parametrize("tp", (2, 4))
    def test_a_kv_head_sized_build_is_refused_above_tp1(self, tp, block_size):
        why = wide_tp_precondition(SEQS, tp, block_size)

        assert why is not None
        assert "set_aiter_persistent_worker_buffers" in why
        assert "num_key_value_heads // tp" in why

    @pytest.mark.parametrize("tp", (2, 4))
    def test_the_deployment_s_own_block_size_is_allowed(self, tp):
        assert wide_tp_precondition(SEQS, tp, 16) is None

    @pytest.mark.parametrize("block_size", KV_HEAD_SIZED_PREPARE_BLOCK_SIZES)
    def test_tp1_is_never_refused_by_it(self, block_size):
        """At one rank `num_key_value_heads // 1` is the head count, and the
        TP1 measurement is a measurement of exactly that work."""
        assert wide_tp_precondition(SEQS, 1, block_size) is None

    def test_regions_none_has_nothing_to_check(self):
        assert wide_tp_precondition(None, 4, 256) is None

    @pytest.mark.parametrize("block_size", KV_HEAD_SIZED_PREPARE_BLOCK_SIZES)
    def test_the_parents_are_held_to_the_same_condition(self, block_size):
        """`SOURCE_27B_TP1` and `conc-v2` transfer on the same argument, so
        they must fail on the same configuration. A guard that only caught the
        newest profile would leave the older names as a way around it."""
        assert wide_tp_precondition(SOURCE_27B_TP1, 2, block_size) is not None
        assert wide_tp_precondition(
            SOURCE_27B_TP1_CONC_V2, 4, block_size) is not None


#: One rank's decode step, priced from one operator so the test is about the
#: regions rather than about a body.
OPS = [{"name": "mm", "input_shapes": [[BUCKET, 64], [64, 64]],
        "dtypes": ["bfloat16"]}]
BODY_SECONDS = 0.011


def _template(tmp_path, rank, width):
    path = tmp_path / f"graph.tp{rank}.json"
    path.write_text(json.dumps({
        "ops": OPS,
        "key": {"topology": [["tp", width]], "rank_coords": [["tp", rank]]},
        "provenance": {
            "batch_spec": {"kind": "decode", "query_lens": [1] * BUCKET,
                           "context_lens": [CONTEXT] * BUCKET},
            "execution": {"capture_bucket": BUCKET},
        },
    }), encoding="utf-8")
    return str(path)


def _prices(tmp_path, rank):
    from atom.compass.runtime.microbench import signature_of

    path = tmp_path / f"prices.tp{rank}.json"
    path.write_text(json.dumps({
        "provenance": {"topology": {"tp": rank}, "registration": "unregistered"},
        "prices": {signature_of(op): {"name": op["name"],
                                      "seconds": BODY_SECONDS,
                                      "occurrences": 1,
                                      "kernels": {"k0": BODY_SECONDS}}
                   for op in OPS},
        "unpriced": {},
    }), encoding="utf-8")
    return str(path)


def _options(tmp_path, width, regions="source-27b-tp1-prefill-seqs",
             block_size=16):
    for rank in range(width):
        _prices(tmp_path, rank)
        _template(tmp_path, rank, width)
    return {"price": str(tmp_path / "prices.json"),
            "template": str(tmp_path / "graph.json"),
            "tp": width, "derive": 0, "require_complete": 0,
            "block_size": block_size, "regions": regions,
            "rank_coords": {"tp": 0}}


def _group(tmp_path, width, **kwargs):
    from atom.compass.runtime.source_oracle import build_source_group

    return build_source_group(**_options(tmp_path, width, **kwargs))


class TestEveryWidthBuildsAndPrices:
    """Through the real factory, not through the preset alone. `regions=none`
    was reached at build time, so the fix has to be checked at build time."""

    @pytest.mark.parametrize("width", WIDTHS)
    def test_the_group_builds_with_this_profile_named(self, tmp_path, width):
        group = _group(tmp_path, width)

        assert group.oracle is not None

    @pytest.mark.parametrize("width", WIDTHS)
    def test_the_step_carries_both_regions(self, tmp_path, width):
        oracle = _group(tmp_path, width).oracle
        target = (oracle if width == 1 else oracle.oracle_for(0))

        cost = target.estimate(_decode(tp=width))

        assert "<prepare>" in cost.breakdown
        assert "<postprocess>" in cost.breakdown

    @pytest.mark.parametrize("width", (2, 4))
    def test_a_wide_step_costs_a_narrow_one_plus_the_broadcast(
            self, tmp_path, width):
        """The whole point of the widening: the wide cell is no longer body
        plus head with the runner's work deleted."""
        narrow = _group(tmp_path, 1).oracle.estimate(_decode(tp=1))
        wide = _group(tmp_path, width).oracle.oracle_for(0).estimate(
            _decode(tp=width))

        assert wide.breakdown["<tp-broadcast>"] == SEQS.tp_broadcast.seconds
        assert wide.seconds - narrow.seconds == pytest.approx(
            SEQS.tp_broadcast.seconds, rel=1e-9)

    @pytest.mark.parametrize("width", (2, 4))
    def test_every_rank_of_the_group_is_priced_with_the_regions(
            self, tmp_path, width):
        group = _group(tmp_path, width)

        for rank in range(width):
            cost = group.oracle.oracle_for(rank).estimate(
                _decode(tp=width, rank=rank))
            assert cost.breakdown["<prepare>"] > 0, rank
            assert cost.breakdown["<tp-broadcast>"] > 0, rank


class TestTheRecordSaysWhichDomainWasSelected:

    @pytest.mark.parametrize("width", WIDTHS)
    def test_the_snapshot_records_the_widened_domain(self, tmp_path, width):
        from atom.compass.runtime.source_oracle import build_source_oracle

        built = build_source_oracle(**_options(tmp_path, width))

        taken = built.oracle.compass_region_snapshot
        assert taken["version"] == SEQS.version
        assert taken["parameters"]["topologies"] == [1, 2, 4]

    def test_a_build_that_would_measure_different_work_is_refused(
            self, tmp_path):
        """The exemption is configuration-conditional, and the configuration
        is checked rather than assumed."""
        from atom.compass.runtime.source_oracle import build_source_oracle

        with pytest.raises(ValueError) as caught:
            build_source_oracle(**_options(tmp_path, 2, block_size=256))

        assert "num_key_value_heads // tp" in str(caught.value)

    def test_the_same_configuration_builds_at_tp1(self, tmp_path):
        from atom.compass.runtime.source_oracle import build_source_oracle

        built = build_source_oracle(**_options(tmp_path, 1, block_size=256))

        assert built.oracle is not None


def test_the_pooled_sentinel_is_untouched_by_the_widening():
    """A guard on the import above: these tests would pass vacuously if the
    pooled group had been removed rather than widened."""
    keys = {key for key, _ in SEQS.prepare_prefill_anchors}
    assert any(seqs == POOLED_SEQS for seqs, _tokens, _out in keys)
