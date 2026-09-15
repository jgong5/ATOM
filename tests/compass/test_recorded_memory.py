"""Sizing a configuration from a record rather than from a device.

A configuration has to exist before it can be sized, which is why the 27B at
TP=1 could not be evaluated at all -- it failed to fit before any timing could
be taken. Reading the five readings back removes that, for configurations some
run has taken them for.
"""

import hashlib
import json

import pytest

from atom.compass.core.memory import (
    RECORD_ROLE,
    MemoryReadings,
    RecordedMemory,
)

CONFIG = {
    "model": "Qwen/Qwen3-0.6B",
    "gpu_memory_utilization": 0.9,
    "max_num_seqs": 512,
    "max_model_len": 40960,
    "kv_cache_dtype": "bf16",
    "block_size": 16,
    "topology": {"tp": 1},
    "rank_coords": {"tp": 0},
}


def _record(tmp_path, name="m.json", **overrides):
    readings = {"total": 200 * 2**30, "free": 190 * 2**30,
                "peak_torch": 2 * 2**30, "non_torch": 2**30,
                "cudagraph_overhead": 2**26}
    readings.update(overrides)
    path = tmp_path / name
    path.write_text(json.dumps({"version": 1, "readings": readings,
                                "config": CONFIG}))
    return str(path)


class TestWhatCountsAsTheSameConfiguration:
    def test_the_readings_come_back(self, tmp_path):
        got = RecordedMemory([_record(tmp_path)]).readings_for(CONFIG)
        assert got == MemoryReadings(200 * 2**30, 190 * 2**30, 2 * 2**30,
                                     2**30, 2**26)

    def test_a_different_rank_is_a_different_configuration(self, tmp_path):
        """At TP=2 the weights are halved and the collective buffers are not,
        so one rank's readings are not another's."""
        source = RecordedMemory([_record(tmp_path)])
        other = dict(CONFIG, rank_coords={"tp": 1})
        assert source.readings_for(other) is None
        assert "no record" in source.refusal(other)

    def test_so_is_a_different_utilization(self, tmp_path):
        source = RecordedMemory([_record(tmp_path)])
        assert source.readings_for(dict(CONFIG, gpu_memory_utilization=0.4)) is None


class TestRecordsThatDescribeTheBoxRatherThanTheConfiguration:
    """`free` is what the neighbours left. Every other term is the
    configuration's."""

    def test_a_budget_bound_record_is_reusable(self, tmp_path):
        source = RecordedMemory([_record(tmp_path)])
        assert source.refusal(CONFIG) is None

    def test_a_free_bound_one_is_refused(self, tmp_path):
        path = _record(tmp_path, free=10 * 2**30)
        source = RecordedMemory([path])
        assert source.readings_for(CONFIG) is not None, "still readable"
        assert "binding term" in source.refusal(CONFIG)

    def test_the_boundary_is_where_the_clamp_bites(self, tmp_path):
        readings = MemoryReadings(
            total=100, free=50, peak_torch=10, non_torch=5,
            cudagraph_overhead=1)
        # budget 90 - (10+5+1+2) = 72 > 50, so free binds
        assert readings.free_was_binding(0.9)
        # budget 40 - 18 = 22 < 50, so it does not
        assert not readings.free_was_binding(0.4)


class TestTermsThatWereOnceOneNumber:
    """`peak_torch` is weights, persistent buffers and peak activations summed.

    A budget derived from it can be right in total while both its terms are
    wrong, so the two readings that split it are recorded -- and a record
    written before they existed still has to load.
    """

    def test_a_record_predating_the_split_still_loads(self, tmp_path):
        path = _record(tmp_path, "old.json")
        readings = RecordedMemory([path]).readings_for(CONFIG)
        assert readings is not None
        assert readings.weights_torch is None
        assert readings.parameter_bytes is None
        assert readings.current_torch is None

    def test_a_record_that_has_them_carries_them_through(self, tmp_path):
        path = _record(tmp_path, "new.json", peak_torch=8, weights_torch=5,
                       parameter_bytes=4, current_torch=6)
        readings = RecordedMemory([path]).readings_for(CONFIG)
        assert (readings.weights_torch, readings.parameter_bytes,
                readings.current_torch) == (5, 4, 6)
        # The split is the reader's arithmetic, not a stored derivation.
        assert readings.peak_torch - readings.current_torch == 2


class TestNonTorchIsAlsoAPropertyOfTheBox:
    """`non_torch` is `(total - free) - reserved`, and `total - free` is
    device-wide, so a neighbour's memory is charged to this configuration --
    the same defect `free` has, unguarded because the number looks like a
    property of the process. On a shared box it stopped being a distorted
    measurement and became a failure to launch."""

    def test_a_plausible_reading_passes(self, tmp_path):
        path = _record(tmp_path, non_torch=6 * 2**30)
        assert RecordedMemory([path]).refusal(
            CONFIG, expected_non_torch=7 * 2**30) is None

    def test_a_reading_that_is_measuring_the_box_is_refused(self, tmp_path):
        """152 GB recorded where this rank had reserved 2.9 GB."""
        path = _record(tmp_path, non_torch=152 * 2**30)
        why = RecordedMemory([path]).refusal(CONFIG,
                                             expected_non_torch=7 * 2**30)
        assert why is not None and "device-wide" in why

    def test_no_expectation_means_no_opinion(self, tmp_path):
        """The guard needs a yardstick; without one it must not invent a
        refusal."""
        path = _record(tmp_path, non_torch=152 * 2**30)
        assert RecordedMemory([path]).refusal(CONFIG) is None

    def test_ranks_that_agree_report_no_spread(self, tmp_path):
        import json

        paths = []
        for rank in (0, 1):
            where = tmp_path / ("r%d.json" % rank)
            config = dict(CONFIG, topology={"tp": 2}, rank_coords={"tp": rank})
            where.write_text(json.dumps({
                "version": 1, "config": config,
                "readings": {"total": 100, "free": 90, "peak_torch": 8,
                             "non_torch": 6 * 2**30, "cudagraph_overhead": 0}}))
            paths.append(str(where))
        source = RecordedMemory(paths)
        want = dict(CONFIG, topology={"tp": 2}, rank_coords={"tp": 0})
        assert source.rank_disagreement(want) == 0

    def test_ranks_that_disagree_measure_the_contamination(self, tmp_path):
        """Ranks do the same work, so a spread is the neighbours arriving on
        some cards and not others -- 192 MiB at TP=4, 640 MiB at TP=8."""
        import json

        paths = []
        for rank, non_torch in ((0, 6 * 2**30), (1, 6 * 2**30 + 192 * 2**20)):
            where = tmp_path / ("r%d.json" % rank)
            config = dict(CONFIG, topology={"tp": 2}, rank_coords={"tp": rank})
            where.write_text(json.dumps({
                "version": 1, "config": config,
                "readings": {"total": 100, "free": 90, "peak_torch": 8,
                             "non_torch": non_torch, "cudagraph_overhead": 0}}))
            paths.append(str(where))
        want = dict(CONFIG, topology={"tp": 2}, rank_coords={"tp": 0})
        assert RecordedMemory(paths).rank_disagreement(want) == 192 * 2**20

    def test_one_rank_alone_cannot_say(self, tmp_path):
        assert RecordedMemory([_record(tmp_path)]).rank_disagreement(CONFIG) is None


class TestTheBytesARecordedBudgetWasServedFrom:
    """R13: a recorded budget can name the artifact it was served out of.

    The same contract the profile path already keeps. `--compass-memory-in`
    hands the runner five readings that go straight into a KV budget, so a run
    that serves them has to be able to say which bytes they were, digested at
    the read rather than restated from the path afterwards.
    """

    def test_each_record_read_is_recorded(self, tmp_path):
        read: list = []
        paths = [_record(tmp_path, "a.json"),
                 _record(tmp_path, "b.json", **{"free": 180 * 2**30})]
        RecordedMemory(paths, collect=read)
        assert [row.role for row in read] == [RECORD_ROLE, RECORD_ROLE]
        assert [row.path for row in read] == paths

    def test_the_digest_is_of_the_bytes_that_were_parsed(self, tmp_path):
        read: list = []
        path = _record(tmp_path)
        RecordedMemory([path], collect=read)
        want = hashlib.sha256(open(path, "rb").read()).hexdigest()
        assert read[0].sha256 == want
        # Rewriting the file afterwards does not move what was attested to.
        open(path, "w").write("{}")
        assert read[0].sha256 == want

    def test_collecting_is_optional(self, tmp_path):
        assert RecordedMemory([_record(tmp_path)]).readings_for(CONFIG)
