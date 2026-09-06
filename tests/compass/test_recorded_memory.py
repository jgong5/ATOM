"""Sizing a configuration from a record rather than from a device.

A configuration has to exist before it can be sized, which is why the 27B at
TP=1 could not be evaluated at all -- it failed to fit before any timing could
be taken. Reading the five readings back removes that, for configurations some
run has taken them for.
"""

import json

import pytest

from atom.compass.core.memory import MemoryReadings, RecordedMemory

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
