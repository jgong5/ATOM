"""A replay that sends part of a trace must not be able to claim the whole one.

`--num-requests` defaulted to 64 on both paths. For a synthetic workload that
is a size; for a trace it is a silent edit. Measured: an 809-request cc-traces
replay sent 64 of them -- 55s of a 5726s timeline -- and wrote a manifest
naming the trace file and its sha256, with nothing to say only 7.9% of it ran.
The coverage and accuracy numbers taken off that run were read for hours as
facts about the trace.

So the default is the whole trace now, and truncation is recorded when asked
for.
"""
import argparse
import importlib.util
import json
from pathlib import Path


def _module():
    spec = importlib.util.spec_from_file_location(
        "replay_trunc_mod",
        Path(__file__).resolve().parents[2] / "scripts/compass/replay.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _trace(tmp_path, n):
    path = tmp_path / "trace.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        for i in range(n):
            fh.write(json.dumps({"arrival_s": float(i), "input_tokens": 640,
                                 "output_tokens": 16, "session": i % 4}) + "\n")
    return path


def _args(**kw):
    base = dict(trace=None, num_requests=None, input_tokens=128,
                output_tokens=32, seed=0, rate=0.0, time_scale=1.0)
    base.update(kw)
    return argparse.Namespace(**base)


class TestATraceIsReplayedWhole:
    def test_no_num_requests_means_every_row(self, tmp_path):
        mod = _module()
        args = _args(trace=str(_trace(tmp_path, 809)))
        assert len(mod._workload(args)) == 809

    def test_the_row_count_on_disk_is_recorded(self, tmp_path):
        mod = _module()
        args = _args(trace=str(_trace(tmp_path, 809)))
        mod._workload(args)
        # The manifest reports this beside what ran; the sha256 names the whole
        # file either way, so this is the only thing that can contradict it.
        assert args.trace_rows == 809


class TestTruncationIsStillPossibleAndStillVisible:
    def test_num_requests_truncates(self, tmp_path):
        mod = _module()
        args = _args(trace=str(_trace(tmp_path, 809)), num_requests=64)
        assert len(mod._workload(args)) == 64
        assert args.trace_rows == 809

    def test_it_keeps_the_earliest_arrivals(self, tmp_path):
        mod = _module()
        args = _args(trace=str(_trace(tmp_path, 100)), num_requests=5)
        rows = mod._workload(args)
        # Sorted by arrival before the cut, so a truncated run is a prefix of
        # the timeline rather than an arbitrary sample of it.
        assert [r["arrival_s"] for r in rows] == [0.0, 1.0, 2.0, 3.0, 4.0]


class TestTheSyntheticPathIsUnchanged:
    def test_it_still_defaults_to_sixty_four(self):
        mod = _module()
        assert len(mod._workload(_args())) == mod._SYNTHETIC_REQUESTS == 64

    def test_and_still_takes_an_explicit_count(self):
        mod = _module()
        assert len(mod._workload(_args(num_requests=7))) == 7
