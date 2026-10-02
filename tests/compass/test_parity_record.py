# SPDX-License-Identifier: MIT
"""The per-step record, taken off ATOM's own scheduler, and the first-divergence compare.

Each run drives the real `Scheduler` and the simulated runner's `forward` in the
engine's order -- schedule, forward, postprocess -- one scheduler per DP rank,
with every prompt distinct.
"""

import json
import queue
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import MockConfig

from atom.compass.parity import ENV, StepRecording, compare, read, request_key
from atom.compass.runner.overrides import NonAllocatingRunner
from atom.model_engine.scheduler import ScheduledBatch, Scheduler
from atom.model_engine.sequence import Sequence, SequenceType
from atom.sampling_params import SamplingParams

PROMPT, BUDGET, BLOCK = 200, 64, 16


def _prompt(i):
    return list(range(5 + 1000 * i, 5 + 1000 * i + PROMPT))


def _config(dp_rank=0, pipeline_parallel_size=1):
    return MockConfig(
        max_num_seqs=8,
        num_kvcache_blocks=4096,
        kv_cache_block_size=BLOCK,
        max_model_len=2048,
        max_num_batched_tokens=BUDGET,
        pipeline_parallel_size=pipeline_parallel_size,
        parallel_config=SimpleNamespace(data_parallel_rank=dp_rank),
    )


class _Runner(NonAllocatingRunner):
    """The simulated runner's `forward`, with the attributes it reads."""

    def __init__(self, config, rank=0):
        self.config, self.rank = config, rank


def _drive(directory, monkeypatch, ranks, prompt=_prompt):
    """Run each DP rank's requests to completion; return its batches and sequences."""
    monkeypatch.setenv(ENV, str(directory))
    built = {}
    for dp_rank, requests in ranks.items():
        config = _config(dp_rank)
        scheduler, runner = Scheduler(config), _Runner(config)
        sequences = [
            Sequence(
                prompt(i),
                BLOCK,
                sampling_params=SamplingParams(max_tokens=3, ignore_eos=True),
            )
            for i in requests
        ]
        for sequence in sequences:
            scheduler.add(sequence)
        batches = []
        while scheduler.has_unfinished_requests():
            assert len(batches) < 100, "the driven sequence does not terminate"
            scheduled = scheduler.schedule()
            if scheduled is None or scheduled[0] is None or not scheduled[0].req_ids:
                continue
            batch, seqs = scheduled
            batches.append(batch)
            reply = runner.forward(batch)
            scheduler.postprocess(
                list(seqs.values()),
                reply,
                stream_output_queue=queue.Queue(),
                batch=batch,
            )
        built[dp_rank] = (batches, {s.id: s for s in sequences}, runner)
    return built


@pytest.mark.parametrize(
    "length, first_rows",
    # At 20 tokens both prompts open in one batch, the second at row 1, and no
    # chunk is a middle one.
    [(PROMPT, [0, 0]), (20, [0, 1])],
)
def test_the_record_is_the_batches_the_scheduler_built(
    tmp_path, monkeypatch, length, first_rows
):
    # An earlier run's file in the same directory is replaced, not appended to.
    _drive(tmp_path, monkeypatch, {0: [2]})
    built = _drive(tmp_path, monkeypatch, {0: [0, 1]}, lambda i: _prompt(i)[:length])
    batches, sequences, runner = built[0]
    keys, expected, opened_at = {}, [], []
    for step, batch in enumerate(batches):
        rows = []
        for i, req_id in enumerate(batch.req_ids):
            start = batch.num_cached_tokens[i]
            window = sequences[req_id].token_ids[
                start : start + batch.num_scheduled_tokens[i]
            ]
            if req_id not in keys:
                opened_at.append(i)
            key = keys.setdefault(req_id, request_key(window))
            rows.append(
                [key, int(batch.num_scheduled_tokens[i]), int(batch.context_lens[i])]
            )
        expected.append(
            {
                "step": step,
                "dp_rank": 0,
                "is_dummy_run": False,
                "produces_output": batch.produces_output(),
                "batch": rows,
            }
        )
    assert len(set(keys.values())) == 2 and opened_at == first_rows
    assert any(not step["produces_output"] for step in expected) == (length == PROMPT)
    assert read(tmp_path) == {0: expected}

    # The dummy batch `ModelRunner.dummy_execution` builds for DP synchronisation.
    seq = Sequence([0], block_size=BLOCK, id=-1)
    seq.type = SequenceType.DECODE
    runner.forward(
        ScheduledBatch(
            seqs={seq.id: seq},
            num_scheduled_tokens=np.array([1], dtype=np.int32),
            total_tokens_num=1,
            total_tokens_num_decode=1,
            total_seqs_num=1,
            total_seqs_num_decode=1,
            is_dummy_run=True,
        )
    )
    assert read(tmp_path)[0][-1] == {
        "step": len(batches),
        "dp_rank": 0,
        "is_dummy_run": True,
        "produces_output": True,
        "batch": [],
    }


def test_two_records_of_one_sequence_compare_equal(tmp_path, monkeypatch):
    _drive(tmp_path / "real", monkeypatch, {0: [0, 1], 1: [2, 3]})
    _drive(tmp_path / "simulated", monkeypatch, {0: [0, 1], 1: [2, 3]})
    report = compare(tmp_path / "real", tmp_path / "simulated")
    assert report["first_divergence"] == {0: None, 1: None}
    ranks = report["request_dp_rank"]
    assert ranks["real"] == ranks["simulated"]
    assert sorted(ranks["real"].values()) == [0, 0, 1, 1]


def test_a_request_moved_to_the_other_rank_names_its_first_step(tmp_path, monkeypatch):
    real = _drive(tmp_path / "real", monkeypatch, {0: [0, 1], 1: [2, 3]})
    _drive(tmp_path / "simulated", monkeypatch, {0: [0], 1: [1, 2, 3]})
    report = compare(tmp_path / "real", tmp_path / "simulated")
    real_record = read(tmp_path / "real")
    first = report["first_divergence"]
    # Rank 0 parts where request 1 is first scheduled on the real side.
    moved = next(
        step for step, b in enumerate(real[0][0]) if any(r == 1 for r in b.req_ids)
    )
    assert first[0]["step"] == moved
    assert first[0]["real"] == real_record[0][moved]
    assert first[1]["step"] == 0
    ranks = report["request_dp_rank"]
    assert [
        (ranks["real"][key], ranks["simulated"][key])
        for key in ranks["real"]
        if ranks["real"][key] != ranks["simulated"][key]
    ] == [(0, 1)]


def test_a_shifted_chunk_boundary_names_its_first_step(tmp_path, monkeypatch):
    _drive(tmp_path / "real", monkeypatch, {0: [0, 1], 1: [2, 3]})
    shutil.copytree(tmp_path / "real", tmp_path / "simulated")
    path = tmp_path / "simulated" / "dp0.jsonl"
    steps = read(tmp_path / "simulated")[0]
    # Request 0's first two chunks, 64 and 64, become 60 and 68.
    assert [s["batch"][0][1:] for s in steps[:2]] == [[64, 64], [64, 128]]
    steps[0]["batch"][0][1:] = [60, 60]
    steps[1]["batch"][0][1:] = [68, 128]
    path.write_text("".join(json.dumps(s) + "\n" for s in steps))
    report = compare(tmp_path / "real", tmp_path / "simulated")
    assert report["first_divergence"][1] is None
    first = report["first_divergence"][0]
    assert first["step"] == 0
    assert first["real"]["batch"][0][1:] == [64, 64]
    assert first["simulated"]["batch"][0][1:] == [60, 60]


def test_two_requests_with_one_first_window_are_refused(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="first scheduled with the same tokens"):
        _drive(tmp_path, monkeypatch, {0: [0, 1]}, prompt=lambda i: _prompt(0))


def test_one_key_on_two_ranks_is_refused(tmp_path, monkeypatch):
    _drive(tmp_path / "real", monkeypatch, {0: [0], 1: [0]})
    _drive(tmp_path / "simulated", monkeypatch, {0: [0]})
    with pytest.raises(ValueError, match="the real record schedules request"):
        compare(tmp_path / "real", tmp_path / "simulated")


def test_an_empty_record_is_refused(tmp_path, monkeypatch):
    _drive(tmp_path / "real", monkeypatch, {0: [0]})
    (tmp_path / "simulated").mkdir()
    with pytest.raises(ValueError, match="the simulated record has no steps"):
        compare(tmp_path / "real", tmp_path / "simulated")


@pytest.mark.parametrize("env, rank", [(None, 0), ("out", 1)])
def test_only_tp_rank_0_records_and_only_when_asked(tmp_path, monkeypatch, env, rank):
    batches = _drive(tmp_path / "source", monkeypatch, {0: [0]})[0][0]
    if env is None:
        monkeypatch.delenv(ENV)
    else:
        monkeypatch.setenv(ENV, str(tmp_path / env))
    _Runner(_config(), rank).forward(batches[0])
    assert not (tmp_path / "out").exists()


def test_pipeline_parallelism_is_refused(tmp_path, monkeypatch):
    batches = _drive(tmp_path / "source", monkeypatch, {0: [0]})[0][0]
    runner = _Runner(_config(pipeline_parallel_size=2))
    with pytest.raises(ValueError, match="each stage would write it"):
        runner.forward(batches[0])


def test_the_mixin_records_from_the_first_forward_after_construction(
    tmp_path, monkeypatch
):
    batches = _drive(tmp_path / "source", monkeypatch, {0: [0]})[0][0]

    class _Base:
        """`ModelRunner`'s shape: construction runs a warmup forward."""

        def __init__(self, rank, config):
            self.rank, self.config = rank, config
            self.forward(batches[1])

        def forward(self, batch):
            return "ran"

    class _Recorded(StepRecording, _Base):
        pass

    monkeypatch.setenv(ENV, str(tmp_path / "out"))
    runner = _Recorded(0, _config())
    assert not (tmp_path / "out").exists()
    assert runner.forward(batches[0]) == "ran"
    assert [s["step"] for s in read(tmp_path / "out")[0]] == [0]
