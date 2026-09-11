# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The two rows that were reporting a number nobody had checked.

`graph pool` read `+0.0%` at TP=1, 2 and 4, and was cited that way. It could
not have read anything else: the derived side mirrors
`_estimate_cudagraph_overhead` and the recorded side *is* that estimator's
output, both computed from the same record. The rows are now split -- an
identity check on the mirror, and the term against the pool capture actually
reserved, which every record has carried since the terms were split.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RECORDS = Path(__file__).parent / "memory_records"


def _script():
    """`validate_memory.py` as a module. It is a script, and lives in
    `scripts/`, so there is no package to import it from."""
    spec = importlib.util.spec_from_file_location(
        "compass_validate_memory", ROOT / "scripts/compass/validate_memory.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record(name: str) -> dict:
    with open(RECORDS / name, encoding="utf-8") as fh:
        return json.load(fh)


def test_the_pool_measurement_is_read_off_the_record():
    """No `--log` needed: the run wrote the measurement into the record.

    The reserved delta is the term -- a captured graph pins its intermediates,
    so what the allocator had to create is what the configuration must budget
    for -- and the allocated delta is beside it because the difference between
    them is segment bookkeeping rather than pinned memory.
    """
    reserved, allocated, sizes = _script().recorded_pool(_record("27b.tp1.memory.json"))
    assert reserved == 127926272
    assert allocated == 110981120
    assert sizes == (1, 2, 4, 8, 16, 32)


def test_a_record_without_a_pool_measurement_says_so_rather_than_zero_out():
    reserved, allocated, sizes = _script().recorded_pool({"readings": {}})
    assert (reserved, allocated, sizes) == (0, 0, ())


def test_the_old_graph_pool_row_was_an_identity():
    """Both sides of the row it replaced are the same arithmetic.

    Pinned so the claim is checkable rather than asserted in a comment: the
    engine's estimate in the record equals `0.2 x (peak_torch -
    current_torch)`, which is exactly what the derived side computed from the
    same two readings.
    """
    module = _script()
    for name in (
        "27b.tp1.memory.json",
        "27b.tp2.rank0.memory.json",
        "27b.tp4.rank0.memory.json",
    ):
        got = _record(name)["readings"]
        warmup_act = got["peak_torch"] - got["current_torch"]
        assert module.graph_pool_bytes(warmup_act) == got["cudagraph_overhead"]


@pytest.mark.parametrize(
    "name,reserved,estimate_over",
    [
        ("27b.tp1.memory.json", 127926272, True),
        ("27b.tp2.rank0.memory.json", 106954752, True),
        ("27b.tp4.rank0.memory.json", 85983232, True),
    ],
)
def test_the_engines_estimate_is_over_on_this_model_and_was_under_on_the_other(
    name, reserved, estimate_over
):
    """The estimator's error changes sign with the model, so it is not a bias.

    `0.2 x peak activations` scales with the model; the pool does not. On the
    0.6B the estimate was 19x under the pool, and on the 27B -- whose warmup
    activations are 2.75 GB -- it is 4.6x over. A constant correction cannot
    fix a term that is wrong in both directions, which is why
    `measured_graph_pool_bytes` models the pool instead of correcting the
    estimate.
    """
    got = _record(name)["readings"]
    assert _script().recorded_pool(_record(name))[0] == reserved
    assert (got["cudagraph_overhead"] > reserved) is estimate_over


def test_the_pinned_bytes_are_the_same_on_both_models_above_width_one():
    """76.0 MiB allocated at TP=2 and TP=4 on the 27B -- the 0.6B's number.

    `DEFAULT_POOL_SHARDED`'s claim is that capture above width one pins a fixed
    set that neither shards nor grows with the ladder, measured as 79 692 800
    bytes on the 0.6B. The 27B, a different model with 64 layers against 28,
    allocates the same count to the byte, which is the first evidence for that
    claim from outside the campaign it was fitted on.
    """
    for name in ("27b.tp2.rank0.memory.json", "27b.tp4.rank0.memory.json"):
        assert _script().recorded_pool(_record(name))[1] == 79692800


def test_a_row_on_another_record_is_not_accused_of_tampering():
    """The note the exclusive record's rows were carrying, wrongly.

    `persistent` and `load_residue` were fitted on the historical TP=1 record.
    Reading the *exclusive* capture, every one of their rows printed `bytes
    altered` -- because the hash differed, which for a different file it must.
    The record is intact; it is simply not that record. Integrity now waits
    for a producer to claim otherwise.
    """
    import dataclasses

    from atom.compass.core.execution_id import (EXECUTION_SCHEMA, ID_INPUTS,
                                                derive_execution_id)
    from atom.compass.core.memory_calibration import (SourceCalibration,
                                                      for_model, producer_key)

    script = _script()
    calib = for_model("Qwen/Qwen3.8-27B")
    config = _record("27b.tp1.exclusive.memory.json")["config"]
    foreign_sha = "0" * 64

    note = script._cal_note(calib, calib.mapping(1), config, foreign_sha, None,
                            "persistent", "held through the allocator")
    assert "residual (run unidentified)" in note
    assert "altered" not in note and "re-serialised" not in note

    # A legacy block naming a host and a pid is not an execution identity, so
    # it cannot promote the note either.
    legacy = {"host": "hjbog-srdc-18", "pid": 695009,
              "started_at": "2026-09-11T08:55:40Z"}
    assert producer_key(legacy) is None

    # And the note that *is* worth printing, once a run claims to be the fit.
    inputs = {"host": "hjbog-srdc-18", "cell": "RESULTS/tp1_source",
              "side": "real", "repeat": 1, "server_pid": 695009,
              "launched_at_ns": 1757580940000000000}
    run = {"execution": {
        "schema": EXECUTION_SCHEMA,
        "execution_id": derive_execution_id(*[inputs[n] for n in ID_INPUTS]),
        "id_inputs": inputs}}
    entry = calib.terms["persistent"]
    identified = SourceCalibration(
        model=calib.model,
        terms={"persistent": dataclasses.replace(
            entry,
            run=dataclasses.replace(entry.run, producers=(producer_key(run),)),
        )},
    )
    note = script._cal_note(identified, identified.mapping(1), config,
                            foreign_sha, run, "persistent", "held")
    assert "re-serialised since it was fitted" in note


def test_the_deep_chunk_is_not_the_warmup_step_even_at_the_same_token_count():
    """The match the token total cannot make.

    `warmup_model` runs fresh sequences: every token scheduled, nothing
    cached. `s27prefdeep` runs the same 16 384 tokens with 98 304 behind them
    -- 7x the KV to read, a different attention branch, and an identical
    `batch_signature`. Scaling its walk to the warmup peak would compare two
    different steps and book the difference as model error.
    """
    script = _script()
    config = {"max_model_len": 262144, "max_num_seqs": 32}

    assert script.warmup_shape(config, 16384) == ((16384,), (16384,))
    assert script.warmup_tokens(config, 16384) == 16384

    def spec(context):
        return {"key": {"batch_signature": [16384]},
                "provenance": {"batch_spec": {"query_lens": [16384],
                                              "context_lens": [context]}}}

    assert script.warmup_mismatch(spec(16384), config, 16384) == ""
    assert "98304" in script.warmup_mismatch(spec(114688), config, 16384)


def test_a_trace_at_another_size_is_still_the_warmup_step(): 
    """Scaling across token counts is the claim, not a mismatch.

    The 0.6B's 3494-token trace reaching its independently measured 4096-token
    warmup peak is the one held-out result the activation term has. A matcher
    strict enough to reject the deep chunk must not reject that.
    """
    script = _script()
    config = {"max_model_len": 40960, "max_num_seqs": 256}
    cold = {"key": {"batch_signature": [3494]},
            "provenance": {"shape": {"num_scheduled_tokens": [3494],
                                     "context_lens": [3494]}}}
    assert script.warmup_shape(config, 4096) == ((4096,), (4096,))
    assert script.warmup_mismatch(cold, config, 4096) == ""


def test_an_unlabelled_graph_is_matched_on_tokens_and_admits_it():
    script = _script()
    config = {"max_model_len": 262144, "max_num_seqs": 32}
    bare = {"key": {"batch_signature": [16384]}}
    assert "tokens are all there is to match" in script.warmup_mismatch(
        bare, config, 16384)
