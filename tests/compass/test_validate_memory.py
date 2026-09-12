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

import hashlib
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


def _attest(profile, **extra) -> dict:
    """What a run would have published about a profile it read.

    `attested` is None here, which is the unattested case -- named on the
    command line, not taken out of a manifest -- so these stay about the term
    under test rather than about the digest checks.
    """
    attest = {"budget_source": "", "profile": str(profile),
              "profile_sha256": "", "attested": None, "lineage": {},
              "num_kvcache_blocks": None, "kind": None}
    attest.update(extra)
    return attest


def _servable_profile(tmp_path, width: int = 1):
    """A profile complete enough that `derived_readings` will answer.

    Not a stub: `derived_readings` fails closed on every missing term, so
    anything short of a real profile refuses before the checks under test are
    reached. The numbers are the 27B's, at the shape the emitter writes.
    """
    calibration = tmp_path / ("calibration.tp%d.json" % width)
    calibration.write_text(json.dumps({
        "persistent": 252339712,
        "non_torch": {str(width): 1157627904},
        "load_residue": {str(width): 14924832},
        "provenance": {
            "persistent": "S27: 27B full engine, the source config",
            "non_torch": "S27: 27B full engine, the source config",
            "load_residue": "S27: 27B full engine, the source config",
        },
    }))
    profile = tmp_path / ("profile.tp%d.json" % width)
    profile.write_text(json.dumps({
        "total": 206141652992,
        "world_size": width,
        "parameters": 55000000000 // width,
        "buffers": 33554432,
        "model_config": str(RECORDS / "qwen3_5_27b.config.json"),
        "compile_mode": "inductor",
        "calibration": str(calibration),
        "provenance": {"model": "Qwen/Qwen3.8-27B"},
    }))
    return profile, calibration


def _digests(*paths) -> dict:
    """What a run's manifest would say about the files a derivation reads.

    The model config is in here too: it is an input to the activation term
    exactly as the calibration is, and the loader refuses anything the
    manifest does not name.
    """
    every = list(paths) + [RECORDS / "qwen3_5_27b.config.json"]
    return {str(path): hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for path in every}


def _plan_entries(script, blob, config_path) -> int:
    """The block count a plan from this record's readings arrives at.

    The same call `kv_rows` makes, so a test can say what the modelled count
    is without restating ATOM's planner.
    """
    with open(config_path, encoding="utf-8") as fh:
        native = json.load(fh)
    config, readings = blob["config"], blob["readings"]
    plan = script.blocks_from_readings(
        native,
        script.MemoryReadings(
            total=int(readings["total"]), free=int(readings["free"]),
            peak_torch=int(readings["peak_torch"]),
            non_torch=int(readings["non_torch"]),
            cudagraph_overhead=int(readings["cudagraph_overhead"])),
        utilization=float(config["gpu_memory_utilization"]),
        max_num_seqs=int(config["max_num_seqs"]),
        tensor_parallel=1, block_size=int(config["block_size"]),
        kv_dtype_bytes=script.KV_DTYPE_BYTES.get(
            str(config.get("kv_cache_dtype")), 2))
    return int(plan.paged_entries)


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


class TestTheGateAcceptanceRunsOn:
    """What `--gate` has to refuse.

    The comparison prints per-term errors and, until now, always exited 0. A
    report nobody can fail is a report, not a gate, and the thing acceptance
    needs is the second one: every non-KV term within 10%, the block count
    within 5%, and a term nothing compared treated as a failure rather than as
    a blank.
    """

    def test_every_term_inside_its_threshold_passes(self):
        script = _script()
        script.SEEN.update({name: 1.0 for name in script.REQUIRED_TERMS})
        assert script.gate("record") is True

    def test_a_term_nothing_compared_is_a_failure_not_a_blank(self):
        """The shape of error this project has been caught by twice.

        A missing row and a passing row look identical in a total. Here the
        uncompared term fails on its own.
        """
        script = _script()
        script.SEEN.update({name: 1.0 for name in script.REQUIRED_TERMS})
        del script.SEEN["non-torch"]
        assert script.gate("record") is False

    def test_the_block_count_is_held_to_the_tighter_threshold(self):
        """7% is a passing term and a failing capacity."""
        script = _script()
        script.SEEN.update({name: 7.0 for name in script.REQUIRED_TERMS
                            if name != "kv blocks"})
        script.SEEN["kv blocks"] = 7.0
        assert script.gate("record") is False
        script.SEEN["kv blocks"] = 4.0
        assert script.gate("record") is True

    def test_the_worst_of_two_rows_for_a_term_is_the_one_gated(self):
        """`activations` prints twice where a graph was traced."""
        script = _script()
        script.note_term("activations", 2.0)
        script.note_term("activations", -30.0)
        assert script.SEEN["activations"] == -30.0

    def test_a_block_count_planned_from_the_record_is_not_coverage(self):
        """Run off the recorded readings the row is an identity.

        The engine planned from exactly those five numbers, so the derivation
        reproduces its count and the row reads +0.00% on every record ever
        written. Counting that as the KV term would pass the 5% gate without
        testing the model at all, so it is left uncovered instead.
        """
        script = _script()
        blob = _record("27b.tp1.memory.json")
        script.kv_rows(blob["config"], blob["readings"], 1, 1, blob,
                       str(RECORDS / "qwen3_5_27b.config.json"))
        assert "kv blocks" not in script.SEEN

    def test_a_profile_for_another_model_is_refused(self, tmp_path):
        """Not a failing comparison -- two unrelated runs.

        The 27B's parameters against the 0.6B's allocator read is +4560%, and
        a gate that reported that as a model error would be reporting on a
        comparison nobody made.
        """
        script = _script()
        profile = tmp_path / "profile.json"
        profile.write_text(json.dumps(
            {"provenance": {"model": "Qwen/Qwen3.8-27B"}}))
        with pytest.raises(SystemExit) as raised:
            script.predicted_terms(_attest(profile),
                                   {"model": "Qwen/Qwen3-0.6B"}, 16384, 1)
        assert "two different models" in str(raised.value)

    def test_the_profile_compared_is_the_one_the_run_read(self, tmp_path):
        """Taken out of what the run published, not off this script's
        command line: the second is a claim about a path made later."""
        script = _script()
        source = tmp_path / "budget_source.json"
        source.write_text(json.dumps({
            "kind": "source-derived",
            "inputs": {"inputs": [
                {"role": "runtime.replay_target", "path": "/t.json",
                 "sha256": "aa"},
                {"role": script.PROFILE_ROLE, "path": "/p.json",
                 "sha256": "bb"}]}}))
        attest = script.profile_from_budget_source(str(source))
        assert (attest["profile"], attest["profile_sha256"]) == ("/p.json", "bb")

    def test_a_budget_no_profile_sized_has_nothing_to_compare(self, tmp_path):
        script = _script()
        source = tmp_path / "budget_source.json"
        source.write_text(json.dumps(
            {"kind": "device-measured", "inputs": {"inputs": []}}))
        with pytest.raises(SystemExit) as raised:
            script.profile_from_budget_source(str(source))
        assert "not sized from a profile" in str(raised.value)

    def test_a_profile_that_moved_under_the_comparison_is_refused(self, tmp_path):
        script = _script()
        profile = tmp_path / "profile.json"
        profile.write_text(json.dumps({"provenance": {}}))
        with pytest.raises(SystemExit) as raised:
            script.predicted_terms(_attest(profile, profile_sha256="0" * 64),
                                   {}, 16384, 1)
        assert "not the profile the run read" in str(raised.value)

    def test_every_attested_input_is_kept_not_just_the_profile(self, tmp_path):
        """The calibration is where every collective constant lives.

        Pulling out the profile row and dropping the rest left the one file
        whose digest was checked as the one file that carries no numbers.
        """
        script = _script()
        source = tmp_path / "budget_source.json"
        source.write_text(json.dumps({
            "kind": "source-derived",
            "inputs": {"inputs": [
                {"role": script.PROFILE_ROLE, "path": "/p.json",
                 "sha256": "bb"},
                {"role": "runtime.memory_model.calibration", "path": "/c.json",
                 "sha256": "cc"},
                {"role": "runtime.memory_model.model_config", "path": "/m.json",
                 "sha256": "dd"}]}}))
        attest = script.profile_from_budget_source(str(source))
        assert attest["attested"] == {"/p.json": "bb", "/c.json": "cc",
                                      "/m.json": "dd"}

    def test_the_budget_is_read_out_of_the_artifact_that_saved_it(
            self, tmp_path):
        """Nothing writes a standalone budget file, and nothing should.

        The runner publishes the object; provenance saves it nested inside the
        per-rank artifact. The readback reads that same object rather than
        anyone adding a second writer and a second schema for the same bytes.
        """
        script = _script()
        budget = {"kind": "source-derived",
                  "inputs": {"inputs": [{"role": script.PROFILE_ROLE,
                                         "path": "/p.json", "sha256": "bb"}]}}
        saved = tmp_path / "provenance.modelled.r0.json"
        saved.write_text(json.dumps(
            {"compass": {"loaded_inputs": {"ranks": [{"budget_source": budget}]}}}))
        attest = script.profile_from_budget_source(str(saved))
        assert attest["profile"] == "/p.json"

    def test_the_other_saved_spelling_is_read_too(self, tmp_path):
        script = _script()
        budget = {"kind": "source-derived",
                  "inputs": {"inputs": [{"role": script.PROFILE_ROLE,
                                         "path": "/p.json", "sha256": "bb"}]}}
        saved = tmp_path / "modelled.r0.json"
        saved.write_text(json.dumps({"run": {"server": {"compass": {
            "loaded_inputs": {"ranks": [{"budget_source": budget}]}}}}}))
        assert script.profile_from_budget_source(str(saved))["profile"] \
            == "/p.json"

    def test_one_physical_record_carries_the_whole_width(self, tmp_path):
        """`ranks[i]` is a physical predictor record, not a TP rank.

        A GPU-free replay writes one record and that record holds the budget
        for all four ranks. Reading it as "rank 0's budget" and then wanting
        three more would be asking for evidence nobody wrote.
        """
        script = _script()
        budget = {"kind": "source-derived",
                  "lineage": {"world_size": 4},
                  "inputs": {"inputs": [{"role": script.PROFILE_ROLE,
                                         "path": "/p4.json", "sha256": "bb"}]}}
        saved = tmp_path / "provenance.modelled.r0.json"
        saved.write_text(json.dumps(
            {"compass": {"loaded_inputs": {"ranks": [{"budget_source": budget}]}}}))
        attest = script.profile_from_budget_source(str(saved), world=4)
        assert attest["lineage"]["world_size"] == 4

    def test_an_artifact_holding_several_budgets_is_not_guessed_at(
            self, tmp_path):
        script = _script()

        def entry(width):
            return {"budget_source": {
                "kind": "source-derived", "lineage": {"world_size": width},
                "inputs": {"inputs": [
                    {"role": script.PROFILE_ROLE,
                     "path": "/p%d.json" % width, "sha256": "bb"}]}}}

        saved = tmp_path / "provenance.modelled.r0.json"
        saved.write_text(json.dumps({"compass": {"loaded_inputs": {
            "ranks": [entry(2), entry(4)]}}}))
        with pytest.raises(SystemExit) as raised:
            script.profile_from_budget_source(str(saved))
        assert "will not guess" in str(raised.value)
        picked = script.profile_from_budget_source(str(saved), world=4)
        assert picked["profile"] == "/p4.json"

    def test_an_artifact_that_saved_no_budget_is_refused(self, tmp_path):
        script = _script()
        saved = tmp_path / "provenance.modelled.r0.json"
        saved.write_text(json.dumps(
            {"compass": {"loaded_inputs": {"ranks": [{"inputs": []}]}}}))
        with pytest.raises(SystemExit) as raised:
            script.profile_from_budget_source(str(saved))
        assert "no entry in it saved a budget_source" in str(raised.value)

    def test_one_path_attested_at_two_digests_is_refused(self, tmp_path):
        """One of the two reads is the one being compared, and there is no
        way to tell which."""
        script = _script()
        source = tmp_path / "budget_source.json"
        source.write_text(json.dumps({
            "kind": "source-derived",
            "inputs": {"inputs": [
                {"role": script.PROFILE_ROLE, "path": "/p.json",
                 "sha256": "bb"},
                {"role": "runtime.memory_model.model_config", "path": "/m.json",
                 "sha256": "dd"},
                {"role": "runtime.memory_model.model_config", "path": "/m.json",
                 "sha256": "ee"}]}}))
        with pytest.raises(SystemExit) as raised:
            script.profile_from_budget_source(str(source))
        assert "two different digests" in str(raised.value)

    def test_a_nested_input_that_moved_is_refused_though_the_outer_did_not(
            self, tmp_path):
        """The failure an outer digest cannot see.

        A calibration can be rewritten without touching a byte of the profile
        that names it, so a profile digest that still matches says nothing
        about the numbers the derivation is about to read.
        """
        script = _script()
        calibration = tmp_path / "calibration.json"
        calibration.write_text(json.dumps({"persistent": 1}))
        load = script.attesting_loader({str(calibration): "0" * 64}, "budget")
        with pytest.raises(SystemExit) as raised:
            load(str(calibration))
        assert "moved under the comparison" in str(raised.value)

    def test_an_input_the_run_never_published_is_refused(self, tmp_path):
        script = _script()
        other = tmp_path / "elsewhere.json"
        other.write_text(json.dumps({}))
        load = script.attesting_loader({}, "budget_source.json")
        with pytest.raises(SystemExit) as raised:
            load(str(other))
        assert "does not attest" in str(raised.value)

    def test_an_attested_input_is_read_once(self, tmp_path):
        """Two reads of one file can disagree, and the second would win.

        `derived_readings` opens the calibration and the returned mapping was
        opened again separately, so the prediction and the row notes could be
        describing different bytes.
        """
        script = _script()
        path = tmp_path / "calibration.json"
        path.write_text(json.dumps({"persistent": 1}))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        load = script.attesting_loader({str(path): digest}, "budget")
        first = load(str(path))
        path.write_text(json.dumps({"persistent": 2}))
        assert load(str(path)) is first

    def test_a_prediction_that_no_longer_reproduces_the_run_is_refused(
            self, tmp_path):
        """Same digests, different answer.

        The run stored what it derived. If re-deriving it here lands somewhere
        else -- a different revision of the model, a different warmup shape --
        then this is not the prediction that sized the run, and charging the
        record for the gap would be charging it for the wrong thing.
        """
        script = _script()
        profile, _ = _servable_profile(tmp_path, 1)
        config = {"model": "Qwen/Qwen3.8-27B"}
        predicted = script.predicted_terms(_attest(profile), config, 16384, 1)
        published = int(predicted["readings"]["non_torch"])
        with pytest.raises(SystemExit) as raised:
            script.predicted_terms(
                _attest(profile, lineage={"non_torch": published + 1}),
                config, 16384, 1)
        assert "not the prediction that sized the run" in str(raised.value)

    def test_a_record_at_another_width_than_the_run_published_is_refused(
            self, tmp_path):
        script = _script()
        profile, _ = _servable_profile(tmp_path, 1)
        with pytest.raises(SystemExit) as raised:
            script.predicted_terms(
                _attest(profile, lineage={"world_size": 2}),
                {"model": "Qwen/Qwen3.8-27B"}, 16384, 1)
        assert "TP=2" in str(raised.value)

    def test_the_calibration_priced_is_the_one_the_derivation_read(
            self, tmp_path):
        """One read, one mapping.

        The row notes quote the calibration and the prediction consumes it. If
        those are two separate `open`s, a file rewritten between them makes
        the notes describe numbers the prediction never saw.
        """
        script = _script()
        profile, calibration = _servable_profile(tmp_path, 1)
        attested = _digests(profile, calibration)
        predicted = script.predicted_terms(
            _attest(profile, attested=attested,
                    profile_sha256=attested[str(profile)]),
            {"model": "Qwen/Qwen3.8-27B"}, 16384, 1)
        assert predicted["calibration"]["persistent"] == 252339712

    def test_a_calibration_that_moved_is_refused_with_the_profile_intact(
            self, tmp_path):
        """The failure the outer digest cannot see, through the real path."""
        script = _script()
        profile, calibration = _servable_profile(tmp_path, 1)
        attested = dict(_digests(profile, calibration),
                        **{str(calibration): "0" * 64})
        with pytest.raises(SystemExit) as raised:
            script.predicted_terms(
                _attest(profile, attested=attested,
                        profile_sha256=attested[str(profile)]),
                {"model": "Qwen/Qwen3.8-27B"}, 16384, 1)
        assert "moved under the comparison" in str(raised.value)

    def test_the_gate_holds_the_reservation_and_the_capture_apart(self):
        """Two questions about the pool, neither standing in for the other.

        `reservation` is `0.2 x` the modelled activations -- the policy that
        actually leaves the KV budget -- and `graph pool` is what capture goes
        on to cost. They differ by 3x on this model, so a gate that covered
        only one of them would report a validated pool either way.
        """
        script = _script()
        assert "reservation" in script.REQUIRED_TERMS
        assert "graph pool" in script.REQUIRED_TERMS

    def test_the_capture_term_comes_from_the_profile_not_a_constant(self):
        """The 104 MiB width constant is superseded and no predictor calls it.

        Gating against it charged the model +26.8% at TP=4 for a formula the
        run never used. The derived side is the source-derived capture replay
        the profile's own calibration publishes.
        """
        script = _script()
        published, note = script.published_capture(
            {"capture_reserved": {"4": {"total": 85983232,
                                        "provenance": "replayed"}}}, 4)
        assert (published, note) == (85983232, "replayed")

    def test_an_unpublished_width_is_not_answered_from_a_neighbour(self):
        """The carry-forward failure the topology module refuses, again."""
        script = _script()
        published, why = script.published_capture(
            {"capture_reserved": {"2": {"total": 106954752}}}, 4)
        assert published is None
        assert "TP=4" in why

    def test_a_profile_publishing_no_capture_prediction_leaves_it_uncovered(self):
        script = _script()
        published, why = script.published_capture({"persistent": 1}, 2)
        assert published is None
        assert "no source-derived capture prediction" in why

    def test_a_record_with_no_readings_fails_rather_than_being_skipped(self):
        """Rank 0 clean and rank 1 silent used to read as a pass.

        Every term that printed was inside tolerance; the run whose numbers
        were missing is the one nobody looked at.
        """
        script = _script()
        script.SEEN.update({name: 1.0 for name in script.REQUIRED_TERMS})
        script.PROBLEMS.append("tp4.memory.tp1.json carries no readings")
        assert script.gate("tp4.memory.tp0.json") is False

    def test_a_requested_record_that_is_not_there_fails_completeness(self):
        script = _script()
        script.SEEN.update({name: 1.0 for name in script.REQUIRED_TERMS})
        script.PROBLEMS.append("mem_*.tp1.json was asked for and matched no file")
        assert script.gate("record") is False

    def test_a_forecast_that_misses_the_real_count_is_still_a_forecast(self):
        """The published count is modelled, and so is the plan: a run whose
        forecast came in 1% over the blocks it actually got is exactly the
        thing this gate exists to measure, not an inconsistency to refuse.

        Requiring the published count to equal the *recorded* one rejected
        every non-exact prediction -- which is every prediction worth making.
        """
        script = _script()
        blob = _record("27b.tp1.memory.json")
        config_path = str(RECORDS / "qwen3_5_27b.config.json")
        plan = _plan_entries(script, blob, config_path)
        # The run forecast `plan` blocks and published that; the device went
        # on to give it 1% fewer. A 1% miss is a pass.
        real = int(plan / 1.01)
        blob = dict(blob, blocks=dict(blob["blocks"], num_kvcache_blocks=real))
        script.kv_rows(blob["config"], blob["readings"], 1, 1, blob,
                       config_path, predicted=blob["readings"], served=plan)
        assert script.PROBLEMS == []
        assert abs(script.SEEN["kv blocks"] - 1.0) < 0.05
        for name in script.REQUIRED_TERMS:
            script.SEEN.setdefault(name, 0.0)
        assert script.gate("a 1% forecast miss") is True

    def test_a_published_count_the_plan_does_not_reproduce_is_refused(self):
        """Identity, not accuracy: both sides here are the model's own.

        The run stored the capacity its forecast arrived at. If re-deriving
        that forecast lands somewhere else, the plan being gated is not the
        plan the run was sized by.
        """
        script = _script()
        blob = _record("27b.tp1.memory.json")
        config_path = str(RECORDS / "qwen3_5_27b.config.json")
        served = _plan_entries(script, blob, config_path) + 1
        script.kv_rows(blob["config"], blob["readings"], 1, 1, blob,
                       config_path, predicted=blob["readings"], served=served)
        assert any("re-deriving its own plan" in problem
                   for problem in script.PROBLEMS)

    def test_a_record_with_no_measured_count_does_not_borrow_the_forecast(self):
        """Comparing the prediction with itself reads +0.00% and tests
        nothing, so the term goes uncovered and the gate fails."""
        script = _script()
        blob = _record("27b.tp1.memory.json")
        config_path = str(RECORDS / "qwen3_5_27b.config.json")
        served = _plan_entries(script, blob, config_path)
        blob = dict(blob, blocks={})
        script.kv_rows(blob["config"], blob["readings"], 1, 1, blob,
                       config_path, predicted=blob["readings"], served=served)
        assert script.SEEN["kv blocks"] is None
        assert script.gate("no measured count") is False
