"""The bridge from a saved acceptance artifact to MEMORY's budget parser.

`validate_memory --budget-source` reads a file whose **root** is one
`compass.memory.budget_source/1` object. Nothing writes such a file: the record
is published by the runner, carried out through `/compass/provenance`, and
saved inside a larger artifact. So the question this answers is whether the
object we already save is the object that parser already accepts -- and if it
is, no writer and no second schema is needed, only a selector.

Nothing here is projected or reshaped. The record comes from the real replay
runner, travels through the real endpoint, is written by the real
`SideRun._check_provenance`, is read back off disk, and the budget object is
handed to the parser **unchanged**. A local projection of the object would
prove only that this test can build one.

The parser is the pinned 99e3943a copy, staged read-only under agent_scratch.
No MEMORY code is edited, and the copy is not on the import path of anything
else.
"""

import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import pytest

from . import test_cc_traces_validate as base
from . import test_replay_memory as mem

validate = base.validate

#: The pinned parser, staged beside this worktree rather than vendored.
PARSER = (Path(__file__).resolve().parents[2] / "agent_scratch"
          / "memval_bridge" / "validate_memory_99e3943a.py")


@pytest.fixture(scope="module")
def memval():
    if not PARSER.exists():
        pytest.skip(f"{PARSER} is not staged here")
    spec = importlib.util.spec_from_file_location("memval_99e3943a", PARSER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001 - environment-dependent
        pytest.skip(f"the pinned validator does not import here: {exc}")
    return module


def _api_payload(monkeypatch, tmp_path):
    """A real runner's record, out through the real provenance endpoint."""
    import asyncio
    import importlib

    from atom.compass.config import CompassConfig
    from atom.compass.replay.runner import ReplayModelRunner

    try:
        api_server = importlib.import_module("atom.entrypoints.openai.api_server")
    except Exception:  # noqa: BLE001 - environment-dependent
        pytest.skip("api_server import unavailable")

    runner = ReplayModelRunner(
        0, mem._config(tmp_path, profile=mem._profile(tmp_path, 1)))
    runner.get_num_blocks()
    manifest = runner.compass_input_manifest()

    engine = types.SimpleNamespace(
        config=types.SimpleNamespace(
            compass_config=CompassConfig(
                enabled=True, mode="predict",
                oracle_qualname=validate.SOURCE_FACTORY,
                oracle_options={}),
            model_config=None, parallel_config=None),
        get_compass_inputs=lambda timeout=10.0: {"ranks": [manifest]})
    monkeypatch.setattr(api_server, "engine", engine)
    said = asyncio.run(api_server.compass_provenance())
    said["tensor_parallel_size"] = 1
    return said


def _saved(tmp_path, said):
    """Written by `SideRun._check_provenance` itself, not by this test.

    The method is bound to a stub carrying the four attributes it reads, so
    the validation it does before saving runs, and the file lands under the
    name and with the formatting the harness actually produces.
    """
    run = _harness()
    cell = tmp_path / "cell"
    cell.mkdir(exist_ok=True)
    side = types.SimpleNamespace(
        cell=cell, side="modelled", plan={"tp": 1}, failures=[])
    side._check_provenance = types.MethodType(
        run.SideRun._check_provenance, side)
    execution = {"config": {}, "artifacts": {}}

    out = side._check_provenance({"repeat": 1, "id": "serve-modelled-1"},
                                 said, execution)

    assert not side.failures, side.failures
    assert out is not None
    return cell / "provenance.modelled.r1.json"


def _harness():
    path = (Path(__file__).resolve().parents[2] / "scripts" / "compass"
            / "cc_traces_run.py")
    spec = importlib.util.spec_from_file_location("compass_run_bridge", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _select(saved_path, index=0):
    """The selector a consumer runs. One key path, no reshaping.

    `.compass.loaded_inputs.ranks[i].budget_source` in the provenance
    artifact. The list is the engine cores that answered -- physical
    predictors -- and not TP rank ordinals: a GPU-free replay stands one
    process in for a logical group, so it has one record whose budget is for
    the whole width. Which width that is, is in the record, not in the index.
    """
    with open(saved_path, encoding="utf-8") as handle:
        blob = json.load(handle)
    ranks = blob["compass"]["loaded_inputs"]["ranks"]
    return ranks[index]["budget_source"]


class TestTheSavedArtifactCarriesWhatTheParserReads:

    def test_the_budget_object_is_at_the_traced_key_path(self, monkeypatch,
                                                         tmp_path):
        saved = _saved(tmp_path, _api_payload(monkeypatch, tmp_path))

        record = _select(saved)

        assert record["schema"] == "compass.memory.budget_source/1"
        assert record["kind"] == "source-derived"
        assert record["inputs"]["inputs"], "the nested manifest the parser reads"

    def test_the_parser_accepts_it_unchanged(self, memval, monkeypatch,
                                             tmp_path):
        """Written out verbatim -- no projection, no added or renamed field --
        because that is what decides whether a writer is needed."""
        saved = _saved(tmp_path, _api_payload(monkeypatch, tmp_path))
        record = _select(saved)
        handed = tmp_path / "budget_source.json"
        handed.write_text(json.dumps(record), encoding="utf-8")

        out = memval.profile_from_budget_source(str(handed))

        assert out["kind"] == "source-derived"

    def test_it_receives_the_exact_profile_path_and_digest(self, memval,
                                                           monkeypatch,
                                                           tmp_path):
        saved = _saved(tmp_path, _api_payload(monkeypatch, tmp_path))
        record = _select(saved)
        handed = tmp_path / "budget_source.json"
        handed.write_text(json.dumps(record), encoding="utf-8")

        out = memval.profile_from_budget_source(str(handed))

        profile = next(row for row in record["inputs"]["inputs"]
                       if row["role"] == "runtime.memory_model")
        assert out["profile"] == profile["path"]
        assert out["profile_sha256"] == profile["sha256"]
        import hashlib
        with open(profile["path"], "rb") as handle:
            assert out["profile_sha256"] == hashlib.sha256(
                handle.read()).hexdigest()

    def test_it_receives_the_nested_attested_inputs(self, memval, monkeypatch,
                                                    tmp_path):
        """The whole manifest, not the profile row. The calibration and the
        model config the profile names are where the collective constants and
        the KV geometry live, and a nested file can move without the profile's
        own bytes moving."""
        saved = _saved(tmp_path, _api_payload(monkeypatch, tmp_path))
        record = _select(saved)
        handed = tmp_path / "budget_source.json"
        handed.write_text(json.dumps(record), encoding="utf-8")

        out = memval.profile_from_budget_source(str(handed))

        roles = {row["role"] for row in record["inputs"]["inputs"]}
        assert any(role.startswith("runtime.memory_model.") for role in roles), (
            "the profile names files of its own")
        for row in record["inputs"]["inputs"]:
            assert out["attested"][os.path.abspath(row["path"])] == row["sha256"]

    def test_it_receives_the_lineage_and_the_served_count(self, memval,
                                                          monkeypatch,
                                                          tmp_path):
        saved = _saved(tmp_path, _api_payload(monkeypatch, tmp_path))
        record = _select(saved)
        handed = tmp_path / "budget_source.json"
        handed.write_text(json.dumps(record), encoding="utf-8")

        out = memval.profile_from_budget_source(str(handed))

        assert out["lineage"] == record["lineage"]
        assert out["num_kvcache_blocks"] == record["num_kvcache_blocks"]
        assert out["num_kvcache_blocks"] > 0, "what the run actually served"


class TestTheSelectorIsAboutRecordsAndNotRankOrdinals:

    def test_the_width_comes_from_the_record_not_the_index(self, monkeypatch,
                                                           tmp_path):
        """A GPU-free replay stands one process in for a logical group, so the
        list has one entry whose budget is for the whole width. Reading the
        index as a TP rank would invent ranks that published nothing."""
        saved = _saved(tmp_path, _api_payload(monkeypatch, tmp_path))
        with open(saved, encoding="utf-8") as handle:
            ranks = json.load(handle)["compass"]["loaded_inputs"]["ranks"]

        assert len(ranks) == 1
        record = ranks[0]["budget_source"]
        assert "world_size" in (record.get("lineage") or {}) or record.get(
            "deployment"), "the width is stated, not positional"

    def test_a_missing_record_is_absent_rather_than_invented(self, monkeypatch,
                                                             tmp_path):
        saved = _saved(tmp_path, _api_payload(monkeypatch, tmp_path))
        with open(saved, encoding="utf-8") as handle:
            ranks = json.load(handle)["compass"]["loaded_inputs"]["ranks"]

        with pytest.raises(IndexError):
            ranks[1]["budget_source"]


class TestWhatTheBridgeCannotCarry:
    """Two boundaries found while verifying the bridge. Both are MEMORY's to
    correct; recorded here so the handoff carries evidence rather than a
    claim, and so a later change to either is noticed here first."""

    def _measured_record(self):
        """A real device-measured record, from the native publication path."""
        from atom.compass.replay import bootstrap

        from .test_native_reference_budget import _Sensors, _sized

        try:
            if not bootstrap.state().get("installed"):
                bootstrap.install("gfx942:sramecc+:xnack-", source="test")
            from atom.compass.runtime import runner
        except Exception as exc:  # noqa: BLE001 - environment-dependent
            pytest.skip(f"the native runner is not importable here: {exc}")
        return _sized(_Sensors(runner))

    def test_a_device_measured_budget_cannot_be_handed_to_this_parser(
            self, memval, tmp_path):
        """`--budget-source` cannot carry the real reference side.

        `profile_from_budget_source` requires a `runtime.memory_model` input,
        and a budget taken off the card read no profile -- that is what makes
        it a measurement. So the flag reaches only the modelled side today.
        The historical `--profile-only` gate published `served=None`, which is
        why nothing noticed.
        """
        record = self._measured_record()
        assert record["kind"] == "device-measured"
        assert not [row for row in record["inputs"]["inputs"]
                    if row["role"] == "runtime.memory_model"]

        handed = tmp_path / "measured.json"
        handed.write_text(json.dumps(record), encoding="utf-8")

        with pytest.raises(SystemExit) as refused:
            memval.profile_from_budget_source(str(handed))

        assert "runtime.memory_model" in str(refused.value)



class TestAnOldRecordKeepsPointingAtWhatItRead:

    def test_the_digest_is_of_the_profile_that_run_loaded(self, memval,
                                                          monkeypatch,
                                                          tmp_path):
        """A profile written later -- `profile_next`, with `capture_reserved`
        appended -- is a different file. An old run's record binds the old
        bytes, and the parser follows the record rather than the name, so
        re-pointing it is neither necessary nor possible by accident."""
        saved = _saved(tmp_path, _api_payload(monkeypatch, tmp_path))
        record = _select(saved)
        handed = tmp_path / "budget_source.json"
        handed.write_text(json.dumps(record), encoding="utf-8")
        before = memval.profile_from_budget_source(str(handed))

        profile = Path(before["profile"])
        moved = json.loads(profile.read_text())
        moved["capture_reserved"] = 1234
        profile.write_text(json.dumps(moved), encoding="utf-8")

        after = memval.profile_from_budget_source(str(handed))

        assert after["profile_sha256"] == before["profile_sha256"], (
            "the record still describes the bytes that run read")
        import hashlib
        with open(profile, "rb") as handle:
            assert hashlib.sha256(handle.read()).hexdigest() != after[
                "profile_sha256"], "and the file on disk has moved away from it"
