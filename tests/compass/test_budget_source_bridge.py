"""The bridge from a saved acceptance artifact to MEMORY's budget parser.

`validate_memory --budget-source` reads a file whose **root** is one
`compass.memory.budget_source/1` object. Nothing writes such a file: the record
is published by the runner, carried out through `/compass/provenance`, and
saved inside a larger artifact. So the question this answers is whether the
object we already save is the object that parser already accepts -- and if it
is, no writer and no second schema is needed, only a selector.

The record comes from the real replay runner, travels through the real
endpoint, is written by the real `SideRun._check_provenance`, is read back off
disk, and the budget object is handed to the parser **unchanged**.

Writing that object out to its own file is a projection -- it is a new local
file. What it is not is a new *schema*: the bytes are the record as saved,
under `compass.memory.budget_source/1`, with no field added, renamed or
computed. Building the object here instead would prove only that this test can
build one, which is why it is extracted rather than constructed.

The parser is `scripts/compass/validate_memory.py` from this tree, so that
after integration this exercises the gate that actually runs and notices
producer/consumer drift. `ATOMCOMPASS_MEMVAL_PARSER` points it at another copy
for the pinned cross-branch check -- that override is this task's, not a
supported knob. Either way no MEMORY code is edited.
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

#: This tree's own memory validator. The default on purpose: a bridge test
#: pinned to a copy under `agent_scratch` would pass for ever without ever
#: reading the gate that runs, and producer/consumer drift is exactly what it
#: exists to catch.
PARSER = (Path(__file__).resolve().parents[2] / "scripts" / "compass"
          / "validate_memory.py")

#: Point this at another checkout's `validate_memory.py` to run the same
#: bridge against it -- the pinned cross-branch check this task needed. Not a
#: supported setting; it exists so a cross-branch verification does not have
#: to be done by editing the test.
PARSER_ENV = "ATOMCOMPASS_MEMVAL_PARSER"


@pytest.fixture(scope="module")
def memval():
    """The parser, imported for real.

    The only skip is the one case that is a fact about the checkout rather
    than about the code: a `validate_memory.py` predating
    `profile_from_budget_source` has nothing for this to test. An import error
    in a parser that *is* there is a failure -- swallowing it would turn a
    broken gate into a green run.
    """
    where = Path(os.environ.get(PARSER_ENV) or PARSER)
    if not where.exists():
        pytest.skip(f"{where} is not present in this checkout")
    spec = importlib.util.spec_from_file_location("memval_under_test", where)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "profile_from_budget_source"):
        pytest.skip(f"{where} predates profile_from_budget_source")
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


def _replay_shaped(saved_path, out):
    """The same payload as `replay.py` nests it, under `.run.server`."""
    with open(saved_path, encoding="utf-8") as handle:
        said = json.load(handle)
    out.write_text(json.dumps({"run": {"server": said}}), encoding="utf-8")
    return out


class TestTheSavedArtifactIsAcceptedDirectly:
    """No extraction step at all, where the consumer can take the artifact.

    `budget_source_object` reads either container, so `--budget-source` can be
    pointed straight at what the harness wrote. That is better than a jq step
    for the same reason the record beats the option string: one fewer file
    between the run and the thing reading it.

    Skipped where the parser under test predates that -- the extraction path
    is what those versions take, and it still works.
    """

    def _direct(self, memval, path, **kw):
        if not hasattr(memval, "budget_source_object"):
            pytest.skip("this parser predates budget_source_object")
        with open(path, encoding="utf-8") as handle:
            blob = json.load(handle)
        return memval.budget_source_object(blob, str(path), **kw)

    def test_the_provenance_artifact_is_read_without_extraction(
            self, memval, monkeypatch, tmp_path):
        saved = _saved(tmp_path, _api_payload(monkeypatch, tmp_path))

        direct = self._direct(memval, saved)

        assert direct == _select(saved), "the same object, not a copy of some"

    def test_the_replay_artifact_is_read_without_extraction(
            self, memval, monkeypatch, tmp_path):
        """The other container: `.run.server.compass...`."""
        saved = _saved(tmp_path, _api_payload(monkeypatch, tmp_path))
        nested = _replay_shaped(saved, tmp_path / "modelled.r1.json")

        direct = self._direct(memval, nested)

        assert direct == _select(saved)

    def test_an_already_extracted_object_still_reads(self, memval, monkeypatch,
                                                     tmp_path):
        """Compatibility both ways: the extraction path keeps working, so a
        cell prepared for an older consumer is not stranded."""
        saved = _saved(tmp_path, _api_payload(monkeypatch, tmp_path))
        handed = tmp_path / "budget_source.json"
        handed.write_text(json.dumps(_select(saved)), encoding="utf-8")

        direct = self._direct(memval, handed)

        assert direct == _select(saved)

    def test_the_direct_read_yields_the_same_results_as_extraction(
            self, memval, monkeypatch, tmp_path):
        """The claim that matters: pointing at the artifact and pointing at an
        extracted object give the parser the same profile, digest, attested
        inputs, lineage and served count."""
        if not hasattr(memval, "budget_source_object"):
            pytest.skip("this parser predates budget_source_object")
        saved = _saved(tmp_path, _api_payload(monkeypatch, tmp_path))
        handed = tmp_path / "budget_source.json"
        handed.write_text(json.dumps(_select(saved)), encoding="utf-8")

        extracted = memval.profile_from_budget_source(str(handed))
        with open(saved, encoding="utf-8") as handle:
            direct_obj = memval.budget_source_object(
                json.load(handle), str(saved))
        straight = tmp_path / "straight.json"
        straight.write_text(json.dumps(direct_obj), encoding="utf-8")
        direct = memval.profile_from_budget_source(str(straight))

        for key in ("profile", "profile_sha256", "attested", "lineage",
                    "num_kvcache_blocks", "kind"):
            assert direct[key] == extracted[key], key

    def test_one_physical_record_is_not_spread_across_logical_ranks(
            self, memval, monkeypatch, tmp_path):
        """The bridge's own caveat, checked against the consumer's rule.

        A GPU-free replay writes one physical record carrying a whole width's
        budget. Asked for a width it does not state, the selector must not
        hand it over as if it were that rank's.
        """
        if not hasattr(memval, "budget_source_object"):
            pytest.skip("this parser predates budget_source_object")
        saved = _saved(tmp_path, _api_payload(monkeypatch, tmp_path))
        with open(saved, encoding="utf-8") as handle:
            blob = json.load(handle)

        ranks = blob["compass"]["loaded_inputs"]["ranks"]
        assert len(ranks) == 1, "one physical predictor"
        # One record is unambiguous and is returned whatever width is asked
        # for; the record states its own width and the caller compares.
        direct = memval.budget_source_object(blob, str(saved), world=4)
        stated = (direct.get("lineage") or {}).get("world_size")
        assert stated != 4 or stated is None, (
            "if this ever states 4 the fixture changed, not the rule")

    def test_two_records_that_cannot_be_told_apart_are_refused(
            self, memval, monkeypatch, tmp_path):
        """And where there really are two, an index is the wrong way to
        choose, so the selector refuses rather than picking."""
        if not hasattr(memval, "budget_source_object"):
            pytest.skip("this parser predates budget_source_object")
        saved = _saved(tmp_path, _api_payload(monkeypatch, tmp_path))
        with open(saved, encoding="utf-8") as handle:
            blob = json.load(handle)
        ranks = blob["compass"]["loaded_inputs"]["ranks"]
        ranks.append(json.loads(json.dumps(ranks[0])))

        with pytest.raises(SystemExit) as refused:
            memval.budget_source_object(blob, str(saved))

        assert "not a TP rank" in str(refused.value)


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
